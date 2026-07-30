"""Automatically search and validate a two-stick Wuji closed-grasp pose.

The search is simulation-only. It first uses many parallel ``hand_grasp``
environments as a batched forward-kinematics evaluator, then re-optimizes the
last CEM iterations through actuator/self-collision rollouts. Stick2's stable
valley pose is searched jointly with the finger joints. The best candidates
are finally ramped onto two dynamic sticks and released. The ranking includes
desired contacts, stick-stick contact region/force, slip, drop, speed, per-joint
tracking error, and actuator torque clipping.

Example:

.. code-block:: bash

    python scripts/debug/hand_grasp_ik_search.py \
        --task hand_grasp \
        --headless \
        --num-envs 512
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PREGRASP_CANDIDATE = (
    PROJECT_ROOT
    / "isaac_neuromeka"
    / "tasks"
    / "manipulation"
    / "hand_grasp"
    / "pregrasp_candidate.json"
)


parser = argparse.ArgumentParser(description="Automatic Wuji two-stick IK/contact search.")
parser.add_argument("--task", type=str, default="hand_grasp")
parser.add_argument("--num-envs", type=int, default=512, help="Parallel CEM population.")
parser.add_argument("--search-iterations", type=int, default=35)
parser.add_argument(
    "--physics-aware-iterations",
    type=int,
    default=10,
    help="Final CEM iterations that use actuator/self-collision rollout instead of teleported FK.",
)
parser.add_argument(
    "--search-ramp-steps",
    type=int,
    default=24,
    help="Physics-aware CEM ramp steps per stage (proximal, then distal).",
)
parser.add_argument(
    "--search-settle-steps",
    type=int,
    default=72,
    help="Physics-aware CEM settle steps after each stage.",
)
parser.add_argument(
    "--search-release-steps",
    type=int,
    default=60,
    help="Short dynamic release scored inside each physics-aware CEM iteration.",
)
parser.add_argument(
    "--joint-limit-margin",
    type=float,
    default=0.10,
    help="Fraction of each joint range excluded at both hard limits during CEM.",
)
parser.add_argument("--elite-fraction", type=float, default=0.08)
parser.add_argument("--validate-k", type=int, default=16)
parser.add_argument(
    "--ramp-steps",
    type=int,
    default=60,
    help="Validation ramp steps per stage (proximal, then distal).",
)
parser.add_argument(
    "--stage-settle-steps",
    type=int,
    default=120,
    help="Validation settle steps after each stage.",
)
parser.add_argument("--release-steps", type=int, default=240, help="Free-stick validation physics steps.")
parser.add_argument("--contact-threshold", type=float, default=0.02)
parser.add_argument("--max-stick-speed", type=float, default=0.50)
parser.add_argument("--max-stick-displacement", type=float, default=0.05)
parser.add_argument("--max-pair-force", type=float, default=5.0)
parser.add_argument("--max-joint-tracking-error", type=float, default=0.25)
parser.add_argument("--tip-region-max-y", type=float, default=-0.050)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--compare-self-collision",
    action="store_true",
    help="Deprecated alias for --self-collision-condition off.",
)
parser.add_argument(
    "--self-collision-condition",
    choices=("on", "off"),
    default=None,
    help="Replay latest candidates once with the selected self-collision setting.",
)
parser.add_argument(
    "--search-self-collision",
    choices=("on", "off"),
    default="on",
    help="Self-collision setting for a full CEM search; keep ON for final validation.",
)
parser.add_argument(
    "--probe-self-collision-pairs",
    action="store_true",
    help="Replay one saved pre-grasp with self-collision ON and log exact hand link pairs.",
)
parser.add_argument(
    "--probe-ring-reach",
    action="store_true",
    help="Sweep only finger4_joint2 around the saved 5/6 pre-grasp after valley settling.",
)
parser.add_argument(
    "--candidate-file",
    type=Path,
    default=DEFAULT_PREGRASP_CANDIDATE,
    help="Canonical pre-grasp JSON used by --probe-self-collision-pairs.",
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=PROJECT_ROOT / "logs" / "debug" / "hand_grasp_ik_search",
)
parser.add_argument("--debug-vis", action="store_true")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


RUN_STAMP = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = args_cli.output_root.expanduser().resolve() / RUN_STAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


STICK_HALF_SIZE = torch.tensor([0.0035, 0.0900, 0.0035])

# Unilateral palm+thumb anchor seed from collision probe 2026-07-27_16-19-09.
# Candidate 36 was dynamically quiet but was not a feasible two-sided valley
# grasp: index-side contact was absent. Keep it only as a search reference.
STICK2_POS_P = torch.tensor([0.0163271502, 0.0250045005, 0.0503145084])
STICK2_QUAT_P = torch.tensor([1.0, 0.0, 0.0, 0.0])
# The first CEM searches saturated both the 16.3 mm palm-normal upper bound and
# the opposing fingers' flexion bounds while remaining 12--16 mm from contact.
# Search up to 30 mm above the seed. The 20 mm bound saturated in the
# 2026-07-27_21-15-49 run while the only missing maintained contact was the
# ring fingertip, so leave room for a preloaded lower-stick contact. The
# palm-z range stays narrow because it did not saturate.
STICK2_SEARCH_LOWER_XZ = torch.tensor([STICK2_POS_P[0] - 0.005, STICK2_POS_P[2]])
STICK2_SEARCH_UPPER_XZ = torch.tensor(
    [STICK2_POS_P[0] + 0.030, STICK2_POS_P[2] + 0.005]
)

# Stick1 closes about a tail-side pivot. The collision probe found the first
# stick-stick contact at the tip at 5 degrees.
PAIR_CLOSE_DEG = 5.0
PAIR_CENTER_OFFSET = 0.020
PAIR_PIVOT_Y = 0.060
# Search the upper stick's radial separation instead of forcing the original
# 20 mm hypothesis. The lower bound keeps the two 7 mm square sticks from
# starting in overlap; the close fraction below is converted to the largest
# tip-touching angle allowed by each sampled separation.
PAIR_CENTER_OFFSET_RANGE = (0.009, 0.030)
PAIR_CLOSE_FRACTION_RANGE = (0.0, 1.0)
PAIR_TIP_LEVER = float(STICK_HALF_SIZE[1]) + PAIR_PIVOT_Y
PAIR_MIN_CENTERLINE_SEPARATION = 2.0 * float(STICK_HALF_SIZE[0])

SEARCH_JOINT_NAMES = [
    f"finger{finger}_joint{joint}"
    for finger in range(1, 5)
    for joint in range(1, 5)
]
PROXIMAL_SEARCH_JOINT_NAMES = [
    name for name in SEARCH_JOINT_NAMES if name.endswith(("joint1", "joint2"))
]
ALL_JOINT_NAMES = [
    f"finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]
ALL_HAND_BODY_NAMES = [
    "palm_link",
    *[
        body_name
        for finger in range(1, 6)
        for body_name in (
            f"finger{finger}_link1",
            f"finger{finger}_link2",
            f"finger{finger}_link3",
            f"finger{finger}_link4",
            f"finger{finger}_tip_link",
        )
    ],
]
BODY_NAMES = [
    "palm_link",
    "finger1_link2",
    "finger1_tip_link",
    "finger2_tip_link",
    "finger3_tip_link",
    "finger4_tip_link",
]

# Collision-mesh-informed pad points in each fingertip link frame. The distal
# pad extends mainly along local -z; these offsets avoid optimizing link-frame
# origins several millimeters behind the physical pad.
TIP_PAD_OFFSETS_B = {
    "finger1_tip_link": (0.0, 0.0, -0.0090),
    "finger2_tip_link": (0.0, 0.0, -0.0140),
    "finger3_tip_link": (0.0, 0.0, -0.0140),
    "finger4_tip_link": (0.0, 0.0, -0.0140),
}
THUMB_MID_PROXY_B = (0.0, 0.0, 0.0160)

CONTACT_SPECS = {
    "thumb_tip_stick1": ("finger1_tip_link", "Stick1"),
    "index_tip_stick1": ("finger2_tip_link", "Stick1"),
    "middle_tip_stick1": ("finger3_tip_link", "Stick1"),
    "ring_tip_stick2": ("finger4_tip_link", "Stick2"),
    "palm_stick2": ("palm_link", "Stick2"),
    "thumb_mid_stick2": ("finger1_link2", "Stick2"),
}
GEOMETRIC_ERROR_LABELS = (
    "thumb_tip_stick1",
    "index_tip_stick1",
    "middle_tip_stick1",
    "ring_tip_stick2",
    "thumb_mid_stick2",
)


@dataclass(frozen=True)
class IkCandidate:
    loss: float
    joint_positions: list[float]
    geometric_errors: list[float]
    stick2_pos_p: list[float]
    pair_center_offset_m: float
    pair_close_deg: float
    search_tracking_errors: list[float]
    min_joint_limit_margin_fraction: float
    source_rank: int | None = None
    search_contact_forces: list[float] | None = None
    search_release_contact_fractions: list[float] | None = None
    search_release_displacements: list[float] | None = None


class ProbeLog:
    def __init__(self, path: Path):
        self._file = path.open("w", encoding="utf-8")

    def write(self, message: str = "") -> None:
        print(message, flush=True)
        self._file.write(message + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def _quat_x(angles: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angles
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), torch.sin(half), zeros, zeros), dim=-1)


def _pair_close_angles(
    pair_center_offsets: torch.Tensor,
    close_fractions: torch.Tensor,
) -> torch.Tensor:
    """Convert [0, 1] close fractions to non-overlapping tip-close angles."""
    sine_limit = torch.clamp(
        (pair_center_offsets - PAIR_MIN_CENTERLINE_SEPARATION) / PAIR_TIP_LEVER,
        min=0.0,
        max=1.0,
    )
    return close_fractions * torch.asin(sine_limit)


def _stick_poses_in_palm(
    count: int,
    device: str,
    stick2_xz: torch.Tensor | None = None,
    pair_center_offsets: torch.Tensor | None = None,
    pair_close_angles: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    stick2_pos = STICK2_POS_P.to(device).unsqueeze(0).repeat(count, 1)
    if stick2_xz is not None:
        stick2_pos[:, 0] = stick2_xz[:, 0]
        stick2_pos[:, 2] = stick2_xz[:, 1]
    stick2_quat = STICK2_QUAT_P.to(device).unsqueeze(0).repeat(count, 1)
    if pair_center_offsets is None:
        pair_center_offsets = torch.full(
            (count,),
            PAIR_CENTER_OFFSET,
            device=device,
            dtype=stick2_pos.dtype,
        )
    if pair_close_angles is None:
        pair_close_angles = torch.full(
            (count,),
            math.radians(PAIR_CLOSE_DEG),
            device=device,
            dtype=stick2_pos.dtype,
        )
    # Palm local +z maps to world +x (finger progression), so the upper
    # Stick1 must be offset along palm z rather than palm x/world z. A positive
    # palm-x rotation (world-z rotation) closes its tip toward Stick2.
    close_quat = _quat_x(pair_close_angles)
    pivot_offset = torch.stack(
        (
            torch.zeros_like(pair_center_offsets),
            torch.full_like(pair_center_offsets, PAIR_PIVOT_Y),
            pair_center_offsets,
        ),
        dim=-1,
    )
    pivot = stick2_pos + pivot_offset
    center_from_pivot = torch.tensor([0.0, -PAIR_PIVOT_Y, 0.0], device=device)
    center_from_pivot = center_from_pivot.unsqueeze(0).repeat(count, 1)
    stick1_pos = pivot + quat_apply(close_quat, center_from_pivot)
    return stick1_pos, close_quat, stick2_pos, stick2_quat


def _compose_world_pose(
    palm_pos_w: torch.Tensor,
    palm_quat_w: torch.Tensor,
    pos_p: torch.Tensor,
    quat_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return palm_pos_w + quat_apply(palm_quat_w, pos_p), quat_mul(palm_quat_w, quat_p)


def _box_surface_error(
    points_p: torch.Tensor,
    center_p: torch.Tensor,
    quat_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = quat_apply_inverse(quat_p, points_p - center_p)
    half = STICK_HALF_SIZE.to(points_p.device)
    signed_components = torch.abs(local) - half
    outside = torch.linalg.norm(torch.clamp(signed_components, min=0.0), dim=-1)
    inside = torch.clamp(torch.max(signed_components, dim=-1).values, max=0.0)
    return torch.abs(outside + inside), local


def _body_proxy_in_palm(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    offset_b: torch.Tensor,
    palm_pos_w: torch.Tensor,
    palm_quat_w: torch.Tensor,
) -> torch.Tensor:
    point_w = body_pos_w + quat_apply(body_quat_w, offset_b)
    return quat_apply_inverse(palm_quat_w, point_w - palm_pos_w)


def _step_direct(scene, sim, steps: int, render: bool) -> None:
    dt = sim.get_physics_dt()
    for _ in range(steps):
        scene.write_data_to_sim()
        sim.step(render=render)
        scene.update(dt)


def _staged_joint_commands(
    start_q: torch.Tensor,
    goal_q: torch.Tensor,
    proximal_ids: list[int],
    proximal_steps: int,
    proximal_settle_steps: int,
    distal_steps: int,
    distal_settle_steps: int,
):
    """Yield collision-safer close commands with a settle after each stage."""
    proximal_goal = start_q.clone()
    proximal_goal[:, proximal_ids] = goal_q[:, proximal_ids]
    for step in range(proximal_steps):
        alpha = (step + 1) / proximal_steps
        yield "proximal", start_q + alpha * (proximal_goal - start_q)
    for _ in range(proximal_settle_steps):
        yield "proximal", proximal_goal
    for step in range(distal_steps):
        alpha = (step + 1) / distal_steps
        yield "distal", proximal_goal + alpha * (goal_q - proximal_goal)
    for _ in range(distal_settle_steps):
        yield "distal", goal_q


def _write_object_poses(
    obj,
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
) -> None:
    obj.write_root_pose_to_sim(torch.cat((pos_w, quat_w), dim=-1))
    obj.write_root_velocity_to_sim(torch.zeros((pos_w.shape[0], 6), device=pos_w.device))


def _root_state_from_physx(obj, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pose_xyzw = obj.root_physx_view.get_transforms().clone().to(device)
    quat_wxyz = torch.cat((pose_xyzw[:, 6:7], pose_xyzw[:, 3:6]), dim=-1)
    velocity = obj.root_physx_view.get_velocities().clone().to(device)
    return pose_xyzw[:, :3], quat_wxyz, velocity


def _sensor_force(scene, sensor_name: str, count: int, device: str) -> torch.Tensor:
    matrix = scene.sensors[sensor_name].data.force_matrix_w
    if matrix is None:
        return torch.zeros(count, device=device)
    return torch.linalg.norm(matrix.reshape(count, -1, 3), dim=-1).sum(dim=-1)


def _first_pair_contact_y(
    scene,
    sensor_name: str,
    stick2,
    unresolved: torch.Tensor,
    first_y: torch.Tensor,
) -> None:
    points = scene.sensors[sensor_name].data.contact_pos_w
    if points is None or not torch.any(unresolved):
        return
    count = points.shape[0]
    flat = points.reshape(count, -1, 3)
    finite = torch.isfinite(flat).all(dim=-1)
    active = unresolved & finite.any(dim=-1)
    if not torch.any(active):
        return
    stick2_pos_w, stick2_quat_w, _ = _root_state_from_physx(stick2, flat.device.type)
    for env_id in torch.nonzero(active, as_tuple=False).flatten().tolist():
        point_w = flat[env_id, finite[env_id]].mean(dim=0)
        point_s2 = quat_apply_inverse(
            stick2_quat_w[env_id].unsqueeze(0),
            (point_w - stick2_pos_w[env_id]).unsqueeze(0),
        )[0]
        first_y[env_id] = point_s2[1]
        unresolved[env_id] = False


def _add_contact_sensors(env_cfg) -> tuple[dict[str, str], str]:
    env_cfg.scene.stick1.spawn.activate_contact_sensors = True
    env_cfg.scene.stick2.spawn.activate_contact_sensors = True
    sensor_names: dict[str, str] = {}
    for label, (body_name, stick_name) in CONTACT_SPECS.items():
        sensor_name = f"ik_{label}"
        setattr(
            env_cfg.scene,
            sensor_name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
                filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{stick_name}"],
                update_period=0.0,
                history_length=1,
                track_pose=True,
                track_contact_points=True,
                max_contact_data_count_per_prim=8,
                debug_vis=args_cli.debug_vis,
            ),
        )
        sensor_names[label] = sensor_name

    pair_name = "ik_stick1_stick2"
    setattr(
        env_cfg.scene,
        pair_name,
        ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Stick1",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
            update_period=0.0,
            history_length=1,
            track_pose=True,
            track_contact_points=True,
            max_contact_data_count_per_prim=8,
            debug_vis=args_cli.debug_vis,
        ),
    )
    return sensor_names, pair_name


def _resolve_ids(scene) -> tuple[dict[str, int], dict[str, int]]:
    body_cfg = SceneEntityCfg("robot", body_names=BODY_NAMES, preserve_order=True)
    joint_cfg = SceneEntityCfg("robot", joint_names=ALL_JOINT_NAMES, preserve_order=True)
    body_cfg.resolve(scene)
    joint_cfg.resolve(scene)
    return (
        {name: idx for name, idx in zip(BODY_NAMES, body_cfg.body_ids)},
        {name: idx for name, idx in zip(ALL_JOINT_NAMES, joint_cfg.joint_ids)},
    )


def _geometric_loss(
    robot,
    body_ids: dict[str, int],
    palm_pos_w: torch.Tensor,
    palm_quat_w: torch.Tensor,
    stick1_pos_p: torch.Tensor,
    stick1_quat_p: torch.Tensor,
    stick2_pos_p: torch.Tensor,
    stick2_quat_p: torch.Tensor,
    q_target: torch.Tensor,
    q_actual: torch.Tensor,
    q_mid: torch.Tensor,
    q_range: torch.Tensor,
    physics_aware: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    count = q_target.shape[0]
    device = q_target.device
    proxies: dict[str, torch.Tensor] = {}
    for body_name, offset in TIP_PAD_OFFSETS_B.items():
        offset_b = torch.tensor(offset, device=device).unsqueeze(0).repeat(count, 1)
        proxies[body_name] = _body_proxy_in_palm(
            robot.data.body_pos_w[:, body_ids[body_name]],
            robot.data.body_quat_w[:, body_ids[body_name]],
            offset_b,
            palm_pos_w,
            palm_quat_w,
        )
    thumb_mid_offset = torch.tensor(THUMB_MID_PROXY_B, device=device).unsqueeze(0).repeat(count, 1)
    proxies["finger1_link2"] = _body_proxy_in_palm(
        robot.data.body_pos_w[:, body_ids["finger1_link2"]],
        robot.data.body_quat_w[:, body_ids["finger1_link2"]],
        thumb_mid_offset,
        palm_pos_w,
        palm_quat_w,
    )

    objectives = [
        ("finger1_tip_link", stick1_pos_p, stick1_quat_p, 1.0),
        ("finger2_tip_link", stick1_pos_p, stick1_quat_p, 1.0),
        ("finger3_tip_link", stick1_pos_p, stick1_quat_p, 1.0),
        ("finger4_tip_link", stick2_pos_p, stick2_quat_p, 1.5),
        ("finger1_link2", stick2_pos_p, stick2_quat_p, 1.0),
    ]
    errors = []
    weighted = []
    axial_penalty = torch.zeros(count, device=device)
    for body_name, center_p, quat_p, weight in objectives:
        error, local = _box_surface_error(proxies[body_name], center_p, quat_p)
        errors.append(error)
        weighted.append(weight * torch.square(error / 0.015))
        axial_penalty += weight * torch.square(torch.relu(torch.abs(local[:, 1]) - 0.075) / 0.015)

    # Keep non-adjacent fingertips from converging to the same mathematical
    # surface point. Actual self-collision/tracking is handled by the final
    # physics-aware CEM iterations below.
    overlap_penalty = torch.zeros(count, device=device)
    separation_specs = (
        ("finger1_tip_link", "finger2_tip_link", 0.005),
        ("finger1_tip_link", "finger3_tip_link", 0.005),
        ("finger2_tip_link", "finger3_tip_link", 0.010),
        ("finger2_tip_link", "finger4_tip_link", 0.008),
        ("finger3_tip_link", "finger4_tip_link", 0.008),
    )
    for first, second, minimum_distance in separation_specs:
        distance = torch.linalg.norm(proxies[first] - proxies[second], dim=-1)
        overlap_penalty += torch.square(
            torch.relu(minimum_distance - distance) / minimum_distance
        )

    normalized_q = 2.0 * (q_target - q_mid) / q_range
    joint_regularizer = 0.05 * torch.mean(torch.square(normalized_q), dim=-1)
    # The hard sampling margin prevents exact-limit solutions. This smooth
    # barrier starts earlier so CEM still prefers comfortably controllable q.
    joint_barrier = 0.50 * torch.mean(
        torch.square(torch.relu(torch.abs(normalized_q) - 0.65) / 0.35),
        dim=-1,
    )

    tracking_error = torch.abs(q_actual - q_target)
    if physics_aware:
        tracking_mean = 0.35 * torch.mean(torch.square(tracking_error / 0.20), dim=-1)
        tracking_peak = 0.75 * torch.square(
            torch.relu(torch.max(tracking_error, dim=-1).values - 0.25) / 0.20
        )
        tracking_penalty = tracking_mean + tracking_peak
    else:
        tracking_penalty = torch.zeros(count, device=device)

    loss = (
        torch.stack(weighted, dim=-1).sum(dim=-1)
        + 0.2 * axial_penalty
        + 0.35 * overlap_penalty
        + joint_regularizer
        + joint_barrier
        + tracking_penalty
    )
    diagnostics = {
        "overlap": overlap_penalty,
        "joint_barrier": joint_barrier,
        "tracking": tracking_penalty,
        "max_tracking_error": torch.max(tracking_error, dim=-1).values,
    }
    return loss, torch.stack(errors, dim=-1), diagnostics


def _select_diverse_candidates(
    archive: list[IkCandidate],
    count: int,
    minimum_distance: float = 0.05,
) -> list[IkCandidate]:
    selected: list[IkCandidate] = []

    def vector(candidate: IkCandidate) -> torch.Tensor:
        # 10x makes a 5 mm position difference comparable to 0.05 rad in q-space.
        return torch.cat(
            (
                torch.tensor(candidate.joint_positions),
                10.0
                * torch.tensor(
                    [
                        candidate.stick2_pos_p[0],
                        candidate.stick2_pos_p[2],
                        candidate.pair_center_offset_m,
                    ]
                ),
                torch.tensor([math.radians(candidate.pair_close_deg)]),
            )
        )

    for candidate in sorted(archive, key=lambda item: item.loss):
        candidate_vector = vector(candidate)
        if all(
            torch.linalg.norm(candidate_vector - vector(other)).item() >= minimum_distance
            for other in selected
        ):
            selected.append(candidate)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for candidate in sorted(archive, key=lambda item: item.loss):
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= count:
                break
    return selected


def _load_latest_candidates(count: int) -> tuple[Path, list[IkCandidate]]:
    for run_dir in sorted(args_cli.output_root.expanduser().resolve().glob("2026-*"), reverse=True):
        summary_path = run_dir / "summary.json"
        if run_dir == OUTPUT_DIR or not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = summary.get("ranked_candidates") or []
        if not rows:
            continue
        candidates = [
            IkCandidate(
                loss=float(row["search_loss"]),
                joint_positions=[
                    float(row["joint_positions"][name]) for name in SEARCH_JOINT_NAMES
                ],
                geometric_errors=list(row["geometric_errors_m"]),
                stick2_pos_p=list(row["stick2_pose_palm"]),
                pair_center_offset_m=float(
                    row.get("pair_center_offset_m", PAIR_CENTER_OFFSET)
                ),
                pair_close_deg=float(row.get("pair_close_deg", PAIR_CLOSE_DEG)),
                search_tracking_errors=list(row["search_tracking_errors_rad"]),
                min_joint_limit_margin_fraction=float(
                    row["min_joint_limit_margin_fraction"]
                ),
                source_rank=int(row["candidate_rank"]),
                search_contact_forces=(
                    None
                    if row.get("search_contact_force_N") is None
                    else [
                        float(row["search_contact_force_N"][label])
                        for label in CONTACT_SPECS
                    ]
                ),
                search_release_contact_fractions=(
                    None
                    if row.get("search_release_contact_fraction") is None
                    else [
                        float(row["search_release_contact_fraction"][label])
                        for label in CONTACT_SPECS
                    ]
                ),
                search_release_displacements=(
                    None
                    if row.get("search_release_displacement_m") is None
                    else [
                        float(value)
                        for value in row["search_release_displacement_m"]
                    ]
                ),
            )
            for row in rows[:count]
        ]
        return run_dir, candidates
    raise FileNotFoundError(
        f"No prior hand_grasp_ik_search summary.json found under {args_cli.output_root}"
    )


def _load_candidate_file(path: Path) -> tuple[dict[str, Any], IkCandidate]:
    path = path.expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    geometry = data["selection_diagnostics"]["geometric_errors_m"]
    candidate = IkCandidate(
        loss=0.0,
        joint_positions=[
            float(data["joint_positions_rad"][name]) for name in SEARCH_JOINT_NAMES
        ],
        geometric_errors=[float(geometry[name]) for name in GEOMETRIC_ERROR_LABELS],
        stick2_pos_p=list(data["stick2_pose_palm"]["position_m"]),
        pair_center_offset_m=float(data["stick_pair"]["center_offset_m"]),
        pair_close_deg=float(data["stick_pair"]["close_angle_deg"]),
        search_tracking_errors=[0.0] * len(SEARCH_JOINT_NAMES),
        min_joint_limit_margin_fraction=float(
            data["selection_diagnostics"]["minimum_joint_limit_margin_fraction"]
        ),
        source_rank=int(data["source_candidate_rank"]),
    )
    return data, candidate


def _run_ring_reach_probe(log: ProbeLog) -> None:
    """Sweep ring ab/adduction once while preserving the saved 5/6 seed."""
    candidate_path = args_cli.candidate_file.expanduser().resolve()
    candidate_data, base_candidate = _load_candidate_file(candidate_path)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.episode_length_s = 1.0e9
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    if hasattr(env_cfg.scene, "lazy_sensor_update"):
        env_cfg.scene.lazy_sensor_update = False
    sensor_names, pair_sensor_name = _add_contact_sensors(env_cfg)

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        _, joint_ids = _resolve_ids(env.scene)
        robot = env.scene["robot"]
        ring_joint2_id = joint_ids["finger4_joint2"]
        soft_limits = robot.data.soft_joint_pos_limits[0, ring_joint2_id]
        joint_range = soft_limits[1] - soft_limits[0]
        probe_margin = 0.02
        lower = soft_limits[0] + probe_margin * joint_range
        upper = soft_limits[1] - probe_margin * joint_range
        if env.num_envs == 1:
            joint2_samples = 0.5 * (lower + upper).unsqueeze(0)
        else:
            joint2_samples = torch.linspace(
                float(lower.item()),
                float(upper.item()),
                env.num_envs,
                device=env.device,
            )
        ring_joint2_index = SEARCH_JOINT_NAMES.index("finger4_joint2")
        candidates: list[IkCandidate] = []
        for env_id in range(env.num_envs):
            joint_positions = list(base_candidate.joint_positions)
            joint_positions[ring_joint2_index] = float(
                joint2_samples[env_id].item()
            )
            candidates.append(
                IkCandidate(
                    loss=0.0,
                    joint_positions=joint_positions,
                    geometric_errors=list(base_candidate.geometric_errors),
                    stick2_pos_p=list(base_candidate.stick2_pos_p),
                    pair_center_offset_m=base_candidate.pair_center_offset_m,
                    pair_close_deg=base_candidate.pair_close_deg,
                    search_tracking_errors=[0.0] * len(SEARCH_JOINT_NAMES),
                    min_joint_limit_margin_fraction=probe_margin,
                    source_rank=env_id + 1,
                )
            )

        log.write(
            f"[ring joint2 reach probe] source={candidate_path.name}, "
            f"envs={env.num_envs}, range="
            f"[{lower.item():.4f}, {upper.item():.4f}]rad, "
            "single deterministic sweep; no CEM"
        )
        rows = _run_physics_validation(
            env,
            candidates,
            joint_ids,
            sensor_names,
            pair_sensor_name,
            log,
            emit_rows=False,
        )
        other_labels = (
            "thumb_tip_stick1",
            "index_tip_stick1",
            "middle_tip_stick1",
            "palm_stick2",
            "thumb_mid_stick2",
        )
        for row in rows:
            fractions = row["contact_fraction"]
            row["ring_joint2_rad"] = row["joint_positions"]["finger4_joint2"]
            row["preserved_other_five"] = all(
                fractions[label] >= 0.10 for label in other_labels
            )
            row["ring_contact_reached"] = (
                fractions["ring_tip_stick2"] >= 0.10
            )
            row["full_topology_reached"] = bool(
                row["preserved_other_five"]
                and row["ring_contact_reached"]
            )

        ranked = sorted(
            rows,
            key=lambda row: (
                not row["full_topology_reached"],
                not row["preserved_other_five"],
                -row["contact_fraction"]["ring_tip_stick2"],
                row["ring_surface_error_m"],
                row["stick1_displacement_m"] + row["stick2_displacement_m"],
                row["max_joint_tracking_error_rad"],
            ),
        )
        preserved = [row for row in rows if row["preserved_other_five"]]
        axis_row = min(
            preserved or rows,
            key=lambda row: row["ring_surface_error_m"],
        )
        axial = axis_row["ring_axial_excess_m"]
        cross = axis_row["ring_cross_section_excess_m"]
        if axial > max(0.003, 1.5 * cross):
            miss_axis = "axial_y"
            recommendation = (
                "Ring miss is mainly beyond a stick end; lengthening is geometrically relevant."
            )
        elif cross > max(0.003, 1.5 * axial):
            miss_axis = "cross_section_xz"
            recommendation = (
                "Ring miss is lateral to the shaft; increasing stick length will not solve it."
            )
        else:
            miss_axis = "mixed_or_surface_contact"
            recommendation = (
                "The miss is mixed; use the raw local coordinates before changing stick length."
            )
        if any(row["full_topology_reached"] for row in rows):
            recommendation = (
                "A 6/6 joint2 value exists; use the best row as the ring target."
            )

        summary = {
            "task": args_cli.task,
            "timestamp": RUN_STAMP,
            "output_dir": str(OUTPUT_DIR),
            "source_candidate_file": str(candidate_path),
            "source_run": candidate_data.get("source_run"),
            "environment_count": env.num_envs,
            "swept_joint": "finger4_joint2",
            "joint_range_rad": [float(lower.item()), float(upper.item())],
            "probe_margin_fraction": probe_margin,
            "ring_contact_count": sum(
                bool(row["ring_contact_reached"]) for row in rows
            ),
            "preserved_other_five_count": sum(
                bool(row["preserved_other_five"]) for row in rows
            ),
            "full_topology_count": sum(
                bool(row["full_topology_reached"]) for row in rows
            ),
            "best_candidate": ranked[0],
            "axis_diagnostic_row": axis_row,
            "miss_axis": miss_axis,
            "symmetric_total_length_extension_m": 2.0 * axial,
            "recommendation": recommendation,
            "top_candidates": ranked[:16],
        }
        _write_json(OUTPUT_DIR / "ring_joint2_sweep.json", {"rows": rows})
        _write_csv(OUTPUT_DIR / "ring_joint2_sweep.csv", rows)
        _write_json(OUTPUT_DIR / "ring_reach_summary.json", summary)

        log.write(
            f" result: ring_contact={summary['ring_contact_count']}/{env.num_envs}, "
            f"other5={summary['preserved_other_five_count']}/{env.num_envs}, "
            f"full6={summary['full_topology_count']}/{env.num_envs}"
        )
        for rank, row in enumerate(ranked[:10], start=1):
            local_mm = [
                round(1000.0 * value, 1)
                for value in row["ring_pad_stick2_local_m"]
            ]
            log.write(
                f" rank={rank:02d} q2={row['ring_joint2_rad']:.4f} "
                f"contacts={row['maintained_contact_count']}/6 "
                f"ring={row['contact_fraction']['ring_tip_stick2']:.3f} "
                f"local_xyz_mm={local_mm} "
                f"axial/cross_mm="
                f"{1000.0 * row['ring_axial_excess_m']:.1f}/"
                f"{1000.0 * row['ring_cross_section_excess_m']:.1f} "
                f"disp_mm=({1000.0 * row['stick1_displacement_m']:.1f},"
                f"{1000.0 * row['stick2_displacement_m']:.1f}) "
                f"track={row['max_joint_tracking_error_rad']:.3f}"
            )
        log.write(
            f" axis={miss_axis}; {recommendation}\n"
            f"Saved {OUTPUT_DIR / 'ring_reach_summary.json'}"
        )
    finally:
        if env is not None:
            env.close()


def _run_self_collision_pair_probe(log: ProbeLog) -> None:
    candidate_path = args_cli.candidate_file.expanduser().resolve()
    candidate_data, candidate = _load_candidate_file(candidate_path)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.episode_length_s = 1.0e9
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    sensor_names: dict[str, str] = {}
    filter_paths = [
        f"{{ENV_REGEX_NS}}/Robot/{body_name}" for body_name in ALL_HAND_BODY_NAMES
    ]
    for body_name in ALL_HAND_BODY_NAMES:
        sensor_name = f"ik_self_{body_name}"
        sensor_names[body_name] = sensor_name
        setattr(
            env_cfg.scene,
            sensor_name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
                filter_prim_paths_expr=filter_paths,
                update_period=0.0,
                history_length=1,
                track_pose=False,
                debug_vis=False,
            ),
        )
    if hasattr(env_cfg.scene, "lazy_sensor_update"):
        env_cfg.scene.lazy_sensor_update = False

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        scene = env.scene
        sim = env.sim
        robot = scene["robot"]
        sensors = {
            body_name: scene.sensors[sensor_name]
            for body_name, sensor_name in sensor_names.items()
        }
        expected_filter_count = len(ALL_HAND_BODY_NAMES)
        for body_name, sensor in sensors.items():
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is None or force_matrix.shape != (
                1,
                1,
                expected_filter_count,
                3,
            ):
                actual_shape = None if force_matrix is None else tuple(force_matrix.shape)
                raise RuntimeError(
                    f"Unexpected self-contact matrix for {body_name}: "
                    f"{actual_shape}, expected "
                    f"(1, 1, {expected_filter_count}, 3)."
                )
        device = env.device
        render = not args_cli.headless
        _, joint_ids = _resolve_ids(scene)
        search_ids = [joint_ids[name] for name in SEARCH_JOINT_NAMES]
        proximal_ids = [joint_ids[name] for name in PROXIMAL_SEARCH_JOINT_NAMES]

        default_q = robot.data.default_joint_pos.clone()
        zero_qd = torch.zeros_like(default_q)
        goal_q = default_q.clone()
        goal_q[0, search_ids] = torch.tensor(
            candidate.joint_positions,
            device=device,
        )

        parking_pos = scene.env_origins + torch.tensor(
            [0.50, 0.0, 0.5 * 0.007 + 0.001],
            device=device,
        )
        parking_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=device,
        ).repeat(1, 1)
        _write_object_poses(scene["stick1"], parking_pos, parking_quat)
        _write_object_poses(
            scene["stick2"],
            parking_pos + torch.tensor([0.10, 0.0, 0.0], device=device),
            parking_quat,
        )

        pair_records: dict[tuple[str, str], dict[str, Any]] = {}
        force_threshold = 1.0e-5

        def _sample_pairs(stage: str, step: int) -> None:
            seen_pairs: set[tuple[str, str]] = set()
            for first, sensor in sensors.items():
                force_matrix = sensor.data.force_matrix_w
                if force_matrix is None:
                    raise RuntimeError(
                        f"Self-contact force matrix is unavailable for {first}."
                    )
                magnitudes = torch.linalg.norm(force_matrix[0, 0], dim=-1)
                active_indices = torch.nonzero(
                    magnitudes > force_threshold,
                    as_tuple=False,
                ).flatten()
                for filter_index in active_indices.tolist():
                    second = ALL_HAND_BODY_NAMES[filter_index]
                    if first == second:
                        continue
                    pair = tuple(sorted((first, second)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    force = float(magnitudes[filter_index].item())
                    record = pair_records.setdefault(
                        pair,
                        {
                            "body_pair": list(pair),
                            "first_stage": stage,
                            "first_step": step,
                            "last_stage": stage,
                            "last_step": step,
                            "active_steps": 0,
                            "max_force_N": 0.0,
                            "last_observed_force_N": 0.0,
                        },
                    )
                    record["last_stage"] = stage
                    record["last_step"] = step
                    record["active_steps"] += 1
                    record["max_force_N"] = max(record["max_force_N"], force)
                    record["last_observed_force_N"] = force

        robot.write_joint_state_to_sim(default_q, zero_qd)
        robot.set_joint_position_target(default_q)
        _step_direct(scene, sim, 2, render)
        pair_records.clear()

        for step, (stage, command_q) in enumerate(
            _staged_joint_commands(
                default_q,
                goal_q,
                proximal_ids,
                args_cli.ramp_steps,
                args_cli.stage_settle_steps,
                args_cli.ramp_steps,
                args_cli.stage_settle_steps,
            ),
            start=1,
        ):
            robot.set_joint_position_target(command_q)
            _step_direct(scene, sim, 1, render)
            _sample_pairs(stage, step)

        final_q = robot.data.joint_pos[0, search_ids]
        tracking = torch.abs(final_q - goal_q[0, search_ids])
        rows = sorted(
            pair_records.values(),
            key=lambda row: (
                row["first_step"],
                -row["max_force_N"],
                row["body_pair"],
            ),
        )
        result = {
            "candidate_file": str(candidate_path),
            "source_run": candidate_data["source_run"],
            "source_candidate_rank": candidate.source_rank,
            "self_collision": "ON",
            "sticks": "parked_outside_hand",
            "measurement": "isaaclab_contact_force_matrix",
            "force_threshold_N": force_threshold,
            "sensor_body_names": ALL_HAND_BODY_NAMES,
            "filter_body_names": ALL_HAND_BODY_NAMES,
            "hand_self_pair_count": len(rows),
            "max_joint_tracking_error_rad": float(torch.max(tracking).item()),
            "worst_tracking_joint": SEARCH_JOINT_NAMES[
                int(torch.argmax(tracking).item())
            ],
            "joint_tracking_error_rad": {
                name: float(tracking[index].item())
                for index, name in enumerate(SEARCH_JOINT_NAMES)
            },
            "pairs": rows,
        }
        output_path = OUTPUT_DIR / "self_collision_pairs.json"
        _write_json(output_path, result)
        log.write(
            f"[self-collision pair probe] candidate={candidate_path.name}, "
            f"pairs={len(rows)}, max_track={result['max_joint_tracking_error_rad']:.3f}"
            f"({result['worst_tracking_joint']})"
        )
        for row in rows:
            log.write(
                f" pair={row['body_pair'][0]} <-> {row['body_pair'][1]} "
                f"first={row['first_stage']}:{row['first_step']} "
                f"max_force={row['max_force_N']:.6f}N "
                f"active_steps={row['active_steps']}"
            )
        log.write(f"Saved pair report to {output_path}")
    finally:
        if env is not None:
            env.close()


def _run_self_collision_condition(
    enabled: bool,
    candidates: list[IkCandidate],
    log: ProbeLog,
) -> list[dict[str, Any]]:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=len(candidates))
    env_cfg.episode_length_s = 1.0e9
    env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = enabled
    if hasattr(env_cfg.scene, "lazy_sensor_update"):
        env_cfg.scene.lazy_sensor_update = False

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        scene = env.scene
        sim = env.sim
        robot = scene["robot"]
        device = env.device
        render = not args_cli.headless
        body_ids, joint_ids = _resolve_ids(scene)
        search_ids = [joint_ids[name] for name in SEARCH_JOINT_NAMES]
        proximal_ids = [joint_ids[name] for name in PROXIMAL_SEARCH_JOINT_NAMES]

        default_q = robot.data.default_joint_pos.clone()
        zero_qd = torch.zeros_like(default_q)
        goal_q = default_q.clone()
        stick2_xz = STICK2_POS_P[[0, 2]].to(device).unsqueeze(0).repeat(len(candidates), 1)
        pair_center_offsets = torch.full(
            (len(candidates),),
            PAIR_CENTER_OFFSET,
            device=device,
        )
        pair_close_angles = torch.full(
            (len(candidates),),
            math.radians(PAIR_CLOSE_DEG),
            device=device,
        )
        for env_id, candidate in enumerate(candidates):
            goal_q[env_id, search_ids] = torch.tensor(
                candidate.joint_positions,
                device=device,
            )
            stick2_xz[env_id] = torch.tensor(
                [candidate.stick2_pos_p[0], candidate.stick2_pos_p[2]],
                device=device,
            )
            pair_center_offsets[env_id] = candidate.pair_center_offset_m
            pair_close_angles[env_id] = math.radians(candidate.pair_close_deg)

        parking_pos = scene.env_origins + torch.tensor(
            [0.50, 0.0, 0.5 * 0.007 + 0.001],
            device=device,
        )
        parking_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=device,
        ).repeat(len(candidates), 1)
        _write_object_poses(scene["stick1"], parking_pos, parking_quat)
        _write_object_poses(
            scene["stick2"],
            parking_pos + torch.tensor([0.10, 0.0, 0.0], device=device),
            parking_quat,
        )

        robot.write_joint_state_to_sim(default_q, zero_qd)
        robot.set_joint_position_target(default_q)
        _step_direct(scene, sim, 2, render)
        for _, command_q in _staged_joint_commands(
            default_q,
            goal_q,
            proximal_ids,
            args_cli.ramp_steps,
            args_cli.stage_settle_steps,
            args_cli.ramp_steps,
            args_cli.stage_settle_steps,
        ):
            robot.set_joint_position_target(command_q)
            _step_direct(scene, sim, 1, render)

        q_actual = robot.data.joint_pos[:, search_ids]
        tracking = torch.abs(q_actual - goal_q[:, search_ids])
        palm_pos_w = robot.data.body_pos_w[:, body_ids["palm_link"]]
        palm_quat_w = robot.data.body_quat_w[:, body_ids["palm_link"]]
        stick1_pos_p, stick1_quat_p, stick2_pos_p, stick2_quat_p = _stick_poses_in_palm(
            len(candidates),
            device,
            stick2_xz,
            pair_center_offsets,
            pair_close_angles,
        )
        lower = robot.data.soft_joint_pos_limits[0, search_ids, 0]
        upper = robot.data.soft_joint_pos_limits[0, search_ids, 1]
        q_mid = 0.5 * (lower + upper)
        q_range = torch.clamp(upper - lower, min=1.0e-4)
        _, errors, _ = _geometric_loss(
            robot,
            body_ids,
            palm_pos_w,
            palm_quat_w,
            stick1_pos_p,
            stick1_quat_p,
            stick2_pos_p,
            stick2_quat_p,
            goal_q[:, search_ids],
            q_actual,
            q_mid,
            q_range,
            False,
        )

        rows: list[dict[str, Any]] = []
        condition = "ON" if enabled else "OFF"
        for env_id, candidate in enumerate(candidates):
            joint_tracking = {
                name: float(tracking[env_id, joint_idx].item())
                for joint_idx, name in enumerate(SEARCH_JOINT_NAMES)
            }
            geometry = {
                name: float(errors[env_id, error_idx].item())
                for error_idx, name in enumerate(GEOMETRIC_ERROR_LABELS)
            }
            row = {
                "self_collision": condition,
                "source_candidate_rank": candidate.source_rank,
                "pair_center_offset_m": candidate.pair_center_offset_m,
                "pair_close_deg": candidate.pair_close_deg,
                "max_joint_tracking_error_rad": max(joint_tracking.values()),
                "joint_tracking_error_rad": joint_tracking,
                "geometric_errors_m": geometry,
                "actual_joint_positions": {
                    name: float(q_actual[env_id, joint_idx].item())
                    for joint_idx, name in enumerate(SEARCH_JOINT_NAMES)
                },
            }
            rows.append(row)
        log.write(
            f" self_collision={condition}: "
            f"median_track={torch.median(torch.max(tracking, dim=-1).values).item():.3f}, "
            f"mean_tip_error_mm={1000.0 * errors[:, :4].mean().item():.1f}"
        )
        return rows
    finally:
        if env is not None:
            env.close()


def _run_self_collision_replay(enabled: bool, log: ProbeLog) -> None:
    source_dir, candidates = _load_latest_candidates(args_cli.validate_k)
    condition = "ON" if enabled else "OFF"
    log.write(
        f"[self-collision replay] condition={condition}, "
        f"source={source_dir.name}, candidates={len(candidates)}"
    )
    rows = _run_self_collision_condition(enabled, candidates, log)
    result = {
        "source_run": str(source_dir),
        "self_collision": condition,
        "candidate_count": len(candidates),
        "rows": rows,
    }
    stem = f"self_collision_{condition.lower()}"
    _write_json(OUTPUT_DIR / f"{stem}.json", result)
    _write_csv(OUTPUT_DIR / f"{stem}.csv", rows)
    log.write(f"Saved replay to {OUTPUT_DIR / f'{stem}.json'}")


def _run_cem(
    env,
    body_ids: dict[str, int],
    joint_ids: dict[str, int],
    sensor_names: dict[str, str],
    log: ProbeLog,
) -> list[IkCandidate]:
    scene = env.scene
    sim = env.sim
    robot = scene["robot"]
    stick1 = scene["stick1"]
    stick2 = scene["stick2"]
    device = env.device
    count = env.num_envs
    render = not args_cli.headless

    search_ids = [joint_ids[name] for name in SEARCH_JOINT_NAMES]
    proximal_ids = [joint_ids[name] for name in PROXIMAL_SEARCH_JOINT_NAMES]
    ring_ids = [
        joint_ids[f"finger4_joint{joint}"]
        for joint in range(1, 5)
    ]
    lower = robot.data.soft_joint_pos_limits[0, search_ids, 0]
    upper = robot.data.soft_joint_pos_limits[0, search_ids, 1]
    q_mid = 0.5 * (lower + upper)
    q_range = torch.clamp(upper - lower, min=1.0e-4)
    if not 0.0 <= args_cli.joint_limit_margin < 0.5:
        raise ValueError("--joint-limit-margin must be in [0, 0.5)")
    if args_cli.search_release_steps <= 0:
        raise ValueError("--search-release-steps must be positive")
    sample_lower = lower + args_cli.joint_limit_margin * q_range
    sample_upper = upper - args_cli.joint_limit_margin * q_range
    mean = q_mid.clone()
    mean[SEARCH_JOINT_NAMES.index("finger1_joint1")] = 0.0
    mean[SEARCH_JOINT_NAMES.index("finger1_joint2")] = 0.4
    mean = torch.maximum(torch.minimum(mean, sample_upper), sample_lower)
    std = 0.30 * q_range
    min_std = 0.02 * q_range
    pose_lower = torch.cat(
        (
            STICK2_SEARCH_LOWER_XZ.to(device),
            torch.tensor(
                [PAIR_CENTER_OFFSET_RANGE[0], PAIR_CLOSE_FRACTION_RANGE[0]],
                device=device,
            ),
        )
    )
    pose_upper = torch.cat(
        (
            STICK2_SEARCH_UPPER_XZ.to(device),
            torch.tensor(
                [PAIR_CENTER_OFFSET_RANGE[1], PAIR_CLOSE_FRACTION_RANGE[1]],
                device=device,
            ),
        )
    )
    pose_range = pose_upper - pose_lower
    pose_mean = 0.5 * (pose_lower + pose_upper)
    pose_std = 0.30 * pose_range
    pose_min_std = 0.02 * pose_range

    default_q = robot.data.default_joint_pos.clone()
    zero_qd = torch.zeros_like(default_q)
    parking_pos = scene.env_origins + torch.tensor(
        [0.50, 0.0, 0.5 * 0.007 + 0.001], device=device
    )
    parking_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(count, 1)
    _write_object_poses(stick1, parking_pos, parking_quat)
    _write_object_poses(
        stick2,
        parking_pos + torch.tensor([0.10, 0.0, 0.0], device=device),
        parking_quat,
    )

    archive: list[IkCandidate] = []
    elite_count = max(1, min(count, int(count * args_cli.elite_fraction)))
    physics_iterations = min(
        args_cli.search_iterations,
        max(0, args_cli.physics_aware_iterations),
    )
    physics_start = args_cli.search_iterations - physics_iterations

    log.write(
        f"[1/2] automatic CEM search: envs={count}, iterations={args_cli.search_iterations}, "
        f"physics_final={physics_iterations}, elite={elite_count}, "
        f"joint_margin={args_cli.joint_limit_margin:.0%}"
    )
    if count < 8:
        log.write(
            " WARNING: fewer than 8 envs gives only a weak sequential/random search; "
            "use --num-envs 512 --headless for the intended automatic search."
        )
    for iteration in range(args_cli.search_iterations):
        physics_aware = iteration >= physics_start
        if physics_aware and iteration == physics_start:
            # Earlier teleported-FK scores are not comparable to actuator-aware
            # scores and must not leak into the final candidate list.
            archive.clear()
            # FK converges to a narrow, high-flexion mode. Re-open the
            # distribution before physical scoring so the last ten iterations
            # can actually escape that mode instead of ranking equally blocked
            # samples.
            mean = torch.maximum(
                torch.minimum(0.50 * mean + 0.50 * q_mid, sample_upper),
                sample_lower,
            )
            std = torch.maximum(std, 0.20 * q_range)
            pose_std = torch.maximum(pose_std, 0.20 * pose_range)
            log.write(
                f" switching to actuator/self-collision rollout: "
                f"proximal={args_cli.search_ramp_steps}+{args_cli.search_settle_steps}, "
                f"distal={args_cli.search_ramp_steps}+{args_cli.search_settle_steps}; "
                f"valley={args_cli.search_release_steps}, "
                f"ring={args_cli.search_ramp_steps}, "
                f"hold={args_cli.search_release_steps}; "
                f"distribution re-expanded"
            )

        samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            (count, len(search_ids)), device=device
        )
        samples = torch.maximum(torch.minimum(samples, sample_upper), sample_lower)
        pose_samples = pose_mean.unsqueeze(0) + pose_std.unsqueeze(0) * torch.randn(
            (count, 4), device=device
        )
        pose_samples = torch.maximum(torch.minimum(pose_samples, pose_upper), pose_lower)
        if count > 1:
            samples[0] = mean
            pose_samples[0] = pose_mean
        full_q = default_q.clone()
        full_q[:, search_ids] = samples
        pair_center_offsets = pose_samples[:, 2]
        pair_close_angles = _pair_close_angles(
            pair_center_offsets,
            pose_samples[:, 3],
        )
        stick1_pos_p, stick1_quat_p, stick2_pos_p, stick2_quat_p = _stick_poses_in_palm(
            count,
            device,
            pose_samples[:, :2],
            pair_center_offsets,
            pair_close_angles,
        )

        if physics_aware:
            robot.write_joint_state_to_sim(default_q, zero_qd)
            robot.set_joint_position_target(default_q)
            pre_release_q = full_q.clone()
            pre_release_q[:, ring_ids] = default_q[:, ring_ids]
            palm_pos_w = robot.data.body_pos_w[:, body_ids["palm_link"]]
            palm_quat_w = robot.data.body_quat_w[:, body_ids["palm_link"]]
            stick1_target_pos_w, stick1_target_quat_w = _compose_world_pose(
                palm_pos_w,
                palm_quat_w,
                stick1_pos_p,
                stick1_quat_p,
            )
            stick2_target_pos_w, stick2_target_quat_w = _compose_world_pose(
                palm_pos_w,
                palm_quat_w,
                stick2_pos_p,
                stick2_quat_p,
            )
            for sensor_name in sensor_names.values():
                scene.sensors[sensor_name].reset()
            for _, command_q in _staged_joint_commands(
                default_q,
                pre_release_q,
                proximal_ids,
                args_cli.search_ramp_steps,
                args_cli.search_settle_steps,
                args_cli.search_ramp_steps,
                args_cli.search_settle_steps,
            ):
                robot.set_joint_position_target(command_q)
                # Keep each sampled stick pose fixed while closing so the
                # actuator-aware phase scores real hand-stick contact instead
                # of distance to an imaginary object parked outside the hand.
                _write_object_poses(
                    stick1,
                    stick1_target_pos_w,
                    stick1_target_quat_w,
                )
                _write_object_poses(
                    stick2,
                    stick2_target_pos_w,
                    stick2_target_quat_w,
                )
                _step_direct(scene, sim, 1, render)
            contact_labels = tuple(CONTACT_SPECS)
            # Let the lower stick settle into the palm/thumb valley before
            # commanding the ring. Closing the ring against the pinned pose
            # first leaves it behind when the free stick drops into the valley.
            for _ in range(args_cli.search_release_steps):
                robot.set_joint_position_target(pre_release_q)
                _step_direct(scene, sim, 1, render)
            for step in range(args_cli.search_ramp_steps):
                alpha = (step + 1) / args_cli.search_ramp_steps
                command_q = pre_release_q + alpha * (full_q - pre_release_q)
                robot.set_joint_position_target(command_q)
                _step_direct(scene, sim, 1, render)
            release_contact_steps = torch.zeros(
                (count, len(contact_labels)),
                device=device,
            )
            for _ in range(args_cli.search_release_steps):
                robot.set_joint_position_target(full_q)
                _step_direct(scene, sim, 1, render)
                release_contact_steps += torch.stack(
                    [
                        _sensor_force(scene, sensor_names[label], count, device)
                        >= args_cli.contact_threshold
                        for label in contact_labels
                    ],
                    dim=-1,
                )
            release_contact_fraction = (
                release_contact_steps / args_cli.search_release_steps
            )
            contact_forces = torch.stack(
                [
                    _sensor_force(scene, sensor_names[label], count, device)
                    for label in contact_labels
                ],
                dim=-1,
            )
            (
                released_stick1_pos,
                released_stick1_quat,
                _,
            ) = _root_state_from_physx(stick1, device)
            (
                released_stick2_pos,
                released_stick2_quat,
                _,
            ) = _root_state_from_physx(stick2, device)
            release_displacement = torch.stack(
                (
                    torch.linalg.norm(
                        released_stick1_pos - stick1_target_pos_w,
                        dim=-1,
                    ),
                    torch.linalg.norm(
                        released_stick2_pos - stick2_target_pos_w,
                        dim=-1,
                    ),
                ),
                dim=-1,
            )
        else:
            robot.write_joint_state_to_sim(full_q, zero_qd)
            robot.set_joint_position_target(full_q)
            _step_direct(scene, sim, 1, render)
            contact_forces = torch.zeros((count, len(CONTACT_SPECS)), device=device)
            release_contact_fraction = torch.zeros_like(contact_forces)
            release_displacement = torch.zeros((count, 2), device=device)

        palm_pos_w = robot.data.body_pos_w[:, body_ids["palm_link"]]
        palm_quat_w = robot.data.body_quat_w[:, body_ids["palm_link"]]
        q_actual = robot.data.joint_pos[:, search_ids]
        if physics_aware:
            # The free sticks settle into the valley during mini-release. Score
            # fingertip geometry against those actual poses, not the pinned
            # pre-release targets; otherwise a 10--20 mm lower-stick shift is
            # invisible to the ring objective.
            geometry_stick1_pos_p = quat_apply_inverse(
                palm_quat_w,
                released_stick1_pos - palm_pos_w,
            )
            geometry_stick2_pos_p = quat_apply_inverse(
                palm_quat_w,
                released_stick2_pos - palm_pos_w,
            )
            palm_quat_inv = palm_quat_w.clone()
            palm_quat_inv[:, 1:] = -palm_quat_inv[:, 1:]
            geometry_stick1_quat_p = quat_mul(
                palm_quat_inv,
                released_stick1_quat,
            )
            geometry_stick2_quat_p = quat_mul(
                palm_quat_inv,
                released_stick2_quat,
            )
        else:
            geometry_stick1_pos_p = stick1_pos_p
            geometry_stick1_quat_p = stick1_quat_p
            geometry_stick2_pos_p = stick2_pos_p
            geometry_stick2_quat_p = stick2_quat_p
        loss, errors, diagnostics = _geometric_loss(
            robot,
            body_ids,
            palm_pos_w,
            palm_quat_w,
            geometry_stick1_pos_p,
            geometry_stick1_quat_p,
            geometry_stick2_pos_p,
            geometry_stick2_quat_p,
            samples,
            q_actual,
            q_mid,
            q_range,
            physics_aware,
        )
        if physics_aware:
            contact_labels = tuple(CONTACT_SPECS)
            # Final free-stick contact force complements the duration-based
            # contact fraction without allowing one finger to dominate.
            static_contact_weights = torch.tensor(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                device=device,
            )
            minimum_force = 0.03
            excessive_force = 2.0
            missing_contact = torch.square(
                torch.relu(minimum_force - contact_forces) / minimum_force
            )
            excessive_contact = torch.square(
                torch.relu(contact_forces - excessive_force) / excessive_force
            )
            contact_penalty = torch.sum(
                static_contact_weights
                * (0.50 * missing_contact + 0.05 * excessive_contact),
                dim=-1,
            )
            required_release_fraction = 0.50
            release_contact_deficit = (
                torch.relu(
                    required_release_fraction - release_contact_fraction
                )
                / required_release_fraction
            )
            # CEM is gradient-free, so use contact count as a lexicographic-like
            # topology objective: a 6/6 candidate must outrank a smoother 5/6
            # candidate. The continuous deficit then orders candidates with
            # the same count and still rewards longer contact.
            missing_maintained_count = torch.sum(
                release_contact_fraction < required_release_fraction,
                dim=-1,
            )
            maintained_contact_penalty = (
                5.0 * missing_maintained_count
                + 1.5 * torch.sum(
                    torch.square(release_contact_deficit),
                    dim=-1,
                )
            )
            displacement_penalty = 0.50 * torch.sum(
                torch.square(release_displacement / 0.010),
                dim=-1,
            )
            loss = (
                loss
                + contact_penalty
                + maintained_contact_penalty
                + displacement_penalty
            )
            diagnostics["contact_count"] = torch.sum(
                contact_forces >= minimum_force,
                dim=-1,
            )
            diagnostics["release_contact_count"] = torch.sum(
                release_contact_fraction >= required_release_fraction,
                dim=-1,
            )
            diagnostics["release_displacement"] = release_displacement
            diagnostics["minimum_contact_force"] = torch.min(
                contact_forces,
                dim=-1,
            ).values
        else:
            diagnostics["contact_count"] = torch.zeros(
                count,
                device=device,
                dtype=torch.int64,
            )
            diagnostics["minimum_contact_force"] = torch.zeros(
                count,
                device=device,
            )
            diagnostics["release_contact_count"] = torch.zeros(
                count,
                device=device,
                dtype=torch.int64,
            )
            diagnostics["release_displacement"] = torch.zeros(
                (count, 2),
                device=device,
            )
        elite_idx = torch.topk(loss, elite_count, largest=False).indices
        elite = samples[elite_idx]
        pose_elite = pose_samples[elite_idx]
        new_mean = elite.mean(dim=0)
        new_std = elite.std(dim=0, unbiased=False)
        new_pose_mean = pose_elite.mean(dim=0)
        new_pose_std = pose_elite.std(dim=0, unbiased=False)
        if elite_count > 1:
            mean = 0.25 * mean + 0.75 * new_mean
            std = torch.maximum(0.30 * std + 0.70 * new_std, min_std)
            pose_mean = 0.25 * pose_mean + 0.75 * new_pose_mean
            pose_std = torch.maximum(0.30 * pose_std + 0.70 * new_pose_std, pose_min_std)
        else:
            # A 1-env GUI run cannot perform population CEM. Keep it usable as
            # a slow sequential random search without collapsing variance.
            mean = 0.50 * mean + 0.50 * new_mean
            std = torch.maximum(0.95 * std, min_std)
            pose_mean = 0.50 * pose_mean + 0.50 * new_pose_mean
            pose_std = torch.maximum(0.95 * pose_std, pose_min_std)

        if physics_aware or physics_iterations == 0:
            keep = min(8, count)
            best_idx = torch.topk(loss, keep, largest=False).indices
            tracking_errors = torch.abs(q_actual - samples)
            margin_fraction = torch.minimum(
                (samples - lower) / q_range,
                (upper - samples) / q_range,
            )
            for env_id in best_idx.tolist():
                archive.append(
                    IkCandidate(
                        loss=float(loss[env_id].item()),
                        joint_positions=samples[env_id].detach().cpu().tolist(),
                        geometric_errors=errors[env_id].detach().cpu().tolist(),
                        stick2_pos_p=stick2_pos_p[env_id].detach().cpu().tolist(),
                        pair_center_offset_m=float(pair_center_offsets[env_id].item()),
                        pair_close_deg=math.degrees(
                            float(pair_close_angles[env_id].item())
                        ),
                        search_tracking_errors=tracking_errors[env_id].detach().cpu().tolist(),
                        min_joint_limit_margin_fraction=float(
                            torch.min(margin_fraction[env_id]).item()
                        ),
                        search_contact_forces=(
                            contact_forces[env_id].detach().cpu().tolist()
                            if physics_aware
                            else None
                        ),
                        search_release_contact_fractions=(
                            release_contact_fraction[env_id]
                            .detach()
                            .cpu()
                            .tolist()
                            if physics_aware
                            else None
                        ),
                        search_release_displacements=(
                            release_displacement[env_id]
                            .detach()
                            .cpu()
                            .tolist()
                            if physics_aware
                            else None
                        ),
                    )
                )
        best = int(torch.argmin(loss).item())
        best_errors_mm = 1000.0 * errors[best]
        best_stick2_xz_mm = 1000.0 * pose_samples[best, :2]
        best_pair_offset_mm = 1000.0 * pair_center_offsets[best]
        best_pair_close_deg = math.degrees(float(pair_close_angles[best].item()))
        log.write(
            f" iter={iteration + 1:02d} mode={'PHYS' if physics_aware else 'FK  '} "
            f"loss={loss[best].item():.4f} "
            f"errors_mm={[round(value, 2) for value in best_errors_mm.tolist()]} "
            f"stick2_xz_mm="
            f"{[round(value, 2) for value in best_stick2_xz_mm.tolist()]} "
            f"pair_offset_mm={best_pair_offset_mm.item():.2f} "
            f"pair_close_deg={best_pair_close_deg:.2f} "
            f"track_max={diagnostics['max_tracking_error'][best].item():.3f} "
            f"contacts={int(diagnostics['contact_count'][best].item())}/6 "
            f"release={int(diagnostics['release_contact_count'][best].item())}/6 "
            f"release_disp_mm="
            f"{[round(1000.0 * value, 1) for value in diagnostics['release_displacement'][best].tolist()]}"
        )

    return _select_diverse_candidates(archive, min(args_cli.validate_k, count))


def _run_physics_validation(
    env,
    candidates: list[IkCandidate],
    joint_ids: dict[str, int],
    sensor_names: dict[str, str],
    pair_sensor_name: str,
    log: ProbeLog,
    emit_rows: bool = True,
) -> list[dict[str, Any]]:
    scene = env.scene
    sim = env.sim
    robot = scene["robot"]
    stick1 = scene["stick1"]
    stick2 = scene["stick2"]
    device = env.device
    count = env.num_envs
    validate_count = len(candidates)
    render = not args_cli.headless

    search_ids = [joint_ids[name] for name in SEARCH_JOINT_NAMES]
    proximal_ids = [joint_ids[name] for name in PROXIMAL_SEARCH_JOINT_NAMES]
    ring_ids = [
        joint_ids[f"finger4_joint{joint}"]
        for joint in range(1, 5)
    ]
    default_q = robot.data.default_joint_pos.clone()
    zero_qd = torch.zeros_like(default_q)
    goal_q = default_q.clone()
    for env_id, candidate in enumerate(candidates):
        goal_q[env_id, search_ids] = torch.tensor(candidate.joint_positions, device=device)
    pre_release_q = goal_q.clone()
    pre_release_q[:, ring_ids] = default_q[:, ring_ids]

    palm_cfg = SceneEntityCfg("robot", body_names=["palm_link"])
    palm_cfg.resolve(scene)
    palm_id = palm_cfg.body_ids[0]
    ring_tip_cfg = SceneEntityCfg("robot", body_names=["finger4_tip_link"])
    ring_tip_cfg.resolve(scene)
    ring_tip_id = ring_tip_cfg.body_ids[0]
    robot.write_joint_state_to_sim(default_q, zero_qd)
    robot.set_joint_position_target(default_q)
    _step_direct(scene, sim, 2, render)

    palm_pos_w = robot.data.body_pos_w[:, palm_id]
    palm_quat_w = robot.data.body_quat_w[:, palm_id]
    stick2_xz = STICK2_POS_P[[0, 2]].to(device).unsqueeze(0).repeat(count, 1)
    pair_center_offsets = torch.full(
        (count,),
        PAIR_CENTER_OFFSET,
        device=device,
    )
    pair_close_angles = torch.full(
        (count,),
        math.radians(PAIR_CLOSE_DEG),
        device=device,
    )
    for env_id, candidate in enumerate(candidates):
        stick2_xz[env_id] = torch.tensor(
            [candidate.stick2_pos_p[0], candidate.stick2_pos_p[2]],
            device=device,
        )
        pair_center_offsets[env_id] = candidate.pair_center_offset_m
        pair_close_angles[env_id] = math.radians(candidate.pair_close_deg)
    stick1_pos_p, stick1_quat_p, stick2_pos_p, stick2_quat_p = _stick_poses_in_palm(
        count,
        device,
        stick2_xz,
        pair_center_offsets,
        pair_close_angles,
    )
    stick1_target_pos_w, stick1_target_quat_w = _compose_world_pose(
        palm_pos_w, palm_quat_w, stick1_pos_p, stick1_quat_p
    )
    stick2_target_pos_w, stick2_target_quat_w = _compose_world_pose(
        palm_pos_w, palm_quat_w, stick2_pos_p, stick2_quat_p
    )
    if validate_count < count:
        parking = scene.env_origins[validate_count:] + torch.tensor(
            [0.50, 0.0, 0.5 * 0.007 + 0.001], device=device
        )
        stick1_target_pos_w[validate_count:] = parking
        stick2_target_pos_w[validate_count:] = parking + torch.tensor([0.10, 0.0, 0.0], device=device)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        stick1_target_quat_w[validate_count:] = identity
        stick2_target_quat_w[validate_count:] = identity

    for sensor_name in [*sensor_names.values(), pair_sensor_name]:
        scene.sensors[sensor_name].reset()

    peak_forces = {label: torch.zeros(count, device=device) for label in sensor_names}
    pair_peak = torch.zeros(count, device=device)
    pair_first_y = torch.full((count,), float("nan"), device=device)
    pair_unresolved = torch.zeros(count, dtype=torch.bool, device=device)
    pair_unresolved[:validate_count] = True
    proximal_stage_tracking_abs = torch.full(
        (count, len(search_ids)),
        float("nan"),
        device=device,
    )

    proximal_ramp_steps = args_cli.ramp_steps
    distal_ramp_steps = args_cli.ramp_steps
    log.write(
        f"\n[2/2] physics validation: candidates={validate_count}, "
        f"proximal={proximal_ramp_steps}+{args_cli.stage_settle_steps}, "
        f"distal={distal_ramp_steps}+{args_cli.stage_settle_steps}, "
        f"valley={args_cli.ramp_steps}, ring={args_cli.ramp_steps}, "
        f"release={args_cli.release_steps}"
    )
    for stage, command_q in _staged_joint_commands(
        default_q,
        pre_release_q,
        proximal_ids,
        proximal_ramp_steps,
        args_cli.stage_settle_steps,
        distal_ramp_steps,
        args_cli.stage_settle_steps,
    ):
        robot.set_joint_position_target(command_q)
        _write_object_poses(stick1, stick1_target_pos_w, stick1_target_quat_w)
        _write_object_poses(stick2, stick2_target_pos_w, stick2_target_quat_w)
        _step_direct(scene, sim, 1, render)
        if stage == "proximal":
            proximal_stage_tracking_abs = torch.abs(
                robot.data.joint_pos[:, search_ids] - pre_release_q[:, search_ids]
            )
        for label, sensor_name in sensor_names.items():
            peak_forces[label] = torch.maximum(
                peak_forces[label], _sensor_force(scene, sensor_name, count, device)
            )
        pair_force = _sensor_force(scene, pair_sensor_name, count, device)
        pair_peak = torch.maximum(pair_peak, pair_force)
        _first_pair_contact_y(scene, pair_sensor_name, stick2, pair_unresolved, pair_first_y)

    initial_stick1_pos, _, _ = _root_state_from_physx(stick1, device)
    initial_stick2_pos, _, _ = _root_state_from_physx(stick2, device)
    # Match the search sequence: first let Stick2 settle on the valley anchors,
    # then close the ring against that settled pose while both sticks are free.
    for _ in range(args_cli.ramp_steps):
        robot.set_joint_position_target(pre_release_q)
        _step_direct(scene, sim, 1, render)
        for label, sensor_name in sensor_names.items():
            peak_forces[label] = torch.maximum(
                peak_forces[label], _sensor_force(scene, sensor_name, count, device)
            )
        pair_force = _sensor_force(scene, pair_sensor_name, count, device)
        pair_peak = torch.maximum(pair_peak, pair_force)
        _first_pair_contact_y(
            scene,
            pair_sensor_name,
            stick2,
            pair_unresolved,
            pair_first_y,
        )
    for step in range(args_cli.ramp_steps):
        alpha = (step + 1) / args_cli.ramp_steps
        command_q = pre_release_q + alpha * (goal_q - pre_release_q)
        robot.set_joint_position_target(command_q)
        _step_direct(scene, sim, 1, render)
        for label, sensor_name in sensor_names.items():
            peak_forces[label] = torch.maximum(
                peak_forces[label], _sensor_force(scene, sensor_name, count, device)
            )
        pair_force = _sensor_force(scene, pair_sensor_name, count, device)
        pair_peak = torch.maximum(pair_peak, pair_force)
        _first_pair_contact_y(
            scene,
            pair_sensor_name,
            stick2,
            pair_unresolved,
            pair_first_y,
        )
    max_speed1 = torch.zeros(count, device=device)
    max_speed2 = torch.zeros(count, device=device)
    max_tracking_abs = torch.zeros((count, len(search_ids)), device=device)
    max_computed_torque = torch.zeros((count, len(search_ids)), device=device)
    max_applied_torque = torch.zeros((count, len(search_ids)), device=device)
    torque_clip_steps = torch.zeros(
        (count, len(search_ids)),
        dtype=torch.int32,
        device=device,
    )
    effort_limits = robot.data.joint_effort_limits[:, search_ids]
    contact_steps = {
        label: torch.zeros(count, dtype=torch.int32, device=device) for label in sensor_names
    }

    for _ in range(args_cli.release_steps):
        robot.set_joint_position_target(goal_q)
        _step_direct(scene, sim, 1, render)
        for label, sensor_name in sensor_names.items():
            force = _sensor_force(scene, sensor_name, count, device)
            peak_forces[label] = torch.maximum(peak_forces[label], force)
            contact_steps[label] += force >= args_cli.contact_threshold
        pair_force = _sensor_force(scene, pair_sensor_name, count, device)
        pair_peak = torch.maximum(pair_peak, pair_force)
        _first_pair_contact_y(scene, pair_sensor_name, stick2, pair_unresolved, pair_first_y)
        _, _, velocity1 = _root_state_from_physx(stick1, device)
        _, _, velocity2 = _root_state_from_physx(stick2, device)
        max_speed1 = torch.maximum(max_speed1, torch.linalg.norm(velocity1[:, :3], dim=-1))
        max_speed2 = torch.maximum(max_speed2, torch.linalg.norm(velocity2[:, :3], dim=-1))
        tracking_abs = torch.abs(robot.data.joint_pos[:, search_ids] - goal_q[:, search_ids])
        computed_torque = torch.abs(robot.data.computed_torque[:, search_ids])
        applied_torque = torch.abs(robot.data.applied_torque[:, search_ids])
        max_tracking_abs = torch.maximum(max_tracking_abs, tracking_abs)
        max_computed_torque = torch.maximum(max_computed_torque, computed_torque)
        max_applied_torque = torch.maximum(max_applied_torque, applied_torque)
        torque_clip_steps += computed_torque > effort_limits + 1.0e-6

    final_stick1_pos, _, _ = _root_state_from_physx(stick1, device)
    final_stick2_pos, final_stick2_quat, _ = _root_state_from_physx(stick2, device)
    displacement1 = torch.linalg.norm(final_stick1_pos - initial_stick1_pos, dim=-1)
    displacement2 = torch.linalg.norm(final_stick2_pos - initial_stick2_pos, dim=-1)
    final_tracking_abs = torch.abs(
        robot.data.joint_pos[:, search_ids] - goal_q[:, search_ids]
    )
    tracking_error = torch.max(final_tracking_abs, dim=-1).values
    torque_clip_fraction = torque_clip_steps.float() / args_cli.release_steps
    palm_pos_w = robot.data.body_pos_w[:, palm_id]
    palm_quat_w = robot.data.body_quat_w[:, palm_id]
    ring_pad_offset_b = torch.tensor(
        TIP_PAD_OFFSETS_B["finger4_tip_link"],
        device=device,
    ).unsqueeze(0).repeat(count, 1)
    ring_pad_p = _body_proxy_in_palm(
        robot.data.body_pos_w[:, ring_tip_id],
        robot.data.body_quat_w[:, ring_tip_id],
        ring_pad_offset_b,
        palm_pos_w,
        palm_quat_w,
    )
    final_stick2_pos_p = quat_apply_inverse(
        palm_quat_w,
        final_stick2_pos - palm_pos_w,
    )
    palm_quat_inv = palm_quat_w.clone()
    palm_quat_inv[:, 1:] = -palm_quat_inv[:, 1:]
    final_stick2_quat_p = quat_mul(palm_quat_inv, final_stick2_quat)
    ring_surface_error, ring_pad_stick2 = _box_surface_error(
        ring_pad_p,
        final_stick2_pos_p,
        final_stick2_quat_p,
    )
    half_size = STICK_HALF_SIZE.to(device)
    ring_axial_excess = torch.relu(
        torch.abs(ring_pad_stick2[:, 1]) - half_size[1]
    )
    ring_cross_section_excess = torch.linalg.norm(
        torch.relu(
            torch.abs(ring_pad_stick2[:, [0, 2]])
            - half_size[[0, 2]]
        ),
        dim=-1,
    )

    rows: list[dict[str, Any]] = []
    required = (
        "thumb_tip_stick1",
        "index_tip_stick1",
        "middle_tip_stick1",
        "ring_tip_stick2",
        "palm_stick2",
        "thumb_mid_stick2",
    )
    for env_id, candidate in enumerate(candidates):
        first_y = float(pair_first_y[env_id].item())
        if math.isnan(first_y):
            pair_region = "none"
        elif first_y <= args_cli.tip_region_max_y:
            pair_region = "tip"
        elif first_y >= 0.5 * PAIR_PIVOT_Y:
            pair_region = "handle"
        else:
            pair_region = "middle"
        fractions = {
            label: float(contact_steps[label][env_id].item() / args_cli.release_steps)
            for label in sensor_names
        }
        maintained_contact_count = sum(
            fractions[label] >= 0.10 for label in required
        )
        peak = {label: float(peak_forces[label][env_id].item()) for label in sensor_names}
        all_contacts = all(fractions[label] >= 0.10 for label in required)
        no_drop = bool(
            final_stick1_pos[env_id, 2] >= 0.40 and final_stick2_pos[env_id, 2] >= 0.40
        )
        stable = bool(
            all_contacts
            and no_drop
            and displacement1[env_id] <= args_cli.max_stick_displacement
            and displacement2[env_id] <= args_cli.max_stick_displacement
            and max_speed1[env_id] <= args_cli.max_stick_speed
            and max_speed2[env_id] <= args_cli.max_stick_speed
            and pair_peak[env_id] <= args_cli.max_pair_force
            and pair_region in ("none", "tip")
            and tracking_error[env_id] <= args_cli.max_joint_tracking_error
        )
        joint_tracking = {
            name: float(final_tracking_abs[env_id, joint_idx].item())
            for joint_idx, name in enumerate(SEARCH_JOINT_NAMES)
        }
        proximal_stage_tracking = {
            name: float(proximal_stage_tracking_abs[env_id, joint_idx].item())
            for joint_idx, name in enumerate(SEARCH_JOINT_NAMES)
            if name in PROXIMAL_SEARCH_JOINT_NAMES
        }
        peak_joint_tracking = {
            name: float(max_tracking_abs[env_id, joint_idx].item())
            for joint_idx, name in enumerate(SEARCH_JOINT_NAMES)
        }
        peak_computed = {
            name: float(max_computed_torque[env_id, joint_idx].item())
            for joint_idx, name in enumerate(SEARCH_JOINT_NAMES)
        }
        peak_applied = {
            name: float(max_applied_torque[env_id, joint_idx].item())
            for joint_idx, name in enumerate(SEARCH_JOINT_NAMES)
        }
        clip_fraction = {
            name: float(torque_clip_fraction[env_id, joint_idx].item())
            for joint_idx, name in enumerate(SEARCH_JOINT_NAMES)
        }
        effort_limit = {
            name: float(effort_limits[env_id, joint_idx].item())
            for joint_idx, name in enumerate(SEARCH_JOINT_NAMES)
        }
        worst_tracking_joint = max(joint_tracking, key=joint_tracking.get)
        row = {
            "candidate_rank": env_id + 1,
            "stable": stable,
            "search_loss": candidate.loss,
            "geometric_errors_m": candidate.geometric_errors,
            "stick2_pose_palm": candidate.stick2_pos_p,
            "pair_center_offset_m": candidate.pair_center_offset_m,
            "pair_close_deg": candidate.pair_close_deg,
            "search_tracking_errors_rad": candidate.search_tracking_errors,
            "min_joint_limit_margin_fraction": candidate.min_joint_limit_margin_fraction,
            "joint_positions": {
                name: value
                for name, value in zip(SEARCH_JOINT_NAMES, candidate.joint_positions)
            },
            "contact_fraction": fractions,
            "maintained_contact_count": maintained_contact_count,
            "search_contact_force_N": (
                None
                if candidate.search_contact_forces is None
                else {
                    label: force
                    for label, force in zip(
                        CONTACT_SPECS,
                        candidate.search_contact_forces,
                    )
                }
            ),
            "search_release_contact_fraction": (
                None
                if candidate.search_release_contact_fractions is None
                else {
                    label: fraction
                    for label, fraction in zip(
                        CONTACT_SPECS,
                        candidate.search_release_contact_fractions,
                    )
                }
            ),
            "search_release_displacement_m": (
                candidate.search_release_displacements
            ),
            "peak_contact_force_N": peak,
            "pair_first_contact_region": pair_region,
            "pair_first_contact_local_y_m": None if math.isnan(first_y) else first_y,
            "pair_peak_force_N": float(pair_peak[env_id].item()),
            "stick1_displacement_m": float(displacement1[env_id].item()),
            "stick2_displacement_m": float(displacement2[env_id].item()),
            "stick1_max_speed_m_s": float(max_speed1[env_id].item()),
            "stick2_max_speed_m_s": float(max_speed2[env_id].item()),
            "max_joint_tracking_error_rad": float(tracking_error[env_id].item()),
            "worst_tracking_joint": worst_tracking_joint,
            "proximal_stage_tracking_error_rad": proximal_stage_tracking,
            "joint_tracking_error_rad": joint_tracking,
            "peak_joint_tracking_error_rad": peak_joint_tracking,
            "peak_computed_torque_Nm": peak_computed,
            "peak_applied_torque_Nm": peak_applied,
            "joint_effort_limit_Nm": effort_limit,
            "torque_clip_fraction": clip_fraction,
            "max_torque_clip_fraction": float(
                torch.max(torque_clip_fraction[env_id]).item()
            ),
            "ring_pad_stick2_local_m": ring_pad_stick2[env_id]
            .detach()
            .cpu()
            .tolist(),
            "ring_surface_error_m": float(ring_surface_error[env_id].item()),
            "ring_axial_excess_m": float(ring_axial_excess[env_id].item()),
            "ring_cross_section_excess_m": float(
                ring_cross_section_excess[env_id].item()
            ),
            "stick1_final_height_m": float(final_stick1_pos[env_id, 2].item()),
            "stick2_final_height_m": float(final_stick2_pos[env_id, 2].item()),
        }
        rows.append(row)
        if emit_rows:
            log.write(
                f" rank={env_id + 1:02d} stable={'YES' if stable else 'NO'} "
                f"geom={1000.0 * candidate.pair_center_offset_m:.1f}mm/"
                f"{candidate.pair_close_deg:.2f}deg "
                f"pair={pair_region}/{row['pair_peak_force_N']:.3f}N "
                f"disp_mm=({1000.0 * row['stick1_displacement_m']:.1f},"
                f"{1000.0 * row['stick2_displacement_m']:.1f}) "
                f"speed=({row['stick1_max_speed_m_s']:.3f},{row['stick2_max_speed_m_s']:.3f}) "
                f"track={row['max_joint_tracking_error_rad']:.3f}"
                f"({row['worst_tracking_joint']}) "
                f"contacts={maintained_contact_count}/6 "
                f"torque_clip={row['max_torque_clip_fraction']:.2f}"
            )
    return rows


def main() -> None:
    log = ProbeLog(OUTPUT_DIR / "probe.log")
    env = None
    try:
        torch.manual_seed(args_cli.seed)
        config = {key: _jsonable(value) for key, value in vars(args_cli).items()}
        config["output_dir"] = str(OUTPUT_DIR)
        config["stick2_pose_palm_search"] = {
            "center_position": STICK2_POS_P.tolist(),
            "quaternion_wxyz": STICK2_QUAT_P.tolist(),
            "palm_x_range": [
                float(STICK2_SEARCH_LOWER_XZ[0]),
                float(STICK2_SEARCH_UPPER_XZ[0]),
            ],
            "palm_z_range": [
                float(STICK2_SEARCH_LOWER_XZ[1]),
                float(STICK2_SEARCH_UPPER_XZ[1]),
            ],
            "source_collision_candidates": [36, 37, 51],
        }
        stick1_pos_p, stick1_quat_p, _, _ = _stick_poses_in_palm(1, args_cli.device)
        config["stick1_pose_palm"] = {
            "position": stick1_pos_p[0].detach().cpu().tolist(),
            "quaternion_wxyz": stick1_quat_p[0].detach().cpu().tolist(),
            "center_offset_axis": "palm +z = world +x",
        }
        config["pair_geometry_search"] = {
            "center_offset_m_range": list(PAIR_CENTER_OFFSET_RANGE),
            "close_fraction_range": list(PAIR_CLOSE_FRACTION_RANGE),
            "tip_lever_m": PAIR_TIP_LEVER,
            "minimum_centerline_separation_m": PAIR_MIN_CENTERLINE_SEPARATION,
            "default_center_offset_m": PAIR_CENTER_OFFSET,
            "default_close_deg": PAIR_CLOSE_DEG,
        }
        _write_json(OUTPUT_DIR / "config.json", config)

        log.write(f"[hand_grasp_ik_search] output: {OUTPUT_DIR}")
        log.write("Automatic batched search; keyboard tuning is not used for candidate generation.")

        if args_cli.probe_ring_reach:
            _run_ring_reach_probe(log)
            return

        if args_cli.probe_self_collision_pairs:
            _run_self_collision_pair_probe(log)
            return

        if args_cli.self_collision_condition is not None or args_cli.compare_self_collision:
            condition = args_cli.self_collision_condition or "off"
            if args_cli.compare_self_collision and args_cli.self_collision_condition is None:
                log.write(
                    "NOTE: --compare-self-collision now runs OFF only because Isaac Sim cannot "
                    "reliably recreate the second ManagerBasedEnv in one process."
                )
            _run_self_collision_replay(condition == "on", log)
            return

        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
        env_cfg.episode_length_s = 1.0e9
        search_self_collision = args_cli.search_self_collision == "on"
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = (
            search_self_collision
        )
        log.write(
            f"[full search] self_collision={args_cli.search_self_collision.upper()}"
        )
        if hasattr(env_cfg.scene, "lazy_sensor_update"):
            env_cfg.scene.lazy_sensor_update = False
        sensor_names, pair_sensor_name = _add_contact_sensors(env_cfg)
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        body_ids, joint_ids = _resolve_ids(env.scene)

        candidates = _run_cem(env, body_ids, joint_ids, sensor_names, log)
        candidate_rows = [
            {
                "candidate_rank": rank + 1,
                "search_loss": candidate.loss,
                "geometric_errors_m": candidate.geometric_errors,
                "stick2_pose_palm": candidate.stick2_pos_p,
                "pair_center_offset_m": candidate.pair_center_offset_m,
                "pair_close_deg": candidate.pair_close_deg,
                "search_tracking_errors_rad": candidate.search_tracking_errors,
                "search_contact_force_N": (
                    None
                    if candidate.search_contact_forces is None
                    else {
                        label: force
                        for label, force in zip(
                            CONTACT_SPECS,
                            candidate.search_contact_forces,
                        )
                    }
                ),
                "search_release_contact_fraction": (
                    None
                    if candidate.search_release_contact_fractions is None
                    else {
                        label: fraction
                        for label, fraction in zip(
                            CONTACT_SPECS,
                            candidate.search_release_contact_fractions,
                        )
                    }
                ),
                "search_release_displacement_m": (
                    candidate.search_release_displacements
                ),
                "min_joint_limit_margin_fraction": candidate.min_joint_limit_margin_fraction,
                "joint_positions": {
                    name: value
                    for name, value in zip(SEARCH_JOINT_NAMES, candidate.joint_positions)
                },
            }
            for rank, candidate in enumerate(candidates)
        ]
        _write_csv(OUTPUT_DIR / "ik_candidates.csv", candidate_rows)

        env.reset()
        validation_rows = _run_physics_validation(
            env,
            candidates,
            joint_ids,
            sensor_names,
            pair_sensor_name,
            log,
        )
        _write_csv(OUTPUT_DIR / "validation.csv", validation_rows)
        ranked = sorted(
            validation_rows,
            key=lambda row: (
                not row["stable"],
                -row["maintained_contact_count"],
                row["stick1_displacement_m"] + row["stick2_displacement_m"],
                row["max_joint_tracking_error_rad"],
                row["search_loss"],
            ),
        )
        summary = {
            "task": args_cli.task,
            "timestamp": RUN_STAMP,
            "output_dir": str(OUTPUT_DIR),
            "candidate_count": len(candidates),
            "stable_count": sum(bool(row["stable"]) for row in validation_rows),
            "best_candidate": ranked[0] if ranked else None,
            "ranked_candidates": ranked,
            "next_step": (
                "inspect the best stable q_ref_closed candidate in GUI"
                if any(row["stable"] for row in validation_rows)
                else "inspect contact fractions and refine the automatic target/search; do not hand-tune blindly"
            ),
        }
        _write_json(OUTPUT_DIR / "summary.json", summary)
        log.write(
            f"\nSaved results; stable={summary['stable_count']}/{summary['candidate_count']}"
        )
        log.write(f"Next: {summary['next_step']}")
    except Exception:
        error_text = traceback.format_exc()
        (OUTPUT_DIR / "error.txt").write_text(error_text, encoding="utf-8")
        log.write("\nERROR — traceback saved to error.txt")
        log.write(error_text)
        raise
    finally:
        if env is not None:
            env.close()
        log.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
