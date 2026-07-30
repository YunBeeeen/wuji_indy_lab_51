"""Probe stick-stick and Stick2-valley collision feasibility for ``hand_grasp``.

This script does not load a policy and does not modify the training task config on disk.
Contact sensors are attached only to the temporary environment configuration used by
this process.

Each run automatically writes:

* ``probe.log``: human-readable progress and verdicts.
* ``config.json``: command-line arguments.
* ``pair_sweep.csv``: Stick1/Stick2 closing-angle sweep.
* ``valley_scan.csv``: Stick2 contact scan around the thumb-index valley.
* ``summary.json``: machine-readable results and compact verdicts.

Example:

.. code-block:: bash

    python scripts/debug/hand_grasp_collision_probe.py \
        --task hand_grasp \
        --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


parser = argparse.ArgumentParser(description="Probe hand_grasp stick and valley collisions.")
parser.add_argument("--task", type=str, default="hand_grasp", help="Gym task id.")
parser.add_argument(
    "--output-root",
    type=Path,
    default=PROJECT_ROOT / "logs" / "debug" / "hand_grasp_collision_probe",
    help="Root directory under which a timestamped result directory is created.",
)
parser.add_argument(
    "--pair-angles-deg",
    type=float,
    nargs="+",
    default=(0.0, 2.0, 4.0, 5.0, 6.0, 8.0),
    help="Stick1 closing angles about the tail-side pivot.",
)
parser.add_argument("--pair-height", type=float, default=0.85, help="World z used to isolate the stick pair.")
parser.add_argument("--pair-center-offset", type=float, default=0.020, help="Open center spacing in meters.")
parser.add_argument(
    "--pair-pivot-y",
    type=float,
    default=0.060,
    help="Tail-side pivot location from Stick1 center along local +y in meters.",
)
parser.add_argument(
    "--pair-contact-threshold",
    type=float,
    default=0.02,
    help="Normal force in N used to classify load-bearing Stick1-Stick2 contact.",
)
parser.add_argument(
    "--pair-settle-steps",
    type=int,
    default=6,
    help="Physics steps used to capture transient Stick1-Stick2 contact.",
)
parser.add_argument(
    "--tip-region-max-y",
    type=float,
    default=-0.050,
    help="A Stick2-local contact y at or below this value is classified as tip contact.",
)
parser.add_argument(
    "--valley-x-offsets-mm",
    type=float,
    nargs="+",
    default=(0.0, 5.0, 10.0, 15.0),
    help="Palm-frame x offsets from the thumb/index-base midpoint.",
)
parser.add_argument(
    "--valley-z-offsets-mm",
    type=float,
    nargs="+",
    default=(-10.0, -5.0, 0.0, 5.0, 10.0),
    help="Palm-frame z offsets from the thumb/index-base midpoint.",
)
parser.add_argument(
    "--thumb-joint1-values",
    type=float,
    nargs="+",
    default=(0.0, 0.5, 1.0),
    help="Coarse finger1_joint1 values used in the valley scan.",
)
parser.add_argument(
    "--thumb-joint2-values",
    type=float,
    nargs="+",
    default=(0.0, 0.4),
    help="Coarse finger1_joint2 values used in the valley scan.",
)
parser.add_argument(
    "--valley-settle-steps",
    type=int,
    default=12,
    help="Physics steps after releasing each Stick2 valley candidate.",
)
parser.add_argument(
    "--valley-contact-threshold",
    type=float,
    default=0.02,
    help="Normal force in N used to classify a valley-link contact.",
)
parser.add_argument(
    "--valley-max-displacement",
    type=float,
    default=0.002,
    help="Maximum Stick2 displacement in meters for the palm-plus-side valley verdict.",
)
parser.add_argument(
    "--valley-max-speed",
    type=float,
    default=0.05,
    help="Maximum final Stick2 linear speed in m/s for the palm-plus-side valley verdict.",
)
parser.add_argument("--top-k", type=int, default=10, help="Number of top valley candidates printed and saved.")
parser.add_argument("--debug-vis", action="store_true", help="Show contact sensor markers in GUI mode.")
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
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402


STICK_LENGTH = 0.18
STICK_THICKNESS = 0.007
STICK_HALF_LENGTH = 0.5 * STICK_LENGTH

VALLEY_BODIES = {
    "palm": "palm_link",
    "thumb_base": "finger1_link1",
    "index_base": "finger2_link1",
    "thumb_mid": "finger1_link2",
    "index_mid": "finger2_link2",
    "index_distal": "finger2_link3",
}


class ProbeLog:
    """Write the same concise probe messages to stdout and ``probe.log``."""

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


def _quat_z(angle_rad: float, *, device: str) -> torch.Tensor:
    half = 0.5 * angle_rad
    return torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)], device=device)


def _apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat.unsqueeze(0), vector.unsqueeze(0))[0]


def _apply_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return quat_apply_inverse(quat.unsqueeze(0), vector.unsqueeze(0))[0]


def _sensor_measurement(scene, sensor_name: str) -> tuple[float, list[float] | None]:
    data = scene.sensors[sensor_name].data
    force_matrix = data.force_matrix_w
    force = 0.0
    if force_matrix is not None:
        force = torch.linalg.norm(force_matrix[0], dim=-1).sum().item()

    contact_point = None
    if data.contact_pos_w is not None:
        points = data.contact_pos_w[0].reshape(-1, 3)
        points = points[torch.isfinite(points).all(dim=-1)]
        if points.numel() > 0:
            contact_point = points.mean(dim=0).detach().cpu().tolist()
    return float(force), contact_point


def _reset_sensors(scene, names: list[str]) -> None:
    for name in names:
        scene.sensors[name].reset()


def _write_object_pose(obj, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
    obj.write_root_pose_to_sim(torch.cat((pos_w, quat_w)).unsqueeze(0))
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=pos_w.device))


def _root_pose_from_physx(obj) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the current actor pose directly, returning position and wxyz quaternion."""
    pose_xyzw = obj.root_physx_view.get_transforms().clone().to(obj.device)
    quat_wxyz = torch.cat((pose_xyzw[:, 6:7], pose_xyzw[:, 3:6]), dim=-1)
    return pose_xyzw[:, :3], quat_wxyz


def _set_gravity_disabled(obj, disabled: bool) -> None:
    count = obj.root_physx_view.count
    values = torch.full((count, 1), disabled, dtype=torch.bool, device="cpu")
    indices = torch.arange(count, dtype=torch.int32, device="cpu")
    obj.root_physx_view.set_disable_gravities(values, indices)


def _step_direct(scene, sim, steps: int, *, render: bool) -> None:
    dt = sim.get_physics_dt()
    for _ in range(steps):
        scene.write_data_to_sim()
        sim.step(render=render)
        scene.update(dt)


def _add_probe_sensors(env_cfg) -> tuple[str, dict[str, str]]:
    env_cfg.scene.stick1.spawn.activate_contact_sensors = True
    env_cfg.scene.stick2.spawn.activate_contact_sensors = True

    pair_name = "probe_stick1_to_stick2"
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

    valley_sensor_names: dict[str, str] = {}
    for label, body_name in VALLEY_BODIES.items():
        sensor_name = f"probe_{label}_to_stick2"
        setattr(
            env_cfg.scene,
            sensor_name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
                update_period=0.0,
                history_length=1,
                track_pose=True,
                track_contact_points=True,
                max_contact_data_count_per_prim=8,
                debug_vis=args_cli.debug_vis,
            ),
        )
        valley_sensor_names[label] = sensor_name
    return pair_name, valley_sensor_names


def _run_pair_sweep(env, pair_sensor_name: str, log: ProbeLog) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene = env.scene
    sim = env.sim
    stick1 = scene["stick1"]
    stick2 = scene["stick2"]
    device = env.device
    render = not args_cli.headless

    env_origin = scene.env_origins[0]
    stick2_pos = env_origin + torch.tensor([0.0, 0.0, args_cli.pair_height], device=device)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    pivot = stick2_pos + torch.tensor(
        [args_cli.pair_center_offset, args_cli.pair_pivot_y, 0.0], device=device
    )
    center_from_pivot = torch.tensor([0.0, -args_cli.pair_pivot_y, 0.0], device=device)
    lower_tip = stick2_pos + torch.tensor([0.0, -STICK_HALF_LENGTH, 0.0], device=device)

    rows: list[dict[str, Any]] = []
    log.write("\n[1/2] Stick1-Stick2 closing-angle sweep")
    # The pair is intentionally isolated in mid-air. Disable gravity only for
    # this sweep so displacement measures contact response instead of free fall.
    _set_gravity_disabled(stick1, True)
    _set_gravity_disabled(stick2, True)
    log.write(
        " angle(deg)  contact  peakF(N)  contact_region  contact_y(m)  "
        "tip_gap_approx(m)  displacement(m)"
    )

    for angle_deg in args_cli.pair_angles_deg:
        _reset_sensors(scene, [pair_sensor_name])
        closing_quat = _quat_z(-math.radians(angle_deg), device=device)
        stick1_pos = pivot + _apply(closing_quat, center_from_pivot)
        _write_object_pose(stick2, stick2_pos, identity)
        _write_object_pose(stick1, stick1_pos, closing_quat)

        force = 0.0
        point_w = None
        for _ in range(args_cli.pair_settle_steps):
            _step_direct(scene, sim, 1, render=render)
            step_force, step_point_w = _sensor_measurement(scene, pair_sensor_name)
            if step_point_w is not None and (point_w is None or step_force >= force):
                point_w = step_point_w
            force = max(force, step_force)

        contact_local = None
        contact_y = None
        contact_region = "none"
        if point_w is not None:
            point_tensor = torch.tensor(point_w, device=device)
            actual_pos_batch, actual_quat_batch = _root_pose_from_physx(stick2)
            actual_pos = actual_pos_batch[0]
            actual_quat = actual_quat_batch[0]
            contact_local_tensor = _apply_inverse(actual_quat, point_tensor - actual_pos)
            contact_local = contact_local_tensor.detach().cpu().tolist()
            contact_y = float(contact_local[1])
            if contact_y <= args_cli.tip_region_max_y:
                contact_region = "tip"
            elif contact_y >= args_cli.pair_pivot_y * 0.5:
                contact_region = "handle"
            else:
                contact_region = "middle"

        upper_tip = stick1_pos + _apply(
            closing_quat, torch.tensor([0.0, -STICK_HALF_LENGTH, 0.0], device=device)
        )
        tip_center_distance = torch.linalg.norm(upper_tip - lower_tip).item()
        tip_gap_approx = tip_center_distance - STICK_THICKNESS
        actual_stick1_pos, _ = _root_pose_from_physx(stick1)
        displacement = torch.linalg.norm(actual_stick1_pos[0] - stick1_pos).item()
        contact_generated = point_w is not None
        load_bearing_contact = force >= args_cli.pair_contact_threshold
        row = {
            "angle_deg": float(angle_deg),
            "peak_force_N": float(force),
            "contact_generated": bool(contact_generated),
            "load_bearing_contact": bool(load_bearing_contact),
            "contact_region": contact_region,
            "contact_point_world_m": point_w,
            "contact_point_stick2_local_m": contact_local,
            "contact_local_y_m": contact_y,
            "tip_center_distance_m": float(tip_center_distance),
            "approx_tip_surface_gap_m": float(tip_gap_approx),
            "stick1_displacement_after_step_m": float(displacement),
        }
        rows.append(row)
        log.write(
            f" {angle_deg:10.2f}  {'YES' if contact_generated else 'NO ':7s}  "
            f"{force:8.4f}  {contact_region:14s}  "
            f"{contact_y if contact_y is not None else float('nan'):12.5f}  "
            f"{tip_gap_approx:17.5f}  {displacement:15.6f}"
        )

    first_contact = next((row for row in rows if row["contact_generated"]), None)
    baseline_contact = bool(rows and rows[0]["contact_generated"])
    first_contact_at_tip = bool(first_contact and first_contact["contact_region"] == "tip")
    passed = not baseline_contact and first_contact_at_tip
    verdict = {
        "passed": passed,
        "baseline_contact": baseline_contact,
        "first_contact_angle_deg": None if first_contact is None else first_contact["angle_deg"],
        "first_contact_region": None if first_contact is None else first_contact["contact_region"],
        "reason": (
            "first detected contact is in the configured tip region"
            if passed
            else "baseline already contacts, no contact was found, or first contact was not at the tip"
        ),
    }
    log.write(f" pair verdict: {'PASS' if passed else 'CHECK'} — {verdict['reason']}")
    _set_gravity_disabled(stick1, False)
    _set_gravity_disabled(stick2, False)
    return rows, verdict


def _run_valley_scan(
    env,
    valley_sensor_names: dict[str, str],
    log: ProbeLog,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scene = env.scene
    sim = env.sim
    robot = scene["robot"]
    stick1 = scene["stick1"]
    stick2 = scene["stick2"]
    device = env.device
    render = not args_cli.headless

    body_cfg = SceneEntityCfg(
        "robot",
        body_names=list(VALLEY_BODIES.values()),
        preserve_order=True,
    )
    joint_cfg = SceneEntityCfg(
        "robot",
        joint_names=["finger1_joint1", "finger1_joint2"],
        preserve_order=True,
    )
    body_cfg.resolve(scene)
    joint_cfg.resolve(scene)
    body_id = {label: idx for label, idx in zip(VALLEY_BODIES, body_cfg.body_ids)}
    thumb_joint1_id, thumb_joint2_id = joint_cfg.joint_ids

    all_sensor_names = list(valley_sensor_names.values())
    env_origin = scene.env_origins[0]
    # Park the unused stick on the ground, away from the hand. Parking a
    # gravity-enabled rigid body in the air makes it visibly fall every time
    # a candidate resets.
    parking_pos = env_origin + torch.tensor(
        [0.50, 0.0, 0.5 * STICK_THICKNESS + 0.001], device=device
    )
    stick2_parking_pos = parking_pos + torch.tensor([0.10, 0.0, 0.0], device=device)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    default_joint_pos = robot.data.default_joint_pos.clone()
    zero_joint_vel = torch.zeros_like(default_joint_pos)

    rows: list[dict[str, Any]] = []
    candidate_id = 0
    log.write("\n[2/2] Stick2 thumb-index valley scan")
    log.write(
        " id  th_j1  th_j2  off_x(mm)  off_z(mm)  contacts  maxF(N)  disp(m)  speed(m/s)  feasible"
    )

    for thumb_joint1 in args_cli.thumb_joint1_values:
        for thumb_joint2 in args_cli.thumb_joint2_values:
            joint_pos = default_joint_pos.clone()
            joint_pos[:, thumb_joint1_id] = thumb_joint1
            joint_pos[:, thumb_joint2_id] = thumb_joint2
            robot.write_joint_state_to_sim(joint_pos, zero_joint_vel)
            robot.set_joint_position_target(joint_pos)
            _write_object_pose(stick1, parking_pos, identity)
            _write_object_pose(stick2, stick2_parking_pos, identity)
            _step_direct(scene, sim, 2, render=render)

            palm_pos = robot.data.body_pos_w[0, body_id["palm"]]
            palm_quat = robot.data.body_quat_w[0, body_id["palm"]]
            thumb_base_pos = robot.data.body_pos_w[0, body_id["thumb_base"]]
            index_base_pos = robot.data.body_pos_w[0, body_id["index_base"]]
            midpoint_w = 0.5 * (thumb_base_pos + index_base_pos)
            midpoint_p = _apply_inverse(palm_quat, midpoint_w - palm_pos)

            for offset_x_mm in args_cli.valley_x_offsets_mm:
                for offset_z_mm in args_cli.valley_z_offsets_mm:
                    candidate_id += 1
                    _reset_sensors(scene, all_sensor_names)
                    offset_p = torch.tensor(
                        [offset_x_mm * 1.0e-3, 0.0, offset_z_mm * 1.0e-3], device=device
                    )
                    candidate_p = midpoint_p + offset_p
                    candidate_w = palm_pos + _apply(palm_quat, candidate_p)
                    target_pos = candidate_w.clone()
                    target_quat = palm_quat.clone()

                    _write_object_pose(stick1, parking_pos, identity)
                    _write_object_pose(stick2, target_pos, target_quat)

                    max_forces = {label: 0.0 for label in valley_sensor_names}
                    contact_points: dict[str, list[float] | None] = {
                        label: None for label in valley_sensor_names
                    }
                    for _ in range(args_cli.valley_settle_steps):
                        _step_direct(scene, sim, 1, render=render)
                        for label, sensor_name in valley_sensor_names.items():
                            force, point_w = _sensor_measurement(scene, sensor_name)
                            if force > max_forces[label]:
                                max_forces[label] = force
                                contact_points[label] = point_w

                    final_pos, _ = _root_pose_from_physx(stick2)
                    displacement = torch.linalg.norm(final_pos - target_pos).item()
                    final_velocity = stick2.root_physx_view.get_velocities().clone().to(device)
                    final_speed = torch.linalg.norm(final_velocity[0, :3]).item()
                    contacted = [
                        label
                        for label, force in max_forces.items()
                        if force >= args_cli.valley_contact_threshold
                    ]
                    thumb_contacted = [label for label in contacted if label.startswith("thumb_")]
                    index_contacted = [label for label in contacted if label.startswith("index_")]
                    two_sided_contact = bool(thumb_contacted and index_contacted)
                    valley_anchor_contact = bool(
                        "palm" in contacted and (thumb_contacted or index_contacted)
                    )
                    max_force = max(max_forces.values(), default=0.0)
                    feasible = bool(
                        valley_anchor_contact
                        and displacement <= args_cli.valley_max_displacement
                        and final_speed <= args_cli.valley_max_speed
                    )
                    score = (
                        10.0 * float(valley_anchor_contact)
                        + 2.0 * float(two_sided_contact)
                        + 0.5 * len(contacted)
                        - 30.0 * displacement
                        - 0.5 * final_speed
                        - 0.05 * max(0.0, max_force - 5.0)
                    )
                    row = {
                        "candidate_id": candidate_id,
                        "thumb_joint1_rad": float(thumb_joint1),
                        "thumb_joint2_rad": float(thumb_joint2),
                        "offset_palm_x_mm": float(offset_x_mm),
                        "offset_palm_z_mm": float(offset_z_mm),
                        "midpoint_palm_m": midpoint_p.detach().cpu().tolist(),
                        "target_stick2_pos_palm_m": candidate_p.detach().cpu().tolist(),
                        "target_stick2_pos_world_m": target_pos.detach().cpu().tolist(),
                        "contacted_links": contacted,
                        "thumb_contacted_links": thumb_contacted,
                        "index_contacted_links": index_contacted,
                        "two_sided_valley_contact": two_sided_contact,
                        "valley_anchor_contact": valley_anchor_contact,
                        "max_forces_N": max_forces,
                        "contact_points_world_m": contact_points,
                        "max_force_N": float(max_force),
                        "stick2_displacement_m": float(displacement),
                        "stick2_final_linear_speed_m_s": float(final_speed),
                        "feasible": feasible,
                        "ranking_score": float(score),
                    }
                    rows.append(row)
                    log.write(
                        f" {candidate_id:3d}  {thumb_joint1:5.2f}  {thumb_joint2:5.2f}  "
                        f"{offset_x_mm:9.1f}  {offset_z_mm:9.1f}  "
                        f"{'+'.join(contacted) if contacted else 'none':32s}  {max_force:7.3f}  "
                        f"{displacement:7.4f}  {final_speed:10.4f}  "
                        f"{'YES' if feasible else 'NO'}"
                    )

    ranked = sorted(rows, key=lambda row: row["ranking_score"], reverse=True)
    top_candidates = ranked[: max(0, args_cli.top_k)]
    feasible_count = sum(bool(row["feasible"]) for row in rows)
    anchor_contact_count = sum(bool(row["valley_anchor_contact"]) for row in rows)
    two_sided_contact_count = sum(bool(row["two_sided_valley_contact"]) for row in rows)
    verdict = {
        "passed": feasible_count > 0,
        "candidate_count": len(rows),
        "anchor_contact_count": anchor_contact_count,
        "two_sided_contact_count": two_sided_contact_count,
        "feasible_count": feasible_count,
        "top_candidates": top_candidates,
        "reason": (
            "at least one stable palm-plus-side valley candidate was found"
            if feasible_count > 0
            else "no candidate met palm-plus-side contact, displacement, and speed limits"
        ),
        "caveat": (
            "Palm-only contact is not a valley success; simultaneous index contact is diagnostic only. "
            "This screen has no ring support, so "
            "it is not a stable-grasp verdict."
        ),
    }
    log.write(f" valley verdict: {'PASS' if verdict['passed'] else 'CHECK'} — {verdict['reason']}")
    log.write(" top valley candidates:")
    for row in top_candidates:
        log.write(
            f"   id={row['candidate_id']} score={row['ranking_score']:.3f} "
            f"feasible={'YES' if row['feasible'] else 'NO'} "
            f"q_thumb=({row['thumb_joint1_rad']:.2f},{row['thumb_joint2_rad']:.2f}) "
            f"offset_mm=({row['offset_palm_x_mm']:.1f},{row['offset_palm_z_mm']:.1f}) "
            f"contacts={row['contacted_links']} disp={row['stick2_displacement_m']:.4f} "
            f"speed={row['stick2_final_linear_speed_m_s']:.4f}"
        )
    return rows, verdict


def main() -> None:
    log = ProbeLog(OUTPUT_DIR / "probe.log")
    env = None
    try:
        config = {key: _jsonable(value) for key, value in vars(args_cli).items()}
        config["output_dir"] = str(OUTPUT_DIR)
        _write_json(OUTPUT_DIR / "config.json", config)

        log.write(f"[hand_grasp_collision_probe] output: {OUTPUT_DIR}")
        log.write("Probe-only contact sensors; the training task config is not modified on disk.")

        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.episode_length_s = 1.0e9
        if hasattr(env_cfg.scene, "lazy_sensor_update"):
            env_cfg.scene.lazy_sensor_update = False
        pair_sensor_name, valley_sensor_names = _add_probe_sensors(env_cfg)

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()

        pair_rows, pair_verdict = _run_pair_sweep(env, pair_sensor_name, log)
        env.reset()
        valley_rows, valley_verdict = _run_valley_scan(env, valley_sensor_names, log)

        summary = {
            "task": args_cli.task,
            "timestamp": RUN_STAMP,
            "output_dir": str(OUTPUT_DIR),
            "pair_sweep": {
                "verdict": pair_verdict,
                "rows": pair_rows,
            },
            "valley_scan": {
                "verdict": valley_verdict,
                "top_candidates": valley_verdict["top_candidates"],
            },
            "next_step": (
                "inspect CHECK conditions before IK"
                if not pair_verdict["passed"] or not valley_verdict["passed"]
                else "use the best valley candidates as fixed Stick2 poses for fingertip IK"
            ),
        }
        _write_csv(OUTPUT_DIR / "pair_sweep.csv", pair_rows)
        _write_csv(OUTPUT_DIR / "valley_scan.csv", valley_rows)
        _write_json(OUTPUT_DIR / "summary.json", summary)
        log.write("\nSaved:")
        for name in ("probe.log", "config.json", "pair_sweep.csv", "valley_scan.csv", "summary.json"):
            log.write(f"  {OUTPUT_DIR / name}")
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


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
