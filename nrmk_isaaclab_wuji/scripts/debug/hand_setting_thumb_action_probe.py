"""Probe thumb-joint residual authority without training a policy.

Only ``finger1_joint2`` receives a non-zero action.  Parallel environments use
different action magnitudes and repeatedly close then reopen the thumb.  The
script records the real action target/position, Stick2 valley geometry, contact
forces, and object speed so action authority can be separated from reward
design.

Example:

.. code-block:: bash

    python scripts/debug/hand_setting_thumb_action_probe.py \
        --task hand_setting \
        --headless
"""

from __future__ import annotations

import argparse
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

parser = argparse.ArgumentParser(
    description="Sweep only Wuji finger1_joint2 through the active residual action."
)
parser.add_argument("--task", type=str, default="hand_setting")
parser.add_argument(
    "--action-magnitudes",
    type=float,
    nargs="+",
    default=(0.10, 0.25, 0.50, 0.75, 1.00),
    help="Absolute raw actions assigned to parallel environments.",
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=3,
    help="Zero-action policy steps before the first pulse.",
)
parser.add_argument(
    "--half-cycle-steps",
    type=int,
    default=8,
    help="Policy steps for closing and then reopening.",
)
parser.add_argument(
    "--cycles",
    type=int,
    default=3,
    help="Number of close/open pulse cycles.",
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=PROJECT_ROOT / "logs/debug/hand_setting_thumb_action_probe",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.action_magnitudes:
    parser.error("at least one action magnitude is required")
if any(value <= 0.0 for value in args_cli.action_magnitudes):
    parser.error("action magnitudes must be positive")
if args_cli.settle_steps < 0:
    parser.error("settle-steps must be non-negative")
if args_cli.half_cycle_steps <= 0 or args_cli.cycles <= 0:
    parser.error("half-cycle-steps and cycles must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402
from isaac_neuromeka.tasks.manipulation.hand_grasp.hand_grasp_env_cfg import (  # noqa: E402
    FUNCTIONAL_CONTACT_GROUPS,
)
from isaac_neuromeka.tasks.manipulation.hand_grasp.mdp import (  # noqa: E402
    _group_forces,
    _object_pair_speeds_relative_to_palm,
    stick2_valley_geometry,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _sample(
    env,
    action_term,
    thumb_action_index: int,
    raw_action: torch.Tensor,
    valley_params: dict[str, Any],
    support_params: dict[str, Any],
    success_params: dict[str, Any],
):
    def snapshot(value: torch.Tensor) -> torch.Tensor:
        """Detach mutable Isaac buffers before storing a time-series sample."""

        return value.detach().clone()

    shaft_distance, axis_error = stick2_valley_geometry(
        env,
        valley_params["palm_cfg"],
        valley_params["stick2_cfg"],
        valley_params["stick2_reference_position_p"],
        valley_params["stick2_reference_quaternion_p"],
        valley_params["valley_point_offset_o"],
        valley_params["stick_half_length"],
    )
    forces = _group_forces(env, FUNCTIONAL_CONTACT_GROUPS)
    (
        stick1_linear_speed,
        stick2_linear_speed,
        stick1_angular_speed,
        stick2_angular_speed,
    ) = _object_pair_speeds_relative_to_palm(
        env,
        success_params["palm_cfg"],
        success_params["stick1_cfg"],
        success_params["stick2_cfg"],
    )
    robot = env.scene["robot"]
    thumb_joint_ids, _ = robot.find_joints(["finger1_joint2"])
    thumb_joint_id = thumb_joint_ids[0]
    target = action_term.joint_pos_target[:, thumb_action_index]
    position = robot.data.joint_pos[:, thumb_joint_id]
    geometry_valid = (
        (
            shaft_distance
            <= float(valley_params["valley_point_error_limit"])
        )
        & (
            axis_error
            <= float(valley_params["valley_axis_error_limit"])
        )
    )
    support_ready = (
        (
            shaft_distance
            <= float(support_params["support_point_error_limit"])
        )
        & (
            axis_error
            <= float(support_params["support_axis_error_limit"])
        )
    )
    anchors_valid = torch.all(
        forces[:, 3:5]
        >= float(support_params["contact_threshold"]),
        dim=-1,
    )
    in_valley = geometry_valid & anchors_valid
    seated = in_valley & (
        forces[:, 5]
        >= float(support_params["contact_threshold"])
    )
    return {
        "raw_action": snapshot(raw_action),
        "thumb_joint2_target_rad": snapshot(target),
        "thumb_joint2_position_rad": snapshot(position),
        "valley_shaft_distance_m": snapshot(shaft_distance),
        "valley_axis_error_rad": snapshot(axis_error),
        "valley_axis_error_deg": snapshot(torch.rad2deg(axis_error)),
        "valley_geometry_valid": snapshot(geometry_valid),
        "valley_support_ready": snapshot(support_ready),
        "in_valley": snapshot(in_valley),
        "seated": snapshot(seated),
        "functional_forces_N": snapshot(forces),
        "stick1_linear_speed_mps": snapshot(stick1_linear_speed),
        "stick2_linear_speed_mps": snapshot(stick2_linear_speed),
        "stick1_angular_speed_radps": snapshot(stick1_angular_speed),
        "stick2_angular_speed_radps": snapshot(stick2_angular_speed),
    }


def main() -> None:
    num_envs = len(args_cli.action_magnitudes)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
    )
    env_cfg.episode_length_s = 1.0e9
    env_cfg.terminations.stick1_dropped.params["minimum_height"] = -1.0
    env_cfg.terminations.stick2_dropped.params["minimum_height"] = -1.0
    env_cfg.terminations.success.params["hold_steps"] = 1_000_000

    output_dir = (
        args_cli.output_root.expanduser().resolve()
        / datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    env = None
    error: BaseException | None = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        action_term = env.action_manager.get_term("hand_action")
        # SceneEntityCfg body/joint IDs are resolved on the copies owned by
        # manager terms.  Use those runtime params rather than the unresolved
        # module constants (whose body_ids are still slice(None)).
        valley_params = env.reward_manager.get_term_cfg(
            "stick2_reference_pose"
        ).params
        support_params = env.reward_manager.get_term_cfg(
            "valley_anchor_support"
        ).params
        success_params = env.reward_manager.get_term_cfg("success").params
        thumb_action_index = list(action_term._joint_names).index(
            "finger1_joint2"
        )
        if isinstance(action_term._scale, torch.Tensor):
            thumb_action_scale = float(
                action_term._scale[0, thumb_action_index].item()
            )
        else:
            thumb_action_scale = float(action_term._scale)
        amplitudes = torch.as_tensor(
            args_cli.action_magnitudes,
            dtype=torch.float32,
            device=env.device,
        )
        actions = torch.zeros(
            (num_envs, env.action_manager.total_action_dim),
            device=env.device,
        )
        records: list[dict[str, Any]] = []

        def step_with(sign: float, phase: str, phase_step: int) -> None:
            actions.zero_()
            actions[:, thumb_action_index] = sign * amplitudes
            env.step(actions)
            sample = _sample(
                env,
                action_term,
                thumb_action_index,
                actions[:, thumb_action_index],
                valley_params,
                support_params,
                success_params,
            )
            records.append(
                {
                    "global_step": len(records),
                    "phase": phase,
                    "phase_step": phase_step,
                    **sample,
                }
            )

        for step in range(args_cli.settle_steps):
            step_with(0.0, "settle", step)
        for cycle in range(args_cli.cycles):
            for step in range(args_cli.half_cycle_steps):
                step_with(1.0, f"close_{cycle}", step)
            for step in range(args_cli.half_cycle_steps):
                step_with(-1.0, f"open_{cycle}", step)

        summaries = []
        for env_id, amplitude in enumerate(args_cli.action_magnitudes):
            shaft_distances = torch.tensor(
                [
                    record["valley_shaft_distance_m"][env_id].item()
                    for record in records
                ]
            )
            axis_errors = torch.tensor(
                [
                    record["valley_axis_error_rad"][env_id].item()
                    for record in records
                ]
            )
            positions = torch.tensor(
                [
                    record["thumb_joint2_position_rad"][env_id].item()
                    for record in records
                ]
            )
            targets = torch.tensor(
                [
                    record["thumb_joint2_target_rad"][env_id].item()
                    for record in records
                ]
            )
            force_history = torch.stack(
                [
                    record["functional_forces_N"][env_id].detach().cpu()
                    for record in records
                ]
            )
            summaries.append(
                {
                    "env_id": env_id,
                    "action_magnitude": amplitude,
                    "thumb_joint2_position_range_rad": [
                        positions.min().item(),
                        positions.max().item(),
                    ],
                    "thumb_joint2_target_range_rad": [
                        targets.min().item(),
                        targets.max().item(),
                    ],
                    "minimum_valley_shaft_distance_mm":
                        1000.0 * shaft_distances.min().item(),
                    "minimum_valley_axis_error_deg":
                        math.degrees(axis_errors.min().item()),
                    "max_functional_forces_N":
                        force_history.max(dim=0).values,
                    "support_ready_steps": sum(
                        int(record["valley_support_ready"][env_id].item())
                        for record in records
                    ),
                    "geometry_valid_steps": sum(
                        int(record["valley_geometry_valid"][env_id].item())
                        for record in records
                    ),
                    "in_valley_steps": sum(
                        int(record["in_valley"][env_id].item())
                        for record in records
                    ),
                    "seated_steps": sum(
                        int(record["seated"][env_id].item())
                        for record in records
                    ),
                }
            )

        result = {
            "task": args_cli.task,
            "controlled_joint": "finger1_joint2",
            "action_semantics":
                "target = current position + raw_action * action_scale",
            "controlled_joint_action_scale": thumb_action_scale,
            "functional_force_order": [
                "thumb_distal_stick1",
                "index_tip_stick1",
                "middle_tip_stick1",
                "palm_stick2",
                "thumb_mid_stick2",
                "ring_tip_stick2",
            ],
            "settings": {
                "action_magnitudes": args_cli.action_magnitudes,
                "settle_steps": args_cli.settle_steps,
                "half_cycle_steps": args_cli.half_cycle_steps,
                "cycles": args_cli.cycles,
                "geometry_shaft_distance_limit_mm":
                    1000.0
                    * valley_params["valley_point_error_limit"],
                "geometry_axis_limit_deg": math.degrees(
                    valley_params["valley_axis_error_limit"]
                ),
                "support_shaft_distance_limit_mm":
                    1000.0
                    * support_params["support_point_error_limit"],
                "support_axis_limit_deg": math.degrees(
                    support_params["support_axis_error_limit"]
                ),
                "contact_threshold_N":
                    support_params["contact_threshold"],
            },
            "summaries": summaries,
            "records": records,
        }
        result_path = output_dir / "thumb_action_probe.json"
        _write_json(result_path, result)

        print(
            "[thumb action probe] "
            "amp | q-range(rad) | min valley mm/deg | "
            "support/geometry/in-valley/seated"
        )
        for summary in summaries:
            q_range = summary["thumb_joint2_position_range_rad"]
            print(
                f"  {summary['action_magnitude']:.2f} | "
                f"[{q_range[0]:+.4f},{q_range[1]:+.4f}] | "
                f"{summary['minimum_valley_shaft_distance_mm']:.1f}/"
                f"{summary['minimum_valley_axis_error_deg']:.1f} | "
                f"{summary['support_ready_steps']}/"
                f"{summary['geometry_valid_steps']}/"
                f"{summary['in_valley_steps']}/"
                f"{summary['seated_steps']}"
            )
        print(f"Saved {result_path}", flush=True)
    except BaseException as exc:
        # SimulationApp.close() may terminate Kit before Python prints a
        # pending exception.  Persist and print it first so a failed probe
        # never looks like a successful, empty run directory.
        error = exc
        error_text = traceback.format_exc()
        (output_dir / "error.txt").write_text(error_text, encoding="utf-8")
        print("\nERROR — traceback saved to error.txt", file=sys.stderr)
        print(error_text, file=sys.stderr)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
