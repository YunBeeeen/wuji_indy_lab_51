"""Diagnose Wuji joint command routing and physical tracking in ``hand_setting``.

The probe does not load a policy.  Parallel environments each command one
selected joint through the task's real ``hand_action`` term.  The default
selects the five Joint4 joints; ``--all-joints`` selects all 20 hand joints.
Identical positive and negative raw-action pulses make it possible to distinguish:

* no policy command / wrong action mapping;
* a valid target that the articulation does not track;
* effort saturation or collision blocking;
* normal joint motion.

Run the normal scene first, then remove object and self-collision constraints:

.. code-block:: bash

    python scripts/debug/hand_setting_joint4_probe.py \
        --task hand_setting --headless

    python scripts/debug/hand_setting_joint4_probe.py \
        --task hand_setting --headless --park-sticks

    python scripts/debug/hand_setting_joint4_probe.py \
        --task hand_setting --headless --park-sticks --disable-self-collision

If a joint is stuck in all three runs, inspect its USD joint/drive mapping.  If
it moves only with ``--park-sticks``, object contact blocks it.  If it moves
only after additionally disabling self-collision, hand self-collision blocks
it.  This script is intentionally a user-run physics probe; Codex only performs
static checks on it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

JOINT4_NAMES = tuple(f"finger{finger}_joint4" for finger in range(1, 6))
ALL_JOINT_NAMES = tuple(
    f"finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)
CONTACT_LABELS = (
    "thumb_distal_stick1",
    "index_tip_stick1",
    "middle_tip_stick1",
    "palm_stick2",
    "thumb_mid_stick2",
    "ring_tip_stick2",
)

parser = argparse.ArgumentParser(
    description="Pulse each Wuji Joint4 through hand_setting's real residual action term."
)
parser.add_argument("--task", type=str, default="hand_setting")
parser.add_argument(
    "--action-magnitude",
    type=float,
    default=1.0,
    help="Absolute raw action used for both pulse directions (default: 1.0).",
)
parser.add_argument(
    "--all-joints",
    action="store_true",
    help="Probe all 20 finger joints instead of only the five Joint4 joints.",
)
for joint_number in range(1, 4):
    parser.add_argument(
        f"--joint{joint_number}-scale",
        type=float,
        default=None,
        help=f"Diagnostic-only residual-scale override for finger1~5_joint{joint_number}.",
    )
parser.add_argument(
    "--joint4-scale",
    type=float,
    default=None,
    help=(
        "Diagnostic-only override for finger1~5_joint4 residual scale. "
        "Use 0.6 to test the full 0.6 Nm drive authority without editing the task cfg."
    ),
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=5,
    help="Zero-action policy steps before the first pulse.",
)
parser.add_argument(
    "--pulse-steps",
    type=int,
    default=12,
    help="Policy steps for each positive/negative pulse.",
)
parser.add_argument(
    "--hold-steps",
    type=int,
    default=4,
    help="Zero-action policy steps between and after pulses.",
)
parser.add_argument(
    "--park-sticks",
    action="store_true",
    help="Continuously park both sticks away from the hand to remove object contact.",
)
parser.add_argument(
    "--disable-self-collision",
    action="store_true",
    help="Disable hand self-collision before scene creation for an isolation A/B.",
)
parser.add_argument(
    "--motion-threshold",
    type=float,
    default=0.005,
    help="Actual joint range below this value is classified as stuck (rad).",
)
parser.add_argument(
    "--target-threshold",
    type=float,
    default=0.05,
    help="Target range required before a no-motion result is classified (rad).",
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=PROJECT_ROOT / "logs/debug/hand_setting_joint4_probe",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not 0.0 < args_cli.action_magnitude <= 1.0:
    parser.error("--action-magnitude must be in (0, 1].")
for joint_number in range(1, 5):
    scale_override = getattr(args_cli, f"joint{joint_number}_scale")
    if scale_override is not None and scale_override <= 0.0:
        parser.error(f"--joint{joint_number}-scale must be positive.")
if args_cli.settle_steps < 1:
    parser.error("--settle-steps must be at least 1.")
if args_cli.pulse_steps < 1 or args_cli.hold_steps < 1:
    parser.error("--pulse-steps and --hold-steps must be at least 1.")
if args_cli.motion_threshold <= 0.0 or args_cli.target_threshold <= 0.0:
    parser.error("motion and target thresholds must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402
from isaac_neuromeka.tasks.manipulation.hand_grasp.hand_grasp_env_cfg import (  # noqa: E402
    FUNCTIONAL_CONTACT_GROUPS,
)
from isaac_neuromeka.tasks.manipulation.hand_grasp.mdp import _group_forces  # noqa: E402


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


def _park_sticks(env) -> None:
    """Teleport both sticks away before every diagnostic sample."""

    offsets = ((-0.40, -0.20, 0.20), (-0.40, 0.20, 0.20))
    for name, offset in zip(("stick1", "stick2"), offsets):
        stick = env.scene[name]
        state = stick.data.default_root_state.clone()
        state[:, :3] = env.scene.env_origins + torch.tensor(
            offset,
            device=env.device,
            dtype=state.dtype,
        )
        state[:, 7:] = 0.0
        stick.write_root_pose_to_sim(state[:, :7])
        stick.write_root_velocity_to_sim(state[:, 7:])


def _term_scale(action_term, env_ids: torch.Tensor, action_cols: torch.Tensor) -> torch.Tensor:
    scale = action_term._scale
    if isinstance(scale, torch.Tensor):
        if scale.ndim == 0:
            return scale.expand_as(env_ids).to(dtype=torch.float32)
        if scale.ndim == 1:
            return scale[action_cols]
        return scale[env_ids, action_cols]
    return torch.full_like(env_ids, float(scale), dtype=torch.float32)


def _joint_tensor(robot, name: str, env_ids: torch.Tensor, joint_ids: torch.Tensor) -> torch.Tensor:
    value = getattr(robot.data, name, None)
    if value is None:
        return torch.zeros_like(env_ids, dtype=robot.data.joint_pos.dtype)
    return value[env_ids, joint_ids]


def _sample(
    env,
    action_term,
    actions: torch.Tensor,
    env_ids: torch.Tensor,
    action_cols: torch.Tensor,
    joint_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Clone one matched Joint4 sample from each parallel environment."""

    robot = env.scene["robot"]
    target = action_term.joint_pos_target[env_ids, action_cols]
    position = robot.data.joint_pos[env_ids, joint_ids]
    velocity = robot.data.joint_vel[env_ids, joint_ids]
    applied_torque = _joint_tensor(robot, "applied_torque", env_ids, joint_ids)
    computed_torque = _joint_tensor(robot, "computed_torque", env_ids, joint_ids)
    effort_limit = robot.data.joint_effort_limits[env_ids, joint_ids]
    velocity_limit = robot.data.joint_vel_limits[env_ids, joint_ids]
    soft_limits = robot.data.soft_joint_pos_limits[env_ids, joint_ids]
    contact_forces = _group_forces(env, FUNCTIONAL_CONTACT_GROUPS)
    processed = action_term._processed_actions[env_ids, action_cols]

    def clone(value: torch.Tensor) -> torch.Tensor:
        return value.detach().clone()

    return {
        "raw_action": clone(actions[env_ids, action_cols]),
        "processed_increment_rad": clone(processed),
        "target_rad": clone(target),
        "position_rad": clone(position),
        "tracking_error_rad": clone(torch.abs(target - position)),
        "velocity_radps": clone(velocity),
        "applied_torque_Nm": clone(applied_torque),
        "computed_torque_Nm": clone(computed_torque),
        "effort_limit_Nm": clone(effort_limit),
        "velocity_limit_radps": clone(velocity_limit),
        "soft_position_limits_rad": clone(soft_limits),
        "functional_forces_N": clone(contact_forces),
    }


def _phase_values(records: list[dict[str, Any]], phase: str, key: str) -> torch.Tensor:
    return torch.stack(
        [record[key].detach().cpu() for record in records if record["phase"] == phase]
    )


def _classify(
    target_range: float,
    position_range: float,
    max_torque_ratio: float,
) -> str:
    if target_range < args_cli.target_threshold:
        return "ACTION_ROUTING_FAIL"
    if position_range >= args_cli.motion_threshold:
        return "MOVES"
    if max_torque_ratio >= 0.80:
        return "STUCK_TORQUE_OR_COLLISION"
    return "DRIVE_NOT_FOLLOWING"


def _scale_for_joint(configured_scale: float | dict[str, float], joint_number: int) -> float:
    """Resolve one representative finger joint from a scalar/regex scale cfg."""

    if not isinstance(configured_scale, dict):
        return float(configured_scale)
    joint_name = f"finger1_joint{joint_number}"
    matches = [
        float(value)
        for pattern, value in configured_scale.items()
        if re.fullmatch(pattern, joint_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one configured scale for {joint_name}, found {len(matches)}: "
            f"{configured_scale}"
        )
    return matches[0]


def main() -> None:
    selected_joint_names = ALL_JOINT_NAMES if args_cli.all_joints else JOINT4_NAMES
    num_envs = len(selected_joint_names)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=num_envs,
    )
    env_cfg.episode_length_s = 1.0e9
    env_cfg.terminations.stick1_dropped.params["minimum_height"] = -1.0
    env_cfg.terminations.stick2_dropped.params["minimum_height"] = -1.0
    env_cfg.terminations.success.params["hold_steps"] = 1_000_000
    scale_overrides = {
        joint_number: getattr(args_cli, f"joint{joint_number}_scale")
        for joint_number in range(1, 5)
    }
    if any(value is not None for value in scale_overrides.values()):
        configured_scale = env_cfg.actions.hand_action.scale
        resolved_scales = {
            joint_number: _scale_for_joint(configured_scale, joint_number)
            for joint_number in range(1, 5)
        }
        for joint_number, value in scale_overrides.items():
            if value is not None:
                resolved_scales[joint_number] = value
        env_cfg.actions.hand_action.scale = {
            f"finger[1-5]_joint{joint_number}": value
            for joint_number, value in resolved_scales.items()
        }
    if args_cli.disable_self_collision:
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False

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
        if args_cli.park_sticks:
            _park_sticks(env)

        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("hand_action")
        action_names = list(action_term._joint_names)
        action_cols = torch.tensor(
            [action_names.index(name) for name in selected_joint_names],
            device=env.device,
            dtype=torch.long,
        )
        if isinstance(action_term._joint_ids, slice):
            action_joint_ids = torch.arange(
                robot.data.joint_pos.shape[1],
                device=env.device,
                dtype=torch.long,
            )[action_term._joint_ids]
        else:
            action_joint_ids = torch.as_tensor(
                action_term._joint_ids,
                device=env.device,
                dtype=torch.long,
            )
        joint_ids = action_joint_ids[action_cols]
        env_ids = torch.arange(num_envs, device=env.device, dtype=torch.long)
        scales = _term_scale(action_term, env_ids, action_cols)
        actions = torch.zeros(
            (num_envs, env.action_manager.total_action_dim),
            device=env.device,
        )
        records: list[dict[str, Any]] = []

        def step(raw_value: float, phase: str, phase_step: int) -> None:
            actions.zero_()
            actions[env_ids, action_cols] = raw_value
            if args_cli.park_sticks:
                _park_sticks(env)
            env.step(actions)
            if args_cli.park_sticks:
                _park_sticks(env)
            records.append(
                {
                    "global_step": len(records),
                    "phase": phase,
                    "phase_step": phase_step,
                    **_sample(
                        env,
                        action_term,
                        actions,
                        env_ids,
                        action_cols,
                        joint_ids,
                    ),
                }
            )

        for phase_step in range(args_cli.settle_steps):
            step(0.0, "settle", phase_step)
        for phase_step in range(args_cli.pulse_steps):
            step(args_cli.action_magnitude, "positive", phase_step)
        for phase_step in range(args_cli.hold_steps):
            step(0.0, "middle_hold", phase_step)
        for phase_step in range(args_cli.pulse_steps):
            step(-args_cli.action_magnitude, "negative", phase_step)
        for phase_step in range(args_cli.hold_steps):
            step(0.0, "final_hold", phase_step)

        positions = torch.stack(
            [record["position_rad"].detach().cpu() for record in records]
        )
        targets = torch.stack(
            [record["target_rad"].detach().cpu() for record in records]
        )
        tracking_errors = torch.stack(
            [record["tracking_error_rad"].detach().cpu() for record in records]
        )
        torques = torch.stack(
            [record["applied_torque_Nm"].detach().cpu() for record in records]
        )
        effort_limits = torch.stack(
            [record["effort_limit_Nm"].detach().cpu() for record in records]
        )
        velocities = torch.stack(
            [record["velocity_radps"].detach().cpu() for record in records]
        )
        forces = torch.stack(
            [record["functional_forces_N"].detach().cpu() for record in records]
        )
        settle_positions = _phase_values(records, "settle", "position_rad")
        positive_positions = _phase_values(records, "positive", "position_rad")
        negative_positions = _phase_values(records, "negative", "position_rad")

        summaries: list[dict[str, Any]] = []
        for index, joint_name in enumerate(selected_joint_names):
            position_range = float(
                (positions[:, index].max() - positions[:, index].min()).item()
            )
            target_range = float(
                (targets[:, index].max() - targets[:, index].min()).item()
            )
            safe_effort = torch.clamp(effort_limits[:, index].abs(), min=1.0e-8)
            max_torque_ratio = float(
                (torques[:, index].abs() / safe_effort).max().item()
            )
            summaries.append(
                {
                    "env_id": index,
                    "joint_name": joint_name,
                    "action_index": int(action_cols[index].item()),
                    "joint_id": int(joint_ids[index].item()),
                    "action_scale": float(scales[index].item()),
                    "initial_position_rad": float(settle_positions[-1, index].item()),
                    "positive_end_position_rad": float(positive_positions[-1, index].item()),
                    "negative_end_position_rad": float(negative_positions[-1, index].item()),
                    "position_range_rad": position_range,
                    "target_range_rad": target_range,
                    "target_min_max_rad": [
                        float(targets[:, index].min().item()),
                        float(targets[:, index].max().item()),
                    ],
                    "max_tracking_error_rad": float(
                        tracking_errors[:, index].max().item()
                    ),
                    "max_abs_velocity_radps": float(
                        velocities[:, index].abs().max().item()
                    ),
                    "max_abs_applied_torque_Nm": float(
                        torques[:, index].abs().max().item()
                    ),
                    "effort_limit_Nm": float(effort_limits[-1, index].item()),
                    "max_effort_ratio": max_torque_ratio,
                    "max_functional_force_N": float(forces[:, index, :].max().item()),
                    "classification": _classify(
                        target_range,
                        position_range,
                        max_torque_ratio,
                    ),
                }
            )

        result = {
            "task": args_cli.task,
            "action_semantics": "target = current position + raw_action * action_scale",
            "scale_overrides": {
                f"joint{joint_number}": value
                for joint_number, value in scale_overrides.items()
                if value is not None
            },
            "joint_per_environment": list(selected_joint_names),
            "functional_force_order": list(CONTACT_LABELS),
            "settings": {
                "action_magnitude": args_cli.action_magnitude,
                "settle_steps": args_cli.settle_steps,
                "pulse_steps": args_cli.pulse_steps,
                "hold_steps": args_cli.hold_steps,
                "park_sticks": args_cli.park_sticks,
                "disable_self_collision": args_cli.disable_self_collision,
                "motion_threshold_rad": args_cli.motion_threshold,
                "target_threshold_rad": args_cli.target_threshold,
            },
            "summaries": summaries,
            "records": records,
        }
        result_path = output_dir / (
            "all_joint_probe.json" if args_cli.all_joints else "joint4_probe.json"
        )
        _write_json(result_path, result)

        print(
            f"[hand_setting {'All-joint' if args_cli.all_joints else 'Joint4'} probe] "
            f"park_sticks={args_cli.park_sticks} "
            f"self_collision={'OFF' if args_cli.disable_self_collision else 'ON'}"
        )
        print(
            " joint                 scale     q0      q+      q-   q_range "
            "tgt_range max_err torque% maxF(N) verdict"
        )
        for summary in summaries:
            print(
                f" {summary['joint_name']:<21}"
                f"{summary['action_scale']:>5.2f}"
                f"{summary['initial_position_rad']:>8.3f}"
                f"{summary['positive_end_position_rad']:>8.3f}"
                f"{summary['negative_end_position_rad']:>8.3f}"
                f"{summary['position_range_rad']:>9.3f}"
                f"{summary['target_range_rad']:>10.3f}"
                f"{summary['max_tracking_error_rad']:>8.3f}"
                f"{100.0 * summary['max_effort_ratio']:>8.1f}"
                f"{summary['max_functional_force_N']:>8.3f} "
                f"{summary['classification']}"
            )
        print(f"Saved {result_path}", flush=True)
    except BaseException as exc:
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
