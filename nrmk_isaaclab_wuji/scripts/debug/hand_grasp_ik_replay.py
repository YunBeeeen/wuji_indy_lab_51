"""Replay the manually found Wuji grasp as a staged joint-space IK endpoint.

The active ``hand_grasp`` reset supplies the thumb-loaded, open-finger start.
For each remaining finger this script closes joint4 then joint3 first, places
the proximal joints while the fingertip stays curled, and only then releases
the distal joints toward the saved pose.  The thumb is relaxed last.  Both
sticks remain dynamic.  PhysX is stepped directly, so task terminations cannot
auto-reset a completed manual pose.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TARGET = (
    PROJECT_ROOT
    / "logs"
    / "debug"
    / "hand_grasp_keyboard"
    / "2026-07-28_10-57-45"
    / "pose_017.json"
)

parser = argparse.ArgumentParser(
    description="Staged joint-space replay of a saved hand_grasp pose."
)
parser.add_argument("--task", type=str, default="hand_grasp")
parser.add_argument("--target-pose", type=Path, default=DEFAULT_TARGET)
parser.add_argument(
    "--distal-steps",
    type=int,
    default=60,
    help="Physics steps for each joint4/joint3 fingertip-closing ramp.",
)
parser.add_argument(
    "--place-steps",
    type=int,
    default=120,
    help="Physics steps to place joint1/joint2 with the fingertip curled.",
)
parser.add_argument(
    "--release-steps",
    type=int,
    default=120,
    help="Physics steps to reopen joint3/joint4 toward the saved target.",
)
parser.add_argument(
    "--thumb-release-steps",
    type=int,
    default=60,
    help="Physics steps for the final thumb relaxation toward the saved target.",
)
parser.add_argument(
    "--hold-steps",
    type=int,
    default=240,
    help="Physics steps to hold the full nominal target before saving.",
)
parser.add_argument(
    "--print-interval",
    type=int,
    default=30,
    help="Physics-step interval for compact state output; 0 disables it.",
)
parser.add_argument(
    "--exit-after",
    action="store_true",
    help="Close after saving instead of leaving the final GUI frame open.",
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=PROJECT_ROOT / "logs" / "debug" / "hand_grasp_ik_replay",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.distal_steps <= 0:
    parser.error("--distal-steps must be positive.")
if args_cli.place_steps <= 0:
    parser.error("--place-steps must be positive.")
if args_cli.release_steps <= 0:
    parser.error("--release-steps must be positive.")
if args_cli.thumb_release_steps <= 0:
    parser.error("--thumb-release-steps must be positive.")
if args_cli.hold_steps < 0:
    parser.error("--hold-steps must be non-negative.")
if args_cli.print_interval < 0:
    parser.error("--print-interval must be non-negative.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402
from isaac_neuromeka.tasks.manipulation.hand_grasp.hand_grasp_env_cfg import (  # noqa: E402
    CONTACT_SENSOR_NAMES,
    HAND_JOINT_NAMES,
    PREGRASP_JOINT_TARGETS,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _tensor_list(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().tolist()


def _sensor_force(scene, sensor_name: str) -> float:
    force_matrix = scene.sensors[sensor_name].data.force_matrix_w
    if force_matrix is None:
        return 0.0
    force = torch.linalg.vector_norm(force_matrix[0], dim=-1).sum()
    return float(force.item())


def main() -> None:
    target_path = args_cli.target_pose.expanduser().resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"Saved target pose not found: {target_path}")
    target_data = _read_json(target_path)
    if target_data.get("joint_names") != HAND_JOINT_NAMES:
        raise ValueError("Saved target joint order does not match HAND_JOINT_NAMES.")

    run_stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args_cli.output_root.expanduser().resolve() / run_stamp
    output_dir.mkdir(parents=True, exist_ok=False)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    scene = env.scene
    sim = env.sim
    robot = scene["robot"]
    stick1 = scene["stick1"]
    stick2 = scene["stick2"]
    joint_ids, resolved_names = robot.find_joints(
        HAND_JOINT_NAMES,
        preserve_order=True,
    )
    if resolved_names != HAND_JOINT_NAMES:
        raise RuntimeError(
            f"Unexpected hand joint order: {resolved_names}; expected {HAND_JOINT_NAMES}"
        )
    palm_ids, _ = robot.find_bodies(["palm_link"], preserve_order=True)
    palm_id = palm_ids[0]
    soft_limits = robot.data.soft_joint_pos_limits[0, joint_ids]

    command_q = torch.tensor(
        PREGRASP_JOINT_TARGETS,
        device=env.device,
        dtype=robot.data.joint_pos.dtype,
    )
    nominal_q = torch.tensor(
        target_data["joint_target_positions_rad"],
        device=env.device,
        dtype=command_q.dtype,
    )
    nominal_q = torch.clamp(
        nominal_q,
        min=soft_limits[:, 0],
        max=soft_limits[:, 1],
    )

    dt = sim.get_physics_dt()
    global_step = 0

    def state_summary(label: str) -> None:
        actual_q = robot.data.joint_pos[0, joint_ids]
        max_error_deg = torch.rad2deg(torch.max(torch.abs(command_q - actual_q)))
        palm_pos_w = robot.data.body_pos_w[:, palm_id]
        palm_quat_w = robot.data.body_quat_w[:, palm_id]
        stick1_pos_p, _ = subtract_frame_transforms(
            palm_pos_w,
            palm_quat_w,
            stick1.data.root_pos_w,
            stick1.data.root_quat_w,
        )
        stick2_pos_p, _ = subtract_frame_transforms(
            palm_pos_w,
            palm_quat_w,
            stick2.data.root_pos_w,
            stick2.data.root_quat_w,
        )
        forces = {
            name: _sensor_force(scene, sensor_name)
            for name, sensor_name in CONTACT_SENSOR_NAMES.items()
        }
        active = sum(force >= 0.02 for force in forces.values())
        print(
            f"[{label}] step={global_step} "
            f"track_max_deg={float(max_error_deg):.2f} "
            f"S1_mm={[round(1000.0 * value, 1) for value in _tensor_list(stick1_pos_p[0])]} "
            f"S2_mm={[round(1000.0 * value, 1) for value in _tensor_list(stick2_pos_p[0])]} "
            f"sensors={active}/{len(forces)}",
            flush=True,
        )

    def physics_step() -> None:
        nonlocal global_step
        robot.set_joint_position_target(command_q.unsqueeze(0), joint_ids=joint_ids)
        scene.write_data_to_sim()
        sim.step(render=not args_cli.headless)
        scene.update(dt)
        global_step += 1

    def ramp(
        label: str,
        indices: tuple[int, ...],
        goal: torch.Tensor,
        steps: int,
    ) -> None:
        index_tensor = torch.tensor(indices, device=env.device, dtype=torch.long)
        start = command_q[index_tensor].clone()
        print(f"[stage] {label}: {steps} steps", flush=True)
        for step in range(steps):
            alpha = float(step + 1) / float(steps)
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            command_q[index_tensor] = start + smooth * (goal - start)
            physics_step()
            if args_cli.print_interval and global_step % args_cli.print_interval == 0:
                state_summary(label)

    try:
        state_summary("reset")
        for finger, name in enumerate(("index", "middle", "ring", "pinky"), start=1):
            base = 4 * finger
            joint4 = (base + 3,)
            joint3 = (base + 2,)
            proximal = (base, base + 1)
            distal = (base + 2, base + 3)
            ramp(
                f"{name}.joint4_close",
                joint4,
                soft_limits[torch.tensor(joint4, device=env.device), 1],
                args_cli.distal_steps,
            )
            ramp(
                f"{name}.joint3_close",
                joint3,
                soft_limits[torch.tensor(joint3, device=env.device), 1],
                args_cli.distal_steps,
            )
            ramp(
                f"{name}.proximal_place",
                proximal,
                nominal_q[torch.tensor(proximal, device=env.device)],
                args_cli.place_steps,
            )
            ramp(
                f"{name}.distal_release",
                distal,
                nominal_q[torch.tensor(distal, device=env.device)],
                args_cli.release_steps,
            )

        thumb = (0, 1, 2, 3)
        ramp(
            "thumb.final_release",
            thumb,
            nominal_q[torch.tensor(thumb, device=env.device)],
            args_cli.thumb_release_steps,
        )

        print(f"[stage] hold: {args_cli.hold_steps} steps", flush=True)
        for _ in range(args_cli.hold_steps):
            physics_step()
            if args_cli.print_interval and global_step % args_cli.print_interval == 0:
                state_summary("hold")

        actual_q = robot.data.joint_pos[0, joint_ids]
        palm_pos_w = robot.data.body_pos_w[:, palm_id]
        palm_quat_w = robot.data.body_quat_w[:, palm_id]
        stick1_pos_p, stick1_quat_p = subtract_frame_transforms(
            palm_pos_w,
            palm_quat_w,
            stick1.data.root_pos_w,
            stick1.data.root_quat_w,
        )
        stick2_pos_p, stick2_quat_p = subtract_frame_transforms(
            palm_pos_w,
            palm_quat_w,
            stick2.data.root_pos_w,
            stick2.data.root_quat_w,
        )
        forces = {
            name: _sensor_force(scene, sensor_name)
            for name, sensor_name in CONTACT_SENSOR_NAMES.items()
        }
        result_path = output_dir / "result.json"
        _write_json(
            result_path,
            {
                "task": args_cli.task,
                "source_target_pose": str(target_path),
                "physics_steps": global_step,
                "joint_names": HAND_JOINT_NAMES,
                "joint_command_positions_rad": _tensor_list(command_q),
                "joint_actual_positions_rad": _tensor_list(actual_q),
                "max_joint_tracking_error_rad": float(
                    torch.max(torch.abs(command_q - actual_q)).item()
                ),
                "stick1_pose_palm": {
                    "position": _tensor_list(stick1_pos_p[0]),
                    "quaternion_wxyz": _tensor_list(stick1_quat_p[0]),
                },
                "stick2_pose_palm": {
                    "position": _tensor_list(stick2_pos_p[0]),
                    "quaternion_wxyz": _tensor_list(stick2_quat_p[0]),
                },
                "contact_forces_n": forces,
                "active_sensor_count_at_0_02n": sum(
                    force >= 0.02 for force in forces.values()
                ),
            },
        )
        state_summary("final")
        print(f"[saved] {result_path}", flush=True)

        if not args_cli.headless and not args_cli.exit_after:
            print("[done] final frame held; close the window or press Ctrl+C.", flush=True)
            while simulation_app.is_running():
                sim.render()
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
