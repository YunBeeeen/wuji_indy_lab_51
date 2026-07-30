"""Interactively tune all 20 Wuji finger joints in the ``hand_grasp`` scene.

This is a GUI-only, policy-free pose editor.  It starts from the open Wuji
joint state by default, keeps an absolute position target for every finger
joint, and lets the user move one selected joint at a time.  The two sticks
remain dynamic so the resulting in-hand motion can be inspected in PhysX.

Key map:

* ``1`` .. ``5``: select finger 1 .. 5
* ``Q`` / ``W`` / ``E`` / ``R``: select joint 1 / 2 / 3 / 4
* ``Left`` / ``A``: decrease the selected joint target
* ``Right`` / ``D``: increase the selected joint target
* ``Z`` / ``X``: halve / double the angular step
* ``O``: command the manual open pose (thumb joint2 fully extended)
* ``P``: command the active manually staged pre-grasp target
* ``Backspace``: reset the scene and return to the requested start pose
* ``Space``: pause/resume physics
* ``V``: print all target and measured joint angles
* ``T``: toggle live joint-angle and palm-local stick-pose output
* ``S``: save joints and both stick poses to a timestamped JSON file
* ``Esc``: save once and quit

By default both sticks are spawned 2 cm farther along the fingers than the
training scene's old placement, and yellow Stick1 is held fixed at that spawn
pose.  Green Stick2 remains dynamic and can be pushed into the valley.
With ``--start-pose pregrasp``, the active training reset is preserved exactly;
use ``--stick1-mode dynamic`` to inspect both freely moving sticks.
Use ``--load-pose path/to/pose.json`` to continue editing a saved state without
recreating its contact path by hand.

Example:

.. code-block:: bash

    python scripts/debug/hand_grasp_keyboard.py \
        --task hand_grasp
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description="Keyboard Wuji joint pose editor for hand_grasp.")
parser.add_argument("--task", type=str, default="hand_grasp")
parser.add_argument(
    "--load-pose",
    type=Path,
    default=None,
    help="Restore joint actual/target and both palm-local stick poses from JSON.",
)
parser.add_argument(
    "--start-pose",
    choices=("open", "pregrasp"),
    default="open",
    help="Joint/object reset state. 'open' disables the training pre-grasp reset.",
)
parser.add_argument(
    "--joint-step-deg",
    type=float,
    default=2.0,
    help="Initial target increment per key press in degrees.",
)
parser.add_argument(
    "--stick1-mode",
    choices=("fixed", "dynamic", "park"),
    default="fixed",
    help="Keep yellow Stick1 fixed at spawn, leave it dynamic, or park it away.",
)
parser.add_argument(
    "--park-stick1",
    action="store_true",
    help="Deprecated alias for --stick1-mode park.",
)
parser.add_argument(
    "--stick-forward-offset",
    type=float,
    default=0.020,
    help="Move both sticks along world +x / palm +z (finger direction), in meters.",
)
parser.add_argument(
    "--stick-height-offset",
    type=float,
    default=0.0,
    help="Move both sticks along world +z (above the palm), in meters.",
)
parser.add_argument(
    "--stick-print-hz",
    type=float,
    default=5.0,
    help="Terminal update rate for live palm-local stick positions; 0 disables it.",
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=PROJECT_ROOT / "logs" / "debug" / "hand_grasp_keyboard",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.headless:
    parser.error("hand_grasp_keyboard.py requires a GUI; do not pass --headless.")
if args_cli.joint_step_deg <= 0.0:
    parser.error("--joint-step-deg must be positive.")
if args_cli.stick_print_hz < 0.0:
    parser.error("--stick-print-hz must be non-negative.")
if args_cli.park_stick1:
    args_cli.stick1_mode = "park"
if args_cli.load_pose is not None:
    args_cli.load_pose = args_cli.load_pose.expanduser().resolve()
    if not args_cli.load_pose.is_file():
        parser.error(f"--load-pose file not found: {args_cli.load_pose}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from pynput import keyboard  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402
from isaac_neuromeka.tasks.manipulation.hand_grasp.hand_grasp_env_cfg import (  # noqa: E402
    HAND_JOINT_NAMES,
    PREGRASP_JOINT_TARGETS,
)


RUN_STAMP = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = args_cli.output_root.expanduser().resolve() / RUN_STAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

FINGER_KEYS = {str(index): index - 1 for index in range(1, 6)}
JOINT_KEYS = {"q": 0, "w": 1, "e": 2, "r": 3}


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


class KeyboardEvents:
    """Forward pynput press events to the simulation thread."""

    def __init__(self) -> None:
        self.events: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self.listener = keyboard.Listener(on_press=self.events.put)
        self.listener.start()

    def drain(self) -> list[Any]:
        result: list[Any] = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except queue.Empty:
                return result

    def stop(self) -> None:
        self.listener.stop()


def _key_char(key: Any) -> str | None:
    try:
        return key.char.lower()
    except (AttributeError, TypeError):
        return None


def _park_stick1(scene) -> None:
    stick1 = scene["stick1"]
    root_state = stick1.data.default_root_state.clone()
    root_state[:, :3] = scene.env_origins + torch.tensor(
        (-0.35, 0.0, 0.01),
        device=scene.env_origins.device,
        dtype=scene.env_origins.dtype,
    )
    root_state[:, 7:] = 0.0
    stick1.write_root_pose_to_sim(root_state[:, :7])
    stick1.write_root_velocity_to_sim(root_state[:, 7:])


def _spawn_stick_with_offset(stick) -> torch.Tensor:
    """Reset one stick to its configured spawn pose plus the manual tuning offset."""
    root_state = stick.data.default_root_state.clone()
    root_state[:, 0] += args_cli.stick_forward_offset
    root_state[:, 2] += args_cli.stick_height_offset
    root_state[:, 7:] = 0.0
    stick.write_root_pose_to_sim(root_state[:, :7])
    stick.write_root_velocity_to_sim(root_state[:, 7:])
    return root_state


def _hold_stick_at_state(stick, root_state: torch.Tensor) -> None:
    """Keep a dynamic rigid object fixed without changing the task USD."""
    stick.write_root_pose_to_sim(root_state[:, :7])
    stick.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))


def _stick_pose_in_palm(robot, palm_id: int, stick) -> tuple[torch.Tensor, torch.Tensor]:
    palm_pos_w = robot.data.body_pos_w[:, palm_id]
    palm_quat_w = robot.data.body_quat_w[:, palm_id]
    return subtract_frame_transforms(
        palm_pos_w,
        palm_quat_w,
        stick.data.root_pos_w,
        stick.data.root_quat_w,
    )


def _print_controls() -> None:
    print(
        "\nKeyboard controls\n"
        "  1..5       select finger\n"
        "  Q/W/E/R    select joint 1/2/3/4\n"
        "  Left/A     decrease selected target\n"
        "  Right/D    increase selected target\n"
        "  Z/X        halve/double step\n"
        "  O/P        manual open pose / old pre-grasp pose\n"
        "  Backspace  reset scene\n"
        "  Space      pause/resume physics\n"
        "  V          print all joint angles\n"
        "  T          toggle live joint angles + stick positions\n"
        "  S          save JSON\n"
        "  Esc        save and quit\n",
        flush=True,
    )


def main() -> None:
    loaded_pose = (
        None
        if args_cli.load_pose is None
        else json.loads(args_cli.load_pose.read_text(encoding="utf-8"))
    )
    if loaded_pose is not None and loaded_pose.get("joint_names") != HAND_JOINT_NAMES:
        raise ValueError("Loaded pose joint order does not match HAND_JOINT_NAMES.")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.episode_length_s = 1.0e9
    if args_cli.start_pose == "open":
        env_cfg.events.reset_pregrasp = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    scene = env.scene
    sim = env.sim
    robot = scene["robot"]
    stick1 = scene["stick1"]
    stick2 = scene["stick2"]

    joint_ids, resolved_names = robot.find_joints(HAND_JOINT_NAMES, preserve_order=True)
    if resolved_names != HAND_JOINT_NAMES:
        raise RuntimeError(
            f"Unexpected hand joint order: {resolved_names}; expected {HAND_JOINT_NAMES}"
        )
    palm_ids, _ = robot.find_bodies(["palm_link"], preserve_order=True)
    palm_id = palm_ids[0]
    soft_limits = robot.data.soft_joint_pos_limits[0, joint_ids].clone()
    default_targets = robot.data.default_joint_pos[0, joint_ids].clone()
    manual_open_targets = default_targets.clone()
    thumb_joint2_index = HAND_JOINT_NAMES.index("finger1_joint2")
    manual_open_targets[thumb_joint2_index] = soft_limits[thumb_joint2_index, 0]
    pregrasp_targets = torch.tensor(
        PREGRASP_JOINT_TARGETS,
        device=env.device,
        dtype=default_targets.dtype,
    )

    selected_finger = 0
    selected_joint = 0
    step_rad = math.radians(args_cli.joint_step_deg)
    paused = False
    quit_requested = False
    live_stick_output = args_cli.stick_print_hz > 0.0
    stick_print_interval = (
        1.0 / args_cli.stick_print_hz if args_cli.stick_print_hz > 0.0 else 0.2
    )
    next_stick_print_time = time.monotonic()
    save_index = 0
    keyboard_events = KeyboardEvents()
    fixed_stick1_state: torch.Tensor | None = None

    def apply_requested_start_pose() -> torch.Tensor:
        nonlocal paused, fixed_stick1_state
        env.reset()
        paused = False
        if loaded_pose is not None:
            target = torch.tensor(
                loaded_pose["joint_target_positions_rad"],
                device=env.device,
                dtype=default_targets.dtype,
            )
            actual = torch.tensor(
                loaded_pose["joint_actual_positions_rad"],
                device=env.device,
                dtype=default_targets.dtype,
            )
            target = torch.clamp(target, min=soft_limits[:, 0], max=soft_limits[:, 1])
            actual = torch.clamp(actual, min=soft_limits[:, 0], max=soft_limits[:, 1])
            zero_joint_velocity = torch.zeros_like(actual).unsqueeze(0)
            robot.write_joint_state_to_sim(
                actual.unsqueeze(0),
                zero_joint_velocity,
                joint_ids=joint_ids,
            )
            robot.set_joint_position_target(target.unsqueeze(0), joint_ids=joint_ids)

            palm_pos_w = robot.data.body_pos_w[:, palm_id]
            palm_quat_w = robot.data.body_quat_w[:, palm_id]
            zero_object_velocity = torch.zeros(
                (1, 6),
                device=env.device,
                dtype=palm_pos_w.dtype,
            )

            def restore_stick(stick, key: str) -> torch.Tensor:
                saved = loaded_pose[key]
                position_p = torch.tensor(
                    saved["position"],
                    device=env.device,
                    dtype=palm_pos_w.dtype,
                ).unsqueeze(0)
                quaternion_p = torch.tensor(
                    saved["quaternion_wxyz"],
                    device=env.device,
                    dtype=palm_pos_w.dtype,
                ).unsqueeze(0)
                pose_w = torch.cat(
                    (
                        palm_pos_w + quat_apply(palm_quat_w, position_p),
                        quat_mul(palm_quat_w, quaternion_p),
                    ),
                    dim=-1,
                )
                stick.write_root_pose_to_sim(pose_w)
                stick.write_root_velocity_to_sim(zero_object_velocity)
                return torch.cat((pose_w, zero_object_velocity), dim=-1)

            restore_stick(stick2, "stick2_pose_palm")
            if args_cli.stick1_mode == "park":
                _park_stick1(scene)
                fixed_stick1_state = None
            else:
                saved_stick1_state = restore_stick(stick1, "stick1_pose_palm")
                fixed_stick1_state = (
                    saved_stick1_state
                    if args_cli.stick1_mode == "fixed"
                    else None
                )
            print(f"[loaded] {args_cli.load_pose}", flush=True)
            return target

        if args_cli.start_pose == "pregrasp":
            target = pregrasp_targets.clone()
            robot.set_joint_position_target(target.unsqueeze(0), joint_ids=joint_ids)
            if args_cli.stick1_mode == "park":
                _park_stick1(scene)
                fixed_stick1_state = None
            else:
                fixed_stick1_state = (
                    stick1.data.root_state_w.clone()
                    if args_cli.stick1_mode == "fixed"
                    else None
                )
            return target

        target = manual_open_targets.clone()
        robot.write_joint_state_to_sim(
            target.unsqueeze(0),
            torch.zeros_like(target).unsqueeze(0),
            joint_ids=joint_ids,
        )
        robot.set_joint_position_target(target.unsqueeze(0), joint_ids=joint_ids)
        _spawn_stick_with_offset(stick2)
        if args_cli.stick1_mode == "park":
            _park_stick1(scene)
            fixed_stick1_state = None
        else:
            stick1_spawn_state = _spawn_stick_with_offset(stick1)
            fixed_stick1_state = (
                stick1_spawn_state if args_cli.stick1_mode == "fixed" else None
            )
        return target

    target_q = apply_requested_start_pose()

    def selected_index() -> int:
        return 4 * selected_finger + selected_joint

    def print_selected(prefix: str = "") -> None:
        index = selected_index()
        actual = float(robot.data.joint_pos[0, joint_ids[index]].item())
        print(
            f"{prefix}{HAND_JOINT_NAMES[index]} "
            f"target={float(target_q[index]):+.4f} rad "
            f"actual={actual:+.4f} rad "
            f"step={math.degrees(step_rad):.3f} deg",
            flush=True,
        )

    def print_all() -> None:
        actual = robot.data.joint_pos[0, joint_ids]
        print("\nCurrent Wuji joints (target / actual rad)", flush=True)
        for finger in range(5):
            values = []
            for joint in range(4):
                index = 4 * finger + joint
                values.append(
                    f"j{joint + 1}={float(target_q[index]):+.3f}/"
                    f"{float(actual[index]):+.3f}"
                )
            print(f"  finger{finger + 1}: " + "  ".join(values), flush=True)

    def print_stick_positions(prefix: str = "[sticks]") -> None:
        stick1_pos_p, _ = _stick_pose_in_palm(robot, palm_id, stick1)
        stick2_pos_p, _ = _stick_pose_in_palm(robot, palm_id, stick2)
        position1_mm = 1000.0 * stick1_pos_p[0]
        position2_mm = 1000.0 * stick2_pos_p[0]
        delta_mm = position1_mm - position2_mm
        transverse_gap_mm = torch.linalg.norm(delta_mm[[0, 2]])
        print(
            f"{prefix} palm_xyz_mm "
            f"S1=({float(position1_mm[0]):+.1f},"
            f"{float(position1_mm[1]):+.1f},"
            f"{float(position1_mm[2]):+.1f}) "
            f"S2=({float(position2_mm[0]):+.1f},"
            f"{float(position2_mm[1]):+.1f},"
            f"{float(position2_mm[2]):+.1f}) "
            f"delta=({float(delta_mm[0]):+.1f},"
            f"{float(delta_mm[1]):+.1f},"
            f"{float(delta_mm[2]):+.1f}) "
            f"transverse_gap={float(transverse_gap_mm):.1f}",
            flush=True,
        )

    def print_joint_positions(prefix: str = "[joints]") -> None:
        actual = robot.data.joint_pos[0, joint_ids]
        fingers = []
        for finger in range(5):
            start = 4 * finger
            values = ",".join(
                f"{float(actual[start + joint]):+.3f}" for joint in range(4)
            )
            fingers.append(f"f{finger + 1}=({values})")
        print(f"{prefix} actual_rad " + " ".join(fingers), flush=True)

    def save_pose(reason: str) -> Path:
        nonlocal save_index
        save_index += 1
        stick1_pos_p, stick1_quat_p = _stick_pose_in_palm(robot, palm_id, stick1)
        stick2_pos_p, stick2_quat_p = _stick_pose_in_palm(robot, palm_id, stick2)
        actual_q = robot.data.joint_pos[0, joint_ids]
        output_path = OUTPUT_DIR / f"pose_{save_index:03d}.json"
        _write_json(
            output_path,
            {
                "reason": reason,
                "task": args_cli.task,
                "start_pose": args_cli.start_pose,
                "loaded_pose": args_cli.load_pose,
                "stick1_mode": args_cli.stick1_mode,
                "stick_forward_offset_m": args_cli.stick_forward_offset,
                "stick_height_offset_m": args_cli.stick_height_offset,
                "selected_joint": HAND_JOINT_NAMES[selected_index()],
                "joint_step_deg": math.degrees(step_rad),
                "joint_names": HAND_JOINT_NAMES,
                "joint_target_positions_rad": target_q,
                "joint_actual_positions_rad": actual_q,
                "stick1_pose_palm": {
                    "position": stick1_pos_p[0],
                    "quaternion_wxyz": stick1_quat_p[0],
                },
                "stick2_pose_palm": {
                    "position": stick2_pos_p[0],
                    "quaternion_wxyz": stick2_quat_p[0],
                },
            },
        )
        print(f"[saved] {output_path}", flush=True)
        return output_path

    _write_json(
        OUTPUT_DIR / "config.json",
        {
            "task": args_cli.task,
            "start_pose": args_cli.start_pose,
            "loaded_pose": args_cli.load_pose,
            "stick1_mode": args_cli.stick1_mode,
            "stick_forward_offset_m": args_cli.stick_forward_offset,
            "stick_height_offset_m": args_cli.stick_height_offset,
            "stick_print_hz": args_cli.stick_print_hz,
            "initial_joint_step_deg": args_cli.joint_step_deg,
            "joint_names": HAND_JOINT_NAMES,
        },
    )
    print(f"[hand_grasp_keyboard] output: {OUTPUT_DIR}", flush=True)
    _print_controls()
    print(
        "Stick output uses palm-local xyz in mm: "
        "+y=stick long axis, +z=finger direction/world +x.",
        flush=True,
    )
    print_selected("[selected] ")

    try:
        dt = sim.get_physics_dt()
        while simulation_app.is_running() and not quit_requested:
            for key in keyboard_events.drain():
                char = _key_char(key)
                if char in FINGER_KEYS:
                    selected_finger = FINGER_KEYS[char]
                    print_selected("[selected] ")
                    continue
                if char in JOINT_KEYS:
                    selected_joint = JOINT_KEYS[char]
                    print_selected("[selected] ")
                    continue

                delta = 0.0
                if key == keyboard.Key.left or char == "a":
                    delta = -step_rad
                elif key == keyboard.Key.right or char == "d":
                    delta = step_rad
                if delta != 0.0:
                    index = selected_index()
                    target_q[index] = torch.clamp(
                        target_q[index] + delta,
                        min=soft_limits[index, 0],
                        max=soft_limits[index, 1],
                    )
                    print_selected("[command] ")
                    continue

                if char == "z":
                    step_rad = max(math.radians(0.125), 0.5 * step_rad)
                    print_selected("[step] ")
                elif char == "x":
                    step_rad = min(math.radians(16.0), 2.0 * step_rad)
                    print_selected("[step] ")
                elif char == "o":
                    target_q = manual_open_targets.clone()
                    print(
                        "[pose] manual open target loaded "
                        "(finger1_joint2 at lower limit)",
                        flush=True,
                    )
                    print_selected("[selected] ")
                elif char == "p":
                    target_q = torch.clamp(
                        pregrasp_targets,
                        min=soft_limits[:, 0],
                        max=soft_limits[:, 1],
                    )
                    print("[pose] active training pre-grasp target loaded", flush=True)
                    print_selected("[selected] ")
                elif char == "v":
                    print_all()
                    print_stick_positions()
                elif char == "t":
                    live_stick_output = not live_stick_output
                    next_stick_print_time = time.monotonic()
                    print(
                        f"[state] live joint/stick output "
                        f"{'ON' if live_stick_output else 'OFF'}",
                        flush=True,
                    )
                elif char == "s":
                    save_pose("manual")
                elif key == keyboard.Key.backspace:
                    target_q = apply_requested_start_pose()
                    print("[reset] scene reset", flush=True)
                    print_selected("[selected] ")
                elif key == keyboard.Key.space:
                    paused = not paused
                    print(f"[physics] {'paused' if paused else 'running'}", flush=True)
                elif key == keyboard.Key.esc:
                    save_pose("quit")
                    quit_requested = True

            robot.set_joint_position_target(target_q.unsqueeze(0), joint_ids=joint_ids)
            if paused:
                sim.render()
                time.sleep(0.01)
            else:
                if fixed_stick1_state is not None:
                    _hold_stick_at_state(stick1, fixed_stick1_state)
                scene.write_data_to_sim()
                sim.step(render=True)
                scene.update(dt)
            now = time.monotonic()
            if live_stick_output and now >= next_stick_print_time:
                print_joint_positions()
                print_stick_positions()
                next_stick_print_time = now + stick_print_interval
    finally:
        keyboard_events.stop()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
