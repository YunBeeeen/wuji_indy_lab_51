from __future__ import annotations

"""Scripted contact/lift probe for Indy-Wuji cube grasp.

This does not load a policy. It runs fixed actions:

1. reset
2. settle with zero action
3. close controlled fingers
4. optionally try simple arm lift candidates while fingers stay closed

The output answers two questions:

- Which hand links contact the cube?
- Does the cube clear the support while thumb + middle contact exists?
"""

import argparse
import itertools
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


parser = argparse.ArgumentParser(description="Check scripted cube contact and lift for Indy-Wuji.")
parser.add_argument("--task", type=str, default="Indy-Wuji-Cube-Grasp", help="Gym task id.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of envs. Keep 1 for readable output.")
parser.add_argument("--settle-steps", type=int, default=30, help="Zero-action steps after reset.")
parser.add_argument("--close-steps", type=int, default=60, help="Finger-close steps before lift.")
parser.add_argument("--lift-steps", type=int, default=60, help="Steps for each lift candidate.")
parser.add_argument("--close-action", type=float, default=1.0, help="Raw action sent to controlled finger joints.")
parser.add_argument(
    "--finger-action",
    type=float,
    nargs="+",
    default=None,
    help=(
        "Finger raw action override. Give 3 values for thumb/index/middle groups, or 12 values in action joint order."
    ),
)
parser.add_argument(
    "--sweep-fingers",
    action="store_true",
    help="Sweep thumb/index/middle close values before trying lift.",
)
parser.add_argument(
    "--finger-values",
    type=float,
    nargs="+",
    default=(0.0, 0.5, 1.0),
    help="Values used by --sweep-fingers for each of thumb/index/middle.",
)
parser.add_argument("--lift-action", type=float, nargs=6, default=None, help="Raw arm action vector for a single lift run.")
parser.add_argument("--sweep-lift", action="store_true", help="Try +/- one-axis arm lift candidates.")
parser.add_argument("--lift-magnitude", type=float, default=1.0, help="Raw action magnitude for --sweep-lift.")
parser.add_argument("--contact-threshold", type=float, default=0.2, help="Contact force threshold in N.")
parser.add_argument(
    "--contact-mode",
    choices=("thumb_middle", "thumb_index", "thumb_any", "tripod"),
    default="thumb_middle",
    help="Which contact pattern counts as GOOD_CONTACT.",
)
parser.add_argument("--lift-threshold", type=float, default=0.005, help="Cube clearance threshold in m.")
parser.add_argument("--render-interval", type=int, default=2, help="Render interval for GUI mode.")
parser.add_argument("--debug-vis", action="store_true", help="Enable contact sensor debug visualization.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Use USD I/O instead of Fabric.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.sensors import ContactSensorCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaac_neuromeka.tasks  # noqa: F401
from isaac_neuromeka.tasks.manipulation.grasp.cube_grasp_env_cfg import BASE_Z, CUBE_HALF


CONTACT_BODIES = {
    "thumb_tip": "finger1_tip_link",
    "index_tip": "finger2_tip_link",
    "index_mid": "finger2_link3",
    "middle_tip": "finger3_tip_link",
    "middle_mid": "finger3_link3",
    "palm": "palm_link",
}


def _sensor_name(label: str) -> str:
    return f"{label}_cube_contact"


def add_contact_sensors(env_cfg, debug_vis: bool) -> list[str]:
    sensor_names: list[str] = []
    for label, body_name in CONTACT_BODIES.items():
        name = _sensor_name(label)
        setattr(
            env_cfg.scene,
            name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
                update_period=0.0,
                history_length=1,
                debug_vis=debug_vis,
                track_pose=True,
            ),
        )
        sensor_names.append(name)

    return sensor_names


def step_env(env, action: torch.Tensor, steps: int) -> None:
    for _ in range(steps):
        env.step(action)


def contact_force(env, sensor_name: str) -> torch.Tensor:
    sensor = env.unwrapped.scene.sensors[sensor_name]
    data = sensor.data
    force_w = getattr(data, "force_matrix_w", None)
    if force_w is None:
        force_w = getattr(data, "net_forces_w", None)
    if force_w is None:
        return torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)

    force = torch.linalg.norm(force_w, dim=-1)
    while force.dim() > 1:
        force = force.sum(dim=-1)
    return force


def cube_clearance(env) -> torch.Tensor:
    cube = env.unwrapped.scene["cube"]
    return cube.data.root_pos_w[:, 2] - CUBE_HALF - BASE_Z


def make_action(
    env,
    close_action: float,
    arm_values: list[float] | tuple[float, ...] | None = None,
    finger_values: list[float] | tuple[float, ...] | None = None,
) -> torch.Tensor:
    unwrapped = env.unwrapped
    action_term = unwrapped.action_manager.get_term("arm_action")
    joint_names = list(action_term._joint_names)
    action = torch.zeros((unwrapped.num_envs, len(joint_names)), device=unwrapped.device)

    if arm_values is not None:
        arm_idx = [i for i, name in enumerate(joint_names) if name.startswith("joint")]
        if len(arm_values) != len(arm_idx):
            raise ValueError(f"Expected {len(arm_idx)} arm values, got {len(arm_values)}")
        for idx, value in zip(arm_idx, arm_values):
            action[:, idx] = float(value)

    finger_idx = [i for i, name in enumerate(joint_names) if name.startswith("finger")]
    if finger_idx:
        if finger_values is None:
            action[:, finger_idx] = close_action
        elif len(finger_values) == 3:
            for idx in finger_idx:
                name = joint_names[idx]
                if name.startswith("finger1"):
                    action[:, idx] = float(finger_values[0])
                elif name.startswith("finger2"):
                    action[:, idx] = float(finger_values[1])
                elif name.startswith("finger3"):
                    action[:, idx] = float(finger_values[2])
        elif len(finger_values) == len(finger_idx):
            for idx, value in zip(finger_idx, finger_values):
                action[:, idx] = float(value)
        else:
            raise ValueError(f"Expected 3 or {len(finger_idx)} finger values, got {len(finger_values)}")
    return action


def contact_ok(forces: dict[str, float], threshold: float, mode: str) -> bool:
    thumb = forces.get("thumb_tip", 0.0) > threshold
    index = max(forces.get("index_tip", 0.0), forces.get("index_mid", 0.0)) > threshold
    middle = max(forces.get("middle_tip", 0.0), forces.get("middle_mid", 0.0)) > threshold
    if mode == "thumb_middle":
        return thumb and middle
    if mode == "thumb_index":
        return thumb and index
    if mode == "thumb_any":
        return thumb and (index or middle)
    if mode == "tripod":
        return thumb and index and middle
    raise ValueError(f"Unknown contact mode: {mode}")


def print_report(env, label: str, sensor_names: list[str], contact_threshold: float, max_clearance: float) -> bool:
    forces = {name.removesuffix("_cube_contact"): contact_force(env, name)[0].item() for name in sensor_names}
    clearance = cube_clearance(env)[0].item()
    good_contact = contact_ok(forces, contact_threshold, args_cli.contact_mode)

    print(f"\n[{label}]", flush=True)
    print(f"cube_clearance={clearance:+.4f} m  max_clearance={max_clearance:+.4f} m", flush=True)
    print(f"GOOD_CONTACT {args_cli.contact_mode}: {good_contact}", flush=True)
    for name in sorted(forces):
        print(f"{name:>20}: {forces[name]:8.4f} N", flush=True)
    return good_contact


def run_candidate(
    env,
    name: str,
    sensor_names: list[str],
    arm_values: list[float] | None,
    finger_values: list[float] | tuple[float, ...] | None,
) -> tuple[bool, bool, float]:
    env.reset()
    zero_action = make_action(env, close_action=0.0)
    close_action = make_action(env, close_action=args_cli.close_action, finger_values=finger_values)
    lift_action = make_action(
        env, close_action=args_cli.close_action, arm_values=arm_values, finger_values=finger_values
    )

    step_env(env, zero_action, args_cli.settle_steps)
    print_report(env, f"{name}: after_settle", sensor_names, args_cli.contact_threshold, cube_clearance(env)[0].item())

    step_env(env, close_action, args_cli.close_steps)
    good_after_close = print_report(
        env, f"{name}: after_close", sensor_names, args_cli.contact_threshold, cube_clearance(env)[0].item()
    )

    max_clearance = cube_clearance(env)[0].item()
    for _ in range(args_cli.lift_steps):
        env.step(lift_action)
        max_clearance = max(max_clearance, cube_clearance(env)[0].item())

    good_final = print_report(env, f"{name}: after_lift", sensor_names, args_cli.contact_threshold, max_clearance)
    lift_success = max_clearance > args_cli.lift_threshold and good_final
    print(f"RESULT {name}: contact_after_close={good_after_close} lift_success={lift_success}", flush=True)
    return good_after_close, lift_success, max_clearance


def lift_candidates() -> list[tuple[str, list[float]]]:
    mag = args_cli.lift_magnitude
    candidates: list[tuple[str, list[float]]] = []
    for joint_id in range(6):
        for sign in (-1.0, 1.0):
            values = [0.0] * 6
            values[joint_id] = sign * mag
            candidates.append((f"joint{joint_id}_{sign:+.1f}", values))
    return candidates


def finger_candidates() -> list[tuple[str, tuple[float, float, float] | None]]:
    if args_cli.finger_action is not None:
        values = tuple(float(v) for v in args_cli.finger_action)
        return [("finger_manual", values)]
    if not args_cli.sweep_fingers:
        return [("finger_all", None)]

    values = [float(v) for v in args_cli.finger_values]
    candidates: list[tuple[str, tuple[float, float, float]]] = []
    for thumb, index, middle in itertools.product(values, repeat=3):
        name = f"f_t{thumb:.2f}_i{index:.2f}_m{middle:.2f}"
        candidates.append((name, (thumb, index, middle)))
    return candidates


def main() -> None:
    print("PROBE: loading env cfg", flush=True)
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.render_interval = args_cli.render_interval
    sensor_names = add_contact_sensors(env_cfg, args_cli.debug_vis)

    print("PROBE: creating env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        print("PROBE: env created", flush=True)
        unwrapped = env.unwrapped
        action_term = unwrapped.action_manager.get_term("arm_action")
        print("\nAction joint order:", flush=True)
        for idx, name in enumerate(action_term._joint_names):
            print(f"{idx:02d}: {name}", flush=True)

        results = []
        arm_candidates = [("close_only", [0.0] * 6)]
        if args_cli.lift_action is not None:
            arm_candidates = [("manual_lift", list(args_cli.lift_action))]
        elif args_cli.sweep_lift:
            arm_candidates += lift_candidates()

        for finger_name, finger_values in finger_candidates():
            for arm_name, arm_values in arm_candidates:
                name = f"{finger_name}/{arm_name}"
                results.append((name, *run_candidate(env, name, sensor_names, arm_values, finger_values)))

        print("\nSummary:", flush=True)
        print(f"{'candidate':>14} {'contact':>8} {'lift':>8} {'max_clearance(m)':>18}", flush=True)
        for name, contact, lift, max_clearance in results:
            print(f"{name:>14} {str(contact):>8} {str(lift):>8} {max_clearance:>18.4f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
