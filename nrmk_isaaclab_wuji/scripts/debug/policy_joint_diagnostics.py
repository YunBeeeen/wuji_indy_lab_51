from __future__ import annotations

"""Policy diagnostics for Indy-Wuji cube grasp.

This script loads an RSL-RL checkpoint, runs the policy, and prints per-joint
tracking diagnostics:

- raw policy action and clipped/applied action
- joint position target, actual position, tracking error
- joint velocity and velocity-limit usage
- applied/computed torque and effort-limit usage
- selected cube/reward metrics

Use this when a policy appears to command a lift but some joints do not follow.
"""

import argparse
import importlib.metadata as metadata
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RSL_RL_DIR = PROJECT_ROOT / "scripts" / "rsl_rl"
for path in (PROJECT_ROOT, RSL_RL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Print per-joint diagnostics for an RSL-RL policy.")
parser.add_argument("--task", type=str, default="Indy-Wuji-Cube-Grasp", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs. Keep 1 for readable output.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config registry entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--steps", type=int, default=240, help="Number of policy steps to run.")
parser.add_argument("--interval", type=int, default=1, help="Print every N policy steps.")
parser.add_argument("--top-k", type=int, default=8, help="Print top K joints by absolute tracking error.")
parser.add_argument(
    "--focus-joints",
    nargs="+",
    default=("joint1", "joint2", "finger1_joint1", "finger2_joint1", "finger3_joint1"),
    help="Joint names always printed in addition to top-k joints.",
)
parser.add_argument("--torque-warn", type=float, default=0.90, help="Warn when |torque|/limit exceeds this ratio.")
parser.add_argument("--vel-warn", type=float, default=0.90, help="Warn when |velocity|/limit exceeds this ratio.")
parser.add_argument("--err-warn", type=float, default=0.30, help="Warn when |target-actual| exceeds this radian value.")
parser.add_argument("--render_interval", type=int, default=2, help="Render interval for GUI playback.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
parser.add_argument("--contact", action="store_true", help="Add cube contact sensors and print contact forces.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaac_neuromeka.mdp as mdp
import isaac_neuromeka.tasks  # noqa: F401


installed_version = metadata.version("rsl-rl-lib")

CONTACT_BODIES = {
    "thumb_tip": "finger1_tip_link",
    "index_tip": "finger2_tip_link",
    "index_mid": "finger2_link3",
    "middle_tip": "finger3_tip_link",
    "middle_mid": "finger3_link3",
    "palm": "palm_link",
}


def _add_contact_sensors(env_cfg) -> list[str]:
    sensor_names: list[str] = []
    for label, body_name in CONTACT_BODIES.items():
        name = f"diag_{label}_cube_contact"
        setattr(
            env_cfg.scene,
            name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
                update_period=0.0,
                history_length=1,
                debug_vis=False,
                track_pose=True,
            ),
        )
        sensor_names.append(name)
    return sensor_names


def _contact_force(env, sensor_name: str) -> torch.Tensor:
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


def _joint_action_rows(env, raw_actions: torch.Tensor):
    """Collect per-joint action diagnostics in action-manager order."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    applied = unwrapped.action_manager.action
    rows = []
    action_offset = 0

    terms = getattr(unwrapped.action_manager, "_terms", {})
    for term_name, term in terms.items():
        action_dim = int(getattr(term, "action_dim", 0))
        if action_dim <= 0:
            continue
        if not hasattr(term, "_joint_names") or not hasattr(term, "_joint_ids"):
            action_offset += action_dim
            continue

        joint_names = list(term._joint_names)
        joint_ids = list(term._joint_ids)
        target = term.processed_actions

        for local_id, (joint_name, joint_id) in enumerate(zip(joint_names, joint_ids)):
            action_id = action_offset + local_id
            if action_id >= raw_actions.shape[1]:
                continue
            actual = robot.data.joint_pos[0, joint_id]
            velocity = robot.data.joint_vel[0, joint_id]
            torque = robot.data.applied_torque[0, joint_id]
            computed_torque = robot.data.computed_torque[0, joint_id]
            effort_limit = robot.data.joint_effort_limits[0, joint_id]
            velocity_limit = robot.data.joint_vel_limits[0, joint_id]
            pos_limits = robot.data.joint_pos_limits[0, joint_id]
            err = target[0, local_id] - actual
            rows.append(
                {
                    "term": term_name,
                    "action_id": action_id,
                    "joint_id": joint_id,
                    "joint": joint_name,
                    "raw": raw_actions[0, action_id],
                    "applied": applied[0, action_id],
                    "target": target[0, local_id],
                    "actual": actual,
                    "err": err,
                    "vel": velocity,
                    "vel_limit": velocity_limit,
                    "torque": torque,
                    "computed_torque": computed_torque,
                    "effort_limit": effort_limit,
                    "pos_lo": pos_limits[0],
                    "pos_hi": pos_limits[1],
                }
            )
        action_offset += action_dim
    return rows


def _ratio(value: torch.Tensor, limit: torch.Tensor) -> float:
    limit_value = float(torch.abs(limit).item())
    if limit_value <= 1.0e-8:
        return 0.0
    return abs(float(value.item())) / limit_value


def _reward_snapshot(env) -> dict[str, tuple[float, float]]:
    """Return term -> (weighted_per_second, raw_estimate_per_second)."""
    manager = env.unwrapped.reward_manager
    names = list(getattr(manager, "_term_names", []))
    cfgs = list(getattr(manager, "_term_cfgs", []))
    step_reward = getattr(manager, "_step_reward", None)
    if step_reward is None:
        return {}

    out: dict[str, tuple[float, float]] = {}
    for idx, (name, cfg) in enumerate(zip(names, cfgs)):
        weighted = float(step_reward[0, idx].item())
        weight = float(cfg.weight)
        raw = weighted / weight if abs(weight) > 1.0e-8 else 0.0
        out[name] = (weighted, raw)
    return out


def _cube_snapshot(env) -> dict[str, float]:
    unwrapped = env.unwrapped
    out: dict[str, float] = {}
    if "cube" not in unwrapped.scene.rigid_objects:
        return out

    cube = unwrapped.scene["cube"]
    out["cube_z"] = float(cube.data.root_pos_w[0, 2].item())
    out["cube_speed"] = float(torch.linalg.norm(cube.data.root_lin_vel_w[0]).item())

    rewards_cfg = getattr(unwrapped.cfg, "rewards", None)
    lift_cfg = getattr(rewards_cfg, "cube_lift", None)
    if lift_cfg is not None:
        params = lift_cfg.params
        clearance = mdp.box_ground_clearance(
            unwrapped,
            params["object_cfg"],
            params.get("object_half_extent", (0.03, 0.03, 0.03)),
            params.get("surface_z", 0.0),
        )
        out["clearance"] = float(clearance[0].item())

    reward_manager = unwrapped.reward_manager
    if hasattr(reward_manager, "_compute_cube_distance_metrics"):
        try:
            metrics = reward_manager._compute_cube_distance_metrics()
        except Exception:
            metrics = {}
        for key in (
            "palm_facing",
            "cage_inside_frac",
            "cage_sdf_mean",
            "cage_sdf_min",
            "cage_span",
            "thumb_index_opposition",
            "thumb_middle_opposition",
            "arm_manipulability",
            "action_track_err",
            "action_delta",
        ):
            if key in metrics:
                out[key] = float(metrics[key][0].item())
    return out


def _print_step_summary(
    step: int,
    raw_actions: torch.Tensor,
    rows: list[dict],
    rewards: dict,
    cube: dict,
    clip_limit: float | None,
) -> None:
    if rows:
        max_err_row = max(rows, key=lambda row: abs(float(row["err"].item())))
        max_torque_ratio = max(_ratio(row["torque"], row["effort_limit"]) for row in rows)
        max_vel_ratio = max(_ratio(row["vel"], row["vel_limit"]) for row in rows)
        max_err = abs(float(max_err_row["err"].item()))
        max_err_joint = max_err_row["joint"]
    else:
        max_err = 0.0
        max_err_joint = "-"
        max_torque_ratio = 0.0
        max_vel_ratio = 0.0

    if clip_limit is None:
        clip_pct = 0.0
    else:
        clip_pct = float((raw_actions.abs() > float(clip_limit)).float().mean().item() * 100.0)

    lift_raw = rewards.get("cube_lift", (0.0, 0.0))[1]
    hold_raw = rewards.get("finger_cage_hold", (0.0, 0.0))[1]
    reach_raw = rewards.get("finger_cage_reach", (0.0, 0.0))[1]
    facing_raw = rewards.get("palm_facing", (0.0, 0.0))[1]
    clearance = cube.get("clearance", float("nan"))
    cage = cube.get("cage_inside_frac", float("nan"))
    cube_speed = cube.get("cube_speed", float("nan"))

    print(
        f"\nSTEP {step:04d} |raw|={raw_actions.abs().mean().item():.3f} "
        f"clip={clip_pct:5.1f}% max_err={max_err:.3f}({max_err_joint}) "
        f"torque%={max_torque_ratio*100:5.1f} vel%={max_vel_ratio*100:5.1f} "
        f"clearance={clearance:+.4f}m cube_speed={cube_speed:.3f} "
        f"cage={cage:.3f} raw[reach/hold/lift/facing]="
        f"{reach_raw:+.4f}/{hold_raw:+.4f}/{lift_raw:+.4f}/{facing_raw:+.4f}",
        flush=True,
    )


def _print_rows(rows: list[dict], focus_joints: set[str]) -> None:
    if not rows:
        return

    by_name = {row["joint"]: row for row in rows}
    selected = sorted(rows, key=lambda row: abs(float(row["err"].item())), reverse=True)[: args_cli.top_k]
    selected_names = {row["joint"] for row in selected}
    for name in focus_joints:
        if name in by_name and name not in selected_names:
            selected.append(by_name[name])
            selected_names.add(name)

    print(
        "  joint                  raw     app  target  actual     err     vel   v%  torque   tq%   comp   limits",
        flush=True,
    )
    for row in selected:
        err = float(row["err"].item())
        vel = float(row["vel"].item())
        torque = float(row["torque"].item())
        computed = float(row["computed_torque"].item())
        vel_pct = _ratio(row["vel"], row["vel_limit"]) * 100.0
        torque_pct = _ratio(row["torque"], row["effort_limit"]) * 100.0
        flags = ""
        if abs(err) >= args_cli.err_warn:
            flags += " E"
        if torque_pct >= args_cli.torque_warn * 100.0:
            flags += " T"
        if vel_pct >= args_cli.vel_warn * 100.0:
            flags += " V"
        print(
            f"  {row['joint']:<20}"
            f"{float(row['raw'].item()):>7.3f}{float(row['applied'].item()):>8.3f}"
            f"{float(row['target'].item()):>8.3f}{float(row['actual'].item()):>8.3f}"
            f"{err:>8.3f}{vel:>8.3f}{vel_pct:>5.0f}"
            f"{torque:>8.2f}{torque_pct:>6.0f}{computed:>8.2f}"
            f" [{float(row['pos_lo'].item()):+.2f},{float(row['pos_hi'].item()):+.2f}]{flags}",
            flush=True,
        )


def _print_contacts(env, sensor_names: list[str]) -> None:
    if not sensor_names:
        return
    values = []
    for sensor_name in sensor_names:
        label = sensor_name.removeprefix("diag_").removesuffix("_cube_contact")
        values.append(f"{label}={_contact_force(env, sensor_name)[0].item():.3f}N")
    print("  contact " + " ".join(values), flush=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed if args_cli.seed is None else args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.render_interval = args_cli.render_interval

    sensor_names: list[str] = []
    if args_cli.contact:
        sensor_names = _add_contact_sensors(env_cfg)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    print(f"[INFO] Loading experiment from: {log_root_path}", flush=True)
    print(f"[INFO] Loading checkpoint: {resume_path}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    focus_joints = set(args_cli.focus_joints or [])

    print(
        "\nLegend: E=tracking error warning, T=torque saturation warning, V=velocity saturation warning.",
        flush=True,
    )
    print("       tq%=|applied_torque|/effort_limit, v%=|joint_vel|/velocity_limit.", flush=True)

    for step in range(args_cli.steps):
        start_time = time.time()
        with torch.inference_mode():
            raw_actions = policy(obs)
            obs, _, _, _ = env.step(raw_actions)

        if step % max(args_cli.interval, 1) == 0:
            rows = _joint_action_rows(env, raw_actions)
            rewards = _reward_snapshot(env)
            cube = _cube_snapshot(env)
            _print_step_summary(step, raw_actions, rows, rewards, cube, agent_cfg.clip_actions)
            _print_rows(rows, focus_joints)
            _print_contacts(env, sensor_names)

        if args_cli.real_time:
            sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
