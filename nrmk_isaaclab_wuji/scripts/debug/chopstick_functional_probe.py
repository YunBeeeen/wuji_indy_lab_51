"""Verify the one-stick functional-grasp target and observation wiring."""

from __future__ import annotations

import argparse
import faulthandler
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe Indy-Wuji-Chopsticks-Grasp constraints.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1, help="Zero-action steps before reading state.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_neuromeka.mdp as mdp  # noqa: E402
import isaac_neuromeka.tasks  # noqa: F401, E402
from isaac_neuromeka.tasks.manipulation.functional_grasp import mdp as fg_mdp  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _policy_obs(obs):
    return obs["policy"] if isinstance(obs, dict) else obs


def _compare(name: str, observed: torch.Tensor, expected: torch.Tensor):
    diff = observed - expected
    print(f"\n[{name}] max_abs_diff={diff.abs().max().item():.6g}", flush=True)
    print(f"obs={observed[0].detach().cpu().tolist()}", flush=True)
    print(f"expected={expected[0].detach().cpu().tolist()}", flush=True)


def main():
    faulthandler.dump_traceback_later(30, repeat=False)
    task = "Indy-Wuji-Chopsticks-Grasp"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=env_cfg).unwrapped

    obs, _ = env.reset()
    zero_action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args.steps):
        obs, _, _, _, _ = env.step(zero_action)
    policy = _policy_obs(obs)

    # 18 joint + 3 stick + 15 fingertips + 3 size + 3 index region
    # + 3 thumb region + 3 hand-stick orientation + 18 previous action = 66.
    s_stick_pos = slice(18, 21)
    s_stick_size = slice(36, 39)
    s_index_error = slice(39, 42)
    s_thumb_error = slice(42, 45)
    s_orientation_error = slice(45, 48)
    s_action = slice(48, 66)

    rewards = env.reward_manager
    index_cfg = rewards.get_term_cfg("index_grip").params
    thumb_cfg = rewards.get_term_cfg("thumb_grip").params
    hold_cfg = rewards.get_term_cfg("finger_cage_hold").params
    palm_cfg = index_cfg["palm_cfg"]
    object_cfg = index_cfg["object_cfg"]
    robot = env.scene[palm_cfg.name]
    stick = env.scene[object_cfg.name]
    palm_id = palm_cfg.body_ids[0]

    expected_stick_pos = stick.data.root_pos_w - robot.data.body_state_w[:, palm_id, :3]
    expected_size = mdp.object_dims(env, object_cfg, fallback_size=(0.02, 0.18, 0.02))
    expected_index_error = fg_mdp.index_grip_error_b(
        env,
        index_cfg["palm_cfg"],
        index_cfg["fingertip_cfg"],
        index_cfg["object_cfg"],
        object_half_extent=index_cfg["object_half_extent"],
        long_axis=index_cfg["long_axis"],
        axial_region=index_cfg["axial_region"],
        surface_axis=index_cfg["surface_axis"],
        surface_sign=index_cfg["surface_sign"],
        surface_offset=index_cfg["surface_offset"],
        surface_tolerance=index_cfg["surface_tolerance"],
    )
    expected_thumb_error = fg_mdp.thumb_grip_error_b(
        env,
        thumb_cfg["palm_cfg"],
        thumb_cfg["fingertip_cfg"],
        thumb_cfg["object_cfg"],
        object_half_extent=thumb_cfg["object_half_extent"],
        long_axis=thumb_cfg["long_axis"],
        axial_region=thumb_cfg["axial_region"],
        surface_axis=thumb_cfg["surface_axis"],
        surface_sign=thumb_cfg["surface_sign"],
        surface_offset=thumb_cfg["surface_offset"],
        surface_tolerance=thumb_cfg["surface_tolerance"],
    )
    expected_orientation_error = fg_mdp.hand_tool_orientation_error_axis_angle(
        env,
        index_cfg["palm_cfg"],
        index_cfg["object_cfg"],
    )
    index_lower_o, index_upper_o = fg_mdp.grip_region_bounds_o(
        env,
        index_cfg["object_half_extent"],
        index_cfg["long_axis"],
        index_cfg["axial_region"],
        index_cfg["surface_axis"],
        index_cfg["surface_sign"],
        index_cfg["surface_offset"],
        index_cfg["surface_tolerance"],
    )
    thumb_lower_o, thumb_upper_o = fg_mdp.grip_region_bounds_o(
        env,
        thumb_cfg["object_half_extent"],
        thumb_cfg["long_axis"],
        thumb_cfg["axial_region"],
        thumb_cfg["surface_axis"],
        thumb_cfg["surface_sign"],
        thumb_cfg["surface_offset"],
        thumb_cfg["surface_tolerance"],
    )
    index_error = torch.norm(expected_index_error, dim=-1)
    thumb_error = torch.norm(expected_thumb_error, dim=-1)
    orientation_error = fg_mdp.hand_tool_orientation_error(env, palm_cfg, object_cfg)
    cage = fg_mdp.balanced_tripod_cage_gate(env, **hold_cfg)
    cage_common = {
        "object_cfg": hold_cfg["object_cfg"],
        "object_half_extent": hold_cfg["object_half_extent"],
        "num_points": hold_cfg["num_points"],
        "sphere_radius": hold_cfg["sphere_radius"],
        "depth_max": hold_cfg["depth_max"],
        "point_fractions": hold_cfg["point_fractions"],
    }
    index_cage = mdp.object_in_finger_cage(
        env, asset_cfg=hold_cfg["index_cage_cfg"], **cage_common
    )
    middle_cage = mdp.object_in_finger_cage(
        env, asset_cfg=hold_cfg["middle_cage_cfg"], **cage_common
    )
    clearance = mdp.box_ground_clearance(
        env,
        object_cfg,
        hold_cfg["object_half_extent"],
        env.cfg.rewards.cube_lift.params["surface_z"],
    )
    if not rewards._cube_metric_enabled:
        raise RuntimeError("CustomRewardManager object metrics are disabled")
    manager_metrics = rewards._compute_cube_distance_metrics()
    expected_manager_metrics = {
        "tripod_index_gate": index_cage,
        "tripod_middle_gate": middle_cage,
        "tripod_gate": cage,
        "index_grip_error": index_error,
        "thumb_grip_error": thumb_error,
        "hand_stick_orientation_error": orientation_error,
        "cube_clearance": clearance,
    }

    print("=== Chopstick functional-grasp probe ===", flush=True)
    print(f"obs_shape={tuple(policy.shape)} action_dim={env.action_manager.total_action_dim}", flush=True)
    print(f"stick_size={expected_size[0].detach().cpu().tolist()}", flush=True)
    print(
        f"index_region_o={index_lower_o[0].detach().cpu().tolist()}.."
        f"{index_upper_o[0].detach().cpu().tolist()}",
        flush=True,
    )
    print(
        f"thumb_region_o={thumb_lower_o[0].detach().cpu().tolist()}.."
        f"{thumb_upper_o[0].detach().cpu().tolist()}",
        flush=True,
    )
    print(
        f"captured_target_q_o_h={env.chopstick_target_palm_quat_o[0].detach().cpu().tolist()}",
        flush=True,
    )
    print(
        f"index_error={index_error[0].item():.4f}m thumb_error={thumb_error[0].item():.4f}m "
        f"orientation_error={torch.rad2deg(orientation_error[0]).item():.2f}deg "
        f"cage(index/middle/balanced)="
        f"{index_cage[0].item():.3f}/{middle_cage[0].item():.3f}/{cage[0].item():.3f} "
        f"clearance={clearance[0].item():+.4f}m",
        flush=True,
    )
    print("\n[CustomRewardManager functional metrics]", flush=True)
    for name, expected in expected_manager_metrics.items():
        actual = manager_metrics[name]
        max_abs_diff = torch.max(torch.abs(actual - expected)).item()
        print(
            f"{name}={actual[0].item():.6f} expected={expected[0].item():.6f} "
            f"max_abs_diff={max_abs_diff:.6g}",
            flush=True,
        )
        if max_abs_diff > 1.0e-6:
            raise AssertionError(f"{name} metric differs from the active constraint")

    _compare("stick_pos obs[18:21]", policy[:, s_stick_pos], expected_stick_pos)
    _compare("stick_size obs[36:39]", policy[:, s_stick_size], expected_size)
    _compare("index_grip_error obs[39:42]", policy[:, s_index_error], expected_index_error)
    _compare("thumb_grip_error obs[42:45]", policy[:, s_thumb_error], expected_thumb_error)
    _compare(
        "hand_stick_orientation_error obs[45:48]",
        policy[:, s_orientation_error],
        expected_orientation_error,
    )
    _compare("action_history obs[48:66]", policy[:, s_action], env.action_manager.prev_action)
    faulthandler.cancel_dump_traceback_later()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
