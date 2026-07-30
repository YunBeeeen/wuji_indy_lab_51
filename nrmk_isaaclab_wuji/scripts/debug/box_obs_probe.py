"""Verify Box-Transport observation slices against simulator source state."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe Indy-Wuji-Box-Transport observation values.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to instantiate.")
parser.add_argument("--steps", type=int, default=1, help="Number of zero-action steps before reading observations.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402
import isaac_neuromeka.mdp as mdp  # noqa: E402
from isaac_neuromeka.tasks.manipulation.functional_grasp import mdp as fg_mdp  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


_HALF_FALLBACK = (0.03, 0.03, 0.03)
_PALM_CFG = SceneEntityCfg("robot", body_names=["palm_link"])
_INDEX_CFG = SceneEntityCfg("robot", body_names=["finger2_tip_link"])
_THUMB_CFG = SceneEntityCfg("robot", body_names=["finger1_tip_link"])
_OBJECT_CFG = SceneEntityCfg("cube")
_INDEX_GRIP_REGION = {
    "long_axis": 1,
    "axial_region": (-0.60, -0.30),
    "surface_axis": 2,
    "surface_sign": 1.0,
    "surface_offset": 0.0,
    "surface_tolerance": 0.005,
}
_THUMB_GRIP_REGION = {
    "long_axis": 1,
    "axial_region": (-0.55, -0.25),
    "surface_axis": 0,
    "surface_sign": 1.0,
    "surface_offset": 0.0,
    "surface_tolerance": 0.005,
}


def _policy_obs(obs):
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def _print_pair(name: str, obs_value: torch.Tensor, expected: torch.Tensor, max_rows: int = 4):
    diff = obs_value - expected
    print(f"\n[{name}]", flush=True)
    print(f"max_abs_diff={diff.abs().max().item():.6g}", flush=True)
    rows = min(obs_value.shape[0], max_rows)
    for i in range(rows):
        print(
            f"env{i}: obs={obs_value[i].detach().cpu().tolist()} "
            f"expected={expected[i].detach().cpu().tolist()} "
            f"diff={diff[i].detach().cpu().tolist()}",
            flush=True,
        )


def main():
    env_cfg = parse_env_cfg("Indy-Wuji-Box-Transport", num_envs=args.num_envs)
    env = gym.make("Indy-Wuji-Box-Transport", cfg=env_cfg).unwrapped

    obs, _ = env.reset()
    zero_action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args.steps):
        obs, _, _, _, _ = env.step(zero_action)

    obs_policy = _policy_obs(obs)

    # BoxTransport policy obs layout:
    # joint_pos(18), cube_pos(3), cube_in_fingertips(15), cube_to_goal(3),
    # box_size(3), box_quat(4), box_ori_to_target(3), index_grip_error(3),
    # thumb_grip_error(3), action_history(18) = 73
    s_cube_pos = slice(18, 21)
    s_cube_to_goal = slice(36, 39)
    s_box_size = slice(39, 42)
    s_box_quat = slice(42, 46)
    s_box_ori_to_target = slice(46, 49)
    s_index_grip_error = slice(49, 52)
    s_thumb_grip_error = slice(52, 55)
    s_action_history = slice(55, 73)

    robot = env.scene["robot"]
    cube = env.scene["cube"]

    palm_id = robot.find_bodies("palm_link")[0][0]
    palm_pos_w = robot.data.body_state_w[:, palm_id, :3]
    command = env.command_manager.get_command("cube_goal")
    goal_w = env.scene.env_origins + command[:, :3]

    expected_cube_pos = cube.data.root_pos_w - palm_pos_w
    expected_cube_to_goal = goal_w - cube.data.root_pos_w
    expected_box_size = env.box_half_extents * 2.0
    expected_box_quat = cube.data.root_quat_w
    expected_box_ori_to_target = mdp.object_ori_error_nearest_sym(env, "cube_goal")
    expected_index_grip_error = fg_mdp.index_grip_error_b(
        env,
        _PALM_CFG,
        _INDEX_CFG,
        _OBJECT_CFG,
        _HALF_FALLBACK,
        **_INDEX_GRIP_REGION,
    )
    expected_thumb_grip_error = fg_mdp.thumb_grip_error_b(
        env,
        _PALM_CFG,
        _THUMB_CFG,
        _OBJECT_CFG,
        _HALF_FALLBACK,
        **_THUMB_GRIP_REGION,
    )
    expected_action_history = env.action_manager.prev_action

    print("=== Box observation probe ===", flush=True)
    print(f"num_envs={env.num_envs}", flush=True)
    print(f"obs_shape={tuple(obs_policy.shape)}", flush=True)
    print(f"action_dim={env.action_manager.total_action_dim}", flush=True)
    print(f"command_shape={tuple(command.shape)}", flush=True)
    print(f"goal_quat_w={command[:, 3:7].detach().cpu().tolist()}", flush=True)
    print(f"box_half_extents={env.box_half_extents.detach().cpu().tolist()}", flush=True)

    _print_pair("cube_pos: obs[18:21] vs cube.root_pos_w - palm.pos_w", obs_policy[:, s_cube_pos], expected_cube_pos)
    _print_pair("cube_to_goal: obs[36:39] vs goal_w - cube.root_pos_w", obs_policy[:, s_cube_to_goal], expected_cube_to_goal)
    _print_pair("box_size: obs[39:42] vs env.box_half_extents * 2", obs_policy[:, s_box_size], expected_box_size)
    _print_pair("box_quat: obs[42:46] vs cube.root_quat_w", obs_policy[:, s_box_quat], expected_box_quat)
    _print_pair(
        "box_ori_to_target: obs[46:49] vs nearest command orientation error",
        obs_policy[:, s_box_ori_to_target],
        expected_box_ori_to_target,
    )
    _print_pair(
        "index_grip_error: obs[49:52] vs object-relative rear region",
        obs_policy[:, s_index_grip_error],
        expected_index_grip_error,
    )
    _print_pair(
        "thumb_grip_error: obs[52:55] vs object-relative rear region",
        obs_policy[:, s_thumb_grip_error],
        expected_thumb_grip_error,
    )
    _print_pair("action_history: obs[55:73] vs prev_action", obs_policy[:, s_action_history], expected_action_history)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
