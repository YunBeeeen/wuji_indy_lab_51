from __future__ import annotations

import pdb  # noqa:F401
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaac_neuromeka.assets.articulation import FiniteArticulation


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

    # #####
    # object: RigidObject = env.scene[object_cfg.name]
    # ee_frame: FrameTransformer = env.scene["ee_frame"]
    # # Target object position: (num_envs, 3)
    # cube_pos_w = object.data.root_pos_w
    # # End-effector position: (num_envs, 3)
    # ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # # Distance of the end-effector to the object: (num_envs,)
    # object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    #
    # return torch.where(torch.logical_and(object.data.root_pos_w[:, 2] > minimal_height, object_ee_distance < 0.3), 1.0, 0.0)
    # #####


def action_w_object_condition(env: ManagerBasedRLEnv, threshold: float, penalty_scale: float) -> torch.Tensor:
    ## compute action L2
    action_l2 = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

    ## compute object-goal distance
    robot: RigidObject = env.scene["robot"]
    object: RigidObject = env.scene["object"]
    command = env.command_manager.get_command("object_pose")
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return torch.where(distance < threshold, action_l2 * penalty_scale, action_l2)


def object_z_distance(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene["robot"]
    object: RigidObject = env.scene["object"]
    command = env.command_manager.get_command("object_pose")
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.abs(des_pos_w[:, 2] - object.data.root_pos_w[:, 2])
    # rewarded if the object is lifted above the threshold
    # return 1 - torch.tanh(distance / std)
    return torch.where(env.episode_length_buf > 30, 1 - torch.tanh(distance / std), 0.0)  # 1s for 30Hz controller


def object_height(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.clamp(object.data.root_pos_w[:, 2], min=0.0, max=0.25)


def object_height_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(
        object.data.root_pos_w[:, 2] > minimal_height, torch.clamp(object.data.root_pos_w[:, 2], min=0.0, max=0.25), 0.0
    )


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return torch.where(env.episode_length_buf > 30, 1 - torch.tanh(distance / std), 0.0)
    # ############
    # n_iter = int(env.common_step_counter / 24)
    # max_iter = 3000
    # max_distance_offset = 0.1
    # distance_offset = max(0, -max_distance_offset / max_iter * n_iter + max_distance_offset)
    # return torch.where(env.episode_length_buf > 30, 1 - torch.tanh((distance - distance_offset) / std), 0.)  # 1s for 30Hz controller
    # ############


def object_goal_distance_w_ee(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    obj_ee_distance = torch.norm(object.data.root_pos_w[:, :3] - ee_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (obj_ee_distance < minimal_height) * (1 - torch.tanh(distance / std))


def object_goal_distance_wo_clip(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return 1 - torch.tanh(distance / std)


def joint_vel_l2_w_object(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L1-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the L1 norm.
    """
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    object_height = cube_pos_w[:, 2]
    return torch.where(torch.logical_and(object_ee_distance < 0.05, object_height > 0.25), 10.0, 1.0) * torch.sum(
        torch.square(robot.data.joint_vel[:, robot_cfg.joint_ids]), dim=1
    )


def finite_joint_vel_l2_w_object(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L1-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the L1 norm.
    """
    # extract the used quantities (to enable type-hinting)
    robot: FiniteArticulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    object_height = cube_pos_w[:, 2]
    return torch.where(torch.logical_and(object_ee_distance < 0.05, object_height > 0.25), 10.0, 1.0) * torch.sum(
        torch.square(robot._finite_joint_vel[:, robot_cfg.joint_ids]), dim=1
    )
