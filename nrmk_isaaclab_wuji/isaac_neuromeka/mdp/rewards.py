from __future__ import annotations

import pdb  # noqa:F401
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaac_neuromeka.assets.articulation import FiniteArticulation


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def end_effector_position_tracking_bounded(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    distance_max: float = 1.0,
) -> torch.Tensor:

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore

    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    distance_bonus = 1.0 - torch.clamp(distance, 0.0, distance_max) / distance_max

    return distance_bonus


def end_effector_orientation_tracking_distance_bounded(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, distance_max: float = 0.5
) -> torch.Tensor:

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore

    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    orientation_bonus = 1.0 - torch.clamp(orientation_error, 0.0, 3.14) / 3.14

    bad_indicies = distance > distance_max

    total_reward = orientation_bonus
    total_reward[bad_indicies] = 0.0

    return total_reward


def end_effector_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the end-effector speed using L2-norm.

    The function computes the end-effector speed as the L2-norm of the end-effector's speed.
    """

    asset: RigidObject = env.scene[asset_cfg.name]

    speed = torch.abs(asset.data.body_state_w[:, asset_cfg.body_ids[0], 7:10])
    return torch.norm(speed, dim=1)


def finite_joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L1-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the L1 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset._finite_joint_vel[:, asset_cfg.joint_ids]), dim=1)


def action_second_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    # TODO: currently broken
    return torch.sum(
        torch.square(
            (env.action_manager.action - env.action_manager.prev_action)
            - (env.action_manager.prev_action - env.action_manager.prevprev_action)
        ),
        dim=1,
    )
