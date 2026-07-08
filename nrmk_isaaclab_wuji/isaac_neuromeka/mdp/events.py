from typing import Dict

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from isaac_neuromeka.assets.articulation import FiniteArticulation
from isaac_neuromeka.terrain.mesh_terrain_importer import MeshTerrainImporter


def randomize_delay(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, delay_step_range: Dict[str, int]):
    """
    Available delay: 0 - (decimation - 1)
    """
    env.delay_steps = torch.randint(
        low=np.clip(delay_step_range.get("low", 0), a_min=0, a_max=env.cfg.decimation - 1),
        high=np.clip(delay_step_range.get("high", 0), a_min=0, a_max=env.cfg.decimation),
        size=(env.num_envs,),
        device=env.device,
    )


# def reset_root_pos(
#     env: ManagerBasedRLEnv,
#     env_ids: torch.Tensor,
#     pose_range: dict[str, tuple[float, float]],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     permute_envs: bool = False,
#     ):

#     # extract the used quantities (to enable type-hinting)
#     asset: FiniteArticulation = env.scene[asset_cfg.name]
#     # get default root state
#     root_states = asset.data.default_root_state[env_ids].clone()

#     # root poses
#     range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
#     ranges = torch.tensor(range_list, device=asset.device)
#     rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

#     base_pos_w = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
#     base_quat_w = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])

#     # root velocities
#     velocities = torch.zeros_like(root_states[:, 7:13])

#     # set into the physics simulation
#     asset.write_root_pose_to_sim(torch.cat([base_pos_w, base_quat_w], dim=-1), env_ids=env_ids)
#     asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_pose_mesh_terrain(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):

    # access the used quantities (to enable type-hinting)
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    terrain: MeshTerrainImporter = env.scene.terrain

    # obtain all flat patches corresponding to the valid poses
    valid_positions: torch.Tensor = terrain.flat_patches

    # sample random valid poses
    ids = torch.randint(0, valid_positions.shape[0], size=(len(env_ids),), device=env.device)
    positions = valid_positions[ids]
    positions += asset.data.default_root_state[env_ids, :3]
    positions[:, 2] += 0.05  # to avoid collision with the terrain

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = asset.data.default_root_state[env_ids, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
