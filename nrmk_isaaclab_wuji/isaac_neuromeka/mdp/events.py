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


def hold_joints_at_default(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    """액션에 없는 관절의 '위치 목표'를 기본 자세로 고정한다.

    리셋은 관절 '상태'만 기본값으로 되돌리고 위치 '목표' 버퍼는 채우지 않는다. 액션 텀은
    자기 관절만 목표를 쓰므로, 나머지 관절은 목표 0을 향해 저절로 움직인다.
    (실측: 접어둔 약지/새끼(1.2rad)가 매 에피소드 시작 직후 0으로 펴지며 파지 지점을 쓸었음)
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    target = asset.data.default_joint_pos[env_ids][:, joint_ids]
    asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)


def randomize_box_dims(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    width_range: tuple[float, float] = (0.03, 0.06),
    ratio_range: tuple[float, float] = (1.5, 3.0),
    base_size: float = 0.06,
    length_axis: int = 1,
):
    """[Box-Transport 전용, prestartup 모드] env마다 비율보존 직육면체 치수를 샘플해 적용.

    - 단면 w×w 정사각 (w ~ U(width_range)), 길이 = w × U(ratio_range) (length_axis 축)
      → 4096 env가 각자 "살짝 뚱뚱이 ~ 길쭉이" 상자를 가짐 (2026-07-16 사수님 방향)
    - sim 시작 전 USD xformOp:scale을 env별로 수정 (isaaclab randomize_rigid_body_scale 패턴).
      정사각 단면의 축 상관관계 때문에 축 독립인 내장 이벤트로는 표현 불가라 자체 구현
    - scene.replicate_physics = False 필수 (env별 지오메트리를 물리 파서가 개별 파싱)
    - 샘플 결과를 env.box_half_extents (N,3) 버퍼로 저장 — 보상/판정/관측/metrics가 읽어감.
      버퍼가 없는 env(큐브 태스크)는 기존 상수 경로 그대로 (하위호환)
    - 질량은 스케일과 무관하게 spawn cfg 값 고정 (밀도가 크기별로 다름 — 의도: 변수 통제)
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, Sdf

    if env.sim.is_playing():
        raise RuntimeError("randomize_box_dims는 prestartup(시뮬 시작 전) 모드 전용임.")

    asset = env.scene[asset_cfg.name]
    n = env.scene.num_envs
    w = torch.empty(n).uniform_(*width_range)
    ratio = torch.empty(n).uniform_(*ratio_range)
    dims = w.unsqueeze(1).repeat(1, 3)
    dims[:, length_axis] = w * ratio
    scales = (dims / base_size).tolist()

    stage = get_current_stage()
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    with Sdf.ChangeBlock():
        for i in range(n):
            prim_path = prim_paths[i]
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            if scale_spec is None:
                scale_spec = Sdf.AttributeSpec(
                    prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3
                )
            scale_spec.default = Gf.Vec3f(*scales[i])

    env.box_half_extents = (dims * 0.5).to(env.device)


def set_box_default_height(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    surface_z: float = 0.0,
):
    """[Box-Transport 전용, startup 모드] env별 기본 스폰 z = 상판 + 자기 반높이.

    치수가 env마다 달라 고정 z로는 작은 상자가 공중 스폰(리셋마다 낙하)되거나 큰 상자가
    상판을 관통 스폰됨. randomize_box_dims의 버퍼를 읽어 default_root_state를 보정함
    (reset_scene_to_default / reset_root_state_uniform이 이 기본값을 씀).
    """
    obj = env.scene[asset_cfg.name]
    obj.data.default_root_state[:, 2] = surface_z + env.box_half_extents[:, 2]
