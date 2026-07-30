from __future__ import annotations

import pdb  # noqa:F401
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (  # noqa: F401
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    # from isaac_neuromeka.env.rl_task_custom_env import CustomManagerBasedRLEnv

from isaaclab.sensors import Camera, RayCaster, RayCasterCamera, TiledCamera

from isaac_neuromeka.assets.articulation import FiniteArticulation


def position_in_world(
    env: ManagerBasedRLEnv, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(body_name)[0][0]
    body_pos_w = asset.data.body_state_w[:, body_idx, :3]
    return body_pos_w


def orientation_in_world(
    env: ManagerBasedRLEnv, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(body_name)[0][0]
    return asset.data.body_state_w[:, body_idx, 3:7]


def lidar_pointcloud(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    body_name: str = "base_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(body_name)[0][0]

    ray_hits_w = sensor.data.ray_hits_w.clone()
    valid = torch.isfinite(ray_hits_w).all(dim=-1)

    base_pos_w = asset.data.body_state_w[:, body_idx, :3]
    base_quat_w = asset.data.body_state_w[:, body_idx, 3:7]
    ray_hits_from_base_w = ray_hits_w - base_pos_w.unsqueeze(1)
    ray_hits_from_base_w = torch.where(
        valid.unsqueeze(-1), ray_hits_from_base_w, torch.zeros_like(ray_hits_from_base_w)
    )

    num_rays = ray_hits_w.shape[1]
    base_quat_w = base_quat_w.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4)
    ray_hits_b = quat_apply_inverse(base_quat_w, ray_hits_from_base_w.reshape(-1, 3)).reshape_as(ray_hits_w)

    return torch.where(valid.unsqueeze(-1), ray_hits_b, torch.full_like(ray_hits_b, float("inf")))


def finite_body_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]

    body_vel_W = asset._finite_body_vel_w[:, asset_cfg.body_ids[0], :]
    body_vel_b = torch.zeros_like(body_vel_W)

    body_vel_b[:, :3] = quat_apply_inverse(asset.data.root_quat_w, body_vel_W[:, :3])
    body_vel_b[:, 3:] = quat_apply_inverse(asset.data.root_quat_w, body_vel_W[:, 3:])

    return body_vel_b


def joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def finite_joint_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return asset._finite_joint_vel[:, asset_cfg.joint_ids]


def joint_pos_history(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return torch.cat(
        (asset._prevprev_joint_pos[:, asset_cfg.joint_ids], asset._prev_joint_pos[:, asset_cfg.joint_ids]), dim=-1
    )


def generated_position_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, :3]


def object_position_relative(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm_link"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    obj = env.scene[object_cfg.name]
    body_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    object_pos_w = obj.data.root_pos_w
    return object_pos_w - body_pos_w


def object_position_relative_to_bodies(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["finger1_tip_link", "finger2_tip_link", "finger3_tip_link", "finger4_tip_link", "finger5_tip_link"],
    ),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    obj = env.scene[object_cfg.name]

    body_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids, :3]
    object_pos_w = obj.data.root_pos_w.unsqueeze(1)
    object_in_bodies_w = object_pos_w - body_pos_w
    return object_in_bodies_w.reshape(env.num_envs, -1)


def object_position_relative_to_bodies_local(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=["finger1_tip_link", "finger2_tip_link", "finger3_tip_link", "finger4_tip_link", "finger5_tip_link"],
    ),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    frame_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["palm_link"]),
) -> torch.Tensor:
    """object − body 상대변위를 frame(기본 palm)의 **로컬 프레임**으로 표현 (회전 불변).

    2026-07-29 신설. 기존 object_position_relative_to_bodies는 world축 상대변위라 손이 회전하면
    같은 파지라도 값이 달라져 학습이 어렵고 일반화가 나쁨. palm-local로 회전변환하면 "손바닥 기준
    스틱이 어디"라 회전 불변 + 손목/손바닥 카메라(egocentric)와도 일치(sim-to-real 친화).
    box/cube 태스크가 쓰는 world축 함수는 그대로 두고 이 변형만 chopstick에 연결한다.
    """
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    obj = env.scene[object_cfg.name]
    frame_asset: FiniteArticulation = env.scene[frame_cfg.name]

    body_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids, :3]          # (N, B, 3)
    object_pos_w = obj.data.root_pos_w.unsqueeze(1)                          # (N, 1, 3)
    disp_w = object_pos_w - body_pos_w                                       # (N, B, 3)

    frame_quat_w = frame_asset.data.body_state_w[:, frame_cfg.body_ids[0], 3:7]  # (N, 4)
    num_bodies = disp_w.shape[1]
    frame_quat = frame_quat_w.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)
    disp_local = quat_apply_inverse(frame_quat, disp_w.reshape(-1, 3))
    return disp_local.reshape(env.num_envs, -1)


def object_orientation_world(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """object 자세 world quaternion raw (N,4).

    2026-07-29. 자세오차는 프레임 독립이라 palm 변환의 이득이 없고, palm으로 바꾸면 손 움직임에
    goal 값이 요동함. 그래서 스틱/goal 자세는 **월드 raw**로 주고 정책이 상대오차를 직접 계산
    (4-대칭은 보상이 처리). 위치는 palm-local(잡으러 갈 때 유용)이지만 자세는 월드가 맞음.
    """
    return env.scene[object_cfg.name].data.root_quat_w


def command_orientation_world(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """goal 자세 world quaternion raw (커맨드 [3:7]) (N,4). object_orientation_world와 같은 월드
    프레임이라 정책이 둘로 상대 자세오차를 바로 계산. 커맨드에 quat 없으면 identity."""
    command = env.command_manager.get_command(command_name)
    if command.shape[-1] >= 7:
        return command[:, 3:7]
    q = torch.zeros(command.shape[0], 4, device=command.device, dtype=command.dtype)
    q[:, 0] = 1.0
    return q


def object_position_error_to_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """goal(커맨드, env-로컬) - 큐브 위치. env_origins 보정 포함.

    object_position_error_to_target의 교체품: 그쪽은 target이 월드 고정점이라 다중 env에서
    env마다 다른 상수가 관측에 들어갔음 (2026-07-15 발견). 커맨드 기반 + 로컬 프레임으로 수정.
    """
    obj = env.scene[object_cfg.name]
    goal_w = env.scene.env_origins + env.command_manager.get_command(command_name)[:, :3]
    return goal_w - obj.data.root_pos_w


def object_position_error_to_target(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_pos: tuple[float, float, float] = (0.55, -0.05, 0.12),
) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    target_pos_w = obj.data.root_pos_w.new_tensor(target_pos).unsqueeze(0)
    return target_pos_w - obj.data.root_pos_w


def joint_vel_history(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return torch.cat(
        (
            asset._prevprev_finite_joint_vel[:, asset_cfg.joint_ids],
            asset._prev_finite_joint_vel[:, asset_cfg.joint_ids],
        ),
        dim=-1,
    )


def action_history(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.action_manager.prev_action
    # return torch.cat((env.action_manager.prev_action, env.action_manager.action), dim=-1)


def position_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.command_manager.get_term("ee_pose").metrics["position_error"].unsqueeze(-1)


def orientation_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.command_manager.get_term("ee_pose").metrics["orientation_error"].unsqueeze(-1)


def op_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return asset._op_state


def body_pose_b(
    env: ManagerBasedRLEnv, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(body_name)[0][0]
    body_pose_w = asset.data.body_state_w[:, body_idx, :7]
    body_pos_b, body_quat_b = subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, body_pose_w[:, :3], body_pose_w[:, 3:]  # w -> r  # w -> b
    )  # r -> b
    body_pose_b = torch.cat((body_pos_b, body_quat_b), dim=-1)  # position + orientation (quaternion)
    return body_pose_b


def body_vel_b(
    env: ManagerBasedRLEnv, body_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.tensor:
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    body_idx = asset.find_bodies(body_name)[0][0]
    body_vel_w = asset.data.body_state_w[:, body_idx, 7:]
    body_vel_b = torch.zeros_like(body_vel_w)
    R_BW = torch.transpose(matrix_from_quat(asset.data.root_quat_w), 1, 2)
    body_vel_b[:, :3] = torch.einsum("bij, bj->bi", R_BW, body_vel_w[:, :3])
    body_vel_b[:, 3:] = torch.einsum("bij, bj->bi", R_BW, body_vel_w[:, 3:])
    return body_vel_b


"""
Privileged information.
"""


def joint_friction(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint friction of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their friction returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return asset.data.joint_friction[:, asset_cfg.joint_ids]


def joint_damping(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint damping of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their damping returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping[:, asset_cfg.joint_ids]


def action_delay_steps(env: ManagerBasedRLEnv) -> torch.Tensor:

    if hasattr(env, "delay_steps"):
        return env.delay_steps.reshape(-1, 1)
    else:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.long)


def image_unnormalized(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float()
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0

    return images.clone()


def object_dims(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    fallback_size: tuple[float, float, float] = (0.06, 0.06, 0.06),
) -> torch.Tensor:
    """물체 전체 치수 (N, 3). Box-Transport 관측용.

    randomize_box_dims가 저장한 env.box_half_extents × 2를 반환. 버퍼가 없으면(고정 크기
    태스크) fallback_size 상수 — 이 경우 상수 채널이므로 관측에 넣는 의미는 없음.
    """
    he = getattr(env, "box_half_extents", None)
    if he is not None:
        return he * 2.0
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w.new_tensor(fallback_size).unsqueeze(0).expand(env.num_envs, -1)


def object_ori_error_nearest_sym(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """최근접 대칭 목표까지의 상대 회전 axis-angle (N, 3). ori 정렬 관측 (2026-07-19).

    raw quat 관측만으로는 정책이 "90° 돌린 자세 = 같은 자세, 지금은 그쪽이 더 가깝다"는
    대칭 구조를 스스로 발굴해야 함 — 실측(WRAP 슬롯 A play): 가까운 대칭을 두고 스폰
    자세로 크게 되돌리려는 행동. 이 관측은 대칭 8개(정사각 단면 프리즘) 중 가장 가까운
    목표 기준의 signed 상대 회전을 axis-angle로 직접 제공한다. 이 값은 회전 명령이 아니라
    0으로 줄여야 하는 오차 벡터다. 벡터 크기 = square_prism_ori_error의 각도와 일치.

    command quaternion에 정사각 프리즘의 8개 등가 대칭을 합성한 뒤 현재 자세에서 가장
    가까운 목표를 고른다. 반환값은 ``q_current * q_target^-1``의 axis-angle이므로
    보정 명령 자체가 아니라 "목표 대비 현재 오차"이며, 정책은 이 벡터를 0으로 줄인다.
    """
    from isaac_neuromeka.mdp.rewards import _SQUARE_PRISM_Y_SYMS, _sym_quat_cache

    obj = env.scene[object_cfg.name]
    quat = obj.data.root_quat_w
    command = env.command_manager.get_command(command_name)
    goal_quat = command[:, 3:7]
    key = (quat.device, quat.dtype)
    syms = _sym_quat_cache.get(key)
    if syms is None:
        syms = torch.tensor(_SQUARE_PRISM_Y_SYMS, device=quat.device, dtype=quat.dtype)
        _sym_quat_cache[key] = syms
    goal_candidates = math_utils.quat_mul(
        goal_quat.unsqueeze(1).expand(-1, syms.shape[0], -1),
        syms.unsqueeze(0).expand(quat.shape[0], -1, -1),
    )
    dots = torch.sum(quat.unsqueeze(1) * goal_candidates, dim=-1)
    nearest = torch.abs(dots).argmax(dim=1)
    target = goal_candidates[torch.arange(quat.shape[0], device=quat.device), nearest]
    # 같은 반구로 정렬 (dot<0이면 -S가 같은 회전의 가까운 표현)
    sign = torch.where(dots.gather(1, nearest.unsqueeze(1)).squeeze(1) < 0, -1.0, 1.0)
    target = target * sign.unsqueeze(1)
    q_err = math_utils.quat_mul(quat, math_utils.quat_inv(target))
    return math_utils.axis_angle_from_quat(q_err)
