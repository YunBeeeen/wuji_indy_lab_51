from __future__ import annotations

import pdb  # noqa:F401
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_mul,
)

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


def _box_signed_distance(
    points_w: torch.Tensor,
    box_pos_w: torch.Tensor,
    box_quat_w: torch.Tensor,
    half_extent: torch.Tensor,
) -> torch.Tensor:
    """점 -> 박스 표면까지의 signed distance. 박스 내부이면 음수.

    박스라서 해석식으로 계산됨. CAD나 사전계산 SDF 불필요.
    입력 (N,P,3) 점 / (N,3) 박스 중심 / (N,4) 박스 회전 / (3,) 반크기 -> 출력 (N,P).
    """
    rel = points_w - box_pos_w.unsqueeze(1)
    quat = box_quat_w.unsqueeze(1).expand(-1, rel.shape[1], -1)
    local = quat_apply_inverse(quat, rel)
    q = local.abs() - half_extent
    outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.clamp(q.max(dim=-1).values, max=0.0) # max -> 제일 0에 가까우니까
    return outside + inside


def cage_points(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, num_points: int) -> torch.Tensor:
    """엄지끝에서 각 대향 body로 선분을 긋고 그 위에 찍는 "파지 간극" 가상점.

    body_names는 [엄지끝, *대향] 순서여야 함 (preserve_order=True 필수).
    논문은 엄지-중지만 써서 6점이지만, 논문에는 r_grasp가 손 회전/손가락 관절각을 붙잡음.
    큐브엔 목표 파지가 없어 r_grasp를 못 쓰므로, 6점만 쓰면 검지가 자유가 되어
    "손바닥이 하늘 + 검지·중지 교차" 자세로도 만점이 나옴 (2026-07-11 실측). 그래서 검지도 포함해 12점.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids, :3]  # type: ignore / 모든 env에 대해 CAGE_BODIES에 해당하는 body 5개만 골라서 그 body들의 position xyz만 가져와라
    if body_pos_w.shape[1] < 2:
        raise ValueError(
            f"cage terms expect [thumb_tip, *opposing] (2 bodies minimum), got {body_pos_w.shape[1]}."
        )
    thumb = body_pos_w[:, 0]    # GPU tensor form / torch.tensor([1,2,3]) -> CPU tensor form
    opposing = body_pos_w[:, 1:]  # (N, M, 3)

    # `num_points` equidistant points strictly between the thumb tip and each opposing body.
    fractions = torch.arange(1, num_points + 1, dtype=thumb.dtype, device=thumb.device) / (num_points + 1) # num_points=3 -> arrange(1,4) -> [1,2,3]/4 -> [0.25, 0.5, 0.75]
    span = opposing - thumb.unsqueeze(1)  # (N, M, 3)
    points = thumb[:, None, None, :] + span.unsqueeze(2) * fractions.view(1, 1, -1, 1)
    return points.reshape(thumb.shape[0], -1, 3)


def _cage_sdf(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    num_points: int,
) -> torch.Tensor:
    """가상점 -> 물체 표면까지의 signed distance. (N, 대향body수 * num_points)"""
    obj: RigidObject = env.scene[object_cfg.name]
    points = cage_points(env, asset_cfg, num_points)
    half = torch.tensor(object_half_extent, dtype=points.dtype, device=points.device)
    return _box_signed_distance(points, obj.data.root_pos_w, obj.data.root_quat_w, half)


def object_in_finger_cage(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
    num_points: int = 3,
    sphere_radius: float = 0.005,
    depth_max: float = 0.02,
) -> torch.Tensor:
    """물체를 손가락 사이에 가두는 것을 보상 (논문 Eq.15, hold).

    가상점을 반지름 sphere_radius의 구로 보고, 그 구가 물체를 파고든 깊이를 보상함.
    손을 오므리면 점들이 물체 안으로 들어가므로 "오므리기"가 직접 보상됨. 접촉센서 불필요.
    거리 reward는 정반대임: 만지면 물체가 밀려나 거리가 늘고 감점 -> 접촉이 손해 -> hover만 함.
    """
    sdf = _cage_sdf(env, asset_cfg, object_cfg, object_half_extent, num_points)
    # 구가 물체를 파고든 깊이. sphere_radius보다 멀면 0, depth_max만큼 들어가면 포화(1).
    penetration = sphere_radius - sdf
    return torch.clamp(penetration / (sphere_radius + depth_max), 0.0, 1.0).mean(dim=1)


class ObjectCageProgressReward(ManagerTermBase):
    """파지 간극을 물체 표면 위로 끌어오는 것을 보상 (논문 Eq.14, reach).

    hold와 "같은" 가상점을 씀. 그래야 물체가 손가락 "사이"로 들어옴.
    손끝을 물체 중심으로 끌면 "엄지만 박고 나머지는 방치"가 최적해가 되고, 그 자세에선 오므려도 안 감싸짐.
    차분형(t-1 vs t) + clamp(min=-1) + reset()에서 기준선 seeding, 셋을 다 해야 함.
    셋 중 하나라도 빠지면 총합이 d(reset)-d(final)로 telescoping되지 않아 swing-out/dawdle 해킹이 남음.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_distance = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    def _mean_sdf(self) -> torch.Tensor:
        p = self.cfg.params
        return _cage_sdf(
            self._env,
            p["asset_cfg"],
            p["object_cfg"],
            p.get("object_half_extent", (0.03, 0.03, 0.03)),
            p.get("num_points", 3),
        ).mean(dim=1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # 기준선은 "리셋 자세"에서 잡음. 첫 __call__에서 잡으면 첫 액션이 기준선을 공짜로 부풀림(= swing-out).
        self._previous_distance[env_ids] = self._mean_sdf()[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
        num_points: int = 3,
        distance_max: float = 0.5,
        palm_cfg: SceneEntityCfg | None = None,
        palm_normal_b: tuple[float, float, float] = (0.19, 0.28, 0.94),
        gate_floor: float = 0.0,
    ) -> torch.Tensor:
        current = self._mean_sdf()
        progress = self._previous_distance - current
        self._previous_distance[:] = current
        reward = torch.clamp(progress / distance_max, min=-1.0, max=1.0)

        # 순서 강제: 손이 물체를 향하지 않으면 "접근"해도 보상이 없음.
        # ★ 전진(양수)에만 게이트를 적용할 것. 후퇴(음수)에도 곱하면 telescoping이 깨짐:
        #   다가감(+0.5) x facing 1.0 = +0.5  |  물러섬(-0.5) x facing 0.0 = 0  <- 페널티 소멸!
        #   -> "겨누고 다가갔다가 방향 버리고 공짜로 물러서기"를 반복하며 reach를 무한 수확함.
        if palm_cfg is not None:
            gate = palm_facing_object(env, palm_cfg, object_cfg, palm_normal_b)
            gate = gate_floor + (1.0 - gate_floor) * gate
            reward = torch.where(reward > 0.0, reward * gate, reward)
        return reward

def box_ground_clearance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
    surface_z: float = 0.0,
) -> torch.Tensor:
    """박스의 "최하 꼭짓점"이 받침면에서 뜬 높이. 한 모서리라도 닿아 있으면 0.

    중심 높이는 lift 신호가 아님: 바닥의 큐브를 짜면 모서리로 세워져 중심만 몇 mm 올라감.
    (실측: 중심 +4.28mm인데 최하 모서리는 -0.04mm로 바닥) 최하점을 봐야 진짜 들었을 때만 지급됨.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    half = torch.tensor(object_half_extent, dtype=torch.float, device=env.device)
    # 박스 자체 좌표계에서의 8개 꼭짓점
    signs = torch.tensor(
        [
            [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0],
        ],
        dtype=torch.float,
        device=env.device,
    )
    corners_b = (signs * half).unsqueeze(0).expand(env.num_envs, -1, -1)  # (N, 8, 3)
    quat = obj.data.root_quat_w.unsqueeze(1).expand(-1, 8, -1)
    corners_w = obj.data.root_pos_w.unsqueeze(1) + quat_apply(quat, corners_b)
    lowest_z = corners_w[..., 2].min(dim=1).values
    return lowest_z - env.scene.env_origins[:, 2] - surface_z


def object_below_surface_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
    surface_z: float = 0.0,
    tolerance: float = 0.05,
) -> torch.Tensor:
    """Penalize forcing the object below its support surface. Range [-1, 0]."""
    clearance = box_ground_clearance(env, object_cfg, object_half_extent, surface_z)
    depth = torch.clamp(-clearance, min=0.0) / tolerance
    return -torch.clamp(depth, max=1.0)


def object_lift_in_cage(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
    num_points: int = 3,
    sphere_radius: float = 0.005,
    depth_max: float = 0.02,
    lift_height: float = 0.08,
    surface_z: float = 0.0,
) -> torch.Tensor:
    """물체를 받침면에서 띄우는 것을 보상. 단, 손가락이 감싸고 있는 동안만 (논문 r_lift).

    "어떤 자세를 진짜 파지로 인정할지" 결정하는 항. 들지 못하는 자세는 파지가 아니므로,
    자세를 지정할 필요 없이 하중을 견디는지만 물으면 됨. 물리가 자세를 결정함.
    막아야 할 편법 2가지: (1) 중심 대신 "최하 모서리"를 봐야 기울이기가 안 통함.
    (2) cage_gate를 곱해야 "파지 없이 튕겨 올리기"가 안 통함. 조밀형이라 1mm 상승에도 gradient가 있음.
    """
    gate = object_in_finger_cage(
        env, asset_cfg, object_cfg, object_half_extent, num_points, sphere_radius, depth_max
    )
    clearance = box_ground_clearance(env, object_cfg, object_half_extent, surface_z)
    lift = torch.clamp(clearance, 0.0, lift_height) / lift_height
    return gate * lift


def arm_manipulability_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    j_max: float = 0.02,
) -> torch.Tensor:
    """팔이 특이점으로 접히는 것을 벌함 (논문 Eq.17, r_MP). 범위 [-1, 0].

    |J| = sqrt(det(J Jt)) (manipulability). j_max 위면 페널티 0, 특이점이면 -1.
    이게 없으면 "손바닥을 물체 쪽으로"를 만족시키는 가장 싼 방법이 "팔을 접어 손목만 돌리기"가 됨.
    접힌 팔은 손을 자유롭게 못 움직여서 물체에 영영 못 감. (실측: manip이 초기 57% -> 13%로 추락,
    큐브 31cm 앞에서 정지). asset_cfg에 body_names=["palm_link"], joint_names=["joint[0-5]"] 필요.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]  # type: ignore
    joint_ids = asset_cfg.joint_ids  # type: ignore

    # fixed-base articulation은 jacobian에 root body가 빠져 있어서 body i가 i-1행에 있음
    jac = asset.root_physx_view.get_jacobians()[:, body_id - 1, :, :][:, :, joint_ids]  # (N, 6, n)
    manip = torch.sqrt(torch.clamp(torch.det(jac @ jac.transpose(1, 2)), min=0.0))

    ratio = torch.clamp(manip, max=j_max) / j_max
    return 1.0 - 2.0 / (1.0 + ratio**3)


def hand_floor_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    clearance: float = 0.02,
    surface_z: float = 0.0,
) -> torch.Tensor:
    """손이 바닥/받침면을 파고드는 것을 벌함. 범위 [-1, 0].

    surface_z + clearance 아래로 내려간 깊이에 비례해 감점함.
    최대가 0이라 "높이 떠 있기"를 유도하진 않고, 받침면을 파고드는 것만 막음.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (N, B)
    limit_z = env.scene.env_origins[:, 2].unsqueeze(-1) + surface_z + clearance
    depth = (limit_z - z).clamp(min=0.0) / clearance  # 파고든 깊이, [0, 1]
    return -depth.clamp(max=1.0).max(dim=-1).values


def palm_facing_object(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    palm_normal_b: tuple[float, float, float] = (0.19, 0.28, 0.94),
) -> torch.Tensor:
    """물체가 손의 "파지 개구부" 안에 있는가. [0, 1]. 1이면 엄지-손가락 사이 정면.

    이름과 달리 palm_normal_b는 손바닥 법선이 **아님**. 물체가 엄지와 손가락 사이로 들어가는
    공간의 방향임. 둘은 65도 어긋나 있고, 파지의 조건은 후자임.
    실측(palm_link 로컬): 손바닥 법선 = (0.965, -0.008, 0.262) ~= +x.
    손가락은 +z로 뻗고(편 상태 검지 z=0.195) 오므리면 한 점(0.065, 0.006, 0.097)으로 모임
    -> 그 사이 공간 = 물체가 들어갈 자리 = (0.19, 0.28, 0.94).
    손바닥 법선(1,0,0)을 쓰면 hold와 상관계수 +0.003(무상관)이라 오히려 파지를 방해함.

    cage 항이 못 보는 것을 봄: 엄지-손가락 선분은 손 방향과 무관하게 물체를 관통할 수 있어서
    "손 옆에 물체를 끼고만 있는" 자세도 cage 만점을 받음.
    축 하나만 제약하고 roll은 자유 -> 대칭 물체의 파지 방식을 고르지 않음.
    이 함수 자체는 metric용. reward로 쓸 땐 반드시 차분형(PalmFacingProgressReward).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    body_id = asset_cfg.body_ids[0]  # type: ignore
    palm_pos_w = asset.data.body_state_w[:, body_id, :3]
    palm_quat_w = asset.data.body_state_w[:, body_id, 3:7]

    normal_b = torch.tensor(palm_normal_b, dtype=torch.float, device=env.device)
    normal_b = normal_b / torch.clamp(torch.norm(normal_b), min=1e-6)
    normal_w = quat_apply(palm_quat_w, normal_b.expand(env.num_envs, 3))

    to_obj = obj.data.root_pos_w - palm_pos_w
    to_obj = to_obj / torch.clamp(torch.norm(to_obj, dim=-1, keepdim=True), min=1e-6)
    return torch.clamp(torch.sum(normal_w * to_obj, dim=-1), 0.0, 1.0)


class PalmFacingProgressReward(ManagerTermBase):
    """손바닥을 물체 쪽으로 "돌리는 것"을 보상. 논문 r_hr과 같은 차분형.

    r(t) = facing(t) - facing(t-1), reset()에서 기준선 seeding.
    총합이 facing(final) - facing(reset)로 telescoping됨 -> 가만히 있으면 정확히 0, 정렬이 깨지면 감점.
    절대형으로 넣었다가 대참사: 겨누기는 접근보다 훨씬 싸서 weight 0.5에서 전체 보상의 98%를 먹고,
    정책이 팔을 접어 31cm 밖에서 겨누기만 함 (manip 13%까지 추락).
    논문의 거의 모든 항이 차분형인 이유가 이것 (절대형은 r_hold 하나뿐). 차분형은 farming 불가.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_facing = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    def _facing(self) -> torch.Tensor:
        p = self.cfg.params
        return palm_facing_object(
            self._env,
            p["asset_cfg"],
            p["object_cfg"],
            p.get("palm_normal_b", (1.0, 0.0, 0.0)),
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # 기준선은 "리셋 자세"에서. cage progress와 같은 이유 (첫 액션이 기준선을 부풀리는 것 방지).
        self._previous_facing[env_ids] = self._facing()[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        palm_normal_b: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> torch.Tensor:
        current = self._facing()
        progress = current - self._previous_facing
        self._previous_facing[:] = current
        return torch.clamp(progress, min=-1.0, max=1.0)


def object_to_target_position_tracking_bounded(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_pos: tuple[float, float, float] = (0.55, -0.05, 0.12),
    distance_max: float = 0.6,
    min_height: float = 0.08,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    target_pos_w = obj.data.root_pos_w.new_tensor(target_pos).unsqueeze(0)

    distance = torch.norm(obj.data.root_pos_w - target_pos_w, dim=1)
    distance_bonus = 1.0 - torch.clamp(distance, 0.0, distance_max) / distance_max
    lifted = obj.data.root_pos_w[:, 2] >= min_height
    return torch.where(lifted, distance_bonus, torch.zeros_like(distance_bonus))


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
