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


# TensorBoard reward 이름은 여기 함수명이 아니라 env_cfg_common.py의 RewTerm 필드명이 됨.
# 예: finger_cage_hold = RewTerm(func=mdp.object_in_finger_cage, ...)
#   -> Episode_Reward/finger_cage_hold
#   -> Episode_Reward_Raw/finger_cage_hold
# Metrics/cube/* 값들은 reward가 아니라 CustomRewardManager에서 따로 기록하는 진단값임.


# TensorBoard: current task reward로 직접 쓰이지 않음. command error helper 성격.
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


# TensorBoard: current task reward로 직접 쓰이지 않음. command error helper 성격.
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


# TensorBoard:
# - Indy-Wuji-Reach: Episode_Reward/end_effector_position_tracking
# - Indy-Wuji-Reach: Episode_Reward_Raw/end_effector_position_tracking
# env_cfg_common.py: RewardsCfg.end_effector_position_tracking 에서 연결됨.
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


# TensorBoard: 직접 reward 이름 없음. 아래 cage reward들의 공통 helper.
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
    # half_extent: (3,) 상수 또는 (N,3) env별 치수 (Box-Transport) — 둘 다 브로드캐스트
    if half_extent.dim() == 1:
        half_extent = half_extent.unsqueeze(0)
    q = local.abs() - half_extent.unsqueeze(1)
    outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.clamp(q.max(dim=-1).values, max=0.0) # max -> 제일 0에 가까우니까
    return outside + inside


# TensorBoard: 직접 reward 이름 없음. finger_cage_reach / finger_cage_hold / cube_lift 공통 helper.
def cage_points(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    num_points: int,
    point_fractions: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """엄지끝에서 각 대향 body로 선분을 긋고 그 위에 찍는 "파지 간극" 가상점.

    body_names는 [엄지끝, *대향] 순서여야 함 (preserve_order=True 필수).
    논문은 엄지-중지만 써서 6점이지만, 논문에는 r_grasp가 손 회전/손가락 관절각을 붙잡음.
    큐브엔 목표 파지가 없어 r_grasp를 못 쓰므로, 6점만 쓰면 검지가 자유가 되어
    "손바닥이 하늘 + 검지·중지 교차" 자세로도 만점이 나옴 (2026-07-11 실측). 그래서 검지도 포함해 12점.

    point_fractions: 선분 위 점 위치 (0=엄지끝, 1=대향 body). 주면 num_points 무시.
    기본(내부 등분 [0.25,0.5,0.75])은 중앙점이 헐렁한 새장에서도 큐브 깊숙이 박혀 포화됨
    -> 손끝이 표면에서 2~3cm 떠도 hold 고점 (2026-07-14 실측: 간격 10cm에서도 고점).
    끝쪽으로 치우친 값(예: 0.1/0.9)을 주면 손끝을 표면까지 가져가야 점수가 나옴.
    1.0까지 보내지 말 것: tip_link 원점은 패드가 아니라 마지막 관절이라 (패드는 2~3cm 앞),
    "원점이 표면에" = 패드가 큐브를 2cm 파고든 상태임.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids, :3]  # type: ignore / 모든 env에 대해 CAGE_BODIES에 해당하는 body 5개만 골라서 그 body들의 position xyz만 가져와라
    if body_pos_w.shape[1] < 2:
        raise ValueError(
            f"cage terms expect [thumb_tip, *opposing] (2 bodies minimum), got {body_pos_w.shape[1]}."
        )
    thumb = body_pos_w[:, 0]    # GPU tensor form / torch.tensor([1,2,3]) -> CPU tensor form
    opposing = body_pos_w[:, 1:]  # (N, M, 3)

    if point_fractions is not None:
        fractions = torch.tensor(point_fractions, dtype=thumb.dtype, device=thumb.device)
    else:
        # `num_points` equidistant points strictly between the thumb tip and each opposing body.
        fractions = torch.arange(1, num_points + 1, dtype=thumb.dtype, device=thumb.device) / (num_points + 1) # num_points=3 -> arrange(1,4) -> [1,2,3]/4 -> [0.25, 0.5, 0.75]
    span = opposing - thumb.unsqueeze(1)  # (N, M, 3)
    points = thumb[:, None, None, :] + span.unsqueeze(2) * fractions.view(1, 1, -1, 1)
    return points.reshape(thumb.shape[0], -1, 3)


# TensorBoard: 직접 reward 이름 없음. finger_cage_reach / finger_cage_hold / cube_lift 공통 helper.
def _cage_sdf(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    num_points: int,
    point_fractions: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """가상점 -> 물체 표면까지의 signed distance. (N, 대향body수 * num_points)"""
    obj: RigidObject = env.scene[object_cfg.name]
    points = cage_points(env, asset_cfg, num_points, point_fractions)
    # env별 치수 버퍼(randomize_box_dims)가 있으면 우선. 없으면 상수 (큐브 태스크 경로 불변)
    half = getattr(env, "box_half_extents", None)
    if half is None:
        half = torch.tensor(object_half_extent, dtype=points.dtype, device=points.device)
    return _box_signed_distance(points, obj.data.root_pos_w, obj.data.root_quat_w, half)


# TensorBoard:
# - Indy-Wuji-Cube-Grasp: Episode_Reward/finger_cage_hold
# - Indy-Wuji-Cube-Grasp: Episode_Reward_Raw/finger_cage_hold
# env_cfg_common.py: CubeGraspRewardsCfg.finger_cage_hold 에서 연결됨.
def object_in_finger_cage(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
    num_points: int = 3,
    sphere_radius: float = 0.005,
    depth_max: float = 0.02,
    point_fractions: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """물체를 손가락 사이에 가두는 것을 보상 (논문 Eq.15, hold).

    가상점을 반지름 sphere_radius의 구로 보고, 그 구가 물체를 파고든 깊이를 보상함.
    손을 오므리면 점들이 물체 안으로 들어가므로 "오므리기"가 직접 보상됨. 접촉센서 불필요.
    거리 reward는 정반대임: 만지면 물체가 밀려나 거리가 늘고 감점 -> 접촉이 손해 -> hover만 함.

    depth_max: 이만큼 파고들면 포화. 끝쪽 point_fractions와 쓸 때는 작게(예: 0.005) 둘 것 —
    크면 접촉 후에도 "더 조여라"가 남는데, 이 손은 40%+ 오므리면 간격이 2.8cm까지 줄며
    6cm 큐브를 수박씨처럼 짜냄 (2026-07-14 실측). "닿으면 만족"으로 포화시키는 게 안전함.
    """
    sdf = _cage_sdf(env, asset_cfg, object_cfg, object_half_extent, num_points, point_fractions)
    # 구가 물체를 파고든 깊이. sphere_radius보다 멀면 0, depth_max만큼 들어가면 포화(1).
    penetration = sphere_radius - sdf
    return torch.clamp(penetration / (sphere_radius + depth_max), 0.0, 1.0).mean(dim=1)


# TensorBoard:
# - Indy-Wuji-Cube-Grasp: Episode_Reward/finger_cage_reach
# - Indy-Wuji-Cube-Grasp: Episode_Reward_Raw/finger_cage_reach
# env_cfg_common.py: CubeGraspRewardsCfg.finger_cage_reach 에서 연결됨.
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
            p.get("point_fractions"),
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
        point_fractions: tuple[float, ...] | None = None,
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

# TensorBoard:
# - reward 이름으로는 직접 기록되지 않음.
# - CustomRewardManager metric: Metrics/cube/cube_clearance, Metrics/cube_final/cube_clearance 등에서 사용됨.
# - object_lift_in_cage 내부 helper로도 사용됨.
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
    half = getattr(env, "box_half_extents", None)
    if half is None:
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
    if half.dim() == 1:
        corners_b = (signs * half).unsqueeze(0).expand(env.num_envs, -1, -1)  # (N, 8, 3)
    else:
        corners_b = signs.unsqueeze(0) * half.unsqueeze(1)  # env별 치수 -> (N, 8, 3)
    quat = obj.data.root_quat_w.unsqueeze(1).expand(-1, 8, -1)
    corners_w = obj.data.root_pos_w.unsqueeze(1) + quat_apply(quat, corners_b)
    lowest_z = corners_w[..., 2].min(dim=1).values
    return lowest_z - env.scene.env_origins[:, 2] - surface_z


# TensorBoard:
# - 현재 7/13 복원 CubeGraspRewardsCfg에는 active term 아님.
# - 예전 cube_support term을 켜면 Episode_Reward/cube_support 로 기록될 수 있는 helper.
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


# TensorBoard:
# - Indy-Wuji-Cube-Grasp: Episode_Reward/cube_lift
# - Indy-Wuji-Cube-Grasp: Episode_Reward_Raw/cube_lift
# env_cfg_common.py: CubeGraspRewardsCfg.cube_lift 에서 연결됨.
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
    point_fractions: tuple[float, ...] | None = None,
) -> torch.Tensor:
    """물체를 받침면에서 띄우는 것을 보상. 단, 손가락이 감싸고 있는 동안만 (논문 r_lift).

    "어떤 자세를 진짜 파지로 인정할지" 결정하는 항. 들지 못하는 자세는 파지가 아니므로,
    자세를 지정할 필요 없이 하중을 견디는지만 물으면 됨. 물리가 자세를 결정함.
    막아야 할 편법 2가지: (1) 중심 대신 "최하 모서리"를 봐야 기울이기가 안 통함.
    (2) cage_gate를 곱해야 "파지 없이 튕겨 올리기"가 안 통함. 조밀형이라 1mm 상승에도 gradient가 있음.
    """
    gate = object_in_finger_cage(
        env, asset_cfg, object_cfg, object_half_extent, num_points, sphere_radius, depth_max,
        point_fractions,
    )
    clearance = box_ground_clearance(env, object_cfg, object_half_extent, surface_z)
    lift = torch.clamp(clearance, 0.0, lift_height) / lift_height
    return gate * lift


# TensorBoard:
# - Episode_Reward/cube_transport 로 기록됨.
# env_cfg_common.py: CubeGraspRewardsCfg.cube_transport 에서 연결됨.
class ObjectToGoalProgressReward(ManagerTermBase):
    """운반 층 (논문 orient(500) 자리): 잡은 채 goal 거리 "신기록"을 깬 만큼 포텐셜 차분 지불.

    - 역수 포텐셜, 전 구간 (2026-07-16, 사용자 설계): φ(d) = eps/(eps + d).
      reward = (φ(현재) − φ(베스트))⁺ × gate. 총액 = φ(최소거리) − φ(시작거리) 고정 → farming 불가.
      "가까울수록 크게"가 연속으로 구현됨: 원거리(30→10cm) ≈ +14, 근거리(10cm→중심) ≈ +89 (w4000).
    - 창(window)은 뒀다가 제거함 (2026-07-16): 스폰 랜덤 ±6/8cm 때문에 "높이는 맞는데
      횡으로 창 밖"인 에피소드가 transport 무신호가 되는 구멍 — 전 구간 역수가 상위호환.
    - best-so-far + 단일 부호 (2026-07-15): 후퇴/왕복/재접근 0원. 낙하 비용은 별도 순수
      페널티(drop_penalty)가 담당 — "페널티면 페널티, 리워드면 리워드" 분리.
    - 도입 이력: 선형 best-so-far는 goal 근처 1cm와 먼 1cm가 같은 값이라 과녁 통과(오버슈트,
      clearance 0.70m 실측)를 못 막았음. 역수형이 근접 흡인을 집중시킴.
    - gate(잡음, 연속값)를 곱함: 잡지 않고 밀거나 던져서 접근시킨 건 지불 안 함.
    - 기준선은 리셋 후 '첫 호출'에서 seeding: reward reset(managers 순서 375)이 command
      resample(381)보다 먼저라 reset()에서 잡으면 이전 에피소드 goal로 오염됨. 첫 스텝
      보상 0은 무해 — 리셋 직후 손은 물체에서 떨어져 있어 첫 액션으로 물체를 못 움직임.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best_phi = torch.zeros(env.num_envs, device=env.device)
        self._pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._pending[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
        num_points: int = 3,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        point_fractions: tuple[float, ...] | None = None,
        potential_eps: float = 0.05,
    ) -> torch.Tensor:
        obj: RigidObject = env.scene[object_cfg.name]
        goal_w = env.scene.env_origins + env.command_manager.get_command(command_name)
        dist = torch.norm(goal_w - obj.data.root_pos_w, dim=1)

        phi = potential_eps / (potential_eps + dist)

        self._best_phi = torch.where(self._pending, phi, self._best_phi)
        self._pending[:] = False
        progress = torch.clamp(phi - self._best_phi, min=0.0)  # 포텐셜 신기록 갱신분만
        self._best_phi = torch.maximum(self._best_phi, phi)

        gate = object_in_finger_cage(
            env, asset_cfg, object_cfg, object_half_extent, num_points, sphere_radius,
            depth_max, point_fractions,
        )
        return progress * gate


# ⚠ 미배선 (2026-07-16): "B 설계" 카드용 부품 — 어느 태스크에도 아직 연결 안 됨.
# B 설계 = lift 은퇴(weight 0, 항은 metrics surface_z 배선 때문에 유지) + transport 일시불 제거
#          + 이 연금 하나로 통합 (gate × φ(d), w~75). 적용은 사용자 신호 대기.
# 연결 시 TensorBoard: Episode_Reward/goal_proximity.
def object_goal_proximity(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
    num_points: int = 3,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
    point_fractions: tuple[float, ...] | None = None,
    potential_eps: float = 0.05,
) -> torch.Tensor:
    """잡은 채(gate) goal에 가까울수록 매 스텝 더 주는 근접 연금 (2026-07-16 통합 설계, 사용자).

    r = gate × φ(d), φ = eps/(eps+d) ∈ (0,1]. lift(높이 연금)와 transport(일시불)를 한 항으로
    통합: goal이 공중이라 "공중 유지"와 "error 축소"가 같은 지급으로 표현됨.
    - 일시불이 없어서 "스침 후 캠핑" 해킹이 불가 — 머물러야만 벌고, 공 안 체류는 0.5s 뒤
      success 종료가 회수 (연금의 캠핑 상한)
    - 크기 안전 근거 (w75, γ=0.99): goal 중심 ~2.5/스텝, 공 바깥 현재가치 ~110 ≪ r_T +500,
      탁자 캠핑 ~0.3/스텝 ≪ 공중 유지 1.5+ (lift의 "공중 유지 연금" 역할 대체)
    - 절대형 양수인데 허용되는 이유: "유지가 어려운 것"(공중 파지 유지) 원칙 충족 + gate 곱
      ("안 잡고 근처 서성"은 0원) + success 종료 마개
    """
    obj: RigidObject = env.scene[object_cfg.name]
    goal_w = env.scene.env_origins + env.command_manager.get_command(command_name)
    dist = torch.norm(goal_w - obj.data.root_pos_w, dim=1)
    gate = object_in_finger_cage(
        env, asset_cfg, object_cfg, object_half_extent, num_points, sphere_radius,
        depth_max, point_fractions,
    )
    return gate * potential_eps / (potential_eps + dist)


# TensorBoard:
# - Episode_Termination/success 로 기록됨 (DoneTerm 필드명 기준)
# - CubeGraspRewardsCfg.transport_success(is_terminated_term)가 이 판정을 한 방 보상으로 변환함.
# env_cfg_common.py: CubeGraspTerminationsCfg.success 에서 연결됨.
class ObjectAtGoalHeld(ManagerTermBase):
    """r_T (운반판): 큐브가 goal 반경 안 + gate 물림을 hold_steps 연속 유지 -> 성공 종료.

    goal이 공중(상판 +10cm 이상, cube_grasp_env_cfg의 커맨드 범위)이라 "들었음"은 자동 함의됨.
    순간 통과(던지기)는 유지 조건에서, 잡지 않은 받치기 등은 gate에서 걸러짐.
    성공 시 즉시 종료가 hold/lift 연금의 마개 (앉아서 버는 상한 << 성공 한 방).
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._count[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
        num_points: int = 3,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        point_fractions: tuple[float, ...] | None = None,
        goal_radius: float = 0.05,
        gate_threshold: float = 0.3,
        hold_steps: int = 15,
    ) -> torch.Tensor:
        obj: RigidObject = env.scene[object_cfg.name]
        goal_w = env.scene.env_origins + env.command_manager.get_command(command_name)
        dist = torch.norm(goal_w - obj.data.root_pos_w, dim=1)
        gate = object_in_finger_cage(
            env, asset_cfg, object_cfg, object_half_extent, num_points, sphere_radius,
            depth_max, point_fractions,
        )
        ok = (dist < goal_radius) & (gate > gate_threshold)
        self._count = torch.where(ok, self._count + 1, torch.zeros_like(self._count))
        return self._count >= hold_steps



# TensorBoard:
# - Indy-Wuji-Cube-Grasp: Episode_Reward/arm_manipulability
# - Indy-Wuji-Cube-Grasp: Episode_Reward_Raw/arm_manipulability
# env_cfg_common.py: CubeGraspRewardsCfg.arm_manipulability 에서 연결됨.
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


# TensorBoard:
# - Indy-Wuji-Cube-Grasp: Episode_Reward/hand_floor
# - Indy-Wuji-Cube-Grasp: Episode_Reward_Raw/hand_floor
# env_cfg_common.py: CubeGraspRewardsCfg.hand_floor 에서 연결됨.
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


# TensorBoard:
# - reward 이름으로는 직접 기록되지 않음.
# - PalmFacingProgressReward 내부 raw 계산으로 쓰이면 Episode_Reward/palm_facing 에 반영됨.
# - ObjectCageProgressReward의 양수 progress gate로 쓰이면 Episode_Reward/finger_cage_reach 에 반영됨.
# - CustomRewardManager metric: Metrics/cube/palm_facing, Metrics/cube_final/palm_facing 등과 같은 개념.
def palm_facing_object(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    palm_normal_b: tuple[float, float, float] = (0.19, 0.28, 0.94),
) -> torch.Tensor:
    """물체가 손의 "파지 개구부" 안에 있는가. [0, 1]. 1이면 엄지-손가락 사이 정면.

    ★ palm_normal_b (0.19, 0.28, 0.94)의 정체와 도출 (2026-07-16 보강 — 이름이 misnomer임):
    이건 손바닥 평면의 법선이 아니라 **"파지 개구부 축"** = 오므린 손끝이 수렴하는 지점을
    향하는 방향이며, 좌표는 전부 **palm_link 로컬 프레임** 기준임. 도출에 쓰인 실측 3개:
      ① 손바닥 평면 법선 실측:      (0.965, -0.008, 0.262) ≈ 로컬 +x
      ② 편 손가락이 뻗는 방향:       ≈ 로컬 +z (편 상태 검지끝 z=0.195)
      ③ 오므릴 때 손끝 수렴점:       (0.065, 0.006, 0.097) 부근 (palm 원점 기준)
    물체가 "오므리면 잡히는" 위치 = 손바닥 앞(+x 성분)이면서 손끝이 모여드는 쪽(+z 위주)의
    사이 공간이고, 그 공간의 중심 방향을 실측 평균한 단위벡터가 (0.19, 0.28, 0.94).
    손바닥 법선(①)과는 65° 어긋남 — Wuji 손은 엄지 배치가 비대칭이라 "손바닥이 보는 곳"과
    "오므림이 닿는 곳"이 다름. gate의 질문은 후자이므로 후자를 씀.
    ⚠ 정확한 평균 산식/측정 스크립트는 유실됨 (2026-07-12경 1회성 probe). 재검증은
    `play.py --show_palm_vectors`(개구부 축 화살표가 물체를 향하는지 눈 확인), 재도출이
    필요하면 hand_geometry.py 방식 FK로 오므림 전후 손끝 변위 방향을 다시 잴 것.
    ⚠ managers.py의 _palm_normal_b와 반드시 동일해야 함 (metric-reward 축 불일치 방지).

    역사적 함정: 손바닥 법선(1,0,0)을 gate 축으로 쓰면 hold와 상관 +0.003(무상관) —
    "손바닥은 마주보는데 오므리면 비껴가는" 자세가 통과해 오히려 파지를 방해함.

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


# TensorBoard:
# - Indy-Wuji-Cube-Grasp: Episode_Reward/palm_facing
# - Indy-Wuji-Cube-Grasp: Episode_Reward_Raw/palm_facing
# env_cfg_common.py: CubeGraspRewardsCfg.palm_facing 에서 연결됨.
class PalmFacingProgressReward(ManagerTermBase):
    """손바닥(파지 개구부)을 물체 쪽으로 "돌리는 것"을 보상. best-so-far 차분 (+ 전용).

    r(t) = (facing(t) - 에피소드 최고 facing)⁺, 신기록 갱신 시에만 지급 + best 갱신.
    총합 = facing(최고) - facing(reset)로 고정 -> farming 불가, 단일 부호
    (2026-07-16 사수님 원칙 통일, 사용자 결정. 이전 ± 차분에서 전환 — 행동 영향은 미미:
    시작 자세가 이미 facing 0.987이라 이 항의 에피소드 예산이 ~0.002이고, 겨눔 "유지"의
    실질 인센티브는 이 항이 아니라 reach의 facing gate가 담당).
    ⚠ (curr − prev)⁺ 단순 클램프는 금지: 하락 무과금 + 상승 재적립이라 손목 진동이
    화폐 발행기가 됨 (왕복마다 재지급). 신기록 조건이 재적립을 차단함.
    절대형은 더 금지: weight 0.5에서 전체 보상의 98%를 먹고 정책이 팔을 접어 31cm 밖에서
    겨누기만 한 대참사 이력 (manip 13%까지 추락).
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best_facing = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

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
        # (palm_facing은 command 무관이라 transport와 달리 reset()에서 바로 seeding 가능)
        self._best_facing[env_ids] = self._facing()[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        palm_normal_b: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> torch.Tensor:
        current = self._facing()
        progress = torch.clamp(current - self._best_facing, min=0.0)  # 신기록 갱신분만
        self._best_facing = torch.maximum(self._best_facing, current)
        return progress


# TensorBoard:
# - 현재 active cube grasp/reach cfg에는 연결되지 않음.
# - 예전/실험용 object goal term을 켜면 cube_goal_tracking 류 이름으로 기록될 수 있음.
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


# TensorBoard:
# - 현재 Indy-Wuji-Reach에서는 비활성 처리됨.
# - 켜면 Episode_Reward/end_effector_orientation_tracking 으로 기록됨.
# - Dual-arm reach cfg에서는 left/right term 이름으로 기록될 수 있음.
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


# TensorBoard:
# - 현재 Indy-Wuji-Reach에서는 비활성 처리됨.
# - 켜면 Episode_Reward/end_effector_speed 로 기록됨.
def end_effector_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the end-effector speed using L2-norm.

    The function computes the end-effector speed as the L2-norm of the end-effector's speed.
    """

    asset: RigidObject = env.scene[asset_cfg.name]

    speed = torch.abs(asset.data.body_state_w[:, asset_cfg.body_ids[0], 7:10])
    return torch.norm(speed, dim=1)


# TensorBoard:
# - 현재 Indy-Wuji-Reach에서는 비활성 처리됨.
# - 켜면 Episode_Reward/joint_vel 로 기록됨.
def finite_joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L1-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the L1 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset._finite_joint_vel[:, asset_cfg.joint_ids]), dim=1)


# TensorBoard:
# - 현재 active cfg에는 연결되지 않음.
# - 켜면 Episode_Reward/action_second_rate 로 기록됨.
def action_second_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    # TODO: currently broken
    return torch.sum(
        torch.square(
            (env.action_manager.action - env.action_manager.prev_action)
            - (env.action_manager.prev_action - env.action_manager.prevprev_action)
        ),
        dim=1,
    )
