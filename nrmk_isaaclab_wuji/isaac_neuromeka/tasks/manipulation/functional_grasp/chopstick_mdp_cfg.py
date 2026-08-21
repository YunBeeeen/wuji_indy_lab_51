"""Standalone MDP configuration for one-stick Skill-A functional grasp.

This task deliberately does not inherit Box-Transport MDP cfg.  It keeps the
validated 12-point cage and 18-D action semantics, but has its own observations,
rewards, success condition, and log directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp
from isaac_neuromeka.tasks.manipulation.functional_grasp import mdp as fg_mdp
from isaac_neuromeka.mdp.rewards import box_ground_clearance, square_prism_ori_error


CHOPSTICK_CAGE_BODIES = SceneEntityCfg(
    "robot",
    body_names=[
        "finger1_tip_link",
        "finger2_tip_link",
        "finger2_link3",
        "finger3_tip_link",
        "finger3_link3",
    ],
    preserve_order=True,
)

CHOPSTICK_INDEX_CAGE_BODIES = SceneEntityCfg(
    "robot",
    body_names=["finger1_tip_link", "finger2_tip_link", "finger2_link3"],
    preserve_order=True,
)

CHOPSTICK_MIDDLE_CAGE_BODIES = SceneEntityCfg(
    "robot",
    body_names=["finger1_tip_link", "finger3_tip_link", "finger3_link3"],
    preserve_order=True,
)

PALM_CFG = SceneEntityCfg("robot", body_names=["palm_link"])
THUMB_CFG = SceneEntityCfg("robot", body_names=["finger1_tip_link"])
INDEX_CFG = SceneEntityCfg("robot", body_names=["finger2_tip_link"])
MIDDLE_CFG = SceneEntityCfg("robot", body_names=["finger3_tip_link"])
RING_CFG = SceneEntityCfg("robot", body_names=["finger4_tip_link"])
PINKY_CFG = SceneEntityCfg("robot", body_names=["finger5_tip_link"])
OBJECT_CFG = SceneEntityCfg("cube")

STICK_HALF_EXTENT = (0.007, 0.09, 0.0035)  # 14×7 단면(두께 7mm) 2026-07-30 (STICK_SIZE와 동반)
POINT_FRACTIONS = (0.1, 0.5, 0.9)

# Semantic regions — 09-42-28 버전으로 복원 (2026-07-24 저녁).
# palm-down 재설계(tip/tail 뒤집기 + 엄지·중지 스왑)를 play로 진단한 결과 **중지가 벌어진 채 누르고
# 엄지·검지 2점 pinch**로 잡음 — 원하는 palm-down tripod가 안 나옴. success 0.83 나왔던 검증된
# 파지(09-42-28)로 되돌리고, 파지는 고정한 채 **랜덤 pose(roll/yaw)만** 도전하기로.
#   tip=+y, tail=-y (파지=음수 axial). 엄지 +x / 중지 -x / 검지 +z.
# (복구용) palm-down 재설계 값: INDEX axial(0.20,0.70)+z / THUMB axial(0.15,0.65)−x /
#   MIDDLE axial(0.15,0.65)+x. 접촉 보상(r_contact) 갖춰지면 재시도.
INDEX_GRIP_REGION = {
    "long_axis": 1,
    "axial_region": (-0.30, 0.0),
    "surface_axis": 2,
    "surface_sign": 1.0,
    "surface_offset": 0.0,
    "surface_tolerance": 0.005,
}

THUMB_GRIP_REGION = {
    "long_axis": 1,
    "axial_region": (-0.20, 0.20),
    "surface_axis": 0,
    "surface_sign": -1.0,
    "surface_offset": 0.0,
    "surface_tolerance": 0.005,
}

MIDDLE_GRIP_REGION = {
    "long_axis": 1,
    "axial_region": (-0.20, 0.20),
    "surface_axis": 0,
    "surface_sign": 1.0,
    "surface_offset": 0.0,
    "surface_tolerance": 0.005,
}

# ── Box-Transport에서 이식: world goal pose 운반 + orientation 매칭 (2026-07-22) ──
# 이미 lift가 되는 chopstick functional grasp 위에 "world 목표 pose로 운반 + 자세 매칭"을 얹음.
# box_mdp_cfg의 Balanced* 구조를 그대로 복사하되, orientation만 sticky latch 대신
# "매 스텝 pos·gate 조건이 참일 때만 지급"하는 결합형으로 교체함 (드리프트 차단, 2026-07-22 설계).


def _goal_pose_from_command(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """7D pose command에서 goal 위치(world)와 quaternion(world)을 뽑는다."""
    command = env.command_manager.get_command(command_name)
    goal_pos_w = env.scene.env_origins + command[:, :3]
    if command.shape[-1] >= 7:
        goal_quat_w = command[:, 3:7]
    else:
        goal_quat_w = torch.zeros(
            command.shape[0], 4, device=command.device, dtype=command.dtype
        )
        goal_quat_w[:, 0] = 1.0
    return goal_pos_w, goal_quat_w


# ── 에피소드 내 goal 재샘플 대응 (2026-07-24, 사수님 지시) ──────────────────────
# resampling_time_range=(5,10)이면 8초 에피소드 도중 goal이 바뀔 수 있음(안 바뀔 수도).
# best-so-far 보상(transport/orientation/fine)은 기준선(_best_*)이 옛 goal에 고착되므로,
# goal이 바뀐 env를 감지해 기준선을 재장전해야 새 goal을 따라간다.
# step 순서상 재샘플은 reward '이후'(manager_based_rl_env.py:208→232)라 다음 스텝에 감지됨.
def _goal_key(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """goal의 위치(3)+자세(4)를 합친 7D pose 키. 재샘플 감지는 이 키로 해야 position 고정 +
    orientation만 랜덤인 경우(현재 chopstick)도 잡힌다. position만 비교하면 ori 재샘플을 놓침."""
    goal_w, goal_quat_w = _goal_pose_from_command(env, command_name)
    return torch.cat([goal_w, goal_quat_w], dim=-1)


def _goal_changed(term, goal_key: torch.Tensor) -> torch.Tensor:
    """이전 스텝 goal(7D pose)과 비교해 바뀐 env를 True로. _goal_cache 없으면 생성(첫 호출 전부 False).
    ⚠ goal_key는 반드시 7D pose(_goal_key)를 넘길 것 — position만 넘기면 ori 재샘플을 못 잡음(2026-07-24 버그)."""
    cache = getattr(term, "_goal_cache", None)
    if cache is None or cache.shape != goal_key.shape:
        term._goal_cache = goal_key.clone()
        return torch.zeros(goal_key.shape[0], dtype=torch.bool, device=goal_key.device)
    changed = (cache - goal_key).abs().sum(dim=-1) > 1.0e-6
    term._goal_cache = goal_key.clone()
    return changed


# ── chopstick 전용 tip/tail 구분 대칭 (2026-07-22) ──
# 젓가락은 tip(+y)과 tail(-y)이 기능적으로 다름 → 길이축(y) 90° 회전 4개만 대칭.
# box의 square_prism 8-대칭은 끝-뒤집기(+y↔-y)를 포함해 tail-forward를 정답으로 인정하는데,
# 젓가락엔 틀림(tip이 goal 방향을 향해야 함) + 정답이 둘로 갈려 모호성. 그래서 4-대칭으로 좁힘.
# box 공유 함수(square_prism_ori_error, object_ori_error_nearest_sym)는 그대로 두고 여기만 씀.
_SQ2 = 0.7071067811865476
_STICK_TIP_SYMS = (
    (1.0, 0.0, 0.0, 0.0),      # identity
    (_SQ2, 0.0, _SQ2, 0.0),    # y축 90°
    (0.0, 0.0, 1.0, 0.0),      # y축 180°
    (_SQ2, 0.0, -_SQ2, 0.0),   # y축 270°  (전부 장축 둘레 회전 → 양 끝 보존, tip 방향 무관)
)
_stick_sym_cache: dict = {}


def _stick_syms(device, dtype) -> torch.Tensor:
    key = (device, dtype)
    syms = _stick_sym_cache.get(key)
    if syms is None:
        syms = torch.tensor(_STICK_TIP_SYMS, device=device, dtype=dtype)
        _stick_sym_cache[key] = syms
    return syms


def stick_tip_ori_error(quat_w: torch.Tensor, goal_quat_w: torch.Tensor) -> torch.Tensor:
    """tip/tail 구분 4-대칭 최소 자세 오차각 [rad] (chopstick 전용).

    box의 square_prism_ori_error와 동일한 geodesic 계산이나, 대칭을 4개(+y 보존)로만 최소화해
    tip 방향을 강제한다. tail-forward는 더 이상 정답으로 인정되지 않음.
    """
    syms = _stick_syms(quat_w.device, quat_w.dtype)
    candidates = math_utils.quat_mul(
        goal_quat_w.unsqueeze(1).expand(-1, syms.shape[0], -1),
        syms.unsqueeze(0).expand(quat_w.shape[0], -1, -1),
    )
    dots = torch.abs(torch.sum(quat_w.unsqueeze(1) * candidates, dim=-1))
    return 2.0 * torch.acos(torch.clamp(dots.max(dim=1).values, max=1.0))


def stick_ori_error_nearest_sym(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = OBJECT_CFG,
) -> torch.Tensor:
    """최근접 4-대칭 목표까지의 상대 회전 axis-angle (N,3) — chopstick obs (tip/tail 구분)."""
    obj = env.scene[object_cfg.name]
    quat = obj.data.root_quat_w
    _, goal_quat = _goal_pose_from_command(env, command_name)
    syms = _stick_syms(quat.device, quat.dtype)
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


class BalancedObjectToGoalProgressReward(ManagerTermBase):
    """Position best-so-far progress, balanced tripod grasp로 gate (box에서 이식)."""

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
        index_cage_cfg: SceneEntityCfg,
        middle_cage_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
        num_points: int = 3,
        point_fractions: tuple[float, ...] | None = None,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        potential_eps: float = 0.05,
    ) -> torch.Tensor:
        obj = env.scene[object_cfg.name]
        goal_w, _ = _goal_pose_from_command(env, command_name)
        distance = torch.norm(goal_w - obj.data.root_pos_w, dim=1)
        phi = potential_eps / (potential_eps + distance)
        # 에피소드 내 goal 재샘플 감지 → 그 env는 다시 pending(새 goal 기준선 재장전).
        self._pending = self._pending | _goal_changed(self, _goal_key(env, command_name))
        # command resample 순서 오염을 피하려 첫 호출/재샘플에서 기준선을 잡음.
        self._best_phi = torch.where(self._pending, phi, self._best_phi)
        self._pending[:] = False
        progress = torch.clamp(phi - self._best_phi, min=0.0)
        self._best_phi = torch.maximum(self._best_phi, phi)
        gate = fg_mdp.balanced_tripod_cage_gate(
            env=env,
            index_cage_cfg=index_cage_cfg,
            middle_cage_cfg=middle_cage_cfg,
            object_cfg=object_cfg,
            object_half_extent=object_half_extent,
            num_points=num_points,
            point_fractions=point_fractions,
            sphere_radius=sphere_radius,
            depth_max=depth_max,
        )
        return progress * gate


class BalancedObjectOrientationCoupledReward(ManagerTermBase):
    """Goal 자세 best-so-far progress를, "지금 이 스텝에 goal에 있고(잡은 채)"일 때만 지급.

    box의 sticky-latch 버전과 달리 latch를 유지하지 않는다. 매 스텝
    ``position_error < activation_distance`` AND ``gate > threshold``를 재확인해서,
    goal에 머문 채로 자세를 개선할 때만 양수를 준다. goal을 벗어나 자세만 맞추는 행위는
    0원이 되어 position과 orientation이 구조적으로 결합됨 (2026-07-22 사용자 설계).
    best_error는 valid 스텝에서만 갱신·유지(리셋 X)해 왕복 파밍을 막는다.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best_error = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self._seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    @property
    def active(self) -> torch.Tensor:
        return self._seen

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._best_error[env_ids] = 0.0
        self._seen[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        index_cage_cfg: SceneEntityCfg,
        middle_cage_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
        num_points: int = 3,
        point_fractions: tuple[float, ...] | None = None,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        activation_distance: float = 0.05,
        activation_gate_threshold: float = 0.3,
        angle_scale: float = 0.7853981633974483,
    ) -> torch.Tensor:
        obj = env.scene[object_cfg.name]
        goal_w, goal_quat_w = _goal_pose_from_command(env, command_name)
        position_error = torch.norm(goal_w - obj.data.root_pos_w, dim=1)
        orientation_error = stick_tip_ori_error(obj.data.root_quat_w, goal_quat_w)
        # (기존 8-대칭, tip/tail 무구분) orientation_error = square_prism_ori_error(obj.data.root_quat_w, goal_quat_w)
        gate = fg_mdp.balanced_tripod_cage_gate(
            env=env,
            index_cage_cfg=index_cage_cfg,
            middle_cage_cfg=middle_cage_cfg,
            object_cfg=object_cfg,
            object_half_extent=object_half_extent,
            num_points=num_points,
            point_fractions=point_fractions,
            sphere_radius=sphere_radius,
            depth_max=depth_max,
        )
        # 에피소드 내 goal 재샘플 감지 → 그 env는 seen 끔(새 goal 자세 기준선 재설정).
        self._seen = self._seen & (~_goal_changed(self, _goal_key(env, command_name)))
        # 지금 이 스텝에 goal 안 + 파지 중일 때만 유효.
        valid = (position_error < activation_distance) & (gate > activation_gate_threshold)
        newly = valid & (~self._seen)
        # 첫 valid 스텝: 기준선만 저장(보상 0).
        self._best_error = torch.where(newly, orientation_error, self._best_error)
        self._seen = self._seen | valid
        scale = max(float(angle_scale), 1.0e-6)
        progress = torch.clamp((self._best_error - orientation_error) / scale, 0.0, 1.0)
        active = valid & (~newly)
        reward = torch.where(active, progress * gate, torch.zeros_like(progress))
        # (c안 2026-07-22) best는 안·밖 모두 갱신, 지급은 active(goal 안)에서만.
        # 밖에서 회전 개선을 best에 저장했다 재진입 때 인출하는 왕복 파밍을 차단 —
        # 밖에서 오차가 낮아지면 best도 함께 내려가 재진입 progress=0이 됨.
        # 따라서 보상을 받으려면 "역대 최저 자세 갱신을 goal 안에 머문 채" 해야 함.
        self._best_error = torch.minimum(self._best_error, orientation_error)
        return reward


class BalancedObjectAtGoalHeld(ManagerTermBase):
    """Goal pose(±ori) + balanced tripod grasp를 hold_steps 연속 유지하면 성공 종료 (box 이식)."""

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
        index_cage_cfg: SceneEntityCfg,
        middle_cage_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
        num_points: int = 3,
        point_fractions: tuple[float, ...] | None = None,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        goal_radius: float = 0.05,
        gate_threshold: float = 0.3,
        hold_steps: int = 15,
        ori_limit: float | None = None,
    ) -> torch.Tensor:
        obj = env.scene[object_cfg.name]
        goal_w, goal_quat_w = _goal_pose_from_command(env, command_name)
        distance = torch.norm(goal_w - obj.data.root_pos_w, dim=1)
        gate = fg_mdp.balanced_tripod_cage_gate(
            env=env,
            index_cage_cfg=index_cage_cfg,
            middle_cage_cfg=middle_cage_cfg,
            object_cfg=object_cfg,
            object_half_extent=object_half_extent,
            num_points=num_points,
            point_fractions=point_fractions,
            sphere_radius=sphere_radius,
            depth_max=depth_max,
        )
        valid = (distance < goal_radius) & (gate > gate_threshold)
        if ori_limit is not None:
            valid &= stick_tip_ori_error(obj.data.root_quat_w, goal_quat_w) < ori_limit
            # (기존 8-대칭) valid &= square_prism_ori_error(obj.data.root_quat_w, goal_quat_w) < ori_limit
        self._count = torch.where(valid, self._count + 1, torch.zeros_like(self._count))
        return self._count >= hold_steps


class GoalReachedBonus(ManagerTermBase):
    """goal 도달(±ori) + 파지를 hold_steps 유지하면 **1회 보너스** 지급. 종료 안 함 (2026-07-24).

    사수님 지시로 success를 termination에서 분리: 성공해도 에피소드를 끝내지 않고, goal당 딱 1회
    보너스를 준다. 그래야 에피소드 내 goal 재샘플로 여러 goal을 연속 처리할 수 있음
    (success 종료면 첫 성공에 리셋돼 재샘플 기회가 사라짐).

    판정 조건은 `BalancedObjectAtGoalHeld`와 동일(pos<goal_radius & gate & ori<ori_limit, hold_steps 유지).
    - `_awarded`: 이 goal에서 이미 보너스를 줬는지 → 같은 goal 반복 성공 파밍 차단.
    - goal 재샘플(`_goal_changed`) 시 `_awarded`/`_count` 재장전 → 새 goal은 다시 성공 가능.
    반환은 bool이 아니라 float 보상(성공한 스텝에 1.0). weight로 크기 조절(weight/30 = 한 방 보너스).
    로깅: TerminationManager에서 빠지므로 Episode_Termination/success는 사라짐 — 성공 빈도는
    Episode_Reward_Raw/transport_success로 봄.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._awarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._count[env_ids] = 0
        self._awarded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        index_cage_cfg: SceneEntityCfg,
        middle_cage_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
        num_points: int = 3,
        point_fractions: tuple[float, ...] | None = None,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        goal_radius: float = 0.05,
        gate_threshold: float = 0.3,
        hold_steps: int = 15,
        ori_limit: float | None = None,
    ) -> torch.Tensor:
        obj = env.scene[object_cfg.name]
        goal_w, goal_quat_w = _goal_pose_from_command(env, command_name)
        # goal 재샘플 시 이 goal의 성공 카운트·지급 이력 재장전.
        changed = _goal_changed(self, _goal_key(env, command_name))
        self._awarded = self._awarded & (~changed)
        self._count = torch.where(changed, torch.zeros_like(self._count), self._count)
        distance = torch.norm(goal_w - obj.data.root_pos_w, dim=1)
        gate = fg_mdp.balanced_tripod_cage_gate(
            env=env,
            index_cage_cfg=index_cage_cfg,
            middle_cage_cfg=middle_cage_cfg,
            object_cfg=object_cfg,
            object_half_extent=object_half_extent,
            num_points=num_points,
            point_fractions=point_fractions,
            sphere_radius=sphere_radius,
            depth_max=depth_max,
        )
        valid = (distance < goal_radius) & (gate > gate_threshold)
        if ori_limit is not None:
            valid &= stick_tip_ori_error(obj.data.root_quat_w, goal_quat_w) < ori_limit
        self._count = torch.where(valid, self._count + 1, torch.zeros_like(self._count))
        newly_success = (self._count >= hold_steps) & (~self._awarded)
        self._awarded = self._awarded | newly_success
        return newly_success.float()


def balanced_object_goal_proximity(
    env: ManagerBasedRLEnv,
    command_name: str,
    index_cage_cfg: SceneEntityCfg,
    middle_cage_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    num_points: int = 3,
    point_fractions: tuple[float, ...] | None = None,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
    potential_eps: float = 0.05,
) -> torch.Tensor:
    """잡은 채(gate) goal 위치에 가까울수록 매 스텝 더 주는 근접 연금 (box에서 이식).

    r = gate × φ(위치거리), φ = eps/(eps+d) ∈ (0,1]. goal을 벗어나면 매 스텝 연금이 끊겨
    "머무는 것"에 붙잡는 힘을 줌. 기본 weight 0 (필요 시 활성화). 캠핑 상한은 success 종료가 담당.
    """
    obj = env.scene[object_cfg.name]
    goal_w, _ = _goal_pose_from_command(env, command_name)
    distance = torch.norm(goal_w - obj.data.root_pos_w, dim=1)
    gate = fg_mdp.balanced_tripod_cage_gate(
        env=env,
        index_cage_cfg=index_cage_cfg,
        middle_cage_cfg=middle_cage_cfg,
        object_cfg=object_cfg,
        object_half_extent=object_half_extent,
        num_points=num_points,
        point_fractions=point_fractions,
        sphere_radius=sphere_radius,
        depth_max=depth_max,
    )
    return gate * potential_eps / (potential_eps + distance)


class FineProximityReward(ManagerTermBase):
    """말단 정밀 수렴용 position 지수 커널 (2026-07-22). best-so-far, 잡은 채(gate)만.

    coarse transport(φ) 위에 얹는 뾰족한 항: K(d)=exp(-d/sigma)는 **d=0에서 기울기 최대**라
    마지막 mm를 조인다(Gaussian은 0에서 평평해 부적합). best-so-far라 왕복/캠핑 파밍 없음.
    ⚠ sigma는 success 반경(5cm)보다 작게 둘 것(임계 안쪽에서 refine). 기본 weight 0.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best_k = torch.zeros(env.num_envs, device=env.device)
        self._pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._pending[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        index_cage_cfg: SceneEntityCfg,
        middle_cage_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
        num_points: int = 3,
        point_fractions: tuple[float, ...] | None = None,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        sigma: float = 0.02,
    ) -> torch.Tensor:
        obj = env.scene[object_cfg.name]
        goal_w, _ = _goal_pose_from_command(env, command_name)
        distance = torch.norm(goal_w - obj.data.root_pos_w, dim=1)
        k = torch.exp(-distance / sigma)
        # 에피소드 내 goal 재샘플 감지 → pending 재장전(새 goal 기준선).
        self._pending = self._pending | _goal_changed(self, _goal_key(env, command_name))
        self._best_k = torch.where(self._pending, k, self._best_k)
        self._pending[:] = False
        progress = torch.clamp(k - self._best_k, min=0.0)
        self._best_k = torch.maximum(self._best_k, k)
        gate = fg_mdp.balanced_tripod_cage_gate(
            env=env,
            index_cage_cfg=index_cage_cfg,
            middle_cage_cfg=middle_cage_cfg,
            object_cfg=object_cfg,
            object_half_extent=object_half_extent,
            num_points=num_points,
            point_fractions=point_fractions,
            sphere_radius=sphere_radius,
            depth_max=depth_max,
        )
        return progress * gate


class FineOrientationReward(ManagerTermBase):
    """말단 정밀 수렴용 orientation 지수 커널 (2026-07-22). best-so-far, pos-게이팅 + (c).

    K(θ)=exp(-θ/sigma), θ=stick_tip_ori_error(4-대칭). **pos<activation_distance & gate일 때만** 지급
    (드리프트 차단, stick_orientation과 같은 커플링). best_k는 goal 안·밖 모두 갱신((c) 방식,
    밖-개선을 재진입 때 인출하는 왕복 파밍 차단). ⚠ sigma는 success 임계(15°=0.262)보다 작게. weight 0 기본.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best_k = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self._seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._best_k[env_ids] = 0.0
        self._seen[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        index_cage_cfg: SceneEntityCfg,
        middle_cage_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
        num_points: int = 3,
        point_fractions: tuple[float, ...] | None = None,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        sigma: float = 0.17453292519943295,
        activation_distance: float = 0.03,
        activation_gate_threshold: float = 0.3,
    ) -> torch.Tensor:
        obj = env.scene[object_cfg.name]
        goal_w, goal_quat_w = _goal_pose_from_command(env, command_name)
        position_error = torch.norm(goal_w - obj.data.root_pos_w, dim=1)
        orientation_error = stick_tip_ori_error(obj.data.root_quat_w, goal_quat_w)
        gate = fg_mdp.balanced_tripod_cage_gate(
            env=env,
            index_cage_cfg=index_cage_cfg,
            middle_cage_cfg=middle_cage_cfg,
            object_cfg=object_cfg,
            object_half_extent=object_half_extent,
            num_points=num_points,
            point_fractions=point_fractions,
            sphere_radius=sphere_radius,
            depth_max=depth_max,
        )
        k = torch.exp(-orientation_error / sigma)
        # 에피소드 내 goal 재샘플 감지 → seen 끔(새 goal 자세 기준선 재설정).
        self._seen = self._seen & (~_goal_changed(self, _goal_key(env, command_name)))
        valid = (position_error < activation_distance) & (gate > activation_gate_threshold)
        newly = valid & (~self._seen)
        self._best_k = torch.where(newly, k, self._best_k)  # 첫 valid에 기준선
        self._seen = self._seen | valid
        progress = torch.clamp(k - self._best_k, min=0.0)
        active = valid & (~newly)
        reward = torch.where(active, progress * gate, torch.zeros_like(progress))
        # (c) best_k는 안·밖 모두 갱신 → 밖에서 개선한 걸 재진입 때 인출 못 함
        self._best_k = torch.maximum(self._best_k, k)
        return reward


# ══════════════════════════════════════════════════════════════════════════════
# "손바닥에 담기" reframe (2026-07-31, 사용자 지시) — world-goal 운반을 대체.
# ──────────────────────────────────────────────────────────────────────────────
# 목표를 "world 목표 pose로 운반"에서 "스틱이 손바닥 안(palm-local 목표점)에 들어오고
# 손바닥이 하늘을 본 채(palm-up) 떨어지지 않기"로 바꿈. 유지해야 할 정밀 자세가 없어
# "잡아서 얼른 뒤집고 손바닥에 위치"가 자연스러운 최적해가 됨.
#
# palm-local 목표점 T: hand_setting 목표(world (0.065,0,0.5195), 손 root (0,0,0.50),
#   HAND_ROOT_ROT로 palm 법선=+z)를 palm_link 로컬로 변환 → world offset (0.065,0,0.0195)에
#   palm 회전 R^T 적용 = (0.02, 0, 0.065). 같은 Wuji 손 USD라 palm_link 로컬 프레임이 동일해
#   우리 로봇(Indy+Wuji)에도 그대로 유효.
# palm-up: palm_link 로컬 +x(=팜 법선)의 world z성분이 1에 가까울수록 손바닥이 하늘을 봄.
PALM_TARGET_LOCAL = (0.02, 0.0, 0.065)


def _stick_in_palm_local(
    env: ManagerBasedRLEnv,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """스틱 위치를 palm_link 로컬 프레임 상대변위로 (회전 불변) + palm world quat 반환."""
    robot = env.scene[palm_cfg.name]
    obj = env.scene[object_cfg.name]
    palm_id = palm_cfg.body_ids[0]
    palm_pos_w = robot.data.body_state_w[:, palm_id, :3]
    palm_quat_w = robot.data.body_state_w[:, palm_id, 3:7]
    disp_w = obj.data.root_pos_w - palm_pos_w
    disp_local = math_utils.quat_apply_inverse(palm_quat_w, disp_w)
    return disp_local, palm_quat_w


def _palm_up_cos(palm_quat_w: torch.Tensor) -> torch.Tensor:
    """palm 법선(로컬 +x)의 world z성분 ∈[-1,1]. 1=손바닥이 완전히 하늘을 봄(yaw 무관)."""
    x_axis = torch.zeros_like(palm_quat_w[:, :3])
    x_axis[:, 0] = 1.0
    normal_w = math_utils.quat_apply(palm_quat_w, x_axis)
    return normal_w[:, 2]


def _lift_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    surface_z: float,
    lo: float,
    hi: float,
) -> torch.Tensor:
    """스틱 clearance가 lo→hi로 오를 때 0→1 램프. palm-in 항이 "공중에 뜬 뒤에만" 켜지게 하는 게이트.

    2026-08-01: palm_up/stick_in_palm이 테이블에서 먼저 손을 뒤집게 만들어(사용자 관측 "잡고 돌려야
    되는데 먼저 돌려버리네") 바닥 스틱을 못 집고 파지가 약해져 영영 못 뜨던 문제 대응. 바닥
    (clearance<lo)에선 0이라 먼저 뒤집을 유인이 사라지고, "잡고 → 들고 → 뒤집기" 순서가 강제됨.
    """
    clearance = box_ground_clearance(env, object_cfg, object_half_extent, surface_z)
    return torch.clamp((clearance - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)


def _goal_proximity_gate(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    command_name: str,
    r_in: float,
    r_out: float,
) -> torch.Tensor:
    """스틱이 목표점(command)에 가까울수록 1로 램프. r_in 이내=1, r_out 밖=0. goal_pos_w=env_origins+cmd[:,:3].

    2026-08-03: palm_up/stick_in_palm이 위치 무관(lift_gate 5cm)이라 "높이 들었다 내려오며 아무 데서나
    뒤집기"로 수렴(run 22-52-13 관측: cube_lift best 0.26m 오버슛 후 하강 중 flip). 뒤집기·담기 보상을
    목표 근접으로 gate해 "목표 위치에 와서 뒤집기"를 강제(사수님). 테이블(dist≈목표높이≥r_out)에선 0이라
    lift_gate처럼 조기 뒤집기도 막음.
    """
    obj = env.scene[object_cfg.name]
    goal_pos_w = env.scene.env_origins + env.command_manager.get_command(command_name)[:, :3]
    dist = torch.norm(obj.data.root_pos_w - goal_pos_w, dim=1)
    return torch.clamp((r_out - dist) / max(r_out - r_in, 1.0e-6), 0.0, 1.0)


class StickInPalmProgressReward(ManagerTermBase):
    """스틱을 palm-local 목표점 T로 데려오는 best-so-far 텔레스코핑, lift + palm-up로 gate.

    φ = eps/(eps+d), d = ||palm-local 스틱위치 − T||. 공중에 뜨고(lift_gate) 손바닥이
    위를 향할수록(palm_up_factor=clamp(cos_up,0,1)) 지급 → "빠르게 뒤집어 손바닥에 담기"를
    dense하게 유도. 2026-08-02(사수님): 뒤집으며 핀치가 풀려도 담기를 계속 보상하려면
    cage로 gate하면 안 됨(핀치 풀리는 순간 0). 그래서 cage gate 제거.
    목표가 상수(palm-local)라 transport와 달리 goal 재샘플 감지 로직이 불필요.
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
        palm_cfg: SceneEntityCfg,
        index_cage_cfg: SceneEntityCfg,
        middle_cage_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        target_local: tuple[float, float, float] = PALM_TARGET_LOCAL,
        object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
        num_points: int = 3,
        point_fractions: tuple[float, ...] | None = None,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        potential_eps: float = 0.05,
        surface_z: float = 0.0,
        lift_gate_lo: float = 0.02,
        lift_gate_hi: float = 0.05,
        command_name: str = "cube_goal",
        goal_gate_r_in: float = 0.06,
        goal_gate_r_out: float = 0.15,
    ) -> torch.Tensor:
        disp_local, palm_quat = _stick_in_palm_local(env, palm_cfg, object_cfg)
        target = torch.as_tensor(target_local, device=disp_local.device, dtype=disp_local.dtype)
        distance = torch.norm(disp_local - target, dim=1)
        phi = potential_eps / (potential_eps + distance)
        self._best_phi = torch.where(self._pending, phi, self._best_phi)
        self._pending[:] = False
        progress = torch.clamp(phi - self._best_phi, min=0.0)
        self._best_phi = torch.maximum(self._best_phi, phi)
        # 2026-08-03: 위치 무관이던 걸 goal_proximity_gate로 "목표점에 와서 담기"로 강제.
        # 2026-08-04: lift_gate 복원(제 버그 수정). goal_gate만으론 테이블에서도 잔여 gate(~0.033)라
        #   빈손 해킹(파지 포기)을 못 막았음(run 23-52-23: palm_up_cos 0.98인데 파지·lift 0). lift_gate는
        #   clearance<2cm에서 정확히 0 → "스틱이 들려야" 담기 보상 켜짐. cage는 계속 뺌(핀치 풀려도 안 죽게).
        goal_gate = _goal_proximity_gate(env, object_cfg, command_name, goal_gate_r_in, goal_gate_r_out)
        lift_gate = _lift_gate(env, object_cfg, object_half_extent, surface_z, lift_gate_lo, lift_gate_hi)
        cos_up = _palm_up_cos(palm_quat)
        palm_up_factor = torch.clamp(cos_up, min=0.0, max=1.0)
        return progress * lift_gate * goal_gate * palm_up_factor


class PalmUpProgressReward(ManagerTermBase):
    """손바닥이 하늘을 보게(palm-up) best-so-far 텔레스코핑, tripod cage로 gate.

    cos_up = palm 법선의 world z성분(1=완전 위). 파지 중일 때만 지급 → "잡은 채 뒤집기"
    (supination) 유도. yaw 무관(법선 z성분만) — 사용자 요구("정밀 자세 유지 불필요, 손바닥만
    하늘로")와 일치. best 초기값 -1이라 첫 스텝은 baseline(보상 0), 이후 개선분만 지급.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best = torch.full((env.num_envs,), -1.0, device=env.device)
        self._pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._pending[env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        palm_cfg: SceneEntityCfg,
        index_cage_cfg: SceneEntityCfg,
        middle_cage_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
        num_points: int = 3,
        point_fractions: tuple[float, ...] | None = None,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        surface_z: float = 0.0,
        lift_gate_lo: float = 0.02,
        lift_gate_hi: float = 0.05,
        command_name: str = "cube_goal",
        goal_gate_r_in: float = 0.06,
        goal_gate_r_out: float = 0.15,
    ) -> torch.Tensor:
        _, palm_quat = _stick_in_palm_local(env, palm_cfg, object_cfg)
        cos_up = _palm_up_cos(palm_quat)
        self._best = torch.where(self._pending, cos_up, self._best)
        self._pending[:] = False
        progress = torch.clamp(cos_up - self._best, min=0.0)
        self._best = torch.maximum(self._best, cos_up)
        # 2026-08-03: cage gate 제거(사수님) — 뒤집으며 핀치 풀려도 flip 보상 안 죽게.
        # 2026-08-04: 단 lift_gate 복원(제 버그 수정). cage도 lift_gate도 없이 goal_gate만 두니
        #   테이블 잔여 gate(~0.033)로 **빈손 뒤집기 해킹** 발생(run 23-52-23: palm_up_cos 0.98인데 파지 0,
        #   palm_dist 0.31m). lift_gate는 clearance<2cm에서 정확히 0 → "들려야 뒤집기 보상"으로 해킹 차단.
        #   cage는 계속 뺌. index/middle_cage_cfg는 cfg 호환 위해 시그니처에만 남김(미사용).
        goal_gate = _goal_proximity_gate(env, object_cfg, command_name, goal_gate_r_in, goal_gate_r_out)
        lift_gate = _lift_gate(env, object_cfg, object_half_extent, surface_z, lift_gate_lo, lift_gate_hi)
        return progress * lift_gate * goal_gate


def stick_throw_penalty(
    env: ManagerBasedRLEnv,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    v0: float = 0.15,
    sigma: float = 0.3,
) -> torch.Tensor:
    """palm 대비 스틱 상대속도가 임계 v0를 넘을수록 커지는 페널티항 ∈[0,1) (weight는 음수로).

    v_rel = ||v_stick − v_palm||. 임계 이하는 0(정상 lift/횡이동 봐줌), 초과분에 지수 포화라
    "많이 멀어질수록 더 페널티"(사용자 설계). v0/sigma는 Metrics/cube*/stick_palm_rel_speed 분포
    측정 후 튜닝 — 초기 weight 0로 두고 이 런에서 분포부터 관찰.
    """
    robot = env.scene[palm_cfg.name]
    obj = env.scene[object_cfg.name]
    palm_id = palm_cfg.body_ids[0]
    palm_lin_vel = robot.data.body_state_w[:, palm_id, 7:10]
    v_rel = torch.norm(obj.data.root_lin_vel_w - palm_lin_vel, dim=-1)
    excess = torch.clamp(v_rel - v0, min=0.0)
    return 1.0 - torch.exp(-excess / max(sigma, 1.0e-6))


class StickInPalmSuccessBonus(ManagerTermBase):
    """손바닥 안(palm-local d<goal_radius) + palm-up(cos>min) + 파지(gate) + 들림(clearance)을
    hold_steps 유지하면 **1회** 보너스 지급. 종료 안 함(GoalReachedBonus와 같은 비종료 방식).

    goal당 1회(_awarded)라 반복 성공 파밍 없음. 여긴 goal이 고정이라 goal 재샘플 로직 불필요.
    로깅: Episode_Reward_Raw/in_palm_success로 성공 빈도 관찰.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._awarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._count[env_ids] = 0
        self._awarded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        palm_cfg: SceneEntityCfg,
        index_cage_cfg: SceneEntityCfg,
        middle_cage_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        target_local: tuple[float, float, float] = PALM_TARGET_LOCAL,
        object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
        num_points: int = 3,
        point_fractions: tuple[float, ...] | None = None,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        goal_radius: float = 0.03,
        command_name: str = "cube_goal",
        goal_radius_w: float = 0.06,
        gate_threshold: float = 0.3,
        palm_up_min: float = 0.9,
        clearance_threshold: float = 0.05,
        surface_z: float = 0.0,
        hold_steps: int = 15,
    ) -> torch.Tensor:
        disp_local, palm_quat = _stick_in_palm_local(env, palm_cfg, object_cfg)
        target = torch.as_tensor(target_local, device=disp_local.device, dtype=disp_local.dtype)
        distance = torch.norm(disp_local - target, dim=1)
        cos_up = _palm_up_cos(palm_quat)
        clearance = box_ground_clearance(env, object_cfg, object_half_extent, surface_z)
        # 2026-08-02: cage gate 조건 제거(사수님). 얇은 스틱은 cage 최대가 ~0.1이라 gate>0.3을
        #   영영 못 넘겨 success가 구조적으로 0이었음. 손바닥에 담기려면 핀치가 풀려야 정상이라
        #   cage 조건은 목표와 모순 → 삭제. gate_threshold/cage cfg는 시그니처 호환 위해만 남김(미사용).
        # 2026-08-02: 월드 위치 조건 추가(사수님). palm_up은 위치 무관이라 팔을 접어 저공에서
        #   팜업(그릇)해도 통과했음. 성공을 "목표점(cube_goal, 적당한 높이) 반경 안에서 팜업+담김"으로
        #   못박으면, 접힌 자세로는 그 높이에 못 닿아 자격 상실 → 자연스러운 편 자세로 유도됨.
        #   goal_pos_w = env_origins + command[:, :3] (transport와 동일 좌표계).
        obj = env.scene[object_cfg.name]
        goal_pos_w = env.scene.env_origins + env.command_manager.get_command(command_name)[:, :3]
        dist_to_goal = torch.norm(obj.data.root_pos_w - goal_pos_w, dim=1)
        valid = (
            (distance < goal_radius)
            & (cos_up > palm_up_min)
            & (clearance > clearance_threshold)
            & (dist_to_goal < goal_radius_w)
        )
        self._count = torch.where(valid, self._count + 1, torch.zeros_like(self._count))
        newly_success = (self._count >= hold_steps) & (~self._awarded)
        self._awarded = self._awarded | newly_success
        return newly_success.float()


@configclass
class ChopstickAcquireCommandsCfg:
    """World goal pose command (position + orientation)으로 스틱을 운반한다.

    box_mdp_cfg의 cube_goal과 동일한 7D pose command. 현재 pos/yaw 고정, 이후 ranges만 넓히면
    랜덤 목표가 됨. pos_z는 env_cfg.__post_init__에서 BASE_Z + 0.20으로 오버라이드.
    """

    cube_goal = mdp.UniformCubeGoalCommandCfg(
        asset_name="cube",
        # 2026-07-24 (사수님): 에피소드(8초) 내 5~10초마다 재샘플 → goal이 도중에 바뀔 수도(안 바뀔 수도).
        #   best-so-far 보상은 _goal_changed로 재장전, success는 GoalReachedBonus(비종료, goal당 1회)로 대응.
        # 2026-07-29 obs 검증 런: goal 자세 고정(roll0/pitch0/yaw0.785398) + 재샘플 없음(1e9).
        #   "됐던" 09-43-43 조건 재현 — obs 하나만 바꿔 검증(랜덤 goal은 2cm에서도 미해결이라 배제).
        #   (07-24 랜덤화 복구용: resampling (5,10) / roll·yaw (1.57, 3.14). 랜덤은 pitch만 4-대칭.)
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        ranges=mdp.UniformCubeGoalCommandCfg.Ranges(
            pos_x=(0.62, 0.62),
            pos_y=(-0.20, -0.20),
            pos_z=(0.45, 0.45),  # placeholder — __post_init__에서 BASE_Z + 0.20
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.7853981633974483, 0.7853981633974483),
            # 2단계(위치)로 넓힐 때: pos_z는 env_cfg.__post_init__의 오버라이드도 같이 범위로 바꿀 것.
            #   예) pos_z 범위 → env_cfg에서 self.commands.cube_goal.ranges.pos_z=(BASE_Z+0.15, BASE_Z+0.25)
            #   pos_x/pos_y는 여기서 직접. 위치는 팔 이동 부담이 커 yaw보다 보수적으로.
        ),
    )


@configclass
class ChopstickAcquireActionsCfg:
    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ChopstickAcquireObservationsCfg:
    """Policy observation: **84** = joint_pos 26 + stick_pos 3 + stick_in_fingertips 15 + stick_size 3
    + stick_to_goal 3 + stick_ori 4 + goal_ori 4 (둘 다 월드 raw quat) + action_history 26.

    2026-07-31 "손바닥에 담기" reframe: **obs는 손대지 않고 보상(goal)만 교체.** lift가 검증된
      16-06-01의 84D obs를 그대로 보존(변수 최소화). stick_to_goal·goal_ori는 stick_transport
      (2026-08-01 lift 부트스트랩으로 재활성)의 world 목표(스폰 위 +20cm)를 가리켜 유효. palm-in
      success는 palm-local(stick_pos)로 별도 판정 — 팜 자세는 joint_pos에 FK로 들어있음.

    2026-07-29 obs 정리(경량화)를 07-25 2cm 베이스라인에 이식:
      · grip_error(9) 제거 — 스틱 정밀 pose+region 기하+FK 필요라 실물에서 못 얻는 oracle.
      · B hand_stick_orientation_error(3) 제거 — 목표가 리셋 우연 자세(arbitrary sim 아티팩트) + oracle.
      · 위치(stick_pos·stick_in_fingertips) → palm-local 프레임(회전 불변 + egocentric).
      · #7 stick_ori_to_target(3, 미리계산 오차) → raw 스틱/goal 자세(월드 quat 4+4). 정책이 상대오차 직접 계산.
    ⚠ 보상·gate(tripod)·영역·두께(2cm)는 07-25 그대로 유지 — 이 런은 "obs만 바꿔서" sim-to-real용
      관측이 2cm 파지·자세맞춤을 여전히 학습하는지 검증하는 용도. 잘 되면 obs 정리 검증됨.
    ⚠ obs dim 변경(91→84)이라 fresh 필수. 파지 shaping REWARD는 sim state로 계산되므로 유지(privileged 허용).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        # palm-local: world축 상대변위 → 손바닥 로컬 프레임 (회전 불변, sim-to-real 친화)
        stick_pos = ObsTerm(
            func=mdp.object_position_relative_to_bodies_local,
            params={"asset_cfg": PALM_CFG, "object_cfg": OBJECT_CFG, "frame_cfg": PALM_CFG},
        )
        stick_in_fingertips = ObsTerm(
            func=mdp.object_position_relative_to_bodies_local,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "finger1_tip_link",
                        "finger2_tip_link",
                        "finger3_tip_link",
                        "finger4_tip_link",
                        "finger5_tip_link",
                    ],
                ),
                "object_cfg": OBJECT_CFG,
                "frame_cfg": PALM_CFG,
            },
        )
        stick_size = ObsTerm(
            func=mdp.object_dims,
            params={"object_cfg": OBJECT_CFG, "fallback_size": (0.02, 0.18, 0.02)},
        )
        # ── 운반 이식(2026-07-22): world goal까지의 위치오차 (oracle 아님 — 알려진 goal - 스틱 위치) ──
        # 2026-08-01: stick_transport(lift 부트스트랩)가 cube_goal(스폰 위 +20cm)을 다시 쓰므로
        #   이 obs는 그 transport 목표오차로 유효. palm-in success는 palm-local(stick_pos)로 별도 판정.
        stick_to_goal = ObsTerm(
            func=mdp.object_position_error_to_command,
            params={"command_name": "cube_goal", "object_cfg": OBJECT_CFG},
        )
        # #7 raw: 미리계산 4-sym 자세오차(oracle 형태) 대신 raw 스틱/goal 자세(월드 quat). 자세오차는
        #   프레임 독립이라 palm 변환 이득 없음 + palm이면 손 움직임에 goal이 요동 → 월드 raw가 맞음.
        stick_ori = ObsTerm(
            func=mdp.object_orientation_world,
            params={"object_cfg": OBJECT_CFG},
        )
        goal_ori = ObsTerm(
            func=mdp.command_orientation_world,
            params={"command_name": "cube_goal"},
        )
        action_history = ObsTerm(func=mdp.action_history)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ChopstickAcquireEventCfg:
    """Fixed-geometry baseline events. Randomization is intentionally off."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    hold_folded_fingers = EventTerm(
        func=mdp.hold_joints_at_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger[4-5]_joint[1-4]"])},
    )
    reset_stick = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": OBJECT_CFG,
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )
    # Capture after reset_all/reset_stick. Startup body poses are not yet the configured reset pose.
    # This baseline has a fixed stick orientation; random-orientation A2 must provide an explicit q_O_H.
    capture_functional_target = EventTerm(
        func=fg_mdp.capture_hand_tool_target,
        mode="reset",
        params={
            "palm_cfg": PALM_CFG,
            "object_cfg": OBJECT_CFG,
            "target_buffer_name": "chopstick_target_palm_quat_o",
        },
    )


@configclass
class ChopstickAcquireRewardsCfg:
    """A1 ladder: cage -> functional constraints -> hold/lift -> terminal."""

    finger_cage_reach = RewTerm(
        func=mdp.ObjectCageProgressReward,
        weight=8.0,
        params={
            "asset_cfg": CHOPSTICK_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "distance_max": 0.5,
        },
    )
    index_grip = RewTerm(
        func=fg_mdp.FingertipGripProgressReward,
        weight=40.0,  # 1.4cm: 2cm 성공값 복귀 (1cm 뭉치기 방지용 150은 되돌림)
        params={
            "palm_cfg": PALM_CFG,
            "fingertip_cfg": INDEX_CFG,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            **INDEX_GRIP_REGION,
            "distance_scale": 0.20,
        },
    )
    thumb_grip = RewTerm(
        func=fg_mdp.FingertipGripProgressReward,
        weight=40.0,  # 1.4cm: 2cm 성공값 복귀 (1cm 뭉치기 방지용 150은 되돌림)
        params={
            "palm_cfg": PALM_CFG,
            "fingertip_cfg": THUMB_CFG,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            **THUMB_GRIP_REGION,
            "distance_scale": 0.20,
        },
    )
    middle_grip = RewTerm(
        func=fg_mdp.FingertipGripProgressReward,
        weight=40.0,  # 1.4cm: 2cm 성공값 복귀 (1cm 뭉치기 방지용 150은 되돌림)
        params={
            "palm_cfg": PALM_CFG,
            "fingertip_cfg": MIDDLE_CFG,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            **MIDDLE_GRIP_REGION,
            "distance_scale": 0.20,
        },
    )
    # 2026-07-31: 약지·새끼도 중지와 같은 +x 면으로 유도(약한 weight). 리워드 없으면 정책이 이들을
    #   물체/중지에 눌러 lift를 막음(play 진단). +x 정렬 = wrap(주먹) 방향, 나중 penta 파지로도 이어짐.
    ring_grip = RewTerm(
        func=fg_mdp.FingertipGripProgressReward,
        weight=15.0,
        params={
            "palm_cfg": PALM_CFG,
            "fingertip_cfg": RING_CFG,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            **MIDDLE_GRIP_REGION,
            "distance_scale": 0.20,
        },
    )
    pinky_grip = RewTerm(
        func=fg_mdp.FingertipGripProgressReward,
        weight=15.0,
        params={
            "palm_cfg": PALM_CFG,
            "fingertip_cfg": PINKY_CFG,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            **MIDDLE_GRIP_REGION,
            "distance_scale": 0.20,
        },
    )
    hand_stick_orientation = RewTerm(
        func=fg_mdp.HandToolOrientationProgressReward,
        weight=0.0,
        params={
            "palm_cfg": PALM_CFG,
            "object_cfg": OBJECT_CFG,
            "target_buffer_name": "chopstick_target_palm_quat_o",
            "angle_scale": 0.7853981633974483,
        },
    )
    finger_cage_hold = RewTerm(
        func=fg_mdp.balanced_tripod_cage_gate,
        weight=5.0,  # 2026-07-30 15→5: 파지는 이미 형성됨(play 확인). 위치무관 '잡고만 있기' 연금을 줄여
        #   캠핑 탈출 — 벌려면 들어야(cube_lift)/goal 가야(transport) 하게. grip 보상 40이 파지는 유지.
        params={
            "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
            "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "sphere_radius": 0.003,  # 14×7: z 반두께 3.5mm에 맞춰 0.005→0.003 (헐거운 z 케이지 조임, 2026-07-30)
            "depth_max": 0.005,
        },
    )
    # Keep the term name ``cube_lift`` because CustomRewardManager reads its
    # surface_z parameter for the generic object-clearance metrics.
    # ⚠ 2026-07-23 재조정: weight 300→150, lift_height 0.15→0.20 (+ 대향 인자).
    #   근거(run 12-13-23 실측): lift가 lift_height에서 포화해 gate×1.0을 매 스텝 영구 지급 →
    #   에피소드 668점(전체 보상의 84%)을 먹고, 목표 도달 항은 6점(0.8%)뿐이라 정책이
    #   "적당히 들고 아무 데서나 버티기"로 수렴함. error_pos가 0.194→0.323으로 단조 악화.
    #   weight 150이면 포화 시 360점 ≈ 도달 가치(transport 160 + success 1000)의 1/3로 내려감.
    #   lift_height 0.20은 goal 높이(clearance 0.19)와 정렬 — 0.15는 goal보다 낮아서 그 위로
    #   올라갈 유인이 0이었음. 최대 지급액은 그대로(gate×1.0)라 연금 상한은 안 커짐.
    #   (이전: 4000 → 150 → 300 → 150)
    cube_lift = RewTerm(
        func=fg_mdp.object_lift_in_balanced_tripod_cage,
        weight=300.0,  # 2026-07-30 100→300: 7mm lift 유도 — 든 채 연금 3배 + 떼는 기울기 3배(들고 유지가 이득)
        params={
            "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
            "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "sphere_radius": 0.003,  # 14×7: z 반두께 3.5mm에 맞춰 0.005→0.003 (헐거운 z 케이지 조임, 2026-07-30)
            "depth_max": 0.005,
            # 2026-07-30: 0.10→0.05. 총 호버 연금(2.4×W=240)은 그대로, 떼는 기울기 W/h만 1000→2000.
            #   1cm 떼면 48점 > cage_hold 전체 연금(36) → "느슨하게 앉기"보다 "떼기"가 이득 → 캠핑 탈출.
            #   goal(BASE_Z+0.20)까진 transport가 마저 올림 — lift는 바닥서 떼는 역할만.
            "lift_height": 0.05,
        },
    )

    # ══ lift 부트스트랩 (2026-08-01) ══════════════════════════════════════════════════
    # 17-57-51(palm-in only)이 2400 iter 돌고도 못 떴음(clearance 0.0005 vs 16-06-01 0.28).
    #   원인: stick_transport가 사실 lift를 일으키던 신호였는데(=스폰 위 +20cm로 곧장 들어올려)
    #   제거해서 파지 gate가 굶고(cage 0.039<0.091) stick_in_palm·palm_up이 다 ~0으로 죽음.
    #   → stick_transport만 순수 lift 부트스트랩으로 재활성. 나머지 5종(orientation/proximity/
    #   fine×2/success)은 rigid world pose/success 강제라 계속 비활성(아래 복구 블록).
    #   협력 구도: transport(스틱을 spawn 위 20cm로 들어올림) + palm-in(뒤집어 손바닥 cup에 담기)
    #   = "들어서 뒤집어 담기". 둘 다 만족점 = 손바닥에 담은 채 20cm 상공 → 충돌 아님.
    stick_transport = RewTerm(
        func=BalancedObjectToGoalProgressReward,
        weight=30000.0,
        params={
            "command_name": "cube_goal",
            "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
            "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "sphere_radius": 0.003,
            "depth_max": 0.005,
            "potential_eps": 0.08,
        },
    )

    # ══ "손바닥에 담기" reframe (2026-07-31) — world-goal의 pose/success를 palm-in으로 교체 ══
    # 스틱을 palm-local 목표점 T로 데려오는 텔레스코핑 (파지 gate). world pose 매칭 대신 손바닥 cup.
    stick_in_palm = RewTerm(
        func=StickInPalmProgressReward,
        weight=30000.0,
        params={
            "palm_cfg": PALM_CFG,
            "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
            "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "target_local": PALM_TARGET_LOCAL,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "sphere_radius": 0.003,
            "depth_max": 0.005,
            "potential_eps": 0.08,
            # 2026-08-03: 목표 근접 gate — 목표점 6cm 안=full, 15cm(목표높이)에서 0. "목표에서 담기".
            "command_name": "cube_goal",
            "goal_gate_r_in": 0.06,
            "goal_gate_r_out": 0.15,
        },
    )
    # 손바닥이 하늘을 보게 뒤집기 텔레스코핑 (파지+목표근접 gate). yaw 무관. stick_orientation 대체.
    palm_up = RewTerm(
        func=PalmUpProgressReward,
        weight=20000.0,
        params={
            "palm_cfg": PALM_CFG,
            "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
            "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "sphere_radius": 0.003,
            "depth_max": 0.005,
            # 2026-08-03: 목표 근접 gate — 목표점 6cm 안=full, 15cm에서 0. "목표에서 뒤집기".
            "command_name": "cube_goal",
            "goal_gate_r_in": 0.06,
            "goal_gate_r_out": 0.15,
        },
    )
    # palm 대비 스틱 상대속도(throw) 페널티. weight 0 — 이 런에서 v_rel 분포 측정 후 v0/weight 튜닝.
    throw_penalty = RewTerm(
        func=stick_throw_penalty,
        weight=0.0,
        params={
            "palm_cfg": PALM_CFG,
            "object_cfg": OBJECT_CFG,
            "v0": 0.15,
            "sigma": 0.3,
        },
    )
    # 성공: 손바닥 안(d<3cm) + palm-up(cos>0.9) + 들림(5cm) + **목표점(cube_goal) 반경 6cm 안**을
    #   0.5s 유지 → 1회 +2000. 비종료. transport_success 대체. surface_z는 __post_init__에서 BASE_Z로.
    #   2026-08-02: 월드 위치 조건(goal_radius_w) 추가 — 저공 그릇 자세를 자격에서 배제(사수님).
    #   목표 높이는 env_cfg에서 cube_goal.ranges.pos_z로 정함(현재 BASE_Z+0.15 = 적당한 높이).
    in_palm_success = RewTerm(
        func=StickInPalmSuccessBonus,
        weight=60000.0,
        params={
            "palm_cfg": PALM_CFG,
            "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
            "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "target_local": PALM_TARGET_LOCAL,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "sphere_radius": 0.003,
            "depth_max": 0.005,
            "goal_radius": 0.03,
            "command_name": "cube_goal",
            "goal_radius_w": 0.06,
            "gate_threshold": 0.3,
            # 2026-08-03: 0.9→0.6(사수님). 4000 iter 돌린 최고 flip도 cos_up 0.69라 0.9는 도달 불가였음.
            #   ≈53° 이내로 팜업이면 성공 인정. 부족하면 임계 근처 shaped 보상으로 발전.
            "palm_up_min": 0.6,
            "clearance_threshold": 0.05,
            "surface_z": 0.0,
            "hold_steps": 15,
        },
    )

    # ══ (복구용) 2026-07-31 reframe으로 비활성화한 world-goal pose/success 5종 ══════════
    #   stick_orientation / goal_proximity / fine_position / fine_orientation / transport_success.
    #   (stick_transport는 lift 부트스트랩으로 위에서 재활성 — 여기 없음. cube_goal 커맨드도 활성.)
    #   되살릴 때: 주석 풀고 reframe 항(stick_in_palm 등)과 목표 충돌 없게 조율할 것.
#     stick_orientation = RewTerm(
#         func=BalancedObjectOrientationCoupledReward,
#         weight=30000.0,
#         params={
#             "command_name": "cube_goal",
#             "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
#             "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
#             "object_cfg": OBJECT_CFG,
#             "object_half_extent": STICK_HALF_EXTENT,
#             "num_points": 3,
#             "point_fractions": POINT_FRACTIONS,
#             "sphere_radius": 0.003,
#             "depth_max": 0.005,
#             "activation_distance": 0.05,
#             "activation_gate_threshold": 0.3,
#             "angle_scale": 0.7853981633974483,
#         },
#     )
#     goal_proximity = RewTerm(
#         func=balanced_object_goal_proximity,
#         weight=20.0,
#         params={
#             "command_name": "cube_goal",
#             "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
#             "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
#             "object_cfg": OBJECT_CFG,
#             "object_half_extent": STICK_HALF_EXTENT,
#             "num_points": 3,
#             "point_fractions": POINT_FRACTIONS,
#             "sphere_radius": 0.003,
#             "depth_max": 0.005,
#             "potential_eps": 0.05,
#         },
#     )
#     fine_position = RewTerm(
#         func=FineProximityReward,
#         weight=3000.0,
#         params={
#             "command_name": "cube_goal",
#             "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
#             "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
#             "object_cfg": OBJECT_CFG,
#             "object_half_extent": STICK_HALF_EXTENT,
#             "num_points": 3,
#             "point_fractions": POINT_FRACTIONS,
#             "sphere_radius": 0.003,
#             "depth_max": 0.005,
#             "sigma": 0.05,
#         },
#     )
#     fine_orientation = RewTerm(
#         func=FineOrientationReward,
#         weight=1500.0,
#         params={
#             "command_name": "cube_goal",
#             "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
#             "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
#             "object_cfg": OBJECT_CFG,
#             "object_half_extent": STICK_HALF_EXTENT,
#             "num_points": 3,
#             "point_fractions": POINT_FRACTIONS,
#             "sphere_radius": 0.003,
#             "depth_max": 0.005,
#             "sigma": 0.2617993877991494,
#             "activation_distance": 0.05,
#             "activation_gate_threshold": 0.3,
#         },
#     )
#     transport_success = RewTerm(
#         func=GoalReachedBonus,
#         weight=60000.0,
#         params={
#             "command_name": "cube_goal",
#             "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
#             "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
#             "object_cfg": OBJECT_CFG,
#             "object_half_extent": STICK_HALF_EXTENT,
#             "num_points": 3,
#             "point_fractions": POINT_FRACTIONS,
#             "sphere_radius": 0.003,
#             "depth_max": 0.005,
#             "goal_radius": 0.05,
#             "gate_threshold": 0.3,
#             "hold_steps": 15,
#             "ori_limit": 0.2617993877991494,
#         },
#     )

    arm_manipulability = RewTerm(
        func=mdp.arm_manipulability_penalty,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["palm_link"], joint_names=["joint[0-5]"]
            ),
            "j_max": 0.02,
        },
    )
    hand_floor = RewTerm(
        func=mdp.hand_floor_penalty,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link", "finger.*"]),
            "clearance": 0.01,
        },
    )
    # ── 페널티 3종 (2026-07-22) ──
    # EE(palm) 선속도 L2 — 급격한 팔 이동 억제
    end_effector_speed = RewTerm(
        func=mdp.end_effector_speed,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["palm_link"])},
    )
    # 이전-현재 action 차의 제곱합 (기존 -0.005 → -0.001)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    # action 2차 rate — CustomActionManager 배선(rl_task_custom_env.py)으로 prevprev_action이
    # 채워져 활성화 가능해짐. action 가속(2차 차분) 억제.
    action_second_rate = RewTerm(func=mdp.action_second_rate_l2, weight=-0.0001)


@configclass
class ChopstickAcquireTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    stick_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": OBJECT_CFG},
    )
    # ── success termination 제거 (2026-07-24, 사수님) ──────────────────────────────
    # 성공 시 종료하면 첫 성공에 리셋돼 에피소드 내 goal 재샘플로 다음 goal을 못 봄.
    # 성공 '보너스'는 유지하되(위 rewards.transport_success = GoalReachedBonus, goal당 1회),
    # 종료는 하지 않음. 종료는 time_out(8초) + stick_dropped만.
    # ⚠ Episode_Termination/success 지표가 사라짐 → 성공 빈도는 Episode_Reward_Raw/transport_success로 봄.
    #
    # (복구용) 기존 success 종료 — BalancedObjectAtGoalHeld, goal_radius 0.05 / gate 0.3 /
    #   hold_steps 15 / ori_limit 0.2617993877991494. 되살리면 transport_success도 is_terminated_term으로.
    # success = DoneTerm(func=BalancedObjectAtGoalHeld, params={... 위 GoalReachedBonus와 동일 ...})
