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
from isaac_neuromeka.mdp.rewards import square_prism_ori_error


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
OBJECT_CFG = SceneEntityCfg("cube")

STICK_HALF_EXTENT = (0.007, 0.09, 0.007)  # 1.4cm(젓가락 2개 다발) 2026-07-30 (STICK_SIZE와 동반)
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
    "axial_region": (-0.60, -0.30),
    "surface_axis": 2,
    "surface_sign": 1.0,
    "surface_offset": 0.0,
    "surface_tolerance": 0.005,
}

THUMB_GRIP_REGION = {
    "long_axis": 1,
    "axial_region": (-0.55, -0.25),
    "surface_axis": 0,
    "surface_sign": -1.0,
    "surface_offset": 0.0,
    "surface_tolerance": 0.005,
}

MIDDLE_GRIP_REGION = {
    "long_axis": 1,
    "axial_region": (-0.55, -0.25),
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
            "sphere_radius": 0.005,  # 1.4cm: 2cm 성공값 복귀 (1cm 때 0.004)
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
        weight=100.0,
        params={
            "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
            "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "sphere_radius": 0.005,  # 1.4cm: 2cm 성공값 복귀 (1cm 때 0.004)
            "depth_max": 0.005,
            # 2026-07-30: 0.10→0.05. 총 호버 연금(2.4×W=240)은 그대로, 떼는 기울기 W/h만 1000→2000.
            #   1cm 떼면 48점 > cage_hold 전체 연금(36) → "느슨하게 앉기"보다 "떼기"가 이득 → 캠핑 탈출.
            #   goal(BASE_Z+0.20)까진 transport가 마저 올림 — lift는 바닥서 떼는 역할만.
            "lift_height": 0.05,
        },
    )

    # ── 운반 이식(2026-07-22): 위치 best-so-far transport + 결합형 orientation ──
    # 위치/각도를 분리하되, orientation은 "goal에 머문 채로만" 지급되는 결합형이라 드리프트 차단.
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
            "sphere_radius": 0.005,  # 1.4cm: 2cm 성공값 복귀 (1cm 때 0.004)
            "depth_max": 0.005,
            "potential_eps": 0.08,
        },  
    )
    stick_orientation = RewTerm(
        func=BalancedObjectOrientationCoupledReward,
        weight=30000.0,
        params={
            "command_name": "cube_goal",
            "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
            "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "sphere_radius": 0.005,  # 1.4cm: 2cm 성공값 복귀 (1cm 때 0.004)
            "depth_max": 0.005,
            # 지금 이 스텝에 goal 3cm 안 + 파지 gate 0.3 넘을 때만 orientation 지급.
            "activation_distance": 0.05,
            "activation_gate_threshold": 0.3,
            "angle_scale": 0.7853981633974483,
        },
    )

    # goal 근접 연금 (box에서 이식, 기본 0). 드리프트 시 연금 손실로 goal 체류에 붙잡는 힘.
    # 활성화 시 per-step ≈ weight/30 × gate × φ. 캠핑 상한은 success 종료가 담당(현재가치 ≪ +1000).
    goal_proximity = RewTerm(
        func=balanced_object_goal_proximity,
        weight=20.0,  # 2026-07-30 75→20: 스틱이 goal 바로 아래라 테이블에서도 받던 연금 축소(캠핑 탈출)
        params={
            "command_name": "cube_goal",
            "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
            "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "sphere_radius": 0.005,  # 1.4cm: 2cm 성공값 복귀 (1cm 때 0.004)
            "depth_max": 0.005,
            "potential_eps": 0.05,
        },
    )

    # 말단 정밀 수렴용 지수 커널 (2026-07-22, weight 0 기본 — 튜닝 후 켤 것).
    # exp(-e/σ)는 e=0에서 기울기 최대라 마지막 mm/도를 조임. best-so-far(파밍 없음).
    # σ는 success 임계보다 작게(임계 안쪽 refine): σ_d=2cm(<5cm), σ_θ=10°(<15°).
    # coarse(transport φ + lift)가 데려온 뒤 말단만 refine — 모양의 사다리.
    fine_position = RewTerm(
        func=FineProximityReward,
        weight=3000.0,
        params={
            "command_name": "cube_goal",
            "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
            "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "sphere_radius": 0.005,  # 1.4cm: 2cm 성공값 복귀 (1cm 때 0.004)
            "depth_max": 0.005,
            "sigma": 0.05,
        },
    )
    fine_orientation = RewTerm(
        func=FineOrientationReward,
        weight=1500.0,
        params={
            "command_name": "cube_goal",
            "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
            "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "sphere_radius": 0.005,  # 1.4cm: 2cm 성공값 복귀 (1cm 때 0.004)
            "depth_max": 0.005,
            "sigma": 0.2617993877991494, #15도
            "activation_distance": 0.05,
            "activation_gate_threshold": 0.3,
        },
    )

    # r_T: goal pose(±15°) + gate 물림 0.5s 유지 → 한 방 +2000 보너스. **종료 안 함** (2026-07-24).
    #   success termination을 분리해 GoalReachedBonus로 교체 — 성공해도 에피소드 유지 → 재샘플로
    #   다음 goal 처리. goal당 1회만 지급(_awarded)이라 같은 goal 반복 성공 파밍 없음.
    #   (기존: func=mdp.is_terminated_term, term_keys="success" — success 종료 참조. 복구용)
    transport_success = RewTerm(
        func=GoalReachedBonus,
        weight=60000.0,
        params={
            "command_name": "cube_goal",
            "index_cage_cfg": CHOPSTICK_INDEX_CAGE_BODIES,
            "middle_cage_cfg": CHOPSTICK_MIDDLE_CAGE_BODIES,
            "object_cfg": OBJECT_CFG,
            "object_half_extent": STICK_HALF_EXTENT,
            "num_points": 3,
            "point_fractions": POINT_FRACTIONS,
            "sphere_radius": 0.005,  # 1.4cm: 2cm 성공값 복귀 (1cm 때 0.004)
            "depth_max": 0.005,
            "goal_radius": 0.05,
            "gate_threshold": 0.3,
            "hold_steps": 15,
            "ori_limit": 0.2617993877991494,
        },
    )
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
