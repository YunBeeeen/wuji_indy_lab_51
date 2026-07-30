"""Indy-Wuji-Box-Transport 전용 MDP 설정 (2026-07-16).

CubeGrasp*Cfg(env_cfg_common.py)의 사본 + 랜덤 직육면체 확장. 큐브 태스크를 동결하기 위해
기존 클래스를 수정하지 않고 복제함 (사용자 지시). 큐브 쪽과의 차이:
  - 관측 +16: box_size(3) + box_quat(4) + box_ori_to_target(3)
    + index/thumb grip-region error(6) -> policy 73
  - 이벤트 +2: randomize_box_dims(prestartup, env별 비율보존 치수) + set_box_default_height(startup)
  - scene.replicate_physics = False 필요 (box_transport_env_cfg.__post_init__에서 설정)

⚠ 활성 태스크: Indy-Wuji-Box-Transport 전용. 큐브 태스크(Indy-Wuji-Cube-Grasp)는
  env_cfg_common.py의 CubeGrasp*Cfg를 씀 — 여기를 고쳐도 큐브 태스크에는 반영 안 됨 (역도 같음).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp
from isaac_neuromeka.tasks.manipulation.functional_grasp import mdp as fg_mdp
from isaac_neuromeka.mdp.rewards import square_prism_ori_error, square_prism_keypoint_goal_distance
from isaaclab.utils.math import quat_apply

# 가상점 12개용 body 목록. env_cfg_common.CAGE_BODIES와 같은 구성이지만 인스턴스를 공유하지
# 않음 — SceneEntityCfg는 resolve 시 내부 상태(body_ids)가 채워지는 가변 객체라 태스크 간
# 공유가 위험함. preserve_order=True 필수 (엄지가 기준점 자리).
BOX_CAGE_BODIES = SceneEntityCfg(
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

# ── 감싸쥐기 cage 부품 (2026-07-19 WRAP 30-point 실험) ──
# 근거 사슬: ori v1(무유도)·v1.1(keypoint 유인) 모두 success 0, 슬롯 A/B 공통 매달림
#   70~93° + play "손가락 2개 불안정 파지" 관찰 → 손끝 pinch는 긴 상자의 중력 토크를
#   못 버팀 (2점 접촉 = 경첩). 힘(depth_max)이 아니라 기하(접촉 분산)의 문제.
# 개편 두 축: 폭(약지/새끼 추가 — 길이축 팔길이 확보. 액션은 커플링으로 이미 오므려짐)
#   + 깊이(근위 마디 link2 — 물체를 손바닥 쪽으로 끌어들여 감싸쥐기 유도).
# 재활성화: 이 정의 아래에 `BOX_CAGE_BODIES = BOX_CAGE_BODIES_WRAP` 할당.
#   (모든 reach/hold/lift/transport gate + success gate가 일괄 전환됨). fresh 필수.
# ⚠ 활성 시: 가상점 평균의 몸이 5→11개라 gate 값이 전반적으로 낮아짐 — success
#   gate_threshold(0.3) 재보정 필요. fresh 초기 Episode_Reward_Raw/finger_cage_hold와
#   낙하율로 판단할 것. 짧은 물체에서는 약지/새끼 점이 못 걸치는 부분 감점도 유의.
BOX_CAGE_BODIES_WRAP = SceneEntityCfg(
    "robot",
    body_names=[
        "finger1_tip_link",  # 엄지끝: 모든 선분의 기준점
        "finger2_tip_link", "finger2_link3", "finger2_link2",
        "finger3_tip_link", "finger3_link3", "finger3_link2",
        "finger4_tip_link", "finger4_link3",
        "finger5_tip_link", "finger5_link3",
    ],
    preserve_order=True,
)

# 2026-07-19 활성했던 WRAP 30-point 코드. 비교/복구용으로 남겨두고 현재는 비활성.
# BOX_CAGE_BODIES = BOX_CAGE_BODIES_WRAP

# 2026-07-20 tip-only 비교 코드: 엄지-검지, 엄지-중지 손끝 선분만 사용.
# 실험에서 한 선분의 평균 지분이 0.25 -> 0.5로 커져 hold 국소최적이 확인되어 비활성화함.
# joint2/3은 여전히 fingertip FK에 기여하므로 policy action에서 빠지지 않음.
BOX_CAGE_BODIES_TIP_ONLY = SceneEntityCfg(
    "robot",
    body_names=[
        "finger1_tip_link",  # thumb tip: every line starts here
        "finger2_tip_link",  # index tip
        "finger3_tip_link",  # middle tip
    ],
    preserve_order=True,
)
# BOX_CAGE_BODIES = BOX_CAGE_BODIES_TIP_ONLY

# 보상/판정의 half_extent 인자는 fallback임 — 실제로는 randomize_box_dims가 저장한
# env.box_half_extents (N,3) 버퍼가 우선함 (rewards.py의 _cage_sdf/box_ground_clearance 참고).
_HALF_FALLBACK = (0.01, 0.09, 0.01)
_POINT_FRACTIONS = (0.1, 0.5, 0.9)

BOX_INDEX_CAGE_BODIES = SceneEntityCfg(
    "robot",
    body_names=["finger1_tip_link", "finger2_tip_link", "finger2_link3"],
    preserve_order=True,
)

BOX_MIDDLE_CAGE_BODIES = SceneEntityCfg(
    "robot",
    body_names=["finger1_tip_link", "finger3_tip_link", "finger3_link3"],
    preserve_order=True,
)

# 2026-07-25 Phase 1 주먹(wrap) 파지: 약지·새끼 cage 추가 (엄지 기준점 공유).
BOX_RING_CAGE_BODIES = SceneEntityCfg(
    "robot",
    body_names=["finger1_tip_link", "finger4_tip_link", "finger4_link3"],
    preserve_order=True,
)
BOX_PINKY_CAGE_BODIES = SceneEntityCfg(
    "robot",
    body_names=["finger1_tip_link", "finger5_tip_link", "finger5_link3"],
    preserve_order=True,
)

_PALM_CFG = SceneEntityCfg("robot", body_names=["palm_link"])
_INDEX_CFG = SceneEntityCfg("robot", body_names=["finger2_tip_link"])
_THUMB_CFG = SceneEntityCfg("robot", body_names=["finger1_tip_link"])
_MIDDLE_CFG = SceneEntityCfg("robot", body_names=["finger3_tip_link"])
_RING_CFG = SceneEntityCfg("robot", body_names=["finger4_tip_link"])
_PINKY_CFG = SceneEntityCfg("robot", body_names=["finger5_tip_link"])
_OBJECT_CFG = SceneEntityCfg("cube")

# 2026-07-25 Phase 1 주먹(wrap) 파지. 2026-07-26_00-04-17의 실제 env.yaml 기준:
# 엄지는 -x면, 나머지 4손가락은 반대 +x면이며 axial stagger로 길이축에 분산한다.
# ⚠ 되돌리기(tripod): _INDEX surface_axis 2(+z) / _THUMB -x / _MIDDLE +x / 약지·새끼 미사용.
#   전체 원본은 box_mdp_cfg.py.keypoint_backup_2026-07-25.
_INDEX_GRIP_REGION = {
    "long_axis": 1,
    "axial_region": (-0.65, -0.35),
    "surface_axis": 0,
    "surface_sign": 1.0,
    "surface_offset": 0.0,
    "surface_tolerance": 0.005,
}
_THUMB_GRIP_REGION = {
    "long_axis": 1,
    "axial_region": (-0.55, -0.25),
    "surface_axis": 0,
    "surface_sign": -1.0,
    "surface_offset": 0.0,
    "surface_tolerance": 0.005,
}
_MIDDLE_GRIP_REGION = {
    "long_axis": 1,
    "axial_region": (-0.55, -0.25),
    "surface_axis": 0,
    "surface_sign": 1.0,
    "surface_offset": 0.0,
    "surface_tolerance": 0.005,
}
_RING_GRIP_REGION = {
    "long_axis": 1,
    "axial_region": (-0.45, -0.15),
    "surface_axis": 0,
    "surface_sign": 1.0,
    "surface_offset": 0.0,
    "surface_tolerance": 0.005,
}
_PINKY_GRIP_REGION = {
    "long_axis": 1,
    "axial_region": (-0.35, -0.05),
    "surface_axis": 0,
    "surface_sign": 1.0,
    "surface_offset": 0.0,
    "surface_tolerance": 0.005,
}


def _goal_pose_from_command(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Box-Transport goal position and quaternion in world coordinates."""
    command = env.command_manager.get_command(command_name)
    goal_pos_w = env.scene.env_origins + command[:, :3]
    if command.shape[-1] >= 7:
        goal_quat_w = command[:, 3:7]
    else:
        goal_quat_w = torch.zeros(
            command.shape[0],
            4,
            device=command.device,
            dtype=command.dtype,
        )
        goal_quat_w[:, 0] = 1.0
    return goal_pos_w, goal_quat_w


# ── 에피소드 내 goal 재샘플 대응 (2026-07-24, 준비) ────────────────────────────
# box 랜덤화 시 resampling_time_range=(5,10)이면 에피소드 도중 goal이 바뀔 수 있음.
# keypoint(_best_d)·success(_count)가 옛 goal에 고착되지 않도록 goal 변화를 감지해 재장전.
# chopstick_mdp_cfg._goal_changed와 동일 로직(box는 별도 파일이라 복제). fresh 전 스모크 필수.
def _goal_key(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """goal 위치(3)+자세(4) 7D pose 키. position만 비교하면 ori-only 재샘플을 놓침(2026-07-24)."""
    goal_w, goal_quat_w = _goal_pose_from_command(env, command_name)
    return torch.cat([goal_w, goal_quat_w], dim=-1)


def _goal_changed(term, goal_key: torch.Tensor) -> torch.Tensor:
    """이전 스텝 goal(7D pose)과 비교해 바뀐 env를 True로. _goal_cache 없으면 생성(첫 호출 전부 False).
    ⚠ goal_key는 반드시 7D pose(_goal_key)를 넘길 것 — position만 넘기면 ori 재샘플을 못 잡음."""
    cache = getattr(term, "_goal_cache", None)
    if cache is None or cache.shape != goal_key.shape:
        term._goal_cache = goal_key.clone()
        return torch.zeros(goal_key.shape[0], dtype=torch.bool, device=goal_key.device)
    changed = (cache - goal_key).abs().sum(dim=-1) > 1.0e-6
    term._goal_cache = goal_key.clone()
    return changed


class BalancedObjectToGoalProgressReward(ManagerTermBase):
    """Position best-so-far progress gated by the balanced tripod grasp.

    기존 ``mdp.ObjectToGoalProgressReward``의 포텐셜 및 reset 동작은 유지하고,
    파지 gate만 12-point 전체 평균에서
    ``min(thumb-index cage, thumb-middle cage)``로 교체한다.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best_phi = torch.zeros(env.num_envs, device=env.device)
        self._pending = torch.ones(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

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
        object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
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

        # command resample 순서 오염을 피하기 위해 첫 호출에서 기준선을 잡는다.
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


class BalancedObjectOrientationProgressReward(ManagerTermBase):
    """Goal-near orientation progress gated by the balanced tripod grasp.

    기존 ``mdp.ObjectOrientationProgressReward``의 latch와 best-so-far 구조는
    유지하고, activation 및 지급에 사용되는 cage gate만 balanced tripod로 통일한다.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best_error = torch.zeros(
            env.num_envs,
            dtype=torch.float,
            device=env.device,
        )
        self._active = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    @property
    def active(self) -> torch.Tensor:
        return self._active

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._best_error[env_ids] = 0.0
        self._active[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        index_cage_cfg: SceneEntityCfg,
        middle_cage_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
        num_points: int = 3,
        point_fractions: tuple[float, ...] | None = None,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        activation_distance: float = 0.10,
        activation_gate_threshold: float = 0.3,
        angle_scale: float = 0.7853981633974483,
    ) -> torch.Tensor:
        obj = env.scene[object_cfg.name]
        goal_w, goal_quat_w = _goal_pose_from_command(env, command_name)

        position_error = torch.norm(goal_w - obj.data.root_pos_w, dim=1)
        orientation_error = square_prism_ori_error(
            obj.data.root_quat_w,
            goal_quat_w,
        )
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

        activate = (
            (position_error < activation_distance)
            & (gate > activation_gate_threshold)
        )
        newly_active = (~self._active) & activate
        self._active |= activate

        # latch가 처음 켜진 step은 현재 오차를 기준선으로만 저장한다.
        self._best_error = torch.where(
            newly_active,
            orientation_error,
            self._best_error,
        )

        scale = max(float(angle_scale), 1.0e-6)
        progress = torch.clamp(
            (self._best_error - orientation_error) / scale,
            min=0.0,
            max=1.0,
        )
        valid = self._active & (~newly_active)
        reward = torch.where(
            valid,
            progress * gate,
            torch.zeros_like(progress),
        )

        self._best_error = torch.where(
            self._active,
            torch.minimum(self._best_error, orientation_error),
            self._best_error,
        )
        return reward


def balanced_object_goal_proximity(
    env: ManagerBasedRLEnv,
    command_name: str,
    index_cage_cfg: SceneEntityCfg,
    middle_cage_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
    num_points: int = 3,
    point_fractions: tuple[float, ...] | None = None,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
    potential_eps: float = 0.05,
) -> torch.Tensor:
    """Balanced tripod grasp를 유지할 때만 goal 근접 연금을 지급한다."""
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


class BalancedObjectAtGoalHeld(ManagerTermBase):
    """Goal pose와 balanced tripod grasp를 연속 유지하면 성공 종료한다.

    기존 Box-Transport 성공 조건과 동일하게 position, object orientation, cage,
    hold_steps만 검사한다. index/thumb grip-region은 실험 분리를 위해 dense reward로만
    유지하며 여기에는 추가하지 않는다.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._count = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )

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
        object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
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
            valid &= (
                square_prism_ori_error(
                    obj.data.root_quat_w,
                    goal_quat_w,
                )
                < ori_limit
            )

        self._count = torch.where(
            valid,
            self._count + 1,
            torch.zeros_like(self._count),
        )
        return self._count >= hold_steps


# ── SimToolReal(2602.16863) keypoint 목표 도달 — box A/B용 (2026-07-22) ──
# transport(위치 φ) + orientation(쿼터니안)을 pose error 한 항으로 융합.
# chopstick은 쿼터니안+exp 유지, box는 이 keypoint로 학습해 비교.
def _stick_tail_tip_pose_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg,
    keypoint_scale: tuple[float, float, float] = (0.03, 0.14, 0.03),
) -> torch.Tensor:
    """[미사용/원형 스틱용] tail/tip 2점(장축 y 위) max 거리 = pose error.

    ⚠ 정사각 스틱은 roll이 90°만 대칭이라 45° roll을 구분 못 하는 이 2점은 under-constrained.
    현재 box는 8-corner + roll 4-대칭(square_prism_y_tip)을 씀. 이 함수는 원형 스틱(roll 무의미) 대비 보존.

    - 순서 고정(tail↔tail, tip↔tip)이라 **앞뒤(flip) 구분** — tail-forward를 정답으로 안 봄.
    - 두 점이 **장축 위**라 roll에 안 움직임 → 정사각 스틱의 roll 대칭 자동 처리(대칭 루프 불필요).
    - 위치(양 끝 동시)+장축 방향을 한 거리에 담음. keypoint_scale=(sx,sy,sz), 점은 로컬 y=±sy/2
      (SimToolReal식 고정 스케일 — 실제 물체 크기와 무관하게 장축 정렬 민감도 일정).
    """
    obj = env.scene[object_cfg.name]
    device = obj.data.root_pos_w.device
    dtype = obj.data.root_pos_w.dtype
    n = env.num_envs
    half_y = 0.5 * keypoint_scale[1]
    kps = torch.tensor(
        [[0.0, -half_y, 0.0], [0.0, half_y, 0.0]], device=device, dtype=dtype
    )  # (2,3): tail(-y), tip(+y)
    kps_b = kps.unsqueeze(0).expand(n, 2, 3)
    goal_pos_w, goal_quat_w = _goal_pose_from_command(env, command_name)
    quat = obj.data.root_quat_w
    cur = obj.data.root_pos_w.unsqueeze(1) + quat_apply(quat.unsqueeze(1).expand(n, 2, 4), kps_b)
    goal = goal_pos_w.unsqueeze(1) + quat_apply(goal_quat_w.unsqueeze(1).expand(n, 2, 4), kps_b)
    return torch.norm(cur - goal, dim=-1).amax(dim=1)  # max over {tail, tip}


class KeypointMaxGoalProgressReward(ManagerTermBase):
    """SimToolReal Eq.2 dense항: r = tripod_gate × max(d_best − d, 0), d = **8-corner max 거리**.

    8 꼭짓점의 max라 모든 꼭짓점이 가까워야(=위치+자세 full, roll 포함) d가 줄어듦 → pos·ori 구조적
    결합(한 축만 맞추고 드리프트 불가). 대칭은 **square_prism_y_tip = roll 4-대칭(90°만, flip 제외)** —
    tail/tip 구분(앞뒤) + 45° roll은 오답(정사각 roll 90° 대칭). best-so-far(d_best 최소)라 파밍 없음.
    tripod_gate는 엄지 대 검지·중지 cage의 min으로 goal 단계의 기준 파지를 사용함.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._best_d = torch.full((env.num_envs,), float("inf"), device=env.device)
        self._pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._pending[env_ids] = True
        self._best_d[env_ids] = float("inf")

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
        symmetry: str = "square_prism_y_tip",
    ) -> torch.Tensor:
        d = square_prism_keypoint_goal_distance(
            env, command_name, object_cfg, object_half_extent, symmetry, reduce="max",
        )
        # 에피소드 내 goal 재샘플 감지 → 그 env는 다시 pending(새 goal 기준선 d로 재장전).
        # d가 inf가 아니라 새 goal 첫 d로 리셋되도록 _pending을 켜면 아래 where가 처리함.
        self._pending = self._pending | _goal_changed(self, _goal_key(env, command_name))
        self._best_d = torch.where(self._pending, d, self._best_d)
        self._pending[:] = False
        progress = torch.clamp(self._best_d - d, min=0.0)
        self._best_d = torch.minimum(self._best_d, d)
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


class KeypointMaxAtGoalHeld(ManagerTermBase):
    """SimToolReal 성공: 8-corner max 거리(roll 4-sym) < keypoint_eps + tripod gate 유지."""

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
        keypoint_eps: float = 0.05,
        gate_threshold: float = 0.3,
        hold_steps: int = 15,
        symmetry: str = "square_prism_y_tip",
    ) -> torch.Tensor:
        d = square_prism_keypoint_goal_distance(
            env, command_name, object_cfg, object_half_extent, symmetry, reduce="max",
        )
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
        valid = (d < keypoint_eps) & (gate > gate_threshold)
        self._count = torch.where(valid, self._count + 1, torch.zeros_like(self._count))
        return self._count >= hold_steps


class KeypointGoalReachedBonus(ManagerTermBase):
    """[준비, 2026-07-24] keypoint 성공 시 **1회 보너스** 지급, 종료 안 함 (box 랜덤화용).

    `KeypointMaxAtGoalHeld`의 판정(d<eps & tripod gate, hold_steps 유지)을 그대로 쓰되, termination이
    아니라 reward로 준다. chopstick의 GoalReachedBonus와 같은 구조 — 성공해도 에피소드 유지 →
    에피소드 내 goal 재샘플로 여러 goal 처리. goal당 1회(_awarded)라 반복 성공 파밍 없음.
    반환은 float(성공 스텝에 1.0). transport_success RewTerm이 is_terminated_term 대신 이걸 쓰게 배선.
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
        keypoint_eps: float = 0.05,
        gate_threshold: float = 0.3,
        hold_steps: int = 15,
        symmetry: str = "square_prism_y_tip",
    ) -> torch.Tensor:
        # goal 재샘플 시 이 goal의 성공 카운트·지급 이력 재장전.
        changed = _goal_changed(self, _goal_key(env, command_name))
        self._awarded = self._awarded & (~changed)
        self._count = torch.where(changed, torch.zeros_like(self._count), self._count)
        d = square_prism_keypoint_goal_distance(
            env, command_name, object_cfg, object_half_extent, symmetry, reduce="max",
        )
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
        valid = (d < keypoint_eps) & (gate > gate_threshold)
        self._count = torch.where(valid, self._count + 1, torch.zeros_like(self._count))
        newly_success = (self._count >= hold_steps) & (~self._awarded)
        self._awarded = self._awarded | newly_success
        return newly_success.float()


@configclass
class BoxTransportCommandsCfg:
    """Command terms for the MDP."""

    # 7D 운반 pose goal. 현재 position/orientation 모두 고정이고, 이후 ranges만 랜덤화함.
    # position z는 box_transport_env_cfg.__post_init__에서 BASE_Z 기준으로 오버라이드.
    cube_goal = mdp.UniformCubeGoalCommandCfg(
        asset_name="cube",
        # 2026-07-25 (전환): 주먹(wrap) 파지 + **고정 palm-up 목표 자세**로 결정. keypoint가 위치·자세
        #   둘 다 매칭(09-43-43에서 err_pos 4cm·ori 3°로 검증된 고정-pose 방식). 랜덤 pose(keypoint가
        #   자세를 못 좁히던 1.57~3.14)는 철회. resample도 고정 goal이라 무의미 → 끔(1e9).
        #   ⚠ yaw 0.785398이 "palm-up"인지는 학습된 wrap grasp의 실제 자세를 측정해 확정할 것(현재 근사).
        #   (직전 랜덤값 백업: box_mdp_cfg.py.keypoint_backup_2026-07-25, roll/yaw (1.57,π) resample(5,10))
        resampling_time_range=(1000000000.0, 1000000000.0),
        debug_vis=True,
        ranges=mdp.UniformCubeGoalCommandCfg.Ranges(
            pos_x=(0.62, 0.62),
            pos_y=(-0.20, -0.20),
            pos_z=(0.45, 0.45),  # placeholder — __post_init__에서 BASE_Z + 0.20
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.785398, 0.785398),
        ),
    )


@configclass
class BoxTransportActionsCfg:
    """Action specifications for the MDP. arm_action은 indy_wuji_box/env_cfg.py에서 채움."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class BoxTransportObservationsCfg:
    """Observation specifications for the MDP. policy = 76 (pose 67 + grip regions 9)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        cube_pos = ObsTerm(
            func=mdp.object_position_relative,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )
        cube_in_fingertips = ObsTerm(
            func=mdp.object_position_relative_to_bodies,
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
                "object_cfg": SceneEntityCfg("cube"),
            },
        )
        cube_to_goal = ObsTerm(
            func=mdp.object_position_error_to_command,
            params={
                "command_name": "cube_goal",
                "object_cfg": SceneEntityCfg("cube"),
            },
        )
        # 2026-07-16 신규: env마다 상자가 달라서 정책이 자기 상자의 치수를 알아야 함
        box_size = ObsTerm(
            func=mdp.object_dims,
            params={"object_cfg": SceneEntityCfg("cube")},
        )
        # 2026-07-16 신규: 직육면체는 좁은 면을 가로질러 잡아야 해서 방향이 기능임.
        # 초기 회전은 아직 고정 — 접촉 중 회전에만 신호가 흐름 ("반쯤 살아있는" 채널,
        # 초기 yaw 랜덤화는 사수님 컨펌 후 켜기로 함 2026-07-15)
        box_quat = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("cube"), "make_quat_unique": True},
        )
        # 2026-07-19 신규 (+3 → policy 67): 최근접 대칭 목표까지의 상대 회전 axis-angle.
        # raw quat만으론 대칭 등가("90° 돌린 자세 = 정답")를 발굴해야 해서 크게 되돌리기가
        # 관찰됨 (play 실측) — 목표 대비 signed 회전 오차를 직접 제공. obs dim 변경 = fresh.
        box_ori_to_target = ObsTerm(
            func=mdp.object_ori_error_nearest_sym,
            params={
                "command_name": "cube_goal",
                "object_cfg": SceneEntityCfg("cube"),
            },
        )

        # 각 fingertip에서 물체 로컬 rear grip region의 최근접점까지의 오차를 palm frame으로
        # 제공함. helper가 env.box_half_extents를 읽으므로 랜덤 box 크기도 자동 반영됨.
        index_grip_error = ObsTerm(
            func=fg_mdp.index_grip_error_b,
            params={
                "palm_cfg": _PALM_CFG,
                "index_cfg": _INDEX_CFG,
                "object_cfg": _OBJECT_CFG,
                "object_half_extent": _HALF_FALLBACK,
                **_INDEX_GRIP_REGION,
            },
        )
        thumb_grip_error = ObsTerm(
            func=fg_mdp.thumb_grip_error_b,
            params={
                "palm_cfg": _PALM_CFG,
                "thumb_cfg": _THUMB_CFG,
                "object_cfg": _OBJECT_CFG,
                "object_half_extent": _HALF_FALLBACK,
                **_THUMB_GRIP_REGION,
            },
        )
        # 2026-07-23 신규 (+3 → policy 76): 중지 grip region 오차. chopstick과 동일 구성.
        middle_grip_error = ObsTerm(
            func=fg_mdp.middle_grip_error_b,
            params={
                "palm_cfg": _PALM_CFG,
                "middle_cfg": _MIDDLE_CFG,
                "object_cfg": _OBJECT_CFG,
                "object_half_extent": _HALF_FALLBACK,
                **_MIDDLE_GRIP_REGION,
            },
        )

        # goal quaternion을 별도 4D로 넣지 않는 이유: 위 상대 axis-angle 3D가 현재 자세와
        # command 자세를 모두 포함하고 대칭까지 해소함. orientation 범위를 랜덤화해도 obs 73 유지.

        action_history = ObsTerm(func=mdp.action_history)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class BoxTransportEventCfg:
    """Configuration for events."""

    # ★ env별 상자 치수 (prestartup = sim 시작 전 USD 조작, 런 내내 고정).
    # 기본값 = 기존 크기 (단면 3~6cm × 길이비 1.5~3, 07-16 사수님 방향) — 슬롯 A(ori v1)가
    # 오버라이드 없이 재현되도록 (2026-07-18 사용자 결정).
    # 크기 확장 (슬롯 B, 2026-07-18 사용자 값: 단면 하한 1.5cm + 길이 최대 20cm =
    # 1.5×1.5×20 젓가락 프록시 포함)은 CLI 오버라이드로:
    #   "env.events.randomize_box.params.width_range=[0.015,0.06]" \
    #   "env.events.randomize_box.params.length_range=[0.0,0.20]"
    # (length_range를 주면 길이가 폭과 독립인 U(1.5×폭, max)로 샘플됨 — events.py 참고)
    # ⚠ 단면 2.8cm 미만은 커플링 손끝 간격 실측 밖 미검증 구간 — 크기-버킷으로 판독.
    randomize_box = EventTerm(
        func=mdp.randomize_box_dims,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "width_range": (0.02, 0.02),
            "ratio_range": (1,1),
            "length_range": (0.18, 0.18),  # None = ratio 방식 (기존 경로). 오버라이드용 자리
            "base_size": 0.06,   # spawn CuboidCfg size와 일치해야 함
            "length_axis": 1,    # 길이는 y (테이블 긴 축)
        },
    )

    # env별 스폰 z = 상판 + 자기 반높이 (surface_z는 env_cfg에서 BASE_Z로 오버라이드)
    set_box_height = EventTerm(
        func=mdp.set_box_default_height,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("cube"), "surface_z": 0.0},
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 약지/새끼 목표 채우기 (리셋~첫 액션 공백 커버, 큐브 태스크와 동일한 이유)
    hold_folded_fingers = EventTerm(
        func=mdp.hold_joints_at_default,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["finger[4-5]_joint[1-4]"])},
    )

    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {
                "x": (-0.06, 0.06),
                "y": (-0.08, 0.08),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
        },
    )


@configclass
class BoxTransportRewardsCfg:
    """Reward terms — reach/hold/lift 뒤 position transport와 orientation을 분리한 구조.

    가중치·파라미터는 2026-07-16 시점의 CubeGraspRewardsCfg 사본. 설계 근거 주석은 원본
    (env_cfg_common.py) 참고. half_extent 인자는 전부 fallback — env별 버퍼가 우선.
    """

    finger_cage_reach = RewTerm(
        func=mdp.ObjectCageProgressReward,
        weight=8.0,
        params={
            "asset_cfg": BOX_CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "distance_max": 0.5,
            "palm_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
            "palm_normal_b": (0.19, 0.28, 0.94),
            "gate_floor": 0.0,
        },
    )

    # Slot B에서 검증한 constraint-based fingertip shaping. 현재 success에는 넣지 않고
    # dense reward로만 사용해 기존 Slot A 사다리에서 파지 자세 개선 효과를 먼저 분리함.
    index_grip = RewTerm(
        func=fg_mdp.FingertipGripProgressReward,
        weight=40.0,
        params={
            "palm_cfg": _PALM_CFG,
            "fingertip_cfg": _INDEX_CFG,
            "object_cfg": _OBJECT_CFG,
            "object_half_extent": _HALF_FALLBACK,
            **_INDEX_GRIP_REGION,
            "distance_scale": 0.20,
        },
    )
    thumb_grip = RewTerm(
        func=fg_mdp.FingertipGripProgressReward,
        weight=40.0,
        params={
            "palm_cfg": _PALM_CFG,
            "fingertip_cfg": _THUMB_CFG,
            "object_cfg": _OBJECT_CFG,
            "object_half_extent": _HALF_FALLBACK,
            **_THUMB_GRIP_REGION,
            "distance_scale": 0.20,
        },
    )
    # 2026-07-23 신규: 중지에도 semantic region shaping (chopstick과 동일, weight 40).
    middle_grip = RewTerm(
        func=fg_mdp.FingertipGripProgressReward,
        weight=40.0,
        params={
            "palm_cfg": _PALM_CFG,
            "fingertip_cfg": _MIDDLE_CFG,
            "object_cfg": _OBJECT_CFG,
            "object_half_extent": _HALF_FALLBACK,
            **_MIDDLE_GRIP_REGION,
            "distance_scale": 0.20,
        },
    )
    # 2026-07-25 wrap: 약지·새끼도 +x면 접촉 유도 (progress형, 검지·중지와 동일 구조).
    ring_grip = RewTerm(
        func=fg_mdp.FingertipGripProgressReward,
        weight=40.0,
        params={
            "palm_cfg": _PALM_CFG,
            "fingertip_cfg": _RING_CFG,
            "object_cfg": _OBJECT_CFG,
            "object_half_extent": _HALF_FALLBACK,
            **_RING_GRIP_REGION,
            "distance_scale": 0.20,
        },
    )
    pinky_grip = RewTerm(
        func=fg_mdp.FingertipGripProgressReward,
        weight=40.0,
        params={
            "palm_cfg": _PALM_CFG,
            "fingertip_cfg": _PINKY_CFG,
            "object_cfg": _OBJECT_CFG,
            "object_half_extent": _HALF_FALLBACK,
            **_PINKY_GRIP_REGION,
            "distance_scale": 0.20,
        },
    )

   # finger_cage_hold = RewTerm(
   #     func=mdp.object_in_finger_cage,
   #     weight=15.0,
   #     params={
   #         "asset_cfg": BOX_CAGE_BODIES,
   #         "object_cfg": SceneEntityCfg("cube"),
   #         "object_half_extent": _HALF_FALLBACK,
   #         "num_points": 3,
   #         "point_fractions": _POINT_FRACTIONS,
   #         "sphere_radius": 0.005,
   #         "depth_max": 0.005,
   #     },
   # )

    # 2026-07-26_00-04-17 기준: hold는 엄지 대 검지·중지·약지의 quad cage.
    # 새끼는 pinky_grip progress만 받고 필수 게이트에서는 제외한다.
    finger_cage_hold = RewTerm(
        func=fg_mdp.balanced_quad_cage_gate,
        weight=15.0,
        params={
            "index_cage_cfg": BOX_INDEX_CAGE_BODIES,
            "middle_cage_cfg": BOX_MIDDLE_CAGE_BODIES,
            "ring_cage_cfg": BOX_RING_CAGE_BODIES,
            "object_cfg": _OBJECT_CFG,
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "sphere_radius": 0.005,
            "depth_max": 0.005,
        },
    )



   # cube_lift = RewTerm(
   #     func=mdp.object_lift_in_cage,
   #     weight=500.0,
   #     params={
   #         "asset_cfg": BOX_CAGE_BODIES,
   #         "object_cfg": SceneEntityCfg("cube"),
   #         "object_half_extent": _HALF_FALLBACK,
   #         "num_points": 3,
   #         "point_fractions": _POINT_FRACTIONS,
   #         "sphere_radius": 0.005,
   #         "depth_max": 0.005,
   #         # 0.08 → 0.20 (2026-07-19): 8cm 포화가 "10cm 호버 = lift 만점" 급여를 만들어
   #         # goal 접근 유인이 죽는 것 실측 (transport 정체 2.6 + clearance만 상승).
   #         # goal 높이(BASE_Z+0.20)까지 상승 매 cm가 연금이 되도록 포화점을 goal로.
   #         "lift_height": 0.08,
   #     },
   # )

    # 2026-07-26_00-04-17 기준: lift도 quad cage를 사용한다.
    cube_lift = RewTerm(
        func=fg_mdp.object_lift_in_balanced_quad_cage,
        weight=150.0,
        params={
            "index_cage_cfg": BOX_INDEX_CAGE_BODIES,
            "middle_cage_cfg": BOX_MIDDLE_CAGE_BODIES,
            "ring_cage_cfg": BOX_RING_CAGE_BODIES,
            "object_cfg": _OBJECT_CFG,
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "lift_height": 0.20,
            "gate_exponent": 1.0,
        },
    )

    # 운반 1단계: 상자 중심의 position error만 best-so-far로 줄임.
    # 위치/각도를 keypoint 하나에 섞으면 원거리에서 위치가 각도 신호를 덮고, 어느 축이
    # 병목인지 진단하기 어려워 분리함. 왕복 이동은 best-so-far라 추가 지급되지 않음.
    # ⚠ SimToolReal A/B (2026-07-22): 위치·자세를 keypoint_goal 한 항으로 융합해 chopstick(쿼터니안+exp)과 비교.
    #   cube_transport·box_orientation은 weight 0으로 끄고(복구용 유지), 아래 keypoint_goal 활성.
    cube_transport = RewTerm(
        func=BalancedObjectToGoalProgressReward,
        weight=0.0,
        params={
            "command_name": "cube_goal",
            "index_cage_cfg": BOX_INDEX_CAGE_BODIES,
            "middle_cage_cfg": BOX_MIDDLE_CAGE_BODIES,
            "object_cfg": _OBJECT_CFG,
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "potential_eps": 0.05,
        },
    )

    # 운반 2단계: position < 5cm + cage gate > 0.3을 한 번 만족하면 latch 활성화.
    # 이후 최근접 대칭 orientation error의 에피소드 최저 기록을 갱신한 만큼만 양수 지급함.
    # 악화와 이전 최저 error까지의 복구는 0이라 왕복 진동으로 반복 적립할 수 없음.
    # 45° 순개선을 raw 합계 약 1로 정규화해 position progress와 동일 weight에서 시작함.
    box_orientation = RewTerm(
        func=BalancedObjectOrientationProgressReward,
        weight=0.0,
        params={
            "command_name": "cube_goal",
            "index_cage_cfg": BOX_INDEX_CAGE_BODIES,
            "middle_cage_cfg": BOX_MIDDLE_CAGE_BODIES,
            "object_cfg": _OBJECT_CFG,
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "activation_distance": 0.05,
            # balanced gate=min(    index, middle)이 0.3을 넘을 때 orientation 단계 활성화.
            "activation_gate_threshold": 0.3,
            "angle_scale": 0.7853981633974483,
        },
    )

    # ── SimToolReal keypoint 목표 (활성, 2026-07-22) — 위 transport·orientation(둘 다 0) 대체 ──
    # r = tripod gate × max(d_best − d, 0), d = 8-corner max 거리(roll 4-sym, flip 제외, m).
    # 00-04-17은 hold/lift만 quad이고 keypoint/success는 엄지 대 검지·중지 tripod를 사용했다.
    # ⚠ weight는 시작값 — d가 m 단위(φ와 스케일 다름)라 실측 보고 조정. keypoint success 판정은 아래 success 참고.
    keypoint_goal = RewTerm(
        func=KeypointMaxGoalProgressReward,
        weight=150000.0,
        params={
            "command_name": "cube_goal",
            "index_cage_cfg": BOX_INDEX_CAGE_BODIES,
            "middle_cage_cfg": BOX_MIDDLE_CAGE_BODIES,
            "object_cfg": _OBJECT_CFG,
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "symmetry": "square_prism_y_tip",
        },
    )

    # B안 통합 연금: gate × φ(d) — "잡은 채 goal 근처에 있는 것"에 매 스텝 지급.
    # A′(lift0, run 2026-07-17_23-15-16) 실측 처방: 일시불 φ는 현금화 후 goal 체류가
    # 무보상이라 내려놓고 hold 파밍 → 정착 실패. 연금은 그 계곡을 메움.
    # 기본 weight 0 (파일 기본 = 검증된 A안 승자 구성). B안 fresh는 CLI 오버라이드로:
    #   env.rewards.cube_lift.weight=0 env.rewards.cube_transport.weight=0 \
    #   env.rewards.goal_proximity.weight=75
    # w75 근거: goal 중심 ~2.5/스텝, 경계 캠핑 현재가치 ≪ r_T +1000 (rewards.py docstring).
    # keypoint 전환(2026-07-22): 150 → 0. 위치-only 연금은 "위치만 가깝고 자세 안 맞춤" 파밍 여지가
    # 있어 keypoint MAX 결합을 깎음(사용자 최종안). 학습 후 goal 근처서 자꾸 놓으면 소량(≤20)만 재활성.
    goal_proximity = RewTerm(
        func=balanced_object_goal_proximity,
        weight=0.0,
        params={
            "command_name": "cube_goal",
            "index_cage_cfg": BOX_INDEX_CAGE_BODIES,
            "middle_cage_cfg": BOX_MIDDLE_CAGE_BODIES,
            "object_cfg": _OBJECT_CFG,
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "potential_eps": 0.05,
        },
    )

    # r_T: keypoint 도달(d<eps) + gate 물림 0.5s 유지 → 한 방 +2000 보너스. **종료 안 함** (2026-07-24).
    #   success termination을 분리해 KeypointGoalReachedBonus로 교체 — 성공해도 에피소드 유지 → 재샘플로
    #   다음 goal 처리. goal당 1회(_awarded)라 반복 성공 파밍 없음. (chopstick GoalReachedBonus와 동일 구조)
    #   (기존: func=mdp.is_terminated_term, term_keys="success" — success 종료 참조. 복구용)
    transport_success = RewTerm(
        func=KeypointGoalReachedBonus,
        weight=60000.0,
        params={
            "command_name": "cube_goal",
            "index_cage_cfg": BOX_INDEX_CAGE_BODIES,
            "middle_cage_cfg": BOX_MIDDLE_CAGE_BODIES,
            "object_cfg": _OBJECT_CFG,
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "keypoint_eps": 0.05,
            "gate_threshold": 0.3,
            "hold_steps": 15,
            "symmetry": "square_prism_y_tip",
        },
    )

    # 낙하 정액 벌금. fresh 단계에서는 0 (탐색 회피 함정 실측, 2026-07-15) —
    # 잡기가 자리 잡은 뒤 resume에서 -3000으로 켜는 2단계 커리큘럼
    drop_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=0.0,
        params={"term_keys": "cube_dropped"},
    )

    # palm_normal_b = "파지 개구부 축" (misnomer: 손바닥 법선 아님).
    # 도출 근거·재검증은 rewards.py palm_facing_object docstring(★) 참고
    palm_facing = RewTerm(
        func=mdp.PalmFacingProgressReward,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
            "object_cfg": SceneEntityCfg("cube"),
            "palm_normal_b": (0.19, 0.28, 0.94),
        },
    )

    arm_manipulability = RewTerm(
        func=mdp.arm_manipulability_penalty,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link"], joint_names=["joint[0-5]"]),
            "j_max": 0.02,
        },
    )

    # clearance 0.02 → 0.01 (2026-07-19): 얇은 스틱(단면 1.5~3cm)은 파지 지점이 상판 위
    # 0.75~1.5cm라 2cm 존이 "잡으러 가기"를 벌함 — 실측: 스틱 런(12-48-25)에서 hand_floor
    # raw 악화(−0.77→−0.82) + 중지 접근 정체(0.14) + cage 관통 0. 1cm면 스틱 파지 높이 합법.
    # 7월 20일 run -> (weight: 1.0, clearance: 0.02)
    # chopsticks -> (weight: 0.0, clearance: 0.01)
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
class BoxTransportTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 낙하 실패 종료 (minimum_height는 env_cfg에서 BASE_Z - 0.05로 오버라이드)
    cube_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube")},
    )

    # ── success termination 제거 (2026-07-24, chopstick과 동일) ────────────────────
    # 성공 시 종료하면 첫 성공에 리셋돼 에피소드 내 goal 재샘플로 다음 goal을 못 봄.
    # 성공 '보너스'는 유지(위 rewards.transport_success = KeypointGoalReachedBonus, goal당 1회),
    # 종료는 time_out(8초) + cube_dropped만. ⚠ Episode_Termination/success 사라짐 →
    # 성공 빈도는 Episode_Reward_Raw/transport_success로 봄.
    #
    # (복구용) 기존 success 종료 — KeypointMaxAtGoalHeld, keypoint_eps 0.05 / gate 0.3 / hold_steps 15 /
    #   symmetry square_prism_y_tip. 되살리면 transport_success도 is_terminated_term(term_keys="success")으로.
    #   (더 옛: BalancedObjectAtGoalHeld, goal_radius 0.05 / ori_limit 0.2617993877991494.)
    # success = DoneTerm(func=KeypointMaxAtGoalHeld, params={... 위 KeypointGoalReachedBonus와 동일 ...})
