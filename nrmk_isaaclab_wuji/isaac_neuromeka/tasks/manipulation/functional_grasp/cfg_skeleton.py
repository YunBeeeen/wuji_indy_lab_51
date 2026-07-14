"""ChopsticksGrasp의 cfg 골격 (아직 배선 안 함 — 구조/흐름만).

완성되면 이 내용을 `common/env_cfg_common.py`(또는 여기)에 정식으로 넣고
`chopsticks_grasp_env_cfg.py`가 `CubeGrasp*Cfg` 대신 이걸 쓰게 바꾼다.

지금 `chopsticks_grasp_env_cfg.py`는 `CubeGrasp*Cfg`를 그대로 공유하고 있고,
`object_half_extent`만 `__post_init__`에서 덮어쓰고 있다 (임시).
"""

from __future__ import annotations

# =============================================================================
# 전체 흐름
# =============================================================================
#
#   [reset] ─ 물체를 랜덤 자세로 스폰 (누워 있거나, 굴러가 있거나)
#      │      팔은 manipulability 높은 중립 자세
#      │
#      ├──> g_local (물체 로컬 상수) ──┐
#      │                              │
#   [step]                            │
#      │                              ▼
#      ├─ 물체 pose ────────> g_world = 물체pose ∘ g_local
#      │       │                      │
#      │       │                      ├─> Δhp, Δhr, Δhj ─> r_hp, r_hr, r_hj ─> r_grasp
#      │       │                      └─> r_T (셋 다 임계 이하 = 성공, 희소, x5000)
#      │       │
#      │       └─> Δo_r (nominal과의 각도) ─────────────> r_orient (x500)
#      │
#      ├─ cage 가상점 ──> r_reach (x1, 기존 재사용)
#      │                 r_hold  (x25, 기존 재사용)
#      │
#      ├─ arm Jacobian ─> r_MP   (x1, 기존 재사용)
#      └─ 물체 높이 ────> r_lift (x1, 기존 재사용)
#
#   [terminate]  (i) r_T 만족 AND 물체가 들림  -> 성공
#                (ii) 물체가 바닥/테이블 밖으로 떨어짐  -> 실패
#                (iii) time_out
#
# =============================================================================


# -----------------------------------------------------------------------------
# 1. Observations  ★ 여기가 큐브 task와 결정적으로 다름
# -----------------------------------------------------------------------------
# @configclass
# class ChopsticksGraspObservationsCfg:
#     class PolicyCfg(ObsGroup):
#         joint_pos      = ObsTerm(...)   # 기존: 제어 관절 18
#         action_history = ObsTerm(...)   # 기존
#
#         # ★ 신규 — 정책이 "목표"를 모르면 갈 수가 없다.
#         #   논문 상태 표현: h = [hp, hr, hj, o_p, o_r, ...]  (손 상태 + 물체 상태 + 목표 파지)
#         target_grasp_pos_b  = ObsTerm(...)   # g_world의 손 기준 상대 위치 (3)
#         target_grasp_rot_b  = ObsTerm(...)   # g_world의 손 기준 상대 회전 (4 또는 6D)
#         target_grasp_joints = ObsTerm(...)   # hj 목표 관절각 (12)
#         object_pos_b        = ObsTerm(...)   # 물체 위치 (손 기준) (3)
#         object_rot_b        = ObsTerm(...)   # 물체 회전 (4 또는 6D)
#
# obs 차원이 늘어남 -> 기존 체크포인트와 호환 불가. 처음부터 학습해야 함.
#
# ★ 상대 좌표(손 기준)로 줄 것. 월드 절대좌표로 주면 일반화가 안 됨.


# -----------------------------------------------------------------------------
# 2. Rewards
# -----------------------------------------------------------------------------
# @configclass
# class ChopsticksGraspRewardsCfg:
#     # ---- r_grasp = r_hp + r_hr + λ·r_hj  (Eq. 8) ----
#     hand_position = RewTerm(func=mdp.HandPositionProgressReward, weight=1.0, ...)
#     hand_rotation = RewTerm(func=mdp.HandRotationProgressReward, weight=1.0, ...)
#     hand_joints   = RewTerm(func=mdp.HandJointProgressReward,    weight=1.0, ...)
#         # λ는 함수 안에서 Δhp/Δhr로부터 계산 (Eq. 12). weight가 아니라 내부 게이트.
#         # ★ λ가 곧 순서 강제: 손이 멀면 λ≈0 -> 손가락을 미리 오므릴 이유가 없음.
#         #   우리가 큐브에서 게이팅으로 흉내내던 것을 논문은 이 λ로 함.
#
#     # ---- r_man = r_reach + r_hold + r_orient  (Eq. 13) ----
#     finger_cage_reach = RewTerm(func=mdp.ObjectCageProgressReward, weight=1.0, ...)
#         # ★ 기존 것 그대로. 단 palm_cfg(게이팅) 파라미터를 넘기지 말 것.
#         #   r_hr이 방향을 직접 보상하므로 게이팅이 필요 없음.
#     finger_cage_hold  = RewTerm(func=mdp.object_in_finger_cage,    weight=25.0, ...)
#     object_orient     = RewTerm(func=mdp.ObjectOrientProgressReward, weight=500.0, ...)
#
#     # ---- 나머지 ----
#     target_reached    = RewTerm(func=mdp.target_grasp_reached, weight=5000.0, ...)   # r_T, 희소
#     object_lift       = RewTerm(func=mdp.object_lift_in_cage,   weight=1.0, ...)     # r_lift
#     arm_manipulability= RewTerm(func=mdp.arm_manipulability_penalty, weight=1.0, ...) # r_MP
#
#     # ---- 우리 추가분 (논문에 없지만 필요) ----
#     hand_floor  = RewTerm(func=mdp.hand_floor_penalty, weight=0.5, ...)
#     action_rate = RewTerm(func=mdp.action_rate_l2,     weight=-0.005)
#
#     # ★ palm_facing 은 제거. r_hr이 상위호환.
#
# 가중치 (논문 p.9):  r_reach 1 / r_hold 25 / r_orient 500 / r_T 5000
#   "Both scaling factors are chosen to be one order of magnitude less than the final reward
#    and the corresponding reward component with higher scaling."  (단계마다 약 20배)
#
# ★ 논문 값을 그대로 쓸 수 있는 이유: 가까운 시작 curriculum이 있기 때문.
#   커리큘럼 없이 r_reach를 1로 죽이면 접근 신호가 사라져 닭-달걀이 됨 (큐브에서 겪음).


# -----------------------------------------------------------------------------
# 3. Terminations  ★ 지금은 time_out 하나뿐. 논문대로 3개.
# -----------------------------------------------------------------------------
# @configclass
# class ChopsticksGraspTerminationsCfg:
#     success   = DoneTerm(func=mdp.grasp_success)   # r_T 만족 AND 물체가 들림
#     dropped   = DoneTerm(func=mdp.object_dropped)  # 물체가 작업영역 밖으로
#     time_out  = DoneTerm(func=mdp.time_out, time_out=True)
#
# 논문: "An episode is terminated when (i) a provided target constraint is satisfied and the
#        object is lifted off the table, (ii) an object falls from the table, or (iii) a maximum
#        number of 200 steps is reached."
#
# ★ 성공 시 조기 종료가 중요함 — 안 그러면 정책이 잡고 나서 뭉개면서 r_hold를 계속 긁음.


# -----------------------------------------------------------------------------
# 4. Events (reset)  ★ 커리큘럼이 여기 들어감
# -----------------------------------------------------------------------------
# @configclass
# class ChopsticksGraspEventCfg:
#     reset_all    = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
#     reset_object = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
#                              params={"pose_range": {...}})   # 위치 + 회전 랜덤
#
# close-start curriculum:
#     - 물체를 **손 5cm 앞**에, **nominal 자세**로 스폰  (논문: "5 cm away from the inner side
#       of the hand", "spawned upright in nominal configuration")
#     - 팔은 **manipulability 높은 중립 자세**
#     - **r_man (= r_reach + r_hold + r_orient) 비활성**  <- 논문이 명시적으로 끔
#       -> 접근/굴리기를 배울 필요가 없고, r_grasp + r_T 만 배움
#     - 성공률 50% 될 때까지
#
# full task:
#     - 전체 난이도. 물체가 누워 있거나 굴러가 있음
#     - r_man 활성
#     - 가까운 시작 run 체크포인트에서 --resume
#
# ★ "손 5cm 앞"은 물체 위치가 아니라 **손의 파지 개구부** 기준임.
#   실측: palm 원점에서 개구부 방향 (0.19, 0.28, 0.94)로 약 0.13 m 지점이 파지 중심.
#   큐브에서 "Easy" task를 만들 때 이걸 안 지켜서 아무 효과가 없었음
#   (손이 공중 67cm에 떠 있는데 물체를 수평으로만 12cm 옮겨서 거리가 2cm밖에 안 줄었음).
