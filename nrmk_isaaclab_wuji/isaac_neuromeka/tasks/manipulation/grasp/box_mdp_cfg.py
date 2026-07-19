"""Indy-Wuji-Box-Transport 전용 MDP 설정 (2026-07-16).

CubeGrasp*Cfg(env_cfg_common.py)의 사본 + 랜덤 직육면체 확장. 큐브 태스크를 동결하기 위해
기존 클래스를 수정하지 않고 복제함 (사용자 지시). 큐브 쪽과의 차이:
  - 관측 +7: box_size(3) + box_quat(4) -> policy 64
  - 이벤트 +2: randomize_box_dims(prestartup, env별 비율보존 치수) + set_box_default_height(startup)
  - scene.replicate_physics = False 필요 (box_transport_env_cfg.__post_init__에서 설정)

⚠ 활성 태스크: Indy-Wuji-Box-Transport 전용. 큐브 태스크(Indy-Wuji-Cube-Grasp)는
  env_cfg_common.py의 CubeGrasp*Cfg를 씀 — 여기를 고쳐도 큐브 태스크에는 반영 안 됨 (역도 같음).
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp

# 가상점 12개용 body 목록. env_cfg_common.CAGE_BODIES와 같은 구성이지만 인스턴스를 공유하지
# 않음 — SceneEntityCfg는 resolve 시 내부 상태(body_ids)가 채워지는 가변 객체라 태스크 간
# 공유가 위험함. preserve_order=True 필수 (엄지가 기준점 자리).
BOX_CAGE_BODIES = SceneEntityCfg(
    "robot",
    body_names=[
        "finger1_tip_link",  # 엄지끝: 모든 선분의 기준점
        "finger2_tip_link",
        "finger2_link3",
        "finger3_tip_link",
        "finger3_link3",
    ],
    preserve_order=True,
)

# ── 감싸쥐기 cage 부품 (v2 후보, 2026-07-19 준비 — 미배선, 스틱 기준선 결과 대기) ──
# 근거 사슬: ori v1(무유도)·v1.1(keypoint 유인) 모두 success 0, 슬롯 A/B 공통 매달림
#   70~93° + play "손가락 2개 불안정 파지" 관찰 → 손끝 pinch는 긴 상자의 중력 토크를
#   못 버팀 (2점 접촉 = 경첩). 힘(depth_max)이 아니라 기하(접촉 분산)의 문제.
# 개편 두 축: 폭(약지/새끼 추가 — 길이축 팔길이 확보. 액션은 커플링으로 이미 오므려짐)
#   + 깊이(근위 마디 link2 — 물체를 손바닥 쪽으로 끌어들여 감싸쥐기 유도).
# 활성화: 이 정의 아래에 `BOX_CAGE_BODIES = BOX_CAGE_BODIES_WRAP` 한 줄 추가
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

# ★ 2026-07-19 활성화 (사용자): 감싸쥐기 라운드 — 슬롯 A(기존 크기+ori) ∥ 슬롯 B(얇은 스틱).
# 되돌리기 = 이 줄 삭제 (손끝 pinch 기준선 복귀).
BOX_CAGE_BODIES = BOX_CAGE_BODIES_WRAP

# 보상/판정의 half_extent 인자는 fallback임 — 실제로는 randomize_box_dims가 저장한
# env.box_half_extents (N,3) 버퍼가 우선함 (rewards.py의 _cage_sdf/box_ground_clearance 참고).
_HALF_FALLBACK = (0.03, 0.03, 0.03)
_POINT_FRACTIONS = (0.1, 0.5, 0.9)


@configclass
class BoxTransportCommandsCfg:
    """Command terms for the MDP."""

    # 운반 goal. 1단계는 고정점 (box_transport_env_cfg.__post_init__에서 BASE_Z 파생 오버라이드)
    cube_goal = mdp.UniformCubeGoalCommandCfg(
        asset_name="cube",
        resampling_time_range=(1.0e9, 1.0e9),  # 에피소드 내 고정 (차분층 telescoping 보호)
        debug_vis=True,
        ranges=mdp.UniformCubeGoalCommandCfg.Ranges(
            pos_x=(0.62, 0.62),
            pos_y=(-0.20, -0.20),
            pos_z=(0.45, 0.45),  # placeholder — __post_init__에서 BASE_Z + 0.20
        ),
    )


@configclass
class BoxTransportActionsCfg:
    """Action specifications for the MDP. arm_action은 indy_wuji_box/env_cfg.py에서 채움."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class BoxTransportObservationsCfg:
    """Observation specifications for the MDP. policy = 64 (큐브 57 + size 3 + quat 4)."""

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

        # ── ③ 랜덤 ori v2 배관 (계획 2026-07-18, AGENTS '확정 진행 순서' — ② 통과 후) ──
        # goal 자세가 상수인 동안 이 obs를 켜면 죽은 채널 (절대 단독 활성 금지).
        # 활성 세트 4개를 "한 fresh"에 함께 투입 (obs dim 64→68 = fresh 필수):
        #  1) commands.py UniformCubeGoalCommand: quat 샘플 추가 (pose command화, 초기엔
        #     yaw만: 대칭 때문에 yaw ∈ (-45°, 45°)면 전 자세 커버). command 출력 3→7.
        #  2) 아래 obs 주석 해제 (goal 자세 — 또는 상자→goal 상대 quat이 더 학습 친화적).
        #  3) rewards.py square_prism_ori_error를 상대 quat(q_goal⁻¹ ⊗ q_box) 시그니처로
        #     확장하고 success(ObjectAtGoalHeld)가 command quat을 읽게 연결.
        #  4) goal 마커를 구슬 → 자세 보이는 ghost box로 (commands.py 마커 cfg).
        # ⚠ 재활성 시 시그니처 대조 필수 (ObjectToGoalProgressReward 파라미터 사고 전례).
        # goal_ori = ObsTerm(
        #     func=mdp.generated_commands,  # command가 quat을 포함하게 된 후 (접점 1 선행)
        #     params={"command_name": "cube_goal"},
        # )

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
            "width_range": (0.03, 0.06),
            "ratio_range": (1.5, 3.0),
            "length_range": None,  # None = ratio 방식 (기존 경로). 오버라이드용 자리
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
    """Reward terms — 사다리: reach(8) < hold(15) < lift(100) < transport(500) < r_T(15000).

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

    finger_cage_hold = RewTerm(
        func=mdp.object_in_finger_cage,
        weight=15.0,
        params={
            "asset_cfg": BOX_CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "sphere_radius": 0.005,
            "depth_max": 0.005,
        },
    )

    cube_lift = RewTerm(
        func=mdp.object_lift_in_cage,
        weight=50.0,
        params={
            "asset_cfg": BOX_CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "lift_height": 0.08,
        },
    )

    # 운반 층 v1.1 (2026-07-18): keypoint φ — 위치+자세를 꼭짓점 8개 거리 하나로 통합.
    # 근거: ori v1(판정만, run 15-37-42)이 iter 4,230 동안 success 0 — 나르는 자세 72~93°로
    # 씨앗 전무 + ori 참조 보상이 없어 내려갈 힘도 없음 → shaping 필요 실측.
    # 자세 오차는 스폰 만점이라 단독 차분항 불가(평생 0원) → 거리에 흡수 (TriFinger 방식,
    # 지급은 기존 best-so-far 일시불 유지). 상세는 KeypointGoalProgressReward docstring.
    # (구 중심거리 버전 ObjectToGoalProgressReward는 큐브 태스크가 계속 사용)
    cube_transport = RewTerm(
        func=mdp.KeypointGoalProgressReward,
        weight=4000.0,
        params={
            "command_name": "cube_goal",
            "asset_cfg": BOX_CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "potential_eps": 0.05,
        },
    )

    # B안 통합 연금: gate × φ(d) — "잡은 채 goal 근처에 있는 것"에 매 스텝 지급.
    # A′(lift0, run 2026-07-17_23-15-16) 실측 처방: 일시불 φ는 현금화 후 goal 체류가
    # 무보상이라 내려놓고 hold 파밍 → 정착 실패. 연금은 그 계곡을 메움.
    # 기본 weight 0 (파일 기본 = 검증된 A안 승자 구성). B안 fresh는 CLI 오버라이드로:
    #   env.rewards.cube_lift.weight=0 env.rewards.cube_transport.weight=0 \
    #   env.rewards.goal_proximity.weight=75
    # w75 근거: goal 중심 ~2.5/스텝, 경계 캠핑 현재가치 ≪ r_T +1000 (rewards.py docstring).
    goal_proximity = RewTerm(
        func=mdp.object_goal_proximity,
        weight=0.0,
        params={
            "command_name": "cube_goal",
            "asset_cfg": BOX_CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "potential_eps": 0.05,
        },
    )

    # 논문 r_T: goal ±5cm + gate 물림 0.5s 유지 -> 한 방 +500 + 즉시 종료
    transport_success = RewTerm(
        func=mdp.is_terminated_term,
        weight=30000.0,
        params={"term_keys": "success"},
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
    hand_floor = RewTerm(
        func=mdp.hand_floor_penalty,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link", "finger.*"]),
            "clearance": 0.01,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)


@configclass
class BoxTransportTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 낙하 실패 종료 (minimum_height는 env_cfg에서 BASE_Z - 0.05로 오버라이드)
    cube_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube")},
    )

    # 논문 r_T의 성공 종료 (운반판): goal 반경 안 + gate 물림 0.5s 유지
    # 2026-07-16 재활성: goal 도달 + 유지 = 성공 종료 (hold_steps 15 = 0.5s "고정")
    success = DoneTerm(
        func=mdp.ObjectAtGoalHeld,
        params={
            "command_name": "cube_goal",
            "asset_cfg": BOX_CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": _HALF_FALLBACK,
            "num_points": 3,
            "point_fractions": _POINT_FRACTIONS,
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "goal_radius": 0.05,
            "gate_threshold": 0.3,
            "hold_steps": 15,  # 0.5s @ 30Hz
            # orientation v1 (2026-07-18): 스폰 자세(월드 정렬) 대비 대칭 최소각 15° 이내
            # 조건 추가 = "기울이거나 굴리지 않고 나르기". goal 자세는 아직 상수라 obs 불변
            # (box_quat이 이미 obs에 있음 — goal 자세 랜덤화 때 obs 확장, AGENTS 로드맵 2).
            # 끄려면 CLI: env.terminations.success.params.ori_limit=null
            "ori_limit": 0.2618,  # 15 deg [rad]
        },
    )
