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

        action_history = ObsTerm(func=mdp.action_history)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class BoxTransportEventCfg:
    """Configuration for events."""

    # ★ env별 상자 치수 (prestartup = sim 시작 전 USD 조작, 런 내내 고정).
    # 단면 3~6cm 정사각 × 길이비 1.5~3 (젓가락만큼 얇지 않은 범위, 사수님 방향).
    # 물리는 창(엄지-중지 2.8~5.8cm) 안에 단면이 들어가도록 폭 상한 6cm.
    randomize_box = EventTerm(
        func=mdp.randomize_box_dims,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "width_range": (0.03, 0.06),
            "ratio_range": (1.5, 3.0),
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
        weight=100.0,
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

    # 운반 층. 현재 선형 best-so-far — 역수형 φ(0.05/(0.05+d)) 전환 설계는 킵 상태
    # (worklog 2026-07-16 "transport-φ 설계 킵" 참고. 오버슈트 재현 시 적용)
    # 2026-07-16 lift 단계 관찰: transport 층 임시 잠금 (사용자 결정 — "일단 잡고 들기부터").
    # 재활성 세트: cube_transport + transport_success + TerminationsCfg.success 3개를 같이 살릴 것.
    # 관측(cube_to_goal)과 goal 커맨드는 유지 -> obs 64 불변이라 재활성 시 체크포인트 이어쓰기 가능.
    # cube_transport = RewTerm(
    #     func=mdp.ObjectToGoalProgressReward,
    #     weight=500.0,
    #     params={
    #         "command_name": "cube_goal",
    #         "asset_cfg": BOX_CAGE_BODIES,
    #         "object_cfg": SceneEntityCfg("cube"),
    #         "object_half_extent": _HALF_FALLBACK,
    #         "num_points": 3,
    #         "point_fractions": _POINT_FRACTIONS,
    #         "sphere_radius": 0.005,
    #         "depth_max": 0.005,
    #         "distance_max": 0.5,
    #     },
    # )

    # 논문 r_T: goal ±5cm + gate 물림 0.5s 유지 -> 한 방 +500 + 즉시 종료 (lift 단계 잠금)
    # transport_success = RewTerm(
    #     func=mdp.is_terminated_term,
    #     weight=15000.0,
    #     params={"term_keys": "success"},
    # )

    # 낙하 정액 벌금. fresh 단계에서는 0 (탐색 회피 함정 실측, 2026-07-15) —
    # 잡기가 자리 잡은 뒤 resume에서 -3000으로 켜는 2단계 커리큘럼
    drop_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=0.0,
        params={"term_keys": "cube_dropped"},
    )

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

    hand_floor = RewTerm(
        func=mdp.hand_floor_penalty,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link", "finger.*"]),
            "clearance": 0.02,
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
    # 2026-07-16 lift 단계 잠금 (transport 재활성 세트의 일부)
    # success = DoneTerm(
    #     func=mdp.ObjectAtGoalHeld,
    #     params={
    #         "command_name": "cube_goal",
    #         "asset_cfg": BOX_CAGE_BODIES,
    #         "object_cfg": SceneEntityCfg("cube"),
    #         "object_half_extent": _HALF_FALLBACK,
    #         "num_points": 3,
    #         "point_fractions": _POINT_FRACTIONS,
    #         "sphere_radius": 0.005,
    #         "depth_max": 0.005,
    #         "goal_radius": 0.05,
    #         "gate_threshold": 0.3,
    #         "hold_steps": 15,  # 0.5s @ 30Hz
    #     },
    # )
