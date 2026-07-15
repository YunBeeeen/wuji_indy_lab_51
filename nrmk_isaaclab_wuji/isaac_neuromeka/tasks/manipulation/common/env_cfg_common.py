from __future__ import annotations

import math
import pdb  # noqa:F401
from dataclasses import MISSING

from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

# from isaaclab.scene import InteractiveSceneCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: F401

import isaac_neuromeka.mdp as mdp

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    class ConFig:
        default_ee_pose = [0.3563, -0.1829, 0.5132]

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # TODO: multiple body names
        resampling_time_range=(6.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(ConFig.default_ee_pose[0], ConFig.default_ee_pose[0] + 0.3),
            pos_y=(ConFig.default_ee_pose[1] - 0.2, ConFig.default_ee_pose[1] + 0.2),
            pos_z=(ConFig.default_ee_pose[2] - 0.3, ConFig.default_ee_pose[2]),
            roll=(0.0, 0.0),
            pitch=(math.pi, math.pi),  # depends on end-effector axis
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # joint_pos = ObsTerm(func=mdp.joint_pos, noise=Gnoise(std=0.01), history_length=3)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        # joint_vel = ObsTerm(func=mdp.finite_joint_vel, noise=Gnoise(std=0.1), history_length=3)
        pose_command = ObsTerm(func=mdp.generated_position_commands, params={"command_name": "ee_pose"})
        # pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})

        action_history = ObsTerm(func=mdp.action_history)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TeacherObsCfg(ObservationsCfg):

    @configclass
    class Proprio(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(std=0.01), history_length=3)
        joint_vel = ObsTerm(func=mdp.finite_joint_vel, noise=Gnoise(std=0.1), history_length=3)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})

        action_history = ObsTerm(func=mdp.action_history)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class Privileged(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_friction = ObsTerm(func=mdp.joint_friction)
        joint_damping = ObsTerm(func=mdp.joint_damping)
        action_delay = ObsTerm(func=mdp.action_delay_steps)
        # TODO: action delay

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprioception = Proprio()
    privileged = Privileged()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # randomize_joint_friction = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names="joint.*"),
    #         "friction_distribution_params": (0.7, 1.3),
    #         "armature_distribution_params": (0.75, 1.25),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "stiffness_range": (94.0, 106.0),  # (100 - 6, 100 + 6)
    #         "damping_range": (17.0, 23.0),  # (20 - 3, 20 + 3)
    #         "operation": "abs",  # if use "reset" + "add", the sampled values are added to previous iter values.
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_delay = EventTerm(
    #     func=mdp.randomize_delay,
    #     mode="reset",
    #     params={
    #         "delay_step_range": {"low": 20, "high": 24}
    #     }
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.end_effector_position_tracking_bounded,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose",
            "distance_max": 0.5,
        },
    )

   # end_effector_orientation_tracking = RewTerm(
   #     func=mdp.end_effector_orientation_tracking_distance_bounded,
   #     weight=0.1,
   #     params={
   #         "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
   #         "command_name": "ee_pose",
   #         "distance_max": 0.25,
   #     },
   # )

    ## regularizers
   # end_effector_speed = RewTerm(
   #     func=mdp.end_effector_speed,
   #     weight=-0.001,
   #     params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},
   # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)

    # action_second_rate = RewTerm(func=mdp.action_second_rate_l2, weight=-0.0001)  # -0.00005

   # joint_vel = RewTerm(
   #     func=mdp.finite_joint_vel_l2,
   #     weight=-0.001,
   #     params={"asset_cfg": SceneEntityCfg("robot")},
   # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)







# -----------------------------------------------------------------------------
# mdp for grasp

@configclass
class CubeGraspCommandsCfg:
    """Command terms for the MDP."""

    # 2026-07-15 운반 goal: 에피소드마다 테이블 위 공중에서 균일 샘플 (리셋에서만 리샘플).
    # z 범위는 cube_grasp_env_cfg.__post_init__가 BASE_Z 파생으로 오버라이드함 (★ 배선 블록).
    # goal이 공중이라 r_T의 "들었음" 조건이 자동 함의됨.
    cube_goal = mdp.UniformCubeGoalCommandCfg(
        asset_name="cube",
        resampling_time_range=(1.0e9, 1.0e9),  # 에피소드 내 고정 (차분층 telescoping 보호)
        debug_vis=True,
        ranges=mdp.UniformCubeGoalCommandCfg.Ranges(
            pos_x=(0.50, 0.72),
            pos_y=(-0.45, 0.05),
            pos_z=(0.35, 0.55),  # placeholder — __post_init__에서 BASE_Z + (0.10, 0.30)
        ),
    )


@configclass
class CubeGraspActionsCfg:
    """Action specifications for the MDP."""
    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class CubeGraspObservationsCfg:
    """Observation specifications for the MDP."""
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
                    body_names=["finger1_tip_link", "finger2_tip_link", "finger3_tip_link", "finger4_tip_link", "finger5_tip_link"],
                ),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )
        # 2026-07-15 고정점 -> 커맨드 goal. 이전 버전은 target이 월드 고정점이라 다중 env에서
        # env마다 다른 상수가 들어갔음 (env_origins 미보정). dim 3 그대로 -> obs 57 유지.
        cube_to_goal = ObsTerm(
            func=mdp.object_position_error_to_command,
            params={
                "command_name": "cube_goal",
                "object_cfg": SceneEntityCfg("cube"),
            },
        )

        action_history = ObsTerm(func=mdp.action_history)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# CubeGraspTeacherObsCfg (teacher/student proprio + privileged obs groups) lived here. It was never
# referenced — the cube grasp task only uses the `policy` group — so it is gone rather than commented
# out. The reach task's TeacherObsCfg is a separate class and is still in use.


@configclass
class CubeGraspEventCfg:
    """Configuration for events."""

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 약지/새끼 목표 채우기: 리셋은 상태만 되돌리고 목표 버퍼는 0이라서, 액션에 없는
    # finger4-5는 매 에피소드 시작 직후 접힘(1.2)에서 0으로 저절로 펴지며 큐브를 쳐냈음
    # (커플링 액션(MimicJointPositionAction)을 쓰는 env에서는 리셋~첫 액션 공백만 메움)
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

    # randomize_joint_friction = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names="joint.*"),
    #         "friction_distribution_params": (0.7, 1.3),
    #         "armature_distribution_params": (0.75, 1.25),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "stiffness_range": (94.0, 106.0),  # (100 - 6, 100 + 6)
    #         "damping_range": (17.0, 23.0),  # (20 - 3, 20 + 3)
    #         "operation": "abs",  # if use "reset" + "add", the sampled values are added to previous iter values.
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_delay = EventTerm(
    #     func=mdp.randomize_delay,
    #     mode="reset",
    #     params={
    #         "delay_step_range": {"low": 20, "high": 24}
    #     }
    # )


CAGE_BODIES = SceneEntityCfg(
    "robot",
    # [엄지끝, *대향]. 엄지끝에서 각 대향 body로 선분을 긋고 등간격 3점 -> 4선분 x 3점 = 가상점 12개.
    # 각 손가락은 두 번 등장함: 끝(핀치 파지)과 중간마디(파워 파지).
    # 논문은 엄지-중지만 써서 6점이지만, 논문에는 r_grasp가 손 회전/손가락 관절각을 붙잡음.
    # 큐브엔 목표 파지가 없어 r_grasp를 못 쓰므로, 6점만 쓰면 검지가 자유가 되어 "손바닥이 하늘 +
    # 검지·중지 교차" 자세로도 만점이 나옴 (2026-07-11 실측). 엄지+검지+중지는 젓가락 그립과 동일.
    # preserve_order=True 필수: 기본값이면 body_ids가 정렬돼서 엄지가 기준점 자리에서 밀려남.
    body_names=[
        "finger1_tip_link",  # 엄지끝: 모든 선분의 기준점
        "finger2_tip_link",
        "finger2_link3",
        "finger3_tip_link",
        "finger3_link3",
    ],
    preserve_order=True,
)


@configclass
class CubeGraspRewardsCfg:
    """Cube grasp reward terms.

    reach/hold/lift가 "같은" 12개 가상점 위에서 동작함. reach가 파지 간극을 큐브 위로 끌어오고,
    hold가 점들이 큐브 안으로 파고드는 것을 보상함 -> "오므리기"가 직접 보상됨 (접촉센서 불필요).

    [절대 다시 넣지 말 것] "손끝 -> 큐브 중심" 거리 reward.
    큐브 중심은 표면에서 3cm 안쪽이라 손끝이 도달 불가능한 목표이고, 엄지 가중치 3배와 결합하면
    "엄지만 박고 나머지 방치"가 최적해가 됨 (실측: thumb 0.017 / index 0.072 / middle 0.078).
    그 자세에선 오므릴수록 가상점이 큐브 밖으로 나감 (강제 오므림 시 inside_frac 0.47 -> 0.40).
    게다가 거리 reward는 접촉을 처벌함 (만지면 큐브가 밀려나 거리가 늘어남).
    """

    finger_cage_reach = RewTerm(
        func=mdp.ObjectCageProgressReward,
        # 2026-07-13_21-57-31 run 값
        weight=8.0,
        params={
            "asset_cfg": CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": (0.03, 0.03, 0.03),
            "num_points": 3,
            # 2026-07-14: 가상점을 손끝 쪽으로 (기본 [0.25,0.5,0.75] -> [0.1,0.5,0.9]).
            # 내부 등분점은 간격 10cm의 헐렁한 새장에서도 큐브에 박혀 고점이 나와서
            # "손끝을 표면까지 가져가라"는 gradient가 없었음. hold/lift와 반드시 같은 점.
            "point_fractions": (0.1, 0.5, 0.9),
            # step당 개선량의 정규화 상수 (거리 임계값 아님).
            # 실제 step당 최대 개선량(약 0.15m)보다 충분히 커야 함. 포화되면 "천천히 접근하기"를 보상함.
            "distance_max": 0.5,
            # 순서 강제 게이트 (양수에만 적용)
            "palm_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
            "palm_normal_b": (0.19, 0.28, 0.94),
            "gate_floor": 0.0,        },
    )

    finger_cage_hold = RewTerm(
        func=mdp.object_in_finger_cage,
        weight=15.0,
        params={
            "asset_cfg": CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": (0.03, 0.03, 0.03),
            "num_points": 3,
            # 2026-07-14: reach와 같은 점 (0.1 = 엄지끝 근처, 0.9 = 대향 손끝 근처).
            # 끝점이 표면에 닿아야(간격 ~6cm) 고점 -> 진짜 오므리기가 직접 보상됨
            "point_fractions": (0.1, 0.5, 0.9),
            # 손가락 굴곡 sweep으로 실측 튜닝함. sphere_radius가 크면 손가락을 벌린 채 큐브가
            # 사이에 있기만 해도 점수가 나와 대비가 죽음.
            # 0.005/0.02 -> 벌림 0.19 / 오므림 0.46 (2.4배).  0.02/0.03 -> 0.30 / 0.49 (1.6배).
            "sphere_radius": 0.005,
            # 2026-07-14: 0.02 -> 0.005. 끝쪽 가상점과 쓰면 0.02는 접촉 후에도 "더 조여라"가
            # 남는데, 이 손은 40%+ 오므리면 간격 2.8cm까지 줄며 6cm 큐브를 짜냄(수박씨).
            # 0.005면 끝점 기준 간격 ~6.2cm(접촉 직후)에서 포화 -> "닿으면 만족"
            "depth_max": 0.005,
        },
    )

    # 논문 r_lift. "어떤 자세를 진짜 파지로 인정할지" 결정하는 항.
    # cage만으로는 하중을 못 견디는 자세도 만점이 나옴 (2026-07-11 run: opposition +0.92,
    # inside_frac 0.84인데 lift는 2mm. 손바닥은 하늘, 손가락은 교차).
    # 자세를 지정하지 않고 "들 수 있는가"만 물음. 드는 자세면 뭐든 진짜 파지임.
    # hold보다 무겁게 (논문 순서 r_T >> r_hold >> r_reach).
    cube_lift = RewTerm(
        func=mdp.object_lift_in_cage,
        weight=100.0,
        params={
            "asset_cfg": CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": (0.03, 0.03, 0.03),
            "num_points": 3,
            # 2026-07-14: hold와 동일 (gate가 곧 hold). 느슨한 gate + lift 가중치 50이면
            # "쳐서 튕겨 올리기"가 열리므로, gate는 "진짜 잡았을 때"만 열리게 조임
            "point_fractions": (0.1, 0.5, 0.9),
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "lift_height": 0.08,
        },
    )

    # 논문 r_T (5000): 성공 한 방 + 즉시 종료 (CubeGraspTerminationsCfg.success 참고).
    # RewardManager가 weight x dt(1/30)를 곱하므로 로그 스케일 한 방은 15000/30 = +500.
    # hold 캠핑의 에피소드 총액 상한(만점이어도 15 x 240 / 30 = 120)을 압도해야
    # "가만히 물고 있기"의 기대수익을 이김 (2026-07-14 실측: 팔로 눌러 정지 캠핑 수렴).
    # 일회성 + 종료라 아무리 커도 farming 불가 (차분형 telescoping과 같은 안전성).
    # 2026-07-15 A/B: 테이블 효과만 분리하기 위해 이번 런에서 임시 주석처리 (성공 종료와 세트).
    # 재활성 시 CubeGraspTerminationsCfg.success와 cube_grasp_env_cfg.py의 surface_z 오버라이드도
    # 같이 살릴 것.
    # lift_success = RewTerm(
    #     func=mdp.is_terminated_term,
    #     weight=15000.0,
    #     params={"term_keys": "success"},
    # )

    # 2026-07-15 운반 층 — 논문 사다리의 orient(500) 자리 번역: hold(15) < 운반 < r_T.
    # best-so-far 차분 (+ 전용): 잡은 채(gate 곱) goal 거리 신기록을 깬 양만 지불.
    # 후퇴/왕복 0원, 도착 서성임 연금 없음. 낙하 비용은 아래 drop_penalty가 별도 담당.
    cube_transport = RewTerm(
        func=mdp.ObjectToGoalProgressReward,
        weight=500.0,
        params={
            "command_name": "cube_goal",
            "asset_cfg": CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": (0.03, 0.03, 0.03),
            "num_points": 3,
            "point_fractions": (0.1, 0.5, 0.9),
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "distance_max": 0.5,
        },
    )

    # 논문 r_T (운반판): goal 반경 안 + 잡은 채 0.5s 유지 -> 한 방 +500 (15000 x dt) + 즉시 종료.
    # 종료가 hold/lift 연금의 마개 (앉아서 버는 상한 << 성공 한 방).
    transport_success = RewTerm(
        func=mdp.is_terminated_term,
        weight=15000.0,
        params={"term_keys": "success"},
    )

    # 2026-07-15 낙하 정액 벌금 (− 전용): cube_dropped 종료 시 한 방 −100 (−3000 x dt).
    # transport를 + 전용(best-so-far)으로 바꾸면서 "놓치면 손해" 압력이 여기로 이사함
    # (파인튜닝 초기: 회당 ~−20 청구서로 낙하율 44%→33% 회복 확인 → 그 5배, r_T의 1/5).
    # 굴러간 거리 비례가 아닌 정액이라 물리 우연에 과세하지 않음.
    drop_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-3000.0,
        params={"term_keys": "cube_dropped"},
    )

    # 자의적 제약이 아니라 물리적 필요조건: 손가락은 손바닥 쪽으로 굽으므로 손바닥 뒤의 물체는 못 감쌈.
    # cage 항은 이걸 못 봄 (선분이 손 방향과 무관하게 큐브를 관통) -> 손바닥이 하늘인데도 cage 만점이 나옴.
    # 법선 축만 제약하고 roll은 자유 -> 대칭 물체의 파지 방식을 고르지 않음.
    # 넣기 전에 도달성 먼저 검증함: 팔 관절 40만개 샘플링 결과 "큐브에 닿으면서 정면(+1.000)"인 자세가 존재함.
    #
    # 논문 r_hr처럼 차분형. 손바닥을 "돌리는 것"에만 지급하고, 겨눈 채 유지하는 것엔 지급 안 함.
    # 절대형(weight 0.5)으로 넣었다가 전체 보상의 98%를 먹음: 겨누기는 접근보다 훨씬 싸서
    # 정책이 팔을 접어 31cm 밖에서 겨누기만 하고 접근을 안 함 (manip이 최적의 13%까지 추락).
    # 차분형은 총액이 (final - reset)으로 고정이라 아무리 weight를 키워도 farming이 불가능함.
    palm_facing = RewTerm(
        func=mdp.PalmFacingProgressReward,
        # 2026-07-13_21-57-31 run 값
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
            "object_cfg": SceneEntityCfg("cube"),
            # 손가락을 오므릴 때 손끝이 이동하는 방향으로 실측한 palm_link의 안쪽 법선
            "palm_normal_b": (0.19, 0.28, 0.94),
        },
    )

    # 논문 Eq.17 (r_MP). 이게 없으면 "손바닥을 큐브 쪽으로"를 만족시키는 가장 싼 방법이
    # "팔을 접어 손목만 돌리기"가 됨. 접힌 팔은 손을 못 움직여서 큐브에 영영 못 감.
    # (실측: manip이 초기 57% -> 13%로 추락, 큐브 31cm 앞에서 정지)
    # 논문대로 스케일 안 함(1.0). 범위 [-1, 0]: 특이점에서 멀면 0.
    arm_manipulability = RewTerm(
        func=mdp.arm_manipulability_penalty,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link"], joint_names=["joint[0-5]"]),
            # 논문은 "관측 최대 |J|의 15%". 실측 최대가 약 0.113이므로 0.017 -> 0.02로 잡음.
            "j_max": 0.02,
        },
    )

    # 큐브가 지면(z=0.03)에 있어서 cage를 만들려면 손끝이 바닥 근처까지 내려와야 함. 조금만 지나치면
    # 팔이 바닥을 뚫을 기세로 밀고 반작용으로 손이 87cm까지 튕겨 오르며 큐브를 67cm 날림 (model_350 실측).
    # 종료 조건이 time_out뿐이라 지금까지 바닥을 쳐도 아무 벌이 없었음.
    # 범위 [-1, 0]이라 최대가 0 -> "안전하게 높이 떠 있기"를 유도하지는 않음.
    hand_floor = RewTerm(
        func=mdp.hand_floor_penalty,
        # 절대형이라 dt 보정 불필요. 최악 -2.0 (reach 1.0의 2배). 더 키우면 정책이 바닥을 피하려고
        # 아예 안 내려와서 지면의 큐브에 영영 못 감 (호버링 실패 모드).
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link", "finger.*"]),
            # 큐브 중심이 z=0.03이므로 2cm까지는 자유롭게 내려갈 수 있어야 감쌀 수 있음
            "clearance": 0.02,
        },
    )

    # 2026-07-15 팔 링크가 바닥에 눕는 것(scoop 웅크림)을 직접 감점. 자세는 브랜치/manip 문제가
    # 아니라 "팔꿈치 링크의 월드 높이" 문제로 판정됨 (manip 0.35~0.5로 r_MP는 데드존).
    # 기준면은 테이블이 아니라 바닥(surface_z 기본 0.0) — 팔꿈치는 테이블 옆 공간에 정상적으로
    # 있을 수 있음 (hand_floor의 BASE_Z 오버라이드와 다른 이유).
    # 링크 선정 실측(시작 자세 z): link1=0.08(항상 낮음, 제외) link2=0.30 link3=0.71
    # link4=0.53 link5=0.56 link6=0.62 -> link[2-6]. 절대 페널티(<=0)라 자세가 좋으면 0.
    #arm_floor = RewTerm(
    #    func=mdp.hand_floor_penalty,
    #    weight=2.0,
    #    params={
    #        "asset_cfg": SceneEntityCfg("robot", body_names=["link[2-6]"]),
    #        "clearance": 0.12,
    #    },
    #)
    
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)



@configclass
class CubeGraspTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 2026-07-15 낙하 실패 종료: 큐브가 상판 아래로 확실히 떨어지면 회수 불가 -> 즉시 리셋.
    # 남은 에피소드 낭비를 끊어 처리량을 올리고, 떨어진 큐브를 쫓아 테이블 아래로 웅크리는
    # 행동(reach 차분이 접근에 지불함)을 차단함. 음수 배경 보상이 없으므로 "일부러 떨구고
    # 리셋" 유인은 없음 (떨구면 hold/lift 연금을 잃는 것 자체가 손해).
    # minimum_height는 cube_grasp_env_cfg.__post_init__가 BASE_Z - 0.05로 오버라이드함.
    cube_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cube")},
    )

    # 2026-07-15 논문 r_T의 성공 종료 (운반판): 큐브가 goal 반경 안 + gate 물림을 0.5s 유지.
    # "들었음"은 goal이 공중이라 자동 함의. 즉시 종료가 hold/lift 연금과 도착 서성임의 마개.
    # gate/가상점 파라미터는 CubeGraspRewardsCfg의 cage 항들과 반드시 동일하게 유지할 것.
    success = DoneTerm(
        func=mdp.ObjectAtGoalHeld,
        params={
            "command_name": "cube_goal",
            "asset_cfg": CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": (0.03, 0.03, 0.03),
            "num_points": 3,
            "point_fractions": (0.1, 0.5, 0.9),
            "sphere_radius": 0.005,
            "depth_max": 0.005,
            "goal_radius": 0.05,
            "gate_threshold": 0.3,
            "hold_steps": 15,  # 0.5s @ 30Hz — 던져 넣기는 유지가 안 됨
        },
    )

    # 논문 r_T의 성공 종료 (2026-07-15). "들어서 유지"가 성공의 정의 — 자세는 지정 안 함.
    # 즉시 종료가 핵심: 성공 후에도/대신에도 hold를 계속 수확하는 경로를 끊음.
    # gate/가상점 파라미터는 CubeGraspRewardsCfg.cube_lift와 반드시 동일하게 유지할 것.
    # 2026-07-15 A/B: 테이블 효과 분리를 위해 임시 주석처리 (lift_success와 세트)
    # success = DoneTerm(
    #     func=mdp.ObjectLiftedHeld,
    #     params={
    #         "asset_cfg": CAGE_BODIES,
    #         "object_cfg": SceneEntityCfg("cube"),
    #         "object_half_extent": (0.03, 0.03, 0.03),
    #         "num_points": 3,
    #         "point_fractions": (0.1, 0.5, 0.9),
    #         "sphere_radius": 0.005,
    #         "depth_max": 0.005,
    #         # lift_height(0.08)와 동일 높이. 캠핑(clearance 0)과 들썩(순간)은 여기서 걸러짐
    #         "min_height": 0.08,
    #         # 2026-07-14 실측: 바닥 캠핑도 gate 0.58까지 나옴 -> gate만으로는 구분 불가,
    #         # 높이+유지와 결합해야 함. 0.3은 "손가락이 표면에 닿아 있음" 수준
    #         "gate_threshold": 0.3,
    #         "hold_steps": 15,  # 0.5s @ 30Hz — fling은 유지가 안 됨
    #     },
    # )






# -----------------------------------------------------------------------------
# mdp for functional_grasp

@configclass
class ChopsticksGraspCommandsCfg:
    """Command terms for the MDP."""
    pass


@configclass
class ChopsticksGraspActionsCfg:
    """Action specifications for the MDP."""
    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ChopsticksGraspObservationsCfg:
    """Observation specifications for the MDP."""
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
                    body_names=["finger1_tip_link", "finger2_tip_link", "finger3_tip_link", "finger4_tip_link", "finger5_tip_link"],
                ),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )
        cube_to_goal = ObsTerm(
            func=mdp.object_position_error_to_target,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "target_pos": (0.55, -0.05, 0.12),
            },
        )

        action_history = ObsTerm(func=mdp.action_history)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# CubeGraspTeacherObsCfg (teacher/student proprio + privileged obs groups) lived here. It was never
# referenced — the cube grasp task only uses the `policy` group — so it is gone rather than commented
# out. The reach task's TeacherObsCfg is a separate class and is still in use.


@configclass
class ChopsticksGraspEventCfg:
    """Configuration for events."""

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (0.5, 1.5),
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 약지/새끼 목표 채우기: 리셋은 상태만 되돌리고 목표 버퍼는 0이라서, 액션에 없는
    # finger4-5는 매 에피소드 시작 직후 접힘(1.2)에서 0으로 저절로 펴지며 큐브를 쳐냈음
    # (커플링 액션(MimicJointPositionAction)을 쓰는 env에서는 리셋~첫 액션 공백만 메움)
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

    # randomize_joint_friction = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names="joint.*"),
    #         "friction_distribution_params": (0.7, 1.3),
    #         "armature_distribution_params": (0.75, 1.25),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "stiffness_range": (94.0, 106.0),  # (100 - 6, 100 + 6)
    #         "damping_range": (17.0, 23.0),  # (20 - 3, 20 + 3)
    #         "operation": "abs",  # if use "reset" + "add", the sampled values are added to previous iter values.
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_delay = EventTerm(
    #     func=mdp.randomize_delay,
    #     mode="reset",
    #     params={
    #         "delay_step_range": {"low": 20, "high": 24}
    #     }
    # )


CAGE_BODIES = SceneEntityCfg(
    "robot",
    # [엄지끝, *대향]. 엄지끝에서 각 대향 body로 선분을 긋고 등간격 3점 -> 4선분 x 3점 = 가상점 12개.
    # 각 손가락은 두 번 등장함: 끝(핀치 파지)과 중간마디(파워 파지).
    # 논문은 엄지-중지만 써서 6점이지만, 논문에는 r_grasp가 손 회전/손가락 관절각을 붙잡음.
    # 큐브엔 목표 파지가 없어 r_grasp를 못 쓰므로, 6점만 쓰면 검지가 자유가 되어 "손바닥이 하늘 +
    # 검지·중지 교차" 자세로도 만점이 나옴 (2026-07-11 실측). 엄지+검지+중지는 젓가락 그립과 동일.
    # preserve_order=True 필수: 기본값이면 body_ids가 정렬돼서 엄지가 기준점 자리에서 밀려남.
    body_names=[
        "finger1_tip_link",  # 엄지끝: 모든 선분의 기준점
        "finger2_tip_link",
        "finger2_link3",
        "finger3_tip_link",
        "finger3_link3",
    ],
    preserve_order=True,
)


@configclass
class ChopsticksGraspRewardsCfg:
    """Cube grasp reward terms.

    reach/hold/lift가 "같은" 12개 가상점 위에서 동작함. reach가 파지 간극을 큐브 위로 끌어오고,
    hold가 점들이 큐브 안으로 파고드는 것을 보상함 -> "오므리기"가 직접 보상됨 (접촉센서 불필요).

    [절대 다시 넣지 말 것] "손끝 -> 큐브 중심" 거리 reward.
    큐브 중심은 표면에서 3cm 안쪽이라 손끝이 도달 불가능한 목표이고, 엄지 가중치 3배와 결합하면
    "엄지만 박고 나머지 방치"가 최적해가 됨 (실측: thumb 0.017 / index 0.072 / middle 0.078).
    그 자세에선 오므릴수록 가상점이 큐브 밖으로 나감 (강제 오므림 시 inside_frac 0.47 -> 0.40).
    게다가 거리 reward는 접촉을 처벌함 (만지면 큐브가 밀려나 거리가 늘어남).
    """

    finger_cage_reach = RewTerm(
        func=mdp.ObjectCageProgressReward,
        # 2026-07-13_21-57-31 run 값
        weight=8.0,
        params={
            "asset_cfg": CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": (0.03, 0.03, 0.03),
            "num_points": 3,
            # step당 개선량의 정규화 상수 (거리 임계값 아님).
            # 실제 step당 최대 개선량(약 0.15m)보다 충분히 커야 함. 포화되면 "천천히 접근하기"를 보상함.
            "distance_max": 0.5,
            # 순서 강제 게이트 (양수에만 적용)
            "palm_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
            "palm_normal_b": (0.19, 0.28, 0.94),
            "gate_floor": 0.0,        },
    )

    finger_cage_hold = RewTerm(
        func=mdp.object_in_finger_cage,
        weight=12.0,
        params={
            "asset_cfg": CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": (0.03, 0.03, 0.03),
            "num_points": 3,
            # 손가락 굴곡 sweep으로 실측 튜닝함. sphere_radius가 크면 손가락을 벌린 채 큐브가
            # 사이에 있기만 해도 점수가 나와 대비가 죽음.
            # 0.005/0.02 -> 벌림 0.19 / 오므림 0.46 (2.4배).  0.02/0.03 -> 0.30 / 0.49 (1.6배).
            "sphere_radius": 0.005,
            "depth_max": 0.02,
        },
    )

    # 논문 r_lift. "어떤 자세를 진짜 파지로 인정할지" 결정하는 항.
    # cage만으로는 하중을 못 견디는 자세도 만점이 나옴 (2026-07-11 run: opposition +0.92,
    # inside_frac 0.84인데 lift는 2mm. 손바닥은 하늘, 손가락은 교차).
    # 자세를 지정하지 않고 "들 수 있는가"만 물음. 드는 자세면 뭐든 진짜 파지임.
    # hold보다 무겁게 (논문 순서 r_T >> r_hold >> r_reach).
    cube_lift = RewTerm(
        func=mdp.object_lift_in_cage,
        weight=100.0,
        params={
            "asset_cfg": CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": (0.03, 0.03, 0.03),
            "num_points": 3,
            "sphere_radius": 0.005,
            "depth_max": 0.02,
            "lift_height": 0.08,
        },
    )

    # 자의적 제약이 아니라 물리적 필요조건: 손가락은 손바닥 쪽으로 굽으므로 손바닥 뒤의 물체는 못 감쌈.
    # cage 항은 이걸 못 봄 (선분이 손 방향과 무관하게 큐브를 관통) -> 손바닥이 하늘인데도 cage 만점이 나옴.
    # 법선 축만 제약하고 roll은 자유 -> 대칭 물체의 파지 방식을 고르지 않음.
    # 넣기 전에 도달성 먼저 검증함: 팔 관절 40만개 샘플링 결과 "큐브에 닿으면서 정면(+1.000)"인 자세가 존재함.
    #
    # 논문 r_hr처럼 차분형. 손바닥을 "돌리는 것"에만 지급하고, 겨눈 채 유지하는 것엔 지급 안 함.
    # 절대형(weight 0.5)으로 넣었다가 전체 보상의 98%를 먹음: 겨누기는 접근보다 훨씬 싸서
    # 정책이 팔을 접어 31cm 밖에서 겨누기만 하고 접근을 안 함 (manip이 최적의 13%까지 추락).
    # 차분형은 총액이 (final - reset)으로 고정이라 아무리 weight를 키워도 farming이 불가능함.
    palm_facing = RewTerm(
        func=mdp.PalmFacingProgressReward,
        # 2026-07-13_21-57-31 run 값
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
            "object_cfg": SceneEntityCfg("cube"),
            # 손가락을 오므릴 때 손끝이 이동하는 방향으로 실측한 palm_link의 안쪽 법선
            "palm_normal_b": (0.19, 0.28, 0.94),
        },
    )

    # 논문 Eq.17 (r_MP). 이게 없으면 "손바닥을 큐브 쪽으로"를 만족시키는 가장 싼 방법이
    # "팔을 접어 손목만 돌리기"가 됨. 접힌 팔은 손을 못 움직여서 큐브에 영영 못 감.
    # (실측: manip이 초기 57% -> 13%로 추락, 큐브 31cm 앞에서 정지)
    # 논문대로 스케일 안 함(1.0). 범위 [-1, 0]: 특이점에서 멀면 0.
    arm_manipulability = RewTerm(
        func=mdp.arm_manipulability_penalty,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link"], joint_names=["joint[0-5]"]),
            # 논문은 "관측 최대 |J|의 15%". 실측 최대가 약 0.113이므로 0.017 -> 0.02로 잡음.
            "j_max": 0.02,
        },
    )

    # 큐브가 지면(z=0.03)에 있어서 cage를 만들려면 손끝이 바닥 근처까지 내려와야 함. 조금만 지나치면
    # 팔이 바닥을 뚫을 기세로 밀고 반작용으로 손이 87cm까지 튕겨 오르며 큐브를 67cm 날림 (model_350 실측).
    # 종료 조건이 time_out뿐이라 지금까지 바닥을 쳐도 아무 벌이 없었음.
    # 범위 [-1, 0]이라 최대가 0 -> "안전하게 높이 떠 있기"를 유도하지는 않음.
    hand_floor = RewTerm(
        func=mdp.hand_floor_penalty,
        # 절대형이라 dt 보정 불필요. 최악 -2.0 (reach 1.0의 2배). 더 키우면 정책이 바닥을 피하려고
        # 아예 안 내려와서 지면의 큐브에 영영 못 감 (호버링 실패 모드).
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["palm_link", "finger.*"]),
            # 큐브 중심이 z=0.03이므로 2cm까지는 자유롭게 내려갈 수 있어야 감쌀 수 있음
            "clearance": 0.02,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)



@configclass
class ChopsticksGraspTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
