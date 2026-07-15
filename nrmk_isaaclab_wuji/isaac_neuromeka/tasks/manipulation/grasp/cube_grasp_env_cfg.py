from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import (
    ReachSceneCfg,
)

# ★ 2026-07-13_21-57-31 run 설정으로 복원함 (그날 지표가 제일 좋았던 run, 9608 iter).
#   그 run의 params/env.yaml 에서 그대로 뽑음.
#   opposition -0.976 -> +0.771,  cage_inside 0 -> 0.680,  hold 0 -> 0.401
#   (다만 cube_lift는 여전히 0이었음)
#
# 그 run의 설정:
#   sim.dt 1/60,  decimation 2 (30 Hz),  episode 8초,  clip_actions 1.0
#   팔: joint1=-0.45  joint2=-1.85  joint3=0.0  joint4=1.2  joint5=2.356194490192345
#   약지/새끼 접음 (finger[4-5]_joint{1,3,4} = 1.2)
#   큐브: (0.62, -0.18, 0.03) 바닥. 6cm, 0.30 kg. 받침면/테이블 없음
#   action scale = 1.0 (스칼라)
# 2026-07-15 테이블 도입. 바닥(0.0) 큐브는 어떤 시작 자세든 손이 바닥까지 내려가야 해서
# 팔 전체가 웅크려졌고, 든 뒤에도 팔을 세울 유인이 없어 그 자세에 머묾 (palm-up scoop +
# 팔꿈치 바닥). 큐브를 손 시작 높이(~0.6m) 근처로 올려 하강량을 60cm -> ~17cm로 줄임.
# lift/floor/success/metrics는 전부 테이블 상판(surface_z=BASE_Z) 기준으로 배선됨
# (__post_init__ 참고. metrics는 managers.py:244가 cube_lift.params에서 자동으로 읽음).
BASE_Z = 0.25                     # 테이블 상판 높이
CUBE_HALF = 0.03
CUBE_POS = (0.62, -0.20)          # 2026-07-13_21-57-31 큐브 x/y (유지)


import isaac_neuromeka.mdp as mdp  # noqa: F401
from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg

# Import common environment configuration
from isaac_neuromeka.tasks.manipulation.common.env_cfg_common import (  # noqa: F401
    CubeGraspActionsCfg,
    CubeGraspCommandsCfg,
    CubeGraspEventCfg,
    CubeGraspObservationsCfg,
    CubeGraspRewardsCfg,
    CubeGraspTerminationsCfg,
)
from isaac_neuromeka.utils.etc import EmptyCfg



@configclass
class CubeGraspSceneCfg(ReachSceneCfg):
    """Scene config for cube grasp smoke tests."""

    # 테이블 (kinematic -> 밀리지 않음). 로봇 베이스(원점)와 겹치지 않게 큐브 중심 기준 0.5m 정사각.
    support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(CUBE_POS[0], CUBE_POS[1], BASE_Z / 2),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 1.5, BASE_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.45, 0.4),
                metallic=0.0,
                roughness=0.8,
            ),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            # 2026-07-13_21-57-31 run의 x/y + 테이블 상판 위 (2026-07-15).
            # 리셋마다 x ±6cm, y ±8cm 랜덤 (events.reset_cube_position) -> 0.5m 상판 안에 안전.
            pos=(CUBE_POS[0], CUBE_POS[1], BASE_Z + CUBE_HALF),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.20),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.4, 1.0),
                metallic=0.0,
                roughness=0.5,
            ),
        ),
    )


@configclass
class CubeGraspEnvCfg(NrmkRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: CubeGraspSceneCfg = CubeGraspSceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: CubeGraspObservationsCfg = CubeGraspObservationsCfg()
    actions: CubeGraspActionsCfg = CubeGraspActionsCfg()
    commands: CubeGraspCommandsCfg = CubeGraspCommandsCfg()
    # MDP settings
    rewards: CubeGraspRewardsCfg | EmptyCfg = CubeGraspRewardsCfg()
    terminations: CubeGraspTerminationsCfg = CubeGraspTerminationsCfg()
    events: CubeGraspEventCfg | EmptyCfg = CubeGraspEventCfg()
    curriculum = EmptyCfg()  # Not used for now
    # CMDP settings
    costs = EmptyCfg()  # Not used for now

    #
    actor_obs_list: list = ["policy"]  # ["proprioception", "point_cloud", "privileged"]
    critic_obs_list: list | None = None  # None: same as actor_obs_list
    teacher_obs_list: list | None = None  # None: same as actor_obs_list

    def __post_init__(self):
        """Post initialization."""
        # task settings
        # 24였음 -> 0.4초/step = 2.5 Hz. 정책이 내는 건 "관절 목표 위치"이고 그 목표를 0.4초 내내
        # 고정한 채 PD가 밀어붙임. 즉 바닥에 닿아도 0.4초가 지나야 알아챔 -> 모든 접촉이 슬램이 됨.
        # (실측: 손이 바닥에 처박고 87cm까지 튕겨 오르길 반복, 큐브가 67cm 날아감)
        # reach는 접촉이 없어서 2.5 Hz로 충분했지만 파지는 접촉이 전부임.
        # 30 Hz는 IsaacLab 공식 접촉 task들의 하한임 (Lift-Cube 50, ShadowHand 40~60, Allegro 30).
        # 15 Hz(dec 4)면 판단 사이에 손이 3.3cm(큐브 반 개) 움직여서 아슬아슬함.
        self.sim.dt = 1.0 / 60.0
        self.decimation = 2  # 33ms/step -> 30 Hz
        self.episode_length_s = 8.0  # -> 240 step/episode (기존 20)
        # 2026-07-10 run이 2**18에서 약 263k patch로 overflow남. 지금은 finger_cage_hold가 손가락을
        # 오므리게 해서 접촉이 더 늘어남. overflow는 크래시가 아니라 "접촉을 조용히 버림" -> 손이
        # 큐브를 통과하고 cage reward가 안 오름 -> "reward 설계가 잘못됨"과 구별이 불가능해짐.
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        # ★ 받침면 기준 배선 (지우면 안 됨): BASE_Z > 0이면 "든 높이"와 "바닥 뚫기"의 기준이
        # 전부 테이블 상판이어야 함. 이게 빠지면 상판 위 큐브의 clearance가 스폰부터 +BASE_Z라
        # lift 보상이 "만점에서 시작"하는 대형 버그가 됨 (2026-07-15 실제로 한 번 지워졌었음).
        # metrics(managers.py:244)는 cube_lift 것을 자동으로 읽어감.
        self.rewards.cube_lift.params["surface_z"] = BASE_Z
        self.rewards.hand_floor.params["surface_z"] = BASE_Z
        # 낙하 종료: 큐브 중심이 상판 5cm 아래 = 테이블 밖으로 확실히 떨어진 상태.
        # 정상 파지 중 큐브 중심은 BASE_Z + 0.03이라 절대 안 걸림.
        self.terminations.cube_dropped.params["minimum_height"] = BASE_Z - 0.05
        # 운반 goal (커리큘럼 1단계, 2026-07-15): 스폰 바로 위 +20cm "고정점" —
        # "잡아서 들고 그 높이에서 멈춰 유지"부터. lo=hi라 매 에피소드 같은 goal.
        # (스폰 랜덤 ±6/8cm 대비 goal은 공칭점 위라 약간의 횡이동 포함 — 무시 가능 수준)
        # 1단계가 되면 2단계로 범위를 박스로 확장:
        #   pos_x = (CUBE_POS[0] - 0.12, CUBE_POS[0] + 0.10)
        #   pos_y = (CUBE_POS[1] - 0.25, CUBE_POS[1] + 0.25)
        #   pos_z = (BASE_Z + 0.10, BASE_Z + 0.30)
        self.commands.cube_goal.ranges.pos_x = (CUBE_POS[0], CUBE_POS[0])
        self.commands.cube_goal.ranges.pos_y = (CUBE_POS[1], CUBE_POS[1])
        self.commands.cube_goal.ranges.pos_z = (BASE_Z + 0.20, BASE_Z + 0.20)
        # (참고) 옛 lift 기반 r_T(ObjectLiftedHeld, 주석 상태)를 되살릴 때만 surface_z 오버라이드
        # 필요. 현재 success는 goal 기반(ObjectAtGoalHeld)이라 surface_z 파라미터가 없음.
        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
