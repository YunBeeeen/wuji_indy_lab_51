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
BASE_Z = 0.0                      # 바닥 (받침면 없음)
CUBE_HALF = 0.03
CUBE_POS = (0.62, -0.18)          # 2026-07-13_21-57-31 큐브 위치


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

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            # 2026-07-13_21-57-31 run 위치. 바닥에 직접 놓임 (받침면/테이블 없음).
            # 리셋마다 x ±6cm, y ±8cm 랜덤 (events.reset_cube_position).
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
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
        self.episode_length_s = 6.0  # -> 240 step/episode (기존 20)
        # 2026-07-10 run이 2**18에서 약 263k patch로 overflow남. 지금은 finger_cage_hold가 손가락을
        # 오므리게 해서 접촉이 더 늘어남. overflow는 크래시가 아니라 "접촉을 조용히 버림" -> 손이
        # 큐브를 통과하고 cage reward가 안 오름 -> "reward 설계가 잘못됨"과 구별이 불가능해짐.
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
