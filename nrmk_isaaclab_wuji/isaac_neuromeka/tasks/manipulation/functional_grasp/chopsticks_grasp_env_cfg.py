"""One-stick acquisition and functional pre-grasp environment.

This is the standalone A1 testbed.  It does not share MDP cfg objects with
Cube-Grasp or Box-Transport, so reward/observation experiments cannot leak
between tasks.  The rigid object is still named ``cube`` only to reuse the
well-tested generic box SDF and metric plumbing.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg
from isaac_neuromeka.tasks.manipulation.functional_grasp.chopstick_mdp_cfg import (
    ChopstickAcquireActionsCfg,
    ChopstickAcquireCommandsCfg,
    ChopstickAcquireEventCfg,
    ChopstickAcquireObservationsCfg,
    ChopstickAcquireRewardsCfg,
    ChopstickAcquireTerminationsCfg,
)
from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import ReachSceneCfg
from isaac_neuromeka.utils.etc import EmptyCfg


BASE_Z = 0.25
STICK_POS = (0.62, -0.20)
STICK_SIZE = (0.014, 0.18, 0.014)  # square-section proxy; +y=tip, -y=tail
# 2026-07-30: 1cm 단일 프록시가 인위적으로 어려워 whack-a-mole → 실제 타깃(7mm 젓가락 2개 다발 ≈ 1.4cm)로 전환.
#   1.4cm은 2cm에 가까워 되던 config가 대체로 전이됨 → 2cm 성공값으로 리사이즈: mass 0.02·offset None·
#   depenet 5.0·solver_vel 1. STICK_HALF_EXTENT(chopstick_mdp_cfg.py)도 (0.007,0.09,0.007) 동반.
#   보상은 2cm 성공 구조 복귀(grip 40·sphere 0.005·cage_hold 5·goal_prox 20·lift 0.05).
#   (1cm 복귀: size 0.01·mass 0.01·offset 0.001/0.0·depenet 0.2·solver_vel 4·half_extent 0.005·sphere 0.004·grip 150.)
STICK_MASS = 0.02
STICK_CONTACT_OFFSET = None
STICK_REST_OFFSET = None
STICK_MAX_DEPENETRATION_VELOCITY = 5.0


@configclass
class ChopsticksGraspSceneCfg(ReachSceneCfg):
    """Table and one fixed-size chopstick proxy."""

    support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(STICK_POS[0], STICK_POS[1], BASE_Z / 2)),
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 1.0, BASE_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.42, 0.38), metallic=0.0, roughness=0.8
            ),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(STICK_POS[0], STICK_POS[1], BASE_Z + STICK_SIZE[2] / 2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=STICK_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,  # 1.4cm (2cm값; 1cm 복귀 시 4)
                max_depenetration_velocity=STICK_MAX_DEPENETRATION_VELOCITY,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=STICK_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=STICK_CONTACT_OFFSET,
                rest_offset=STICK_REST_OFFSET,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.55, 0.08), metallic=0.0, roughness=0.55
            ),
        ),
    )


@configclass
class ChopsticksGraspEnvCfg(NrmkRLEnvCfg):
    """A1: fixed one-stick functional-grasp baseline."""

    scene: ChopsticksGraspSceneCfg = ChopsticksGraspSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ChopstickAcquireObservationsCfg = ChopstickAcquireObservationsCfg()
    actions: ChopstickAcquireActionsCfg = ChopstickAcquireActionsCfg()
    commands: ChopstickAcquireCommandsCfg = ChopstickAcquireCommandsCfg()
    rewards: ChopstickAcquireRewardsCfg | EmptyCfg = ChopstickAcquireRewardsCfg()
    terminations: ChopstickAcquireTerminationsCfg = ChopstickAcquireTerminationsCfg()
    events: ChopstickAcquireEventCfg | EmptyCfg = ChopstickAcquireEventCfg()
    curriculum = EmptyCfg()
    costs = EmptyCfg()

    actor_obs_list: list = ["policy"]
    critic_obs_list: list | None = None
    teacher_obs_list: list | None = None

    def __post_init__(self):
        self.sim.dt = 1.0 / 60.0
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        self.rewards.cube_lift.params["surface_z"] = BASE_Z
        self.rewards.hand_floor.params["surface_z"] = BASE_Z
        self.terminations.stick_dropped.params["minimum_height"] = BASE_Z - 0.05

        # 운반 이식(2026-07-22): world goal = 스폰 위 +20cm, 자세 yaw 45° 고정.
        # pos_x/pos_y는 cfg에서 STICK_POS(0.62, -0.20)로 이미 고정, z만 여기서 오버라이드.
        # 이후 ranges를 넓히면 랜덤 목표(위치/자세)가 됨.
        self.commands.cube_goal.ranges.pos_z = (BASE_Z + 0.20, BASE_Z + 0.20)

        self.viewer.eye = (2.5, 2.5, 2.0)
