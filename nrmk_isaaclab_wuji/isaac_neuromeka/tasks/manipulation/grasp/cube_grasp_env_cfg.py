from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import (
    ReachSceneCfg,
)


import isaac_neuromeka.mdp as mdp  # noqa: F401
from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg

# Import common environment configuration
from isaac_neuromeka.tasks.manipulation.common.env_cfg_common import (  # noqa: F401
    CubeGraspActionsCfg,
    CubeGraspCommandsCfg,
    CubeGraspEventCfg,
    CubeGraspObservationsCfg,
    CubeGraspRewardsCfg,
    CubeGraspTeacherObsCfg,
    CubeGraspTerminationsCfg,
)
from isaac_neuromeka.utils.etc import EmptyCfg



@configclass
class CubeGraspSceneCfg(ReachSceneCfg):
    """Scene config for cube grasp smoke tests."""

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, -0.18, 0.03),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
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
        self.decimation = 24  # physics dt is 1/60, so this is a 0.4 s step -> 2.5 Hz control
        self.episode_length_s = 8.0
        # A 2026-07-10 run overflowed 2**18 at ~263k patches. finger_cage_hold now drives the fingers
        # to close on the cube, adding finger-cube and finger-finger contacts on top of that. Overflow
        # does not crash — PhysX silently drops contacts, so the hand would pass through the cube and
        # the cage reward would never rise, which is indistinguishable from a badly shaped reward.
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
