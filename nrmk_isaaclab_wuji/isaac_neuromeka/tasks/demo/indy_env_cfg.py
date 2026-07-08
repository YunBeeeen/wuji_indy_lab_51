from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

# observation space
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (  # noqa: F401
    ContactSensor,
    ContactSensorCfg,
    FrameTransformer,
    FrameTransformerCfg,
)

# terrain
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp
from isaac_neuromeka.assets import INDY7_CFG
from isaac_neuromeka.assets.articulation import FiniteArticulationCfg
from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg  # TODO: move one level up

# action space
from isaac_neuromeka.mdp.actions import CustomJointPositionAction
from isaac_neuromeka.utils.etc import EmptyCfg

# command


@configclass
class IndyDemoSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robots
    robot: FiniteArticulationCfg = INDY7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # target object
    obstacle = None

    # contact sensor
    contact_sensors = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link[2-6]",
        update_period=0.0,
        debug_vis=False,
        track_pose=True,
        track_air_time=False,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RobotStatesCfg(ObsGroup):
        q = ObsTerm(func=mdp.joint_pos)
        qdot = ObsTerm(func=mdp.finite_joint_vel)
        p = ObsTerm(func=mdp.body_pose_b, params={"body_name": "tcp"})
        pdot = ObsTerm(func=mdp.body_vel_b, params={"body_name": "tcp"})
        op_state = ObsTerm(func=mdp.op_state)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # @configclass
    # class VisionStatesCfg(RobotStatesCfg):
    #     # ik_solution = ObsTerm(func=mdp.ik_solution, params={"command_name": "ee_pose"}) # TODO: fix
    #     point_cloud = ObsTerm(func=mdp.point_cloud_flat)
    #     camera_position_b = ObsTerm(func=mdp.camera_position_b, params={"source_name": "box_raycast"})
    #     def __post_init__(self):
    #         self.enable_corruption = False
    #         self.concatenate_terms = False

    # observation groups
    policy = RobotStatesCfg()


## THIS IS ONLY USED FOR SANITY CHECKS AND TESTING
@configclass
class ObservationRSLRL(ObservationsCfg):
    @configclass
    class RobotStatesFlatCfg(ObservationsCfg.RobotStatesCfg):
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy = RobotStatesFlatCfg()


# @configclass
# class ReachCommandsCfg:
#     ee_pose = ZeroPoseCommandCfg(
#         asset_name="robot",
#         body_name="tcp",
#         debug_vis=True
#         )

#     contact_mode = mdp.ContactModeCommandCfg(
#         class_type=mdp.ContactModeCommand,
#         resampling_time_range=(5.0, 10.0),
#         contact_mode_prob=0.4,
#         )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        class_type=CustomJointPositionAction,
        asset_name="robot",
        joint_names=["joint[0-5]"],
        scale=1.0,
        use_default_offset=False,
    )
    gripper_action: ActionTerm | None = None


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

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="joint.*"),
            "friction_distribution_params": (0.7, 1.3),
            "armature_distribution_params": (0.75, 1.25),
            "operation": "abs",
            "distribution": "uniform",
        },
    )


@configclass
class IndyDeployEnvCfg(NrmkRLEnvCfg):

    scene = IndyDemoSceneCfg(num_envs=1, env_spacing=3.0)
    observations = ObservationsCfg()
    commands = EmptyCfg()
    actions = ActionsCfg()
    rewards = EmptyCfg()
    events = EventCfg()
    curriculum = EmptyCfg()  # Not used for now
    costs = EmptyCfg()  # Not used for now
    terminations = EmptyCfg()  # Not used for now

    actor_obs_list = ["policy"]

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override
        self.decimation = 5  # 20 Hz
        self.sim.render_interval = 8
        self.sim.dt = 1.0 / 100.0

        self.episode_length_s = 6000.0


## THIS IS ONLY USED FOR SANITY CHECKS AND TESTING
@configclass
class IndyDeployEnvRSLRL(IndyDeployEnvCfg):
    """Configuration for the Indy deployment environment with RSLRL observations."""

    observations = ObservationRSLRL()


# @configclass
# class VisionDemoEnvCfg(IndyVisionCMDPEnvCfg):

#     actor_obs_list: list = ["policy"]
#     critic_obs_list: list | None = None
#     teacher_obs_list: list | None = None

#     observations = ObservationsCfg()
#     commands = ReachCommandsCfg()
#     rewards = EmptyCfg()

#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # override
#         self.episode_length_s = 6000.

#         self.actions.arm_action = mdp.JointPositionActionCfg(
#             class_type=CustomJointPositionAction,
#             asset_name="robot", joint_names=["joint[0-5]"], scale=1.0, use_default_offset=False
#         )

#         terain_cfg = TerrainGeneratorCfg(
#             curriculum=False,
#             size=(2.0, 2.0),
#             border_width=0.0,
#             num_rows=1,num_cols=1, # for 4000 envs
#             difficulty_range=(0.25, 1.0),
#             sub_terrains={
#                 "boxes": MeshBoxTerrainCfg(
#                     high_prob=0.1,
#                     proportion=0.5,
#                     grid_width=0.3,
#                     grid_height_range=(-0.05, 0.3),
#                     low_height_ratio = 0.3,
#                     platform_width=0.5,
#                     robot_range_width = 0.25
#                 )
#             },
#         )

#         self.scene.terrain.terrain_generator = terain_cfg
