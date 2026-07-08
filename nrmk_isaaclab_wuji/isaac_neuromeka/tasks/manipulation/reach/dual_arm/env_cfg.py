from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

# from isaaclab.assets import RigidObjectCfg
# import isaaclab.sim as sim_utils
# from isaaclab.sensors import CameraCfg
import isaac_neuromeka.mdp as mdp

##
# Pre-defined configs
##
from isaac_neuromeka.assets import DUAL_ARM_CFG
from isaac_neuromeka.assets.articulation import FiniteArticulationCfg
from isaac_neuromeka.mdp.actions import CustomJointPositionAction
from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # robots
    robot: FiniteArticulationCfg = None

    # target object
    obstacle = None

    # contact sensor
    contact_sensors_left_arm = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_arm_link[2-6]",
        update_period=0.0,
        debug_vis=False,
        track_pose=True,
        track_air_time=False,
    )

    contact_sensors_right_arm = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_arm_link[2-6]",
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


##
# Environment configuration
##
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Gnoise(std=0.05), history_length=3)
        joint_vel = ObsTerm(func=mdp.finite_joint_vel, noise=Gnoise(std=0.5), history_length=3)
        pose_command_left = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_pose"})
        pose_command_right = ObsTerm(func=mdp.generated_commands, params={"command_name": "right_ee_pose"})

        action_history = ObsTerm(func=mdp.action_history)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class DualArmRewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    left_ee_pose_Tracking = RewTerm(
        func=mdp.end_effector_position_tracking_bounded,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="left_arm_link6"), "command_name": "left_ee_pose"},
    )

    left_ee_orientation_tracking = RewTerm(
        func=mdp.end_effector_orientation_tracking_distance_bounded,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="left_arm_link6"), "command_name": "left_ee_pose"},
    )

    right_ee_pose_Tracking = RewTerm(
        func=mdp.end_effector_position_tracking_bounded,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_arm_link6"), "command_name": "right_ee_pose"},
    )

    right_ee_orientation_tracking = RewTerm(
        func=mdp.end_effector_orientation_tracking_distance_bounded,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_arm_link6"), "command_name": "right_ee_pose"},
    )

    # ## regularizers
    left_ee_speed = RewTerm(
        func=mdp.end_effector_speed,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="left_arm_link6")},
    )

    right_ee_speed = RewTerm(
        func=mdp.end_effector_speed,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="right_arm_link6")},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    # action_second_rate = RewTerm(func=mdp.action_second_rate_l2, weight=-0.0001)  # -0.00005


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        class_type=CustomJointPositionAction,
        asset_name="robot",
        joint_names=[".*_arm_.*"],
        scale=0.5,
        use_default_offset=True,
    )
    base_action: ActionTerm = mdp.JointPositionActionCfg(
        class_type=CustomJointPositionAction,
        asset_name="robot",
        joint_names=["base_joint"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class TwoArmCommandsCfg:

    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # TODO: multiple body names
        resampling_time_range=(6.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5),
            pos_y=(0.2, 0.5),
            pos_z=(0.25, 0.75),
            roll=(math.pi * 0.5 - 0.3, math.pi * 0.5 + 0.3),
            pitch=(math.pi - 0.1, math.pi + 0.1),
            yaw=(-0.3, 0.3),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # TODO: multiple body names
        resampling_time_range=(6.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5),
            pos_y=(-0.5, -0.2),
            pos_z=(0.25, 0.75),
            roll=(-math.pi * 0.5 - 0.3, -math.pi * 0.5 + 0.3),
            pitch=(math.pi - 0.1, math.pi + 0.1),
            yaw=(-0.3, 0.3),
        ),
    )


@configclass
class DualArmReachEnvCfg(ReachEnvCfg):

    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=3.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = TwoArmCommandsCfg()
    rewards = DualArmRewardsCfg()

    def __post_init__(self):
        """Post initialization."""
        # task settings
        self.decimation = 24  # 5hz # 30 Hz (4)
        self.episode_length_s = 8.0
        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)

        # switch robot to indy
        self.scene.robot = DUAL_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override command generator body
        # end-effector is along z-direction
        # self.commands.left_ee_pose.body_name = "left_arm_tcp"
        self.commands.left_ee_pose.body_name = "left_arm_link6"
        # self.commands.right_ee_pose.body_name = "right_arm_tcp"
        self.commands.right_ee_pose.body_name = "right_arm_link6"
