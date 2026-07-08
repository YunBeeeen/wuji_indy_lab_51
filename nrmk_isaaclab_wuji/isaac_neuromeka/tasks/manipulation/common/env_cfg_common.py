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
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Gnoise(std=0.01), history_length=3)
        joint_vel = ObsTerm(func=mdp.finite_joint_vel, noise=Gnoise(std=0.1), history_length=3)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})

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

    end_effector_orientation_tracking = RewTerm(
        func=mdp.end_effector_orientation_tracking_distance_bounded,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "command_name": "ee_pose",
            "distance_max": 0.25,
        },
    )

    ## regularizers
    end_effector_speed = RewTerm(
        func=mdp.end_effector_speed,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING)},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)

    # action_second_rate = RewTerm(func=mdp.action_second_rate_l2, weight=-0.0001)  # -0.00005

    joint_vel = RewTerm(
        func=mdp.finite_joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
