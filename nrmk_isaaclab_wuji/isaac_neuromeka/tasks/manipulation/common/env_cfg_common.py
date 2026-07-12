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
    pass


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


@configclass
# class CubeGraspTeacherObsCfg(ObservationsCfg):
#
#    @configclass
#    class Proprio(ObsGroup):
#        """Observations for policy group."""

        # observation terms (order preserved)
#        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Gnoise(std=0.01), history_length=3)
 #       joint_vel = ObsTerm(func=mdp.finite_joint_vel, noise=Gnoise(std=0.1), history_length=3)
  #      pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})

#        action_history = ObsTerm(func=mdp.action_history)

#        def __post_init__(self):
 #           self.enable_corruption = True
  #          self.concatenate_terms = True

#    @configclass
 #   class Privileged(ObsGroup):
  #      """Observations for policy group."""

        # observation terms (order preserved)
#        joint_friction = ObsTerm(func=mdp.joint_friction)
 #       joint_damping = ObsTerm(func=mdp.joint_damping)
  #      action_delay = ObsTerm(func=mdp.action_delay_steps)
        # TODO: action delay

#        def __post_init__(self):
 #           self.enable_corruption = False
  #          self.concatenate_terms = True

#    proprioception = Proprio()
 #   privileged = Privileged()


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
    # [thumb_tip, *opposing]: a line is drawn from the thumb tip to each opposing body, and each
    # finger appears twice — its tip (pinch grasp) and its mid-phalanx (secure grasp). 4 lines x 3
    # points = 12 cage points.
    #
    # The paper uses the thumb-middle pair alone (6 points) because its r_grasp term separately pins
    # the hand rotation and every finger joint to a target grasp. A symmetric cube has no target
    # grasp, so we have no r_grasp — and with only thumb-middle, the index was left entirely
    # unconstrained and the policy found a palm-up, fingers-crossed pose that still scored a perfect
    # cage. Listing the index too is the paper's own suggested extension, and thumb+index+middle is
    # the chopstick grip we are ultimately after.
    #
    # preserve_order: SceneEntityCfg sorts body_ids by default, which would silently move the thumb
    # out of slot 0 and turn some other finger into the anchor.
    body_names=[
        "finger1_tip_link",  # thumb tip: the anchor every line starts from
        "finger2_tip_link",
        "finger2_link3",
        "finger3_tip_link",
        "finger3_link3",
    ],
    preserve_order=True,
)


@configclass
class CubeGraspRewardsCfg:
    """Reward terms for the MDP.

    Both terms act on the *same* virtual points spanning the grasp aperture — the gap between the
    thumb tip and the middle finger — which is what makes them compose. reach (Eq. 14) drags that
    aperture onto the cube; hold (Eq. 15) pays for the points sinking into it, so closing the hand
    is rewarded directly with no contact sensing.

    Every earlier reward drove *fingertips* at the cube's *centre*. That target is 3 cm inside a
    6 cm cube, so it is unreachable, and with the thumb weighted 3:1:1 the cheapest way to satisfy
    it was to bury the thumb and leave the other fingers behind (measured: thumb 0.017 m off the
    surface, index 0.072, middle 0.078). From that pose the thumb->middle segment does not straddle
    the cube, so closing the hand pushed the cage points back *out* — forcing the fingers shut drove
    cage_inside_frac down from 0.47 to 0.40. The policy was right to refuse to close; the reward was
    wrong. Centre-seeking distance terms also punish contact, since touching a free cube shoves it
    away and the distance grows. Do not reintroduce them.
    """

    finger_cage_reach = RewTerm(
        func=mdp.ObjectCageProgressReward,
        # Kept well below finger_cage_hold. The paper scales rewards by their position in the
        # sub-task sequence (r_T >> r_orient >> r_hold >> r_reach) precisely to stop the policy from
        # settling for the easier earlier sub-task, which is exactly what happened when reach
        # outweighed hold.
        weight=0.3,
        params={
            "asset_cfg": CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": (0.03, 0.03, 0.03),
            "num_points": 3,
            # Normalises the per-step improvement. Keep it above the largest realistic per-step gain
            # (~0.15 m during the approach) so the term never saturates: a saturating progress reward
            # pays per *step* that banks an improvement, which rewards dawdling.
            "distance_max": 0.5,
        },
    )

    finger_cage_hold = RewTerm(
        func=mdp.object_in_finger_cage,
        weight=1.0,
        params={
            "asset_cfg": CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": (0.03, 0.03, 0.03),
            "num_points": 3,
            # Tuned by sweeping this term over a flexion sweep of the settled 2026-07-10 policy:
            # a larger sphere_radius pays out just for having the cube between open fingers. At
            # 0.005/0.02 an open hand scores 0.19 and a closed one 0.46 (2.4x); at 0.02/0.03 it was
            # 0.30 -> 0.49 (1.6x), i.e. mostly a constant the critic subtracts out.
            "sphere_radius": 0.005,
            "depth_max": 0.02,
        },
    )

    # The paper's r_lift, and the term that decides which grasps count. The cage geometry alone can
    # be satisfied by a pose that never takes the cube's weight — a 2026-07-11 run reached
    # opposition +0.92 and inside_frac 0.84 while lifting the cube by 2 mm, with the palm turned up
    # and the fingers crossed. Rather than prescribe a hand orientation to rule that out (arbitrary:
    # a symmetric cube has no functional grasp to hit, which is the whole reason the paper's r_hr and
    # r_hj are unavailable to us), just require the hand to carry the thing. Any pose that lifts the
    # cube is a real grasp.
    #
    # Weighted above hold, per the paper's ordering r_T >> r_orient >> r_hold >> r_reach.
    cube_lift = RewTerm(
        func=mdp.object_lift_in_cage,
        weight=3.0,
        params={
            "asset_cfg": CAGE_BODIES,
            "object_cfg": SceneEntityCfg("cube"),
            "object_half_extent": (0.03, 0.03, 0.03),
            "num_points": 3,
            "sphere_radius": 0.005,
            "depth_max": 0.02,
            "initial_height": 0.03,  # the cube's spawn height; reset only randomises x/y
            "lift_height": 0.08,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0003)



@configclass
class CubeGraspTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
