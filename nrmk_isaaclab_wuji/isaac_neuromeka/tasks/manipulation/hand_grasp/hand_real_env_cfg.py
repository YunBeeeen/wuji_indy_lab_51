"""Sim-to-real observation variant of the successful ``hand_move`` task.

The policy observation changes and the hand actuator gains are overridden with
the real-hand tuning table.  Reset, 20D residual action, OPEN/CLOSE command,
disturbances and floating-root control are inherited from
:class:`HandMoveEnvCfg`.  Reward weights stay inherited, while explicit Stick2
reference-orientation checks still compare the directed shaft axis rather than
forcing one shaft-roll angle.  Contact loss does not terminate training so the
policy can observe and learn recovery; physical stick drops and time-out remain active.

No simulator-only joint or rigid-body velocity is exposed.  Consecutive
position/pose samples are supplied instead so the policy may infer motion from
the same information that can be retained by a real controller and vision
pipeline.
"""

from __future__ import annotations

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
)
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp

from . import hand_real_mdp
from . import hand_move_mdp
from . import mdp as hand_grasp_mdp
from .hand_grasp_env_cfg import (
    FINGERTIPS,
    HAND_JOINT_NAMES,
    HAND_JOINTS,
    PALM,
    PREGRASP_STICK1_QUATERNION_P,
    PREGRASP_STICK2_QUATERNION_P,
    STICK_1,
    STICK_2,
)
from .hand_move_env_cfg import (
    HandMoveCommandsCfg,
    HandMoveEnvCfg,
    HandMoveTerminationsCfg,
)


HAND_REAL_POLICY_OBS_DIM = 105
HAND_REAL_ORIENTATION_ERROR_MODE = "directed_axis"

# Task-local current-joint residual scale.  Joint1/2 retain the successful
# 0.1-rad contract, Joint3 uses 0.2 rad, and Joint4 uses the 0.15-rad step used
# to tune its real-hand Kp/Kd.  The actuator effort_limit_sim remains the final
# torque cap.
HAND_REAL_ACTION_SCALE: dict[str, float] = {
    joint_name: (
        0.2
        if joint_name.endswith("joint3")
        else 0.15
        if joint_name.endswith("joint4")
        else 0.1
    )
    for joint_name in HAND_JOINT_NAMES
}

# Task-local settings for the real-hand lineage.  Keep these separate from
# WUJI_RIGHT_CFG so hand_grasp/hand_move/hand_setting retain their existing
# actuator physics.  Kp/Kd come from the standalone tuner; the effort table
# below restores the official per-joint Wuji URDF limits.
HAND_REAL_STIFFNESS: dict[str, float] = {
    "finger1_joint1": 1.7,
    "finger1_joint2": 2.7,
    "finger1_joint3": 0.75,
    "finger1_joint4": 1.0,
    "finger2_joint1": 2.4,
    "finger2_joint2": 0.7,
    "finger2_joint3": 0.75,
    "finger2_joint4": 1.3,
    "finger3_joint1": 2.4,
    "finger3_joint2": 0.7,
    "finger3_joint3": 0.75,
    "finger3_joint4": 1.3,
    "finger4_joint1": 2.4,
    "finger4_joint2": 0.7,
    "finger4_joint3": 0.75,
    "finger4_joint4": 1.3,
    "finger5_joint1": 2.4,
    "finger5_joint2": 0.7,
    "finger5_joint3": 0.75,
    "finger5_joint4": 1.15,
}

HAND_REAL_DAMPING: dict[str, float] = {
    "finger1_joint1": 0.04,
    "finger1_joint2": 0.05,
    "finger1_joint3": 0.0015,
    "finger1_joint4": 0.0015,
    "finger2_joint1": 0.055,
    "finger2_joint2": 0.02,
    "finger2_joint3": 0.0015,
    "finger2_joint4": 0.0005,
    "finger3_joint1": 0.055,
    "finger3_joint2": 0.02,
    "finger3_joint3": 0.0015,
    "finger3_joint4": 0.0001,
    "finger4_joint1": 0.055,
    "finger4_joint2": 0.02,
    "finger4_joint3": 0.0015,
    "finger4_joint4": 0.0001,
    "finger5_joint1": 0.055,
    "finger5_joint2": 0.02,
    "finger5_joint3": 0.0015,
    "finger5_joint4": 0.0001,
}

# Official per-joint limits from the Wuji right-hand URDF.  Keep this table
# task-local: hand_real/hand_final target the real hand, while the other hand
# tasks retain the shared actuator experiment settings.
HAND_REAL_EFFORT_LIMITS: dict[str, float] = {
    "finger1_joint1": 0.4452,
    "finger1_joint2": 0.4259,
    "finger1_joint3": 0.1888,
    "finger1_joint4": 0.1468,
    "finger2_joint1": 0.6188,
    "finger2_joint2": 0.1822,
    "finger2_joint3": 0.2251,
    "finger2_joint4": 0.2170,
    "finger3_joint1": 0.6494,
    "finger3_joint2": 0.1827,
    "finger3_joint3": 0.2078,
    "finger3_joint4": 0.2018,
    "finger4_joint1": 0.6389,
    "finger4_joint2": 0.1832,
    "finger4_joint3": 0.2249,
    "finger4_joint4": 0.2044,
    "finger5_joint1": 0.6441,
    "finger5_joint2": 0.1798,
    "finger5_joint3": 0.2384,
    "finger5_joint4": 0.1866,
}

if (
    set(HAND_REAL_STIFFNESS) != set(HAND_JOINT_NAMES)
    or set(HAND_REAL_DAMPING) != set(HAND_JOINT_NAMES)
    or set(HAND_REAL_EFFORT_LIMITS) != set(HAND_JOINT_NAMES)
    or set(HAND_REAL_ACTION_SCALE) != set(HAND_JOINT_NAMES)
):
    raise ValueError(
        "hand_real action/Kp/Kd/effort maps must contain exactly the 20 hand joints"
    )


@configclass
class HandRealObservationsCfg:
    """105D observation using joint/quaternion-pose history instead of velocity.

    Isaac Lab's history buffer returns samples oldest-to-newest.  Each
    ``history_length=2`` term is therefore laid out as ``[previous, current]``.
    On the first observation after reset, both slots contain the reset value;
    this represents zero inferred motion without injecting a fake zero pose.

    Layout::

        joint_pos_history   40 = q_(t-1), q_t (20 each, limit-normalized)
        fingertip_pos       15 = current palm-frame xyz for five fingertips
        stick1_pose_history 14 = state_(t-1), state_t (palm-frame xyz + wxyz)
        stick2_pose_history 14 = state_(t-1), state_t (palm-frame xyz + wxyz)
        last_action         20 = action that produced the current state
        open_close_mode      2 = OPEN/CLOSE one-hot command
        total              105

    The history is advanced only by ObservationManager calls with
    ``update_history=True`` (reset and one call after each policy step).
    Read-only recorder/debug observation calls therefore cannot consume an
    extra history slot.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_history = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": HAND_JOINTS},
            history_length=2,
            flatten_history_dim=True,
        )
        fingertip_pos = ObsTerm(
            func=hand_grasp_mdp.fingertip_positions_in_palm,
            params={"palm_cfg": PALM, "fingertip_cfg": FINGERTIPS},
        )
        stick1_pose_history = ObsTerm(
            func=hand_real_mdp.canonical_object_pose_in_palm,
            params={
                "palm_cfg": PALM,
                "object_cfg": STICK_1,
                "reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
            },
            history_length=2,
            flatten_history_dim=True,
        )
        stick2_pose_history = ObsTerm(
            func=hand_real_mdp.canonical_object_pose_in_palm,
            params={
                "palm_cfg": PALM,
                "object_cfg": STICK_2,
                "reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
            },
            history_length=2,
            flatten_history_dim=True,
        )
        last_action = ObsTerm(func=hand_real_mdp.last_applied_action)
        open_close_mode = ObsTerm(
            func=hand_grasp_mdp.open_close_mode,
            params={"command_name": "open_close"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HandRealCommandsCfg(HandMoveCommandsCfg):
    """Use an explicit neutral mode during the five-second pregrasp stage."""

    open_close = hand_move_mdp.HandMoveOpenCloseCommandCfg(
        neutral_before_open_close=True,
    )


@configclass
class HandRealTerminationsCfg(HandMoveTerminationsCfg):
    """Match the successful hand_move rollout contract without contact-loss reset."""

    # The successful 2026-08-08 hand_move lineage learned through transient
    # contact loss.  Keep the rollout alive so hand_real can learn the same
    # recovery behavior; actual Stick1/Stick2 drops and time-out stay inherited.
    functional_contact_lost = None


def apply_hand_real_contract(env_cfg) -> None:
    """Apply deploy observation/action semantics and real-hand actuator settings."""
    env_cfg.rewards.stick2_reference_pose.params["orientation_error_mode"] = (
        HAND_REAL_ORIENTATION_ERROR_MODE
    )
    env_cfg.rewards.success.params["orientation_error_mode"] = (
        HAND_REAL_ORIENTATION_ERROR_MODE
    )
    env_cfg.terminations.success.params["orientation_error_mode"] = (
        HAND_REAL_ORIENTATION_ERROR_MODE
    )

    # JointAction parses dictionary scales by resolved physical joint name.
    # Assign every joint explicitly because unspecified dictionary entries
    # default to 1.0 in Isaac Lab's JointAction implementation.
    hand_action = env_cfg.actions.hand_action.replace(
        scale=dict(HAND_REAL_ACTION_SCALE)
    )
    env_cfg.actions = env_cfg.actions.replace(hand_action=hand_action)

    finger_actuator = env_cfg.scene.robot.actuators["fingers"]
    env_cfg.scene.robot = env_cfg.scene.robot.replace(
        actuators={
            **env_cfg.scene.robot.actuators,
            "fingers": finger_actuator.replace(
                stiffness=HAND_REAL_STIFFNESS,
                damping=HAND_REAL_DAMPING,
                effort_limit_sim=HAND_REAL_EFFORT_LIMITS,
            ),
        }
    )


@configclass
class HandRealEnvCfg(HandMoveEnvCfg):
    """``hand_move`` dynamics and objective with a realizable 105D policy input."""

    observations: HandRealObservationsCfg = HandRealObservationsCfg()
    commands: HandRealCommandsCfg = HandRealCommandsCfg()
    terminations: HandRealTerminationsCfg = HandRealTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep this helper reusable by hand_final, whose cube task inherits
        # from HandObjectEnvCfg rather than from this class.
        apply_hand_real_contract(self)
