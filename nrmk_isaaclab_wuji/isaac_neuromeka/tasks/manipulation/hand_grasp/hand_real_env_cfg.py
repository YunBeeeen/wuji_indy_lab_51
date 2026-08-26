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

import math

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp

from . import hand_real_mdp
from . import hand_move_mdp
from . import mdp as hand_grasp_mdp
from .hand_grasp_env_cfg import (
    CLOSE_TIP_GAP,
    FINGERTIPS,
    FUNCTIONAL_CONTACT_GROUPS,
    HAND_JOINT_NAMES,
    HAND_JOINTS,
    PALM,
    PREGRASP_STICK1_POSITION_P,
    PREGRASP_STICK1_QUATERNION_P,
    PREGRASP_STICK2_QUATERNION_P,
    PREGRASP_STICK2_POSITION_P,
    STICK_1,
    STICK_2,
    STICK_SIZE,
    STICK_TIP_OFFSET_O,
    TIP_AXIAL_OFFSET_STICK2,
    TIP_AXIAL_SIGMA,
    TIP_LATERAL_SIGMA,
    TIP_SEPARATION_DIRECTION_STICK2,
)
from .hand_move_env_cfg import (
    HandMoveCommandsCfg,
    HandMoveEnvCfg,
    HandMoveEventCfg,
    HandMoveRewardsCfg,
    HandMoveSceneCfg,
    HandMoveTerminationsCfg,
)


HAND_REAL_POLICY_OBS_DIM = 105
HAND_REAL_ORIENTATION_ERROR_MODE = "directed_axis"
HAND_REAL_TIP_CONTACT_SENSOR = "stick1_stick2_tip_contact"
HAND_REAL_TIP_FORCE_SATURATION_N = 0.30
HAND_REAL_TIP_FORCE_WEIGHT = 20.0
# Semantic order: thumb--Stick1, index--Stick1, middle--Stick1,
# palm--Stick2, thumb-middle--Stick2, ring--Stick2.  Only the index contact
# must now reach 0.30 N for full dense credit; the 0.02 N contact predicate is
# intentionally unchanged.
HAND_REAL_FUNCTIONAL_FORCE_SCALES_N = (
    0.10,
    0.30,
    0.10,
    0.10,
    0.10,
    0.10,
)

# Parked reset-noise ranges for a later curriculum after the initial
# model_4500 tip-force fine-tuning stage.  Probability zero below makes the
# active reset deterministic even though the ranges remain documented here.
HAND_REAL_STICK_POSITION_NOISE_M = ((-0.005, 0.005),) * 3
HAND_REAL_STICK_ORIENTATION_NOISE_RAD = (
    (-math.radians(5.0), math.radians(5.0)),
) * 3
# Initial force fine-tuning isolates one variable: deterministic pose_005 reset.
# Keep the parked ranges above for the later reset-noise curriculum.
HAND_REAL_STICK_RESET_NOISE_PROBABILITY = 0.0

# Restore the exact six tip/palm groups used by the 2026-08-18 model_4500.
# hand_real2 owns its link4 topology independently.
HAND_REAL_FUNCTIONAL_CONTACT_GROUPS = FUNCTIONAL_CONTACT_GROUPS

# Recorded pose_005 was the previous reset and joint-reference target.  Keep it
# explicit and independent from PREGRASP_JOINT_POSITIONS, which is now the
# 4 mm-clearance reset pose used for acquisition experiments.
HAND_REAL_PREV_JOINT_TARGET = (
    0.5377866626,
    0.8436813951,
    0.0377136655,
    -0.0000001810,
    0.7017297745,
    0.0553143807,
    1.1822255850,
    1.4215219021,
    0.4649881423,
    -0.0292181600,
    1.6272000000,
    1.1032750607,
    0.9151425958,
    -0.0129909236,
    1.3248542547,
    0.3182539344,
    0.7154092789,
    0.0788998753,
    1.6272000000,
    0.2546040118,
)

# model_4500 target: recorded pose_005, with no link4-driven J1 flexion.
HAND_REAL_JOINT_REFERENCE = HAND_REAL_PREV_JOINT_TARGET

# hand_real is now the zero-rotation OPEN/CLOSE continuation of hand_real2.
# Keep this task-local so edits to HAND_MOVE_SCHEDULE cannot silently enable
# rotation here.  hand_real2 retains its own independent schedule.
HAND_REAL_OPEN_CLOSE_SCHEDULE = hand_move_mdp.HandMoveScheduleCfg(
    range_x=(0.0, 0.0),
    range_y=(0.0, 0.0),
    range_z=(0.0, 0.0),
)
HAND_REAL_OPEN_CLOSE_SCHEDULE.validate()

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
    """Zero-rotation OPEN/CLOSE continuation of the acquired grasp."""

    root_orientation = hand_move_mdp.HandMoveRootOrientationCommandCfg(
        schedule=HAND_REAL_OPEN_CLOSE_SCHEDULE,
    )

    open_close = hand_move_mdp.HandMoveOpenCloseCommandCfg(
        schedule=HAND_REAL_OPEN_CLOSE_SCHEDULE,
        neutral_before_open_close=True,
    )


@configclass
class HandRealEventCfg(HandMoveEventCfg):
    """Task-local noisy reset used to fine-tune the established hand_real policy."""

    # Previous deterministic wiring, preserved explicitly for rollback:
    # reset_pregrasp = HandMoveEventCfg.reset_pregrasp
    reset_pregrasp = EventTerm(
        func=hand_real_mdp.reset_to_noisy_functional_pregrasp,
        mode="reset",
        params={
            "hand_cfg": HAND_JOINTS,
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            # Keep the model_4500 pose_005 hand reset; only the two stick poses
            # receive the task-local fine-tuning noise below.
            "joint_positions": HAND_REAL_PREV_JOINT_TARGET,
            "stick1_position_p": PREGRASP_STICK1_POSITION_P,
            "stick1_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
            "stick2_position_p": PREGRASP_STICK2_POSITION_P,
            "stick2_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
            "position_noise_m": HAND_REAL_STICK_POSITION_NOISE_M,
            "orientation_noise_rad": HAND_REAL_STICK_ORIENTATION_NOISE_RAD,
            "probability": HAND_REAL_STICK_RESET_NOISE_PROBABILITY,
        },
    )


@configclass
class HandRealTipForceSceneCfg(HandMoveSceneCfg):
    """The model_4500 scene plus one Stick1--Stick2 force sensor."""

    stick1_stick2_tip_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Stick1",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )


@configclass
class HandRealRewardsCfg(HandMoveRewardsCfg):
    """The model_4500 reward set plus one CLOSE-only tip-force term."""

    tip_press_force = RewTerm(
        func=hand_real_mdp.tip_press_force,
        weight=HAND_REAL_TIP_FORCE_WEIGHT,
        params={
            "command_name": "open_close",
            "tip_contact_sensor_name": HAND_REAL_TIP_CONTACT_SENSOR,
            "sensor_groups": HAND_REAL_FUNCTIONAL_CONTACT_GROUPS,
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "stick1_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick2_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick_thickness": STICK_SIZE[0],
            "close_target_gap": CLOSE_TIP_GAP,
            "reference_separation_direction_stick2": (
                TIP_SEPARATION_DIRECTION_STICK2
            ),
            "reference_axial_offset_stick2": TIP_AXIAL_OFFSET_STICK2,
            "force_saturation": HAND_REAL_TIP_FORCE_SATURATION_N,
            "functional_force_scale": HAND_REAL_FUNCTIONAL_FORCE_SCALES_N,
            "gap_sigma": 0.003,
            "lateral_sigma": TIP_LATERAL_SIGMA,
            "axial_sigma": TIP_AXIAL_SIGMA,
        },
    )


@configclass
class HandRealTerminationsCfg(HandMoveTerminationsCfg):
    """Match the successful hand_move rollout contract without contact-loss reset."""

    # The successful 2026-08-08 hand_move lineage learned through transient
    # contact loss.  Keep the rollout alive so hand_real can learn the same
    # recovery behavior; actual Stick1/Stick2 drops and time-out stay inherited.
    functional_contact_lost = None


def apply_hand_real_actuator_contract(env_cfg) -> None:
    """Apply the real hand's action scale, joint clamp, gains, and effort caps."""
    # JointAction parses dictionary scales by resolved physical joint name.
    # Assign every joint explicitly because unspecified dictionary entries
    # default to 1.0 in Isaac Lab's JointAction implementation.
    hand_action = env_cfg.actions.hand_action.replace(
        scale=dict(HAND_REAL_ACTION_SCALE),
        # Use the articulation's official per-joint lower limits.  In
        # particular, do not resurrect hand_setting's old cosmetic Joint4=0
        # floor, which made the distal action space differ from hand_real.
        joint_position_lower_overrides=None,
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


def apply_hand_real_contract(env_cfg) -> None:
    """Apply deploy reward semantics plus the real-hand actuator contract."""
    env_cfg.rewards.stick2_reference_pose.params["orientation_error_mode"] = (
        HAND_REAL_ORIENTATION_ERROR_MODE
    )
    env_cfg.rewards.success.params["orientation_error_mode"] = (
        HAND_REAL_ORIENTATION_ERROR_MODE
    )
    env_cfg.terminations.success.params["orientation_error_mode"] = (
        HAND_REAL_ORIENTATION_ERROR_MODE
    )
    apply_hand_real_actuator_contract(env_cfg)


@configclass
class HandRealEnvCfg(HandMoveEnvCfg):
    """``hand_move`` dynamics and objective with a realizable 105D policy input."""

    # model_4500's six tip/palm sensors plus one reward-only stick--stick sensor.
    # Keep the later link4/wrong-contact scene confined to hand_real2.
    scene: HandRealTipForceSceneCfg = HandRealTipForceSceneCfg(
        num_envs=4096,
        env_spacing=1.0,
    )
    observations: HandRealObservationsCfg = HandRealObservationsCfg()
    commands: HandRealCommandsCfg = HandRealCommandsCfg()
    events: HandRealEventCfg = HandRealEventCfg()
    rewards: HandRealRewardsCfg = HandRealRewardsCfg()
    terminations: HandRealTerminationsCfg = HandRealTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep this helper reusable by hand_final, whose cube task inherits
        # from HandObjectEnvCfg rather than from this class.
        apply_hand_real_contract(self)
        HAND_REAL_OPEN_CLOSE_SCHEDULE.validate()
        self.episode_length_s = HAND_REAL_OPEN_CLOSE_SCHEDULE.episode_length_s

        # Restore model_4500's tip-only contact and pose_005 q-reference
        # contract.  This first force fine-tuning stage keeps both reset noise
        # and external disturbance disabled.
        self.events.stick_disturbance.params["sensor_groups"] = (
            HAND_REAL_FUNCTIONAL_CONTACT_GROUPS
        )
        self.events.stick_disturbance.params.pop("group_reduction", None)
        # External disturbance is a later curriculum stage.
        self.events.stick_disturbance.params["probability"] = 0.0
        self.rewards.joint_reference.params["reference_joint_positions"] = (
            HAND_REAL_JOINT_REFERENCE
        )
        # hand_real only: HandReal2RewardsCfg does not define tip_press_force
        # and therefore retains the inherited uniform 0.10 N contact scale.
        if hasattr(self.rewards, "tip_press_force"):
            self.rewards.functional_contact_min.params["force_scale"] = (
                HAND_REAL_FUNCTIONAL_FORCE_SCALES_N
            )
