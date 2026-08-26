"""Memoryless staged learning from an open hand to a chopstick setting."""

from __future__ import annotations

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp
from isaac_neuromeka.mdp.actions import CustomResidualJointActionCfg
from isaac_neuromeka.utils.etc import EmptyCfg

from . import hand_real_mdp
from . import mdp as hand_grasp_mdp
from .hand_grasp_env_cfg import (
    FINGERTIPS,
    FUNCTIONAL_CONTACT_GROUPS,
    HAND_JOINT_NAMES,
    HAND_JOINTS,
    PALM,
    PREGRASP_JOINT_POSITIONS,
    PREGRASP_STICK1_POSITION_P,
    PREGRASP_STICK1_QUATERNION_P,
    PREGRASP_STICK2_POSITION_P,
    PREGRASP_STICK2_QUATERNION_P,
    STICK1_PIVOT_OFFSET_O,
    STICK_1,
    STICK_2,
    STICK_SIZE,
    HandGraspActionsCfg,
    HandGraspEnvCfg,
    HandGraspSceneCfg,
)
from .hand_real_env_cfg import apply_hand_real_actuator_contract


# Task-local reset positions validated with the keyboard tool.  Writing the
# absolute values is clearer than expressing them as offsets from the inherited
# HandGraspSceneCfg defaults, which hand_grasp itself replaces with pose_005 at
# reset.  The two dynamic sticks remain parallel and 20 mm apart.
SETTING_STICK1_RESET_POS = (0.075, 0.0, 0.5195)
SETTING_STICK2_RESET_POS = (0.055, 0.0, 0.5195)

# These task-local sensors do not change the six functional contacts.  They
# detect the unwanted solution where any part of the index finger wraps around
# and supports Stick2 instead of placing its tip between the sticks on Stick1.
INDEX_WRONG_STICK2_SENSOR_NAMES = (
    "index_link1_stick2_wrong",
    "index_link2_stick2_wrong",
    "index_link3_stick2_wrong",
    "index_link4_stick2_wrong",
    "index_tip_stick2_wrong",
)
INDEX_WRONG_STICK2_SENSOR_GROUPS = (INDEX_WRONG_STICK2_SENSOR_NAMES,)

# The three of the six semantic contacts that require actual fingertip
# placement.  FUNCTIONAL_CONTACT_GROUPS is ordered
# (thumb_distal_stick1, index_tip_stick1, middle_tip_stick1,
#  palm_stick2, thumb_mid_stick2, ring_tip_stick2), so this is 1, 2 and 5.
#
# The other three are free.  The palm has no actuator at all and the thumb
# carries 2.34x the torque and 3.9x the stiffness of the other fingers, so
# "rest Stick2 on the palm and press with the thumb" wins all three of them
# without moving a finger.  Measured on 2026-08-26_03-27-36 at iteration 331:
#
#     palm_stick2          2.654 N      index_tip_stick1   0.000 N (max 0.000)
#     thumb_mid_stick2     2.584 N      ring_tip_stick2    0.000 N (max 0.000)
#     thumb_distal_stick1  1.238 N      middle_tip_stick1  0.005 N (max 0.123)
#
# functional_contact_count sat at 2.98 for 200 iterations.  That is the whole
# "wall of three": three contacts are free and the other three are not.
# Which face of which shaft each fingertip belongs on, in that stick's local
# frame.  (stick_cfg_key, surface_axis, surface_sign); axis 2 is the 3.5 mm
# half-extent cross-section axis, so +1 is the upper face and -1 the lower.
#
#   index  -> Stick1 upper   (already recorded in
#                             SETTING_INDEX_STICK1_UPPER_SURFACE_PARAMS)
#   middle -> Stick1 lower   (user, 2026-08-26)
#   ring   -> Stick2 lower   (user, 2026-08-26)
#
# Nothing in the reward reads this yet.  Both contact_group_strength and
# body_box_surface_distance are magnitudes, so the objective cannot tell an
# upper-face contact from a lower-face one and pays 61 points either way.  The
# Metrics/hand_setting/*_face_z diagnostics added on the same day are what
# measures it; wire this in only once those confirm the current topology.
SETTING_FINGERTIP_TARGET_FACES = (
    ("stick1", 2, +1.0),
    ("stick1", 2, -1.0),
    ("stick2", 2, -1.0),
)

SETTING_FINGERTIP_CONTACT_GROUPS = (
    FUNCTIONAL_CONTACT_GROUPS[1],
    FUNCTIONAL_CONTACT_GROUPS[2],
    FUNCTIONAL_CONTACT_GROUPS[5],
)


@configclass
class HandSettingSceneCfg(HandGraspSceneCfg):
    """Add setting-only sensors for unintended Index--Stick2 contacts."""

    index_link1_stick2_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger2_link1",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    index_link2_stick2_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger2_link2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    index_link3_stick2_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger2_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    index_link4_stick2_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger2_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    index_tip_stick2_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger2_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )

# Phase-boundary reset for the first feasibility experiment.  This is an
# initial state, not a hidden grasp preload: reset_hand_joint_state writes the
# same value to the simulated joint state and its PD target.  The only
# non-zero joint is the thumb joint used to open the thumb-index valley.
#
# A later robustness A/B may replace -0.1659 with zero or a reset
# distribution.  Keeping the value here explicit makes that a single-variable
# experiment instead of silently changing the reward or action semantics.
SETTING_OPEN_JOINT_POSITIONS = (
    0.0,
    -0.1659,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)

@configclass
class HandSettingActionsCfg(HandGraspActionsCfg):
    """Placeholder action replaced by the hand_real contract in post-init."""

    # Keep the stable residual range on joints 1--3.  Joint 4 needs a larger
    # one-step position lead to overcome its common distal-joint dead zone;
    # increasing only joint 4 avoids the instability seen with scale=0.3 on
    # all twenty joints.
    hand_action = CustomResidualJointActionCfg(
        asset_name="robot",
        joint_names=HAND_JOINT_NAMES,
        preserve_order=True,
        scale=0.1,
        #scale={
        #    "finger[1-5]_joint[1-4]": 0.1,
            # Direct drive A/B: 0.3 rad capped Joint4 torque near 0.3 Nm and
            # left middle/ring/little motionless.  At 0.5 rad all five Joint4
            # joints moved with about 80% of the 0.6 Nm effort limit; 0.6 rad
            # merely pushed them to about 96%, so 0.5 keeps useful headroom.
            # "finger[1-5]_joint4": 0.5,
        #},
        clamp_to_limits=True,
        # Keep the articulation's official lower limits.  Post-init also
        # replaces the scale with hand_real's per-joint residual table.
        joint_position_lower_overrides=None,
    )

THUMB_DISTAL = SceneEntityCfg(
    "robot",
    body_names=["finger1_link3"],
)
INDEX_TIP = SceneEntityCfg(
    "robot",
    body_names=["finger2_tip_link"],
)
MIDDLE_TIP = SceneEntityCfg(
    "robot",
    body_names=["finger3_tip_link"],
)
RING_TIP = SceneEntityCfg(
    "robot",
    body_names=["finger4_tip_link"],
)

# Joint reference recorded in hand_setting/2026-08-10_18-30-36 params/env.yaml.
# The shared PREGRASP_JOINT_POSITIONS was later replaced by the 4 mm-clearance
# pose that hand_real now resets to, so the two differ by RMS 1.90 deg
# (max 4.47 deg on finger2_joint2).  Kept here, unused, so restoring the exact
# 08-10 reward target is a one-line switch in the four joint-reference terms
# below.  Do NOT edit PREGRASP_JOINT_POSITIONS itself: hand_grasp, hand_move,
# hand_real and hand_object all read it.
SETTING_LEGACY_0810_JOINT_REFERENCE = (
    0.5377866626, 0.8436813951, 0.0377136655, -0.0000001810,
    0.7017297745, 0.0553143807, 1.1822255850, 1.4215219021,
    0.4649881423, -0.0292181600, 1.6298730373, 1.1032750607,
    0.9151425958, -0.0129909236, 1.3248542547, 0.3182539344,
    0.7154092789, 0.0788998753, 1.6281884909, 0.2546040118,
)

# Thumb acquisition is handled by the pivot/contact terms.  These are the
# sixteen non-thumb joints that need a stronger acquisition q guide.  The
# little finger has no direct stick-contact target, but it must still reach the
# all-20 transition pose and mechanically support the ring finger.
SETTING_MISSING_FINGER_JOINT_NAMES = [
    f"finger{finger}_joint{joint}"
    for finger in range(2, 6)
    for joint in range(1, 5)
]
SETTING_MISSING_FINGER_JOINTS = SceneEntityCfg(
    "robot",
    joint_names=SETTING_MISSING_FINGER_JOINT_NAMES,
    preserve_order=True,
)
SETTING_MISSING_FINGER_JOINT_POSITIONS = PREGRASP_JOINT_POSITIONS[4:20]

STICK_HALF_EXTENT = tuple(0.5 * value for value in STICK_SIZE)
SETTING_LONG_AXIS = 1
SETTING_STAGE1_GATE_PARAMS = {
    "palm_cfg": PALM,
    "stick1_cfg": STICK_1,
    "stick2_cfg": STICK_2,
    "thumb_cfg": THUMB_DISTAL,
    "stick1_reference_position_p": PREGRASP_STICK1_POSITION_P,
    "stick1_reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
    "stick2_reference_position_p": PREGRASP_STICK2_POSITION_P,
    "stick2_reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
    "stick1_half_extent": STICK_HALF_EXTENT,
    "long_axis": SETTING_LONG_AXIS,
    "pivot_station": STICK1_PIVOT_OFFSET_O[SETTING_LONG_AXIS],
    "position_sigma": 0.10,
    "orientation_sigma": 1.5707963268,
    "thumb_sigma": 0.02,
    # Reject the premature-contact solution seen around pair_score ~= 0.57,
    # while allowing the recently observed ~=0.66 acquisition state through.
    # Measured on 2026-08-25_21-06-54 at iteration 1280: pair_score mean 0.617,
    # episode max 0.720, and the metric stage1_ready 0.596 -- that metric is
    # exactly "pair >= 0.65" here, because the loose sigma=0.06 thumb kernel it
    # uses never fails at these distances.  So the stick side already clears
    # this threshold about 60% of the time and is left alone.
    "pair_score_threshold": 0.65,
    # TEMPORARY DIAGNOSTIC, 2026-08-25.  Original value 0.35 (d <= 21.0 mm).
    #
    # Every reward except two_stick_reference_min and reference_thumb_pivot_min
    # sits behind this gate, and the gate has never opened on the 105D runs, so
    # the linear + fine joint-reference rewards below have literally never been
    # evaluated.  Relaxing this is the cheapest way to find out whether that new
    # structure does anything; it is NOT a claim that a 32 mm thumb is
    # acceptable.  Same run at iteration 1280:
    #
    #     thumb_pivot_distance  mean 34.1 mm   episode best 29.0 mm
    #     thumb_score(sigma 0.02)  mean 0.182  episode best 0.234
    #
    # 0.20 corresponds to d <= 32.2 mm, roughly 3 mm of headroom over the
    # typical episode best, and the Stage-1 latch only needs one qualifying
    # step per episode.  If stage1_unlocked is still ~0 after about 200
    # iterations, drop to 0.15 (d <= 37.9 mm) rather than tightening anything
    # else.  Restore 0.35 before this becomes a baseline, and fix the thumb
    # itself -- reference_thumb_pivot_min is a min() that currently selects the
    # stick-pose branch, so the thumb has exactly zero reward gradient.
    "thumb_score_threshold": 0.20,
}
# Strict Stage-2 readiness remains a memoryless diagnostic: Stage 1 must be
# valid and every joint must be within fifteen degrees of pose_005.  Contact
# shaping below uses a one-way per-episode Stage-1 latch so transient stick
# motion during finger closure does not erase the next-stage learning signal.
# 2026-08-10_18-30-36 baseline: the Stage-1 reference terms only decay after
# all twenty joints enter the original strict five-degree band.  Stage-2
# contact itself is parked below, but this threshold is still part of the
# active Stage-1 reward functions' weight schedule.
SETTING_STAGE2_JOINT_ERROR_THRESHOLD = 0.0872664626
# Contact shaping begins weakly at 0.8 rad all-joint RMSE and reaches full
# strength at the 15-degree threshold.  The strict Stage-2 readiness
# metric remains the memoryless maximum-error test above.
SETTING_STAGE2_CONTACT_START_RMSE = 0.80
SETTING_SEMANTIC_APPROACH_RANGE = 0.08
# Long-range companion to the range above.  Measured on 2026-08-26_09-52-55 at
# iteration 532, the three fingertip surface distances were index 49.6 mm,
# middle 19.9 mm and ring 92.9 mm.  clip(1 - d/0.08) is flat zero beyond 80 mm,
# so the ring finger -- the one that has never once touched Stick2 -- sat
# outside the term entirely with no gradient at all, and semantic_approach_min
# read 0.007.  0.30 keeps a constant 3.3 per metre out to 300 mm.
SETTING_SEMANTIC_APPROACH_COARSE_RANGE = 0.30
# Keep semantic contacts on the central 160 mm of the 180 mm shaft.  This
# rejects end-cap reward farming without forcing one exact contact point.
SETTING_AXIAL_HALF_LENGTH = 0.08
SETTING_INDEX_BETWEEN_CORE_PARAMS = {
    "index_cfg": INDEX_TIP,
    "stick_half_extent": STICK_HALF_EXTENT,
    "axial_half_length": SETTING_AXIAL_HALF_LENGTH,
    # At the validated 20 mm centerline separation, the inner Stick1 face is
    # around coordinate 0.175.  A 0.15 soft margin therefore gives full credit
    # near the correct face while rejecting either outer side.
    "between_margin_fraction": 0.15,
    "stick1_proximity_sigma": 0.04,
}
# Current A/B: Index belongs on Stick1 local +z (the established upper face),
# but it need not enter the gap between Stick1 and Stick2.  The target spans
# the central 160 mm shaft and the whole face rather than prescribing a point.
SETTING_INDEX_STICK1_UPPER_SURFACE_PARAMS = {
    "index_cfg": INDEX_TIP,
    "object_half_extent": STICK_HALF_EXTENT,
    "axial_half_length": SETTING_AXIAL_HALF_LENGTH,
    "surface_axis": 2,
    "surface_sign": 1.0,
    "tangent_margin": 0.01,
    "region_sigma": 0.005,
}
SETTING_INDEX_BETWEEN_PROGRESS_START = 0.25
SETTING_INDEX_BETWEEN_READY_THRESHOLD = 0.75
SETTING_STAGE2_PARAMS = {
    "asset_cfg": HAND_JOINTS,
    "reference_joint_positions": PREGRASP_JOINT_POSITIONS,
    "joint_error_threshold": SETTING_STAGE2_JOINT_ERROR_THRESHOLD,
    "joint_error_start_threshold": SETTING_STAGE2_CONTACT_START_RMSE,
    # Keep between out of the Stage-2 hard gate for this reward-only A/B.
    # The experiment should shape Index motion without changing readiness.
    # "index_cfg": INDEX_TIP,
    # "index_between_axial_half_length": SETTING_AXIAL_HALF_LENGTH,
    # "index_between_margin_fraction": 0.15,
    # "index_stick1_proximity_sigma": 0.04,
    # "index_between_progress_start": (
    #     SETTING_INDEX_BETWEEN_PROGRESS_START
    # ),
    # "index_between_ready_threshold": (
    #     SETTING_INDEX_BETWEEN_READY_THRESHOLD
    # ),
    **SETTING_STAGE1_GATE_PARAMS,
}

SETTING_COMPLETION_PARAMS = {
    "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
    "palm_cfg": PALM,
    "stick1_cfg": STICK_1,
    "stick2_cfg": STICK_2,
    "thumb_distal_cfg": THUMB_DISTAL,
    "index_tip_cfg": INDEX_TIP,
    "middle_tip_cfg": MIDDLE_TIP,
    "ring_tip_cfg": RING_TIP,
    "stick1_reference_position_p": PREGRASP_STICK1_POSITION_P,
    "stick1_reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
    "stick2_reference_position_p": PREGRASP_STICK2_POSITION_P,
    "stick2_reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
    "long_axis": SETTING_LONG_AXIS,
    "axial_half_length": SETTING_AXIAL_HALF_LENGTH,
}

SETTING_SUCCESS_PARAMS = {
    **SETTING_COMPLETION_PARAMS,
    "contact_threshold": 0.02,
    "stick1_position_error_limit": 0.02,
    "stick1_orientation_error_limit": 0.3490658504,
    # Stick2 is the lower anchor rail, so its translation tolerance is tighter.
    "stick2_position_error_limit": 0.015,
    "stick2_orientation_error_limit": 0.3490658504,
    "linear_speed_limit": 0.15,
    "angular_speed_limit": 3.0,
    "hold_steps": 30,
}

# Parked finite-shaft proxy from the previous A/B.  Keep the definition for
# reproducibility and offline diagnostics, but do not use it in the active
# reward or state gate.  The simpler active definition below reuses the stable
# Stick2 pose already validated in hand_grasp.
STICK2_VALLEY_REFERENCE_POINT_OFFSET_O = (0.0, -0.06, 0.0)
STICK2_VALLEY_GEOMETRY_PARAMS = {
    "palm_cfg": PALM,
    "stick2_cfg": STICK_2,
    "stick2_reference_position_p": PREGRASP_STICK2_POSITION_P,
    "stick2_reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
    "valley_point_offset_o": STICK2_VALLEY_REFERENCE_POINT_OFFSET_O,
    "stick_half_length": STICK_HALF_EXTENT[SETTING_LONG_AXIS],
    # Contact support disambiguates outside-valley false positives, so the
    # geometric corridor can allow small pose variation around pose_005.
    "valley_point_error_limit": 0.01,
    "valley_axis_error_limit": 0.2617993878,
}
STICK2_VALLEY_ANCHOR_GROUPS = (
    FUNCTIONAL_CONTACT_GROUPS[3],
    FUNCTIONAL_CONTACT_GROUPS[4],
)
STICK2_IN_VALLEY_PARAMS = {
    "anchor_groups": STICK2_VALLEY_ANCHOR_GROUPS,
    "palm_cfg": PALM,
    "stick2_cfg": STICK_2,
    "stick2_reference_position_p": PREGRASP_STICK2_POSITION_P,
    "stick2_reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
    "contact_threshold": 0.02,
    "position_error_limit": 0.015,
    "orientation_error_limit": 0.3490658504,
}
STICK2_POSE_SUPPORT_PARAMS = {
    **STICK2_IN_VALLEY_PARAMS,
    # The force-shaping corridor is deliberately looser than the strict
    # 15 mm / 20 deg in-valley gate.
    "support_position_error_limit": 0.03,
    "support_orientation_error_limit": 0.5235987756,
}

HAND_SETTING_POLICY_OBS_DIM = 105


@configclass
class HandSettingObservationsCfg:
    """105D deployable state matching ``hand_real`` with neutral mode.

    ``hand_setting`` has no OPEN/CLOSE command manager.  It appends the fixed
    neutral/setting state ``[0, 0]`` in the same final two slots where
    ``hand_real`` supplies its OPEN/CLOSE command:

    40 joint positions (previous/current, limit-normalized)
    + 15 current fingertip positions
    + 28 stick poses (previous/current palm-frame xyz+wxyz for both sticks)
    + 20 actions that produced the current state
    + 2 neutral mode values.

    History samples are oldest-to-newest.  Immediately after reset both slots
    contain the reset sample, matching ``hand_real`` and representing zero
    inferred motion without simulator-only joint or rigid-body velocity.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor/critic observation group shared by all environments."""

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
        open_close_mode = ObsTerm(func=hand_real_mdp.neutral_open_close_mode)

        def __post_init__(self):
            """Concatenate the terms in the documented 105D order."""

            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HandSettingEventCfg:
    """Reset the open hand and two sticks without preloading the grasp."""

    # HandSettingEnvCfg sets task-local absolute positions x=0.075/0.055, so
    # reset restores the validated parallel dynamic sticks with their 20 mm
    # separation and zero velocity.
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    # Active acquisition reset: the hand starts open while the sticks use the
    # task-local world spawn above.  PREGRASP_JOINT_POSITIONS remains only the
    # 4 mm-clearance joint-reference target in the reward terms below.
    reset_open_hand = EventTerm(
        func=hand_grasp_mdp.reset_hand_joint_state,
        mode="reset",
        params={
            "hand_cfg": HAND_JOINTS,
            "joint_positions": SETTING_OPEN_JOINT_POSITIONS,
        },
    )
def _region_term(
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    weight: float = 0.0,
) -> RewTerm:
    """Configure existing shaft approach shaping without adding reward terms."""

    return RewTerm(
        func=hand_grasp_mdp.body_box_shaft_region_proximity,
        weight=weight,
        params={
            "body_cfg": body_cfg,
            "object_cfg": object_cfg,
            "object_half_extent": STICK_HALF_EXTENT,
            "long_axis": SETTING_LONG_AXIS,
            "axial_half_length": SETTING_AXIAL_HALF_LENGTH,
            # Broad only for approach shaping.  Final completion still uses
            # hard semantic contact pairs and the tighter pose gates below.
            "sigma": 0.05,
        },
    )


def _valley_region_term(
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    weight: float,
) -> RewTerm:
    """Reuse one region term after force-validated Stick2 valley entry."""
    return RewTerm(
        func=hand_grasp_mdp.seated_gated_body_box_shaft_region_proximity,
        weight=weight,
        params={
            "body_cfg": body_cfg,
            "object_cfg": object_cfg,
            "object_half_extent": STICK_HALF_EXTENT,
            "long_axis": SETTING_LONG_AXIS,
            "axial_half_length": SETTING_AXIAL_HALF_LENGTH,
            "sigma": 0.05,
            **STICK2_IN_VALLEY_PARAMS,
        },
    )


def _contact_term(
    sensor_group: tuple[str, ...],
    weight: float = 0.0,
) -> RewTerm:
    """Expose one saturated semantic contact as an independent reward tag."""

    # Keeping the term at weight zero preserves its TensorBoard tag for the
    # valley-only A/B without letting it influence the policy.
    return RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=weight,
        params={
            "sensor_groups": (sensor_group,),
            "force_scale": 0.10,
            "reduction": "mean",
        },
    )


def _valley_contact_term(
    sensor_groups: tuple[tuple[str, ...], ...],
    weight: float,
    reduction: str = "mean",
) -> RewTerm:
    """Reuse contact shaping after force-validated Stick2 valley entry."""
    return RewTerm(
        func=hand_grasp_mdp.seated_gated_contact_group_strength,
        weight=weight,
        params={
            "target_groups": sensor_groups,
            "force_scale": 0.10,
            "reduction": reduction,
            **STICK2_IN_VALLEY_PARAMS,
        },
    )


@configclass
class HandSettingRewardsCfg:
    """2026-08-11_17-45-05 coarse + Index-between reward baseline."""

    # 2026-08-26: long-range + finish, the same split just applied to the
    # sixteen joint references, now applied to the stick pose.
    #
    # Run 2026-08-26_00-24-02 opened the Stage-1 gate (stage1_unlocked 0 ->
    # 0.38, mean_reward 12.26 at iteration 455) and then lost it in eighteen
    # iterations: the thumb released (1.68 N -> 0.02 N) and the stick pose it
    # had been holding by pressing fell to 34.2 mm / 61.5 deg.  Under the
    # single tight kernel that state scores exp(-3.42 - 4.28) = 4.5e-4 with a
    # position gradient of 0.00054 per mm, so nothing could pull it back.  The
    # run has been frozen at mean_reward 0.07 for seventy iterations since.
    #
    #     state              coarse(0.10/90deg)   fine(0.01/0.25)
    #     34.2 mm / 61.5 deg        0.359            0.00045
    #     18.6 mm / 26.6 deg        0.617            0.02420
    #      5.0 mm /  5.7 deg        0.893            0.40657
    #
    # Coarse owns recovery, fine owns convergence, and they are added rather
    # than multiplied so tuning one sigma does not deflate the other.  The two
    # weights sum to the previous 12, so the value at the reference pose is
    # unchanged and this is a shape change, not a budget change.
    two_stick_reference_coarse_min = RewTerm(
        func=hand_grasp_mdp.ObjectPairReferencePoseMinTracking,
        # Deliberately the Stage-1 gate kernel.  The term that has to pull the
        # sticks back is then measuring the same thing the gate asks for, and
        # the gate threshold 0.65 reads directly off this term's value.
        weight=3.0,
        params={
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "stick1_reference_position_p": PREGRASP_STICK1_POSITION_P,
            "stick1_reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
            "stick2_reference_position_p": PREGRASP_STICK2_POSITION_P,
            "stick2_reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
            "position_sigma": 0.10,
            "orientation_sigma": 1.5707963268,
            # Rebase.  Without this the untouched spawn pose scores 0.348 and
            # the term became a do-nothing annuity that beat every behaviour
            # the policy had ever found (run 2026-08-26_02-11-08: mean_reward
            # frozen at 16.49 for 340 iterations, 99.8% of it from this term).
            # At 0.20 the idle pose still pays 0.185 -- about 4.4 points per
            # episode against roughly 135 for held contact -- and the gradient
            # stays alive down to about 50 mm and 100 deg, worse than anything
            # measured so far.
            "score_floor": 0.20,
        },
    )
    two_stick_reference_min = RewTerm(
        func=hand_grasp_mdp.ObjectPairReferencePoseMinTracking,
        # Unchanged kernel.  Keeping the term name keeps its TensorBoard tag
        # comparable with every earlier run.
        weight=3.0,
        params={
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "stick1_reference_position_p": PREGRASP_STICK1_POSITION_P,
            "stick1_reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
            "stick2_reference_position_p": PREGRASP_STICK2_POSITION_P,
            "stick2_reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
            "position_sigma": 0.01,
            "orientation_sigma": 0.25,
        },
    )
    # Parked after the fine-min A/B slowed Stage-1 acquisition.  Keep the
    # exact term here for rollback; the active baseline is the coarse term.
    # two_stick_reference_fine_min = RewTerm(
    #     func=hand_grasp_mdp.ObjectPairReferencePoseMinTracking,
    #     weight=6.0,
    #     params={
    #         "palm_cfg": PALM,
    #         "stick1_cfg": STICK_1,
    #         "stick2_cfg": STICK_2,
    #         "stick1_reference_position_p": PREGRASP_STICK1_POSITION_P,
    #         "stick1_reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
    #         "stick2_reference_position_p": PREGRASP_STICK2_POSITION_P,
    #         "stick2_reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
    #         "position_sigma": 0.002,
    #         "orientation_sigma": 0.05,
    #     },
    # )
    reference_thumb_pivot_min = RewTerm(
        func=hand_grasp_mdp.ObjectPairReferenceThumbStationMinTracking,
        # This term can increase only when both-stick alignment and thumb
        # approach improve together; thumb-only pivot chasing cannot farm it.
        #
        # 4 -> 1 on 2026-08-26.  It is a min() of the pair score on the tight
        # kernel against the thumb score, and at the measured stick pose the
        # pair branch scores 0.045 and always wins, so the thumb side of this
        # term has had a gradient of exactly zero.  It earned 1.53 of a nominal
        # 32 points on 2026-08-26_09-52-55; that headroom is structural, not
        # reachable.  The freed weight funds the long-range approach term below.
        # Fix the min() before giving this weight back.
        weight=1.0,
        params={
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "thumb_cfg": THUMB_DISTAL,
            "stick1_reference_position_p": PREGRASP_STICK1_POSITION_P,
            "stick1_reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
            "stick2_reference_position_p": PREGRASP_STICK2_POSITION_P,
            "stick2_reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
            "stick1_half_extent": STICK_HALF_EXTENT,
            "long_axis": SETTING_LONG_AXIS,
            "pivot_station": STICK1_PIVOT_OFFSET_O[SETTING_LONG_AXIS],
            "position_sigma": 0.01,
            "orientation_sigma": 0.25,
            "thumb_sigma": 0.06,
        },
    )
    stage1_joint_reference = RewTerm(
        func=hand_grasp_mdp.stage1_gated_joint_reference_tracking,
        # Preserve the existing all-20 exponential prior, including the
        # little finger that mechanically supports the ring finger.
        weight=3.0,
        params={
            "asset_cfg": HAND_JOINTS,
            "reference_joint_positions": PREGRASP_JOINT_POSITIONS,
            "joint_sigma": 0.80,
            # Effective weight: 8 before the Stage-2 gate and 2 while it is
            # active.  The weak all-20 prior includes the little finger.
            "stage2_reference_weight_ratio": 0.25,
            "stage2_joint_error_threshold": (
                SETTING_STAGE2_JOINT_ERROR_THRESHOLD
            ),
            **SETTING_STAGE1_GATE_PARAMS,
        },
    )
    # 2026-08-25: the single sigma=0.80 exponential guide for the sixteen
    # non-thumb joints is replaced by linear-for-range + exponential-for-finish.
    # Measured on 2026-08-25_21-06-54 at iteration 725, where the worst joint
    # (finger3_joint3) sat at 2.126 rad:
    #
    #     exp sigma 0.80   score 0.000857   gradient 0.005692
    #     linear R = 2.50  score 0.149600   gradient 0.400000   (70x)
    #
    # The term is 0.5 * mean + 0.5 * min, so that worst joint owns half of it,
    # and under the exponential that half was dead from reset onward.  The two
    # weights sum to the previous 12 on purpose: the value at q_ref is
    # unchanged, so this is a shape change and not a budget change.
    #
    # The trade is real and deliberate.  Between 0.4 and 1.0 rad the old
    # exponential was steeper (4.0-6.6 versus 1.27 here); that band already has
    # plenty of gradient and is not where the runs stall.  To buy it back, move
    # weight from fine to linear -- 8/4 and 12/8 were the other candidates.
    stage1_missing_joint_linear = RewTerm(
        func=hand_grasp_mdp.stage1_gated_joint_reference_linear_mean_min,
        weight=3.0,
        params={
            "asset_cfg": SETTING_MISSING_FINGER_JOINTS,
            "reference_joint_positions": (
                SETTING_MISSING_FINGER_JOINT_POSITIONS
            ),
            # Must stay above the largest observed error (2.13 rad) or the
            # blocking joint scores a flat zero and the gradient dies again.
            # Same range the signed-progress potential uses, so the two terms
            # cannot disagree about how far "far" is.
            "joint_linear_range": 2.50,
            "stage2_asset_cfg": HAND_JOINTS,
            "stage2_reference_joint_positions": PREGRASP_JOINT_POSITIONS,
            "stage2_joint_error_threshold": (
                SETTING_STAGE2_JOINT_ERROR_THRESHOLD
            ),
            # Acquisition scaffold: it fades inside the five-degree band and
            # returns automatically when that gate fails.
            "stage2_reference_weight_ratio": 0.0,
            **SETTING_STAGE1_GATE_PARAMS,
        },
    )
    stage1_missing_joint_fine = RewTerm(
        func=hand_grasp_mdp.stage1_gated_joint_reference_mean_min,
        weight=3.0,
        params={
            "asset_cfg": SETTING_MISSING_FINGER_JOINTS,
            "reference_joint_positions": (
                SETTING_MISSING_FINGER_JOINT_POSITIONS
            ),
            # Finish only.  exp(-(e/0.15)^2) is under 0.01 beyond 0.32 rad
            # (18 deg), so this term is silent across the whole approach and
            # then dominates: at the five-degree Stage-2 band its gradient is
            # 17.6 against the linear term's 1.27.
            "joint_sigma": 0.15,
            "stage2_asset_cfg": HAND_JOINTS,
            "stage2_reference_joint_positions": PREGRASP_JOINT_POSITIONS,
            "stage2_joint_error_threshold": (
                SETTING_STAGE2_JOINT_ERROR_THRESHOLD
            ),
            # This one must NOT fade.  It is the maintenance reward for the
            # converged pose, and it is also what keeps crossing into the
            # five-degree band from cutting the joint income by 90%.
            "stage2_reference_weight_ratio": 1.0,
            **SETTING_STAGE1_GATE_PARAMS,
        },
    )
    # Previous per-step linear annuity, parked for a one-line A/B rollback.
    # It remained positive at the observed 1--2 rad errors, so a policy could
    # hold a distant pose under the Stage-1 gate and collect it every step.
    #stage1_missing_joint_linear_reference = RewTerm(
    #    func=hand_grasp_mdp.stage1_gated_joint_reference_linear_mean_min,
    #    weight=100.0,
    #    params={
    #        "asset_cfg": SETTING_MISSING_FINGER_JOINTS,
    #        "reference_joint_positions": (
    #            SETTING_MISSING_FINGER_JOINT_POSITIONS
    #        ),
    #        "joint_linear_range": 2.50,
    #        "stage2_asset_cfg": HAND_JOINTS,
    #        "stage2_reference_joint_positions": PREGRASP_JOINT_POSITIONS,
    #        "stage2_joint_error_threshold": (
    #            SETTING_STAGE2_JOINT_ERROR_THRESHOLD
    #        ),
    #        "stage2_reference_weight_ratio": 0.0,
    #        **SETTING_STAGE1_GATE_PARAMS,
    #    },
    #)
    # 2026-08-25: best-so-far -> signed progress.  Same potential, same
    # weights, same budget; only the sign policy changes.  Rollback is one
    # line: put Stage1MissingJointBestSoFar back and rename the term.
    stage1_missing_joint_signed_progress = RewTerm(
        func=hand_grasp_mdp.Stage1MissingJointSignedProgress,
        # Each of the sixteen non-thumb joints owns an independent signed
        # difference.  The episode total telescopes to
        # (Phi_final - Phi_unlock), still at most one normalized score unit,
        # so at 30 Hz weight 3000 kept the same ~100-point episode budget the
        # best-so-far term had.  Halved with the rest of the scaffolding when
        # functional contact became the anchor: 1500 caps this at 50 points.
        weight=1500.0,
        params={
            "asset_cfg": SETTING_MISSING_FINGER_JOINTS,
            "reference_joint_positions": (
                SETTING_MISSING_FINGER_JOINT_POSITIONS
            ),
            "joint_linear_range": 2.50,
            # Index/Middle/Ring keep full progress credit.  Little remains in
            # the weak all-20 prior and the strict Stage-2 gate, but receives
            # only 25% credit here so it does not close across Ring's route
            # before Ring reaches Stick2.  Normalization inside the term keeps
            # the total best-so-far reward budget unchanged.
            "joint_reward_weights": (
                1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
                0.25, 0.25, 0.25, 0.25,
            ),
            **SETTING_STAGE1_GATE_PARAMS,
        },
    )
    # Active 2026-08-10 baseline guide: after Stage-1 unlock, route
    # Index/Middle toward Stick1 and Ring toward Stick2 without a hard
    # between-sticks constraint.
    stage1_semantic_surface_approach = RewTerm(
        func=hand_grasp_mdp.Stage1SemanticSurfaceApproach,
        weight=1.0,
        params={
            "index_cfg": INDEX_TIP,
            "middle_cfg": MIDDLE_TIP,
            "ring_cfg": RING_TIP,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "stick_half_extent": STICK_HALF_EXTENT,
            "approach_range": SETTING_SEMANTIC_APPROACH_RANGE,
            **SETTING_STAGE1_GATE_PARAMS,
        },
    )
    # Long-range half of the same split used for the joint references and the
    # stick pose: a wide kernel that carries the approach, added to rather than
    # replacing the narrow one that finishes it.  Same function, same fingers,
    # only approach_range and weight differ.
    #
    # The metric family Metrics/hand_setting/semantic_approach_* stays wired to
    # the 0.08 term above by name, so those numbers remain comparable with
    # every earlier run.
    stage1_semantic_surface_approach_coarse = RewTerm(
        func=hand_grasp_mdp.Stage1SemanticSurfaceApproach,
        weight=3.0,
        params={
            "index_cfg": INDEX_TIP,
            "middle_cfg": MIDDLE_TIP,
            "ring_cfg": RING_TIP,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "stick_half_extent": STICK_HALF_EXTENT,
            "approach_range": SETTING_SEMANTIC_APPROACH_COARSE_RANGE,
            **SETTING_STAGE1_GATE_PARAMS,
        },
    )
    # === The objective ===========================================
    # 2026-08-26: functional contact is what this task is for, and until now
    # it carried no reward at all -- the two Stage-2 contact terms were parked
    # on 2026-08-25 to reproduce the 08-10 six-term baseline and never came
    # back.  Every run since has optimized scaffolding only, which is how a
    # single wide-kernel shaping term ended up outscoring the best behaviour
    # any run had found (2026-08-26_02-11-08).
    #
    # contact_group_strength saturates each contact force at force_scale and
    # reduces over the six semantic groups, so its raw value is already in
    # [0, 1].  Weights are set from that: at 30 Hz over a 240-step episode one
    # unit of weight held at raw 1 is worth 8 reward points.
    #
    #   objective   min 40 + mean 5   = 45   -> 360 pt held for a full episode
    #   scaffolding all seven terms   = 20   -> 160 pt
    #   signed progress               = 1500 ->  50 pt (telescoping, once)
    #
    # Deliberately ungated.  The min over six groups is zero unless all six
    # contacts are live at the same instant, so it cannot be farmed by pressing
    # one pair -- which is exactly what the thumb was doing at 1.5 N in
    # 2026-08-26_00-24-02.  The mean can be partially farmed at 1/6 per contact,
    # so it is kept small: one pressed contact is worth 6.7 points.
    functional_contact_min = RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=17.0,
        params={
            "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "force_scale": 0.10,
            "reduction": "min",
        },
    )
    # Do not cut this again.  This six-way mean is the *early* ladder: it is
    # the only thing paying for the first three contacts, which are the ones
    # that stabilize the stick pose and get the Stage-1 gate open in the first
    # place.  At weight 5 one contact is worth 6.7 points and run
    # 2026-08-26_03-27-36 climbed 1.0 -> 1.79 -> 2.64 -> 2.98 contacts on it.
    # Dropping it to 2 made a rung worth 2.7 points and run 2026-08-26_04-06-25
    # stopped dead at 0.996 contacts from iteration 18 to 183 -- PPO was
    # healthy throughout (std 0.24-0.30, episode length 240), there was simply
    # nothing left to climb.  The weight that had been moved to
    # functional_contact_hard_mean was money placed on a rung four to seven
    # centimetres out of reach.
    functional_contact_mean = RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=5.0,
        params={
            "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "force_scale": 0.10,
            "reduction": "mean",
        },
    )
    # 2026-08-26: the rung between three contacts and six.
    #
    # min() over all six is zero until the last one lands, so from three to
    # five it contributes no gradient at all, and the only thing paying for a
    # fourth contact was the six-way mean at one sixth of weight 5 -- 6.7
    # points.  Against that, closing a finger onto a stick risks the Stage-1
    # gate, and every gated term put together was earning 55.2 points on
    # 2026-08-26_03-27-36.  A fourth contact was an 8.3-to-1 losing trade and
    # the policy correctly refused it for 200 iterations.
    #
    # Averaging over only the three fingertip contacts makes each one worth
    # (1/3) * 23 * 8 = 61 points, which clears the 55 points at risk.  The
    # weight comes out of the other two so the objective total stays 45:
    # min 40 -> 20, mean 5 -> 2, this 23.  Dropping the six-way mean also stops
    # paying 19.4 points for the three contacts that were always free.
    functional_contact_hard_mean = RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        # 23 -> 8 on 2026-08-26.  The remaining 15 moved to the face-aware term
        # below.  This face-blind share is deliberately kept so a wrong-face
        # contact still pays something: the middle tip was earning 50 points on
        # the wrong face of Stick1, and zeroing that outright is the same
        # all-or-nothing move that has stalled this task every time it was
        # tried.  A wrong-face contact is now worth 21.3 points and moving it
        # to the correct face is worth +40.0, with no cliff in between.
        weight=8.0,
        params={
            "sensor_groups": SETTING_FINGERTIP_CONTACT_GROUPS,
            "force_scale": 0.10,
            "reduction": "mean",
        },
    )
    # 2026-08-26: the same three fingertips, paid only on the face each one
    # belongs on.  Targets are recorded in SETTING_FINGERTIP_TARGET_FACES:
    # index on Stick1 upper, middle on Stick1 lower, ring on Stick2 lower.
    #
    # Measured on 2026-08-26_10-59-50 at iteration 608:
    #
    #     index   face_z +54.75 mm   0.000 N   correct side, 55.5 mm away
    #     middle  face_z +16.10 mm   0.649 N   WRONG side, and in index's place
    #     ring    face_z -38.39 mm   0.000 N   correct side, 59.9 mm away
    #
    # The index tip has never registered contact in any run, and the middle tip
    # sitting on its face is the most likely reason.  all_joint_reference_rmse
    # also regressed 0.43 -> 0.66 while that wrong-face grasp formed.
    functional_contact_hard_face = RewTerm(
        func=hand_grasp_mdp.fingertip_face_contact_strength,
        weight=15.0,
        params={
            "index_cfg": INDEX_TIP,
            "middle_cfg": MIDDLE_TIP,
            "ring_cfg": RING_TIP,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "index_sensor": SETTING_FINGERTIP_CONTACT_GROUPS[0][0],
            "middle_sensor": SETTING_FINGERTIP_CONTACT_GROUPS[1][0],
            "ring_sensor": SETTING_FINGERTIP_CONTACT_GROUPS[2][0],
            "object_half_extent": STICK_HALF_EXTENT,
            "surface_axis": 2,
            "index_surface_sign": 1.0,
            "middle_surface_sign": -1.0,
            "ring_surface_sign": -1.0,
            "force_scale": 0.10,
        },
    )
    # Parked after run 2026-08-12_00-04-29: this guide collapsed Stage-1
    # acquisition (pair score ~0.20, stage1_ready ~0) instead of correcting
    # Index topology.  Let the existing semantic approach and progressively
    # activated functional-contact rewards perform that correction.
    # stage1_index_stick1_upper_surface = RewTerm(
    #     func=hand_grasp_mdp.Stage1IndexStick1SurfaceTracking,
    #     weight=4.0,
    #     params={
    #         **SETTING_INDEX_STICK1_UPPER_SURFACE_PARAMS,
    #         **SETTING_STAGE1_GATE_PARAMS,
    #     },
    # )
    # 2026-08-11_17-45-05 baseline.  Parked on 2026-08-25 to restore the exact
    # six-term reward of hand_setting/2026-08-10_18-30-36, the run that opened
    # the Stage-1 gate from scratch (unlocked 0.33 by iter 234, thumb pivot
    # 18.6 mm at iter 936).  The 105D runs carrying this term plus the two
    # stage2 contact terms sat at unlocked 0.000 / 32 mm through iter 775.
    # stage1_index_between = RewTerm(
    #     func=hand_grasp_mdp.Stage1IndexBetweenTracking,
    #     weight=4.0,
    #     params={
    #         **SETTING_INDEX_BETWEEN_CORE_PARAMS,
    #         **SETTING_STAGE1_GATE_PARAMS,
    #         "asset_cfg": HAND_JOINTS,
    #         "reference_joint_positions": PREGRASP_JOINT_POSITIONS,
    #         "joint_error_threshold": (
    #             SETTING_STAGE2_JOINT_ERROR_THRESHOLD
    #         ),
    #         "joint_error_start_threshold": (
    #             SETTING_STAGE2_CONTACT_START_RMSE
    #         ),
    #     },
    # )
    # Index's intended semantic pair is Index-tip--Stick1 only.  Saturate the
    # maximum force over all five Index links against Stick2 so wrapping both
    # sticks cannot remain an equally rewarded shortcut.  This is deliberately
    # weak relative to the +5 mean / +20 minimum desired-contact terms: it
    # selects topology after approach without teaching Index to stay away.
    # Post-08-10 topology A/B, parked while reproducing 18-30-36 Stage 1.
    # index_wrong_stick2_contact = RewTerm(
    #     func=hand_grasp_mdp.contact_group_strength,
    #     weight=-2.0,
    #     params={
    #         "sensor_groups": INDEX_WRONG_STICK2_SENSOR_GROUPS,
    #         "force_scale": 0.10,
    #         "reduction": "mean",
    #     },
    # )
    # Same 2026-08-11 baseline, parked with stage1_index_between above.  These
    # never paid in the 105D open-hand runs anyway: their ramp needs all-joint
    # q-RMSE below 0.80 rad and those runs plateaued at 0.87 rad, so
    # Episode_Reward for both read exactly 0.0000 for the whole run.
    # stage2_contact_mean = RewTerm(
    #     func=hand_grasp_mdp.Stage2ContactGroupStrength,
    #     weight=5.0,
    #     params={
    #         "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
    #         "force_scale": 0.10,
    #         "reduction": "mean",
    #         **SETTING_STAGE2_PARAMS,
    #     },
    # )
    # stage2_contact_min = RewTerm(
    #     func=hand_grasp_mdp.Stage2ContactGroupStrength,
    #     weight=20.0,
    #     params={
    #         "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
    #         "force_scale": 0.10,
    #         "reduction": "min",
    #         **SETTING_STAGE2_PARAMS,
    #     },
    # )
    #success = RewTerm(
    #    func=hand_grasp_mdp.FunctionalSettingHeld,
    #    weight=30000,
    #    params=SETTING_SUCCESS_PARAMS,
    #)

    # Inactive reward terms are deliberately commented out instead of
    # registering them with weight=0.  This keeps the current Reward Manager
    # honest.  Keep the definitions below as a ready-to-restore archive for
    # later comparison stages.
    #
    # joint_reference = RewTerm(
    #     func=hand_grasp_mdp.JointReferenceTracking,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": HAND_JOINTS,
    #         "reference_joint_positions": PREGRASP_JOINT_POSITIONS,
    #         "sigma": 0.80,
    #     },
    # )
    # stick1_reference_pose = RewTerm(
    #     func=hand_grasp_mdp.ObjectReferencePoseTracking,
    #     weight=0.0,
    #     params={
    #         "palm_cfg": PALM,
    #         "object_cfg": STICK_1,
    #         "reference_position_p": PREGRASP_STICK1_POSITION_P,
    #         "reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
    #         "position_sigma": 0.10,
    #         "orientation_sigma": 1.5707963268,
    #     },
    # )
    # valley_anchor_support = RewTerm(
    #     func=hand_grasp_mdp.stick2_pose_anchor_support_strength,
    #     weight=0.0,
    #     params={
    #         **STICK2_POSE_SUPPORT_PARAMS,
    #         "force_scale": 0.10,
    #         "position_sigma": 0.10,
    #         "orientation_sigma": 1.5707963268,
    #     },
    # )
    # thumb_distal_region = _region_term(THUMB_DISTAL, STICK_1, weight=0.0)
    # index_tip_region = _valley_region_term(INDEX_TIP, STICK_1, weight=0.0)
    # middle_tip_region = _valley_region_term(MIDDLE_TIP, STICK_1, weight=0.0)
    # ring_tip_region = _region_term(RING_TIP, STICK_2, weight=0.0)
    # thumb_distal_contact = _valley_contact_term(
    #     (FUNCTIONAL_CONTACT_GROUPS[0],), weight=0.0
    # )
    # index_tip_contact = _valley_contact_term(
    #     (FUNCTIONAL_CONTACT_GROUPS[1],), weight=0.0
    # )
    # middle_tip_contact = _valley_contact_term(
    #     (FUNCTIONAL_CONTACT_GROUPS[2],), weight=0.0
    # )
    # ring_tip_contact = _contact_term(
    #     FUNCTIONAL_CONTACT_GROUPS[5], weight=0.0
    # )
    # functional_contact_min = _valley_contact_term(
    #     FUNCTIONAL_CONTACT_GROUPS, weight=0.0, reduction="min"
    # )
    # setting_completion = RewTerm(
    #     func=hand_grasp_mdp.setting_completion_strength,
    #     weight=0.0,
    #     params={
    #         **SETTING_COMPLETION_PARAMS,
    #         "force_scale": 0.10,
    #         "position_sigma": 0.02,
    #         "orientation_sigma": 0.3490658504,
    #     },
    # )
    # setting_stability = RewTerm(
    #     func=hand_grasp_mdp.setting_grasp_stability,
    #     weight=0.0,
    #     params={
    #         **SETTING_COMPLETION_PARAMS,
    #         "force_scale": 0.10,
    #         "position_sigma": 0.02,
    #         "orientation_sigma": 0.3490658504,
    #         "linear_speed_scale": 0.10,
    #         "angular_speed_scale": 2.0,
    #     },
    # )
    # success = RewTerm(
    #     func=hand_grasp_mdp.FunctionalSettingHeld,
    #     weight=0.0,
    #     params=SETTING_SUCCESS_PARAMS,
    # )
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=0.0)


@configclass
class HandSettingTerminationsCfg:
    """Terminate on timeout or a physically dropped stick."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    stick1_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.40, "asset_cfg": STICK_1},
    )
    stick2_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.40, "asset_cfg": STICK_2},
    )
    # Final-setting success is diagnostic during the current coarse/fine pose
    # acquisition stage.  Do not end the episode without an explicit success
    # reward: that would shorten collection of the remaining positive shaping
    # rewards and can make success less attractive.  Preserve this block for a
    # later strict validation stage.
    # success = DoneTerm(
    #     func=hand_grasp_mdp.FunctionalSettingHeld,
    #     params=SETTING_SUCCESS_PARAMS,
    # )


@configclass
class HandSettingEnvCfg(HandGraspEnvCfg):
    """Open hand → 4 mm-clearance q_ref → functional grasp."""

    # Metrics use the same geometric/contact contract as the parked success
    # termination without requiring that termination to remain active.
    hand_setting_metric_params: dict[str, object] = SETTING_SUCCESS_PARAMS

    # The five setting-only Index--Stick2 sensors are reward-only privileged
    # inputs.  They do not enter the 105D actor observation or alter contacts.
    scene: HandSettingSceneCfg = HandSettingSceneCfg(
        num_envs=4096,
        env_spacing=1.0,
    )
    observations: HandSettingObservationsCfg = HandSettingObservationsCfg()
    actions: HandSettingActionsCfg = HandSettingActionsCfg()
    commands = EmptyCfg()
    rewards: HandSettingRewardsCfg = HandSettingRewardsCfg()
    terminations: HandSettingTerminationsCfg = HandSettingTerminationsCfg()
    events: HandSettingEventCfg = HandSettingEventCfg()

    def __post_init__(self):
        """Apply shared hand physics, then set the single-transition horizon."""

        super().__post_init__()
        # The setting policy is the predecessor of hand_real.  Keep the full
        # controllable joint range, residual step sizes, Kp/Kd, and effort caps
        # identical so the learned transition does not rely on different hand
        # physics before handoff.
        apply_hand_real_actuator_contract(self)
        # Keep the task-local stick placement that leaves the open thumb free
        # to reach its pivot before the remaining fingers close toward q_ref.
        self.scene.stick1 = self.scene.stick1.replace(
            init_state=self.scene.stick1.init_state.replace(
                pos=SETTING_STICK1_RESET_POS
            )
        )
        self.scene.stick2 = self.scene.stick2.replace(
            init_state=self.scene.stick2.init_state.replace(
                pos=SETTING_STICK2_RESET_POS
            )
        )
        # Keep the successful hand_grasp physics/action setup, but allow the
        # one-shot setting transition slightly more time than a 10 s mode.
        self.episode_length_s = 8.0
