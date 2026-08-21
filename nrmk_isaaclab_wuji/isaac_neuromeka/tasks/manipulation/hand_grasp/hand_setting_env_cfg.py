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
    """Task-local current-joint residual action for stable setting control."""

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
        # Signed residuals may extend Joint4 toward zero, but never command a
        # distal target below the neutral pose.
        joint_position_lower_overrides={
            "finger1_joint4": 0.0,
            "finger2_joint4": 0.0,
            "finger3_joint4": 0.0,
            "finger4_joint4": 0.0,
            "finger5_joint4": 0.0,
        },
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
    "pair_score_threshold": 0.65,
    "thumb_score_threshold": 0.35,
}
# Strict Stage-2 readiness remains a memoryless diagnostic: Stage 1 must be
# valid and every joint must be within fifteen degrees of pose_005.  Contact
# shaping below uses a one-way per-episode Stage-1 latch so transient stick
# motion during finger closure does not erase the next-stage learning signal.
SETTING_STAGE2_JOINT_ERROR_THRESHOLD = 0.2617993878
# Contact shaping begins weakly at 0.8 rad all-joint RMSE and reaches full
# strength at the 15-degree threshold.  The strict Stage-2 readiness
# metric remains the memoryless maximum-error test above.
SETTING_STAGE2_CONTACT_START_RMSE = 0.80
SETTING_SEMANTIC_APPROACH_RANGE = 0.08
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
    """Reset the dynamic sticks and hand without injecting contact force."""

    # HandSettingEnvCfg sets task-local absolute positions x=0.075/0.055, so
    # reset restores the validated parallel dynamic sticks with their 20 mm
    # separation and zero velocity.
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
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
    """Acquire pose_005, then hand off smoothly to semantic contacts."""

    two_stick_reference_min = RewTerm(
        func=hand_grasp_mdp.ObjectPairReferencePoseMinTracking,
        # Coarse acquisition/maintenance kernel.  Keep this broad enough to
        # produce a useful signal before either stick reaches the fine basin.
        weight=12.0,
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
        weight=8.0,
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
        weight=8.0,
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
    stage1_missing_joint_reference = RewTerm(
        func=hand_grasp_mdp.stage1_gated_joint_reference_mean_min,
        # Preserve the existing mean/min exponential guide for the sixteen
        # non-thumb joints as the closer-range reference reward.
        weight=12.0,
        params={
            "asset_cfg": SETTING_MISSING_FINGER_JOINTS,
            "reference_joint_positions": (
                SETTING_MISSING_FINGER_JOINT_POSITIONS
            ),
            "joint_sigma": 0.80,
            "stage2_asset_cfg": HAND_JOINTS,
            "stage2_reference_joint_positions": PREGRASP_JOINT_POSITIONS,
            "stage2_joint_error_threshold": (
                SETTING_STAGE2_JOINT_ERROR_THRESHOLD
            ),
            # This strong acquisition guide disappears while Stage 2 is valid
            # and automatically returns if the current 15-degree gate fails.
            "stage2_reference_weight_ratio": 0.0,
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
    stage1_missing_joint_best_so_far = RewTerm(
        func=hand_grasp_mdp.Stage1MissingJointBestSoFar,
        # Each of the sixteen non-thumb joints owns an independent best score.
        # Their normalized weighted progress still pays at most one score unit
        # per episode. At 30 Hz, weight 3000 therefore preserves the previous
        # episode-level maximum budget of about 100 reward points.
        weight=3000.0,
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
    # Parked for the contact-topology A/B.  Joint reference plus exact contact
    # rewards should determine the route without prescribing broad surfaces.
    # stage1_semantic_surface_approach = RewTerm(
    #     func=hand_grasp_mdp.Stage1SemanticSurfaceApproach,
    #     weight=2.0,
    #     params={
    #         "index_cfg": INDEX_TIP,
    #         "middle_cfg": MIDDLE_TIP,
    #         "ring_cfg": RING_TIP,
    #         "stick1_cfg": STICK_1,
    #         "stick2_cfg": STICK_2,
    #         "stick_half_extent": STICK_HALF_EXTENT,
    #         "approach_range": SETTING_SEMANTIC_APPROACH_RANGE,
    #         **SETTING_STAGE1_GATE_PARAMS,
    #     },
    # )
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
    # Parked in the same A/B: do not prescribe an inter-stick Index route.
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
    index_wrong_stick2_contact = RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=-2.0,
        params={
            "sensor_groups": INDEX_WRONG_STICK2_SENSOR_GROUPS,
            "force_scale": 0.10,
            "reduction": "mean",
        },
    )
    stage2_contact_mean = RewTerm(
        func=hand_grasp_mdp.Stage2ContactGroupStrength,
        # Half of hand_grasp's final contact weights.  The RMSE progress gate
        # makes this much weaker still while q is far from pose_005.
        weight=5.0,
        params={
            "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "force_scale": 0.10,
            "reduction": "mean",
            **SETTING_STAGE2_PARAMS,
        },
    )
    stage2_contact_min = RewTerm(
        func=hand_grasp_mdp.Stage2ContactGroupStrength,
        weight=20.0,
        params={
            "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "force_scale": 0.10,
            "reduction": "min",
            **SETTING_STAGE2_PARAMS,
        },
    )
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
    """Open hand → q_ref acquisition → functional-contact grasp."""

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
        # Do not alter HandGraspSceneCfg globally.  These task-local absolute
        # positions leave the open thumb free to reach its pivot before
        # contacting Stick2.
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
