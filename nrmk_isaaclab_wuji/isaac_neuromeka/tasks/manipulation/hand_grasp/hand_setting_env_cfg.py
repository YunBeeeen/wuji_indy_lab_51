"""Single-phase learning from an open hand to a functional chopstick setting."""

from __future__ import annotations

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp
from isaac_neuromeka.utils.etc import EmptyCfg

from . import mdp as hand_grasp_mdp
from .hand_grasp_env_cfg import (
    FINGERTIPS,
    FUNCTIONAL_CONTACT_GROUPS,
    HAND_JOINTS,
    PALM,
    PREGRASP_JOINT_POSITIONS,
    PREGRASP_STICK1_POSITION_P,
    PREGRASP_STICK1_QUATERNION_P,
    PREGRASP_STICK2_POSITION_P,
    PREGRASP_STICK2_QUATERNION_P,
    STICK_1,
    STICK_2,
    STICK_SIZE,
    HandGraspActionsCfg,
    HandGraspEnvCfg,
    HandGraspSceneCfg,
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

STICK_HALF_EXTENT = tuple(0.5 * value for value in STICK_SIZE)
SETTING_LONG_AXIS = 1
# Keep semantic contacts on the central 160 mm of the 180 mm shaft.  This
# rejects end-cap reward farming without forcing one exact contact point.
SETTING_AXIAL_HALF_LENGTH = 0.08

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


@configclass
class HandSettingObservationsCfg:
    """101D state; there is no command or hidden phase variable.

    20 joint positions + 20 joint velocities + 15 fingertip positions
    + 14 stick poses + 12 palm-relative stick velocities + 20 previous actions.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor/critic observation group shared by all environments."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": HAND_JOINTS},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": HAND_JOINTS},
            scale=0.2,
        )
        fingertip_pos = ObsTerm(
            func=hand_grasp_mdp.fingertip_positions_in_palm,
            params={"palm_cfg": PALM, "fingertip_cfg": FINGERTIPS},
        )
        stick1_pose = ObsTerm(
            func=hand_grasp_mdp.object_pose_in_palm,
            params={"palm_cfg": PALM, "object_cfg": STICK_1},
        )
        stick2_pose = ObsTerm(
            func=hand_grasp_mdp.object_pose_in_palm,
            params={"palm_cfg": PALM, "object_cfg": STICK_2},
        )
        stick1_velocity = ObsTerm(
            func=hand_grasp_mdp.object_velocity_in_palm,
            params={"palm_cfg": PALM, "object_cfg": STICK_1},
            scale=0.2,
        )
        stick2_velocity = ObsTerm(
            func=hand_grasp_mdp.object_velocity_in_palm,
            params={"palm_cfg": PALM, "object_cfg": STICK_2},
            scale=0.2,
        )
        action_history = ObsTerm(func=mdp.action_history)

        def __post_init__(self):
            """Concatenate the terms in the documented 101D order."""

            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HandSettingEventCfg:
    """Reset the dynamic sticks and hand without injecting contact force."""

    # The scene defaults restore two dynamic, parallel sticks at world
    # x=0.055/0.035, y=0, z=0.5195 with zero velocity.
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
) -> RewTerm:
    """Create broad central-shaft approach shaping for one semantic link."""

    return RewTerm(
        func=hand_grasp_mdp.body_box_shaft_region_proximity,
        weight=2.0,
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


def _contact_term(sensor_group: tuple[str, ...]) -> RewTerm:
    """Expose one saturated semantic contact as an independent reward tag."""

    # Six terms with weight 5/6 are exactly equivalent to the former
    # weight-5 mean, while TensorBoard now shows which semantic contact is
    # missing instead of only the aggregate.
    return RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=5.0 / 6.0,
        params={
            "sensor_groups": (sensor_group,),
            "force_scale": 0.10,
            "reduction": "mean",
        },
    )


@configclass
class HandSettingRewardsCfg:
    """One continuous reward ladder from approach to stable six-contact grasp."""

    # Weak posture prior only.  The contact/pose terms remain free to find a
    # nearby solution instead of copying pose_005 joint-for-joint.
    joint_reference = RewTerm(
        func=hand_grasp_mdp.JointReferenceTracking,
        weight=2.0,
        params={
            "asset_cfg": HAND_JOINTS,
            "reference_joint_positions": PREGRASP_JOINT_POSITIONS,
            # At the open reset sigma=0.30 produced only ~2.1e-4 raw reward.
            # 0.80 leaves a useful weak signal (~0.30) without fixing q exactly.
            "sigma": 0.80,
        },
    )
    stick1_reference_pose = RewTerm(
        func=hand_grasp_mdp.ObjectReferencePoseTracking,
        weight=8.0,
        params={
            "palm_cfg": PALM,
            "object_cfg": STICK_1,
            "reference_position_p": PREGRASP_STICK1_POSITION_P,
            "reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
            "position_sigma": 0.10,
            "orientation_sigma": 1.5707963268,
        },
    )
    stick2_reference_pose = RewTerm(
        func=hand_grasp_mdp.ObjectReferencePoseTracking,
        weight=12.0,
        params={
            "palm_cfg": PALM,
            "object_cfg": STICK_2,
            "reference_position_p": PREGRASP_STICK2_POSITION_P,
            "reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
            "position_sigma": 0.10,
            "orientation_sigma": 1.5707963268,
        },
    )

    # Dense approach signals use the correct link-object pair and a broad
    # central shaft region.  No palm/opening normal is used.
    thumb_distal_region = _region_term(THUMB_DISTAL, STICK_1)
    index_tip_region = _region_term(INDEX_TIP, STICK_1)
    middle_tip_region = _region_term(MIDDLE_TIP, STICK_1)
    ring_tip_region = _region_term(RING_TIP, STICK_2)

    # Their sum is weight-5 mean contact shaping.  Splitting the terms exposes
    # every force channel in Episode_Reward_Raw without changing the objective.
    thumb_distal_contact = _contact_term(FUNCTIONAL_CONTACT_GROUPS[0])
    index_tip_contact = _contact_term(FUNCTIONAL_CONTACT_GROUPS[1])
    middle_tip_contact = _contact_term(FUNCTIONAL_CONTACT_GROUPS[2])
    palm_anchor_contact = _contact_term(FUNCTIONAL_CONTACT_GROUPS[3])
    thumb_mid_anchor_contact = _contact_term(FUNCTIONAL_CONTACT_GROUPS[4])
    ring_tip_contact = _contact_term(FUNCTIONAL_CONTACT_GROUPS[5])
    # Hard min is the completion pressure: one missing group makes it zero.
    functional_contact_min = RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=20.0,
        params={
            "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "force_scale": 0.10,
            "reduction": "min",
        },
    )
    setting_completion = RewTerm(
        func=hand_grasp_mdp.setting_completion_strength,
        weight=30.0,
        params={
            **SETTING_COMPLETION_PARAMS,
            "force_scale": 0.10,
            "position_sigma": 0.02,
            "orientation_sigma": 0.3490658504,
        },
    )
    setting_stability = RewTerm(
        func=hand_grasp_mdp.setting_grasp_stability,
        weight=50.0,
        params={
            **SETTING_COMPLETION_PARAMS,
            "force_scale": 0.10,
            "position_sigma": 0.02,
            "orientation_sigma": 0.3490658504,
            "linear_speed_scale": 0.10,
            "angular_speed_scale": 2.0,
        },
    )
    success = RewTerm(
        func=hand_grasp_mdp.FunctionalSettingHeld,
        weight=30000.0,
        params=SETTING_SUCCESS_PARAMS,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)


@configclass
class HandSettingTerminationsCfg:
    """Terminate on timeout, either dropped stick, or a held final setting."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    stick1_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.40, "asset_cfg": STICK_1},
    )
    stick2_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.40, "asset_cfg": STICK_2},
    )
    success = DoneTerm(
        func=hand_grasp_mdp.FunctionalSettingHeld,
        params=SETTING_SUCCESS_PARAMS,
    )


@configclass
class HandSettingEnvCfg(HandGraspEnvCfg):
    """Open hand + aligned dynamic sticks → validated functional grasp."""

    scene: HandGraspSceneCfg = HandGraspSceneCfg(
        num_envs=4096,
        env_spacing=1.0,
    )
    observations: HandSettingObservationsCfg = HandSettingObservationsCfg()
    actions: HandGraspActionsCfg = HandGraspActionsCfg()
    commands = EmptyCfg()
    rewards: HandSettingRewardsCfg = HandSettingRewardsCfg()
    terminations: HandSettingTerminationsCfg = HandSettingTerminationsCfg()
    events: HandSettingEventCfg = HandSettingEventCfg()

    def __post_init__(self):
        """Apply shared hand physics, then set the single-transition horizon."""

        super().__post_init__()
        # Keep the successful hand_grasp physics/action setup, but allow the
        # one-shot setting transition slightly more time than a 10 s mode.
        self.episode_length_s = 15.0
