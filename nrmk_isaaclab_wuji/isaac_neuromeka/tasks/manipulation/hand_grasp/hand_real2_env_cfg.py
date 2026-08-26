"""Thumb-anchored acquisition variant of :mod:`hand_real_env_cfg`.

``hand_real`` remains the established pose_005 reset task.  ``hand_real2`` now
keeps that same recorded pregrasp reset while testing a stricter functional
contact topology.  The joint and stick references, 105D observation, action, reward functions and
weights, actuator contract, command implementations, disturbance implementation,
and terminations remain inherited.  Task-local overrides own the command
schedule and disturbance parameters.  Index requires both tip and distal
actuated link4 contact; Middle and Ring are tip-only.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import hand_move_mdp, hand_real2_mdp
from . import mdp as hand_grasp_mdp
from .hand_grasp_env_cfg import (
    HAND_JOINTS,
    PALM,
    PREGRASP_JOINT_POSITIONS,
    PREGRASP_STICK1_POSITION_P,
    PREGRASP_STICK1_QUATERNION_P,
    PREGRASP_STICK2_POSITION_P,
    PREGRASP_STICK2_QUATERNION_P,
    STICK_1,
    STICK_2,
)
from .hand_real_env_cfg import (
    HAND_REAL_PREV_JOINT_TARGET,
    HandRealCommandsCfg,
    HandRealEnvCfg,
)
from .hand_move_env_cfg import HandMoveEventCfg, HandMoveRewardsCfg, HandMoveSceneCfg


HAND_REAL2_FUNCTIONAL_CONTACT_GROUPS = (
    ("thumb_distal_stick1",),
    ("index_tip_stick1", "index_link4_stick1"),
    ("middle_tip_stick1",),
    ("palm_stick2",),
    ("thumb_mid_stick2",),
    ("ring_tip_stick2",),
)

HAND_REAL2_LINK4_ACQUISITION_GROUPS = (
    ("index_link4_stick1",),
    ("ring_link4_stick2",),
)

HAND_REAL2_RING_LITTLE_SUPPORT_GROUPS = (
    ("little_tip_ring_tip",),
)

HAND_REAL2_RING_WRONG_STICK1_SENSOR_NAMES = (
    "ring_link1_stick1_wrong",
    "ring_link2_stick1_wrong",
    "ring_link3_stick1_wrong",
    "ring_link4_stick1_wrong",
    "ring_tip_stick1_wrong",
)
HAND_REAL2_RING_WRONG_STICK1_GROUPS = (
    HAND_REAL2_RING_WRONG_STICK1_SENSOR_NAMES,
)

HAND_REAL2_LITTLE_WRONG_STICK2_SENSOR_NAMES = (
    "little_link1_stick2_wrong",
    "little_link2_stick2_wrong",
    "little_link3_stick2_wrong",
    "little_link4_stick2_wrong",
    "little_tip_stick2_wrong",
)
HAND_REAL2_LITTLE_WRONG_STICK2_GROUPS = (
    HAND_REAL2_LITTLE_WRONG_STICK2_SENSOR_NAMES,
)

# Keep the current target A/B unchanged while hand_real returns to model_4500:
# recorded pose_005 with Index/Middle J1 flexed further.  This task-local tuple
# prevents future hand_real target rollbacks from changing hand_real2 again.
HAND_REAL2_JOINT_REFERENCE = (
    *HAND_REAL_PREV_JOINT_TARGET[:4],
    0.8217297745,  # Index J1: pose_005 + 0.12 rad.
    *HAND_REAL_PREV_JOINT_TARGET[5:8],
    0.5649881423,  # Middle J1: pose_005 + 0.10 rad.
    *HAND_REAL_PREV_JOINT_TARGET[9:],
)


@configclass
class HandReal2CommandsCfg(HandRealCommandsCfg):
    """hand_real command semantics driven by the hand_real2-local schedule."""

    root_orientation = hand_move_mdp.HandMoveRootOrientationCommandCfg(
        schedule=hand_real2_mdp.HAND_REAL2_SCHEDULE,
    )
    open_close = hand_move_mdp.HandMoveOpenCloseCommandCfg(
        schedule=hand_real2_mdp.HAND_REAL2_SCHEDULE,
        neutral_before_open_close=True,
    )


@configclass
class HandReal2EventCfg(HandMoveEventCfg):
    """Inherited reset with task-local disturbance curriculum knobs."""

    reset_pregrasp = EventTerm(
        func=hand_real2_mdp.reset_to_noisy_functional_pregrasp,
        mode="reset",
        params={
            "hand_cfg": HAND_JOINTS,
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "joint_positions": PREGRASP_JOINT_POSITIONS,
            "stick1_position_p": PREGRASP_STICK1_POSITION_P,
            "stick1_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
            "stick2_position_p": PREGRASP_STICK2_POSITION_P,
            "stick2_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
            "position_noise_m": hand_real2_mdp.HAND_REAL2_STICK_POSITION_NOISE_M,
            "orientation_noise_rad": hand_real2_mdp.HAND_REAL2_STICK_ORIENTATION_NOISE_RAD,
            "probability": hand_real2_mdp.HAND_REAL2_STICK_RESET_NOISE_PROBABILITY,
        },
    )

    stick_disturbance = EventTerm(
        func=hand_move_mdp.StickDisturbance,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=False,
        params={
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "sensor_groups": HAND_REAL2_FUNCTIONAL_CONTACT_GROUPS,
            "group_reduction": "min",
            "contact_threshold": 0.02,
            "time_range_s": hand_real2_mdp.HAND_REAL2_DISTURBANCE_TIME_RANGE_S,
            "duration_s": hand_real2_mdp.HAND_REAL2_DISTURBANCE_DURATION_S,
            "force_range_n": hand_real2_mdp.HAND_REAL2_DISTURBANCE_FORCE_RANGE_N,
            "probability": hand_real2_mdp.HAND_REAL2_DISTURBANCE_PROBABILITY,
            "application_offset_o": (
                hand_real2_mdp.HAND_REAL2_DISTURBANCE_APPLICATION_OFFSET_O
            ),
        },
    )


@configclass
class HandReal2RewardsCfg(HandMoveRewardsCfg):
    """Add a hard six-group completion signal without changing hand_real."""

    # Give each missing link4 independent credit.  At force_scale=0.10 N,
    # either link4 alone can earn half of this term and both earn the full term.
    link4_acquisition = RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=20.0,
        params={
            "sensor_groups": HAND_REAL2_LINK4_ACQUISITION_GROUPS,
            "force_scale": 0.10,
            "reduction": "mean",
        },
    )

    functional_contact_complete = RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=20.0,
        params={
            "sensor_groups": HAND_REAL2_FUNCTIONAL_CONTACT_GROUPS,
            "force_scale": 0.10,
            "reduction": "min",
            "group_reduction": "min",
        },
    )

    full_contact_hold = RewTerm(
        func=hand_grasp_mdp.full_contact_bonus,
        weight=20.0,
        params={
            "sensor_groups": HAND_REAL2_FUNCTIONAL_CONTACT_GROUPS,
            "contact_threshold": 0.02,
            "group_reduction": "min",
        },
    )

    # Weak auxiliary support only.  It is deliberately excluded from the six
    # functional groups, success, full-contact hold, and contact-loss logic.
    ring_little_tip_support = RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=2.0,
        params={
            "sensor_groups": HAND_REAL2_RING_LITTLE_SUPPORT_GROUPS,
            "force_scale": 0.10,
            "reduction": "mean",
        },
    )

    # Select the intended topology after approach: no part of Ring should
    # support Stick1.  Max-over-links means any wrong Ring--Stick1 contact is
    # visible, while the small bounded weight avoids dominating acquisition.
    ring_wrong_stick1_contact = RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=-5.0,
        params={
            "sensor_groups": HAND_REAL2_RING_WRONG_STICK1_GROUPS,
            "force_scale": 0.10,
            "reduction": "mean",
        },
    )

    # Little should support Ring, not replace Ring by directly carrying
    # Stick2.  Max-over-links exposes any such shortcut while the small,
    # bounded weight keeps the auxiliary Ring--Little support signal usable.
    little_wrong_stick2_contact = RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=-2.0,
        params={
            "sensor_groups": HAND_REAL2_LITTLE_WRONG_STICK2_GROUPS,
            "force_scale": 0.10,
            "reduction": "mean",
        },
    )


@configclass
class HandReal2SceneCfg(HandMoveSceneCfg):
    """Require tip+link4 for Index and tip-only for Middle/Ring."""

    # Preserve the historical sensor keys so every inherited reward and metric
    # remains wired identically.  Only the monitored collision body differs
    # from hand_real/hand_move.
    index_tip_stick1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger2_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    index_link4_stick1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger2_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    middle_tip_stick1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger3_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    middle_link4_stick1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger3_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    ring_tip_stick2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger4_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    ring_link4_stick2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger4_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    little_tip_ring_tip = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger5_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/finger4_tip_link"],
        update_period=0.0,
    )
    ring_link1_stick1_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger4_link1",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    ring_link2_stick1_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger4_link2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    ring_link3_stick1_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger4_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    ring_link4_stick1_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger4_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    ring_tip_stick1_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger4_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    little_link1_stick2_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger5_link1",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    little_link2_stick2_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger5_link2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    little_link3_stick2_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger5_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    little_link4_stick2_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger5_link4",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    little_tip_stick2_wrong = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger5_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )

# Parked 10 mm acquisition reset.  The active hand_real2 reset is again the
# inherited pose_005 pregrasp; Thumb in this parked candidate is exact pose_005.
# HAND_REAL2_ACQUISITION_RESET_JOINT_POSITIONS = (
#     0.5377866626, 0.8436813951, 0.0377136655, -0.0000001810,
#     0.7640267118, 0.2402315297, 1.0542420894, 1.3523163263,
#     0.3177517346, 0.0630920132, 1.5268040966, 1.0945922575,
#     0.7820721520, 0.0735689166, 1.3201216836, 0.3228168438,
#     0.8608684063, -0.0609072480, 1.6258423449, 0.2498187101,
# )

# Parked 15 mm acquisition reset (previous hand_real2 A/B).  Thumb was also
# exact pose_005; only the four non-thumb fingers differed from the active
# 10 mm reset above.
# HAND_REAL2_ACQUISITION_RESET_JOINT_POSITIONS = (
#     0.5377866626, 0.8436813951, 0.0377136655, -0.0000001810,
#     0.7986974050, 0.3165175430, 0.9751217095, 1.3052689966,
#     0.2591424098, 0.1003195813, 1.4607863573, 1.0788861118,
#     0.7232182992, 0.1150754580, 1.3007612181, 0.3188584974,
#     0.9376087315, -0.1281184761, 1.6074278879, 0.2422029568,
# )


@configclass
class HandReal2EnvCfg(HandRealEnvCfg):
    """Deploy-compatible hand_real with task-local acquisition curriculum."""

    # Privileged diagnostics only: no actor observation, reward, termination,
    # or physical collision property depends on these sensors.
    scene: HandReal2SceneCfg = HandReal2SceneCfg(
        num_envs=4096,
        env_spacing=1.0,
    )
    commands: HandReal2CommandsCfg = HandReal2CommandsCfg()
    events: HandReal2EventCfg = HandReal2EventCfg()
    rewards: HandReal2RewardsCfg = HandReal2RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # HandRealEnvCfg initializes from HAND_MOVE_SCHEDULE.  Reassert the
        # task-local default so source-level schedule edits stay isolated.
        hand_real2_mdp.HAND_REAL2_SCHEDULE.validate()
        self.episode_length_s = hand_real2_mdp.HAND_REAL2_SCHEDULE.episode_length_s
        # hand_real2 keeps six semantic contacts.  Index uses min(tip, link4),
        # while Middle and Ring are tip-only.  Every inherited consumer must
        # see the same group list and reduction.
        self.events.stick_disturbance.params["sensor_groups"] = (
            HAND_REAL2_FUNCTIONAL_CONTACT_GROUPS
        )
        self.events.stick_disturbance.params["group_reduction"] = "min"
        self.events.stick_disturbance.params["probability"] = (
            hand_real2_mdp.HAND_REAL2_DISTURBANCE_PROBABILITY
        )
        self.rewards.functional_contact_min.params["sensor_groups"] = (
            HAND_REAL2_FUNCTIONAL_CONTACT_GROUPS
        )
        self.rewards.joint_reference.params["reference_joint_positions"] = (
            HAND_REAL2_JOINT_REFERENCE
        )
        self.rewards.joint_reference.params["deactivate_sensor_groups"] = (
            HAND_REAL2_FUNCTIONAL_CONTACT_GROUPS
        )
        self.rewards.joint_reference.params["deactivate_contact_threshold"] = 0.02
        self.rewards.joint_reference.params["deactivate_group_reduction"] = "min"
        # Acquisition shaping: Index link4 remains part of the hard topology;
        # Ring link4 is only a soft preference.  Ring can therefore complete
        # its semantic contact through the tip even when link4 is unloaded.
        self.rewards.functional_contact_min.params["group_reduction"] = (
            "partial_and_bonus"
        )
        self.rewards.functional_contact_min.params["reduction"] = "mean"
        # The additional completion term restores the original weakest-contact
        # pressure and pays only as all six hard-AND groups become loaded.
        self.rewards.functional_contact_complete.params["sensor_groups"] = (
            HAND_REAL2_FUNCTIONAL_CONTACT_GROUPS
        )
        self.rewards.mode_grasp_stability.params["sensor_groups"] = (
            HAND_REAL2_FUNCTIONAL_CONTACT_GROUPS
        )
        self.rewards.mode_grasp_stability.params["group_reduction"] = "min"
        self.rewards.success.params["sensor_groups"] = (
            HAND_REAL2_FUNCTIONAL_CONTACT_GROUPS
        )
        self.rewards.success.params["group_reduction"] = "min"
        self.terminations.success.params["sensor_groups"] = (
            HAND_REAL2_FUNCTIONAL_CONTACT_GROUPS
        )
        self.terminations.success.params["group_reduction"] = "min"
        # The 4 mm-clearance joint reset and noisy stick poses remain owned by
        # HandReal2EventCfg; hand_real's pose_005 rollback does not affect them.
