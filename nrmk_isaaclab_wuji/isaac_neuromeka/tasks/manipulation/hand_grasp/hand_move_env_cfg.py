"""Floating-root hand rotation on top of the ``hand_grasp`` grasp (2026-08-05).

``hand_move`` starts each episode from the validated ``hand_grasp`` functional
chopsticks grasp, rotates the *floating* Wuji hand root to a randomly sampled
goal orientation, and then runs the OPEN/CLOSE sequence at that orientation.
The finger policy has to keep the functional grasp through the rotation and
still open and close afterwards.

Episode script (15 s, all boundaries derived from ``HAND_MOVE_SCHEDULE``):

===========  ==================================  ==========
time         root orientation command            OPEN/CLOSE
===========  ==================================  ==========
``[0, 2)``   ``q_start`` (hold, stabilise)       OPEN
``[2, 4)``   ``SLERP(q_start, q_goal, alpha)``   OPEN
``[4, 5)``   ``q_goal`` (settling)               OPEN
``[5, 15)``  ``q_goal``                          5 x 2 s alternating
===========  ==================================  ==========

Only the SLERP finishes at 4 s.  ``q_cmd`` stays pinned at ``q_goal`` and the
root PD controller plus its PhysX wrench write keep running to the end of the
episode.

What the policy controls: a 20D current-joint residual finger command, and
nothing else.  The root is moved by a scripted trajectory plus a wrench-based
PD controller - never by teleporting a pose or zeroing a velocity.  A fixed
``pose_005``-reference residual was tested and parked because it preserved the
grasp but failed to reach the 20 mm OPEN command.

Unchanged from ``hand_grasp``: stick geometry/mass/friction/contact settings,
the ``pose_005`` functional-grasp reset, the six contact sensors, the whole
103D observation, every reward term and weight, and the termination set.

Changed for this task, deliberately:
    * ``fix_root_link=False`` plus the root hold/tracking controller;
    * ``episode_length_s`` 10 s -> 15 s;
    * the OPEN/CLOSE command is a scripted schedule instead of ``hand_grasp``'s
      0.5 s random resampling;
    * an extra ``root_orientation`` command term (not observed by the policy).

Relationship to ``hand_grasp``
------------------------------
This module sits next to ``hand_setting_env_cfg.py`` and follows the same
pattern that task already uses: a task-local copy of every configuration
*class*, so ``hand_move`` can diverge without ever touching the baseline, while
the physical *constants* are imported from ``hand_grasp_env_cfg`` instead of
being re-typed.  Re-typing ``pose_005`` and the palm-frame stick references by
hand is the most likely way to make the two tasks silently differ.
``hand_grasp`` is a frozen baseline, so this import is stable; if it ever has
to change, copy the constants into this module first.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp
from isaac_neuromeka.assets import WUJI_RIGHT_CFG
from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg
from isaac_neuromeka.mdp.actions import (
    CustomResidualJointActionCfg,
    ReferenceResidualJointActionCfg,
)
from isaac_neuromeka.utils.etc import EmptyCfg

# Shared, unmodified MDP implementation of the baseline task.
from . import mdp as hand_grasp_mdp
from .hand_grasp_env_cfg import (
    CLOSE_TIP_GAP,
    FINGERTIPS,
    FUNCTIONAL_CONTACT_GROUPS,
    HAND_JOINT_NAMES,
    HAND_JOINTS,
    HAND_ROOT_POS,
    HAND_ROOT_ROT,
    MODE_HELD_PARAMS,
    PALM,
    PREGRASP_JOINT_POSITIONS,
    PREGRASP_STICK1_POSITION_P,
    PREGRASP_STICK1_QUATERNION_P,
    PREGRASP_STICK2_POSITION_P,
    PREGRASP_STICK2_QUATERNION_P,
    SMALL_CONTACT_OFFSET,
    SMALL_MAX_DEPENETRATION_VELOCITY,
    SMALL_REST_OFFSET,
    STICK1_PIVOT_OFFSET_O,
    STICK_1,
    STICK_2,
    STICK_ROT,
    STICK_SIZE,
    STICK_TIP_OFFSET_O,
    STICK_Z,
    TIP_AXIAL_OFFSET_STICK2,
    TIP_AXIAL_SIGMA,
    TIP_LATERAL_SIGMA,
    TIP_SEPARATION_DIRECTION_STICK2,
)
from . import hand_move_mdp
from .hand_move_mdp import HAND_MOVE_SCHEDULE
from .hand_move_root_actions import HandRootHoldActionCfg

# Task-local OPEN target from the successful
# hand_move/2026-08-08_09-29-27 model_6800 run.  Do not change the shared
# hand_grasp OPEN target when reproducing the hand_move baseline.
HAND_MOVE_OPEN_TARGET_GAP = 0.017

# Parked fixed-reference A/B scale.  Although +/-0.30 rad stabilized the grasp,
# run 2026-08-09_22-29-38 retained about 16 mm OPEN error (roughly 4 mm actual
# gap for a 20 mm command), so the range was not sufficient for this task.
HAND_MOVE_REFERENCE_RESIDUAL_SCALE = 0.30


def _stick_cfg(name: str, pos_x: float, color: tuple[float, float, float]) -> RigidObjectCfg:
    """Same 7 mm / 10 g chopstick proxy as ``hand_grasp``."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(pos_x, 0.0, STICK_Z),
            rot=STICK_ROT,
        ),
        spawn=sim_utils.CuboidCfg(
            size=STICK_SIZE,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=SMALL_MAX_DEPENETRATION_VELOCITY,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=SMALL_CONTACT_OFFSET,
                rest_offset=SMALL_REST_OFFSET,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                metallic=0.0,
                roughness=0.5,
            ),
        ),
    )


# Floating-base USD overlay, in the same directory as the collision-filtered
# one that ``WUJI_RIGHT_CFG`` uses.  Derived from that path so the two cannot
# drift apart if the asset ever moves.  See the header of the .usda for why a
# plain ``fix_root_link=False`` is not sufficient.
_WUJI_RIGHT_FLOATING_USD = os.path.join(
    os.path.dirname(WUJI_RIGHT_CFG.spawn.usd_path), "wuji_right_floating.usda"
)


@configclass
class HandMoveSceneCfg(InteractiveSceneCfg):
    """``HandGraspSceneCfg`` with the articulation root released from the world.

    Releasing the root needs *two* changes, not one.  The Wuji USD was imported
    from URDF with a fixed base, so it already contains a world fixed joint and
    - this is the part that bites - ``PhysicsArticulationRootAPI`` is applied to
    that **joint** prim rather than to a body::

        /wujihand_right_v1_0_2/root_joint | PhysicsFixedJoint
            body0: []  body1: [.../palm_link]
            appliedSchemas: ['PhysicsArticulationRootAPI']

    ``ArticulationRootPropertiesCfg.fix_root_link=False`` only sets that joint's
    ``physics:jointEnabled`` to false (``schemas.py:180-184``); it leaves the
    articulation root on the now-disabled joint.  PhysX then registers no
    articulation there and ``Articulation._initialize_impl`` raises
    "Failed to create articulation at: .../Robot/root_joint".

    So this task also swaps in ``wuji_right_floating.usda``, a non-destructive
    overlay that deactivates ``root_joint`` and moves the articulation root onto
    ``palm_link`` (the actual root rigid body), while still referencing the
    collision-filtered overlay.  ``fix_root_link=False`` is kept for intent: with
    the joint deactivated there is no world fixed joint left to find, so it is a
    no-op.

    ``WUJI_RIGHT_CFG``, the shared USD files and ``hand_grasp`` are untouched.
    Gravity is already disabled on the hand in ``WUJI_RIGHT_CFG``, so releasing
    the root does not make it fall.
    """

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    robot = WUJI_RIGHT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=WUJI_RIGHT_CFG.spawn.replace(
            # ★ Floating-base overlay instead of the fixed-base one.
            usd_path=_WUJI_RIGHT_FLOATING_USD,
            rigid_props=WUJI_RIGHT_CFG.spawn.rigid_props.replace(
                max_depenetration_velocity=SMALL_MAX_DEPENETRATION_VELOCITY,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=SMALL_CONTACT_OFFSET,
                rest_offset=SMALL_REST_OFFSET,
            ),
            articulation_props=WUJI_RIGHT_CFG.spawn.articulation_props.replace(
                solver_velocity_iteration_count=4,
                # ★ The only physical difference from hand_grasp.
                fix_root_link=False,
            ),
        ),
        init_state=WUJI_RIGHT_CFG.init_state.replace(
            pos=HAND_ROOT_POS,
            rot=HAND_ROOT_ROT,
        ),
    )
    stick1 = _stick_cfg("Stick1", 0.055, (0.95, 0.62, 0.12))
    stick2 = _stick_cfg("Stick2", 0.035, (0.95, 0.62, 0.12))
    #stick2 = _stick_cfg("Stick2", 0.035, (0.12, 0.72, 0.32))

    thumb_distal_stick1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger1_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    index_tip_stick1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger2_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    middle_tip_stick1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger3_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    ring_tip_stick2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger4_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    palm_stick2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/palm_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    thumb_mid_stick2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger1_link2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )


@configclass
class HandMoveActionsCfg:
    """20D current-joint residual, plus the scripted root controller.

    ``hand_action`` is declared first so its slice of the action vector stays
    ``[0:20]``, identical to ``hand_grasp``.  ``root_action`` reports
    ``action_dim = 0``, so the total action dimension remains 20: the root is
    driven by a scripted trajectory, never by the policy.
    """

    # Active baseline restored after the fixed-reference A/B below failed to
    # open: q_target = current_q + 0.1 * action.  It can move farther than
    # 0.1 rad over multiple policy steps while retaining joint-limit clamping.
    hand_action = CustomResidualJointActionCfg(
        asset_name="robot",
        joint_names=HAND_JOINT_NAMES,
        preserve_order=True,
        scale=0.1,
        clamp_to_limits=True,
        # 2026-08-17: the five Joint4 floors at 0.0 rad are lifted.  They were
        # cosmetic -- distal hyperextension looks unnatural for a human hand --
        # and the chopstick task is not expected to need that pose anyway.  But
        # play traces show the policy paying for them: finger1_joint4 requests a
        # negative residual on *every* sampled step in all three sessions
        # (2026-08-12 and current lineages alike), gets clamped to exactly 0.000
        # and therefore produces exactly 0.000 N.m, while thumb_distal_stick1 is
        # one of the six functional contacts.  A residual action can never
        # accumulate past a clamp, so that joint stayed inert for whole
        # episodes.  `clamp_to_limits` still bounds every target by the
        # articulation's own soft limits, which as of today are the official
        # per-joint values (Joint4 lower ~= -0.468 .. -0.478 rad).
        joint_position_lower_overrides=None,
    )

    # Parked paper-style A/B: q_target = q_pose005 + 0.3 * action.
    # It improved contact retention but converged to CLOSE in both modes:
    # around iteration 1000, final full-contact was ~96% while OPEN error was
    # still ~16 mm.  Keep the exact block for a future mode-conditioned
    # OPEN/CLOSE reference experiment; do not enable it in hand_grasp yet.
    # hand_action = ReferenceResidualJointActionCfg(
    #     asset_name="robot",
    #     joint_names=HAND_JOINT_NAMES,
    #     preserve_order=True,
    #     scale=HAND_MOVE_REFERENCE_RESIDUAL_SCALE,
    #     clamp_to_limits=True,
    #     reference_positions=PREGRASP_JOINT_POSITIONS,
    # )

    # Floating-root position hold + attitude tracking.  The attitude target is
    # the scripted ``q_cmd`` of the ``root_orientation`` command term, so no
    # angular-velocity integration takes place and nothing drifts on top of the
    # SLERP trajectory.  The controller and the wrench write run every physics
    # step for the whole 15 s episode.
    root_action = HandRootHoldActionCfg(
        asset_name="robot",
        rotation_action_dim=0,
        orientation_command_name="root_orientation",
    )


@configclass
class HandMoveCommandsCfg:
    """Scripted episode script: root orientation trajectory + OPEN/CLOSE.

    ``open_close`` keeps its name so the 2D one-hot observation term, the
    reward terms and the success/termination terms bind to it exactly as in
    ``hand_grasp``.  ``root_orientation`` is an extra term whose command is a
    quaternion; it is deliberately *not* part of the policy observation.
    """

    # hand_grasp's 0.5 s random resampling is intentionally NOT reused here.
    open_close = hand_move_mdp.HandMoveOpenCloseCommandCfg()

    root_orientation = hand_move_mdp.HandMoveRootOrientationCommandCfg()


@configclass
class HandMoveObservationsCfg:
    """103D policy state, identical to ``hand_grasp``.

    Dimension breakdown:
        20 normalized joint positions
      + 20 joint velocities
      + 15 palm-frame fingertip positions (5 fingertips x xyz)
      + 14 palm-frame stick poses (2 sticks x [xyz + quaternion])
      + 12 palm-frame relative stick velocities (2 sticks x [linear + angular])
      + 20 previous actions
      +  2 OPEN/CLOSE one-hot mode
      = 103.

    ``action_history`` mirrors ``action_manager.prev_action``.  The root is
    driven by a scripted controller and consumes no policy action, so it stays
    at 20 and the total stays at 103.

    Deliberately absent (this is the point of the first experiment): the root
    quaternion, the goal quaternion, the orientation error, the phase, the
    SLERP alpha and the elapsed time.  The question being asked is whether the
    existing palm-relative state alone is enough to survive a random root
    rotation.

    TODO(next iteration, only if this fails in the matching way):
      * if success depends strongly on *which* final orientation was sampled,
        add a 3D projected-gravity observation - the fingers currently cannot
        anticipate which way the sticks will be pulled;
      * if failures cluster *during* the rotation rather than after it, add a
        3D palm-frame root angular velocity observation.
    Do not pre-wire either of them as an inactive term.
    """

    @configclass
    class PolicyCfg(ObsGroup):
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
        open_close_mode = ObsTerm(
            func=hand_grasp_mdp.open_close_mode,
            params={"command_name": "open_close"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Chopstick disturbance (2026-08-07).  Additive experiment: nothing else in this
# task changes, so a run with these on and a run with them off is a clean A/B on
# the question "was never *experiencing* a displaced stick the problem?".
#
# All four knobs live here so a sweep only ever edits this block.
# ---------------------------------------------------------------------------

# When the pulse may fire, in seconds of episode time.  Deliberately after the
# rotation window (0-2 hold, 2-4 SLERP, 4-5 settling) so a failure cannot be
# confused with a rotation-tracking failure.  Assumes the final 15 s curriculum
# stage; HAND_MOVE_SCHEDULE.episode_length_s is NOT touched by this feature.
DISTURBANCE_TIME_RANGE_S = (0, 12.0)

# How long the force pushes.  Short enough to read as a knock rather than a
# sustained load.
DISTURBANCE_DURATION_S = 0.10

# PROVISIONAL - calibrate so disturbance causes ~1-2 functional contact losses
# without dropping the stick.
#
#   too weak   : all 6 contacts survive
#   right      : 6 -> 4~5 contacts briefly, stick still in the hand, recoverable
#   too strong : large slip, far outside the functional grasp, or a drop
#
# Starting point is deliberately tiny: the stick is 0.01 kg, so 0.02 N is about
# 0.2 g of weight, and 0.10 N over 0.10 s is an impulse of 0.01 N*s = 1 m/s of
# free-stick velocity change.  Sweep upward from here in play/debug; the
# diagnostics below (contacts_before / minimum_contacts_after) are what tells
# you which band you are in.
# Keep a valid conservative band even while probability is zero.  The first
# fresh hand_real stage is disturbance-free; this range becomes active only
# when DISTURBANCE_PROBABILITY is explicitly enabled in a later stage.
DISTURBANCE_FORCE_RANGE_N = (0.02, 0.1)
#DISTURBANCE_FORCE_RANGE_N = (0.05, 0.3)
#DISTURBANCE_FORCE_RANGE_N = (0.3, 0.9)
#DISTURBANCE_FORCE_RANGE_N = (0.9, 1.2)

# Fraction of episodes that get a pulse at all.  Below 1.0 the policy still sees
# undisturbed episodes, which keeps the original behaviour represented.
DISTURBANCE_PROBABILITY = 0.0


@configclass
class HandMoveEventCfg:
    """Identical reset to ``hand_grasp``; correct for a floating root as-is.

    ``reset_scene_to_default`` writes the articulation root pose *and* zeroes
    the root linear/angular velocity from ``init_state`` (``HAND_ROOT_POS`` /
    ``HAND_ROOT_ROT`` plus the env origin), and it runs before
    ``reset_pregrasp``.  Writing the root pose invalidates the cached body-pose
    buffer, so the ``body_pos_w``/``body_quat_w`` that ``reset_pregrasp`` reads
    for the palm already reflect the new root pose, and the two sticks are
    placed from the correct palm frame.  No task-local reset function is needed.
    """

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_pregrasp = EventTerm(
        func=hand_grasp_mdp.reset_to_functional_pregrasp,
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
        },
    )

    # Per-step hook for the disturbance pulse state machine.
    #
    # interval_range_s = (0, 0) makes EventManager fire this on every policy step
    # for every environment (event_manager.py:210-232).  That is only the hook -
    # StickDisturbance decides from its own per-env state and the episode clock
    # whether a pulse is running, so "at most once per episode" does not depend
    # on the interval mechanism.  Plain interval *scheduling* is unusable here
    # precisely because it resamples and re-fires.
    stick_disturbance = EventTerm(
        func=hand_move_mdp.StickDisturbance,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=False,
        params={
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            # Same six groups and threshold the grasp reward uses, so the
            # diagnostic "contact count" means the same thing there and here.
            # Read-only: this term never feeds reward or termination.
            "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "contact_threshold": 0.02,
            "time_range_s": DISTURBANCE_TIME_RANGE_S,
            "duration_s": DISTURBANCE_DURATION_S,
            "force_range_n": DISTURBANCE_FORCE_RANGE_N,
            "probability": DISTURBANCE_PROBABILITY,
        },
    )


@configclass
class HandMoveRewardsCfg:
    """Unchanged ``hand_grasp`` reward set, weights included."""

    joint_reference = RewTerm(
        func=hand_grasp_mdp.JointReferenceTracking,
        weight=2.0,
        params={
            "asset_cfg": HAND_JOINTS,
            "reference_joint_positions": PREGRASP_JOINT_POSITIONS,
            "sigma": 0.20,
        },
    )
    stick1_pivot = RewTerm(
        func=hand_grasp_mdp.ObjectPointReferenceTracking,
        weight=10.0,
        params={
            "palm_cfg": PALM,
            "object_cfg": STICK_1,
            "point_o": STICK1_PIVOT_OFFSET_O,
            "reference_position_p": PREGRASP_STICK1_POSITION_P,
            "reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
            "sigma": 0.01,
        },
    )
    stick2_reference_pose = RewTerm(
        func=hand_grasp_mdp.ObjectReferencePoseTracking,
        weight=15.0,
        params={
            "palm_cfg": PALM,
            "object_cfg": STICK_2,
            "reference_position_p": PREGRASP_STICK2_POSITION_P,
            "reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
            "position_sigma": 0.01,
            "orientation_sigma": 0.1745329252,
        },
    )
    open_tip_gap = RewTerm(
        func=hand_grasp_mdp.mode_tip_gap_tracking,
        weight=20.0,
        params={
            "command_name": "open_close",
            "mode_index": 0,
            "target_gap": HAND_MOVE_OPEN_TARGET_GAP,
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "stick1_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick2_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick_thickness": STICK_SIZE[0],
            "reference_separation_direction_stick2": TIP_SEPARATION_DIRECTION_STICK2,
            "reference_axial_offset_stick2": TIP_AXIAL_OFFSET_STICK2,
            "sigma": 0.005,
            # lateral 은 tip_lateral 독립 항으로 분리 (2026-08-07).
            # 셋이 한 지수를 공유하면 서로의 gradient 를 곱으로 깎는다.
            "lateral_sigma": None,
            "axial_sigma": TIP_AXIAL_SIGMA,
            # 교차(옆으로 지나감) 상태가 gap=0 만점으로 보고되던 것을 막는다.
            "clamp_gap": False,
        },
    )
    close_tip_gap = RewTerm(
        func=hand_grasp_mdp.mode_tip_gap_tracking,
        weight=20.0,
        params={
            "command_name": "open_close",
            "mode_index": 1,
            "target_gap": CLOSE_TIP_GAP,
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "stick1_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick2_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick_thickness": STICK_SIZE[0],
            "reference_separation_direction_stick2": TIP_SEPARATION_DIRECTION_STICK2,
            "reference_axial_offset_stick2": TIP_AXIAL_OFFSET_STICK2,
            "sigma": 0.005,
            # lateral 은 tip_lateral 독립 항으로 분리 (2026-08-07).
            # 셋이 한 지수를 공유하면 서로의 gradient 를 곱으로 깎는다.
            "lateral_sigma": None,
            "axial_sigma": TIP_AXIAL_SIGMA,
            # 교차(옆으로 지나감) 상태가 gap=0 만점으로 보고되던 것을 막는다.
            "clamp_gap": False,
        },
    )
    functional_contact_min = RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=20.0,
        params={
            "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "force_scale": 0.10,
            "reduction": "min",
        },
    )
    mode_grasp_stability = RewTerm(
        func=hand_grasp_mdp.mode_grasp_stability,
        weight=50.0,
        params={
            "command_name": "open_close",
            "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "stick1_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick2_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick_thickness": STICK_SIZE[0],
            "open_target_gap": HAND_MOVE_OPEN_TARGET_GAP,
            "close_target_gap": CLOSE_TIP_GAP,
            "reference_separation_direction_stick2": TIP_SEPARATION_DIRECTION_STICK2,
            "reference_axial_offset_stick2": TIP_AXIAL_OFFSET_STICK2,
            "gap_sigma": 0.003,
            "lateral_sigma": None,      # tip_lateral 독립 항으로 분리
            "axial_sigma": TIP_AXIAL_SIGMA,
            "clamp_gap": False,
            "force_scale": 0.10,
            "contact_threshold": 0.02,
            "linear_speed_scale": 0.10,
            "angular_speed_scale": 2.0,
        },
    )
    # 2026-08-07: 두 스틱이 lateral(개폐 평면 밖)로 벌어지는 것을 막는다.
    #
    # 왜 별도 항인가: 3D 에서 두 직선이 만나려면 위치와 방향이 **둘 다** 맞아야
    # 한다.  위치 쪽(팁 두 점의 평면법선 성분)은 이미 open/close_tip_gap 과
    # mode_grasp_stability 안의 lateral_error 가 잡고 있었지만, 방향 쪽(stick1
    # 축의 평면법선 성분)은 **아무도 재지 않았다** - _tip_geometry_from_palm_poses
    # 는 stick1 의 샤프트 축을 계산조차 하지 않는다.  그래서 스틱이 비스듬히
    # 기운 채로 닫혀 맞물리지 못하고 미끄러졌다.
    #
    # 왜 기존 지수에 안 넣는가: mode_tip_gap_tracking 은 gap/lateral/axial 을
    # **하나의 지수**에 넣어 곱하므로, 거기에 항을 더하면 skew 가 나쁠 때 gap 의
    # gradient 까지 같이 깎인다.  덧셈으로 분리해야 서로 안 깎는다.
    #
    # 왜 모드 게이트가 없는가: 실측 skew 가 OPEN 5.31 / CLOSE 5.28 deg 로 같다.
    # 개폐와 무관한 상시 편향이고, 이미 기운 채로 들어오면 CLOSE 에서 고칠 수 없다.
    #
    # sigma 0.05 rad(2.9 deg)는 실측 3.05 deg 근처 - Laplacian 은 e/sigma = 1 에서
    # gradient 가 최대다.  weight 10 x 15 s = 예산 150 점 (close_tip_gap 200 점 대비).
    # 진단은 Metrics/hand_grasp/stick_axis_skew_angle.
    # 2026-08-07: lateral 의 **위치** 절반.  방향 절반은 아래 stick_axis_lateral.
    #
    # 원래 open/close_tip_gap 과 mode_grasp_stability 의 공유 지수 안에 있었는데,
    # 그 항들이 최대의 5% 수준이라 lateral gradient 가 같은 0.05 로 깎여 아무 일도
    # 못 했다.  실측 tip_lateral_error 9.14 mm - 스틱 단면이 7 mm 이므로 두 단면이
    # 겹치는 부분이 아예 없어 닫으면 서로 지나간다(교차).
    #
    # gap 항은 이걸 못 막는다: transverse_distance 가 norm 이라 "옆으로 9mm" 와
    # "개폐방향으로 9mm" 를 구분하지 못하고, support 를 빼면 둘 다 gap=0(완벽한
    # CLOSE)으로 보고된다.  독립 항으로 빼야 gap 상태와 무관하게 gradient 가 산다.
    #
    # sigma 는 TIP_LATERAL_ERROR_LIMIT(5mm) - success 종료가 이미 쓰는 합격선이고
    # 겹침이 사라지는 7mm 보다 안쪽이다.  모드 게이트 없음: 실측이 OPEN/CLOSE 에서
    # 같다(8.30 vs 8.26 mm).
    tip_lateral = RewTerm(
        func=hand_grasp_mdp.tip_lateral_alignment,
        weight=15.0,
        params={
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "stick1_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick2_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick_thickness": STICK_SIZE[0],
            "reference_separation_direction_stick2": TIP_SEPARATION_DIRECTION_STICK2,
            "sigma": TIP_LATERAL_SIGMA,
        },
    )
    stick_axis_lateral = RewTerm(
        func=hand_grasp_mdp.stick_axis_lateral_alignment,
        weight=10.0,
        params={
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "reference_separation_direction_stick2": TIP_SEPARATION_DIRECTION_STICK2,
            "sigma": 0.05,
        },
    )
    angular_speed_excess = RewTerm(
        func=hand_grasp_mdp.object_pair_angular_speed_excess_l2,
        weight=-0.1,
        params={
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "angular_speed_limit": 7.0,
            "max_excess": 10.0,
        },
    )
    success = RewTerm(
        func=hand_grasp_mdp.OpenCloseModeHeld,
        weight=30000.0,
        params={
            **MODE_HELD_PARAMS,
            "open_target_gap": HAND_MOVE_OPEN_TARGET_GAP,
            # 보상이 clamp_gap=False 로 도니 성공 판정도 같은 정의를 써야 한다.
            # 안 그러면 교차 상태를 보상은 깎고 성공은 인정한다 (2026-08-07).
            "clamp_gap": False,
            "hold_steps": 30,
            "one_shot_per_mode": True,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)


@configclass
class HandMoveTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    stick1_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.20, "asset_cfg": STICK_1},
    )
    stick2_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.20, "asset_cfg": STICK_2},
    )
    # After a load-bearing six-contact grasp has been established, terminate a
    # sustained loss of any one functional contact.  The lower release
    # threshold and 15-step debounce tolerate force chatter and brief
    # OPEN/CLOSE transition flicker.
    functional_contact_lost = DoneTerm(
        func=hand_grasp_mdp.FunctionalContactLoss,
        params={
            "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "acquire_threshold": 0.10,
            "release_threshold": 0.05,
            "acquire_hold_steps": 5,
            "minimum_retained_contacts": 6,
            "loss_hold_steps": 15,
        },
    )
    success = DoneTerm(
        func=hand_grasp_mdp.OpenCloseModeHeld,
        params={
            **MODE_HELD_PARAMS,
            "open_target_gap": HAND_MOVE_OPEN_TARGET_GAP,
            # 보상이 clamp_gap=False 로 도니 성공 판정도 같은 정의를 써야 한다.
            # 안 그러면 교차 상태를 보상은 깎고 성공은 인정한다 (2026-08-07).
            "clamp_gap": False,
            # Unreachable on purpose: a successful mode must not reset the
            # continuous OPEN/CLOSE sequence.
            "hold_steps": 1_000_000,
        },
    )


@configclass
class HandMoveEnvCfg(NrmkRLEnvCfg):
    """OPEN/CLOSE learning around the validated grasp, on a floating hand root."""

    scene: HandMoveSceneCfg = HandMoveSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: HandMoveObservationsCfg = HandMoveObservationsCfg()
    actions: HandMoveActionsCfg = HandMoveActionsCfg()
    commands: HandMoveCommandsCfg = HandMoveCommandsCfg()
    rewards: HandMoveRewardsCfg = HandMoveRewardsCfg()
    terminations: HandMoveTerminationsCfg = HandMoveTerminationsCfg()
    events: HandMoveEventCfg = HandMoveEventCfg()
    curriculum = EmptyCfg()
    costs = EmptyCfg()

    actor_obs_list: list = ["policy"]
    critic_obs_list: list | None = None
    teacher_obs_list: list | None = None

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        # 15 s = 2 s hold + 2 s SLERP + 1 s settling + 5 x 2 s OPEN/CLOSE.
        # Taken from the shared schedule so the two can never disagree; the
        # schedule itself asserts that decomposition.
        HAND_MOVE_SCHEDULE.validate()
        self.episode_length_s = HAND_MOVE_SCHEDULE.episode_length_s
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        self.viewer.eye = (0.55, 0.55, 0.78)
        self.viewer.lookat = (0.045, 0.0, 0.51)
