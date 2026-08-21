"""Task-local MDP terms for ``hand_object`` (2026-08-06).

``hand_object`` fine-tunes the ``hand_move`` policy so that the two chopsticks
actually pinch a 10 mm cube with real contact force and keep holding it once
the support underneath the cube is taken away.

Everything in this module is additive.  No ``hand_grasp`` or ``hand_move``
file, function or class is modified; the pieces that can be reused are
subclassed instead of copied, so the validated timing/SLERP/manual-override
machinery cannot drift.

Policy interface is frozen
--------------------------
The actor still sees the same 103D observation and still emits the same 20D
finger residual as ``hand_move``.  The cube, the contact forces and the phase
are used **only** by rewards, terminations and metrics.  A ``hand_move``
checkpoint therefore loads with no shape change - that is the whole point of
this task, and adding a single cube observation would break it.

Episode script (9 s, all boundaries from :data:`HAND_OBJECT_SCHEDULE`).  The
support column is the one thing here that is *not* on the clock - it goes as
soon as the grasp is real, and at the deadline below if it never becomes real::

    ===========  ==================================  ==========  ==============
    time         root pose target                    OPEN/CLOSE  support
    ===========  ==================================  ==========  ==============
    [0.0, 0.5)   spawn pose (hold)                   OPEN        up
    [0.5, 1.25)  align x/y + SLERP at spawn height   OPEN        up
    [1.25, 2.0)  descend straight down to the cube   OPEN        up
    [2.0, 2.5)   goal pose (settle)                  OPEN        up
    [2.5, 5.5)   goal pose                           CLOSE       up until the
                                                                 grasp is real
    <trigger>    goal pose                           CLOSE       retracting 0.5s
    ... 9.0]     goal pose                           CLOSE       down
    ===========  ==================================  ==========  ==============

``<trigger>`` is whenever both sticks have been loaded on the cube for
``retract_trigger_debounce_steps`` consecutive steps, or 5.5 s, whichever comes
first.  So the unsupported hold window is *at least* 3.0 s and longer for a
policy that grips early - which is the intended pressure to grip early.

The goal pose is *not* random here: every environment and every episode flies to
the same calibrated position and orientation, which is what makes a fixed cube
position meaningful.

What the calibration actually determines
---------------------------------------
The hand's spawn pose and the cube's position are **both fixed** - the spawn
because it is the validated functional grasp, the cube because the task is to
reach it.  So the entire calibration is one thing: **the root pose the hand
flies to** during the rotation window, chosen so the distal tips end up
straddling the cube.  The hand translates as well as rotating; turning on the
spot leaves the tips swinging on an arc about the palm that never arrives.

:data:`HAND_OBJECT_TARGET_ROOT_POS_E`, :data:`HAND_OBJECT_TARGET_EULER_RAD`,
:data:`HAND_OBJECT_CUBE_POS_E` and :data:`HAND_OBJECT_SUPPORT_POS_E` have never
been measured.  Guessing them and calling the task finished would produce a run
whose geometry nobody checked, so they are ``None`` and the training config
raises a clear error.  Fly the hand with the keyboard, press ``P``, and paste
the two lines it prints.  :data:`HAND_OBJECT_FORCE_SATURATION_N` is in the same
state but has a provisional value, because unlike the geometry it cannot be
determined without running the environment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import (
    CommandTerm,
    CommandTermCfg,
    ManagerTermBase,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
)

from .mdp import _group_forces
from .hand_move_mdp import (
    CLOSE_INDEX,
    quat_slerp_batch,
    _quat_geodesic_angle,
    OPEN_INDEX,
    HandMoveOpenCloseCommand,
    HandMoveOpenCloseCommandCfg,
    HandMoveRootOrientationCommand,
    HandMoveRootOrientationCommandCfg,
    HandMoveScheduleCfg,
    _normalize_quat,
)

# ---------------------------------------------------------------------------
# CALIBRATION CONSTANTS - UNCALIBRATED, measured by a human, never by code.
#
# All positions are **env-local** (the ``_E`` suffix): the value written into
# ``RigidObjectCfg.init_state.pos``, which Isaac Lab offsets by the per-env
# origin when it replicates the scene.  Never store a world coordinate here -
# it would put every environment except env 0 in the wrong place.
# ---------------------------------------------------------------------------

# Where the hand has to end up, env-local.  The spawn pose and the cube are
# both fixed, so this pair is the whole calibration: it is the root pose the
# scripted trajectory flies to during the rotation window, chosen so the distal
# tips straddle the cube once it gets there.
#
# The hand **translates as well as rotating**.  Holding the position and only
# turning leaves the tips swinging on an arc about the palm, which will not
# arrive at a cube that is 12 cm away.
HAND_OBJECT_TARGET_ROOT_POS_E: tuple[float, float, float] | None = (0.0685, -0.0400, 0.3650)

# Relative rotation about the hand's **own** axes, applied to the validated
# reset orientation as ``q_goal = q_start (x) q_delta``, in radians
# ``(roll, pitch, yaw)``.  Measured at reset the hand's local z points along
# world +x, so its "yaw" swings the tips through the world y-z plane - it is not
# a rotation about the world vertical.  Read this straight off the ``P`` key.
HAND_OBJECT_TARGET_EULER_RAD: tuple[float, float, float] | None = (0.0934, 0.3472, -1.0331)

# Cube centre, env-local.  Measured 2026-08-06 as the distal-tip midpoint at the
# goal pose above: the operator flew the hand to a pose that grips well, and the
# cube is placed where the tips end up.  Fixed from here on - if the geometry
# ever needs adjusting, move this and the support, never the validated grasp.
HAND_OBJECT_CUBE_POS_E: tuple[float, float, float] | None = (0.1468, 0.0865, 0.4114)

# Support centre, env-local.  Its top face must sit exactly one cube
# half-height plus the clearance below the cube centre; the env config derives
# and checks that, so only the x/y and the column height are free here.
HAND_OBJECT_SUPPORT_POS_E: tuple[float, float, float] | None = (0.1468, 0.0865, 0.2030)

# Directional inward force at which the strict squeeze reward saturates.  This
# also feeds the support-retract trigger, so initial contact acquisition must
# not change it.
HAND_OBJECT_FORCE_SATURATION_N: float = 0.05  # PROVISIONAL / UNCALIBRATED

# Acquisition-only scale for the magnitude of the two cube-filtered contact
# forces.  It deliberately ignores direction so it can bridge two-sided touch
# to the stricter closing-axis squeeze without lowering the retract threshold.
HAND_OBJECT_CONTACT_ACQUISITION_SATURATION_N: float = 0.005

# The cube objective must not be purchasable by releasing one of the six
# hand--stick support contacts.  This is the same load-bearing scale used by
# hand_move's functional-contact reward and acquisition latch.
HAND_OBJECT_FUNCTIONAL_FORCE_SCALE_N: float = 0.10


def calibration_is_complete() -> bool:
    """True once the three measured constants above have been filled in."""
    return (
        HAND_OBJECT_TARGET_ROOT_POS_E is not None
        and HAND_OBJECT_TARGET_EULER_RAD is not None
        and HAND_OBJECT_CUBE_POS_E is not None
        and HAND_OBJECT_SUPPORT_POS_E is not None
    )


def missing_calibration_names() -> tuple[str, ...]:
    """Names of the constants still unmeasured, for the fail-fast message."""
    missing = []
    if HAND_OBJECT_TARGET_ROOT_POS_E is None:
        missing.append("HAND_OBJECT_TARGET_ROOT_POS_E")
    if HAND_OBJECT_TARGET_EULER_RAD is None:
        missing.append("HAND_OBJECT_TARGET_EULER_RAD")
    if HAND_OBJECT_CUBE_POS_E is None:
        missing.append("HAND_OBJECT_CUBE_POS_E")
    if HAND_OBJECT_SUPPORT_POS_E is None:
        missing.append("HAND_OBJECT_SUPPORT_POS_E")
    return tuple(missing)


# ---------------------------------------------------------------------------
# Contact-force sign convention.
#
# These are NOT guesses.  ``ContactSensorData.force_matrix_w`` comes from
# ``contact_physx_view.get_contact_force_matrix``, whose own docstring does not
# state which body the force acts on.  The convention is pinned down by the
# installed isaacsim test ``test_rigid_prim_view.contact_force_test``:
#
#     setup     : gravity g = -10; Box (1 kg) on the ground,
#                 TopBox (1 kg) resting on Box;
#                 sensor view = Box, filter = TopBox
#     assertion : forces_matrix[:, 0, :] == [0, 0, g] == [0, 0, -10]
#
# TopBox weighs 10 N and presses *down* on Box, and the reported vector is
# -10 z.  So the matrix holds the force exerted **by the filtered body on the
# sensor body**, in world coordinates.
#
# Applying that here, with ``closing_axis_w`` pointing from the Stick2 tip
# towards the Stick1 tip:
#
#   * sensor Stick1, filter Cube.  Stick1 squeezing pushes the cube along
#     -closing_axis; the cube's reaction on Stick1 is along +closing_axis.
#     A squeeze therefore reads dot(f1, axis) > 0  ->  sign +1.
#   * sensor Stick2, filter Cube.  The cube pushes Stick2 away from Stick1,
#     i.e. along -closing_axis, so a squeeze reads dot(f2, axis) < 0
#     ->  sign -1.
#
# The calibration logging prints the raw vectors and both dot products so this
# derivation can be confirmed against a real pinch rather than trusted.  If the
# printout shows both inward forces negative during an obvious squeeze, flip
# both signs here.
# ---------------------------------------------------------------------------
STICK1_CUBE_FORCE_SIGN: float = +1.0
STICK2_CUBE_FORCE_SIGN: float = -1.0


@configclass
class HandObjectScheduleCfg(HandMoveScheduleCfg):
    """Single tuning point for the whole ``hand_object`` episode.

    Inherits the ``hand_move`` field names so
    :class:`HandObjectRootOrientationCommand` can reuse the parent's SLERP and
    phase code unchanged; only the values and the extra support fields differ.

    Naming note: ``open_close_start_time_s`` is inherited and here means "the
    instant CLOSE begins".  There is no alternation in this task - the hand
    opens, rotates, then closes once and holds.
    """

    # -- The single calibrated goal pose, read from the module constants so
    #    there is exactly one place to edit.
    target_root_pos_e: tuple[float, float, float] | None = HAND_OBJECT_TARGET_ROOT_POS_E
    target_euler_rad: tuple[float, float, float] | None = HAND_OBJECT_TARGET_EULER_RAD

    # -- The approach is two stages, not one straight line.
    #
    #   stage 1  (first ``approach_fraction`` of the move window)
    #            turn to the goal orientation and line up x/y, all at the spawn
    #            height.  Nothing is near the cube yet, so a large motion here
    #            is free.
    #   stage 2  (the rest)
    #            descend straight down to the cube, orientation already fixed.
    #
    # A single straight line from spawn to goal would arrive at the cube on a
    # diagonal while still rotating, which sweeps the tips sideways through the
    # place the cube is sitting.  Coming down vertically onto a settled pose is
    # both easier to reason about and closer to how the real approach would go.
    approach_fraction: float = 0.5

    # -- Timing, in seconds.  Provisional; tuned by watching the calibration run.
    initial_hold_time_s: float = 0.5
    rotation_interpolation_time_s: float = 1.5
    rotation_settling_time_s: float = 0.5
    open_close_start_time_s: float = 2.5  # = 0.5 + 1.5 + 0.5, i.e. CLOSE begins

    # Inherited from ``HandMoveScheduleCfg`` and vestigial here: this task never
    # alternates OPEN/CLOSE, so there is no segment decomposition to satisfy.
    # ``validate()`` below deliberately does not call ``super().validate()``,
    # which is what would otherwise force ``open_close_start + segment x n ==
    # episode_length``.  They are kept only because the parent's phase code
    # reads them.
    open_close_segment_time_s: float = 6.5
    num_open_close_segments: int = 1
    episode_length_s: float = 9.0

    # -- Support retract deadline.
    #
    # The retract is **condition-triggered, with this as the fallback**: the
    # column starts sliding as soon as the grasp is real (see
    # ``HandObjectSupportCommandCfg.retract_trigger_*``), and at this time at
    # the latest whether or not anything was gripped.
    #
    # The fallback is not optional.  With a pure condition trigger, "never form
    # a grasp" means the column never leaves, the cube never falls, the drop
    # termination never fires, and ``bilateral_cube_force`` can be farmed for
    # the whole episode at zero risk - i.e. avoiding the cube becomes the safe
    # optimum, which is exactly the failure already observed on 2026-08-06.
    # The deadline guarantees the test always happens; the condition only lets
    # a policy that *did* grip take it earlier, and being earlier pays because
    # the hold window is what is left of the episode.
    support_retract_deadline_time_s: float = 7.5
    support_retract_duration_s: float = 0.5
    # How far the column drops, in metres.
    #
    # This has to be large enough that an *unheld* cube falls clear of the
    # column, not merely large enough to open a gap under a held one.  The
    # column sits directly beneath the cube, so a short retract just lowers the
    # pedestal and the cube lands back on it: at 20 mm the cube could only ever
    # fall 20 mm, which is less than ``cube_drop_height_m``, so a drop was never
    # detected and the policy learned that ignoring the cube costs nothing
    # (2026-08-06 run 23-48-32: holding stayed 0.000 for 3557 iterations while
    # cube_height never went below 0.357 m).
    #
    # 150 mm clears the column entirely.  At the configured 0.5 s that is
    # 300 mm/s, still a smooth slide rather than a teleport.
    support_retract_distance_m: float = 0.15

    # -- Cube drop detection (geometry based, never a single lost contact).
    #    Distance from the Stick2 tip beyond which the cube counts as gone.
    cube_drop_distance_m: float = 0.05
    #    Drop below the cube's own reset height that also counts as gone.
    #    Must stay well under ``support_retract_distance_m`` - a cube that can
    #    only fall as far as the column travels can never trip a larger
    #    threshold.  ``validate()`` enforces the relationship.
    cube_drop_height_m: float = 0.03
    #    Consecutive policy steps the condition must hold before terminating.
    cube_drop_debounce_steps: int = 3

    # -- Diagnostics.
    debug_env_ids: tuple[int, ...] = (0,)

    @property
    def support_retract_end_time_s(self) -> float:
        """Latest instant the column can still be moving.

        Worst case, not the usual case: an environment whose grasp trips the
        condition early finishes its retract well before this.
        """
        return self.support_retract_deadline_time_s + self.support_retract_duration_s

    @property
    def close_hold_time_s(self) -> float:
        """Seconds of CLOSE before the retract deadline - time to form a grasp."""
        return self.support_retract_deadline_time_s - self.open_close_start_time_s

    @property
    def hold_phase_time_s(self) -> float:
        """Shortest possible unsupported hold window, i.e. the deadline case."""
        return self.episode_length_s - self.support_retract_end_time_s

    @property
    def slerp_end_time_s(self) -> float:
        """Redeclared from the parent; inheriting it alone breaks ``configclass``.

        ``_custom_post_init`` walks ``dir(obj)`` and deep-copies every member,
        skipping properties - but it looks the name up in
        ``obj.__class__.__dict__`` (``configclass.py:398``), which only holds
        *this* class's own attributes.  A property defined solely on the parent
        is therefore not recognised as one, and the deep-copy turns into
        ``setattr`` on a read-only attribute::

            AttributeError: property 'slerp_end_time_s' of
            'HandObjectScheduleCfg' object has no setter

        Restating it here puts it in this class's ``__dict__`` so the check
        finds it.  Any future property added to ``HandMoveScheduleCfg`` will
        need the same treatment.
        """
        return self.initial_hold_time_s + self.rotation_interpolation_time_s

    def validate(self) -> None:
        """Timing checks for this task.

        Deliberately does **not** call ``super().validate()``: the parent's
        check is ``open_close_start + segment x n == episode_length``, which
        describes an alternating OPEN/CLOSE schedule this task does not have.
        The phase boundaries that matter here are checked directly below.
        """
        if self.rotation_interpolation_time_s <= 0.0:
            raise ValueError(
                "HandObjectScheduleCfg.rotation_interpolation_time_s must be positive."
            )
        if self.open_close_start_time_s < (
            self.initial_hold_time_s
            + self.rotation_interpolation_time_s
            + self.rotation_settling_time_s
        ):
            raise ValueError(
                "HandObjectScheduleCfg: CLOSE must begin after the hand has arrived "
                f"and settled ({self.open_close_start_time_s} < "
                f"{self.initial_hold_time_s} + {self.rotation_interpolation_time_s} + "
                f"{self.rotation_settling_time_s}); closing while still flying sweeps "
                "the tips through the cube."
            )
        if self.support_retract_duration_s <= 0.0:
            raise ValueError(
                "HandObjectScheduleCfg.support_retract_duration_s must be positive."
            )
        if self.support_retract_deadline_time_s < self.open_close_start_time_s:
            raise ValueError(
                "HandObjectScheduleCfg: the support may only retract after CLOSE has "
                f"begun ({self.support_retract_deadline_time_s} < "
                f"{self.open_close_start_time_s}); retracting earlier means the policy "
                "is asked to hold a cube it was never told to grip."
            )
        if self.support_retract_end_time_s >= self.episode_length_s:
            raise ValueError(
                "HandObjectScheduleCfg: the retract must finish strictly before the "
                f"episode ends ({self.support_retract_end_time_s} >= "
                f"{self.episode_length_s}), otherwise the hold phase has zero length "
                "and the hold reward can never be earned."
            )
        if self.support_retract_distance_m <= 0.0:
            raise ValueError(
                "HandObjectScheduleCfg.support_retract_distance_m must be positive."
            )
        if self.support_retract_distance_m <= 2.0 * self.cube_drop_height_m:
            raise ValueError(
                "HandObjectScheduleCfg: the support must retract well past the drop "
                f"threshold ({self.support_retract_distance_m} <= "
                f"2 x {self.cube_drop_height_m}). The column sits directly under the "
                "cube, so an unheld cube falls only as far as the column travels; if "
                "that is not clearly more than cube_drop_height_m the drop is never "
                "detected, the cube quietly rests on the lowered column, and ignoring "
                "it becomes the optimal policy."
            )
        if not 0.0 < self.approach_fraction < 1.0:
            raise ValueError(
                "HandObjectScheduleCfg.approach_fraction must be strictly between "
                f"0 and 1 (got {self.approach_fraction}); it splits the move window "
                "into the align stage and the descent stage."
            )
        if self.cube_drop_debounce_steps < 1:
            raise ValueError(
                "HandObjectScheduleCfg.cube_drop_debounce_steps must be at least 1."
            )


# Shared schedule instance.  Edit this one object to retune the task.
HAND_OBJECT_SCHEDULE = HandObjectScheduleCfg()


# ---------------------------------------------------------------------------
# Command terms.
# ---------------------------------------------------------------------------


class HandObjectRootOrientationCommand(HandMoveRootOrientationCommand):
    """``hand_move``'s root trajectory with one fixed goal instead of a random one.

    Only :meth:`_resample_command` changes.  The SLERP, the phase computation,
    the convergence metrics, the frame markers and the manual-play override are
    all inherited, so the play/calibration keyboard path behaves exactly as it
    does in ``hand_move``.
    """

    cfg: HandObjectRootOrientationCommandCfg

    def __init__(self, cfg: HandObjectRootOrientationCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Fail-fast on unmeasured geometry.
        #
        # This check belongs here rather than in the env config's
        # ``__post_init__`` because ``hydra_task_config`` builds the config
        # before applying command-line overrides, so a raise there would fire
        # even for the calibration session that is meant to opt out.  Command
        # terms are constructed inside ``gym.make``, which already has the
        # overridden config, so ``env.require_calibration=false`` works here.
        if getattr(env.cfg, "require_calibration", True) and not calibration_is_complete():
            missing = ", ".join(missing_calibration_names())
            raise ValueError(
                "HandObject goal-pose/cube/support calibration is not completed: "
                f"{missing} are still None in hand_object_mdp.py.\n"
                "Run the calibration session first:\n"
                "  python scripts/rsl_rl/play.py --task hand_object --num_envs 1 \\\n"
                "      --manual_root --load_run <hand_move run, e.g. 2026-08-06_00-12-28> \\\n"
                "      env.episode_length_s=300.0\n"
                "Fly the hand with I/K/J/L/U/O and Q/E/W/S/A/D until "
                "'cube - tip midpoint' is ~0, then press P and paste the two "
                "printed lines into hand_object_mdp.py."
            )
        self._uncalibrated_warned = False

        # 2026-08-08: 적응형 목표가 실제로 무엇을 하는지 보기 위한 진단 3종.
        #
        #   adaptive_goal_shift            고정 상수 대비 목표를 얼마나 옮겼나
        #                                  = 에피소드마다 스틱이 얼마나 다르게 놓이나
        #   position_error_at_close_start  루트가 목표에 도착했나 (PD 추종)
        #   tip_to_cube_at_close_start     팁 중점이 큐브에 도착했나  <- 결정적
        #
        # 마지막 것이 핵심이다. 루트 위치 오차만 보면 "PD 가 잘 따라갔다"까지만 알 수
        # 있고, 정작 중요한 "그래서 팁이 큐브 옆에 있나"는 못 본다. 두 값이 갈리면
        # 원인이 루트인지 손 안의 스틱인지 바로 구분된다.
        self._reset_goal = torch.zeros((self.num_envs, 3), device=self.device)
        for name in (
            "adaptive_goal_shift",
            "goal_drift_during_approach",
            "position_error_at_close_start",
            "tip_to_cube_at_close_start",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self._close_start_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def __str__(self) -> str:
        schedule = self.cfg.schedule
        pos = schedule.target_root_pos_e
        euler = schedule.target_euler_rad
        pose_text = (
            "UNCALIBRATED"
            if pos is None or euler is None
            else (
                f"pos {tuple(round(v, 4) for v in pos)}, "
                f"euler_deg {tuple(round(math.degrees(v), 2) for v in euler)}"
            )
        )
        return (
            "HandObjectRootOrientationCommand:\n"
            f"\thold {schedule.initial_hold_time_s}s"
            f" -> move {schedule.rotation_interpolation_time_s}s"
            f" -> settle {schedule.rotation_settling_time_s}s\n"
            f"\tfixed goal root pose (env-local, relative euler): {pose_text}"
        )

    """
    Scripted position target, read by ``HandRootHoldAction``.
    """

    @staticmethod
    def _smoothstep(value: torch.Tensor) -> torch.Tensor:
        return value * value * (3.0 - 2.0 * value)

    @property
    def waypoint_pos_w(self) -> torch.Tensor:
        """End of stage 1: the goal x/y held at the spawn height, ``(N, 3)``.

        Derived rather than calibrated - it is the goal position with its z
        replaced by the start's, so stage 2 is a pure vertical descent by
        construction and no second constant can drift out of step with the
        first.
        """
        waypoint = self._position_goal.clone()
        waypoint[:, 2] = self._position_start[:, 2]
        return waypoint

    @property
    def target_pos_w(self) -> torch.Tensor | None:
        """World position the root PD controller should hold, shape ``(N, 3)``.

        Two stages, split at ``approach_fraction`` of the move window:

        * **align** - travel to the goal x/y at the spawn height while the
          orientation SLERP runs.  The hand is nowhere near the cube, so this
          large motion costs nothing.
        * **descend** - drop straight down onto the cube with the orientation
          already settled.

        The two ramps are written as a sum because stage 1 is complete before
        stage 2 begins (``a1`` saturates at 1 exactly where ``a2`` leaves 0), so
        the sum is the same as a branch but has no boundary case to get wrong.

        Returns ``None`` under manual override, and that is not a detail:
        ``HandRootHoldAction`` copies this into its position-target buffer on
        every physics step, which is the same buffer the keyboard writes
        through ``add_target_position_delta``.  Without the ``None`` the
        translation keys appear dead - each press is undone before the next
        frame - while the rotation keys keep working, because those write into
        the command term's own quaternion buffer instead.
        """
        if self._manual_override:
            return None
        fraction = self.cfg.schedule.approach_fraction
        alpha = self._alpha
        a1 = self._smoothstep(torch.clamp(alpha / fraction, min=0.0, max=1.0))
        a2 = self._smoothstep(
            torch.clamp((alpha - fraction) / (1.0 - fraction), min=0.0, max=1.0)
        )
        waypoint = self.waypoint_pos_w
        return (
            self._position_start
            + a1 * (waypoint - self._position_start)
            + a2 * (self._position_goal - waypoint)
        )

    def _solve_goal_position(
        self, env_ids, quat_goal: torch.Tensor
    ) -> torch.Tensor:
        """Root position that puts the *current* tip midpoint on the cube.

        ``d_r`` is the root-frame offset from the root to the tip midpoint, i.e.
        exactly the thing this command does not control.  Solving

            p_goal_w = cube_w - q_goal . d_r

        cancels it out.  Rigid-offset inverse geometry, nothing learned.
        """
        tip1_w, tip2_w = stick_tip_positions_w(
            self._env, self.cfg.stick1_cfg, self.cfg.stick2_cfg, self.cfg.tip_offset_o
        )
        midpoint_w = 0.5 * (tip1_w[env_ids] + tip2_w[env_ids])
        root_pos = self._robot.data.root_link_pos_w[env_ids]
        root_quat = _normalize_quat(self._robot.data.root_link_quat_w[env_ids])
        offset_root = quat_apply_inverse(root_quat, midpoint_w - root_pos)
        cube_w = self._env.scene[self.cfg.cube_cfg.name].data.root_pos_w[env_ids]
        # Aim slightly past the cube; see ``tip_target_offset_w``.
        target_w = cube_w + torch.as_tensor(
            self.cfg.tip_target_offset_w, dtype=cube_w.dtype, device=cube_w.device
        )
        # ``goal_root_offset_r`` shifts the solved root pose along the hand's
        # own axes at the goal orientation, so "+z by 7 mm" stays 7 mm along
        # the root link's local +z however the hand is turned.  Folding it into
        # the same rotation is exact:
        #     p_goal_w = target_w - q_goal . d_r + q_goal . o_r
        #              = target_w - q_goal . (d_r - o_r)
        offset_root = offset_root - torch.as_tensor(
            self.cfg.goal_root_offset_r, dtype=cube_w.dtype, device=cube_w.device
        )
        return target_w - quat_apply(quat_goal, offset_root)

    def _refresh_goal_position(self, alpha: torch.Tensor) -> None:
        """Re-solve the goal while the hand is still lining up, then freeze.

        Measuring once at reset is not enough: the fingers keep acting for the
        whole 2 s approach, so the sticks shift in the hand and the offset the
        goal was solved from goes stale.  Re-solving every step keeps the target
        matched to where the sticks *currently* are.

        Frozen once the descent begins (``alpha >= approach_fraction``).  Stage 2
        is a pure vertical drop onto the cube and it is the phase where the tips
        actually arrive; letting the target keep moving there would have the root
        chasing the sticks at exactly the moment precision matters.  Stage 1 is
        a coarse traverse at spawn height, so refining through it is free.

        The correction is negative feedback - the goal moves opposite to the
        offset drift - so it converges rather than running away.
        """
        if not getattr(self.cfg, "adaptive_position", False):
            return
        active = (alpha < self.cfg.schedule.approach_fraction).squeeze(-1)
        env_ids = torch.nonzero(active).flatten()
        if env_ids.numel() == 0:
            return
        goal = self._solve_goal_position(env_ids, self._quat_goal[env_ids])
        self._position_goal[env_ids] = goal
        self.metrics["goal_drift_during_approach"][env_ids] = torch.linalg.vector_norm(
            goal - self._reset_goal[env_ids], dim=-1
        )

    def _update_command(self) -> None:
        """Advance the pose trajectory, with the rotation finished by stage 1.

        The parent drives orientation off ``self._alpha`` directly, which would
        keep the hand turning all the way through the descent.  Here the turn has
        to be over before the hand comes down, so the orientation runs on its own
        ``alpha / approach_fraction`` clock while ``self._alpha`` stays the
        *overall* move progress that :attr:`target_pos_w` needs.
        """
        if self._manual_override:
            # The keyboard owns the buffers in manual play; leaving them alone
            # is what stops the scripted trajectory from fighting the operator.
            self._phase[:] = 0
            return
        schedule = self.cfg.schedule
        elapsed = self.elapsed_time

        overall = torch.clamp(
            (elapsed - schedule.initial_hold_time_s)
            / schedule.rotation_interpolation_time_s,
            min=0.0,
            max=1.0,
        )
        self._alpha[:] = overall.unsqueeze(-1)
        self._refresh_goal_position(self._alpha)

        # CLOSE 시작 시점(= 접근이 끝난 시점)에 한 번만 latch.
        newly = (~self._close_start_latched) & (
            elapsed >= schedule.open_close_start_time_s
        )
        if bool(newly.any()):
            tip1_w, tip2_w = stick_tip_positions_w(
                self._env, self.cfg.stick1_cfg, self.cfg.stick2_cfg, self.cfg.tip_offset_o
            )
            midpoint_w = 0.5 * (tip1_w + tip2_w)
            cube_w = self._env.scene[self.cfg.cube_cfg.name].data.root_pos_w
            tip_gap = torch.linalg.vector_norm(midpoint_w - cube_w, dim=-1)
            root_gap = torch.linalg.vector_norm(
                self._robot.data.root_link_pos_w - self._position_goal, dim=-1
            )
            self.metrics["tip_to_cube_at_close_start"] = torch.where(
                newly, tip_gap, self.metrics["tip_to_cube_at_close_start"]
            )
            self.metrics["position_error_at_close_start"] = torch.where(
                newly, root_gap, self.metrics["position_error_at_close_start"]
            )
            self._close_start_latched |= newly

        orientation_raw = torch.clamp(
            overall / schedule.approach_fraction, min=0.0, max=1.0
        )
        if schedule.use_smoothstep:
            orientation_alpha = self._smoothstep(orientation_raw)
        else:
            orientation_alpha = orientation_raw

        if schedule.use_slerp:
            self._quat_command[:] = quat_slerp_batch(
                self._quat_start, self._quat_goal, orientation_alpha.unsqueeze(-1)
            )
        else:
            use_goal = (orientation_raw >= 1.0).unsqueeze(-1)
            self._quat_command[:] = torch.where(
                use_goal, self._quat_goal, self._quat_start
            )

        self._phase[:] = self._compute_phase(elapsed)

    @property
    def position_goal_w(self) -> torch.Tensor:
        return self._position_goal

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Capture the reset pose and aim at the one calibrated goal pose.

        Runs after the ``reset``-mode events, so the root pose read here is the
        post-``reset_to_functional_pregrasp`` one.  Nothing is sampled: every
        environment and every episode flies to the same place, which is what
        makes a *fixed* cube position meaningful.
        """
        count = len(env_ids)
        if count == 0:
            return
        schedule = self.cfg.schedule
        self._close_start_latched[env_ids] = False

        # Lazily sized: the parent has no position-goal buffer.
        if not hasattr(self, "_position_goal"):
            self._position_goal = torch.zeros((self.num_envs, 3), device=self.device)

        position_start = self._robot.data.root_link_pos_w[env_ids].clone()
        self._position_start[env_ids] = position_start
        quat_start = _normalize_quat(self._robot.data.root_link_quat_w[env_ids].clone())
        self._quat_start[env_ids] = quat_start

        if self._manual_override:
            # Calibration / manual play: the operator owns the target, so the
            # goal is seeded to the reset pose and left alone.
            self._quat_goal[env_ids] = quat_start
            self._quat_command[env_ids] = quat_start
            self._position_goal[env_ids] = position_start
            self._delta_euler[env_ids] = 0.0
            self._alpha[env_ids] = 0.0
            self._phase[env_ids] = 0
            self._previous_elapsed[env_ids] = -1.0
            self.metrics["adaptive_goal_shift"][env_ids] = 0.0
            self.metrics["goal_drift_during_approach"][env_ids] = 0.0
            self._reset_goal[env_ids] = position_start
            return

        target_pos_e = schedule.target_root_pos_e
        target_euler = schedule.target_euler_rad
        if target_pos_e is None or target_euler is None:
            # Uncalibrated, and the operator has said so by passing
            # ``env.require_calibration=false`` - the ``__init__`` guard is what
            # stops a *training* run from getting here.  Raising again would
            # make the calibration session impossible to start: the manual
            # override is enabled by ``attach()``, which runs after the wrapper
            # has already triggered this first reset.  So hold the spawn pose
            # and let the keyboard take over a moment later.
            if not self._uncalibrated_warned:
                self._uncalibrated_warned = True
                print(
                    "[WARN] hand_object: goal root pose is UNCALIBRATED; the scripted"
                    " trajectory will hold the spawn pose. Fly the hand with the"
                    " keyboard and press P to read the values to paste.",
                    flush=True,
                )
            self._quat_goal[env_ids] = quat_start
            self._quat_command[env_ids] = quat_start
            self._position_goal[env_ids] = position_start
            self._delta_euler[env_ids] = 0.0
            self._alpha[env_ids] = 0.0
            self._phase[env_ids] = 0
            self._previous_elapsed[env_ids] = -1.0
            self.metrics["adaptive_goal_shift"][env_ids] = 0.0
            self.metrics["goal_drift_during_approach"][env_ids] = 0.0
            self._reset_goal[env_ids] = position_start
            return

        # env-local -> world.  Storing a world coordinate in the constant would
        # put every environment except env 0 in the wrong place.
        goal_local = torch.as_tensor(
            target_pos_e, dtype=position_start.dtype, device=self.device
        )
        fixed_goal = self._env.scene.env_origins[env_ids] + goal_local

        delta = torch.as_tensor(
            target_euler, dtype=position_start.dtype, device=self.device
        ).expand(count, -1).clone()
        self._delta_euler[env_ids] = delta

        quat_delta = quat_from_euler_xyz(delta[:, 0], delta[:, 1], delta[:, 2])
        # q_goal = q_start (x) q_delta: the rotation is about the hand's own
        # axes, matching hand_move and matching what the calibration keys do.
        quat_goal = _normalize_quat(quat_mul(quat_start, quat_delta))
        self._quat_goal[env_ids] = quat_goal
        self._quat_command[env_ids] = quat_start

        # ------------------------------------------------------------------
        # Adaptive goal position (2026-08-08).
        #
        # The tip midpoint is not something this command controls.  It is
        #     root pose  x  where the sticks happen to sit in the hand,
        # and the second factor varies by millimetres every episode
        # (``stick1_pivot_error`` 1.5-5.0 mm, ``tip_lateral_error`` 11-15 mm -
        # both larger than the 5 mm cube).  Flying to a *fixed* root pose
        # therefore lands the tips somewhere different each time, and the policy
        # cannot correct for it: the 103D observation is proprioceptive only, so
        # it never learns where the cube is relative to the tips.
        #
        # So the goal is inverted instead of memorised.  Measure the offset from
        # the root to the tip midpoint *now* (the sticks have just been placed by
        # ``reset_to_functional_pregrasp``), express it in the root frame, and
        # solve for the root position that puts that midpoint on the cube:
        #
        #     d_r      = q_start^-1 (m_w - p_start)      offset, root frame
        #     p_goal_w = cube_w - q_goal d_r
        #
        # Rigid-offset inverse geometry, nothing learned.  The observation, the
        # action and every reward stay exactly as they were - only the number the
        # scripted trajectory flies to changes, and it now depends on the episode.
        #
        # ``HAND_OBJECT_TARGET_ROOT_POS_E`` is kept as the fallback and as the
        # reference the ``adaptive_goal_shift`` metric measures against, so how
        # much this is actually correcting stays visible.
        if getattr(self.cfg, "adaptive_position", False):
            adaptive_goal = self._solve_goal_position(env_ids, quat_goal)
            self._position_goal[env_ids] = adaptive_goal
            self._reset_goal[env_ids] = adaptive_goal
            self.metrics["adaptive_goal_shift"][env_ids] = torch.linalg.vector_norm(
                adaptive_goal - fixed_goal, dim=-1
            )
        else:
            self._position_goal[env_ids] = fixed_goal
            self._reset_goal[env_ids] = fixed_goal
            self.metrics["adaptive_goal_shift"][env_ids] = 0.0
        self.metrics["goal_drift_during_approach"][env_ids] = 0.0

        self._alpha[env_ids] = 0.0
        self._phase[env_ids] = 0
        self._previous_elapsed[env_ids] = -1.0

        self.metrics["goal_delta_roll"][env_ids] = delta[:, 0]
        self.metrics["goal_delta_pitch"][env_ids] = delta[:, 1]
        self.metrics["goal_delta_yaw"][env_ids] = delta[:, 2]
        self.metrics["goal_geodesic_angle"][env_ids] = _quat_geodesic_angle(
            quat_goal, quat_start
        )


@configclass
class HandObjectRootOrientationCommandCfg(HandMoveRootOrientationCommandCfg):
    """Configuration for :class:`HandObjectRootOrientationCommand`."""

    class_type: type = HandObjectRootOrientationCommand
    schedule: HandObjectScheduleCfg = HAND_OBJECT_SCHEDULE

    # Solve the goal root position from where the sticks actually are, instead
    # of flying to a memorised constant.  See ``_resample_command`` for why.
    # Set False to fall back to ``schedule.target_root_pos_e``.
    adaptive_position: bool = True
    stick1_cfg: SceneEntityCfg = SceneEntityCfg("stick1")
    stick2_cfg: SceneEntityCfg = SceneEntityCfg("stick2")
    cube_cfg: SceneEntityCfg = SceneEntityCfg("object")
    tip_offset_o: tuple[float, float, float] = (0.0, 0.09, 0.0)

    # Where to aim the tip midpoint, as a **world-frame** offset from the cube
    # centre.  Not a calibration of the geometry - the adaptive solve already
    # lands the midpoint on the cube - but a deliberate overshoot found by hand
    # to make the grip more reliable (2026-08-08).
    #
    # World frame on purpose: this was found with the keyboard, and the manual
    # translation keys are world-axis (`hand_move_manual_control.py:50-61`,
    # `J = (axis 1, +1.0)`).  So "+1 cm in the J direction" is exactly
    # ``(0.0, +0.01, 0.0)`` here, independent of how the hand is oriented.
    tip_target_offset_w: tuple[float, float, float] = (0.0, 0.01, 0.0)

    # Extra translation of the solved goal root position, expressed in the
    # **hand's own frame at the goal orientation** - unlike
    # ``tip_target_offset_w`` above, which is world-frame.
    #
    # Use this when the correction is "push the hand further along the way the
    # fingers point", which is a body-frame statement: the root link's local +z
    # is the finger-extension direction (world +x at reset, see HAND_ROOT_ROT in
    # hand_grasp_env_cfg.py), local +x is the palm-plane normal.
    #
    # Zero here keeps ``hand_object`` on its 2026-08-08 geometry; ``hand_final``
    # overrides it.
    goal_root_offset_r: tuple[float, float, float] = (0.0, 0.0, 0.0)
    debug_vis: bool = HAND_OBJECT_SCHEDULE.debug_vis


class HandObjectOpenCloseCommand(HandMoveOpenCloseCommand):
    """OPEN through the rotation, then CLOSE once and hold.

    ``hand_move`` starts CLOSE and alternates every 2 s.  Here the hand has to
    arrive at the cube with the tips *apart*, so the episode starts OPEN and
    switches to CLOSE exactly once, at ``open_close_start_time_s``.  The switch
    is unconditional - grasp quality never gates it, exactly as in ``hand_move``.

    The one-hot encoding, the command name and the manual override are all
    inherited, so the 103D observation and every reward binding are unchanged.
    """

    cfg: HandObjectOpenCloseCommandCfg

    def __str__(self) -> str:
        schedule = self.cfg.schedule
        return (
            "HandObjectOpenCloseCommand:\n"
            f"\tOPEN until {schedule.open_close_start_time_s}s, then CLOSE and hold"
        )

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        count = len(env_ids)
        if count == 0:
            return
        if self._manual_override:
            # Keep the operator's latched selection across resets.
            self._command[env_ids, OPEN_INDEX] = float(self._manual_mode_index == OPEN_INDEX)
            self._command[env_ids, CLOSE_INDEX] = float(self._manual_mode_index == CLOSE_INDEX)
            self._segment[env_ids] = 0
            return
        # Episodes begin OPEN - the opposite of hand_move.
        self._first_mode_open[env_ids] = True
        self._command[env_ids, OPEN_INDEX] = 1.0
        self._command[env_ids, CLOSE_INDEX] = 0.0
        self._segment[env_ids] = 0

    def _update_command(self) -> None:
        if self._manual_override:
            return
        schedule = self.cfg.schedule
        elapsed = self._env.episode_length_buf.float() * self._env.step_dt

        close = elapsed >= schedule.open_close_start_time_s
        self._command[:, OPEN_INDEX] = (~close).float()
        self._command[:, CLOSE_INDEX] = close.float()
        self._segment[:] = close.long()


@configclass
class HandObjectOpenCloseCommandCfg(HandMoveOpenCloseCommandCfg):
    """Configuration for :class:`HandObjectOpenCloseCommand`."""

    class_type: type = HandObjectOpenCloseCommand
    schedule: HandObjectScheduleCfg = HAND_OBJECT_SCHEDULE


class HandObjectSupportCommand(CommandTerm):
    """Lowers the cube's support column so the grasp has to hold on its own.

    While the column is up, a cube resting on it stays put whether or not the
    chopsticks are really gripping, so every grasp reward is satisfiable by
    doing nothing.  Taking the column away is what turns "looks like a grasp"
    into "is a grasp", which is why this term exists at all.

    **When** it moves: as soon as the grasp is real, and at
    ``support_retract_deadline_time_s`` at the latest.  A fixed time alone is
    wrong because the column can then vanish while the fingers are still
    closing, which asks the policy to hold something it never got hold of.  A
    condition alone is worse: never gripping would mean never being tested, so
    the cube could be ignored for a whole episode while
    ``bilateral_cube_force`` was farmed at zero risk.  Condition-or-deadline
    keeps both properties - the test always happens, and a policy that already
    has the cube is not punished for the clock.

    The trigger is **latched per environment**: once the column starts moving
    it never goes back up, however the grasp evolves afterwards.  Letting it
    rise again would hand the load back mid-episode and make the hold reward
    meaningless.

    How it moves, and why not some other way:

    * **Lowered, not deleted, and never switched non-colliding.**  Removing a
      prim mid-episode or flipping its collision off makes the cube's support
      vanish between two physics steps; the cube then starts the hold phase
      already falling.  Sliding it away over ``support_retract_duration_s``
      hands the load over gradually.
    * **Per-environment state.**  ``_current_offset`` is ``(num_envs,)``.  Every
      environment reaches its own retract phase from its own episode clock, so
      one environment resetting must not move another one's column.
    * **Writes only while something changes.**  Once the column is fully down
      the term stops writing poses; a kinematic body simply stays where it was
      last put, and ``reset_scene_to_default`` restores it on reset.

    ``retracted_gate`` is what the hold reward and the drop termination read:
    1.0 only once the column is *fully* clear, never part-way through.
    """

    cfg: HandObjectSupportCommandCfg

    def __init__(self, cfg: HandObjectSupportCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.cfg.schedule.validate()
        self._support: RigidObject = env.scene[cfg.support_name]

        if not 0.0 < cfg.retract_trigger_force_fraction <= 1.0:
            raise ValueError(
                "HandObjectSupportCommandCfg.retract_trigger_force_fraction must lie "
                f"in (0, 1] (got {cfg.retract_trigger_force_fraction}); it is the "
                "share of force_saturation that counts as 'gripped'."
            )
        if cfg.retract_trigger_debounce_steps < 1:
            raise ValueError(
                "HandObjectSupportCommandCfg.retract_trigger_debounce_steps must be "
                "at least 1; a single noisy contact step must not drop the column."
            )

        num_envs, device = self.num_envs, self.device
        # Metres already travelled downwards, per environment.
        self._current_offset = torch.zeros(num_envs, device=device)
        self._last_written_offset = torch.zeros(num_envs, device=device)
        # Dummy command: this term drives the scene, it does not feed the policy.
        self._command = torch.zeros(num_envs, 1, device=device)

        # Retract trigger state, all per environment.
        self._retract_started = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._trigger_streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._trigger_time_s = torch.zeros(num_envs, device=device)
        self._triggered_by_grasp = torch.zeros(num_envs, device=device)
        self._force_threshold = (
            cfg.retract_trigger_force_fraction * cfg.force_saturation
        )

        for name in (
            "retract_progress",
            "retracted_fraction",
            "support_offset_m",
            # When the column started moving, and whether the grasp or the clock
            # started it.  These two answer "is the policy ever gripping before
            # the deadline?" directly, which is the thing the deadline hides.
            "retract_trigger_time_s",
            "retract_by_grasp",
        ):
            self.metrics[name] = torch.zeros(num_envs, device=device)

        # Manual play / calibration only; training never touches these.
        self._manual_override = False
        self._manual_retracting = False

    def __str__(self) -> str:
        schedule = self.cfg.schedule
        return (
            "HandObjectSupportCommand:\n"
            f"\tretract {schedule.support_retract_distance_m * 1000:.0f}mm over "
            f"{schedule.support_retract_duration_s}s\n"
            f"\ttriggered by min(inward) >= {self._force_threshold:.4f}N "
            f"({self.cfg.retract_trigger_force_fraction:.2f} x saturation) for "
            f"{self.cfg.retract_trigger_debounce_steps} steps in CLOSE,\n"
            f"\tor at {schedule.support_retract_deadline_time_s}s at the latest"
        )

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """Not a policy command; present because ``CommandTerm`` requires it."""
        return self._command

    @property
    def retract_progress(self) -> torch.Tensor:
        """0 before the retract, 1 once the column is fully down. Shape ``(N,)``."""
        return self._current_offset / self.cfg.schedule.support_retract_distance_m

    @property
    def retracted_gate(self) -> torch.Tensor:
        """1.0 only where the column is completely clear of the cube."""
        return (self.retract_progress >= 1.0 - 1.0e-6).float()

    @property
    def support_offset_m(self) -> torch.Tensor:
        return self._current_offset

    """
    Manual play / calibration override.
    """

    @property
    def manual_override(self) -> bool:
        return self._manual_override

    def enable_manual_override(self, enabled: bool = True) -> None:
        """Stop the timed retract; the operator triggers it with a key instead.

        Without this, a manual play session would lose its support at the deadline no
        matter where the operator had flown the hand to, and the cube would be
        on the floor before it could be approached.
        """
        self._manual_override = bool(enabled)
        if self._manual_override:
            self._manual_retracting = False

    def trigger_manual_retract(self, retracting: bool = True) -> None:
        """Start (or cancel) the retract by hand. Ignored outside manual mode."""
        self._manual_retracting = bool(retracting)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Put the column back up for the environments that just reset.

        ``reset_scene_to_default`` has already written the support's
        ``init_state`` pose, so only the bookkeeping has to follow it back to
        zero; writing the pose again here would be redundant.
        """
        if len(env_ids) == 0:
            return
        self._current_offset[env_ids] = 0.0
        self._last_written_offset[env_ids] = 0.0
        self._retract_started[env_ids] = False
        self._trigger_streak[env_ids] = 0
        self._trigger_time_s[env_ids] = 0.0
        self._triggered_by_grasp[env_ids] = 0.0
        if self._manual_override:
            self._manual_retracting = False

    def _grasp_is_real(self) -> torch.Tensor:
        """``True`` where both sticks are genuinely loaded against the cube.

        Uses the same closing-axis projection as the force reward, so "gripped
        enough for the column to go" and "rewarded for gripping" cannot drift
        apart.  The threshold is expressed as a *fraction* of
        ``force_saturation`` rather than as its own newton value: that constant
        is still provisional, and a second independently-guessed force number
        would be one more thing to calibrate.
        """
        cfg = self.cfg
        close_gate = self._env.command_manager.get_command(
            cfg.open_close_command_name
        )[:, CLOSE_INDEX]
        inward1, inward2, _, _, _ = cube_inward_forces(
            self._env,
            cfg.stick1_cfg,
            cfg.stick2_cfg,
            cfg.tip_offset_o,
            cfg.stick1_sensor_name,
            cfg.stick2_sensor_name,
        )
        # ``minimum`` for the same reason the reward uses it: one stick mashing
        # the cube into the column is not a grasp, and pulling the column out
        # from under that would just drop the cube.
        return (close_gate > 0.5) & (
            torch.minimum(inward1, inward2) >= self._force_threshold
        )

    def _update_command(self) -> None:
        schedule = self.cfg.schedule
        distance = schedule.support_retract_distance_m
        step = distance * self._env.step_dt / schedule.support_retract_duration_s

        if self._manual_override:
            if self._manual_retracting:
                target = torch.clamp(self._current_offset + step, max=distance)
            else:
                target = self._current_offset
        else:
            elapsed = self._env.episode_length_buf.float() * self._env.step_dt

            gripped = self._grasp_is_real()
            self._trigger_streak = torch.where(
                gripped,
                self._trigger_streak + 1,
                torch.zeros_like(self._trigger_streak),
            )
            by_grasp = self._trigger_streak >= self.cfg.retract_trigger_debounce_steps
            by_deadline = elapsed >= schedule.support_retract_deadline_time_s

            # Latch: record why and when, but only on the step it first fires.
            newly = (~self._retract_started) & (by_grasp | by_deadline)
            self._trigger_time_s = torch.where(newly, elapsed, self._trigger_time_s)
            self._triggered_by_grasp = torch.where(
                newly, (by_grasp & ~by_deadline).float(), self._triggered_by_grasp
            )
            self._retract_started |= by_grasp | by_deadline

            # Incremental, so the offset is monotone by construction: the column
            # never rises again even if the grasp is lost a step later.
            target = torch.where(
                self._retract_started,
                torch.clamp(self._current_offset + step, max=distance),
                self._current_offset,
            )

        self._current_offset[:] = target
        self._write_support_pose()

    def _write_support_pose(self) -> None:
        """Push the current offset into the sim, but only when it moved.

        A kinematic body keeps whatever pose it was last given, so re-writing an
        unchanged pose every step is pure cost.  During the 0.5 s retract this
        writes on 15 consecutive steps and then goes quiet.
        """
        moved = (self._current_offset - self._last_written_offset).abs() > 1.0e-9
        if not bool(moved.any()):
            return
        default = self._support.data.default_root_state.clone()
        pose = default[:, :7]
        pose[:, :3] += self._env.scene.env_origins
        pose[:, 2] -= self._current_offset
        self._support.write_root_pose_to_sim(pose)
        self._last_written_offset[:] = self._current_offset

    def _update_metrics(self) -> None:
        self.metrics["retract_progress"][:] = self.retract_progress
        self.metrics["retracted_fraction"][:] = self.retracted_gate
        self.metrics["support_offset_m"][:] = self._current_offset
        self.metrics["retract_trigger_time_s"][:] = self._trigger_time_s
        self.metrics["retract_by_grasp"][:] = self._triggered_by_grasp


@configclass
class HandObjectSupportCommandCfg(CommandTermCfg):
    """Configuration for :class:`HandObjectSupportCommand`."""

    class_type: type = HandObjectSupportCommand

    support_name: str = "object_support"
    schedule: HandObjectScheduleCfg = HAND_OBJECT_SCHEDULE

    # -- Retract trigger.  The entities below are only read to recompute the
    #    same inward forces the reward uses; they are filled in from the env cfg
    #    so the stick names, tip offset and sensor names have one definition.
    #
    #    Note these are *not* resolved by the manager: only term ``params``
    #    dicts get ``SceneEntityCfg.resolve()`` called on them
    #    (``manager_base.py:398``).  Nothing here needs ``body_ids``, so plain
    #    names are enough - do not add a body-name lookup without resolving.
    open_close_command_name: str = "open_close"
    stick1_cfg: SceneEntityCfg = MISSING
    stick2_cfg: SceneEntityCfg = MISSING
    tip_offset_o: tuple[float, float, float] = MISSING
    stick1_sensor_name: str = MISSING
    stick2_sensor_name: str = MISSING
    force_saturation: float = HAND_OBJECT_FORCE_SATURATION_N

    # Share of ``force_saturation`` that counts as "actually gripped".  A
    # fraction and not its own newton constant, so it inherits whatever
    # calibration ``force_saturation`` eventually gets instead of adding a
    # second unmeasured number.
    retract_trigger_force_fraction: float = 0.5
    # Consecutive policy steps the condition must hold.  Contact forces flicker
    # as the tips settle; one lucky step must not pull the column away.
    retract_trigger_debounce_steps: int = 50

    # Deterministic function of elapsed time; the timer must never fire.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False


# ---------------------------------------------------------------------------
# Geometry and force helpers.  Shared by rewards, terminations and metrics so
# there is exactly one definition of "the closing axis" and "the inward force".
# ---------------------------------------------------------------------------


def stick_tip_positions_w(
    env: ManagerBasedEnv,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    tip_offset_o: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """World positions of both distal chopstick tips, shape ``(N, 3)`` each.

    ``tip_offset_o`` is the same local ``+y`` offset the ``hand_grasp`` rewards
    use, so "tip" means the same point everywhere in the project.
    """
    stick1: RigidObject = env.scene[stick1_cfg.name]
    stick2: RigidObject = env.scene[stick2_cfg.name]
    offset = torch.as_tensor(
        tip_offset_o, dtype=stick1.data.root_pos_w.dtype, device=stick1.data.root_pos_w.device
    ).expand(env.num_envs, -1)
    tip1 = stick1.data.root_pos_w + quat_apply(stick1.data.root_quat_w, offset)
    tip2 = stick2.data.root_pos_w + quat_apply(stick2.data.root_quat_w, offset)
    return tip1, tip2


def closing_axis_w(tip1_w: torch.Tensor, tip2_w: torch.Tensor) -> torch.Tensor:
    """Unit vector from the Stick2 tip towards the Stick1 tip, shape ``(N, 3)``.

    Stick2 is the passive reference rail and Stick1 is the stick that closes,
    so this points "the way Stick1 came from" - the direction the cube's
    reaction pushes Stick1 when the pinch is real.

    The tips can coincide (a fully closed empty grasp), so the norm is clamped
    rather than divided by directly; a degenerate axis yields a zero-ish vector
    whose dot products are ~0, i.e. no force reward, which is the right answer.
    """
    delta = tip1_w - tip2_w
    norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    return delta / torch.clamp(norm, min=1.0e-8)


def _filtered_contact_force_w(env: ManagerBasedEnv, sensor_name: str) -> torch.Tensor:
    """Sum of a sensor's cube-filtered contact forces, shape ``(N, 3)``.

    ``force_matrix_w`` is ``(N, B, M, 3)`` - bodies by filtered bodies.  Both B
    and M are 1 here (one stick, one cube), but summing over them keeps this
    correct if either ever gains an entry.  Returns zeros when the sensor has
    no filter configured, so a mis-wired scene degrades to "no force" instead
    of raising inside the reward manager.
    """
    sensor = env.scene.sensors[sensor_name]
    force_matrix = sensor.data.force_matrix_w
    if force_matrix is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    return force_matrix.sum(dim=(1, 2))


def cube_inward_forces(
    env: ManagerBasedEnv,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    tip_offset_o: tuple[float, float, float],
    stick1_sensor_name: str,
    stick2_sensor_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve both stick-cube contact forces onto the closing axis.

    Returns ``(inward1, inward2, axis, raw1, raw2)`` where the inward forces are
    scalars in newtons, positive when that stick is genuinely pressing the cube.

    The projection is the point of this function.  ``norm(force)`` would reward
    a stick that drags the cube sideways or shoves it off the support just as
    much as one that squeezes it; only the component along the line joining the
    two tips actually contributes to a pinch.
    """
    tip1, tip2 = stick_tip_positions_w(env, stick1_cfg, stick2_cfg, tip_offset_o)
    axis = closing_axis_w(tip1, tip2)
    raw1 = _filtered_contact_force_w(env, stick1_sensor_name)
    raw2 = _filtered_contact_force_w(env, stick2_sensor_name)
    inward1 = STICK1_CUBE_FORCE_SIGN * torch.sum(raw1 * axis, dim=-1)
    inward2 = STICK2_CUBE_FORCE_SIGN * torch.sum(raw2 * axis, dim=-1)
    return inward1, inward2, axis, raw1, raw2


def bilateral_force_score(
    inward1: torch.Tensor,
    inward2: torch.Tensor,
    force_saturation: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(score1, score2, min(score1, score2))``, each in ``[0, 1]``.

    ``minimum`` and not a mean or a sum: a pinch needs *both* sticks loaded.
    A mean would pay full marks for one stick mashing the cube against the
    support with the other nowhere near it, which is exactly the degenerate
    behaviour this task has to avoid.  Taking the weaker of the two makes the
    reward equal to "how much of a real pinch is there".

    Clamping at 1 makes the reward saturate: past ``force_saturation`` extra
    squeeze earns nothing, so there is no gradient pushing the policy to crush
    the cube, and no separate over-force penalty is needed yet.
    """
    scale = max(float(force_saturation), 1.0e-9)
    score1 = torch.clamp(inward1 / scale, min=0.0, max=1.0)
    score2 = torch.clamp(inward2 / scale, min=0.0, max=1.0)
    return score1, score2, torch.minimum(score1, score2)


def cube_relative_state(
    env: ManagerBasedEnv,
    cube_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    tip_offset_o: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cube pose/velocity **relative to the Stick2 tip**.

    Returns ``(distance, linear_speed, angular_speed)``.

    Everything is relative on purpose.  The hand is free-floating and the whole
    point of the task is that the grasp survives the hand being moved, so a
    reward built on absolute world position would punish exactly the motion it
    is supposed to allow.  Stick2 is the passive reference rail, which makes it
    the natural frame: if the cube is still where it was against Stick2, the
    grasp held.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    stick2: RigidObject = env.scene[stick2_cfg.name]
    offset = torch.as_tensor(
        tip_offset_o, dtype=cube.data.root_pos_w.dtype, device=cube.data.root_pos_w.device
    ).expand(env.num_envs, -1)
    tip2 = stick2.data.root_pos_w + quat_apply(stick2.data.root_quat_w, offset)

    distance = torch.linalg.vector_norm(cube.data.root_pos_w - tip2, dim=-1)
    linear_speed = torch.linalg.vector_norm(
        cube.data.root_lin_vel_w - stick2.data.root_lin_vel_w, dim=-1
    )
    angular_speed = torch.linalg.vector_norm(
        cube.data.root_ang_vel_w - stick2.data.root_ang_vel_w, dim=-1
    )
    return distance, linear_speed, angular_speed


# ---------------------------------------------------------------------------
# Rewards.
# ---------------------------------------------------------------------------


def functional_grasp_gate(
    env: ManagerBasedRLEnv,
    sensor_groups: tuple[tuple[str, ...], ...],
    force_scale: float = HAND_OBJECT_FUNCTIONAL_FORCE_SCALE_N,
) -> torch.Tensor:
    """Return the weakest normalized hand--stick functional contact.

    The minimum makes all six semantic contacts necessary.  Unlike a hard
    boolean threshold, the clipped 0--1 score still gives a recovery gradient
    while a weak contact is being rebuilt; losing any contact completely makes
    every cube reward zero.
    """
    if force_scale <= 0.0:
        raise ValueError("force_scale must be positive")
    return torch.min(
        torch.clamp(
            _group_forces(env, sensor_groups) / force_scale,
            min=0.0,
            max=1.0,
        ),
        dim=-1,
    ).values


def bilateral_cube_force(
    env: ManagerBasedRLEnv,
    command_name: str,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    cube_cfg: SceneEntityCfg,
    tip_offset_o: tuple[float, float, float],
    stick1_sensor_name: str,
    stick2_sensor_name: str,
    functional_sensor_groups: tuple[tuple[str, ...], ...],
    force_saturation: float = HAND_OBJECT_FORCE_SATURATION_N,
    functional_force_scale: float = HAND_OBJECT_FUNCTIONAL_FORCE_SCALE_N,
) -> torch.Tensor:
    """Reward a two-sided squeeze only while the six-stick grasp survives.

    Deliberately *not* gated on the support being retracted: the squeeze has to
    be learned while the cube is still supported, otherwise the policy has no
    way to discover the grasp before the floor drops out from under it.

    ``cube_cfg`` is unused in the computation and kept in the signature so the
    reward manager resolves and validates the cube entity along with the rest -
    a scene missing the cube then fails at construction, not mid-run.
    """
    del cube_cfg
    close_gate = env.command_manager.get_command(command_name)[:, CLOSE_INDEX]
    inward1, inward2, _, _, _ = cube_inward_forces(
        env, stick1_cfg, stick2_cfg, tip_offset_o, stick1_sensor_name, stick2_sensor_name
    )
    _, _, score = bilateral_force_score(inward1, inward2, force_saturation)
    grasp_gate = functional_grasp_gate(
        env,
        functional_sensor_groups,
        functional_force_scale,
    )
    return close_gate * grasp_gate * score


def bilateral_cube_contact_acquisition(
    env: ManagerBasedRLEnv,
    command_name: str,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    tip_offset_o: tuple[float, float, float],
    stick1_sensor_name: str,
    stick2_sensor_name: str,
    force_saturation: float = HAND_OBJECT_CONTACT_ACQUISITION_SATURATION_N,
) -> torch.Tensor:
    """Reward two-sided raw cube contact before a directional pinch exists.

    Taking the minimum requires both sticks to load the cube.  Force direction
    is intentionally ignored only in this low-weight acquisition bridge; the
    existing bilateral force and hold rewards retain the strict closing-axis
    projection used by the final objective.
    """
    if force_saturation <= 0.0:
        raise ValueError("force_saturation must be positive")
    close_gate = env.command_manager.get_command(command_name)[:, CLOSE_INDEX]
    _, _, _, raw1, raw2 = cube_inward_forces(
        env,
        stick1_cfg,
        stick2_cfg,
        tip_offset_o,
        stick1_sensor_name,
        stick2_sensor_name,
    )
    score1 = torch.clamp(
        torch.linalg.vector_norm(raw1, dim=-1) / force_saturation,
        min=0.0,
        max=1.0,
    )
    score2 = torch.clamp(
        torch.linalg.vector_norm(raw2, dim=-1) / force_saturation,
        min=0.0,
        max=1.0,
    )
    return close_gate * torch.minimum(score1, score2)


def cube_relative_stability(
    env: ManagerBasedRLEnv,
    support_command_name: str,
    cube_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    tip_offset_o: tuple[float, float, float],
    functional_sensor_groups: tuple[tuple[str, ...], ...],
    reference_distance_sigma: float = 0.01,
    linear_speed_sigma: float = 0.05,
    functional_force_scale: float = HAND_OBJECT_FUNCTIONAL_FORCE_SCALE_N,
) -> torch.Tensor:
    """Reward a stable cube only with support gone and six contacts intact.

    Gated on the support being *fully* down.  Before that the column is holding
    the cube steady all by itself, so a reward paid then would be measuring the
    scenery rather than the grasp.

    The relative *angular* speed was dropped on 2026-08-07 for the same reason
    it was dropped from :func:`cube_hold` - see that docstring.  In short, the
    5 mm / 3 g cube has ``I = 1.25e-8 kg m^2``, so contact jitter alone reads
    ~10 rad/s, and the quantity is a norm so shaking that nets to zero rotation
    still averages large.  Both terms have to drop it together: leaving it here
    would make "the cube is steady" mean two different things in two rewards
    that are supposed to describe the same state.
    """
    support = env.command_manager.get_term(support_command_name)
    distance, linear_speed, _ = cube_relative_state(
        env, cube_cfg, stick2_cfg, tip_offset_o
    )
    score = torch.exp(
        -distance / reference_distance_sigma
        - linear_speed / linear_speed_sigma
    )
    grasp_gate = functional_grasp_gate(
        env,
        functional_sensor_groups,
        functional_force_scale,
    )
    return support.retracted_gate * grasp_gate * score


def cube_hold(
    env: ManagerBasedRLEnv,
    command_name: str,
    support_command_name: str,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    cube_cfg: SceneEntityCfg,
    tip_offset_o: tuple[float, float, float],
    stick1_sensor_name: str,
    stick2_sensor_name: str,
    functional_sensor_groups: tuple[tuple[str, ...], ...],
    force_saturation: float = HAND_OBJECT_FORCE_SATURATION_N,
    reference_distance_sigma: float = 0.01,
    linear_speed_sigma: float = 0.05,
    functional_force_scale: float = HAND_OBJECT_FUNCTIONAL_FORCE_SCALE_N,
) -> torch.Tensor:
    """The actual objective: functional grasp plus a stable cube hold.

    ``close x retracted x functional-grasp x cube-force x stability`` - a
    product, so every factor is necessary and none of them can be traded
    against the others.  Paid every policy step for as long as the hold lasts
    rather than once at the end, so holding longer is strictly better and
    there is no incentive to end early.

    No angular-speed factor (removed 2026-08-07)
    -------------------------------------------
    ``cube_relative_state`` also returns a relative angular speed, and it used
    to divide into this exponent with ``angular_speed_sigma = 1.0``.  It made
    the whole reward unreachable: with the force-gated metric
    ``Metrics/hand_object/hold_angular_speed`` reading **10.43 rad/s while the
    cube was genuinely pinched**, that single factor contributed 10.43 of a
    12.04 total exponent and pinned ``stability`` at 6e-6.  ``cube_hold`` earned
    0.084 of a 1800-point budget for 200 iterations.

    That reading is almost certainly not macroscopic rotation:

    * The cube is 5 mm and 3 g, so ``I = m a^2 / 6 = 1.25e-8 kg m^2``.  A 0.07 N
      contact acting 2.5 mm off centre gives ``alpha = 14,000 rad/s^2`` - one
      physics step of unbalanced contact is worth ~117 rad/s.  10 rad/s is the
      noise floor for a body this small, not a motion.
    * The quantity is ``norm(w_cube - w_stick2)``, always positive, so jitter
      that nets to zero rotation still averages to a large number.  It measures
      shaking, not turning.
    * ``hold_distance`` sat at 8.07 mm with a 8.60 mm maximum.  A cube actually
      spinning at 1.7 rev/s between square-section tips would swing the tip gap
      by up to 41% as faces and corners alternate; a 0.5 mm spread says it is
      held still.

    Penalising solver noise inside a *product* is worse than a badly tuned
    sigma: no policy can reduce it, so the objective stays at zero whatever else
    improves.  ``linear_speed`` is kept - it reads 0.040 m/s (exponent 0.80),
    a sane scale, and slipping is what actually loses the cube.
    """
    close_gate = env.command_manager.get_command(command_name)[:, CLOSE_INDEX]
    support = env.command_manager.get_term(support_command_name)
    inward1, inward2, _, _, _ = cube_inward_forces(
        env, stick1_cfg, stick2_cfg, tip_offset_o, stick1_sensor_name, stick2_sensor_name
    )
    _, _, force_score = bilateral_force_score(inward1, inward2, force_saturation)
    distance, linear_speed, _ = cube_relative_state(
        env, cube_cfg, stick2_cfg, tip_offset_o
    )
    stability = torch.exp(
        -distance / reference_distance_sigma
        - linear_speed / linear_speed_sigma
    )
    grasp_gate = functional_grasp_gate(
        env,
        functional_sensor_groups,
        functional_force_scale,
    )
    return (
        close_gate
        * support.retracted_gate
        * grasp_gate
        * force_score
        * stability
    )


# ---------------------------------------------------------------------------
# Termination.
# ---------------------------------------------------------------------------


class CubeDropped(ManagerTermBase):
    """Fail the episode once the cube is unambiguously gone.

    Three conditions have to hold together, and each one rules out a specific
    false positive:

    * **CLOSE** - during OPEN the hand is not even trying to hold anything.
    * **support fully retracted** - while the column is up the cube cannot fall,
      and half-way through the retract it is still partly supported.
    * **geometry, debounced** - the cube must be far from the Stick2 tip *or*
      well below its own reset height, for several consecutive steps.  A single
      lost contact, or one noisy step, is not a drop; sticks bobble constantly.

    There is deliberately **no drop penalty**.  Termination alone already
    removes all future reward, and a large explicit penalty is the standard way
    to teach a policy never to attempt the grasp in the first place.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        schedule = cfg.params["schedule"]
        self._debounce = int(schedule.cube_drop_debounce_steps)
        self._counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._reference_height: torch.Tensor | None = None

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._counter[:] = 0
        else:
            self._counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        support_command_name: str,
        cube_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        tip_offset_o: tuple[float, float, float],
        schedule: HandObjectScheduleCfg,
    ) -> torch.Tensor:
        cube: RigidObject = env.scene[cube_cfg.name]
        if self._reference_height is None:
            # The reset height of the cube, taken from its configured spawn so
            # it is independent of where the episode has got to.
            self._reference_height = cube.data.default_root_state[:, 2].clone()

        close_gate = env.command_manager.get_command(command_name)[:, CLOSE_INDEX] > 0.5
        support = env.command_manager.get_term(support_command_name)
        retracted = support.retracted_gate > 0.5

        distance, _, _ = cube_relative_state(env, cube_cfg, stick2_cfg, tip_offset_o)
        height = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        gone = (distance > schedule.cube_drop_distance_m) | (
            height < self._reference_height - schedule.cube_drop_height_m
        )

        candidate = close_gate & retracted & gone
        self._counter = torch.where(
            candidate, self._counter + 1, torch.zeros_like(self._counter)
        )
        return self._counter >= self._debounce


# ---------------------------------------------------------------------------
# Calibration / debug reporting.  Never called from the training loop.
# ---------------------------------------------------------------------------


def calibration_report(
    env: ManagerBasedRLEnv,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    cube_cfg: SceneEntityCfg,
    tip_offset_o: tuple[float, float, float],
    stick1_sensor_name: str,
    stick2_sensor_name: str,
    # Optional: ``hand_play`` has a table instead of the retracting column, so
    # both the prim and the command term are absent there.  Passing None drops
    # the two support lines and keeps the rest of the readout working.
    support_cfg: SceneEntityCfg | None = None,
    command_name: str = "open_close",
    support_command_name: str | None = "support",
    orientation_command_name: str = "root_orientation",
    force_saturation: float = HAND_OBJECT_FORCE_SATURATION_N,
    env_id: int = 0,
) -> str:
    """The hand geometry block, plus the cube, the support and the forces.

    Two things get measured here, and both need a human looking at them:

    1. **Where the hand has to end up.**  The spawn pose and the cube are both
       fixed, so the calibration is one goal root pose: fly the hand with the
       keyboard until the distal tips straddle the cube, then paste the two
       lines this prints into ``hand_object_mdp.py``.  ``cube - tip midpoint``
       is the number to drive towards zero.
    2. **What a holding grasp reads.**  CLOSE on the cube, press ``V`` to drop
       the support, and watch ``min(inward1, inward2)``.  Compare attempts that
       hold against ones that drop, and set ``HAND_OBJECT_FORCE_SATURATION_N``
       near the bottom of the holding range.

    Both raw force vectors and both dot products are printed so the sign
    convention documented at the top of this module can be *checked* rather
    than trusted: during a real squeeze both inward forces must be positive.
    """
    from .hand_move_mdp import geometry_report

    index = int(env_id)
    origin = env.scene.env_origins[index]
    cube: RigidObject = env.scene[cube_cfg.name]
    support: RigidObject | None = (
        env.scene[support_cfg.name] if support_cfg is not None else None
    )

    tip1, tip2 = stick_tip_positions_w(env, stick1_cfg, stick2_cfg, tip_offset_o)
    axis = closing_axis_w(tip1, tip2)
    inward1, inward2, _, raw1, raw2 = cube_inward_forces(
        env, stick1_cfg, stick2_cfg, tip_offset_o, stick1_sensor_name, stick2_sensor_name
    )
    score1, score2, bilateral = bilateral_force_score(inward1, inward2, force_saturation)
    distance, linear_speed, angular_speed = cube_relative_state(
        env, cube_cfg, stick2_cfg, tip_offset_o
    )

    midpoint = 0.5 * (tip1[index] + tip2[index])
    cube_pos = cube.data.root_pos_w[index]
    to_cube = cube_pos - midpoint

    support_term = (
        env.command_manager.get_term(support_command_name)
        if support_command_name is not None
        and support_command_name in env.command_manager.active_terms
        else None
    )
    mode = env.command_manager.get_command(command_name)[index]

    def support_lines(where: str) -> list[str]:
        """The two support rows, or nothing when the scene has no column."""
        if where == "geometry":
            if support is None:
                return []
            return [
                f"  support centre           : "
                f"{vec(support.data.root_pos_w[index], origin)}"
                "   <- HAND_OBJECT_SUPPORT_POS_E"
            ]
        if support_term is None:
            return []
        return [
            f"  support retract           : "
            f"{float(support_term.retract_progress[index]):.3f}"
            f"  (offset {float(support_term.support_offset_m[index]) * 1000:.1f} mm, "
            f"retracted "
            f"{'YES' if float(support_term.retracted_gate[index]) > 0.5 else 'no'})"
        ]

    def vec(tensor, sub=None) -> str:
        values = tensor - sub if sub is not None else tensor
        return "(" + ", ".join(f"{float(v):+.4f}" for v in values) + ")"

    hand_block = geometry_report(
        env,
        stick1_cfg=stick1_cfg,
        stick2_cfg=stick2_cfg,
        tip_offset_o=tip_offset_o,
        orientation_command_name=orientation_command_name,
        env_id=index,
    )

    return "\n".join(
        [
            hand_block,
            "--- cube / support " + "-" * 49,
            f"  cube centre (FIXED)      : {vec(cube_pos, origin)}"
            "   <- HAND_OBJECT_CUBE_POS_E",
            *support_lines("geometry"),
            f"  cube - tip midpoint      : {vec(to_cube)}"
            f"   |d| = {float(torch.linalg.vector_norm(to_cube)) * 1000:6.2f} mm"
            "   <- 0 으로 만드는 게 목표",
            f"  cube - Stick1 tip        : "
            f"{float(torch.linalg.vector_norm(cube_pos - tip1[index])) * 1000:6.2f} mm",
            f"  cube - Stick2 tip        : "
            f"{float(torch.linalg.vector_norm(cube_pos - tip2[index])) * 1000:6.2f} mm",
            "--- force (압착 중이면 inward 둘 다 > 0 이어야 함) " + "-" * 18,
            f"  raw Stick1 <- Cube       : {vec(raw1[index])}  "
            f"|F| = {float(torch.linalg.vector_norm(raw1[index])):.5f} N",
            f"  raw Stick2 <- Cube       : {vec(raw2[index])}  "
            f"|F| = {float(torch.linalg.vector_norm(raw2[index])):.5f} N",
            f"  dot(F1, axis) x {STICK1_CUBE_FORCE_SIGN:+.0f}      : "
            f"{float(torch.sum(raw1[index] * axis[index])):+.5f} N",
            f"  dot(F2, axis) x {STICK2_CUBE_FORCE_SIGN:+.0f}      : "
            f"{float(torch.sum(raw2[index] * axis[index])):+.5f} N",
            f"  inward1 / inward2        : {float(inward1[index]):+.5f} / "
            f"{float(inward2[index]):+.5f} N",
            f"  min(inward1, inward2)    : "
            f"{float(torch.minimum(inward1, inward2)[index]):+.5f} N"
            "   <- HAND_OBJECT_FORCE_SATURATION_N 은 이 값으로 정함",
            f"  score1 / score2 / bilat  : {float(score1[index]):.3f} / "
            f"{float(score2[index]):.3f} / {float(bilateral[index]):.3f}"
            f"   (saturation {force_saturation:.4f} N, PROVISIONAL)",
            "--- state " + "-" * 58,
            f"  mode                     : "
            f"{'OPEN' if float(mode[OPEN_INDEX]) > 0.5 else 'CLOSE'}",
            *support_lines("state"),
            f"  cube height (env-local z): {float(cube_pos[2] - origin[2]):.4f} m",
            f"  cube vs Stick2 tip       : {float(distance[index]) * 1000:.2f} mm, "
            f"{float(linear_speed[index]):.4f} m/s, {float(angular_speed[index]):.4f} rad/s",
            "-" * 68,
        ]
    )
