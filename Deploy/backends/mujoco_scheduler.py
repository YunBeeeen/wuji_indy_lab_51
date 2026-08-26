# [backend/MuJoCo] 목표 하나를 정확히 1/30초 유지하는 스케줄러 + glide_to_pose/settle 기동 동작.
"""MuJoCo-only timing and target-hold scheduling.

The sim-to-sim contract is *the hold duration*, not a particular substep count.
One policy target must be held for exactly ``POLICY_DT`` seconds; how many
integration steps MuJoCo takes inside that window is a numerical-accuracy
choice, not a property of the task.

Isaac trains at ``sim.dt = 1/120`` with ``decimation = 4``, but PhysX also runs
``solver_position_iteration_count = 16`` inside each of those steps
(``hand_grasp_env_cfg.py``).  MuJoCo's Newton solver does not converge contacts
that way, so reproducing Isaac's *accuracy* takes a finer timestep rather than
an identical one.  Requiring both engines to use the same ``dt`` would be like
requiring two different ODE solvers to use the same step size.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..common.policy_contract import COMMAND_TARGET_LIMITS, POLICY_DT, soft_command_limits
from ..common.timing import StageTimer

if TYPE_CHECKING:
    from .mujoco_wuji import MujocoWujiHand
    from ..policy.policy_runner import PolicyRunner


# Default numerical settings for the contact-rich two-stick grasp.  A 10 g stick
# pinched between convex-hull fingertips is not resolved at 8.3 ms with an
# explicit integrator, and the deploy (Isaac-tuned) gains are stiff enough that
# the explicit stability bound is violated there.
#
# 2026-08-18 convergence study (full 15 s episode, deploy gains, implicitfast,
# Stick1 displacement from the Isaac reset reference):
#
#     1/120   189.9 mm     1/960    86.1 mm
#     1/240   182.7 mm     1/1920  195.9 mm
#     1/480   diverged     1/3840  196.1 mm
#
# 1/480 diverges and 1/960 is an outlier, not a converged answer; 1/1920 and
# 1/3840 agree.  64 substeps is therefore the smallest defensible default.
# Re-run this study after the Isaac/MuJoCo collision-geometry mismatch is
# resolved, because the current converged answer is a drop either way.
MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP = 64
MUJOCO_PHYSICS_DT = POLICY_DT / MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP
# MuJoCo's recommended default.  RK4, which the vendor MJCF declares, does not
# support implicit damping and is not recommended with contacts.
MUJOCO_INTEGRATOR = "implicitfast"

SUPPORTED_INTEGRATORS = ("euler", "rk4", "implicit", "implicitfast")

# Retained so older call sites keep working; new code should read
# ``backend.physics_substeps``.
MUJOCO_PHYSICS_STEPS_PER_POLICY_STEP = MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP


def physics_dt_for_substeps(substeps: int) -> float:
    """Return the timestep that divides one policy step into ``substeps``."""

    if int(substeps) != substeps or substeps < 1:
        raise ValueError(f"Physics substeps must be a positive integer, got {substeps!r}.")
    return POLICY_DT / int(substeps)


def validate_hold_schedule(timestep: float, substeps: int) -> None:
    """Fail unless ``substeps`` integration steps span exactly one policy step."""

    if not np.isclose(timestep * substeps, POLICY_DT, atol=0.0, rtol=1.0e-12):
        raise RuntimeError(
            f"MuJoCo scheduling does not match the {1.0 / POLICY_DT:.1f} Hz policy "
            f"contract: {substeps} x {timestep:.9f}s = {timestep * substeps:.9f}s "
            f"!= {POLICY_DT:.9f}s."
        )


validate_hold_schedule(MUJOCO_PHYSICS_DT, MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP)


class MujocoScheduler:
    """Hold one policy target for exactly one policy step of physics time."""

    def __init__(
        self,
        backend: "MujocoWujiHand",
        *,
        viewer=None,
        realtime: bool = False,
    ) -> None:
        self.backend = backend
        self.viewer = viewer
        self.realtime = realtime
        # Read the substep count from the backend so the two cannot drift.
        self.substeps = int(getattr(backend, "physics_substeps", MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP))
        self.physics_dt = float(backend.model.opt.timestep)
        validate_hold_schedule(self.physics_dt, self.substeps)
        self._realtime_anchor: float | None = None
        self._policy_steps_run = 0
        #: The physics hold is where a MuJoCo tick's time actually goes -- 64
        #: substeps per policy step.  Without it the tick report shows a large
        #: unaccounted remainder and points at nothing.
        self.timing = StageTimer(budget_ms=1000.0 * POLICY_DT, name="mujoco")

    def hold_policy_target(self, pin_sticks=None) -> None:
        """Integrate one policy step, optionally paced to the wall clock.

        Pacing is one sleep per POLICY step against an absolute deadline, not
        one sleep per physics substep.  Sleeping ``physics_dt`` each substep
        looks right but is not achievable: at 1/1920 s that asks for 0.52 ms,
        below the scheduler's granularity, so each call overshoots and the
        error multiplies by the substep count.  Measured against a real hand,
        the simulator fell minutes behind.

        Physics is unaffected either way -- the integrator does not know about
        wall-clock time -- so trajectories and CSVs are identical with and
        without ``realtime``.
        """

        target = self.backend.control_snapshot()
        if self.realtime and self._realtime_anchor is None:
            self._realtime_anchor = time.monotonic()

        for _ in range(self.substeps):
            self.backend.step(1)
            if pin_sticks is not None:
                # Every substep, not once per policy step: a 10 g stick falls
                # 0.13 mm in one substep but 8.5 mm in a policy step, and the
                # contact impulse from re-teleporting it that far is exactly the
                # PhysX-explosion pattern CLAUDE.md warns about.  Re-applying
                # the SAME pose each substep keeps penetration at its reset
                # value instead of accumulating.
                self.backend.set_stick_poses_in_palm(pin_sticks)
            if not np.array_equal(target, self.backend.control_snapshot()):
                raise RuntimeError("MuJoCo target changed during the policy-step hold.")

        # One viewer sync per policy step: at 30 Hz that is already above the
        # rate a person can see, and syncing every substep was costing more
        # than the physics.
        if self.viewer is not None:
            self.viewer.sync()

        self._policy_steps_run += 1
        if self.realtime:
            deadline = self._realtime_anchor + self._policy_steps_run * POLICY_DT
            wait = deadline - time.monotonic()
            if wait > 0:
                time.sleep(wait)

    def run_policy_tick(self, runner: "PolicyRunner", pin_sticks=None):
        with self.timing.stage("command"):
            decoded = runner.command()
        with self.timing.stage("physics_hold"):
            self.hold_policy_target(pin_sticks=pin_sticks)
        with self.timing.stage("observe"):
            observation = runner.observe_after_hold()
        return decoded, observation

    # ------------------------------------------------------------------ #
    # Bring-up moves.  Deliberately the same names and meanings as
    # RealWujiScheduler so the MuJoCo and hardware entry points read alike:
    # glide to the start pose, let it settle, and only then hand over to the
    # policy.  MuJoCo walks the target at the POLICY rate rather than the
    # hardware command rate, because one target held for one policy step is
    # this backend's whole scheduling contract.
    # ------------------------------------------------------------------ #

    def glide_to_pose(
        self,
        q_target_all: npt.ArrayLike,
        *,
        seconds: float,
        limit_fraction: float = 1.0,
        pin_sticks=None,
        report=None,
    ):
        """Walk the commanded target linearly to ``q_target_all``.

        The hardware reason for walking rather than commanding the endpoint is
        rate: a fixed target plus a LowPass is an exponential, fastest at its
        first tick.  MuJoCo has no such filter, so a step command would be even
        more violent -- the servo would chase the full displacement inside one
        policy step.  Walking gives ``displacement / seconds`` throughout, which
        is what the real hand does and therefore what sim-to-sim must compare.

        Returns ``(elapsed_s, q_actual, error)``.
        """

        q_start = self.backend.read_joint_positions()
        q_goal = np.asarray(q_target_all, dtype=np.float32)
        if q_goal.shape != q_start.shape:
            raise ValueError(f"Goal must be {q_start.shape}, got {q_goal.shape}.")
        if not np.isfinite(q_goal).all():
            raise ValueError("Goal must be finite.")
        limits = soft_command_limits(limit_fraction)
        # Clamp the ENDPOINT only.  A joint resting on a mechanical stop reads
        # slightly outside the software limit, and interpolating from that raw
        # measurement would make the very first commanded target out of range.
        q_goal = np.clip(q_goal, limits[:, 0], limits[:, 1]).astype(np.float32)

        steps = max(1, int(round(float(seconds) / POLICY_DT)))
        for step in range(1, steps + 1):
            fraction = step / steps
            target = np.clip(
                q_start + fraction * (q_goal - q_start), limits[:, 0], limits[:, 1]
            ).astype(np.float32)
            self.backend.write_joint_position_targets(target)
            self.hold_policy_target(pin_sticks=pin_sticks)
            if report is not None:
                actual = self.backend.read_joint_positions()
                report(step * POLICY_DT, actual, actual - q_goal)

        actual = self.backend.read_joint_positions()
        return steps * POLICY_DT, actual, actual - q_goal

    def settle(self, seconds: float, *, pin_sticks=None, report=None):
        """Hold the present target so the servo and contacts come to rest.

        Separate from ``glide_to_pose`` on purpose: arriving at a target is not
        the same as being settled at it, and on hardware this is the window a
        person uses to place the chopsticks.  Returns ``(q_actual, drift)``,
        drift being how far q moved during the hold.
        """

        q_before = self.backend.read_joint_positions()
        steps = max(1, int(round(float(seconds) / POLICY_DT)))
        for step in range(1, steps + 1):
            self.hold_policy_target(pin_sticks=pin_sticks)
            if report is not None:
                actual = self.backend.read_joint_positions()
                report(step * POLICY_DT, actual, actual - q_before)
        q_after = self.backend.read_joint_positions()
        return q_after, q_after - q_before
