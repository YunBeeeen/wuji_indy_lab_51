"""90 Hz command loop with a 30 Hz policy tick, for the real hand.

Single thread on purpose.  Two threads would need a lock around the target
buffer and would make the policy-to-command phase nondeterministic; here the
relationship is fixed and checkable: 90/30 = 3 exactly, so policy runs on every
third command tick and the two other ticks re-send the same target.

    tick 0   read q -> observation -> ONNX -> residual target -> send
    tick 1   send the same target
    tick 2   send the same target
    tick 3   read q -> ...

Timing uses absolute monotonic deadlines rather than accumulated sleeps, the
same way ``move_middle_j1.py`` paces its 100 Hz loop: sleeping ``1/rate`` each
pass accumulates every scheduling overshoot, and at 90 Hz that drifts fast.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..common.policy_contract import COMMAND_TARGET_LIMITS, POLICY_DT, soft_command_limits

DEFAULT_COMMAND_HZ = 90.0
POLICY_HZ = 1.0 / POLICY_DT


def policy_divider(command_hz: float) -> int:
    """Command ticks per policy tick.  Must be an exact integer."""

    ratio = command_hz * POLICY_DT
    divider = int(round(ratio))
    if divider < 1 or not np.isclose(ratio, divider, atol=0.0, rtol=1e-9):
        raise ValueError(
            f"command rate {command_hz} Hz is not an integer multiple of the "
            f"{POLICY_HZ:.1f} Hz policy rate (ratio {ratio:.6f}). Pick e.g. 90 Hz."
        )
    return divider


@dataclass
class LoopTiming:
    """Measured pacing, so drift is reported rather than assumed absent."""

    command_periods_ms: list[float] = field(default_factory=list)
    policy_inference_ms: list[float] = field(default_factory=list)
    late_ticks: int = 0

    def summary(self) -> str:
        if not self.command_periods_ms:
            return "no ticks recorded"
        periods = np.asarray(self.command_periods_ms)
        text = (
            f"command period: mean {periods.mean():.3f} ms, "
            f"p95 {np.percentile(periods, 95):.3f} ms, max {periods.max():.3f} ms; "
            f"late ticks {self.late_ticks}/{len(periods)}"
        )
        if self.policy_inference_ms:
            inference = np.asarray(self.policy_inference_ms)
            text += (
                f"; policy step: mean {inference.mean():.3f} ms, "
                f"max {inference.max():.3f} ms"
            )
        return text


class RealWujiScheduler:
    """Pace the command loop and fire the policy on an exact subdivision."""

    def __init__(self, backend, command_hz: float = DEFAULT_COMMAND_HZ):
        self.backend = backend
        self.command_hz = float(command_hz)
        self.divider = policy_divider(self.command_hz)
        self.dt = 1.0 / self.command_hz
        self.timing = LoopTiming()

    def run(self, duration_s: float, controller, on_policy_tick=None,
            on_command_tick=None) -> int:
        """Drive the loop for ``duration_s``.

        ``on_policy_tick(policy_index, elapsed_s)`` runs on every ``divider``-th
        tick and is expected to leave a new target in the backend.  Every tick,
        policy or not, re-sends whatever target is currently stored.

        ``on_command_tick(tick, elapsed_s)`` runs on EVERY tick and is for
        diagnostics only -- it must not touch the target.  The policy has to
        sample q at its own rate, because the observation history is defined as
        two samples one policy step apart; logging faster changes nothing about
        that and exposes what happens between policy steps, which is otherwise
        invisible.
        """

        total_ticks = max(1, int(round(duration_s * self.command_hz)))
        start = time.monotonic()
        previous = start
        policy_index = 0

        for tick in range(total_ticks):
            if tick % self.divider == 0:
                if on_policy_tick is not None:
                    began = time.monotonic()
                    on_policy_tick(policy_index, began - start)
                    self.timing.policy_inference_ms.append(
                        (time.monotonic() - began) * 1000.0
                    )
                policy_index += 1

            self.backend.publish_latest_target(controller)
            if on_command_tick is not None:
                on_command_tick(tick, time.monotonic() - start)

            now = time.monotonic()
            self.timing.command_periods_ms.append((now - previous) * 1000.0)
            previous = now

            deadline = start + (tick + 1) * self.dt
            wait = deadline - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                self.timing.late_ticks += 1
        return policy_index

    def glide_to_pose(
        self,
        q_target_all: npt.ArrayLike,
        controller,
        *,
        joint_indices,
        seconds: float,
        tolerance_rad: float,
        stable_seconds: float,
        timeout_seconds: float,
        report=None,
        limit_fraction: float = 1.0,
    ):
        """Move at a BOUNDED rate by walking the target, then confirm arrival.

        ``move_to_start_pose`` sets the final target immediately and lets the
        LowPass shape the approach.  That is smooth but not slow: an exponential
        is fastest at t=0, at ``displacement / tau``.  Returning 1.18 rad with a
        0.5 Hz filter therefore starts at 3.7 rad/s -- quicker than the policy's
        own 3 rad/s ceiling, which is alarming to stand next to.

        Here the commanded target is interpolated over ``seconds``, so the
        commanded rate is ``displacement / seconds`` from the first tick to the
        last.  Use this whenever the displacement is large and the motion should
        look deliberate; use ``move_to_start_pose`` when the target is close and
        letting the servo settle is the point.
        """

        q_start = self.backend.read_joint_positions()
        q_goal = np.asarray(q_target_all, dtype=np.float32)
        if q_goal.shape != q_start.shape:
            raise ValueError(f"Goal must be {q_start.shape}, got {q_goal.shape}.")

        # Clamp both ends into the command space before interpolating.  A joint
        # resting against its mechanical stop reads a hair past the software
        # limit -- finger3_joint3 came back 1.6804522 against a 1.680047 ceiling
        # -- and starting a trajectory from that raw measurement asks the
        # backend to command a value it is right to refuse.  Constraining the
        # trajectory endpoints is not the same as silently clipping a policy
        # output; every intermediate point is then in range by construction.
        limits = soft_command_limits(limit_fraction)
        lower, upper = limits[:, 0], limits[:, 1]
        q_start = np.clip(q_start, lower, upper).astype(np.float32)
        q_goal = np.clip(q_goal, lower, upper).astype(np.float32)

        travel = float(np.abs(q_goal - q_start).max())
        allowance = travel + max(0.05, 0.1 * travel)
        glide_ticks = max(1, int(round(seconds * self.command_hz)))
        start = time.monotonic()
        previous = start
        reached_since = None
        report_every = max(1, int(round(self.command_hz / 3.0)))
        tick = 0

        while True:
            alpha = min(1.0, (tick + 1) / glide_ticks)
            self.backend.write_joint_position_targets(
                ((1.0 - alpha) * q_start + alpha * q_goal).astype(np.float32),
                max_step_rad=allowance,
            )
            self.backend.publish_latest_target(controller)
            now_tick = time.monotonic()
            self.timing.command_periods_ms.append((now_tick - previous) * 1000.0)
            previous = now_tick

            if tick % self.divider == 0:
                q_actual = self.backend.read_joint_positions()
                error = np.abs(q_actual[joint_indices] - q_goal[joint_indices])
                max_error = float(error.max())
                now = time.monotonic()
                elapsed = now - start

                # Only start judging arrival once the target has stopped moving.
                if alpha >= 1.0 and max_error <= tolerance_rad:
                    if reached_since is None:
                        reached_since = now
                    elif now - reached_since >= stable_seconds:
                        return elapsed, q_actual, error
                else:
                    reached_since = None

                if elapsed > seconds + timeout_seconds:
                    raise RuntimeError(
                        f"Glide did not settle within {seconds:.1f}+{timeout_seconds:.1f} s.\n"
                        f"  tolerance      {tolerance_rad:.4f} rad\n"
                        f"  max error      {max_error:.6f} rad\n"
                        f"  actual   (cmd) {np.round(q_actual[joint_indices], 6).tolist()}\n"
                        f"  desired  (cmd) {np.round(q_goal[joint_indices], 6).tolist()}\n"
                        f"  per-joint err  {np.round(error, 6).tolist()}"
                    )

                if report is not None and tick % report_every == 0:
                    report(elapsed, q_actual[joint_indices], error, max_error)

            tick += 1
            deadline = start + tick * self.dt
            wait = deadline - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                self.timing.late_ticks += 1

    def move_to_start_pose(
        self,
        q_target_all: npt.ArrayLike,
        controller,
        *,
        joint_indices,
        tolerance_rad: float,
        stable_seconds: float,
        timeout_seconds: float,
        report=None,
        limit_fraction: float = 1.0,
    ):
        """Drive to a FIXED start target and wait until the encoders confirm it.

        This is not a ramp and not a policy step.  The target is written once
        and then re-sent unchanged at the command rate; the LowPass filter and
        the firmware servo produce the motion.  Nothing here interpolates the
        target, runs inference, or applies the +-0.1 rad residual rule -- that
        rule belongs to the policy phase and only to it.

        Both phases transmit at the same wire rate for the same reason (90 Hz is
        the realtime command rate), but they generate the target completely
        differently:

            start preparation   fixed target, unchanged for the whole move
            policy execution    new target every 30 Hz from q_current + 0.1*a

        Arrival is judged on ``joint_indices`` only -- the joints actually being
        commanded -- and must hold inside ``tolerance_rad`` continuously for
        ``stable_seconds``; a single sample brushing the threshold on the way
        past is not arrival.
        """

        q_target = np.asarray(q_target_all, dtype=np.float32)
        q_now = self.backend.read_joint_positions()
        if q_target.shape != q_now.shape:
            raise ValueError(f"Start target must be {q_now.shape}, got {q_target.shape}.")
        # Same reason as glide_to_pose: a joint on its stop reads past the limit.
        _soft = soft_command_limits(limit_fraction)
        q_target = np.clip(
            q_target, _soft[:, 0], _soft[:, 1]
        ).astype(np.float32)

        # The move legitimately spans the whole displacement, so it states its
        # own allowance rather than inheriting the per-policy-step slew guard.
        travel = float(np.abs(q_target - q_now).max())
        self.backend.write_joint_position_targets(
            q_target, max_step_rad=travel + max(0.05, 0.1 * travel)
        )

        start = time.monotonic()
        previous = start
        reached_since = None
        report_every = max(1, int(round(self.command_hz / 3.0)))
        tick = 0
        while True:
            # Same target, every tick, never interpolated.
            self.backend.publish_latest_target(controller)
            now_tick = time.monotonic()
            self.timing.command_periods_ms.append((now_tick - previous) * 1000.0)
            previous = now_tick

            # Read at the policy rate rather than every command tick: it is the
            # load the policy phase will impose, so the link is exercised the
            # same way here.
            if tick % self.divider == 0:
                q_actual = self.backend.read_joint_positions()
                error = np.abs(q_actual[joint_indices] - q_target[joint_indices])
                max_error = float(error.max())
                now = time.monotonic()
                elapsed = now - start

                if max_error <= tolerance_rad:
                    if reached_since is None:
                        reached_since = now
                    elif now - reached_since >= stable_seconds:
                        return elapsed, q_actual, error
                else:
                    reached_since = None

                if elapsed > timeout_seconds:
                    lines = [
                        f"Start pose not reached within {timeout_seconds:.1f} s.",
                        f"  tolerance      {tolerance_rad:.4f} rad",
                        f"  max error      {max_error:.6f} rad",
                        f"  actual   (cmd) {np.round(q_actual[joint_indices], 6).tolist()}",
                        f"  desired  (cmd) {np.round(q_target[joint_indices], 6).tolist()}",
                        f"  per-joint err  {np.round(error, 6).tolist()}",
                    ]
                    raise RuntimeError("\n".join(lines))

                if report is not None and tick % report_every == 0:
                    report(elapsed, q_actual[joint_indices], error, max_error)

            tick += 1
            deadline = start + tick * self.dt
            wait = deadline - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            else:
                self.timing.late_ticks += 1
