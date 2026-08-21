"""Real Wuji Hand backend for the middle-finger reach contract.

Deliberately imports no MuJoCo: ``wujihandpy`` lives in the ``wuji_hw``
environment, which has no mujoco, and the two backends must stay separable.

What this is NOT
----------------
It is not a mirror of ``MujocoWujiHand``.  Two of that class's operations have
no hardware counterpart and are absent here on purpose:

* ``reset(q)`` -- a simulator teleport.  A real hand can only be *moved*, so
  the start pose is reached by ramping (see ``real_wuji_scheduler.ramp_to``).
* controller gains -- MuJoCo installs Kp/Kd explicitly.  ``wujihandpy`` exposes
  a position command and a LowPass command filter; what servo gains the
  firmware runs underneath is UNVERIFIED.  Nothing here pretends otherwise, so
  the sim-to-real plant mismatch stays visible rather than assumed away.

SDK boundary
------------
``wujihandpy`` speaks ``(5, 4)`` finger-major arrays; the policy contract
speaks canonical ``(20,)``.  Every crossing goes through
``finger_reach.finger_major_grid`` / ``from_finger_major_grid`` with an
explicit shape assert -- the two layouts happen to agree today, and that is
exactly the kind of coincidence that should be checked rather than trusted.

Verified against ``/home/lsc/wuji_test/move_middle_j1.py``, which moved
finger3_joint1 on the physical hand.  No SDK call outside that example plus the
inspected ``wujihandpy`` API surface is used.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..policy.finger_reach import (
    FINGERS,
    REACH_ACTION_SCALE_RAD,
    JOINTS_PER_FINGER,
    MIDDLE_POLICY_INDICES,
    finger_major_grid,
    from_finger_major_grid,
)
from ..common.backend_protocol import BackendHealth
from ..common.policy_contract import COMMAND_TARGET_LIMITS, POLICY_JOINT_NAMES

HAND_JOINTS = FINGERS * JOINTS_PER_FINGER
# finger3 is row 2 of the SDK grid; derived from the canonical indices rather
# than written down, so a change to REACH_FINGER_INDEX cannot desync them.
MIDDLE_GRID_ROWS = sorted({int(i) // JOINTS_PER_FINGER for i in MIDDLE_POLICY_INDICES})


def full_enable_mask() -> npt.NDArray[np.bool_]:
    """Enable every joint -- what a twenty-joint grasp policy needs.

    Separate from ``middle_enable_mask`` on purpose.  finger_reach could leave
    sixteen motors off, so an error could only ever move one finger; a grasp
    policy cannot, and that is a real step up in what a bug can do.
    """

    return np.ones((FINGERS, JOINTS_PER_FINGER), dtype=bool)


def middle_enable_mask(joint: int | None = None) -> npt.NDArray[np.bool_]:
    """Enable only the middle finger, or a single one of its joints.

    ``joint`` is 1-based (J1..J4) and exists for the per-joint bring-up check
    that confirms canonical finger3_jointN really is the hardware's J(N).
    """

    mask = np.zeros((FINGERS, JOINTS_PER_FINGER), dtype=bool)
    for row in MIDDLE_GRID_ROWS:
        if joint is None:
            mask[row, :] = True
        else:
            if not 1 <= joint <= JOINTS_PER_FINGER:
                raise ValueError(f"joint must be 1..{JOINTS_PER_FINGER}, got {joint}.")
            mask[row, joint - 1] = True
    return mask


class RealWujiHand:
    """Canonical-radian I/O against the physical hand.

    ``step_guard_reference="measured"`` keeps the original residual-policy
    safety check: each target is bounded relative to the latest measured q.

    ``step_guard_reference="command"`` is for precomputed trajectory replay:
    each new target is bounded relative to the previous commanded target, so
    normal servo lag/contact preload does not trip the slew guard.
    """

    def __init__(
        self,
        read_source: str = "controller",
        max_step_rad: float | None = None,
        step_guard_reference: str = "measured",
    ):
        import wujihandpy

        if read_source not in ("controller", "hand"):
            raise ValueError(f"read_source must be 'controller' or 'hand', got {read_source!r}.")
        if step_guard_reference not in ("measured", "command"):
            raise ValueError(
                "step_guard_reference must be 'measured' or 'command', "
                f"got {step_guard_reference!r}."
            )

        self._wujihandpy = wujihandpy
        self.read_source = read_source
        self.step_guard_reference = step_guard_reference
        self.hand = wujihandpy.Hand()
        self.controller = None
        self.enabled_mask: npt.NDArray[np.bool_] | None = None
        self.latest_target = np.zeros(HAND_JOINTS, dtype=np.float32)
        self._target_initialized = False
        # Independent slew guard.  A correct reach policy can only move a joint
        # by REACH_ACTION_SCALE_RAD per policy step, so at that value this is a
        # no-op and the sim/real contract is unchanged.  It exists to catch the
        # cases the contract cannot: a corrupt ONNX, a mis-shaped observation,
        # or a decode bug asking the hand for a large jump in one step.
        self.max_step_rad = (
            float(REACH_ACTION_SCALE_RAD) if max_step_rad is None else float(max_step_rad)
        )
        self._last_read_q: npt.NDArray[np.float32] | None = None
        # Set by safe_stop().  Sticky on purpose: a run that hit a safe stop
        # must not look like a clean one in the log afterwards.
        self.safe_stopped = False
        self.safe_stop_reason: str | None = None

    # -- identity ---------------------------------------------------------
    def joint_identifiers(self) -> tuple[str, ...]:
        return POLICY_JOINT_NAMES

    def describe(self) -> str:
        # read_handedness() is NOT the contract's source of handedness.  On the
        # connected RIGHT hand it reports 0, which TactileHandedness maps to
        # LEFT -- the field belongs to the tactile glove, or is simply unset.
        # The RIGHT-hand contract comes from the pinned wuji-description
        # hand/body RIGHT model, so print this as raw diagnostics and label it.
        parts = [
            "contract=RIGHT (wuji-description hand/body RIGHT)",
            f"read_handedness()={self.hand.read_handedness()} <- tactile field, not the hand body",
        ]
        for label, call in (("sn", "read_product_sn"), ("fw", "read_firmware_version")):
            getter = getattr(self.hand, call, None)
            if getter is not None:
                try:
                    parts.append(f"{label}={getter()}")
                except Exception as exc:  # pragma: no cover - diagnostics only
                    parts.append(f"{label}=<{type(exc).__name__}>")
        return ", ".join(str(p) for p in parts)

    # -- reads ------------------------------------------------------------
    def read_joint_positions(self) -> npt.NDArray[np.float32]:
        """Return the measured 20 joint angles in canonical order, radians."""

        if self.read_source == "controller" and self.controller is not None:
            grid = self.controller.get_joint_actual_position()
        else:
            grid = self.hand.read_joint_actual_position()
        grid = np.asarray(grid, dtype=np.float64)
        if grid.shape != (FINGERS, JOINTS_PER_FINGER):
            raise RuntimeError(
                f"SDK returned joint positions with shape {grid.shape}, "
                f"expected {(FINGERS, JOINTS_PER_FINGER)}."
            )
        if not np.isfinite(grid).all():
            raise RuntimeError("SDK returned non-finite joint positions.")
        q = from_finger_major_grid(grid.astype(np.float32))
        self._last_read_q = q
        return q

    def read_hardware_limits(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Return the hand's own reported limits, canonical order."""

        lower = from_finger_major_grid(
            np.asarray(self.hand.read_joint_lower_limit(), dtype=np.float32)
        )
        upper = from_finger_major_grid(
            np.asarray(self.hand.read_joint_upper_limit(), dtype=np.float32)
        )
        return lower, upper

    # -- writes -----------------------------------------------------------
    def write_joint_position_targets(
        self, targets_policy_order: npt.ArrayLike, max_step_rad: float | None = None
    ) -> None:
        """Store a canonical 20-vector target.  Does not transmit.

        Transmission happens in ``publish_latest_target`` at the command rate,
        so the policy rate and the wire rate stay independent.  Out-of-range
        input is rejected rather than clipped: the residual decoder already
        clamps to ``COMMAND_TARGET_LIMITS``, so anything outside it means a bug
        upstream, and silently trimming it on the way to hardware would hide it.
        """

        targets = np.asarray(targets_policy_order, dtype=np.float32)
        if targets.shape != (HAND_JOINTS,):
            raise ValueError(f"Targets must have shape {(HAND_JOINTS,)}, got {targets.shape}.")
        if not np.isfinite(targets).all():
            raise ValueError("Targets must be finite.")
        lower, upper = COMMAND_TARGET_LIMITS[:, 0], COMMAND_TARGET_LIMITS[:, 1]
        if np.any(targets < lower) or np.any(targets > upper):
            bad = np.flatnonzero((targets < lower) | (targets > upper)).tolist()
            raise ValueError(
                f"Targets outside COMMAND_TARGET_LIMITS at policy indices {bad}: "
                f"{targets[bad].tolist()}"
            )
        limit = self.max_step_rad if max_step_rad is None else float(max_step_rad)

        if limit > 0.0:
            if self.step_guard_reference == "measured":
                # Original residual-policy guard:
                # q_target is defined relative to the measured q, so reject a
                # target that is unexpectedly far from the latest measurement.
                reference = self._last_read_q
                reference_name = "measured position"
            else:
                # Replay / precomputed-trajectory guard:
                # only reject a discontinuous command jump.  Servo lag, contact
                # preload and LowPass delay are allowed to make actual q differ
                # substantially from the commanded target.
                reference = self.latest_target if self._target_initialized else None
                reference_name = "previous command"

            if reference is not None:
                step = targets - reference
                excessive = np.abs(step) > limit + 1e-5

                if excessive.any():
                    names = [
                        POLICY_JOINT_NAMES[i]
                        for i in np.flatnonzero(excessive)
                    ]
                    raise RuntimeError(
                        f"Target changes by "
                        f"{np.round(step[excessive], 5).tolist()} rad "
                        f"from the {reference_name} at {names}, "
                        f"beyond the {limit:.4f} rad allowance. "
                        "Refusing to send it to hardware."
                    )

        self.latest_target = targets.copy()
        self._target_initialized = True

    def prime_target_to_current(self) -> npt.NDArray[np.float32]:
        """Latch the present pose as the target before any motor is enabled.

        ``move_middle_j1.py`` does this too: writing the target first means
        enabling cannot produce a step toward a stale setpoint.
        """

        q = self.read_joint_positions()
        held = np.clip(q, COMMAND_TARGET_LIMITS[:, 0], COMMAND_TARGET_LIMITS[:, 1]).astype(
            np.float32
        )
        self.write_joint_position_targets(held)
        self.hand.write_joint_target_position(finger_major_grid(held).astype(np.float64))
        return q

    def publish_latest_target(self, controller=None) -> None:
        """Send the stored target once, at the command rate."""

        if not self._target_initialized:
            raise RuntimeError("No target has been written yet; call prime_target_to_current().")
        sink = controller if controller is not None else self.controller
        grid = finger_major_grid(self.latest_target).astype(np.float64)
        if sink is None:
            self.hand.write_joint_target_position(grid)
        else:
            sink.set_joint_target_position(grid)

    # -- protection -------------------------------------------------------
    def read_joint_temperatures(self) -> npt.NDArray[np.float32]:
        """Return the 20 motor temperatures in Celsius, canonical order.

        **This is a BLOCKING SDO read** (``read_joint_temperature(timeout=0.5)``)
        and must not be called from inside the command loop.  It is the same
        class of call ``measure_io_timing`` reports separately as "blocking SDO
        read for reference -- not used by the loop": a command tick has an
        11.1 ms budget at 90 Hz, and this can spend far more than that.

        The SDK also offers ``get_joint_temperature()``, with no timeout
        argument, which returns a locally cached value and would be cheap.  It
        is NOT used here: what populates that cache is not documented and has
        not been measured, so it could return zeros that a temperature guard
        would read as "cold".  A validated slow read in a place where slowness
        is harmless beats an unvalidated fast one inside a safety check.

        Callers must therefore use this only where a late tick does not matter
        -- while holding a constant target, not while a policy is running.

        It exists because holding the pregrasp pose WITHOUT chopsticks stalls
        the fingers against each other: measured 2026-08-19, current saturated
        at the 1.5 A limit for 96 s and finger1_joint2 reached 88.4 C.
        """

        return from_finger_major_grid(
            np.asarray(self.hand.read_joint_temperature(), dtype=np.float32)
        )

    def read_joint_efforts(self) -> npt.NDArray[np.float32]:
        """Return per-joint effort in AMPERES, canonical order.

        Controller-only, and deliberately not falling back to ``hand``:
        wujihandpy 1.7.0 exposes no actual-effort call on ``Hand`` at all
        (``read_joint_effort_limit`` is the configured LIMIT, not a measurement).
        ``IController.get_joint_actual_effort`` is the single source, and being
        a ``get_*`` it reads the same upstream stream as
        ``get_joint_actual_position`` -- so it is cheap enough for the policy
        loop, unlike ``read_joint_temperatures``.

        Named 'effort' by the SDK but it is current, not torque: it saturates at
        exactly the 1.5000 A current limit.  Torque would need a motor constant
        this project has not measured, so nothing here converts it.
        """

        if self.controller is None:
            raise RuntimeError(
                "Joint effort is only available through the realtime controller; "
                "wujihandpy 1.7.0 has no actual-effort call on Hand."
            )
        return from_finger_major_grid(
            np.asarray(self.controller.get_joint_actual_effort(), dtype=np.float32)
        )

    def read_current_limits(self) -> npt.NDArray[np.float32]:
        return from_finger_major_grid(
            np.asarray(self.hand.read_joint_current_limit(), dtype=np.float32)
        )

    def write_current_limit(self, amps: float) -> None:
        """Set every joint's current limit before any motor is enabled.

        The unit is amperes, not N.m -- the vendor's own example passes 1.5 to
        an ``effort_limit`` named field, and torque only follows from current
        through a motor constant this project has not measured.  So this caps
        current directly and makes no claim about the resulting torque.
        """

        if not np.isfinite(amps) or amps <= 0.0:
            raise ValueError(f"Current limit must be positive and finite, got {amps}.")
        grid = np.full((FINGERS, JOINTS_PER_FINGER), float(amps), dtype=np.float64)
        self.hand.write_joint_current_limit(grid)

    def run_vendor_latency_test(self, seconds: float = 3.0) -> list[str]:
        """Run the SDK's own latency test and return what it wrote to the log.

        ``start_latency_test`` / ``stop_latency_test`` return None, so whatever
        they measure goes to the SDK log under ``~/.wuji/log/``.  This exists
        because our own timing numbers only prove how often Python CALLED the
        API: ``set_joint_target_position`` returns in 0.039 ms, far too fast for
        a USB round trip, so it is almost certainly enqueuing into the realtime
        stream rather than waiting for the wire.  The vendor's test is the only
        available way to see the transport itself.

        Undocumented, so this reports the log verbatim instead of interpreting
        it.  Safe with motors off.
        """

        import time
        from pathlib import Path

        log_dir = Path.home() / ".wuji" / "log"
        before = set(log_dir.glob("*.log")) if log_dir.is_dir() else set()
        sizes = {p: p.stat().st_size for p in before}

        self.hand.start_latency_test()
        try:
            time.sleep(float(seconds))
        finally:
            self.hand.stop_latency_test()

        if not log_dir.is_dir():
            return [f"(no SDK log directory at {log_dir})"]
        newest = max(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, default=None)
        if newest is None:
            return ["(no SDK log file found)"]
        with newest.open(errors="replace") as handle:
            handle.seek(sizes.get(newest, 0))
            fresh = [line.rstrip() for line in handle if line.strip()]
        return fresh or [f"(latency test wrote nothing new to {newest})"]

    def measure_io_timing(self, samples: int = 200, lowpass_hz: float = 0.5) -> dict[str, float]:
        """Time the calls the control loop actually makes, with motors OFF.

        This opens a realtime controller and measures
        ``controller.set_joint_target_position`` / ``get_joint_actual_position``
        -- the streaming path.  Measuring ``hand.write_joint_target_position``
        instead reports the blocking SDO path, which is an order of magnitude
        slower and says nothing about the loop: it reported 30 ms per write on a
        link that then ran 90 Hz with zero late ticks.

        Safe with every motor disabled: nothing is energised, and the target is
        primed to the present pose first so an accidental enable commands no
        motion.
        """

        import time

        q = self.prime_target_to_current()
        grid = finger_major_grid(self.latest_target).astype(np.float64)
        blocking_reads, stream_reads, stream_writes = [], [], []

        began = time.monotonic()
        self.hand.read_joint_actual_position()
        blocking_read_ms = (time.monotonic() - began) * 1000.0

        with self.realtime_controller(lowpass_hz, enable_upstream=True) as controller:
            previous_source, self.controller = self.read_source, controller
            try:
                for _ in range(int(samples)):
                    began = time.monotonic()
                    controller.set_joint_target_position(grid)
                    stream_writes.append((time.monotonic() - began) * 1000.0)
                    began = time.monotonic()
                    controller.get_joint_actual_position()
                    stream_reads.append((time.monotonic() - began) * 1000.0)
                    time.sleep(1.0 / 90.0)
            finally:
                self.controller = None
                self.read_source = previous_source

        reads, writes = np.asarray(stream_reads), np.asarray(stream_writes)
        return {
            "read_mean_ms": float(reads.mean()),
            "read_p95_ms": float(np.percentile(reads, 95)),
            "read_max_ms": float(reads.max()),
            "write_mean_ms": float(writes.mean()),
            "write_p95_ms": float(np.percentile(writes, 95)),
            "write_max_ms": float(writes.max()),
            "blocking_read_ms": blocking_read_ms,
            "samples": float(samples),
            "q_drift_rad": float(np.abs(self.read_joint_positions() - q).max()),
        }

    # -- motor state ------------------------------------------------------
    def enable(self, mask: npt.ArrayLike) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (FINGERS, JOINTS_PER_FINGER):
            raise ValueError(f"Enable mask must be {(FINGERS, JOINTS_PER_FINGER)}, got {mask.shape}.")
        self.hand.write_joint_enabled(mask)
        self.enabled_mask = mask

    def disable(self) -> None:
        """Turn every motor off.  Safe to call twice, and from a finally block."""

        self.hand.write_joint_enabled(False)
        self.enabled_mask = None

    # -- WujiBackend boundary ---------------------------------------------
    def health(self) -> BackendHealth:
        """Report whether this backend can still be believed.

        Diagnostic, not a safety interlock -- the one call site is the end of a
        smoke test.  It deliberately uses only SDK calls that hardware runs have
        already exercised: there is no fault-flag or status API in wujihandpy
        1.7.0 that has been validated here, and inventing one would report
        health that was never measured.
        """

        if self.safe_stopped:
            return BackendHealth(
                False, f"safe stop is latched: {self.safe_stop_reason}", False
            )
        try:
            q = self.read_joint_positions()
        except Exception as exc:  # SDK/transport failure is exactly the case
            return BackendHealth(False, f"joint read failed: {type(exc).__name__}: {exc}", False)
        lower, upper = COMMAND_TARGET_LIMITS[:, 0], COMMAND_TARGET_LIMITS[:, 1]
        # A joint resting on a mechanical stop reads slightly past the software
        # limit (measured 0.4 mrad), so this reports rather than fails.
        outside = np.flatnonzero((q < lower) | (q > upper))
        enabled = "no motor enabled" if self.enabled_mask is None else (
            f"{int(np.count_nonzero(self.enabled_mask))}/{HAND_JOINTS} joints enabled"
        )
        if outside.size:
            names = [POLICY_JOINT_NAMES[i] for i in outside]
            overshoot = float(
                np.max(np.maximum(lower - q, q - upper)[outside])
            )
            return BackendHealth(
                True,
                f"{enabled}; {len(names)} joint(s) past the command limit by up to "
                f"{overshoot * 1000:.2f} mrad: {names}",
                True,
            )
        return BackendHealth(True, f"{enabled}; all joints inside the command limits", True)

    def safe_stop(self, reason: str = "unspecified") -> None:
        """Freeze the present command.  Does NOT disable, does NOT re-latch.

        Called by ``PolicyRunner`` when perception goes stale or lost -- i.e.
        mid-run, with the hand possibly holding two chopsticks.  Three things
        this must not do:

        * ``disable()`` -- motors off drops whatever is being held.  Ending the
          run is the operator's decision (Ctrl+C -> ``finally: disable()``), and
          "stop trusting the policy" is not the same event.
        * re-latch to the measured q -- ``q_target - q`` IS the grip preload
          (that gap is what presses the sticks).  Setting the target to where
          the joints currently sit releases exactly that force.
        * raise -- this runs on an error path that is already unwinding.

        Holding requires that SOMETHING keeps publishing the frozen target: the
        firmware's behaviour when commands simply stop arriving is UNVERIFIED
        and is not assumed here.  The caller's abort handler owns that loop.
        """

        self.safe_stopped = True
        self.safe_stop_reason = str(reason)
        try:
            if not self._target_initialized:
                # Nothing was ever commanded, so there is no grasp to preserve;
                # the present pose is the only defensible target.
                self.prime_target_to_current()
            if self.controller is not None:
                self.publish_latest_target(self.controller)
        except Exception:
            # Never mask the original failure with a teardown error.
            pass

    def realtime_controller(self, lowpass_hz: float, enable_upstream: bool):
        """Return the SDK's realtime controller context manager.

        ``enable_upstream`` must be true to read joint state through the
        controller during the loop.  ``move_middle_j1.py`` used false because it
        never read back; the reach policy is a residual on the measured q, so it
        does.
        """

        return self.hand.realtime_controller(
            enable_upstream=enable_upstream,
            filter=self._wujihandpy.filter.LowPass(cutoff_freq=float(lowpass_hz)),
        )
