"""Periodic, joint-limit-aware position step command generation."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class StepSample:
    """One requested and joint-limit-clamped target sample."""

    requested_target: float
    applied_target: float
    clamped: bool
    phase: str


class PeriodicStepSignal:
    """Alternate between ``q0`` and ``q0 + direction * amplitude``."""

    def __init__(
        self,
        amplitude: float = 0.2,
        period: float = 1.0,
        initial_delay: float = 0.25,
        direction: int = 1,
        repeat: bool = True,
    ) -> None:
        self.configure(amplitude, period, initial_delay, direction, repeat)
        self.q0 = 0.0
        self.active = False
        self.start_time = 0.0

    def configure(
        self,
        amplitude: float,
        period: float,
        initial_delay: float,
        direction: int,
        repeat: bool,
    ) -> None:
        """Validate and store step parameters."""

        values = (amplitude, period, initial_delay)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Step parameters must be finite.")
        if amplitude < 0.0:
            raise ValueError("Step amplitude must be non-negative; use direction for sign.")
        if period <= 0.0:
            raise ValueError("Step period must be positive.")
        if initial_delay < 0.0:
            raise ValueError("Initial delay must be non-negative.")
        if direction not in (-1, 1):
            raise ValueError("Step direction must be +1 or -1.")
        self.amplitude = float(amplitude)
        self.period = float(period)
        self.initial_delay = float(initial_delay)
        self.direction = int(direction)
        self.repeat = bool(repeat)

    def restart(self, simulation_time: float, q0: float, active: bool = True) -> None:
        """Capture a new baseline and restart the phase at its low target."""

        self.q0 = float(q0)
        self.start_time = float(simulation_time)
        self.active = bool(active)

    def pause(self) -> None:
        """Stop waveform advancement and command the low target."""

        self.active = False

    def sample(self, simulation_time: float, lower_limit: float, upper_limit: float) -> StepSample:
        """Return the current square-wave target, clamped to the joint limits."""

        elapsed = max(float(simulation_time) - self.start_time, 0.0)
        phase = "paused"
        requested = self.q0
        if self.active:
            if elapsed < self.initial_delay:
                phase = "delay_low"
            else:
                wave_time = elapsed - self.initial_delay
                if not self.repeat and wave_time >= self.period:
                    self.active = False
                    phase = "complete_low"
                else:
                    phase_time = wave_time % self.period
                    if phase_time < 0.5 * self.period:
                        phase = "low"
                    else:
                        phase = "high"
                        requested = self.q0 + self.direction * self.amplitude
        applied = min(max(requested, lower_limit), upper_limit)
        return StepSample(
            requested_target=requested,
            applied_target=applied,
            clamped=not math.isclose(requested, applied, rel_tol=0.0, abs_tol=1.0e-12),
            phase=phase,
        )
