"""Online step-response metrics independent of Isaac Sim and Qt."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import fmean
from typing import Any


@dataclass(slots=True)
class StepMetrics:
    """Metrics for one low-to-high or high-to-low target transition."""

    start_time: float
    elapsed_time: float
    start_position: float
    target_position: float
    current_error: float
    absolute_position_error: float
    maximum_absolute_error: float
    peak_velocity: float
    peak_absolute_effort: float | None
    percentage_overshoot: float
    rise_time: float | None
    settling_time: float | None
    steady_state_error: float
    effort_saturation_ratio: float
    sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StepResponseTracker:
    """Track 10%-90% rise, overshoot, settling, and effort use online."""

    def __init__(self, commanded_amplitude: float = 0.2) -> None:
        self.commanded_amplitude = abs(float(commanded_amplitude))
        self.last_completed: StepMetrics | None = None
        self.reset()

    def reset(self) -> None:
        """Forget the active transition while retaining no stale metrics."""

        self.active = False
        self.start_time = 0.0
        self.start_position = 0.0
        self.target_position = 0.0
        self._change = 0.0
        self._max_abs_error = 0.0
        self._peak_velocity = 0.0
        self._peak_effort: float | None = None
        self._max_overshoot = 0.0
        self._rise_10_time: float | None = None
        self._rise_90_time: float | None = None
        self._settling_candidate: float | None = None
        self._errors: list[float] = []
        self._saturated_samples = 0
        self._samples = 0
        self.last_completed = None

    def begin_transition(self, simulation_time: float, position: float, target: float) -> StepMetrics | None:
        """Finalize any active transition and start a new target transition."""

        completed = self.finalize(simulation_time) if self.active else None
        self.active = not math.isclose(position, target, rel_tol=0.0, abs_tol=1.0e-12)
        self.start_time = float(simulation_time)
        self.start_position = float(position)
        self.target_position = float(target)
        self._change = self.target_position - self.start_position
        self.commanded_amplitude = abs(self._change)
        self._max_abs_error = abs(self._change)
        self._peak_velocity = 0.0
        self._peak_effort = None
        self._max_overshoot = 0.0
        self._rise_10_time = None
        self._rise_90_time = None
        self._settling_candidate = None
        self._errors = []
        self._saturated_samples = 0
        self._samples = 0
        return completed

    @property
    def settling_tolerance(self) -> float:
        """Two percent of the step, with a 1 mrad lower bound."""

        return max(self.commanded_amplitude * 0.02, 0.001)

    def update(
        self,
        simulation_time: float,
        position: float,
        velocity: float,
        effort: float | None,
        saturated: bool,
    ) -> None:
        """Accumulate one sample for the current transition."""

        if not self.active:
            return
        elapsed = max(float(simulation_time) - self.start_time, 0.0)
        error = self.target_position - float(position)
        abs_error = abs(error)
        self._errors.append(error)
        self._samples += 1
        self._max_abs_error = max(self._max_abs_error, abs_error)
        self._peak_velocity = max(self._peak_velocity, abs(float(velocity)))
        if effort is not None and math.isfinite(effort):
            value = abs(float(effort))
            self._peak_effort = value if self._peak_effort is None else max(self._peak_effort, value)
        self._saturated_samples += int(bool(saturated))

        if abs(self._change) > 1.0e-12:
            progress = (float(position) - self.start_position) / self._change
            if self._rise_10_time is None and progress >= 0.10:
                self._rise_10_time = elapsed
            if self._rise_90_time is None and progress >= 0.90:
                self._rise_90_time = elapsed
            direction = 1.0 if self._change > 0.0 else -1.0
            overshoot = max(direction * (float(position) - self.target_position), 0.0)
            self._max_overshoot = max(self._max_overshoot, overshoot)

        if abs_error <= self.settling_tolerance:
            if self._settling_candidate is None:
                self._settling_candidate = elapsed
        else:
            self._settling_candidate = None

    def snapshot(self, simulation_time: float) -> StepMetrics | None:
        """Return current metrics without ending the transition."""

        if not self.active:
            return None
        error = self._errors[-1] if self._errors else self.target_position - self.start_position
        tail_count = max(1, int(math.ceil(0.10 * len(self._errors))))
        steady = fmean(self._errors[-tail_count:]) if self._errors else error
        rise = (
            self._rise_90_time - self._rise_10_time
            if self._rise_10_time is not None and self._rise_90_time is not None
            else None
        )
        overshoot = (
            100.0 * self._max_overshoot / abs(self._change)
            if abs(self._change) > 1.0e-12
            else 0.0
        )
        return StepMetrics(
            start_time=self.start_time,
            elapsed_time=max(float(simulation_time) - self.start_time, 0.0),
            start_position=self.start_position,
            target_position=self.target_position,
            current_error=error,
            absolute_position_error=abs(error),
            maximum_absolute_error=self._max_abs_error,
            peak_velocity=self._peak_velocity,
            peak_absolute_effort=self._peak_effort,
            percentage_overshoot=overshoot,
            rise_time=rise,
            settling_time=self._settling_candidate,
            steady_state_error=steady,
            effort_saturation_ratio=(self._saturated_samples / self._samples if self._samples else 0.0),
            sample_count=self._samples,
        )

    def finalize(self, simulation_time: float) -> StepMetrics | None:
        """Finish and retain the current transition."""

        metrics = self.snapshot(simulation_time)
        if metrics is not None:
            self.last_completed = metrics
        self.active = False
        return metrics
