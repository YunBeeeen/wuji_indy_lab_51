"""Configurable Auto Tune trial metrics and hard safety checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class TrialSample:
    """One physics-step sample from an outward Auto Tune transition."""

    elapsed_time: float
    target_position: float
    actual_position: float
    joint_velocity: float
    computed_effort: float | None
    applied_effort: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrialMetrics:
    """Metrics and exact hard-constraint result for one direction/repeat."""

    direction: str
    repeat_index: int
    requested_target: float
    applied_target: float
    target_clamped: bool
    actual_step_amplitude: float
    settling_time: float | None
    percentage_overshoot: float
    steady_state_error: float
    rms_computed_effort: float | None
    rms_applied_effort: float | None
    peak_computed_effort: float | None
    peak_applied_effort: float | None
    maximum_velocity: float
    saturation_count: int
    saturation_ratio: float
    sample_count: int
    hard_constraint_passed: bool
    hard_failure_reasons: tuple[str, ...]
    time_series: tuple[TrialSample, ...]

    def to_dict(self, *, include_time_series: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result["hard_failure_reasons"] = list(self.hard_failure_reasons)
        if not include_time_series:
            result.pop("time_series", None)
        else:
            result["time_series"] = [sample.to_dict() for sample in self.time_series]
        return result


def _rms(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    if not data:
        return None
    return math.sqrt(sum(value * value for value in data) / len(data))


def hard_failure_reasons_for_sample(
    sample: TrialSample,
    *,
    effort_limit: float,
    torque_match_tolerance: float,
    maximum_velocity: float,
    lower_limit: float,
    upper_limit: float,
    allow_torque_saturation: bool = False,
) -> tuple[str, ...]:
    """Return exact non-relaxable failures for one physics sample."""

    reasons: list[str] = []
    scalar_values = (
        sample.elapsed_time,
        sample.target_position,
        sample.actual_position,
        sample.joint_velocity,
    )
    if not all(math.isfinite(value) for value in scalar_values):
        reasons.append("non-finite position, velocity, target, or time")
    if abs(sample.joint_velocity) > maximum_velocity + 1.0e-9:
        reasons.append(
            f"maximum velocity exceeded: {abs(sample.joint_velocity):.6g} > "
            f"{maximum_velocity:.6g} rad/s"
        )
    if sample.actual_position < lower_limit - 1.0e-4 or sample.actual_position > upper_limit + 1.0e-4:
        reasons.append("hard joint position limit violated")
    if sample.computed_effort is None or sample.applied_effort is None:
        reasons.append("computed/applied effort signal unavailable")
    elif not math.isfinite(sample.computed_effort) or not math.isfinite(sample.applied_effort):
        reasons.append("non-finite computed/applied effort")
    elif not allow_torque_saturation and (
        abs(sample.computed_effort) > effort_limit + torque_match_tolerance
        or abs(sample.computed_effort - sample.applied_effort) > torque_match_tolerance
    ):
        reasons.append("torque saturation detected")
    return tuple(reasons)


def evaluate_trial(
    samples: list[TrialSample],
    *,
    direction: str,
    repeat_index: int,
    start_position: float,
    requested_target: float,
    applied_target: float,
    target_clamped: bool,
    settling_tolerance: float,
    settling_hold_time: float,
    effort_limit: float,
    torque_match_tolerance: float,
    maximum_velocity: float,
    lower_limit: float,
    upper_limit: float,
    allow_torque_saturation: bool = False,
) -> TrialMetrics:
    """Evaluate a complete or hard-failed trial from actual physics samples.

    Settling time is the beginning of the final uninterrupted interval that is
    inside tolerance for at least ``settling_hold_time``.  A later excursion
    invalidates an earlier temporary entry.  Steady-state error is the mean
    absolute error over the final max(10% of trial, settling hold) window.
    """

    if not samples:
        raise ValueError("Cannot evaluate an Auto Tune trial without samples.")
    change = applied_target - start_position
    if abs(change) <= 1.0e-12:
        raise ValueError("Applied Auto Tune step amplitude is zero.")

    failure_reasons: list[str] = []
    saturation_count = 0
    computed_values: list[float] = []
    applied_values: list[float] = []
    maximum_observed_velocity = 0.0
    overshoot_distance = 0.0
    final_in_band_start: float | None = None
    direction_sign = 1.0 if change > 0.0 else -1.0

    def fail(reason: str) -> None:
        if reason not in failure_reasons:
            failure_reasons.append(reason)

    for sample in samples:
        sample_failures = hard_failure_reasons_for_sample(
            sample,
            effort_limit=effort_limit,
            torque_match_tolerance=torque_match_tolerance,
            maximum_velocity=maximum_velocity,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            allow_torque_saturation=allow_torque_saturation,
        )
        for reason in sample_failures:
            if reason != "torque saturation detected":
                fail(reason)
        maximum_observed_velocity = max(maximum_observed_velocity, abs(sample.joint_velocity))
        if sample.computed_effort is None or sample.applied_effort is None:
            pass
        elif not math.isfinite(sample.computed_effort) or not math.isfinite(sample.applied_effort):
            pass
        else:
            computed_values.append(sample.computed_effort)
            applied_values.append(sample.applied_effort)
            saturated = (
                abs(sample.computed_effort) > effort_limit + torque_match_tolerance
                or abs(sample.computed_effort - sample.applied_effort) > torque_match_tolerance
            )
            if saturated:
                saturation_count += 1
        error = applied_target - sample.actual_position
        if abs(error) <= settling_tolerance:
            if final_in_band_start is None:
                final_in_band_start = sample.elapsed_time
        else:
            final_in_band_start = None
        overshoot_distance = max(
            overshoot_distance,
            max(direction_sign * (sample.actual_position - applied_target), 0.0),
        )

    if saturation_count and not allow_torque_saturation:
        fail(f"torque saturation detected in {saturation_count} sample(s)")
    end_time = samples[-1].elapsed_time
    settling_time = None
    if final_in_band_start is not None and end_time - final_in_band_start + 1.0e-12 >= settling_hold_time:
        settling_time = final_in_band_start

    tail_duration = max(settling_hold_time, 0.10 * max(end_time, 0.0))
    tail_start = max(0.0, end_time - tail_duration)
    tail_errors = [
        abs(applied_target - sample.actual_position)
        for sample in samples
        if sample.elapsed_time + 1.0e-12 >= tail_start
    ]
    steady_state_error = sum(tail_errors) / len(tail_errors) if tail_errors else abs(change)
    percentage_overshoot = 100.0 * overshoot_distance / abs(change)
    return TrialMetrics(
        direction=direction,
        repeat_index=repeat_index,
        requested_target=requested_target,
        applied_target=applied_target,
        target_clamped=target_clamped,
        actual_step_amplitude=abs(change),
        settling_time=settling_time,
        percentage_overshoot=percentage_overshoot,
        steady_state_error=steady_state_error,
        rms_computed_effort=_rms(computed_values),
        rms_applied_effort=_rms(applied_values),
        peak_computed_effort=(max((abs(value) for value in computed_values), default=None)),
        peak_applied_effort=(max((abs(value) for value in applied_values), default=None)),
        maximum_velocity=maximum_observed_velocity,
        saturation_count=saturation_count,
        saturation_ratio=saturation_count / len(samples),
        sample_count=len(samples),
        hard_constraint_passed=not failure_reasons,
        hard_failure_reasons=tuple(failure_reasons),
        time_series=tuple(samples),
    )
