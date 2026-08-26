"""Second-order closed-loop identification from measured joint step responses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

from .metrics import TrialMetrics, TrialSample
from .result import CandidateResult


@dataclass(frozen=True, slots=True)
class SecondOrderEstimate:
    """Local effective joint model inferred at one tested Kp/Kd operating point."""

    natural_frequency: float
    damping_ratio: float
    effective_inertia: float
    passive_damping: float
    normalized_rmse: float
    method: str
    source_candidate_id: int
    source_direction: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _unit_step_response(time_value: float, natural_frequency: float, damping_ratio: float) -> float:
    """Return the canonical second-order unit-step response at ``time_value``."""

    t = max(float(time_value), 0.0)
    wn = float(natural_frequency)
    zeta = float(damping_ratio)
    if zeta < 1.0 - 1.0e-8:
        root = math.sqrt(max(1.0 - zeta * zeta, 1.0e-12))
        wd = wn * root
        return 1.0 - math.exp(-zeta * wn * t) * (
            math.cos(wd * t) + zeta * math.sin(wd * t) / root
        )
    if zeta <= 1.0 + 1.0e-8:
        return 1.0 - math.exp(-wn * t) * (1.0 + wn * t)
    root = math.sqrt(zeta * zeta - 1.0)
    r1 = -wn * (zeta - root)
    r2 = -wn * (zeta + root)
    denominator = r1 - r2
    return 1.0 + (r2 * math.exp(r1 * t) - r1 * math.exp(r2 * t)) / denominator


def _logspace(low: float, high: float, count: int) -> list[float]:
    if count <= 1 or math.isclose(low, high):
        return [math.sqrt(low * high)]
    low_log = math.log(low)
    span = math.log(high) - low_log
    return [math.exp(low_log + span * index / (count - 1)) for index in range(count)]


def _downsample(samples: list[tuple[float, float]], maximum: int = 180) -> list[tuple[float, float]]:
    if len(samples) <= maximum:
        return samples
    indexes = {
        round(index * (len(samples) - 1) / (maximum - 1))
        for index in range(maximum)
    }
    return [samples[index] for index in sorted(indexes)]


def _normalized_samples(trial: TrialMetrics) -> list[tuple[float, float]]:
    sign = 1.0 if trial.direction == "positive" else -1.0
    change = sign * float(trial.actual_step_amplitude)
    if abs(change) <= 1.0e-12:
        return []
    start_position = float(trial.applied_target) - change
    normalized = [
        (
            float(sample.elapsed_time),
            (float(sample.actual_position) - start_position) / change,
        )
        for sample in trial.time_series
        if math.isfinite(sample.elapsed_time) and math.isfinite(sample.actual_position)
    ]
    normalized.sort(key=lambda item: item[0])
    return _downsample(normalized)


def estimate_trial_response(
    trial: TrialMetrics,
    *,
    kp: float,
    kd: float,
    source_candidate_id: int,
) -> SecondOrderEstimate | None:
    """Fit a local second-order model without SciPy or an assumed link inertia.

    A small deterministic grid fits the canonical unit-step response.  The
    resulting closed-loop ``wn`` and ``zeta`` are then combined with the gain
    used for the trial to estimate ``J_eff`` and passive damping.  Saturated or
    barely moving trials are deliberately rejected because they are not valid
    linear step responses.
    """

    if not trial.hard_constraint_passed or trial.saturation_count or kp <= 1.0e-10:
        return None
    samples = _normalized_samples(trial)
    if len(samples) < 8:
        return None
    duration = samples[-1][0] - samples[0][0]
    if duration <= 0.0:
        return None
    response_span = max(value for _, value in samples) - min(value for _, value in samples)
    response_peak = max(abs(value) for _, value in samples)
    if response_span < 0.03 and response_peak < 0.05:
        return None

    time_steps = [
        later[0] - earlier[0]
        for earlier, later in zip(samples, samples[1:])
        if later[0] > earlier[0]
    ]
    dt = min(time_steps) if time_steps else duration / max(len(samples) - 1, 1)
    wn_low = max(0.15 / duration, 1.0e-3)
    # Keep several physics samples per fitted oscillation; faster content is
    # indistinguishable from numerical/impact noise in this telemetry.
    wn_high = max(wn_low * 1.01, min(0.20 * math.pi / max(dt, 1.0e-6), 250.0))
    wn_values = _logspace(wn_low, wn_high, 72)
    zeta_values = [0.10 + 0.05 * index for index in range(39)]  # 0.10 .. 2.00

    best: tuple[float, float, float] | None = None
    for zeta in zeta_values:
        for wn in wn_values:
            squared_error = 0.0
            for time_value, actual in samples:
                predicted = _unit_step_response(time_value, wn, zeta)
                squared_error += (predicted - actual) ** 2
            rmse = math.sqrt(squared_error / len(samples))
            if best is None or rmse < best[0]:
                best = (rmse, wn, zeta)
    if best is None:
        return None
    rmse, wn, zeta = best
    scale = max(response_peak, 0.25)
    normalized_rmse = rmse / scale
    if not math.isfinite(normalized_rmse) or normalized_rmse > 0.45:
        return None
    effective_inertia = float(kp) / (wn * wn)
    passive_damping = max(0.0, 2.0 * zeta * effective_inertia * wn - float(kd))
    if not all(math.isfinite(value) for value in (effective_inertia, passive_damping)):
        return None
    return SecondOrderEstimate(
        natural_frequency=wn,
        damping_ratio=zeta,
        effective_inertia=effective_inertia,
        passive_damping=passive_damping,
        normalized_rmse=normalized_rmse,
        method="deterministic_second_order_step_fit",
        source_candidate_id=int(source_candidate_id),
        source_direction=trial.direction,
    )


def estimate_candidate_response(result: CandidateResult) -> SecondOrderEstimate | None:
    """Return the best valid repeat/direction model for one candidate."""

    estimates = [
        estimate_trial_response(
            trial,
            kp=result.kp,
            kd=result.kd,
            source_candidate_id=result.candidate.candidate_id,
        )
        for trial in result.trials
    ]
    valid = [estimate for estimate in estimates if estimate is not None]
    return min(valid, key=lambda item: item.normalized_rmse, default=None)


def best_response_estimate(results: Iterable[CandidateResult]) -> SecondOrderEstimate | None:
    """Choose the most trustworthy model from safe measured candidates."""

    estimates = [estimate_candidate_response(result) for result in results]
    valid = [estimate for estimate in estimates if estimate is not None]
    return min(valid, key=lambda item: (item.normalized_rmse, -item.source_candidate_id), default=None)
