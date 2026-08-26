"""Preset/custom normalized ranking and strict fallback selection."""

from __future__ import annotations

import math
from typing import Iterable

from .config import AutoTunePreset, METRIC_NAMES, ResolvedAutoTuneConfig
from .constraints import classify_candidate
from .result import AutoTuneOutcome, CandidateResult, CandidateStatus


PRESET_WEIGHTS: dict[AutoTunePreset, dict[str, float]] = {
    AutoTunePreset.BALANCED: {
        "settling_time": 6,
        "overshoot": 5,
        "rms_applied_effort": 4,
        "steady_state_error": 3,
        "peak_applied_effort": 2,
        "gain_magnitude": 1,
    },
    AutoTunePreset.FAST_RESPONSE: {
        "settling_time": 6,
        "steady_state_error": 5,
        "overshoot": 4,
        "peak_applied_effort": 3,
        "rms_applied_effort": 2,
        "gain_magnitude": 1,
    },
    AutoTunePreset.SMOOTH_RESPONSE: {
        "overshoot": 6,
        "settling_time": 5,
        "peak_applied_effort": 4,
        "rms_applied_effort": 3,
        "steady_state_error": 2,
        "gain_magnitude": 1,
    },
    AutoTunePreset.LOW_TORQUE: {
        "rms_applied_effort": 6,
        "peak_applied_effort": 5,
        "settling_time": 4,
        "overshoot": 3,
        "steady_state_error": 2,
        "gain_magnitude": 1,
    },
    AutoTunePreset.CONVERGENCE_FIRST: {
        "settling_time": 10,
        "steady_state_error": 8,
        "overshoot": 3,
        "rms_applied_effort": 1,
        "peak_applied_effort": 1,
        "gain_magnitude": 0.5,
    },
}


def resolve_ranking_weights(
    config: ResolvedAutoTuneConfig,
) -> tuple[dict[str, float], str | None]:
    """Normalize custom/preset weights; all-zero custom falls back visibly."""

    warning = None
    if config.preset is AutoTunePreset.CUSTOM:
        raw = {name: max(0.0, float(config.custom_weights.get(name, 0.0))) for name in METRIC_NAMES}
        if sum(raw.values()) <= 0.0:
            warning = "All custom weights were zero; Balanced preset was used."
            raw = dict(PRESET_WEIGHTS[AutoTunePreset.BALANCED])
    else:
        raw = dict(PRESET_WEIGHTS.get(config.preset, PRESET_WEIGHTS[AutoTunePreset.BALANCED]))
    total = sum(raw.values())
    return ({name: value / total for name, value in raw.items()}, warning)


def _metric_value(result: CandidateResult, name: str, config: ResolvedAutoTuneConfig) -> float:
    if name == "settling_time":
        return result.settling_time if result.settling_time is not None else config.hold_duration * 2.0
    if name == "overshoot":
        return result.percentage_overshoot
    if name == "rms_applied_effort":
        return result.rms_applied_effort if result.rms_applied_effort is not None else math.inf
    if name == "peak_applied_effort":
        return result.peak_applied_effort if result.peak_applied_effort is not None else math.inf
    if name == "steady_state_error":
        return result.steady_state_error
    if name == "gain_magnitude":
        kp_span = max(config.kp_max - config.kp_min, 1.0e-12)
        kd_span = max(config.kd_max - config.kd_min, 1.0e-12)
        kp_norm = (result.kp - config.kp_min) / kp_span
        kd_norm = (result.kd - config.kd_min) / kd_span
        return math.sqrt(kp_norm * kp_norm + kd_norm * kd_norm)
    raise KeyError(name)


def _score_candidates(
    candidates: list[CandidateResult],
    config: ResolvedAutoTuneConfig,
    weights: dict[str, float],
) -> None:
    if not candidates:
        return
    metric_values: dict[str, list[float]] = {
        name: [_metric_value(candidate, name, config) for candidate in candidates]
        for name in METRIC_NAMES
    }
    for name, values in metric_values.items():
        finite = [value for value in values if math.isfinite(value)]
        replacement = (max(finite) * 2.0 + 1.0) if finite else 1.0
        metric_values[name] = [value if math.isfinite(value) else replacement for value in values]
    for index, candidate in enumerate(candidates):
        normalized: dict[str, float] = {}
        for name in METRIC_NAMES:
            values = metric_values[name]
            low, high = min(values), max(values)
            normalized[name] = 0.0 if math.isclose(low, high) else (values[index] - low) / (high - low)
        candidate.normalized_scores = normalized
        candidate.total_score = sum(weights[name] * normalized[name] for name in METRIC_NAMES)


def rank_candidates(
    candidates: Iterable[CandidateResult],
    config: ResolvedAutoTuneConfig,
) -> AutoTuneOutcome:
    """Pick feasible first; otherwise select a clearly labelled safe fallback."""

    results = [classify_candidate(candidate, config) for candidate in candidates]
    weights, warning = resolve_ranking_weights(config)
    hard_passed = [candidate for candidate in results if candidate.hard_constraint_passed]
    _score_candidates(hard_passed, config, weights)
    feasible = [candidate for candidate in hard_passed if candidate.performance_constraint_passed]
    best_feasible = min(
        feasible,
        key=lambda candidate: (candidate.total_score if candidate.total_score is not None else math.inf, candidate.candidate.candidate_id),
        default=None,
    )
    if best_feasible is not None:
        return AutoTuneOutcome(
            candidates=results,
            best_feasible=best_feasible,
            best_fallback=None,
            selected=best_feasible,
            fully_feasible=True,
            selection_reason="Best fully feasible candidate by normalized preset/custom score.",
            ranking_warning=warning,
        )

    settled = [candidate for candidate in hard_passed if candidate.settling_time is not None]
    if settled:
        fallback = min(
            settled,
            key=lambda candidate: (
                float(candidate.settling_time),
                candidate.total_score if candidate.total_score is not None else math.inf,
                candidate.candidate.candidate_id,
            ),
        )
        reason = (
            "No fully feasible candidate; selected the fastest settled candidate that passed every hard constraint."
        )
    elif hard_passed:
        fallback = min(
            hard_passed,
            key=lambda candidate: (
                candidate.steady_state_error,
                candidate.total_score if candidate.total_score is not None else math.inf,
                candidate.candidate.candidate_id,
            ),
        )
        reason = (
            "No candidate settled; selected the smallest steady-state-error candidate that passed every hard constraint."
        )
    else:
        fallback = None
        reason = "No candidate passed the non-relaxable hard constraints."
    if fallback is not None:
        fallback.status = CandidateStatus.FALLBACK
    return AutoTuneOutcome(
        candidates=results,
        best_feasible=None,
        best_fallback=fallback,
        selected=fallback,
        fully_feasible=False,
        selection_reason=reason,
        ranking_warning=warning,
    )
