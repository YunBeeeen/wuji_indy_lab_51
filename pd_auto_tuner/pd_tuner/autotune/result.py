"""Serializable candidates, aggregate results, and final Auto Tune outcome."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .metrics import TrialMetrics


class CandidateStatus(str, Enum):
    """Visible Auto Tune candidate classifications."""

    PENDING = "PENDING"
    FEASIBLE = "FEASIBLE"
    FALLBACK = "FALLBACK"
    HARD_CONSTRAINT_FAILED = "HARD_CONSTRAINT_FAILED"
    PERFORMANCE_CONSTRAINT_FAILED = "PERFORMANCE_CONSTRAINT_FAILED"
    NOT_SETTLED = "NOT_SETTLED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One deterministic staged Kp/Kd point and its generation evidence."""

    candidate_id: int
    kp: float
    kd: float
    stage: str
    predicted_effort_fraction: float | None = None
    generation_reason: str = ""
    model_estimate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateResult:
    """Conservative aggregation across directions and repeats."""

    candidate: CandidateSpec
    direction: str
    trials: list[TrialMetrics] = field(default_factory=list)
    status: CandidateStatus = CandidateStatus.PENDING
    settling_time: float | None = None
    percentage_overshoot: float = 0.0
    steady_state_error: float = 0.0
    rms_computed_effort: float | None = None
    rms_applied_effort: float | None = None
    peak_computed_effort: float | None = None
    peak_applied_effort: float | None = None
    maximum_velocity: float = 0.0
    saturation_count: int = 0
    hard_constraint_passed: bool = False
    hard_failure_reasons: list[str] = field(default_factory=list)
    performance_constraint_passed: bool = False
    performance_violations: list[str] = field(default_factory=list)
    normalized_scores: dict[str, float] = field(default_factory=dict)
    total_score: float | None = None
    representative_response: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def aggregate(
        cls,
        candidate: CandidateSpec,
        direction: str,
        trials: list[TrialMetrics],
    ) -> "CandidateResult":
        """Use the worst repeat/direction; RMS effort also uses maximum."""

        if not trials:
            return cls(candidate=candidate, direction=direction, status=CandidateStatus.CANCELLED)
        failures: list[str] = []
        for trial in trials:
            for reason in trial.hard_failure_reasons:
                if reason not in failures:
                    failures.append(reason)
        all_settled = all(trial.settling_time is not None for trial in trials)
        settling = max(float(trial.settling_time) for trial in trials) if all_settled else None

        def optional_max(name: str) -> float | None:
            values = [getattr(trial, name) for trial in trials if getattr(trial, name) is not None]
            return max(float(value) for value in values) if values else None

        representative = max(
            trials,
            key=lambda trial: (
                trial.settling_time is None,
                trial.settling_time or 0.0,
                trial.steady_state_error,
                trial.percentage_overshoot,
            ),
        )
        hard_passed = all(trial.hard_constraint_passed for trial in trials)
        return cls(
            candidate=candidate,
            direction=direction,
            trials=trials,
            status=(
                CandidateStatus.PENDING if hard_passed else CandidateStatus.HARD_CONSTRAINT_FAILED
            ),
            settling_time=settling,
            percentage_overshoot=max(trial.percentage_overshoot for trial in trials),
            steady_state_error=max(trial.steady_state_error for trial in trials),
            rms_computed_effort=optional_max("rms_computed_effort"),
            rms_applied_effort=optional_max("rms_applied_effort"),
            peak_computed_effort=optional_max("peak_computed_effort"),
            peak_applied_effort=optional_max("peak_applied_effort"),
            maximum_velocity=max(trial.maximum_velocity for trial in trials),
            saturation_count=sum(trial.saturation_count for trial in trials),
            hard_constraint_passed=hard_passed,
            hard_failure_reasons=failures,
            representative_response=[sample.to_dict() for sample in representative.time_series],
        )

    @property
    def kp(self) -> float:
        return self.candidate.kp

    @property
    def kd(self) -> float:
        return self.candidate.kd

    def to_dict(self, *, include_trial_time_series: bool = False) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "direction": self.direction,
            "trials": [
                trial.to_dict(include_time_series=include_trial_time_series) for trial in self.trials
            ],
            "status": self.status.value,
            "settling_time": self.settling_time,
            "percentage_overshoot": self.percentage_overshoot,
            "steady_state_error": self.steady_state_error,
            "rms_computed_effort": self.rms_computed_effort,
            "rms_applied_effort": self.rms_applied_effort,
            "peak_computed_effort": self.peak_computed_effort,
            "peak_applied_effort": self.peak_applied_effort,
            "maximum_velocity": self.maximum_velocity,
            "saturation_count": self.saturation_count,
            "hard_constraint_passed": self.hard_constraint_passed,
            "hard_failure_reasons": list(self.hard_failure_reasons),
            "performance_constraint_passed": self.performance_constraint_passed,
            "performance_violations": list(self.performance_violations),
            "normalized_scores": dict(self.normalized_scores),
            "total_score": self.total_score,
            "representative_response": list(self.representative_response),
        }


@dataclass(slots=True)
class AutoTuneOutcome:
    """Ranked final result, including explicit no-feasible fallback state."""

    candidates: list[CandidateResult]
    best_feasible: CandidateResult | None
    best_fallback: CandidateResult | None
    selected: CandidateResult | None
    fully_feasible: bool
    selection_reason: str
    ranking_warning: str | None = None
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "best_feasible_candidate_id": (
                self.best_feasible.candidate.candidate_id if self.best_feasible else None
            ),
            "best_fallback_candidate_id": (
                self.best_fallback.candidate.candidate_id if self.best_fallback else None
            ),
            "selected_candidate_id": self.selected.candidate.candidate_id if self.selected else None,
            "fully_feasible": self.fully_feasible,
            "selection_reason": self.selection_reason,
            "ranking_warning": self.ranking_warning,
            "cancelled": self.cancelled,
        }
