"""Hard and user-performance candidate classification."""

from __future__ import annotations

from .config import ResolvedAutoTuneConfig
from .result import CandidateResult, CandidateStatus


def classify_candidate(
    result: CandidateResult,
    config: ResolvedAutoTuneConfig,
) -> CandidateResult:
    """Classify without ever relaxing a hard safety constraint."""

    if result.status is CandidateStatus.CANCELLED:
        return result
    if not result.hard_constraint_passed:
        result.status = CandidateStatus.HARD_CONSTRAINT_FAILED
        result.performance_constraint_passed = False
        return result
    violations: list[str] = []
    if result.settling_time is None:
        violations.append(
            f"not settled within hold duration {config.hold_duration:.6g} s"
        )
    elif result.settling_time > config.target_settling_time + 1.0e-12:
        violations.append(
            f"settling time {result.settling_time:.6g} s > "
            f"requested {config.target_settling_time:.6g} s"
        )
    if (
        config.maximum_overshoot is not None
        and result.percentage_overshoot > config.maximum_overshoot + 1.0e-12
    ):
        violations.append(
            f"overshoot {result.percentage_overshoot:.6g}% > "
            f"requested {config.maximum_overshoot:.6g}%"
        )
    if (
        config.maximum_steady_state_error is not None
        and result.steady_state_error > config.maximum_steady_state_error + 1.0e-12
    ):
        violations.append(
            f"steady-state error {result.steady_state_error:.6g} rad > "
            f"requested {config.maximum_steady_state_error:.6g} rad"
        )
    result.performance_violations = violations
    result.performance_constraint_passed = not violations
    if not violations:
        result.status = CandidateStatus.FEASIBLE
    elif result.settling_time is None:
        result.status = CandidateStatus.NOT_SETTLED
    else:
        result.status = CandidateStatus.PERFORMANCE_CONSTRAINT_FAILED
    return result
