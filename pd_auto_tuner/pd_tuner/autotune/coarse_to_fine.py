"""Deterministic response-identification-guided PD search."""

from __future__ import annotations

import math

from .config import AutoTunePreset, AutoTuneTorquePolicy
from .identification import SecondOrderEstimate, best_response_estimate
from .result import CandidateResult, CandidateSpec
from .search_base import AutoTuneSearchStrategy


def _deduplicate(values: list[float]) -> list[float]:
    result: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = round(float(value), 14)
        if key not in seen:
            seen.add(key)
            result.append(float(value))
    return result


class CoarseToFineSearch(AutoTuneSearchStrategy):
    """Tune like an operator: safe Kp probes, model target, then corrections.

    The strategy does not enumerate a rectangular Kp/Kd grid.  It first runs a
    few increasing Kp-only probes inside the initial step-torque limit.  Their
    measured time series identify a local second-order ``wn``, ``zeta``,
    effective inertia, and passive damping.  The requested settling time then
    gives a calculated Kp/Kd target.  Remaining trials are one-at-a-time
    corrections: increase Kp for slow/non-settling responses and increase Kd
    for excess overshoot.  Exact simulator torque checks remain authoritative.
    """

    @property
    def name(self) -> str:
        return "response_identification_guided_v3"

    @property
    def effective_budget(self) -> int:
        return self.config.search_budget

    @property
    def response_velocity(self) -> float:
        return self.config.step_amplitude / self.config.target_settling_time

    @property
    def effort_fraction_cap(self) -> float:
        return 3.0 if self.config.torque_policy is AutoTuneTorquePolicy.ALLOW_CLIPPING else 0.95

    @property
    def safe_kp_high(self) -> float:
        if self.config.torque_policy is AutoTuneTorquePolicy.ALLOW_CLIPPING:
            return self.config.kp_max
        exact_cap = (
            self.config.effort_limit - self.config.torque_match_tolerance
        ) / self.config.step_amplitude
        return min(self.config.kp_max, exact_cap)

    def _effort_fraction(self, kp: float, kd: float) -> float:
        envelope = kp * self.config.step_amplitude + kd * self.response_velocity
        return envelope / self.config.effort_limit

    def _bounded(self, kp: float, kd: float) -> tuple[float, float] | None:
        kp = min(max(float(kp), self.config.kp_min), self.safe_kp_high)
        kd = min(max(float(kd), self.config.kd_min), self.config.kd_max)
        if kp <= 0.0:
            return None
        if self.config.torque_policy is AutoTuneTorquePolicy.STRICT_NO_SATURATION:
            damping_headroom = (
                self.effort_fraction_cap * self.config.effort_limit
                - kp * self.config.step_amplitude
            )
            safe_kd = max(0.0, damping_headroom) / max(self.response_velocity, 1.0e-12)
            kd = min(kd, safe_kd)
        if kd < self.config.kd_min - 1.0e-12:
            return None
        return kp, kd

    def _spec(
        self,
        candidate_id: int,
        kp: float,
        kd: float,
        stage: str,
        reason: str,
        estimate: SecondOrderEstimate | None = None,
    ) -> CandidateSpec:
        return CandidateSpec(
            candidate_id=candidate_id,
            kp=float(kp),
            kd=float(kd),
            stage=stage,
            predicted_effort_fraction=float(self._effort_fraction(kp, kd)),
            generation_reason=reason,
            model_estimate=estimate.to_dict() if estimate is not None else None,
        )

    @staticmethod
    def _tested_pairs(results: list[CandidateResult]) -> set[tuple[float, float]]:
        return {(round(item.kp, 12), round(item.kd, 12)) for item in results}

    def _target_damping_ratio(self) -> float:
        return {
            AutoTunePreset.FAST_RESPONSE: 0.70,
            AutoTunePreset.SMOOTH_RESPONSE: 1.00,
            AutoTunePreset.LOW_TORQUE: 0.90,
            AutoTunePreset.CONVERGENCE_FIRST: 0.75,
        }.get(self.config.preset, 0.80)

    def _gains_from_model(
        self,
        estimate: SecondOrderEstimate,
        frequency_scale: float = 1.0,
        damping_scale: float = 1.0,
    ) -> tuple[float, float] | None:
        zeta_target = self._target_damping_ratio() * damping_scale
        wn_target = (
            4.0 / (max(zeta_target, 1.0e-6) * self.config.target_settling_time)
        ) * frequency_scale
        kp = estimate.effective_inertia * wn_target * wn_target
        kd = (
            2.0 * zeta_target * estimate.effective_inertia * wn_target
            - estimate.passive_damping
        )
        return self._bounded(kp, kd)

    def coarse_candidates(self) -> list[CandidateSpec]:
        """Return at most four increasing effort-safe Kp-only probes."""

        count = min(self.effective_budget, 4)
        if count <= 0 or self.safe_kp_high < self.config.kp_min:
            return []
        fractions_by_count = {
            1: (0.60,),
            2: (0.35, 0.85),
            3: (0.25, 0.55, 0.90),
            4: (0.20, 0.45, 0.70, 0.95),
        }
        low = max(self.config.kp_min, 0.0)
        high = self.safe_kp_high
        values = [low + fraction * (high - low) for fraction in fractions_by_count[count]]
        current = self.config.current_kp
        if 0.0 < current <= high and current >= low:
            closest = min(range(len(values)), key=lambda index: abs(values[index] - current))
            values[closest] = current
        values = sorted(_deduplicate([value for value in values if value > 0.0]))
        return [
            self._spec(
                index + 1,
                kp,
                self.config.kd_min,
                "kp_probe",
                (
                    "Increasing Kp-only probe; initial P-effort is "
                    f"{100.0 * kp * self.config.step_amplitude / self.config.effort_limit:.1f}% "
                    "of the selected effort limit."
                ),
            )
            for index, kp in enumerate(values)
        ]

    def damping_candidates(
        self,
        kp_results: list[CandidateResult],
        already_tested: list[CandidateResult],
    ) -> list[CandidateSpec]:
        """Calculate target gains from identified dynamics, with safe fallback."""

        remaining = self.effective_budget - len(already_tested)
        if remaining <= 0:
            return []
        safe_results = [item for item in kp_results if item.hard_constraint_passed]
        if not safe_results:
            return []
        estimate = best_response_estimate(safe_results)
        existing = self._tested_pairs(already_tested)
        generated: list[tuple[float, float, str, str]] = []
        if estimate is not None:
            variants = (
                (1.00, 1.00, "identified_target", "Kp/Kd calculated from measured wn and zeta."),
                (1.00, 0.80, "identified_damping", "Calculated target with 20% less damping."),
                (1.00, 1.20, "identified_damping", "Calculated target with 20% more damping."),
                (0.88, 1.00, "identified_frequency", "Calculated target with 12% lower natural frequency."),
                (1.12, 1.00, "identified_frequency", "Calculated target with 12% higher natural frequency."),
            )
            for frequency_scale, damping_scale, stage, reason in variants:
                gains = self._gains_from_model(estimate, frequency_scale, damping_scale)
                if gains is not None:
                    generated.append((*gains, stage, reason))
        else:
            # A non-oscillating, clipped, or barely moving response cannot
            # identify wn/zeta reliably. Continue like a cautious operator:
            # use the best observed Kp and introduce one small damping probe.
            source = min(
                safe_results,
                key=lambda item: (
                    item.settling_time is None,
                    item.settling_time if item.settling_time is not None else math.inf,
                    item.steady_state_error,
                    item.percentage_overshoot,
                ),
            )
            kd_probe = max(
                self.config.current_kd,
                self.config.kd_min + 0.20 * (self.config.kd_max - self.config.kd_min),
            )
            gains = self._bounded(source.kp, kd_probe)
            if gains is not None:
                generated.append(
                    (
                        *gains,
                        "damping_probe",
                        "Step model was not identifiable; add one conservative Kd probe at the best safe Kp.",
                    )
                )

        specs: list[CandidateSpec] = []
        for kp, kd, stage, reason in generated:
            key = (round(kp, 12), round(kd, 12))
            if key in existing:
                continue
            existing.add(key)
            specs.append(
                self._spec(
                    len(already_tested) + len(specs) + 1,
                    kp,
                    kd,
                    stage,
                    reason,
                    estimate,
                )
            )
            if len(specs) >= min(remaining, 5):
                break
        return specs

    def fine_candidates(
        self,
        best: CandidateResult,
        already_tested: list[CandidateResult],
    ) -> list[CandidateSpec]:
        """Return one human-like correction based on the measured failure."""

        if len(already_tested) >= self.effective_budget:
            return []
        existing = self._tested_pairs(already_tested)
        estimate = best_response_estimate(
            item for item in already_tested if item.hard_constraint_passed
        )
        overshoot_limit = self.config.maximum_overshoot
        steady_limit = self.config.maximum_steady_state_error
        slow_or_not_settled = (
            best.settling_time is None
            or best.settling_time > self.config.target_settling_time
        )
        excessive_steady_error = (
            steady_limit is not None and best.steady_state_error > steady_limit
        )
        excessive_overshoot = (
            overshoot_limit is not None
            and best.percentage_overshoot > overshoot_limit
        )

        proposals: list[tuple[float, float, str, str]] = []
        if excessive_overshoot:
            for factor in (1.20, 1.40, 1.65):
                kd = max(
                    best.kd * factor,
                    best.kd + 0.05 * factor * (self.config.kd_max - self.config.kd_min),
                )
                proposals.append(
                    (
                        best.kp,
                        kd,
                        "adaptive_kd",
                        "Measured overshoot exceeded its limit, so Kd was increased while Kp was held.",
                    )
                )
        elif slow_or_not_settled or excessive_steady_error:
            for factor in (1.15, 1.30, 1.50):
                kp = best.kp * factor
                # Preserve approximately the same damping ratio while Kp is
                # increased; Kd scales with sqrt(Kp) for fixed inertia.
                kd = best.kd * math.sqrt(factor)
                proposals.append(
                    (
                        kp,
                        kd,
                        "adaptive_kp",
                        "Response was slow/not settled or retained steady error, so Kp was increased.",
                    )
                )
            if best.kd > self.config.kd_min:
                proposals.append(
                    (
                        best.kp,
                        0.80 * best.kd,
                        "adaptive_kd",
                        "Kp reached its useful range; test whether excess damping caused the slow response.",
                    )
                )
        elif self.config.preset is AutoTunePreset.LOW_TORQUE:
            for factor in (0.90, 0.80):
                proposals.append(
                    (
                        best.kp * factor,
                        best.kd * math.sqrt(factor),
                        "adaptive_effort",
                        "Feasible response: reduce gains while preserving approximate damping ratio.",
                    )
                )
        elif self.config.preset is AutoTunePreset.SMOOTH_RESPONSE:
            for factor in (1.10, 1.20):
                proposals.append(
                    (
                        best.kp,
                        max(best.kd * factor, best.kd + 0.02 * (self.config.kd_max - self.config.kd_min)),
                        "adaptive_smooth",
                        "Feasible response: test slightly higher damping for smoother motion.",
                    )
                )
        else:
            for factor in (0.95, 1.08):
                proposals.append(
                    (
                        best.kp * factor,
                        best.kd * math.sqrt(factor),
                        "adaptive_balance",
                        "Feasible response: verify a nearby lower-gain/faster response trade-off.",
                    )
                )

        for kp, kd, stage, reason in proposals:
            gains = self._bounded(kp, kd)
            if gains is None:
                continue
            bounded_kp, bounded_kd = gains
            key = (round(bounded_kp, 12), round(bounded_kd, 12))
            if key in existing:
                continue
            return [
                self._spec(
                    len(already_tested) + 1,
                    bounded_kp,
                    bounded_kd,
                    stage,
                    reason,
                    estimate,
                )
            ]
        return []
