"""Response-guided search orchestration shared by GUI and headless runs."""

from __future__ import annotations

import math

from .coarse_to_fine import CoarseToFineSearch
from .config import ResolvedAutoTuneConfig
from .ranking import rank_candidates
from .result import AutoTuneOutcome, CandidateResult, CandidateSpec
from .search_base import AutoTuneSearchStrategy


class AutoTuneController:
    """Run safe Kp probes, identified targets, then measured corrections."""

    def __init__(
        self,
        config: ResolvedAutoTuneConfig,
        strategy: AutoTuneSearchStrategy | None = None,
    ) -> None:
        self.config = config
        self.strategy = strategy or CoarseToFineSearch(config)
        self.pending = self.strategy.coarse_candidates()
        self.results: list[CandidateResult] = []
        self.search_stage = "kp_probe"

    @property
    def tested_count(self) -> int:
        return len(self.results)

    @property
    def planned_count(self) -> int:
        strategy_limit = getattr(self.strategy, "effective_budget", self.config.search_budget)
        return min(strategy_limit, len(self.results) + len(self.pending))

    def next_candidate(self) -> CandidateSpec | None:
        """Return the next point and expand each stage only after evidence."""

        while not self.pending:
            if self.search_stage == "kp_probe":
                self._generate_model_candidates()
            elif self.search_stage == "model_guided":
                self.search_stage = "adaptive_refine"
                self._generate_next_correction()
            elif self.search_stage == "adaptive_refine":
                self._generate_next_correction()
            else:
                return None
            if self.search_stage == "complete":
                return None
        return self.pending.pop(0)

    def add_result(self, result: CandidateResult) -> None:
        if len(self.results) >= self.config.search_budget:
            raise RuntimeError("Auto Tune search budget exceeded.")
        self.results.append(result)

    @staticmethod
    def _seed_sort_key(result: CandidateResult) -> tuple[float, ...]:
        """Prefer usable response, then response quality, without unsafe gains."""

        return (
            0.0 if result.performance_constraint_passed else 1.0,
            0.0 if result.settling_time is not None else 1.0,
            float(result.settling_time) if result.settling_time is not None else math.inf,
            float(result.steady_state_error),
            float(result.percentage_overshoot),
            float(result.total_score) if result.total_score is not None else math.inf,
            float(result.candidate.candidate_id),
        )

    def _safe_kp_results(self) -> list[CandidateResult]:
        kp_results = [
            result for result in self.results if result.candidate.stage == "kp_probe"
        ]
        if not kp_results:
            return []
        # Classification/normalization is reused, but only hard-passed Kp
        # trials can seed the damping stage.
        rank_candidates(kp_results, self.config)
        safe = [result for result in kp_results if result.hard_constraint_passed]
        safe.sort(key=self._seed_sort_key)
        return safe

    def _generate_model_candidates(self) -> None:
        if len(self.results) >= self.config.search_budget:
            self.search_stage = "complete"
            return
        kp_results = self._safe_kp_results()
        if not kp_results:
            # If every Kp-only response violated a hard condition, there is no
            # defensible linear response from which to identify the joint.
            self.search_stage = "complete"
            return
        self.pending = self.strategy.damping_candidates(kp_results, self.results)
        self.search_stage = "model_guided"
        if not self.pending:
            self.search_stage = "adaptive_refine"
            self._generate_next_correction()

    def _generate_next_correction(self) -> None:
        if len(self.results) >= self.config.search_budget:
            self.search_stage = "complete"
            return
        provisional = rank_candidates(self.results, self.config)
        best = provisional.selected
        if best is None:
            self.search_stage = "complete"
            return
        self.pending = self.strategy.fine_candidates(best, self.results)
        self.search_stage = "adaptive_refine" if self.pending else "complete"

    def finalize(self) -> AutoTuneOutcome:
        return rank_candidates(self.results, self.config)
