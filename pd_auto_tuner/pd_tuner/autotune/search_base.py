"""Extensible deterministic Auto Tune search strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .config import ResolvedAutoTuneConfig
from .result import CandidateResult, CandidateSpec


class AutoTuneSearchStrategy(ABC):
    """Candidate-generation interface for future Bayesian/CMA-ES strategies."""

    def __init__(self, config: ResolvedAutoTuneConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable serializer name."""

    @abstractmethod
    def coarse_candidates(self) -> list[CandidateSpec]:
        """Return deterministic Kp-screen candidates."""

    @abstractmethod
    def damping_candidates(
        self,
        kp_results: list[CandidateResult],
        already_tested: list[CandidateResult],
    ) -> list[CandidateSpec]:
        """Infer a local model and return calculated Kp/Kd candidates."""

    @abstractmethod
    def fine_candidates(
        self,
        best: CandidateResult,
        already_tested: list[CandidateResult],
    ) -> list[CandidateSpec]:
        """Return the next measured-response correction candidate."""
