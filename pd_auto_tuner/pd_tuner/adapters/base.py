"""Stable adapter interface used by the simulator and GUI layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GainState:
    """Runtime PD and effort-limit state for one articulation joint."""

    stiffness: float
    damping: float
    effort_limit: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AppliedGainState:
    """Requested and simulator-confirmed values after one update."""

    joint_name: str
    requested: GainState
    applied: GainState
    actuator_group: str
    scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_name": self.joint_name,
            "requested": self.requested.to_dict(),
            "applied": self.applied.to_dict(),
            "actuator_group": self.actuator_group,
            "scope": self.scope,
        }


@dataclass(frozen=True, slots=True)
class EffortSignals:
    """Effort signals actually exposed by the selected Isaac Lab version."""

    computed_effort: float | None
    applied_effort: float | None
    measured_joint_effort: float | None = None


@dataclass(frozen=True, slots=True)
class JointInfo:
    """Resolved metadata for one spawned articulation joint."""

    name: str
    index: int
    actuator_group: str | None
    actuator_type: str | None
    tunable: bool
    lower_limit: float
    upper_limit: float
    velocity_limit: float
    original_gain: GainState

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["original_gain"] = self.original_gain.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class ActuatorGroupInfo:
    """Resolved actuator group and its matching actual articulation joints."""

    name: str
    actuator_type: str
    joint_names_expr: tuple[str, ...]
    joint_names: tuple[str, ...]
    joint_indices: tuple[int, ...]
    tunable: bool
    stiffness_source_type: str
    damping_source_type: str
    effort_limit_source_type: str
    runtime_update_scope: str
    limitation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActuatorTuningAdapter(ABC):
    """Public interface that isolates Isaac Lab runtime API differences."""

    @property
    @abstractmethod
    def joint_infos(self) -> tuple[JointInfo, ...]:
        """Return all spawned articulation joints and their actuator mapping."""

    @property
    @abstractmethod
    def actuator_groups(self) -> tuple[ActuatorGroupInfo, ...]:
        """Return resolved actuator group metadata."""

    @property
    @abstractmethod
    def warnings(self) -> tuple[str, ...]:
        """Return all mapping and compatibility warnings."""

    @abstractmethod
    def list_tunable_joints(self) -> list[str]:
        """List joints supporting position control and runtime PD updates."""

    @abstractmethod
    def read_gains(self, joint_index: int) -> GainState:
        """Read simulator-confirmed gains for one global joint index."""

    @abstractmethod
    def apply_gains(
        self,
        joint_index: int,
        stiffness: float,
        damping: float,
        effort_limit: float,
        *,
        apply_to_group: bool = False,
    ) -> list[AppliedGainState]:
        """Apply and read back gains at a physics-step boundary."""

    @abstractmethod
    def restore_original_gains(self, joint_index: int, *, apply_to_group: bool = False) -> list[AppliedGainState]:
        """Restore the gain snapshot captured immediately after spawn."""

    @abstractmethod
    def read_effort_signals(self, joint_index: int) -> EffortSignals:
        """Return available effort signals without inventing measured data."""
