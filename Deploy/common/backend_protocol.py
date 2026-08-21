"""The boundary a backend must satisfy.  An interface, not a backend.

Lives beside the other contracts rather than in ``backends/`` because
``policy/`` types the runner against it, and ``policy/`` must stay importable
where no simulator and no hand SDK exist.  The implementations -- MuJoCo, the
physical hand -- are in ``backends/`` and import this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy.typing as npt
import numpy as np


@dataclass(frozen=True)
class BackendHealth:
    ok: bool
    message: str
    sample_fresh: bool


@runtime_checkable
class WujiBackend(Protocol):
    """Canonical-radian robot I/O only; storage mapping stays inside backend."""

    def joint_identifiers(self) -> tuple[str, ...]: ...

    def read_joint_positions(self) -> npt.NDArray[np.float32]: ...

    def write_joint_position_targets(self, targets_policy_order: npt.ArrayLike) -> None:
        """Write an already-decoded canonical policy-order target in radians."""
        ...

    # Deliberately NOT part of the boundary.  Fingertip position is a policy
    # input contract solved from q against the trained URDF, not something a
    # backend reports -- see observation_adapter.  MuJoCo still exposes its own
    # sites for model-consistency checks and logging; the real hand has no
    # Cartesian sensor to expose at all.

    def health(self) -> BackendHealth: ...

    def safe_stop(self) -> None: ...
