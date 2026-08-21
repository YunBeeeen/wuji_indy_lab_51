"""Canonical 105D observation builder with policy-step history.

``obs[40:55]`` is solved here from measured joint angles rather than taken from
the backend.  Fingertip position is a policy-input contract, not a backend
measurement: the policy was trained against Isaac's tip frames, so every
backend must be shown those frames even when it simulates a different model.
Asking the backend produced MuJoCo's own vendor-description tips, which put the
thumb 3.0 mm and the index 0.7 mm away from what the policy learned.

A rigid body makes this free: ``site position == FK(q)`` to 6.7e-08 m within one
model (``run_policy --validate-fk``), so nothing is lost by solving it.  The
real hand has no Cartesian sensor and could only ever have done it this way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..common.fingertip_fk import POLICY_TIP_FRAME_URDF, WujiHand1FingertipFK
from ..common.perception import PoseState, StickPosePair7D, StickPoseProvider, SyntheticStickPoseProvider
from ..common.policy_contract import (
    ACTION_DIM,
    OBSERVATION_DIM,
    OBSERVATION_SLICES,
    mode_one_hot,
    normalize_joint_positions,
)


_POLICY_FINGERTIP_FK: WujiHand1FingertipFK | None = None


def policy_fingertip_fk() -> WujiHand1FingertipFK:
    """Return the shared trained-contract FK, parsing the URDF once."""

    global _POLICY_FINGERTIP_FK
    if _POLICY_FINGERTIP_FK is None:
        _POLICY_FINGERTIP_FK = WujiHand1FingertipFK(POLICY_TIP_FRAME_URDF)
    return _POLICY_FINGERTIP_FK


@dataclass
class PolicyObservationAdapter:
    """Maintain oldest-to-newest q/stick history independent of a backend."""

    mode: str = "open"
    stick_provider: StickPoseProvider = field(default_factory=SyntheticStickPoseProvider)
    fingertip_fk: WujiHand1FingertipFK = field(default_factory=policy_fingertip_fk)

    def __post_init__(self) -> None:
        self.set_mode(self.mode)
        self._initialized = False
        self._q_previous = np.zeros(ACTION_DIM, dtype=np.float32)
        self._q_current = np.zeros(ACTION_DIM, dtype=np.float32)
        self._fingertips = np.zeros(15, dtype=np.float32)
        self._stick1_previous = np.zeros(7, dtype=np.float32)
        self._stick1_current = np.zeros(7, dtype=np.float32)
        self._stick2_previous = np.zeros(7, dtype=np.float32)
        self._stick2_current = np.zeros(7, dtype=np.float32)
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._last_stick_sample: StickPosePair7D | None = None

    def set_mode(self, mode: str) -> None:
        mode_one_hot(mode)  # validate before mutating state
        self.mode = mode.strip().lower()

    def reset(self, q_current: npt.ArrayLike) -> None:
        q = self._validate_vector(q_current, ACTION_DIM, "q_current")
        tips = self._fingertips_from_joints(q)
        self.stick_provider.reset()
        sticks = self.stick_provider.sample()
        self._last_stick_sample = sticks
        self._q_previous = q.copy()
        self._q_current = q.copy()
        self._fingertips = tips.copy()
        self._stick1_previous = sticks.stick1.copy()
        self._stick1_current = sticks.stick1.copy()
        self._stick2_previous = sticks.stick2.copy()
        self._stick2_current = sticks.stick2.copy()
        self._last_action.fill(0.0)
        self._initialized = True

    def advance(
        self,
        q_current: npt.ArrayLike,
        last_action: npt.ArrayLike,
    ) -> None:
        if not self._initialized:
            raise RuntimeError("Call reset() before advance().")
        q = self._validate_vector(q_current, ACTION_DIM, "q_current")
        tips = self._fingertips_from_joints(q)
        action = self._validate_vector(last_action, ACTION_DIM, "last_action")
        sticks = self.stick_provider.sample()
        self._last_stick_sample = sticks
        self._q_previous = self._q_current.copy()
        self._q_current = q.copy()
        self._fingertips = tips.copy()
        self._stick1_previous = self._stick1_current.copy()
        self._stick1_current = sticks.stick1.copy()
        self._stick2_previous = self._stick2_current.copy()
        self._stick2_current = sticks.stick2.copy()
        self._last_action = action.copy()

    def build(self) -> npt.NDArray[np.float32]:
        if not self._initialized:
            raise RuntimeError("Call reset() before build().")
        observation = np.empty(OBSERVATION_DIM, dtype=np.float32)
        values = {
            "joint_previous": normalize_joint_positions(self._q_previous),
            "joint_current": normalize_joint_positions(self._q_current),
            "fingertips": self._fingertips,
            "stick1_previous": self._stick1_previous,
            "stick1_current": self._stick1_current,
            "stick2_previous": self._stick2_previous,
            "stick2_current": self._stick2_current,
            "last_action": self._last_action,
            "mode": mode_one_hot(self.mode),
        }
        for name, value in values.items():
            observation[OBSERVATION_SLICES[name].slice] = value
        if observation.dtype != np.float32 or not np.isfinite(observation).all():
            raise AssertionError("Policy observation must be finite float32.")
        return np.ascontiguousarray(observation)

    def debug_slices(self) -> dict[str, npt.NDArray[np.float32]]:
        observation = self.build()
        return {name: observation[term.slice].copy() for name, term in OBSERVATION_SLICES.items()}

    @property
    def perception_state(self) -> PoseState:
        if self._last_stick_sample is None:
            raise RuntimeError("No stick sample is available before reset().")
        return self._last_stick_sample.state

    @property
    def perception_timestamp_s(self) -> float:
        if self._last_stick_sample is None:
            raise RuntimeError("No stick sample is available before reset().")
        return self._last_stick_sample.timestamp_s

    def _fingertips_from_joints(self, q: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Solve obs[40:55] in the trained tip frames from measured q."""

        return self._validate_vector(
            self.fingertip_fk.fingertip_positions_in_palm(q), 15, "fingertips_in_palm"
        )

    @staticmethod
    def _validate_vector(value: npt.ArrayLike, size: int, label: str) -> npt.NDArray[np.float32]:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (size,):
            raise ValueError(f"{label} must have shape {(size,)}, got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"{label} must be finite.")
        return array
