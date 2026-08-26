# [policy/legacy] 2026-08-13 hand_final 전용 101D 관측·0.1 rad 액션 계약.
"""Frozen adapter for the pre-105D ``hand_final`` actor.

This module exists so the 2026-08-13 policy can be deployed without weakening
or conditionally mutating the active 105D contract.  Its constants come from
the saved run
``hand_final/2026-08-13_14-15-09(최종)/params/env.yaml`` and the local URDF
that produced that run's Isaac USD.

The differences are a single inseparable contract:

* 101D observation: each stick is ``palm xyz + directed local +Y axis``;
* joint normalization uses the old local-URDF placeholder limits;
* every residual action has a uniform 0.1 rad scale;

Output safety is deliberately NOT frozen to the old simulator limits.  Every
policy deployed on this physical hand uses the one hardware command envelope:
the connected hand's factory limits multiplied by ``COMMAND_LIMIT_RATIO``.

Do not import these values into ``policy_contract.py``.  That file is the
active 105D contract; keeping this compatibility path here is what prevents an
old checkpoint from silently changing new policies, or vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from ..common.fingertip_fk import ISAAC_URDF, WujiHand1FingertipFK
from ..common.perception import (
    PoseState,
    StickPosePair7D,
    StickPoseProvider,
    SyntheticStickPoseProvider,
)
from ..common.policy_contract import (
    ACTION_CLIP,
    ACTION_DIM,
    COMMAND_TARGET_LIMITS,
    ObservationSlice,
)
from ..common.stick_pose import quaternion_to_rotation_matrix_wxyz
from ..common.timing import StageTimer
from .action_adapter import DecodedAction


LEGACY_HAND_FINAL_RUN = "hand_final/2026-08-13_14-15-09(최종)"
LEGACY_OBSERVATION_DIM = 101
LEGACY_ACTION_DIM = 20
LEGACY_STICK_REPRESENTATION = "palm xyz + directed local +Y axis"

# Exact soft-limit normalization table of the local URDF imported by the old
# Isaac USD.  The repeated 1.6272 rad upper is the historical 93.2317-degree
# placeholder, not the later vendor table and not the connected hand's factory
# table.  joint_pos_limit_normalized() used this table with no clipping.
LEGACY_OBSERVATION_NORMALIZATION_LIMITS = np.asarray(
    [
        (-0.04480, 1.65080), (-0.16590, 0.93390),
        (-0.49320, 1.62720), (-0.49320, 1.62720),
        (-0.32695, 1.63595), (-0.49500, 0.49500),
        (-0.49320, 1.62720), (-0.49320, 1.62720),
        (-0.32695, 1.63595), (-0.49500, 0.49500),
        (-0.49320, 1.62720), (-0.49320, 1.62720),
        (-0.32695, 1.63595), (-0.49500, 0.49500),
        (-0.49320, 1.62720), (-0.49320, 1.62720),
        (-0.32695, 1.63595), (-0.49500, 0.49500),
        (-0.49320, 1.62720), (-0.49320, 1.62720),
    ],
    dtype=np.float32,
)
LEGACY_OBSERVATION_NORMALIZATION_LIMITS.setflags(write=False)

LEGACY_ACTION_SCALE_RAD = np.full(LEGACY_ACTION_DIM, 0.1, dtype=np.float32)
LEGACY_ACTION_SCALE_RAD.setflags(write=False)

# Observation normalization remains checkpoint-specific, but target safety is
# hardware-specific.  Do not intersect this with the old URDF range or restore
# the historical Joint4 floor: doing either makes this actor the sole policy
# with a command envelope narrower than REAL_HAND_FACTORY_LIMITS * 0.95 and can
# turn a legal 0.1-rad residual into a larger corrective jump when measured q
# lies outside that extra boundary.
LEGACY_DEPLOY_TARGET_LIMITS = COMMAND_TARGET_LIMITS.copy()
LEGACY_DEPLOY_TARGET_LIMITS.setflags(write=False)

# Exact reset recorded in the saved 2026-08-13 env.yaml.  This is intentionally
# not today's ISAAC_PREGRASP_JOINT_POSITIONS_RAD: a checkpoint owns its reset.
LEGACY_PREGRASP_JOINT_POSITIONS_RAD = np.asarray(
    [
        0.5377866626, 0.8436813951, 0.0377136655, -0.0000001810,
        0.7017297745, 0.0553143807, 1.1822255850, 1.4215219021,
        0.4649881423, -0.0292181600, 1.6298730373, 1.1032750607,
        0.9151425958, -0.0129909236, 1.3248542547, 0.3182539344,
        0.7154092789, 0.0788998753, 1.6281884909, 0.2546040118,
    ],
    dtype=np.float32,
)
LEGACY_PREGRASP_JOINT_POSITIONS_RAD.setflags(write=False)

LEGACY_OBSERVATION_SLICES: dict[str, ObservationSlice] = {
    "joint_previous": ObservationSlice(0, 20, "previous legacy-limit-normalized q"),
    "joint_current": ObservationSlice(20, 40, "current legacy-limit-normalized q"),
    "fingertips": ObservationSlice(40, 55, "current trained tip-frame xyz in palm"),
    "stick1_previous": ObservationSlice(55, 61, "previous Stick1 xyz + directed +Y"),
    "stick1_current": ObservationSlice(61, 67, "current Stick1 xyz + directed +Y"),
    "stick2_previous": ObservationSlice(67, 73, "previous Stick2 xyz + directed +Y"),
    "stick2_current": ObservationSlice(73, 79, "current Stick2 xyz + directed +Y"),
    "last_action": ObservationSlice(79, 99, "last clipped ActionManager action"),
    "mode": ObservationSlice(99, 101, "OPEN/CLOSE one-hot command"),
}


def legacy_observation_csv_columns() -> list[str]:
    columns = ["" for _ in range(LEGACY_OBSERVATION_DIM)]
    for name, term in LEGACY_OBSERVATION_SLICES.items():
        for offset, index in enumerate(range(term.start, term.stop)):
            columns[index] = f"obs_{name}_{offset:02d}"
    if any(not column for column in columns):
        raise RuntimeError("Legacy observation slices leave an unnamed CSV column.")
    return columns


def normalize_legacy_joint_positions(
    joint_positions: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    q = _finite_vector(joint_positions, LEGACY_ACTION_DIM, "canonical joint positions")
    lower = LEGACY_OBSERVATION_NORMALIZATION_LIMITS[:, 0]
    upper = LEGACY_OBSERVATION_NORMALIZATION_LIMITS[:, 1]
    center = (lower + upper) * np.float32(0.5)
    return (np.float32(2.0) * (q - center) / (upper - lower)).astype(np.float32)


def legacy_mode_one_hot(mode: str) -> npt.NDArray[np.float32]:
    normalized = mode.strip().lower()
    if normalized == "open":
        return np.asarray([1.0, 0.0], dtype=np.float32)
    if normalized == "close":
        return np.asarray([0.0, 1.0], dtype=np.float32)
    raise ValueError(
        f"Legacy hand_final mode {mode!r} is invalid; the 2026-08-13 actor "
        "was trained only with 'open' and 'close', not neutral [0,0]."
    )


def pose7d_to_legacy_directed_axis(pose: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Convert palm ``xyz+wxyz`` to the old ``xyz+R(q)@local_y`` feature."""

    value = _finite_vector(pose, 7, "StickPose7D")
    directed_local_y = quaternion_to_rotation_matrix_wxyz(value[3:7])[:, 1]
    return np.concatenate((value[:3], directed_local_y)).astype(np.float32)


def decode_legacy_hand_final_action(
    q_current_policy_order: npt.ArrayLike,
    raw_policy_action: npt.ArrayLike,
) -> DecodedAction:
    """Reproduce the old uniform-0.1 residual action under hardware safety."""

    q_current = _finite_vector(
        q_current_policy_order, LEGACY_ACTION_DIM, "canonical joint positions"
    )
    onnx_action = _finite_vector(raw_policy_action, LEGACY_ACTION_DIM, "ONNX action")
    clipped_action = np.clip(onnx_action, -ACTION_CLIP, ACTION_CLIP).astype(np.float32)
    unclamped_target = (
        q_current + LEGACY_ACTION_SCALE_RAD * clipped_action
    ).astype(np.float32)
    position_target = np.clip(
        unclamped_target,
        LEGACY_DEPLOY_TARGET_LIMITS[:, 0],
        LEGACY_DEPLOY_TARGET_LIMITS[:, 1],
    ).astype(np.float32)
    return DecodedAction(
        onnx_action=onnx_action.copy(),
        action_manager_action=clipped_action,
        unclamped_target=unclamped_target,
        position_target=position_target,
        action_was_clipped=np.not_equal(onnx_action, clipped_action),
        target_was_clamped=np.not_equal(unclamped_target, position_target),
    )


_LEGACY_FINGERTIP_FK: WujiHand1FingertipFK | None = None


def legacy_policy_fingertip_fk() -> WujiHand1FingertipFK:
    global _LEGACY_FINGERTIP_FK
    if _LEGACY_FINGERTIP_FK is None:
        # Name the historical Isaac source directly instead of following the
        # active POLICY_TIP_FRAME_URDF alias if that alias changes later.
        _LEGACY_FINGERTIP_FK = WujiHand1FingertipFK(ISAAC_URDF)
    return _LEGACY_FINGERTIP_FK


@dataclass
class LegacyHandFinal101ObservationAdapter:
    """Build exactly the 101 floats consumed by the 2026-08-13 actor."""

    observation_dim: ClassVar[int] = LEGACY_OBSERVATION_DIM
    observation_slices: ClassVar[dict[str, ObservationSlice]] = LEGACY_OBSERVATION_SLICES

    mode: str = "open"
    stick_provider: StickPoseProvider = field(default_factory=SyntheticStickPoseProvider)
    fingertip_fk: WujiHand1FingertipFK = field(default_factory=legacy_policy_fingertip_fk)
    timing: StageTimer = field(default_factory=lambda: StageTimer(name="obs"))

    def __post_init__(self) -> None:
        self.set_mode(self.mode)
        self._initialized = False
        self._q_previous = np.zeros(LEGACY_ACTION_DIM, dtype=np.float32)
        self._q_current = np.zeros(LEGACY_ACTION_DIM, dtype=np.float32)
        self._fingertips = np.zeros(15, dtype=np.float32)
        self._stick1_previous = np.zeros(6, dtype=np.float32)
        self._stick1_current = np.zeros(6, dtype=np.float32)
        self._stick2_previous = np.zeros(6, dtype=np.float32)
        self._stick2_current = np.zeros(6, dtype=np.float32)
        self._last_action = np.zeros(LEGACY_ACTION_DIM, dtype=np.float32)
        self._last_stick_sample: StickPosePair7D | None = None

    def set_mode(self, mode: str) -> None:
        legacy_mode_one_hot(mode)
        self.mode = mode.strip().lower()

    def reset(self, q_current: npt.ArrayLike) -> None:
        q = _finite_vector(q_current, LEGACY_ACTION_DIM, "q_current")
        tips = self._fingertips_from_joints(q)
        self.stick_provider.reset()
        with self.timing.stage("stick_sample"):
            sticks = self.stick_provider.sample()
        stick1, stick2 = self._legacy_sticks(sticks)
        self._last_stick_sample = sticks
        self._q_previous = q.copy()
        self._q_current = q.copy()
        self._fingertips = tips.copy()
        self._stick1_previous = stick1.copy()
        self._stick1_current = stick1.copy()
        self._stick2_previous = stick2.copy()
        self._stick2_current = stick2.copy()
        self._last_action.fill(0.0)
        self._initialized = True

    def advance(self, q_current: npt.ArrayLike, last_action: npt.ArrayLike) -> None:
        if not self._initialized:
            raise RuntimeError("Call reset() before advance().")
        q = _finite_vector(q_current, LEGACY_ACTION_DIM, "q_current")
        action = _finite_vector(last_action, LEGACY_ACTION_DIM, "last_action")
        tips = self._fingertips_from_joints(q)
        with self.timing.stage("stick_sample"):
            sticks = self.stick_provider.sample()
        stick1, stick2 = self._legacy_sticks(sticks)
        self._last_stick_sample = sticks
        self._q_previous = self._q_current.copy()
        self._q_current = q.copy()
        self._fingertips = tips.copy()
        self._stick1_previous = self._stick1_current.copy()
        self._stick1_current = stick1.copy()
        self._stick2_previous = self._stick2_current.copy()
        self._stick2_current = stick2.copy()
        self._last_action = action.copy()

    def build(self) -> npt.NDArray[np.float32]:
        if not self._initialized:
            raise RuntimeError("Call reset() before build().")
        observation = np.empty(LEGACY_OBSERVATION_DIM, dtype=np.float32)
        values = {
            "joint_previous": normalize_legacy_joint_positions(self._q_previous),
            "joint_current": normalize_legacy_joint_positions(self._q_current),
            "fingertips": self._fingertips,
            "stick1_previous": self._stick1_previous,
            "stick1_current": self._stick1_current,
            "stick2_previous": self._stick2_previous,
            "stick2_current": self._stick2_current,
            "last_action": self._last_action,
            "mode": legacy_mode_one_hot(self.mode),
        }
        for name, value in values.items():
            observation[LEGACY_OBSERVATION_SLICES[name].slice] = value
        if observation.dtype != np.float32 or not np.isfinite(observation).all():
            raise AssertionError("Legacy policy observation must be finite float32.")
        return np.ascontiguousarray(observation)

    def debug_slices(self) -> dict[str, npt.NDArray[np.float32]]:
        observation = self.build()
        return {
            name: observation[term.slice].copy()
            for name, term in LEGACY_OBSERVATION_SLICES.items()
        }

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

    def _fingertips_from_joints(
        self, q: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        with self.timing.stage("fingertip_fk"):
            tips = self.fingertip_fk.fingertip_positions_in_palm(q)
        return _finite_vector(tips, 15, "fingertips_in_palm")

    @staticmethod
    def _legacy_sticks(
        sticks: StickPosePair7D,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return (
            pose7d_to_legacy_directed_axis(sticks.stick1),
            pose7d_to_legacy_directed_axis(sticks.stick2),
        )


def legacy_contract_summary() -> str:
    return "\n".join(
        [
            "legacy hand_final 2026-08-13 (isolated compatibility adapter)",
            f"observation: {LEGACY_OBSERVATION_DIM}D, {LEGACY_STICK_REPRESENTATION}",
            "joint normalization: old local-URDF placeholder limits",
            "action: clip +/-1, q_current + 0.1 rad * action",
            "hardware target clamp: connected-hand factory limits * 0.95",
        ]
    )


def _finite_vector(
    value: npt.ArrayLike, size: int, label: str
) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{label} must have shape {(size,)}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite.")
    return array


if max(term.stop for term in LEGACY_OBSERVATION_SLICES.values()) != LEGACY_OBSERVATION_DIM:
    raise RuntimeError("Legacy observation layout is not 101D.")
