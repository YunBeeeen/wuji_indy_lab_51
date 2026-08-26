# [policy] ONNX 입력 폭으로 active 105D와 격리된 legacy 101D 계약을 선택.
"""Composition root for supported chopstick-grasp actor interfaces.

The adapters themselves stay independent.  This small selector is the only
place that knows both exist, allowing the existing CLIs to auto-detect a graph
without adding a ``--legacy`` switch that can be set incorrectly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import numpy.typing as npt

from ..common.perception import StickPoseProvider
from ..common.policy_contract import (
    ACTION_DIM,
    ACTION_SCALE_RAD,
    OBSERVATION_DIM,
    OBSERVATION_SLICES,
    ObservationSlice,
    contract_summary,
    observation_csv_columns,
)
from .action_adapter import DecodedAction, decode_policy_action
from .legacy_hand_final_101 import (
    LEGACY_ACTION_DIM,
    LEGACY_ACTION_SCALE_RAD,
    LEGACY_OBSERVATION_DIM,
    LEGACY_OBSERVATION_SLICES,
    LEGACY_PREGRASP_JOINT_POSITIONS_RAD,
    LegacyHandFinal101ObservationAdapter,
    decode_legacy_hand_final_action,
    legacy_contract_summary,
    legacy_observation_csv_columns,
)
from .observation_adapter import PolicyObservationAdapter
from .onnx_policy import OnnxPolicy


ActionDecoder = Callable[[npt.ArrayLike, npt.ArrayLike], DecodedAction]


@dataclass(frozen=True)
class GraspPolicyContract:
    """All policy-facing choices that must move together for one actor."""

    key: str
    observation_dim: int
    action_dim: int
    observation_slices: Mapping[str, ObservationSlice]
    action_scale_rad: npt.NDArray[np.float32]
    adapter_type: type
    action_decoder: ActionDecoder
    csv_columns_factory: Callable[[], list[str]]
    summary_factory: Callable[[], str]
    supported_modes: tuple[str, ...]
    default_pregrasp: npt.NDArray[np.float32] | None = None

    def make_observation_adapter(
        self, *, mode: str, stick_provider: StickPoseProvider
    ):
        return self.adapter_type(mode=mode, stick_provider=stick_provider)

    def observation_csv_columns(self) -> list[str]:
        return self.csv_columns_factory()

    def summary(self) -> str:
        return self.summary_factory()


CURRENT_105D_CONTRACT = GraspPolicyContract(
    key="hand_real_105d",
    observation_dim=OBSERVATION_DIM,
    action_dim=ACTION_DIM,
    observation_slices=OBSERVATION_SLICES,
    action_scale_rad=ACTION_SCALE_RAD,
    adapter_type=PolicyObservationAdapter,
    action_decoder=decode_policy_action,
    csv_columns_factory=observation_csv_columns,
    summary_factory=contract_summary,
    supported_modes=("open", "close", "neutral"),
)

LEGACY_HAND_FINAL_101D_CONTRACT = GraspPolicyContract(
    key="hand_final_2026-08-13_101d",
    observation_dim=LEGACY_OBSERVATION_DIM,
    action_dim=LEGACY_ACTION_DIM,
    observation_slices=LEGACY_OBSERVATION_SLICES,
    action_scale_rad=LEGACY_ACTION_SCALE_RAD,
    adapter_type=LegacyHandFinal101ObservationAdapter,
    action_decoder=decode_legacy_hand_final_action,
    csv_columns_factory=legacy_observation_csv_columns,
    summary_factory=legacy_contract_summary,
    supported_modes=("open", "close"),
    default_pregrasp=LEGACY_PREGRASP_JOINT_POSITIONS_RAD,
)

_CONTRACTS_BY_OBSERVATION_DIM = {
    CURRENT_105D_CONTRACT.observation_dim: CURRENT_105D_CONTRACT,
    LEGACY_HAND_FINAL_101D_CONTRACT.observation_dim: LEGACY_HAND_FINAL_101D_CONTRACT,
}


def grasp_contract_for_dimensions(
    observation_dim: int, action_dim: int
) -> GraspPolicyContract:
    contract = _CONTRACTS_BY_OBSERVATION_DIM.get(int(observation_dim))
    if contract is None or int(action_dim) != contract.action_dim:
        supported = ", ".join(
            f"{item.observation_dim}D->{item.action_dim}D"
            for item in _CONTRACTS_BY_OBSERVATION_DIM.values()
        )
        raise RuntimeError(
            f"Unsupported grasp actor interface {observation_dim}D->{action_dim}D; "
            f"supported: {supported}."
        )
    return contract


def inspect_fixed_onnx_dimensions(path: str | Path) -> tuple[int, int]:
    """Read one fixed-batch ONNX interface before choosing its adapter."""

    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("onnxruntime is required to inspect the policy.") from exc

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"ONNX policy does not exist: {resolved}")
    session = ort.InferenceSession(str(resolved), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            f"Expected one ONNX input/output, got {len(inputs)}/{len(outputs)}."
        )
    input_meta, output_meta = inputs[0], outputs[0]
    if input_meta.type != "tensor(float)" or output_meta.type != "tensor(float)":
        raise RuntimeError(
            f"Expected float ONNX tensors, got {input_meta.type}/{output_meta.type}."
        )
    if (
        not isinstance(input_meta.shape, list)
        or not isinstance(output_meta.shape, list)
        or len(input_meta.shape) != 2
        or len(output_meta.shape) != 2
        or input_meta.shape[0] != 1
        or output_meta.shape[0] != 1
        or not isinstance(input_meta.shape[1], int)
        or not isinstance(output_meta.shape[1], int)
    ):
        raise RuntimeError(
            "Grasp actor must have fixed shapes [1,obs] -> [1,action], got "
            f"{input_meta.shape} -> {output_meta.shape}."
        )
    return int(input_meta.shape[1]), int(output_meta.shape[1])


def load_grasp_policy(path: str | Path) -> tuple[OnnxPolicy, GraspPolicyContract]:
    observation_dim, action_dim = inspect_fixed_onnx_dimensions(path)
    contract = grasp_contract_for_dimensions(observation_dim, action_dim)
    policy = OnnxPolicy(path, contract.observation_dim, contract.action_dim)
    return policy, contract
