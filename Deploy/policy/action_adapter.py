# [policy] 정책 출력 -> 관절 목표. clip(±1) -> q + 관절별 스케일 -> 명령 한계 clamp.
"""정책 출력을 백엔드 공통 관절 잔차 목표로 변환."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..common.policy_contract import (
    ACTION_CLIP,
    ACTION_DIM,
    ACTION_SCALE_RAD,
    COMMAND_TARGET_LIMITS,
)


@dataclass(frozen=True)
class DecodedAction:
    """네트워크 출력, 적용 액션, 최종 관절 목표 구분 보관."""

    onnx_action: npt.NDArray[np.float32]
    action_manager_action: npt.NDArray[np.float32]
    unclamped_target: npt.NDArray[np.float32]
    position_target: npt.NDArray[np.float32]
    action_was_clipped: npt.NDArray[np.bool_]
    target_was_clamped: npt.NDArray[np.bool_]


def decode_policy_action(
    q_current_policy_order: npt.ArrayLike,
    raw_policy_action: npt.ArrayLike,
) -> DecodedAction:
    """학습 계약에 따라 액션 clip, 관절별 스케일, 명령 한계 적용.

    입력 ``q_current``: 정규화 값이 아닌 실제 관절각(rad).
    """

    q_current = np.asarray(q_current_policy_order, dtype=np.float32)
    onnx_action = np.asarray(raw_policy_action, dtype=np.float32)
    if q_current.shape != (ACTION_DIM,):
        raise ValueError(f"Expected q shape {(ACTION_DIM,)}, got {q_current.shape}.")
    if onnx_action.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape {(ACTION_DIM,)}, got {onnx_action.shape}.")
    if not np.isfinite(q_current).all() or not np.isfinite(onnx_action).all():
        raise ValueError("Joint state and ONNX action must both be finite.")

    clipped_action = np.clip(onnx_action, -ACTION_CLIP, ACTION_CLIP).astype(np.float32)
    unclamped_target = (q_current + ACTION_SCALE_RAD * clipped_action).astype(np.float32)
    position_target = np.clip(
        unclamped_target,
        COMMAND_TARGET_LIMITS[:, 0],
        COMMAND_TARGET_LIMITS[:, 1],
    ).astype(np.float32)
    return DecodedAction(
        onnx_action=onnx_action.copy(),
        action_manager_action=clipped_action,
        unclamped_target=unclamped_target,
        position_target=position_target,
        action_was_clipped=np.not_equal(onnx_action, clipped_action),
        target_was_clamped=np.not_equal(unclamped_target, position_target),
    )
