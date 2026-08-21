"""Canonical residual policy-action processing, independent of any backend."""

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
    """Keep network output, ActionManager input, and final position target distinct."""

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
    """Decode actor output according to the canonical training contract.

    ``RslRlVecEnvWrapper`` first clips the actor output to ``[-1, 1]``.  The
    common action adapter then computes
    ``actual q [rad] + ACTION_SCALE_RAD * action`` elementwise and clamps the
    target to the distinct command limits.  The scale is per joint, not a single
    scalar (Joint3 ``0.2``, Joint4 ``0.15``, the rest ``0.1``), matching Isaac's
    ``HAND_REAL_ACTION_SCALE``.  Normalized q is never accepted or used by this
    function.
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
