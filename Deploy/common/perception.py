# [common] 스틱 포즈 제공자가 갖춰야 할 인터페이스(PoseState/StickPoseProvider) + 합성 스텁. 카메라 아님.
"""정책이 요구하는 스틱 포즈 제공자와 유효 상태의 공통 인터페이스.
실제 카메라 구현은 ``vision/``에 배치. 합성 제공자는 배선 확인에만 사용."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable
import time

import numpy as np
import numpy.typing as npt

from .stick_pose import STICK_POSE_DIM, normalize_quaternion_wxyz


class PoseState(str, Enum):
    VALID = "valid"
    HOLD = "hold"
    STALE = "stale"
    LOST = "lost"
    REINIT = "reinit"


@dataclass(frozen=True)
class StickPosePair7D:
    stick1: npt.NDArray[np.float32]
    stick2: npt.NDArray[np.float32]
    timestamp_s: float
    state: PoseState = PoseState.VALID
    fresh: bool = True


@runtime_checkable
class StickPoseProvider(Protocol):
    representation: str

    def reset(self) -> None: ...
    def sample(self) -> StickPosePair7D: ...


class SyntheticStickPoseProvider:
    representation = "StickPose7D"

    def __init__(self, stick1: npt.ArrayLike | None = None, stick2: npt.ArrayLike | None = None):
        self._stick1 = _pose7d(
            [0.0250743479, 0.0242451150, 0.0969612077,
             0.4618085623, -0.0092124203, -0.1713383496, -0.8702247143]
            if stick1 is None else stick1,
            "stick1",
        )
        self._stick2 = _pose7d(
            [0.0355986878, 0.0160842165, 0.0733669698,
             0.2051235586, -0.6018196344, -0.4935579300, -0.5934122205]
            if stick2 is None else stick2,
            "stick2",
        )

    def reset(self) -> None:
        pass

    def sample(self) -> StickPosePair7D:
        return StickPosePair7D(self._stick1.copy(), self._stick2.copy(), time.monotonic())


class VisionStickPoseProvider:
    representation = "PENDING_VISION_RUNTIME"

    def reset(self) -> None:
        raise NotImplementedError("Use the configured ArUco provider implementation.")

    def sample(self) -> StickPosePair7D:
        raise NotImplementedError("Use the configured ArUco provider implementation.")


def _pose7d(value: npt.ArrayLike, label: str) -> npt.NDArray[np.float32]:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (STICK_POSE_DIM,) or not np.isfinite(pose).all():
        raise ValueError(f"{label} must be a finite StickPose7D vector.")
    pose[3:] = normalize_quaternion_wxyz(pose[3:])
    return pose.astype(np.float32)
