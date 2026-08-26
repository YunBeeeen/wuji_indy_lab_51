# [backend/MuJoCo] 시뮬 카메라 프레임 소스와 스틱 그라운드트루스 제공자.
"""MuJoCo camera and ground-truth stick providers."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from ..common.perception import PoseState, StickPosePair7D
from ..common.isaac_reset import STICK_REFERENCE_QUATERNIONS_PALM_WXYZ
from ..common.stick_pose import canonicalize_square_stick_quaternion


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    timestamp_s: float
    fresh: bool


class MujocoGroundTruthStickPoseProvider:
    representation = "StickPose7D"

    def __init__(self, backend) -> None:
        self.backend = backend

    def reset(self) -> None:
        pass

    def sample(self) -> StickPosePair7D:
        poses = self.backend.get_stick_poses_in_palm()
        for index in range(2):
            poses[index, 3:] = canonicalize_square_stick_quaternion(
                poses[index, 3:], STICK_REFERENCE_QUATERNIONS_PALM_WXYZ[index]
            )
        return StickPosePair7D(
            poses[0].copy(), poses[1].copy(), float(self.backend.data.time), PoseState.VALID, True
        )


class MujocoCameraSource:
    """Render the calibrated D435 RGB view at a maximum fresh rate of 15 Hz."""

    def __init__(self, backend, width: int = 1280, height: int = 720, fps: float = 15.0):
        import mujoco

        self.backend = backend
        self.width = width
        self.height = height
        self.period_s = 1.0 / fps
        self.renderer = mujoco.Renderer(backend.model, height=height, width=width)
        self._last_time = -np.inf
        self._last_rgb: np.ndarray | None = None

    def reset(self) -> None:
        self._last_time = -np.inf
        self._last_rgb = None

    def capture(self) -> CameraFrame:
        timestamp = float(self.backend.data.time)
        fresh = self._last_rgb is None or timestamp - self._last_time >= self.period_s - 1.0e-9
        if fresh:
            self.renderer.update_scene(self.backend.data, camera="d435_rgb")
            self._last_rgb = np.asarray(self.renderer.render(), dtype=np.uint8).copy()
            self._last_time = timestamp
        return CameraFrame(self._last_rgb.copy(), self._last_time, fresh)

    def close(self) -> None:
        self.renderer.close()
