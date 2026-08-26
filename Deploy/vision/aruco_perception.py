# [vision] OpenCV ArUco/IPPE로 스틱 7D 포즈를 내는 제공자. 점프 게이트와 HOLD/STALE/LOST 상태기계 포함.
"""OpenCV ArUco/IPPE StickPose7D provider shared by simulated and real RGB sources."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..common.perception import PoseState, StickPosePair7D
from ..common.isaac_reset import STICK_REFERENCE_QUATERNIONS_PALM_WXYZ
from .sim_aruco import (MARKER_IDS_BY_STICK, STICK_PRIMARY_MARKER_SIZE_M, T_MARKER_STICK_BY_ID, T_PALM_CAMERA)
from ..common.stick_pose import (
    canonicalize_square_stick_quaternion,
    pose_matrix_to_xyz_wxyz,
    quaternion_geodesic_error_deg,
    quaternion_to_rotation_matrix_wxyz,
)


DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets/d435_rgb_814412070582_1280x720_15hz.json"
)


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    matrix: np.ndarray
    distortion: np.ndarray

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CALIBRATION_PATH):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        matrix = np.asarray(
            [[values["fx"], 0.0, values["cx"]],
             [0.0, values["fy"], values["cy"]],
             [0.0, 0.0, 1.0]], dtype=np.float64,
        )
        return cls(values["width"], values["height"], matrix,
                   np.asarray(values["distortion_coefficients"], dtype=np.float64))


@dataclass
class _Track:
    raw_pose: np.ndarray | None = None
    filtered_pose: np.ndarray | None = None
    missed: int = 0
    lost: bool = True


class ArucoStickPoseProvider:
    representation = "StickPose7D"

    def __init__(
        self,
        camera_source,
        calibration: CameraCalibration | None = None,
        *,
        max_reprojection_error_px: float = 1.3,
        max_position_jump_m: float = 0.070,
        max_rotation_jump_deg: float = 35.0,
        position_alpha: float = 0.45,
        rotation_alpha: float = 0.35,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("ArucoStickPoseProvider requires OpenCV with cv2.aruco.") from exc
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
            raise RuntimeError("Installed OpenCV does not provide ArucoDetector.")
        self.cv2 = cv2
        self.camera_source = camera_source
        self.calibration = calibration or CameraCalibration.load()
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # A 19 mm marker at the calibrated mounting distance occupies less
        # than OpenCV's default minimum perimeter fraction in 1280x720.
        parameters.minMarkerPerimeterRate = 0.005
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.max_reprojection_error_px = max_reprojection_error_px
        self.max_position_jump_m = max_position_jump_m
        self.max_rotation_jump_deg = max_rotation_jump_deg
        self.position_alpha = position_alpha
        self.rotation_alpha = rotation_alpha
        self._tracks = [_Track(), _Track()]
        half = STICK_PRIMARY_MARKER_SIZE_M / 2.0
        self._object_points = np.asarray(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )
        self.last_reprojection_errors = np.full(2, np.nan, dtype=np.float64)

    def reset(self) -> None:
        self.camera_source.reset()
        self._tracks = [_Track(), _Track()]

    def sample(self) -> StickPosePair7D:
        frame = self.camera_source.capture()
        if not frame.fresh and all(track.filtered_pose is not None for track in self._tracks):
            return self._pair(frame.timestamp_s, PoseState.HOLD, False)
        corners, ids, _ = self.detector.detectMarkers(frame.rgb)
        detected = {} if ids is None else {int(marker_id): corner.reshape(4, 2) for marker_id, corner in zip(ids.ravel(), corners)}
        states = []
        for index, marker_ids in enumerate(MARKER_IDS_BY_STICK):
            estimates = [
                self._estimate_pose(detected.get(marker_id), index, marker_id)
                for marker_id in marker_ids
                if marker_id in detected
            ]
            estimates = [item for item in estimates if item[0] is not None]
            if estimates:
                pose, reprojection = min(estimates, key=lambda item: item[1])
            else:
                pose, reprojection = None, np.nan
            self.last_reprojection_errors[index] = reprojection
            states.append(self._update_track(index, pose))
        severity = {
            PoseState.VALID: 0,
            PoseState.REINIT: 0,
            PoseState.HOLD: 1,
            PoseState.STALE: 2,
            PoseState.LOST: 3,
        }
        state = max(states, key=severity.__getitem__)
        if any(track.filtered_pose is None for track in self._tracks):
            raise RuntimeError(f"ArUco stick pose unavailable: {[state.value for state in states]}")
        return self._pair(frame.timestamp_s, state, frame.fresh)

    def _estimate_pose(self, corners, stick_index: int, marker_id: int):
        if corners is None:
            return None, np.nan
        result = self.cv2.solvePnPGeneric(
            self._object_points, np.asarray(corners, dtype=np.float64),
            self.calibration.matrix, self.calibration.distortion,
            flags=self.cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not result[0]:
            return None, np.nan
        errors = np.asarray(result[3], dtype=np.float64).reshape(-1)
        candidates = []
        for rvec, tvec, error in zip(result[1], result[2], errors):
            if float(error) > self.max_reprojection_error_px:
                continue
            rotation, _ = self.cv2.Rodrigues(rvec)
            t_camera_marker = np.eye(4)
            t_camera_marker[:3, :3] = rotation
            t_camera_marker[:3, 3] = np.asarray(tvec).reshape(3)
            pose = pose_matrix_to_xyz_wxyz(
                T_PALM_CAMERA @ t_camera_marker @ T_MARKER_STICK_BY_ID[marker_id]
            )
            track = self._tracks[stick_index]
            if track.raw_pose is not None and not track.lost:
                if np.linalg.norm(pose[:3] - track.raw_pose[:3]) > self.max_position_jump_m:
                    continue
                if quaternion_geodesic_error_deg(pose[3:], track.raw_pose[3:]) > self.max_rotation_jump_deg:
                    continue
            score = float(error)
            if track.raw_pose is not None and not track.lost:
                score += np.linalg.norm(pose[:3] - track.raw_pose[:3])
            candidates.append((score, pose, float(error)))
        if not candidates:
            return None, np.nan
        _, pose, error = min(candidates, key=lambda item: item[0])
        pose[3:] = canonicalize_square_stick_quaternion(
            pose[3:], STICK_REFERENCE_QUATERNIONS_PALM_WXYZ[stick_index]
        )
        return pose, error

    def _update_track(self, index: int, pose):
        track = self._tracks[index]
        if pose is None:
            track.missed += 1
            if track.missed <= 3 and track.filtered_pose is not None:
                return PoseState.HOLD
            if track.missed <= 5 and track.filtered_pose is not None:
                return PoseState.STALE
            track.lost = True
            return PoseState.LOST
        reinit = track.lost
        track.raw_pose = pose.copy()
        track.missed = 0
        track.lost = False
        if reinit or track.filtered_pose is None:
            track.filtered_pose = pose.copy()
            return PoseState.REINIT
        track.filtered_pose[:3] += self.position_alpha * (pose[:3] - track.filtered_pose[:3])
        track.filtered_pose[3:] = _slerp(track.filtered_pose[3:], pose[3:], self.rotation_alpha)
        return PoseState.VALID

    def _pair(self, timestamp_s: float, state: PoseState, fresh: bool):
        return StickPosePair7D(
            self._tracks[0].filtered_pose.astype(np.float32).copy(),
            self._tracks[1].filtered_pose.astype(np.float32).copy(),
            timestamp_s, state, fresh,
        )


def _slerp(a, b, alpha):
    q0 = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    q1 = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return (result / np.linalg.norm(result)).astype(np.float32)
    theta = np.arccos(dot)
    result = (np.sin((1-alpha)*theta)*q0 + np.sin(alpha*theta)*q1) / np.sin(theta)
    return (result / np.linalg.norm(result)).astype(np.float32)
