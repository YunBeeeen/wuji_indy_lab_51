# [vision] MuJoCo 안에 렌더링하는 가짜 ArUco 장면. 마커 검출 코드를 정답 아는 상태로 채점하는 테스트 픽스처.
"""A PRETEND camera rig, used only to test the marker-detection code.

None of these numbers describe the lab.  They describe a scene rendered inside
MuJoCo: one camera at a made-up pose, and ArUco markers stuck on the chopsticks
at a made-up layout.  The point is that inside a simulator the true stick pose
is known exactly, so the detection and PnP maths can be graded against it --
which no real camera image can do.

The names say so: ``HAND_PALM_HEIGHT_TEMP_M`` is a placeholder and
``SIM_MARKER_LAYOUT_CANDIDATE`` is a candidate.  They were previously mixed in
with Isaac's real reset numbers in one file, which is how the two Base
conventions got confused on 2026-08-21 (here Base +X is up; the physical rig in
``deploy_rig.py`` uses a robot base where +Z is up).

Used by ``tests/test_vision.py`` and ``run_policy --validate-aruco``.  The
deployed stick-pose provider must use ``deploy_rig.py`` instead.
"""

from __future__ import annotations

import numpy as np

from ..common.isaac_reset import (
    ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ,
    MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ,
    STICK_REFERENCE_QUATERNIONS_PALM_WXYZ,
    STICK_SIZE_M,
)

_UNUSED_STICK_SIZE_M = np.asarray([0.007, 0.180, 0.007], dtype=np.float64)
STICK_MASS_KG = 0.010
STICK_PRIMARY_MARKER_SIZE_M = 0.019
PRIMARY_MARKER_IDS = (0, 2)
# ID1/ID3 will share the -90 mm shaft station but use an approximately 48 deg
# neighboring roll.  No secondary transform is created before physical
# dual-marker calibration.
SECONDARY_MARKER_ROLL_APPROX_DEG = 48.0
# SIMULATION_LAYOUT_CANDIDATE only: axially staggered /| marker arrangement
# selected by rendered reset-view detection.  It is not a
# measured real marker calibration.  (stick index, center local Y, roll about
# directed shaft +Y in degrees, tab protrusion beyond the stick surface).  All
# paper extents remain at Y<=+31.4 mm, leaving the +32..+90 mm tip/contact zone
# clear.  Each pair's roll separation is exactly 48 degrees.  Protrusion is
# zero: every marker is flush with the stick surface.  The Palm-Z visibility
# audit found no full-range one-camera solution under these constraints.
SIM_MARKER_LAYOUT_CANDIDATE = {
    0: (0, -0.070, -48.0, 0.000),
    1: (0, -0.040, 0.0, 0.000),
    2: (1, -0.070, -18.0, 0.000),
    3: (1, +0.020, 30.0, 0.000),
}
MARKER_IDS_BY_STICK = ((0, 1), (2, 3))
D435_NOMINAL_BODY_SIZE_M = np.asarray([0.090, 0.025, 0.025], dtype=np.float64)
HAND_DISPLAY_RGBA = np.asarray([0.025, 0.025, 0.030, 1.0], dtype=np.float32)

# Camera2 is deliberately a reset-tail auxiliary view, not a second calibrated
# full-workspace tracker.  Its optical axis is horizontal (0 deg downward) so
# the mount can be installed level.  Height is the midpoint of the rendered
# reset ID0/ID2 marker centers (0.1210/0.1294 m in Base +X), rounded to an
# installable 0.125 m.  These values are SIMULATION_CANDIDATE, not calibration.
CAMERA2_TAIL_RESET_HEIGHT_CANDIDATE_M = 0.125
CAMERA2_TAIL_RESET_SIDE_Y_CANDIDATE_M = 0.200
CAMERA2_TAIL_RESET_CENTER_Z_CANDIDATE_M = 0.060
CAMERA2_DOWN_ANGLE_CANDIDATE_DEG = 0.0

# Physical-testbed layout measured relative to the Base frame.  None of these
# values may be used to derive or modify the calibrated camera optical transform.
BASE_PLATE_WIDTH_Y_M = 0.630
BASE_PLATE_DEPTH_Z_M = 0.625
BASE_PLATE_NEGATIVE_Z_EDGE_M = -0.160
# Measured height difference between the lower floor supporting the 4040 and
# the plate top supporting the 2020.  Base X=0 is the plate top.
BASE_PLATE_THICKNESS_M = 0.012
CAMERA_SUPPORT_FLOOR_X_M = -BASE_PLATE_THICKNESS_M
HAND_PALM_HEIGHT_TEMP_M = 0.15
HAND_PROFILE_SIZE_M = 0.020
HAND_PROFILE_HEIGHT_APPROX_M = 0.30
HAND_PROFILE_CENTER_Y_M = 0.0
HAND_PROFILE_CENTER_Z_M = 0.0
CAMERA_SUPPORT_PROFILE_SIZE_M = 0.040
CAMERA_SUPPORT_HEIGHT_APPROX_M = 0.52
# The 4040 center is 29 cm from the -Y edge and 34 cm from the +Y edge.
CAMERA_SUPPORT_CENTER_Y_M = -0.025
# It sits immediately outside the plate's +Z edge.
BASE_PLATE_POSITIVE_Z_EDGE_M = (
    BASE_PLATE_NEGATIVE_Z_EDGE_M + BASE_PLATE_DEPTH_Z_M
)
CAMERA_SUPPORT_CENTER_Z_M = (
    BASE_PLATE_POSITIVE_Z_EDGE_M + CAMERA_SUPPORT_PROFILE_SIZE_M / 2.0
)
CAMERA_BRACKET_RADIUS_APPROX_M = 0.0075
DEBUG_FRAME_AXIS_LENGTH_M = 0.050

# MuJoCo world convention used by the derived scene: Base +X(up), +Y(right),
# +Z(forward) map to world +Z, +Y, -X respectively.
R_WORLD_BASE = np.asarray(
    [[0.0, 0.0, -1.0],
     [0.0, 1.0, 0.0],
     [1.0, 0.0, 0.0]], dtype=np.float64,
)

T_MARKER_STICK = np.eye(4, dtype=np.float64)
T_MARKER_STICK[:3, 3] = [0.0, 0.090, -0.0035]


def _candidate_marker_transforms() -> dict[int, np.ndarray]:
    """Return marker->stick transforms for the uncalibrated simulation layout."""

    result: dict[int, np.ndarray] = {}
    half_section = STICK_SIZE_M[0] / 2.0
    for marker_id, (_, center_y, roll_deg, protrusion) in SIM_MARKER_LAYOUT_CANDIDATE.items():
        angle = np.deg2rad(roll_deg)
        rotation_s_m = np.asarray(
            [[np.cos(angle), 0.0, np.sin(angle)],
             [0.0, 1.0, 0.0],
             [-np.sin(angle), 0.0, np.cos(angle)]], dtype=np.float64,
        )
        normal_s = rotation_s_m[:, 2]
        surface_distance = (
            half_section * (abs(normal_s[0]) + abs(normal_s[2])) + protrusion
        )
        t_stick_marker = np.eye(4, dtype=np.float64)
        t_stick_marker[:3, :3] = rotation_s_m
        t_stick_marker[:3, 3] = normal_s * surface_distance
        t_stick_marker[1, 3] = center_y
        result[marker_id] = np.linalg.inv(t_stick_marker)
    return result


T_MARKER_STICK_BY_ID = _candidate_marker_transforms()
T_BASE_CAMERA = np.asarray(
    [[-0.002011, -0.705937, -0.708272, 0.423442],
     [-0.999958, 0.007752, -0.004887, 0.009551],
     [0.008941, 0.708233, -0.705923, 0.421189],
     [0.0, 0.0, 0.0, 1.0]], dtype=np.float64,
)
T_BASE_PALM = np.eye(4, dtype=np.float64)
T_BASE_PALM[:3, 3] = [HAND_PALM_HEIGHT_TEMP_M, 0.0, 0.0]
T_PALM_CAMERA = np.linalg.inv(T_BASE_PALM) @ T_BASE_CAMERA

def frontal_camera_validation_stick_poses(
    camera_x_positions: tuple[float, float] = (-0.12, 0.10),
    camera_y: float = 0.092,
    camera_z: float = 0.45,
) -> np.ndarray:
    """Return a deterministic two-marker-visible diagnostic pose in Palm."""

    from ..common.stick_pose import pose_matrix_to_xyz_wxyz

    poses = np.empty((2, 7), dtype=np.float64)
    for index, camera_x in enumerate(camera_x_positions):
        t_camera_marker = np.eye(4, dtype=np.float64)
        # OpenCV: +X image-right, +Y image-down, +Z forward.  Marker +Z faces
        # the camera in this diagnostic placement.
        t_camera_marker[:3, :3] = np.diag([1.0, -1.0, -1.0])
        t_camera_marker[:3, 3] = [camera_x, camera_y, camera_z]
        marker_id = PRIMARY_MARKER_IDS[index]
        poses[index] = pose_matrix_to_xyz_wxyz(
            T_PALM_CAMERA @ t_camera_marker @ T_MARKER_STICK_BY_ID[marker_id]
        )
    return poses

for _array in (D435_NOMINAL_BODY_SIZE_M, HAND_DISPLAY_RGBA, R_WORLD_BASE,
               T_MARKER_STICK, T_BASE_CAMERA, T_BASE_PALM, T_PALM_CAMERA):
    _array.setflags(write=False)
for _transform in T_MARKER_STICK_BY_ID.values():
    _transform.setflags(write=False)
