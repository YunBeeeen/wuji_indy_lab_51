"""Isaac's own reset numbers.  Real values, shared by every backend.

Where the twenty finger joints start (the pregrasp pose) and where the two
chopsticks sit relative to the palm at that moment.  These come out of the
Isaac environment the policy was trained in, so MuJoCo and the physical hand
both have to start from them or they are not reproducing the same episode.

Nothing here is a camera, a marker, or a guess.  The simulated ArUco fixture
that used to share this file lives in ``vision/sim_aruco.py``, and the physical
camera rig lives in ``vision/deploy_rig.py``.
"""

from __future__ import annotations

import numpy as np

#: Stick geometry, from the Isaac scene.
STICK_SIZE_M = np.asarray([0.007, 0.180, 0.007], dtype=np.float64)
STICK_MASS_KG = 0.010

# Mirrors PREGRASP_JOINT_POSITIONS in hand_grasp_env_cfg.py.  Indices 10 and 18
# are back to the recorded pose_005 value 1.6272 as of 2026-08-18: the connected
# hand's factory limits (1.680047 / 1.675141) admit it with ~0.05 rad to spare,
# so the 2026-08-17 clamp to the narrower vendor description was not needed.
# Both simulators now carry the real limits, so no clamping is expected.
# CAVEAT: 1.6272 rad == 93.2317 deg was the old placeholder upper, i.e. these
# two joints were SATURATED when pose_005 was recorded.  Re-record the pregrasp
# against the real limits before treating this pose as final.
ISAAC_PREGRASP_JOINT_POSITIONS_RAD = np.asarray(
    [0.5377866626, 0.8436813951, 0.0377136655, -0.0000001810,
     0.7017297745, 0.0553143807, 1.1822255850, 1.4215219021,
     0.4649881423, -0.0292181600, 1.6272000000, 1.1032750607,
     0.9151425958, -0.0129909236, 1.3248542547, 0.3182539344,
     0.7154092789, 0.0788998753, 1.6272000000, 0.2546040118], dtype=np.float32,
)
ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ = np.asarray(
    [[0.0250743479, 0.0242451150, 0.0969612077,
      0.4618085623, -0.0092124203, -0.1713383496, -0.8702247143],
     [0.0355986878, 0.0160842165, 0.0733669698,
      0.2051235586, -0.6018196344, -0.4935579300, -0.5934122205]], dtype=np.float64,
)
STICK_REFERENCE_QUATERNIONS_PALM_WXYZ = ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ[:, 3:].copy()


def _primary_marker_up_symmetric_reset_poses() -> np.ndarray:
    """Choose the square-symmetric reset roll with ID0/ID2 on top.

    Primary marker centers remain at stick-local Y=-90 mm, Z=+3.5 mm.  The
    square section admits four policy-equivalent shaft rolls; select the one
    whose local +Z marker normal points most toward Palm/Base +X (up).
    """

    from .stick_pose import (
        quaternion_multiply_wxyz,
        quaternion_to_rotation_matrix_wxyz,
    )

    poses = ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ.copy()
    up_p = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    for stick_index, source_pose in enumerate(ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ):
        best_score = -np.inf
        best_quaternion = source_pose[3:].copy()
        for quarter_turn in range(4):
            half_angle = quarter_turn * np.pi / 4.0
            roll_y = np.asarray(
                [np.cos(half_angle), 0.0, np.sin(half_angle), 0.0],
                dtype=np.float64,
            )
            candidate = quaternion_multiply_wxyz(source_pose[3:], roll_y)
            rotation_p_s = quaternion_to_rotation_matrix_wxyz(candidate)
            marker_normal_p = rotation_p_s[:, 2]
            score = float(marker_normal_p @ up_p)
            if score > best_score:
                best_score = score
                best_quaternion = candidate
        poses[stick_index, 3:] = best_quaternion
    return poses


MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ = (
    _primary_marker_up_symmetric_reset_poses()
)


def _primary_marker_up_symmetric_reset_poses() -> np.ndarray:
    """Choose the square-symmetric reset roll with ID0/ID2 on top.

    Primary marker centers remain at stick-local Y=-90 mm, Z=+3.5 mm.  The
    square section admits four policy-equivalent shaft rolls; select the one
    whose local +Z marker normal points most toward Palm/Base +X (up).
    """

    from .stick_pose import (
        quaternion_multiply_wxyz,
        quaternion_to_rotation_matrix_wxyz,
    )

    poses = ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ.copy()
    up_p = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    for stick_index, source_pose in enumerate(ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ):
        best_score = -np.inf
        best_quaternion = source_pose[3:].copy()
        for quarter_turn in range(4):
            half_angle = quarter_turn * np.pi / 4.0
            roll_y = np.asarray(
                [np.cos(half_angle), 0.0, np.sin(half_angle), 0.0],
                dtype=np.float64,
            )
            candidate = quaternion_multiply_wxyz(source_pose[3:], roll_y)
            rotation_p_s = quaternion_to_rotation_matrix_wxyz(candidate)
            marker_normal_p = rotation_p_s[:, 2]
            score = float(marker_normal_p @ up_p)
            if score > best_score:
                best_score = score
                best_quaternion = candidate
        poses[stick_index, 3:] = best_quaternion
    return poses


MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ = (
    _primary_marker_up_symmetric_reset_poses()
)



for _array in (STICK_SIZE_M, ISAAC_PREGRASP_JOINT_POSITIONS_RAD,
               ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ,
               STICK_REFERENCE_QUATERNIONS_PALM_WXYZ,
               MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ):
    _array.setflags(write=False)
