#!/usr/bin/env python3
# [vision] Stick1 단독 추적기(마커 2개 DUAL/SINGLE 분기 + workspace prior). 사용자 작성, 원본 유지.

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
import os

import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# CONFIG
# ============================================================

WIDTH = 1280
HEIGHT = 720
FPS = 30

MARKER_A_ID = 0
MARKER_B_ID = 1

MARKER_SIZE_M = 0.019
MARKER_AXIS_LENGTH_M = 0.015


def _draw_frame_axes_if_visible(vis, K, dist, rvec, tvec, length, thickness=2):
    """Avoid OpenCV's per-frame warning when an axis leaves the image."""
    points = np.asarray(
        [[0.0, 0.0, 0.0], [length, 0.0, 0.0],
         [0.0, length, 0.0], [0.0, 0.0, length]],
        dtype=np.float64,
    )
    try:
        projected, _ = cv2.projectPoints(points, rvec, tvec, K, dist)
    except cv2.error:
        return False
    xy = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    height, width = vis.shape[:2]
    if not (
        np.isfinite(xy).all()
        and np.all(xy[:, 0] >= 0.0) and np.all(xy[:, 0] < float(width))
        and np.all(xy[:, 1] >= 0.0) and np.all(xy[:, 1] < float(height))
    ):
        return False
    cv2.drawFrameAxes(vis, K, dist, rvec, tvec, length, thickness)
    return True

# ============================================================
# Workspace orientation prior
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

WORKSPACE_REFERENCE_CSV = (
    BASE_DIR
    / "references"
    / "stick1_workspace_reference_final.csv"
)

WORKSPACE_PRIOR_MAX_ANGLE_DEG = 40.0
WORKSPACE_PRIOR_MIN_MARGIN_DEG = 15.0

# True  = prior가 애매하면 기존처럼 reprojection 최소 후보로 임시 fallback
# False = prior가 애매하면 해당 SINGLE을 출력하지 않음 (더 보수적)
WORKSPACE_PRIOR_FALLBACK_TO_REPROJ = True

TARGET_IDS = {
    MARKER_A_ID,
    MARKER_B_ID,
}


# ============================================================
# SINGLE PNP GATE
# ============================================================

SINGLE_MAX_REPROJ_ERROR_PX = 1.3
SINGLE_MAX_POSITION_JUMP_M = 0.035     # 70 -> 35 mm/frame
SINGLE_MAX_ROTATION_JUMP_DEG = 17.5    # 35 -> 17.5 deg/frame

SINGLE_HISTORY_RESET_MISSES = 6        # 3 -> 6
SINGLE_REJECT_RESET_FRAMES = 6         # 3 -> 6


# ============================================================
# DUAL GATE
# ============================================================

DUAL_MAX_REPROJ_ERROR_PX = 1.3
DUAL_MAX_POSITION_JUMP_M = 0.010       # 20 -> 10 mm/frame
DUAL_MAX_ROTATION_JUMP_DEG = 7.5       # 15 -> 7.5 deg/frame

DUAL_HISTORY_RESET_MISSES = 6          # 3 -> 6
DUAL_REJECT_RESET_FRAMES = 6           # 3 -> 6


# ============================================================
# FINAL FILTER
# ============================================================

POS_ALPHA = 0.8
ROT_ALPHA = 0.8

FINAL_FILTER_RESET_MISSES = 16         # 8 -> 16


# ============================================================
# ONLINE HANDOFF CORRECTION
# ============================================================

ONLINE_CORR_POS_ALPHA = 0.1340
ONLINE_CORR_ROT_ALPHA = 0.1340

ONLINE_CORR_MAX_UPDATE_TRANS_M = 0.020
ONLINE_CORR_MAX_UPDATE_ROT_DEG = 8.0


# ============================================================
# CAMERA -> BASE
# ============================================================

T_BASE_CAMERA = np.array(
    [
        [ 0.011009927,  0.713786390, -0.700276924,  0.937431906],
        [ 0.999937040, -0.009377283,  0.006163071, -0.111095107],
        [-0.002167578, -0.700300689, -0.713844693,  0.435281684],
        [ 0.0,          0.0,          0.0,          1.0],
    ],
    dtype=np.float64,
)

#FALLBACK_T_BASE_CAMERA = np.array(
#    [
#        [0.0, +0.70710678, -0.70710678, 0.936189],
#        [1.0,  0.0,         0.0,        -0.109551],
#        [0.0, -0.70710678, -0.70710678, 0.425442],
#        [0.0,  0.0,         0.0,        1.0],
#    ],
#    dtype=np.float64,
#)


# ============================================================
# PHYSICAL MARKER -> STICK GEOMETRY
#
# Convention:
#
# T_A_B maps coordinates in B into A
#
# Stick frame:
#   origin = geometric center
#   +Y = tail -> tip
#   +Z = Marker0 face outward direction
#
# Marker0:
#   marker center -> stick center
#   [0, +88.0, -3.5] mm
#
# Marker1:
#   marker center is 37.5 mm tip-side of stick center
#   perpendicular face, ideal 90 deg
# ============================================================

T_M0_S = np.eye(
    4,
    dtype=np.float64,
)

T_M0_S[:3, :3] = np.eye(
    3,
    dtype=np.float64,
)

T_M0_S[:3, 3] = np.array(
    [
        -0.0085,
        +0.0890,
        -0.0035,
    ],
    dtype=np.float64,
)


T_M1_S = np.eye(
    4,
    dtype=np.float64,
)

T_M1_S[:3, :3] = np.array(
    [
        [0.0, 0.0, +1.0],
        [0.0, 1.0,  0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

T_M1_S[:3, 3] = np.array(
    [
        -0.0085,
        -0.0720,
        -0.0035,
    ],
    dtype=np.float64,
)


# ============================================================
# MARKER CORNERS
# ============================================================

HALF_MARKER = MARKER_SIZE_M / 2.0

MARKER_OBJECT_POINTS = np.array(
    [
        [-HALF_MARKER, +HALF_MARKER, 0.0],
        [+HALF_MARKER, +HALF_MARKER, 0.0],
        [+HALF_MARKER, -HALF_MARKER, 0.0],
        [-HALF_MARKER, -HALF_MARKER, 0.0],
    ],
    dtype=np.float64,
)


# ============================================================
# DEBUG
# ============================================================

DEBUG_LOG_DIR = (
    Path(__file__).resolve().parent
    / "logs"
    / "stick1"
)

LIVE_PRINT_INTERVAL_SEC = 1.0
DEBUG_FLUSH_EVERY_FRAME = True


# ============================================================
# TRANSFORM HELPERS
# ============================================================

def make_transform(R, t):

    T = np.eye(
        4,
        dtype=np.float64,
    )

    T[:3, :3] = np.asarray(
        R,
        dtype=np.float64,
    ).reshape(3, 3)

    T[:3, 3] = np.asarray(
        t,
        dtype=np.float64,
    ).reshape(3)

    return T
    
def load_workspace_orientation_references(csv_path):
    """
    Workspace reference CSV에서
    Marker0 / Marker1의 CAMERA-frame quaternion들을 읽는다.

    return:
        {
            0: [q0, q1, ...],
            1: [q0, q1, ...],
        }
    """

    refs = {
        0: [],
        1: [],
    }

    if not os.path.exists(csv_path):
        print()
        print("[WORKSPACE PRIOR WARNING]")
        print("Reference CSV not found:")
        print(csv_path)
        print()

        return refs

    import pandas as pd

    df = pd.read_csv(csv_path)

    required = [
        "marker_id",
        "marker_cam_qw",
        "marker_cam_qx",
        "marker_cam_qy",
        "marker_cam_qz",
    ]

    for name in required:
        if name not in df.columns:
            raise RuntimeError(
                f"Workspace reference CSV missing column: {name}"
            )

    for _, row in df.iterrows():

        marker_id = int(row["marker_id"])

        if marker_id not in refs:
            continue

        q = np.array(
            [
                row["marker_cam_qw"],
                row["marker_cam_qx"],
                row["marker_cam_qy"],
                row["marker_cam_qz"],
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(q)):
            continue

        norm = np.linalg.norm(q)

        if norm < 1e-12:
            continue

        q /= norm

        refs[marker_id].append(q)

    print()
    print("=" * 80)
    print("WORKSPACE ORIENTATION PRIOR")
    print("=" * 80)
    print("CSV:", csv_path)
    print(f"Marker0 references: {len(refs[0])}")
    print(f"Marker1 references: {len(refs[1])}")
    print("=" * 80)
    print()

    return refs
    
def select_candidate_from_workspace_prior(
    candidates,
    marker_id,
    workspace_refs,
    max_angle_deg=40.0,
    min_margin_deg=15.0,
):
    """
    IPPE candidate 중 workspace reference orientation에
    가장 가까운 물리 branch를 선택한다.

    candidates:
        get_ippe_candidates()가 반환하는 candidate list

    return:
        selected_candidate, debug_info

    선택을 확신할 수 없으면:
        None, debug_info
    """

    refs = workspace_refs.get(marker_id, [])

    debug = {
        "accepted": False,
        "reason": None,
        "best_branch": None,
        "best_angle_deg": None,
        "second_angle_deg": None,
        "margin_deg": None,
        "candidate_results": [],
    }

    if candidates is None or len(candidates) == 0:
        debug["reason"] = "NO_CANDIDATES"
        return None, debug

    if len(refs) == 0:
        debug["reason"] = "NO_WORKSPACE_REFS"
        return None, debug

    results = []

    for candidate in candidates:

        T_CAMERA_MARKER = candidate["T_CAMERA_MARKER"]

        q_candidate = rotation_matrix_to_quaternion_wxyz(
            T_CAMERA_MARKER[:3, :3]
        )

        ref_angles = [
            quaternion_angle_deg(
                q_candidate,
                q_ref,
            )
            for q_ref in refs
        ]

        nearest_angle = float(
            np.min(ref_angles)
        )

        nearest_ref_index = int(
            np.argmin(ref_angles)
        )

        results.append(
            {
                "candidate": candidate,
                "branch": candidate["branch"],
                "nearest_angle_deg": nearest_angle,
                "nearest_ref_index": nearest_ref_index,
                "reproj": candidate["reproj"],
            }
        )

    # workspace orientation distance 우선
    # reprojection은 tie breaker
    results.sort(
        key=lambda x: (
            x["nearest_angle_deg"],
            x["reproj"],
        )
    )

    debug["candidate_results"] = results

    best = results[0]

    if len(results) >= 2:
        second = results[1]
        second_angle = second["nearest_angle_deg"]
    else:
        second_angle = float("inf")

    best_angle = best["nearest_angle_deg"]
    margin = second_angle - best_angle

    debug["best_branch"] = best["branch"]
    debug["best_angle_deg"] = best_angle
    debug["second_angle_deg"] = second_angle
    debug["margin_deg"] = margin

    # --------------------------------------------------------
    # Safety gate
    #
    # 정상 workspace에 충분히 가까워야 하고
    # 다른 후보보다 명확하게 좋아야 함.
    # --------------------------------------------------------

    if best_angle > max_angle_deg:
        debug["reason"] = "TOO_FAR_FROM_WORKSPACE"
        return None, debug

    if margin < min_margin_deg:
        debug["reason"] = "AMBIGUOUS_WORKSPACE_PRIOR"
        return None, debug

    debug["accepted"] = True
    debug["reason"] = "WORKSPACE_PRIOR"

    return best["candidate"], debug
    
def invert_transform(T):

    T = np.asarray(
        T,
        dtype=np.float64,
    )

    R = T[:3, :3]
    t = T[:3, 3]

    out = np.eye(
        4,
        dtype=np.float64,
    )

    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t

    return out


def transform_points(T, points):

    points = np.asarray(
        points,
        dtype=np.float64,
    ).reshape(-1, 3)

    return (
        points @ T[:3, :3].T
        + T[:3, 3]
    )


# ============================================================
# QUATERNION
# ============================================================

def rotation_matrix_to_quaternion_wxyz(R):

    R = np.asarray(
        R,
        dtype=np.float64,
    ).reshape(3, 3)

    q = np.empty(
        4,
        dtype=np.float64,
    )

    trace = np.trace(R)

    if trace > 0.0:

        s = np.sqrt(
            trace + 1.0
        ) * 2.0

        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s

    elif (
        R[0, 0] > R[1, 1]
        and
        R[0, 0] > R[2, 2]
    ):

        s = np.sqrt(
            1.0
            + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        ) * 2.0

        q[0] = (
            R[2, 1] - R[1, 2]
        ) / s

        q[1] = 0.25 * s

        q[2] = (
            R[0, 1] + R[1, 0]
        ) / s

        q[3] = (
            R[0, 2] + R[2, 0]
        ) / s

    elif R[1, 1] > R[2, 2]:

        s = np.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2.0

        q[0] = (
            R[0, 2] - R[2, 0]
        ) / s

        q[1] = (
            R[0, 1] + R[1, 0]
        ) / s

        q[2] = 0.25 * s

        q[3] = (
            R[1, 2] + R[2, 1]
        ) / s

    else:

        s = np.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2.0

        q[0] = (
            R[1, 0] - R[0, 1]
        ) / s

        q[1] = (
            R[0, 2] + R[2, 0]
        ) / s

        q[2] = (
            R[1, 2] + R[2, 1]
        ) / s

        q[3] = 0.25 * s

    q /= np.linalg.norm(q)

    return q


def quaternion_wxyz_to_rotation_matrix(q):

    q = np.asarray(
        q,
        dtype=np.float64,
    ).reshape(4)

    q = q / np.linalg.norm(q)

    w, x, y, z = q

    return np.array(
        [
            [
                1.0 - 2.0 * (y*y + z*z),
                2.0 * (x*y - z*w),
                2.0 * (x*z + y*w),
            ],
            [
                2.0 * (x*y + z*w),
                1.0 - 2.0 * (x*x + z*z),
                2.0 * (y*z - x*w),
            ],
            [
                2.0 * (x*z - y*w),
                2.0 * (y*z + x*w),
                1.0 - 2.0 * (x*x + y*y),
            ],
        ],
        dtype=np.float64,
    )


def quaternion_angle_deg(q1, q2):

    q1 = np.asarray(
        q1,
        dtype=np.float64,
    )

    q2 = np.asarray(
        q2,
        dtype=np.float64,
    )

    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = abs(
        float(
            np.dot(
                q1,
                q2,
            )
        )
    )

    dot = np.clip(
        dot,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            2.0
            * np.arccos(dot)
        )
    )


def quaternion_slerp(
    q0,
    q1,
    alpha,
):

    q0 = np.asarray(
        q0,
        dtype=np.float64,
    )

    q1 = np.asarray(
        q1,
        dtype=np.float64,
    )

    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(
        np.dot(
            q0,
            q1,
        )
    )

    if dot < 0.0:

        q1 = -q1
        dot = -dot

    dot = np.clip(
        dot,
        -1.0,
        1.0,
    )

    if dot > 0.9995:

        q = (
            (1.0 - alpha) * q0
            + alpha * q1
        )

        return q / np.linalg.norm(q)

    theta_0 = np.arccos(dot)

    theta = (
        theta_0
        * alpha
    )

    sin_theta_0 = np.sin(
        theta_0
    )

    s0 = (
        np.sin(
            theta_0 - theta
        )
        / sin_theta_0
    )

    s1 = (
        np.sin(theta)
        / sin_theta_0
    )

    q = (
        s0 * q0
        + s1 * q1
    )

    return q / np.linalg.norm(q)


# ============================================================
# TRANSFORM DIFFERENCE
# ============================================================
def rotation_difference_deg_from_T(T_a, T_b):
    """
    두 4x4 transform의 rotation 차이 [deg].
    """
    R_a = np.asarray(T_a[:3, :3], dtype=np.float64)
    R_b = np.asarray(T_b[:3, :3], dtype=np.float64)

    R_rel = R_a.T @ R_b

    cos_theta = (
        np.trace(R_rel) - 1.0
    ) * 0.5

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(cos_theta)
        )
    )
    
def transform_difference(
    T_a,
    T_b,
):

    dp = float(
        np.linalg.norm(
            T_a[:3, 3]
            - T_b[:3, 3]
        )
    )

    qa = (
        rotation_matrix_to_quaternion_wxyz(
            T_a[:3, :3]
        )
    )

    qb = (
        rotation_matrix_to_quaternion_wxyz(
            T_b[:3, :3]
        )
    )

    dr = quaternion_angle_deg(
        qa,
        qb,
    )

    return dp, dr


def smooth_transform(
    previous_T,
    new_T,
    pos_alpha,
    rot_alpha,
):

    if previous_T is None:

        return new_T.copy()

    p = (
        pos_alpha
        * new_T[:3, 3]
        +
        (1.0 - pos_alpha)
        * previous_T[:3, 3]
    )

    q0 = (
        rotation_matrix_to_quaternion_wxyz(
            previous_T[:3, :3]
        )
    )

    q1 = (
        rotation_matrix_to_quaternion_wxyz(
            new_T[:3, :3]
        )
    )

    q = quaternion_slerp(
        q0,
        q1,
        rot_alpha,
    )

    return make_transform(
        quaternion_wxyz_to_rotation_matrix(q),
        p,
    )


# ============================================================
# PHYSICAL GEOMETRY
# ============================================================

def build_marker_to_stick_transforms():

    marker_to_stick = {
        MARKER_A_ID:
            T_M0_S.copy(),

        MARKER_B_ID:
            T_M1_S.copy(),
    }

    # debug only:
    # T_M0_M1 = T_M0_S @ T_S_M1

    T_M0_M1 = (
        T_M0_S
        @ invert_transform(
            T_M1_S
        )
    )

    return (
        T_M0_M1,
        marker_to_stick,
    )


def build_dual_object_points(
    marker_to_stick,
):

    T_S_M0 = invert_transform(
        marker_to_stick[
            MARKER_A_ID
        ]
    )

    T_S_M1 = invert_transform(
        marker_to_stick[
            MARKER_B_ID
        ]
    )

    object_a = transform_points(
        T_S_M0,
        MARKER_OBJECT_POINTS,
    )

    object_b = transform_points(
        T_S_M1,
        MARKER_OBJECT_POINTS,
    )

    return object_a, object_b


# ============================================================
# IPPE SINGLE-MARKER CANDIDATES
# ============================================================

def get_ippe_candidates(
    image_points,
    K,
    dist_coeffs,
):

    image_points = np.asarray(
        image_points,
        dtype=np.float64,
    ).reshape(4, 2)

    result = cv2.solvePnPGeneric(
        MARKER_OBJECT_POINTS,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not result[0]:

        return []

    rvecs = result[1]
    tvecs = result[2]

    output = []

    for branch, (
        rvec,
        tvec,
    ) in enumerate(
        zip(
            rvecs,
            tvecs,
        )
    ):

        rvec = np.asarray(
            rvec,
            dtype=np.float64,
        ).reshape(3, 1)

        tvec = np.asarray(
            tvec,
            dtype=np.float64,
        ).reshape(3, 1)

        R_CAMERA_MARKER, _ = (
            cv2.Rodrigues(
                rvec
            )
        )

        T_CAMERA_MARKER = (
            make_transform(
                R_CAMERA_MARKER,
                tvec,
            )
        )

        projected, _ = (
            cv2.projectPoints(
                MARKER_OBJECT_POINTS,
                rvec,
                tvec,
                K,
                dist_coeffs,
            )
        )

        projected = projected.reshape(
            4,
            2,
        )

        reproj = float(
            np.mean(
                np.linalg.norm(
                    projected
                    - image_points,
                    axis=1,
                )
            )
        )

        output.append(
            {
                "branch":
                    branch,

                "rvec":
                    rvec,

                "tvec":
                    tvec,

                "reproj":
                    reproj,

                "T_CAMERA_MARKER":
                    T_CAMERA_MARKER,
            }
        )

    return output


# ============================================================
# IPPE AMBIGUITY DIAGNOSTIC
# ============================================================

def summarize_ippe_candidates(
    candidates,
):

    result = {
        "candidate_count":
            len(candidates),

        "reproj0":
            None,

        "reproj1":
            None,

        "position_separation_mm":
            None,

        "rotation_separation_deg":
            None,
    }

    if len(candidates) >= 1:

        result[
            "reproj0"
        ] = candidates[0][
            "reproj"
        ]

    if len(candidates) >= 2:

        result[
            "reproj1"
        ] = candidates[1][
            "reproj"
        ]

        T0 = candidates[0][
            "T_CAMERA_MARKER"
        ]

        T1 = candidates[1][
            "T_CAMERA_MARKER"
        ]

        dp, dr = transform_difference(
            T0,
            T1,
        )

        result[
            "position_separation_mm"
        ] = dp * 1000.0

        result[
            "rotation_separation_deg"
        ] = dr

    return result


# ============================================================
# SELECT SINGLE PNP
# ============================================================

def select_single_candidate(
    candidates,
    T_BASE_CAMERA,
    T_MARKER_STICK,
    previous_position,
    previous_quaternion,
):

    if not candidates:

        return None

    processed = []

    for candidate in candidates:

        T_BASE_MARKER = (
            T_BASE_CAMERA
            @ candidate[
                "T_CAMERA_MARKER"
            ]
        )

        T_BASE_STICK = (
            T_BASE_MARKER
            @ T_MARKER_STICK
        )

        position = (
            T_BASE_STICK[
                :3,
                3
            ].copy()
        )

        quaternion = (
            rotation_matrix_to_quaternion_wxyz(
                T_BASE_STICK[
                    :3,
                    :3
                ]
            )
        )

        if (
            previous_quaternion
            is not None
            and
            np.dot(
                previous_quaternion,
                quaternion,
            ) < 0.0
        ):

            quaternion = -quaternion

        if (
            previous_position
            is None
            or
            previous_quaternion
            is None
        ):

            position_jump = 0.0
            rotation_jump = 0.0

        else:

            position_jump = float(
                np.linalg.norm(
                    position
                    - previous_position
                )
            )

            rotation_jump = (
                quaternion_angle_deg(
                    previous_quaternion,
                    quaternion,
                )
            )

        reject_reasons = []

        if (
            candidate["reproj"]
            > SINGLE_MAX_REPROJ_ERROR_PX
        ):

            reject_reasons.append(
                "REPROJ"
            )

        if (
            previous_position
            is not None
            and
            position_jump
            > SINGLE_MAX_POSITION_JUMP_M
        ):

            reject_reasons.append(
                "POSITION"
            )

        if (
            previous_quaternion
            is not None
            and
            rotation_jump
            > SINGLE_MAX_ROTATION_JUMP_DEG
        ):

            reject_reasons.append(
                "ROTATION"
            )

        accepted = (
            len(reject_reasons)
            == 0
        )

        score = (
            candidate["reproj"]
            / SINGLE_MAX_REPROJ_ERROR_PX
        )

        if previous_position is not None:

            score += (
                position_jump
                / SINGLE_MAX_POSITION_JUMP_M
            )

        if previous_quaternion is not None:

            score += (
                rotation_jump
                / SINGLE_MAX_ROTATION_JUMP_DEG
            )

        processed.append(
            {
                **candidate,

                "T_BASE_MARKER":
                    T_BASE_MARKER,

                "T_BASE_STICK":
                    T_BASE_STICK,

                "position":
                    position,

                "quaternion":
                    quaternion,

                "position_jump":
                    position_jump,

                "rotation_jump":
                    rotation_jump,

                "reject_reasons":
                    reject_reasons,

                "accepted":
                    accepted,

                "score":
                    float(score),
            }
        )

    valid = [
        x
        for x in processed
        if x["accepted"]
    ]

    if not valid:

        return None

    return min(
        valid,
        key=lambda x:
            x["score"],
    )


# ============================================================
# DUAL 8-CORNER ESTIMATION
# ============================================================

def estimate_dual_pose(
    image_points_a,
    image_points_b,
    object_points_a,
    object_points_b,
    K,
    dist_coeffs,
    T_BASE_CAMERA,
    T_CAMERA_BASE,
    previous_position,
    previous_quaternion,
):

    object_points = np.vstack(
        [
            object_points_a,
            object_points_b,
        ]
    ).astype(
        np.float64
    )

    image_points = np.vstack(
        [
            np.asarray(
                image_points_a,
                dtype=np.float64,
            ).reshape(4, 2),

            np.asarray(
                image_points_b,
                dtype=np.float64,
            ).reshape(4, 2),
        ]
    )

    use_guess = (
        previous_position
        is not None
        and
        previous_quaternion
        is not None
    )

    if use_guess:

        T_BASE_STICK_PREV = (
            make_transform(
                quaternion_wxyz_to_rotation_matrix(
                    previous_quaternion
                ),
                previous_position,
            )
        )

        T_CAMERA_STICK_PREV = (
            T_CAMERA_BASE
            @ T_BASE_STICK_PREV
        )

        rvec_guess, _ = (
            cv2.Rodrigues(
                T_CAMERA_STICK_PREV[
                    :3,
                    :3
                ]
            )
        )

        tvec_guess = (
            T_CAMERA_STICK_PREV[
                :3,
                3
            ]
            .reshape(3, 1)
            .copy()
        )

        success, rvec, tvec = (
            cv2.solvePnP(
                object_points,
                image_points,
                K,
                dist_coeffs,
                rvec_guess,
                tvec_guess,
                True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        )

    else:

        success, rvec, tvec = (
            cv2.solvePnP(
                object_points,
                image_points,
                K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        )

    if not success:

        return None

    rvec = np.asarray(
        rvec,
        dtype=np.float64,
    ).reshape(3, 1)

    tvec = np.asarray(
        tvec,
        dtype=np.float64,
    ).reshape(3, 1)

    R_CAMERA_STICK, _ = (
        cv2.Rodrigues(rvec)
    )

    T_CAMERA_STICK = (
        make_transform(
            R_CAMERA_STICK,
            tvec,
        )
    )

    projected, _ = (
        cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            K,
            dist_coeffs,
        )
    )

    projected = projected.reshape(
        -1,
        2,
    )

    errors = np.linalg.norm(
        projected
        - image_points,
        axis=1,
    )

    reproj_mean = float(
        np.mean(errors)
    )

    reproj_max = float(
        np.max(errors)
    )

    T_BASE_STICK = (
        T_BASE_CAMERA
        @ T_CAMERA_STICK
    )

    position = (
        T_BASE_STICK[
            :3,
            3
        ].copy()
    )

    quaternion = (
        rotation_matrix_to_quaternion_wxyz(
            T_BASE_STICK[
                :3,
                :3
            ]
        )
    )

    if (
        previous_quaternion
        is not None
        and
        np.dot(
            previous_quaternion,
            quaternion,
        ) < 0.0
    ):

        quaternion = -quaternion

    if (
        previous_position
        is None
        or
        previous_quaternion
        is None
    ):

        position_jump = 0.0
        rotation_jump = 0.0

    else:

        position_jump = float(
            np.linalg.norm(
                position
                - previous_position
            )
        )

        rotation_jump = (
            quaternion_angle_deg(
                previous_quaternion,
                quaternion,
            )
        )

    reject_reasons = []

    if (
        reproj_mean
        > DUAL_MAX_REPROJ_ERROR_PX
    ):

        reject_reasons.append(
            "REPROJ"
        )

    if (
        previous_position
        is not None
        and
        position_jump
        > DUAL_MAX_POSITION_JUMP_M
    ):

        reject_reasons.append(
            "POSITION"
        )

    if (
        previous_quaternion
        is not None
        and
        rotation_jump
        > DUAL_MAX_ROTATION_JUMP_DEG
    ):

        reject_reasons.append(
            "ROTATION"
        )

    return {
        "accepted":
            (
                len(reject_reasons)
                == 0
            ),

        "reject_reasons":
            reject_reasons,

        "T_CAMERA_STICK":
            T_CAMERA_STICK,

        "T_BASE_STICK":
            T_BASE_STICK,

        "position":
            position,

        "quaternion":
            quaternion,

        "rvec":
            rvec,

        "tvec":
            tvec,

        "reproj_mean":
            reproj_mean,

        "reproj_max":
            reproj_max,

        "position_jump":
            position_jump,

        "rotation_jump":
            rotation_jump,
    }


# ============================================================
# MARKER-LEVEL SINGLE vs DUAL DIAGNOSTIC
#
# Direct:
#
# T_C_M = single marker IPPE
#
# DUAL implied:
#
# T_B_M =
# T_B_S @ T_S_M
#
# T_C_M =
# T_C_B @ T_B_M
#
# This isolates marker PnP error from the stick-center offset.
# ============================================================
def calculate_marker_image_geometry(
    image_points,
    T_CAMERA_MARKER=None,
):
    """
    image_points: shape (4,2)

    Returns:
      edge lengths [px]
      diagonals [px]
      polygon area [px^2]
      apparent tilt angle [deg]
    """

    if image_points is None:
        return None

    p = np.asarray(
        image_points,
        dtype=np.float64,
    ).reshape(4, 2)

    # ArUco corner order 그대로
    edges = np.array(
        [
            np.linalg.norm(p[1] - p[0]),
            np.linalg.norm(p[2] - p[1]),
            np.linalg.norm(p[3] - p[2]),
            np.linalg.norm(p[0] - p[3]),
        ],
        dtype=np.float64,
    )

    diag0 = float(
        np.linalg.norm(
            p[2] - p[0]
        )
    )

    diag1 = float(
        np.linalg.norm(
            p[3] - p[1]
        )
    )

    area = float(
        cv2.contourArea(
            p.astype(
                np.float32
            )
        )
    )

    edge_mean = float(
        np.mean(edges)
    )

    edge_min = float(
        np.min(edges)
    )

    edge_max = float(
        np.max(edges)
    )

    edge_ratio = (
        edge_max / edge_min
        if edge_min > 1e-9
        else None
    )

    tilt_deg = None

    if T_CAMERA_MARKER is not None:

        # marker local +Z normal in camera coordinates
        normal_camera = (
            T_CAMERA_MARKER[
                :3,
                :3
            ][:, 2]
        )

        normal_camera = (
            normal_camera
            / np.linalg.norm(
                normal_camera
            )
        )

        # Camera optical axis = +Z
        cos_angle = np.clip(
            abs(
                normal_camera[2]
            ),
            0.0,
            1.0,
        )

        tilt_deg = float(
            np.degrees(
                np.arccos(
                    cos_angle
                )
            )
        )

    return {
        "edge0_px": float(edges[0]),
        "edge1_px": float(edges[1]),
        "edge2_px": float(edges[2]),
        "edge3_px": float(edges[3]),

        "edge_mean_px": edge_mean,
        "edge_min_px": edge_min,
        "edge_max_px": edge_max,
        "edge_ratio": edge_ratio,

        "diag0_px": diag0,
        "diag1_px": diag1,

        "area_px2": area,

        "tilt_deg": tilt_deg,
    }
    
def calculate_marker_diagnostic(
    single,
    dual,
    T_MARKER_STICK,
    T_CAMERA_BASE,
):

    if (
        single is None
        or
        dual is None
        or
        not dual["accepted"]
    ):

        return None

    T_BASE_MARKER_DIRECT = (
        single[
            "T_BASE_MARKER"
        ]
    )

    T_STICK_MARKER = (
        invert_transform(
            T_MARKER_STICK
        )
    )

    T_BASE_MARKER_DUAL = (
        dual[
            "T_BASE_STICK"
        ]
        @ T_STICK_MARKER
    )

    T_CAMERA_MARKER_DIRECT = (
        single[
            "T_CAMERA_MARKER"
        ]
    )

    T_CAMERA_MARKER_DUAL = (
        T_CAMERA_BASE
        @ T_BASE_MARKER_DUAL
    )

    p_direct_cam = (
        T_CAMERA_MARKER_DIRECT[
            :3,
            3
        ].copy()
    )

    p_dual_cam = (
        T_CAMERA_MARKER_DUAL[
            :3,
            3
        ].copy()
    )

    delta_cam = (
        p_direct_cam
        - p_dual_cam
    )

    distance_cam = float(
        np.linalg.norm(
            delta_cam
        )
    )

    _, rotation_difference = (
        transform_difference(
            T_CAMERA_MARKER_DIRECT,
            T_CAMERA_MARKER_DUAL,
        )
    )

    return {
        "direct_cam":
            p_direct_cam,

        "dual_cam":
            p_dual_cam,

        "delta_cam":
            delta_cam,

        "distance_mm":
            distance_cam
            * 1000.0,

        "rotation_difference_deg":
            rotation_difference,
    }

def build_single_from_specific_candidate(
    candidate,
    T_BASE_CAMERA,
    T_MARKER_STICK,
    marker_reference_quaternion_camera=None,
    marker_reference_source="NONE",
):

    if candidate is None:
        return None

    # Basic reprojection gate
    if (
        candidate["reproj"]
        > SINGLE_MAX_REPROJ_ERROR_PX
    ):
        return None

    # --------------------------------------------------------
    # CAMERA <- MARKER
    # --------------------------------------------------------

    T_CAMERA_MARKER = (
        candidate[
            "T_CAMERA_MARKER"
        ]
    )

    marker_quaternion_camera = (
        rotation_matrix_to_quaternion_wxyz(
            T_CAMERA_MARKER[
                :3,
                :3
            ]
        )
    )

    marker_reference_rotation_deg = None

    if (
        marker_reference_quaternion_camera
        is not None
    ):

        # Quaternion sign canonicalization
        if (
            np.dot(
                marker_reference_quaternion_camera,
                marker_quaternion_camera,
            )
            < 0.0
        ):
            marker_quaternion_camera = (
                -marker_quaternion_camera
            )

        marker_reference_rotation_deg = (
            quaternion_angle_deg(
                marker_reference_quaternion_camera,
                marker_quaternion_camera,
            )
        )

    # --------------------------------------------------------
    # BASE <- MARKER
    # --------------------------------------------------------

    T_BASE_MARKER = (
        T_BASE_CAMERA
        @ T_CAMERA_MARKER
    )

    # --------------------------------------------------------
    # BASE <- STICK
    # --------------------------------------------------------

    T_BASE_STICK = (
        T_BASE_MARKER
        @ T_MARKER_STICK
    )

    position = (
        T_BASE_STICK[
            :3,
            3
        ].copy()
    )

    quaternion = (
        rotation_matrix_to_quaternion_wxyz(
            T_BASE_STICK[
                :3,
                :3
            ]
        )
    )

    # Score is used only if later two singles must be compared.
    score = (
        candidate["reproj"]
        / SINGLE_MAX_REPROJ_ERROR_PX
    )

    if (
        marker_reference_rotation_deg
        is not None
    ):
        score += (
            marker_reference_rotation_deg
            / 180.0
        )

    return {
        **candidate,

        "T_BASE_MARKER":
            T_BASE_MARKER,

        "T_BASE_STICK":
            T_BASE_STICK,

        "position":
            position,

        "quaternion":
            quaternion,

        "accepted":
            True,

        "score":
            float(score),

        "position_jump":
            0.0,

        "rotation_jump":
            0.0,

        "reject_reasons":
            [],

        # NEW
        "marker_quaternion_camera":
            marker_quaternion_camera,

        "marker_reference_source":
            marker_reference_source,

        "marker_reference_rotation_deg":
            marker_reference_rotation_deg,
    }


def select_physical_branch(
    marker_id,
    candidates,
    T_BASE_CAMERA,
    T_MARKER_STICK,
    T_CAMERA_BASE,
    dual,
    previous_marker_quaternion_camera,
    workspace_refs,
):
    """
    Generic physical IPPE branch selector for BOTH Marker0 and Marker1.

    Priority:
      1) DUAL-implied marker orientation
      2) previous physically selected marker orientation (HISTORY)
      3) calibrated workspace orientation prior (fresh start)
      4) optional reprojection fallback if prior is inconclusive

    All orientation comparisons are performed in CAMERA <- MARKER frame.
    """

    if candidates is None or len(candidates) == 0:
        return None

    # ========================================================
    # 1. Build orientation reference
    # ========================================================

    reference_quaternion = None
    reference_source = "NONE"

    if dual is not None and dual.get("accepted", False):

        # T_S_M : STICK <- MARKER
        T_STICK_MARKER = invert_transform(
            T_MARKER_STICK
        )

        # BASE <- MARKER, implied by DUAL stick pose
        T_BASE_MARKER_DUAL = (
            dual["T_BASE_STICK"]
            @ T_STICK_MARKER
        )

        # CAMERA <- MARKER, implied by DUAL
        T_CAMERA_MARKER_DUAL = (
            T_CAMERA_BASE
            @ T_BASE_MARKER_DUAL
        )

        reference_quaternion = (
            rotation_matrix_to_quaternion_wxyz(
                T_CAMERA_MARKER_DUAL[:3, :3]
            )
        )

        reference_source = "DUAL"

    elif previous_marker_quaternion_camera is not None:

        reference_quaternion = (
            previous_marker_quaternion_camera.copy()
        )

        reference_source = "HISTORY"

    # ========================================================
    # 2. Basic reprojection gate
    # ========================================================

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate["reproj"] <= SINGLE_MAX_REPROJ_ERROR_PX
    ]

    if len(valid_candidates) == 0:
        return None

    # ========================================================
    # 3. Fresh startup: no DUAL and no HISTORY
    #    -> workspace orientation prior
    # ========================================================

    if reference_quaternion is None:

        chosen, prior_debug = (
            select_candidate_from_workspace_prior(
                valid_candidates,
                marker_id,
                workspace_refs,
                max_angle_deg=WORKSPACE_PRIOR_MAX_ANGLE_DEG,
                min_margin_deg=WORKSPACE_PRIOR_MIN_MARGIN_DEG,
            )
        )

        if chosen is not None:

            print()
            print(
                f"M{marker_id} FRESH-START WORKSPACE PRIOR:"
            )

            for item in prior_debug["candidate_results"]:
                print(
                    f"  branch {item['branch']} "
                    f"| nearest_ref={item['nearest_angle_deg']:.2f}deg "
                    f"| reproj={item['reproj']:.3f}px"
                )

            print(
                f"  SELECTED = branch {prior_debug['best_branch']}"
            )
            print(
                f"  margin   = {prior_debug['margin_deg']:.2f}deg"
            )

            result = build_single_from_specific_candidate(
                chosen,
                T_BASE_CAMERA,
                T_MARKER_STICK,
                None,
                "WORKSPACE_PRIOR",
            )

            if result is not None:
                # For terminal/debug display: distance to nearest workspace ref.
                result["marker_reference_rotation_deg"] = float(
                    prior_debug["best_angle_deg"]
                )
                result["workspace_prior_debug"] = prior_debug

            return result

        # ----------------------------------------------------
        # Workspace prior could not decide confidently
        # ----------------------------------------------------

        print()
        print(
            f"[M{marker_id} WORKSPACE PRIOR FAILED]"
        )
        print(
            "reason =",
            prior_debug["reason"],
        )

        if prior_debug["best_angle_deg"] is not None:
            print(
                f"best angle = {prior_debug['best_angle_deg']:.2f}deg"
            )

        if prior_debug["margin_deg"] is not None:
            print(
                f"margin = {prior_debug['margin_deg']:.2f}deg"
            )

        if not WORKSPACE_PRIOR_FALLBACK_TO_REPROJ:
            return None

        # Temporary continuity fallback: preserve previous behavior.
        chosen = min(
            valid_candidates,
            key=lambda c: c["reproj"],
        )

        result = build_single_from_specific_candidate(
            chosen,
            T_BASE_CAMERA,
            T_MARKER_STICK,
            None,
            "WORKSPACE_PRIOR_FAILED_REPROJ",
        )

        if result is not None:
            result["workspace_prior_debug"] = prior_debug

        return result

    # ========================================================
    # 4. DUAL / HISTORY physical branch selection
    #
    # Compare MARKER orientation itself.
    # DO NOT choose based on stick-center continuity here.
    # ========================================================

    ranked = []

    for candidate in valid_candidates:

        q_candidate = (
            rotation_matrix_to_quaternion_wxyz(
                candidate["T_CAMERA_MARKER"][:3, :3]
            )
        )

        dR = quaternion_angle_deg(
            reference_quaternion,
            q_candidate,
        )

        ranked.append(
            (
                dR,
                candidate["reproj"],
                candidate,
            )
        )

    # 1st: physical orientation distance
    # 2nd: reprojection error
    ranked.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    best_dR, _, chosen = ranked[0]

    result = build_single_from_specific_candidate(
        chosen,
        T_BASE_CAMERA,
        T_MARKER_STICK,
        reference_quaternion,
        reference_source,
    )

    if result is not None:
        result["marker_reference_rotation_deg"] = float(best_dR)

    return result


def compare_ippe_candidates_to_dual(
    candidates,
    dual,
    T_MARKER_STICK,
    T_CAMERA_BASE,
):

    if (
        candidates is None
        or len(candidates) == 0
        or dual is None
        or not dual["accepted"]
    ):
        return None

    # DUAL stick pose -> DUAL-implied marker pose
    T_STICK_MARKER = invert_transform(
        T_MARKER_STICK
    )

    T_BASE_MARKER_DUAL = (
        dual["T_BASE_STICK"]
        @ T_STICK_MARKER
    )

    T_CAMERA_MARKER_DUAL = (
        T_CAMERA_BASE
        @ T_BASE_MARKER_DUAL
    )

    results = []

    for candidate in candidates:

        dp, dr = transform_difference(
            candidate["T_CAMERA_MARKER"],
            T_CAMERA_MARKER_DUAL,
        )

        results.append(
            {
                "branch": candidate["branch"],
                "reproj": candidate["reproj"],
                "dp_mm": dp * 1000.0,
                "dr_deg": dr,
            }
        )

    return results
    
# ============================================================
# ONLINE CORRECTION
# ============================================================

def make_correction_state():

    return {
        "T":
            None,

        "last_update_time":
            None,
    }


def update_online_correction(
    state,
    single,
    dual,
    now,
):

    candidate = (
        invert_transform(
            single[
                "T_BASE_STICK"
            ]
        )
        @ dual[
            "T_BASE_STICK"
        ]
    )

    if state["T"] is None:

        state["T"] = (
            candidate.copy()
        )

        state[
            "last_update_time"
        ] = now

        return True

    dp, dr = transform_difference(
        state["T"],
        candidate,
    )

    if (
        dp
        > ONLINE_CORR_MAX_UPDATE_TRANS_M
        or
        dr
        > ONLINE_CORR_MAX_UPDATE_ROT_DEG
    ):

        return False

    state["T"] = smooth_transform(
        state["T"],
        candidate,
        ONLINE_CORR_POS_ALPHA,
        ONLINE_CORR_ROT_ALPHA,
    )

    state[
        "last_update_time"
    ] = now

    return True


def apply_online_correction(
    single,
    state,
):

    output = dict(
        single
    )

    if state["T"] is None:

        output[
            "online_corrected"
        ] = False

        return output

    T = (
        single[
            "T_BASE_STICK"
        ]
        @ state[
            "T"
        ]
    )

    output[
        "T_BASE_STICK"
    ] = T

    output[
        "position"
    ] = (
        T[
            :3,
            3
        ].copy()
    )

    output[
        "quaternion"
    ] = (
        rotation_matrix_to_quaternion_wxyz(
            T[
                :3,
                :3
            ]
        )
    )

    output[
        "online_corrected"
    ] = True

    return output


# ============================================================
# FINAL SOURCE
# ============================================================

def choose_final_source(
    singles,
    dual,
):

    if (
        dual is not None
        and
        dual["accepted"]
    ):

        return (
            "DUAL",
            "DUAL_VALID_PRIORITY",
            dual,
        )

    s0 = singles.get(
        MARKER_A_ID
    )

    s1 = singles.get(
        MARKER_B_ID
    )

    if (
        s0 is not None
        and
        s1 is not None
    ):

        if (
            s0["score"]
            <= s1["score"]
        ):

            suffix = (
                "CORRECTED"
                if s0.get(
                    "online_corrected",
                    False,
                )
                else "RAW"
            )

            return (
                "SINGLE_0",
                f"BEST_SINGLE_A_{suffix}",
                s0,
            )

        suffix = (
            "CORRECTED"
            if s1.get(
                "online_corrected",
                False,
            )
            else "RAW"
        )

        return (
            "SINGLE_1",
            f"BEST_SINGLE_B_{suffix}",
            s1,
        )

    if s0 is not None:

        suffix = (
            "CORRECTED"
            if s0.get(
                "online_corrected",
                False,
            )
            else "RAW"
        )

        return (
            "SINGLE_0",
            f"SINGLE_A_ONLY_{suffix}",
            s0,
        )

    if s1 is not None:

        suffix = (
            "CORRECTED"
            if s1.get(
                "online_corrected",
                False,
            )
            else "RAW"
        )

        return (
            "SINGLE_1",
            f"SINGLE_B_ONLY_{suffix}",
            s1,
        )

    return (
        "NONE",
        "NO_VALID_ESTIMATOR",
        None,
    )


# ============================================================
# CSV LOGGER
# ============================================================
class DebugLogger:

    def __init__(
        self,
        output_dir,
    ):

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        stamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        self.path = (
            output_dir
            / f"stick1_online_handoff_{stamp}.csv"
        )

        self.file = open(
            self.path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self.fields = [

            "timestamp",
            "frame",

            "actual_source",
            "reason",

            "detected_a",
            "detected_b",

            # ================================================
            # existing estimator information
            # ================================================

            "single_a_valid",
            "single_a_score",
            "single_a_reproj",

            "single_b_valid",
            "single_b_score",
            "single_b_reproj",

            "dual_attempted",
            "dual_valid",
            "dual_reproj_mean",
            "dual_reproj_max",
            "dual_position_jump_mm",
            "dual_rotation_jump_deg",

            # ================================================
            # RAW stick poses
            # ================================================

            "s0_x",
            "s0_y",
            "s0_z",

            "s1_x",
            "s1_y",
            "s1_z",

            "dual_x",
            "dual_y",
            "dual_z",

            # ================================================
            # NEW: S0 IPPE diagnostics
            # ================================================

            "s0_branch",
            "s0_ippe_reproj0",
            "s0_ippe_reproj1",
            "s0_ippe_position_sep_mm",
            "s0_ippe_rotation_sep_deg",

            # ================================================
            # NEW: S1 IPPE diagnostics
            # ================================================

            "s1_branch",
            "s1_ippe_reproj0",
            "s1_ippe_reproj1",
            "s1_ippe_position_sep_mm",
            "s1_ippe_rotation_sep_deg",

            # ================================================
            # NEW:
            # direct Marker0 vs DUAL-implied Marker0
            # CAMERA FRAME
            # ================================================

            "m0_direct_cam_x",
            "m0_direct_cam_y",
            "m0_direct_cam_z",

            "m0_dual_cam_x",
            "m0_dual_cam_y",
            "m0_dual_cam_z",

            "m0_delta_cam_x_mm",
            "m0_delta_cam_y_mm",
            "m0_delta_cam_z_mm",

            "m0_marker_difference_mm",
            "m0_marker_rotation_difference_deg",

            # ================================================
            # NEW:
            # direct Marker1 vs DUAL-implied Marker1
            # CAMERA FRAME
            # ================================================

            "m1_direct_cam_x",
            "m1_direct_cam_y",
            "m1_direct_cam_z",

            "m1_dual_cam_x",
            "m1_dual_cam_y",
            "m1_dual_cam_z",

            "m1_delta_cam_x_mm",
            "m1_delta_cam_y_mm",
            "m1_delta_cam_z_mm",

            "m1_marker_difference_mm",
            "m1_marker_rotation_difference_deg",

            # ================================================
            # corrections
            # ================================================

            "corr0_valid",
            "corr0_age_sec",

            "corr1_valid",
            "corr1_age_sec",

            # ================================================
            # selected pose BEFORE final filter
            # ================================================

            "raw_x",
            "raw_y",
            "raw_z",

            "raw_qw",
            "raw_qx",
            "raw_qy",
            "raw_qz",

            # ================================================
            # final output
            # ================================================

            "filtered_x",
            "filtered_y",
            "filtered_z",

            "filtered_qw",
            "filtered_qx",
            "filtered_qy",
            "filtered_qz",
            
            # Marker0 image geometry
            "m0_edge0_px",
            "m0_edge1_px",
            "m0_edge2_px",
            "m0_edge3_px",
            "m0_edge_mean_px",
            "m0_edge_ratio",
            "m0_area_px2",
            "m0_tilt_deg",

            # Marker1 image geometry
            "m1_edge0_px",
            "m1_edge1_px",
            "m1_edge2_px",
            "m1_edge3_px",
            "m1_edge_mean_px",
            "m1_edge_ratio",
            "m1_area_px2",
            "m1_tilt_deg",
        ]

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self.fields,
        )

        self.writer.writeheader()
        self.file.flush()

        print(
            f"[DEBUG CSV] "
            f"{self.path}"
        )

    def write(
        self,
        row,
    ):

        safe = {}

        for field in self.fields:

            value = row.get(
                field,
                None,
            )

            if value is None:

                safe[field] = ""

            elif isinstance(
                value,
                (
                    np.floating,
                    np.integer,
                ),
            ):

                safe[field] = (
                    value.item()
                )

            else:

                safe[field] = value

        self.writer.writerow(
            safe
        )

        if DEBUG_FLUSH_EVERY_FRAME:

            self.file.flush()

    def close(self):

        if not self.file.closed:

            self.file.flush()
            self.file.close()

            print(
                f"[DEBUG CSV SAVED] "
                f"{self.path}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    np.set_printoptions(
        precision=6,
        suppress=True,
    )
        
    # ============================================================
    # ★ 기존 workspace reference를 읽어서 fresh-start prior로 사용
    # ============================================================
    workspace_orientation_refs = (
        load_workspace_orientation_references(
            WORKSPACE_REFERENCE_CSV
        )
    )
    
    # ============================================================
    # DUAL / TRUSTED-SINGLE reference logger
    # ============================================================

    reference_sample_id = 0

    REFERENCE_LOG_DIR = "/home/lsc/FoundationPose/stick1_debug_logs"
    os.makedirs(REFERENCE_LOG_DIR, exist_ok=True)

    reference_csv_path = os.path.join(
        REFERENCE_LOG_DIR,
        f"stick1_workspace_reference_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    reference_fields = [
        "sample_id",
        "marker_id",
        "source",

        # marker pose in CAMERA
        "marker_cam_x_m",
        "marker_cam_y_m",
        "marker_cam_z_m",

        "marker_cam_qw",
        "marker_cam_qx",
        "marker_cam_qy",
        "marker_cam_qz",

        # corresponding stick pose in BASE
        "stick_base_x_m",
        "stick_base_y_m",
        "stick_base_z_m",

        "stick_base_qw",
        "stick_base_qx",
        "stick_base_qy",
        "stick_base_qz",
    ]

    with open(reference_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=reference_fields,
        )
        writer.writeheader()


    # ============================================================
    # Trusted SINGLE chain state
    #
    # False:
    #   fresh SINGLE이라서 branch가 맞다는 보장이 없음
    #
    # True:
    #   DUAL로 seed된 후 SINGLE이 연속적으로 정상 추적 중
    # ============================================================

    trusted_single = {
        0: False,
        1: False,
    }

    trusted_prev_marker_T = {
        0: None,
        1: None,
    }

    trusted_miss_count = {
        0: 0,
        1: 0,
    }


    # reference 수집용으로는 보수적으로
    TRUSTED_SINGLE_MAX_ROT_STEP_DEG = 35.0

    # 15 Hz 기준 3 frame ≈ 0.2 sec
    # 이것보다 오래 marker가 끊기면 trusted chain 폐기
    TRUSTED_SINGLE_MAX_MISSES = 3


    print()
    print("[REFERENCE CSV]", reference_csv_path)
    print("r = save DUAL / trusted SINGLE reference")
    
    # --------------------------------------------------------
    # Camera extrinsic
    # --------------------------------------------------------

    T_CAMERA_BASE = (
        invert_transform(
            T_BASE_CAMERA
        )
    )

    # --------------------------------------------------------
    # Physical marker geometry
    # --------------------------------------------------------

    (
        T_M0_M1,
        MARKER_TO_STICK,
    ) = (
        build_marker_to_stick_transforms()
    )

    (
        DUAL_OBJECT_A,
        DUAL_OBJECT_B,
    ) = (
        build_dual_object_points(
            MARKER_TO_STICK
        )
    )

    print()
    print("=" * 80)
    print(
        "STICK1 DUAL + SINGLE PNP DIAGNOSTIC"
    )
    print("=" * 80)

    print(
        f"Markers: "
        f"{MARKER_A_ID}, "
        f"{MARKER_B_ID}"
    )

    print(
        f"Camera: "
        f"{WIDTH}x{HEIGHT}@{FPS}"
    )

    print(
        "Camera extrinsic: "
        "HARDCODED T_BASE_CAMERA"
    )

    print()

    print(
        "Pair geometry T_M0_M1 translation [mm]:"
    )

    print(
        T_M0_M1[
            :3,
            3
        ]
        * 1000.0
    )

    print()

    print(
        "Marker0 -> Stick center [mm]:"
    )

    print(
        T_M0_S[
            :3,
            3
        ]
        * 1000.0
    )

    print()

    print(
        "Marker1 -> Stick center [mm]:"
    )

    print(
        T_M1_S[
            :3,
            3
        ]
        * 1000.0
    )

    print()

    print(
        "NEW DIAGNOSTIC:"
    )

    print(
        "  SINGLE marker IPPE pose"
    )

    print(
        "          vs"
    )

    print(
        "  DUAL-implied marker pose"
    )

    print(
        "  -> camera XYZ difference"
    )

    print(
        "  -> camera Z depth error"
    )

    print(
        "  -> rotation difference"
    )

    print(
        "  -> IPPE branch ambiguity"
    )

    print()

    print(
        "q = quit"
    )

    print(
        "x = reset histories + correction"
    )

    print(
        "p = print physical transforms"
    )
    print("r = save DUAL / trusted SINGLE reference")
    print("=" * 80)

    # --------------------------------------------------------
    # ArUco
    # --------------------------------------------------------

    dictionary = (
        cv2.aruco
        .getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
    )

    params = (
        cv2.aruco
        .DetectorParameters()
    )

    params.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_APRILTAG
    )

    detector = (
        cv2.aruco.ArucoDetector(
            dictionary,
            params,
        )
    )

    # --------------------------------------------------------
    # RealSense
    # --------------------------------------------------------

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        WIDTH,
        HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    started = False

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    previous_single_position = {
        MARKER_A_ID:
            None,

        MARKER_B_ID:
            None,
    }

    previous_single_quaternion = {
        MARKER_A_ID:
            None,

        MARKER_B_ID:
            None,
    }

    marker_miss_count = {
        MARKER_A_ID:
            0,

        MARKER_B_ID:
            0,
    }

    single_reject_streak = {
        MARKER_A_ID:
            0,

        MARKER_B_ID:
            0,
    }
    trusted_single[0] = False
    trusted_single[1] = False

    trusted_prev_marker_T[0] = None
    trusted_prev_marker_T[1] = None

    trusted_miss_count[0] = 0
    trusted_miss_count[1] = 0
    
    previous_dual_position = None
    previous_dual_quaternion = None
    # --------------------------------------------------------
    # Generic physical IPPE branch history for M0 / M1
    #
    # Stores CAMERA-frame MARKER orientation for each marker,
    # NOT stick orientation.
    # --------------------------------------------------------

    previous_marker_quaternion_camera = {
        MARKER_A_ID: None,
        MARKER_B_ID: None,
    }
    dual_miss_count = 0
    dual_reject_streak = 0

    correction_state = {
        MARKER_A_ID:
            make_correction_state(),

        MARKER_B_ID:
            make_correction_state(),
    }

    final_filtered_position = None
    final_filtered_quaternion = None

    final_invalid_streak = 0

    logger = DebugLogger(
        DEBUG_LOG_DIR
    )

    frame_index = 0
    last_live_print = 0.0

    try:

        profile = pipeline.start(
            config
        )

        started = True

        # ----------------------------------------------------
        # Intrinsics
        # ----------------------------------------------------

        color_profile = (
            profile
            .get_stream(
                rs.stream.color
            )
            .as_video_stream_profile()
        )

        intr = (
            color_profile
            .get_intrinsics()
        )

        K = np.array(
            [
                [
                    intr.fx,
                    0.0,
                    intr.ppx,
                ],
                [
                    0.0,
                    intr.fy,
                    intr.ppy,
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            ],
            dtype=np.float64,
        )

        dist_coeffs = np.asarray(
            intr.coeffs,
            dtype=np.float64,
        )

        while True:

            frames = (
                pipeline.wait_for_frames()
            )

            color_frame = (
                frames.get_color_frame()
            )

            if not color_frame:

                continue

            now = time.time()

            frame = np.asanyarray(
                color_frame.get_data()
            )

            vis = frame.copy()

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            frame_index += 1

            # =================================================
            # DETECT
            # =================================================

            corners, ids, _ = (
                detector.detectMarkers(
                    gray
                )
            )

            detected = {}

            if ids is not None:

                cv2.aruco.drawDetectedMarkers(
                    vis,
                    corners,
                    ids,
                )

                for corner, marker_id in zip(
                    corners,
                    ids.flatten(),
                ):

                    marker_id = int(
                        marker_id
                    )

                    if (
                        marker_id
                        not in TARGET_IDS
                    ):

                        continue

                    detected[
                        marker_id
                    ] = (
                        np.asarray(
                            corner,
                            dtype=np.float64,
                        )
                        .reshape(4, 2)
                    )

            # =================================================
            # MISS HANDLING
            # =================================================

            for marker_id in (
                MARKER_A_ID,
                MARKER_B_ID,
            ):

                if marker_id in detected:

                    if (
                        marker_miss_count[
                            marker_id
                        ]
                        >=
                        SINGLE_HISTORY_RESET_MISSES
                    ):

                        previous_single_position[
                            marker_id
                        ] = None

                        previous_single_quaternion[
                            marker_id
                        ] = None

                        # Tracking gap: do not keep stale physical-branch history.
                        previous_marker_quaternion_camera[
                            marker_id
                        ] = None

                    marker_miss_count[
                        marker_id
                    ] = 0

                else:

                    marker_miss_count[
                        marker_id
                    ] += 1

            # =================================================
            # SINGLES
            # =================================================

            single_raw = {}
            candidate_diagnostics = {}
            candidate_lists = {}

            for marker_id in (
                MARKER_A_ID,
                MARKER_B_ID,
            ):

                if marker_id not in detected:

                    continue

                candidates = (
                    get_ippe_candidates(
                        detected[
                            marker_id
                        ],
                        K,
                        dist_coeffs,
                    )
                )
                
                candidate_lists[
                    marker_id
                ] = candidates
                
                candidate_diagnostics[
                    marker_id
                ] = (
                    summarize_ippe_candidates(
                        candidates
                    )
                )

                selected = (
                    select_single_candidate(
                        candidates,
                        T_BASE_CAMERA,
                        MARKER_TO_STICK[
                            marker_id
                        ],
                        previous_single_position[
                            marker_id
                        ],
                        previous_single_quaternion[
                            marker_id
                        ],
                    )
                )

                if selected is None:

                    single_reject_streak[
                        marker_id
                    ] += 1

                    if (
                        single_reject_streak[
                            marker_id
                        ]
                        >=
                        SINGLE_REJECT_RESET_FRAMES
                    ):

                        previous_single_position[
                            marker_id
                        ] = None

                        previous_single_quaternion[
                            marker_id
                        ] = None

                        # Repeated rejection: force fresh physical re-seeding.
                        previous_marker_quaternion_camera[
                            marker_id
                        ] = None

                        single_reject_streak[
                            marker_id
                        ] = 0

                        selected = (
                            select_single_candidate(
                                candidates,
                                T_BASE_CAMERA,
                                MARKER_TO_STICK[
                                    marker_id
                                ],
                                None,
                                None,
                            )
                        )

                else:

                    single_reject_streak[
                        marker_id
                    ] = 0

                if selected is None:

                    continue

                previous_single_position[
                    marker_id
                ] = (
                    selected[
                        "position"
                    ].copy()
                )

                previous_single_quaternion[
                    marker_id
                ] = (
                    selected[
                        "quaternion"
                    ].copy()
                )

                single_raw[
                    marker_id
                ] = selected

                _draw_frame_axes_if_visible(
                    vis,
                    K,
                    dist_coeffs,
                    selected[
                        "rvec"
                    ],
                    selected[
                        "tvec"
                    ],
                    MARKER_AXIS_LENGTH_M,
                    2,
                )

            # =================================================
            # DUAL
            # =================================================

            dual = None

            both_detected = (
                MARKER_A_ID
                in detected

                and

                MARKER_B_ID
                in detected
            )

            if both_detected:

                dual_miss_count = 0

                dual = (
                    estimate_dual_pose(
                        detected[
                            MARKER_A_ID
                        ],
                        detected[
                            MARKER_B_ID
                        ],
                        DUAL_OBJECT_A,
                        DUAL_OBJECT_B,
                        K,
                        dist_coeffs,
                        T_BASE_CAMERA,
                        T_CAMERA_BASE,
                        previous_dual_position,
                        previous_dual_quaternion,
                    )
                )

                if (
                    dual is not None
                    and
                    dual["accepted"]
                ):

                    previous_dual_position = (
                        dual[
                            "position"
                        ].copy()
                    )

                    previous_dual_quaternion = (
                        dual[
                            "quaternion"
                        ].copy()
                    )

                    dual_reject_streak = 0

                else:

                    dual_reject_streak += 1

                    if (
                        dual_reject_streak
                        >=
                        DUAL_REJECT_RESET_FRAMES
                    ):

                        previous_dual_position = None
                        previous_dual_quaternion = None

                        dual_reject_streak = 0

            else:

                dual_miss_count += 1

                if (
                    dual_miss_count
                    >=
                    DUAL_HISTORY_RESET_MISSES
                ):

                    previous_dual_position = None
                    previous_dual_quaternion = None

            # =================================================
            # GENERIC PHYSICAL IPPE BRANCH SELECTION (M0 + M1)
            #
            # For each marker:
            #   DUAL > HISTORY > WORKSPACE PRIOR
            #
            # The generic selector replaces the legacy SINGLE
            # result so BOTH markers receive identical branch
            # ambiguity protection.
            # =================================================

            branch_before = {
                MARKER_A_ID: None,
                MARKER_B_ID: None,
            }

            branch_after = {
                MARKER_A_ID: None,
                MARKER_B_ID: None,
            }

            branch_reference_source = {
                MARKER_A_ID: "NONE",
                MARKER_B_ID: "NONE",
            }

            branch_reference_rotation_deg = {
                MARKER_A_ID: None,
                MARKER_B_ID: None,
            }

            for marker_id in (
                MARKER_A_ID,
                MARKER_B_ID,
            ):

                old_single = single_raw.get(
                    marker_id
                )

                if old_single is not None:
                    branch_before[marker_id] = (
                        old_single["branch"]
                    )

                physical = select_physical_branch(
                    marker_id,
                    candidate_lists.get(marker_id),
                    T_BASE_CAMERA,
                    MARKER_TO_STICK[marker_id],
                    T_CAMERA_BASE,
                    dual,
                    previous_marker_quaternion_camera[marker_id],
                    workspace_orientation_refs,
                )

                if physical is None:
                    # If the physical selector deliberately refuses
                    # a detected marker (e.g. fail-safe prior mode),
                    # do not silently keep the legacy branch.
                    if marker_id in candidate_lists:
                        single_raw.pop(marker_id, None)
                    continue

                # Replace legacy SINGLE estimate completely.
                single_raw[marker_id] = physical

                branch_after[marker_id] = (
                    physical["branch"]
                )

                branch_reference_source[marker_id] = (
                    physical.get(
                        "marker_reference_source",
                        "NONE",
                    )
                )

                branch_reference_rotation_deg[marker_id] = (
                    physical.get(
                        "marker_reference_rotation_deg",
                        None,
                    )
                )

                # Store the physically selected CAMERA-frame
                # marker orientation for future marker-only frames.
                previous_marker_quaternion_camera[marker_id] = (
                    physical[
                        "marker_quaternion_camera"
                    ].copy()
                )

                # Keep the legacy stick-pose temporal selector aligned
                # with the physically selected branch too.
                previous_single_position[marker_id] = (
                    physical["position"].copy()
                )

                previous_single_quaternion[marker_id] = (
                    physical["quaternion"].copy()
                )

            image_geom_0 = None
            image_geom_1 = None

            s0_for_geom = single_raw.get(
                MARKER_A_ID
            )

            s1_for_geom = single_raw.get(
                MARKER_B_ID
            )

            if (
                MARKER_A_ID in detected
                and
                s0_for_geom is not None
            ):
                image_geom_0 = (
                    calculate_marker_image_geometry(
                        detected[
                            MARKER_A_ID
                        ],
                        s0_for_geom[
                            "T_CAMERA_MARKER"
                        ],
                    )
                )

            if (
                MARKER_B_ID in detected
                and
                s1_for_geom is not None
            ):
                image_geom_1 = (
                    calculate_marker_image_geometry(
                        detected[
                            MARKER_B_ID
                        ],
                        s1_for_geom[
                            "T_CAMERA_MARKER"
                        ],
                    )
                )
                            
            # ============================================================
            # TRUSTED SINGLE CHAIN UPDATE
            #
            # DUAL이 있으면 그것을 ground-truth seed로 사용.
            # DUAL이 없으면 직전 trusted marker orientation과
            # 현재 SINGLE orientation이 연속적인지 확인.
            # ============================================================

            if dual is not None and dual.get("accepted", False):

                T_BASE_STICK_TRUST = dual["T_BASE_STICK"]

                # DUAL stick pose -> each marker pose
                T_BASE_M0_TRUST = (
                    T_BASE_STICK_TRUST
                    @ invert_transform(T_M0_S)
                )

                T_BASE_M1_TRUST = (
                    T_BASE_STICK_TRUST
                    @ invert_transform(T_M1_S)
                )

                T_CAMERA_M0_TRUST = (
                    T_CAMERA_BASE
                    @ T_BASE_M0_TRUST
                )

                T_CAMERA_M1_TRUST = (
                    T_CAMERA_BASE
                    @ T_BASE_M1_TRUST
                )

                # DUAL이므로 두 marker 모두 trusted seed 획득
                trusted_single[0] = True
                trusted_single[1] = True

                trusted_prev_marker_T[0] = (
                    T_CAMERA_M0_TRUST.copy()
                )

                trusted_prev_marker_T[1] = (
                    T_CAMERA_M1_TRUST.copy()
                )

                trusted_miss_count[0] = 0
                trusted_miss_count[1] = 0


            else:

                # --------------------------------------------------------
                # DUAL 없음:
                # 기존 DUAL seed를 SINGLE이 연속적으로 이어가는지 확인
                # --------------------------------------------------------

                for marker_id in [0, 1]:

                    current_single = single_raw.get(marker_id)

                    single_valid = (
                        current_single is not None
                        and current_single.get("accepted", False)
                        and current_single.get(
                            "T_CAMERA_MARKER",
                            None,
                        ) is not None
                    )

                    if single_valid:

                        trusted_miss_count[marker_id] = 0

                        if (
                            trusted_single[marker_id]
                            and trusted_prev_marker_T[marker_id]
                            is not None
                        ):

                            T_current_marker = (
                                current_single[
                                    "T_CAMERA_MARKER"
                                ]
                            )

                            dR_trusted = (
                                rotation_difference_deg_from_T(
                                    trusted_prev_marker_T[
                                        marker_id
                                    ],
                                    T_current_marker,
                                )
                            )

                            if (
                                dR_trusted
                                <=
                                TRUSTED_SINGLE_MAX_ROT_STEP_DEG
                            ):

                                # 정상적으로 연속 추적 중
                                trusted_prev_marker_T[
                                    marker_id
                                ] = T_current_marker.copy()

                            else:

                                # 갑자기 큰 회전:
                                # branch flip 또는 추적 재획득 가능성
                                trusted_single[
                                    marker_id
                                ] = False

                                trusted_prev_marker_T[
                                    marker_id
                                ] = None

                                print(
                                    f"[TRUST LOST M{marker_id}] "
                                    f"rotation jump "
                                    f"{dR_trusted:.2f} deg"
                                )

                        # 처음부터 SINGLE만 보인 경우:
                        # trusted_single=False 그대로 유지.
                        # 절대 자기 자신으로 seed하면 안 됨.

                    else:

                        trusted_miss_count[
                            marker_id
                        ] += 1

                        if (
                            trusted_miss_count[marker_id]
                            >
                            TRUSTED_SINGLE_MAX_MISSES
                        ):

                            if trusted_single[marker_id]:

                                print(
                                    f"[TRUST LOST M{marker_id}] "
                                    "tracking gap too long"
                                )

                            trusted_single[
                                marker_id
                            ] = False

                            trusted_prev_marker_T[
                                marker_id
                            ] = None
            marker_diag_0 = (
                calculate_marker_diagnostic(
                    single_raw.get(
                        MARKER_A_ID
                    ),
                    dual,
                    MARKER_TO_STICK[
                        MARKER_A_ID
                    ],
                    T_CAMERA_BASE,
                )
            )

            marker_diag_1 = (
                calculate_marker_diagnostic(
                    single_raw.get(
                        MARKER_B_ID
                    ),
                    dual,
                    MARKER_TO_STICK[
                        MARKER_B_ID
                    ],
                    T_CAMERA_BASE,
                )
            )
            candidates_vs_dual = {}

            for marker_id in (
                MARKER_A_ID,
                MARKER_B_ID,
            ):

                candidates_vs_dual[marker_id] = (
                    compare_ippe_candidates_to_dual(
                        candidate_lists.get(marker_id),
                        dual,
                        MARKER_TO_STICK[marker_id],
                        T_CAMERA_BASE,
                    )
                )
            # =================================================
            # UPDATE ONLINE CORRECTION
            # =================================================

            if (
                dual is not None
                and
                dual["accepted"]
            ):

                for marker_id in (
                    MARKER_A_ID,
                    MARKER_B_ID,
                ):

                    if (
                        marker_id
                        in single_raw
                    ):

                        update_online_correction(
                            correction_state[
                                marker_id
                            ],
                            single_raw[
                                marker_id
                            ],
                            dual,
                            now,
                        )

            # =================================================
            # APPLY ONLINE CORRECTION
            # =================================================

            single_corrected = {}

            for marker_id, estimator in (
                single_raw.items()
            ):

                single_corrected[
                    marker_id
                ] = (
                    apply_online_correction(
                        estimator,
                        correction_state[
                            marker_id
                        ],
                    )
                )

            # =================================================
            # SOURCE SELECTION
            # =================================================

            (
                actual_source,
                reason,
                selected_estimator,
            ) = (
                choose_final_source(
                    single_corrected,
                    dual,
                )
            )

            if selected_estimator is None:

                raw_position = None
                raw_quaternion = None

            else:

                raw_position = (
                    selected_estimator[
                        "position"
                    ].copy()
                )

                raw_quaternion = (
                    selected_estimator[
                        "quaternion"
                    ].copy()
                )

            # =================================================
            # FINAL FILTER
            # =================================================

            if raw_position is not None:

                final_invalid_streak = 0

                if (
                    final_filtered_position
                    is None
                ):

                    final_filtered_position = (
                        raw_position.copy()
                    )

                else:

                    final_filtered_position = (
                        POS_ALPHA
                        * raw_position
                        +
                        (1.0 - POS_ALPHA)
                        * final_filtered_position
                    )

                if (
                    final_filtered_quaternion
                    is None
                ):

                    final_filtered_quaternion = (
                        raw_quaternion.copy()
                    )

                else:

                    if (
                        np.dot(
                            final_filtered_quaternion,
                            raw_quaternion,
                        )
                        < 0.0
                    ):

                        raw_quaternion = (
                            -raw_quaternion
                        )

                    final_filtered_quaternion = (
                        quaternion_slerp(
                            final_filtered_quaternion,
                            raw_quaternion,
                            ROT_ALPHA,
                        )
                    )

            else:

                final_invalid_streak += 1

                if (
                    final_invalid_streak
                    >=
                    FINAL_FILTER_RESET_MISSES
                ):

                    final_filtered_position = None
                    final_filtered_quaternion = None

            # =================================================
            # LIVE TERMINAL
            # =================================================

            if (
                now
                - last_live_print
                >=
                LIVE_PRINT_INTERVAL_SEC
            ):

                print()
                print(
                    "============ LIVE ============"
                )

                print(
                    f"SOURCE : {actual_source}"
                )

                print(
                    f"WHY    : {reason}"
                )

                s1 = single_raw.get(
                    MARKER_B_ID
                )

                s1_diag = (
                    candidate_diagnostics.get(
                        MARKER_B_ID
                    )
                )

                if (
                    s1 is not None
                    and
                    s1_diag is not None
                ):

                    print(
                        "S1 IPPE: "
                        f"branch={s1['branch']} "
                        f"| selected repr="
                        f"{s1['reproj']:.3f}px"
                    )

                    print(
                        "         "
                        f"repr0={s1_diag['reproj0']} "
                        f"| repr1={s1_diag['reproj1']}"
                    )

                    if (
                        s1_diag[
                            "position_separation_mm"
                        ]
                        is not None
                    ):

                        print(
                            "         "
                            f"solutions dP="
                            f"{s1_diag['position_separation_mm']:.2f}mm "
                            f"| dR="
                            f"{s1_diag['rotation_separation_deg']:.2f}deg"
                        )

                if marker_diag_1 is not None:

                    d = marker_diag_1

                    print(
                        "M1 DIRECT vs DUAL:"
                    )

                    print(
                        "  dCAM [mm] = "
                        f"["
                        f"{d['delta_cam'][0]*1000:+.2f}, "
                        f"{d['delta_cam'][1]*1000:+.2f}, "
                        f"{d['delta_cam'][2]*1000:+.2f}"
                        f"]"
                    )

                    print(
                        "  3D="
                        f"{d['distance_mm']:.2f}mm "
                        "| dR="
                        f"{d['rotation_difference_deg']:.2f}deg"
                    )

                    print(
                        "  direct CAMERA Z = "
                        f"{d['direct_cam'][2]*1000:.2f}mm"
                    )

                    print(
                        "  dual   CAMERA Z = "
                        f"{d['dual_cam'][2]*1000:.2f}mm"
                    )

                if (
                    final_filtered_position
                    is not None
                ):

                    p = (
                        final_filtered_position
                    )

                    q = (
                        final_filtered_quaternion
                    )

                    print(
                        "XYZ    : "
                        f"{p[0]*1000:+.2f}, "
                        f"{p[1]*1000:+.2f}, "
                        f"{p[2]*1000:+.2f} mm"
                    )

                    print(
                        "QUAT   : "
                        f"{q[0]:+.5f}, "
                        f"{q[1]:+.5f}, "
                        f"{q[2]:+.5f}, "
                        f"{q[3]:+.5f}"
                    )

                print(
                    "=============================="
                )

                last_live_print = now
                for marker_id in (
                    MARKER_A_ID,
                    MARKER_B_ID,
                ):

                    if branch_after[marker_id] is not None:

                        dr_value = (
                            branch_reference_rotation_deg[marker_id]
                        )

                        dr_text = (
                            "NONE"
                            if dr_value is None
                            else f"{dr_value:.2f}deg"
                        )

                        print(
                            f"M{marker_id} PHYSICAL BRANCH:"
                        )

                        print(
                            f"  old branch = {branch_before[marker_id]}"
                        )

                        print(
                            f"  new branch = {branch_after[marker_id]}"
                        )

                        print(
                            "  reference  = "
                            f"{branch_reference_source[marker_id]}"
                        )

                        print(
                            f"  dR(ref)    = {dr_text}"
                        )

                    marker_candidates_vs_dual = (
                        candidates_vs_dual.get(marker_id)
                    )

                    if marker_candidates_vs_dual is not None:

                        print(
                            f"M{marker_id} IPPE CANDIDATES vs DUAL:"
                        )

                        selected_single = single_raw.get(marker_id)

                        for result in marker_candidates_vs_dual:

                            selected_text = ""

                            if (
                                selected_single is not None
                                and result["branch"]
                                == selected_single["branch"]
                            ):
                                selected_text = "  <-- SELECTED"

                            print(
                                f"  branch {result['branch']} "
                                f"| reproj={result['reproj']:.3f}px "
                                f"| dP={result['dp_mm']:.2f}mm "
                                f"| dR={result['dr_deg']:.2f}deg"
                                f"{selected_text}"
                            )

                        
            # =================================================
            # SIMPLE GUI
            # =================================================

            cv2.putText(
                vis,
                f"SOURCE: {actual_source}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if marker_diag_1 is not None:

                cv2.putText(
                    vis,
                    (
                        "M1 direct-dual: "
                        f"{marker_diag_1['distance_mm']:.2f} mm "
                        f"/ "
                        f"{marker_diag_1['rotation_difference_deg']:.2f} deg"
                    ),
                    (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    vis,
                    (
                        "M1 depth diff: "
                        f"{marker_diag_1['delta_cam'][2]*1000:+.2f} mm"
                    ),
                    (20, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # =================================================
            # CSV HELPERS
            # =================================================

            s0 = single_raw.get(
                MARKER_A_ID
            )

            s1 = single_raw.get(
                MARKER_B_ID
            )

            cdiag0 = (
                candidate_diagnostics.get(
                    MARKER_A_ID,
                    {},
                )
            )

            cdiag1 = (
                candidate_diagnostics.get(
                    MARKER_B_ID,
                    {},
                )
            )

            corr0 = correction_state[
                MARKER_A_ID
            ]

            corr1 = correction_state[
                MARKER_B_ID
            ]

            corr0_age = (
                now
                - corr0[
                    "last_update_time"
                ]
                if corr0[
                    "last_update_time"
                ]
                is not None
                else None
            )

            corr1_age = (
                now
                - corr1[
                    "last_update_time"
                ]
                if corr1[
                    "last_update_time"
                ]
                is not None
                else None
            )

            # =================================================
            # CSV WRITE
            # =================================================

            row = {

                "timestamp":
                    now,

                "frame":
                    frame_index,

                "actual_source":
                    actual_source,

                "reason":
                    reason,

                "detected_a":
                    int(
                        MARKER_A_ID
                        in detected
                    ),

                "detected_b":
                    int(
                        MARKER_B_ID
                        in detected
                    ),

                "single_a_valid":
                    int(
                        s0 is not None
                    ),

                "single_a_score":
                    (
                        s0["score"]
                        if s0 is not None
                        else None
                    ),

                "single_a_reproj":
                    (
                        s0["reproj"]
                        if s0 is not None
                        else None
                    ),

                "single_b_valid":
                    int(
                        s1 is not None
                    ),

                "single_b_score":
                    (
                        s1["score"]
                        if s1 is not None
                        else None
                    ),

                "single_b_reproj":
                    (
                        s1["reproj"]
                        if s1 is not None
                        else None
                    ),

                "dual_attempted":
                    int(
                        dual is not None
                    ),

                "dual_valid":
                    int(
                        dual is not None
                        and
                        dual[
                            "accepted"
                        ]
                    ),

                "dual_reproj_mean":
                    (
                        dual[
                            "reproj_mean"
                        ]
                        if dual is not None
                        else None
                    ),

                "dual_reproj_max":
                    (
                        dual[
                            "reproj_max"
                        ]
                        if dual is not None
                        else None
                    ),

                "dual_position_jump_mm":
                    (
                        dual[
                            "position_jump"
                        ]
                        * 1000.0
                        if dual is not None
                        else None
                    ),

                "dual_rotation_jump_deg":
                    (
                        dual[
                            "rotation_jump"
                        ]
                        if dual is not None
                        else None
                    ),

                # ------------------------------------------------
                # raw stick poses
                # ------------------------------------------------

                "s0_x":
                    (
                        s0["position"][0]
                        if s0 is not None
                        else None
                    ),

                "s0_y":
                    (
                        s0["position"][1]
                        if s0 is not None
                        else None
                    ),

                "s0_z":
                    (
                        s0["position"][2]
                        if s0 is not None
                        else None
                    ),

                "s1_x":
                    (
                        s1["position"][0]
                        if s1 is not None
                        else None
                    ),

                "s1_y":
                    (
                        s1["position"][1]
                        if s1 is not None
                        else None
                    ),

                "s1_z":
                    (
                        s1["position"][2]
                        if s1 is not None
                        else None
                    ),

                "dual_x":
                    (
                        dual["position"][0]
                        if (
                            dual is not None
                            and
                            dual["accepted"]
                        )
                        else None
                    ),

                "dual_y":
                    (
                        dual["position"][1]
                        if (
                            dual is not None
                            and
                            dual["accepted"]
                        )
                        else None
                    ),

                "dual_z":
                    (
                        dual["position"][2]
                        if (
                            dual is not None
                            and
                            dual["accepted"]
                        )
                        else None
                    ),

                # ------------------------------------------------
                # S0 IPPE
                # ------------------------------------------------

                "s0_branch":
                    (
                        s0["branch"]
                        if s0 is not None
                        else None
                    ),

                "s0_ippe_reproj0":
                    cdiag0.get(
                        "reproj0"
                    ),

                "s0_ippe_reproj1":
                    cdiag0.get(
                        "reproj1"
                    ),

                "s0_ippe_position_sep_mm":
                    cdiag0.get(
                        "position_separation_mm"
                    ),

                "s0_ippe_rotation_sep_deg":
                    cdiag0.get(
                        "rotation_separation_deg"
                    ),

                # ------------------------------------------------
                # S1 IPPE
                # ------------------------------------------------

                "s1_branch":
                    (
                        s1["branch"]
                        if s1 is not None
                        else None
                    ),

                "s1_ippe_reproj0":
                    cdiag1.get(
                        "reproj0"
                    ),

                "s1_ippe_reproj1":
                    cdiag1.get(
                        "reproj1"
                    ),

                "s1_ippe_position_sep_mm":
                    cdiag1.get(
                        "position_separation_mm"
                    ),

                "s1_ippe_rotation_sep_deg":
                    cdiag1.get(
                        "rotation_separation_deg"
                    ),

                "corr0_valid":
                    int(
                        corr0["T"]
                        is not None
                    ),

                "corr0_age_sec":
                    corr0_age,

                "corr1_valid":
                    int(
                        corr1["T"]
                        is not None
                    ),

                "corr1_age_sec":
                    corr1_age,

                # ------------------------------------------------
                # output raw
                # ------------------------------------------------

                "raw_x":
                    (
                        raw_position[0]
                        if raw_position
                        is not None
                        else None
                    ),

                "raw_y":
                    (
                        raw_position[1]
                        if raw_position
                        is not None
                        else None
                    ),

                "raw_z":
                    (
                        raw_position[2]
                        if raw_position
                        is not None
                        else None
                    ),

                "raw_qw":
                    (
                        raw_quaternion[0]
                        if raw_quaternion
                        is not None
                        else None
                    ),

                "raw_qx":
                    (
                        raw_quaternion[1]
                        if raw_quaternion
                        is not None
                        else None
                    ),

                "raw_qy":
                    (
                        raw_quaternion[2]
                        if raw_quaternion
                        is not None
                        else None
                    ),

                "raw_qz":
                    (
                        raw_quaternion[3]
                        if raw_quaternion
                        is not None
                        else None
                    ),

                # ------------------------------------------------
                # final filtered
                # ------------------------------------------------

                "filtered_x":
                    (
                        final_filtered_position[0]
                        if final_filtered_position
                        is not None
                        else None
                    ),

                "filtered_y":
                    (
                        final_filtered_position[1]
                        if final_filtered_position
                        is not None
                        else None
                    ),

                "filtered_z":
                    (
                        final_filtered_position[2]
                        if final_filtered_position
                        is not None
                        else None
                    ),

                "filtered_qw":
                    (
                        final_filtered_quaternion[0]
                        if final_filtered_quaternion
                        is not None
                        else None
                    ),

                "filtered_qx":
                    (
                        final_filtered_quaternion[1]
                        if final_filtered_quaternion
                        is not None
                        else None
                    ),

                "filtered_qy":
                    (
                        final_filtered_quaternion[2]
                        if final_filtered_quaternion
                        is not None
                        else None
                    ),

                "filtered_qz":
                    (
                        final_filtered_quaternion[3]
                        if final_filtered_quaternion
                        is not None
                        else None
                    ),
            }

            # =================================================
            # MARKER 0 DIAGNOSTIC CSV
            # =================================================

            if marker_diag_0 is not None:

                d = marker_diag_0

                row.update(
                    {
                        "m0_direct_cam_x":
                            d[
                                "direct_cam"
                            ][0],

                        "m0_direct_cam_y":
                            d[
                                "direct_cam"
                            ][1],

                        "m0_direct_cam_z":
                            d[
                                "direct_cam"
                            ][2],

                        "m0_dual_cam_x":
                            d[
                                "dual_cam"
                            ][0],

                        "m0_dual_cam_y":
                            d[
                                "dual_cam"
                            ][1],

                        "m0_dual_cam_z":
                            d[
                                "dual_cam"
                            ][2],

                        "m0_delta_cam_x_mm":
                            d[
                                "delta_cam"
                            ][0]
                            * 1000.0,

                        "m0_delta_cam_y_mm":
                            d[
                                "delta_cam"
                            ][1]
                            * 1000.0,

                        "m0_delta_cam_z_mm":
                            d[
                                "delta_cam"
                            ][2]
                            * 1000.0,

                        "m0_marker_difference_mm":
                            d[
                                "distance_mm"
                            ],

                        "m0_marker_rotation_difference_deg":
                            d[
                                "rotation_difference_deg"
                            ],
                    }
                )

            # =================================================
            # MARKER 1 DIAGNOSTIC CSV
            # =================================================

            if marker_diag_1 is not None:

                d = marker_diag_1

                row.update(
                    {
                        "m1_direct_cam_x":
                            d[
                                "direct_cam"
                            ][0],

                        "m1_direct_cam_y":
                            d[
                                "direct_cam"
                            ][1],

                        "m1_direct_cam_z":
                            d[
                                "direct_cam"
                            ][2],

                        "m1_dual_cam_x":
                            d[
                                "dual_cam"
                            ][0],

                        "m1_dual_cam_y":
                            d[
                                "dual_cam"
                            ][1],

                        "m1_dual_cam_z":
                            d[
                                "dual_cam"
                            ][2],

                        "m1_delta_cam_x_mm":
                            d[
                                "delta_cam"
                            ][0]
                            * 1000.0,

                        "m1_delta_cam_y_mm":
                            d[
                                "delta_cam"
                            ][1]
                            * 1000.0,

                        "m1_delta_cam_z_mm":
                            d[
                                "delta_cam"
                            ][2]
                            * 1000.0,

                        "m1_marker_difference_mm":
                            d[
                                "distance_mm"
                            ],

                        "m1_marker_rotation_difference_deg":
                            d[
                                "rotation_difference_deg"
                            ],
                    }
                )
            # =================================================
            # IMAGE GEOMETRY CSV
            # =================================================

            if image_geom_0 is not None:

                row.update(
                    {
                        "m0_edge0_px":
                            image_geom_0["edge0_px"],

                        "m0_edge1_px":
                            image_geom_0["edge1_px"],

                        "m0_edge2_px":
                            image_geom_0["edge2_px"],

                        "m0_edge3_px":
                            image_geom_0["edge3_px"],

                        "m0_edge_mean_px":
                            image_geom_0["edge_mean_px"],

                        "m0_edge_ratio":
                            image_geom_0["edge_ratio"],

                        "m0_area_px2":
                            image_geom_0["area_px2"],

                        "m0_tilt_deg":
                            image_geom_0["tilt_deg"],
                    }
                )


            if image_geom_1 is not None:

                row.update(
                    {
                        "m1_edge0_px":
                            image_geom_1["edge0_px"],

                        "m1_edge1_px":
                            image_geom_1["edge1_px"],

                        "m1_edge2_px":
                            image_geom_1["edge2_px"],

                        "m1_edge3_px":
                            image_geom_1["edge3_px"],

                        "m1_edge_mean_px":
                            image_geom_1["edge_mean_px"],

                        "m1_edge_ratio":
                            image_geom_1["edge_ratio"],

                        "m1_area_px2":
                            image_geom_1["area_px2"],

                        "m1_tilt_deg":
                            image_geom_1["tilt_deg"],
                    }
                )


            logger.write(row)

            # =================================================
            # GUI
            # =================================================

            cv2.imshow(
                "Stick1 PnP Diagnostic",
                vis,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                break

            elif key == ord("p"):

                print()
                print(
                    "T_M0_S:"
                )

                print(
                    T_M0_S
                )

                print()
                print(
                    "T_M1_S:"
                )

                print(
                    T_M1_S
                )

                print()
                print(
                    "T_M0_M1:"
                )

                print(
                    T_M0_M1
                )

            elif key == ord("x"):
            

                previous_single_position = {
                    MARKER_A_ID:
                        None,

                    MARKER_B_ID:
                        None,
                }

                previous_single_quaternion = {
                    MARKER_A_ID:
                        None,

                    MARKER_B_ID:
                        None,
                }

                marker_miss_count = {
                    MARKER_A_ID:
                        0,

                    MARKER_B_ID:
                        0,
                }

                single_reject_streak = {
                    MARKER_A_ID:
                        0,

                    MARKER_B_ID:
                        0,
                }

                previous_dual_position = None
                previous_dual_quaternion = None

                previous_marker_quaternion_camera = {
                    MARKER_A_ID: None,
                    MARKER_B_ID: None,
                }

                trusted_single[MARKER_A_ID] = False
                trusted_single[MARKER_B_ID] = False

                trusted_prev_marker_T[MARKER_A_ID] = None
                trusted_prev_marker_T[MARKER_B_ID] = None

                trusted_miss_count[MARKER_A_ID] = 0
                trusted_miss_count[MARKER_B_ID] = 0

                dual_miss_count = 0
                dual_reject_streak = 0

                correction_state = {
                    MARKER_A_ID:
                        make_correction_state(),

                    MARKER_B_ID:
                        make_correction_state(),
                }

                final_filtered_position = None
                final_filtered_quaternion = None
                final_invalid_streak = 0

                print(
                    "[FULL RESET]"
                )
            elif key == ord("r"):

                rows_to_save = []

                # ========================================================
                # CASE 1:
                # Current DUAL valid
                #
                # → M0/M1 둘 다 DUAL-implied reference 저장
                # ========================================================

                if (
                    dual is not None
                    and dual.get("accepted", False)
                ):

                    T_BASE_STICK_REF = (
                        dual["T_BASE_STICK"].copy()
                    )

                    q_base_stick = (
                        rotation_matrix_to_quaternion_wxyz(
                            T_BASE_STICK_REF[:3, :3]
                        )
                    )

                    p_base_stick = (
                        T_BASE_STICK_REF[:3, 3]
                    )

                    for marker_id, T_MARKER_STICK in [
                        (0, T_M0_S),
                        (1, T_M1_S),
                    ]:

                        T_BASE_MARKER_REF = (
                            T_BASE_STICK_REF
                            @ invert_transform(
                                T_MARKER_STICK
                            )
                        )

                        T_CAMERA_MARKER_REF = (
                            T_CAMERA_BASE
                            @ T_BASE_MARKER_REF
                        )

                        p_cam_marker = (
                            T_CAMERA_MARKER_REF[:3, 3]
                        )

                        q_cam_marker = (
                            rotation_matrix_to_quaternion_wxyz(
                                T_CAMERA_MARKER_REF[
                                    :3,
                                    :3,
                                ]
                            )
                        )

                        rows_to_save.append(
                            {
                                "marker_id": marker_id,
                                "source": "DUAL",

                                "marker_cam_x_m":
                                    p_cam_marker[0],
                                "marker_cam_y_m":
                                    p_cam_marker[1],
                                "marker_cam_z_m":
                                    p_cam_marker[2],

                                "marker_cam_qw":
                                    q_cam_marker[0],
                                "marker_cam_qx":
                                    q_cam_marker[1],
                                "marker_cam_qy":
                                    q_cam_marker[2],
                                "marker_cam_qz":
                                    q_cam_marker[3],

                                "stick_base_x_m":
                                    p_base_stick[0],
                                "stick_base_y_m":
                                    p_base_stick[1],
                                "stick_base_z_m":
                                    p_base_stick[2],

                                "stick_base_qw":
                                    q_base_stick[0],
                                "stick_base_qx":
                                    q_base_stick[1],
                                "stick_base_qy":
                                    q_base_stick[2],
                                "stick_base_qz":
                                    q_base_stick[3],
                            }
                        )


                # ========================================================
                # CASE 2:
                # DUAL 없음
                #
                # trusted chain이 살아있는 SINGLE marker들을 저장
                # ========================================================

                else:

                    for marker_id in [0, 1]:

                        # 이 marker가 DUAL에서 seed된 trusted chain인지
                        if not trusted_single[marker_id]:
                            continue

                        single_ref = single_raw.get(marker_id)

                        # 현재 frame에서 SINGLE pose가 실제로 유효한지
                        if (
                            single_ref is None
                            or not single_ref.get("accepted", False)
                            or single_ref.get("T_CAMERA_MARKER", None) is None
                            or single_ref.get("T_BASE_STICK", None) is None
                        ):
                            continue

                        T_CAMERA_MARKER_REF = (
                            single_ref["T_CAMERA_MARKER"].copy()
                        )

                        T_BASE_STICK_REF = (
                            single_ref["T_BASE_STICK"].copy()
                        )

                        p_cam_marker = (
                            T_CAMERA_MARKER_REF[:3, 3]
                        )

                        q_cam_marker = (
                            rotation_matrix_to_quaternion_wxyz(
                                T_CAMERA_MARKER_REF[:3, :3]
                            )
                        )

                        p_base_stick = (
                            T_BASE_STICK_REF[:3, 3]
                        )

                        q_base_stick = (
                            rotation_matrix_to_quaternion_wxyz(
                                T_BASE_STICK_REF[:3, :3]
                            )
                        )

                        rows_to_save.append(
                            {
                                "marker_id": marker_id,

                                "source":
                                    "TRUSTED_SINGLE_HISTORY",

                                "marker_cam_x_m":
                                    p_cam_marker[0],
                                "marker_cam_y_m":
                                    p_cam_marker[1],
                                "marker_cam_z_m":
                                    p_cam_marker[2],

                                "marker_cam_qw":
                                    q_cam_marker[0],
                                "marker_cam_qx":
                                    q_cam_marker[1],
                                "marker_cam_qy":
                                    q_cam_marker[2],
                                "marker_cam_qz":
                                    q_cam_marker[3],

                                "stick_base_x_m":
                                    p_base_stick[0],
                                "stick_base_y_m":
                                    p_base_stick[1],
                                "stick_base_z_m":
                                    p_base_stick[2],

                                "stick_base_qw":
                                    q_base_stick[0],
                                "stick_base_qx":
                                    q_base_stick[1],
                                "stick_base_qy":
                                    q_base_stick[2],
                                "stick_base_qz":
                                    q_base_stick[3],
                            }
                        )
                # ========================================================
                # SAVE
                # ========================================================

                if not rows_to_save:

                    print()
                    print("[REFERENCE NOT SAVED]")
                    print(
                        "Trusted M0    : "
                        f"{trusted_single[0]}"
                    )

                    print(
                        "Trusted M1    : "
                        f"{trusted_single[1]}"
                    )

                    print(
                        "Need DUAL or a continuous "
                        "DUAL-seeded trusted SINGLE."
                    )
                    print()

                else:

                    with open(
                        reference_csv_path,
                        "a",
                        newline="",
                    ) as f:

                        writer = csv.DictWriter(
                            f,
                            fieldnames=reference_fields,
                        )

                        for row in rows_to_save:

                            reference_sample_id += 1

                            row["sample_id"] = (
                                reference_sample_id
                            )

                            writer.writerow(row)

                            print()
                            print("=" * 80)

                            print(
                                f"[REFERENCE SAVED "
                                f"#{reference_sample_id}]"
                            )

                            print(
                                f"MARKER : "
                                f"{row['marker_id']}"
                            )

                            print(
                                f"SOURCE : "
                                f"{row['source']}"
                            )

                            print(
                                "CAM XYZ [mm] : "
                                f"{row['marker_cam_x_m'] * 1000:+.2f}, "
                                f"{row['marker_cam_y_m'] * 1000:+.2f}, "
                                f"{row['marker_cam_z_m'] * 1000:+.2f}"
                            )

                            print(
                                "CAM QUAT wxyz: "
                                f"{row['marker_cam_qw']:+.6f}, "
                                f"{row['marker_cam_qx']:+.6f}, "
                                f"{row['marker_cam_qy']:+.6f}, "
                                f"{row['marker_cam_qz']:+.6f}"
                            )

                            print("=" * 80)

                    print()
                    print(
                        "CSV:",
                        reference_csv_path,
                    )
                    print()
    finally:

        logger.close()

        if started:

            pipeline.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
