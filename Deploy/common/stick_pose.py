"""Canonical StickPose7D transforms matching the active Isaac observation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


STICK_POSE_DIM = 7
SQRT_HALF = np.sqrt(0.5)
SQUARE_STICK_Y_SYMMETRIES_WXYZ = np.asarray(
    [
        (1.0, 0.0, 0.0, 0.0),
        (SQRT_HALF, 0.0, SQRT_HALF, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (SQRT_HALF, 0.0, -SQRT_HALF, 0.0),
    ],
    dtype=np.float64,
)


def normalize_quaternion_wxyz(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    result = np.asarray(q, dtype=np.float64)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("Quaternion must be a finite wxyz vector.")
    norm = float(np.linalg.norm(result))
    if norm < 1.0e-12:
        raise ValueError("Quaternion norm is zero.")
    return result / norm


def quaternion_multiply_wxyz(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.float64]:
    aw, ax, ay, az = normalize_quaternion_wxyz(a)
    bw, bx, by, bz = normalize_quaternion_wxyz(b)
    return np.asarray(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def canonicalize_square_stick_quaternion(
    quaternion_wxyz: npt.ArrayLike,
    reference_wxyz: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """Apply Isaac's four local-+Y roll symmetries, then enforce ``w >= 0``."""

    q = normalize_quaternion_wxyz(quaternion_wxyz)
    reference = normalize_quaternion_wxyz(reference_wxyz)
    candidates = np.stack(
        [quaternion_multiply_wxyz(q, symmetry) for symmetry in SQUARE_STICK_Y_SYMMETRIES_WXYZ]
    )
    selected = candidates[int(np.argmax(np.abs(candidates @ reference)))].copy()
    if selected[0] < 0.0:
        selected *= -1.0
    return selected.astype(np.float32)


def pose_matrix_to_xyz_wxyz(transform: npt.ArrayLike) -> npt.NDArray[np.float32]:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Pose transform must be finite 4x4.")
    return np.concatenate((matrix[:3, 3], rotation_matrix_to_quaternion_wxyz(matrix[:3, :3]))).astype(
        np.float32
    )


def rotation_matrix_to_quaternion_wxyz(rotation: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("Rotation must be finite 3x3.")
    # Numerically stable eigenvector construction; sign is canonicalized later.
    k = np.asarray(
        [
            [r[0, 0] - r[1, 1] - r[2, 2], r[0, 1] + r[1, 0], r[0, 2] + r[2, 0], r[2, 1] - r[1, 2]],
            [r[0, 1] + r[1, 0], r[1, 1] - r[0, 0] - r[2, 2], r[1, 2] + r[2, 1], r[0, 2] - r[2, 0]],
            [r[0, 2] + r[2, 0], r[1, 2] + r[2, 1], r[2, 2] - r[0, 0] - r[1, 1], r[1, 0] - r[0, 1]],
            [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], r[0, 0] + r[1, 1] + r[2, 2]],
        ]
    ) / 3.0
    values, vectors = np.linalg.eigh(k)
    xyzw = vectors[:, int(np.argmax(values))]
    return normalize_quaternion_wxyz(xyzw[[3, 0, 1, 2]])


def quaternion_to_rotation_matrix_wxyz(q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    w, x, y, z = normalize_quaternion_wxyz(q)
    return np.asarray(
        [[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
         [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
         [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]],
        dtype=np.float64,
    )


def quaternion_geodesic_error_deg(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    dot = abs(float(np.dot(normalize_quaternion_wxyz(a), normalize_quaternion_wxyz(b))))
    return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))
