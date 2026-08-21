"""Generate near-current pregrasp IK candidates without running Isaac physics.

The four functional fingertip pads are displaced along the outward normal of
their assigned reset stick.  Thumb displacement is half the requested nominal
clearance; the little finger is unchanged because it has no direct functional
stick contact in the active topology.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from Deploy.contract.fingertip_fk import (
    POLICY_TIP_FRAME_URDF,
    WujiHand1FingertipFK,
    _axis_angle_transform,
)
from Deploy.contract.policy_contract import REAL_HAND_FACTORY_LIMITS
from Deploy.contract.isaac_reset import ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ
from Deploy.stick_pose import quaternion_to_rotation_matrix_wxyz


Q0 = np.asarray(
    [
        0.5377866626, 0.8436813951, 0.0377136655, -0.0000001810,
        0.7017297745, 0.0553143807, 1.1822255850, 1.4215219021,
        0.4649881423, -0.0292181600, 1.6272000000, 1.1032750607,
        0.9151425958, -0.0129909236, 1.3248542547, 0.3182539344,
        0.7154092789, 0.0788998753, 1.6272000000, 0.2546040118,
    ],
    dtype=np.float64,
)

PAD_OFFSETS = {
    0: np.asarray([0.0, 0.0, -0.0090]),
    1: np.asarray([0.0, 0.0, -0.0140]),
    2: np.asarray([0.0, 0.0, -0.0140]),
    3: np.asarray([0.0, 0.0, -0.0140]),
}
STICK_FOR_FINGER = {0: 0, 1: 0, 2: 0, 3: 1}
STICK_HALF = np.asarray([0.0035, 0.0900, 0.0035])


def tip_transform(fk: WujiHand1FingertipFK, finger: int, q4: np.ndarray) -> np.ndarray:
    q = Q0.copy()
    q[4 * finger : 4 * finger + 4] = q4
    transform = np.eye(4)
    for joint in fk._chain_to_palm(fk.tip_link_names[finger]):
        transform = transform @ joint.origin
        if joint.joint_type == "revolute":
            transform = transform @ _axis_angle_transform(joint.axis, q[joint.policy_index])
    return transform


def pad_position(fk: WujiHand1FingertipFK, finger: int, q4: np.ndarray) -> np.ndarray:
    transform = tip_transform(fk, finger, q4)
    return transform[:3, 3] + transform[:3, :3] @ PAD_OFFSETS[finger]


def outward_normal(point_p: np.ndarray, stick_index: int) -> np.ndarray:
    pose = ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ[stick_index]
    rotation = quaternion_to_rotation_matrix_wxyz(pose[3:])
    local = rotation.T @ (point_p - pose[:3])
    # The nearest rectangular-prism face gives the local outward direction.
    face_margin = np.abs(np.abs(local) - STICK_HALF)
    axis = int(np.argmin(face_margin))
    normal_local = np.zeros(3)
    normal_local[axis] = 1.0 if local[axis] >= 0.0 else -1.0
    return rotation @ normal_local


def numerical_jacobian(fk, finger, q4, eps=1.0e-6):
    base = pad_position(fk, finger, q4)
    jac = np.empty((3, 4))
    for column in range(4):
        shifted = q4.copy()
        shifted[column] += eps
        jac[:, column] = (pad_position(fk, finger, shifted) - base) / eps
    return jac


def solve_family(fk, finger: int, displacement_m: float) -> list[np.ndarray]:
    start = Q0[4 * finger : 4 * finger + 4]
    p0 = pad_position(fk, finger, start)
    target = p0 + displacement_m * outward_normal(p0, STICK_FOR_FINGER[finger])
    bounds = REAL_HAND_FACTORY_LIMITS[4 * finger : 4 * finger + 4].T.astype(np.float64)

    def correct(seed, preference, preference_weight):
        def residual(q4):
            position = (pad_position(fk, finger, q4) - target) / 1.0e-4
            regularizer = preference_weight * (q4 - preference)
            return np.concatenate([position, regularizer])

        return least_squares(
            residual, seed, bounds=bounds, xtol=1.0e-13, ftol=1.0e-13,
            gtol=1.0e-13, max_nfev=3000,
        ).x

    centre = correct(start, start, 0.02)
    _, _, vh = np.linalg.svd(numerical_jacobian(fk, finger, centre))
    null = vh[-1]
    null /= np.linalg.norm(null)
    variants = [centre]
    for sign in (-1.0, 1.0):
        preferred = np.clip(centre + sign * 0.04 * null, bounds[0], bounds[1])
        variants.append(correct(preferred, preferred, 0.08))
    return variants


def main() -> None:
    fk = WujiHand1FingertipFK(POLICY_TIP_FRAME_URDF)
    names = ("thumb", "index", "middle", "ring")
    for clearance_mm in (1, 2, 3, 4):
        families = []
        for finger in range(4):
            scale = 0.5 if finger == 0 else 1.0
            families.append(solve_family(fk, finger, clearance_mm * scale * 1.0e-3))
        print(f"CLEARANCE {clearance_mm} mm (thumb {0.5 * clearance_mm:.1f} mm)")
        for variant in range(3):
            q = Q0.copy()
            achieved = []
            for finger, family in enumerate(families):
                q4 = family[variant]
                q[4 * finger : 4 * finger + 4] = q4
                p0 = pad_position(fk, finger, Q0[4 * finger : 4 * finger + 4])
                normal = outward_normal(p0, STICK_FOR_FINGER[finger])
                achieved.append(1000.0 * np.dot(pad_position(fk, finger, q4) - p0, normal))
            delta = q - Q0
            print(
                f"  candidate {variant + 1}: max|dq|={np.max(np.abs(delta)):.6f} rad, "
                f"||dq||={np.linalg.norm(delta):.6f} rad, "
                + " ".join(f"{name}={value:.4f}mm" for name, value in zip(names, achieved))
            )
            print("    (" + ", ".join(f"{value:.10f}" for value in q) + ",)")


if __name__ == "__main__":
    main()
