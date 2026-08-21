"""Evaluate fixed-angle +Palm-Y Camera2 height candidates in rendered RGB."""

from __future__ import annotations

import argparse

import cv2
import mujoco
import numpy as np

from Deploy.backends.mujoco_wuji import MujocoWujiHand
from Deploy.vision.sim_aruco import (R_WORLD_BASE, T_BASE_PALM)
from Deploy.common.stick_pose import (
    quaternion_multiply_wxyz,
    rotation_matrix_to_quaternion_wxyz,
)


ASSEMBLY_PALM_Z_DEG = (-90, -75, -60, -45, -30, -15, 0)
DEFAULT_HEIGHTS_BASE_X_M = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
CAMERA2_FIXED_DOWN_ANGLE_DEG = 45.0
# Nominal tail/grasp-region aim point.  This is a mount-search target, not a
# policy frame or calibration constant.
CAMERA2_TARGET_PALM_M = np.asarray([0.025, 0.0, 0.035], dtype=np.float64)


def _detector():
    parameters = cv2.aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.005
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def _camera_rotation_base(position_b: np.ndarray, target_b: np.ndarray) -> np.ndarray:
    """Return OpenCV camera axes in Base: +X right, +Y down, +Z forward."""

    forward = target_b - position_b
    forward /= np.linalg.norm(forward)
    base_up = np.asarray([1.0, 0.0, 0.0])
    right = np.cross(forward, base_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack((right, down, forward))


def _fixed_down_rotation_base(down_angle_deg: float) -> np.ndarray:
    """OpenCV camera axes for a +Base-Y mount looking toward -Base-Y."""

    angle = np.deg2rad(down_angle_deg)
    forward = np.asarray([-np.sin(angle), -np.cos(angle), 0.0])
    base_up = np.asarray([1.0, 0.0, 0.0])
    right = np.cross(forward, base_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack((right, down, forward))


def _set_camera2_mount(
    hand,
    height_base_x_m: float,
    down_angle_deg: float,
    side_y_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(
        hand.model, mujoco.mjtObj.mjOBJ_BODY, "d435_camera2_candidate_mount"
    )
    target_b = T_BASE_PALM[:3, 3] + CAMERA2_TARGET_PALM_M
    if side_y_m is None:
        height_delta = height_base_x_m - target_b[0]
        if height_delta <= 0:
            raise ValueError("Camera2 height must be above its nominal target.")
        if abs(down_angle_deg) < 1e-9:
            raise ValueError("A horizontal camera requires --side-y.")
        side_distance = height_delta / np.tan(np.deg2rad(down_angle_deg))
        position_b = target_b + np.asarray([height_delta, side_distance, 0.0])
        rotation_b_cv = _camera_rotation_base(position_b, target_b)
    else:
        position_b = np.asarray([height_base_x_m, side_y_m, target_b[2]])
        rotation_b_cv = _fixed_down_rotation_base(down_angle_deg)
    rotation_w_mj = R_WORLD_BASE @ rotation_b_cv @ np.diag([1.0, -1.0, -1.0])
    hand.model.body_pos[body_id] = R_WORLD_BASE @ position_b
    hand.model.body_quat[body_id] = rotation_matrix_to_quaternion_wxyz(rotation_w_mj)
    return position_b, rotation_b_cv


def _assembly_snapshot(hand):
    palm_id = mujoco.mj_name2id(
        hand.model, mujoco.mjtObj.mjOBJ_BODY, "right_palm_link"
    )
    stick_joint_ids = [
        mujoco.mj_name2id(hand.model, mujoco.mjtObj.mjOBJ_JOINT, f"stick{i}_free")
        for i in (1, 2)
    ]
    stick_qpos = []
    for joint_id in stick_joint_ids:
        address = int(hand.model.jnt_qposadr[joint_id])
        stick_qpos.append(hand.data.qpos[address:address + 7].copy())
    return (
        palm_id,
        hand.model.body_pos[palm_id].copy(),
        hand.model.body_quat[palm_id].copy(),
        stick_joint_ids,
        stick_qpos,
    )


def _set_assembly_rotation(hand, snapshot, angle_deg: float) -> None:
    palm_id, palm_position_w, palm_quaternion_w, joint_ids, stick_qpos = snapshot
    axis_w = R_WORLD_BASE @ np.asarray([0.0, 0.0, 1.0])
    angle = np.deg2rad(angle_deg)
    skew = np.asarray(
        [[0.0, -axis_w[2], axis_w[1]],
         [axis_w[2], 0.0, -axis_w[0]],
         [-axis_w[1], axis_w[0], 0.0]]
    )
    rotation_w = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    rotation_quaternion = rotation_matrix_to_quaternion_wxyz(rotation_w)
    hand.model.body_pos[palm_id] = palm_position_w
    hand.model.body_quat[palm_id] = quaternion_multiply_wxyz(
        rotation_quaternion, palm_quaternion_w
    )
    for joint_id, source in zip(joint_ids, stick_qpos):
        address = int(hand.model.jnt_qposadr[joint_id])
        hand.data.qpos[address:address + 3] = (
            palm_position_w + rotation_w @ (source[:3] - palm_position_w)
        )
        hand.data.qpos[address + 3:address + 7] = quaternion_multiply_wxyz(
            rotation_quaternion, source[3:]
        )
    mujoco.mj_forward(hand.model, hand.data)


def _detected_ids(renderer, hand, camera_name: str, detector) -> set[int]:
    renderer.update_scene(hand.data, camera=camera_name)
    rgb = np.asarray(renderer.render(), dtype=np.uint8)
    _, ids, _ = detector.detectMarkers(rgb)
    return set() if ids is None else {int(value) for value in ids.ravel()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=DEFAULT_HEIGHTS_BASE_X_M,
        help="Candidate optical-center heights in Base +X meters.",
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=ASSEMBLY_PALM_Z_DEG,
        help="Palm-Z assembly angles in degrees.",
    )
    parser.add_argument(
        "--down-angle",
        type=float,
        default=CAMERA2_FIXED_DOWN_ANGLE_DEG,
        help="Fixed downward viewing angle in degrees from horizontal.",
    )
    parser.add_argument(
        "--side-y",
        type=float,
        nargs="+",
        default=None,
        help="Optional fixed Base +Y optical-center positions; required at 0 deg.",
    )
    args = parser.parse_args()

    hand = MujocoWujiHand()
    snapshot = _assembly_snapshot(hand)
    detector = _detector()
    renderer = mujoco.Renderer(hand.model, height=720, width=1280)
    sample_count = len(args.angles)
    print(
        "height_m  side_y_m  both_ok  stick1_ok  stick2_ok  failed_angles "
        f"(samples={sample_count})"
    )
    try:
        side_candidates = args.side_y if args.side_y is not None else (None,)
        for height in args.heights:
            for side_y in side_candidates:
                position_b, _ = _set_camera2_mount(
                    hand, height, args.down_angle, side_y
                )
                both_count = stick1_count = stick2_count = 0
                failed = []
                for angle in args.angles:
                    _set_assembly_rotation(hand, snapshot, angle)
                    camera1 = _detected_ids(renderer, hand, "d435_rgb", detector)
                    camera2 = _detected_ids(
                        renderer, hand, "d435_camera2_candidate", detector
                    )
                    combined = camera1 | camera2
                    stick1_ok = bool(combined & {0, 1})
                    stick2_ok = bool(combined & {2, 3})
                    stick1_count += int(stick1_ok)
                    stick2_count += int(stick2_ok)
                    both_count += int(stick1_ok and stick2_ok)
                    if not (stick1_ok and stick2_ok):
                        failed.append(angle)
                print(
                    f"{height:8.3f}  {position_b[1]:8.3f}  {both_count:9d}  "
                    f"{stick1_count:11d}  {stick2_count:11d}  {failed}"
                )
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
