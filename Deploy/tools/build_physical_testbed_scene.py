"""Generate approximate fixture geometry from named scene-contract values."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from Deploy.vision.sim_aruco import (BASE_PLATE_DEPTH_Z_M, BASE_PLATE_NEGATIVE_Z_EDGE_M, BASE_PLATE_THICKNESS_M, BASE_PLATE_WIDTH_Y_M, CAMERA_BRACKET_RADIUS_APPROX_M, CAMERA_SUPPORT_HEIGHT_APPROX_M, CAMERA_SUPPORT_CENTER_Y_M, CAMERA_SUPPORT_CENTER_Z_M, CAMERA_SUPPORT_FLOOR_X_M, CAMERA_SUPPORT_PROFILE_SIZE_M, DEBUG_FRAME_AXIS_LENGTH_M, HAND_PROFILE_CENTER_Y_M, HAND_PROFILE_CENTER_Z_M, HAND_PROFILE_HEIGHT_APPROX_M, HAND_PROFILE_SIZE_M, R_WORLD_BASE, T_BASE_CAMERA, T_BASE_PALM)


OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "assets/wuji_description/hand/body/mjcf/physical_testbed.inc.xml"
)


def _world(point_base: np.ndarray) -> np.ndarray:
    return R_WORLD_BASE @ np.asarray(point_base, dtype=np.float64)


def _v(values) -> str:
    return " ".join(f"{float(value):.9f}" for value in values)


def _axis_geoms(prefix: str, origin_b: np.ndarray, rotation_b_f: np.ndarray) -> list[str]:
    colors = ((1, 0, 0, 1), (0, 1, 0, 1), (0, 0.35, 1, 1))
    names = ("x", "y", "z")
    lines: list[str] = []
    origin_w = _world(origin_b)
    for index, (name, color) in enumerate(zip(names, colors)):
        endpoint_w = _world(
            origin_b + DEBUG_FRAME_AXIS_LENGTH_M * rotation_b_f[:, index]
        )
        lines.append(
            f'  <geom name="{prefix}_axis_{name}" type="capsule" '
            f'fromto="{_v(origin_w)} {_v(endpoint_w)}" size="0.002" '
            f'rgba="{_v(color)}" contype="0" conaffinity="0" mass="0" group="4"/>'
        )
    return lines


def main() -> None:
    floor_center_b = np.asarray(
        [
            -BASE_PLATE_THICKNESS_M / 2.0,
            0.0,
            BASE_PLATE_NEGATIVE_Z_EDGE_M + BASE_PLATE_DEPTH_Z_M / 2.0,
        ]
    )
    floor_halfsize_w = np.asarray(
        [
            BASE_PLATE_DEPTH_Z_M / 2.0,
            BASE_PLATE_WIDTH_Y_M / 2.0,
            BASE_PLATE_THICKNESS_M / 2.0,
        ]
    )

    hand_profile_center_b = np.asarray(
        [
            HAND_PROFILE_HEIGHT_APPROX_M / 2.0,
            HAND_PROFILE_CENTER_Y_M,
            HAND_PROFILE_CENTER_Z_M,
        ]
    )
    hand_profile_halfsize_w = np.asarray(
        [HAND_PROFILE_SIZE_M / 2.0, HAND_PROFILE_SIZE_M / 2.0,
         HAND_PROFILE_HEIGHT_APPROX_M / 2.0]
    )

    camera_position_b = T_BASE_CAMERA[:3, 3]
    camera_support_center_b = np.asarray(
        [CAMERA_SUPPORT_FLOOR_X_M + CAMERA_SUPPORT_HEIGHT_APPROX_M / 2.0,
         CAMERA_SUPPORT_CENTER_Y_M, CAMERA_SUPPORT_CENTER_Z_M]
    )
    camera_support_halfsize_w = np.asarray(
        [CAMERA_SUPPORT_PROFILE_SIZE_M / 2.0,
         CAMERA_SUPPORT_PROFILE_SIZE_M / 2.0,
         CAMERA_SUPPORT_HEIGHT_APPROX_M / 2.0]
    )
    bracket_support_b = np.asarray(
        [camera_position_b[0], CAMERA_SUPPORT_CENTER_Y_M, CAMERA_SUPPORT_CENTER_Z_M]
    )

    lines = [
        "<mujocoinclude>",
        "  <!-- APPROXIMATE/TEMPORARY support geometry; calibrated frames win. -->",
        f'  <geom name="base_plate_collision" type="box" pos="{_v(_world(floor_center_b))}" '
        f'size="{_v(floor_halfsize_w)}" rgba="0.56 0.59 0.62 1" '
        'contype="1" conaffinity="1" friction="0.8 0.02 0.001"/>',
        f'  <geom name="hand_2020_profile_visual" type="box" '
        f'pos="{_v(_world(hand_profile_center_b))}" size="{_v(hand_profile_halfsize_w)}" '
        'rgba="0.62 0.64 0.67 1" contype="0" conaffinity="0" mass="0" group="3"/>',
        f'  <geom name="camera_4040_profile_visual" type="box" '
        f'pos="{_v(_world(camera_support_center_b))}" size="{_v(camera_support_halfsize_w)}" '
        'rgba="0.52 0.54 0.57 1" contype="0" conaffinity="0" mass="0" group="3"/>',
        f'  <geom name="camera_bracket_visual" type="capsule" '
        f'fromto="{_v(_world(bracket_support_b))} {_v(_world(camera_position_b))}" '
        f'size="{CAMERA_BRACKET_RADIUS_APPROX_M:.9f}" rgba="0.18 0.18 0.20 1" '
        'contype="0" conaffinity="0" mass="0" group="3"/>',
    ]
    lines.extend(_axis_geoms("base", np.zeros(3), np.eye(3)))
    lines.extend(_axis_geoms("palm", T_BASE_PALM[:3, 3], T_BASE_PALM[:3, :3]))
    lines.extend(_axis_geoms(
        "camera_optical", T_BASE_CAMERA[:3, 3], T_BASE_CAMERA[:3, :3]
    ))
    lines.append("</mujocoinclude>")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
