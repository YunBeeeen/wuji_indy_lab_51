# [tool] 19mm ArUco 마커 비주얼을 배포용 MJCF에 심는다.
"""Install exact 19 mm OpenCV ArUco visuals in the derived deployment MJCF."""

from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np

from Deploy.common.isaac_reset import STICK_SIZE_M
from Deploy.vision.sim_aruco import (MARKER_IDS_BY_STICK, SIM_MARKER_LAYOUT_CANDIDATE)


ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "assets/wuji_description/hand/body/mjcf/right_with_tip_sites.xml"


def _marker_lines(indent: str, marker_id: int) -> list[str]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = cv2.aruco.generateImageMarker(dictionary, marker_id, 600)
    bits = image.reshape(6, 100, 6, 100).mean((1, 3)) < 127
    _, center_y, roll_deg, protrusion = SIM_MARKER_LAYOUT_CANDIDATE[marker_id]
    angle = np.deg2rad(roll_deg)
    normal = np.asarray([np.sin(angle), 0.0, np.cos(angle)])
    distance = (
        STICK_SIZE_M[0] / 2.0 * (abs(normal[0]) + abs(normal[2]))
        + protrusion
    )
    center = normal * distance
    center[1] = center_y
    quat = [np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0]
    child = indent + "  "
    lines = [
        f'{indent}<body name="aruco{marker_id}_candidate_mount" '
        f'pos="{center[0]:.9f} {center[1]:.9f} {center[2]:.9f}" '
        f'quat="{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}">',
        f'{child}<geom name="aruco{marker_id}_paper" type="box" '
        'pos="0 0 -0.00001" size="0.0114 0.0114 0.00001" '
        'rgba="1 1 1 1" contype="0" conaffinity="0" mass="0"/>'
    ]
    cell = 0.019 / 6.0
    for row, col in np.argwhere(bits):
        x = (float(col) - 2.5) * cell
        y = (2.5 - float(row)) * cell
        lines.append(
            f'{child}<geom name="aruco{marker_id}_r{row}c{col}" type="box" '
            f'pos="{x:.9f} {y:.9f} 0.00001" '
            f'size="{cell/2:.9f} {cell/2:.9f} 0.00001" rgba="0 0 0 1" '
            'contype="0" conaffinity="0" mass="0"/>'
        )
    lines.append(f"{indent}</body>")
    return lines


def main() -> None:
    lines = MJCF.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    installed_sticks: set[int] = set()
    skipped_candidate_body_depth = 0
    for line in lines:
        if skipped_candidate_body_depth:
            skipped_candidate_body_depth += line.count("<body ")
            skipped_candidate_body_depth -= line.count("</body>")
            continue
        if any(token in line for token in (
            'name="aruco_plane"', 'name="aruco0_texture"', 'name="aruco2_texture"',
            'name="aruco0_material"', 'name="aruco2_material"',
        )):
            continue
        match = re.search(r'name="aruco([0-3])_(?:candidate_mount|visual|paper|r\d+c\d+)"', line)
        if match:
            marker_id = int(match.group(1))
            stick_index = 0 if marker_id < 2 else 1
            if stick_index not in installed_sticks:
                indent = line[: len(line) - len(line.lstrip())]
                for candidate_id in MARKER_IDS_BY_STICK[stick_index]:
                    output.extend(_marker_lines(indent, candidate_id))
                installed_sticks.add(stick_index)
            if "candidate_mount" in line:
                skipped_candidate_body_depth = 1
            continue
        output.append(line)
    if installed_sticks != {0, 1}:
        raise RuntimeError(f"Expected marker anchors on both sticks, found {installed_sticks}")
    MJCF.write_text("\n".join(output) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
