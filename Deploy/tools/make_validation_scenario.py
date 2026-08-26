# [tool] 세 백엔드가 똑같이 재생할 결정적 타깃 시퀀스를 생성.
"""Generate the deterministic target sequence the three backends replay.

Isaac, MuJoCo and the real hand must not each sample their own targets, or the
trajectories are not comparable.  This writes one file they all read.

Targets are checked against the middle finger's actual reachable set (URDF FK
swept over the connected hand's factory limits) rather than merely against the
sampling box, so a scenario cannot silently contain a point the finger cannot
occupy.
"""

from __future__ import annotations

import argparse
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..policy.finger_reach import (
    FINGER_REACH_RESET_JOINT_POSITIONS,
    MIDDLE_JOINT_NAMES,
    MIDDLE_POLICY_INDICES,
    REACH_RANGE_M,
    MIDDLE_COMMAND_TARGET_LIMITS,
)
from ..common.fingertip_fk import DEFAULT_URDF_PATH
from ..common.policy_contract import POLICY_DT, POLICY_JOINT_NAMES


def _rpy(rpy):
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _middle_tip_samples(resolution: int = 9) -> np.ndarray:
    """Sweep the four middle joints and return palm-frame tip positions."""

    root = ET.parse(DEFAULT_URDF_PATH).getroot()
    joints = {}
    for element in root.findall("joint"):
        origin = element.find("origin")
        axis = element.find("axis")
        transform = np.eye(4)
        if origin is not None:
            transform[:3, :3] = _rpy([float(v) for v in origin.attrib.get("rpy", "0 0 0").split()])
            transform[:3, 3] = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
        joints[element.find("child").attrib["link"]] = dict(
            name=element.attrib["name"].replace("right_", ""),
            parent=element.find("parent").attrib["link"],
            transform=transform,
            axis=np.asarray([float(v) for v in axis.attrib.get("xyz", "0 0 0").split()])
            if axis is not None else np.zeros(3),
            joint_type=element.attrib["type"],
        )
    chain, link = [], "right_finger3_tip_link"
    while link in joints:
        chain.append(joints[link])
        link = joints[link]["parent"]
    chain.reverse()

    grid = [np.linspace(lo, hi, resolution) for lo, hi in MIDDLE_COMMAND_TARGET_LIMITS]
    samples = []
    for combination in itertools.product(*grid):
        angles = dict(zip([n for n in MIDDLE_JOINT_NAMES], combination))
        pose = np.eye(4)
        for entry in chain:
            local = entry["transform"].copy()
            if entry["joint_type"] == "revolute":
                axis = entry["axis"] / np.linalg.norm(entry["axis"])
                theta = angles[entry["name"]]
                skew = np.asarray([[0, -axis[2], axis[1]],
                                   [axis[2], 0, -axis[0]],
                                   [-axis[1], axis[0], 0]])
                local[:3, :3] = local[:3, :3] @ (
                    np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * skew @ skew
                )
            pose = pose @ local
        samples.append(pose[:3, 3])
    return np.asarray(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", type=int, default=4)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-reach-error-mm", type=float, default=3.0)
    args = parser.parse_args()

    reachable = _middle_tip_samples()
    rng = np.random.default_rng(args.seed)
    chosen: list[np.ndarray] = []
    rejected = 0
    while len(chosen) < args.targets:
        candidate = np.asarray(
            [rng.uniform(*REACH_RANGE_M[axis]) for axis in ("x", "y", "z")]
        )
        distance = np.linalg.norm(reachable - candidate, axis=1).min()
        if distance * 1000.0 <= args.max_reach_error_mm:
            chosen.append(candidate)
        else:
            rejected += 1
        if rejected > 20000:
            raise RuntimeError("Could not find enough reachable targets; widen the tolerance.")

    reset = np.asarray(FINGER_REACH_RESET_JOINT_POSITIONS, dtype=float)
    scenario = {
        "_comment": (
            "Deterministic reach scenario replayed identically by Isaac, MuJoCo "
            "and the real hand. Joint values are canonical finger-major order "
            "(finger1..5 x joint1..4), which is also the Wuji SDK's order."
        ),
        "controlled_joints": list(MIDDLE_JOINT_NAMES),
        "policy_hz": round(1.0 / POLICY_DT, 6),
        "target_duration_s": args.duration,
        "targets_palm_m": [[float(v) for v in t] for t in chosen],
        "initial_all_joint_q": [float(v) for v in reset],
        "initial_middle_q": [float(reset[i]) for i in MIDDLE_POLICY_INDICES],
        "other_joint_hold_positions": {
            name: float(reset[index])
            for index, name in enumerate(POLICY_JOINT_NAMES)
            if index not in set(MIDDLE_POLICY_INDICES.tolist())
        },
        "max_reach_error_mm": args.max_reach_error_mm,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenario, indent=2) + "\n")

    print(f"wrote {len(chosen)} targets -> {args.output}")
    print(f"  rejected {rejected} unreachable candidates "
          f"(tolerance {args.max_reach_error_mm} mm)")
    for index, target in enumerate(chosen):
        distance = np.linalg.norm(reachable - target, axis=1).min()
        print(f"  target[{index}] palm = [{target[0]:+.4f}, {target[1]:+.4f}, {target[2]:+.4f}] m"
              f"   nearest reachable {distance*1000:.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
