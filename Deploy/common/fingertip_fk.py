"""Standalone Wuji Hand 1 fingertip FK parsed from a pinned URDF.

There are TWO Wuji URDFs in this repository and they do not agree, so every
caller must say which one it means -- that is why ``source`` has no default.

``finger1_tip_link`` sits 3.0 mm apart between them and ``finger2_tip_link``
0.7 mm; the joint axes and every intermediate frame agree once composed, so the
difference is purely where the fixed tip link is mounted.  Measured 2026-08-21
over reset / pregrasp / limit / random poses, the offset is constant to
0.04 mm, i.e. a fixed bias rather than a kinematic error.

Which one to use depends on what the number is FOR:

* ``POLICY_TIP_FRAME_URDF`` for anything feeding ``obs[40:55]``.  The policy
  was trained against Isaac's USD, which was imported from the local URDF, so
  that is the trained contract regardless of which model the backend runs.
* ``OFFICIAL_URDF`` only to check the vendor description against MuJoCo, which
  loads that same description.

Middle-finger-only work cannot tell them apart (``finger3`` agrees to
0.001 mm), which is why finger_reach never exposed this.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import numpy.typing as npt

from .policy_contract import ACTION_DIM, CANONICAL_JOINTS


#: The Deploy package root, one level up now that this sits in common/.
_DEPLOY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class UrdfSource:
    """One URDF plus the link/joint naming convention it uses."""

    path: Path
    joint_prefix: str
    palm_link: str
    label: str

    def tip_link_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.joint_prefix}finger{finger}_tip_link" for finger in range(1, 6)
        )


#: Pinned vendor description.  MuJoCo loads this same model.
OFFICIAL_URDF = UrdfSource(
    path=_DEPLOY_ROOT / "assets/wuji_description/hand/body/urdf/right.urdf",
    joint_prefix="right_",
    palm_link="right_palm_link",
    label="official wuji-description",
)

#: The URDF Isaac's ``wuji_right_filtered.usda`` was imported from.  Referenced
#: in place rather than copied so an edit to it cannot silently diverge from the
#: trained contract -- ``tests`` pins the tip offsets numerically.
ISAAC_URDF = UrdfSource(
    path=(
        _DEPLOY_ROOT.parent
        / "nrmk_isaaclab_wuji/isaac_neuromeka/assets/model/urdf/wuji_right/wuji_right.urdf"
    ),
    joint_prefix="",
    palm_link="palm_link",
    label="local Isaac import source",
)

#: The frames the trained policy actually saw.  Anything that builds a policy
#: observation must use this one on EVERY backend.
POLICY_TIP_FRAME_URDF = ISAAC_URDF

#: Retained for tools that parse the vendor XML directly for their own reasons.
DEFAULT_URDF_PATH = OFFICIAL_URDF.path


@dataclass(frozen=True)
class UrdfJointTransform:
    name: str
    parent: str
    child: str
    joint_type: str
    origin: npt.NDArray[np.float64]
    axis: npt.NDArray[np.float64]
    policy_index: int | None


class WujiHand1FingertipFK:
    """Return five tip-link origins in the palm frame of a chosen URDF.

    ``source`` is required: see the module docstring for why picking the wrong
    one silently biases the thumb by 3 mm.
    """

    def __init__(self, source: UrdfSource):
        if not isinstance(source, UrdfSource):
            raise TypeError(
                "source must be an UrdfSource (POLICY_TIP_FRAME_URDF for policy "
                f"observations, OFFICIAL_URDF for vendor checks), got {type(source).__name__}."
            )
        self.source = source
        self.urdf_path = Path(source.path).resolve()
        self.palm_link = source.palm_link
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        root = ET.parse(self.urdf_path).getroot()
        official_urdf_name_to_policy = {
            f"{source.joint_prefix}{joint.canonical_name}": joint.policy_index
            for joint in CANONICAL_JOINTS
        }
        self._by_child: dict[str, UrdfJointTransform] = {}
        for element in root.findall("joint"):
            name = element.attrib["name"]
            parent = element.find("parent").attrib["link"]
            child = element.find("child").attrib["link"]
            origin_element = element.find("origin")
            xyz = _vector(origin_element.attrib.get("xyz", "0 0 0"))
            rpy = _vector(origin_element.attrib.get("rpy", "0 0 0"))
            origin = np.eye(4, dtype=np.float64)
            origin[:3, :3] = _rpy_matrix(rpy)
            origin[:3, 3] = xyz
            axis_element = element.find("axis")
            axis = _vector(axis_element.attrib.get("xyz", "0 0 0")) if axis_element is not None else np.zeros(3)
            transform = UrdfJointTransform(
                name=name,
                parent=parent,
                child=child,
                joint_type=element.attrib["type"],
                origin=origin,
                axis=axis,
                policy_index=official_urdf_name_to_policy.get(name),
            )
            if child in self._by_child:
                raise RuntimeError(f"URDF child link {child!r} has two parent joints.")
            self._by_child[child] = transform

        self.tip_link_names = source.tip_link_names()
        resolved = {x.policy_index for x in self._by_child.values() if x.policy_index is not None}
        if resolved != set(range(ACTION_DIM)):
            raise RuntimeError(f"URDF did not resolve all canonical joints: {sorted(resolved)}")
        for tip in self.tip_link_names:
            self._chain_to_palm(tip)

    def fingertip_positions_in_palm(
        self, q_policy: npt.ArrayLike
    ) -> npt.NDArray[np.float32]:
        q = np.asarray(q_policy, dtype=np.float64)
        if q.shape != (ACTION_DIM,) or not np.isfinite(q).all():
            raise ValueError("Canonical q must be a finite vector of shape (20,).")
        tips = []
        for tip_link in self.tip_link_names:
            transform = np.eye(4, dtype=np.float64)
            for joint in self._chain_to_palm(tip_link):
                transform = transform @ joint.origin
                if joint.joint_type == "revolute":
                    transform = transform @ _axis_angle_transform(
                        joint.axis, float(q[joint.policy_index])
                    )
                elif joint.joint_type != "fixed":
                    raise RuntimeError(f"Unsupported joint type {joint.joint_type!r}.")
            tips.append(transform[:3, 3])
        return np.asarray(tips, dtype=np.float32).reshape(15)

    def _chain_to_palm(self, child_link: str) -> tuple[UrdfJointTransform, ...]:
        reverse_chain = []
        current = child_link
        while current != self.palm_link:
            joint = self._by_child.get(current)
            if joint is None:
                raise RuntimeError(
                    f"No URDF chain from {child_link!r} to {self.palm_link!r} "
                    f"in {self.source.label}; stopped at {current!r}."
                )
            reverse_chain.append(joint)
            current = joint.parent
        return tuple(reversed(reverse_chain))


def _vector(text: str) -> npt.NDArray[np.float64]:
    result = np.fromstring(text, sep=" ", dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid URDF vector {text!r}.")
    return result


def _rpy_matrix(rpy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _axis_angle_transform(axis: npt.NDArray[np.float64], angle: float) -> npt.NDArray[np.float64]:
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError("Revolute joint axis cannot be zero.")
    x, y, z = axis / norm
    c, s, one_c = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    rotation = np.asarray(
        [
            [c + x*x*one_c, x*y*one_c - z*s, x*z*one_c + y*s],
            [y*x*one_c + z*s, c + y*y*one_c, y*z*one_c - x*s],
            [z*x*one_c - y*s, z*y*one_c + x*s, c + z*z*one_c],
        ]
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    return result
