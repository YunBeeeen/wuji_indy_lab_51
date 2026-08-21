"""Shared Wuji right-hand actuator parameters and PD tuning slots."""

from __future__ import annotations


WUJI_RIGHT_JOINT_NAMES: tuple[str, ...] = tuple(
    f"finger{finger_index}_joint{joint_index}"
    for finger_index in range(1, 6)
    for joint_index in range(1, 5)
)


# Per-joint torque limits from Wuji Technology's right-hand URDF.  The project
# USD drops the upstream ``right_`` name prefix, so the keys below use the joint
# names that actually appear in the spawned articulation.
WUJI_RIGHT_URDF_EFFORT_LIMITS: dict[str, float] = {
    "finger1_joint1": 0.4452,
    "finger1_joint2": 0.4259,
    "finger1_joint3": 0.3000,
    "finger1_joint4": 0.5000,
    # real: "finger1_joint3": 0.1888,
    # real: "finger1_joint4": 0.1468,

    "finger2_joint1": 0.6188,
    "finger2_joint2": 0.1822,
    "finger2_joint3": 0.3000,
    "finger2_joint4": 0.5000,
    # real: "finger2_joint3": 0.2251,
    # real: "finger2_joint4": 0.2170,

    "finger3_joint1": 0.6494,
    "finger3_joint2": 0.1827,
    "finger3_joint3": 0.3000,
    "finger3_joint4": 0.5000,
    # real: "finger3_joint3": 0.2078,
    # real: "finger3_joint4": 0.2018,

    "finger4_joint1": 0.6389,
    "finger4_joint2": 0.1832,
    "finger4_joint3": 0.3000,
    "finger4_joint4": 0.5000,
    # real: "finger4_joint3": 0.2249,
    # real: "finger4_joint4": 0.2044,

    "finger5_joint1": 0.6441,
    "finger5_joint2": 0.1798,
    "finger5_joint3": 0.3000,
    "finger5_joint4": 0.5000,
    # real: "finger5_joint3": 0.2384,
    # real: "finger5_joint4": 0.1866,
}


# Blank per-joint slots for values measured with the PD tuner.  ``None`` keeps
# that asset's existing fallback; entering a number overrides only that joint.
WUJI_RIGHT_TUNED_STIFFNESS: dict[str, float | None] = {
    "finger1_joint1": 1.7,
    "finger1_joint2": 2.7,
    "finger1_joint3": 1.0,
    "finger1_joint4": 1.5,
    "finger2_joint1": 2.4,
    "finger2_joint2": 0.7,
    "finger2_joint3": 1.0,
    "finger2_joint4": 1.5,
    "finger3_joint1": 2.4,
    "finger3_joint2": 0.7,
    "finger3_joint3": 1.0,
    "finger3_joint4": 1.5,
    "finger4_joint1": 2.4,
    "finger4_joint2": 0.7,
    "finger4_joint3": 1.0,
    "finger4_joint4": 1.5,
    "finger5_joint1": 2.4,
    "finger5_joint2": 0.7,
    "finger5_joint3": 1.0,
    "finger5_joint4": 1.5,
}

WUJI_RIGHT_TUNED_DAMPING: dict[str, float | None] = {
    "finger1_joint1": 0.04,
    "finger1_joint2": 0.05,
    "finger1_joint3": 0.005,
    "finger1_joint4": 0.0,
    "finger2_joint1": 0.055,
    "finger2_joint2": 0.02,
    "finger2_joint3": 0.005,
    "finger2_joint4": 0.0,
    "finger3_joint1": 0.055,
    "finger3_joint2": 0.02,
    "finger3_joint3": 0.005,
    "finger3_joint4": 0.0,
    "finger4_joint1": 0.055,
    "finger4_joint2": 0.02,
    "finger4_joint3": 0.005,
    "finger4_joint4": 0.0,
    "finger5_joint1": 0.055,
    "finger5_joint2": 0.02,
    "finger5_joint3": 0.005,
    "finger5_joint4": 0.0,
}


def resolve_wuji_right_tuned_parameter(
    tuned_values: dict[str, float | None], fallback_values: dict[str, float]
) -> dict[str, float]:
    """Resolve a complete 20-joint parameter map from optional tuned values."""

    expected = set(WUJI_RIGHT_JOINT_NAMES)
    if set(tuned_values) != expected or set(fallback_values) != expected:
        raise ValueError("Wuji PD parameter maps must contain exactly the 20 actuated hand joints.")
    return {
        joint_name: (
            fallback_values[joint_name]
            if tuned_values[joint_name] is None
            else float(tuned_values[joint_name])
        )
        for joint_name in WUJI_RIGHT_JOINT_NAMES
    }
