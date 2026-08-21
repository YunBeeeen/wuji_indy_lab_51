"""Standalone Wuji right-hand asset configuration."""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from isaac_neuromeka.assets.articulation import FiniteArticulation, FiniteArticulationCfg
from isaac_neuromeka.assets.wuji_actuator_parameters import (
    WUJI_RIGHT_JOINT_NAMES,
    WUJI_RIGHT_TUNED_DAMPING,
    WUJI_RIGHT_TUNED_STIFFNESS,
    WUJI_RIGHT_URDF_EFFORT_LIMITS,
    resolve_wuji_right_tuned_parameter,
)


# The accompanying MuJoCo model excludes palm_link against each finger*_link2,
# while the standalone URDF-derived USD has no equivalent pair filters. Load a
# non-destructive overlay that aligns those five structural contact exceptions.
_WUJI_RIGHT_USD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model",
    "urdf",
    "wuji_right",
    "wuji_right",
    "wuji_right_filtered.usda",
)

_WUJI_RIGHT_DEFAULT_STIFFNESS = {
    joint_name: (2.0 if joint_name.endswith(("joint1", "joint2")) else 1.0)
    for joint_name in WUJI_RIGHT_JOINT_NAMES
}
_WUJI_RIGHT_DEFAULT_DAMPING = {joint_name: 0.05 for joint_name in WUJI_RIGHT_JOINT_NAMES}


WUJI_RIGHT_CFG = FiniteArticulationCfg(
    class_type=FiniteArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=_WUJI_RIGHT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64.0 / math.pi * 180.0,
            max_depenetration_velocity=5.0,
            max_contact_impulse=100.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=FiniteArticulationCfg.InitialStateCfg(
        # 2026-08-17: ``finger1_joint1`` cannot default to 0.0 any more.  The
        # official Wuji Hand 1 description gives that joint a *positive* lower
        # limit (+0.0475 rad), unlike the older local URDF which allowed
        # -0.0448, so ``Articulation._validate_cfg`` rejects a 0.0 default.  Use
        # a small margin above the limit rather than the limit itself.  Every
        # hand task overwrites these values at reset with ``pose_005`` anyway;
        # this default only has to be a legal spawn state.
        #
        # The three patterns are deliberately disjoint: overlapping regex keys
        # in ``joint_pos`` resolve ambiguously.
        joint_pos={
            "finger1_joint1": 0.05,
            "finger1_joint[2-4]": 0.0,
            "finger[2-5]_joint[1-4]": 0.0,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["finger[1-5]_joint[1-4]"],
            effort_limit_sim=WUJI_RIGHT_URDF_EFFORT_LIMITS,
            velocity_limit=12.0,
            stiffness=resolve_wuji_right_tuned_parameter(
                WUJI_RIGHT_TUNED_STIFFNESS, _WUJI_RIGHT_DEFAULT_STIFFNESS
            ),
            damping=resolve_wuji_right_tuned_parameter(
                WUJI_RIGHT_TUNED_DAMPING, _WUJI_RIGHT_DEFAULT_DAMPING
            ),
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
