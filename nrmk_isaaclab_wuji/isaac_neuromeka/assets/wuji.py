"""Standalone Wuji right-hand asset configuration."""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from isaac_neuromeka.assets.articulation import FiniteArticulation, FiniteArticulationCfg


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
        joint_pos={"finger[1-5]_joint[1-4]": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["finger[1-5]_joint[1-4]"],
            effort_limit=0.6,
            velocity_limit=12.0,
            stiffness={
                "finger[1-5]_joint[1-2]": 2.0,
                "finger[1-5]_joint[3-4]": 1.0,
            },
            damping=0.05,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
