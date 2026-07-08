import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from isaac_neuromeka.assets.articulation import (
    FiniteArticulation,
    FiniteArticulationCfg,
)

##
# Configuration
##

DUAL_ARM_CFG = FiniteArticulationCfg(
    class_type=FiniteArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/dual_icon3/dual_icon3_edited.usd",
        # usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/indy7_simplified/indy7_simplified.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=FiniteArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={  # TODO: update
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_arm.*"],
            velocity_limit=10.0,
            effort_limit=431.97,
            stiffness=100.0,
            damping=10.0,
        ),
        "body": ImplicitActuatorCfg(
            joint_names_expr=["base_joint"],
            velocity_limit=1.0,
            effort_limit=1000.00,
            stiffness=500.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
