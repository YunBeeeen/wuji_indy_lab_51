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

MOBY_CFG = FiniteArticulationCfg(
    class_type=FiniteArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/moby/moby.usd",
        activate_contact_sensors=True,  # TODO
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # (Indy control framework already includes gravity compensation) 모비는?
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=FiniteArticulationCfg.InitialStateCfg(
        joint_pos={
            ".*_rot_joint": 0.0,
            ".*_tract_joint": 0.0,
            # Arm joints
            "joint0": 0.0,
            "joint1": 0.0,
            "joint2": -1.57079,
            "joint3": 0.0,
            "joint4": -1.57079,
            "joint5": 0.0,
        },
    ),
    ## TODO: all these limits should be double-checked
    actuators={
        "tract_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*_tract_joint"],
            velocity_limit=5.0,
            effort_limit=100.0,
            stiffness=0.0,  # -> only velocity control
            damping=25.0,
        ),
        "rotation_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*_rot_joint"],
            velocity_limit=2.0,
            effort_limit=100.0,  # 60.0
            stiffness=100.0,  # 100.0
            damping=20.0,  # 20.0
        ),
        # Arm joints
        "arm0": ImplicitActuatorCfg(
            joint_names_expr=["joint[0-1]"],
            velocity_limit=2.775073510670984,
            effort_limit=431.97,
            stiffness=100.0,
            damping=20.0,
        ),
        "arm1": ImplicitActuatorCfg(
            joint_names_expr=["joint2"],
            velocity_limit=2.775073510670984,
            effort_limit=197.23,
            stiffness=100.0,
            damping=20.0,
        ),
        "arm2": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-5]"],
            velocity_limit=3.2986722862692828,
            effort_limit=79.79,
            stiffness=100.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
