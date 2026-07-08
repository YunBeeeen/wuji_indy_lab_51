import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from isaac_neuromeka.assets.articulation import (
    FiniteArticulation,
    FiniteArticulationCfg,
)

NAMI_CFG = FiniteArticulationCfg(
    class_type=FiniteArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/NAMI/NAMI_wo_upper_body_collision.usd",
        # usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/NAMI/NAMI.usd",  # Use this if you want to include upper body collision, but it may cause difficultly navigating in narrow spaces
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # Gravity compensation is used for real-world deployment
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=FiniteArticulationCfg.InitialStateCfg(
        joint_pos={
            # Upper body
            "Joint_L0": 0.0,
            "Joint_L1": 0.0,
            "Joint_L2_R": 0.0,
            "Joint_L3_R": 1.2,
            "Joint_L4_R": 0.0,
            "Joint_L5_R": 0.3,
            "Joint_L6_R": 0.0,
            "Joint_L7_R": 0.0,
            "Joint_L8_R": 0.0,
            "Joint_L2_L": 0.0,
            "Joint_L3_L": -1.2,
            "Joint_L4_L": 0.0,
            "Joint_L5_L": -0.3,
            "Joint_L6_L": 0.0,
            "Joint_L7_L": 0.0,
            "Joint_L8_L": 0.0,
            "Joint_L2_U": 0.0,
            "Joint_L3_U": 0.0,
            # Mobile base
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
    ),
    actuators={
        # Upper body
        "waist_joint_yaw": ImplicitActuatorCfg(
            joint_names_expr=["Joint_L0"],
            velocity_limit_sim=1.308996939,
            effort_limit_sim=235.0,
            stiffness=500.0,
            damping=200.0,
        ),
        "waist_joint_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Joint_L1"],
            velocity_limit_sim=0.523598776,
            effort_limit_sim=1000.0,
            stiffness=700.0,
            damping=300.0,
        ),
        "neck_joint_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Joint_L2_U"],
            velocity_limit_sim=2.617993878,
            effort_limit_sim=126.0,
            stiffness=600.0,
            damping=250.0,
        ),
        "neck_joint_yaw": ImplicitActuatorCfg(
            joint_names_expr=["Joint_L3_U"],
            velocity_limit_sim=2.617993878,
            effort_limit_sim=40.0,
            stiffness=170.0,
            damping=50.0,
        ),
        "arm_joint_0": ImplicitActuatorCfg(
            joint_names_expr=["Joint_L2_R", "Joint_L2_L"],
            velocity_limit_sim=2.617993878,
            effort_limit_sim=360.0,
            stiffness=100.0,
            damping=20.0,
        ),
        "arm_joint_1": ImplicitActuatorCfg(
            joint_names_expr=["Joint_L3_R", "Joint_L3_L"],
            velocity_limit_sim=2.617993878,
            effort_limit_sim=360.0,
            stiffness=200.0,
            damping=40.0,
        ),
        "arm_joint_2": ImplicitActuatorCfg(
            joint_names_expr=["Joint_L4_R", "Joint_L4_L"],
            velocity_limit_sim=2.617993878,
            effort_limit_sim=135.0,
            stiffness=250.0,
            damping=20.0,
        ),
        "arm_joint_3": ImplicitActuatorCfg(
            joint_names_expr=["Joint_L5_R", "Joint_L5_L"],
            velocity_limit_sim=2.617993878,
            effort_limit_sim=135.0,
            stiffness=150.0,
            damping=20.0,
        ),
        "arm_joint_4": ImplicitActuatorCfg(
            joint_names_expr=["Joint_L6_R", "Joint_L6_L"],
            velocity_limit_sim=2.617993878,
            effort_limit_sim=60.0,
            stiffness=250.0,
            damping=20.0,
        ),
        "arm_joint_5": ImplicitActuatorCfg(
            joint_names_expr=["Joint_L7_R", "Joint_L7_L"],
            velocity_limit_sim=2.617993878,
            effort_limit_sim=60.0,
            stiffness=250.0,
            damping=20.0,
        ),
        "arm_joint_6": ImplicitActuatorCfg(
            joint_names_expr=["Joint_L8_R", "Joint_L8_L"],
            velocity_limit_sim=2.617993878,
            effort_limit_sim=60.0,
            stiffness=250.0,
            damping=20.0,
        ),
        # Mobile base
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            velocity_limit=2.0,
            effort_limit=50.0,
            stiffness=0.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
