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

ZEN_CFG = FiniteArticulationCfg(
    class_type=FiniteArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/paxini/paxini.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=FiniteArticulationCfg.InitialStateCfg(
        joint_pos={
            # Wheel joints
            "r_wheel_joint": 0.0,
            "l_wheel_joint": 0.0,
            # Rotation joint
            "rotate_joint": 0.0,
            # Waist joints
            "waist_1_joint": -0.9075,  # (-2.286, 0.471) → (-2.286 + 0.471) / 2 = -0.9075
            "waist_2_joint": 0.384,  # (-1.518, 2.286) → (-1.518 + 2.286) / 2 = 0.384
            # Neck and head joints
            "neck_joint": 0.0,
            "head_joint": 0.0,
            # Right arm joints
            "rarm_1_joint": 0.0,  # (-2.81, 2.81) → 0.0
            "rarm_2_joint": 0.938,  # (-0.314, 2.19) → (2.19 + (-0.314)) / 2 = 0.938
            "rarm_3_joint": 0.0,  # (-2.81, 2.81) → 0.0
            "rarm_4_joint": -1.0715,  # (-2.37, 0.227) → (0.227 + (-2.37)) / 2 = -1.0715
            "rarm_5_joint": 0.0,  # (-2.81, 2.81) → 0.0
            "rarm_6_joint": 0.0,  # (-0.715, 0.715) → 0.0
            # Left arm joints
            "larm_1_joint": 0.0,  # (-2.81, 2.81) → 0.0
            "larm_2_joint": -0.938,  # (-2.19, 0.314) → (0.314 + (-2.19)) / 2 = -0.938
            "larm_3_joint": 0.0,  # (-2.81, 2.81) → 0.0
            "larm_4_joint": -1.0715,  # (-2.37, 0.227) → (0.227 + (-2.37)) / 2 = -1.0715
            "larm_5_joint": 0.0,  # (-2.81, 2.81) → 0.0
            "larm_6_joint": 0.0,  # (-0.715, 0.715) → 0.0
        },
    ),
    actuators={
        # # Wheel joints - velocity control
        # "wheel_joints": ImplicitActuatorCfg(
        #     joint_names_expr=["r_wheel_joint", "l_wheel_joint"],
        #     velocity_limit_sim=2.0,
        #     effort_limit_sim=50.0,
        #     stiffness=0.0,  # velocity control only
        #     damping=20.0,
        # ),
        # Rotation joint
        "rotation_joint": ImplicitActuatorCfg(
            joint_names_expr=["rotate_joint"],
            velocity_limit_sim=2.0,
            effort_limit_sim=841.0,
            stiffness=1000.0,
            damping=200.0,
        ),
        "waist_joint1": ImplicitActuatorCfg(
            joint_names_expr=["waist_1_joint"],
            velocity_limit_sim=2.0,
            effort_limit_sim=1080.0,
            stiffness=1000.0,
            damping=120.0,
        ),
        "waist_joint2": ImplicitActuatorCfg(
            joint_names_expr=["waist_2_joint"],
            velocity_limit_sim=2.0,
            effort_limit_sim=841.0,
            stiffness=800.0,
            damping=100.0,
        ),
        # Neck and head joints
        "head_joints": ImplicitActuatorCfg(
            joint_names_expr=["neck_joint", "head_joint"],
            velocity_limit_sim=3.0,
            effort_limit_sim=35.0,
            stiffness=100.0,
            damping=20.0,
        ),
        "arm_0": ImplicitActuatorCfg(
            joint_names_expr=["rarm_[1-2]_joint", "larm_[1-2]_joint"],
            velocity_limit_sim=3.0,
            effort_limit_sim=191.0,
            stiffness=100.0,
            damping=20.0,
        ),
        "arm_1": ImplicitActuatorCfg(
            joint_names_expr=["rarm_[3-4]_joint", "larm_[3-4]_joint"],
            velocity_limit_sim=3.0,
            effort_limit_sim=143.0,
            stiffness=80.0,
            damping=20.0,
        ),
        "arm_2": ImplicitActuatorCfg(
            joint_names_expr=["rarm_5_joint", "larm_5_joint"],
            velocity_limit_sim=2.5,  # Indy arm2
            effort_limit_sim=70.0,  # Indy arm2
            stiffness=70.0,
            damping=10.0,
        ),
        "arm_3": ImplicitActuatorCfg(
            joint_names_expr=["rarm_6_joint", "larm_6_joint"],  # todo: ADD 7
            velocity_limit_sim=3.0,  # Indy arm2
            effort_limit_sim=35.0,  # Indy arm2
            stiffness=70.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=0.97,
)
