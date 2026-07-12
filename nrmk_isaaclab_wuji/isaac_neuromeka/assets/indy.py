import math
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

INDY7_CFG = FiniteArticulationCfg(
    class_type=FiniteArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/indy7/indy7.usd",
        # usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/indy7_simplified/indy7_simplified.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # (Indy control framework already includes gravity compensation)
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=FiniteArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint0": 0.0,
            "joint1": 0.0,
            "joint2": -1.57079,
            "joint3": 0.0,
            "joint4": -1.57079,
            "joint5": 0.0,
        },
    ),
    actuators={
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
    soft_joint_pos_limit_factor=1.0,
)

INDY7_ORBIT_ALLEGRO_CFG = FiniteArticulationCfg(
    class_type=FiniteArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/indy7_orbit_allegro_hand/indy7_orbit_allegro_hand.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        # disable self-collision
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=FiniteArticulationCfg.InitialStateCfg(
        joint_pos={
            # indy7
            "joint0": 0.0,
            "joint1": -0.649,
            "joint2": -2.064,
            "joint3": 0.0,
            "joint4": 1.11199,
            "joint5": 2.356194490192345,
            # allegro hand
            "index_joint_[0-3]": 0.0,
            "middle_joint_[0-3]": 0.0,
            "ring_joint_[0-3]": 0.0,
            "thumb_joint_0": 0.28,
            "thumb_joint_[1-3]": 0.0,
        },
    ),
    actuators={
        # indy7
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
        # allegro hand
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["index_joint_[0-3]", "middle_joint_[0-3]", "ring_joint_[0-3]", "thumb_joint_[0-3]"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

INDY7_WUJI_RIGHT_CFG = FiniteArticulationCfg(
    class_type=FiniteArticulation,
    spawn=sim_utils.UsdFileCfg(
        # active: fidelity tier, arm collision simplified and hand collision restored from *_collision.STL convex hulls
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/indy7_wuji_right/indy7_wuji_right_simplified.usd",
        # fallback: quick-start tier, arm + hand both simplified (cube colliders)
        # usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/indy7_wuji_right/indy7_wuji_right_all_simplified.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=FiniteArticulationCfg.InitialStateCfg(
        joint_pos={
            # indy7
            "joint0": 0.0,
            "joint1": -0.649,
            "joint2": -2.064,
            "joint3": 0.0,
            "joint4": 1.11199,
            "joint5": 2.356194490192345,
            # wuji hand
            "finger[1-5]_joint[1-4]": 0.0,
        },
    ),
    actuators={
        # indy7
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
        # wuji hand
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["finger[1-5]_joint[1-4]"],
            effort_limit=0.6,
            velocity_limit=12.0,
            stiffness=8.0,
            damping=0.5,
            friction=0.02,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
