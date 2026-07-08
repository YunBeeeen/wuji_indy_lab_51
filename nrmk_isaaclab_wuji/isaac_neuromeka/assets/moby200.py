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

MOBY200_CFG = FiniteArticulationCfg(
    class_type=FiniteArticulation,
    # fixed_base=False,
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/moby2_0722/moby2_0722.usd",      #/model/usd/moby2_C/moby2_C.usd"                 /model/moby_col.usd  , /model/usd/moby2/moby.usd
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/model/usd/moby200/moby200.usd",  # /model/usd/moby2_C/moby2_C.usd"                 /model/moby_col.usd  , /model/usd/moby2/moby.usd
        activate_contact_sensors=True,  # TODO
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,  # TODO: think about this. Maybe False if we have upper body without gravity comp
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=FiniteArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
            # "rear_castor_mount_to_rear_castor_dummy_joint": 1.50,
            # "rear_castor_dummy_to_rear_castor_wheel_joint": 0.0,
            # "front_castor_mount_to_front_castor_dummy_joint": 1.50,
            # "front_castor_dummy_to_front_castor_wheel_joint": 0.0,
        },
    ),
    actuators={
        "wheel_joints": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            velocity_limit=2.0,
            effort_limit=50.0,
            stiffness=0.0,
            damping=20.0,
        ),
        # 캐스터는 수동 joint → actuator 없음
    },
    soft_joint_pos_limit_factor=0.95,
)
