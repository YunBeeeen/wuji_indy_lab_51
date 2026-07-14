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
            max_depenetration_velocity=5.0,
            max_contact_impulse=100.0,
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
        # wuji hand — 제조사 값 기준 (2026-07-14 검증)
        # effort: URDF 스펙 0.2~1.0 N·m의 중간값 0.6. 이 값으로 0.30kg 하중 유지가 확인됨.
        #   단, 그 "4/4 들기"는 palm-up 자세에서 새끼/손바닥 받침 hold였음 (GUI로 확인 —
        #   엄지·중지는 닿지도 않음). 엄지+중지 집게는 미검증. "힘은 충분"의 근거로만 쓸 것
        # kp/kd: 제조사 right.xml은 kp 2/2/1/0.8, kd 0.05. 기존 20/0.5는 그 10~25배라
        #   오차 1.7°면 토크 포화 -> bang-bang 떨림 + 큐브 후려침. 제조사 값이 성공 창도 넓음
        #   (오므림 0.32~0.60 대부분 4/4 vs 기존은 0.40~0.50만). joint3/4는 1.0으로 통일
        # ★ 약지/새끼(finger4-5)는 예외로 뻣뻣하게: 정책 액션에 없어서 gain만으로 자세를
        #   유지해야 함. kp 1~2로는 접힘(1.2)을 못 버티고 저절로 펴지며(-0.5까지) 파지 지점을
        #   쓸었음 (2026-07-14 실측). chopsticks/functional_grasp의 접힌 자세 유지용으로 필요.
        #   cube grasp는 커플링 액션이 finger4-5를 매 스텝 구동하므로 env_cfg에서 전 손가락을
        #   제조사 값으로 override함 (indy_wuji/env_cfg.py 참고)
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["finger[1-5]_joint[1-4]"],
            effort_limit=0.6,
            velocity_limit=12.0,
            stiffness={
                "finger[1-3]_joint[1-2]": 2.0,
                "finger[1-3]_joint[3-4]": 1.0,
                "finger[4-5]_joint[1-4]": 20.0,
            },
            damping={
                "finger[1-3]_joint[1-4]": 0.05,
                "finger[4-5]_joint[1-4]": 0.5,
            },
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
