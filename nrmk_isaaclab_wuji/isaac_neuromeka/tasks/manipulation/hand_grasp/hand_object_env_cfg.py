"""Object-pick scene built on the validated Wuji two-stick grasp."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp as hand_grasp_mdp
from .hand_grasp_env_cfg import (
    HandGraspEnvCfg,
    HandGraspEventCfg,
    HandGraspSceneCfg,
    PREGRASP_STICK1_POSITION_P,
    PREGRASP_STICK1_QUATERNION_P,
    PREGRASP_STICK2_POSITION_P,
    PREGRASP_STICK2_QUATERNION_P,
    SMALL_CONTACT_OFFSET,
    SMALL_MAX_DEPENETRATION_VELOCITY,
    SMALL_REST_OFFSET,
    STICK_TIP_OFFSET_O,
)


# Rotate the complete validated hand/stick state about world x so the mean
# chopstick long axis is horizontal.  This leaves the in-hand relative poses
# unchanged and moves the distal tips away from the palm, making room for a
# vertical support below the grasp object.
HAND_OBJECT_ROOT_ROT = (
    0.3543931173,
    0.6118868526,
    0.3543931173,
    0.6118868526,
)

OBJECT_SIZE = (0.010, 0.010, 0.010)
OBJECT_MASS = 0.002
SUPPORT_SIZE = (0.006, 0.006, 0.500)

OBJECT = SceneEntityCfg("object")
OBJECT_SUPPORT = SceneEntityCfg("object_support")
_BASE_HAND_SCENE = HandGraspSceneCfg(num_envs=1, env_spacing=1.0)


@configclass
class HandObjectGraspSceneCfg(HandGraspSceneCfg):
    """The existing two-stick hand plus a supported 10 mm dynamic cube."""

    robot = _BASE_HAND_SCENE.robot.replace(
        init_state=_BASE_HAND_SCENE.robot.init_state.replace(
            rot=HAND_OBJECT_ROOT_ROT,
        ),
    )

    object_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObjectSupport",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.1135, 0.1015, 0.2775),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=SUPPORT_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=SMALL_CONTACT_OFFSET,
                rest_offset=SMALL_REST_OFFSET,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.35, 0.35, 0.38),
                metallic=0.0,
                roughness=0.8,
            ),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.1135, 0.1015, 0.5325),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_SIZE,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=SMALL_MAX_DEPENETRATION_VELOCITY,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=SMALL_CONTACT_OFFSET,
                rest_offset=SMALL_REST_OFFSET,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.42, 0.95),
                metallic=0.0,
                roughness=0.5,
            ),
        ),
    )


@configclass
class HandObjectGraspEventCfg(HandGraspEventCfg):
    """Reset the supported object after restoring the validated stick pose."""

    reset_object_for_tip_grasp = EventTerm(
        func=hand_grasp_mdp.reset_object_between_stick_tips,
        mode="reset",
        params={
            "palm_cfg": SceneEntityCfg("robot", body_names=["palm_link"]),
            "object_cfg": OBJECT,
            "support_cfg": OBJECT_SUPPORT,
            "stick1_position_p": PREGRASP_STICK1_POSITION_P,
            "stick1_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
            "stick2_position_p": PREGRASP_STICK2_POSITION_P,
            "stick2_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
            "stick1_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick2_tip_offset_o": STICK_TIP_OFFSET_O,
            "object_size": OBJECT_SIZE,
            "support_height": SUPPORT_SIZE[2],
        },
    )


@configclass
class HandObjectGraspEnvCfg(HandGraspEnvCfg):
    """Environment-only object-pick stage; policy MDP is unchanged for now."""

    scene: HandObjectGraspSceneCfg = HandObjectGraspSceneCfg(
        num_envs=4096,
        env_spacing=1.0,
    )
    events: HandObjectGraspEventCfg = HandObjectGraspEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (0.55, 0.65, 0.68)
        self.viewer.lookat = (0.11, 0.10, 0.53)
