from __future__ import annotations

import isaac_neuromeka.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaac_neuromeka.assets import INDY7_WUJI_RIGHT_CFG
from isaac_neuromeka.mdp.actions import CustomJointPositionAction
from isaac_neuromeka.tasks.manipulation.grasp.cube_grasp_env_cfg import (  # noqa: F401
    CubeGraspEnvCfg,
)

##
# Environment configuration
##

@configclass
class Indy7WujiCubeGraspEnvCfg(CubeGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = INDY7_WUJI_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.render_interval = self.decimation
        self.scene.robot.init_state.joint_pos.update(
            {
                "joint1": -0.45,
                "joint2": -1.85,
                "joint4": 1.20,
            }
        )

        controlled_joint_names = ["joint[0-5]", "finger[1-3]_joint[1-4]"]

        def controlled_joint_cfg():
            return SceneEntityCfg("robot", joint_names=controlled_joint_names)

        self.observations.policy.joint_pos.params = {"asset_cfg": controlled_joint_cfg()}

        self.actions.arm_action = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot",
            joint_names=controlled_joint_names,
            # Targets are absolute (default + scale * action), so scale is the reach radius around
            # the default pose, not just a smoothing knob. At 0.1 the arm cannot cover the 0.7 m to
            # the cube within the 20-step episode.
            scale=0.2,
            use_default_offset=True,
        )
