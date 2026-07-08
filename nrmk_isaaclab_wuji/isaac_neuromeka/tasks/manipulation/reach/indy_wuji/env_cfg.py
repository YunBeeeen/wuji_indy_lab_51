from __future__ import annotations

import isaac_neuromeka.mdp as mdp
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaac_neuromeka.assets import INDY7_WUJI_RIGHT_CFG
from isaac_neuromeka.mdp.actions import CustomJointPositionAction
from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import (  # noqa: F401
    ObservationsCfg,
    ReachEnvCfg,
    TeacherObsCfg,
)

##
# Environment configuration
##


@configclass
class Indy7WujiReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = INDY7_WUJI_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["tcp"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["tcp"]
        self.rewards.end_effector_speed.params["asset_cfg"].body_names = ["tcp"]
        self.actions.arm_action = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot",
            joint_names=["joint[0-5]"],
            scale=0.2,
            use_default_offset=True,
        )
        self.commands.ee_pose.body_name = "tcp"


@configclass
class Indy7WujiReachTeacherEnvCfg(Indy7WujiReachEnvCfg):
    observations = TeacherObsCfg()

    actor_obs_list: list = ["proprioception", "privileged"]
    critic_obs_list: list | None = None
    teacher_obs_list: list | None = None
