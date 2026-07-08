from __future__ import annotations

import isaac_neuromeka.mdp as mdp
from isaaclab.managers import SceneEntityCfg
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
        self.sim.render_interval = self.decimation

        arm_joint_names = ["joint[0-5]"]

        def arm_joint_cfg():
            return SceneEntityCfg("robot", joint_names=arm_joint_names)

        # "tcp" is only a leftover non-rigid frame on link6 from the bare indy7 arm asset;
        # the wuji hand's actual attachment body is palm_link, so track that instead.
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["palm_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["palm_link"]
        self.rewards.end_effector_speed.params["asset_cfg"].body_names = ["palm_link"]
        self.rewards.joint_vel.params["asset_cfg"] = arm_joint_cfg()

        self.observations.policy.joint_pos.params = {"asset_cfg": arm_joint_cfg()}
        self.observations.policy.joint_vel.params = {"asset_cfg": arm_joint_cfg()}

        if hasattr(self.observations, "proprioception"):
            self.observations.proprioception.joint_pos.params = {"asset_cfg": arm_joint_cfg()}
            self.observations.proprioception.joint_vel.params = {"asset_cfg": arm_joint_cfg()}
        if hasattr(self.observations, "privileged"):
            self.observations.privileged.joint_friction.params = {"asset_cfg": arm_joint_cfg()}
            self.observations.privileged.joint_damping.params = {"asset_cfg": arm_joint_cfg()}

        self.actions.arm_action = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot",
            joint_names=arm_joint_names,
            scale=0.2,
            use_default_offset=True,
        )
        self.commands.ee_pose.body_name = "palm_link"


@configclass
class Indy7WujiReachTeacherEnvCfg(Indy7WujiReachEnvCfg):
    observations = TeacherObsCfg()

    actor_obs_list: list = ["proprioception", "privileged"]
    critic_obs_list: list | None = None
    teacher_obs_list: list | None = None


@configclass
class Indy7WujiReachStudentEnvCfg(Indy7WujiReachTeacherEnvCfg):
    actor_obs_list: list = ["proprioception"]
    teacher_obs_list: list = ["proprioception", "privileged"]


@configclass
class Indy7WujiReachCMDPEnvCfg(Indy7WujiReachEnvCfg):
    observations = TeacherObsCfg()

    actor_obs_list: list = ["proprioception", "privileged"]
    critic_obs_list: list | None = None
    teacher_obs_list: list | None = None

    def __post_init__(self):
        super().__post_init__()
