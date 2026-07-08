##
# Environment configuration
##

from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.utils import configclass


@configclass
class RLEnvWithIKCfg(ManagerBasedRLEnvCfg):
    ik_method: str = "dls"
    ik_body_name: str = "tcp"
    ik_cmd_name: str = "ee_pose"
    # TODO: move IK params to here


@configclass
class NrmkRLEnvCfg(RLEnvWithIKCfg):
    """Configuration for a reinforcement learning environment."""

    # ui settings
    ui_window_class_type: type | None = ManagerBasedRLEnvWindow

    # general settings
    is_finite_horizon: bool = False
    episode_length_s: float = MISSING

    # environment settings
    rewards: object = MISSING
    terminations: object = MISSING
    curriculum: object = MISSING
    commands: object = MISSING

    # custom elements
    costs: object | None = None
    pointcloud: object | None = None

    # New for NRMK-RL
    actor_obs_list: list = ["policy"]
    critic_obs_list: list | None = None
    teacher_obs_list: list | None = None

    # IK settings
    ik_method: str = "dls"
    ik_body_name: str = "tcp"

    def __post_init__(self):
        """Post initialization."""
        # task settings
        self.decimation = 24
        self.sim.render_interval = 8
        self.sim.dt = 1.0 / 120.0
