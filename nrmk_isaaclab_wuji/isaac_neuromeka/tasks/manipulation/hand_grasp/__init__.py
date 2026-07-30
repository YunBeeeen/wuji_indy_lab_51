"""Gym registrations for the isolated Wuji-hand chopstick tasks.

The three task IDs intentionally represent different experiment boundaries:

* ``hand_grasp`` learns OPEN/CLOSE motion from an already established grasp.
* ``hand_setting`` learns the preceding open-hand-to-functional-grasp transition.
* ``hand_grasp_object`` is the object-interaction scene scaffold built on the
  established grasp; it is not the canonical OPEN/CLOSE training task.

Keeping separate task IDs prevents checkpoints with different observation
dimensions or success definitions from being loaded into the wrong MDP.
"""

import gymnasium as gym

from . import learning
from .hand_grasp_env_cfg import HandGraspEnvCfg
from .hand_object_env_cfg import HandObjectGraspEnvCfg
from .hand_setting_env_cfg import HandSettingEnvCfg


gym.register(
    id="hand_grasp",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandGraspEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:HandGraspPPORunnerCfg",
    },
)


gym.register(
    id="hand_grasp_object",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandObjectGraspEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:HandObjectGraspPPORunnerCfg"
        ),
    },
)


gym.register(
    id="hand_setting",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandSettingEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:HandSettingPPORunnerCfg"
        ),
    },
)
