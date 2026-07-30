import gymnasium as gym

from . import env_cfg, learning

##
# Register Gym environments.
##

gym.register(
    id="Indy-Wuji-Chopsticks-Grasp",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.Indy7WujiChopsticksGraspEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ChopstickAcquirePPORunnerCfg",
    },
)
