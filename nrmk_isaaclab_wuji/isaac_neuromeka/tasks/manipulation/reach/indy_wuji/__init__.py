import gymnasium as gym

from . import env_cfg, learning

##
# Register Gym environments.
##

gym.register(
    id="Indy-Wuji-Reach",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.Indy7WujiReachEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
    },
)
