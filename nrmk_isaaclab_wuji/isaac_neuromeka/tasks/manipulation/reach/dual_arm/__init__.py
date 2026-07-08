import gymnasium as gym

from . import env_cfg, learning

##
# Register Gym environments.
##

gym.register(
    id="Dual-Arm-Reach",
    # entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.DualArmReachEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
        "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachPPORunnerCfg",
    },
)
