import gymnasium as gym

from . import indy_env_cfg, learning, moby_env_cfg, nami_env_cfg, zen_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Indy-Deploy",
    # entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": indy_env_cfg.IndyDeployEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
        "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachPPORunnerCfg",
    },
)

gym.register(
    id="Moby-Deploy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": moby_env_cfg.MobyDeployEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
        "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachPPORunnerCfg",
    },
)

gym.register(
    id="Nami-Nav-Deploy",
    entry_point="isaac_neuromeka.env.tracking_viewer_env:TrackingViewerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": nami_env_cfg.NamiNavDeployEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
        "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachPPORunnerCfg",
    },
)

gym.register(
    id="Zen-Deploy",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": zen_env_cfg.ZenDeployEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
        "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachPPORunnerCfg",
    },
)
