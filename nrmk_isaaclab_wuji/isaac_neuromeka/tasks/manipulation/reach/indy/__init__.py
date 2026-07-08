import gymnasium as gym

from . import env_cfg, learning

##
# Register Gym environments.
##

gym.register(
    id="Indy-Reach",
    # entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.Indy7ReachEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
        "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachPPORunnerCfg",
    },
)


## To be released in the future.

# gym.register(
#     id="Indy-Reach-Teacher",
#     entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": env_cfg.Indy7ReachTeacherEnvCfg,
#         "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachTeacherPPORunnerCfg",
#     },
# )

# gym.register(
#     id="Indy-Reach-Student",
#     entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": env_cfg.Indy7ReachStudentEnvCfg,
#         "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:DistillationRunnerCfg",
#     },
# )

# gym.register(
#     id="Indy-Reach-CMDP",
#     entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": env_cfg.Indy7ReachCMDPEnvCfg,
#         "nrmk_rl_cfg_entry_point": f"{learning.__name__}.nrmk_rl_cfg:ReachP3ORunnerCfg",
#     },
# )
