"""PPO scaffold for the Wuji hand-only two-stick environment."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HandGraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    obs_groups = {"policy": ["policy"], "actor": ["policy"], "critic": ["policy"]}
    max_iterations = 50000
    save_interval = 50
    experiment_name = "hand_grasp"
    run_name = ""
    resume = False
    empirical_normalization = False
    clip_actions = 1.0
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HandObjectGraspPPORunnerCfg(HandGraspPPORunnerCfg):
    """Separate log namespace for the object-pick scene."""

    experiment_name = "hand_grasp_object"


@configclass
class HandSettingPPORunnerCfg(HandGraspPPORunnerCfg):
    """Separate log namespace for the phase 2→3 hand-setting task."""

    experiment_name = "hand_setting"
