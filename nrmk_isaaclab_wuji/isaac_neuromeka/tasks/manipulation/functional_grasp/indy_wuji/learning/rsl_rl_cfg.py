from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    max_iterations = 50000
    save_interval = 50
    experiment_name = "indy_wuji_chopsticks_grasp"
    run_name = ""
    resume = False
    empirical_normalization = False
    # 없으면 None -> action에 상한이 전혀 없음. 정책 출력이 그대로 관절 목표가 되어 발산함
    # (실측 |a| 평균 1.5, |Δa| 최대 9.66). action이 [-1,1] 안에서 의미를 갖도록 scale=1.0과 짝지음.
    clip_actions = 1.0
    # deprecated pre-4.0 field; scripts/rsl_rl/{train,play}.py call handle_deprecated_rsl_rl_cfg()
    # which derives `actor`/`critic` (with correct distribution_cfg) from this automatically.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
