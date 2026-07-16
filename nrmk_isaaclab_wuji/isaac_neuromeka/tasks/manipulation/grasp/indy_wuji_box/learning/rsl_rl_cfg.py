"""Box-Transport용 PPO 러너 설정. grasp/indy_wuji/learning/rsl_rl_cfg.py의 사본 (2026-07-16).

experiment_name을 분리해 로그/체크포인트 폴더가 큐브 태스크와 섞이지 않게 함
(과거 run 섞임 -> checkpoint 선택 사고 이력).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BoxTransportPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    max_iterations = 50000
    save_interval = 50
    experiment_name = "indy_wuji_box_transport"
    run_name = ""
    resume = False
    empirical_normalization = False
    # action 상한 (scale=1.0과 짝. 없으면 발산 — 큐브 태스크 실측)
    clip_actions = 1.0
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
