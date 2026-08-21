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


@configclass
class FingerReachPPORunnerCfg(HandSettingPPORunnerCfg):
    """진단용 finger_reach 전용 로그 네임스페이스(2026-08-04)."""

    experiment_name = "finger_reach"


@configclass
class HandSetting0731PPORunnerCfg(HandSettingPPORunnerCfg):
    """07-31 baseline 격리(07-31 보상 + 현재 kp/kd) 전용 로그 네임스페이스(2026-08-05)."""

    experiment_name = "hand_setting_0731"


@configclass
class HandMovePPORunnerCfg(HandGraspPPORunnerCfg):
    """floating-root hand_move 전용 로그 네임스페이스(2026-08-05).

    하이퍼파라미터는 검증된 hand_grasp 러너를 그대로 상속하고 로그 경로만 분리한다.
    """

    experiment_name = "hand_move"


@configclass
class HandRealPPORunnerCfg(HandMovePPORunnerCfg):
    """Separate namespace for the 105D quaternion-history real observation."""

    experiment_name = "hand_real"


@configclass
class HandObjectPPORunnerCfg(HandMovePPORunnerCfg):
    """hand_object fine-tuning 전용 로그 네임스페이스(2026-08-06).

    hand_move 러너를 상속하되 experiment_name 만 분리한다. 같은 이름을 쓰면
    logs/rsl_rl/hand_move/ 아래에 섞여 들어가 어느 런이 어느 태스크인지 구분이
    안 되고, --init_checkpoint 로 hand_move 체크포인트를 고를 때도 헷갈린다.

    네트워크 구조(actor/critic hidden dims, activation)와 obs_groups 는 절대
    바꾸지 않는다 — 바꾸는 순간 hand_move 체크포인트가 shape mismatch 로
    로드되지 않아 fine-tuning 자체가 불가능해진다.
    """

    experiment_name = "hand_object"

    # learning_rate 하나만 다르지만 블록 전체를 다시 쓴다.
    # configclass 가 클래스 속성을 dataclass 필드로 바꿔서 `HandMovePPORunnerCfg.algorithm`
    # 접근 자체가 AttributeError 이고, `__post_init__` 에서 self.algorithm.learning_rate 를
    # 고치는 것은 더 나쁘다 — configclass 는 사용자 __post_init__ 을 먼저 실행하고 그 "다음에"
    # 멤버를 deepcopy 하므로(configclass.py:93, _custom_post_init) 아직 공유 상태인 기본
    # 객체를 건드리게 되어 HandGrasp/HandMove/HandSetting 의 LR 까지 같이 바뀐다.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=8,
        num_mini_batches=4,
        # 1e-3 -> 3e-4. schedule 이 "adaptive" 라 desired_kl 에 맞춰 곧 재조정되지만,
        # 초기 몇 번의 큰 gradient 가 기존 파지 능력부터 망가뜨리는 것을 피한다.
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HandFinalPPORunnerCfg(HandObjectPPORunnerCfg):
    """hand_object fine-tuning settings in a separate hand_final namespace."""

    experiment_name = "hand_final"
