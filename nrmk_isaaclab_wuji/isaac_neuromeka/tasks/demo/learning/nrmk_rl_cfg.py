import os

from isaaclab.utils import configclass

from isaac_neuromeka.learning.algorithm_cfg import NrmkP3OCfg, NrmkPPOCfg
from isaac_neuromeka.learning.runner_cfg import P3ORunnerCfg, PPORunnerCfg


@configclass
class ReachPPORunnerCfg(PPORunnerCfg):
    num_steps_per_env = 24
    experiment_name = "demo"
    wandb_project = "demo"
    logger = "wandb"

    policy = f"{os.path.dirname(os.path.abspath(__file__))}/model/cmdp.yaml"

    algorithm = NrmkPPOCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=False,
        clip_param=0.2,
        entropy_coef=0.007,
        num_learning_epochs=3,  # 5
        num_mini_batches=1,  # 4
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ReachP3ORunnerCfg(P3ORunnerCfg):

    experiment_name = "demo"
    wandb_project = "demo"
    logger = "wandb"

    policy = f"{os.path.dirname(os.path.abspath(__file__))}/model/cmdp.yaml"

    algorithm = NrmkP3OCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=False,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=3,
        num_mini_batches=1,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        constraint_delay=50,  # Start cost optimization after N steps
        kappa_init=1.0,
        kappa_max=2.0,
        kappa_exp=1.001,  # exponentially growing kappa. 1.0: fixed
        cost_thresholds=0.0,  # or list: [0.0, 0.0, 1.0, ...]
        critic_learning_rate=2.5e-4,
        critic_learning_rate_decay=1.0,
    )
