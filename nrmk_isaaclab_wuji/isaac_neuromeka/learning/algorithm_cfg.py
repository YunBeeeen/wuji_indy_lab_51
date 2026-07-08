from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class NrmkPPOCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

@configclass
class NrmkCMDPAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "P3O"
    """The algorithm class name. Default is P3O."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""
    
    
    # use_clipped_cost_loss: bool = MISSING
    # """Whether to use clipped cost loss."""
    constraint_delay: int = MISSING

    
@configclass
class NrmkDistillationCfg:
    class_name: str = "PolicyDistillation"
    num_learning_epochs: int = 1
    num_mini_batches: int = 1
    learning_rate: float = 1e-3
    learning_rate_decay: float = 0.999
    weight_decay: float = 1e-10
    max_grad_norm: float = 1.0
    latent_loss_coeff: float = 0.5
    

@configclass
class NrmkP3OCfg(NrmkCMDPAlgorithmCfg):
    critic_learning_rate: float = MISSING
    critic_learning_rate_decay: float = MISSING
    kappa_init: float | list[float] = MISSING
    kappa_max: float | list[float] = MISSING
    kappa_exp: float = MISSING    
    cost_thresholds: float | list[float] = MISSING
    """cost threshold for CMDP algorithms (Epsilon).""" 
    