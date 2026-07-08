import os
from dataclasses import MISSING
from typing import Literal
from isaaclab.utils import configclass

from isaac_neuromeka.learning.algorithm_cfg import NrmkPPOCfg, NrmkDistillationCfg, NrmkP3OCfg


@configclass
class NrmkRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    device: str = "cuda"
    runner_cls_name: str = MISSING  # Runner class name
    num_steps_per_env: int = MISSING # Number of steps per environment per update
    max_iterations: int = MISSING # Maximum number of iterations
    policy: str = MISSING # Path to the policy configuration file
    algorithm: NrmkPPOCfg = MISSING

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""
    
    neptune_project: str = "orbit"
    wandb_project: str = "orbit"

    ##
    # Loading parameters
    ##

    resume: bool = False
    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).
    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).
    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """
    

@configclass
class PPORunnerCfg(NrmkRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = ""
    wandb_project = ""
    logger = "wandb"
    
    runner_cls_name = "OnPolicyRunner"

    policy = f"{os.path.dirname(os.path.abspath(__file__))}/model/mlp.yaml"
    
    algorithm = NrmkPPOCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=False,
        clip_param=0.2,
        entropy_coef=0.007,
        num_learning_epochs=3,  # 5
        num_mini_batches=1,  # 4
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0
    )

    
@configclass
class DistillationRunnerCfg(NrmkRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = ""
    wandb_project = ""
    logger = "wandb"
    
    runner_cls_name = "OnPolicyDistillationRunner"

    policy = f"{os.path.dirname(os.path.abspath(__file__))}/model/student.yaml"
    teacher_policy = f"{os.path.dirname(os.path.abspath(__file__))}/model/teacher.yaml"
    
    # TODO: automate
    teacher_checkpoint_path = ""
    
    algorithm = NrmkDistillationCfg(
        num_learning_epochs=3,
        num_mini_batches=1,
        learning_rate=1e-3,
        learning_rate_decay=0.999,
        weight_decay=1e-10,
        max_grad_norm=1.0,
    )
    
@configclass
class P3ORunnerCfg(NrmkRlOnPolicyRunnerCfg):
    
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = ""
    wandb_project = ""
    logger = "wandb"
    
    policy = f"{os.path.dirname(os.path.abspath(__file__))}/model/cmdp.yaml"
    
    runner_cls_name = "OnPolicyRunnerCMDP"
    
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
        constraint_delay = 50, # Start cost optimization after N steps
        kappa_init = 0.5,
        kappa_max = 2.0,
        kappa_exp = 1.001, # exponentially growing kappa. 1.0: fixed
        cost_thresholds = 0.0, # or list: [0.0, 0.0, 1.0, ...]
        critic_learning_rate=5.0e-4,
        critic_learning_rate_decay=0.9999
    )
    
    
    
