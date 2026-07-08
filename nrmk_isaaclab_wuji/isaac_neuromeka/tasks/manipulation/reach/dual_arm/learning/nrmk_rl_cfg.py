import os

from isaaclab.utils import configclass

from isaac_neuromeka.learning.runner_cfg import PPORunnerCfg, DistillationRunnerCfg, P3ORunnerCfg
from isaac_neuromeka.learning.algorithm_cfg import NrmkPPOCfg, NrmkDistillationCfg, NrmkP3OCfg

@configclass
class ReachPPORunnerCfg(PPORunnerCfg):
    num_steps_per_env = 24
    experiment_name = "indy_reach"
    wandb_project = "indy_IK"
    logger = "wandb"
    
    policy = f"{os.path.dirname(os.path.abspath(__file__))}/model/teacher.yaml"
    
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
        max_grad_norm=1.0
    )


@configclass
class ReachTeacherPPORunnerCfg(PPORunnerCfg):
    policy = f"{os.path.dirname(os.path.abspath(__file__))}/model/teacher.yaml"
    
    experiment_name = "indy_reach_teacher"
    wandb_project = "indy_reach_teacher"
    
    algorithm = NrmkPPOCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=False,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=3,  # 5
        num_mini_batches=1,  # 4
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.97,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0
    )
    

    
@configclass
class DistillationRunnerCfg(DistillationRunnerCfg):
    experiment_name = "indy_student"
    wandb_project = "indy_student"
    logger = "wandb"

    policy = f"{os.path.dirname(os.path.abspath(__file__))}/model/student.yaml"
    teacher_policy = f"{os.path.dirname(os.path.abspath(__file__))}/model/teacher.yaml"
    
    # TODO: automate
    teacher_checkpoint_path = "/home/joonho/isaac_ws/orbit.neuromeka/isaac_neuromeka/logs/nrmk_rl/indy_reach_teacher/2024-06-21_17-08-44/model_500.pt" 
    
    algorithm = NrmkDistillationCfg(
        num_learning_epochs=3,
        num_mini_batches=1,
        learning_rate=1e-3,
        learning_rate_decay=0.999,
        weight_decay=1e-10,
        max_grad_norm=1.0,
    )
    
    
@configclass
class ReachP3ORunnerCfg(P3ORunnerCfg):
    
    experiment_name = "indy_reach_cmdp"
    wandb_project = "indy_reach_cmdp"
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
        constraint_delay = 50, # Start cost optimization after N steps
        kappa_init = 1.0,
        kappa_max = 2.0,
        kappa_exp = 1.001, # exponentially growing kappa. 1.0: fixed
        cost_thresholds = 0.0, # or list: [0.0, 0.0, 1.0, ...]
        critic_learning_rate=2.5e-4,
        critic_learning_rate_decay=1.0
    )
    