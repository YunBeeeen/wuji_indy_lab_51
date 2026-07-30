# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import pdb  # noqa:F401

# from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
# from isaaclab.managers.manager_term_cfg import RewardTermCfg
# from isaaclab.utils import configclass
# from dataclasses import MISSING
# import carb
# import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import CommandManager, CurriculumManager, TerminationManager

from isaac_neuromeka.env.managers import CustomActionManager, CustomRewardManager, SceneEntityCfg

# custom cfg
from isaac_neuromeka.env.rl_task_env_cfg import RLEnvWithIKCfg

# from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
# from isaaclab.managers import (
#     EventManager,
#     ObservationManager,
#     CommandManager,
#     CurriculumManager,
#     RewardManager,
#     TerminationManager,
#     RecorderManager
# )


"""
Environment with IK Solver
"""


from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms


class CustomManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-based RL env that logs both weighted and raw reward terms."""

    def load_managers(self):
        # note: this order follows IsaacLab's ManagerBasedRLEnv but swaps in CustomRewardManager.
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        ManagerBasedEnv.load_managers(self)

        # 표준 ActionManager를 CustomActionManager로 교체 — prevprev_action 추적(action_second_rate_l2용).
        # 표준 매니저 생성 직후 교체. action term init은 articulation에 write하지 않아 재생성 안전.
        self.action_manager = CustomActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager (custom, prevprev): ", self.action_manager)

        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)

        self.reward_manager = CustomRewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)

        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        self._configure_gym_env_spaces()

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")


# Only supports single body for now.
class RLEnvWithIK(ManagerBasedRLEnv):
    cfg: RLEnvWithIKCfg

    def __pre_manager_init__(self):

        self.ik_method = self.cfg.ik_method
        self.ik_body_name = self.cfg.ik_body_name
        self.ik_cmd_name = self.cfg.ik_cmd_name
        ik_params: dict[str, float] | None = {
            "pinv": {"k_val": 1.0},
            "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
            "trans": {"k_val": 1.0},
            "dls": {"lambda_val": 1.0},
        }

        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method=self.ik_method,
            ik_params=ik_params,
        )

        self.ik_solver = DifferentialIKController(diff_ik_cfg, num_envs=self.num_envs, device=self.device)

        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=[self.ik_body_name])
        robot_entity_cfg.resolve(self.scene)

        self.robot = self.scene[robot_entity_cfg.name]
        self.ee_idx = self.robot.find_bodies(self.ik_body_name)[0][0]
        num_joints = self.robot.num_joints

        self.ik_solution = torch.zeros((self.num_envs, num_joints), device=self.device)
        self.ik_solution_clip = torch.zeros((self.num_envs, num_joints), device=self.device)

        if self.robot.is_fixed_base:
            self.jacobi_idx = robot_entity_cfg.body_ids[0] - 1
        else:
            self.jacobi_idx = robot_entity_cfg.body_ids[0]

        self.joint_pos_target = torch.zeros((self.num_envs, num_joints), device=self.device)
        self.prev_joint_pos_target = torch.zeros((self.num_envs, num_joints), device=self.device)

    def _update_ik(self):
        self.ik_solver.reset()

        command = self.command_manager.get_term(self.ik_cmd_name).pose_command_b
        self.ik_solver.set_command(command)

        # obtain quantities from simulation
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.jacobi_idx, :, :]
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_idx, 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        joint_pos = self.robot.data.joint_pos[:, :]

        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        # compute the joint commands
        self.ik_solution = self.ik_solver.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # clip the ik solution
        diff = self.ik_solution - joint_pos
        diff = torch.clamp(diff, -0.1, 0.1)
        self.ik_solution_clip = joint_pos + diff

    def _pre_observation_compute_step(self):
        super()._pre_observation_compute_step()
        self._update_ik()
