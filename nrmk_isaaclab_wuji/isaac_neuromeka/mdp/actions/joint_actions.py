from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions import JointAction, actions_cfg
from isaaclab.managers.action_manager import ActionTerm

# from isaac_neuromeka.env.rl_task_custom_env import CustomManagerBasedRLEnv, RLEnvWithIK
from isaac_neuromeka.env.rl_task_custom_env import RLEnvWithIK

if TYPE_CHECKING:
    from .action_cfgs import ClampedJointActionCfg, ResidualJointActionCfg


class FixedJointPositionAction(ActionTerm):
    """Zero-dimension action term that holds selected joints at their default positions."""

    cfg: actions_cfg.JointPositionActionCfg

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._processed_actions = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._export_IO_descriptor = False

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.to(self.device)
        self._processed_actions = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        target = self._asset.data.default_joint_pos[:, self._joint_ids][env_ids]
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids, env_ids=env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self.apply_actions(env_ids=env_ids)


class CustomJointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedRLEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        else:
            self._offset = torch.zeros_like(self._asset.data.default_joint_pos[:, self._joint_ids])  # TODO (remove)

    def apply_actions(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        target = self.processed_actions[env_ids, :]

        # if self._joint_ids is not None and self._joint_ids != slice(None):
        #         env_ids = env_ids.unsqueeze(-1)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids, env_ids=env_ids)

    # def reset(self, env_ids: Sequence[int] | None = None) -> None:
    #     if env_ids is None:
    #         env_ids = slice(None)
    #     target = self._offset[env_ids, :]

    #     if self._joint_ids is not None and self._joint_ids != slice(None):
    #         env_ids = env_ids.unsqueeze(-1)
    #     self._asset.set_joint_position_target(target, joint_ids=self._joint_ids, env_ids=env_ids)  # TODO: fix required


class JointResidualAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: ResidualJointActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ResidualJointActionCfg, env: ManagerBasedRLEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # self.joint_target_history = HistoryBuffer(env.num_envs,
        #                                           self._num_joints,
        #                                           max_len=5,
        #                                           device=self.device)

        self.joint_pos_target = torch.zeros_like(self._asset.data.joint_pos)

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        self.joint_pos_target = self._asset.data.joint_pos + self._processed_actions

        # self.joint_target_history(self.joint_pos_target)

    def apply_actions(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        if self._joint_ids is not None and self._joint_ids != slice(None):
            env_ids = env_ids.unsqueeze(-1)

        self._asset.set_joint_position_target(
            self.joint_pos_target[env_ids], joint_ids=self._joint_ids, env_ids=env_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self.joint_pos_target[env_ids] = self._asset.data.joint_pos[env_ids, self._joint_ids]

        if self._joint_ids is not None and self._joint_ids != slice(None):
            env_ids = env_ids.unsqueeze(-1)
        self._asset.set_joint_position_target(
            self.joint_pos_target[env_ids], joint_ids=self._joint_ids, env_ids=env_ids
        )  # TODO: fix required


class IKResidualAction(JointResidualAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    _env: RLEnvWithIK
    cfg: ResidualJointActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ResidualJointActionCfg, env: RLEnvWithIK):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset

        cmd_name = cfg.cmd_name
        self.command_term = env.command_manager.get_term(cmd_name)
        self.command_manager = env.command_manager

    def apply_actions(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        if self._joint_ids is not None and self._joint_ids != slice(None):
            env_ids = env_ids.unsqueeze(-1)

        residual_action = self.processed_actions
        # # joint_pos_des = self.command_term.ik_solution[env_ids] + residual_action
        # joint_pos_des = self.command_term.ik_solution[env_ids]

        joint_pos_des = self._env.ik_solution_clip + residual_action

        self._asset.set_joint_position_target(joint_pos_des[env_ids], joint_ids=self._joint_ids, env_ids=env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # target = self._offset[env_ids, :]
        joint_pos = self._asset.data.joint_pos[env_ids, self._joint_ids]

        if self._joint_ids is not None and self._joint_ids != slice(None):
            env_ids = env_ids.unsqueeze(-1)
        self._asset.set_joint_position_target(
            joint_pos, joint_ids=self._joint_ids, env_ids=env_ids
        )  # TODO: fix required


class JointVelocityAction(JointAction):
    cfg: actions_cfg.JointActionCfg

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.joint_vel_target = torch.zeros_like(self._asset.data.joint_vel)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset
        self.joint_vel_target = self._processed_actions

    def apply_actions(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        self._asset.set_joint_velocity_target(
            self.joint_vel_target[env_ids], joint_ids=self._joint_ids, env_ids=env_ids
        )


class ClampedJointPositionAction(CustomJointPositionAction):
    cfg: ClampedJointActionCfg

    def __init__(self, cfg: ClampedJointActionCfg, env: ManagerBasedRLEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        self.range = cfg.clamp_range

    def apply_actions(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        target = self.processed_actions[env_ids, :]
        target = torch.clamp(target, self.range[0], self.range[1])
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids, env_ids=env_ids)
