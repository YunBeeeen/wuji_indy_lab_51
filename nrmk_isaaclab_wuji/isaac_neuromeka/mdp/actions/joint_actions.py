from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions import JointAction, actions_cfg
from isaaclab.managers.action_manager import ActionTerm

# from isaac_neuromeka.env.rl_task_custom_env import CustomManagerBasedRLEnv, RLEnvWithIK
from isaac_neuromeka.env.rl_task_custom_env import RLEnvWithIK

if TYPE_CHECKING:
    from .action_cfgs import ClampedJointActionCfg, MimicJointActionCfg, ResidualJointActionCfg


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


class MimicJointPositionAction(CustomJointPositionAction):
    """커플링 위치 액션: 액션에 없는 follower 관절이 액션(source) 관절의 목표를 따라간다.

    논문 하드웨어(Schunk SIH)의 기계식 관절 커플링을 흉내냄 — 제어 DOF를 늘리지 않고
    감싸는 손가락을 늘림. follower는 action_dim/관측에 포함되지 않으며, apply 시점에
    follower 목표 = source 목표 + (follower 기본자세 - source 기본자세) 로 복사받는다.

    부수 효과: 리셋이 관절 '상태'만 되돌리고 목표 버퍼를 안 채워서 액션 밖 관절이
    저절로 움직이던 문제(2026-07-14, hold_joints_at_default 참고)가 follower에 한해
    리셋~첫 액션 사이 한 스텝으로 줄어든다.
    """

    cfg: MimicJointActionCfg

    def __init__(self, cfg: MimicJointActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not cfg.mimic:
            raise ValueError(
                "MimicJointPositionAction: cfg.mimic이 비어 있음. CustomJointPositionAction을 쓸 것."
            )
        follower_ids, follower_names = self._asset.find_joints(
            list(cfg.mimic.keys()), preserve_order=True
        )
        overlap = set(follower_names) & set(self._joint_names)
        if overlap:
            raise ValueError(f"mimic follower가 액션 관절에도 있음: {sorted(overlap)}")

        src_indices: list[int] = []
        for name in follower_names:
            src = cfg.mimic[name]
            if src not in self._joint_names:
                raise ValueError(
                    f"mimic source '{src}' (follower '{name}')가 액션 관절 목록에 없음: {self._joint_names}"
                )
            src_indices.append(self._joint_names.index(src))

        self._mimic_joint_ids = follower_ids
        self._mimic_src_idx = torch.tensor(src_indices, dtype=torch.long, device=self.device)
        action_joint_ids = self._joint_ids
        if isinstance(action_joint_ids, slice):
            action_joint_ids = list(range(self._asset.num_joints))
        src_joint_ids = [action_joint_ids[i] for i in src_indices]
        default = self._asset.data.default_joint_pos
        self._mimic_offset = default[:, follower_ids] - default[:, src_joint_ids]

        # 관절 '목표' 클램프 (물리 한계는 그대로 -> 접촉에 밀리는 수동 순응성 유지).
        # 목적: 음수 목표로 손가락이 과신전 되어 벌어진 채 바닥에 박히는 실패 모드 차단
        # (2026-07-14 실측: kp 2로는 바닥 접촉을 못 이기고 못 접힌 채 에피소드 종료).
        # process_actions에서 클램프하므로 mimic follower도 클램프된 목표를 복사받음.
        self._clamp_active = bool(cfg.target_clamp)
        if self._clamp_active:
            n = len(self._joint_names)
            self._clamp_min = torch.full((n,), -torch.inf, device=self.device)
            self._clamp_max = torch.full((n,), torch.inf, device=self.device)
            matched = {expr: False for expr in cfg.target_clamp}
            for j, name in enumerate(self._joint_names):
                for expr, (lo, hi) in cfg.target_clamp.items():
                    if re.fullmatch(expr, name):
                        if lo is not None:
                            self._clamp_min[j] = lo
                        if hi is not None:
                            self._clamp_max[j] = hi
                        matched[expr] = True
            unmatched = [e for e, m in matched.items() if not m]
            if unmatched:
                raise ValueError(f"target_clamp 패턴이 액션 관절에 하나도 안 맞음: {unmatched}")

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        if self._clamp_active:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clamp_min, max=self._clamp_max
            )

    def apply_actions(self, env_ids: Sequence[int] | None = None):
        super().apply_actions(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        target = self.processed_actions[env_ids][:, self._mimic_src_idx] + self._mimic_offset[env_ids]
        self._asset.set_joint_position_target(target, joint_ids=self._mimic_joint_ids, env_ids=env_ids)


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
