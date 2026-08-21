from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions import JointAction, actions_cfg
from isaaclab.managers.action_manager import ActionTerm

# from isaac_neuromeka.env.rl_task_custom_env import CustomManagerBasedRLEnv, RLEnvWithIK
from isaac_neuromeka.env.rl_task_custom_env import RLEnvWithIK

if TYPE_CHECKING:
    from .action_cfgs import (
        ClampedJointActionCfg,
        CustomResidualJointActionCfg,
        MimicJointActionCfg,
        ReferenceResidualJointActionCfg,
        ResidualJointActionCfg,
    )


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

    def apply_actions(self, env_ids: Sequence[int] | None = None):
        super().apply_actions(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        target = self.processed_actions[env_ids][:, self._mimic_src_idx] + self._mimic_offset[env_ids]
        self._asset.set_joint_position_target(target, joint_ids=self._mimic_joint_ids, env_ids=env_ids)


class CustomResidualJointPositionAction(JointAction):
    """잔차(residual) 위치 액션 — 뉴로메카 예제 `JointResidualAction`과 같은 구조 (2026-07-23).

        processed = action * scale + offset      (offset 기본 0)
        joint_pos_target = 현재 관절각 + processed

    예제와 동일하게 `JointAction`을 직접 상속하고, 목표를 `_processed_actions`가 아니라 별도
    `joint_pos_target` 버퍼에 두며, `process_actions`에서 목표를 만들고 `apply_actions`는
    그 버퍼를 쓰기만 한다. 예제와 다른 곳은 아래 넷뿐이고 전부 예제 그대로 두면 이 태스크에서
    깨지거나 잘못 도는 부분이다.

    1. `joint_pos_target` 폭을 `num_joints`(26)가 아니라 **action_dim**(현재 18)으로 잡음.
       예제는 `torch.zeros_like(data.joint_pos)`(N,26)에 (N,18)을 더해 26관절 중 일부만
       제어하면 RuntimeError. (예제는 전 관절 제어 전제로 쓰였음)
    2. `apply_actions`의 `env_ids.unsqueeze(-1)` 제거. 예제는 `env_ids=None`(→`slice(None)`)일 때
       `slice.unsqueeze`로 AttributeError. 인덱스 브로드캐스트는 `set_joint_position_target`이
       이미 내부에서 처리함(articulation.py:1098-1099).
    3. `clamp_to_limits` 옵션 추가(기본 True) — 예제에는 없음. 잔차는 기준점이 실측 관절각이라
       목표가 limit을 넘어봐야 최대 `scale`이지만, 그만큼 관절 스토퍼를 계속 미는 토크
       (≈ kp × scale)가 남는다. clamp하면 그게 사라진다. 예제 그대로 가려면 False로 둘 것.
    4. `joint_position_lower_overrides` 옵션 추가(기본 None) — 선택한 관절의
       PD target에 task-local 하한을 더한다. raw/processed action의 부호는 바꾸지
       않으므로 굽힌 관절을 다시 펼 수 있다.

    ⚠ 절대형(`MimicJointPositionAction`)과 scale의 의미가 다르다.
       절대형: 기본자세로부터의 변위 상한 / 잔차형: **스텝당 증분**. implicit PD 기준 대략

           최대 관절속도  v ≈ (kp / kd) × scale   [rad/s]
           최대 유지토크  τ ≈ kp × scale

       즉 손가락 파지력이 scale에 직접 묶인다(막힌 관절의 정상상태 오차가 scale에서 멈춤).
       절대형은 막힌 관절의 오차가 최대 1 rad까지 벌어져 effort_limit(0.6)까지 포화했다.

    ⚠ `processed_actions`는 예제와 같이 **증분**이다(절대 목표가 아님). 반면
       `env/managers.py`의 `action_track_err`와 `scripts/rsl_rl/play.py --print_action`의
       target 열은 `processed_actions`를 절대 목표로 읽는다. 이 term을 배선할 때 그 두 곳이
       `joint_pos_target`을 읽도록 같이 고칠 것.
    """

    cfg: CustomResidualJointActionCfg

    def __init__(self, cfg: CustomResidualJointActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 예제와 동일하게 cfg.offset(기본 0.0)이 그대로 offset이 된다. default_joint_pos를
        # 더하지 않는 것이 절대형과의 핵심 차이.
        self.joint_pos_target = torch.zeros_like(self._raw_actions)
        if cfg.clamp_to_limits:
            # 한 번만 캐시. 랜덤화 이벤트가 관절 limit을 다시 쓰면 이 캐시도 갱신해야 함.
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            self._limit_lo = limits[..., 0].clone()
            self._limit_hi = limits[..., 1].clone()
        else:
            self._limit_lo = None
            self._limit_hi = None

        lower_overrides = cfg.joint_position_lower_overrides or {}
        if lower_overrides:
            unknown_joints = sorted(set(lower_overrides) - set(self._joint_names))
            if unknown_joints:
                raise ValueError(
                    "CustomResidualJointPositionAction lower-limit overrides contain "
                    f"joints outside this action term: {unknown_joints}. "
                    f"Resolved joints: {self._joint_names}"
                )

            # If full articulation-limit clamping is disabled, start from
            # unbounded limits and apply only the explicitly requested floors.
            if self._limit_lo is None:
                self._limit_lo = torch.full_like(self._raw_actions, -torch.inf)
                self._limit_hi = torch.full_like(self._raw_actions, torch.inf)

            for joint_name, configured_lower in lower_overrides.items():
                lower = float(configured_lower)
                if not math.isfinite(lower):
                    raise ValueError(
                        "CustomResidualJointPositionAction lower limit must be finite: "
                        f"{joint_name}={configured_lower}"
                    )
                action_index = self._joint_names.index(joint_name)
                if torch.any(lower > self._limit_hi[:, action_index]):
                    raise ValueError(
                        "CustomResidualJointPositionAction lower limit exceeds the "
                        f"upper limit for {joint_name}: lower={lower}"
                    )
                # Do not relax a stricter articulation limit. This option only
                # narrows the allowed target range.
                self._limit_lo[:, action_index] = torch.maximum(
                    self._limit_lo[:, action_index],
                    torch.tensor(lower, device=self.device, dtype=self._limit_lo.dtype),
                )

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset
        target = self._asset.data.joint_pos[:, self._joint_ids] + self._processed_actions
        if self._limit_lo is not None:
            target = torch.clamp(target, self._limit_lo, self._limit_hi)
        self.joint_pos_target = target

    def apply_actions(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._asset.set_joint_position_target(
            self.joint_pos_target[env_ids], joint_ids=self._joint_ids, env_ids=env_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)  # raw_actions = 0
        if env_ids is None:
            env_ids = slice(None)
        self.joint_pos_target[env_ids] = self._asset.data.joint_pos[:, self._joint_ids][env_ids]
        self._asset.set_joint_position_target(
            self.joint_pos_target[env_ids], joint_ids=self._joint_ids, env_ids=env_ids
        )


class ReferenceResidualJointPositionAction(CustomResidualJointPositionAction):
    """Fine residual control around a fixed preload-producing joint command."""

    cfg: ReferenceResidualJointActionCfg

    def __init__(self, cfg: ReferenceResidualJointActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reference = torch.as_tensor(
            cfg.reference_positions,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        if reference.numel() != self.action_dim:
            raise ValueError(
                "ReferenceResidualJointPositionAction expected "
                f"{self.action_dim} reference positions, got {reference.numel()}."
            )
        self._reference = reference.reshape(1, -1).expand(self.num_envs, -1)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset
        target = self._reference + self._processed_actions
        if self._limit_lo is not None:
            target = torch.clamp(target, self._limit_lo, self._limit_hi)
        self.joint_pos_target = target

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.joint_pos_target[env_ids] = self._reference[env_ids]
        self._asset.set_joint_position_target(
            self.joint_pos_target[env_ids],
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )


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
