from collections.abc import Sequence

import torch
from isaaclab.assets.articulation import Articulation, ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (  # noqa: F401
    axis_angle_from_quat,
    combine_frame_transforms,
    quat_conjugate,
    quat_error_magnitude,
    quat_mul,
)

from isaac_neuromeka.utils.history_buffer import HistoryBuffer


@configclass
class FiniteArticulationCfg(ArticulationCfg):
    history_len: int = 50


class FiniteArticulation(Articulation):

    cfg: FiniteArticulationCfg

    def __init__(self, cfg):
        super().__init__(cfg)
        self.joint_target_history = None
        self.joint_pos_history = None
        self.joint_vel_history = None

        self.history_len = self.cfg.history_len

    def reset(self, env_ids: Sequence[int] | slice | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._prevprev_joint_pos[env_ids] = 0
        self._prev_joint_pos[env_ids] = 0
        self._prevprev_finite_joint_vel[env_ids] = 0
        self._prev_finite_joint_vel[env_ids] = 0
        self._finite_joint_vel[env_ids] = 0
        self._op_state[env_ids] = 0

        self.buffer_filled[env_ids] = False

        new_body_state = self._data.body_state_w[env_ids]
        self._preprev_body_pose_w[env_ids] = new_body_state[:, :, :7]
        self._prev_body_pose_w[env_ids] = new_body_state[:, :, :7]
        self._finite_body_vel_w[env_ids] = new_body_state[:, :, 7:13]

        self.joint_target_history.reset(env_ids)
        self.joint_pos_history.reset(env_ids)
        self.joint_vel_history.reset(env_ids)

    def _create_buffers(self):
        super()._create_buffers()
        self._prevprev_joint_pos = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._prev_joint_pos = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._prevprev_finite_joint_vel = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._prev_finite_joint_vel = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._finite_joint_vel = torch.zeros(self.num_instances, self.num_joints, device=self.device)
        self._op_state = torch.zeros(self.num_instances, 1, device=self.device)  # used for demo

        self._preprev_body_pose_w = torch.zeros(self.num_instances, self.num_bodies, 3 + 4, device=self.device)
        self._prev_body_pose_w = torch.zeros(self.num_instances, self.num_bodies, 3 + 4, device=self.device)
        self._finite_body_vel_w = torch.zeros(self.num_instances, self.num_bodies, 6, device=self.device)

        self.joint_target_history = HistoryBuffer(
            self.num_instances, self.num_joints, max_len=self.history_len, device=self.device
        )
        self.joint_pos_history = HistoryBuffer(
            self.num_instances, self.num_joints, max_len=self.history_len, device=self.device
        )
        self.joint_vel_history = HistoryBuffer(
            self.num_instances, self.num_joints, max_len=self.history_len, device=self.device
        )

        self.buffer_filled = torch.zeros(self.num_instances, dtype=torch.bool, device=self.device)

    def write_joint_state_to_sim(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        joint_ids: Sequence[int] | slice | None = None,
        env_ids: Sequence[int] | slice | None = None,
    ):
        super().write_joint_state_to_sim(position, velocity, joint_ids, env_ids)
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)
        self._prevprev_joint_pos[env_ids, joint_ids] = position
        self._prev_joint_pos[env_ids, joint_ids] = position
        self._prevprev_finite_joint_vel[env_ids, joint_ids] = velocity
        self._prev_finite_joint_vel[env_ids, joint_ids] = velocity
        self._finite_joint_vel[env_ids, joint_ids] = velocity

    def update(self, dt: float):

        super().update(dt)

        self._finite_joint_vel[:] = (self._data.joint_pos - self._prev_joint_pos) / dt
        self._finite_body_vel_w[:, :, :3] = (self._data.body_state_w[:, :, :3] - self._prev_body_pose_w[:, :, :3]) / dt

        new_idx = ~self.buffer_filled
        self._finite_joint_vel[new_idx] = self._data.joint_vel[new_idx]
        self._finite_body_vel_w[new_idx, :, :3] = self._data.body_state_w[new_idx, :, 7:10]
        self.buffer_filled[new_idx] = True

        ## THIS IMPLEMENTATION OF ANGULAR VEL IS WRONG. TODO: FIX IT
        # Compute angular velocity using finite differences from quaternions
        # # Extract current and previous quaternions
        # quat_current = self._data.body_state_w[:, :, 3:7]  # [num_instances, num_bodies, 4]
        # quat_prev = self._prev_body_pose_w[:, :, 3:7]

        # q_rel = quat_mul(quat_current, quat_conjugate(quat_prev))     # (...,4)
        # axis_angle = axis_angle_from_quat(q_rel)
        # ang_vel = axis_angle / dt

        self._finite_body_vel_w[:, :, 3:] = self._data.body_state_w[:, :, 10:13]  # [num_instances, num_bodies, 3]

        # history buffers

        self.joint_pos_history(self._data.joint_pos[:])
        self.joint_vel_history(self._finite_joint_vel[:])
        self.joint_target_history(self._data.joint_pos_target)

        self._prevprev_joint_pos[:] = self._prev_joint_pos[:]
        self._prev_joint_pos[:] = self._data.joint_pos[:]
        self._prevprev_finite_joint_vel[:] = self._prev_finite_joint_vel[:]
        self._prev_finite_joint_vel[:] = self._finite_joint_vel[:]

        self._preprev_body_pose_w[:] = self._prev_body_pose_w[:]
        self._prev_body_pose_w[:] = self._data.body_state_w[:, :, :7]
