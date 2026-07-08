import pdb  # noqa:F401
from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique
from pynput.keyboard import Key

from isaac_neuromeka.utils.helper import KeyboardListener


@configclass
class DefaultUniformPoseCommandCfg(UniformPoseCommandCfg):
    default_ee_pose: list = MISSING


class KeyboardPoseCommand(UniformPoseCommand):
    class DeltaCommand:
        pos_x: float = MISSING
        pos_y: float = MISSING
        pos_z: float = MISSING
        yaw: float = MISSING

    def __init__(self, cfg: DefaultUniformPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.keyboard_listener = KeyboardListener(
            key_targets=[Key.left, Key.up, Key.right, Key.down, "a", "w", "d", "s"]
        )

        n_bins = 20.0
        self.delta_command = self.DeltaCommand()
        self.delta_command.pos_x = (self.cfg.ranges.pos_x[1] - self.cfg.ranges.pos_x[0]) / n_bins
        self.delta_command.pos_y = (self.cfg.ranges.pos_y[1] - self.cfg.ranges.pos_y[0]) / n_bins
        self.delta_command.pos_z = (self.cfg.ranges.pos_z[1] - self.cfg.ranges.pos_z[0]) / n_bins
        self.delta_command.yaw = (self.cfg.ranges.yaw[1] - self.cfg.ranges.yaw[0]) / n_bins

        self.euler_angles = torch.zeros_like(self.pose_command_b[:, :3])

    def _resample_command(self, env_ids: Sequence[int]):
        # initialize to zero command
        # -- position
        self.pose_command_b[env_ids, 0] = self.cfg.default_ee_pose[0]
        self.pose_command_b[env_ids, 1] = self.cfg.default_ee_pose[1]
        self.pose_command_b[env_ids, 2] = self.cfg.default_ee_pose[2]
        # -- orientation
        if len(self.cfg.default_ee_pose) == 6:
            self.euler_angles[env_ids, 0] = self.cfg.default_ee_pose[3]
            self.euler_angles[env_ids, 1] = self.cfg.default_ee_pose[4]
            self.euler_angles[env_ids, 2] = self.cfg.default_ee_pose[5]
        quat = quat_from_euler_xyz(
            self.euler_angles[env_ids, 0], self.euler_angles[env_ids, 1], self.euler_angles[env_ids, 2]
        )
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        keyboard_data = self.keyboard_listener.get_key_states()
        if keyboard_data["updated"]:
            # update position
            if keyboard_data["value"][Key.left]:
                self.pose_command_b[:, 0] -= self.delta_command.pos_x
            if keyboard_data["value"][Key.right]:
                self.pose_command_b[:, 0] += self.delta_command.pos_x
            if keyboard_data["value"][Key.up]:
                self.pose_command_b[:, 1] -= self.delta_command.pos_y
            if keyboard_data["value"][Key.down]:
                self.pose_command_b[:, 1] += self.delta_command.pos_y
            if keyboard_data["value"]["w"]:
                self.pose_command_b[:, 2] += self.delta_command.pos_z
            if keyboard_data["value"]["s"]:
                self.pose_command_b[:, 2] -= self.delta_command.pos_z

            self.pose_command_b[:, :3] = torch.clamp(
                self.pose_command_b[:, :3],
                min=torch.tensor(
                    [self.cfg.ranges.pos_x[0], self.cfg.ranges.pos_y[0], self.cfg.ranges.pos_z[0]],
                    device=self.pose_command_b.device,
                ),
                max=torch.tensor(
                    [self.cfg.ranges.pos_x[1], self.cfg.ranges.pos_y[1], self.cfg.ranges.pos_z[1]],
                    device=self.pose_command_b.device,
                ),
            )

            # update orientation
            if keyboard_data["value"]["a"]:
                self.euler_angles[:, 2] -= self.delta_command.yaw
            if keyboard_data["value"]["d"]:
                self.euler_angles[:, 2] += self.delta_command.yaw

            self.euler_angles[:, 2] = torch.clamp(
                self.euler_angles[:, 2], min=self.cfg.ranges.yaw[0], max=self.cfg.ranges.yaw[1]
            )

            quat = quat_from_euler_xyz(self.euler_angles[:, 0], self.euler_angles[:, 1], self.euler_angles[:, 2])
            self.pose_command_b[:, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat


class EmptyPoseCommand(UniformPoseCommand):
    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass
