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


# =============================================================================
# 큐브 운반 goal 커맨드 (2026-07-15)
# =============================================================================
import isaaclab.sim as _sim_utils  # noqa: E402
from isaaclab.managers import CommandTerm, CommandTermCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402


class UniformCubeGoalCommand(CommandTerm):
    """에피소드마다 큐브 운반 goal 위치를 env-로컬 박스에서 균일 샘플.

    command는 (N, 3) env-로컬 위치. 월드 좌표가 필요하면 env_origins를 더할 것 —
    기존 object_position_error_to_target이 이 보정을 빼먹어서 다중 env에서 goal 관측이
    env마다 다른 상수(사실상 잡음 채널)였던 것을 고치는 구현임 (2026-07-15 발견).
    resampling_time_range를 에피소드보다 길게 두면 리셋에서만 리샘플됨 (에피소드 내 고정
    -> 운반 차분 보상의 telescoping이 깨지지 않음).
    """

    cfg: "UniformCubeGoalCommandCfg"

    def __init__(self, cfg: "UniformCubeGoalCommandCfg", env):
        super().__init__(cfg, env)
        self.cube = env.scene[cfg.asset_name]
        self.goal_pos_e = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.goal_pos_e

    def _resample_command(self, env_ids: Sequence[int]):
        r = self.cfg.ranges
        for i, (lo, hi) in enumerate((r.pos_x, r.pos_y, r.pos_z)):
            self.goal_pos_e[env_ids, i] = (
                torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
            )

    def _update_command(self):
        pass

    def _update_metrics(self):
        goal_w = self._env.scene.env_origins + self.goal_pos_e
        self.metrics["error_pos"] = torch.norm(goal_w - self.cube.data.root_pos_w, dim=1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_marker_cfg)
            self.goal_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_visualizer.visualize(translations=self._env.scene.env_origins + self.goal_pos_e)


@configclass
class UniformCubeGoalCommandCfg(CommandTermCfg):
    class_type: type = UniformCubeGoalCommand

    asset_name: str = "cube"

    @configclass
    class Ranges:
        pos_x: tuple[float, float] = MISSING
        pos_y: tuple[float, float] = MISSING
        pos_z: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/cube_goal",
        markers={
            "sphere": _sim_utils.SphereCfg(
                radius=0.02,
                visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.3)),
            )
        },
    )
