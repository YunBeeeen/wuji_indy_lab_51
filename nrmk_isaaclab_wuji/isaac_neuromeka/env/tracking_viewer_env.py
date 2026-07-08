from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.utils import math as math_utils


class YawTrackingViewportCameraController(ViewportCameraController):
    """Viewport controller that rotates camera offsets with the tracked asset yaw."""

    def __init__(self, env: ManagerBasedRLEnv, cfg):
        self.viewer_quat: torch.Tensor | None = None
        super().__init__(env, cfg)

    def update_view_to_world(self):
        self.viewer_quat = None
        super().update_view_to_world()

    def update_view_to_env(self):
        self.viewer_quat = None
        super().update_view_to_env()

    def update_view_to_asset_root(self, asset_name: str):
        if self.cfg.asset_name != asset_name:
            asset_entities = [*self._env.scene.rigid_objects.keys(), *self._env.scene.articulations.keys()]
            if asset_name not in asset_entities:
                raise ValueError(f"Asset '{asset_name}' is not in the scene. Available entities: {asset_entities}.")
        self.cfg.asset_name = asset_name
        self.cfg.origin_type = "asset_root"
        asset = self._env.scene[self.cfg.asset_name]
        self.viewer_origin = asset.data.root_pos_w[self.cfg.env_index]
        self.viewer_quat = asset.data.root_quat_w[self.cfg.env_index].view(1, 4)
        self.update_view_location()

    def update_view_to_asset_body(self, asset_name: str, body_name: str):
        if self.cfg.asset_name != asset_name:
            asset_entities = [*self._env.scene.rigid_objects.keys(), *self._env.scene.articulations.keys()]
            if asset_name not in asset_entities:
                raise ValueError(f"Asset '{asset_name}' is not in the scene. Available entities: {asset_entities}.")
        asset = self._env.scene[asset_name]
        if body_name not in asset.body_names:
            raise ValueError(
                f"'{body_name}' is not a body of Asset '{asset_name}'. Available bodies: {asset.body_names}."
            )
        body_id, _ = asset.find_bodies(body_name)
        self.cfg.asset_name = asset_name
        self.cfg.body_name = body_name
        self.cfg.origin_type = "asset_body"
        self.viewer_origin = asset.data.body_pos_w[self.cfg.env_index, body_id].view(3)
        self.viewer_quat = asset.data.body_quat_w[self.cfg.env_index, body_id].view(1, 4)
        self.update_view_location()

    def update_view_location(self, eye: Sequence[float] | None = None, lookat: Sequence[float] | None = None):
        if eye is not None:
            self.default_cam_eye = np.asarray(eye)
        if lookat is not None:
            self.default_cam_lookat = np.asarray(lookat)

        viewer_origin = self.viewer_origin.detach().cpu().numpy()
        if self.viewer_quat is not None and self.cfg.origin_type in {"asset_root", "asset_body"}:
            quat = self.viewer_quat
            cam_eye_offset = torch.as_tensor(self.default_cam_eye, device=quat.device, dtype=quat.dtype).view(1, 3)
            cam_target_offset = torch.as_tensor(self.default_cam_lookat, device=quat.device, dtype=quat.dtype).view(
                1, 3
            )
            cam_eye = viewer_origin + math_utils.quat_apply_yaw(quat, cam_eye_offset)[0].detach().cpu().numpy()
            cam_target = viewer_origin + math_utils.quat_apply_yaw(quat, cam_target_offset)[0].detach().cpu().numpy()
        else:
            cam_eye = viewer_origin + self.default_cam_eye
            cam_target = viewer_origin + self.default_cam_lookat

        self._env.sim.set_camera_view(eye=cam_eye, target=cam_target)


class TrackingViewerEnv(ManagerBasedRLEnv):
    """Manager-based env that swaps in the yaw-tracking viewport controller for robot."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        if self.viewport_camera_controller is not None:
            del self.viewport_camera_controller
            self.viewport_camera_controller = YawTrackingViewportCameraController(self, self.cfg.viewer)
