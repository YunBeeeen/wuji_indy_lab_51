"""MDP terms for the hand-only two-stick functional pre-grasp task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import (
    CommandTerm,
    CommandTermCfg,
    ManagerTermBase,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_mul,
    subtract_frame_transforms,
)


class OpenCloseModeCommand(CommandTerm):
    """Alternate OPEN/CLOSE within an episode after a balanced initial sample.

    The command is a two-element one-hot vector:
    ``[1, 0]`` for OPEN and ``[0, 1]`` for CLOSE.
    """

    cfg: "OpenCloseModeCommandCfg"

    def __init__(self, cfg: "OpenCloseModeCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self.metrics["open_fraction"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["close_fraction"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        current = self._command[env_ids]
        initialized = torch.sum(current, dim=-1) > 0.5
        initial_open = (
            torch.rand(len(env_ids), device=self.device)
            < self.cfg.open_probability
        )
        open_mode = torch.where(
            initialized,
            current[:, 1] > 0.5,
            initial_open,
        )
        self._command[env_ids, 0] = open_mode.float()
        self._command[env_ids, 1] = (~open_mode).float()

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        self.metrics["open_fraction"][:] = self._command[:, 0]
        self.metrics["close_fraction"][:] = self._command[:, 1]


@configclass
class OpenCloseModeCommandCfg(CommandTermCfg):
    """Configuration for the per-episode OPEN/CLOSE command."""

    class_type: type = OpenCloseModeCommand
    open_probability: float = 0.5


def reset_hand_joint_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    hand_cfg: SceneEntityCfg,
    joint_positions: tuple[float, ...],
) -> None:
    """Reset a hand to a measured/open state with target equal to state.

    No contact preload is injected: the position target starts at the same
    joint position written into PhysX.  Any sustained grasp force must
    therefore be produced by the policy after reset.
    """
    robot: Articulation = env.scene[hand_cfg.name]
    joint_pos = torch.as_tensor(
        joint_positions,
        device=env.device,
        dtype=robot.data.joint_pos.dtype,
    ).expand(env_ids.numel(), -1)
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=hand_cfg.joint_ids,
        env_ids=env_ids,
    )
    robot.set_joint_position_target(
        joint_pos,
        joint_ids=hand_cfg.joint_ids,
        env_ids=env_ids,
    )
    robot.set_joint_velocity_target(
        joint_vel,
        joint_ids=hand_cfg.joint_ids,
        env_ids=env_ids,
    )


def reset_to_functional_pregrasp(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    hand_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    joint_positions: tuple[float, ...],
    stick1_position_p: tuple[float, float, float],
    stick1_quaternion_p: tuple[float, float, float, float],
    stick2_position_p: tuple[float, float, float],
    stick2_quaternion_p: tuple[float, float, float, float],
) -> None:
    """Restore the validated in-hand state without injecting a PD preload."""
    robot: Articulation = env.scene[hand_cfg.name]
    stick1: RigidObject = env.scene[stick1_cfg.name]
    stick2: RigidObject = env.scene[stick2_cfg.name]

    count = env_ids.numel()
    device = env.device
    joint_pos = torch.as_tensor(joint_positions, device=device).expand(count, -1)
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        joint_ids=hand_cfg.joint_ids,
        env_ids=env_ids,
    )
    robot.set_joint_position_target(
        joint_pos,
        joint_ids=hand_cfg.joint_ids,
        env_ids=env_ids,
    )
    robot.set_joint_velocity_target(
        joint_vel,
        joint_ids=hand_cfg.joint_ids,
        env_ids=env_ids,
    )

    palm_id = palm_cfg.body_ids[0]
    palm_pos_w = robot.data.body_pos_w[env_ids, palm_id]
    palm_quat_w = robot.data.body_quat_w[env_ids, palm_id]

    stick1_pos_p = torch.as_tensor(
        stick1_position_p,
        device=device,
        dtype=palm_pos_w.dtype,
    ).expand(count, -1)
    stick2_pos_p = torch.as_tensor(
        stick2_position_p,
        device=device,
        dtype=palm_pos_w.dtype,
    ).expand(count, -1)
    stick1_quat_p = torch.as_tensor(
        stick1_quaternion_p,
        device=device,
        dtype=palm_pos_w.dtype,
    ).expand(count, -1)
    stick2_quat_p = torch.as_tensor(
        stick2_quaternion_p,
        device=device,
        dtype=palm_pos_w.dtype,
    ).expand(count, -1)

    stick1_pose_w = torch.cat(
        (
            palm_pos_w + quat_apply(palm_quat_w, stick1_pos_p),
            quat_mul(palm_quat_w, stick1_quat_p),
        ),
        dim=-1,
    )
    stick2_pose_w = torch.cat(
        (
            palm_pos_w + quat_apply(palm_quat_w, stick2_pos_p),
            quat_mul(palm_quat_w, stick2_quat_p),
        ),
        dim=-1,
    )
    zero_velocity = torch.zeros(
        (count, 6),
        device=device,
        dtype=palm_pos_w.dtype,
    )
    stick1.write_root_pose_to_sim(stick1_pose_w, env_ids=env_ids)
    stick1.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    stick2.write_root_pose_to_sim(stick2_pose_w, env_ids=env_ids)
    stick2.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def reset_object_between_stick_tips(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    support_cfg: SceneEntityCfg,
    stick1_position_p: tuple[float, float, float],
    stick1_quaternion_p: tuple[float, float, float, float],
    stick2_position_p: tuple[float, float, float],
    stick2_quaternion_p: tuple[float, float, float, float],
    stick1_tip_offset_o: tuple[float, float, float],
    stick2_tip_offset_o: tuple[float, float, float],
    object_size: tuple[float, float, float],
    support_height: float,
) -> None:
    """Place a dynamic object at the reference distal-tip midpoint.

    The support is a narrow world-vertical kinematic post whose top face meets
    the object's bottom face.  Object pose is intentionally not added to the
    policy observation or reward in this environment-only stage.
    """
    robot: Articulation = env.scene[palm_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    support: RigidObject = env.scene[support_cfg.name]

    count = env_ids.numel()
    device = env.device
    palm_id = palm_cfg.body_ids[0]
    palm_pos_w = robot.data.body_pos_w[env_ids, palm_id]
    palm_quat_w = robot.data.body_quat_w[env_ids, palm_id]
    dtype = palm_pos_w.dtype

    def reference_tip_p(
        position_p: tuple[float, float, float],
        quaternion_p: tuple[float, float, float, float],
        tip_offset_o: tuple[float, float, float],
    ) -> torch.Tensor:
        position = torch.as_tensor(
            position_p,
            device=device,
            dtype=dtype,
        ).expand(count, -1)
        quaternion = torch.as_tensor(
            quaternion_p,
            device=device,
            dtype=dtype,
        ).expand(count, -1)
        tip_offset = torch.as_tensor(
            tip_offset_o,
            device=device,
            dtype=dtype,
        ).expand(count, -1)
        return position + quat_apply(quaternion, tip_offset)

    tip1_p = reference_tip_p(
        stick1_position_p,
        stick1_quaternion_p,
        stick1_tip_offset_o,
    )
    tip2_p = reference_tip_p(
        stick2_position_p,
        stick2_quaternion_p,
        stick2_tip_offset_o,
    )
    object_position_w = palm_pos_w + quat_apply(
        palm_quat_w,
        0.5 * (tip1_p + tip2_p),
    )

    identity_quaternion = torch.zeros(
        (count, 4),
        device=device,
        dtype=dtype,
    )
    identity_quaternion[:, 0] = 1.0
    object_pose_w = torch.cat(
        (object_position_w, identity_quaternion),
        dim=-1,
    )

    support_position_w = object_position_w.clone()
    support_position_w[:, 2] -= 0.5 * (
        float(object_size[2]) + float(support_height)
    )
    support_pose_w = torch.cat(
        (support_position_w, identity_quaternion),
        dim=-1,
    )
    zero_velocity = torch.zeros(
        (count, 6),
        device=device,
        dtype=dtype,
    )

    support.write_root_pose_to_sim(support_pose_w, env_ids=env_ids)
    support.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
    obj.write_root_pose_to_sim(object_pose_w, env_ids=env_ids)
    obj.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def _sensor_force(env: ManagerBasedEnv, sensor_name: str) -> torch.Tensor:
    """Return the summed filtered contact-force magnitude for each environment."""
    force_matrix = env.scene.sensors[sensor_name].data.force_matrix_w
    if force_matrix is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.linalg.vector_norm(force_matrix, dim=-1).sum(dim=(-1, -2))


def _group_forces(
    env: ManagerBasedEnv,
    sensor_groups: tuple[tuple[str, ...], ...],
) -> torch.Tensor:
    """Return one force per semantic contact group, using OR within a group."""
    groups = []
    for sensor_names in sensor_groups:
        forces = torch.stack(
            [_sensor_force(env, sensor_name) for sensor_name in sensor_names],
            dim=-1,
        )
        groups.append(torch.max(forces, dim=-1).values)
    return torch.stack(groups, dim=-1)


def _object_pose_in_palm(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one rigid object's palm-frame position and quaternion."""
    robot: Articulation = env.scene[palm_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    palm_id = palm_cfg.body_ids[0]
    return subtract_frame_transforms(
        robot.data.body_pos_w[:, palm_id],
        robot.data.body_quat_w[:, palm_id],
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )


def _object_pair_speeds_relative_to_palm(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return each stick's linear/angular speed relative to the moving palm."""
    robot: Articulation = env.scene[palm_cfg.name]
    stick1: RigidObject = env.scene[stick1_cfg.name]
    stick2: RigidObject = env.scene[stick2_cfg.name]
    palm_id = palm_cfg.body_ids[0]

    palm_pos_w = robot.data.body_pos_w[:, palm_id]
    palm_lin_vel_w = robot.data.body_lin_vel_w[:, palm_id]
    palm_ang_vel_w = robot.data.body_ang_vel_w[:, palm_id]

    def relative_speeds(obj: RigidObject) -> tuple[torch.Tensor, torch.Tensor]:
        offset_w = obj.data.root_pos_w - palm_pos_w
        relative_lin_vel_w = (
            obj.data.root_lin_vel_w
            - palm_lin_vel_w
            - torch.cross(palm_ang_vel_w, offset_w, dim=-1)
        )
        relative_ang_vel_w = obj.data.root_ang_vel_w - palm_ang_vel_w
        return (
            torch.linalg.vector_norm(relative_lin_vel_w, dim=-1),
            torch.linalg.vector_norm(relative_ang_vel_w, dim=-1),
        )

    stick1_linear_speed, stick1_angular_speed = relative_speeds(stick1)
    stick2_linear_speed, stick2_angular_speed = relative_speeds(stick2)
    return (
        stick1_linear_speed,
        stick2_linear_speed,
        stick1_angular_speed,
        stick2_angular_speed,
    )


def _object_point_in_palm(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    point_o: tuple[float, float, float],
) -> torch.Tensor:
    """Return an object-local point expressed in the palm frame."""
    position_p, quaternion_p = _object_pose_in_palm(
        env,
        palm_cfg,
        object_cfg,
    )
    point = torch.as_tensor(
        point_o,
        dtype=position_p.dtype,
        device=position_p.device,
    ).expand(env.num_envs, -1)
    return position_p + quat_apply(quaternion_p, point)


def _tip_geometry_from_palm_poses(
    stick1_pos_p: torch.Tensor,
    stick1_quat_p: torch.Tensor,
    stick2_pos_p: torch.Tensor,
    stick2_quat_p: torch.Tensor,
    stick1_tip_offset_o: tuple[float, float, float],
    stick2_tip_offset_o: tuple[float, float, float],
    stick_thickness: float,
    reference_separation_direction_stick2: (
        tuple[float, float, float] | torch.Tensor | None
    ) = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return distal surface gap, lateral error, and Stick2-axis offset.

    Local ``+y`` remains the semantic distal direction.  Stick2 is the
    stabilized reference rail, so the center delta is projected off its current
    shaft axis.  Each square cross-section's support radius is then projected
    onto the instantaneous transverse separation direction, accounting for
    cross-section rotation without relying on a fixed palm/opening normal.

    The optional lateral error measures only the component perpendicular to the
    validated Stick1-to-Stick2 separation direction expressed in Stick2 local
    coordinates.  Axial slip along local ``+y`` is excluded.
    """
    dtype = stick1_pos_p.dtype
    device = stick1_pos_p.device
    num_envs = stick1_pos_p.shape[0]

    stick1_tip_offset = torch.as_tensor(
        stick1_tip_offset_o,
        dtype=dtype,
        device=device,
    ).expand(num_envs, -1)
    stick2_tip_offset = torch.as_tensor(
        stick2_tip_offset_o,
        dtype=dtype,
        device=device,
    ).expand(num_envs, -1)
    stick1_tip_p = stick1_pos_p + quat_apply(
        stick1_quat_p,
        stick1_tip_offset,
    )
    stick2_tip_p = stick2_pos_p + quat_apply(
        stick2_quat_p,
        stick2_tip_offset,
    )

    local_x = torch.tensor((1.0, 0.0, 0.0), dtype=dtype, device=device).expand(
        num_envs, -1
    )
    local_y = torch.tensor((0.0, 1.0, 0.0), dtype=dtype, device=device).expand(
        num_envs, -1
    )
    local_z = torch.tensor((0.0, 0.0, 1.0), dtype=dtype, device=device).expand(
        num_envs, -1
    )
    stick1_x_p = quat_apply(stick1_quat_p, local_x)
    stick1_z_p = quat_apply(stick1_quat_p, local_z)
    stick2_x_p = quat_apply(stick2_quat_p, local_x)
    stick2_y_p = quat_apply(stick2_quat_p, local_y)
    stick2_z_p = quat_apply(stick2_quat_p, local_z)

    tip_delta_p = stick1_tip_p - stick2_tip_p
    axial_delta = torch.sum(
        tip_delta_p * stick2_y_p,
        dim=-1,
        keepdim=True,
    )
    transverse_delta_p = tip_delta_p - axial_delta * stick2_y_p
    transverse_distance = torch.linalg.vector_norm(
        transverse_delta_p,
        dim=-1,
    )
    separation_direction_p = transverse_delta_p / torch.clamp(
        transverse_distance.unsqueeze(-1),
        min=1.0e-8,
    )

    half_thickness = 0.5 * stick_thickness
    stick1_support = half_thickness * (
        torch.abs(torch.sum(separation_direction_p * stick1_x_p, dim=-1))
        + torch.abs(torch.sum(separation_direction_p * stick1_z_p, dim=-1))
    )
    stick2_support = half_thickness * (
        torch.abs(torch.sum(separation_direction_p * stick2_x_p, dim=-1))
        + torch.abs(torch.sum(separation_direction_p * stick2_z_p, dim=-1))
    )
    surface_gap = torch.clamp(
        transverse_distance - stick1_support - stick2_support,
        min=0.0,
    )

    if reference_separation_direction_stick2 is None:
        lateral_error = torch.zeros_like(surface_gap)
    else:
        reference_direction = torch.as_tensor(
            reference_separation_direction_stick2,
            dtype=dtype,
            device=device,
        )
        reference_xz = reference_direction[[0, 2]]
        reference_xz = reference_xz / torch.clamp(
            torch.linalg.vector_norm(reference_xz),
            min=1.0e-8,
        )
        tip_delta_stick2 = quat_apply_inverse(
            stick2_quat_p,
            tip_delta_p,
        )
        # In the 2-D cross-section plane, (-z, x) is perpendicular to
        # the validated separation direction (x, z).
        lateral_error = torch.abs(
            -reference_xz[1] * tip_delta_stick2[:, 0]
            + reference_xz[0] * tip_delta_stick2[:, 2]
        )

    return surface_gap, lateral_error, axial_delta.squeeze(-1)


def _tip_surface_gap(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick1_tip_offset_o: tuple[float, float, float],
    stick2_tip_offset_o: tuple[float, float, float],
    stick_thickness: float,
) -> torch.Tensor:
    """Return the transverse surface gap between the distal cross-sections."""
    stick1_pos_p, stick1_quat_p = _object_pose_in_palm(
        env,
        palm_cfg,
        stick1_cfg,
    )
    stick2_pos_p, stick2_quat_p = _object_pose_in_palm(
        env,
        palm_cfg,
        stick2_cfg,
    )
    surface_gap, _, _ = _tip_geometry_from_palm_poses(
        stick1_pos_p,
        stick1_quat_p,
        stick2_pos_p,
        stick2_quat_p,
        stick1_tip_offset_o,
        stick2_tip_offset_o,
        stick_thickness,
    )
    return surface_gap


def _tip_surface_gap_and_lateral_error(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick1_tip_offset_o: tuple[float, float, float],
    stick2_tip_offset_o: tuple[float, float, float],
    stick_thickness: float,
    reference_separation_direction_stick2: tuple[float, float, float]
    | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gap and sideways miss using one shared pose calculation."""
    stick1_pos_p, stick1_quat_p = _object_pose_in_palm(
        env,
        palm_cfg,
        stick1_cfg,
    )
    stick2_pos_p, stick2_quat_p = _object_pose_in_palm(
        env,
        palm_cfg,
        stick2_cfg,
    )
    surface_gap, lateral_error, _ = _tip_geometry_from_palm_poses(
        stick1_pos_p,
        stick1_quat_p,
        stick2_pos_p,
        stick2_quat_p,
        stick1_tip_offset_o,
        stick2_tip_offset_o,
        stick_thickness,
        reference_separation_direction_stick2,
    )
    return surface_gap, lateral_error


def _pose_errors(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    reference_position: torch.Tensor,
    reference_quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Euclidean position error and sign-invariant quaternion angle."""
    position_error = torch.linalg.vector_norm(
        position - reference_position,
        dim=-1,
    )
    quaternion_dot = torch.abs(
        torch.sum(quaternion * reference_quaternion, dim=-1)
    )
    orientation_error = 2.0 * torch.acos(
        torch.clamp(quaternion_dot, min=0.0, max=1.0)
    )
    return position_error, orientation_error


class JointReferenceTracking(ManagerTermBase):
    """Bounded joint-state prior around a measured functional grasp."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reference = torch.as_tensor(
            cfg.params["reference_joint_positions"],
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        reference_joint_positions: tuple[float, ...],
        sigma: float = 0.20,
    ) -> torch.Tensor:
        del reference_joint_positions
        robot: Articulation = env.scene[asset_cfg.name]
        joint_error = (
            robot.data.joint_pos[:, asset_cfg.joint_ids] - self._reference
        )
        return torch.exp(
            -torch.mean(torch.square(joint_error / sigma), dim=-1)
        )


class ObjectReferencePoseTracking(ManagerTermBase):
    """Bounded palm-frame pose tracking for one dynamic stick."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reference_position = torch.as_tensor(
            cfg.params["reference_position_p"],
            device=env.device,
        )
        self._reference_quaternion = torch.as_tensor(
            cfg.params["reference_quaternion_p"],
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        palm_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        reference_position_p: tuple[float, float, float],
        reference_quaternion_p: tuple[float, float, float, float],
        position_sigma: float = 0.01,
        orientation_sigma: float = 0.1745329252,
    ) -> torch.Tensor:
        del reference_position_p, reference_quaternion_p
        position, quaternion = _object_pose_in_palm(
            env,
            palm_cfg,
            object_cfg,
        )
        position_error, orientation_error = _pose_errors(
            position,
            quaternion,
            self._reference_position,
            self._reference_quaternion,
        )
        return torch.exp(
            -position_error / position_sigma
            -orientation_error / orientation_sigma
        )


class ObjectReferencePoseDeviation(ManagerTermBase):
    """Bounded palm-frame position or orientation deviation penalty."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reference_position = torch.as_tensor(
            cfg.params["reference_position_p"],
            device=env.device,
        )
        self._reference_quaternion = torch.as_tensor(
            cfg.params["reference_quaternion_p"],
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        palm_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        reference_position_p: tuple[float, float, float],
        reference_quaternion_p: tuple[float, float, float, float],
        component: str,
        position_sigma: float = 0.005,
        orientation_sigma: float = 0.1745329252,
    ) -> torch.Tensor:
        del reference_position_p, reference_quaternion_p
        position, quaternion = _object_pose_in_palm(
            env,
            palm_cfg,
            object_cfg,
        )
        position_error, orientation_error = _pose_errors(
            position,
            quaternion,
            self._reference_position,
            self._reference_quaternion,
        )
        if component == "position":
            return 1.0 - torch.exp(-position_error / position_sigma)
        if component == "orientation":
            return 1.0 - torch.exp(
                -orientation_error / orientation_sigma
            )
        raise ValueError(f"Unsupported pose-deviation component: {component}")


class ObjectPointReferenceTracking(ManagerTermBase):
    """Track one object-local support point instead of freezing the full pose."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        reference_position = torch.as_tensor(
            cfg.params["reference_position_p"],
            device=env.device,
        )
        reference_quaternion = torch.as_tensor(
            cfg.params["reference_quaternion_p"],
            device=env.device,
        )
        point = torch.as_tensor(
            cfg.params["point_o"],
            device=env.device,
        )
        self._reference_point_p = reference_position + quat_apply(
            reference_quaternion.unsqueeze(0),
            point.unsqueeze(0),
        )[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        palm_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        point_o: tuple[float, float, float],
        reference_position_p: tuple[float, float, float],
        reference_quaternion_p: tuple[float, float, float, float],
        sigma: float = 0.01,
    ) -> torch.Tensor:
        del reference_position_p, reference_quaternion_p
        point_p = _object_point_in_palm(
            env,
            palm_cfg,
            object_cfg,
            point_o,
        )
        error = torch.linalg.vector_norm(
            point_p - self._reference_point_p,
            dim=-1,
        )
        return torch.exp(-error / sigma)


class ModeTipGapTracking(ManagerTermBase):
    """Track the requested gap only while contacts and Stick2 anchor hold."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stick2_reference_position = torch.as_tensor(
            cfg.params["stick2_reference_position_p"],
            device=env.device,
        )
        self._stick2_reference_quaternion = torch.as_tensor(
            cfg.params["stick2_reference_quaternion_p"],
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        mode_index: int,
        target_gap: float,
        sensor_groups: tuple[tuple[str, ...], ...],
        palm_cfg: SceneEntityCfg,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        stick1_tip_offset_o: tuple[float, float, float],
        stick2_tip_offset_o: tuple[float, float, float],
        stick_thickness: float,
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        sigma: float = 0.005,
        contact_force_scale: float = 0.02,
        anchor_position_sigma: float = 0.005,
        anchor_orientation_sigma: float = 0.1745329252,
    ) -> torch.Tensor:
        del (
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
        )
        mode_gate = env.command_manager.get_command(command_name)[:, mode_index]
        contact_gate = torch.min(
            torch.clamp(
                _group_forces(env, sensor_groups) / contact_force_scale,
                min=0.0,
                max=1.0,
            ),
            dim=-1,
        ).values
        stick1_position, stick1_quaternion = _object_pose_in_palm(
            env,
            palm_cfg,
            stick1_cfg,
        )
        stick2_position, stick2_quaternion = _object_pose_in_palm(
            env,
            palm_cfg,
            stick2_cfg,
        )
        gap, _, _ = _tip_geometry_from_palm_poses(
            stick1_position,
            stick1_quaternion,
            stick2_position,
            stick2_quaternion,
            stick1_tip_offset_o,
            stick2_tip_offset_o,
            stick_thickness,
        )
        position_error, orientation_error = _pose_errors(
            stick2_position,
            stick2_quaternion,
            self._stick2_reference_position,
            self._stick2_reference_quaternion,
        )
        anchor_gate = torch.exp(
            -position_error / anchor_position_sigma
            -orientation_error / anchor_orientation_sigma
        )
        return (
            mode_gate
            * contact_gate
            * anchor_gate
            * torch.exp(-torch.abs(gap - target_gap) / sigma)
        )


def mode_tip_gap_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    mode_index: int,
    target_gap: float,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick1_tip_offset_o: tuple[float, float, float],
    stick2_tip_offset_o: tuple[float, float, float],
    stick_thickness: float,
    reference_separation_direction_stick2: tuple[float, float, float],
    reference_axial_offset_stick2: float,
    sigma: float = 0.005,
    lateral_sigma: float = 0.005,
    axial_sigma: float = 0.005,
) -> torch.Tensor:
    """Track mode gap plus transverse and axial distal-tip alignment."""
    mode_gate = env.command_manager.get_command(command_name)[:, mode_index]
    stick1_position, stick1_quaternion = _object_pose_in_palm(
        env,
        palm_cfg,
        stick1_cfg,
    )
    stick2_position, stick2_quaternion = _object_pose_in_palm(
        env,
        palm_cfg,
        stick2_cfg,
    )
    gap, lateral_error, axial_offset = _tip_geometry_from_palm_poses(
        stick1_position,
        stick1_quaternion,
        stick2_position,
        stick2_quaternion,
        stick1_tip_offset_o,
        stick2_tip_offset_o,
        stick_thickness,
        reference_separation_direction_stick2,
    )
    axial_error = torch.abs(
        axial_offset - float(reference_axial_offset_stick2)
    )
    return mode_gate * torch.exp(
        -torch.abs(gap - target_gap) / sigma
        - lateral_error / lateral_sigma
        - axial_error / axial_sigma
    )


def tip_lateral_deviation(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick1_tip_offset_o: tuple[float, float, float],
    stick2_tip_offset_o: tuple[float, float, float],
    stick_thickness: float,
    reference_separation_direction_stick2: tuple[float, float, float],
    lateral_sigma: float = 0.005,
) -> torch.Tensor:
    """Return a bounded penalty for sideways distal-tip misalignment.

    This term is deliberately independent of contact and OPEN/CLOSE mode.
    Breaking contact therefore cannot make a large lateral deviation free.
    """
    _, lateral_error = _tip_surface_gap_and_lateral_error(
        env,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        stick1_tip_offset_o,
        stick2_tip_offset_o,
        stick_thickness,
        reference_separation_direction_stick2,
    )
    return 1.0 - torch.exp(-lateral_error / lateral_sigma)


def contact_group_strength(
    env: ManagerBasedEnv,
    sensor_groups: tuple[tuple[str, ...], ...],
    force_scale: float = 0.10,
    reduction: str = "mean",
) -> torch.Tensor:
    """Dense contact strength; multiple sensors in a group implement logical OR."""
    strengths = torch.clamp(
        _group_forces(env, sensor_groups) / force_scale,
        min=0.0,
        max=1.0,
    )
    if reduction == "mean":
        return torch.mean(strengths, dim=-1)
    if reduction == "min":
        return torch.min(strengths, dim=-1).values
    raise ValueError(f"Unsupported contact reduction: {reduction}")


def gated_contact_group_strength(
    env: ManagerBasedEnv,
    target_groups: tuple[tuple[str, ...], ...],
    gate_groups: tuple[tuple[str, ...], ...],
    force_scale: float = 0.10,
    gate_threshold: float = 0.02,
) -> torch.Tensor:
    """Reward target contacts only while every prerequisite contact is valid."""
    target_strength = torch.mean(
        torch.clamp(
            _group_forces(env, target_groups) / force_scale,
            min=0.0,
            max=1.0,
        ),
        dim=-1,
    )
    gate_valid = torch.all(
        _group_forces(env, gate_groups) >= gate_threshold,
        dim=-1,
    )
    return target_strength * gate_valid.float()


def body_box_surface_proximity(
    env: ManagerBasedEnv,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    sigma: float = 0.02,
) -> torch.Tensor:
    """Dense exponential proximity from one body origin to an oriented box surface."""
    robot: Articulation = env.scene[body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, body_cfg.body_ids[0]]
    rel_o = quat_apply_inverse(
        obj.data.root_quat_w,
        body_pos_w - obj.data.root_pos_w,
    )
    half_extent = torch.as_tensor(
        object_half_extent,
        dtype=rel_o.dtype,
        device=rel_o.device,
    )
    outside = torch.clamp(torch.abs(rel_o) - half_extent, min=0.0)
    surface_distance = torch.linalg.vector_norm(outside, dim=-1)
    return torch.exp(-surface_distance / sigma)


def body_box_shaft_region_proximity(
    env: ManagerBasedEnv,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    long_axis: int = 1,
    axial_half_length: float = 0.075,
    sigma: float = 0.02,
) -> torch.Tensor:
    """Approach shaping for the correct link and the usable stick shaft.

    The score is one near the oriented box and inside the central axial
    interval.  It does not prescribe one surface normal or one exact contact
    point, so nearby valid contacts remain interchangeable.
    """
    robot: Articulation = env.scene[body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, body_cfg.body_ids[0]]
    rel_o = quat_apply_inverse(
        obj.data.root_quat_w,
        body_pos_w - obj.data.root_pos_w,
    )
    half_extent = torch.as_tensor(
        object_half_extent,
        dtype=rel_o.dtype,
        device=rel_o.device,
    )
    outside = torch.clamp(torch.abs(rel_o) - half_extent, min=0.0)
    surface_distance = torch.linalg.vector_norm(outside, dim=-1)
    axial_excess = torch.clamp(
        torch.abs(rel_o[:, long_axis]) - axial_half_length,
        min=0.0,
    )
    region_distance = torch.sqrt(
        torch.square(surface_distance) + torch.square(axial_excess)
    )
    return torch.exp(-region_distance / sigma)


def _body_in_box_shaft_region(
    env: ManagerBasedEnv,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    long_axis: int,
    axial_half_length: float,
) -> torch.Tensor:
    """Return whether one body origin lies inside the allowed axial interval."""
    robot: Articulation = env.scene[body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, body_cfg.body_ids[0]]
    rel_o = quat_apply_inverse(
        obj.data.root_quat_w,
        body_pos_w - obj.data.root_pos_w,
    )
    return torch.abs(rel_o[:, long_axis]) <= axial_half_length


def _setting_geometry(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    thumb_distal_cfg: SceneEntityCfg,
    index_tip_cfg: SceneEntityCfg,
    middle_tip_cfg: SceneEntityCfg,
    ring_tip_cfg: SceneEntityCfg,
    stick1_reference_position_p: tuple[float, float, float],
    stick1_reference_quaternion_p: tuple[float, float, float, float],
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    long_axis: int,
    axial_half_length: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return final stick pose errors and the four narrow region checks."""
    stick1_position, stick1_quaternion = _object_pose_in_palm(
        env,
        palm_cfg,
        stick1_cfg,
    )
    stick2_position, stick2_quaternion = _object_pose_in_palm(
        env,
        palm_cfg,
        stick2_cfg,
    )
    stick1_ref_position = torch.as_tensor(
        stick1_reference_position_p,
        device=env.device,
        dtype=stick1_position.dtype,
    )
    stick1_ref_quaternion = torch.as_tensor(
        stick1_reference_quaternion_p,
        device=env.device,
        dtype=stick1_quaternion.dtype,
    )
    stick2_ref_position = torch.as_tensor(
        stick2_reference_position_p,
        device=env.device,
        dtype=stick2_position.dtype,
    )
    stick2_ref_quaternion = torch.as_tensor(
        stick2_reference_quaternion_p,
        device=env.device,
        dtype=stick2_quaternion.dtype,
    )
    stick1_position_error, stick1_orientation_error = _pose_errors(
        stick1_position,
        stick1_quaternion,
        stick1_ref_position,
        stick1_ref_quaternion,
    )
    stick2_position_error, stick2_orientation_error = _pose_errors(
        stick2_position,
        stick2_quaternion,
        stick2_ref_position,
        stick2_ref_quaternion,
    )
    region_valid = (
        _body_in_box_shaft_region(
            env,
            thumb_distal_cfg,
            stick1_cfg,
            long_axis,
            axial_half_length,
        )
        & _body_in_box_shaft_region(
            env,
            index_tip_cfg,
            stick1_cfg,
            long_axis,
            axial_half_length,
        )
        & _body_in_box_shaft_region(
            env,
            middle_tip_cfg,
            stick1_cfg,
            long_axis,
            axial_half_length,
        )
        & _body_in_box_shaft_region(
            env,
            ring_tip_cfg,
            stick2_cfg,
            long_axis,
            axial_half_length,
        )
    )
    return (
        stick1_position_error,
        stick1_orientation_error,
        stick2_position_error,
        stick2_orientation_error,
        region_valid,
    )


def setting_completion_strength(
    env: ManagerBasedEnv,
    sensor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    thumb_distal_cfg: SceneEntityCfg,
    index_tip_cfg: SceneEntityCfg,
    middle_tip_cfg: SceneEntityCfg,
    ring_tip_cfg: SceneEntityCfg,
    stick1_reference_position_p: tuple[float, float, float],
    stick1_reference_quaternion_p: tuple[float, float, float, float],
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    force_scale: float = 0.10,
    position_sigma: float = 0.02,
    orientation_sigma: float = 0.3490658504,
    long_axis: int = 1,
    axial_half_length: float = 0.08,
) -> torch.Tensor:
    """Completion shaping: all six contacts, final pose, and valid shaft regions."""
    contact_min = torch.min(
        torch.clamp(
            _group_forces(env, sensor_groups) / force_scale,
            min=0.0,
            max=1.0,
        ),
        dim=-1,
    ).values
    (
        stick1_position_error,
        stick1_orientation_error,
        stick2_position_error,
        stick2_orientation_error,
        region_valid,
    ) = _setting_geometry(
        env,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        thumb_distal_cfg,
        index_tip_cfg,
        middle_tip_cfg,
        ring_tip_cfg,
        stick1_reference_position_p,
        stick1_reference_quaternion_p,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        long_axis,
        axial_half_length,
    )
    pose_gate = torch.exp(
        -(stick1_position_error + stick2_position_error) / position_sigma
        -(stick1_orientation_error + stick2_orientation_error)
        / orientation_sigma
    )
    return contact_min * pose_gate * region_valid.float()


def setting_grasp_stability(
    env: ManagerBasedEnv,
    sensor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    thumb_distal_cfg: SceneEntityCfg,
    index_tip_cfg: SceneEntityCfg,
    middle_tip_cfg: SceneEntityCfg,
    ring_tip_cfg: SceneEntityCfg,
    stick1_reference_position_p: tuple[float, float, float],
    stick1_reference_quaternion_p: tuple[float, float, float, float],
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    force_scale: float = 0.10,
    position_sigma: float = 0.02,
    orientation_sigma: float = 0.3490658504,
    linear_speed_scale: float = 0.10,
    angular_speed_scale: float = 2.0,
    long_axis: int = 1,
    axial_half_length: float = 0.08,
) -> torch.Tensor:
    """Reward a quiet final setting without suppressing the approach motion."""
    completion = setting_completion_strength(
        env,
        sensor_groups,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        thumb_distal_cfg,
        index_tip_cfg,
        middle_tip_cfg,
        ring_tip_cfg,
        stick1_reference_position_p,
        stick1_reference_quaternion_p,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        force_scale,
        position_sigma,
        orientation_sigma,
        long_axis,
        axial_half_length,
    )
    (
        stick1_linear_speed,
        stick2_linear_speed,
        stick1_angular_speed,
        stick2_angular_speed,
    ) = _object_pair_speeds_relative_to_palm(
        env,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
    )
    quiet = torch.exp(
        -torch.maximum(stick1_linear_speed, stick2_linear_speed)
        / linear_speed_scale
        -torch.maximum(stick1_angular_speed, stick2_angular_speed)
        / angular_speed_scale
    )
    return completion * quiet


def object_pair_angular_speed_excess_l2(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    angular_speed_limit: float = 3.0,
    max_excess: float = 10.0,
) -> torch.Tensor:
    """Squared excess of the faster palm-relative stick angular speed."""
    _, _, stick1_angular_speed, stick2_angular_speed = (
        _object_pair_speeds_relative_to_palm(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
        )
    )
    angular_speed = torch.maximum(
        stick1_angular_speed,
        stick2_angular_speed,
    )
    excess = torch.clamp(
        angular_speed - angular_speed_limit,
        min=0.0,
        max=max_excess,
    )
    return torch.square(excess)


def full_grasp_stability(
    env: ManagerBasedEnv,
    sensor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    force_scale: float = 0.10,
    contact_threshold: float = 0.02,
    linear_speed_scale: float = 0.10,
    angular_speed_scale: float = 2.0,
) -> torch.Tensor:
    """Reward quiet two-stick motion only while every contact group is engaged."""
    group_forces = _group_forces(env, sensor_groups)
    contacts_valid = torch.all(
        group_forces >= contact_threshold,
        dim=-1,
    )
    contact_gate = torch.min(
        torch.clamp(
            group_forces / force_scale,
            min=0.0,
            max=1.0,
        ),
        dim=-1,
    ).values
    (
        stick1_linear_speed,
        stick2_linear_speed,
        stick1_angular_speed,
        stick2_angular_speed,
    ) = _object_pair_speeds_relative_to_palm(
        env,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
    )
    linear_speed = torch.maximum(
        stick1_linear_speed,
        stick2_linear_speed,
    )
    angular_speed = torch.maximum(
        stick1_angular_speed,
        stick2_angular_speed,
    )
    quiet = torch.exp(
        -linear_speed / linear_speed_scale
        -angular_speed / angular_speed_scale
    )
    return contacts_valid.float() * contact_gate * quiet


class ModeGraspStability(ManagerTermBase):
    """Reward quiet mode tracking only while Stick2 remains the fixed rail."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stick2_reference_position = torch.as_tensor(
            cfg.params["stick2_reference_position_p"],
            device=env.device,
        )
        self._stick2_reference_quaternion = torch.as_tensor(
            cfg.params["stick2_reference_quaternion_p"],
            device=env.device,
        )
        self._reference_separation_direction_stick2 = torch.as_tensor(
            cfg.params["reference_separation_direction_stick2"],
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_groups: tuple[tuple[str, ...], ...],
        palm_cfg: SceneEntityCfg,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        stick1_tip_offset_o: tuple[float, float, float],
        stick2_tip_offset_o: tuple[float, float, float],
        stick_thickness: float,
        open_target_gap: float,
        close_target_gap: float,
        reference_separation_direction_stick2: tuple[float, float, float],
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        gap_sigma: float = 0.005,
        lateral_sigma: float = 0.005,
        anchor_position_sigma: float = 0.005,
        anchor_orientation_sigma: float = 0.1745329252,
        force_scale: float = 0.10,
        contact_threshold: float = 0.02,
        linear_speed_scale: float = 0.10,
        angular_speed_scale: float = 2.0,
    ) -> torch.Tensor:
        del (
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
            reference_separation_direction_stick2,
        )
        base_stability = full_grasp_stability(
            env,
            sensor_groups,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            force_scale,
            contact_threshold,
            linear_speed_scale,
            angular_speed_scale,
        )
        command = env.command_manager.get_command(command_name)
        target_gap = (
            command[:, 0] * open_target_gap
            + command[:, 1] * close_target_gap
        )
        stick1_position, stick1_quaternion = _object_pose_in_palm(
            env,
            palm_cfg,
            stick1_cfg,
        )
        stick2_position, stick2_quaternion = _object_pose_in_palm(
            env,
            palm_cfg,
            stick2_cfg,
        )
        tip_gap, lateral_error, _ = _tip_geometry_from_palm_poses(
            stick1_position,
            stick1_quaternion,
            stick2_position,
            stick2_quaternion,
            stick1_tip_offset_o,
            stick2_tip_offset_o,
            stick_thickness,
            self._reference_separation_direction_stick2,
        )
        mode_gate = torch.exp(
            -torch.abs(tip_gap - target_gap) / gap_sigma
            - lateral_error / lateral_sigma
        )
        position_error, orientation_error = _pose_errors(
            stick2_position,
            stick2_quaternion,
            self._stick2_reference_position,
            self._stick2_reference_quaternion,
        )
        anchor_gate = torch.exp(
            -position_error / anchor_position_sigma
            -orientation_error / anchor_orientation_sigma
        )
        return base_stability * mode_gate * anchor_gate


def mode_grasp_stability(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick1_tip_offset_o: tuple[float, float, float],
    stick2_tip_offset_o: tuple[float, float, float],
    stick_thickness: float,
    open_target_gap: float,
    close_target_gap: float,
    reference_separation_direction_stick2: tuple[float, float, float],
    reference_axial_offset_stick2: float,
    gap_sigma: float = 0.005,
    lateral_sigma: float = 0.005,
    axial_sigma: float = 0.005,
    force_scale: float = 0.10,
    contact_threshold: float = 0.02,
    linear_speed_scale: float = 0.10,
    angular_speed_scale: float = 2.0,
) -> torch.Tensor:
    """Reward quiet six-contact gap, lateral, and axial mode tracking."""
    base_stability = full_grasp_stability(
        env,
        sensor_groups,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        force_scale,
        contact_threshold,
        linear_speed_scale,
        angular_speed_scale,
    )
    command = env.command_manager.get_command(command_name)
    target_gap = (
        command[:, 0] * open_target_gap
        + command[:, 1] * close_target_gap
    )
    stick1_position, stick1_quaternion = _object_pose_in_palm(
        env,
        palm_cfg,
        stick1_cfg,
    )
    stick2_position, stick2_quaternion = _object_pose_in_palm(
        env,
        palm_cfg,
        stick2_cfg,
    )
    tip_gap, lateral_error, axial_offset = _tip_geometry_from_palm_poses(
        stick1_position,
        stick1_quaternion,
        stick2_position,
        stick2_quaternion,
        stick1_tip_offset_o,
        stick2_tip_offset_o,
        stick_thickness,
        reference_separation_direction_stick2,
    )
    axial_error = torch.abs(
        axial_offset - float(reference_axial_offset_stick2)
    )
    mode_gate = torch.exp(
        -torch.abs(tip_gap - target_gap) / gap_sigma
        - lateral_error / lateral_sigma
        - axial_error / axial_sigma
    )
    return base_stability * mode_gate


class FunctionalSettingHeld(ManagerTermBase):
    """Detect a complete, quiet six-contact functional setting."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stable_steps = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stable_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_groups: tuple[tuple[str, ...], ...],
        palm_cfg: SceneEntityCfg,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        thumb_distal_cfg: SceneEntityCfg,
        index_tip_cfg: SceneEntityCfg,
        middle_tip_cfg: SceneEntityCfg,
        ring_tip_cfg: SceneEntityCfg,
        stick1_reference_position_p: tuple[float, float, float],
        stick1_reference_quaternion_p: tuple[float, float, float, float],
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        contact_threshold: float = 0.02,
        stick1_position_error_limit: float = 0.02,
        stick1_orientation_error_limit: float = 0.3490658504,
        stick2_position_error_limit: float = 0.015,
        stick2_orientation_error_limit: float = 0.3490658504,
        linear_speed_limit: float = 0.15,
        angular_speed_limit: float = 3.0,
        long_axis: int = 1,
        axial_half_length: float = 0.08,
        hold_steps: int = 30,
    ) -> torch.Tensor:
        (
            stick1_position_error,
            stick1_orientation_error,
            stick2_position_error,
            stick2_orientation_error,
            region_valid,
        ) = _setting_geometry(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            thumb_distal_cfg,
            index_tip_cfg,
            middle_tip_cfg,
            ring_tip_cfg,
            stick1_reference_position_p,
            stick1_reference_quaternion_p,
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
            long_axis,
            axial_half_length,
        )
        contacts_valid = torch.all(
            _group_forces(env, sensor_groups) >= contact_threshold,
            dim=-1,
        )
        (
            stick1_linear_speed,
            stick2_linear_speed,
            stick1_angular_speed,
            stick2_angular_speed,
        ) = _object_pair_speeds_relative_to_palm(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
        )
        stable = (
            contacts_valid
            & region_valid
            & (stick1_position_error <= stick1_position_error_limit)
            & (
                stick1_orientation_error
                <= stick1_orientation_error_limit
            )
            & (stick2_position_error <= stick2_position_error_limit)
            & (
                stick2_orientation_error
                <= stick2_orientation_error_limit
            )
            & (
                torch.maximum(stick1_linear_speed, stick2_linear_speed)
                <= linear_speed_limit
            )
            & (
                torch.maximum(stick1_angular_speed, stick2_angular_speed)
                <= angular_speed_limit
            )
        )
        self._stable_steps = torch.where(
            stable,
            self._stable_steps + 1,
            torch.zeros_like(self._stable_steps),
        )
        return self._stable_steps == hold_steps


class OpenCloseModeHeld(ManagerTermBase):
    """Track a requested mode held with a valid functional grasp.

    As a termination term this returns ``True`` after ``hold_steps``.  As a
    reward term, ``one_shot_per_mode=True`` emits one pulse per OPEN/CLOSE
    interval and rearms automatically when the command changes.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stable_steps = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        self._awarded = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        self._last_mode = torch.full(
            (env.num_envs,),
            -1,
            dtype=torch.long,
            device=env.device,
        )
        stick1_position_reference = torch.as_tensor(
            cfg.params["stick1_reference_position_p"],
            device=env.device,
        )
        stick1_quaternion_reference = torch.as_tensor(
            cfg.params["stick1_reference_quaternion_p"],
            device=env.device,
        )
        stick1_pivot_offset = torch.as_tensor(
            cfg.params["stick1_pivot_offset_o"],
            device=env.device,
        )
        self._stick1_pivot_reference = (
            stick1_position_reference
            + quat_apply(
                stick1_quaternion_reference.unsqueeze(0),
                stick1_pivot_offset.unsqueeze(0),
            )[0]
        )
        self._stick2_position_reference = torch.as_tensor(
            cfg.params["stick2_reference_position_p"],
            device=env.device,
        )
        self._stick2_quaternion_reference = torch.as_tensor(
            cfg.params["stick2_reference_quaternion_p"],
            device=env.device,
        )
        self._tip_separation_direction_stick2 = torch.as_tensor(
            cfg.params["reference_separation_direction_stick2"],
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stable_steps[env_ids] = 0
        self._awarded[env_ids] = False
        self._last_mode[env_ids] = -1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_groups: tuple[tuple[str, ...], ...],
        palm_cfg: SceneEntityCfg,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        stick1_pivot_offset_o: tuple[float, float, float],
        stick1_tip_offset_o: tuple[float, float, float],
        stick2_tip_offset_o: tuple[float, float, float],
        stick_thickness: float,
        open_target_gap: float,
        close_target_gap: float,
        stick1_reference_position_p: tuple[float, float, float],
        stick1_reference_quaternion_p: tuple[float, float, float, float],
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        reference_separation_direction_stick2: tuple[float, float, float],
        contact_threshold: float = 0.02,
        pivot_error_limit: float = 0.015,
        tip_gap_error_limit: float = 0.005,
        lateral_error_limit: float = 0.005,
        stick2_position_error_limit: float = 0.005,
        stick2_orientation_error_limit: float = 0.1745329252,
        linear_speed_limit: float = 0.15,
        angular_speed_limit: float = 3.0,
        hold_steps: int = 30,
        one_shot_per_mode: bool = False,
    ) -> torch.Tensor:
        del (
            stick1_reference_position_p,
            stick1_reference_quaternion_p,
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
            reference_separation_direction_stick2,
        )
        contacts_valid = torch.all(
            _group_forces(env, sensor_groups) >= contact_threshold,
            dim=-1,
        )
        stick1_pivot = _object_point_in_palm(
            env,
            palm_cfg,
            stick1_cfg,
            stick1_pivot_offset_o,
        )
        stick2_position, stick2_quaternion = _object_pose_in_palm(
            env,
            palm_cfg,
            stick2_cfg,
        )
        stick1_pivot_error = torch.linalg.vector_norm(
            stick1_pivot - self._stick1_pivot_reference,
            dim=-1,
        )
        stick2_position_error, stick2_orientation_error = _pose_errors(
            stick2_position,
            stick2_quaternion,
            self._stick2_position_reference,
            self._stick2_quaternion_reference,
        )
        command = env.command_manager.get_command(command_name)
        mode = torch.argmax(command, dim=-1)
        mode_changed = mode != self._last_mode
        self._stable_steps = torch.where(
            mode_changed,
            torch.zeros_like(self._stable_steps),
            self._stable_steps,
        )
        self._awarded = self._awarded & (~mode_changed)
        self._last_mode = mode
        target_gap = (
            command[:, 0] * open_target_gap
            + command[:, 1] * close_target_gap
        )
        tip_gap, lateral_error = _tip_surface_gap_and_lateral_error(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            stick1_tip_offset_o,
            stick2_tip_offset_o,
            stick_thickness,
            self._tip_separation_direction_stick2,
        )
        mode_valid = (
            torch.abs(tip_gap - target_gap) <= tip_gap_error_limit
        )
        geometry_valid = (
            (stick1_pivot_error <= pivot_error_limit)
            & (stick2_position_error <= stick2_position_error_limit)
            & (
                stick2_orientation_error
                <= stick2_orientation_error_limit
            )
            & mode_valid
            & (lateral_error <= lateral_error_limit)
        )
        (
            stick1_linear_speed,
            stick2_linear_speed,
            stick1_angular_speed,
            stick2_angular_speed,
        ) = _object_pair_speeds_relative_to_palm(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
        )
        linear_speed = torch.maximum(
            stick1_linear_speed,
            stick2_linear_speed,
        )
        angular_speed = torch.maximum(
            stick1_angular_speed,
            stick2_angular_speed,
        )
        valid = (
            contacts_valid
            & geometry_valid
            & (linear_speed <= linear_speed_limit)
            & (angular_speed <= angular_speed_limit)
        )
        self._stable_steps = torch.where(
            valid,
            self._stable_steps + 1,
            torch.zeros_like(self._stable_steps),
        )
        reached = self._stable_steps >= hold_steps
        if not one_shot_per_mode:
            return reached
        newly_reached = reached & (~self._awarded)
        self._awarded = self._awarded | newly_reached
        return newly_reached.float()


class FunctionalContactLoss(ManagerTermBase):
    """Terminate a sustained multi-contact collapse after grasp acquisition.

    The latch prevents reset loops before the policy has first established the
    validated six-contact topology.  Separate acquire/release thresholds add
    hysteresis, and a short loss grace period tolerates transient mode-change
    contact flicker.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._acquire_steps = torch.zeros(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        self._loss_steps = torch.zeros_like(self._acquire_steps)
        self._latched = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._acquire_steps[env_ids] = 0
        self._loss_steps[env_ids] = 0
        self._latched[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_groups: tuple[tuple[str, ...], ...],
        acquire_threshold: float = 0.02,
        release_threshold: float = 0.01,
        acquire_hold_steps: int = 5,
        minimum_retained_contacts: int = 5,
        loss_hold_steps: int = 10,
    ) -> torch.Tensor:
        group_forces = _group_forces(env, sensor_groups)
        all_acquired = torch.all(
            group_forces >= acquire_threshold,
            dim=-1,
        )
        self._acquire_steps = torch.where(
            (~self._latched) & all_acquired,
            self._acquire_steps + 1,
            torch.zeros_like(self._acquire_steps),
        )
        self._latched = (
            self._latched
            | (self._acquire_steps >= acquire_hold_steps)
        )

        retained_contacts = torch.sum(
            group_forces >= release_threshold,
            dim=-1,
        )
        collapsed = retained_contacts < minimum_retained_contacts
        self._loss_steps = torch.where(
            self._latched & collapsed,
            self._loss_steps + 1,
            torch.zeros_like(self._loss_steps),
        )
        return self._loss_steps >= loss_hold_steps


def open_close_mode(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Return the OPEN/CLOSE one-hot command for policy observation."""
    return env.command_manager.get_command(command_name)


def fingertip_positions_in_palm(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    fingertip_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return all fingertip positions expressed in the palm frame."""
    robot: Articulation = env.scene[palm_cfg.name]
    palm_pos_w = robot.data.body_pos_w[:, palm_cfg.body_ids[0]]
    palm_quat_w = robot.data.body_quat_w[:, palm_cfg.body_ids[0]]
    fingertip_pos_w = robot.data.body_pos_w[:, fingertip_cfg.body_ids]

    num_tips = fingertip_pos_w.shape[1]
    palm_pos_w = palm_pos_w.unsqueeze(1)
    palm_quat_w = palm_quat_w.unsqueeze(1).expand(-1, num_tips, -1)
    positions_b = quat_apply_inverse(palm_quat_w, fingertip_pos_w - palm_pos_w)
    return positions_b.reshape(env.num_envs, -1)


def object_pose_in_palm(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an object's position and quaternion expressed in the palm frame."""
    pos_b, quat_b = _object_pose_in_palm(env, palm_cfg, object_cfg)
    return torch.cat((pos_b, quat_b), dim=-1)


def object_velocity_in_palm(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an object's linear and angular velocity relative to the palm."""
    robot: Articulation = env.scene[palm_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    palm_id = palm_cfg.body_ids[0]

    palm_pos_w = robot.data.body_pos_w[:, palm_id]
    palm_quat_w = robot.data.body_quat_w[:, palm_id]
    palm_lin_vel_w = robot.data.body_lin_vel_w[:, palm_id]
    palm_ang_vel_w = robot.data.body_ang_vel_w[:, palm_id]

    offset_w = obj.data.root_pos_w - palm_pos_w
    relative_lin_vel_w = (
        obj.data.root_lin_vel_w
        - palm_lin_vel_w
        - torch.cross(palm_ang_vel_w, offset_w, dim=-1)
    )
    relative_ang_vel_w = obj.data.root_ang_vel_w - palm_ang_vel_w
    return torch.cat(
        (
            quat_apply_inverse(palm_quat_w, relative_lin_vel_w),
            quat_apply_inverse(palm_quat_w, relative_ang_vel_w),
        ),
        dim=-1,
    )
