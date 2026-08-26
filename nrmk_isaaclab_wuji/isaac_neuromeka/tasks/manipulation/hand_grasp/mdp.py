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


# [hand_grasp] Alternating OPEN/CLOSE command used by mode-conditioned grasp.
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
# [hand_grasp] Configuration for the active OPEN/CLOSE command.
class OpenCloseModeCommandCfg(CommandTermCfg):
    """Configuration for the per-episode OPEN/CLOSE command."""

    class_type: type = OpenCloseModeCommand
    open_probability: float = 0.5


# [hand_setting] Open-hand reset with no hidden PD preload.
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


# [hand_grasp] Reset directly to the validated functional grasp.
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

# [hand_setting] 진단용: gate 없는 단순 관절 위치 트래킹(reach식). 주어진 관절이 고정 목표
# 자세에 얼마나 가까운지 exp로 보상. stage1 게이트·접촉·페이드 다 뺀 순수 트래킹이라, "정책이
# 현재 PD에서 손가락을 명령 위치로 몰아갈 수 있나"를 격리 검증하는 용도(2026-08-04).
def joint_reference_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    reference_joint_positions: tuple[float, ...],
    joint_sigma: float,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    reference = torch.as_tensor(
        reference_joint_positions,
        dtype=robot.data.joint_pos.dtype,
        device=env.device,
    )
    joint_error = robot.data.joint_pos[:, asset_cfg.joint_ids] - reference
    return torch.exp(
        -torch.mean(torch.square(joint_error / joint_sigma), dim=-1)
    )


# [hand_setting] 진단용 mini-reach: 손끝(body)이 FingerTipReachCommand 목표점에 얼마나 가까운지
# exp 보상. target_w = env_origins + command[:,:3] (env-로컬 저장). (2026-08-04)
def body_reach_command_tracking(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    command_name: str,
    sigma: float,
) -> torch.Tensor:
    robot: Articulation = env.scene[body_cfg.name]
    tip_w = robot.data.body_pos_w[:, body_cfg.body_ids[0]]
    target_w = (
        env.scene.env_origins
        + env.command_manager.get_command(command_name)[:, :3]
    )
    dist = torch.norm(tip_w - target_w, dim=1)
    return torch.exp(-dist / sigma)


# [hand_setting] 진단용 obs: 손끝 → 목표점 오차 벡터(world). 정책이 "어디로 가야 하나"를 알게.
def body_reach_command_error(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    robot: Articulation = env.scene[body_cfg.name]
    tip_w = robot.data.body_pos_w[:, body_cfg.body_ids[0]]
    target_w = (
        env.scene.env_origins
        + env.command_manager.get_command(command_name)[:, :3]
    )
    return target_w - tip_w


# [hand_grasp_object] Spawn the later object-grasp stage between the tips.
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


# [shared: hand_grasp + hand_setting] Read one contact sensor's force.
def _sensor_force(env: ManagerBasedEnv, sensor_name: str) -> torch.Tensor:
    """Return the summed filtered contact-force magnitude for each environment."""
    force_matrix = env.scene.sensors[sensor_name].data.force_matrix_w
    if force_matrix is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.linalg.vector_norm(force_matrix, dim=-1).sum(dim=(-1, -2))


# [shared: hand_grasp + hand_setting] Combine semantic contact sensor groups.
def _group_forces(
    env: ManagerBasedEnv,
    sensor_groups: tuple[tuple[str, ...], ...],
    group_reduction: str = "max",
) -> torch.Tensor:
    """Return one force per semantic group, using OR/max or AND/min inside it."""
    groups = []
    for sensor_names in sensor_groups:
        forces = torch.stack(
            [_sensor_force(env, sensor_name) for sensor_name in sensor_names],
            dim=-1,
        )
        if group_reduction == "max":
            groups.append(torch.max(forces, dim=-1).values)
        elif group_reduction == "min":
            groups.append(torch.min(forces, dim=-1).values)
        else:
            raise ValueError(
                "group_reduction must be 'max' or 'min', got "
                f"{group_reduction!r}"
            )
    return torch.stack(groups, dim=-1)


# [shared: hand_grasp + hand_setting] Express one object pose in the palm frame.
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


# [shared: hand_grasp + hand_setting] Measure both sticks relative to the palm.
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


# [hand_grasp] Express the moving-stick pivot point in the palm frame.
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


# [hand_grasp] Compute OPEN/CLOSE tip gap, lateral miss, and axial offset.
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
    clamp_gap: bool = True,
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

    ``clamp_gap`` (2026-08-07) controls whether the surface gap floors at zero.

    Rigid collision keeps the two cross-sections from interpenetrating, so in
    any *valid* grasp ``transverse_distance`` is at least the sum of the two
    support radii and the raw gap is non-negative.  A **negative** raw gap means
    the tips are closer together than two half-widths while not colliding -
    which can only happen when they are offset far enough sideways to slide past
    each other.  In other words the negative branch is a crossing detector.

    With the clamp on, every one of those crossing states reports ``gap = 0``,
    i.e. a *perfect* CLOSE, and the reward has no way to object.  Measured
    2026-08-07: ``tip_lateral_error`` reached 9.14 mm against a 7 mm stick
    section, so the sections could not overlap at all, yet the gap term was
    scoring full marks.  Passing ``clamp_gap=False`` turns that into an error of
    the same magnitude and the tracking exponent penalises it.

    The default stays ``True`` so ``hand_grasp`` and ``hand_setting`` keep the
    behaviour they were tuned under; only ``hand_move`` and its descendants opt
    out.
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
    surface_gap = transverse_distance - stick1_support - stick2_support
    if clamp_gap:
        surface_gap = torch.clamp(surface_gap, min=0.0)

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


# [hand_grasp] Lightweight tip-gap helper used by play diagnostics.
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


# [hand_grasp] Shared tip geometry for mode success and legacy diagnostics.
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
    clamp_gap: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gap and sideways miss using one shared pose calculation.

    ``clamp_gap`` is forwarded; see :func:`_tip_geometry_from_palm_poses`.  The
    success termination has to use the *same* setting as the reward, otherwise
    a crossing state that the reward penalises can still be declared a success.
    """
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
        clamp_gap=clamp_gap,
    )
    return surface_gap, lateral_error


# [shared: hand_grasp + hand_setting] Compute palm-frame position/orientation error.
def _pose_errors(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    reference_position: torch.Tensor,
    reference_quaternion: torch.Tensor,
    orientation_error_mode: str = "quaternion",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Euclidean position error and the configured orientation error.

    ``quaternion`` preserves the historical full-orientation comparison.
    ``directed_axis`` compares only local ``+y`` and deliberately does not use
    ``abs(dot)``: tip and tail remain distinct while shaft roll is ignored.
    """
    position_error = torch.linalg.vector_norm(
        position - reference_position,
        dim=-1,
    )
    if orientation_error_mode == "quaternion":
        quaternion_dot = torch.abs(
            torch.sum(quaternion * reference_quaternion, dim=-1)
        )
        orientation_error = 2.0 * torch.acos(
            torch.clamp(quaternion_dot, min=0.0, max=1.0)
        )
    elif orientation_error_mode == "directed_axis":
        local_y = torch.zeros_like(position)
        local_y[:, 1] = 1.0
        current_axis = quat_apply(quaternion, local_y)
        reference_axis = quat_apply(
            reference_quaternion.expand_as(quaternion),
            local_y,
        )
        orientation_error = torch.acos(
            torch.clamp(
                torch.sum(current_axis * reference_axis, dim=-1),
                min=-1.0,
                max=1.0,
            )
        )
    else:
        raise ValueError(
            "orientation_error_mode must be 'quaternion' or 'directed_axis', "
            f"got {orientation_error_mode!r}"
        )
    return position_error, orientation_error


# [hand_setting, parked] Historical finite-shaft valley proxy.
def _stick_valley_geometry(
    position_p: torch.Tensor,
    quaternion_p: torch.Tensor,
    reference_position_p: torch.Tensor,
    reference_quaternion_p: torch.Tensor,
    valley_point_offset_o: torch.Tensor,
    stick_half_length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare a finite shaft segment and directed axis to a valley reference.

    Full-pose scalar tolerances are too broad for a 7 mm stick and do not say
    whether the shaft actually passes through the validated thumb-index
    valley.  ``valley_point_offset_o`` identifies that fixed point from the
    saved reference pose only.  The current stick may intersect it with any
    station on its physical centerline, so axial sliding is deliberately free
    during insertion.  Roll around the shaft is also ignored.
    """
    reference_point_p = reference_position_p + quat_apply(
        reference_quaternion_p,
        valley_point_offset_o,
    )

    local_long_axis = torch.zeros_like(position_p)
    local_long_axis[:, 1] = 1.0
    current_axis_p = quat_apply(quaternion_p, local_long_axis)
    valley_from_center = reference_point_p - position_p
    closest_station = torch.sum(
        valley_from_center * current_axis_p,
        dim=-1,
    )
    closest_station = torch.clamp(
        closest_station,
        min=-stick_half_length,
        max=stick_half_length,
    )
    closest_point_p = (
        position_p + closest_station.unsqueeze(-1) * current_axis_p
    )
    point_error = torch.linalg.vector_norm(
        closest_point_p - reference_point_p,
        dim=-1,
    )

    reference_axis_p = quat_apply(
        reference_quaternion_p,
        local_long_axis,
    )
    axis_dot = torch.sum(current_axis_p * reference_axis_p, dim=-1)
    axis_error = torch.acos(
        torch.clamp(axis_dot, min=-1.0, max=1.0)
    )
    return point_error, axis_error


# [hand_setting, parked/debug] Public wrapper for the finite-shaft valley probe.
def stick2_valley_geometry(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    valley_point_offset_o: tuple[float, float, float],
    stick_half_length: float = 0.09,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite-shaft distance and directed-axis valley errors."""
    position, quaternion = _object_pose_in_palm(
        env,
        palm_cfg,
        stick2_cfg,
    )
    reference_position = torch.as_tensor(
        stick2_reference_position_p,
        dtype=position.dtype,
        device=position.device,
    ).expand_as(position)
    reference_quaternion = torch.as_tensor(
        stick2_reference_quaternion_p,
        dtype=quaternion.dtype,
        device=quaternion.device,
    ).expand_as(quaternion)
    valley_offset = torch.as_tensor(
        valley_point_offset_o,
        dtype=position.dtype,
        device=position.device,
    ).expand_as(position)
    return _stick_valley_geometry(
        position,
        quaternion,
        reference_position,
        reference_quaternion,
        valley_offset,
        stick_half_length,
    )


# [hand_grasp] Active weak joint prior; its hand_setting term is parked.
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
        deactivate_sensor_groups: tuple[tuple[str, ...], ...] | None = None,
        deactivate_contact_threshold: float = 0.02,
        deactivate_group_reduction: str = "max",
    ) -> torch.Tensor:
        del reference_joint_positions
        robot: Articulation = env.scene[asset_cfg.name]
        joint_error = (
            robot.data.joint_pos[:, asset_cfg.joint_ids] - self._reference
        )
        score = torch.exp(
            -torch.mean(torch.square(joint_error / sigma), dim=-1)
        )
        if deactivate_sensor_groups is None:
            return score
        full_contact = torch.all(
            _group_forces(
                env,
                deactivate_sensor_groups,
                deactivate_group_reduction,
            )
            >= deactivate_contact_threshold,
            dim=-1,
        )
        # Memoryless acquisition prior: disappear at 6/6, return immediately
        # if any semantic contact is lost so recovery still has a pose guide.
        return (~full_contact).to(dtype=score.dtype) * score


# [shared: hand_grasp + hand_setting] Active palm-frame stick-pose reward.
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
        orientation_error_mode: str = "quaternion",
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
            orientation_error_mode,
        )
        return torch.exp(
            -position_error / position_sigma
            -orientation_error / orientation_sigma
        )


# [hand_setting] Compute the weaker of two palm-frame reference-pose scores.
def _object_pair_reference_pose_min_score(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick1_reference_position: torch.Tensor,
    stick1_reference_quaternion: torch.Tensor,
    stick2_reference_position: torch.Tensor,
    stick2_reference_quaternion: torch.Tensor,
    position_sigma: float,
    orientation_sigma: float,
) -> torch.Tensor:
    """Return the lower full-pose score so one aligned stick cannot dominate."""
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
    stick1_position_error, stick1_orientation_error = _pose_errors(
        stick1_position,
        stick1_quaternion,
        stick1_reference_position,
        stick1_reference_quaternion,
    )
    stick2_position_error, stick2_orientation_error = _pose_errors(
        stick2_position,
        stick2_quaternion,
        stick2_reference_position,
        stick2_reference_quaternion,
    )
    stick1_score = torch.exp(
        -stick1_position_error / position_sigma
        -stick1_orientation_error / orientation_sigma
    )
    stick2_score = torch.exp(
        -stick2_position_error / position_sigma
        -stick2_orientation_error / orientation_sigma
    )
    return torch.minimum(stick1_score, stick2_score)


# [hand_setting] Active two-stick reference score determined by the worse pose.
class ObjectPairReferencePoseMinTracking(ManagerTermBase):
    """Track both palm-frame stick poses without allowing one-stick farming."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stick1_reference_position = torch.as_tensor(
            cfg.params["stick1_reference_position_p"],
            device=env.device,
        )
        self._stick1_reference_quaternion = torch.as_tensor(
            cfg.params["stick1_reference_quaternion_p"],
            device=env.device,
        )
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
        palm_cfg: SceneEntityCfg,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        stick1_reference_position_p: tuple[float, float, float],
        stick1_reference_quaternion_p: tuple[float, float, float, float],
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        position_sigma: float = 0.10,
        orientation_sigma: float = 1.5707963268,
        score_floor: float = 0.0,
    ) -> torch.Tensor:
        """Return the weaker stick's pose score, optionally rebased.

        ``score_floor`` subtracts a constant and renormalizes:
        ``clamp((score - floor) / (1 - floor), 0, 1)``.  The default 0.0 is the
        identity, so every existing caller is unchanged.

        This exists because a wide kernel pays a large constant for doing
        nothing.  Run 2026-08-26_02-11-08 measured that exactly: a sigma
        0.10 m / 90 deg term at weight 6 scored 0.348 on the untouched spawn
        pose, worth 16.7 points per episode, against the 12.3 points the best
        behaviour any previous run found had earned.  The policy stopped moving
        at iteration 100 and mean_reward sat at 16.49 for the next 340
        iterations.  Rebasing removes that annuity while keeping -- in fact
        amplifying by 1/(1 - floor) -- the long-range gradient the wide kernel
        was added for.

        Keep the floor strictly below the score of any state the policy still
        has to recover from: at and below the floor this term is flat zero, so
        a floor set at the idle score would leave worse states with no pull.
        """
        del (
            stick1_reference_position_p,
            stick1_reference_quaternion_p,
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
        )
        score = _object_pair_reference_pose_min_score(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            self._stick1_reference_position,
            self._stick1_reference_quaternion,
            self._stick2_reference_position,
            self._stick2_reference_quaternion,
            position_sigma,
            orientation_sigma,
        )
        if score_floor == 0.0:
            return score
        if not 0.0 <= score_floor < 1.0:
            raise ValueError("score_floor must be in [0, 1)")
        return torch.clamp(
            (score - score_floor) / (1.0 - score_floor),
            min=0.0,
            max=1.0,
        )


# [hand_setting] Active two-stick-gated thumb approach to Stick1's pivot station.
class ObjectPairReferenceThumbStationMinTracking(
    ObjectPairReferencePoseMinTracking
):
    """Approach the pivot only while preserving the weaker stick-pose score."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        palm_cfg: SceneEntityCfg,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        thumb_cfg: SceneEntityCfg,
        stick1_reference_position_p: tuple[float, float, float],
        stick1_reference_quaternion_p: tuple[float, float, float, float],
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        stick1_half_extent: tuple[float, float, float],
        long_axis: int = 1,
        pivot_station: float = -0.06,
        position_sigma: float = 0.10,
        orientation_sigma: float = 1.5707963268,
        thumb_sigma: float = 0.02,
    ) -> torch.Tensor:
        pair_score = super().__call__(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            stick1_reference_position_p,
            stick1_reference_quaternion_p,
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
            position_sigma,
            orientation_sigma,
        )

        thumb_distance = body_box_axial_station_distance(
            env,
            thumb_cfg,
            stick1_cfg,
            stick1_half_extent,
            long_axis,
            pivot_station,
        )
        thumb_score = torch.exp(-thumb_distance / thumb_sigma)
        return torch.minimum(pair_score, thumb_score)


# [hand_setting] Measure a body-origin proxy to one finite stick shaft station.
def body_box_axial_station_distance(
    env: ManagerBasedEnv,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    long_axis: int = 1,
    axial_station: float = -0.06,
) -> torch.Tensor:
    """Return transverse box distance combined with axial-station error."""
    robot: Articulation = env.scene[body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_position_w = robot.data.body_pos_w[:, body_cfg.body_ids[0]]
    body_position_o = quat_apply_inverse(
        obj.data.root_quat_w,
        body_position_w - obj.data.root_pos_w,
    )
    half_extent = torch.as_tensor(
        object_half_extent,
        dtype=body_position_o.dtype,
        device=body_position_o.device,
    )
    outside = torch.clamp(
        torch.abs(body_position_o) - half_extent,
        min=0.0,
    )
    transverse_mask = torch.ones_like(half_extent)
    transverse_mask[long_axis] = 0.0
    transverse_distance = torch.linalg.vector_norm(
        outside * transverse_mask,
        dim=-1,
    )
    axial_error = torch.abs(
        body_position_o[:, long_axis] - axial_station
    )
    return torch.sqrt(
        torch.square(transverse_distance) + torch.square(axial_error)
    )


# [hand_setting] Open the contact stage after stick-pair and pivot acquisition.
def setting_stage1_gate(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    thumb_cfg: SceneEntityCfg,
    stick1_reference_position_p: tuple[float, float, float],
    stick1_reference_quaternion_p: tuple[float, float, float, float],
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    stick1_half_extent: tuple[float, float, float],
    long_axis: int = 1,
    pivot_station: float = -0.06,
    position_sigma: float = 0.10,
    orientation_sigma: float = 1.5707963268,
    thumb_sigma: float = 0.02,
    pair_score_threshold: float = 0.50,
    thumb_score_threshold: float = 0.35,
) -> torch.Tensor:
    """Return a memoryless hard gate derived entirely from the observed state."""
    stick1_reference_position = torch.as_tensor(
        stick1_reference_position_p,
        dtype=torch.float,
        device=env.device,
    )
    stick1_reference_quaternion = torch.as_tensor(
        stick1_reference_quaternion_p,
        dtype=torch.float,
        device=env.device,
    )
    stick2_reference_position = torch.as_tensor(
        stick2_reference_position_p,
        dtype=torch.float,
        device=env.device,
    )
    stick2_reference_quaternion = torch.as_tensor(
        stick2_reference_quaternion_p,
        dtype=torch.float,
        device=env.device,
    )
    pair_score = _object_pair_reference_pose_min_score(
        env,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        stick1_reference_position,
        stick1_reference_quaternion,
        stick2_reference_position,
        stick2_reference_quaternion,
        position_sigma,
        orientation_sigma,
    )
    thumb_distance = body_box_axial_station_distance(
        env,
        thumb_cfg,
        stick1_cfg,
        stick1_half_extent,
        long_axis,
        pivot_station,
    )
    thumb_score = torch.exp(-thumb_distance / thumb_sigma)
    return (
        (pair_score >= pair_score_threshold)
        & (thumb_score >= thumb_score_threshold)
    ).float()


# [hand_setting] Open contact learning while all 20 joints currently match q_ref.
def setting_stage2_gate(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    reference_joint_positions: tuple[float, ...],
    joint_error_threshold: float,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    thumb_cfg: SceneEntityCfg,
    stick1_reference_position_p: tuple[float, float, float],
    stick1_reference_quaternion_p: tuple[float, float, float, float],
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    stick1_half_extent: tuple[float, float, float],
    long_axis: int,
    pivot_station: float,
    position_sigma: float,
    orientation_sigma: float,
    thumb_sigma: float,
    pair_score_threshold: float,
    thumb_score_threshold: float,
) -> torch.Tensor:
    """Return a memoryless hard gate for the q_ref-to-contact transition."""
    if joint_error_threshold <= 0.0:
        raise ValueError("joint_error_threshold must be positive")
    stage1_ready = setting_stage1_gate(
        env,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        thumb_cfg,
        stick1_reference_position_p,
        stick1_reference_quaternion_p,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        stick1_half_extent,
        long_axis,
        pivot_station,
        position_sigma,
        orientation_sigma,
        thumb_sigma,
        pair_score_threshold,
        thumb_score_threshold,
    ).bool()
    robot: Articulation = env.scene[asset_cfg.name]
    reference = torch.as_tensor(
        reference_joint_positions,
        dtype=robot.data.joint_pos.dtype,
        device=env.device,
    )
    joint_count = len(asset_cfg.joint_ids)
    if reference.numel() != joint_count:
        raise ValueError(
            "Stage-2 reference size must match the selected joints: "
            f"reference={reference.numel()}, joint_count={joint_count}"
        )
    max_joint_error = torch.max(
        torch.abs(robot.data.joint_pos[:, asset_cfg.joint_ids] - reference),
        dim=-1,
    ).values
    return (
        stage1_ready
        & (max_joint_error <= joint_error_threshold)
    ).float()

# [hand_setting] Gate one joint-pose prior and weaken it after Stage 2 unlocks.
def stage1_gated_joint_reference_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    reference_joint_positions: tuple[float, ...],
    joint_sigma: float,
    stage2_reference_weight_ratio: float,
    stage2_joint_error_threshold: float,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    thumb_cfg: SceneEntityCfg,
    stick1_reference_position_p: tuple[float, float, float],
    stick1_reference_quaternion_p: tuple[float, float, float, float],
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    stick1_half_extent: tuple[float, float, float],
    long_axis: int,
    pivot_station: float,
    position_sigma: float,
    orientation_sigma: float,
    thumb_sigma: float,
    pair_score_threshold: float,
    thumb_score_threshold: float,
) -> torch.Tensor:
    """Gate the all-joint exponential prior and apply its Stage-2 fade."""
    if joint_sigma <= 0.0:
        raise ValueError("joint_sigma must be positive")
    gate = setting_stage1_gate(
        env,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        thumb_cfg,
        stick1_reference_position_p,
        stick1_reference_quaternion_p,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        stick1_half_extent,
        long_axis,
        pivot_station,
        position_sigma,
        orientation_sigma,
        thumb_sigma,
        pair_score_threshold,
        thumb_score_threshold,
    )
    robot: Articulation = env.scene[asset_cfg.name]
    reference = torch.as_tensor(
        reference_joint_positions,
        dtype=robot.data.joint_pos.dtype,
        device=env.device,
    )
    joint_error = robot.data.joint_pos[:, asset_cfg.joint_ids] - reference
    score = torch.exp(
        -torch.mean(torch.square(joint_error / joint_sigma), dim=-1)
    )
    if not 0.0 <= stage2_reference_weight_ratio <= 1.0:
        raise ValueError("stage2_reference_weight_ratio must be in [0, 1]")
    stage2_gate = setting_stage2_gate(
        env,
        asset_cfg,
        reference_joint_positions,
        stage2_joint_error_threshold,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        thumb_cfg,
        stick1_reference_position_p,
        stick1_reference_quaternion_p,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        stick1_half_extent,
        long_axis,
        pivot_station,
        position_sigma,
        orientation_sigma,
        thumb_sigma,
        pair_score_threshold,
        thumb_score_threshold,
    )
    weight_scale = 1.0 - (
        1.0 - stage2_reference_weight_ratio
    ) * stage2_gate
    active_gate = torch.maximum(gate, stage2_gate)
    return active_gate * weight_scale * score


# [hand_setting] Combine dense and worst-joint q-reference guidance after Stage 1.
def stage1_gated_joint_reference_mean_min(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    reference_joint_positions: tuple[float, ...],
    joint_sigma: float,
    stage2_asset_cfg: SceneEntityCfg,
    stage2_reference_joint_positions: tuple[float, ...],
    stage2_joint_error_threshold: float,
    stage2_reference_weight_ratio: float,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    thumb_cfg: SceneEntityCfg,
    stick1_reference_position_p: tuple[float, float, float],
    stick1_reference_quaternion_p: tuple[float, float, float, float],
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    stick1_half_extent: tuple[float, float, float],
    long_axis: int,
    pivot_station: float,
    position_sigma: float,
    orientation_sigma: float,
    thumb_sigma: float,
    pair_score_threshold: float,
    thumb_score_threshold: float,
) -> torch.Tensor:
    """Exponentially guide selected joints without ignoring the worst one."""
    if joint_sigma <= 0.0:
        raise ValueError("joint_sigma must be positive")
    gate = setting_stage1_gate(
        env,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        thumb_cfg,
        stick1_reference_position_p,
        stick1_reference_quaternion_p,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        stick1_half_extent,
        long_axis,
        pivot_station,
        position_sigma,
        orientation_sigma,
        thumb_sigma,
        pair_score_threshold,
        thumb_score_threshold,
    )
    robot: Articulation = env.scene[asset_cfg.name]
    reference = torch.as_tensor(
        reference_joint_positions,
        dtype=robot.data.joint_pos.dtype,
        device=env.device,
    )
    joint_count = len(asset_cfg.joint_ids)
    if reference.numel() != joint_count:
        raise ValueError(
            "Stage-1 reference size must match the selected joints: "
            f"reference={reference.numel()}, joint_count={joint_count}"
        )
    joint_error = robot.data.joint_pos[:, asset_cfg.joint_ids] - reference
    # Score every joint separately.  The dense mean rewards broad progress;
    # the minimum prevents three easy joints from hiding one straight distal
    # joint, which caused the observed three-contact local optimum.
    joint_scores = torch.exp(-torch.square(joint_error / joint_sigma))
    dense_score = torch.mean(joint_scores, dim=-1)
    completion_score = torch.min(joint_scores, dim=-1).values
    score = 0.5 * dense_score + 0.5 * completion_score

    if not 0.0 <= stage2_reference_weight_ratio <= 1.0:
        raise ValueError("stage2_reference_weight_ratio must be in [0, 1]")
    stage2_gate = setting_stage2_gate(
        env,
        stage2_asset_cfg,
        stage2_reference_joint_positions,
        stage2_joint_error_threshold,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        thumb_cfg,
        stick1_reference_position_p,
        stick1_reference_quaternion_p,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        stick1_half_extent,
        long_axis,
        pivot_station,
        position_sigma,
        orientation_sigma,
        thumb_sigma,
        pair_score_threshold,
        thumb_score_threshold,
    )
    weight_scale = 1.0 - (
        1.0 - stage2_reference_weight_ratio
    ) * stage2_gate
    active_gate = torch.maximum(gate, stage2_gate)
    return active_gate * weight_scale * score


# [hand_setting] Add a long-range linear guide for the sixteen non-thumb joints.
def stage1_gated_joint_reference_linear_mean_min(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    reference_joint_positions: tuple[float, ...],
    joint_linear_range: float,
    stage2_asset_cfg: SceneEntityCfg,
    stage2_reference_joint_positions: tuple[float, ...],
    stage2_joint_error_threshold: float,
    stage2_reference_weight_ratio: float,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    thumb_cfg: SceneEntityCfg,
    stick1_reference_position_p: tuple[float, float, float],
    stick1_reference_quaternion_p: tuple[float, float, float, float],
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    stick1_half_extent: tuple[float, float, float],
    long_axis: int,
    pivot_station: float,
    position_sigma: float,
    orientation_sigma: float,
    thumb_sigma: float,
    pair_score_threshold: float,
    thumb_score_threshold: float,
) -> torch.Tensor:
    """Linearly guide selected joints without ignoring the worst one."""
    if joint_linear_range <= 0.0:
        raise ValueError("joint_linear_range must be positive")
    gate = setting_stage1_gate(
        env,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        thumb_cfg,
        stick1_reference_position_p,
        stick1_reference_quaternion_p,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        stick1_half_extent,
        long_axis,
        pivot_station,
        position_sigma,
        orientation_sigma,
        thumb_sigma,
        pair_score_threshold,
        thumb_score_threshold,
    )
    robot: Articulation = env.scene[asset_cfg.name]
    reference = torch.as_tensor(
        reference_joint_positions,
        dtype=robot.data.joint_pos.dtype,
        device=env.device,
    )
    joint_count = len(asset_cfg.joint_ids)
    if reference.numel() != joint_count:
        raise ValueError(
            "Stage-1 reference size must match the selected joints: "
            f"reference={reference.numel()}, joint_count={joint_count}"
        )
    joint_abs_error = torch.abs(
        robot.data.joint_pos[:, asset_cfg.joint_ids] - reference
    )
    joint_scores = torch.clamp(
        1.0 - joint_abs_error / joint_linear_range,
        min=0.0,
        max=1.0,
    )
    dense_score = torch.mean(joint_scores, dim=-1)
    completion_score = torch.min(joint_scores, dim=-1).values
    score = 0.5 * dense_score + 0.5 * completion_score

    if not 0.0 <= stage2_reference_weight_ratio <= 1.0:
        raise ValueError("stage2_reference_weight_ratio must be in [0, 1]")
    stage2_gate = setting_stage2_gate(
        env,
        stage2_asset_cfg,
        stage2_reference_joint_positions,
        stage2_joint_error_threshold,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        thumb_cfg,
        stick1_reference_position_p,
        stick1_reference_quaternion_p,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        stick1_half_extent,
        long_axis,
        pivot_station,
        position_sigma,
        orientation_sigma,
        thumb_sigma,
        pair_score_threshold,
        thumb_score_threshold,
    )
    weight_scale = 1.0 - (
        1.0 - stage2_reference_weight_ratio
    ) * stage2_gate
    active_gate = torch.maximum(gate, stage2_gate)
    return active_gate * weight_scale * score


# [hand_setting] Reward only new missing-joint pose records after Stage 1.
class Stage1MissingJointBestSoFar(ManagerTermBase):
    """Pay independent per-joint q-reference records after Stage 1.

    Stage 1 acts as a one-way unlock, not a per-step gate.  This lets the
    policy keep receiving an approach signal while the other fingers begin to
    touch and perturb the sticks.  The live stick-pose/thumb rewards still
    enforce anchor maintenance.  Each selected joint owns its own best score,
    so progress by one joint is not suppressed by the worst of the other
    fifteen joints.  A normalized optional per-joint weighting changes their
    relative priority without changing the term's total reward budget. Seeding
    every best score at unlock prevents an artificial payout for progress made
    before Stage 1 was acquired.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stage1_unlocked = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        joint_count = len(cfg.params["reference_joint_positions"])
        if joint_count <= 0:
            raise ValueError("At least one reference joint is required")
        joint_reward_weights = cfg.params.get("joint_reward_weights")
        if joint_reward_weights is None:
            joint_reward_weights = (1.0,) * joint_count
        if len(joint_reward_weights) != joint_count:
            raise ValueError(
                "joint_reward_weights must match the selected joints: "
                f"weights={len(joint_reward_weights)}, joint_count={joint_count}"
            )
        self._joint_reward_weights = torch.as_tensor(
            joint_reward_weights,
            dtype=torch.float,
            device=env.device,
        )
        if torch.any(self._joint_reward_weights < 0.0):
            raise ValueError("joint_reward_weights must be non-negative")
        self._joint_reward_weight_sum = torch.sum(self._joint_reward_weights)
        if float(self._joint_reward_weight_sum.item()) <= 0.0:
            raise ValueError("At least one joint_reward_weight must be positive")
        self._best_joint_scores = torch.zeros(
            (env.num_envs, joint_count),
            dtype=torch.float,
            device=env.device,
        )
        self._current_joint_scores = torch.zeros_like(
            self._best_joint_scores
        )

    @property
    def stage1_unlocked(self) -> torch.Tensor:
        """Return the per-environment Stage-1 acquisition latch."""
        return self._stage1_unlocked

    @property
    def best_score(self) -> torch.Tensor:
        """Return the mean of the independent per-joint best scores."""
        return torch.mean(self._best_joint_scores, dim=-1)

    @property
    def best_joint_scores(self) -> torch.Tensor:
        """Return each selected joint's best linear score this episode."""
        return self._best_joint_scores

    @property
    def current_joint_scores(self) -> torch.Tensor:
        """Return each selected joint's current linear reference score."""
        return self._current_joint_scores

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stage1_unlocked[env_ids] = False
        self._best_joint_scores[env_ids] = 0.0
        self._current_joint_scores[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        reference_joint_positions: tuple[float, ...],
        joint_linear_range: float,
        palm_cfg: SceneEntityCfg,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        thumb_cfg: SceneEntityCfg,
        stick1_reference_position_p: tuple[float, float, float],
        stick1_reference_quaternion_p: tuple[float, float, float, float],
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        stick1_half_extent: tuple[float, float, float],
        long_axis: int,
        pivot_station: float,
        position_sigma: float,
        orientation_sigma: float,
        thumb_sigma: float,
        pair_score_threshold: float,
        thumb_score_threshold: float,
        joint_reward_weights: tuple[float, ...] | None = None,
    ) -> torch.Tensor:
        if joint_linear_range <= 0.0:
            raise ValueError("joint_linear_range must be positive")

        stage1_ready = setting_stage1_gate(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            thumb_cfg,
            stick1_reference_position_p,
            stick1_reference_quaternion_p,
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
            stick1_half_extent,
            long_axis,
            pivot_station,
            position_sigma,
            orientation_sigma,
            thumb_sigma,
            pair_score_threshold,
            thumb_score_threshold,
        ).bool()

        robot: Articulation = env.scene[asset_cfg.name]
        reference = torch.as_tensor(
            reference_joint_positions,
            dtype=robot.data.joint_pos.dtype,
            device=env.device,
        )
        joint_count = len(asset_cfg.joint_ids)
        if reference.numel() != joint_count:
            raise ValueError(
                "Stage-1 reference size must match the selected joints: "
                f"reference={reference.numel()}, joint_count={joint_count}"
            )

        joint_abs_error = torch.abs(
            robot.data.joint_pos[:, asset_cfg.joint_ids] - reference
        )
        joint_scores = torch.clamp(
            1.0 - joint_abs_error / joint_linear_range,
            min=0.0,
            max=1.0,
        )
        if joint_scores.shape != self._best_joint_scores.shape:
            raise ValueError(
                "Stage-1 joint-score shape changed after initialization: "
                f"current={tuple(joint_scores.shape)}, "
                f"expected={tuple(self._best_joint_scores.shape)}"
            )
        self._current_joint_scores = joint_scores

        newly_unlocked = (~self._stage1_unlocked) & stage1_ready
        self._stage1_unlocked |= stage1_ready

        # The unlock step establishes each joint's baseline and pays zero.
        self._best_joint_scores = torch.where(
            newly_unlocked.unsqueeze(-1),
            joint_scores,
            self._best_joint_scores,
        )
        joint_progress = torch.clamp(
            joint_scores - self._best_joint_scores,
            min=0.0,
            max=1.0,
        )
        joint_progress *= self._stage1_unlocked.unsqueeze(-1).float()
        self._best_joint_scores = torch.where(
            self._stage1_unlocked.unsqueeze(-1),
            torch.maximum(self._best_joint_scores, joint_scores),
            self._best_joint_scores,
        )
        # Normalize by the configured weight sum so the term's maximum remains
        # one regardless of the A/B weighting.  Here the little finger is made
        # deliberately slower so it does not occupy the ring finger's route.
        return torch.sum(
            joint_progress * self._joint_reward_weights,
            dim=-1,
        ) / self._joint_reward_weight_sum


# [hand_setting] Pay the signed per-joint q-reference change after Stage 1.
class Stage1MissingJointSignedProgress(Stage1MissingJointBestSoFar):
    """Pay ``Phi(q_t) - Phi(q_{t-1})`` per joint instead of only new records.

    Same potential, same weights, same normalized budget as
    :class:`Stage1MissingJointBestSoFar`; only the sign policy differs.  The
    episode total telescopes to ``Phi_final - Phi_unlock`` rather than
    ``Phi_best - Phi_unlock``, so the term pays for the pose the policy is
    actually holding rather than the best pose it ever touched.

    Why this task wants the signed form:

    * The objective is maintenance, not contact.  All six functional contacts
      must hold simultaneously, and 2026-08-10_18-30-36 ended with per-joint
      ``best 0.921`` against ``current 0.780`` (gap 0.141).  About 42% of that
      run's 33.3 reward points were paid for peaks the policy did not hold.
    * Stationary farming stays impossible: holding still pays exactly zero
      because consecutive scores are equal.  An advance/retreat round trip
      also nets exactly zero, because the retreat is charged at the same rate
      the advance paid.
    * ``best_so_far`` needs sixteen hidden per-joint records that never appear
      in the 105D observation, so equal observations can carry different
      returns.  The signed form needs only ``Phi(q_{t-1})``, and the actor's
      ``joint_pos_history`` term already contains both ``q_{t-1}`` and ``q_t``.

    Two rules this implementation deliberately follows:

    * No discount factor.  ``gamma * Phi' - Phi`` would charge
      ``(gamma - 1) * Phi`` every step just for existing, which at weight 3000
      and 30 Hz is about -1.0 per step, and would make dropping a stick to end
      the episode early the cheapest way to stop the bleeding.  The plain
      difference telescopes exactly and has neither problem.
    * No asymmetric scaling.  Charging retreat at a fraction of the advance
      rate reopens farming: an advance/retreat cycle would net the difference
      every time.  Only the symmetric form is safe.

    ``best_joint_scores`` keeps tracking the per-episode maximum, but it is now
    diagnostic only and no longer touches the payout.  Keeping it means the
    ``best`` vs ``current`` gap stays plotted on the same axis as the
    best-so-far runs, which is exactly the A/B readout for this change.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_joint_scores = torch.zeros_like(self._best_joint_scores)

    @property
    def prev_joint_scores(self) -> torch.Tensor:
        """Return the previous-step scores that define the signed payout."""
        return self._prev_joint_scores

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._prev_joint_scores[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        reference_joint_positions: tuple[float, ...],
        joint_linear_range: float,
        palm_cfg: SceneEntityCfg,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        thumb_cfg: SceneEntityCfg,
        stick1_reference_position_p: tuple[float, float, float],
        stick1_reference_quaternion_p: tuple[float, float, float, float],
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        stick1_half_extent: tuple[float, float, float],
        long_axis: int,
        pivot_station: float,
        position_sigma: float,
        orientation_sigma: float,
        thumb_sigma: float,
        pair_score_threshold: float,
        thumb_score_threshold: float,
        joint_reward_weights: tuple[float, ...] | None = None,
    ) -> torch.Tensor:
        if joint_linear_range <= 0.0:
            raise ValueError("joint_linear_range must be positive")

        stage1_ready = setting_stage1_gate(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            thumb_cfg,
            stick1_reference_position_p,
            stick1_reference_quaternion_p,
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
            stick1_half_extent,
            long_axis,
            pivot_station,
            position_sigma,
            orientation_sigma,
            thumb_sigma,
            pair_score_threshold,
            thumb_score_threshold,
        ).bool()

        robot: Articulation = env.scene[asset_cfg.name]
        reference = torch.as_tensor(
            reference_joint_positions,
            dtype=robot.data.joint_pos.dtype,
            device=env.device,
        )
        joint_count = len(asset_cfg.joint_ids)
        if reference.numel() != joint_count:
            raise ValueError(
                "Stage-1 reference size must match the selected joints: "
                f"reference={reference.numel()}, joint_count={joint_count}"
            )

        joint_abs_error = torch.abs(
            robot.data.joint_pos[:, asset_cfg.joint_ids] - reference
        )
        joint_scores = torch.clamp(
            1.0 - joint_abs_error / joint_linear_range,
            min=0.0,
            max=1.0,
        )
        if joint_scores.shape != self._best_joint_scores.shape:
            raise ValueError(
                "Stage-1 joint-score shape changed after initialization: "
                f"current={tuple(joint_scores.shape)}, "
                f"expected={tuple(self._best_joint_scores.shape)}"
            )
        self._current_joint_scores = joint_scores

        newly_unlocked = (~self._stage1_unlocked) & stage1_ready
        self._stage1_unlocked |= stage1_ready
        unlocked = self._stage1_unlocked.unsqueeze(-1)

        # The unlock step establishes the baseline and pays zero, matching the
        # best-so-far seeding so pre-Stage-1 motion is never paid for.
        self._prev_joint_scores = torch.where(
            newly_unlocked.unsqueeze(-1),
            joint_scores,
            self._prev_joint_scores,
        )
        # Signed on purpose: no clamp.  Losing ground costs exactly what
        # gaining it paid, which is the whole point of this term.
        joint_delta = (joint_scores - self._prev_joint_scores) * unlocked.float()
        self._prev_joint_scores = torch.where(
            unlocked,
            joint_scores,
            self._prev_joint_scores,
        )

        # Diagnostic only from here down; the payout above never reads it.
        self._best_joint_scores = torch.where(
            newly_unlocked.unsqueeze(-1),
            joint_scores,
            self._best_joint_scores,
        )
        self._best_joint_scores = torch.where(
            unlocked,
            torch.maximum(self._best_joint_scores, joint_scores),
            self._best_joint_scores,
        )

        # Same normalization as the best-so-far term, so weight 3000 keeps its
        # meaning and the two A/B runs share one reward scale.
        return torch.sum(
            joint_delta * self._joint_reward_weights,
            dim=-1,
        ) / self._joint_reward_weight_sum


# [hand_setting] Pay fingertip contact only on the face that tip belongs on.
def fingertip_face_contact_strength(
    env: ManagerBasedEnv,
    index_cfg: SceneEntityCfg,
    middle_cfg: SceneEntityCfg,
    ring_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    index_sensor: str,
    middle_sensor: str,
    ring_sensor: str,
    object_half_extent: tuple[float, float, float],
    surface_axis: int = 2,
    index_surface_sign: float = 1.0,
    middle_surface_sign: float = -1.0,
    ring_surface_sign: float = -1.0,
    force_scale: float = 0.10,
) -> torch.Tensor:
    """Mean saturated fingertip contact, weighted by being on the right face.

    ``contact_group_strength`` is a force magnitude and
    ``body_box_surface_distance`` is an unsigned distance, so nothing in the
    objective could tell which face of a shaft a fingertip was pressing.  Run
    2026-08-26_10-59-50 measured the consequence: the middle tip settled at
    face_z +16.1 mm on Stick1 at 0.649 N -- the upper face, where it belongs on
    the lower one -- and the index tip sat further out on that same side at
    +54.75 mm and had never once registered contact.  The middle finger was
    occupying the index finger's place and being paid 50 of the 161 points the
    policy earned.

    Each tip is scaled by ``clamp(sign * face_coordinate / half_extent, 0, 1)``:
    one on the correct face, zero on the wrong one or on a side face, and a
    gradient in between as it comes around the shaft.  The scaling is on the
    payout only, so this is meant to be registered *alongside* a face-blind
    contact term rather than replacing it -- a wrong-face contact then still
    earns the face-blind share instead of dropping to zero, and moving to the
    correct face is a gain rather than the recovery of a loss.  Every hard
    all-or-nothing structure in this task has stalled or collapsed
    (functional_contact_min has never once paid, and the Stage-1 gate cost run
    2026-08-26_00-24-02 its entire reward in eighteen iterations).

    Per the manager's resolution rules the five SceneEntityCfg arguments are
    separate named parameters; a list of them would silently keep body_ids at
    None.  The sensor arguments are plain strings and are safe to group.
    """
    if surface_axis not in (0, 1, 2):
        raise ValueError("surface_axis must be 0, 1, or 2")
    if force_scale <= 0.0:
        raise ValueError("force_scale must be positive")
    half_extent = float(object_half_extent[surface_axis])
    if half_extent <= 0.0:
        raise ValueError("object_half_extent must be positive on surface_axis")

    scores = []
    for body_cfg, object_cfg, sensor_name, surface_sign in (
        (index_cfg, stick1_cfg, index_sensor, index_surface_sign),
        (middle_cfg, stick1_cfg, middle_sensor, middle_surface_sign),
        (ring_cfg, stick2_cfg, ring_sensor, ring_surface_sign),
    ):
        if surface_sign not in (-1.0, 1.0):
            raise ValueError("surface signs must be -1 or 1")
        contact = torch.clamp(
            _sensor_force(env, sensor_name) / force_scale,
            min=0.0,
            max=1.0,
        )
        robot: Articulation = env.scene[body_cfg.name]
        obj: RigidObject = env.scene[object_cfg.name]
        body_position_o = quat_apply_inverse(
            obj.data.root_quat_w,
            robot.data.body_pos_w[:, body_cfg.body_ids[0]]
            - obj.data.root_pos_w,
        )
        face_factor = torch.clamp(
            surface_sign * body_position_o[:, surface_axis] / half_extent,
            min=0.0,
            max=1.0,
        )
        scores.append(contact * face_factor)
    return torch.mean(torch.stack(scores, dim=-1), dim=-1)


# [hand_setting] Measure whether the index tip lies between the two stick shafts.
def index_between_sticks_geometry(
    env: ManagerBasedEnv,
    index_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick_half_extent: tuple[float, float, float],
    long_axis: int = 1,
    axial_half_length: float = 0.08,
    between_margin_fraction: float = 0.15,
    stick1_proximity_sigma: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return index position and score proxies for the inter-stick corridor.

    ``between_coordinate`` is the projection of the index-tip body origin onto
    the local line from the closest Stick1 shaft point to the closest Stick2
    shaft point.  Zero is Stick1's centerline and one is Stick2's centerline.
    The slab score is high only inside that interval, while the Stick1 shaft
    score keeps the index near its assigned functional-contact surface instead
    of rewarding any point in the infinite plane between the sticks.
    """
    if long_axis not in (0, 1, 2):
        raise ValueError("long_axis must be 0, 1, or 2")
    if axial_half_length <= 0.0:
        raise ValueError("axial_half_length must be positive")
    if not 0.0 < between_margin_fraction <= 0.5:
        raise ValueError("between_margin_fraction must be in (0, 0.5]")
    if stick1_proximity_sigma <= 0.0:
        raise ValueError("stick1_proximity_sigma must be positive")

    robot: Articulation = env.scene[index_cfg.name]
    stick1: RigidObject = env.scene[stick1_cfg.name]
    stick2: RigidObject = env.scene[stick2_cfg.name]
    index_position_w = robot.data.body_pos_w[:, index_cfg.body_ids[0]]
    local_axis = torch.zeros(
        3,
        dtype=index_position_w.dtype,
        device=index_position_w.device,
    )
    local_axis[long_axis] = 1.0

    def closest_shaft_point(stick: RigidObject) -> torch.Tensor:
        axis_w = quat_apply(
            stick.data.root_quat_w,
            local_axis.expand(env.num_envs, -1),
        )
        station = torch.sum(
            (index_position_w - stick.data.root_pos_w) * axis_w,
            dim=-1,
        )
        station = torch.clamp(
            station,
            min=-axial_half_length,
            max=axial_half_length,
        )
        return stick.data.root_pos_w + station.unsqueeze(-1) * axis_w

    stick1_shaft_point = closest_shaft_point(stick1)
    stick2_shaft_point = closest_shaft_point(stick2)
    separation = stick2_shaft_point - stick1_shaft_point
    separation_sq = torch.sum(torch.square(separation), dim=-1)
    valid_separation = separation_sq > 1.0e-12
    between_coordinate = torch.sum(
        (index_position_w - stick1_shaft_point) * separation,
        dim=-1,
    ) / torch.clamp(separation_sq, min=1.0e-12)
    between_coordinate = torch.where(
        valid_separation,
        between_coordinate,
        torch.zeros_like(between_coordinate),
    )
    stick1_side_score = torch.clamp(
        between_coordinate / between_margin_fraction,
        min=0.0,
        max=1.0,
    )
    stick2_side_score = torch.clamp(
        (1.0 - between_coordinate) / between_margin_fraction,
        min=0.0,
        max=1.0,
    )
    slab_score = torch.minimum(stick1_side_score, stick2_side_score)
    slab_score *= valid_separation.float()

    stick1_shaft_score = body_box_shaft_region_proximity(
        env,
        index_cfg,
        stick1_cfg,
        stick_half_extent,
        long_axis,
        axial_half_length,
        stick1_proximity_sigma,
    )
    between_score = torch.minimum(slab_score, stick1_shaft_score)
    return (
        between_coordinate,
        slab_score,
        stick1_shaft_score,
        between_score,
    )


# [hand_setting] Score a fingertip against one selected finite stick face.
def body_box_surface_region_geometry(
    env: ManagerBasedEnv,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    long_axis: int = 1,
    axial_half_length: float = 0.08,
    surface_axis: int = 2,
    surface_sign: float = 1.0,
    tangent_margin: float = 0.01,
    region_sigma: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return local face coordinate, finite-region distance, and score.

    The target is a broad rectangular patch on one selected object face, not
    one exact point.  The long-axis range rejects end-cap farming and the
    tangent margin tolerates the fingertip body's finite collision geometry.
    """
    if long_axis not in (0, 1, 2) or surface_axis not in (0, 1, 2):
        raise ValueError("long_axis and surface_axis must be 0, 1, or 2")
    if long_axis == surface_axis:
        raise ValueError("long_axis and surface_axis must differ")
    if surface_sign not in (-1.0, 1.0):
        raise ValueError("surface_sign must be -1 or 1")
    if axial_half_length <= 0.0:
        raise ValueError("axial_half_length must be positive")
    if tangent_margin < 0.0:
        raise ValueError("tangent_margin must be non-negative")
    if region_sigma <= 0.0:
        raise ValueError("region_sigma must be positive")

    robot: Articulation = env.scene[body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_position_w = robot.data.body_pos_w[:, body_cfg.body_ids[0]]
    body_position_o = quat_apply_inverse(
        obj.data.root_quat_w,
        body_position_w - obj.data.root_pos_w,
    )
    half_extent = torch.as_tensor(
        object_half_extent,
        dtype=body_position_o.dtype,
        device=body_position_o.device,
    )
    tangent_axis = next(
        axis for axis in range(3) if axis not in (long_axis, surface_axis)
    )
    surface_target = surface_sign * half_extent[surface_axis]
    surface_error = torch.abs(
        body_position_o[:, surface_axis] - surface_target
    )
    axial_excess = torch.clamp(
        torch.abs(body_position_o[:, long_axis]) - axial_half_length,
        min=0.0,
    )
    tangent_excess = torch.clamp(
        torch.abs(body_position_o[:, tangent_axis])
        - (half_extent[tangent_axis] + tangent_margin),
        min=0.0,
    )
    region_distance = torch.sqrt(
        torch.square(surface_error)
        + torch.square(axial_excess)
        + torch.square(tangent_excess)
    )
    region_score = torch.exp(-region_distance / region_sigma)
    return body_position_o[:, surface_axis], region_distance, region_score


# [hand_setting] Penalize Index--Stick2 support only outside the intended gap.
def index_stick2_contact_outside_between(
    env: ManagerBasedEnv,
    sensor_groups: tuple[tuple[str, ...], ...],
    index_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick_half_extent: tuple[float, float, float],
    force_scale: float = 0.10,
    long_axis: int = 1,
    axial_half_length: float = 0.08,
    between_margin_fraction: float = 0.15,
    stick1_proximity_sigma: float = 0.04,
) -> torch.Tensor:
    """Return bounded wrong-contact strength, exempting a valid between pose."""
    contact_strength = contact_group_strength(
        env,
        sensor_groups,
        force_scale,
        reduction="mean",
    )
    _, _, _, between_score = index_between_sticks_geometry(
        env,
        index_cfg,
        stick1_cfg,
        stick2_cfg,
        stick_half_extent,
        long_axis,
        axial_half_length,
        between_margin_fraction,
        stick1_proximity_sigma,
    )
    return contact_strength * (1.0 - between_score)


def _joint_reference_rmse_progress(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    reference_joint_positions: tuple[float, ...],
    joint_error_threshold: float,
    joint_error_start_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all-joint RMSE and its normalized Stage-2 progress."""
    if joint_error_threshold < 0.0:
        raise ValueError("joint_error_threshold must be non-negative")
    if joint_error_start_threshold <= joint_error_threshold:
        raise ValueError(
            "joint_error_start_threshold must exceed joint_error_threshold"
        )
    robot: Articulation = env.scene[asset_cfg.name]
    reference = torch.as_tensor(
        reference_joint_positions,
        dtype=robot.data.joint_pos.dtype,
        device=env.device,
    )
    joint_count = len(asset_cfg.joint_ids)
    if reference.numel() != joint_count:
        raise ValueError(
            "Stage-2 reference size must match the selected joints: "
            f"reference={reference.numel()}, joint_count={joint_count}"
        )
    joint_error = robot.data.joint_pos[:, asset_cfg.joint_ids] - reference
    joint_rmse = torch.sqrt(torch.mean(torch.square(joint_error), dim=-1))
    joint_progress = torch.clamp(
        (joint_error_start_threshold - joint_rmse)
        / (joint_error_start_threshold - joint_error_threshold),
        min=0.0,
        max=1.0,
    )
    return joint_rmse, joint_progress


# [hand_setting] Guide the index into the inter-stick corridor after Stage 1.
class Stage1IndexBetweenTracking(ManagerTermBase):
    """Use Index-between only as a fading Stage-1-to-Stage-2 scaffold."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stage1_unlocked = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        self._handoff_scale = torch.zeros(
            env.num_envs,
            dtype=torch.float,
            device=env.device,
        )
        self._stick1_reference_position = torch.as_tensor(
            cfg.params["stick1_reference_position_p"],
            device=env.device,
        )
        self._stick1_reference_quaternion = torch.as_tensor(
            cfg.params["stick1_reference_quaternion_p"],
            device=env.device,
        )
        self._stick2_reference_position = torch.as_tensor(
            cfg.params["stick2_reference_position_p"],
            device=env.device,
        )
        self._stick2_reference_quaternion = torch.as_tensor(
            cfg.params["stick2_reference_quaternion_p"],
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stage1_unlocked[env_ids] = False
        self._handoff_scale[env_ids] = 0.0

    @property
    def handoff_scale(self) -> torch.Tensor:
        """Return the live multiplier applied to the between scaffold."""
        return self._handoff_scale

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        index_cfg: SceneEntityCfg,
        stick_half_extent: tuple[float, float, float],
        axial_half_length: float,
        between_margin_fraction: float,
        stick1_proximity_sigma: float,
        palm_cfg: SceneEntityCfg,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        thumb_cfg: SceneEntityCfg,
        stick1_reference_position_p: tuple[float, float, float],
        stick1_reference_quaternion_p: tuple[float, float, float, float],
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        stick1_half_extent: tuple[float, float, float],
        long_axis: int,
        pivot_station: float,
        position_sigma: float,
        orientation_sigma: float,
        thumb_sigma: float,
        pair_score_threshold: float,
        thumb_score_threshold: float,
        asset_cfg: SceneEntityCfg,
        reference_joint_positions: tuple[float, ...],
        joint_error_threshold: float,
        joint_error_start_threshold: float,
    ) -> torch.Tensor:
        stage1_ready = setting_stage1_gate(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            thumb_cfg,
            stick1_reference_position_p,
            stick1_reference_quaternion_p,
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
            stick1_half_extent,
            long_axis,
            pivot_station,
            position_sigma,
            orientation_sigma,
            thumb_sigma,
            pair_score_threshold,
            thumb_score_threshold,
        ).bool()
        self._stage1_unlocked |= stage1_ready

        pair_score = _object_pair_reference_pose_min_score(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            self._stick1_reference_position,
            self._stick1_reference_quaternion,
            self._stick2_reference_position,
            self._stick2_reference_quaternion,
            position_sigma,
            orientation_sigma,
        )
        # The one-way latch preserves the directional signal after acquisition,
        # but this live factor prevents moving both sticks around the fingertip
        # from becoming an easier way to farm the between score.
        pair_maintenance = torch.clamp(
            pair_score / pair_score_threshold,
            min=0.0,
            max=1.0,
        )
        _, _, _, between_score = index_between_sticks_geometry(
            env,
            index_cfg,
            stick1_cfg,
            stick2_cfg,
            stick_half_extent,
            long_axis,
            axial_half_length,
            between_margin_fraction,
            stick1_proximity_sigma,
        )
        _, joint_progress = _joint_reference_rmse_progress(
            env,
            asset_cfg,
            reference_joint_positions,
            joint_error_threshold,
            joint_error_start_threshold,
        )
        # Between is only a route scaffold.  It starts at full strength after
        # Stage 1 is acquired, then vanishes continuously as the same q_ref
        # progress that activates Stage-2 contact rises from zero to one.
        self._handoff_scale = (
            self._stage1_unlocked.float() * (1.0 - joint_progress)
        )
        return (
            self._handoff_scale
            * pair_maintenance
            * between_score
        )


# [hand_setting] Guide Index to Stick1's selected upper face after Stage 1.
class Stage1IndexStick1SurfaceTracking(ManagerTermBase):
    """Reward a finite Stick1 face region without requiring inter-stick entry."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stage1_unlocked = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        self._stick1_reference_position = torch.as_tensor(
            cfg.params["stick1_reference_position_p"],
            device=env.device,
        )
        self._stick1_reference_quaternion = torch.as_tensor(
            cfg.params["stick1_reference_quaternion_p"],
            device=env.device,
        )
        self._stick2_reference_position = torch.as_tensor(
            cfg.params["stick2_reference_position_p"],
            device=env.device,
        )
        self._stick2_reference_quaternion = torch.as_tensor(
            cfg.params["stick2_reference_quaternion_p"],
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stage1_unlocked[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        index_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float],
        axial_half_length: float,
        surface_axis: int,
        surface_sign: float,
        tangent_margin: float,
        region_sigma: float,
        palm_cfg: SceneEntityCfg,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        thumb_cfg: SceneEntityCfg,
        stick1_reference_position_p: tuple[float, float, float],
        stick1_reference_quaternion_p: tuple[float, float, float, float],
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        stick1_half_extent: tuple[float, float, float],
        long_axis: int,
        pivot_station: float,
        position_sigma: float,
        orientation_sigma: float,
        thumb_sigma: float,
        pair_score_threshold: float,
        thumb_score_threshold: float,
    ) -> torch.Tensor:
        stage1_ready = setting_stage1_gate(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            thumb_cfg,
            stick1_reference_position_p,
            stick1_reference_quaternion_p,
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
            stick1_half_extent,
            long_axis,
            pivot_station,
            position_sigma,
            orientation_sigma,
            thumb_sigma,
            pair_score_threshold,
            thumb_score_threshold,
        ).bool()
        self._stage1_unlocked |= stage1_ready

        pair_score = _object_pair_reference_pose_min_score(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            self._stick1_reference_position,
            self._stick1_reference_quaternion,
            self._stick2_reference_position,
            self._stick2_reference_quaternion,
            position_sigma,
            orientation_sigma,
        )
        pair_maintenance = torch.clamp(
            pair_score / pair_score_threshold,
            min=0.0,
            max=1.0,
        )
        _, _, surface_score = body_box_surface_region_geometry(
            env,
            index_cfg,
            stick1_cfg,
            object_half_extent,
            long_axis,
            axial_half_length,
            surface_axis,
            surface_sign,
            tangent_margin,
            region_sigma,
        )
        return self._stage1_unlocked.float() * pair_maintenance * surface_score


# [hand_setting] Weak live semantic proximity after Stage-1 acquisition.
class Stage1SemanticSurfaceApproach(ManagerTermBase):
    """Keep the three missing fingertips near their assigned sticks.

    Unlike the q-reference acquisition term, this is deliberately a live
    current-state score rather than best-so-far progress.  If joint tracking
    moves a fingertip away after Stage 1, returning to the previous distance
    must immediately recover reward.  The Stage-1 latch only controls when the
    weak guide starts; it does not establish a distance baseline.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stage1_unlocked = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stage1_unlocked[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        index_cfg: SceneEntityCfg,
        middle_cfg: SceneEntityCfg,
        ring_cfg: SceneEntityCfg,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        stick_half_extent: tuple[float, float, float],
        approach_range: float,
        palm_cfg: SceneEntityCfg,
        thumb_cfg: SceneEntityCfg,
        stick1_reference_position_p: tuple[float, float, float],
        stick1_reference_quaternion_p: tuple[float, float, float, float],
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        stick1_half_extent: tuple[float, float, float],
        long_axis: int,
        pivot_station: float,
        position_sigma: float,
        orientation_sigma: float,
        thumb_sigma: float,
        pair_score_threshold: float,
        thumb_score_threshold: float,
    ) -> torch.Tensor:
        if approach_range <= 0.0:
            raise ValueError("approach_range must be positive")

        stage1_ready = setting_stage1_gate(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            thumb_cfg,
            stick1_reference_position_p,
            stick1_reference_quaternion_p,
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
            stick1_half_extent,
            long_axis,
            pivot_station,
            position_sigma,
            orientation_sigma,
            thumb_sigma,
            pair_score_threshold,
            thumb_score_threshold,
        ).bool()
        self._stage1_unlocked |= stage1_ready

        distances = torch.stack(
            (
                body_box_surface_distance(
                    env,
                    index_cfg,
                    stick1_cfg,
                    stick_half_extent,
                ),
                body_box_surface_distance(
                    env,
                    middle_cfg,
                    stick1_cfg,
                    stick_half_extent,
                ),
                body_box_surface_distance(
                    env,
                    ring_cfg,
                    stick2_cfg,
                    stick_half_extent,
                ),
            ),
            dim=-1,
        )
        scores = torch.clamp(
            1.0 - distances / approach_range,
            min=0.0,
            max=1.0,
        )
        mean_score = torch.mean(scores, dim=-1)
        min_score = torch.min(scores, dim=-1).values
        return self._stage1_unlocked.float() * (
            0.5 * mean_score + 0.5 * min_score
        )


# [hand_setting, parked] Historical finite-centerline approach reward.
class StickValleyCenterlineTracking(ManagerTermBase):
    """Bring any physical Stick2 shaft station through the valley."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reference_position = torch.as_tensor(
            cfg.params["stick2_reference_position_p"],
            device=env.device,
        )
        self._reference_quaternion = torch.as_tensor(
            cfg.params["stick2_reference_quaternion_p"],
            device=env.device,
        )
        self._valley_offset = torch.as_tensor(
            cfg.params["valley_point_offset_o"],
            device=env.device,
        )
        self._stick_half_length = float(
            cfg.params.get("stick_half_length", 0.09)
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        palm_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        valley_point_offset_o: tuple[float, float, float],
        stick_half_length: float = 0.09,
        point_sigma: float = 0.05,
        axis_sigma: float = 1.0,
        valley_point_error_limit: float = 0.005,
        valley_axis_error_limit: float = 0.1745329252,
    ) -> torch.Tensor:
        del (
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
            valley_point_offset_o,
            stick_half_length,
            valley_point_error_limit,
            valley_axis_error_limit,
        )
        position, quaternion = _object_pose_in_palm(
            env,
            palm_cfg,
            stick2_cfg,
        )
        point_error, axis_error = _stick_valley_geometry(
            position,
            quaternion,
            self._reference_position.expand_as(position),
            self._reference_quaternion.expand_as(quaternion),
            self._valley_offset.expand_as(position),
            self._stick_half_length,
        )
        point_score = torch.exp(-point_error / point_sigma)
        axis_score = torch.exp(-axis_error / axis_sigma)
        # The previous arithmetic mean allowed the policy to preserve one
        # easy component while abandoning the other.  A hard minimum makes
        # the weaker of translation and shaft-axis alignment determine the
        # score.  PPO does not differentiate this expression through the
        # simulator, so the non-smooth equality point is not a concern.
        return torch.minimum(point_score, axis_score)


# [hand_setting, parked] Historical split pose-deviation penalty.
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


# [hand_grasp] Active Stick1 pivot-preservation reward.
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


# [hand_grasp, parked] Older contact/anchor-gated mode-gap reward.
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


# [hand_grasp] Active OPEN/CLOSE gap, lateral, and axial tracking reward.
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
    lateral_sigma: float | None = 0.005,
    axial_sigma: float = 0.005,
    clamp_gap: bool = True,
) -> torch.Tensor:
    """Track mode gap plus transverse and axial distal-tip alignment.

    ``lateral_sigma=None`` drops the lateral term from this exponent
    (2026-08-07).  All three errors sharing one exponent means they multiply:
    when lateral is bad the *gap* gradient is scaled down by the same factor, so
    neither improves.  ``hand_move`` moves lateral out to its own additive term
    (``tip_lateral_alignment``) and passes ``None`` here; ``hand_grasp`` and
    ``hand_setting`` keep the combined form they were tuned under.
    """
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
        clamp_gap=clamp_gap,
    )
    axial_error = torch.abs(
        axial_offset - float(reference_axial_offset_stick2)
    )
    exponent = (
        torch.abs(gap - target_gap) / sigma + axial_error / axial_sigma
    )
    if lateral_sigma is not None:
        exponent = exponent + lateral_error / lateral_sigma
    return mode_gate * torch.exp(-exponent)


# [hand_move] Keep the two distal tips from sliding past each other sideways.
def tip_lateral_alignment(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick1_tip_offset_o: tuple[float, float, float],
    stick2_tip_offset_o: tuple[float, float, float],
    stick_thickness: float,
    reference_separation_direction_stick2: tuple[float, float, float],
    sigma: float = 0.005,
) -> torch.Tensor:
    """Reward the two tip *points* staying in the opening/closing plane.

    This is the position half of "lateral"; ``stick_axis_lateral_alignment`` is
    the direction half.  Two lines in 3D meet only if both agree, and until
    2026-08-07 only the position half was measured - inside a shared exponent
    where it could not act.

    Why it had to come out of that exponent
    ---------------------------------------
    ``mode_tip_gap_tracking`` and ``mode_grasp_stability`` divide gap, lateral
    and axial into **one** exponent, so the three multiply.  With the term
    already sitting near 5% of its maximum, the lateral gradient was scaled by
    that same 0.05 and could not pull anything back.  As an additive term it
    keeps full gradient no matter how the gap is doing.

    What it prevents, concretely
    ----------------------------
    The sticks are 7 mm square.  Once the tips are offset sideways by more than
    that the two sections cannot overlap at all: closing drives them *past* each
    other rather than together.  Measured 2026-08-07 on ``hand_move``:
    ``tip_lateral_error`` 9.14 mm - crossing, visibly.

    The gap term cannot object, because ``transverse_distance`` is a norm and
    does not know which direction the tips are apart in.  Nine millimetres of
    sideways offset and nine millimetres of closing distance look identical to
    it, and after the support radii are subtracted both report ``gap = 0``, a
    perfect CLOSE.  This term is what makes the difference visible.

    ``sigma`` defaults to ``TIP_LATERAL_ERROR_LIMIT`` (5 mm), the value the
    success termination already treats as the pass mark, and comfortably below
    the 7 mm section width where overlap is lost entirely.
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
    return torch.exp(-lateral_error / max(float(sigma), 1.0e-8))


# [hand_move] Keep the two shafts in the opening/closing plane.
def stick_axis_lateral_alignment(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    reference_separation_direction_stick2: tuple[float, float, float],
    sigma: float = 0.05, #radian
) -> torch.Tensor:
    """Reward Stick1's shaft **direction** staying in the open/close plane.

    Why this exists as its own term
    -------------------------------
    Two lines in 3D meet only if both their positions and their directions
    agree.  ``tip_lateral_error`` (inside ``mode_tip_gap_tracking`` and
    ``mode_grasp_stability``) covers the position half: it measures how far the
    two *tip points* sit apart along the plane normal.  Nothing measured the
    direction half - ``_tip_geometry_from_palm_poses`` never even computes
    Stick1's shaft axis - so the sticks were free to close while tilted out of
    plane, meeting at an angle and sliding rather than pinching.

    Measured 2026-08-07 on ``hand_move`` at it 7402: 3.05 deg mean skew that
    never dropped below 0.18 deg, i.e. a standing bias rather than jitter, and
    identical in OPEN (5.31) and CLOSE (5.28) - which is why this term carries
    **no mode gate**.  Closing from an already-skewed pose cannot be fixed
    during CLOSE.

    What is deliberately *not* constrained
    -------------------------------------
    Only the plane-normal component.  The in-plane angle between the shafts is
    the opening/closing degree of freedom itself (measured swing roughly -9 to
    +7 deg), so a term driving the axes parallel would fight the mode command.
    That is the whole reason this projects onto ``n`` instead of using
    ``dot(axis1, axis2)``.

    The plane is spanned by Stick2's shaft and
    ``reference_separation_direction_stick2``; its normal is the same direction
    ``lateral_error`` projects the tip delta onto, so "lateral" means one thing
    whether it is measured on the tips or on the shafts.

    Note on the target
    ------------------
    Target is zero even though the validated ``pose_005`` sits at about 5.2 deg
    (``Metrics/hand_grasp/reset_axis_skew_angle``).  That pose was validated for
    *holding the sticks*, never for *pinching with the tips*, and the reference
    direction itself was derived from its tip positions - which pins the tip
    lateral offset to zero there but says nothing about the axes.  The policy
    also drifted from 5.2 to 3.05 deg unprompted, so nothing appears to be
    holding it at the reset value.
    """
    stick1_position_p, stick1_quaternion_p = _object_pose_in_palm(
        env, palm_cfg, stick1_cfg
    )
    stick2_position_p, stick2_quaternion_p = _object_pose_in_palm(
        env, palm_cfg, stick2_cfg
    )
    del stick1_position_p, stick2_position_p  # direction only; position is the other half

    num_envs = stick1_quaternion_p.shape[0]
    dtype = stick1_quaternion_p.dtype
    device = stick1_quaternion_p.device
    local_y = torch.tensor((0.0, 1.0, 0.0), dtype=dtype, device=device).expand(
        num_envs, -1
    )
    stick1_axis_p = quat_apply(stick1_quaternion_p, local_y)
    stick1_axis_stick2 = quat_apply_inverse(stick2_quaternion_p, stick1_axis_p)

    reference = torch.as_tensor(
        reference_separation_direction_stick2, dtype=dtype, device=device
    )
    reference_xz = reference[[0, 2]]
    reference_xz = reference_xz / torch.clamp(
        torch.linalg.vector_norm(reference_xz), min=1.0e-8
    )
    # Plane normal (-ref_z, 0, ref_x).  Stick2's own axis is (0, 1, 0) in this
    # frame and so lies in the plane by construction; only Stick1 can leave it.
    out_of_plane = (
        -reference_xz[1] * stick1_axis_stick2[:, 0]
        + reference_xz[0] * stick1_axis_stick2[:, 2]
    )
    # Unit axes, so the normal component is the sine of the angle to the plane.
    skew_angle = torch.asin(torch.clamp(torch.abs(out_of_plane), min=0.0, max=1.0))
    return torch.exp(-skew_angle / max(float(sigma), 1.0e-8))


# [hand_grasp, parked] Standalone lateral penalty retained for comparison.
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


# [hand_grasp] Active six-contact strength reward.
def _normalized_group_force_scales(
    force_scale: float | tuple[float, ...],
    group_count: int,
) -> tuple[float, ...]:
    """Return one positive force scale per semantic contact group."""
    if isinstance(force_scale, (int, float)):
        scales = (float(force_scale),) * group_count
    else:
        scales = tuple(float(value) for value in force_scale)
        if len(scales) != group_count:
            raise ValueError(
                "Per-group force_scale must match sensor_groups: "
                f"got {len(scales)} scales for {group_count} groups."
            )
    if any(value <= 0.0 for value in scales):
        raise ValueError(f"force_scale values must be positive, got {scales}.")
    return scales


def contact_group_strength(
    env: ManagerBasedEnv,
    sensor_groups: tuple[tuple[str, ...], ...],
    force_scale: float | tuple[float, ...] = 0.10,
    reduction: str = "mean",
    group_reduction: str = "max",
) -> torch.Tensor:
    """Dense contact strength with configurable within-group credit.

    ``force_scale`` may be one shared scalar or one value per semantic group.
    ``max`` and ``min`` preserve the historical OR and hard-AND semantics.
    ``mean_strength`` first saturates each physical contact independently and
    then averages them.  ``partial_and_bonus`` preserves singleton groups and,
    for a two-surface group, returns ``0.25 * sum + 0.5 * min``.  Therefore one
    loaded surface earns 0.25 while both loaded surfaces earn 1.0.
    """
    force_scales = _normalized_group_force_scales(
        force_scale, len(sensor_groups)
    )
    if group_reduction in ("mean_strength", "partial_and_bonus"):
        group_strengths = []
        for group_index, sensor_names in enumerate(sensor_groups):
            sensor_strengths = torch.stack(
                [
                    torch.clamp(
                        _sensor_force(env, sensor_name)
                        / force_scales[group_index],
                        min=0.0,
                        max=1.0,
                    )
                    for sensor_name in sensor_names
                ],
                dim=-1,
            )
            if group_reduction == "partial_and_bonus" and len(sensor_names) > 1:
                if len(sensor_names) != 2:
                    raise ValueError(
                        "partial_and_bonus requires singleton or two-sensor groups"
                    )
                partial_credit = 0.25 * torch.sum(sensor_strengths, dim=-1)
                both_contact_bonus = 0.5 * torch.min(
                    sensor_strengths, dim=-1
                ).values
                group_strengths.append(partial_credit + both_contact_bonus)
            else:
                group_strengths.append(torch.mean(sensor_strengths, dim=-1))
        strengths = torch.stack(group_strengths, dim=-1)
    else:
        group_forces = _group_forces(env, sensor_groups, group_reduction)
        scale_tensor = torch.as_tensor(
            force_scales,
            device=group_forces.device,
            dtype=group_forces.dtype,
        ).unsqueeze(0)
        strengths = torch.clamp(
            group_forces / scale_tensor,
            min=0.0,
            max=1.0,
        )
    if reduction == "mean":
        return torch.mean(strengths, dim=-1)
    if reduction == "min":
        return torch.min(strengths, dim=-1).values
    raise ValueError(f"Unsupported contact reduction: {reduction}")


# [hand_real2] Per-step bonus for retaining every semantic contact.
def full_contact_bonus(
    env: ManagerBasedEnv,
    sensor_groups: tuple[tuple[str, ...], ...],
    contact_threshold: float = 0.02,
    group_reduction: str = "max",
) -> torch.Tensor:
    """Return one while every hard contact group exceeds the threshold.

    This is deliberately memoryless and does not terminate on contact loss, so
    a policy that drops a contact can continue acting and learn recovery.
    """
    group_forces = _group_forces(env, sensor_groups, group_reduction)
    return torch.all(group_forces >= contact_threshold, dim=-1).to(
        dtype=group_forces.dtype
    )


# [hand_setting] Apply partial or all-six contact strength only after Stage 1.
def stage1_gated_contact_group_strength(
    env: ManagerBasedEnv,
    sensor_groups: tuple[tuple[str, ...], ...],
    force_scale: float,
    reduction: str,
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    thumb_cfg: SceneEntityCfg,
    stick1_reference_position_p: tuple[float, float, float],
    stick1_reference_quaternion_p: tuple[float, float, float, float],
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    stick1_half_extent: tuple[float, float, float],
    long_axis: int,
    pivot_station: float,
    position_sigma: float,
    orientation_sigma: float,
    thumb_sigma: float,
    pair_score_threshold: float,
    thumb_score_threshold: float,
) -> torch.Tensor:
    """Build functional contacts without allowing pre-alignment contact farming."""
    gate = setting_stage1_gate(
        env,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        thumb_cfg,
        stick1_reference_position_p,
        stick1_reference_quaternion_p,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        stick1_half_extent,
        long_axis,
        pivot_station,
        position_sigma,
        orientation_sigma,
        thumb_sigma,
        pair_score_threshold,
        thumb_score_threshold,
    )
    strength = contact_group_strength(
        env,
        sensor_groups,
        force_scale,
        reduction,
    )
    return gate * strength


# [hand_setting] Contact shaping with a one-way Stage-1 hand-off.
class Stage2ContactGroupStrength(ManagerTermBase):
    """Keep contact shaping alive after Stage 1 is acquired once per episode."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stage1_unlocked = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )
        self._contact_progress = torch.zeros(
            env.num_envs,
            dtype=torch.float,
            device=env.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stage1_unlocked[env_ids] = False
        self._contact_progress[env_ids] = 0.0

    @property
    def stage1_unlocked(self) -> torch.Tensor:
        """Return the per-environment one-way Stage-1 latch."""
        return self._stage1_unlocked

    @property
    def contact_progress(self) -> torch.Tensor:
        """Return the multiplier currently applied to contact strength."""
        return self._contact_progress

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_groups: tuple[tuple[str, ...], ...],
        force_scale: float,
        reduction: str,
        asset_cfg: SceneEntityCfg,
        reference_joint_positions: tuple[float, ...],
        joint_error_threshold: float,
        palm_cfg: SceneEntityCfg,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        thumb_cfg: SceneEntityCfg,
        stick1_reference_position_p: tuple[float, float, float],
        stick1_reference_quaternion_p: tuple[float, float, float, float],
        stick2_reference_position_p: tuple[float, float, float],
        stick2_reference_quaternion_p: tuple[float, float, float, float],
        stick1_half_extent: tuple[float, float, float],
        long_axis: int,
        pivot_station: float,
        position_sigma: float,
        orientation_sigma: float,
        thumb_sigma: float,
        pair_score_threshold: float,
        thumb_score_threshold: float,
        joint_error_start_threshold: float | None = None,
        index_cfg: SceneEntityCfg | None = None,
        index_between_axial_half_length: float = 0.08,
        index_between_margin_fraction: float = 0.15,
        index_stick1_proximity_sigma: float = 0.04,
        index_between_progress_start: float = 0.25,
        index_between_ready_threshold: float = 0.75,
    ) -> torch.Tensor:
        stage1_ready = setting_stage1_gate(
            env,
            palm_cfg,
            stick1_cfg,
            stick2_cfg,
            thumb_cfg,
            stick1_reference_position_p,
            stick1_reference_quaternion_p,
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
            stick1_half_extent,
            long_axis,
            pivot_station,
            position_sigma,
            orientation_sigma,
            thumb_sigma,
            pair_score_threshold,
            thumb_score_threshold,
        ).bool()
        self._stage1_unlocked |= stage1_ready

        if joint_error_start_threshold is None:
            robot: Articulation = env.scene[asset_cfg.name]
            reference = torch.as_tensor(
                reference_joint_positions,
                dtype=robot.data.joint_pos.dtype,
                device=env.device,
            )
            joint_abs_error = torch.abs(
                robot.data.joint_pos[:, asset_cfg.joint_ids] - reference
            )
            joint_progress = (
                torch.max(joint_abs_error, dim=-1).values
                <= joint_error_threshold
            ).float()
        else:
            _, joint_progress = _joint_reference_rmse_progress(
                env,
                asset_cfg,
                reference_joint_positions,
                joint_error_threshold,
                joint_error_start_threshold,
            )

        contact_progress = self._stage1_unlocked.float() * joint_progress
        if index_cfg is not None:
            if not (
                0.0 <= index_between_progress_start
                < index_between_ready_threshold
                <= 1.0
            ):
                raise ValueError(
                    "index-between progress thresholds must satisfy "
                    "0 <= start < ready <= 1"
                )
            _, _, _, index_between_score = index_between_sticks_geometry(
                env,
                index_cfg,
                stick1_cfg,
                stick2_cfg,
                stick1_half_extent,
                long_axis,
                index_between_axial_half_length,
                index_between_margin_fraction,
                index_stick1_proximity_sigma,
            )
            index_between_progress = torch.clamp(
                (index_between_score - index_between_progress_start)
                / (
                    index_between_ready_threshold
                    - index_between_progress_start
                ),
                min=0.0,
                max=1.0,
            )
            contact_progress *= index_between_progress

        self._contact_progress = contact_progress
        return contact_progress * contact_group_strength(
            env,
            sensor_groups,
            force_scale,
            reduction,
        )


# [parked] Generic prerequisite-gated contact shaping helper.
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


# [hand_setting, parked] Pose-plus-contact Stick2 seating gate.
def stick2_seated_gate(
    env: ManagerBasedEnv,
    anchor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    contact_threshold: float = 0.02,
    position_error_limit: float = 0.015,
    orientation_error_limit: float = 0.3490658504,
) -> torch.Tensor:
    """Return one only after Stick2 is physically seated in its valley pose.

    Pair contact alone is insufficient because palm/Stick2 and
    thumb-middle/Stick2 can both fire on the outside of the valley.  The
    palm-relative pose gate prevents that false positive without adding a
    hidden phase variable.
    """
    anchors_valid = torch.all(
        _group_forces(env, anchor_groups) >= contact_threshold,
        dim=-1,
    )
    position, quaternion = _object_pose_in_palm(
        env,
        palm_cfg,
        stick2_cfg,
    )
    reference_position = torch.as_tensor(
        stick2_reference_position_p,
        dtype=position.dtype,
        device=position.device,
    )
    reference_quaternion = torch.as_tensor(
        stick2_reference_quaternion_p,
        dtype=quaternion.dtype,
        device=quaternion.device,
    )
    position_error, orientation_error = _pose_errors(
        position,
        quaternion,
        reference_position,
        reference_quaternion,
    )
    pose_valid = (
        (position_error <= position_error_limit)
        & (orientation_error <= orientation_error_limit)
    )
    return (anchors_valid & pose_valid).float()


# [hand_setting, parked] Prepared palm/thumb support reward near Stick2 reference.
def stick2_pose_anchor_support_strength(
    env: ManagerBasedEnv,
    anchor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    force_scale: float = 0.10,
    position_sigma: float = 0.10,
    orientation_sigma: float = 1.5707963268,
    contact_threshold: float = 0.02,
    position_error_limit: float = 0.015,
    orientation_error_limit: float = 0.3490658504,
    support_position_error_limit: float = 0.03,
    support_orientation_error_limit: float = 0.5235987756,
) -> torch.Tensor:
    """Reward reciprocal Stick2 support only near its known stable hand pose.

    The stable ``hand_grasp`` reset already supplies a validated Stick2
    palm-relative pose.  Using that pose as the semantic valley avoids an
    additional hand-space proxy.  A loose pose gate exposes the two-sided
    support signal before the strict seated gate, while ``min`` prevents the
    easy palm contact from compensating for missing thumb-middle support.
    """
    del contact_threshold, position_error_limit, orientation_error_limit
    position, quaternion = _object_pose_in_palm(
        env,
        palm_cfg,
        stick2_cfg,
    )
    reference_position = torch.as_tensor(
        stick2_reference_position_p,
        dtype=position.dtype,
        device=position.device,
    )
    reference_quaternion = torch.as_tensor(
        stick2_reference_quaternion_p,
        dtype=quaternion.dtype,
        device=quaternion.device,
    )
    position_error, orientation_error = _pose_errors(
        position,
        quaternion,
        reference_position,
        reference_quaternion,
    )
    pose_score = torch.minimum(
        torch.exp(-position_error / position_sigma),
        torch.exp(-orientation_error / orientation_sigma),
    )
    support_ready = (
        (position_error <= support_position_error_limit)
        & (orientation_error <= support_orientation_error_limit)
    )
    anchor_strength = contact_group_strength(
        env,
        anchor_groups,
        force_scale,
        reduction="min",
    )
    return pose_score * anchor_strength * support_ready.float()


# [hand_setting, parked] Unlock later contact rewards after Stick2 seating.
def seated_gated_contact_group_strength(
    env: ManagerBasedEnv,
    target_groups: tuple[tuple[str, ...], ...],
    anchor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    force_scale: float = 0.10,
    reduction: str = "mean",
    contact_threshold: float = 0.02,
    position_error_limit: float = 0.015,
    orientation_error_limit: float = 0.3490658504,
) -> torch.Tensor:
    """Expose target contact reward only after the lower stick is seated."""
    target_strength = contact_group_strength(
        env,
        target_groups,
        force_scale,
        reduction,
    )
    seated = stick2_seated_gate(
        env,
        anchor_groups,
        palm_cfg,
        stick2_cfg,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        contact_threshold,
        position_error_limit,
        orientation_error_limit,
    )
    return target_strength * seated


# [hand_setting metric] Body-origin distance used only for approach diagnosis.
def body_box_surface_distance(
    env: ManagerBasedEnv,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
) -> torch.Tensor:
    """Measure a body-origin proxy to the nearest oriented-box surface."""
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
    return torch.linalg.vector_norm(outside, dim=-1)


# [hand_setting, parked] Broad link-to-stick surface proximity reward.
def body_box_surface_proximity(
    env: ManagerBasedEnv,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    sigma: float = 0.02,
) -> torch.Tensor:
    """Dense exponential proximity from one body origin to an oriented box surface."""
    surface_distance = body_box_surface_distance(
        env,
        body_cfg,
        object_cfg,
        object_half_extent,
    )
    return torch.exp(-surface_distance / sigma)


# [hand_setting, parked] Central-shaft semantic proximity reward.
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


# [hand_setting, parked] Unlock shaft proximity only after Stick2 seating.
def seated_gated_body_box_shaft_region_proximity(
    env: ManagerBasedEnv,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    anchor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    long_axis: int = 1,
    axial_half_length: float = 0.075,
    sigma: float = 0.02,
    contact_threshold: float = 0.02,
    position_error_limit: float = 0.015,
    orientation_error_limit: float = 0.3490658504,
) -> torch.Tensor:
    """Enable one fingertip approach signal only after Stick2 seating."""
    proximity = body_box_shaft_region_proximity(
        env,
        body_cfg,
        object_cfg,
        object_half_extent,
        long_axis,
        axial_half_length,
        sigma,
    )
    seated = stick2_seated_gate(
        env,
        anchor_groups,
        palm_cfg,
        stick2_cfg,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        contact_threshold,
        position_error_limit,
        orientation_error_limit,
    )
    return proximity * seated


# [hand_setting, parked] Hard gate for the historical finite-shaft proxy.
def stick2_valley_geometry_gate(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    valley_point_offset_o: tuple[float, float, float],
    stick_half_length: float = 0.09,
    valley_point_error_limit: float = 0.005,
    valley_axis_error_limit: float = 0.1745329252,
) -> torch.Tensor:
    """Return one when the Stick2 centerline occupies the valley corridor."""
    point_error, axis_error = stick2_valley_geometry(
        env,
        palm_cfg,
        stick2_cfg,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        valley_point_offset_o,
        stick_half_length,
    )
    return (
        (point_error <= valley_point_error_limit)
        & (axis_error <= valley_axis_error_limit)
    ).float()


# [hand_setting, parked] Historical shaft geometry plus reciprocal-contact gate.
def stick2_in_valley_gate(
    env: ManagerBasedEnv,
    anchor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    valley_point_offset_o: tuple[float, float, float],
    stick_half_length: float = 0.09,
    contact_threshold: float = 0.02,
    valley_point_error_limit: float = 0.01,
    valley_axis_error_limit: float = 0.2617993878,
) -> torch.Tensor:
    """Require both valley geometry and reciprocal physical anchor support.

    The anchor contacts used here must have their own ungated shaping reward.
    This avoids the former circular dependency in which the policy needed the
    contacts to unlock the very actions/rewards that could create them.
    """
    geometry_valid = stick2_valley_geometry_gate(
        env,
        palm_cfg,
        stick2_cfg,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        valley_point_offset_o,
        stick_half_length,
        valley_point_error_limit,
        valley_axis_error_limit,
    )
    anchors_valid = torch.all(
        _group_forces(env, anchor_groups) >= contact_threshold,
        dim=-1,
    )
    return geometry_valid * anchors_valid.float()


# [hand_setting, parked] Historical finite-shaft anchor support reward.
def valley_anchor_support_strength(
    env: ManagerBasedEnv,
    anchor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    valley_point_offset_o: tuple[float, float, float],
    stick_half_length: float = 0.09,
    force_scale: float = 0.10,
    point_sigma: float = 0.05,
    axis_sigma: float = 1.0,
    contact_threshold: float = 0.02,
    valley_point_error_limit: float = 0.01,
    valley_axis_error_limit: float = 0.2617993878,
    support_point_error_limit: float = 0.02,
    support_axis_error_limit: float = 0.5235987756,
) -> torch.Tensor:
    """Reward two-sided Stick2 support inside a loose valley corridor.

    The loose corridor opens before the hard ``stick2_in_valley`` gate and
    supplies the palm/thumb-middle force signal needed to complete seating.
    ``min`` prevents the already-easy palm contact from compensating for a
    missing thumb-middle contact.
    """
    del (
        contact_threshold,
        valley_point_error_limit,
        valley_axis_error_limit,
    )
    point_error, axis_error = stick2_valley_geometry(
        env,
        palm_cfg,
        stick2_cfg,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        valley_point_offset_o,
        stick_half_length,
    )
    geometry_score = torch.minimum(
        torch.exp(-point_error / point_sigma),
        torch.exp(-axis_error / axis_sigma),
    )
    anchor_strength = contact_group_strength(
        env,
        anchor_groups,
        force_scale,
        reduction="min",
    )
    # An exponential score never reaches exactly zero.  Without this loose
    # corridor, the policy could press Stick2 against the outside of the hand
    # and collect a small anchor-force annuity forever.  The corridor is wider
    # than the final in-valley gate, so it still supplies force shaping before
    # exact seating.
    support_ready = (
        (point_error <= support_point_error_limit)
        & (axis_error <= support_axis_error_limit)
    )
    return geometry_score * anchor_strength * support_ready.float()


# [hand_setting, parked] Historical valley-gated contact reward.
def valley_gated_contact_group_strength(
    env: ManagerBasedEnv,
    target_groups: tuple[tuple[str, ...], ...],
    anchor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    valley_point_offset_o: tuple[float, float, float],
    stick_half_length: float = 0.09,
    force_scale: float = 0.10,
    reduction: str = "mean",
    contact_threshold: float = 0.02,
    valley_point_error_limit: float = 0.01,
    valley_axis_error_limit: float = 0.2617993878,
) -> torch.Tensor:
    """Expose remaining contact shaping after force-validated valley entry."""
    target_strength = contact_group_strength(
        env,
        target_groups,
        force_scale,
        reduction,
    )
    in_valley = stick2_in_valley_gate(
        env,
        anchor_groups,
        palm_cfg,
        stick2_cfg,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        valley_point_offset_o,
        stick_half_length,
        contact_threshold,
        valley_point_error_limit,
        valley_axis_error_limit,
    )
    return target_strength * in_valley


# [hand_setting, parked] Historical valley-gated shaft proximity reward.
def valley_gated_body_box_shaft_region_proximity(
    env: ManagerBasedEnv,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    anchor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick2_reference_position_p: tuple[float, float, float],
    stick2_reference_quaternion_p: tuple[float, float, float, float],
    valley_point_offset_o: tuple[float, float, float],
    stick_half_length: float = 0.09,
    long_axis: int = 1,
    axial_half_length: float = 0.075,
    sigma: float = 0.02,
    contact_threshold: float = 0.02,
    valley_point_error_limit: float = 0.01,
    valley_axis_error_limit: float = 0.2617993878,
) -> torch.Tensor:
    """Enable fingertip approach after force-validated valley entry."""
    proximity = body_box_shaft_region_proximity(
        env,
        body_cfg,
        object_cfg,
        object_half_extent,
        long_axis,
        axial_half_length,
        sigma,
    )
    in_valley = stick2_in_valley_gate(
        env,
        anchor_groups,
        palm_cfg,
        stick2_cfg,
        stick2_reference_position_p,
        stick2_reference_quaternion_p,
        valley_point_offset_o,
        stick_half_length,
        contact_threshold,
        valley_point_error_limit,
        valley_axis_error_limit,
    )
    return proximity * in_valley


# [hand_setting] Active shaft-region check for metrics and success termination.
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


# [hand_setting] Active final-pose and shaft-region validator.
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


# [hand_setting, parked] Prepared pose/region/contact completion reward.
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


# [hand_setting, parked] Prepared quiet completion reward.
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


# [hand_grasp] Active palm-relative stick angular-speed penalty.
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


# [hand_grasp] Six-contact quietness base used by active mode stability.
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
    group_reduction: str = "max",
) -> torch.Tensor:
    """Reward quiet two-stick motion only while every contact group is engaged."""
    group_forces = _group_forces(env, sensor_groups, group_reduction)
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


# [hand_grasp, parked] Older anchor-gated mode-stability implementation.
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


# [hand_grasp] Active six-contact OPEN/CLOSE stability reward.
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
    lateral_sigma: float | None = 0.005,
    axial_sigma: float = 0.005,
    force_scale: float = 0.10,
    contact_threshold: float = 0.02,
    linear_speed_scale: float = 0.10,
    angular_speed_scale: float = 2.0,
    clamp_gap: bool = True,
    group_reduction: str = "max",
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
        group_reduction,
    )
    command = env.command_manager.get_command(command_name)
    command_valid = torch.clamp(command.sum(dim=-1), min=0.0, max=1.0)
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
        clamp_gap=clamp_gap,
    )
    axial_error = torch.abs(
        axial_offset - float(reference_axial_offset_stick2)
    )
    # lateral_sigma=None 이면 lateral 을 이 지수에서 뺀다 (2026-08-07).
    # 셋이 한 지수를 공유하면 서로의 gradient 를 곱으로 깎는다 -> hand_move 는
    # tip_lateral_alignment 독립 항으로 옮겼다.  mode_tip_gap_tracking 과 동일.
    exponent = (
        torch.abs(tip_gap - target_gap) / gap_sigma + axial_error / axial_sigma
    )
    if lateral_sigma is not None:
        exponent = exponent + lateral_error / lateral_sigma
    return command_valid * base_stability * torch.exp(-exponent)


# [hand_setting] Active strict success termination and stable-step metric.
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


# [hand_grasp] Active per-mode success pulse and unreachable metric termination.
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
        orientation_error_mode: str = "quaternion",
        # 보상과 같은 gap 정의를 써야 한다.  clamp 가 켜져 있으면 옆으로 지나가는
        # (교차) 상태가 gap=0 으로 보고돼, 보상은 깎으면서 성공은 인정하는 모순이
        # 생긴다.  기본 True 로 hand_grasp/hand_setting 은 그대로.
        clamp_gap: bool = True,
        group_reduction: str = "max",
    ) -> torch.Tensor:
        del (
            stick1_reference_position_p,
            stick1_reference_quaternion_p,
            stick2_reference_position_p,
            stick2_reference_quaternion_p,
            reference_separation_direction_stick2,
        )
        contacts_valid = torch.all(
            _group_forces(env, sensor_groups, group_reduction) >= contact_threshold,
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
            orientation_error_mode,
        )
        command = env.command_manager.get_command(command_name)
        command_valid = command.sum(dim=-1) > 0.0
        mode = torch.where(
            command_valid,
            torch.argmax(command, dim=-1),
            torch.full_like(self._last_mode, -1),
        )
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
            clamp_gap=clamp_gap,
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
            command_valid
            & contacts_valid
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


# [hand_move] Post-acquisition functional-contact loss termination.
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
        group_reduction: str = "max",
    ) -> torch.Tensor:
        group_forces = _group_forces(env, sensor_groups, group_reduction)
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


# [hand_grasp] Expose the OPEN/CLOSE command in policy observation.
def open_close_mode(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Return the OPEN/CLOSE one-hot command for policy observation."""
    return env.command_manager.get_command(command_name)


# [shared: hand_grasp + hand_setting] Fingertip-position observation.
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


# [shared: hand_grasp + hand_setting] Stick-pose observation.
def object_pose_in_palm(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return an object's position and quaternion expressed in the palm frame."""
    pos_b, quat_b = _object_pose_in_palm(env, palm_cfg, object_cfg)
    return torch.cat((pos_b, quat_b), dim=-1)


# [shared: hand_grasp + hand_setting] Palm-relative stick-velocity observation.
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
