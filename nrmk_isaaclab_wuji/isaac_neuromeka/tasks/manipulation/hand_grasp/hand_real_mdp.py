"""Deployment-compatible observation helpers for :mod:`hand_real`.

The real controller can retain the last command locally and can estimate a
stick pose from vision, but it cannot read simulator joint/object velocities.
This module contains the deployable observation adapters.  Physics and actions
remain inherited from ``hand_move``; the active observation retains a
symmetry-canonical palm-frame quaternion while ``hand_real`` still selects the
directed-axis orientation mode for its reward and success validator so one
particular shaft-roll angle is not demanded by the objective.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, quat_unique

from . import mdp as hand_grasp_mdp


_SQRT_HALF = 0.7071067811865476
# The 7 x 180 x 7 mm proxy is invariant to quarter turns about local +y.
# These symmetries preserve +y, so the distal tip and tail are never swapped.
_SQUARE_STICK_Y_SYMMETRIES = (
    (1.0, 0.0, 0.0, 0.0),
    (_SQRT_HALF, 0.0, _SQRT_HALF, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (_SQRT_HALF, 0.0, -_SQRT_HALF, 0.0),
)
_symmetry_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def tip_press_force(
    env: ManagerBasedRLEnv,
    command_name: str,
    tip_contact_sensor_name: str,
    sensor_groups: tuple[tuple[str, ...], ...],
    palm_cfg: SceneEntityCfg,
    stick1_cfg: SceneEntityCfg,
    stick2_cfg: SceneEntityCfg,
    stick1_tip_offset_o: tuple[float, float, float],
    stick2_tip_offset_o: tuple[float, float, float],
    stick_thickness: float,
    close_target_gap: float,
    reference_separation_direction_stick2: tuple[float, float, float],
    reference_axial_offset_stick2: float,
    force_saturation: float = 0.30,
    functional_force_scale: float | tuple[float, ...] = 0.10,
    gap_sigma: float = 0.003,
    lateral_sigma: float = 0.005,
    axial_sigma: float = 0.005,
    group_reduction: str = "max",
) -> torch.Tensor:
    """Reward a loaded CLOSE tip press while retaining the grasp.

    Stick1 is one rigid cuboid, so its Stick2-filtered sensor reports every
    stick--stick contact rather than a spatially cropped final few millimetres.
    Multiplying the load by the active distal gap/lateral/axial geometry rejects
    shaft-contact farming.  The weakest functional hand-contact score is a
    second gate, preventing the policy from trading away its established grasp
    merely to squeeze the two sticks together.

    Force credit saturates at ``force_saturation``; there is no incentive to
    press harder than the requested load and overheat the physical hand.
    """
    for name, value in (
        ("force_saturation", force_saturation),
        ("gap_sigma", gap_sigma),
        ("lateral_sigma", lateral_sigma),
        ("axial_sigma", axial_sigma),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}.")

    # NEUTRAL [0, 0] keeps the inherited pose/contact objective unchanged.
    # Tip preload is a CLOSE-only instruction and must never fight OPEN.
    close_gate = env.command_manager.get_command(command_name)[:, 1]
    force_score = torch.clamp(
        hand_grasp_mdp._sensor_force(env, tip_contact_sensor_name)
        / force_saturation,
        min=0.0,
        max=1.0,
    )
    functional_forces = hand_grasp_mdp._group_forces(
        env, sensor_groups, group_reduction
    )
    functional_scales = torch.as_tensor(
        hand_grasp_mdp._normalized_group_force_scales(
            functional_force_scale, len(sensor_groups)
        ),
        device=functional_forces.device,
        dtype=functional_forces.dtype,
    ).unsqueeze(0)
    functional_gate = torch.min(
        torch.clamp(
            functional_forces / functional_scales,
            min=0.0,
            max=1.0,
        ),
        dim=-1,
    ).values

    stick1_position, stick1_quaternion = hand_grasp_mdp._object_pose_in_palm(
        env, palm_cfg, stick1_cfg
    )
    stick2_position, stick2_quaternion = hand_grasp_mdp._object_pose_in_palm(
        env, palm_cfg, stick2_cfg
    )
    gap, lateral_error, axial_offset = (
        hand_grasp_mdp._tip_geometry_from_palm_poses(
            stick1_position,
            stick1_quaternion,
            stick2_position,
            stick2_quaternion,
            stick1_tip_offset_o,
            stick2_tip_offset_o,
            stick_thickness,
            reference_separation_direction_stick2,
            clamp_gap=False,
        )
    )
    axial_error = torch.abs(
        axial_offset - float(reference_axial_offset_stick2)
    )
    geometry_gate = torch.exp(
        -torch.abs(gap - close_target_gap) / gap_sigma
        - lateral_error / lateral_sigma
        - axial_error / axial_sigma
    )
    return close_gate * functional_gate * geometry_gate * force_score


def reset_to_noisy_functional_pregrasp(
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
    position_noise_m: tuple[tuple[float, float], ...],
    orientation_noise_rad: tuple[tuple[float, float], ...],
    probability: float,
) -> None:
    """Apply hand_real's normal pregrasp reset, then perturb both sticks."""
    hand_grasp_mdp.reset_to_functional_pregrasp(
        env, env_ids, hand_cfg, palm_cfg, stick1_cfg, stick2_cfg,
        joint_positions, stick1_position_p, stick1_quaternion_p,
        stick2_position_p, stick2_quaternion_p,
    )
    count = env_ids.numel()
    if count == 0 or probability <= 0.0:
        return
    if len(position_noise_m) != 3 or len(orientation_noise_rad) != 3:
        raise ValueError("Stick reset position/orientation noise must each have xyz ranges.")

    robot: Articulation = env.scene[hand_cfg.name]
    sticks: tuple[RigidObject, RigidObject] = (
        env.scene[stick1_cfg.name], env.scene[stick2_cfg.name]
    )
    palm_id = palm_cfg.body_ids[0]
    palm_pos_w = robot.data.body_pos_w[env_ids, palm_id]
    palm_quat_w = robot.data.body_quat_w[env_ids, palm_id]
    dtype, device = palm_pos_w.dtype, env.device
    enabled = (torch.rand((count, 1), device=device) < min(float(probability), 1.0)).to(dtype)
    pos_low = torch.tensor([r[0] for r in position_noise_m], device=device, dtype=dtype)
    pos_high = torch.tensor([r[1] for r in position_noise_m], device=device, dtype=dtype)
    rot_low = torch.tensor([r[0] for r in orientation_noise_rad], device=device, dtype=dtype)
    rot_high = torch.tensor([r[1] for r in orientation_noise_rad], device=device, dtype=dtype)
    zero_velocity = torch.zeros((count, 6), device=device, dtype=dtype)

    for stick, nominal_position, nominal_quaternion in zip(
        sticks,
        (stick1_position_p, stick2_position_p),
        (stick1_quaternion_p, stick2_quaternion_p),
        strict=True,
    ):
        position_noise = (
            pos_low + torch.rand((count, 3), device=device) * (pos_high - pos_low)
        ) * enabled
        euler_noise = (
            rot_low + torch.rand((count, 3), device=device) * (rot_high - rot_low)
        ) * enabled
        delta_quat = quat_from_euler_xyz(
            euler_noise[:, 0], euler_noise[:, 1], euler_noise[:, 2]
        )
        position_p = torch.as_tensor(nominal_position, device=device, dtype=dtype).expand(count, -1)
        quaternion_p = torch.as_tensor(nominal_quaternion, device=device, dtype=dtype).expand(count, -1)
        pose_w = torch.cat(
            (
                palm_pos_w + quat_apply(palm_quat_w, position_p + position_noise),
                quat_mul(palm_quat_w, quat_mul(quaternion_p, delta_quat)),
            ),
            dim=-1,
        )
        stick.write_root_pose_to_sim(pose_w, env_ids=env_ids)
        stick.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)


def _square_stick_quaternion_nearest_reference(
    quaternion_p: torch.Tensor,
    reference_quaternion_p: tuple[float, float, float, float],
) -> torch.Tensor:
    """Fold four local-y roll symmetries onto the pose_005 branch.

    Multiplying the measured object quaternion by each local symmetry generates
    four physically equivalent square-stick poses.  Selecting the candidate
    nearest the fixed palm-frame reference gives simulation and vision the same
    deterministic representative without changing the observation dimension.
    """
    key = (quaternion_p.device, quaternion_p.dtype)
    symmetries = _symmetry_cache.get(key)
    if symmetries is None:
        symmetries = torch.as_tensor(
            _SQUARE_STICK_Y_SYMMETRIES,
            dtype=quaternion_p.dtype,
            device=quaternion_p.device,
        )
        _symmetry_cache[key] = symmetries

    quaternion_p = quaternion_p / torch.clamp(
        torch.linalg.vector_norm(quaternion_p, dim=-1, keepdim=True),
        min=1.0e-8,
    )
    reference = torch.as_tensor(
        reference_quaternion_p,
        dtype=quaternion_p.dtype,
        device=quaternion_p.device,
    )
    reference = reference / torch.clamp(
        torch.linalg.vector_norm(reference),
        min=1.0e-8,
    )
    candidates = quat_mul(
        quaternion_p.unsqueeze(1).expand(-1, len(symmetries), -1),
        symmetries.unsqueeze(0).expand(quaternion_p.shape[0], -1, -1),
    )
    scores = torch.abs(
        torch.sum(candidates * reference.view(1, 1, 4), dim=-1)
    )
    nearest = torch.argmax(scores, dim=-1)
    selected = candidates[
        torch.arange(quaternion_p.shape[0], device=quaternion_p.device),
        nearest,
    ]
    # Preserve the existing w >= 0 quaternion convention after symmetry fold.
    return quat_unique(selected)


def canonical_object_pose_in_palm(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    reference_quaternion_p: tuple[float, float, float, float],
) -> torch.Tensor:
    """Return symmetry-folded palm-frame ``xyz + wxyz`` for the 105D policy.

    The four square-section roll symmetries are mapped to the pose_005 branch,
    then the quaternion double-cover is resolved with ``w >= 0``.  Tip/tail is
    deliberately preserved and Stick1/Stick2 identity remains the tracker's
    responsibility.  The same reference branch and quaternion convention must
    be reproduced by the real vision bridge.
    """

    pose = hand_grasp_mdp.object_pose_in_palm(env, palm_cfg, object_cfg)
    quaternion = _square_stick_quaternion_nearest_reference(
        pose[:, 3:7],
        reference_quaternion_p,
    )
    return torch.cat((pose[:, :3], quaternion), dim=-1)


def object_position_and_directed_axis_in_palm(
    env: ManagerBasedEnv,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the parked 101D-era ``xyz + directed local-y axis`` state.

    Local ``+y`` points from tail to distal tip for both stick proxies.  Keeping
    its sign therefore preserves tip/tail identity while making every rotation
    about the shaft axis observationally equivalent.  A three-component unit
    vector is used instead of yaw/pitch so the real vision contract has no
    Euler singularity or quaternion sign/quarter-turn branch to reproduce.
    """

    pose = hand_grasp_mdp.object_pose_in_palm(env, palm_cfg, object_cfg)
    local_y = torch.zeros_like(pose[:, :3])
    local_y[:, 1] = 1.0
    directed_axis_p = quat_apply(pose[:, 3:7], local_y)
    return torch.cat((pose[:, :3], directed_axis_p), dim=-1)


def last_applied_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the 20D policy action just executed for the returned state.

    ``ActionManager.process_action(a_t)`` stores ``a_t`` in ``action`` and the
    older ``a_(t-1)`` in ``prev_action`` before physics stepping.  Observations
    are computed after that physics step, so ``action`` is the command that
    caused the current state; using ``prev_action`` here would add one extra
    policy-step delay.
    """

    return env.action_manager.action


def neutral_open_close_mode(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the explicit 2D neutral/setting mode ``[0, 0]``.

    This keeps a setting-only task shape-compatible with the 105D hand_real
    actor without creating an OPEN/CLOSE command term.  The zero pair is a
    deliberate third semantic state learned from setting/reference rewards.
    """

    return torch.zeros((env.num_envs, 2), device=env.device)
