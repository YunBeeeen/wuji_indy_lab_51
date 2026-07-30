"""Object-relative targets for the one-stick functional-grasp task.

The A1 experiment uses constraint-based fingertip regions and a palm orientation
relative to the stick. No full hand joint pose is prescribed.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def _half_extents(
    env,
    fallback: tuple[float, float, float],
    dtype: torch.dtype,
) -> torch.Tensor:
    half = getattr(env, "box_half_extents", None)
    if half is None:
        half = torch.tensor(fallback, dtype=dtype, device=env.device).unsqueeze(0)
        half = half.expand(env.num_envs, -1)
    elif half.dim() == 1:
        half = half.unsqueeze(0).expand(env.num_envs, -1)
    return half


def grip_target_position_o(
    env,
    object_half_extent: tuple[float, float, float],
    long_axis: int = 1,
    grip_fraction: float = -0.45,
    surface_axis: int = 2,
    surface_sign: float = 1.0,
    surface_offset: float = 0.0,
) -> torch.Tensor:
    """Return the semantic index target in the object frame.

    ``grip_fraction`` is normalized by the half length: -1 is the tail, +1 is
    the tip.  The default -0.45 puts the index in the rear grip region while
    preserving the tip-tail direction for later chopstick use.
    """
    if long_axis == surface_axis:
        raise ValueError("long_axis and surface_axis must differ")
    if not -1.0 <= grip_fraction <= 1.0:
        raise ValueError(f"grip_fraction must be in [-1, 1], got {grip_fraction}")
    if surface_sign not in (-1.0, 1.0):
        raise ValueError(f"surface_sign must be -1 or 1, got {surface_sign}")

    half = _half_extents(env, object_half_extent, torch.float)
    target_o = torch.zeros(env.num_envs, 3, dtype=half.dtype, device=env.device)
    target_o[:, long_axis] = grip_fraction * half[:, long_axis]
    target_o[:, surface_axis] = surface_sign * half[:, surface_axis] + surface_offset
    return target_o


def grip_target_position_w(
    env,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    long_axis: int = 1,
    grip_fraction: float = -0.45,
    surface_axis: int = 2,
    surface_sign: float = 1.0,
    surface_offset: float = 0.0,
) -> torch.Tensor:
    """Transform the semantic index target from object to world coordinates."""
    obj = env.scene[object_cfg.name]
    target_o = grip_target_position_o(
        env,
        object_half_extent,
        long_axis,
        grip_fraction,
        surface_axis,
        surface_sign,
        surface_offset,
    )
    return obj.data.root_pos_w + math_utils.quat_apply(obj.data.root_quat_w, target_o)


def grip_region_bounds_o(
    env,
    object_half_extent: tuple[float, float, float],
    long_axis: int = 1,
    axial_region: tuple[float, float] = (-0.60, -0.30),
    surface_axis: int = 2,
    surface_sign: float = 1.0,
    surface_offset: float = 0.0,
    surface_tolerance: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lower/upper bounds of a rectangular grip region in object frame.

    ``axial_region`` is normalized by the object's half length. The remaining
    tangent axis spans the object face, while ``surface_tolerance`` gives the
    fingertip a narrow band around the selected surface.
    """
    if long_axis == surface_axis:
        raise ValueError("long_axis and surface_axis must differ")
    if not 0 <= long_axis < 3 or not 0 <= surface_axis < 3:
        raise ValueError("long_axis and surface_axis must be in [0, 2]")
    if len(axial_region) != 2 or axial_region[0] > axial_region[1]:
        raise ValueError(f"axial_region must be an ordered pair, got {axial_region}")
    if axial_region[0] < -1.0 or axial_region[1] > 1.0:
        raise ValueError(f"axial_region must lie in [-1, 1], got {axial_region}")
    if surface_sign not in (-1.0, 1.0):
        raise ValueError(f"surface_sign must be -1 or 1, got {surface_sign}")
    if surface_tolerance < 0.0:
        raise ValueError(f"surface_tolerance must be non-negative, got {surface_tolerance}")

    half = _half_extents(env, object_half_extent, torch.float)
    lower = -half.clone()
    upper = half.clone()
    lower[:, long_axis] = axial_region[0] * half[:, long_axis]
    upper[:, long_axis] = axial_region[1] * half[:, long_axis]
    surface = surface_sign * half[:, surface_axis] + surface_offset
    lower[:, surface_axis] = surface - surface_tolerance
    upper[:, surface_axis] = surface + surface_tolerance
    return lower, upper


def fingertip_grip_region_error_b(
    env,
    palm_cfg: SceneEntityCfg,
    fingertip_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    long_axis: int = 1,
    axial_region: tuple[float, float] = (-0.60, -0.30),
    surface_axis: int = 2,
    surface_sign: float = 1.0,
    surface_offset: float = 0.0,
    surface_tolerance: float = 0.005,
) -> torch.Tensor:
    """Vector from a fingertip to the nearest point in its grip region, in palm frame."""
    robot = env.scene[palm_cfg.name]
    obj = env.scene[object_cfg.name]
    palm_id = palm_cfg.body_ids[0]
    fingertip_id = fingertip_cfg.body_ids[0]

    fingertip_w = robot.data.body_state_w[:, fingertip_id, :3]
    fingertip_o = math_utils.quat_apply_inverse(
        obj.data.root_quat_w,
        fingertip_w - obj.data.root_pos_w,
    )
    lower_o, upper_o = grip_region_bounds_o(
        env,
        object_half_extent,
        long_axis,
        axial_region,
        surface_axis,
        surface_sign,
        surface_offset,
        surface_tolerance,
    )
    nearest_o = torch.minimum(torch.maximum(fingertip_o, lower_o), upper_o)
    error_o = nearest_o - fingertip_o
    error_w = math_utils.quat_apply(obj.data.root_quat_w, error_o)
    palm_quat_w = robot.data.body_state_w[:, palm_id, 3:7]
    return math_utils.quat_apply_inverse(palm_quat_w, error_w)


def fingertip_grip_region_error(
    env,
    palm_cfg: SceneEntityCfg,
    fingertip_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    **kwargs,
) -> torch.Tensor:
    """Scalar distance from a fingertip to its object-relative grip region."""
    del palm_cfg  # A distance is rotation invariant; avoid two unnecessary frame transforms.
    object_half_extent = kwargs.pop("object_half_extent", (0.01, 0.09, 0.01))
    robot = env.scene[fingertip_cfg.name]
    obj = env.scene[object_cfg.name]
    fingertip_w = robot.data.body_state_w[:, fingertip_cfg.body_ids[0], :3]
    fingertip_o = math_utils.quat_apply_inverse(
        obj.data.root_quat_w,
        fingertip_w - obj.data.root_pos_w,
    )
    lower_o, upper_o = grip_region_bounds_o(env, object_half_extent, **kwargs)
    nearest_o = torch.minimum(torch.maximum(fingertip_o, lower_o), upper_o)
    return torch.norm(nearest_o - fingertip_o, dim=-1)


def index_grip_error_b(
    env,
    palm_cfg: SceneEntityCfg,
    index_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    long_axis: int = 1,
    axial_region: tuple[float, float] = (-0.60, -0.30),
    surface_axis: int = 2,
    surface_sign: float = 1.0,
    surface_offset: float = 0.0,
    surface_tolerance: float = 0.005,
) -> torch.Tensor:
    """Vector from the index tip to the nearest point in its grip region."""
    return fingertip_grip_region_error_b(
        env,
        palm_cfg,
        index_cfg,
        object_cfg,
        object_half_extent,
        long_axis,
        axial_region,
        surface_axis,
        surface_sign,
        surface_offset,
        surface_tolerance,
    )


def index_grip_error(
    env,
    palm_cfg: SceneEntityCfg,
    index_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    **kwargs,
) -> torch.Tensor:
    """Scalar index target error in metres."""
    return torch.norm(index_grip_error_b(env, palm_cfg, index_cfg, object_cfg, **kwargs), dim=-1)


def thumb_grip_error_b(
    env,
    palm_cfg: SceneEntityCfg,
    thumb_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    long_axis: int = 1,
    axial_region: tuple[float, float] = (-0.55, -0.25),
    surface_axis: int = 0,
    surface_sign: float = 1.0,
    surface_offset: float = 0.0,
    surface_tolerance: float = 0.005,
) -> torch.Tensor:
    """Vector from the thumb tip to the nearest point in its grip region."""
    return fingertip_grip_region_error_b(
        env,
        palm_cfg,
        thumb_cfg,
        object_cfg,
        object_half_extent,
        long_axis,
        axial_region,
        surface_axis,
        surface_sign,
        surface_offset,
        surface_tolerance,
    )


def thumb_grip_error(
    env,
    palm_cfg: SceneEntityCfg,
    thumb_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    **kwargs,
) -> torch.Tensor:
    """Scalar thumb grip-region error in metres."""
    return torch.norm(thumb_grip_error_b(env, palm_cfg, thumb_cfg, object_cfg, **kwargs), dim=-1)


def middle_grip_error_b(
    env,
    palm_cfg: SceneEntityCfg,
    middle_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    long_axis: int = 1,
    axial_region: tuple[float, float] = (-0.55, -0.25),
    surface_axis: int = 0,
    surface_sign: float = -1.0,
    surface_offset: float = 0.0,
    surface_tolerance: float = 0.005,
) -> torch.Tensor:
    """Vector from the middle tip to the nearest point in its grip region."""
    return fingertip_grip_region_error_b(
        env,
        palm_cfg,
        middle_cfg,
        object_cfg,
        object_half_extent,
        long_axis,
        axial_region,
        surface_axis,
        surface_sign,
        surface_offset,
        surface_tolerance,
    )


def middle_grip_error(
    env,
    palm_cfg: SceneEntityCfg,
    middle_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    **kwargs,
) -> torch.Tensor:
    """Scalar middle grip-region error in metres."""
    return torch.norm(middle_grip_error_b(env, palm_cfg, middle_cfg, object_cfg, **kwargs), dim=-1)


def hand_orientation_in_object(
    env,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Palm quaternion expressed in the object frame, ``q_O_H``."""
    robot = env.scene[palm_cfg.name]
    obj = env.scene[object_cfg.name]
    palm_id = palm_cfg.body_ids[0]
    _, quat_o_h = math_utils.subtract_frame_transforms(
        obj.data.root_pos_w,
        obj.data.root_quat_w,
        robot.data.body_state_w[:, palm_id, :3],
        robot.data.body_state_w[:, palm_id, 3:7],
    )
    return quat_o_h


def capture_hand_tool_target(
    env,
    env_ids: torch.Tensor | None,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    target_buffer_name: str = "chopstick_target_palm_quat_o",
    target_quat_o: tuple[float, float, float, float] | None = None,
) -> None:
    """Capture or assign the desired object-relative palm orientation after reset.

    The baseline captures the configured pre-grasp orientation instead of
    embedding an unexplained quaternion.  A measured target can later be passed
    with ``target_quat_o`` without changing observation or action dimensions.
    """
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is None:
        env_ids = all_env_ids
    elif isinstance(env_ids, slice):
        env_ids = all_env_ids[env_ids]
    if target_quat_o is None:
        target = hand_orientation_in_object(env, palm_cfg, object_cfg)[env_ids].clone()
    else:
        target = torch.tensor(target_quat_o, dtype=torch.float, device=env.device)
        target = target / torch.clamp(torch.norm(target), min=1.0e-6)
        target = target.unsqueeze(0).expand(len(env_ids), -1).clone()

    buffer = getattr(env, target_buffer_name, None)
    if buffer is None:
        buffer = torch.zeros(env.num_envs, 4, dtype=target.dtype, device=env.device)
        buffer[:, 0] = 1.0
        setattr(env, target_buffer_name, buffer)
    buffer[env_ids] = target


def target_hand_orientation_in_object(
    env,
    target_buffer_name: str = "chopstick_target_palm_quat_o",
    fallback_quat_o: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Read the captured target, with an explicit fallback for static analysis/tests."""
    target = getattr(env, target_buffer_name, None)
    if target is not None:
        return target
    target = torch.tensor(fallback_quat_o, dtype=torch.float, device=env.device)
    target = target / torch.clamp(torch.norm(target), min=1.0e-6)
    return target.unsqueeze(0).expand(env.num_envs, -1)


def hand_tool_orientation_error_axis_angle(
    env,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    target_buffer_name: str = "chopstick_target_palm_quat_o",
    fallback_quat_o: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Signed axis-angle error from target to current ``q_O_H``."""
    current = hand_orientation_in_object(env, palm_cfg, object_cfg)
    target = target_hand_orientation_in_object(env, target_buffer_name, fallback_quat_o)
    error_quat = math_utils.quat_mul(current, math_utils.quat_inv(target))
    return math_utils.axis_angle_from_quat(error_quat)


def hand_tool_orientation_error(
    env,
    palm_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    target_buffer_name: str = "chopstick_target_palm_quat_o",
    fallback_quat_o: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Geodesic hand-stick relative orientation error in radians."""
    current = hand_orientation_in_object(env, palm_cfg, object_cfg)
    target = target_hand_orientation_in_object(env, target_buffer_name, fallback_quat_o)
    return math_utils.quat_error_magnitude(current, target)
