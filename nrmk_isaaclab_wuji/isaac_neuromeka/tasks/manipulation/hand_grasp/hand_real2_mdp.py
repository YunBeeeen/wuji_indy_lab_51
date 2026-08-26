"""Task-local command and disturbance settings for :mod:`hand_real2_env_cfg`.

The implementations remain shared with ``hand_move_mdp``.  Only the mutable
curriculum knobs are owned here so experiments on hand_real2 cannot silently
change hand_real/hand_move, and vice versa.
"""

from __future__ import annotations

import math

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from . import mdp as hand_grasp_mdp
from .hand_move_mdp import HandMoveScheduleCfg


# First-stage default: no root rotation.  Keep the complete 15 s command
# schedule here; acquisition-only training truncates the environment at 5 s
# with ``env.episode_length_s=5.0`` and therefore sees neutral [0, 0] only.
HAND_REAL2_SCHEDULE = HandMoveScheduleCfg(
    range_x=(0.0, 0.0),
    range_y=(0.0, 0.0),
    range_z=(0.0, 0.0),
    initial_hold_time_s=2.0,
    rotation_interpolation_time_s=2.0,
    rotation_settling_time_s=1.0,
    open_close_start_time_s=5.0,
    open_close_segment_time_s=2.0,
    num_open_close_segments=5,
    episode_length_s=15.0,
    use_slerp=True,
    use_smoothstep=True,
    first_mode_open_probability=0.5,
)
HAND_REAL2_SCHEDULE.validate()


# Disturbance curriculum knobs.  These intentionally mirror the conservative
# first hand_real stage but no longer alias hand_move_env_cfg module constants.
HAND_REAL2_DISTURBANCE_TIME_RANGE_S = (2.5, 4.0)
HAND_REAL2_DISTURBANCE_DURATION_S = 0.10
HAND_REAL2_DISTURBANCE_FORCE_RANGE_N = (0.01, 0.03)
# 2026-08-24: isolate the enlarged reset-pose distribution first.  The weak
# distal-tip disturbance remains parked for the following curriculum stage.
HAND_REAL2_DISTURBANCE_PROBABILITY = 0.0
# The Cuboid center is the rigid-body/CoM origin and local +Y is the 180 mm
# shaft axis, so +90 mm is the distal geometric tip.  StickDisturbance converts
# this point force to the equivalent CoM force plus r x F torque.
HAND_REAL2_DISTURBANCE_APPLICATION_OFFSET_O = (0.0, 0.09, 0.0)


# Reset-pose noise curriculum.  Translation is sampled independently for each
# stick along palm-frame xyz.  Rotation is an independent object-local xyz
# Euler perturbation composed onto the recorded pose_005 quaternion.
# 2026-08-24 run 18-25-19 condition: +/-5 mm, +/-5 deg at probability 1.0.
# Intermediate full-population stage that was used before the 1 cm/10 deg stage:
# HAND_REAL2_STICK_POSITION_NOISE_M = ((-0.0025, 0.0025),) * 3
# HAND_REAL2_STICK_ORIENTATION_NOISE_RAD = (
#     (-math.radians(3.0), math.radians(3.0)),
# ) * 3
# Attempted next stage (2026-08-25); preserve it for a later retry:
# HAND_REAL2_STICK_POSITION_NOISE_M = ((-0.01, 0.01),) * 3
# HAND_REAL2_STICK_ORIENTATION_NOISE_RAD = (
#     (-math.radians(10.0), math.radians(10.0)),
# ) * 3
# Active recovery stage: restore the 2026-08-24_18-25-19 distribution.
HAND_REAL2_STICK_POSITION_NOISE_M = ((-0.005, 0.005),) * 3
HAND_REAL2_STICK_ORIENTATION_NOISE_RAD = (
    (-math.radians(5.0), math.radians(5.0)),
) * 3
HAND_REAL2_STICK_RESET_NOISE_PROBABILITY = 1.0


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
    """Apply the normal pregrasp reset, then independently perturb both sticks."""
    hand_grasp_mdp.reset_to_functional_pregrasp(
        env,
        env_ids,
        hand_cfg,
        palm_cfg,
        stick1_cfg,
        stick2_cfg,
        joint_positions,
        stick1_position_p,
        stick1_quaternion_p,
        stick2_position_p,
        stick2_quaternion_p,
    )

    count = env_ids.numel()
    if count == 0 or probability <= 0.0:
        return
    if len(position_noise_m) != 3 or len(orientation_noise_rad) != 3:
        raise ValueError("Stick reset position/orientation noise must each have xyz ranges.")

    robot: Articulation = env.scene[hand_cfg.name]
    sticks: tuple[RigidObject, RigidObject] = (
        env.scene[stick1_cfg.name],
        env.scene[stick2_cfg.name],
    )
    palm_id = palm_cfg.body_ids[0]
    palm_pos_w = robot.data.body_pos_w[env_ids, palm_id]
    palm_quat_w = robot.data.body_quat_w[env_ids, palm_id]
    dtype = palm_pos_w.dtype
    device = env.device

    enabled = (
        torch.rand((count, 1), device=device) < min(float(probability), 1.0)
    ).to(dtype=dtype)
    pos_low = torch.tensor([r[0] for r in position_noise_m], device=device, dtype=dtype)
    pos_high = torch.tensor([r[1] for r in position_noise_m], device=device, dtype=dtype)
    rot_low = torch.tensor([r[0] for r in orientation_noise_rad], device=device, dtype=dtype)
    rot_high = torch.tensor([r[1] for r in orientation_noise_rad], device=device, dtype=dtype)
    nominal_positions = (stick1_position_p, stick2_position_p)
    nominal_quaternions = (stick1_quaternion_p, stick2_quaternion_p)
    zero_velocity = torch.zeros((count, 6), device=device, dtype=dtype)

    for stick, nominal_position, nominal_quaternion in zip(
        sticks, nominal_positions, nominal_quaternions, strict=True
    ):
        position_noise = (pos_low + torch.rand((count, 3), device=device) * (pos_high - pos_low)) * enabled
        euler_noise = (rot_low + torch.rand((count, 3), device=device) * (rot_high - rot_low)) * enabled
        delta_quat = quat_from_euler_xyz(
            euler_noise[:, 0], euler_noise[:, 1], euler_noise[:, 2]
        )
        position_p = torch.as_tensor(nominal_position, device=device, dtype=dtype).expand(count, -1)
        quaternion_p = torch.as_tensor(nominal_quaternion, device=device, dtype=dtype).expand(count, -1)
        noisy_position_p = position_p + position_noise
        noisy_quaternion_p = quat_mul(quaternion_p, delta_quat)
        pose_w = torch.cat(
            (
                palm_pos_w + quat_apply(palm_quat_w, noisy_position_p),
                quat_mul(palm_quat_w, noisy_quaternion_p),
            ),
            dim=-1,
        )
        stick.write_root_pose_to_sim(pose_w, env_ids=env_ids)
        stick.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
