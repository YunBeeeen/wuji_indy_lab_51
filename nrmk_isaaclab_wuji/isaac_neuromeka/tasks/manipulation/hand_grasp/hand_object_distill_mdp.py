"""Frozen observation/action adapters for 103D-to-105D hand distillation.

The teacher checkpoint belongs to
``hand_object/2026-08-08_20-39-52(성공)/model_300.pt``.  Its joint-position
inputs were normalized with the old local-URDF placeholder limits and its
action-history input was expressed in uniform-0.1-rad action units.  The
current simulator uses this physical hand's factory limits and the deployable
student uses per-joint residual scales, so neither quantity can be read through
the current generic observation term without changing the teacher contract.

These constants are task-local on purpose.  The active ``hand_object`` and
``hand_real`` observations remain untouched.
"""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


TEACHER_POLICY_OBS_DIM = 103
STUDENT_POLICY_OBS_DIM = 105
TEACHER_ACTION_SCALE_RAD = 0.1
STUDENT_ACTION_SCALE_RAD = (0.1, 0.1, 0.2, 0.15) * 5

# Exact pose_005 reset from the saved successful run's params/env.yaml.  The
# two values slightly above 1.6272 were legal under the imported USD at the
# time; the current physical-hand limits also admit them.
TEACHER_PREGRASP_JOINT_POSITIONS = (
    0.5377866626,
    0.8436813951,
    0.0377136655,
    -0.0000001810,
    0.7017297745,
    0.0553143807,
    1.1822255850,
    1.4215219021,
    0.4649881423,
    -0.0292181600,
    1.6298730373,
    1.1032750607,
    0.9151425958,
    -0.0129909236,
    1.3248542547,
    0.3182539344,
    0.7154092789,
    0.0788998753,
    1.6281884909,
    0.2546040118,
)

# Exact soft-limit table used by joint_pos_limit_normalized() when the teacher
# was trained.  This is the historical local-URDF table, not today's connected
# hand limits.  The function deliberately does not clip normalized values,
# matching Isaac Lab's original affine observation.
TEACHER_JOINT_NORMALIZATION_LIMITS_RAD = (
    (-0.04480, 1.65080),
    (-0.16590, 0.93390),
    (-0.49320, 1.62720),
    (-0.49320, 1.62720),
    (-0.32695, 1.63595),
    (-0.49500, 0.49500),
    (-0.49320, 1.62720),
    (-0.49320, 1.62720),
    (-0.32695, 1.63595),
    (-0.49500, 0.49500),
    (-0.49320, 1.62720),
    (-0.49320, 1.62720),
    (-0.32695, 1.63595),
    (-0.49500, 0.49500),
    (-0.49320, 1.62720),
    (-0.49320, 1.62720),
    (-0.32695, 1.63595),
    (-0.49500, 0.49500),
    (-0.49320, 1.62720),
    (-0.49320, 1.62720),
)

if len(TEACHER_PREGRASP_JOINT_POSITIONS) != 20:
    raise ValueError("teacher pregrasp must contain exactly 20 joints")
if len(TEACHER_JOINT_NORMALIZATION_LIMITS_RAD) != 20:
    raise ValueError("teacher normalization table must contain exactly 20 joints")


def teacher_joint_pos_limit_normalized(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return current q in the successful 103D teacher's old normalization."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    if joint_pos.shape[-1] != len(TEACHER_JOINT_NORMALIZATION_LIMITS_RAD):
        raise RuntimeError(
            "teacher joint observation resolved the wrong width: "
            f"{joint_pos.shape[-1]} != 20"
        )
    limits = torch.as_tensor(
        TEACHER_JOINT_NORMALIZATION_LIMITS_RAD,
        dtype=joint_pos.dtype,
        device=joint_pos.device,
    )
    lower = limits[:, 0]
    upper = limits[:, 1]
    center = 0.5 * (lower + upper)
    return 2.0 * (joint_pos - center) / (upper - lower)


def teacher_previous_action(
    env: ManagerBasedEnv,
    student_action_scale_rad: tuple[float, ...],
) -> torch.Tensor:
    """Express the env's previous 105D-student action in old teacher units.

    The teacher-driven runner sends a mapped action ``a_old*0.1/s_new`` to the
    current action term.  ActionManager therefore stores that mapped value.
    Multiplying by ``s_new/0.1`` reconstructs the exact old 20D action-history
    feature expected by the teacher.
    """
    previous_action = env.action_manager.prev_action
    scale = torch.as_tensor(
        student_action_scale_rad,
        dtype=previous_action.dtype,
        device=previous_action.device,
    )
    if previous_action.shape[-1] != scale.numel():
        raise RuntimeError(
            "teacher previous-action width does not match its scale vector: "
            f"{previous_action.shape[-1]} != {scale.numel()}"
        )
    return previous_action * (scale / TEACHER_ACTION_SCALE_RAD)
