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
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul, quat_unique

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
