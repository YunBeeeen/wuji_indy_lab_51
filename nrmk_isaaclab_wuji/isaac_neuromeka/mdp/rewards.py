from __future__ import annotations

import pdb  # noqa:F401
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaac_neuromeka.assets.articulation import FiniteArticulation


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def end_effector_position_tracking_bounded(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    distance_max: float = 1.0,
) -> torch.Tensor:

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore

    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    distance_bonus = 1.0 - torch.clamp(distance, 0.0, distance_max) / distance_max

    return distance_bonus


def _box_signed_distance(
    points_w: torch.Tensor,
    box_pos_w: torch.Tensor,
    box_quat_w: torch.Tensor,
    half_extent: torch.Tensor,
) -> torch.Tensor:
    """Signed distance from world points to an oriented box surface. Negative inside the box.

    Args:
        points_w: (N, P, 3) query points in world frame.
        box_pos_w: (N, 3) box center.
        box_quat_w: (N, 4) box orientation (w, x, y, z).
        half_extent: (3,) box half-extents in its own frame.

    Returns:
        (N, P) signed distances.
    """
    rel = points_w - box_pos_w.unsqueeze(1)
    quat = box_quat_w.unsqueeze(1).expand(-1, rel.shape[1], -1)
    local = quat_apply_inverse(quat, rel)
    q = local.abs() - half_extent
    outside = torch.norm(torch.clamp(q, min=0.0), dim=-1)
    inside = torch.clamp(q.max(dim=-1).values, max=0.0)
    return outside + inside


def cage_points(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, num_points: int) -> torch.Tensor:
    """Virtual grasp-aperture points between the thumb tip and every opposing finger body.

    ``asset_cfg.body_names`` must resolve to ``[thumb_tip, *opposing]`` in that order (so
    ``preserve_order=True`` is required). One line is drawn from the thumb tip to each opposing
    body, carrying ``num_points`` equidistant points. Following Dexterous Pre-grasp Manipulation,
    list each finger twice — its tip (where an object can be pinch-grasped) and its mid-phalanx
    (where an object is held more securely).

    The paper uses one finger pair (thumb-middle, 6 points) because its ``r_grasp`` term separately
    pins the hand's rotation and every finger joint to a target grasp. We have no target grasp for a
    symmetric cube and so no ``r_grasp``; with only the thumb-middle line, the index finger was left
    completely unconstrained and the policy settled into a palm-up pose with the fingers crossed
    that still scored a perfect cage. Listing the index as well is the paper's own suggested
    extension ("it is straightforward to utilize several finger pairs at the same time") and it is
    what the chopstick grasp needs anyway.

    Returns:
        (N, len(opposing) * num_points, 3) world points.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids, :3]  # type: ignore
    if body_pos_w.shape[1] < 2:
        raise ValueError(
            f"cage terms expect [thumb_tip, *opposing] (2 bodies minimum), got {body_pos_w.shape[1]}."
        )
    thumb = body_pos_w[:, 0]
    opposing = body_pos_w[:, 1:]  # (N, M, 3)

    # `num_points` equidistant points strictly between the thumb tip and each opposing body.
    fractions = torch.arange(1, num_points + 1, dtype=thumb.dtype, device=thumb.device) / (num_points + 1)
    span = opposing - thumb.unsqueeze(1)  # (N, M, 3)
    points = thumb[:, None, None, :] + span.unsqueeze(2) * fractions.view(1, 1, -1, 1)
    return points.reshape(thumb.shape[0], -1, 3)


def _cage_sdf(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    num_points: int,
) -> torch.Tensor:
    """(N, M * num_points) signed distance from the cage points to the object surface."""
    obj: RigidObject = env.scene[object_cfg.name]
    points = cage_points(env, asset_cfg, num_points)
    half = torch.tensor(object_half_extent, dtype=points.dtype, device=points.device)
    return _box_signed_distance(points, obj.data.root_pos_w, obj.data.root_quat_w, half)


def object_in_finger_cage(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
    num_points: int = 3,
    sphere_radius: float = 0.02,
    depth_max: float = 0.03,
) -> torch.Tensor:
    """Reward caging the object between the thumb and an opposing finger (paper Eq. 15).

    Each cage point carries a sphere of ``sphere_radius``; the reward is how deeply those spheres
    overlap the object. Closing the hand pulls the points toward each other and pushes them inside
    the object, so enclosing is rewarded directly — no contact sensing needed. A distance-to-object
    reward does the opposite: touching a free object shoves it away, which *costs* reward, so
    distance shaping alone drives the hand to hover rather than grasp.
    """
    sdf = _cage_sdf(env, asset_cfg, object_cfg, object_half_extent, num_points)
    # How far each virtual sphere reaches into the object. Zero once a point sits more than
    # `sphere_radius` outside the surface; saturates once it is `depth_max` deep inside.
    penetration = sphere_radius - sdf
    return torch.clamp(penetration / (sphere_radius + depth_max), 0.0, 1.0).mean(dim=1)


class ObjectCageProgressReward(ManagerTermBase):
    """Reward closing the grasp aperture onto the object surface (paper Eq. 14).

    Acts on the *same* cage points as :func:`object_in_finger_cage`, which is the whole point: it
    pulls the gap between the thumb and the opposing finger onto the object, so the object ends up
    *between* the fingers. Driving fingertips at the object centre instead produces a hand that
    pokes the object with its thumb while the other fingers hang back — a pose from which closing
    the hand can only push the cage points back out of the object.

    Differential (t-1 vs t), *not* clamped at zero, and baselined at reset:

    * retreating costs reward, instead of being free;
    * the episode total telescopes to ``d(reset) - d(final)``, which depends only on where the hand
      ends up. No pacing trick or first-step swing-out can inflate it.

    A best-so-far variant with a floor at zero has neither property: it pays for the *excursion*
    rather than the end state, so the policy learns to fling the arm away on step 1 (raising the
    baseline it gets paid to recover) and to dawdle so that more steps each bank a new best.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_distance = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    def _mean_sdf(self) -> torch.Tensor:
        p = self.cfg.params
        return _cage_sdf(
            self._env,
            p["asset_cfg"],
            p["object_cfg"],
            p.get("object_half_extent", (0.03, 0.03, 0.03)),
            p.get("num_points", 3),
        ).mean(dim=1)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # Baseline at the reset pose, before the policy acts. Seeding on the first __call__ instead
        # would let the first action inflate the baseline for free, which is exactly the swing-out.
        self._previous_distance[env_ids] = self._mean_sdf()[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
        num_points: int = 3,
        distance_max: float = 0.5,
    ) -> torch.Tensor:
        current = self._mean_sdf()
        progress = self._previous_distance - current
        self._previous_distance[:] = current
        return torch.clamp(progress / distance_max, min=-1.0, max=1.0)

def object_lift_in_cage(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.03, 0.03, 0.03),
    num_points: int = 3,
    sphere_radius: float = 0.005,
    depth_max: float = 0.02,
    initial_height: float = 0.03,
    lift_height: float = 0.08,
) -> torch.Tensor:
    """Reward lifting the object, but only for as long as the fingers actually cage it.

    This is the term that decides which grasps count. A pose that satisfies the cage geometry but
    cannot take the object's weight is not a grasp, and no amount of pose shaping can tell the two
    apart — asking the hand to carry the thing can. The paper adds ``r_lift`` for precisely this
    reason: without it a policy can "satisfy the constraint without actually stably grasping the
    object" (fake success).

    So we deliberately do *not* prescribe a hand orientation or a nominal finger shape here. The
    object is symmetric and there is no functional grasp to hit — the paper's ``r_hr``/``r_hj`` exist
    to hold a drill *so its trigger can be pulled*, which is a requirement we simply don't have. Any
    pose that lifts the cube is a legitimate grasp; let the load decide.

    Gating on the cage is what stops the obvious cheat, which is to flick the object upward without
    ever holding it.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    gate = object_in_finger_cage(
        env, asset_cfg, object_cfg, object_half_extent, num_points, sphere_radius, depth_max
    )
    # Dense in height, so even the millimetre of lift the current policy already produces carries a
    # gradient. A sparse "is it above N cm" term would sit at exactly zero forever and teach nothing.
    height = obj.data.root_pos_w[:, 2] - initial_height
    lift = torch.clamp(height, 0.0, lift_height) / lift_height
    return gate * lift


def object_to_target_position_tracking_bounded(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    target_pos: tuple[float, float, float] = (0.55, -0.05, 0.12),
    distance_max: float = 0.6,
    min_height: float = 0.08,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    target_pos_w = obj.data.root_pos_w.new_tensor(target_pos).unsqueeze(0)

    distance = torch.norm(obj.data.root_pos_w - target_pos_w, dim=1)
    distance_bonus = 1.0 - torch.clamp(distance, 0.0, distance_max) / distance_max
    lifted = obj.data.root_pos_w[:, 2] >= min_height
    return torch.where(lifted, distance_bonus, torch.zeros_like(distance_bonus))


def end_effector_orientation_tracking_distance_bounded(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, distance_max: float = 0.5
) -> torch.Tensor:

    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore

    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    orientation_bonus = 1.0 - torch.clamp(orientation_error, 0.0, 3.14) / 3.14

    bad_indicies = distance > distance_max

    total_reward = orientation_bonus
    total_reward[bad_indicies] = 0.0

    return total_reward


def end_effector_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the end-effector speed using L2-norm.

    The function computes the end-effector speed as the L2-norm of the end-effector's speed.
    """

    asset: RigidObject = env.scene[asset_cfg.name]

    speed = torch.abs(asset.data.body_state_w[:, asset_cfg.body_ids[0], 7:10])
    return torch.norm(speed, dim=1)


def finite_joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L1-kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the L1 norm.
    """
    # extract the used quantities (to enable type-hinting)
    asset: FiniteArticulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset._finite_joint_vel[:, asset_cfg.joint_ids]), dim=1)


def action_second_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    # TODO: currently broken
    return torch.sum(
        torch.square(
            (env.action_manager.action - env.action_manager.prev_action)
            - (env.action_manager.prev_action - env.action_manager.prevprev_action)
        ),
        dim=1,
    )
