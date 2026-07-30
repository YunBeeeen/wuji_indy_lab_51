"""Functional-grasp rewards for the one-stick A1 experiment."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from isaac_neuromeka.mdp.rewards import box_ground_clearance, object_in_finger_cage

from .target_grasp import (
    fingertip_grip_region_error,
    hand_tool_orientation_error,
    index_grip_error,
    thumb_grip_error,
)


def balanced_tripod_cage_gate(
    env: ManagerBasedRLEnv,
    index_cage_cfg: SceneEntityCfg,
    middle_cage_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    num_points: int = 3,
    point_fractions: tuple[float, ...] | None = None,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
) -> torch.Tensor:
    """Require both thumb-index and thumb-middle cage groups to engage."""
    index_gate = object_in_finger_cage(
        env,
        index_cage_cfg,
        object_cfg,
        object_half_extent,
        num_points,
        sphere_radius,
        depth_max,
        point_fractions,
    )
    middle_gate = object_in_finger_cage(
        env,
        middle_cage_cfg,
        object_cfg,
        object_half_extent,
        num_points,
        sphere_radius,
        depth_max,
        point_fractions,
    )
    return torch.minimum(index_gate, middle_gate)


def object_lift_in_balanced_tripod_cage(
    env: ManagerBasedRLEnv,
    index_cage_cfg: SceneEntityCfg,
    middle_cage_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    num_points: int = 3,
    point_fractions: tuple[float, ...] | None = None,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
    lift_height: float = 0.08,
    surface_z: float = 0.0,
) -> torch.Tensor:
    """Reward clearance only while both functional cage groups remain engaged."""
    gate = balanced_tripod_cage_gate(
        env,
        index_cage_cfg,
        middle_cage_cfg,
        object_cfg,
        object_half_extent,
        num_points,
        point_fractions,
        sphere_radius,
        depth_max,
    )
    clearance = box_ground_clearance(env, object_cfg, object_half_extent, surface_z)
    lift = torch.clamp(clearance, 0.0, lift_height) / lift_height
    return gate * lift


# ── 5-finger wrap gate (2026-07-25, Phase 1 주먹 파지) ──────────────────────────
# tripod(엄지 대 검지·중지 2쌍) → penta(엄지 대 검지·중지·약지·새끼 4쌍)로 확장.
# 엄지가 한 면, 나머지 4손가락이 반대 면을 감싸는 주먹 파지를 강제. min이라 4쌍이 모두
# 물려야 게이트가 열림 → lift·hold가 5손가락 wrap을 요구. tripod 함수는 그대로 보존(되돌리기).
def balanced_penta_cage_gate(
    env: ManagerBasedRLEnv,
    index_cage_cfg: SceneEntityCfg,
    middle_cage_cfg: SceneEntityCfg,
    ring_cage_cfg: SceneEntityCfg,
    pinky_cage_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    num_points: int = 3,
    point_fractions: tuple[float, ...] | None = None,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
) -> torch.Tensor:
    """Require thumb-vs-{index,middle,ring,pinky} cages to all engage (min)."""
    gate = None
    for cage_cfg in (index_cage_cfg, middle_cage_cfg, ring_cage_cfg, pinky_cage_cfg):
        g = object_in_finger_cage(
            env,
            cage_cfg,
            object_cfg,
            object_half_extent,
            num_points,
            sphere_radius,
            depth_max,
            point_fractions,
        )
        gate = g if gate is None else torch.minimum(gate, g)
    return gate


def object_lift_in_balanced_penta_cage(
    env: ManagerBasedRLEnv,
    index_cage_cfg: SceneEntityCfg,
    middle_cage_cfg: SceneEntityCfg,
    ring_cage_cfg: SceneEntityCfg,
    pinky_cage_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    num_points: int = 3,
    point_fractions: tuple[float, ...] | None = None,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
    lift_height: float = 0.08,
    surface_z: float = 0.0,
    gate_exponent: float = 1.0,
) -> torch.Tensor:
    """Reward clearance only while the 5-finger wrap remains engaged.

    ``gate_exponent`` > 1 이면 느슨한 게이트에서 lift 지급이 급감(gate^k) — "꽉 물어야
    든다"를 부드럽게 강제. 기본 1.0(선형)이라 penta 게이트만으로 시작하고, 느슨-lift가
    드롭을 유발하면 그때 k=2~3으로 올리면 됨(코드 수정 없이 cfg 인자만).
    """
    gate = balanced_penta_cage_gate(
        env,
        index_cage_cfg,
        middle_cage_cfg,
        ring_cage_cfg,
        pinky_cage_cfg,
        object_cfg,
        object_half_extent,
        num_points,
        point_fractions,
        sphere_radius,
        depth_max,
    )
    clearance = box_ground_clearance(env, object_cfg, object_half_extent, surface_z)
    lift = torch.clamp(clearance, 0.0, lift_height) / lift_height
    return torch.pow(gate, gate_exponent) * lift


# ── N-finger wrap gate 공용 헬퍼 (2026-07-26) ──────────────────────────────────
# 엄지 대 여러 손가락 cage의 min. cage_cfgs는 파이썬 리스트지만 **공개 함수는 반드시
# SceneEntityCfg를 개별 named 파라미터로 받아야** 매니저가 resolve한다(manager_base.py:398
# 은 params의 직접 SceneEntityCfg만 resolve, 리스트 안은 안 함). 그래서 공개 함수에서
# 이 헬퍼로 리스트를 만들어 넘기는 구조로 둔다.
def _min_cage_gate(
    env: ManagerBasedRLEnv,
    cage_cfgs,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    num_points: int,
    point_fractions: tuple[float, ...] | None,
    sphere_radius: float,
    depth_max: float,
) -> torch.Tensor:
    gate = None
    for cage_cfg in cage_cfgs:
        g = object_in_finger_cage(
            env,
            cage_cfg,
            object_cfg,
            object_half_extent,
            num_points,
            sphere_radius,
            depth_max,
            point_fractions,
        )
        gate = g if gate is None else torch.minimum(gate, g)
    return gate


# ── 4-finger wrap gate (2026-07-26) ────────────────────────────────────────────
# penta에서 새끼(pinky)를 뺀 버전: 엄지 대 검지·중지·약지 3쌍. 새끼는 엄지 대향이
# 해부학적으로 가장 어려워 안 붙어 penta 게이트를 0으로 끌던 문제(21-44-48 실측) 대응.
# 새끼는 게이트에서 빼되 pinky_grip(약한 progress)으로만 "오면 좋고" 유도.
def balanced_quad_cage_gate(
    env: ManagerBasedRLEnv,
    index_cage_cfg: SceneEntityCfg,
    middle_cage_cfg: SceneEntityCfg,
    ring_cage_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    num_points: int = 3,
    point_fractions: tuple[float, ...] | None = None,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
) -> torch.Tensor:
    """Require thumb-vs-{index,middle,ring} cages to all engage (min)."""
    return _min_cage_gate(
        env,
        (index_cage_cfg, middle_cage_cfg, ring_cage_cfg),
        object_cfg,
        object_half_extent,
        num_points,
        point_fractions,
        sphere_radius,
        depth_max,
    )


def object_lift_in_balanced_quad_cage(
    env: ManagerBasedRLEnv,
    index_cage_cfg: SceneEntityCfg,
    middle_cage_cfg: SceneEntityCfg,
    ring_cage_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    num_points: int = 3,
    point_fractions: tuple[float, ...] | None = None,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
    lift_height: float = 0.08,
    surface_z: float = 0.0,
    gate_exponent: float = 1.0,
) -> torch.Tensor:
    """Reward clearance only while the 4-finger wrap (thumb+index+middle+ring) engages."""
    gate = balanced_quad_cage_gate(
        env,
        index_cage_cfg,
        middle_cage_cfg,
        ring_cage_cfg,
        object_cfg,
        object_half_extent,
        num_points,
        point_fractions,
        sphere_radius,
        depth_max,
    )
    clearance = box_ground_clearance(env, object_cfg, object_half_extent, surface_z)
    lift = torch.clamp(clearance, 0.0, lift_height) / lift_height
    return torch.pow(gate, gate_exponent) * lift


def _maintained_fingertip_proximity_gate(
    env: ManagerBasedRLEnv,
    palm_cfg: SceneEntityCfg,
    fingertip_cfgs: tuple[SceneEntityCfg, ...],
    grip_regions: tuple[dict, ...],
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float],
    proximity_near: float,
    proximity_far: float,
) -> torch.Tensor:
    """Smooth minimum gate over current fingertip-to-region distances."""
    if proximity_far <= proximity_near:
        raise ValueError(
            f"proximity_far ({proximity_far}) must be greater than proximity_near ({proximity_near})"
        )

    scale = proximity_far - proximity_near
    gate = None
    for fingertip_cfg, grip_region in zip(fingertip_cfgs, grip_regions, strict=True):
        distance = fingertip_grip_region_error(
            env,
            palm_cfg,
            fingertip_cfg,
            object_cfg,
            object_half_extent=object_half_extent,
            **grip_region,
        )
        current = torch.clamp((proximity_far - distance) / scale, min=0.0, max=1.0)
        gate = current if gate is None else torch.minimum(gate, current)
    return gate


def maintained_tripod_cage_gate(
    env: ManagerBasedRLEnv,
    index_cage_cfg: SceneEntityCfg,
    middle_cage_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    thumb_fingertip_cfg: SceneEntityCfg,
    index_fingertip_cfg: SceneEntityCfg,
    middle_fingertip_cfg: SceneEntityCfg,
    thumb_grip_region: dict,
    index_grip_region: dict,
    middle_grip_region: dict,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    num_points: int = 3,
    point_fractions: tuple[float, ...] | None = None,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
    proximity_near: float = 0.02,
    proximity_far: float = 0.10,
) -> torch.Tensor:
    """Tripod cage multiplied by maintained thumb/index/middle region proximity."""
    cage_gate = balanced_tripod_cage_gate(
        env,
        index_cage_cfg,
        middle_cage_cfg,
        object_cfg,
        object_half_extent,
        num_points,
        point_fractions,
        sphere_radius,
        depth_max,
    )
    proximity_gate = _maintained_fingertip_proximity_gate(
        env,
        palm_cfg,
        (thumb_fingertip_cfg, index_fingertip_cfg, middle_fingertip_cfg),
        (thumb_grip_region, index_grip_region, middle_grip_region),
        object_cfg,
        object_half_extent,
        proximity_near,
        proximity_far,
    )
    return cage_gate * proximity_gate


def object_lift_in_maintained_tripod_cage(
    env: ManagerBasedRLEnv,
    index_cage_cfg: SceneEntityCfg,
    middle_cage_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    thumb_fingertip_cfg: SceneEntityCfg,
    index_fingertip_cfg: SceneEntityCfg,
    middle_fingertip_cfg: SceneEntityCfg,
    thumb_grip_region: dict,
    index_grip_region: dict,
    middle_grip_region: dict,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    num_points: int = 3,
    point_fractions: tuple[float, ...] | None = None,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
    proximity_near: float = 0.02,
    proximity_far: float = 0.10,
    lift_height: float = 0.08,
    surface_z: float = 0.0,
) -> torch.Tensor:
    """Reward clearance only while the maintained tripod gate is engaged."""
    gate = maintained_tripod_cage_gate(
        env,
        index_cage_cfg,
        middle_cage_cfg,
        palm_cfg,
        thumb_fingertip_cfg,
        index_fingertip_cfg,
        middle_fingertip_cfg,
        thumb_grip_region,
        index_grip_region,
        middle_grip_region,
        object_cfg,
        object_half_extent,
        num_points,
        point_fractions,
        sphere_radius,
        depth_max,
        proximity_near,
        proximity_far,
    )
    clearance = box_ground_clearance(env, object_cfg, object_half_extent, surface_z)
    return gate * (torch.clamp(clearance, 0.0, lift_height) / lift_height)


def maintained_quad_cage_gate(
    env: ManagerBasedRLEnv,
    index_cage_cfg: SceneEntityCfg,
    middle_cage_cfg: SceneEntityCfg,
    ring_cage_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    thumb_fingertip_cfg: SceneEntityCfg,
    index_fingertip_cfg: SceneEntityCfg,
    middle_fingertip_cfg: SceneEntityCfg,
    ring_fingertip_cfg: SceneEntityCfg,
    thumb_grip_region: dict,
    index_grip_region: dict,
    middle_grip_region: dict,
    ring_grip_region: dict,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    num_points: int = 3,
    point_fractions: tuple[float, ...] | None = None,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
    proximity_near: float = 0.02,
    proximity_far: float = 0.10,
) -> torch.Tensor:
    """Quad cage multiplied by maintained thumb/index/middle/ring region proximity."""
    cage_gate = balanced_quad_cage_gate(
        env,
        index_cage_cfg,
        middle_cage_cfg,
        ring_cage_cfg,
        object_cfg,
        object_half_extent,
        num_points,
        point_fractions,
        sphere_radius,
        depth_max,
    )
    proximity_gate = _maintained_fingertip_proximity_gate(
        env,
        palm_cfg,
        (
            thumb_fingertip_cfg,
            index_fingertip_cfg,
            middle_fingertip_cfg,
            ring_fingertip_cfg,
        ),
        (thumb_grip_region, index_grip_region, middle_grip_region, ring_grip_region),
        object_cfg,
        object_half_extent,
        proximity_near,
        proximity_far,
    )
    return cage_gate * proximity_gate


def object_lift_in_maintained_quad_cage(
    env: ManagerBasedRLEnv,
    index_cage_cfg: SceneEntityCfg,
    middle_cage_cfg: SceneEntityCfg,
    ring_cage_cfg: SceneEntityCfg,
    palm_cfg: SceneEntityCfg,
    thumb_fingertip_cfg: SceneEntityCfg,
    index_fingertip_cfg: SceneEntityCfg,
    middle_fingertip_cfg: SceneEntityCfg,
    ring_fingertip_cfg: SceneEntityCfg,
    thumb_grip_region: dict,
    index_grip_region: dict,
    middle_grip_region: dict,
    ring_grip_region: dict,
    object_cfg: SceneEntityCfg,
    object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
    num_points: int = 3,
    point_fractions: tuple[float, ...] | None = None,
    sphere_radius: float = 0.005,
    depth_max: float = 0.005,
    proximity_near: float = 0.02,
    proximity_far: float = 0.10,
    lift_height: float = 0.08,
    surface_z: float = 0.0,
    gate_exponent: float = 1.0,
) -> torch.Tensor:
    """Reward clearance only while the maintained quad gate is engaged."""
    gate = maintained_quad_cage_gate(
        env,
        index_cage_cfg,
        middle_cage_cfg,
        ring_cage_cfg,
        palm_cfg,
        thumb_fingertip_cfg,
        index_fingertip_cfg,
        middle_fingertip_cfg,
        ring_fingertip_cfg,
        thumb_grip_region,
        index_grip_region,
        middle_grip_region,
        ring_grip_region,
        object_cfg,
        object_half_extent,
        num_points,
        point_fractions,
        sphere_radius,
        depth_max,
        proximity_near,
        proximity_far,
    )
    clearance = box_ground_clearance(env, object_cfg, object_half_extent, surface_z)
    lift = torch.clamp(clearance, 0.0, lift_height) / lift_height
    return torch.pow(gate, gate_exponent) * lift


class FingertipGripProgressReward(ManagerTermBase):
    """Signed progress toward an object-relative fingertip grip region."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_error = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    def _error(self) -> torch.Tensor:
        p = self.cfg.params
        return fingertip_grip_region_error(
            self._env,
            p["palm_cfg"],
            p["fingertip_cfg"],
            p["object_cfg"],
            object_half_extent=p.get("object_half_extent", (0.01, 0.09, 0.01)),
            long_axis=p.get("long_axis", 1),
            axial_region=p.get("axial_region", (-0.60, -0.30)),
            surface_axis=p.get("surface_axis", 2),
            surface_sign=p.get("surface_sign", 1.0),
            surface_offset=p.get("surface_offset", 0.0),
            surface_tolerance=p.get("surface_tolerance", 0.005),
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_error[env_ids] = self._error()[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
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
        distance_scale: float = 0.20,
    ) -> torch.Tensor:
        del env, palm_cfg, fingertip_cfg, object_cfg, object_half_extent
        del long_axis, axial_region, surface_axis, surface_sign, surface_offset, surface_tolerance
        current = self._error()
        progress = (self._previous_error - current) / distance_scale
        self._previous_error[:] = current
        return torch.clamp(progress, min=-1.0, max=1.0)


class HandToolOrientationProgressReward(ManagerTermBase):
    """Signed progress of palm orientation relative to the stick."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_error = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    def _error(self) -> torch.Tensor:
        p = self.cfg.params
        return hand_tool_orientation_error(
            self._env,
            p["palm_cfg"],
            p["object_cfg"],
            target_buffer_name=p.get("target_buffer_name", "chopstick_target_palm_quat_o"),
            fallback_quat_o=p.get("fallback_quat_o", (1.0, 0.0, 0.0, 0.0)),
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_error[env_ids] = self._error()[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        palm_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        target_buffer_name: str = "chopstick_target_palm_quat_o",
        fallback_quat_o: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        angle_scale: float = 0.7853981633974483,
    ) -> torch.Tensor:
        del env, palm_cfg, object_cfg, target_buffer_name, fallback_quat_o
        current = self._error()
        progress = (self._previous_error - current) / angle_scale
        self._previous_error[:] = current
        return torch.clamp(progress, min=-1.0, max=1.0)


class FunctionalGraspHeld(ManagerTermBase):
    """Terminal condition for a lifted, stable constraint-based one-stick grasp."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stable_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        palm_cfg: SceneEntityCfg,
        index_cfg: SceneEntityCfg,
        thumb_cfg: SceneEntityCfg,
        index_cage_cfg: SceneEntityCfg,
        middle_cage_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg,
        object_half_extent: tuple[float, float, float] = (0.01, 0.09, 0.01),
        index_target: dict | None = None,
        thumb_target: dict | None = None,
        target_buffer_name: str = "chopstick_target_palm_quat_o",
        fallback_quat_o: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        num_points: int = 3,
        point_fractions: tuple[float, ...] | None = None,
        sphere_radius: float = 0.005,
        depth_max: float = 0.005,
        index_error_limit: float = 0.02,
        thumb_error_limit: float = 0.02,
        orientation_error_limit: float = 0.2617993877991494,
        gate_threshold: float = 0.3,
        clearance_threshold: float = 0.05,
        surface_z: float = 0.0,
        hold_steps: int = 15,
    ) -> torch.Tensor:
        if index_target is None:
            index_target = {}
        if thumb_target is None:
            thumb_target = {"surface_axis": 0, "surface_sign": 1.0}
        index_err = index_grip_error(
            env,
            palm_cfg,
            index_cfg,
            object_cfg,
            object_half_extent=object_half_extent,
            **index_target,
        )
        thumb_err = thumb_grip_error(
            env,
            palm_cfg,
            thumb_cfg,
            object_cfg,
            object_half_extent=object_half_extent,
            **thumb_target,
        )
        orientation_err = hand_tool_orientation_error(
            env,
            palm_cfg,
            object_cfg,
            target_buffer_name=target_buffer_name,
            fallback_quat_o=fallback_quat_o,
        )
        gate = balanced_tripod_cage_gate(
            env,
            index_cage_cfg,
            middle_cage_cfg,
            object_cfg,
            object_half_extent,
            num_points,
            point_fractions,
            sphere_radius,
            depth_max,
        )
        clearance = box_ground_clearance(env, object_cfg, object_half_extent, surface_z)
        valid = (
            (index_err < index_error_limit)
            & (thumb_err < thumb_error_limit)
            & (orientation_err < orientation_error_limit)
            & (gate > gate_threshold)
            & (clearance > clearance_threshold)
        )
        self._stable_steps = torch.where(valid, self._stable_steps + 1, torch.zeros_like(self._stable_steps))
        return self._stable_steps >= hold_steps
