"""Runtime tuner for Isaac Lab implicit PD actuators."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .base import (
    ActuatorGroupInfo,
    ActuatorTuningAdapter,
    AppliedGainState,
    EffortSignals,
    GainState,
    JointInfo,
)
from .version_compat import validate_articulation_api


class IsaacLabImplicitActuatorAdapter(ActuatorTuningAdapter):
    """Resolve real joints and update implicit PhysX drives through public APIs."""

    def __init__(self, articulation: Any, articulation_cfg: Any) -> None:
        # Imports are intentionally delayed until the Isaac Sim child is initialized.
        import torch
        from isaaclab.actuators import ImplicitActuator

        validate_articulation_api(articulation)
        self._torch = torch
        self._articulation = articulation
        self._cfg = articulation_cfg
        self._implicit_type = ImplicitActuator
        self._warnings: list[str] = []
        self._groups: list[ActuatorGroupInfo] = []
        self._joint_to_groups: dict[int, list[str]] = defaultdict(list)
        self._joint_to_actuator: dict[int, tuple[str, Any, int]] = {}
        self._resolve_groups()
        self._original_gains = {
            index: self._read_gain_tensor(index) for index in range(self._articulation.num_joints)
        }
        self._joint_infos = self._build_joint_infos()

    @staticmethod
    def _indices_to_list(indices: Any, count: int) -> list[int]:
        if isinstance(indices, slice):
            return list(range(count))[indices]
        if hasattr(indices, "detach"):
            return [int(value) for value in indices.detach().cpu().flatten().tolist()]
        return [int(value) for value in indices]

    def _resolve_groups(self) -> None:
        actual_names = list(self._articulation.joint_names)
        cfg_actuators = getattr(self._cfg, "actuators", {})
        runtime_group_names = set(self._articulation.actuators)
        for group_name, cfg in cfg_actuators.items():
            if group_name not in runtime_group_names:
                expressions = getattr(cfg, "joint_names_expr", ()) or ()
                if isinstance(expressions, str):
                    expressions = (expressions,)
                self._warnings.append(
                    f"Configured actuator group {group_name!r} expressions {tuple(expressions)!r} "
                    "matched no actual joints."
                )
        for group_name, actuator in self._articulation.actuators.items():
            indices = self._indices_to_list(actuator.joint_indices, self._articulation.num_joints)
            resolved_names = tuple(actual_names[index] for index in indices)
            cfg = cfg_actuators.get(group_name)
            raw_expressions = getattr(cfg, "joint_names_expr", ()) or ()
            expressions = (raw_expressions,) if isinstance(raw_expressions, str) else tuple(raw_expressions)
            tunable = isinstance(actuator, self._implicit_type)
            limitation = None
            if not tunable:
                limitation = (
                    f"Unsupported actuator type {type(actuator).__name__}: runtime stiffness/damping "
                    "tuning is disabled; state monitoring remains available."
                )
                self._warnings.append(f"Actuator group {group_name!r}: {limitation}")
            if not resolved_names:
                self._warnings.append(
                    f"Actuator group {group_name!r} expressions {expressions!r} matched no actual joints."
                )
            self._groups.append(
                ActuatorGroupInfo(
                    name=group_name,
                    actuator_type=type(actuator).__name__,
                    joint_names_expr=expressions,
                    joint_names=resolved_names,
                    joint_indices=tuple(indices),
                    tunable=tunable,
                    stiffness_source_type=type(getattr(cfg, "stiffness", None)).__name__,
                    damping_source_type=type(getattr(cfg, "damping", None)).__name__,
                    effort_limit_source_type=type(
                        getattr(cfg, "effort_limit_sim", None)
                        if getattr(cfg, "effort_limit_sim", None) is not None
                        else getattr(cfg, "effort_limit", None)
                    ).__name__,
                    runtime_update_scope="joint_and_group" if tunable else "monitoring_only",
                    limitation=limitation,
                )
            )
            for local_index, global_index in enumerate(indices):
                self._joint_to_groups[global_index].append(group_name)
                if global_index not in self._joint_to_actuator:
                    self._joint_to_actuator[global_index] = (group_name, actuator, local_index)

        for index, groups in self._joint_to_groups.items():
            if len(groups) > 1:
                self._warnings.append(
                    f"Joint {actual_names[index]!r} is matched by multiple actuator groups: {groups}. "
                    f"The first group {groups[0]!r} is used for tuning."
                )
        for index, name in enumerate(actual_names):
            if index not in self._joint_to_groups:
                self._warnings.append(f"Joint {name!r} has no actuator group and is monitoring-only.")

    def _read_gain_tensor(self, joint_index: int) -> GainState:
        data = self._articulation.data
        return GainState(
            stiffness=float(data.joint_stiffness[0, joint_index].item()),
            damping=float(data.joint_damping[0, joint_index].item()),
            effort_limit=float(data.joint_effort_limits[0, joint_index].item()),
        )

    def _build_joint_infos(self) -> tuple[JointInfo, ...]:
        data = self._articulation.data
        infos: list[JointInfo] = []
        for index, name in enumerate(self._articulation.joint_names):
            mapping = self._joint_to_actuator.get(index)
            group_name = mapping[0] if mapping else None
            actuator = mapping[1] if mapping else None
            infos.append(
                JointInfo(
                    name=name,
                    index=index,
                    actuator_group=group_name,
                    actuator_type=type(actuator).__name__ if actuator is not None else None,
                    tunable=isinstance(actuator, self._implicit_type),
                    lower_limit=float(data.joint_pos_limits[0, index, 0].item()),
                    upper_limit=float(data.joint_pos_limits[0, index, 1].item()),
                    velocity_limit=float(data.joint_vel_limits[0, index].item()),
                    original_gain=self._original_gains[index],
                )
            )
        return tuple(infos)

    @property
    def joint_infos(self) -> tuple[JointInfo, ...]:
        return self._joint_infos

    @property
    def actuator_groups(self) -> tuple[ActuatorGroupInfo, ...]:
        return tuple(self._groups)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    def list_tunable_joints(self) -> list[str]:
        return [info.name for info in self._joint_infos if info.tunable]

    def read_gains(self, joint_index: int) -> GainState:
        self._validate_joint_index(joint_index)
        return self._read_gain_tensor(joint_index)

    @staticmethod
    def _validate_gain_values(stiffness: float, damping: float, effort_limit: float) -> GainState:
        values = (float(stiffness), float(damping), float(effort_limit))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Stiffness, damping, and effort limit must be finite.")
        if values[0] < 0.0 or values[1] < 0.0 or values[2] <= 0.0:
            raise ValueError("Require stiffness >= 0, damping >= 0, and effort limit > 0.")
        return GainState(*values)

    def _validate_joint_index(self, joint_index: int) -> None:
        if not 0 <= int(joint_index) < self._articulation.num_joints:
            raise IndexError(f"Joint index out of range: {joint_index}")

    def _group_joint_indices(self, joint_index: int) -> list[int]:
        mapping = self._joint_to_actuator.get(joint_index)
        if mapping is None:
            raise RuntimeError(f"Joint {self._articulation.joint_names[joint_index]!r} has no actuator group.")
        group_name = mapping[0]
        return list(next(group.joint_indices for group in self._groups if group.name == group_name))

    def _sync_actuator_model(self, joint_index: int, gains: GainState) -> None:
        mapping = self._joint_to_actuator[joint_index]
        _, actuator, local_index = mapping
        if not isinstance(actuator, self._implicit_type):
            raise RuntimeError(
                f"Unsupported actuator type {type(actuator).__name__}; only implicit PD drives are tunable."
            )
        actuator.stiffness[:, local_index] = gains.stiffness
        actuator.damping[:, local_index] = gains.damping
        actuator.effort_limit_sim[:, local_index] = gains.effort_limit
        actuator.effort_limit[:, local_index] = gains.effort_limit

    def apply_gains(
        self,
        joint_index: int,
        stiffness: float,
        damping: float,
        effort_limit: float,
        *,
        apply_to_group: bool = False,
    ) -> list[AppliedGainState]:
        self._validate_joint_index(joint_index)
        requested = self._validate_gain_values(stiffness, damping, effort_limit)
        mapping = self._joint_to_actuator.get(joint_index)
        if mapping is None or not isinstance(mapping[1], self._implicit_type):
            kind = type(mapping[1]).__name__ if mapping else "unmapped"
            raise RuntimeError(f"Joint is monitoring-only; unsupported actuator type: {kind}.")
        indices = self._group_joint_indices(joint_index) if apply_to_group else [joint_index]
        for index in indices:
            self._articulation.write_joint_stiffness_to_sim(requested.stiffness, joint_ids=[index])
            self._articulation.write_joint_damping_to_sim(requested.damping, joint_ids=[index])
            self._articulation.write_joint_effort_limit_to_sim(requested.effort_limit, joint_ids=[index])
            self._sync_actuator_model(index, requested)

        results: list[AppliedGainState] = []
        for index in indices:
            group_name = self._joint_to_actuator[index][0]
            results.append(
                AppliedGainState(
                    joint_name=self._articulation.joint_names[index],
                    requested=requested,
                    applied=self._read_gain_tensor(index),
                    actuator_group=group_name,
                    scope="actuator_group" if apply_to_group else "joint",
                )
            )
        return results

    def restore_original_gains(
        self, joint_index: int, *, apply_to_group: bool = False
    ) -> list[AppliedGainState]:
        self._validate_joint_index(joint_index)
        indices = self._group_joint_indices(joint_index) if apply_to_group else [joint_index]
        results: list[AppliedGainState] = []
        for index in indices:
            gains = self._original_gains[index]
            results.extend(
                self.apply_gains(
                    index,
                    gains.stiffness,
                    gains.damping,
                    gains.effort_limit,
                    apply_to_group=False,
                )
            )
        return results

    def read_effort_signals(self, joint_index: int) -> EffortSignals:
        self._validate_joint_index(joint_index)
        data = self._articulation.data
        computed = getattr(data, "computed_torque", None)
        applied = getattr(data, "applied_torque", None)
        return EffortSignals(
            computed_effort=(float(computed[0, joint_index].item()) if computed is not None else None),
            applied_effort=(float(applied[0, joint_index].item()) if applied is not None else None),
            measured_joint_effort=None,
        )
