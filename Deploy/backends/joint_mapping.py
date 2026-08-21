"""Strict physical-name mapping between policy and MuJoCo arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..common.policy_contract import POLICY_JOINT_NAMES


# MuJoCo storage identifiers are intentionally outside the canonical contract.
MUJOCO_JOINT_NAMES = tuple(f"right_{name}" for name in POLICY_JOINT_NAMES)


@dataclass(frozen=True)
class JointMapping:
    """All forward/reverse maps required by the deployment invariant."""

    policy_to_mujoco_qpos: npt.NDArray[np.int32]
    mujoco_qpos_to_policy: npt.NDArray[np.int32]
    policy_to_mujoco_dof: npt.NDArray[np.int32]
    policy_to_mujoco_actuator: npt.NDArray[np.int32]
    mujoco_actuator_to_policy: npt.NDArray[np.int32]

    # Upper-case aliases make the requested invariant names explicit without
    # introducing a second mutable source of truth.
    @property
    def POLICY_TO_MUJOCO_QPOS(self) -> npt.NDArray[np.int32]:
        return self.policy_to_mujoco_qpos

    @property
    def MUJOCO_QPOS_TO_POLICY(self) -> npt.NDArray[np.int32]:
        return self.mujoco_qpos_to_policy

    @property
    def POLICY_TO_MUJOCO_ACTUATOR(self) -> npt.NDArray[np.int32]:
        return self.policy_to_mujoco_actuator

    @property
    def MUJOCO_ACTUATOR_TO_POLICY(self) -> npt.NDArray[np.int32]:
        return self.mujoco_actuator_to_policy


def build_joint_mapping(model) -> JointMapping:
    """Resolve every joint and position actuator by name or fail loudly."""

    import mujoco

    joint_names = _all_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    expected = set(MUJOCO_JOINT_NAMES)
    actual = set(joint_names)
    missing = sorted(expected - actual)
    if missing or len(joint_names) != len(actual):
        raise RuntimeError(
            "MuJoCo policy joint set is invalid: "
            f"missing={missing}, duplicate_names="
            f"{len(joint_names) != len(actual)}"
        )
    for joint_id, name in enumerate(joint_names):
        if name not in expected and int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError(f"Unexpected non-policy MuJoCo joint {name!r}.")

    qpos_by_policy: list[int] = []
    dof_by_policy: list[int] = []
    actuator_by_policy: list[int] = []
    actuator_names = _all_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    if len(set(actuator_names)) != len(actuator_names):
        raise RuntimeError("MuJoCo actuator names are not unique.")

    for joint_name in MUJOCO_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Missing MuJoCo joint {joint_name!r}.")
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise RuntimeError(f"Policy joint {joint_name!r} is not a 1-DoF hinge.")
        qpos_by_policy.append(int(model.jnt_qposadr[joint_id]))
        dof_by_policy.append(int(model.jnt_dofadr[joint_id]))

        matching_actuators = []
        for actuator_id in range(model.nu):
            if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            if int(model.actuator_trnid[actuator_id, 0]) == joint_id:
                matching_actuators.append(actuator_id)
        if len(matching_actuators) != 1:
            raise RuntimeError(
                f"Expected exactly one actuator for {joint_name!r}, got {matching_actuators}."
            )
        actuator_id = matching_actuators[0]
        _assert_position_actuator(model, actuator_id, joint_name)
        actuator_by_policy.append(actuator_id)

    if len(set(qpos_by_policy)) != len(MUJOCO_JOINT_NAMES):
        raise RuntimeError("Two policy joints resolve to the same MuJoCo qpos address.")
    if len(set(actuator_by_policy)) != len(MUJOCO_JOINT_NAMES):
        raise RuntimeError("Two policy joints resolve to the same MuJoCo actuator.")
    if set(actuator_by_policy) != set(range(model.nu)):
        extra = sorted(set(range(model.nu)) - set(actuator_by_policy))
        raise RuntimeError(f"Unexpected MuJoCo actuators outside the policy map: {extra}.")

    policy_to_qpos = np.asarray(qpos_by_policy, dtype=np.int32)
    qpos_to_policy = np.full(model.nq, -1, dtype=np.int32)
    for policy_index, qpos_address in enumerate(policy_to_qpos):
        qpos_to_policy[qpos_address] = policy_index
    policy_to_actuator = np.asarray(actuator_by_policy, dtype=np.int32)
    actuator_to_policy = np.full(model.nu, -1, dtype=np.int32)
    for policy_index, actuator_id in enumerate(policy_to_actuator):
        actuator_to_policy[actuator_id] = policy_index
    return JointMapping(
        policy_to_mujoco_qpos=policy_to_qpos,
        mujoco_qpos_to_policy=qpos_to_policy,
        policy_to_mujoco_dof=np.asarray(dof_by_policy, dtype=np.int32),
        policy_to_mujoco_actuator=policy_to_actuator,
        mujoco_actuator_to_policy=actuator_to_policy,
    )


def format_joint_mapping(model, mapping: JointMapping) -> str:
    """Render the complete 20-row policy/MuJoCo map."""

    import mujoco

    lines = ["[POLICY / MUJOCO JOINT MAP]"]
    for policy_index, canonical_name in enumerate(POLICY_JOINT_NAMES):
        actuator_id = int(mapping.policy_to_mujoco_actuator[policy_index])
        actuator_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        lines.extend(
            [
                "",
                f"Policy[{policy_index:02d}] {canonical_name}",
                f"    MuJoCo    : {MUJOCO_JOINT_NAMES[policy_index]}",
                f"    qpos addr : {int(mapping.policy_to_mujoco_qpos[policy_index])}",
                f"    dof addr  : {int(mapping.policy_to_mujoco_dof[policy_index])}",
                f"    actuator  : {actuator_id} ({actuator_name})",
            ]
        )
    return "\n".join(lines)


def _all_names(model, object_type, count: int) -> list[str]:
    import mujoco

    names = []
    for object_id in range(count):
        name = mujoco.mj_id2name(model, object_type, object_id)
        if not name:
            raise RuntimeError(f"Unnamed MuJoCo object {object_type} id={object_id}.")
        names.append(name)
    return names


def _assert_position_actuator(model, actuator_id: int, joint_name: str) -> None:
    """Recognize MuJoCo's compiled ``<position>`` affine actuator form."""

    import mujoco

    gain = float(model.actuator_gainprm[actuator_id, 0])
    bias_q = float(model.actuator_biasprm[actuator_id, 1])
    bias_qd = float(model.actuator_biasprm[actuator_id, 2])
    is_position = (
        int(model.actuator_gaintype[actuator_id]) == int(mujoco.mjtGain.mjGAIN_FIXED)
        and int(model.actuator_biastype[actuator_id]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
        and gain > 0.0
        and np.isclose(bias_q, -gain)
        # MuJoCo's <position kv="..."> shorthand stores -Kd here.  Zero is
        # valid too, but a positive velocity-feedback coefficient is not.
        and bias_qd <= 0.0
    )
    if not is_position:
        raise RuntimeError(
            f"Actuator {actuator_id} for {joint_name!r} is not a MuJoCo position actuator."
        )
