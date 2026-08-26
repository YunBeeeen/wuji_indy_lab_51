# [backend/MuJoCo] 고정 베이스 Wuji Hand MuJoCo 백엔드 — 리셋, 관절 읽기/쓰기, 스틱 고정, 게인 주입.
"""Real-ready fixed-base Wuji Hand 1 MuJoCo backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..common.backend_protocol import BackendHealth
from .joint_mapping import JointMapping, MUJOCO_JOINT_NAMES, build_joint_mapping
from .mujoco_scheduler import (
    MUJOCO_INTEGRATOR,
    MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP,
    SUPPORTED_INTEGRATORS,
    physics_dt_for_substeps,
    validate_hold_schedule,
)
from ..common.policy_contract import (
    ACTION_DIM,
    COMMAND_TARGET_LIMITS,
    DEPLOY_DAMPING_NMS_PER_RAD,
    DEPLOY_EFFORT_LIMITS_NM,
    DEPLOY_STIFFNESS_NM_PER_RAD,
    OFFICIAL_NOMINAL_PHYSICAL_LIMITS,
    REAL_HAND_FACTORY_LIMITS,
    PALM_FRAME_NAME,
    POLICY_JOINT_NAMES,
    WUJI_DESCRIPTION_REVISION,
)
from ..common.isaac_reset import (ISAAC_PREGRASP_JOINT_POSITIONS_RAD, MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ)
from ..vision.sim_aruco import HAND_DISPLAY_RGBA
from ..common.stick_pose import quaternion_multiply_wxyz


TIP_SITE_NAMES = tuple(f"finger{finger}_tip" for finger in range(1, 6))


_MJCF_DIR = Path(__file__).resolve().parents[1] / "assets/wuji_description/hand/body/mjcf"

# The chopstick-grasp scene: hand, two free sticks, D435 camera, testbed.
DEFAULT_MODEL_PATH = _MJCF_DIR / "right_with_tip_sites.xml"
# The finger-reach diagnostic scene: the fixed hand and nothing else.  A reach
# probe compares joint and fingertip trajectories, so a stray dynamic body would
# add contacts the Isaac finger_reach scene does not have.
FINGER_REACH_MODEL_PATH = _MJCF_DIR / "right_reach.xml"


class MujocoWujiHand:
    """Implement the Wuji backend boundary with the pinned official MJCF."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        physical_limit_tolerance_rad: float = 0.02,
        controller_gains: str = "vendor",
        physics_substeps: int = MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP,
        integrator: str = MUJOCO_INTEGRATOR,
    ):
        """Build the fixed-base hand backend.

        ``controller_gains="vendor"`` (default) keeps the pinned MJCF's own
        identified position-servo gains untouched.  As of 2026-08-18 these are
        the plant contract: they are the manufacturer's identification of this
        hardware, so both simulators and the real hand should agree on them.

        ``controller_gains="isaac_tuned"`` installs the hand-tuned values from
        ``hand_real_env_cfg.py`` instead.  Those were tuned in Isaac Sim against
        a plant with no rotor armature, are 1.5~7.2x stiffer, and are retained
        only to reproduce policies trained before the switch.

        ``physics_substeps`` and ``integrator`` are numerical-accuracy settings,
        not part of the policy contract: the timestep is always derived so that
        the substeps span exactly one 30 Hz policy step.  They exist because
        MuJoCo's solver needs a finer step than PhysX to resolve the same
        contact problem, not because the trained timing changed.
        """
        import mujoco

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Wuji MJCF does not exist: {self.model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self._apply_hand_display_color()
        if not np.isfinite(physical_limit_tolerance_rad) or physical_limit_tolerance_rad < 0:
            raise ValueError("Physical-limit tolerance must be finite and non-negative.")
        self.physical_limit_tolerance_rad = float(physical_limit_tolerance_rad)
        self.mapping: JointMapping = build_joint_mapping(self.model)
        self._assert_fixed_base()
        self._apply_physical_limits()
        integrator_name = str(integrator).strip().lower()
        if integrator_name not in SUPPORTED_INTEGRATORS:
            raise ValueError(
                f"integrator must be one of {SUPPORTED_INTEGRATORS}, got {integrator!r}."
            )
        self.physics_substeps = int(physics_substeps)
        self.integrator = integrator_name
        self.model.opt.timestep = physics_dt_for_substeps(self.physics_substeps)
        self.model.opt.integrator = {
            "euler": mujoco.mjtIntegrator.mjINT_EULER,
            "rk4": mujoco.mjtIntegrator.mjINT_RK4,
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        }[integrator_name]
        validate_hold_schedule(self.model.opt.timestep, self.physics_substeps)

        # Accept the pre-2026-08-18 names so older commands keep working.
        controller_gains = {"official": "vendor", "deploy": "isaac_tuned"}.get(
            controller_gains, controller_gains
        )
        if controller_gains not in ("vendor", "isaac_tuned"):
            raise ValueError(
                "controller_gains must be 'vendor' or 'isaac_tuned', got "
                f"{controller_gains!r}."
            )
        self.controller_gains = controller_gains

        # Official mass, inertia, geometry, armature and joint damping are always
        # preserved.  ctrlrange always becomes the policy command contract
        # (including the Joint4 floor).  The position-servo gains follow
        # ``controller_gains``: MuJoCo's compiled <position> form stores kp in
        # gainprm[0], -kp in biasprm[1] and -kv in biasprm[2].  The default
        # leaves them exactly as the vendor identified them.
        for policy_index, actuator_id in enumerate(self.mapping.policy_to_mujoco_actuator):
            self.model.actuator_ctrlrange[actuator_id] = COMMAND_TARGET_LIMITS[policy_index]
            self.model.actuator_ctrllimited[actuator_id] = 1
            if controller_gains == "isaac_tuned":
                stiffness = float(DEPLOY_STIFFNESS_NM_PER_RAD[policy_index])
                damping = float(DEPLOY_DAMPING_NMS_PER_RAD[policy_index])
                effort = float(DEPLOY_EFFORT_LIMITS_NM[policy_index])
                self.model.actuator_gainprm[actuator_id, 0] = stiffness
                self.model.actuator_biasprm[actuator_id, 1] = -stiffness
                self.model.actuator_biasprm[actuator_id, 2] = -damping
                self.model.actuator_forcerange[actuator_id] = (-effort, effort)
                self.model.actuator_forcelimited[actuator_id] = 1

        self.data = mujoco.MjData(self.model)
        self._palm_body_id = self._body_id(PALM_FRAME_NAME)
        self._tip_site_ids = np.asarray(
            [self._site_id(name) for name in TIP_SITE_NAMES], dtype=np.int32
        )
        if self.has_sticks:
            self._stick_joint_ids = np.asarray(
                [self._joint_id("stick1_free"), self._joint_id("stick2_free")], dtype=np.int32
            )
            self._stick_body_ids = np.asarray(
                [self._body_id("stick1"), self._body_id("stick2")], dtype=np.int32
            )
        else:
            self._stick_joint_ids = np.zeros(0, dtype=np.int32)
            self._stick_body_ids = np.zeros(0, dtype=np.int32)
        self.physics_step_count = 0
        self.last_reset_clamped = np.zeros(ACTION_DIM, dtype=np.bool_)
        # Sticky, mirroring RealWujiHand: a run that hit a safe stop must not
        # look like a clean one afterwards.
        self.safe_stopped = False
        self.safe_stop_reason: str | None = None
        self.reset()

    def _apply_hand_display_color(self) -> None:
        """Render every fixed-hand geom black without changing its dynamics."""

        import mujoco

        palm_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, PALM_FRAME_NAME
        )
        if palm_id < 0:
            raise RuntimeError(f"Missing hand root body {PALM_FRAME_NAME!r}.")
        for geom_id, body_id in enumerate(self.model.geom_bodyid):
            ancestor = int(body_id)
            while ancestor > 0 and ancestor != palm_id:
                ancestor = int(self.model.body_parentid[ancestor])
            if ancestor == palm_id:
                self.model.geom_rgba[geom_id] = HAND_DISPLAY_RGBA

    def joint_identifiers(self) -> tuple[str, ...]:
        return POLICY_JOINT_NAMES

    def reset(self, q_policy_order: npt.ArrayLike | None = None) -> None:
        import mujoco

        if q_policy_order is None:
            q_policy_order = ISAAC_PREGRASP_JOINT_POSITIONS_RAD
        requested = _finite_vector(q_policy_order, "reset q")
        lower = REAL_HAND_FACTORY_LIMITS[:, 0]
        upper = REAL_HAND_FACTORY_LIMITS[:, 1]
        applied = np.clip(requested, lower, upper)
        self.last_reset_clamped = np.not_equal(requested, applied)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.mapping.policy_to_mujoco_qpos] = applied
        self.data.qvel[self.mapping.policy_to_mujoco_dof] = 0.0
        # A reset may contain physically valid negative Joint4 q.  The initial
        # commanded hold must still obey the narrower policy command contract.
        initial_target = np.clip(
            applied, COMMAND_TARGET_LIMITS[:, 0], COMMAND_TARGET_LIMITS[:, 1]
        )
        mujoco.mj_forward(self.model, self.data)
        palm_position = self.data.xpos[self._palm_body_id].copy()
        palm_rotation = self.data.xmat[self._palm_body_id].reshape(3, 3).copy()
        palm_quaternion = self.data.xquat[self._palm_body_id].copy()
        for stick_index, joint_id in enumerate(self._stick_joint_ids):
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            pose_p = MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ[stick_index]
            self.data.qpos[qpos_address:qpos_address + 3] = palm_position + palm_rotation @ pose_p[:3]
            self.data.qpos[qpos_address + 3:qpos_address + 7] = quaternion_multiply_wxyz(
                palm_quaternion, pose_p[3:]
            )
            dof_address = int(self.model.jnt_dofadr[joint_id])
            self.data.qvel[dof_address:dof_address + 6] = 0.0
        self.write_joint_position_targets(initial_target)
        mujoco.mj_forward(self.model, self.data)
        self.physics_step_count = 0

    def read_joint_positions(self) -> npt.NDArray[np.float32]:
        result = np.asarray(
            self.data.qpos[self.mapping.policy_to_mujoco_qpos], dtype=np.float32
        ).copy()
        if not np.isfinite(result).all():
            raise RuntimeError("MuJoCo returned non-finite joint positions.")
        return result

    def write_joint_position_targets(self, targets_policy_order: npt.ArrayLike) -> None:
        targets = _finite_vector(targets_policy_order, "position targets")
        lower = COMMAND_TARGET_LIMITS[:, 0]
        upper = COMMAND_TARGET_LIMITS[:, 1]
        if np.any(targets < lower) or np.any(targets > upper):
            bad = np.flatnonzero((targets < lower) | (targets > upper)).tolist()
            raise ValueError(f"Backend received targets outside COMMAND_TARGET_LIMITS: {bad}")
        self.data.ctrl[self.mapping.policy_to_mujoco_actuator] = targets

    def get_fingertip_positions_in_palm(self) -> npt.NDArray[np.float32]:
        palm_position = self.data.xpos[self._palm_body_id]
        palm_rotation = self.data.xmat[self._palm_body_id].reshape(3, 3)
        site_world = self.data.site_xpos[self._tip_site_ids]
        result = ((site_world - palm_position) @ palm_rotation).astype(np.float32).reshape(15)
        if not np.isfinite(result).all():
            raise RuntimeError("MuJoCo returned non-finite fingertip positions.")
        return result

    def get_stick_poses_in_palm(self) -> npt.NDArray[np.float32]:
        """Return both stick poses in the palm frame; grasp scene only."""

        if not self.has_sticks:
            raise RuntimeError(
                "get_stick_poses_in_palm() requires the grasp scene; this model has no sticks."
            )
        result = np.empty((2, 7), dtype=np.float32)
        palm_position = self.data.xpos[self._palm_body_id]
        palm_rotation = self.data.xmat[self._palm_body_id].reshape(3, 3)
        inverse_palm_quaternion = self.data.xquat[self._palm_body_id].copy()
        inverse_palm_quaternion[1:] *= -1.0
        for index, body_id in enumerate(self._stick_body_ids):
            result[index, :3] = palm_rotation.T @ (self.data.xpos[body_id] - palm_position)
            result[index, 3:] = quaternion_multiply_wxyz(
                inverse_palm_quaternion, self.data.xquat[body_id]
            )
        return result

    def set_stick_poses_in_palm(self, poses: npt.ArrayLike) -> None:
        """Set free-stick state for MuJoCo scene/vision validation only."""
        if not self.has_sticks:
            raise RuntimeError(
                "set_stick_poses_in_palm() requires the grasp scene; this model has no sticks."
            )

        import mujoco

        values = np.asarray(poses, dtype=np.float64)
        if values.shape != (2, 7) or not np.isfinite(values).all():
            raise ValueError("Stick poses must be a finite (2, 7) xyz+wxyz array.")
        palm_position = self.data.xpos[self._palm_body_id].copy()
        palm_rotation = self.data.xmat[self._palm_body_id].reshape(3, 3).copy()
        palm_quaternion = self.data.xquat[self._palm_body_id].copy()
        for index, joint_id in enumerate(self._stick_joint_ids):
            norm = float(np.linalg.norm(values[index, 3:]))
            if norm < 1.0e-12:
                raise ValueError(f"Stick {index + 1} quaternion norm is zero.")
            quaternion_p = values[index, 3:] / norm
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qpos_address:qpos_address + 3] = (
                palm_position + palm_rotation @ values[index, :3]
            )
            self.data.qpos[qpos_address + 3:qpos_address + 7] = quaternion_multiply_wxyz(
                palm_quaternion, quaternion_p
            )
            dof_address = int(self.model.jnt_dofadr[joint_id])
            self.data.qvel[dof_address:dof_address + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, count: int = 1) -> None:
        import mujoco

        if count < 1:
            raise ValueError("Physics step count must be positive.")
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
            self.physics_step_count += 1
        self.assert_finite()
        self.assert_within_physical_limits()

    def control_snapshot(self) -> npt.NDArray[np.float64]:
        """Return policy-ordered MuJoCo controls for scheduler diagnostics."""

        return self.data.ctrl[self.mapping.policy_to_mujoco_actuator].copy()

    def health(self) -> BackendHealth:
        if self.safe_stopped:
            return BackendHealth(
                False, f"safe stop is latched: {self.safe_stop_reason}", True
            )
        try:
            self.assert_finite()
        except RuntimeError as exc:
            return BackendHealth(False, str(exc), True)
        return BackendHealth(True, "MuJoCo state finite", True)

    def safe_stop(self, reason: str = "unspecified") -> None:
        """Freeze the present command.  Deliberately changes nothing.

        ``data.ctrl`` persists until something overwrites it, so stopping is
        exactly "stop writing".  This used to re-latch the target to the
        measured q, which looks equivalent and is not: ``q_target - q`` IS the
        grip preload, so moving the target onto the joints releases the force
        holding the sticks.  The real backend must not do that either, and the
        two must mean the same thing for sim-to-sim to be worth anything.
        """

        self.safe_stopped = True
        self.safe_stop_reason = str(reason)

    # Compatibility names used by the existing CLI.
    read_joint_positions_policy_order = read_joint_positions
    apply_position_targets = write_joint_position_targets
    fingertip_positions_in_palm = get_fingertip_positions_in_palm

    def assert_finite(self) -> None:
        if not (
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(self.data.ctrl).all()
        ):
            raise RuntimeError("MuJoCo state or control became non-finite.")

    def assert_within_physical_limits(self) -> None:
        q = self.read_joint_positions()
        lower = REAL_HAND_FACTORY_LIMITS[:, 0]
        upper = REAL_HAND_FACTORY_LIMITS[:, 1]
        violation = np.maximum(lower - q, q - upper)
        if float(np.max(violation)) > self.physical_limit_tolerance_rad:
            bad = np.flatnonzero(violation > self.physical_limit_tolerance_rad).tolist()
            raise RuntimeError(f"Physical joint-limit violation beyond tolerance: {bad}")

    def model_summary(self) -> str:
        kp_min, kp_max, kd_min, kd_max = self.gain_summary()
        return (
            f"model: {self.model_path}\n"
            f"wuji-description: {WUJI_DESCRIPTION_REVISION}\n"
            f"nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}, "
            f"njnt={self.model.njnt}, nbody={self.model.nbody}, nsite={self.model.nsite}\n"
            f"fixed_base=True, timestep={self.model.opt.timestep:.10f}s, "
            f"physics_hz={1.0 / self.model.opt.timestep:.1f}, "
            f"integrator={self.integrator}, substeps/policy_step={self.physics_substeps}\n"
            f"scene: official fixed Wuji Hand 1"
            + (", two free 10 g sticks, D435 camera" if self.has_sticks else " only (reach diagnostic)")
            + "; no Indy7"
            f"\ncontroller gains: {self.controller_gains} "
            f"(Kp {kp_min:.4g}~{kp_max:.4g}, Kd {kd_min:.4g}~{kd_max:.4g})"
            f"\nIsaac pregrasp reset clamp indices: "
            f"{np.flatnonzero(self.last_reset_clamped).tolist()}"
        )

    def gain_summary(self) -> tuple[float, float, float, float]:
        """Return (kp_min, kp_max, kd_min, kd_max) as actually compiled."""

        actuator_ids = self.mapping.policy_to_mujoco_actuator
        kp = self.model.actuator_gainprm[actuator_ids, 0]
        kd = -self.model.actuator_biasprm[actuator_ids, 2]
        return float(kp.min()), float(kp.max()), float(kd.min()), float(kd.max())

    def _assert_fixed_base(self) -> None:
        import mujoco

        unexpected = [
            index for index in range(self.model.njnt)
            if int(self.model.jnt_type[index]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_FREE)
            )
        ]
        hinge_count = int(np.sum(self.model.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE)))
        free_count = int(np.sum(self.model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE)))
        # Two scenes are supported and they must be unambiguous: the grasp scene
        # carries exactly two free sticks, the reach scene carries none.
        if unexpected or hinge_count != ACTION_DIM or free_count not in (0, 2):
            raise RuntimeError(
                "Model must be the fixed 20-hinge Hand 1 with either two free "
                f"sticks or none: unexpected={unexpected}, hinge={hinge_count}, "
                f"free={free_count}."
            )
        self.has_sticks = free_count == 2
        if self.model.nu != ACTION_DIM:
            raise RuntimeError(f"Expected 20 actuators, got {self.model.nu}.")

    def _apply_physical_limits(self) -> None:
        """Verify the MJCF's provenance, then widen it to this hand's real ROM.

        The pinned MJCF ships the vendor description's ranges, which are
        narrower than the connected hand on all 20 joints.  Leaving them would
        make PhysX-vs-MuJoCo disagree at the stops and would clamp commands the
        contract now allows, so the compiled ranges are replaced in memory with
        ``REAL_HAND_FACTORY_LIMITS``.  The source file is untouched, and the
        provenance check still fails loudly if the vendored MJCF ever changes.
        """

        joint_ids = np.empty(ACTION_DIM, dtype=np.int32)
        ranges = np.empty((ACTION_DIM, 2), dtype=np.float64)
        for policy_index, qpos_address in enumerate(self.mapping.policy_to_mujoco_qpos):
            matches = np.flatnonzero(self.model.jnt_qposadr == qpos_address)
            if matches.size != 1:
                raise RuntimeError(f"Cannot resolve joint for qpos address {qpos_address}.")
            joint_ids[policy_index] = int(matches[0])
            ranges[policy_index] = self.model.jnt_range[int(matches[0])]
        if not np.allclose(ranges, OFFICIAL_NOMINAL_PHYSICAL_LIMITS, atol=5.0e-7, rtol=0.0):
            raise RuntimeError(
                "Vendored MJCF joint ranges no longer match the pinned description; "
                "re-audit OFFICIAL_NOMINAL_PHYSICAL_LIMITS before overriding."
            )
        for policy_index, joint_id in enumerate(joint_ids):
            self.model.jnt_range[int(joint_id)] = REAL_HAND_FACTORY_LIMITS[policy_index]

    def _body_id(self, name: str) -> int:
        import mujoco
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f"Missing MuJoCo body {name!r}.")
        return body_id

    def _site_id(self, name: str) -> int:
        import mujoco
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise RuntimeError(f"Missing MuJoCo tip site {name!r}.")
        return site_id

    def _joint_id(self, name: str) -> int:
        import mujoco
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Missing MuJoCo joint {name!r}.")
        return joint_id


def _finite_vector(value: npt.ArrayLike, label: str) -> npt.NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (ACTION_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite policy-order vector of shape (20,).")
    return result


def make_finger_reach_backend(**kwargs):
    """Build the reach backend on the stickless scene.

    Lives here rather than beside the reach contract because it CHOOSES a
    backend, and ``policy/`` must not: that layer has to stay importable in an
    environment with no simulator in it.

    The grasp scene is a different environment: two free 10 g sticks, a D435
    camera and a testbed plate.  Reusing it here would let sticks fall, bounce
    and contact the hand during a probe whose whole output is a joint and
    fingertip trajectory, and the Isaac finger_reach scene parks its sticks
    2 m away for exactly that reason.  Keeping the two MuJoCo scenes separate
    makes that structural instead of something the caller has to remember.
    """

    kwargs.setdefault("model_path", FINGER_REACH_MODEL_PATH)
    # MuJoCo joint limits are soft constraints, so a policy that presses a stop
    # settles slightly past it.  The reach policy does exactly that on
    # finger3_joint3 -- it learned to sit at the upper limit -- and the measured
    # steady press is 23.4 mrad (1.34 deg).  The backend's 20 mrad default would
    # abort a physically fine rollout.  This guard exists to catch instability,
    # not to re-check the command contract, which write_joint_position_targets
    # already enforces exactly.
    kwargs.setdefault("physical_limit_tolerance_rad", 0.05)
    backend = MujocoWujiHand(**kwargs)
    if backend.has_sticks:
        raise RuntimeError(
            "The reach backend must run on a stickless scene; got a grasp scene."
        )
    return backend
