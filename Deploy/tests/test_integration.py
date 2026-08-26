# [test] MuJoCo 백엔드를 실제로 띄워서 도는 통합 테스트.
from __future__ import annotations

import unittest

import numpy as np

from Deploy.policy.action_adapter import decode_policy_action
from Deploy.common.backend_protocol import BackendHealth
from Deploy.common.fingertip_fk import (
    ISAAC_URDF,
    OFFICIAL_URDF,
    POLICY_TIP_FRAME_URDF,
    WujiHand1FingertipFK,
)
from Deploy.backends.joint_mapping import MUJOCO_JOINT_NAMES
from Deploy.backends.mujoco_scheduler import (
    MUJOCO_INTEGRATOR,
    MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP,
    MujocoScheduler,
    physics_dt_for_substeps,
    validate_hold_schedule,
)
from Deploy.backends.mujoco_wuji import DEFAULT_MODEL_PATH, TIP_SITE_NAMES, MujocoWujiHand
from Deploy.policy.observation_adapter import PolicyObservationAdapter
from Deploy.policy.policy_runner import PolicyRunner
from Deploy.common.policy_contract import (
    DEPLOY_DAMPING_NMS_PER_RAD,
    DEPLOY_EFFORT_LIMITS_NM,
    DEPLOY_STIFFNESS_NM_PER_RAD,
    ACTION_DIM,
    COMMAND_TARGET_LIMITS,
    DEFAULT_RESET_JOINT_POSITIONS,
    JOINT4_POLICY_INDICES,
    OFFICIAL_NOMINAL_PHYSICAL_LIMITS,
    REAL_HAND_FACTORY_LIMITS,
    PALM_FRAME_NAME,
    POLICY_JOINT_NAMES,
    OBSERVATION_SLICES,
)


class LocalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hand = MujocoWujiHand()

    def setUp(self) -> None:
        self.hand.reset()

    def test_official_model_topology_and_unique_name_mapping(self) -> None:
        import mujoco

        self.assertEqual(self.hand.model.nq, ACTION_DIM + 14)
        self.assertEqual(self.hand.model.nv, ACTION_DIM + 12)
        self.assertEqual(self.hand.model.nu, ACTION_DIM)
        self.assertEqual(self.hand.model.njnt, ACTION_DIM + 2)
        self.assertEqual(len(set(self.hand.mapping.policy_to_mujoco_qpos.tolist())), ACTION_DIM)
        self.assertEqual(len(set(self.hand.mapping.policy_to_mujoco_dof.tolist())), ACTION_DIM)
        self.assertEqual(len(set(self.hand.mapping.policy_to_mujoco_actuator.tolist())), ACTION_DIM)
        names = tuple(
            mujoco.mj_id2name(self.hand.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.hand.model.njnt)
        )
        self.assertEqual(names[:ACTION_DIM], MUJOCO_JOINT_NAMES)
        self.assertEqual(set(names[ACTION_DIM:]), {"stick1_free", "stick2_free"})
        self.assertFalse(any("indy" in (name or "").lower() for name in names))
        self.assertGreaterEqual(
            mujoco.mj_name2id(self.hand.model, mujoco.mjtObj.mjOBJ_BODY, PALM_FRAME_NAME), 0
        )
        for site_name in TIP_SITE_NAMES:
            self.assertGreaterEqual(
                mujoco.mj_name2id(self.hand.model, mujoco.mjtObj.mjOBJ_SITE, site_name), 0
            )
        self.assertEqual(self.hand.joint_identifiers(), POLICY_JOINT_NAMES)

    def test_isaac_tuned_mode_installs_the_hand_tuned_gains(self) -> None:
        """Only the commanded controller changes; the official plant does not."""

        import mujoco

        pristine_path = DEFAULT_MODEL_PATH.with_name("right.xml")
        pristine = mujoco.MjModel.from_xml_path(str(pristine_path))
        self.hand = MujocoWujiHand(controller_gains="isaac_tuned")
        actuator_ids = self.hand.mapping.policy_to_mujoco_actuator
        dof_ids = self.hand.mapping.policy_to_mujoco_dof

        # Mass distribution stays exactly the vendor model in every mode.
        np.testing.assert_allclose(self.hand.model.dof_armature[dof_ids], pristine.dof_armature)
        np.testing.assert_allclose(self.hand.model.dof_damping[dof_ids], pristine.dof_damping)
        np.testing.assert_allclose(
            self.hand.model.actuator_ctrlrange[actuator_ids], COMMAND_TARGET_LIMITS
        )

        # Default deploy mode installs the Isaac-tuned gains the policy trained
        # against, in the compiled <position> representation.
        self.assertEqual(self.hand.controller_gains, "isaac_tuned")
        np.testing.assert_allclose(
            self.hand.model.actuator_gainprm[actuator_ids, 0], DEPLOY_STIFFNESS_NM_PER_RAD
        )
        np.testing.assert_allclose(
            self.hand.model.actuator_biasprm[actuator_ids, 1], -DEPLOY_STIFFNESS_NM_PER_RAD
        )
        np.testing.assert_allclose(
            -self.hand.model.actuator_biasprm[actuator_ids, 2], DEPLOY_DAMPING_NMS_PER_RAD
        )
        np.testing.assert_allclose(
            self.hand.model.actuator_forcerange[actuator_ids, 1], DEPLOY_EFFORT_LIMITS_NM
        )
        # Effort limits are the official URDF values, so this one table must not
        # have moved away from the vendor model.
        np.testing.assert_allclose(
            self.hand.model.actuator_forcerange[actuator_ids],
            pristine.actuator_forcerange[actuator_ids],
        )
        # The gains genuinely differ from the vendor identification, otherwise
        # this test would be silently vacuous.
        self.assertFalse(
            np.allclose(
                self.hand.model.actuator_gainprm[actuator_ids, 0],
                pristine.actuator_gainprm[actuator_ids, 0],
            )
        )

    def test_official_gain_mode_restores_the_vendor_identification(self) -> None:
        import mujoco

        pristine_path = DEFAULT_MODEL_PATH.with_name("right.xml")
        pristine = mujoco.MjModel.from_xml_path(str(pristine_path))
        vendor = MujocoWujiHand()  # vendor gains are now the default
        actuator_ids = vendor.mapping.policy_to_mujoco_actuator
        np.testing.assert_allclose(
            vendor.model.actuator_gainprm[actuator_ids], pristine.actuator_gainprm[actuator_ids]
        )
        np.testing.assert_allclose(
            vendor.model.actuator_biasprm[actuator_ids], pristine.actuator_biasprm[actuator_ids]
        )
        np.testing.assert_allclose(
            vendor.model.actuator_forcerange[actuator_ids],
            pristine.actuator_forcerange[actuator_ids],
        )
        np.testing.assert_allclose(
            vendor.model.actuator_ctrlrange[actuator_ids], COMMAND_TARGET_LIMITS
        )
        with self.assertRaises(ValueError):
            MujocoWujiHand(controller_gains="nonsense")

    def test_command_validation_and_joint4_physical_command_separation(self) -> None:
        # The five Joint4 command floors were lifted on 2026-08-18 to match
        # Isaac, so a small negative Joint4 target is now legal.  What must
        # still be rejected is a target outside the articulation limits.
        q = DEFAULT_RESET_JOINT_POSITIONS.copy()
        q[JOINT4_POLICY_INDICES] = -0.005
        self.hand.reset(q)
        np.testing.assert_allclose(
            self.hand.read_joint_positions()[JOINT4_POLICY_INDICES], -0.005, atol=1e-7
        )
        self.hand.write_joint_position_targets(q)
        decoded = decode_policy_action(q, np.zeros(ACTION_DIM, dtype=np.float32))
        np.testing.assert_allclose(
            decoded.position_target[JOINT4_POLICY_INDICES], -0.005, atol=1e-7
        )
        self.hand.write_joint_position_targets(decoded.position_target)

        beyond = q.copy()
        beyond[JOINT4_POLICY_INDICES] = (
            COMMAND_TARGET_LIMITS[JOINT4_POLICY_INDICES, 0] - 0.05
        )
        with self.assertRaisesRegex(ValueError, "COMMAND_TARGET_LIMITS"):
            self.hand.write_joint_position_targets(beyond)

    def test_explicit_sites_match_standalone_official_urdf_fk(self) -> None:
        # Model consistency only: MuJoCo loads the vendor description, so its
        # sites must agree with that file's FK.  This is NOT the policy's tip
        # contract -- see test_policy_observation_uses_isaac_tip_frames.
        fk = WujiHand1FingertipFK(OFFICIAL_URDF)
        lower = OFFICIAL_NOMINAL_PHYSICAL_LIMITS[:, 0]
        upper = OFFICIAL_NOMINAL_PHYSICAL_LIMITS[:, 1]
        for fraction in (0.1, 0.5, 0.9):
            q = lower + np.float32(fraction) * (upper - lower)
            self.hand.reset(q)
            np.testing.assert_allclose(
                self.hand.get_fingertip_positions_in_palm(),
                fk.fingertip_positions_in_palm(q),
                atol=2e-6,
                rtol=0.0,
            )

    def test_observation_and_exact_four_step_target_hold(self) -> None:
        adapter = PolicyObservationAdapter(mode="open")
        adapter.reset(self.hand.read_joint_positions())
        observation = adapter.build()
        self.assertEqual(observation.shape, (105,))
        self.assertEqual(observation.dtype, np.float32)

        decoded = decode_policy_action(
            self.hand.read_joint_positions(), np.zeros(ACTION_DIM, dtype=np.float32)
        )
        self.hand.write_joint_position_targets(decoded.position_target)
        held = self.hand.data.ctrl.copy()
        start = self.hand.physics_step_count
        scheduler = MujocoScheduler(self.hand)
        scheduler.hold_policy_target()
        np.testing.assert_array_equal(self.hand.data.ctrl, held)
        self.assertEqual(self.hand.physics_step_count - start, self.hand.physics_substeps)
        self.assertTrue(self.hand.health().ok)

    def test_hold_duration_is_the_contract_not_the_substep_count(self) -> None:
        """Any substep count is valid as long as the hold spans one policy step."""

        from Deploy.common.policy_contract import POLICY_DT

        for substeps in (4, 8, 16, 30):
            hand = MujocoWujiHand(physics_substeps=substeps)
            self.assertEqual(hand.physics_substeps, substeps)
            self.assertAlmostEqual(
                hand.model.opt.timestep * substeps, POLICY_DT, places=12
            )
            scheduler = MujocoScheduler(hand)
            self.assertEqual(scheduler.substeps, substeps)
            start = hand.physics_step_count
            scheduler.hold_policy_target()
            self.assertEqual(hand.physics_step_count - start, substeps)
        with self.assertRaises(ValueError):
            MujocoWujiHand(physics_substeps=0)
        with self.assertRaises(ValueError):
            MujocoWujiHand(integrator="verlet")
        with self.assertRaises(RuntimeError):
            validate_hold_schedule(physics_dt_for_substeps(16), 4)

    def test_default_numerics_are_implicitfast_and_self_consistent(self) -> None:
        """Assert the defaults agree with each other, not a magic frequency."""

        import mujoco

        from Deploy.common.policy_contract import POLICY_DT

        self.assertEqual(self.hand.integrator, MUJOCO_INTEGRATOR)
        self.assertEqual(
            int(self.hand.model.opt.integrator), int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        )
        self.assertEqual(self.hand.physics_substeps, MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP)
        self.assertAlmostEqual(
            self.hand.model.opt.timestep,
            physics_dt_for_substeps(MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP),
            places=12,
        )
        # The vendor MJCF declares rk4 at 2 ms; the backend must have overridden
        # both, otherwise the convergence study's premise no longer holds.
        pristine = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL_PATH.with_name("right.xml")))
        self.assertNotEqual(int(self.hand.model.opt.integrator), int(pristine.opt.integrator))
        self.assertLess(self.hand.model.opt.timestep, POLICY_DT / 4.0)

    def test_backend_neutral_policy_runner_and_mode(self) -> None:
        class ZeroPolicy:
            def infer(self, observation):
                self.observation = np.asarray(observation).copy()
                return np.zeros(ACTION_DIM, dtype=np.float32)

        policy = ZeroPolicy()
        runner = PolicyRunner(self.hand, policy, PolicyObservationAdapter())
        reset_observation = runner.reset()
        runner.set_mode("close")
        decoded = runner.command()
        MujocoScheduler(self.hand).hold_policy_target()
        next_observation = runner.observe_after_hold()
        self.assertEqual(reset_observation.shape, (105,))
        np.testing.assert_array_equal(policy.observation[-2:], [0.0, 1.0])
        np.testing.assert_array_equal(
            next_observation[OBSERVATION_SLICES["last_action"].slice],
            decoded.action_manager_action,
        )

    def test_fake_and_mujoco_backend_share_common_action_semantics(self) -> None:
        class FixedPolicy:
            def __init__(self, action):
                self.action = action

            def infer(self, observation):
                return self.action.copy()

        class FakeBackend:
            def __init__(self, q, tips):
                self.q = q.copy()
                self.tips = tips.copy()
                self.target = None

            def joint_identifiers(self): return POLICY_JOINT_NAMES
            def read_joint_positions(self): return self.q.copy()
            def write_joint_position_targets(self, q): self.target = np.asarray(q).copy()
            def health(self): return BackendHealth(True, "fake healthy", True)
            def safe_stop(self, reason=""): pass

        q = DEFAULT_RESET_JOINT_POSITIONS.copy()
        self.hand.reset(q)
        tips = self.hand.get_fingertip_positions_in_palm()
        action = np.linspace(-1.5, 1.5, ACTION_DIM, dtype=np.float32)
        fake = FakeBackend(q, tips)
        mujoco_runner = PolicyRunner(
            self.hand, FixedPolicy(action), PolicyObservationAdapter()
        )
        fake_runner = PolicyRunner(fake, FixedPolicy(action), PolicyObservationAdapter())
        mujoco_runner.reset()
        fake_runner.reset()
        mujoco_decoded = mujoco_runner.command()
        fake_decoded = fake_runner.command()
        np.testing.assert_array_equal(
            mujoco_decoded.action_manager_action, fake_decoded.action_manager_action
        )
        np.testing.assert_array_equal(
            mujoco_decoded.position_target, fake_decoded.position_target
        )
        np.testing.assert_array_equal(fake.target, mujoco_decoded.position_target)

    def test_reach_and_grasp_scenes_are_separate_environments(self) -> None:
        """A reach probe must not run in the scene that contains the sticks."""

        import mujoco

        from Deploy.policy.finger_reach import (
            FINGER_REACH_RESET_JOINT_POSITIONS,
        )
        from Deploy.backends.mujoco_wuji import FINGER_REACH_MODEL_PATH

        from Deploy.backends.mujoco_wuji import make_finger_reach_backend

        reach = make_finger_reach_backend()
        self.assertFalse(reach.has_sticks)
        self.assertEqual(reach.model_path.name, FINGER_REACH_MODEL_PATH.name)
        # No free joints at all: nothing in the scene can fall, roll or touch
        # the hand while a trajectory is being recorded.
        self.assertEqual(reach.model.nq, 20)
        self.assertEqual(reach.model.nv, 20)
        for name in ("stick1", "stick2"):
            self.assertLess(
                mujoco.mj_name2id(reach.model, mujoco.mjtObj.mjOBJ_BODY, name), 0
            )
        # The tip sites the observation and the logs rely on survive.
        for site_name in TIP_SITE_NAMES:
            self.assertGreaterEqual(
                mujoco.mj_name2id(reach.model, mujoco.mjtObj.mjOBJ_SITE, site_name), 0
            )
        # Stick accessors fail loudly instead of returning something meaningless.
        with self.assertRaisesRegex(RuntimeError, "no sticks"):
            reach.get_stick_poses_in_palm()

        # The grasp scene keeps its sticks, and the two scenes agree on the hand.
        grasp = MujocoWujiHand()
        self.assertTrue(grasp.has_sticks)
        # Same hand in both scenes: identical joint ranges in policy order.
        np.testing.assert_allclose(
            reach.model.jnt_range[reach.mapping.policy_to_mujoco_qpos],
            grasp.model.jnt_range[grasp.mapping.policy_to_mujoco_qpos],
        )

        # Reach resets to the OPEN hand, not the curled grasp pregrasp.
        reach.reset(FINGER_REACH_RESET_JOINT_POSITIONS)
        np.testing.assert_allclose(
            reach.read_joint_positions(), FINGER_REACH_RESET_JOINT_POSITIONS, atol=1e-6
        )
        self.assertGreater(
            float(np.abs(grasp.read_joint_positions() - FINGER_REACH_RESET_JOINT_POSITIONS).max()),
            1.5,
        )


if __name__ == "__main__":
    unittest.main()
