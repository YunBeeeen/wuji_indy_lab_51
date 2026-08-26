from __future__ import annotations

import re
import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np

from Deploy.policy.action_adapter import decode_policy_action
from Deploy.common.backend_protocol import BackendHealth
from Deploy.common.fingertip_fk import (
    ISAAC_URDF,
    OFFICIAL_URDF,
    POLICY_TIP_FRAME_URDF,
    UrdfSource,
    WujiHand1FingertipFK,
)
from Deploy.policy.observation_adapter import PolicyObservationAdapter
from Deploy.common.perception import PoseState, StickPosePair7D
from Deploy.policy.policy_runner import PolicyRunner
from Deploy.common.policy_contract import (
    ACTION_DIM,
    ACTION_SCALE_RAD,
    COMMAND_TARGET_LIMITS,
    JOINT4_POLICY_INDICES,
    OBSERVATION_DIM,
    OBSERVATION_NORMALIZATION_LIMITS,
    OBSERVATION_SLICES,
    OFFICIAL_NOMINAL_PHYSICAL_LIMITS,
    mode_one_hot,
    normalize_joint_positions,
    observation_csv_columns,
    validate_factory_limits,
)
from Deploy.backends.real_wuji_backend import (
    PENDING_REAL_VALIDATION,
    VERIFIED_ON_HARDWARE,
    pending_validation_report,
)


class SequenceProvider:
    representation = "StickPose7D"

    def __init__(self) -> None:
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def sample(self) -> StickPosePair7D:
        shift = np.float32(self.index * 0.001)
        self.index += 1
        return StickPosePair7D(
            np.asarray([shift, 0, 0, 1, 0, 0, 0], dtype=np.float32),
            np.asarray([0, shift, 0, 1, 0, 0, 0], dtype=np.float32),
            float(self.index),
        )


class FakeBackend:
    def __init__(self, q: np.ndarray) -> None:
        self.q = np.asarray(q, dtype=np.float32).copy()
        self.tips = np.linspace(0.0, 0.014, 15, dtype=np.float32)
        self.written_target = None

    def joint_identifiers(self):
        return tuple(f"finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5))

    def read_joint_positions(self):
        return self.q.copy()

    def write_joint_position_targets(self, target):
        self.written_target = np.asarray(target, dtype=np.float32).copy()

    def get_fingertip_positions_in_palm(self):
        return self.tips.copy()

    def health(self):
        return BackendHealth(True, "fake healthy", True)

    def safe_stop(self, reason: str = "unspecified"):
        pass


class PolicyContractTests(unittest.TestCase):
    def test_fake_backend_runs_complete_common_policy_step(self) -> None:
        class FakePolicy:
            def infer(self, observation):
                self.observation = np.asarray(observation).copy()
                action = np.zeros(ACTION_DIM, dtype=np.float32)
                action[0] = 2.0
                action[3] = -1.0
                return action

        q = (
            OBSERVATION_NORMALIZATION_LIMITS[:, 0]
            + OBSERVATION_NORMALIZATION_LIMITS[:, 1]
        ) / np.float32(2.0)
        q[3] = -0.005
        backend = FakeBackend(q)
        policy = FakePolicy()
        runner = PolicyRunner(
            backend,
            policy,
            PolicyObservationAdapter(stick_provider=SequenceProvider()),
        )
        self.assertEqual(runner.reset().shape, (105,))
        decoded = runner.command()
        np.testing.assert_array_equal(backend.written_target, decoded.position_target)
        # Until 2026-08-18 this floored at 0.0.  The Joint4 command floors are
        # lifted, so a negative residual from q=-0.005 now reaches the target
        # unclamped and only the articulation lower limit bounds it.
        self.assertLess(float(backend.written_target[3]), 0.0)
        self.assertGreaterEqual(
            float(backend.written_target[3]),
            float(COMMAND_TARGET_LIMITS[3, 0]),
        )
        backend.q = backend.written_target.copy()
        observation = runner.observe_after_hold()
        self.assertEqual(observation.shape, (105,))
        np.testing.assert_array_equal(
            observation[OBSERVATION_SLICES["last_action"].slice], decoded.action_manager_action
        )

    def test_stale_perception_blocks_policy_command_and_safe_stops(self) -> None:
        class StaleProvider(SequenceProvider):
            def sample(self):
                sample = super().sample()
                return StickPosePair7D(
                    sample.stick1, sample.stick2, sample.timestamp_s, PoseState.STALE, False
                )

        class PolicyThatMustNotRun:
            def infer(self, observation):
                raise AssertionError("Policy inference must not run on stale perception")

        backend = FakeBackend(np.zeros(ACTION_DIM, dtype=np.float32))
        backend.stopped = None
        backend.safe_stop = lambda reason="": setattr(backend, "stopped", reason)
        runner = PolicyRunner(
            backend,
            PolicyThatMustNotRun(),
            PolicyObservationAdapter(stick_provider=StaleProvider()),
        )
        runner.reset()
        with self.assertRaisesRegex(RuntimeError, "stale perception"):
            runner.command()
        # The backend is told WHY, so a frozen hand can be explained from the
        # log without correlating timestamps against the tracker's output.
        self.assertEqual(backend.stopped, "stale perception")

    def test_common_imports_and_fake_tick_without_mujoco(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = r'''import sys
sys.modules["mujoco"] = None
import numpy as np
from Deploy.common.policy_contract import ACTION_DIM, OBSERVATION_NORMALIZATION_LIMITS
from Deploy.policy.observation_adapter import PolicyObservationAdapter
from Deploy.policy.action_adapter import decode_policy_action
from Deploy.common.fingertip_fk import WujiHand1FingertipFK
from Deploy.policy.policy_runner import PolicyRunner
from Deploy.common.backend_protocol import BackendHealth
class Backend:
    def __init__(self):
        limits = OBSERVATION_NORMALIZATION_LIMITS
        self.q = ((limits[:, 0] + limits[:, 1]) / 2).astype(np.float32)
        self.target = None
    def joint_identifiers(self): return tuple(str(i) for i in range(ACTION_DIM))
    def read_joint_positions(self): return self.q.copy()
    def write_joint_position_targets(self, q): self.target = np.asarray(q).copy()
    def get_fingertip_positions_in_palm(self): return np.zeros(15, np.float32)
    def health(self): return BackendHealth(True, "ok", True)
    def safe_stop(self, reason=""): pass
class Policy:
    def infer(self, observation): return np.zeros(ACTION_DIM, np.float32)
backend = Backend()
runner = PolicyRunner(backend, Policy(), PolicyObservationAdapter())
assert runner.reset().shape == (105,)
runner.command()
assert backend.target.shape == (20,)
runner.observe_after_hold()
assert sys.modules["mujoco"] is None
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_backend_files_do_not_duplicate_policy_semantics(self) -> None:
        package = Path(__file__).resolve().parents[1]
        source = "\n".join(
            (package / name).read_text(encoding="utf-8")
            for name in ("backends/mujoco_wuji.py", "backends/real_wuji_backend.py")
        )
        for forbidden in (
            "normalize_joint_positions",
            "decode_policy_action",
            "ACTION_SCALE_RAD",
            "ACTION_CLIP",
        ):
            self.assertNotIn(forbidden, source)

    def test_three_limit_tables_are_explicit_and_independent(self) -> None:
        for limits in (
            OFFICIAL_NOMINAL_PHYSICAL_LIMITS,
            OBSERVATION_NORMALIZATION_LIMITS,
            COMMAND_TARGET_LIMITS,
        ):
            self.assertEqual(limits.shape, (ACTION_DIM, 2))
            self.assertFalse(limits.flags.writeable)
        self.assertFalse(np.shares_memory(
            OFFICIAL_NOMINAL_PHYSICAL_LIMITS, OBSERVATION_NORMALIZATION_LIMITS
        ))
        self.assertFalse(np.shares_memory(
            OBSERVATION_NORMALIZATION_LIMITS, COMMAND_TARGET_LIMITS
        ))
        self.assertTrue(np.all(
            OFFICIAL_NOMINAL_PHYSICAL_LIMITS[JOINT4_POLICY_INDICES, 0] < 0.0
        ))
        # The five Joint4 command floors were lifted on 2026-08-18 to match
        # Isaac.  The two tables were equal until 2026-08-22, when
        # COMMAND_LIMIT_RATIO narrowed the command range for hardware safety.
        # Normalization must NOT follow: it defines what the network's inputs
        # mean, and rescaling it would feed the policy unseen numbers.
        from Deploy.common.policy_contract import (
            COMMAND_LIMIT_RATIO, REAL_HAND_FACTORY_LIMITS,
        )

        np.testing.assert_array_equal(
            OBSERVATION_NORMALIZATION_LIMITS, REAL_HAND_FACTORY_LIMITS
        )
        np.testing.assert_allclose(
            COMMAND_TARGET_LIMITS,
            REAL_HAND_FACTORY_LIMITS * COMMAND_LIMIT_RATIO, atol=1e-7,
        )
        # Containment, both sides.  `limit * ratio` only shrinks a range that
        # straddles zero -- assert the precondition rather than trusting it.
        self.assertTrue(np.all(REAL_HAND_FACTORY_LIMITS[:, 0] <= 0.0))
        self.assertTrue(np.all(REAL_HAND_FACTORY_LIMITS[:, 1] >= 0.0))
        self.assertTrue(np.all(
            COMMAND_TARGET_LIMITS[:, 0] >= OBSERVATION_NORMALIZATION_LIMITS[:, 0]))
        self.assertTrue(np.all(
            COMMAND_TARGET_LIMITS[:, 1] <= OBSERVATION_NORMALIZATION_LIMITS[:, 1]))
        self.assertTrue(np.all(COMMAND_TARGET_LIMITS[JOINT4_POLICY_INDICES, 0] < 0.0))

    def test_limit_normalization_endpoints_center_and_no_clip(self) -> None:
        lower = OBSERVATION_NORMALIZATION_LIMITS[:, 0]
        upper = OBSERVATION_NORMALIZATION_LIMITS[:, 1]
        center = (lower + upper) / np.float32(2.0)
        np.testing.assert_allclose(normalize_joint_positions(lower), -1.0, atol=2e-7)
        np.testing.assert_allclose(normalize_joint_positions(center), 0.0, atol=2e-7)
        np.testing.assert_allclose(normalize_joint_positions(upper), 1.0, atol=2e-7)
        outside = upper + np.float32(0.1) * (upper - lower)
        self.assertTrue(np.all(normalize_joint_positions(outside) > 1.0))

    def test_history_reset_advance_and_105d_layout(self) -> None:
        adapter = PolicyObservationAdapter(mode="open", stick_provider=SequenceProvider())
        lower = OBSERVATION_NORMALIZATION_LIMITS[:, 0]
        upper = OBSERVATION_NORMALIZATION_LIMITS[:, 1]
        q0 = (lower + upper) / np.float32(2.0)
        q1 = q0 + np.float32(0.01)
        adapter.reset(q0)
        reset_obs = adapter.build()
        np.testing.assert_array_equal(
            reset_obs[OBSERVATION_SLICES["joint_previous"].slice],
            reset_obs[OBSERVATION_SLICES["joint_current"].slice],
        )
        np.testing.assert_array_equal(
            reset_obs[OBSERVATION_SLICES["stick1_previous"].slice],
            reset_obs[OBSERVATION_SLICES["stick1_current"].slice],
        )

        action = np.linspace(-1.0, 1.0, ACTION_DIM, dtype=np.float32)
        adapter.advance(q1, action)
        obs = adapter.build()
        np.testing.assert_allclose(
            obs[OBSERVATION_SLICES["joint_previous"].slice], normalize_joint_positions(q0)
        )
        np.testing.assert_allclose(
            obs[OBSERVATION_SLICES["joint_current"].slice], normalize_joint_positions(q1)
        )
        self.assertNotEqual(
            float(obs[OBSERVATION_SLICES["stick1_previous"].start]),
            float(obs[OBSERVATION_SLICES["stick1_current"].start]),
        )
        np.testing.assert_array_equal(obs[OBSERVATION_SLICES["last_action"].slice], action)
        self.assertEqual(obs.shape, (OBSERVATION_DIM,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(obs.flags.c_contiguous)
        self.assertTrue(np.isfinite(obs).all())

    def test_modes(self) -> None:
        np.testing.assert_array_equal(mode_one_hot("open"), [1.0, 0.0])
        np.testing.assert_array_equal(mode_one_hot("close"), [0.0, 1.0])

    def test_action_uses_actual_radians_clip_residual_and_command_floor(self) -> None:
        q = ((OBSERVATION_NORMALIZATION_LIMITS[:, 0] +
              OBSERVATION_NORMALIZATION_LIMITS[:, 1]) / np.float32(2.0)).copy()
        q[3] = np.float32(-0.005)  # physically observable, but not commandable
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[0] = 2.0
        action[3] = -0.5
        decoded = decode_policy_action(q, action)
        self.assertEqual(decoded.action_manager_action[0], np.float32(1.0))
        self.assertAlmostEqual(
            float(decoded.unclamped_target[0]),
            float(q[0] + ACTION_SCALE_RAD[0]),
            places=6,
        )
        # Joint4 uses the 0.15 rad step, not the Joint1/Joint2 0.1 rad step.
        self.assertAlmostEqual(float(decoded.unclamped_target[3]), -0.08, places=6)
        # -0.08 rad used to clamp to the 0.0 floor; the floors are lifted and
        # the articulation lower limit is far below, so it passes through.
        self.assertAlmostEqual(float(decoded.position_target[3]), -0.08, places=6)
        self.assertTrue(decoded.action_was_clipped[0])
        self.assertFalse(decoded.target_was_clamped[3])

    def test_action_scale_is_per_joint_and_matches_isaac(self) -> None:
        """Isaac hand_real retuned Joint3/Joint4 against larger PD steps."""
        self.assertEqual(ACTION_SCALE_RAD.shape, (ACTION_DIM,))
        self.assertFalse(ACTION_SCALE_RAD.flags.writeable)
        expected = np.asarray(
            [0.1, 0.1, 0.2, 0.15] * 5,
            dtype=np.float32,
        )
        np.testing.assert_allclose(ACTION_SCALE_RAD, expected)

        # A full-scale action must move each joint by exactly its own scale.
        q = np.zeros(ACTION_DIM, dtype=np.float32)
        decoded = decode_policy_action(q, np.ones(ACTION_DIM, dtype=np.float32))
        np.testing.assert_allclose(decoded.unclamped_target, expected, atol=1e-6)

    def test_factory_limits_are_compared_not_adopted(self) -> None:
        before = OBSERVATION_NORMALIZATION_LIMITS.copy()
        result = validate_factory_limits(
            OFFICIAL_NOMINAL_PHYSICAL_LIMITS[:, 0] + 0.001,
            OFFICIAL_NOMINAL_PHYSICAL_LIMITS[:, 1] - 0.001,
        )
        self.assertEqual(result["lower_minus_nominal"].shape, (20,))
        np.testing.assert_array_equal(OBSERVATION_NORMALIZATION_LIMITS, before)

    def test_real_backend_states_what_is_and_is_not_measured(self) -> None:
        # The blanket refusal this replaces was correct when nothing had been
        # driven.  finger_reach has since driven four joints closed-loop, so the
        # honest form is an itemized record rather than a locked door -- and the
        # unmeasured items must stay named.
        self.assertTrue(VERIFIED_ON_HARDWARE)
        self.assertTrue(PENDING_REAL_VALIDATION)
        report = pending_validation_report()
        self.assertIn("[VERIFIED]", report)
        self.assertIn("[NOT MEASURED]", report)
        self.assertIn("watchdog", report)
        # The thermal incident is a hardware fact, not a note: keep it in the
        # header every twenty-joint run prints.
        self.assertIn("88.4 C", report)


if __name__ == "__main__":
    unittest.main()


class FingerReachContractTests(unittest.TestCase):
    """The 15D/4D reach contract, which must not inherit grasp-task details."""

    def test_canonical_indices_and_sdk_finger_major_grid(self) -> None:
        from Deploy.policy.finger_reach import (
            MIDDLE_JOINT_NAMES,
            MIDDLE_POLICY_INDICES,
            finger_major_grid,
            from_finger_major_grid,
        )
        from Deploy.common.policy_contract import POLICY_JOINT_NAMES

        self.assertEqual(
            [POLICY_JOINT_NAMES[i] for i in MIDDLE_POLICY_INDICES], list(MIDDLE_JOINT_NAMES)
        )
        # The vendor SDK streams 20 joints finger-major, which is the canonical
        # order, so the (5,4) view must be a reshape and never a reordering.
        flat = np.arange(20, dtype=np.float32)
        grid = finger_major_grid(flat)
        self.assertEqual(grid.shape, (5, 4))
        np.testing.assert_array_equal(grid[2], MIDDLE_POLICY_INDICES.astype(np.float32))
        np.testing.assert_array_equal(from_finger_major_grid(grid), flat)

    def test_reach_command_limits_are_the_real_hand_range(self) -> None:
        from Deploy.policy.finger_reach import (
            MIDDLE_COMMAND_TARGET_LIMITS,
            MIDDLE_POLICY_INDICES,
        )
        from Deploy.common.policy_contract import REAL_HAND_FACTORY_LIMITS

        # No distal floor anywhere: Isaac, MuJoCo and the real hand share one
        # action space, which is the connected hand's own factory range.
        # Explicitly the UNSCALED table -- the grasp task's COMMAND_LIMIT_RATIO
        # must not reach the reach probe, whose CLIs apply their own 0.95 via
        # --limit-margin.  Stacking them would make it an unannounced 0.90.
        np.testing.assert_allclose(
            MIDDLE_COMMAND_TARGET_LIMITS,
            REAL_HAND_FACTORY_LIMITS[MIDDLE_POLICY_INDICES],
        )
        from Deploy.common.policy_contract import COMMAND_TARGET_LIMITS

        self.assertFalse(np.allclose(
            MIDDLE_COMMAND_TARGET_LIMITS,
            COMMAND_TARGET_LIMITS[MIDDLE_POLICY_INDICES]),
            "reach limits must not be the ratio-scaled grasp table")
        self.assertLess(MIDDLE_COMMAND_TARGET_LIMITS[3, 0], 0.0)

    def test_action_is_clipped_residual_on_actual_radians(self) -> None:
        from Deploy.policy.finger_reach import (
            MIDDLE_COMMAND_TARGET_LIMITS,
            REACH_ACTION_SCALE_RAD,
            decode_reach_action,
        )

        q = np.asarray([0.4, 0.0, 1.0, 0.5], dtype=np.float32)
        decoded = decode_reach_action(q, np.asarray([2.0, -2.0, 0.5, -0.5], dtype=np.float32))
        np.testing.assert_allclose(decoded.clipped_action, [1.0, -1.0, 0.5, -0.5])
        np.testing.assert_allclose(
            decoded.unclamped_target, q + REACH_ACTION_SCALE_RAD * decoded.clipped_action, atol=1e-7
        )
        # The reach scale is a single 0.1, not the per-joint grasp table.
        self.assertAlmostEqual(float(REACH_ACTION_SCALE_RAD), 0.1)
        # Joint4 may extend below zero: the clamp is the hand's factory range.
        extended = decode_reach_action(
            np.asarray([0.0, 0.0, 0.0, 0.01], dtype=np.float32),
            np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float32),
        )
        self.assertLess(extended.position_target[3], 0.0)
        np.testing.assert_allclose(
            extended.position_target[3], extended.unclamped_target[3], atol=1e-7
        )
        self.assertTrue(
            np.all(decoded.position_target >= MIDDLE_COMMAND_TARGET_LIMITS[:, 0] - 1e-6)
        )

    def test_history_layout_and_reset_semantics(self) -> None:
        from Deploy.policy.finger_reach import (
            FingerReachObservationAdapter,
            REACH_OBSERVATION_SLICES,
            normalize_middle_joints,
        )

        adapter = FingerReachObservationAdapter()
        adapter.set_target(np.asarray([0.03, 0.01, 0.10], dtype=np.float32))
        q0 = np.asarray([0.4, 0.0, 1.0, 0.5], dtype=np.float32)
        adapter.reset(q0)
        observation = adapter.build()
        self.assertEqual(observation.shape, (15,))
        self.assertEqual(observation.dtype, np.float32)
        # After reset both history slots hold the reset sample.
        np.testing.assert_allclose(
            observation[REACH_OBSERVATION_SLICES["q_previous"]],
            observation[REACH_OBSERVATION_SLICES["q_current"]],
        )
        np.testing.assert_allclose(
            observation[REACH_OBSERVATION_SLICES["q_previous"]], normalize_middle_joints(q0)
        )
        np.testing.assert_allclose(observation[REACH_OBSERVATION_SLICES["target_palm"]],
                                   [0.03, 0.01, 0.10], atol=1e-7)
        np.testing.assert_allclose(observation[REACH_OBSERVATION_SLICES["last_action"]], 0.0)

        q1 = np.asarray([0.5, 0.1, 1.1, 0.6], dtype=np.float32)
        action = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        adapter.advance(q1, action)
        advanced = adapter.build()
        np.testing.assert_allclose(
            advanced[REACH_OBSERVATION_SLICES["q_previous"]], normalize_middle_joints(q0)
        )
        np.testing.assert_allclose(
            advanced[REACH_OBSERVATION_SLICES["q_current"]], normalize_middle_joints(q1)
        )
        np.testing.assert_allclose(advanced[REACH_OBSERVATION_SLICES["last_action"]], action)

    def test_normalization_matches_isaac_equation_on_the_middle_subset(self) -> None:
        from Deploy.policy.finger_reach import MIDDLE_POLICY_INDICES, normalize_middle_joints
        from Deploy.common.policy_contract import (
            OBSERVATION_NORMALIZATION_LIMITS,
            normalize_joint_positions,
        )

        q20 = np.linspace(-0.2, 1.2, 20).astype(np.float32)
        np.testing.assert_allclose(
            normalize_middle_joints(q20[MIDDLE_POLICY_INDICES]),
            normalize_joint_positions(q20)[MIDDLE_POLICY_INDICES],
            atol=1e-6,
        )
        limits = OBSERVATION_NORMALIZATION_LIMITS[MIDDLE_POLICY_INDICES]
        np.testing.assert_allclose(normalize_middle_joints(limits[:, 0]), -1.0, atol=1e-6)
        np.testing.assert_allclose(normalize_middle_joints(limits[:, 1]), 1.0, atol=1e-6)


class CommandLimitMarginTests(unittest.TestCase):
    """The 0.95 deploy margin: what it changes, and what it must not."""

    def test_soft_limits_keep_the_centre_and_shrink_the_half_range(self) -> None:
        from Deploy.common.policy_contract import COMMAND_TARGET_LIMITS, soft_command_limits

        soft = soft_command_limits(0.95)
        hard = COMMAND_TARGET_LIMITS
        np.testing.assert_allclose(soft.sum(axis=1), hard.sum(axis=1), atol=1e-6)
        np.testing.assert_allclose(
            soft[:, 1] - soft[:, 0], 0.95 * (hard[:, 1] - hard[:, 0]), rtol=1e-6
        )
        self.assertTrue(np.all(soft[:, 0] > hard[:, 0]))
        self.assertTrue(np.all(soft[:, 1] < hard[:, 1]))
        self.assertFalse(soft.flags.writeable)
        np.testing.assert_array_equal(soft_command_limits(1.0), hard)

    def test_margin_never_mutates_the_trained_contract(self) -> None:
        """A margin is layered on; the tables that define training stay put."""

        from Deploy.common import policy_contract as pc

        before = pc.COMMAND_TARGET_LIMITS.copy()
        before_norm = pc.OBSERVATION_NORMALIZATION_LIMITS.copy()
        pc.soft_command_limits(0.9)
        np.testing.assert_array_equal(pc.COMMAND_TARGET_LIMITS, before)
        np.testing.assert_array_equal(
            pc.OBSERVATION_NORMALIZATION_LIMITS, before_norm
        )
        pc.validate_contract()

    def test_rejects_a_fraction_outside_zero_to_one(self) -> None:
        from Deploy.common.policy_contract import soft_command_limits

        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError):
                soft_command_limits(bad)

    def test_margin_clamps_the_policy_target_but_not_what_it_asked_for(self) -> None:
        """The record of the policy's request must survive the safety clamp."""

        from Deploy.policy.finger_reach import (
            MIDDLE_COMMAND_TARGET_LIMITS,
            decode_reach_action,
            middle_soft_command_limits,
        )

        soft = middle_soft_command_limits(0.95)
        q = MIDDLE_COMMAND_TARGET_LIMITS[:, 1].astype(np.float32)
        push = np.ones(4, dtype=np.float32)

        hard_decoded = decode_reach_action(q, push)
        soft_decoded = decode_reach_action(q, push, soft)

        np.testing.assert_allclose(
            hard_decoded.position_target, MIDDLE_COMMAND_TARGET_LIMITS[:, 1], atol=1e-6
        )
        np.testing.assert_allclose(soft_decoded.position_target, soft[:, 1], atol=1e-6)
        self.assertTrue(np.all(soft_decoded.position_target < hard_decoded.position_target))
        # Same request, same record -- only the clamp differs.
        np.testing.assert_array_equal(soft_decoded.raw_action, hard_decoded.raw_action)
        np.testing.assert_array_equal(soft_decoded.clipped_action, hard_decoded.clipped_action)
        np.testing.assert_array_equal(
            soft_decoded.unclamped_target, hard_decoded.unclamped_target
        )

    def test_default_runner_reproduces_the_trained_clamp_exactly(self) -> None:
        """Nothing changes unless a margin is explicitly asked for."""

        from Deploy.policy.finger_reach import (
            MIDDLE_COMMAND_TARGET_LIMITS,
            middle_soft_command_limits,
        )

        np.testing.assert_array_equal(
            middle_soft_command_limits(1.0), MIDDLE_COMMAND_TARGET_LIMITS
        )

    def test_a_hand_parked_past_the_soft_edge_is_held_inside_it(self) -> None:
        """Seeding from a real pose must not command the hold to stay outside."""

        from Deploy.policy.finger_reach import (
            FingerReachObservationAdapter,
            MiddleFingerReachRunner,
        )
        from Deploy.common.policy_contract import COMMAND_TARGET_LIMITS, soft_command_limits

        class _Backend:
            def __init__(self) -> None:
                self.written = None

            def write_joint_position_targets(self, targets):
                self.written = np.asarray(targets, dtype=np.float32)

        backend = _Backend()
        runner = MiddleFingerReachRunner(
            backend, policy=None, observation_adapter=FingerReachObservationAdapter(),
            limit_fraction=0.95,
        )
        runner.seed_from_current_state(COMMAND_TARGET_LIMITS[:, 1].astype(np.float32))
        soft = soft_command_limits(0.95)
        self.assertTrue(np.all(backend.written <= soft[:, 1] + 1e-6))
        np.testing.assert_allclose(backend.written, soft[:, 1], atol=1e-6)


class ReachHistoryOrderingTests(unittest.TestCase):
    """At command() time, the observation's q_current must BE the residual base.

    MuJoCo satisfies this for free: command() builds the observation and reads
    the residual base back to back, with the plant advancing only afterwards in
    hold_policy_target().  A hardware loop has no such natural barrier, so if
    the history is advanced at the end of a policy tick instead of the start of
    the next one, the policy infers from q_(k-1) while the residual is computed
    from q_k -- one policy step apart, silently.
    """

    class _Backend:
        """Moves only when a target is published, like real hardware."""

        def __init__(self):
            from Deploy.common.policy_contract import POLICY_JOINT_NAMES

            self.q = np.zeros(len(POLICY_JOINT_NAMES), dtype=np.float32)
            self.target = self.q.copy()

        def read_joint_positions(self):
            return self.q.copy()

        def write_joint_position_targets(self, targets):
            self.target = np.asarray(targets, dtype=np.float32)

        def publish(self):
            """Stand-in for one policy step of plant motion."""
            self.q = (0.5 * self.q + 0.5 * self.target).astype(np.float32)

    class _RecordingPolicy:
        """Remember the q_current slice each observation carried."""

        def __init__(self):
            self.seen_q_current = []

        def infer(self, observation):
            from Deploy.policy.finger_reach import REACH_OBSERVATION_SLICES

            self.seen_q_current.append(
                np.asarray(observation)[REACH_OBSERVATION_SLICES["q_current"]].copy()
            )
            return np.full(4, 0.5, dtype=np.float32)

    def _runner(self):
        from Deploy.policy.finger_reach import (
            FingerReachObservationAdapter,
            MiddleFingerReachRunner,
        )

        backend = self._Backend()
        policy = self._RecordingPolicy()
        runner = MiddleFingerReachRunner(
            backend, policy, FingerReachObservationAdapter()
        )
        runner.seed_from_current_state(backend.read_joint_positions())
        runner.set_target(np.asarray([0.035, 0.010, 0.100], dtype=np.float32))
        return backend, runner, policy

    @staticmethod
    def _normalized(q_middle):
        from Deploy.policy.finger_reach import normalize_middle_joints

        return normalize_middle_joints(q_middle)

    def test_advancing_after_command_lags_the_observation_by_one_step(self) -> None:
        """The bug this guards against, stated as an executable fact."""

        from Deploy.policy.finger_reach import MIDDLE_POLICY_INDICES

        backend, runner, policy = self._runner()
        residual_bases = []
        for _ in range(4):
            residual_bases.append(backend.read_joint_positions()[MIDDLE_POLICY_INDICES])
            runner.command()
            runner.observe_after_hold()   # WRONG: nothing moved in between
            backend.publish()

        mismatches = [
            i
            for i, (seen, base) in enumerate(zip(policy.seen_q_current, residual_bases))
            if not np.allclose(seen, self._normalized(base), atol=1e-6)
        ]
        self.assertTrue(
            mismatches,
            "ordering was expected to desynchronize observation and residual base",
        )

    def test_advancing_before_command_keeps_them_the_same_sample(self) -> None:
        from Deploy.policy.finger_reach import MIDDLE_POLICY_INDICES

        backend, runner, policy = self._runner()
        residual_bases = []
        for tick in range(4):
            q_all = backend.read_joint_positions()
            if tick > 0:
                runner.observe_after_hold(q_all)   # the previous target was applied
            residual_bases.append(q_all[MIDDLE_POLICY_INDICES])
            runner.command(q_all)
            backend.publish()

        for index, (seen, base) in enumerate(zip(policy.seen_q_current, residual_bases)):
            np.testing.assert_allclose(
                seen, self._normalized(base), atol=1e-6,
                err_msg=f"policy tick {index} inferred from a different sample",
            )
        # And the history is genuinely two distinct samples, not a repeat.
        self.assertGreater(
            float(
                np.abs(
                    runner.observations.q_current - runner.observations.q_previous
                ).max()
            ),
            1e-4,
        )

    def test_injected_measurement_is_used_for_both_history_and_residual(self) -> None:
        from Deploy.policy.finger_reach import MIDDLE_POLICY_INDICES, decode_reach_action

        backend, runner, _ = self._runner()
        runner.command()

        injected = backend.read_joint_positions()
        injected[MIDDLE_POLICY_INDICES] += 0.05
        runner.observe_after_hold(injected)
        np.testing.assert_allclose(
            runner.observations.q_current, injected[MIDDLE_POLICY_INDICES]
        )

        decoded = runner.command(injected)
        expected = decode_reach_action(
            injected[MIDDLE_POLICY_INDICES], decoded.raw_action
        )
        np.testing.assert_allclose(decoded.position_target, expected.position_target)
        self.assertFalse(
            np.allclose(
                injected[MIDDLE_POLICY_INDICES],
                backend.read_joint_positions()[MIDDLE_POLICY_INDICES],
            )
        )


class PolicyTipFrameTests(unittest.TestCase):
    """Pin which URDF supplies obs[40:55].

    Two vendor releases of the Wuji URDF are in this repository and they define
    ``*_tip_link`` differently.  The physical part lands in the same palm-frame
    place either way (<=0.33 mm), so this is not a geometry problem -- but the
    LINK ORIGIN is what an observation reports, and those differ by 3.0 mm at
    the thumb.  The policy learned Isaac's, so Isaac's is the contract.
    """

    # Measured 2026-08-21 from the two files' *_tip_fixed origins.  Exact by
    # construction: official minus isaac, along the distal axis.
    EXPECTED_TIP_FRAME_DELTA_MM = (3.0, 0.7, 0.0, 0.0, 0.0)

    def setUp(self) -> None:
        self.official = WujiHand1FingertipFK(OFFICIAL_URDF)
        self.trained = WujiHand1FingertipFK(POLICY_TIP_FRAME_URDF)
        limits = OBSERVATION_NORMALIZATION_LIMITS
        self.q = ((limits[:, 0] + limits[:, 1]) / 2).astype(np.float32)

    def test_policy_tip_frame_is_the_isaac_urdf(self) -> None:
        self.assertIs(POLICY_TIP_FRAME_URDF, ISAAC_URDF)

    def test_source_argument_is_required(self) -> None:
        # The bug this guards against was an implicit default silently meaning
        # "official".  Every caller must now state which model it means.
        with self.assertRaises(TypeError):
            WujiHand1FingertipFK()
        with self.assertRaises(TypeError):
            WujiHand1FingertipFK(str(OFFICIAL_URDF.path))

    def test_two_urdfs_differ_by_the_pinned_tip_offsets(self) -> None:
        delta = (
            self.official.fingertip_positions_in_palm(self.q)
            - self.trained.fingertip_positions_in_palm(self.q)
        ).reshape(5, 3)
        distance_mm = np.linalg.norm(delta, axis=1) * 1.0e3
        np.testing.assert_allclose(
            distance_mm, self.EXPECTED_TIP_FRAME_DELTA_MM, atol=0.05
        )

    def test_tip_frame_offset_is_constant_over_the_workspace(self) -> None:
        # A fixed link-mount difference, not a joint-axis error: if this ever
        # becomes pose-dependent the two files diverged in the chain itself.
        lower = OBSERVATION_NORMALIZATION_LIMITS[:, 0]
        upper = OBSERVATION_NORMALIZATION_LIMITS[:, 1]
        generator = np.random.default_rng(0)
        distances = []
        for _ in range(8):
            q = (lower + generator.random(ACTION_DIM) * (upper - lower)).astype(np.float32)
            delta = (
                self.official.fingertip_positions_in_palm(q)
                - self.trained.fingertip_positions_in_palm(q)
            ).reshape(5, 3)
            distances.append(np.linalg.norm(delta, axis=1) * 1.0e3)
        spread = np.ptp(np.stack(distances), axis=0)
        self.assertLess(float(spread.max()), 0.05)

    def test_observation_fingertips_come_from_the_trained_urdf(self) -> None:
        adapter = PolicyObservationAdapter(mode="open")
        adapter.reset(self.q)
        fingertips = adapter.build()[OBSERVATION_SLICES["fingertips"].slice]
        np.testing.assert_allclose(
            fingertips, self.trained.fingertip_positions_in_palm(self.q), atol=0.0
        )
        self.assertFalse(
            np.allclose(
                fingertips, self.official.fingertip_positions_in_palm(self.q), atol=1e-4
            )
        )

    def test_runner_never_asks_the_backend_for_fingertips(self) -> None:
        # A backend's own model's tips are not the policy's contract, so the
        # boundary must not carry them -- see backend_protocol.
        class TipsWouldBeWrong(FakeBackend):
            def get_fingertip_positions_in_palm(self):
                raise AssertionError("PolicyRunner must solve fingertips from q.")

        class ZeroPolicy:
            def infer(self, observation):
                return np.zeros(ACTION_DIM, dtype=np.float32)

        backend = TipsWouldBeWrong(self.q)
        runner = PolicyRunner(backend, ZeroPolicy(), PolicyObservationAdapter())
        runner.reset()
        runner.command()
        runner.observe_after_hold()

    def test_urdf_sources_name_distinct_files_that_exist(self) -> None:
        for source in (OFFICIAL_URDF, ISAAC_URDF):
            self.assertIsInstance(source, UrdfSource)
            self.assertTrue(source.path.is_file(), source.path)
        self.assertNotEqual(OFFICIAL_URDF.path, ISAAC_URDF.path)


class OneReadPerPolicyStepTests(unittest.TestCase):
    """The observation and the residual base must be one sample.

    ``q_target = q_current + scale * action`` only means anything if
    ``q_current`` is the q the policy was shown.  Reading twice satisfies that
    in a deterministic simulator by luck -- nothing runs in between -- and
    breaks on hardware, where two SDK reads differ by encoder noise.
    """

    class CountingBackend(FakeBackend):
        def __init__(self, q):
            super().__init__(q)
            self.reads = 0

        def read_joint_positions(self):
            self.reads += 1
            return super().read_joint_positions()

    class ZeroPolicy:
        def infer(self, observation):
            return np.zeros(ACTION_DIM, dtype=np.float32)

    def setUp(self) -> None:
        limits = OBSERVATION_NORMALIZATION_LIMITS
        self.q = ((limits[:, 0] + limits[:, 1]) / 2).astype(np.float32)

    def test_command_does_not_read_the_backend(self) -> None:
        backend = self.CountingBackend(self.q)
        runner = PolicyRunner(backend, self.ZeroPolicy(), PolicyObservationAdapter())
        runner.reset()
        backend.reads = 0
        runner.command()
        self.assertEqual(backend.reads, 0)
        runner.observe_after_hold()
        self.assertEqual(backend.reads, 1)

    def test_residual_base_is_the_observed_sample(self) -> None:
        # Move the backend AFTER observe_after_hold.  The next command must
        # still decode against the observed q, not against the new position.
        backend = self.CountingBackend(self.q)
        runner = PolicyRunner(backend, self.ZeroPolicy(), PolicyObservationAdapter())
        runner.reset()
        runner.command()
        runner.observe_after_hold()
        observed = backend.read_joint_positions()

        backend.q = (observed + np.float32(0.02)).astype(np.float32)
        decoded = runner.command()
        # Zero action -> target == the q the observation was built from.
        np.testing.assert_allclose(decoded.unclamped_target, observed, atol=0.0)

    def test_callers_may_inject_their_single_reading(self) -> None:
        # Hardware takes exactly one reading per policy tick and hands the same
        # array to both calls.
        backend = self.CountingBackend(self.q)
        runner = PolicyRunner(backend, self.ZeroPolicy(), PolicyObservationAdapter())
        sample = backend.read_joint_positions()
        runner.reset(sample)
        backend.reads = 0
        runner.command()
        runner.observe_after_hold(sample)
        self.assertEqual(backend.reads, 0)

    def test_first_tick_decodes_against_the_sample_it_was_seeded_with(self) -> None:
        """One sample per tick, including the first.

        The entry points seed with ``reset(q)`` and then loop.  If the first
        iteration reads again, the observation and residual come from the seed
        while the backend's slew guard checks the newer read -- and the drift
        between them lands on top of a legal step.  On an unpowered, settling
        hand that drift was 2.35 mrad, which pushed a 0.150 rad joint4 move to
        0.15235 and had it refused as illegal.
        """

        backend = self.CountingBackend(self.q)
        runner = PolicyRunner(backend, self.ZeroPolicy(), PolicyObservationAdapter())
        seeded = backend.read_joint_positions()
        runner.reset(seeded)

        # The hand drifts between the seed and the first tick.
        backend.q = (seeded + np.float32(0.00235)).astype(np.float32)

        decoded = runner.command()
        # Zero action -> the target IS the sample it decoded against.
        np.testing.assert_allclose(decoded.unclamped_target, seeded, atol=0.0)
        np.testing.assert_allclose(runner.q_current, seeded, atol=0.0)

    def test_q_current_is_a_copy_not_the_live_buffer(self) -> None:
        backend = self.CountingBackend(self.q)
        runner = PolicyRunner(backend, self.ZeroPolicy(), PolicyObservationAdapter())
        runner.reset(backend.read_joint_positions())
        snapshot = runner.q_current
        snapshot[:] = 99.0
        self.assertFalse(np.allclose(runner.q_current, 99.0))

    def test_command_before_reset_is_refused(self) -> None:
        backend = self.CountingBackend(self.q)
        runner = PolicyRunner(backend, self.ZeroPolicy(), PolicyObservationAdapter())
        with self.assertRaisesRegex(RuntimeError, "reset"):
            runner.command()


class ObservationLogContractTests(unittest.TestCase):
    """The obs a run logs must be the obs its action came from.

    This is what makes a log replayable: feed the logged observation back
    through the same ONNX and the logged action must come out.  The MuJoCo
    runner logged the observation ``run_policy_tick`` returns, which is built
    AFTER the physics hold and is therefore the NEXT tick's input -- a
    one-step offset that made every replay disagree by ~0.18 while both runs
    were individually fine (2026-08-22).  Nothing in a trajectory plot shows
    that; only the replay does.
    """

    class ObsEchoPolicy:
        """Action determined by the observation, so an offset cannot pass."""

        def infer(self, observation):
            observation = np.asarray(observation, dtype=np.float32)
            return np.tanh(observation[:ACTION_DIM]).astype(np.float32)

    def setUp(self) -> None:
        limits = OBSERVATION_NORMALIZATION_LIMITS
        self.q = ((limits[:, 0] + limits[:, 1]) / 2).astype(np.float32)

    def test_last_observation_is_the_command_input(self) -> None:
        backend = FakeBackend(self.q)
        runner = PolicyRunner(backend, self.ObsEchoPolicy(), PolicyObservationAdapter())
        runner.reset()
        for _ in range(3):
            decoded = runner.command()
            np.testing.assert_allclose(
                decoded.onnx_action,
                self.ObsEchoPolicy().infer(runner.last_observation),
                atol=0.0,
                err_msg="last_observation is not the input command() ran on.")
            backend.q = (backend.q + np.float32(0.01)).astype(np.float32)
            runner.observe_after_hold()

    def test_reset_clears_the_cached_observation(self) -> None:
        backend = FakeBackend(self.q)
        runner = PolicyRunner(backend, self.ObsEchoPolicy(), PolicyObservationAdapter())
        runner.reset()
        runner.command()
        self.assertIsNotNone(runner.last_observation)
        runner.reset()
        self.assertIsNone(runner.last_observation,
                          "A stale observation must not survive a reset.")

    def test_both_runners_log_the_command_input(self) -> None:
        """Source check: neither runner may log a post-step observation."""

        for name in ("run_hand_policy_real.py", "run_mujoco_policy.py"):
            source = (Path(__file__).resolve().parents[1] / "run" / name).read_text()
            self.assertIn("list(runner.last_observation)", source, name)
            self.assertNotIn("+ list(observation)", source, name)

    def test_observation_columns_cover_every_element(self) -> None:
        columns = observation_csv_columns()
        self.assertEqual(len(columns), OBSERVATION_DIM)
        self.assertEqual(len(set(columns)), OBSERVATION_DIM, "duplicate column name")
        for term in OBSERVATION_SLICES.values():
            for offset, index in enumerate(range(term.start, term.stop)):
                self.assertTrue(columns[index].endswith(f"_{offset:02d}"))


class PregraspFromRunTests(unittest.TestCase):
    """A checkpoint's pregrasp comes from its own run, not from the constant.

    ``ISAAC_PREGRASP_JOINT_POSITIONS_RAD`` is one run's pose and it also anchors
    ``ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ`` -- the stick poses were recorded
    with the hand in exactly that pose.  Editing the constant to chase a new
    checkpoint would silently invalidate MuJoCo's initial grasp geometry, which
    is why the real runner reads the pose per run instead.
    """

    YAML = """\
events:
  reset_pregrasp:
    func: mdp:reset_to_functional_pregrasp
    params:
      joint_positions: !!python/tuple
{items}
  other_event:
    func: mdp:something_else
"""

    def _write(self, tmp, values):
        items = "\n".join(f"      - {v}" for v in values)
        params = tmp / "params"
        params.mkdir(parents=True, exist_ok=True)
        (params / "env.yaml").write_text(self.YAML.format(items=items))
        return tmp

    def test_reads_twenty_joints_and_stops_at_the_next_key(self) -> None:
        import tempfile

        from Deploy.common.isaac_reset import read_pregrasp_from_env_yaml

        values = [round(0.01 * i, 4) for i in range(20)]
        with tempfile.TemporaryDirectory() as name:
            run = self._write(Path(name), values)
            np.testing.assert_allclose(read_pregrasp_from_env_yaml(run), values,
                                       atol=1e-6)
            # Same run, addressed by the paths a person actually has to hand.
            np.testing.assert_allclose(
                read_pregrasp_from_env_yaml(run / "params" / "env.yaml"),
                values, atol=1e-6)
            (run / "exported").mkdir()
            (run / "exported" / "policy.onnx").write_bytes(b"")
            np.testing.assert_allclose(
                read_pregrasp_from_env_yaml(run / "exported" / "policy.onnx"),
                values, atol=1e-6)

    def test_scientific_notation_survives(self) -> None:
        """Isaac writes ``-1.81e-07``; a plain [0-9.] regex drops it silently."""

        import tempfile

        from Deploy.common.isaac_reset import read_pregrasp_from_env_yaml

        values = [0.5377866626, 0.8436813951, 0.0377136655, -1.81e-07] + [0.0] * 16
        with tempfile.TemporaryDirectory() as name:
            run = self._write(Path(name), values)
            np.testing.assert_allclose(read_pregrasp_from_env_yaml(run), values,
                                       atol=1e-9)

    def test_short_list_is_an_error_not_a_pad(self) -> None:
        import tempfile

        from Deploy.common.isaac_reset import read_pregrasp_from_env_yaml

        with tempfile.TemporaryDirectory() as name:
            run = self._write(Path(name), [0.1] * 19)
            with self.assertRaises(ValueError):
                read_pregrasp_from_env_yaml(run)

    def test_the_constant_is_not_mutated(self) -> None:
        import tempfile

        from Deploy.common.isaac_reset import (
            ISAAC_PREGRASP_JOINT_POSITIONS_RAD, read_pregrasp_from_env_yaml,
        )

        before = ISAAC_PREGRASP_JOINT_POSITIONS_RAD.copy()
        with tempfile.TemporaryDirectory() as name:
            read_pregrasp_from_env_yaml(self._write(Path(name), [0.3] * 20))
        np.testing.assert_array_equal(ISAAC_PREGRASP_JOINT_POSITIONS_RAD, before)


class ClosureRebindingTests(unittest.TestCase):
    """No accumulator in a per-tick closure may be rebound with ``+=``.

    ``on_policy_tick`` is a nested function, so ``total += x`` on a name from
    the enclosing scope makes that name LOCAL for the whole function and the
    first call raises UnboundLocalError.  It cannot be caught by importing or
    by a dry run of the surrounding code -- only by executing the tick -- so on
    2026-08-23 it reached the hardware and killed a run before step 0, after
    ENABLE, GLIDE and a 45 s chopstick insert had already been paid for.

    ``action_peak["sum"] += ...`` is fine (mutating a dict is not a rebinding);
    ``current_sum[:] += ...`` is fine (in-place on a slice).  A BARE ``name +=``
    where ``name`` is not assigned in the function is the bug, so that is what
    this scans for.
    """

    #: Functions whose bodies run once per policy step, inside a closure.
    TICK_FUNCTIONS = ("on_policy_tick", "on_hold_tick", "report_move")

    def _sources(self):
        package = Path(__file__).resolve().parents[1]
        for path in sorted((package / "run").glob("*.py")):
            yield path, path.read_text()

    def test_no_bare_augmented_assignment_to_an_outer_name(self) -> None:
        import ast

        for path, source in self._sources():
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if node.name not in self.TICK_FUNCTIONS:
                    continue
                declared = {
                    name
                    for child in ast.walk(node)
                    for name in (
                        getattr(child, "names", [])
                        if isinstance(child, (ast.Nonlocal, ast.Global)) else []
                    )
                }
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                declared.add(target.id)
                    elif isinstance(child, ast.arg):
                        declared.add(child.arg)
                for child in ast.walk(node):
                    if not isinstance(child, ast.AugAssign):
                        continue
                    if not isinstance(child.target, ast.Name):
                        continue  # obj[...] += / obj.attr += are in-place
                    if child.target.id in declared:
                        continue
                    self.fail(
                        f"{path.name}:{child.lineno} in {node.name}(): "
                        f"`{child.target.id} +=` rebinds an outer name and will "
                        f"raise UnboundLocalError on the first tick.  Use "
                        f"`{child.target.id}[:] +=`, a dict, or `nonlocal`.")

    def test_the_scan_would_catch_the_real_bug(self) -> None:
        """Guard against the scan silently matching nothing."""

        import ast

        source = (
            "def outer():\n"
            "    total = 0\n"
            "    def on_policy_tick(i):\n"
            "        total += i\n"
        )
        tree = ast.parse(source)
        found = [
            child
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "on_policy_tick"
            for child in ast.walk(node)
            if isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name)
        ]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].target.id, "total")


class GlideStaysInsideCommandLimitsTests(unittest.TestCase):
    """Every commanded sample of a glide, not just its endpoints.

    ``COMMAND_LIMIT_RATIO`` made joints sitting exactly ON a bound routine:
    three of them clamp to their upper.  With ``q_start == q_goal == upper``,
    ``(1 - a) * q_start + a * q_goal`` is float32 arithmetic under NumPy 2's
    weak promotion (a Python float times a float32 array stays float32), and it
    rounds ONE ULP past the bound.  The backend rejects out-of-range targets by
    design, so on 2026-08-23 that killed a run at glide tick 1 -- after ENABLE
    and a 45 s chopstick insert.

    The endpoints were already clamped.  That is what made the bug invisible:
    clamping the ends is not the same as keeping the path inside the box.
    """

    class RejectingBackend:
        """Mirrors RealWujiHand: refuses a target outside the command limits."""

        def __init__(self, q):
            self.q = np.asarray(q, dtype=np.float32)
            self.published = 0
            self.worst_overshoot = 0.0

        def read_joint_positions(self):
            return self.q.copy()

        def write_joint_position_targets(self, targets, max_step_rad=None):
            targets = np.asarray(targets, dtype=np.float32)
            lower = COMMAND_TARGET_LIMITS[:, 0]
            upper = COMMAND_TARGET_LIMITS[:, 1]
            over = float(np.max(np.concatenate([targets - upper, lower - targets])))
            self.worst_overshoot = max(self.worst_overshoot, over)
            if np.any(targets < lower) or np.any(targets > upper):
                bad = np.flatnonzero((targets < lower) | (targets > upper)).tolist()
                raise ValueError(
                    f"Targets outside COMMAND_TARGET_LIMITS at policy indices "
                    f"{bad}: {targets[bad].tolist()}")
            self.q = targets.copy()

        def publish_latest_target(self, controller):
            self.published += 1

    def test_glide_from_and_to_the_upper_bound(self) -> None:
        from Deploy.backends.real_wuji_scheduler import RealWujiScheduler

        # Both ends past the upper bound, so both clamp exactly onto it -- the
        # 2026-08-23 situation for finger3/4/5_joint3.
        upper = COMMAND_TARGET_LIMITS[:, 1]
        measured = (upper + np.float32(0.033)).astype(np.float32)
        goal = (upper + np.float32(0.031)).astype(np.float32)

        backend = self.RejectingBackend(measured)
        scheduler = RealWujiScheduler(backend, command_hz=90.0)
        scheduler.glide_to_pose(
            goal, controller=None, joint_indices=np.arange(ACTION_DIM),
            seconds=0.5, tolerance_rad=1.0, stable_seconds=0.0,
            timeout_seconds=2.0)
        self.assertLessEqual(backend.worst_overshoot, 0.0)
        self.assertGreater(backend.published, 1)

    def test_the_float32_blend_really_does_overshoot(self) -> None:
        """Guard against the test above passing for the wrong reason."""

        # Reproduce the ARRAY path the scheduler uses: under NEP 50 a Python
        # float times a float32 array stays float32.  Which alpha rounds up
        # depends on the bit pattern, so scan rather than pick one.
        bound = COMMAND_TARGET_LIMITS[:, 1].astype(np.float32)
        ticks = 45  # 0.5 s at 90 Hz, as the glide test above
        naive_over = max(
            float(np.max(((1.0 - a) * bound + a * bound).astype(np.float32) - bound))
            for a in (min(1.0, (t + 1) / ticks) for t in range(ticks))
        )
        self.assertGreater(naive_over, 0.0,
                           "float32 blend no longer overshoots; the glide test "
                           "above would then pass for the wrong reason")
        widened_over = max(
            float(np.max(np.clip(
                (1.0 - a) * bound.astype(np.float64) + a * bound.astype(np.float64),
                COMMAND_TARGET_LIMITS[:, 0], bound).astype(np.float32) - bound))
            for a in (min(1.0, (t + 1) / ticks) for t in range(ticks))
        )
        self.assertLessEqual(widened_over, 0.0)


class SafeStopSemanticsTests(unittest.TestCase):
    """safe_stop freezes the command; it must not release the grip.

    ``q_target - q`` is the preload -- that gap is the force pressing the
    sticks.  Re-latching the target onto the measured q looks like "hold still"
    and is actually "let go", which is the worst possible response to losing
    sight of the object being held.
    """

    def test_mujoco_safe_stop_changes_no_target(self) -> None:
        from Deploy.backends.mujoco_wuji import MujocoWujiHand

        hand = MujocoWujiHand()
        hand.reset()
        # Command a preload: a target deliberately away from the measured q.
        q = hand.read_joint_positions()
        target = np.clip(
            q + np.float32(0.05), COMMAND_TARGET_LIMITS[:, 0], COMMAND_TARGET_LIMITS[:, 1]
        ).astype(np.float32)
        hand.write_joint_position_targets(target)
        before = hand.control_snapshot()

        hand.safe_stop("test")

        np.testing.assert_array_equal(hand.control_snapshot(), before)
        self.assertTrue(hand.safe_stopped)
        self.assertEqual(hand.safe_stop_reason, "test")

    def test_mujoco_health_reports_a_latched_safe_stop(self) -> None:
        from Deploy.backends.mujoco_wuji import MujocoWujiHand

        hand = MujocoWujiHand()
        self.assertTrue(hand.health().ok)
        hand.safe_stop("lost perception")
        health = hand.health()
        self.assertFalse(health.ok)
        self.assertIn("lost perception", health.message)

    def test_safe_stop_is_sticky_across_a_reset(self) -> None:
        # A run that hit a safe stop must not look clean afterwards.
        from Deploy.backends.mujoco_wuji import MujocoWujiHand

        hand = MujocoWujiHand()
        hand.safe_stop("lost perception")
        hand.reset()
        self.assertTrue(hand.safe_stopped)


class RealEntryPointGuardTests(unittest.TestCase):
    """Guards that must hold before the entry point ever touches hardware.

    Checked by parsing rather than by running: the module imports wujihandpy,
    which does not exist in the environment the contract tests run in, and the
    point of these guards is that they fire BEFORE a hand is opened.
    """

    SOURCE = Path(__file__).resolve().parents[1] / "run/run_hand_policy_real.py"

    def setUp(self) -> None:
        self.text = self.SOURCE.read_text(encoding="utf-8")

    def test_synthetic_sticks_are_refused_before_the_hand_is_opened(self) -> None:
        refusal = self.text.index("allow_synthetic_sticks")
        construction = self.text.index("RealWujiHand(read_source=")
        self.assertLess(
            refusal, construction,
            "The synthetic-stick refusal must come before the hand is constructed.",
        )

    def test_insert_wait_is_bounded_and_temperature_watched(self) -> None:
        # The unloaded pregrasp pose is a stall.  An unbounded 'press Enter'
        # is exactly the 96 s that reached 88.4 C.
        self.assertIn("DEFAULT_INSERT_TIMEOUT_S", self.text)
        self.assertIn("read_joint_temperatures", self.text)
        self.assertIn("temperature_limit", self.text)

    def test_blocking_temperature_read_stays_out_of_the_policy_loop(self) -> None:
        # read_joint_temperatures is a blocking SDO read with a 0.5 s timeout.
        # A command tick is 11.1 ms at 90 Hz, so it belongs only where the
        # target is constant.
        run_phase = self.text[self.text.index("def on_policy_tick("):]
        run_phase = run_phase[:run_phase.index("print(f\"\\n[RUN]")]
        self.assertNotIn("read_joint_temperatures", run_phase)

    def test_insert_wait_does_not_use_the_policy_scheduler(self) -> None:
        # Publishing there is hand-rolled so an overrunning temperature read
        # cannot pollute the timing statistics of the policy run.
        block = self.text[self.text.index("def wait_for_chopsticks("):]
        block = block[:block.index("def main()")]
        self.assertIn("backend.publish_latest_target(controller)", block)
        self.assertNotIn("scheduler.run(", block)

    def test_home_return_runs_inside_the_open_controller(self) -> None:
        """The return needs a LIVE controller, so it cannot sit in the outer finally.

        It did, and on the path it existed for -- an aborted run -- it fired
        after ``with backend.realtime_controller(...)`` had already closed and
        died with "Controller is closed."  It belongs in a finally INSIDE that
        context, before the outer one disables the motors.
        """

        opened = self.text.index("with backend.realtime_controller(")
        opened = self.text.index("with backend.realtime_controller(", opened + 1)
        outer_finally = self.text.rindex("    finally:")
        ret = self.text.index("[RETURN]   시작 자세로")
        self.assertGreater(ret, opened, "복귀가 컨트롤러 컨텍스트보다 앞에 있다")
        self.assertLess(ret, outer_finally, "복귀가 컨트롤러가 닫힌 뒤에 있다")

        # It must still run when the body raised, i.e. from a finally.
        block = self.text[opened:outer_finally]
        self.assertIn("finally:", block)
        # Ctrl+C is an emergency stop and skips it.
        self.assertIn("_ctrl_c_pending[0]", block)
        self.assertIn("_ctrl_c_pending[0] = True", block)
        # A failed return must not skip the disable that follows.
        self.assertIn("except Exception as exc:", block)

    def test_disable_still_follows_the_return(self) -> None:
        tail = self.text[self.text.rindex("    finally:"):]
        self.assertIn("backend.disable()", tail)
        self.assertNotIn("[RETURN]", tail)

    def test_temperature_is_read_before_motors_are_enabled(self) -> None:
        # The glide presses the thumb for the whole of --glide-seconds; a hand
        # that started warm from the previous run never gets to shed that.
        temp = self.text.index("[TEMP]     시작 온도")
        enable = self.text.index("backend.enable(mask)")
        self.assertLess(temp, enable)

    def test_arrival_is_judged_by_size_on_both_glides(self) -> None:
        """A quarter of a degree short must not fail a run; a jam still must.

        Real position servos leave a steady-state offset.  Measured 2026-08-21:
        outbound, finger1_joint2 held 0.045 rad for 25 s; homeward,
        finger2_joint2 held 0.035 rad against a 0.030 tolerance -- while every
        other joint sat inside 0.007.  The outbound glide got the size-based
        verdict first and the return did not, so a successful 300-step run ended
        with "move_home.py 로 복구하세요" over 0.28 deg.  One helper serves both.
        """

        self.assertIn("--start-abort-rad", self.text)
        self.assertIn("def settle_verdict(", self.text)
        self.assertIn("도착 판정 실패했지만 진행", self.text)
        verdict = self.text[self.text.index("def settle_verdict("):]
        verdict = verdict[:verdict.index("\ndef ")]
        self.assertIn("start_abort_rad", verdict)
        self.assertIn("raise RuntimeError(", verdict)
        self.assertIn('settle_verdict(backend, pregrasp, args, "GLIDE")', self.text)
        self.assertIn('settle_verdict(backend, q_now, args, "RETURN")', self.text)

    def test_disable_is_in_a_finally(self) -> None:
        tail = self.text[self.text.index("    finally:"):]
        self.assertIn("backend.disable()", tail)

    def test_safe_stop_abort_keeps_publishing(self) -> None:
        # Freezing the target only holds if something keeps sending it; the
        # firmware's behaviour when commands stop is unverified.
        block = self.text[self.text.index("[SAFE STOP]"):]
        self.assertIn("scheduler.run(", block)

    def test_full_enable_mask_is_used_not_the_middle_finger_one(self) -> None:
        self.assertIn("full_enable_mask()", self.text)
        self.assertNotIn("middle_enable_mask", self.text)

    def test_the_perception_source_is_actually_started(self) -> None:
        # The camera provider defers opening to start() so its refusals can fire
        # with nothing connected.  That deferral is only safe if the entry point
        # calls it -- forgetting to cost a crash on the first hardware run, at
        # the first sample(), after the hand had already been opened.
        # Count CALL sites, not the definition, which contains the same text.
        calls = [l for l in self.text.split("\n")
                 if "_start_provider(provider)" in l and not l.lstrip().startswith("def ")]
        # Both paths: read-only builds observations too, so it needs cameras.
        self.assertEqual(len(calls), 2, calls)

    def test_neither_loop_reads_twice_on_its_first_tick(self) -> None:
        # Both entry-point loops seed with reset() and must reuse that sample on
        # tick 0 rather than taking a second reading.
        for marker in ("if step == 0:", "if policy_index == 0:"):
            index = self.text.index(marker)
            # Read to the matching else:, so a long explanatory comment between
            # the branch and its body cannot make this pass or fail by accident.
            branch = self.text[index:self.text.index("else:", index)]
            self.assertIn("runner.q_current", branch, marker)
            self.assertNotIn("backend.read_joint_positions()", branch, marker)

    def test_the_read_only_loop_runs_at_the_policy_rate(self) -> None:
        # Free-running, the loop measures per-step cost, not whether 30 Hz is
        # held -- 30 Hz is never attempted.  Printing a budget the loop does not
        # keep is how a latency number gets misread as a rate problem, which is
        # exactly what happened on the first instrumented hardware run.
        body = self.text[self.text.index("def _read_only_loop("):]
        body = body[:body.index("\ndef ")]
        self.assertIn("POLICY_DT", body)
        self.assertIn("time.sleep(wait)", body)
        self.assertIn('timer.stage("total")', body)

    def test_read_only_opens_a_controller_but_never_enables_or_publishes(self) -> None:
        # Opening the controller makes read_joint_positions take the same branch
        # the drive loop takes; without it the tick is dominated by a blocking
        # SDO read the drive loop never pays.  It must still move nothing.
        block = self.text[self.text.index("[CONTROLLER]"):]
        block = block[:block.index("finally:\n            _stop_provider")]
        self.assertIn("backend.realtime_controller(", block)
        self.assertNotIn("backend.enable(", block)
        self.assertNotIn("publish_latest_target", block)
        # And it is released again, so the drive path cannot inherit it.
        self.assertIn("backend.controller = None", block)

    def test_cameras_open_before_the_motors(self) -> None:
        # So the INSERT wait can show whether the tracker picked up the
        # chopsticks that were just placed -- deciding that after the run has
        # been committed to is too late.  Also keeps the 1.2 s warm-up out of
        # the gap between settle and the first policy step.
        drive = self.text[self.text.index("# ---------------- DRIVE"):]
        start = drive.index("_start_provider(provider)")
        enable = drive.index("backend.enable(mask)")
        self.assertLess(start, enable)
        # ...and not a second time at SEED.
        seed = drive[drive.index("# ---------------- SEED"):]
        seed = seed[:seed.index("# ---------------- RUN")]
        self.assertNotIn("_start_provider", seed)

    def test_perception_state_is_counted_during_the_run(self) -> None:
        # HOLD rides through silently by design, so "the run did not abort"
        # only rules out STALE/LOST.  Whether tracking ever dropped has to be
        # counted, or the question cannot be answered afterwards.
        self.assertIn("[PERCEPTION]", self.text)
        self.assertIn("perception[seen] = perception.get(seen, 0) + 1", self.text)
        self.assertIn("끊긴 적 없음", self.text)

    def test_keys_are_read_during_the_safe_stop_hold(self) -> None:
        """The hold loop must poll stdin too.

        When safe_stop fires the policy loop exits, so the mode keys read inside
        on_policy_tick stop being polled.  On 2026-08-22 the operator pressed
        o/c during that hold and nothing happened -- the terminal had received
        the keys, but no code was reading them.  Ctrl+C was the only way out,
        and that skips the return glide.
        """

        block = self.text[self.text.index("[SAFE STOP]"):]
        block = block[:block.index("# ---------------- 지연 보고")]
        self.assertIn("on_command_tick=on_hold_tick", block)
        self.assertIn("read_key()", block)
        self.assertIn('hold["quit"]', block)
        # Ctrl+C there is still an emergency stop, so it skips the return.
        self.assertIn("_ctrl_c_pending[0] = True", block)

    def test_camera_display_is_scaled_and_rate_limited(self) -> None:
        # Two 1280x720 windows cost 21.5 ms p95 and blow the 11.1 ms command
        # tick; pollKey does not help because the cost is Qt pushing pixels.
        # Half scale is 5.9 ms, every sixth tick is 4.5 ms.
        self.assertIn("--camera-scale", self.text)
        self.assertIn("--camera-every", self.text)
        self.assertIn("camera_scale=args.camera_scale", self.text)
        self.assertIn("camera_every=args.camera_every", self.text)

    def test_the_perception_source_is_closed_on_every_path(self) -> None:
        # Two RealSense pipelines and two threads; leaking them wedges the next
        # run with a device-busy error rather than a clear message.
        calls = [l for l in self.text.split("\n")
                 if "_stop_provider(provider)" in l and not l.lstrip().startswith("def ")]
        self.assertEqual(len(calls), 2, calls)
        tail = self.text[self.text.rindex("    finally:"):]
        self.assertIn("_stop_provider(provider)", tail)


class DeployRigProvenanceTests(unittest.TestCase):
    """The lab rig is separate from the simulated ArUco scene, and labelled.

    Two of the rig's inputs are not measurements: the hand mounting yaw is the
    tracker's own stated candidate, and q6 is typed in by a person rather than
    read from the arm.  Both rotate the palm frame, and the palm frame is the
    frame every stick observation is expressed in, so a guess promoted to a
    contract here would be invisible everywhere downstream.
    """

    def setUp(self) -> None:
        from Deploy.vision import deploy_rig

        self.rig = deploy_rig

    def test_unverified_inputs_are_named(self) -> None:
        pending = set(self.rig.unverified_inputs())
        self.assertIn("TOTAL_YAW_DEG", pending)
        self.assertIn("q6_deg", pending)
        self.assertNotIn("T_BASE_CAMERA_MAIN", pending)
        # The mount yaw alone is NOT listed: it is not separately identifiable,
        # so asking anyone to "verify 155 deg" would send them after a quantity
        # that does not exist.
        self.assertNotIn("HAND_MOUNT_YAW_OFFSET_DEG", pending)

    def test_only_the_yaw_sum_is_identifiable(self) -> None:
        # Same axis, and the mount translation runs along it, so the split is
        # bookkeeping.  If this ever stops holding, the split becomes real and
        # the provenance table is wrong.
        rig = self.rig
        for q6 in (0.0, 25.000097, -40.0, 130.0):
            split = (
                rig.T_BASE_J6
                @ rig.rotation_z(np.deg2rad(q6))
                @ rig.t_link6_hand()
            )
            merged = rig.T_BASE_J6 @ rig.rotation_z(
                np.deg2rad(q6 + rig.HAND_MOUNT_YAW_OFFSET_DEG)
            )
            merged[:3, 3] = split[:3, 3]  # translation compared separately below
            np.testing.assert_allclose(split[:3, :3], merged[:3, :3], atol=1e-12)
        # The translation is along the rotation axis, hence yaw-independent.
        a = rig.t_base_hand(0.0)[:3, 3] - rig.T_BASE_J6[:3, 3]
        b = rig.t_base_hand(90.0)[:3, 3] - rig.T_BASE_J6[:3, 3]
        np.testing.assert_allclose(np.linalg.norm(a), np.linalg.norm(b), atol=1e-12)

    def test_palm_up_residual_is_recorded_not_silently_optimised(self) -> None:
        # The palm-up optimum is 182.724 deg and the rig uses 180.000097.  The
        # difference is kept visible because it may belong to T_BASE_J6 rather
        # than to the yaw; quietly adopting the optimum would hide that.
        self.assertAlmostEqual(self.rig.TOTAL_YAW_DEG, 180.000097, places=6)
        self.assertNotAlmostEqual(
            self.rig.TOTAL_YAW_DEG, self.rig.PALM_UP_OPTIMUM_TOTAL_YAW_DEG, places=2
        )
        self.assertGreater(self.rig.PALM_UP_RESIDUAL_DEG, 0.0)

    def test_unverified_geometry_is_reported_every_run(self) -> None:
        """Warns, never refuses -- and names every unverified input.

        This used to raise unless the caller opted out.  A refusal that every
        run opts out of teaches people to type the flag, not to read the
        reason, so the message is now unconditional and free.  What must not
        regress is that the message still names each unverified input: a
        warning that omits one is worse than no warning.
        """

        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.rig.assert_deployable()
        printed = buffer.getvalue()
        self.assertTrue(self.rig.unverified_inputs(),
                        "test is vacuous once everything is measured")
        for name in self.rig.unverified_inputs():
            self.assertIn(name, printed)

        silenced = io.StringIO()
        with contextlib.redirect_stdout(silenced):
            self.rig.assert_deployable(acknowledge_candidates=True)
        self.assertEqual(silenced.getvalue(), "")

    def test_q6_has_no_default(self) -> None:
        with self.assertRaises(TypeError):
            self.rig.t_base_hand()
        with self.assertRaisesRegex(ValueError, "joint-6"):
            self.rig.t_base_hand(None)

    def test_palm_frame_actually_turns_with_q6(self) -> None:
        # If this were ever a no-op, the "q6 is a candidate" warning would be
        # pointless and someone would rightly delete it.
        a = self.rig.t_base_hand(25.0)
        b = self.rig.t_base_hand(35.0)
        self.assertFalse(np.allclose(a, b))

    def test_hand_camera_inverts_consistently(self) -> None:
        for camera in ("main", "side"):
            hand_camera = self.rig.t_hand_camera(camera, 25.0)
            rebuilt = self.rig.t_base_hand(25.0) @ hand_camera
            np.testing.assert_allclose(
                rebuilt, self.rig.T_BASE_CAMERA[camera], atol=1e-12
            )

    def test_simulated_scene_is_not_the_deployed_rig(self) -> None:
        # scene_contract's camera is a rendered one with a placeholder palm.
        # Merging them would silently change what test_vision validates.
        from Deploy.vision import sim_aruco

        self.assertFalse(
            np.allclose(sim_aruco.T_BASE_CAMERA, self.rig.T_BASE_CAMERA_MAIN)
        )
        self.assertEqual(sim_aruco.HAND_PALM_HEIGHT_TEMP_M, 0.15)


class PackageLayerTests(unittest.TestCase):
    """Imports must run one way: run -> backends -> policy -> common.

    This is not tidiness.  The three environments hold different packages --
    ``wuji_mujoco`` has mujoco and no wujihandpy, ``wuji_hw`` the reverse, and
    the vision trackers need cv2 plus pyrealsense2.  A backwards import is how a
    contract module ends up dragging in a simulator, which breaks the hardware
    environment at import time rather than at the point of use.

    Checked by reading source, so it holds for modules this environment cannot
    import at all.
    """

    PACKAGE = Path(__file__).resolve().parents[1]
    #: What each layer is permitted to import from.
    ALLOWED = {
        "common": set(),
        "policy": {"common"},
        "vision": {"common"},
        "backends": {"common", "policy", "vision"},
        "run": {"common", "policy", "vision", "backends", "models"},
        "models": set(),
    }
    #: Third-party packages a layer may not name at module level.
    FORBIDDEN_IMPORTS = {
        "common": ("mujoco", "wujihandpy", "cv2", "pyrealsense2"),
        "policy": ("mujoco", "wujihandpy", "cv2", "pyrealsense2"),
    }

    def _modules(self, layer):
        for path in sorted((self.PACKAGE / layer).glob("*.py")):
            if path.name == "__init__.py":
                continue
            yield path

    def test_layer_imports_run_one_way(self) -> None:
        pattern = re.compile(r"^\s*from \.\.(\w+)[. ]", re.M)
        for layer, allowed in self.ALLOWED.items():
            for path in self._modules(layer):
                for other in set(pattern.findall(path.read_text(encoding="utf-8"))):
                    self.assertIn(
                        other, allowed,
                        f"{layer}/{path.name} imports from {other}/, which "
                        f"{layer}/ may not depend on.",
                    )

    def test_contract_and_policy_name_no_simulator_or_sdk(self) -> None:
        for layer, forbidden in self.FORBIDDEN_IMPORTS.items():
            for path in self._modules(layer):
                text = path.read_text(encoding="utf-8")
                for name in forbidden:
                    self.assertNotRegex(
                        text, rf"^\s*(import|from) {name}\b",
                        f"{layer}/{path.name} imports {name} at module level.",
                    )

    def test_every_layer_is_a_package(self) -> None:
        for layer in self.ALLOWED:
            self.assertTrue((self.PACKAGE / layer / "__init__.py").is_file(), layer)

    def test_the_real_backend_never_reaches_the_simulator(self) -> None:
        # The rule that actually broke a run once: wuji_hw has no mujoco.
        for name in ("real_wuji.py", "real_wuji_scheduler.py", "real_wuji_backend.py"):
            text = (self.PACKAGE / "backends" / name).read_text(encoding="utf-8")
            self.assertNotRegex(text, r"^\s*(import|from) mujoco\b")
            self.assertNotIn("mujoco_wuji", text)


class StageTimerTests(unittest.TestCase):
    """The latency accounting itself, since a wrong timer misdirects debugging."""

    def setUp(self) -> None:
        from Deploy.common.timing import StageTimer

        self.StageTimer = StageTimer

    def test_aggregate_label_is_not_named_as_the_bottleneck(self) -> None:
        # "total" is the wrapper; ranking it would always win and say nothing.
        t = self.StageTimer(budget_ms=33.3)
        for _ in range(5):
            t.record("total", 20.0)
            t.record("slow_bit", 15.0)
            t.record("fast_bit", 1.0)
        self.assertEqual(t.slowest()[0], "slow_bit")

    def test_unaccounted_time_is_reported(self) -> None:
        # The finding this module exists for: parts summing well under the
        # total means time is going somewhere nobody instrumented.
        t = self.StageTimer(budget_ms=33.3)
        for _ in range(5):
            t.record("total", 12.0)
            t.record("known", 1.0)
        self.assertAlmostEqual(t.unaccounted_ms(), 11.0, places=6)
        self.assertIn("미계상", t.report())

    def test_a_timer_with_only_a_total_reports_no_gap(self) -> None:
        # Nothing to account for, so "unaccounted == total" would be noise.
        t = self.StageTimer(budget_ms=33.3)
        for _ in range(5):
            t.record("total", 12.0)
        self.assertIsNone(t.unaccounted_ms())
        self.assertNotIn("미계상", t.report())

    def test_gauges_are_kept_apart_from_durations(self) -> None:
        # A fast sample() over a stale frame is fast and wrong at once; only a
        # gauge can say so, and it must not be summed as if it were a duration.
        t = self.StageTimer(budget_ms=33.3)
        t.record("total", 5.0)
        t.record("work", 5.0)
        t.gauge("frame_age_ms", 90.0)
        self.assertAlmostEqual(t.unaccounted_ms(), 0.0, places=6)
        self.assertNotEqual(t.slowest()[0], "frame_age_ms")
        self.assertIn("게이지", t.report())

    def test_non_finite_gauges_are_dropped(self) -> None:
        t = self.StageTimer()
        t.gauge("x", float("nan"))
        t.gauge("x", None)
        self.assertEqual(t.stats("x"), {})

    def test_over_budget_counts_only_the_late_ticks(self) -> None:
        t = self.StageTimer(budget_ms=10.0)
        for value in (5.0, 9.9, 10.1, 50.0):
            t.record("total", value)
        self.assertEqual(t.over_budget(), 2)

    def test_csv_row_matches_csv_columns(self) -> None:
        t = self.StageTimer()
        t.record("a", 1.0)
        t.gauge("g", 2.0)
        self.assertEqual(len(t.csv_columns()), len(t.csv_row()))
        self.assertEqual(t.csv_columns(), ("ms_a", "g_g"))


class SlewGuardTests(unittest.TestCase):
    """The guard must not second-guess the action scale.

    Its job is catching what the contract cannot -- a corrupt graph, a
    mis-shaped observation, a decode bug asking for a large jump.  Set below the
    contract's own per-joint step it rejects CORRECT output instead, which is
    how a first hardware run stopped with finger5_joint3 asking for a perfectly
    legal 0.1326 rad against a 0.05 allowance inherited from the four-joint
    reach task.
    """

    def setUp(self) -> None:
        from Deploy.backends.real_wuji import _step_limit

        self._step_limit = _step_limit

    def test_contract_scale_is_accepted_as_a_per_joint_limit(self) -> None:
        limit = self._step_limit(ACTION_SCALE_RAD)
        np.testing.assert_allclose(limit, ACTION_SCALE_RAD)
        self.assertEqual(limit.shape, (ACTION_DIM,))

    def test_a_scalar_still_works(self) -> None:
        self.assertEqual(self._step_limit(0.05).ndim, 0)

    def test_wrong_length_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._step_limit(np.zeros(4, dtype=np.float32))

    def test_the_default_guard_permits_every_legal_policy_step(self) -> None:
        # A full-scale action in either direction is legal by construction.
        # If the default guard rejects any of it, the guard is wrong.
        from Deploy.policy.action_adapter import decode_policy_action

        limits = OBSERVATION_NORMALIZATION_LIMITS
        q = ((limits[:, 0] + limits[:, 1]) / 2).astype(np.float32)
        for sign in (+1.0, -1.0):
            decoded = decode_policy_action(
                q, np.full(ACTION_DIM, sign, dtype=np.float32)
            )
            step = np.abs(decoded.position_target - q)
            self.assertTrue(
                np.all(step <= ACTION_SCALE_RAD + 1e-5),
                f"legal step exceeds the contract scale: {step}",
            )

    def test_the_entry_point_defaults_to_the_contract_scale(self) -> None:
        text = (Path(__file__).resolve().parents[1]
                / "run/run_hand_policy_real.py").read_text(encoding="utf-8")
        self.assertIn("ACTION_SCALE_RAD if args.max_step_rad is None", text)
        self.assertIn('"--max-step-rad", type=float, default=None', text)
