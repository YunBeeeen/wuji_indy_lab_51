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
        # Isaac (``joint_position_lower_overrides=None``).  Command range must
        # now equal the observation range: the policy was trained with a single
        # clamp derived from the articulation limits.
        np.testing.assert_array_equal(
            COMMAND_TARGET_LIMITS, OBSERVATION_NORMALIZATION_LIMITS
        )
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
        np.testing.assert_allclose(
            MIDDLE_COMMAND_TARGET_LIMITS,
            REAL_HAND_FACTORY_LIMITS[MIDDLE_POLICY_INDICES],
        )
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
        pc.soft_command_limits(0.9)
        np.testing.assert_array_equal(pc.COMMAND_TARGET_LIMITS, before)
        np.testing.assert_array_equal(
            pc.COMMAND_TARGET_LIMITS, pc.OBSERVATION_NORMALIZATION_LIMITS
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

    def test_command_before_reset_is_refused(self) -> None:
        backend = self.CountingBackend(self.q)
        runner = PolicyRunner(backend, self.ZeroPolicy(), PolicyObservationAdapter())
        with self.assertRaisesRegex(RuntimeError, "reset"):
            runner.command()


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

    def test_deployment_gate_refuses_silently_using_candidates(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unverified geometry"):
            self.rig.assert_deployable()
        self.rig.assert_deployable(acknowledge_candidates=True)

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
