"""Unit tests for the isolated 2026-08-13 hand_final compatibility path."""

from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from Deploy.common.perception import PoseState, StickPosePair7D
from Deploy.common.fingertip_fk import ISAAC_URDF
from Deploy.common.isaac_reset import read_pregrasp_from_env_yaml
from Deploy.common.policy_contract import COMMAND_TARGET_LIMITS, OBSERVATION_DIM
from Deploy.policy.action_adapter import decode_policy_action
from Deploy.policy.grasp_policy_contract import (
    CURRENT_105D_CONTRACT,
    LEGACY_HAND_FINAL_101D_CONTRACT,
    grasp_contract_for_dimensions,
    load_grasp_policy,
)
from Deploy.policy.legacy_hand_final_101 import (
    LEGACY_ACTION_SCALE_RAD,
    LEGACY_DEPLOY_TARGET_LIMITS,
    LEGACY_OBSERVATION_DIM,
    LEGACY_OBSERVATION_NORMALIZATION_LIMITS,
    LEGACY_OBSERVATION_SLICES,
    LEGACY_PREGRASP_JOINT_POSITIONS_RAD,
    LegacyHandFinal101ObservationAdapter,
    decode_legacy_hand_final_action,
    legacy_observation_csv_columns,
    normalize_legacy_joint_positions,
    pose7d_to_legacy_directed_axis,
)
from Deploy.policy.observation_adapter import PolicyObservationAdapter
from Deploy.policy.policy_runner import PolicyRunner


class _FakeFingertipFK:
    def fingertip_positions_in_palm(self, q):
        q = np.asarray(q, dtype=np.float32)
        return (np.arange(15, dtype=np.float32) * 0.001 + q[0]).astype(np.float32)


class _SequenceStickProvider:
    representation = "StickPose7D"

    def __init__(self, samples):
        self.samples = list(samples)
        self.index = 0

    def reset(self):
        self.index = 0

    def sample(self):
        sample = self.samples[min(self.index, len(self.samples) - 1)]
        self.index += 1
        return sample


def _stick_sample(stick1, stick2, timestamp=1.0):
    return StickPosePair7D(
        np.asarray(stick1, dtype=np.float32),
        np.asarray(stick2, dtype=np.float32),
        timestamp,
        PoseState.VALID,
    )


class _Backend:
    def __init__(self):
        self.q = np.zeros(20, dtype=np.float32)
        self.target = None
        self.stop_reason = None

    def read_joint_positions(self):
        return self.q.copy()

    def write_joint_position_targets(self, target):
        self.target = np.asarray(target, dtype=np.float32).copy()

    def safe_stop(self, reason):
        self.stop_reason = reason


class _OnesPolicy:
    def infer(self, observation):
        return np.ones(20, dtype=np.float32)


class LegacyObservationTests(unittest.TestCase):
    def test_layout_is_exactly_101d_without_gaps(self):
        self.assertEqual(LEGACY_OBSERVATION_DIM, 101)
        covered = []
        for term in LEGACY_OBSERVATION_SLICES.values():
            covered.extend(range(term.start, term.stop))
        self.assertEqual(covered, list(range(101)))
        self.assertEqual(len(legacy_observation_csv_columns()), 101)

    def test_frozen_normalization_table_matches_old_local_urdf(self):
        root = ET.parse(ISAAC_URDF.path).getroot()
        by_name = {
            joint.attrib["name"]: (
                float(joint.find("limit").attrib["lower"]),
                float(joint.find("limit").attrib["upper"]),
            )
            for joint in root.findall("joint")
            if joint.find("limit") is not None
        }
        names = [f"finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]
        actual = np.asarray([by_name[name] for name in names], dtype=np.float32)
        np.testing.assert_array_equal(actual, LEGACY_OBSERVATION_NORMALIZATION_LIMITS)

    def test_reset_and_advance_match_oldest_to_newest_history(self):
        identity = [0.01, 0.02, 0.03, 1.0, 0.0, 0.0, 0.0]
        half = np.sqrt(0.5)
        z90 = [0.04, 0.05, 0.06, half, 0.0, 0.0, half]
        provider = _SequenceStickProvider(
            [
                _stick_sample(identity, identity, 1.0),
                _stick_sample(z90, z90, 2.0),
            ]
        )
        adapter = LegacyHandFinal101ObservationAdapter(
            mode="open", stick_provider=provider, fingertip_fk=_FakeFingertipFK()
        )
        q0 = LEGACY_OBSERVATION_NORMALIZATION_LIMITS[:, 0]
        adapter.reset(q0)
        first = adapter.build()
        np.testing.assert_allclose(first[:20], -1.0, atol=2e-7)
        np.testing.assert_array_equal(first[20:40], first[:20])
        np.testing.assert_allclose(first[55:61], [0.01, 0.02, 0.03, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(first[55:61], first[61:67])
        np.testing.assert_array_equal(first[67:73], first[73:79])
        np.testing.assert_array_equal(first[79:99], np.zeros(20, dtype=np.float32))
        np.testing.assert_array_equal(first[99:101], [1.0, 0.0])

        q1 = LEGACY_OBSERVATION_NORMALIZATION_LIMITS[:, 1]
        action = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
        adapter.advance(q1, action)
        second = adapter.build()
        np.testing.assert_allclose(second[:20], -1.0, atol=2e-7)
        np.testing.assert_allclose(second[20:40], 1.0, atol=2e-7)
        np.testing.assert_allclose(second[55:61], [0.01, 0.02, 0.03, 0.0, 1.0, 0.0])
        np.testing.assert_allclose(second[61:67], [0.04, 0.05, 0.06, -1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_array_equal(second[79:99], action)

    def test_directed_axis_ignores_quaternion_double_cover(self):
        pose = np.asarray([0.1, 0.2, 0.3, 0.5, -0.5, 0.5, -0.5], dtype=np.float32)
        opposite = pose.copy()
        opposite[3:] *= -1.0
        np.testing.assert_allclose(
            pose7d_to_legacy_directed_axis(pose),
            pose7d_to_legacy_directed_axis(opposite),
            atol=1e-7,
        )

    def test_neutral_mode_is_rejected_for_old_actor(self):
        provider = _SequenceStickProvider(
            [_stick_sample([0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0])]
        )
        with self.assertRaisesRegex(ValueError, "not neutral"):
            LegacyHandFinal101ObservationAdapter(
                mode="neutral", stick_provider=provider, fingertip_fk=_FakeFingertipFK()
            )


class LegacyActionTests(unittest.TestCase):
    def test_uniform_scale_and_common_hardware_target_limits(self):
        np.testing.assert_allclose(LEGACY_ACTION_SCALE_RAD, 0.1, atol=1e-8)
        np.testing.assert_array_equal(LEGACY_DEPLOY_TARGET_LIMITS, COMMAND_TARGET_LIMITS)
        q = np.zeros(20, dtype=np.float32)
        raw = np.full(20, -0.5, dtype=np.float32)
        decoded = decode_legacy_hand_final_action(q, raw)
        np.testing.assert_allclose(decoded.unclamped_target, -0.05, atol=1e-7)
        expected = np.clip(
            decoded.unclamped_target,
            LEGACY_DEPLOY_TARGET_LIMITS[:, 0],
            LEGACY_DEPLOY_TARGET_LIMITS[:, 1],
        )
        np.testing.assert_array_equal(decoded.position_target, expected)
        np.testing.assert_allclose(decoded.position_target, -0.05, atol=1e-7)

    def test_runner_injects_legacy_decoder_but_current_default_is_unchanged(self):
        sample = _stick_sample(
            [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]
        )
        legacy_adapter = LegacyHandFinal101ObservationAdapter(
            mode="open",
            stick_provider=_SequenceStickProvider([sample]),
            fingertip_fk=_FakeFingertipFK(),
        )
        backend = _Backend()
        legacy_runner = PolicyRunner(
            backend,
            _OnesPolicy(),
            legacy_adapter,
            action_decoder=decode_legacy_hand_final_action,
        )
        legacy_runner.reset()
        decoded = legacy_runner.command()
        np.testing.assert_allclose(decoded.unclamped_target, 0.1, atol=1e-7)
        np.testing.assert_array_equal(backend.target, decoded.position_target)

        current_adapter = PolicyObservationAdapter(
            mode="open",
            stick_provider=_SequenceStickProvider([sample]),
            fingertip_fk=_FakeFingertipFK(),
        )
        current_runner = PolicyRunner(_Backend(), _OnesPolicy(), current_adapter)
        self.assertIs(current_runner.action_decoder, decode_policy_action)


class ContractSelectionTests(unittest.TestCase):
    def test_dimension_selection_is_unambiguous(self):
        self.assertIs(
            grasp_contract_for_dimensions(101, 20), LEGACY_HAND_FINAL_101D_CONTRACT
        )
        self.assertIs(
            grasp_contract_for_dimensions(OBSERVATION_DIM, 20), CURRENT_105D_CONTRACT
        )
        with self.assertRaisesRegex(RuntimeError, "Unsupported grasp actor"):
            grasp_contract_for_dimensions(103, 20)

    def test_legacy_contract_owns_saved_reset(self):
        self.assertIsNone(CURRENT_105D_CONTRACT.default_pregrasp)
        np.testing.assert_array_equal(
            LEGACY_HAND_FINAL_101D_CONTRACT.default_pregrasp,
            LEGACY_PREGRASP_JOINT_POSITIONS_RAD,
        )
        self.assertAlmostEqual(float(LEGACY_PREGRASP_JOINT_POSITIONS_RAD[10]), 1.6298730373)

        env_yaml = (
            Path(__file__).resolve().parents[2]
            / "nrmk_isaaclab_wuji/logs/rsl_rl/hand_final"
            / "2026-08-13_14-15-09(최종)/params/env.yaml"
        )
        if env_yaml.is_file():
            np.testing.assert_array_equal(
                read_pregrasp_from_env_yaml(env_yaml),
                LEGACY_PREGRASP_JOINT_POSITIONS_RAD,
            )

    def test_saved_2026_08_13_graph_auto_selects_legacy_contract(self):
        graph = (
            Path(__file__).resolve().parents[2]
            / "nrmk_isaaclab_wuji/logs/rsl_rl/hand_final"
            / "2026-08-13_14-15-09(최종)/exported/policy.onnx"
        )
        if not graph.is_file():
            self.skipTest("Archived 2026-08-13 ONNX is not present.")
        try:
            policy, contract = load_grasp_policy(graph)
        except RuntimeError as exc:
            if "onnxruntime is required" in str(exc):
                self.skipTest(str(exc))
            raise
        self.assertIs(contract, LEGACY_HAND_FINAL_101D_CONTRACT)
        self.assertEqual(policy.input.shape, [1, 101])
        self.assertEqual(policy.output.shape, [1, 20])

    def test_current_105d_graph_still_selects_active_contract(self):
        graph = (
            Path(__file__).resolve().parents[1]
            / "models/hand_real_2026-08-18_23-57-25_model4500.onnx"
        )
        if not graph.is_file():
            self.skipTest("Current 105D ONNX is not present.")
        try:
            policy, contract = load_grasp_policy(graph)
        except RuntimeError as exc:
            if "onnxruntime is required" in str(exc):
                self.skipTest(str(exc))
            raise
        self.assertIs(contract, CURRENT_105D_CONTRACT)
        self.assertEqual(policy.input.shape, [1, 105])
        self.assertEqual(policy.output.shape, [1, 20])


if __name__ == "__main__":
    unittest.main()
