"""Isaac-free tests for deterministic Auto Tune behavior."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from pd_tuner.autotune.coarse_to_fine import CoarseToFineSearch
from pd_tuner.autotune.controller import AutoTuneController
from pd_tuner.autotune.config import (
    AutoTuneDirection,
    AutoTunePreset,
    AutoTuneRequest,
    AutoTuneTorquePolicy,
    JointTuningContext,
    resolve_autotune_config,
)
from pd_tuner.autotune.metrics import (
    TrialMetrics,
    TrialSample,
    evaluate_trial,
    hard_failure_reasons_for_sample,
)
from pd_tuner.autotune.identification import estimate_candidate_response
from pd_tuner.autotune.ranking import rank_candidates, resolve_ranking_weights
from pd_tuner.autotune.result import CandidateResult, CandidateSpec
from pd_tuner.autotune.serialization import build_autotune_document_from_payload


def context() -> JointTuningContext:
    return JointTuningContext(
        joint_name="joint_a",
        joint_index=2,
        actuator_group="implicit",
        lower_limit=-1.0,
        upper_limit=1.0,
        velocity_limit=3.0,
        current_position=0.0,
        current_kp=1.0,
        current_kd=0.05,
        current_effort_limit=0.2,
        physics_dt=0.01,
    )


def config(**values):
    return resolve_autotune_config(AutoTuneRequest(**values), context())


def trial(
    *,
    direction: str = "positive",
    settling: float | None = 0.2,
    overshoot: float = 1.0,
    steady: float = 0.0005,
    rms: float = 0.05,
    peak: float = 0.1,
    velocity: float = 1.0,
    hard: bool = True,
) -> TrialMetrics:
    sample = TrialSample(0.1, 0.1, 0.1, velocity, peak, peak)
    return TrialMetrics(
        direction=direction,
        repeat_index=0,
        requested_target=0.1,
        applied_target=0.1,
        target_clamped=False,
        actual_step_amplitude=0.1,
        settling_time=settling,
        percentage_overshoot=overshoot,
        steady_state_error=steady,
        rms_computed_effort=rms,
        rms_applied_effort=rms,
        peak_computed_effort=peak,
        peak_applied_effort=peak,
        maximum_velocity=velocity,
        saturation_count=0 if hard else 1,
        saturation_ratio=0.0 if hard else 1.0,
        sample_count=1,
        hard_constraint_passed=hard,
        hard_failure_reasons=() if hard else ("torque saturation detected",),
        time_series=(sample,),
    )


def result(
    candidate_id: int,
    kp: float,
    kd: float,
    **trial_values,
) -> CandidateResult:
    return CandidateResult.aggregate(
        CandidateSpec(candidate_id, kp, kd, "coarse"),
        "positive",
        [trial(**trial_values)],
    )


def second_order_trial(
    *,
    kp: float,
    natural_frequency: float = 8.0,
    damping_ratio: float = 0.6,
    kd: float = 0.05,
    duration: float = 2.0,
    dt: float = 0.01,
) -> TrialMetrics:
    """Create an unsaturated analytical second-order response for unit tests."""

    samples = []
    root = (1.0 - damping_ratio * damping_ratio) ** 0.5
    damped_frequency = natural_frequency * root
    steps = round(duration / dt)
    for index in range(steps + 1):
        time_value = index * dt
        response = 1.0 - math.exp(-damping_ratio * natural_frequency * time_value) * (
            math.cos(damped_frequency * time_value)
            + damping_ratio * math.sin(damped_frequency * time_value) / root
        )
        position = 0.1 * response
        samples.append(TrialSample(time_value, 0.1, position, 0.0, 0.05, 0.05))
    return TrialMetrics(
        direction="positive",
        repeat_index=0,
        requested_target=0.1,
        applied_target=0.1,
        target_clamped=False,
        actual_step_amplitude=0.1,
        settling_time=1.0,
        percentage_overshoot=9.5,
        steady_state_error=0.0,
        rms_computed_effort=0.03,
        rms_applied_effort=0.03,
        peak_computed_effort=0.05,
        peak_applied_effort=0.05,
        maximum_velocity=1.0,
        saturation_count=0,
        saturation_ratio=0.0,
        sample_count=len(samples),
        hard_constraint_passed=True,
        hard_failure_reasons=(),
        time_series=tuple(samples),
    )


class ConfigTests(unittest.TestCase):
    def test_blank_defaults_and_kp_auto_max(self) -> None:
        resolved = config()
        self.assertAlmostEqual(resolved.step_amplitude, 0.1)
        self.assertAlmostEqual(resolved.target_settling_time, 0.5)
        self.assertAlmostEqual(resolved.hold_duration, 2.0)
        self.assertAlmostEqual(resolved.settling_tolerance, 0.002)
        self.assertEqual(resolved.repeats, 2)
        self.assertEqual(resolved.search_budget, 12)
        self.assertAlmostEqual(resolved.kp_max, 0.95 * 0.2 / 0.1)
        self.assertGreaterEqual(resolved.kd_max, 4.0 * context().current_kd)
        self.assertEqual(resolved.value_sources["effort_limit"].value, "ASSET")

    def test_direction_trial_sequence(self) -> None:
        self.assertEqual(AutoTuneDirection.POSITIVE.signs, (1,))
        self.assertEqual(AutoTuneDirection.NEGATIVE.signs, (-1,))
        self.assertEqual(AutoTuneDirection.BIDIRECTIONAL.signs, (1, -1))
        resolved = config(direction=AutoTuneDirection.BIDIRECTIONAL)
        self.assertEqual(set(resolved.applied_targets), {"positive", "negative"})

    def test_zero_amplitude_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "too small or zero"):
            config(step_amplitude=0.0)

    def test_kp_min_that_guarantees_initial_saturation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "guarantees initial position-step torque saturation"):
            config(kp_min=2.1, kp_max=3.0)

    def test_convergence_first_expands_kp_range_and_allows_saturating_kp_min(self) -> None:
        resolved = config(
            torque_policy=AutoTuneTorquePolicy.ALLOW_CLIPPING,
            kp_min=2.1,
        )
        self.assertAlmostEqual(resolved.kp_max, 3.0 * 0.2 / 0.1)
        self.assertEqual(resolved.torque_policy, AutoTuneTorquePolicy.ALLOW_CLIPPING)


class MetricTests(unittest.TestCase):
    def test_computed_applied_mismatch_is_saturation(self) -> None:
        sample = TrialSample(0.0, 0.1, 0.0, 0.0, 0.15, 0.10)
        reasons = hard_failure_reasons_for_sample(
            sample,
            effort_limit=0.2,
            torque_match_tolerance=2.0e-5,
            maximum_velocity=3.0,
            lower_limit=-1.0,
            upper_limit=1.0,
        )
        self.assertIn("torque saturation detected", reasons)

    def test_single_saturated_sample_fails_whole_trial(self) -> None:
        samples = [
            TrialSample(0.0, 0.1, 0.0, 0.0, 0.1, 0.1),
            TrialSample(0.1, 0.1, 0.1, 0.0, 0.21, 0.2),
        ]
        metrics = evaluate_trial(
            samples,
            direction="positive",
            repeat_index=0,
            start_position=0.0,
            requested_target=0.1,
            applied_target=0.1,
            target_clamped=False,
            settling_tolerance=0.002,
            settling_hold_time=0.1,
            effort_limit=0.2,
            torque_match_tolerance=2.0e-5,
            maximum_velocity=3.0,
            lower_limit=-1.0,
            upper_limit=1.0,
        )
        self.assertFalse(metrics.hard_constraint_passed)
        self.assertEqual(metrics.saturation_count, 1)

    def test_allow_clipping_reports_saturation_but_keeps_hard_pass(self) -> None:
        samples = [
            TrialSample(0.0, 0.1, 0.0, 0.0, 0.3, 0.2),
            TrialSample(0.1, 0.1, 0.1, 0.0, 0.3, 0.2),
            TrialSample(0.2, 0.1, 0.1, 0.0, 0.0, 0.0),
        ]
        metrics = evaluate_trial(
            samples,
            direction="positive",
            repeat_index=0,
            start_position=0.0,
            requested_target=0.1,
            applied_target=0.1,
            target_clamped=False,
            settling_tolerance=0.002,
            settling_hold_time=0.1,
            effort_limit=0.2,
            torque_match_tolerance=2.0e-5,
            maximum_velocity=3.0,
            lower_limit=-1.0,
            upper_limit=1.0,
            allow_torque_saturation=True,
        )
        self.assertTrue(metrics.hard_constraint_passed)
        self.assertEqual(metrics.saturation_count, 2)
        self.assertGreater(metrics.saturation_ratio, 0.0)

    def test_settling_overshoot_steady_state_and_effort(self) -> None:
        samples = [
            TrialSample(0.0, 0.1, 0.0, 0.8, 0.10, 0.10),
            TrialSample(0.1, 0.1, 0.08, 0.4, 0.08, 0.08),
            TrialSample(0.2, 0.1, 0.11, 0.2, 0.06, 0.06),
            TrialSample(0.3, 0.1, 0.101, 0.02, 0.04, 0.04),
            TrialSample(0.4, 0.1, 0.1005, 0.0, 0.02, 0.02),
        ]
        metrics = evaluate_trial(
            samples,
            direction="positive",
            repeat_index=0,
            start_position=0.0,
            requested_target=0.1,
            applied_target=0.1,
            target_clamped=False,
            settling_tolerance=0.002,
            settling_hold_time=0.1,
            effort_limit=0.2,
            torque_match_tolerance=2.0e-5,
            maximum_velocity=3.0,
            lower_limit=-1.0,
            upper_limit=1.0,
        )
        self.assertAlmostEqual(metrics.settling_time or -1.0, 0.3)
        self.assertAlmostEqual(metrics.percentage_overshoot, 10.0)
        self.assertLess(metrics.steady_state_error, 0.002)
        self.assertAlmostEqual(metrics.peak_applied_effort or 0.0, 0.1)
        self.assertGreater(metrics.rms_applied_effort or 0.0, 0.0)

    def test_bidirectional_uses_worst_case(self) -> None:
        aggregated = CandidateResult.aggregate(
            CandidateSpec(1, 1.0, 0.1, "coarse"),
            "bidirectional",
            [
                trial(direction="positive", settling=0.2, overshoot=1.0, rms=0.04),
                trial(direction="negative", settling=0.4, overshoot=4.0, rms=0.08),
            ],
        )
        self.assertAlmostEqual(aggregated.settling_time or 0.0, 0.4)
        self.assertAlmostEqual(aggregated.percentage_overshoot, 4.0)
        self.assertAlmostEqual(aggregated.rms_applied_effort or 0.0, 0.08)


class RankingAndSearchTests(unittest.TestCase):
    def test_preset_and_custom_weight_normalization(self) -> None:
        balanced, warning = resolve_ranking_weights(config())
        self.assertAlmostEqual(sum(balanced.values()), 1.0)
        self.assertIsNone(warning)
        custom_cfg = config(
            preset=AutoTunePreset.CUSTOM,
            custom_weights={"settling_time": 2.0, "overshoot": 1.0},
        )
        custom, warning = resolve_ranking_weights(custom_cfg)
        self.assertAlmostEqual(custom["settling_time"], 2.0 / 3.0)
        self.assertIsNone(warning)
        zero_cfg = config(preset=AutoTunePreset.CUSTOM)
        _weights, warning = resolve_ranking_weights(zero_cfg)
        self.assertIn("Balanced", warning or "")
        convergence, warning = resolve_ranking_weights(
            config(preset=AutoTunePreset.CONVERGENCE_FIRST)
        )
        self.assertGreater(convergence["settling_time"], convergence["rms_applied_effort"])
        self.assertIsNone(warning)

    def test_feasible_and_low_torque_preset_selection(self) -> None:
        cfg = config(
            preset=AutoTunePreset.LOW_TORQUE,
            maximum_overshoot=5.0,
            maximum_steady_state_error=0.002,
        )
        high_torque = result(1, 1.0, 0.1, settling=0.15, rms=0.15, peak=0.18)
        low_torque = result(2, 0.8, 0.1, settling=0.25, rms=0.03, peak=0.06)
        outcome = rank_candidates([high_torque, low_torque], cfg)
        self.assertTrue(outcome.fully_feasible)
        self.assertEqual(outcome.selected.candidate.candidate_id, 2)

    def test_fallback_fastest_settled_then_smallest_error(self) -> None:
        cfg = config(target_settling_time=0.1)
        slow = result(1, 1.0, 0.1, settling=0.4)
        less_slow = result(2, 1.1, 0.1, settling=0.3)
        outcome = rank_candidates([slow, less_slow], cfg)
        self.assertFalse(outcome.fully_feasible)
        self.assertEqual(outcome.selected.candidate.candidate_id, 2)
        no_settle_a = result(3, 0.5, 0.0, settling=None, steady=0.03)
        no_settle_b = result(4, 0.7, 0.0, settling=None, steady=0.01)
        outcome = rank_candidates([no_settle_a, no_settle_b], cfg)
        self.assertEqual(outcome.selected.candidate.candidate_id, 4)
        self.assertIn("No candidate settled", outcome.selection_reason)

    def test_hard_failed_candidate_never_selected(self) -> None:
        safe = result(1, 0.5, 0.0, settling=None, steady=0.02)
        unsafe = result(2, 2.0, 0.0, settling=0.01, hard=False)
        outcome = rank_candidates([safe, unsafe], config())
        self.assertEqual(outcome.selected.candidate.candidate_id, 1)

    def test_budget_cap_and_deterministic_generation(self) -> None:
        cfg = config(search_budget=17)
        first = CoarseToFineSearch(cfg).coarse_candidates()
        second = CoarseToFineSearch(cfg).coarse_candidates()
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 17)
        measured = [
            CandidateResult.aggregate(item, "positive", [trial(settling=None, steady=0.01)])
            for item in first
        ]
        fine = CoarseToFineSearch(cfg).fine_candidates(measured[-1], measured)
        self.assertLessEqual(len(first) + len(fine), 17)
        self.assertEqual(
            len({(item.kp, item.kd) for item in first + fine}),
            len(first + fine),
        )

    def test_response_guided_search_does_safe_kp_probes_first(self) -> None:
        cfg = config(search_budget=20)
        strategy = CoarseToFineSearch(cfg)
        kp_candidates = strategy.coarse_candidates()
        self.assertTrue(kp_candidates)
        self.assertLessEqual(len(kp_candidates), 4)
        self.assertTrue(all(item.stage == "kp_probe" for item in kp_candidates))
        self.assertTrue(all(item.kd == cfg.kd_min for item in kp_candidates))
        self.assertTrue(
            all((item.predicted_effort_fraction or 0.0) <= 0.95 + 1.0e-12 for item in kp_candidates)
        )
        fractions = [item.predicted_effort_fraction for item in kp_candidates]
        self.assertEqual(fractions, sorted(fractions))

        measured = [
            CandidateResult.aggregate(
                item,
                "positive",
                [second_order_trial(kp=item.kp, kd=item.kd)],
            )
            for item in kp_candidates
        ]
        model_candidates = strategy.damping_candidates(measured, measured)
        self.assertTrue(model_candidates)
        self.assertTrue(all(item.stage.startswith("identified_") for item in model_candidates))
        self.assertTrue(all(item.model_estimate for item in model_candidates))
        self.assertTrue(
            all((item.predicted_effort_fraction or 0.0) <= 0.95 + 1.0e-12 for item in model_candidates)
        )

    def test_second_order_identification_recovers_known_response(self) -> None:
        candidate = CandidateSpec(1, 2.0, 0.05, "kp_probe")
        measured = CandidateResult.aggregate(
            candidate,
            "positive",
            [second_order_trial(kp=2.0, kd=0.05, natural_frequency=8.0, damping_ratio=0.6)],
        )
        estimate = estimate_candidate_response(measured)
        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertAlmostEqual(estimate.natural_frequency, 8.0, delta=0.8)
        self.assertAlmostEqual(estimate.damping_ratio, 0.6, delta=0.1)
        self.assertAlmostEqual(estimate.effective_inertia, 2.0 / 64.0, delta=0.008)

    def test_controller_identifies_only_after_safe_kp_results(self) -> None:
        cfg = config(search_budget=12)
        controller = AutoTuneController(cfg)
        initial = list(controller.pending)
        self.assertTrue(initial)
        for expected in initial:
            candidate = controller.next_candidate()
            self.assertEqual(candidate, expected)
            controller.add_result(
                CandidateResult.aggregate(
                    candidate,
                    "positive",
                    [second_order_trial(kp=candidate.kp, kd=candidate.kd)],
                )
            )
        calculated = controller.next_candidate()
        self.assertIsNotNone(calculated)
        self.assertTrue(calculated.stage.startswith("identified_"))
        self.assertIsNotNone(calculated.model_estimate)

    def test_adaptive_correction_raises_kp_or_kd_from_measured_failure(self) -> None:
        cfg = config(search_budget=12, maximum_overshoot=5.0)
        strategy = CoarseToFineSearch(cfg)
        slow = result(1, 0.8, 0.02, settling=None, steady=0.01)
        kp_fix = strategy.fine_candidates(slow, [slow])
        self.assertTrue(kp_fix)
        self.assertEqual(kp_fix[0].stage, "adaptive_kp")
        self.assertGreater(kp_fix[0].kp, slow.kp)

        oscillatory = result(1, 0.8, 0.02, settling=0.2, overshoot=20.0)
        kd_fix = strategy.fine_candidates(oscillatory, [oscillatory])
        self.assertTrue(kd_fix)
        self.assertEqual(kd_fix[0].stage, "adaptive_kd")
        self.assertAlmostEqual(kd_fix[0].kp, oscillatory.kp)
        self.assertGreater(kd_fix[0].kd, oscillatory.kd)

    def test_full_staged_sequence_is_deterministic_and_budget_capped(self) -> None:
        def run_ids() -> list[tuple[float, float, str]]:
            cfg = config(search_budget=17)
            controller = AutoTuneController(cfg)
            sequence: list[tuple[float, float, str]] = []
            while True:
                candidate = controller.next_candidate()
                if candidate is None:
                    break
                sequence.append((candidate.kp, candidate.kd, candidate.stage))
                controller.add_result(
                    CandidateResult.aggregate(candidate, "positive", [trial(settling=0.3)])
                )
            self.assertLessEqual(len(sequence), cfg.search_budget)
            self.assertIn("kp_probe", {item[2] for item in sequence})
            self.assertIn("damping_probe", {item[2] for item in sequence})
            return sequence

        self.assertEqual(run_ids(), run_ids())

    def test_json_round_trip(self) -> None:
        cfg = config()
        outcome = rank_candidates([result(1, 1.0, 0.05)], cfg)
        payload = build_autotune_document_from_payload(
            metadata={"asset_file": "/tmp/robot.py", "asset_cfg_name": "ROBOT", "physics_dt": 0.01},
            version_info={"isaac_sim": "test", "isaac_lab": "test"},
            resolved_configuration=cfg.to_dict(),
            outcome=outcome.to_dict(),
            original_gains={"stiffness": 1.0, "damping": 0.05, "effort_limit": 0.2},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded, payload)
        self.assertEqual(loaded["schema"], "isaaclab_pd_autotune_v1")


if __name__ == "__main__":
    unittest.main()
