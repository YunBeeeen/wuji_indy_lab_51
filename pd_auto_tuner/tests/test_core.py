"""Simulator- and Qt-independent unit tests for portable core logic."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from pd_tuner.asset_loader import discover_asset_files
from pd_tuner.gain_io import build_gain_document, load_json, save_json
from pd_tuner.metrics import StepResponseTracker
from pd_tuner.step_signal import PeriodicStepSignal


class StepSignalTests(unittest.TestCase):
    def test_square_wave_and_clamp(self) -> None:
        signal = PeriodicStepSignal(amplitude=0.3, period=2.0, initial_delay=0.5, direction=1)
        signal.restart(10.0, q0=0.9, active=True)
        self.assertEqual(signal.sample(10.25, -1.0, 1.0).phase, "delay_low")
        self.assertEqual(signal.sample(10.75, -1.0, 1.0).phase, "low")
        high = signal.sample(11.75, -1.0, 1.0)
        self.assertEqual(high.phase, "high")
        self.assertAlmostEqual(high.requested_target, 1.2)
        self.assertAlmostEqual(high.applied_target, 1.0)
        self.assertTrue(high.clamped)


class MetricTests(unittest.TestCase):
    def test_negative_direction_rise_and_overshoot(self) -> None:
        tracker = StepResponseTracker(0.2)
        tracker.begin_transition(0.0, position=1.0, target=0.8)
        for index, position in enumerate((0.98, 0.9, 0.82, 0.79, 0.8)):
            tracker.update(0.1 * (index + 1), position, -0.5, -0.2, False)
        result = tracker.finalize(0.5)
        assert result is not None
        self.assertIsNotNone(result.rise_time)
        self.assertAlmostEqual(result.percentage_overshoot, 5.0, places=5)
        self.assertTrue(math.isfinite(result.steady_state_error))


class PersistenceTests(unittest.TestCase):
    def test_gain_round_trip_and_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "robot.py"
            asset.write_text("ROBOT = object()\n", encoding="utf-8")
            (root / "__init__.py").write_text("", encoding="utf-8")
            self.assertEqual(discover_asset_files(root), [asset])
            document = build_gain_document(
                str(asset), "ROBOT_CFG", 1.0 / 120.0, {"joint": {"stiffness": 1.0, "damping": 0.1, "effort_limit": 2.0}}
            )
            path = save_json(root / "gains.json", document)
            self.assertEqual(load_json(path), document)


if __name__ == "__main__":
    unittest.main()
