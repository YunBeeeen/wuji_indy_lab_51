"""Bounded graph history and streaming CSV persistence."""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import asdict
from pathlib import Path
import shutil
from typing import Any

from .messages import TelemetryPacket


CSV_FIELDS = (
    "simulation_time",
    "joint_name",
    "target_position",
    "actual_position",
    "joint_velocity",
    "position_error",
    "computed_effort",
    "applied_effort",
    "measured_joint_effort",
    "stiffness",
    "damping",
    "effort_limit",
    "saturated",
)


class TimeSeriesBuffer:
    """Timestamp-trimmed telemetry ring buffer for real-time plotting."""

    def __init__(self, history_seconds: float = 10.0) -> None:
        self.history_seconds = float(history_seconds)
        self.samples: deque[TelemetryPacket] = deque()

    def set_history_seconds(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Graph history must be positive.")
        self.history_seconds = float(value)
        self._trim()

    def append(self, packet: TelemetryPacket) -> None:
        if self.samples and packet.joint_name != self.samples[-1].joint_name:
            self.clear()
        self.samples.append(packet)
        self._trim()

    def _trim(self) -> None:
        if not self.samples:
            return
        cutoff = self.samples[-1].simulation_time - self.history_seconds
        while self.samples and self.samples[0].simulation_time < cutoff:
            self.samples.popleft()

    def clear(self) -> None:
        self.samples.clear()

    def arrays(self) -> dict[str, list[float]]:
        """Return plot-ready Python arrays without requiring NumPy in core code."""

        result: dict[str, list[float]] = {
            "time": [],
            "target": [],
            "position": [],
            "velocity": [],
            "computed_effort": [],
            "applied_effort": [],
            "effort_limit_positive": [],
            "effort_limit_negative": [],
        }
        for sample in self.samples:
            result["time"].append(sample.simulation_time)
            result["target"].append(sample.target_position)
            result["position"].append(sample.actual_position)
            result["velocity"].append(sample.joint_velocity)
            result["computed_effort"].append(
                float("nan") if sample.computed_effort is None else sample.computed_effort
            )
            result["applied_effort"].append(
                float("nan") if sample.applied_effort is None else sample.applied_effort
            )
            result["effort_limit_positive"].append(sample.effort_limit)
            result["effort_limit_negative"].append(-sample.effort_limit)
        return result


class CsvStreamLogger:
    """Write full-session telemetry incrementally instead of retaining it in RAM."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._rows_since_flush = 0

    def append(self, packet: TelemetryPacket) -> None:
        values = asdict(packet)
        self._writer.writerow({name: values.get(name) for name in CSV_FIELDS})
        self._rows_since_flush += 1
        if self._rows_since_flush >= 100:
            self._handle.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def export(self, destination: Path) -> Path:
        self._handle.flush()
        target = destination.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path, target)
        return target
