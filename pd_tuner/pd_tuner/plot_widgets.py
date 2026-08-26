"""External pyqtgraph widgets for real-time actuator telemetry."""

from __future__ import annotations

from typing import Any

import pyqtgraph as pg
from PySide6 import QtWidgets

from .data_buffer import TimeSeriesBuffer


class TelemetryPlots(QtWidgets.QWidget):
    """Three vertically stacked plots with physically distinct units."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=False)
        layout = QtWidgets.QVBoxLayout(self)
        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics)

        self.position_plot = self.graphics.addPlot(row=0, col=0, title="Joint position")
        self.position_plot.setLabel("left", "Joint position", units="rad")
        self.position_plot.setLabel("bottom", "Simulation time", units="s")
        self.position_plot.showGrid(x=True, y=True, alpha=0.25)
        self.position_plot.addLegend()
        self.target_curve = self.position_plot.plot(pen=pg.mkPen("#f0a020", width=2), name="target_position")
        self.position_curve = self.position_plot.plot(pen=pg.mkPen("#40a0ff", width=2), name="actual_position")

        self.velocity_plot = self.graphics.addPlot(row=1, col=0, title="Joint velocity")
        self.velocity_plot.setLabel("left", "Joint velocity", units="rad/s")
        self.velocity_plot.setLabel("bottom", "Simulation time", units="s")
        self.velocity_plot.showGrid(x=True, y=True, alpha=0.25)
        self.velocity_plot.addLegend()
        self.velocity_curve = self.velocity_plot.plot(pen=pg.mkPen("#60d080", width=2), name="actual_joint_velocity")
        self.velocity_zero = self.velocity_plot.plot(pen=pg.mkPen("#888888", style=pg.QtCore.Qt.PenStyle.DashLine), name="zero")

        self.effort_plot = self.graphics.addPlot(row=2, col=0, title="Joint effort")
        self.effort_plot.setLabel("left", "Joint effort", units="N·m")
        self.effort_plot.setLabel("bottom", "Simulation time", units="s")
        self.effort_plot.showGrid(x=True, y=True, alpha=0.25)
        self.effort_plot.addLegend()
        self.computed_curve = self.effort_plot.plot(
            pen=pg.mkPen("#d060ff", width=2), name="computed_effort (PD estimate)"
        )
        self.applied_curve = self.effort_plot.plot(
            pen=pg.mkPen("#ff5050", width=2), name="applied_effort (clipped)"
        )
        limit_pen = pg.mkPen("#d0d0d0", style=pg.QtCore.Qt.PenStyle.DashLine)
        self.positive_limit_curve = self.effort_plot.plot(pen=limit_pen, name="+effort_limit")
        self.negative_limit_curve = self.effort_plot.plot(pen=limit_pen, name="-effort_limit")

    def clear(self) -> None:
        """Clear every curve without changing axes or legends."""

        for curve in (
            self.target_curve,
            self.position_curve,
            self.velocity_curve,
            self.velocity_zero,
            self.computed_curve,
            self.applied_curve,
            self.positive_limit_curve,
            self.negative_limit_curve,
        ):
            curve.setData([], [])

    def update_from_buffer(self, buffer: TimeSeriesBuffer) -> None:
        """Refresh plot curves from the bounded graph history."""

        arrays: dict[str, Any] = buffer.arrays()
        times = arrays["time"]
        self.target_curve.setData(times, arrays["target"])
        self.position_curve.setData(times, arrays["position"])
        self.velocity_curve.setData(times, arrays["velocity"])
        self.velocity_zero.setData(times, [0.0] * len(times))
        self.computed_curve.setData(times, arrays["computed_effort"])
        self.applied_curve.setData(times, arrays["applied_effort"])
        self.positive_limit_curve.setData(times, arrays["effort_limit_positive"])
        self.negative_limit_curve.setData(times, arrays["effort_limit_negative"])
