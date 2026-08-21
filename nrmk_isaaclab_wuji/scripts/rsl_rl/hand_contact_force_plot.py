"""External real-time hand contact-force plot used by ``play.py``.

The GUI intentionally runs in a separate Python subprocess.  Isaac Sim owns
the parent process and sends newline-delimited JSON samples through stdin; this
module imports no Isaac APIs.  Closing the plot only closes the pipe -- play
continues normally.
"""

from __future__ import annotations

import argparse
import atexit
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


class HandContactPlotPublisher:
    """Non-blocking JSON-lines publisher backed by a bounded writer queue."""

    _STOP = object()

    def __init__(self, history_seconds: float = 10.0, force_threshold_n: float = 0.01):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--history-seconds",
            str(max(float(history_seconds), 1.0)),
            "--force-threshold",
            str(max(float(force_threshold_n), 0.0)),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=2)
        self._closed = False
        self._writer = threading.Thread(
            target=self._writer_loop,
            name="hand-contact-plot-writer",
            daemon=True,
        )
        self._writer.start()
        atexit.register(self.close)

    @property
    def running(self) -> bool:
        """Whether the external graph process is still alive."""
        return not self._closed and self._process.poll() is None

    def publish(
        self,
        simulation_time: float,
        body_names: list[str],
        force_magnitudes_n: list[float],
    ) -> None:
        """Queue the latest sample without ever blocking the simulation loop."""
        if not self.running:
            return
        packet = {
            "type": "sample",
            "simulation_time": float(simulation_time),
            "body_names": list(body_names),
            "force_magnitudes_n": [float(value) for value in force_magnitudes_n],
        }
        try:
            self._queue.put_nowait(packet)
        except queue.Full:
            # Telemetry is disposable; preserve the newest state rather than
            # allowing a stalled GUI to stall PhysX.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(packet)
            except queue.Full:
                pass

    def close(self) -> None:
        """Close stdin and reap the graph process, with a bounded fallback."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(self._STOP)
            except queue.Full:
                pass
        self._writer.join(timeout=1.0)
        if self._process.poll() is None:
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=1.0)

    def _writer_loop(self) -> None:
        stream = self._process.stdin
        if stream is None:
            return
        try:
            while True:
                packet = self._queue.get()
                if packet is self._STOP:
                    break
                stream.write(json.dumps(packet, separators=(",", ":")) + "\n")
                stream.flush()
        except (BrokenPipeError, OSError):
            # The operator may close the plot while play keeps running.
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass


def _run_gui(history_seconds: float, force_threshold_n: float) -> int:
    try:
        import numpy as np
        import pyqtgraph as pg
        from PySide6 import QtCore, QtGui, QtWidgets
    except ModuleNotFoundError as exc:
        print(
            "[hand-contact-plot:error] Missing GUI dependency: "
            f"{exc.name}. Install in the Isaac Lab Python environment with:\n"
            "  python -m pip install PySide6 pyqtgraph numpy",
            file=sys.stderr,
            flush=True,
        )
        return 2

    functional_defaults = {
        "palm_link",
        "finger1_link2",
        "finger1_link3",
        "finger2_tip_link",
        "finger3_tip_link",
        "finger4_tip_link",
    }

    class HandContactForceWindow(QtWidgets.QMainWindow):
        """One magnitude plot with selectable per-link curves plus sum/max."""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Hand Contact Forces — play.py")
            self.resize(1280, 720)
            self._history_seconds = max(float(history_seconds), 1.0)
            self._force_threshold_n = max(float(force_threshold_n), 0.0)
            self._times: list[float] = []
            self._series: dict[str, list[float]] = {}
            self._curves: dict[str, Any] = {}
            self._items: dict[str, QtWidgets.QListWidgetItem] = {}
            self._dirty = False

            root = QtWidgets.QWidget()
            self.setCentralWidget(root)
            layout = QtWidgets.QHBoxLayout(root)

            controls = QtWidgets.QWidget()
            controls.setMaximumWidth(330)
            controls_layout = QtWidgets.QVBoxLayout(controls)
            self._status = QtWidgets.QLabel("Waiting for play.py telemetry …")
            self._status.setWordWrap(True)
            controls_layout.addWidget(self._status)

            history_row = QtWidgets.QHBoxLayout()
            history_row.addWidget(QtWidgets.QLabel("History [s]"))
            self._history_spin = QtWidgets.QDoubleSpinBox()
            self._history_spin.setRange(1.0, 120.0)
            self._history_spin.setDecimals(1)
            self._history_spin.setValue(self._history_seconds)
            self._history_spin.valueChanged.connect(self._set_history)
            history_row.addWidget(self._history_spin)
            controls_layout.addLayout(history_row)

            button_row = QtWidgets.QHBoxLayout()
            all_button = QtWidgets.QPushButton("All")
            none_button = QtWidgets.QPushButton("None")
            active_button = QtWidgets.QPushButton("Active")
            clear_button = QtWidgets.QPushButton("Clear")
            all_button.clicked.connect(lambda: self._set_all_checks(True))
            none_button.clicked.connect(lambda: self._set_all_checks(False))
            active_button.clicked.connect(self._select_active)
            clear_button.clicked.connect(self._clear_history)
            for button in (all_button, none_button, active_button, clear_button):
                button_row.addWidget(button)
            controls_layout.addLayout(button_row)

            controls_layout.addWidget(QtWidgets.QLabel("Per-link curves"))
            self._body_list = QtWidgets.QListWidget()
            self._body_list.itemChanged.connect(self._on_item_changed)
            controls_layout.addWidget(self._body_list, stretch=1)
            self._note = QtWidgets.QLabel(
                "sum |F_link| is the sum of link-force magnitudes, not a vector sum.\n"
                "Signals are contact forces, not actuator torque/root-PD wrench."
            )
            self._note.setWordWrap(True)
            controls_layout.addWidget(self._note)
            layout.addWidget(controls)

            self._plot = pg.PlotWidget(title="Hand-link net contact-force magnitude")
            self._plot.setLabel("left", "Contact force", units="N")
            self._plot.setLabel("bottom", "Simulation time", units="s")
            self._plot.showGrid(x=True, y=True, alpha=0.25)
            self._plot.addLegend(offset=(10, 10))
            self._sum_curve = self._plot.plot(
                name="sum |F_link|",
                pen=pg.mkPen((255, 255, 255), width=3),
            )
            self._max_curve = self._plot.plot(
                name="max |F_link|",
                pen=pg.mkPen((255, 80, 255), width=3),
            )
            layout.addWidget(self._plot, stretch=1)

            self._stdin_notifier = QtCore.QSocketNotifier(
                sys.stdin.fileno(), QtCore.QSocketNotifier.Type.Read
            )
            self._stdin_notifier.activated.connect(self._read_one_packet)
            self._refresh_timer = QtCore.QTimer(self)
            self._refresh_timer.timeout.connect(self._refresh_plot)
            self._refresh_timer.start(50)  # 20 Hz GUI refresh

        def _set_history(self, value: float) -> None:
            self._history_seconds = float(value)
            self._prune_history()
            self._dirty = True

        def _ensure_bodies(self, body_names: list[str]) -> None:
            for body_name in body_names:
                if body_name in self._series:
                    continue
                self._series[body_name] = [0.0] * len(self._times)
                index = len(self._curves)
                pen = pg.mkPen(pg.intColor(index, hues=max(len(body_names), 1)), width=2)
                curve = self._plot.plot(name=body_name, pen=pen)
                self._curves[body_name] = curve
                item = QtWidgets.QListWidgetItem(body_name)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                checked = body_name in functional_defaults
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if checked
                    else QtCore.Qt.CheckState.Unchecked
                )
                curve.setVisible(checked)
                self._items[body_name] = item
                self._body_list.addItem(item)

        @QtCore.Slot()
        def _read_one_packet(self, *_args) -> None:
            line = sys.stdin.readline()
            if not line:
                self._stdin_notifier.setEnabled(False)
                self._status.setText("play.py disconnected — plot will close.")
                QtCore.QTimer.singleShot(150, QtWidgets.QApplication.instance().quit)
                return
            try:
                packet = json.loads(line)
                if packet.get("type") != "sample":
                    return
                body_names = [str(name) for name in packet["body_names"]]
                values = [float(value) for value in packet["force_magnitudes_n"]]
                if len(body_names) != len(values):
                    raise ValueError("body_names/force_magnitudes_n length mismatch")
                self._append_sample(float(packet["simulation_time"]), body_names, values)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._status.setText(f"Malformed telemetry ignored: {exc}")

        def _append_sample(self, time_s: float, body_names: list[str], values: list[float]) -> None:
            self._ensure_bodies(body_names)
            value_map = dict(zip(body_names, values))
            self._times.append(time_s)
            for body_name, series in self._series.items():
                series.append(value_map.get(body_name, 0.0))
            self._prune_history()
            if values:
                max_index = int(np.argmax(values))
                active_count = sum(value >= self._force_threshold_n for value in values)
                self._status.setText(
                    f"t={time_s:.2f}s | max={body_names[max_index]} "
                    f"{values[max_index]:.3f}N | sum={sum(values):.3f}N | "
                    f"active={active_count}/{len(values)}"
                )
            self._latest_values = value_map
            self._dirty = True

        def _prune_history(self) -> None:
            if not self._times:
                return
            cutoff = self._times[-1] - self._history_seconds
            drop_count = 0
            while drop_count < len(self._times) and self._times[drop_count] < cutoff:
                drop_count += 1
            if drop_count:
                del self._times[:drop_count]
                for series in self._series.values():
                    del series[:drop_count]

        def _refresh_plot(self) -> None:
            if not self._dirty or not self._times:
                return
            times = np.asarray(self._times, dtype=float)
            matrix = np.asarray([self._series[name] for name in self._series], dtype=float)
            self._sum_curve.setData(times, matrix.sum(axis=0))
            self._max_curve.setData(times, matrix.max(axis=0))
            for body_name, curve in self._curves.items():
                if curve.isVisible():
                    curve.setData(times, np.asarray(self._series[body_name], dtype=float))
            self._dirty = False

        def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
            curve = self._curves.get(item.text())
            if curve is not None:
                curve.setVisible(item.checkState() == QtCore.Qt.CheckState.Checked)
                self._dirty = True

        def _set_all_checks(self, checked: bool) -> None:
            state = QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked
            for item in self._items.values():
                item.setCheckState(state)

        def _select_active(self) -> None:
            latest = getattr(self, "_latest_values", {})
            for body_name, item in self._items.items():
                state = (
                    QtCore.Qt.CheckState.Checked
                    if latest.get(body_name, 0.0) >= self._force_threshold_n
                    else QtCore.Qt.CheckState.Unchecked
                )
                item.setCheckState(state)

        def _clear_history(self) -> None:
            self._times.clear()
            for series in self._series.values():
                series.clear()
            self._sum_curve.clear()
            self._max_curve.clear()
            for curve in self._curves.values():
                curve.clear()
            self._dirty = False

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Hand Contact Force Plot")
    window = HandContactForceWindow()
    window.show()
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-seconds", type=float, default=10.0)
    parser.add_argument("--force-threshold", type=float, default=0.01)
    args = parser.parse_args()
    return _run_gui(args.history_seconds, args.force_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
