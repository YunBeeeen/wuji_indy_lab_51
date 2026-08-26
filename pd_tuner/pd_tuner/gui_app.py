"""PySide6 parent process for the external actuator tuning window."""

from __future__ import annotations

from datetime import datetime
import math
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Any, Callable

from PySide6 import QtCore, QtGui, QtWidgets

from .asset_loader import default_asset_directories, discover_asset_files
from .asset_inspector import run_asset_inspector
from .data_buffer import CsvStreamLogger, TimeSeriesBuffer
from .gain_io import build_gain_document, load_json, save_json
from .messages import CommandKind, ControlCommand, EventKind, EventPacket, StartConfig, TelemetryPacket
from .plot_widgets import TelemetryPlots
from .simulator_process import run_simulator_process


class SliderSpinBox(QtWidgets.QWidget):
    """Synchronized linear slider and floating-point editor."""

    valueChanged = QtCore.Signal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        decimals: int,
        parent: QtWidgets.QWidget | None = None,
        *,
        logarithmic: bool = True,
        smallest_positive: float = 1.0e-6,
    ) -> None:
        super().__init__(parent)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.logarithmic = bool(logarithmic)
        self.smallest_positive = max(float(smallest_positive), 1.0e-12)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 10000)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(self.minimum, self.maximum)
        self.spin.setSingleStep(max((self.maximum - self.minimum) / 1000.0, 10 ** (-decimals)))
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spin)
        self.slider.valueChanged.connect(self._slider_changed)
        self.spin.valueChanged.connect(self._spin_changed)

    def _slider_changed(self, position: int) -> None:
        value = self._value_from_slider(position)
        with QtCore.QSignalBlocker(self.spin):
            self.spin.setValue(value)
        self.valueChanged.emit(value)

    def _spin_changed(self, value: float) -> None:
        with QtCore.QSignalBlocker(self.slider):
            self.slider.setValue(self._slider_from_value(float(value)))
        self.valueChanged.emit(float(value))

    def _value_from_slider(self, position: int) -> float:
        if not self.logarithmic:
            return self.minimum + (self.maximum - self.minimum) * position / 10000.0
        if self.minimum == 0.0 and position == 0:
            return 0.0
        first = 1 if self.minimum == 0.0 else 0
        ratio = (position - first) / max(10000 - first, 1)
        low = self.smallest_positive if self.minimum == 0.0 else self.minimum
        return math.exp(math.log(low) + ratio * (math.log(self.maximum) - math.log(low)))

    def _slider_from_value(self, value: float) -> int:
        if not self.logarithmic:
            ratio = (value - self.minimum) / max(self.maximum - self.minimum, 1.0e-12)
            return round(10000 * ratio)
        if self.minimum == 0.0 and value <= 0.0:
            return 0
        first = 1 if self.minimum == 0.0 else 0
        low = self.smallest_positive if self.minimum == 0.0 else self.minimum
        clipped = min(max(value, low), self.maximum)
        ratio = (math.log(clipped) - math.log(low)) / (math.log(self.maximum) - math.log(low))
        return round(first + ratio * (10000 - first))

    def setValue(self, value: float) -> None:  # noqa: N802 - Qt naming convention
        self.spin.setValue(min(max(float(value), self.minimum), self.maximum))

    def value(self) -> float:
        return float(self.spin.value())


def _readonly_line() -> QtWidgets.QLineEdit:
    line = QtWidgets.QLineEdit("—")
    line.setReadOnly(True)
    return line


class PdTunerWindow(QtWidgets.QMainWindow):
    """Own the external GUI, IPC queues, and spawned Isaac Sim process."""

    def __init__(self, args: Any, package_root: Path) -> None:
        super().__init__()
        self.args = args
        self.package_root = package_root
        self.outputs_root = (
            Path(args.output_directory).expanduser().resolve()
            if args.output_directory
            else package_root / "outputs"
        )
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        self.setWindowTitle("Isaac Lab Actuator PD Tuner")
        self.resize(1500, 940)

        self.process: mp.Process | None = None
        self.inspector_process: mp.Process | None = None
        self.inspector_queue: Any = None
        self.pending_inspection: tuple[str, str | None] | None = None
        self.active_inspection: tuple[str, str | None] | None = None
        self.inspector_started_wall_time: float | None = None
        self.preferred_asset_cfg_name: str | None = args.asset_cfg_name
        self.control_queue: Any = None
        self.telemetry_queue: Any = None
        self.event_queue: Any = None
        self.shutdown_event: Any = None
        self.command_sequence = 0
        self.metadata: dict[str, Any] = {}
        self.joint_metadata: dict[str, dict[str, Any]] = {}
        self.current_gains: dict[str, dict[str, float]] = {}
        self.graph_buffer = TimeSeriesBuffer(args.history_seconds)
        self.csv_logger: CsvStreamLogger | None = None
        self.last_csv_path: Path | None = None
        self.last_telemetry_wall_time: float | None = None
        self.last_completed_metrics: dict[str, Any] = {}
        self._updating_joint = False

        self._build_ui()
        self._wire_ui()
        self._load_initial_arguments()

        self.gain_debounce = QtCore.QTimer(self)
        self.gain_debounce.setSingleShot(True)
        self.gain_debounce.setInterval(50)
        self.gain_debounce.timeout.connect(self._send_gain_update)
        self.step_debounce = QtCore.QTimer(self)
        self.step_debounce.setSingleShot(True)
        self.step_debounce.setInterval(100)
        self.step_debounce.timeout.connect(self._send_step_configuration)
        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self._poll_ipc)
        self.poll_timer.start(20)
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(lambda: self.plots.update_from_buffer(self.graph_buffer))
        self.plot_timer.start(40)
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self._refresh_communication_status)
        self.status_timer.start(250)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.addWidget(self._build_start_group())
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_control_scroll())
        self.plots = TelemetryPlots()
        splitter.addWidget(self.plots)
        splitter.setSizes([550, 950])
        outer.addWidget(splitter, stretch=1)
        self.statusBar().showMessage("Select a project/asset/config, then start simulation.")

    def _build_start_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Model selection and simulation start")
        grid = QtWidgets.QGridLayout(group)
        self.project_root_edit = QtWidgets.QLineEdit()
        self.project_root_button = QtWidgets.QPushButton("Browse…")
        self.asset_directory_combo = QtWidgets.QComboBox()
        self.asset_directory_combo.setEditable(True)
        self.asset_directory_button = QtWidgets.QPushButton("Browse…")
        self.asset_file_combo = QtWidgets.QComboBox()
        self.asset_file_combo.setEditable(True)
        self.asset_file_button = QtWidgets.QPushButton("File…")
        self.asset_cfg_combo = QtWidgets.QComboBox()
        self.refresh_assets_button = QtWidgets.QPushButton("Refresh")
        self.device_edit = QtWidgets.QComboBox()
        self.device_edit.setEditable(True)
        self.device_edit.addItems(["cuda:0", "cpu"])
        self.physics_dt_spin = QtWidgets.QDoubleSpinBox()
        self.physics_dt_spin.setDecimals(8)
        self.physics_dt_spin.setRange(0.00001, 0.1)
        self.physics_dt_spin.setValue(float(self.args.physics_dt))
        self.initial_pose_combo = QtWidgets.QComboBox()
        self.initial_pose_combo.addItems(["asset_default", "zeros", "joint_limit_midpoint"])
        self.effort_override_check = QtWidgets.QCheckBox("Override")
        self.effort_override_spin = QtWidgets.QDoubleSpinBox()
        self.effort_override_spin.setRange(0.000001, 1.0e7)
        self.effort_override_spin.setDecimals(6)
        self.effort_override_spin.setEnabled(False)
        self.start_sim_button = QtWidgets.QPushButton("Start Simulation")
        self.start_sim_button.setStyleSheet("font-weight: bold")
        self.start_sim_button.setEnabled(False)

        rows = [
            ("Project root", self.project_root_edit, self.project_root_button),
            ("Asset directory", self.asset_directory_combo, self.asset_directory_button),
            ("Asset Python file", self.asset_file_combo, self.asset_file_button),
            ("ArticulationCfg", self.asset_cfg_combo, self.refresh_assets_button),
        ]
        for row, (label, widget, button) in enumerate(rows):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            grid.addWidget(widget, row, 1, 1, 3)
            grid.addWidget(button, row, 4)
        grid.addWidget(QtWidgets.QLabel("Device"), 0, 5)
        grid.addWidget(self.device_edit, 0, 6)
        grid.addWidget(QtWidgets.QLabel("Physics dt [s]"), 1, 5)
        grid.addWidget(self.physics_dt_spin, 1, 6)
        grid.addWidget(QtWidgets.QLabel("Initial joint pose"), 2, 5)
        grid.addWidget(self.initial_pose_combo, 2, 6)
        grid.addWidget(QtWidgets.QLabel("Global effort limit"), 3, 5)
        effort_row = QtWidgets.QHBoxLayout()
        effort_row.addWidget(self.effort_override_check)
        effort_row.addWidget(self.effort_override_spin)
        grid.addLayout(effort_row, 3, 6)
        grid.addWidget(self.start_sim_button, 0, 7, 4, 1)
        return group

    def _build_control_scroll(self) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.addWidget(self._build_status_group())
        layout.addWidget(self._build_joint_group())
        layout.addWidget(self._build_gain_group())
        layout.addWidget(self._build_step_group())
        layout.addWidget(self._build_metrics_group())
        layout.addWidget(self._build_buttons_group())
        layout.addWidget(self._build_environment_group())
        layout.addStretch(1)
        scroll.setWidget(body)
        scroll.setMinimumWidth(500)
        return scroll

    def _build_status_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Simulation status")
        form = QtWidgets.QFormLayout(group)
        self.sim_state_label = QtWidgets.QLabel("stopped")
        self.sim_time_label = QtWidgets.QLabel("0.000 s")
        self.render_status_label = QtWidgets.QLabel("—")
        self.model_status_label = QtWidgets.QLabel("—")
        self.config_status_label = QtWidgets.QLabel("—")
        self.device_status_label = QtWidgets.QLabel("—")
        self.latency_label = QtWidgets.QLabel("—")
        self.queue_label = QtWidgets.QLabel("—")
        self.last_telemetry_label = QtWidgets.QLabel("never")
        for label, widget in (
            ("State", self.sim_state_label),
            ("Simulation time", self.sim_time_label),
            ("Rendering", self.render_status_label),
            ("Model", self.model_status_label),
            ("Config", self.config_status_label),
            ("Device", self.device_status_label),
            ("IPC latency", self.latency_label),
            ("Queue state", self.queue_label),
            ("Last telemetry", self.last_telemetry_label),
        ):
            form.addRow(label, widget)
        return group

    def _build_joint_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Joint selection")
        form = QtWidgets.QFormLayout(group)
        self.joint_combo = QtWidgets.QComboBox()
        self.joint_index_line = _readonly_line()
        self.actuator_group_line = _readonly_line()
        self.position_limits_line = _readonly_line()
        self.velocity_limit_line = _readonly_line()
        self.original_gains_line = _readonly_line()
        self.current_position_line = _readonly_line()
        self.current_velocity_line = _readonly_line()
        self.current_effort_line = _readonly_line()
        self.saturation_label = QtWidgets.QLabel("—")
        for label, widget in (
            ("Joint", self.joint_combo),
            ("Joint index", self.joint_index_line),
            ("Actuator group", self.actuator_group_line),
            ("Position limits", self.position_limits_line),
            ("Velocity limit", self.velocity_limit_line),
            ("Original Kp / Kd / limit", self.original_gains_line),
            ("Current position", self.current_position_line),
            ("Current velocity", self.current_velocity_line),
            ("Current effort", self.current_effort_line),
            ("Effort status", self.saturation_label),
        ):
            form.addRow(label, widget)
        return group

    def _build_gain_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Gain tuning (acknowledged values only)")
        form = QtWidgets.QFormLayout(group)
        self.kp_control = SliderSpinBox(0.0, 100000.0, 6, smallest_positive=1.0e-4)
        self.kd_control = SliderSpinBox(0.0, 10000.0, 6, smallest_positive=1.0e-5)
        self.effort_control = SliderSpinBox(0.000001, 1000000.0, 6)
        self.applied_gain_label = QtWidgets.QLabel("—")
        form.addRow("Stiffness Kp", self.kp_control)
        form.addRow("Damping Kd", self.kd_control)
        form.addRow("Effort limit", self.effort_control)
        form.addRow("Simulator applied", self.applied_gain_label)
        return group

    def _build_step_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Periodic joint-position step")
        form = QtWidgets.QFormLayout(group)
        self.step_amplitude_spin = QtWidgets.QDoubleSpinBox()
        self.step_amplitude_spin.setRange(0.0, 1000.0)
        self.step_amplitude_spin.setDecimals(6)
        self.step_amplitude_spin.setValue(self.args.step_amplitude)
        self.step_period_spin = QtWidgets.QDoubleSpinBox()
        self.step_period_spin.setRange(0.00001, 3600.0)
        self.step_period_spin.setDecimals(6)
        self.step_period_spin.setValue(self.args.step_period)
        self.initial_delay_spin = QtWidgets.QDoubleSpinBox()
        self.initial_delay_spin.setRange(0.0, 3600.0)
        self.initial_delay_spin.setDecimals(6)
        self.initial_delay_spin.setValue(self.args.initial_delay)
        self.direction_combo = QtWidgets.QComboBox()
        self.direction_combo.addItem("Positive (+)", 1)
        self.direction_combo.addItem("Negative (-)", -1)
        self.repeat_check = QtWidgets.QCheckBox("Repeat")
        self.repeat_check.setChecked(True)
        self.history_spin = QtWidgets.QDoubleSpinBox()
        self.history_spin.setRange(0.5, 3600.0)
        self.history_spin.setValue(self.args.history_seconds)
        self.requested_target_line = _readonly_line()
        self.applied_target_line = _readonly_line()
        self.clamp_status_label = QtWidgets.QLabel("—")
        buttons = QtWidgets.QHBoxLayout()
        self.start_step_button = QtWidgets.QPushButton("Start step")
        self.pause_step_button = QtWidgets.QPushButton("Pause step")
        self.restart_step_button = QtWidgets.QPushButton("Restart step")
        buttons.addWidget(self.start_step_button)
        buttons.addWidget(self.pause_step_button)
        buttons.addWidget(self.restart_step_button)
        for label, widget in (
            ("Amplitude [rad]", self.step_amplitude_spin),
            ("Period [s]", self.step_period_spin),
            ("Initial delay [s]", self.initial_delay_spin),
            ("Direction", self.direction_combo),
            ("Repeat", self.repeat_check),
            ("Graph history [s]", self.history_spin),
            ("Requested target", self.requested_target_line),
            ("Applied target", self.applied_target_line),
            ("Clamp status", self.clamp_status_label),
        ):
            form.addRow(label, widget)
        form.addRow(buttons)
        return group

    def _build_metrics_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Step-response metrics")
        form = QtWidgets.QFormLayout(group)
        self.current_metrics_label = QtWidgets.QLabel("No active transition")
        self.current_metrics_label.setWordWrap(True)
        self.completed_metrics_label = QtWidgets.QLabel("No completed transition")
        self.completed_metrics_label.setWordWrap(True)
        form.addRow("Current step", self.current_metrics_label)
        form.addRow("Last completed", self.completed_metrics_label)
        return group

    def _build_buttons_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Control and persistence")
        grid = QtWidgets.QGridLayout(group)
        self.pause_sim_button = QtWidgets.QPushButton("Pause Simulation")
        self.resume_sim_button = QtWidgets.QPushButton("Resume Simulation")
        self.stop_sim_button = QtWidgets.QPushButton("Stop Simulation")
        self.reset_joint_button = QtWidgets.QPushButton("Reset Selected Joint")
        self.reset_all_button = QtWidgets.QPushButton("Reset All Joints")
        self.clear_graph_button = QtWidgets.QPushButton("Clear Graph")
        self.restore_gains_button = QtWidgets.QPushButton("Restore Original Gains")
        self.apply_group_button = QtWidgets.QPushButton("Apply Gains to Actuator Group")
        self.save_gains_button = QtWidgets.QPushButton("Save Tuned Gains")
        self.export_csv_button = QtWidgets.QPushButton("Export CSV")
        self.save_session_button = QtWidgets.QPushButton("Save Session")
        self.load_session_button = QtWidgets.QPushButton("Load Session")
        buttons = [
            self.pause_sim_button,
            self.resume_sim_button,
            self.stop_sim_button,
            self.reset_joint_button,
            self.reset_all_button,
            self.clear_graph_button,
            self.restore_gains_button,
            self.apply_group_button,
            self.save_gains_button,
            self.export_csv_button,
            self.save_session_button,
            self.load_session_button,
        ]
        for index, button in enumerate(buttons):
            grid.addWidget(button, index // 2, index % 2)
        return group

    def _build_environment_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Environment information / warnings")
        layout = QtWidgets.QVBoxLayout(group)
        self.environment_text = QtWidgets.QPlainTextEdit()
        self.environment_text.setReadOnly(True)
        self.environment_text.setMaximumBlockCount(500)
        self.environment_text.setMinimumHeight(130)
        layout.addWidget(self.environment_text)
        return group

    def _wire_ui(self) -> None:
        self.project_root_button.clicked.connect(self._browse_project_root)
        self.project_root_edit.editingFinished.connect(self._refresh_project_root)
        self.asset_directory_button.clicked.connect(self._browse_asset_directory)
        self.asset_file_button.clicked.connect(self._browse_asset_file)
        self.refresh_assets_button.clicked.connect(self._refresh_asset_configs)
        self.asset_directory_combo.currentIndexChanged.connect(
            lambda _index: self._refresh_asset_files(self.asset_directory_combo.currentText())
        )
        self.asset_directory_combo.lineEdit().editingFinished.connect(
            lambda: self._refresh_asset_files(self.asset_directory_combo.currentText())
        )
        self.asset_file_combo.currentIndexChanged.connect(
            lambda _index: self._refresh_asset_configs(self.asset_file_combo.currentText())
        )
        self.asset_file_combo.lineEdit().editingFinished.connect(self._refresh_asset_configs)
        self.effort_override_check.toggled.connect(self.effort_override_spin.setEnabled)
        self.start_sim_button.clicked.connect(self._start_simulation)
        self.joint_combo.currentIndexChanged.connect(self._joint_changed)
        for control in (self.kp_control, self.kd_control, self.effort_control):
            control.valueChanged.connect(lambda _value: self.gain_debounce.start())
        for control in (self.step_amplitude_spin, self.step_period_spin, self.initial_delay_spin):
            control.valueChanged.connect(lambda _value: self.step_debounce.start())
        self.direction_combo.currentIndexChanged.connect(lambda _index: self.step_debounce.start())
        self.repeat_check.toggled.connect(lambda _checked: self.step_debounce.start())
        self.history_spin.valueChanged.connect(self.graph_buffer.set_history_seconds)
        self.start_step_button.clicked.connect(self._start_step)
        self.pause_step_button.clicked.connect(lambda: self._send_command(CommandKind.PAUSE_STEP))
        self.restart_step_button.clicked.connect(self._restart_step)
        self.pause_sim_button.clicked.connect(lambda: self._send_command(CommandKind.PAUSE_SIMULATION))
        self.resume_sim_button.clicked.connect(lambda: self._send_command(CommandKind.RESUME_SIMULATION))
        self.stop_sim_button.clicked.connect(self._stop_simulation)
        self.reset_joint_button.clicked.connect(lambda: self._send_command(CommandKind.RESET_SELECTED_JOINT))
        self.reset_all_button.clicked.connect(lambda: self._send_command(CommandKind.RESET_ALL_JOINTS))
        self.clear_graph_button.clicked.connect(self._clear_graph)
        self.restore_gains_button.clicked.connect(lambda: self._send_command(CommandKind.RESTORE_ORIGINAL_GAINS))
        self.apply_group_button.clicked.connect(self._apply_gains_to_group)
        self.save_gains_button.clicked.connect(self._save_gains)
        self.export_csv_button.clicked.connect(self._export_csv)
        self.save_session_button.clicked.connect(self._save_session)
        self.load_session_button.clicked.connect(self._load_session_dialog)

    def _load_initial_arguments(self) -> None:
        if self.args.session:
            self._apply_session(load_json(self.args.session))
            return
        project_root = Path(self.args.project_root).expanduser().resolve() if self.args.project_root else Path.cwd()
        self.project_root_edit.setText(str(project_root))
        directories = default_asset_directories(project_root)
        with QtCore.QSignalBlocker(self.asset_directory_combo), QtCore.QSignalBlocker(self.asset_file_combo):
            self.asset_directory_combo.addItems(str(path) for path in directories)
            if self.args.asset_directory:
                self.asset_directory_combo.setCurrentText(str(Path(self.args.asset_directory).expanduser().resolve()))
            if self.args.asset_file:
                self.asset_file_combo.setCurrentText(str(Path(self.args.asset_file).expanduser().resolve()))
        if self.args.asset_file:
            self._refresh_asset_configs(str(Path(self.args.asset_file).expanduser().resolve()))
        elif self.args.asset_directory:
            self._refresh_asset_files(self.asset_directory_combo.currentText())
        elif directories:
            self._refresh_asset_files(str(directories[0]))
        self.device_edit.setCurrentText(self.args.device)
        if self.args.effort_limit is not None:
            self.effort_override_check.setChecked(True)
            self.effort_override_spin.setValue(self.args.effort_limit)

    def _browse_project_root(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select project root", self.project_root_edit.text())
        if directory:
            self.project_root_edit.setText(directory)
            self._refresh_project_root()

    @QtCore.Slot()
    def _refresh_project_root(self) -> None:
        """Rescan conventional asset directories after the project root changes."""

        root_text = self.project_root_edit.text().strip()
        if not root_text:
            return
        root = Path(root_text).expanduser()
        if not root.is_dir():
            self.statusBar().showMessage(f"Project root does not exist: {root}", 5000)
            return
        candidates = default_asset_directories(root)
        with QtCore.QSignalBlocker(self.asset_directory_combo):
            self.asset_directory_combo.clear()
            self.asset_directory_combo.addItems(str(path) for path in candidates)
        if candidates:
            self.asset_directory_combo.setCurrentText(str(candidates[0]))
            self._refresh_asset_files(str(candidates[0]))
        else:
            self.asset_file_combo.clear()
            self.asset_cfg_combo.clear()
            self.start_sim_button.setEnabled(False)
            self.statusBar().showMessage(
                "No conventional asset directory found; use Asset directory or File…",
                5000,
            )

    def _browse_asset_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select asset directory", self.asset_directory_combo.currentText()
        )
        if directory:
            self.asset_directory_combo.setCurrentText(directory)
            self._refresh_asset_files(directory)

    def _browse_asset_file(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select asset Python file", self.asset_directory_combo.currentText(), "Python files (*.py)"
        )
        if filename:
            self.asset_file_combo.setCurrentText(filename)
            self._refresh_asset_configs(filename)

    @QtCore.Slot(str)
    def _refresh_asset_files(self, text: str | None = None) -> None:
        directory = Path(text or self.asset_directory_combo.currentText()).expanduser()
        files = discover_asset_files(directory)
        current = self.asset_file_combo.currentText()
        with QtCore.QSignalBlocker(self.asset_file_combo):
            self.asset_file_combo.clear()
            self.asset_file_combo.addItems(str(path) for path in files)
            if current in [str(path) for path in files]:
                self.asset_file_combo.setCurrentText(current)
        if self.asset_file_combo.count():
            self._refresh_asset_configs(self.asset_file_combo.currentText())

    @QtCore.Slot()
    @QtCore.Slot(str)
    def _refresh_asset_configs(self, _text: str | None = None) -> None:
        filename = self.asset_file_combo.currentText().strip()
        if not filename or not Path(filename).is_file():
            return
        request = (filename, self.project_root_edit.text().strip() or None)
        if self.inspector_process is not None and self.inspector_process.is_alive():
            if request == self.active_inspection:
                self.statusBar().showMessage("This asset is already being inspected.")
                return
            self.pending_inspection = request
            self.statusBar().showMessage("Asset inspection is busy; the newest selection is queued.")
            return
        self._start_asset_inspection(*request)

    def _start_asset_inspection(self, filename: str, project_root: str | None) -> None:
        context = mp.get_context("spawn")
        self.inspector_queue = context.Queue(maxsize=2)
        self.inspector_process = context.Process(
            target=run_asset_inspector,
            args=(filename, project_root, self.inspector_queue),
            name="pd-tuner-asset-inspector",
        )
        self.inspector_process.start()
        self.active_inspection = (filename, project_root)
        self.inspector_started_wall_time = time.time()
        self.asset_cfg_combo.clear()
        self.asset_cfg_combo.addItem("Inspecting with Isaac Lab…")
        self.start_sim_button.setEnabled(False)
        self.statusBar().showMessage(f"Inspecting {Path(filename).name} in an isolated Isaac process…")

    def _poll_asset_inspector(self) -> None:
        if self.inspector_process is None:
            return
        if (
            self.inspector_process.is_alive()
            and self.inspector_started_wall_time is not None
            and time.time() - self.inspector_started_wall_time > 60.0
        ):
            self.inspector_process.terminate()
            self.inspector_process.join(timeout=5.0)
            self.asset_cfg_combo.clear()
            self.start_sim_button.setEnabled(False)
            self.environment_text.appendPlainText(
                "[ASSET IMPORT ERROR] Inspector timed out after 60 seconds. "
                "Check the child console and project root."
            )
        result = None
        if self.inspector_queue is not None:
            try:
                result = self.inspector_queue.get_nowait()
            except queue.Empty:
                pass
        if result is not None:
            current = self.asset_cfg_combo.currentText()
            preferred = self.preferred_asset_cfg_name or (
                current if current != "Inspecting with Isaac Lab…" else None
            )
            self.asset_cfg_combo.clear()
            if result["ok"]:
                for summary in result["summaries"]:
                    self.asset_cfg_combo.addItem(summary["name"], summary)
                if preferred:
                    index = self.asset_cfg_combo.findText(preferred)
                    if index >= 0:
                        self.asset_cfg_combo.setCurrentIndex(index)
                        self.preferred_asset_cfg_name = None
                if not result["summaries"]:
                    self.environment_text.appendPlainText(
                        f"No ArticulationCfg objects found in {result['asset_file']}"
                    )
                self.statusBar().showMessage(
                    f"Found {len(result['summaries'])} ArticulationCfg object(s).", 5000
                )
            else:
                self.environment_text.appendPlainText(f"[ASSET IMPORT ERROR] {result['error']}")
                self.statusBar().showMessage("Asset import failed; see Environment information.", 10000)
            self.start_sim_button.setEnabled(bool(result["ok"] and result["summaries"]))
        if not self.inspector_process.is_alive():
            self.inspector_process.join(timeout=0.1)
            self.inspector_process.close()
            self.inspector_process = None
            self.inspector_queue = None
            self.active_inspection = None
            self.inspector_started_wall_time = None
            if self.pending_inspection is not None:
                request = self.pending_inspection
                self.pending_inspection = None
                self._start_asset_inspection(*request)

    def _stop_asset_inspector(self) -> None:
        if self.inspector_process is None:
            return
        self.inspector_process.join(timeout=15.0)
        if self.inspector_process.is_alive():
            self.inspector_process.terminate()
            self.inspector_process.join(timeout=5.0)
        self.inspector_process.close()
        self.inspector_process = None
        self.inspector_queue = None
        self.active_inspection = None
        self.inspector_started_wall_time = None
        self.pending_inspection = None

    def _make_start_config(self) -> StartConfig:
        filename = self.asset_file_combo.currentText().strip()
        cfg_name = self.asset_cfg_combo.currentText().strip()
        if not filename or not Path(filename).is_file():
            raise ValueError("Select a valid asset Python file.")
        if not cfg_name:
            raise ValueError("Select an ArticulationCfg.")
        return StartConfig(
            project_root=self.project_root_edit.text().strip() or None,
            asset_file=filename,
            asset_cfg_name=cfg_name,
            device=self.device_edit.currentText().strip(),
            physics_dt=float(self.physics_dt_spin.value()),
            render=not self.args.headless,
            headless=bool(self.args.headless),
            initial_pose_mode=self.initial_pose_combo.currentText(),
            effort_limit_override=(
                float(self.effort_override_spin.value()) if self.effort_override_check.isChecked() else None
            ),
            selected_joint=self.args.joint,
            step_amplitude=float(self.step_amplitude_spin.value()),
            step_period=float(self.step_period_spin.value()),
            initial_delay=float(self.initial_delay_spin.value()),
            step_direction=int(self.direction_combo.currentData()),
            repeat_step=self.repeat_check.isChecked(),
            gain_config=self.args.gain_config,
        )

    def _start_simulation(self) -> None:
        self._stop_asset_inspector()
        if self.process is not None:
            self._stop_simulation(wait=True)
        try:
            config = self._make_start_config()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Cannot start simulation", str(exc))
            return
        context = mp.get_context("spawn")
        self.control_queue = context.Queue(maxsize=256)
        self.telemetry_queue = context.Queue(maxsize=4096)
        self.event_queue = context.Queue(maxsize=256)
        self.shutdown_event = context.Event()
        self.process = context.Process(
            target=run_simulator_process,
            args=(config, self.control_queue, self.telemetry_queue, self.event_queue, self.shutdown_event),
            name="pd-tuner-isaac-sim",
        )
        self.process.start()
        self.sim_state_label.setText("starting")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_csv_path = self.outputs_root / "sessions" / f"pd_tuning_{stamp}.csv"
        self.csv_logger = CsvStreamLogger(self.last_csv_path)
        self.statusBar().showMessage("Isaac Sim child is starting…")

    def _stop_simulation(self, _checked: bool = False, *, wait: bool = False) -> None:
        if self.process is None:
            return
        if self.shutdown_event is not None:
            self.shutdown_event.set()
        self._send_command(CommandKind.STOP)
        if wait:
            self.process.join(timeout=15.0)
            if self.process.is_alive():
                self.environment_text.appendPlainText(
                    "Simulation did not finish cleanup in 15 s; terminating the child process."
                )
                self.process.terminate()
                self.process.join(timeout=5.0)
            self._finish_process()

    def _finish_process(self) -> None:
        if self.process is not None:
            with contextlib_suppress(Exception):
                self.process.close()
        self.process = None
        if self.csv_logger is not None:
            self.csv_logger.close()
            self.csv_logger = None
        if self.sim_state_label.text() != "error":
            self.sim_state_label.setText("stopped")
        self.shutdown_event = None

    def _send_command(self, kind: CommandKind, payload: dict[str, Any] | None = None) -> None:
        if self.control_queue is None or self.process is None or not self.process.is_alive():
            return
        self.command_sequence += 1
        command = ControlCommand(
            kind=kind,
            payload=payload or {},
            sequence=self.command_sequence,
            wall_time_sent=time.time(),
        )
        try:
            self.control_queue.put_nowait(command)
        except queue.Full:
            self.statusBar().showMessage("Control queue is full; command was not sent.", 5000)

    def _poll_ipc(self) -> None:
        self._poll_asset_inspector()
        if self.event_queue is not None:
            for _ in range(200):
                try:
                    packet: EventPacket = self.event_queue.get_nowait()
                except queue.Empty:
                    break
                self._handle_event(packet)
        if self.telemetry_queue is not None:
            newest: TelemetryPacket | None = None
            for _ in range(2000):
                try:
                    packet = self.telemetry_queue.get_nowait()
                except queue.Empty:
                    break
                newest = packet
                self.graph_buffer.append(packet)
                if self.csv_logger is not None:
                    self.csv_logger.append(packet)
            if newest is not None:
                self._handle_telemetry(newest)
        if self.process is not None and not self.process.is_alive():
            exit_code = self.process.exitcode
            if self.sim_state_label.text() not in ("stopped", "error"):
                self.sim_state_label.setText("error" if exit_code else "stopped")
                self.environment_text.appendPlainText(f"Simulation process exited with code {exit_code}.")
            self._finish_process()

    def _handle_event(self, packet: EventPacket) -> None:
        if packet.kind == EventKind.STATE:
            self.sim_state_label.setText(str(packet.payload.get("state", "unknown")))
        elif packet.kind == EventKind.VERSION_INFO:
            self.environment_text.appendPlainText(
                "Environment: " + ", ".join(f"{key}={value}" for key, value in packet.payload.items())
            )
        elif packet.kind == EventKind.MODEL_METADATA:
            self._install_metadata(packet.payload)
        elif packet.kind == EventKind.JOINT_SELECTED:
            if packet.payload.get("clear_graph"):
                self._clear_graph()
            if "joint" in packet.payload:
                name = packet.payload["joint"]["name"]
                with QtCore.QSignalBlocker(self.joint_combo):
                    self.joint_combo.setCurrentIndex(self.joint_combo.findData(name))
                self._display_joint(name)
            if "gains" in packet.payload:
                self._set_gain_controls(packet.payload["gains"])
        elif packet.kind == EventKind.GAIN_APPLIED:
            name = packet.payload["joint_name"]
            applied = packet.payload["applied"]
            self.current_gains[name] = applied
            if self.joint_combo.currentData() == name:
                self._set_gain_controls(applied)
                self.applied_gain_label.setText(
                    f"Kp={applied['stiffness']:.6g}, Kd={applied['damping']:.6g}, "
                    f"limit={applied['effort_limit']:.6g} ({packet.payload['scope']})"
                )
        elif packet.kind == EventKind.STEP_METRICS:
            self.last_completed_metrics = packet.payload.get("last_completed", {})
            self.completed_metrics_label.setText(self._format_metrics(self.last_completed_metrics))
        elif packet.kind in (EventKind.WARNING, EventKind.ERROR):
            level = packet.kind.value.upper()
            self.environment_text.appendPlainText(f"[{level}] {packet.payload.get('message', packet.payload)}")
            if packet.kind == EventKind.ERROR:
                self.statusBar().showMessage(str(packet.payload.get("message", "Simulation error")), 10000)

    def _install_metadata(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.joint_metadata = {joint["name"]: joint for joint in metadata["joints"]}
        self.current_gains = {
            name: dict(joint["original_gain"])
            for name, joint in self.joint_metadata.items()
            if joint["tunable"]
        }
        self.joint_combo.clear()
        for joint in metadata["joints"]:
            if joint["tunable"]:
                group = joint["actuator_group"] or "unmapped"
                self.joint_combo.addItem(f"{joint['name']} [{group}]", joint["name"])
        self.model_status_label.setText(Path(metadata["asset_file"]).name)
        self.config_status_label.setText(metadata["asset_cfg_name"])
        self.device_status_label.setText(metadata["device"])
        self.render_status_label.setText("enabled" if metadata["rendering"] else "headless")
        self.physics_dt_spin.setValue(float(metadata["physics_dt"]))
        if not metadata["fixed_base"]:
            self.environment_text.appendPlainText("[WARNING] Floating-base articulation: base motion can corrupt tuning.")
        self.statusBar().showMessage(f"Loaded {len(metadata['joints'])} joints; simulation running.")

    def _joint_changed(self, index: int) -> None:
        if index < 0 or self._updating_joint:
            return
        name = self.joint_combo.itemData(index)
        if name:
            self._display_joint(name)
            self._send_command(CommandKind.SELECT_JOINT, {"joint_name": name})

    def _display_joint(self, name: str) -> None:
        joint = self.joint_metadata.get(name)
        if not joint:
            return
        self.joint_index_line.setText(str(joint["index"]))
        self.actuator_group_line.setText(f"{joint['actuator_group']} ({joint['actuator_type']})")
        self.position_limits_line.setText(f"[{joint['lower_limit']:.6g}, {joint['upper_limit']:.6g}] rad")
        self.velocity_limit_line.setText(f"{joint['velocity_limit']:.6g} rad/s")
        original = joint["original_gain"]
        self.original_gains_line.setText(
            f"{original['stiffness']:.6g} / {original['damping']:.6g} / {original['effort_limit']:.6g}"
        )
        self._set_gain_controls(self.current_gains.get(name, original))

    def _set_gain_controls(self, gains: dict[str, float]) -> None:
        self._updating_joint = True
        try:
            with QtCore.QSignalBlocker(self.kp_control), QtCore.QSignalBlocker(self.kd_control), QtCore.QSignalBlocker(
                self.effort_control
            ):
                self.kp_control.setValue(gains["stiffness"])
                self.kd_control.setValue(gains["damping"])
                self.effort_control.setValue(gains["effort_limit"])
        finally:
            self._updating_joint = False

    def _send_gain_update(self) -> None:
        if self._updating_joint:
            return
        self._send_command(
            CommandKind.APPLY_GAINS,
            {
                "stiffness": self.kp_control.value(),
                "damping": self.kd_control.value(),
                "effort_limit": self.effort_control.value(),
            },
        )

    def _apply_gains_to_group(self) -> None:
        group = self.actuator_group_line.text()
        answer = QtWidgets.QMessageBox.question(
            self,
            "Apply to actuator group",
            f"Apply the displayed gains to every joint in {group}?",
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._send_command(
                CommandKind.APPLY_GAINS_TO_GROUP,
                {
                    "stiffness": self.kp_control.value(),
                    "damping": self.kd_control.value(),
                    "effort_limit": self.effort_control.value(),
                },
            )

    def _send_step_configuration(self) -> None:
        self._send_command(
            CommandKind.CONFIGURE_STEP,
            {
                "amplitude": float(self.step_amplitude_spin.value()),
                "period": float(self.step_period_spin.value()),
                "initial_delay": float(self.initial_delay_spin.value()),
                "direction": int(self.direction_combo.currentData()),
                "repeat": self.repeat_check.isChecked(),
            },
        )

    def _start_step(self) -> None:
        self._send_step_configuration()
        self._send_command(CommandKind.START_STEP)

    def _restart_step(self) -> None:
        self._send_step_configuration()
        self._send_command(CommandKind.RESTART_STEP)
        self._clear_graph()

    def _handle_telemetry(self, packet: TelemetryPacket) -> None:
        now = time.time()
        self.last_telemetry_wall_time = now
        self.sim_time_label.setText(f"{packet.simulation_time:.3f} s")
        self.current_position_line.setText(f"{packet.actual_position:+.6f} rad")
        self.current_velocity_line.setText(f"{packet.joint_velocity:+.6f} rad/s")
        effort_parts = []
        if packet.computed_effort is not None:
            effort_parts.append(f"computed={packet.computed_effort:+.5g}")
        if packet.applied_effort is not None:
            effort_parts.append(f"applied={packet.applied_effort:+.5g}")
        self.current_effort_line.setText(", ".join(effort_parts) + " N·m" if effort_parts else "unavailable")
        self.saturation_label.setText("SATURATED" if packet.saturated else "not saturated")
        self.saturation_label.setStyleSheet("color: red; font-weight: bold" if packet.saturated else "")
        self.requested_target_line.setText(f"{packet.requested_target_position:+.6f} rad")
        self.applied_target_line.setText(f"{packet.target_position:+.6f} rad")
        self.clamp_status_label.setText("CLAMPED TO JOINT LIMIT" if packet.clamp_status else "not clamped")
        self.current_metrics_label.setText(self._format_metrics(packet.current_metrics))
        latency_ms = max(0.0, (now - packet.wall_time_sent) * 1000.0)
        self.latency_label.setText(f"{latency_ms:.1f} ms")
        self.current_gains[packet.joint_name] = {
            "stiffness": packet.stiffness,
            "damping": packet.damping,
            "effort_limit": packet.effort_limit,
        }

    @staticmethod
    def _format_metrics(metrics: dict[str, Any]) -> str:
        if not metrics:
            return "No active transition"
        def value(key: str, unit: str = "", precision: int = 4) -> str:
            item = metrics.get(key)
            if item is None:
                return "Not settled" if key == "settling_time" else "—"
            return f"{float(item):.{precision}g}{unit}"
        return (
            f"error={value('current_error', ' rad')}, |error|={value('absolute_position_error', ' rad')}, "
            f"max|error|={value('maximum_absolute_error', ' rad')}\n"
            f"peak velocity={value('peak_velocity', ' rad/s')}, peak effort={value('peak_absolute_effort', ' N·m')}, "
            f"overshoot={value('percentage_overshoot', '%')}\n"
            f"rise={value('rise_time', ' s')}, settling={value('settling_time', ' s')}, "
            f"steady error={value('steady_state_error', ' rad')}, saturation={value('effort_saturation_ratio', '', 3)}"
        )

    def _refresh_communication_status(self) -> None:
        if self.last_telemetry_wall_time is None:
            self.last_telemetry_label.setText("never")
        else:
            age = time.time() - self.last_telemetry_wall_time
            self.last_telemetry_label.setText(f"{age:.2f} s ago")
            if age > 3.0 and self.process is not None and self.process.is_alive() and self.sim_state_label.text() == "running":
                self.statusBar().showMessage("Telemetry timeout: no sample for more than 3 seconds.")
        sizes = []
        for label, ipc_queue in (("ctl", self.control_queue), ("tel", self.telemetry_queue), ("evt", self.event_queue)):
            try:
                sizes.append(f"{label}={ipc_queue.qsize() if ipc_queue is not None else 0}")
            except (NotImplementedError, AttributeError):
                sizes.append(f"{label}=?")
        self.queue_label.setText(", ".join(sizes))

    def _clear_graph(self) -> None:
        self.graph_buffer.clear()
        self.plots.clear()

    def _save_gains(self) -> None:
        if not self.metadata:
            return
        default = self.outputs_root / "gains" / f"{Path(self.metadata['asset_file']).stem}_tuned_gains.json"
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save tuned gains", str(default), "JSON (*.json)")
        if not filename:
            return
        document = build_gain_document(
            self.metadata["asset_file"],
            self.metadata["asset_cfg_name"],
            float(self.metadata["physics_dt"]),
            self.current_gains,
        )
        save_json(filename, document)
        self.statusBar().showMessage(f"Saved gains: {filename}", 5000)

    def _export_csv(self) -> None:
        if self.last_csv_path is None:
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export telemetry CSV", str(self.last_csv_path), "CSV (*.csv)")
        if filename:
            if self.csv_logger is not None:
                self.csv_logger.export(Path(filename))
            else:
                import shutil
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.last_csv_path, filename)
            self.statusBar().showMessage(f"Exported CSV: {filename}", 5000)

    def _session_document(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root_edit.text().strip(),
            "asset_directory": self.asset_directory_combo.currentText().strip(),
            "asset_file": self.asset_file_combo.currentText().strip(),
            "asset_cfg_name": self.asset_cfg_combo.currentText().strip(),
            "device": self.device_edit.currentText().strip(),
            "physics_dt": float(self.physics_dt_spin.value()),
            "initial_pose_mode": self.initial_pose_combo.currentText(),
            "effort_limit_override": (
                float(self.effort_override_spin.value()) if self.effort_override_check.isChecked() else None
            ),
            "joint": self.joint_combo.currentData(),
            "step_amplitude": float(self.step_amplitude_spin.value()),
            "step_period": float(self.step_period_spin.value()),
            "initial_delay": float(self.initial_delay_spin.value()),
            "step_direction": int(self.direction_combo.currentData()),
            "repeat_step": self.repeat_check.isChecked(),
            "history_seconds": float(self.history_spin.value()),
        }

    def _save_session(self) -> None:
        default = self.outputs_root / "sessions" / "pd_tuner_session.json"
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save session", str(default), "JSON (*.json)")
        if filename:
            save_json(filename, self._session_document())

    def _load_session_dialog(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load session", str(self.outputs_root), "JSON (*.json)")
        if filename:
            self._apply_session(load_json(filename))

    def _apply_session(self, session: dict[str, Any]) -> None:
        self.project_root_edit.setText(str(session.get("project_root", "")))
        with QtCore.QSignalBlocker(self.asset_directory_combo), QtCore.QSignalBlocker(self.asset_file_combo):
            self.asset_directory_combo.setCurrentText(str(session.get("asset_directory", "")))
            self.asset_file_combo.setCurrentText(str(session.get("asset_file", "")))
        self.preferred_asset_cfg_name = str(session.get("asset_cfg_name", "")) or None
        self._refresh_asset_configs()
        self.device_edit.setCurrentText(str(session.get("device", "cuda:0")))
        self.physics_dt_spin.setValue(float(session.get("physics_dt", self.args.physics_dt)))
        self.initial_pose_combo.setCurrentText(str(session.get("initial_pose_mode", "asset_default")))
        effort = session.get("effort_limit_override")
        self.effort_override_check.setChecked(effort is not None)
        if effort is not None:
            self.effort_override_spin.setValue(float(effort))
        self.args.joint = session.get("joint")
        self.step_amplitude_spin.setValue(float(session.get("step_amplitude", self.args.step_amplitude)))
        self.step_period_spin.setValue(float(session.get("step_period", self.args.step_period)))
        self.initial_delay_spin.setValue(float(session.get("initial_delay", self.args.initial_delay)))
        direction_index = self.direction_combo.findData(int(session.get("step_direction", 1)))
        self.direction_combo.setCurrentIndex(max(direction_index, 0))
        self.repeat_check.setChecked(bool(session.get("repeat_step", True)))
        self.history_spin.setValue(float(session.get("history_seconds", self.args.history_seconds)))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt override
        self._stop_asset_inspector()
        self._stop_simulation(wait=True)
        event.accept()


class contextlib_suppress:
    """Tiny local suppress context to keep GUI import dependencies minimal."""

    def __init__(self, *exceptions: type[BaseException]) -> None:
        self.exceptions = exceptions

    def __enter__(self) -> None:
        return None

    def __exit__(self, exception_type: Any, exception: Any, traceback_object: Any) -> bool:
        return exception_type is not None and issubclass(exception_type, self.exceptions)


def run_gui(args: Any, package_root: Path) -> int:
    """Create and run the external Qt application."""

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PdTunerWindow(args, package_root)
    window.show()
    return int(app.exec())
