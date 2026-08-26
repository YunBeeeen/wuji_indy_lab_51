"""Auto Tune controls, progress display, and candidate table."""

from __future__ import annotations

import math
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

from ..autotune.config import METRIC_NAMES


class OptionalNumberEdit(QtWidgets.QWidget):
    """Blank-capable numeric field with an explicit value-source badge."""

    changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QtWidgets.QLineEdit()
        self.edit.setClearButtonEnabled(True)
        validator = QtGui.QDoubleValidator(self)
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        self.edit.setValidator(validator)
        self.source = QtWidgets.QLabel("AUTO")
        self.source.setMinimumWidth(58)
        self.source.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.source.setStyleSheet("font-weight: bold; color: #4f81bd")
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.source)
        self.edit.textChanged.connect(self._text_changed)

    def _text_changed(self, text: str) -> None:
        if text.strip():
            self.source.setText("USER")
            self.source.setStyleSheet("font-weight: bold; color: #c08020")
        self.changed.emit()

    def value(self) -> float | None:
        text = self.edit.text().strip()
        if not text:
            return None
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("Auto Tune numeric inputs must be finite.")
        return value

    def set_auto(self, value: float, source: str) -> None:
        self.edit.setPlaceholderText(f"Auto: {value:.7g}")
        if not self.edit.text().strip():
            self.source.setText(source)
            color = "#4f81bd" if source == "AUTO" else "#4f9b58"
            self.source.setStyleSheet(f"font-weight: bold; color: {color}")

    def load_value(self, value: Any, source: str = "SESSION") -> None:
        with QtCore.QSignalBlocker(self.edit):
            self.edit.setText("" if value in (None, "") else f"{float(value):.12g}")
        self.source.setText(source if value not in (None, "") else "AUTO")


class AutoTunePanel(QtWidgets.QScrollArea):
    """Pure GUI presentation; all search and physics remain in the child."""

    configurationChanged = QtCore.Signal()

    COLUMNS = (
        "Rank",
        "Status",
        "Search stage",
        "Predicted effort [%]",
        "Generation reason",
        "Identified wn [rad/s]",
        "Identified zeta",
        "Kp",
        "Kd",
        "Direction",
        "Settling [s]",
        "Overshoot [%]",
        "Steady error [rad]",
        "RMS computed [N·m]",
        "RMS applied [N·m]",
        "Peak computed [N·m]",
        "Peak applied [N·m]",
        "Max velocity [rad/s]",
        "Saturation count",
        "Total score",
        "Failure reason",
    )

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setMinimumWidth(620)
        body = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(body)
        self.fields: dict[str, OptionalNumberEdit] = {}
        self.scalar_sources = {"repeats": "AUTO", "search_budget": "AUTO"}
        self.candidate_rows: list[dict[str, Any]] = []
        self.layout.addWidget(self._build_inputs())
        self.layout.addWidget(self._build_ranking())
        self.layout.addWidget(self._build_resolved())
        self.layout.addWidget(self._build_progress())
        self.layout.addWidget(self._build_results())
        self.layout.addStretch(1)
        self.setWidget(body)

    def _field(self, name: str, form: QtWidgets.QFormLayout, label: str) -> OptionalNumberEdit:
        field = OptionalNumberEdit()
        field.changed.connect(self.configurationChanged)
        self.fields[name] = field
        form.addRow(label, field)
        return field

    def _build_inputs(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Auto Tune inputs (blank = automatic default)")
        form = QtWidgets.QFormLayout(group)
        self.joint_combo = QtWidgets.QComboBox()
        form.addRow("Joint", self.joint_combo)
        self.torque_policy_combo = QtWidgets.QComboBox()
        self.torque_policy_combo.addItem(
            "Strict: reject any saturation", "strict_no_saturation"
        )
        self.torque_policy_combo.addItem(
            "Convergence first: allow actuator clipping", "allow_clipping"
        )
        self.torque_policy_combo.setToolTip(
            "Allow clipping never raises actual applied effort above effort_limit; "
            "it only permits computed_effort to exceed the limit."
        )
        form.addRow("Torque policy", self.torque_policy_combo)
        self._field("effort_limit", form, "Effort limit [N·m]")
        self._field("step_amplitude", form, "Step amplitude [rad]")
        self.direction_combo = QtWidgets.QComboBox()
        self.direction_combo.addItem("Positive only", "positive")
        self.direction_combo.addItem("Negative only", "negative")
        self.direction_combo.addItem("Bidirectional", "bidirectional")
        form.addRow("Direction", self.direction_combo)
        self._field("target_settling_time", form, "Target settling time [s]")
        self._field("hold_duration", form, "Hold duration [s]")
        self._field("settling_tolerance", form, "Settling tolerance [rad]")
        self._field("settling_hold_time", form, "Settling minimum hold [s]")

        self.overshoot_enabled = QtWidgets.QCheckBox("Enforce")
        self.overshoot_enabled.setChecked(True)
        overshoot = self._field("maximum_overshoot", form, "Maximum overshoot [%]")
        overshoot.layout().insertWidget(0, self.overshoot_enabled)
        self.overshoot_enabled.toggled.connect(overshoot.edit.setEnabled)
        self.overshoot_enabled.toggled.connect(overshoot.source.setEnabled)
        self.sse_enabled = QtWidgets.QCheckBox("Enforce")
        self.sse_enabled.setChecked(True)
        sse = self._field("maximum_steady_state_error", form, "Maximum steady-state error [rad]")
        sse.layout().insertWidget(0, self.sse_enabled)
        self.sse_enabled.toggled.connect(sse.edit.setEnabled)
        self.sse_enabled.toggled.connect(sse.source.setEnabled)
        self._field("maximum_velocity", form, "Maximum velocity [rad/s]")

        kp_row = QtWidgets.QWidget()
        kp_layout = QtWidgets.QHBoxLayout(kp_row)
        kp_layout.setContentsMargins(0, 0, 0, 0)
        for name in ("kp_min", "kp_max"):
            field = OptionalNumberEdit()
            field.changed.connect(self.configurationChanged)
            self.fields[name] = field
            kp_layout.addWidget(QtWidgets.QLabel("min" if name.endswith("min") else "max"))
            kp_layout.addWidget(field, 1)
        form.addRow("Kp range", kp_row)
        kd_row = QtWidgets.QWidget()
        kd_layout = QtWidgets.QHBoxLayout(kd_row)
        kd_layout.setContentsMargins(0, 0, 0, 0)
        for name in ("kd_min", "kd_max"):
            field = OptionalNumberEdit()
            field.changed.connect(self.configurationChanged)
            self.fields[name] = field
            kd_layout.addWidget(QtWidgets.QLabel("min" if name.endswith("min") else "max"))
            kd_layout.addWidget(field, 1)
        form.addRow("Kd range", kd_row)

        self.repeats_edit = QtWidgets.QLineEdit()
        self.repeats_edit.setPlaceholderText("Auto: 2")
        self.repeats_edit.setValidator(QtGui.QIntValidator(1, 1000, self))
        self.budget_edit = QtWidgets.QLineEdit()
        self.budget_edit.setPlaceholderText("Auto: 12")
        self.budget_edit.setValidator(QtGui.QIntValidator(1, 100000, self))
        form.addRow("Repeats per direction", self.repeats_edit)
        form.addRow("Search budget", self.budget_edit)
        for widget in (
            self.direction_combo,
            self.torque_policy_combo,
            self.overshoot_enabled,
            self.sse_enabled,
            self.repeats_edit,
            self.budget_edit,
        ):
            signal = widget.currentIndexChanged if isinstance(widget, QtWidgets.QComboBox) else (
                widget.toggled if isinstance(widget, QtWidgets.QCheckBox) else widget.textChanged
            )
            signal.connect(self.configurationChanged)
        self.repeats_edit.textEdited.connect(
            lambda _text: self.scalar_sources.__setitem__("repeats", "USER")
        )
        self.budget_edit.textEdited.connect(
            lambda _text: self.scalar_sources.__setitem__("search_budget", "USER")
        )
        return group

    def _build_ranking(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Candidate selection priority")
        layout = QtWidgets.QVBoxLayout(group)
        self.preset_combo = QtWidgets.QComboBox()
        for label, value in (
            ("Balanced", "balanced"),
            ("Fast response", "fast_response"),
            ("Smooth response", "smooth_response"),
            ("Low torque", "low_torque"),
            ("Convergence first", "convergence_first"),
            ("Custom weights", "custom"),
        ):
            self.preset_combo.addItem(label, value)
        layout.addWidget(self.preset_combo)
        self.weights_group = QtWidgets.QGroupBox("Advanced custom weights (internally normalized)")
        form = QtWidgets.QFormLayout(self.weights_group)
        self.weight_edits: dict[str, QtWidgets.QDoubleSpinBox] = {}
        labels = {
            "settling_time": "Settling time",
            "overshoot": "Overshoot",
            "rms_applied_effort": "RMS applied effort",
            "peak_applied_effort": "Peak applied effort",
            "steady_state_error": "Steady-state error",
            "gain_magnitude": "Gain magnitude",
        }
        for name in METRIC_NAMES:
            edit = QtWidgets.QDoubleSpinBox()
            edit.setRange(0.0, 1.0e6)
            edit.setDecimals(3)
            edit.valueChanged.connect(self.configurationChanged)
            self.weight_edits[name] = edit
            form.addRow(labels[name], edit)
        self.weights_group.setEnabled(False)
        layout.addWidget(self.weights_group)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        return group

    def _preset_changed(self, _index: int) -> None:
        self.weights_group.setEnabled(self.preset_combo.currentData() == "custom")
        if self.preset_combo.currentData() == "convergence_first":
            allow_index = self.torque_policy_combo.findData("allow_clipping")
            if allow_index >= 0:
                self.torque_policy_combo.setCurrentIndex(allow_index)
        self.configurationChanged.emit()

    def _build_resolved(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Resolved Auto Tune Configuration")
        layout = QtWidgets.QVBoxLayout(group)
        self.resolved_text = QtWidgets.QPlainTextEdit()
        self.resolved_text.setReadOnly(True)
        self.resolved_text.setMinimumHeight(180)
        layout.addWidget(self.resolved_text)
        return group

    def _build_progress(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Auto Tune progress")
        layout = QtWidgets.QVBoxLayout(group)
        self.progress_text = QtWidgets.QLabel("Not started")
        self.progress_text.setWordWrap(True)
        layout.addWidget(self.progress_text)
        row = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start Auto Tune")
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.resume_button = QtWidgets.QPushButton("Resume")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        for button in (self.start_button, self.pause_button, self.resume_button, self.cancel_button):
            row.addWidget(button)
        layout.addLayout(row)
        return group

    def _build_results(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Auto Tune candidate results")
        layout = QtWidgets.QVBoxLayout(group)
        self.result_summary = QtWidgets.QLabel("No result")
        self.result_summary.setWordWrap(True)
        layout.addWidget(self.result_summary)
        self.table = QtWidgets.QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(260)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        self.apply_best_button = QtWidgets.QPushButton("Apply Best")
        self.apply_selected_button = QtWidgets.QPushButton("Apply Selected")
        self.restore_button = QtWidgets.QPushButton("Restore Original Gains")
        self.save_button = QtWidgets.QPushButton("Save Auto Tune Result")
        self.export_button = QtWidgets.QPushButton("Export Auto Tune CSV")
        for button in (
            self.apply_best_button,
            self.apply_selected_button,
            self.restore_button,
            self.save_button,
            self.export_button,
        ):
            row.addWidget(button)
        layout.addLayout(row)
        return group

    def request_mapping(self) -> dict[str, Any]:
        """Return raw blank-capable inputs for shared child-side resolution."""

        values = {name: field.value() for name, field in self.fields.items()}
        values.update(
            direction=str(self.direction_combo.currentData()),
            overshoot_enabled=self.overshoot_enabled.isChecked(),
            steady_state_error_enabled=self.sse_enabled.isChecked(),
            repeats=(int(self.repeats_edit.text()) if self.repeats_edit.text().strip() else None),
            search_budget=(int(self.budget_edit.text()) if self.budget_edit.text().strip() else None),
            preset=str(self.preset_combo.currentData()),
            torque_policy=str(self.torque_policy_combo.currentData()),
            custom_weights={name: edit.value() for name, edit in self.weight_edits.items()},
            value_sources={
                **{
                    name: field.source.text()
                    for name, field in self.fields.items()
                    if field.edit.text().strip()
                },
                **(
                    {"repeats": self.scalar_sources["repeats"]}
                    if self.repeats_edit.text().strip()
                    else {}
                ),
                **(
                    {"search_budget": self.scalar_sources["search_budget"]}
                    if self.budget_edit.text().strip()
                    else {}
                ),
            },
        )
        return values

    def load_request(self, values: dict[str, Any]) -> None:
        """Load optional inputs from a session while preserving their source."""

        for name, field in self.fields.items():
            field.load_value(values.get(name), "SESSION")
        direction = self.direction_combo.findData(values.get("direction", "positive"))
        self.direction_combo.setCurrentIndex(max(direction, 0))
        self.overshoot_enabled.setChecked(bool(values.get("overshoot_enabled", True)))
        self.sse_enabled.setChecked(bool(values.get("steady_state_error_enabled", True)))
        self.repeats_edit.setText("" if values.get("repeats") in (None, "") else str(values["repeats"]))
        self.budget_edit.setText(
            "" if values.get("search_budget") in (None, "") else str(values["search_budget"])
        )
        if values.get("repeats") not in (None, ""):
            self.scalar_sources["repeats"] = "SESSION"
        if values.get("search_budget") not in (None, ""):
            self.scalar_sources["search_budget"] = "SESSION"
        preset = self.preset_combo.findData(values.get("preset", "balanced"))
        self.preset_combo.setCurrentIndex(max(preset, 0))
        torque_policy = self.torque_policy_combo.findData(
            values.get("torque_policy", "strict_no_saturation")
        )
        self.torque_policy_combo.setCurrentIndex(max(torque_policy, 0))
        weights = values.get("custom_weights", {})
        for name, edit in self.weight_edits.items():
            edit.setValue(float(weights.get(name, 0.0)))

    def populate_joints(self, joints: list[dict[str, Any]], selected: str | None) -> None:
        with QtCore.QSignalBlocker(self.joint_combo):
            self.joint_combo.clear()
            for joint in joints:
                if joint.get("tunable"):
                    group = joint.get("actuator_group") or "unmapped"
                    self.joint_combo.addItem(f"{joint['name']} [{group}]", joint["name"])
            index = self.joint_combo.findData(selected)
            self.joint_combo.setCurrentIndex(max(index, 0))

    def install_resolved(self, resolved: dict[str, Any]) -> None:
        sources = resolved.get("value_sources", {})
        for name, field in self.fields.items():
            if name in resolved and resolved[name] is not None:
                field.set_auto(float(resolved[name]), str(sources.get(name, "AUTO")))
        lines = [
            f"joint = {resolved.get('joint_name')} [{resolved.get('actuator_group')}]",
            f"q0 = {resolved.get('q0'):.7g} rad",
        ]
        display_order = (
            "effort_limit", "current_kp", "current_kd", "step_amplitude", "target_settling_time", "hold_duration",
            "settling_tolerance", "settling_hold_time", "maximum_overshoot",
            "maximum_steady_state_error", "maximum_velocity", "kp_min", "kp_max",
            "kd_min", "kd_max", "repeats", "search_budget", "torque_match_tolerance",
        )
        for name in display_order:
            value = resolved.get(name)
            source = sources.get(name, "DERIVED")
            lines.append(f"{name} = {value}  [{source}]")
        lines.append(f"direction = {resolved.get('direction')}")
        lines.append(f"torque_policy = {resolved.get('torque_policy')}")
        lines.append(f"requested targets = {resolved.get('requested_targets')}")
        lines.append(f"applied targets = {resolved.get('applied_targets')}")
        lines.append(f"clamped = {resolved.get('target_clamped')}")
        for note in resolved.get("resolution_notes", {}).values():
            lines.append(f"NOTE: {note}")
        self.resolved_text.setPlainText("\n".join(lines))

    def set_progress(self, payload: dict[str, Any]) -> None:
        candidate = payload.get("current_candidate") or payload.get("candidate") or {}
        failures = payload.get("hard_failure_reason") or payload.get("failure_reason") or "—"
        self.progress_text.setText(
            f"state={payload.get('state', 'running')} | candidate "
            f"{payload.get('candidate_number', payload.get('tested_count', 0))}/"
            f"{payload.get('search_budget', '?')} | stage={candidate.get('stage', payload.get('search_stage', '—'))} | "
            f"predicted effort={self._format_percent(candidate.get('predicted_effort_fraction'))}\n"
            f"Kp={candidate.get('kp', '—')} | Kd={candidate.get('kd', '—')}\n"
            f"reason={candidate.get('generation_reason') or '—'}\n"
            f"identified model={candidate.get('model_estimate') or '—'}\n"
            f"direction={payload.get('direction', '—')} | repeat={payload.get('repeat', '—')}/"
            f"{payload.get('repeats', '—')} | sim time={payload.get('elapsed_simulation_time', '—')}\n"
            f"best feasible={payload.get('best_feasible_candidate_id')} | "
            f"best fallback={payload.get('best_fallback_candidate_id')} | "
            f"hard rejected={payload.get('rejected_candidate_count', 0)} | failure={failures}"
        )

    @staticmethod
    def _format(value: Any) -> str:
        if value is None:
            return "Not settled"
        if isinstance(value, float):
            return f"{value:.7g}"
        return str(value)

    @staticmethod
    def _format_percent(value: Any) -> str:
        if value is None:
            return "—"
        return f"{100.0 * float(value):.3g}%"

    def install_outcome(self, outcome: dict[str, Any]) -> None:
        candidates = list(outcome.get("candidates", []))
        selected_id = outcome.get("selected_candidate_id")
        candidates.sort(
            key=lambda item: (
                0 if item["candidate"]["candidate_id"] == selected_id else 1,
                item.get("total_score") if item.get("total_score") is not None else math.inf,
                item["candidate"]["candidate_id"],
            )
        )
        self.candidate_rows = candidates
        self.table.setRowCount(len(candidates))
        for row, result in enumerate(candidates):
            candidate = result["candidate"]
            estimate = candidate.get("model_estimate") or {}
            failure = result.get("hard_failure_reasons") or result.get("performance_violations") or []
            values = (
                row + 1,
                result.get("status"),
                candidate.get("stage"),
                self._format_percent(candidate.get("predicted_effort_fraction")),
                candidate.get("generation_reason"),
                estimate.get("natural_frequency"),
                estimate.get("damping_ratio"),
                candidate.get("kp"),
                candidate.get("kd"),
                result.get("direction"),
                result.get("settling_time"),
                result.get("percentage_overshoot"),
                result.get("steady_state_error"),
                result.get("rms_computed_effort"),
                result.get("rms_applied_effort"),
                result.get("peak_computed_effort"),
                result.get("peak_applied_effort"),
                result.get("maximum_velocity"),
                result.get("saturation_count"),
                result.get("total_score"),
                " | ".join(failure),
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(self._format(value))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, candidate["candidate_id"])
                self.table.setItem(row, column, item)
        feasible = bool(outcome.get("fully_feasible"))
        prefix = "FEASIBLE result" if feasible else "No fully feasible candidate found"
        selected_result = next(
            (
                item
                for item in candidates
                if item["candidate"]["candidate_id"] == selected_id
            ),
            None,
        )
        detail = ""
        if selected_result is not None:
            detail = "\nSatisfied:\n✓ All non-relaxable hard constraints"
            violations = selected_result.get("performance_violations", [])
            if violations:
                detail += "\nViolated:\n" + "\n".join(f"✗ {item}" for item in violations)
            else:
                detail += "\n✓ All requested performance constraints"
        self.result_summary.setText(
            f"{prefix}\n{outcome.get('selection_reason', '')}\n"
            f"Selected candidate: {selected_id}; warning: {outcome.get('ranking_warning') or 'none'}"
            f"{detail}"
        )
        if candidates:
            self.table.selectRow(0)

    def selected_candidate(self) -> dict[str, Any] | None:
        row = self.table.currentRow()
        if 0 <= row < len(self.candidate_rows):
            return self.candidate_rows[row]
        return None
