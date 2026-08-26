"""Typed IPC messages shared by the GUI and Isaac Sim processes."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any


class CommandKind(str, Enum):
    """Commands accepted by the simulation child process."""

    STOP = "stop"
    PAUSE_SIMULATION = "pause_simulation"
    RESUME_SIMULATION = "resume_simulation"
    SELECT_JOINT = "select_joint"
    APPLY_GAINS = "apply_gains"
    APPLY_GAINS_TO_GROUP = "apply_gains_to_group"
    RESTORE_ORIGINAL_GAINS = "restore_original_gains"
    CONFIGURE_STEP = "configure_step"
    START_STEP = "start_step"
    PAUSE_STEP = "pause_step"
    RESTART_STEP = "restart_step"
    RESET_SELECTED_JOINT = "reset_selected_joint"
    RESET_ALL_JOINTS = "reset_all_joints"


class EventKind(str, Enum):
    """Reliable low-rate events emitted by the simulation process."""

    STATE = "state"
    MODEL_METADATA = "model_metadata"
    GAIN_APPLIED = "gain_applied"
    JOINT_SELECTED = "joint_selected"
    STEP_METRICS = "step_metrics"
    WARNING = "warning"
    ERROR = "error"
    VERSION_INFO = "version_info"


@dataclass(slots=True)
class StartConfig:
    """Immutable configuration used to create one simulation process."""

    project_root: str | None
    asset_file: str
    asset_cfg_name: str
    device: str = "cuda:0"
    physics_dt: float = 1.0 / 120.0
    render: bool = True
    headless: bool = False
    initial_pose_mode: str = "asset_default"
    effort_limit_override: float | None = None
    selected_joint: str | None = None
    step_amplitude: float = 0.2
    step_period: float = 1.0 / 6.0
    initial_delay: float = 0.25
    step_direction: int = 1
    repeat_step: bool = True
    telemetry_hz: float = 100.0
    velocity_safety_threshold: float = 50.0
    saturation_safety_duration: float = 2.0
    gain_config: str | None = None

    def normalized(self) -> "StartConfig":
        """Return a copy with filesystem paths normalized for the child."""

        values = {item.name: getattr(self, item.name) for item in fields(self)}
        values.update(
            project_root=(
                str(Path(self.project_root).expanduser().resolve())
                if self.project_root
                else None
            ),
            asset_file=str(Path(self.asset_file).expanduser().resolve()),
            gain_config=(
                str(Path(self.gain_config).expanduser().resolve())
                if self.gain_config
                else None
            ),
        )
        return StartConfig(**values)


@dataclass(slots=True)
class ControlCommand:
    """One command delivered at a physics-step boundary."""

    kind: CommandKind
    payload: dict[str, Any] = field(default_factory=dict)
    sequence: int = 0
    wall_time_sent: float = 0.0


@dataclass(slots=True)
class EventPacket:
    """Low-rate state, metadata, acknowledgement, warning, or error event."""

    kind: EventKind
    payload: dict[str, Any] = field(default_factory=dict)
    wall_time_sent: float = 0.0


@dataclass(slots=True)
class TelemetryPacket:
    """One sampled joint state published without blocking physics."""

    simulation_time: float
    joint_name: str
    joint_index: int
    target_position: float
    requested_target_position: float
    actual_position: float
    joint_velocity: float
    position_error: float
    computed_effort: float | None
    applied_effort: float | None
    measured_joint_effort: float | None
    effort_limit: float
    stiffness: float
    damping: float
    saturated: bool
    clamp_status: bool
    step_phase: str
    simulation_state: str
    current_metrics: dict[str, Any] = field(default_factory=dict)
    wall_time_sent: float = 0.0
