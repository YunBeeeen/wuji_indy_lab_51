"""Auto Tune request parsing, default resolution, and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any


class ValueSource(str, Enum):
    """Origin of one resolved Auto Tune value."""

    AUTO = "AUTO"
    USER = "USER"
    ASSET = "ASSET"
    SESSION = "SESSION"


class AutoTuneDirection(str, Enum):
    """Step directions evaluated for every candidate."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    BIDIRECTIONAL = "bidirectional"

    @property
    def signs(self) -> tuple[int, ...]:
        if self is AutoTuneDirection.POSITIVE:
            return (1,)
        if self is AutoTuneDirection.NEGATIVE:
            return (-1,)
        return (1, -1)


class AutoTunePreset(str, Enum):
    """Built-in lower-is-better ranking profiles."""

    BALANCED = "balanced"
    FAST_RESPONSE = "fast_response"
    SMOOTH_RESPONSE = "smooth_response"
    LOW_TORQUE = "low_torque"
    CONVERGENCE_FIRST = "convergence_first"
    CUSTOM = "custom"


class AutoTuneTorquePolicy(str, Enum):
    """Whether actuator clipping rejects a candidate or is only reported."""

    STRICT_NO_SATURATION = "strict_no_saturation"
    ALLOW_CLIPPING = "allow_clipping"


METRIC_NAMES = (
    "settling_time",
    "overshoot",
    "rms_applied_effort",
    "peak_applied_effort",
    "steady_state_error",
    "gain_magnitude",
)


@dataclass(frozen=True, slots=True)
class JointTuningContext:
    """Spawned-joint state needed to resolve safe automatic defaults."""

    joint_name: str
    joint_index: int
    actuator_group: str
    lower_limit: float
    upper_limit: float
    velocity_limit: float
    current_position: float
    current_kp: float
    current_kd: float
    current_effort_limit: float
    physics_dt: float


@dataclass(slots=True)
class AutoTuneRequest:
    """Optional user/session values; ``None`` means resolve automatically."""

    effort_limit: float | None = None
    step_amplitude: float | None = None
    target_settling_time: float | None = None
    hold_duration: float | None = None
    settling_tolerance: float | None = None
    settling_hold_time: float | None = None
    maximum_overshoot: float | None = None
    overshoot_enabled: bool = True
    maximum_steady_state_error: float | None = None
    steady_state_error_enabled: bool = True
    maximum_velocity: float | None = None
    kp_min: float | None = None
    kp_max: float | None = None
    kd_min: float | None = None
    kd_max: float | None = None
    direction: AutoTuneDirection = AutoTuneDirection.POSITIVE
    repeats: int | None = None
    search_budget: int | None = None
    preset: AutoTunePreset = AutoTunePreset.BALANCED
    torque_policy: AutoTuneTorquePolicy = AutoTuneTorquePolicy.STRICT_NO_SATURATION
    custom_weights: dict[str, float] = field(default_factory=dict)
    value_sources: dict[str, ValueSource] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "AutoTuneRequest":
        """Parse a GUI/headless JSON mapping without accepting unknown enums."""

        sources = {
            str(key): ValueSource(str(value))
            for key, value in dict(values.get("value_sources", {})).items()
        }
        numeric_names = (
            "effort_limit",
            "step_amplitude",
            "target_settling_time",
            "hold_duration",
            "settling_tolerance",
            "settling_hold_time",
            "maximum_overshoot",
            "maximum_steady_state_error",
            "maximum_velocity",
            "kp_min",
            "kp_max",
            "kd_min",
            "kd_max",
        )
        kwargs: dict[str, Any] = {
            name: (None if values.get(name) in (None, "") else float(values[name]))
            for name in numeric_names
        }
        kwargs.update(
            overshoot_enabled=bool(values.get("overshoot_enabled", True)),
            steady_state_error_enabled=bool(values.get("steady_state_error_enabled", True)),
            direction=AutoTuneDirection(str(values.get("direction", AutoTuneDirection.POSITIVE.value))),
            repeats=(None if values.get("repeats") in (None, "") else int(values["repeats"])),
            search_budget=(
                None if values.get("search_budget") in (None, "") else int(values["search_budget"])
            ),
            preset=AutoTunePreset(str(values.get("preset", AutoTunePreset.BALANCED.value))),
            torque_policy=AutoTuneTorquePolicy(
                str(
                    values.get(
                        "torque_policy",
                        AutoTuneTorquePolicy.STRICT_NO_SATURATION.value,
                    )
                )
            ),
            custom_weights={
                str(key): float(value) for key, value in dict(values.get("custom_weights", {})).items()
            },
            value_sources=sources,
        )
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["direction"] = self.direction.value
        result["preset"] = self.preset.value
        result["torque_policy"] = self.torque_policy.value
        result["value_sources"] = {key: value.value for key, value in self.value_sources.items()}
        return result


@dataclass(frozen=True, slots=True)
class ResolvedAutoTuneConfig:
    """Validated values used by the search and simulator trial state machine."""

    joint_name: str
    joint_index: int
    actuator_group: str
    q0: float
    lower_limit: float
    upper_limit: float
    effort_limit: float
    current_kp: float
    current_kd: float
    step_amplitude: float
    direction: AutoTuneDirection
    requested_targets: dict[str, float]
    applied_targets: dict[str, float]
    target_clamped: dict[str, bool]
    target_settling_time: float
    hold_duration: float
    settling_tolerance: float
    settling_hold_time: float
    maximum_overshoot: float | None
    maximum_steady_state_error: float | None
    maximum_velocity: float
    kp_min: float
    kp_max: float
    kd_min: float
    kd_max: float
    repeats: int
    search_budget: int
    preset: AutoTunePreset
    torque_policy: AutoTuneTorquePolicy
    custom_weights: dict[str, float]
    torque_match_tolerance: float
    stabilization_duration: float
    return_duration: float
    physics_dt: float
    value_sources: dict[str, ValueSource]
    resolution_notes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["direction"] = self.direction.value
        result["preset"] = self.preset.value
        result["torque_policy"] = self.torque_policy.value
        result["value_sources"] = {key: value.value for key, value in self.value_sources.items()}
        return result


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _source(request: AutoTuneRequest, name: str, automatic: ValueSource) -> ValueSource:
    if getattr(request, name, None) is None:
        return automatic
    return request.value_sources.get(name, ValueSource.USER)


def resolve_autotune_config(
    request: AutoTuneRequest,
    context: JointTuningContext,
) -> ResolvedAutoTuneConfig:
    """Resolve every blank field and reject unsafe or ambiguous configurations.

    The automatic Kd upper bound does not assume an unknown link inertia.  It
    allocates half of the effort limit to damping at the average velocity
    required to traverse the requested step within the target settling time.
    If the current Kd is positive, four times that value is also included so
    the existing setting is not accidentally excluded. In strict mode torque
    matching remains a hard per-sample constraint. Convergence-first mode
    retains the applied effort cap but permits and reports actuator clipping.
    """

    notes: dict[str, str] = {}
    sources: dict[str, ValueSource] = {}
    position_range = _finite("joint position range", context.upper_limit - context.lower_limit)
    if position_range <= 0.0:
        raise ValueError("Selected joint has no positive finite position range.")

    effort = _finite(
        "effort limit",
        context.current_effort_limit if request.effort_limit is None else request.effort_limit,
    )
    if effort <= 0.0:
        raise ValueError("Effort limit must be greater than zero.")
    sources["effort_limit"] = _source(request, "effort_limit", ValueSource.ASSET)

    amplitude = _finite(
        "step amplitude",
        min(0.1, 0.10 * position_range) if request.step_amplitude is None else request.step_amplitude,
    )
    if amplitude <= 1.0e-9:
        raise ValueError("Step amplitude is too small or zero; cannot create a PD search range.")
    sources["step_amplitude"] = _source(request, "step_amplitude", ValueSource.AUTO)

    target_settling = _finite(
        "target settling time",
        0.5 if request.target_settling_time is None else request.target_settling_time,
    )
    if target_settling <= 0.0:
        raise ValueError("Target settling time must be positive.")
    sources["target_settling_time"] = _source(request, "target_settling_time", ValueSource.AUTO)

    hold = _finite(
        "hold duration",
        max(2.0, 3.0 * target_settling) if request.hold_duration is None else request.hold_duration,
    )
    sources["hold_duration"] = _source(request, "hold_duration", ValueSource.AUTO)
    tolerance = _finite(
        "settling tolerance",
        max(amplitude * 0.02, 0.001)
        if request.settling_tolerance is None
        else request.settling_tolerance,
    )
    sources["settling_tolerance"] = _source(request, "settling_tolerance", ValueSource.AUTO)
    settle_hold = _finite(
        "settling hold time",
        0.1 if request.settling_hold_time is None else request.settling_hold_time,
    )
    sources["settling_hold_time"] = _source(request, "settling_hold_time", ValueSource.AUTO)
    if tolerance <= 0.0 or settle_hold <= 0.0:
        raise ValueError("Settling tolerance and hold time must be positive.")
    if hold <= max(settle_hold, context.physics_dt):
        raise ValueError("Hold duration must exceed settling hold time and physics dt.")

    maximum_overshoot = None
    if request.overshoot_enabled:
        maximum_overshoot = _finite(
            "maximum overshoot",
            5.0 if request.maximum_overshoot is None else request.maximum_overshoot,
        )
        if maximum_overshoot < 0.0:
            raise ValueError("Maximum overshoot cannot be negative.")
        sources["maximum_overshoot"] = _source(request, "maximum_overshoot", ValueSource.AUTO)
    maximum_sse = None
    if request.steady_state_error_enabled:
        maximum_sse = _finite(
            "maximum steady-state error",
            tolerance
            if request.maximum_steady_state_error is None
            else request.maximum_steady_state_error,
        )
        if maximum_sse < 0.0:
            raise ValueError("Maximum steady-state error cannot be negative.")
        sources["maximum_steady_state_error"] = _source(
            request, "maximum_steady_state_error", ValueSource.AUTO
        )

    asset_velocity = _finite("asset velocity limit", context.velocity_limit)
    if asset_velocity <= 0.0:
        raise ValueError("Selected joint has no positive finite velocity limit.")
    if request.maximum_velocity is None:
        maximum_velocity = asset_velocity
        sources["maximum_velocity"] = ValueSource.ASSET
    else:
        requested_velocity = _finite("maximum velocity", request.maximum_velocity)
        if requested_velocity <= 0.0:
            raise ValueError("Maximum velocity must be positive.")
        maximum_velocity = min(requested_velocity, asset_velocity)
        sources["maximum_velocity"] = request.value_sources.get("maximum_velocity", ValueSource.USER)
        if requested_velocity > asset_velocity:
            notes["maximum_velocity"] = (
                f"User value {requested_velocity:g} rad/s was limited by asset value "
                f"{asset_velocity:g} rad/s."
            )

    kp_min = _finite("Kp min", 0.0 if request.kp_min is None else request.kp_min)
    effort_fraction_cap = (
        3.0 if request.torque_policy is AutoTuneTorquePolicy.ALLOW_CLIPPING else 0.95
    )
    kp_max = _finite(
        "Kp max",
        effort_fraction_cap * effort / amplitude
        if request.kp_max is None
        else request.kp_max,
    )
    sources["kp_min"] = _source(request, "kp_min", ValueSource.AUTO)
    sources["kp_max"] = _source(request, "kp_max", ValueSource.AUTO)
    kd_min = _finite("Kd min", 0.0 if request.kd_min is None else request.kd_min)
    sources["kd_min"] = _source(request, "kd_min", ValueSource.AUTO)
    if request.kd_max is None:
        response_velocity = amplitude / target_settling
        if response_velocity <= 0.0 or not math.isfinite(response_velocity):
            raise ValueError("Cannot derive Kd range from amplitude and target settling time.")
        torque_envelope = 0.5 * effort / response_velocity
        current_envelope = 4.0 * context.current_kd if context.current_kd > 0.0 else 0.0
        kd_max = max(torque_envelope, current_envelope)
        sources["kd_max"] = ValueSource.AUTO
        notes["kd_max"] = (
            "Auto Kd max allocates 50% of effort_limit to damping at "
            "step_amplitude/target_settling_time and includes 4×current Kd when positive: "
            f"torque envelope={torque_envelope:.6g}, current envelope={current_envelope:.6g}."
        )
    else:
        kd_max = _finite("Kd max", request.kd_max)
        sources["kd_max"] = request.value_sources.get("kd_max", ValueSource.USER)
    if kp_min < 0.0 or kd_min < 0.0 or kp_max < kp_min or kd_max < kd_min:
        raise ValueError("Require 0 <= Kp min <= Kp max and 0 <= Kd min <= Kd max.")
    if kp_max == 0.0:
        raise ValueError("Kp search range contains only zero and cannot drive a position step.")
    initial_step_kp_cap = (effort - max(1.0e-6, effort * 1.0e-4)) / amplitude
    if (
        request.torque_policy is AutoTuneTorquePolicy.STRICT_NO_SATURATION
        and kp_min > initial_step_kp_cap + 1.0e-12
    ):
        raise ValueError(
            "Kp min guarantees initial position-step torque saturation: "
            f"Kp_min*amplitude={kp_min * amplitude:.6g} N·m, "
            f"available effort={effort:.6g} N·m. Lower Kp min or step amplitude."
        )
    if request.torque_policy is AutoTuneTorquePolicy.ALLOW_CLIPPING:
        notes["torque_policy"] = (
            "Convergence-first mode permits computed_effort to exceed effort_limit and "
            "differ from applied_effort. The actuator still clips actual applied effort; "
            "non-finite state, position-limit, and velocity violations remain hard failures."
        )
        notes["kp_effort_screen"] = (
            "Kp screening may use up to 300% predicted computed-effort demand; clipping "
            "and saturation_count remain visible in every result."
        )
    else:
        notes["kp_effort_screen"] = (
            "Kp screening is performed first with Kd fixed at Kd min. Candidate Kp values "
            "are spaced by predicted initial P-effort use; theoretical no-saturation cap="
            f"{initial_step_kp_cap:.6g}."
        )

    repeats = 2 if request.repeats is None else int(request.repeats)
    budget = 12 if request.search_budget is None else int(request.search_budget)
    sources["repeats"] = _source(request, "repeats", ValueSource.AUTO)
    sources["search_budget"] = _source(request, "search_budget", ValueSource.AUTO)
    if repeats <= 0 or budget <= 0:
        raise ValueError("Repeats and search budget must be positive integers.")
    notes["search_strategy"] = (
        "Response-guided search uses increasing safe Kp probes, identifies a local "
        "second-order model from measured step data, calculates a target Kp/Kd, then "
        "raises Kp for slow/non-settling responses or Kd for excess overshoot. "
        "The search budget is a maximum, so the strategy may stop early."
    )

    requested_targets: dict[str, float] = {}
    applied_targets: dict[str, float] = {}
    target_clamped: dict[str, bool] = {}
    for sign in request.direction.signs:
        name = "positive" if sign > 0 else "negative"
        requested_target = context.current_position + sign * amplitude
        applied_target = min(max(requested_target, context.lower_limit), context.upper_limit)
        if abs(applied_target - context.current_position) <= 1.0e-9:
            raise ValueError(f"{name.title()} step has zero range after joint-limit clamping.")
        requested_targets[name] = requested_target
        applied_targets[name] = applied_target
        target_clamped[name] = not math.isclose(
            requested_target, applied_target, rel_tol=0.0, abs_tol=1.0e-12
        )

    weights = {name: float(request.custom_weights.get(name, 0.0)) for name in METRIC_NAMES}
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("Custom ranking weights must be finite and non-negative.")

    torque_tolerance = max(1.0e-6, effort * 1.0e-4)
    return ResolvedAutoTuneConfig(
        joint_name=context.joint_name,
        joint_index=context.joint_index,
        actuator_group=context.actuator_group,
        q0=context.current_position,
        lower_limit=context.lower_limit,
        upper_limit=context.upper_limit,
        effort_limit=effort,
        current_kp=context.current_kp,
        current_kd=context.current_kd,
        step_amplitude=amplitude,
        direction=request.direction,
        requested_targets=requested_targets,
        applied_targets=applied_targets,
        target_clamped=target_clamped,
        target_settling_time=target_settling,
        hold_duration=hold,
        settling_tolerance=tolerance,
        settling_hold_time=settle_hold,
        maximum_overshoot=maximum_overshoot,
        maximum_steady_state_error=maximum_sse,
        maximum_velocity=maximum_velocity,
        kp_min=kp_min,
        kp_max=kp_max,
        kd_min=kd_min,
        kd_max=kd_max,
        repeats=repeats,
        search_budget=budget,
        preset=request.preset,
        torque_policy=request.torque_policy,
        custom_weights=weights,
        torque_match_tolerance=torque_tolerance,
        stabilization_duration=max(0.25, 12.0 * context.physics_dt),
        return_duration=max(0.25, settle_hold),
        physics_dt=context.physics_dt,
        value_sources=sources,
        resolution_notes=notes,
    )
