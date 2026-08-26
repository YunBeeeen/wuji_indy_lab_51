"""Isaac Sim child-process entry point and standalone articulation loop."""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import queue
import time
import traceback
from typing import Any

from .gain_io import load_json
from .messages import CommandKind, ControlCommand, EventKind, EventPacket, StartConfig, TelemetryPacket
from .metrics import StepResponseTracker
from .step_signal import PeriodicStepSignal


def _put_latest(output_queue: Any, value: Any) -> None:
    """Put without blocking physics, dropping the oldest queued sample if full."""

    try:
        output_queue.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        output_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        output_queue.put_nowait(value)
    except queue.Full:
        pass


def _send_event(event_queue: Any, kind: EventKind, payload: dict[str, Any]) -> None:
    packet = EventPacket(kind=kind, payload=payload, wall_time_sent=time.time())
    _put_latest(event_queue, packet)
    if kind in (EventKind.WARNING, EventKind.ERROR):
        print(f"[pd_tuner:{kind.value}] {payload.get('message', payload)}", flush=True)


def _finite_or_none(value: float | None) -> bool:
    return value is None or math.isfinite(value)


def _validate_start_config(cfg: StartConfig) -> None:
    """Reject invalid numeric startup values before creating SimulationApp."""

    finite_values = {
        "physics_dt": cfg.physics_dt,
        "step_amplitude": cfg.step_amplitude,
        "step_period": cfg.step_period,
        "initial_delay": cfg.initial_delay,
        "telemetry_hz": cfg.telemetry_hz,
        "velocity_safety_threshold": cfg.velocity_safety_threshold,
        "saturation_safety_duration": cfg.saturation_safety_duration,
    }
    if not all(math.isfinite(float(value)) for value in finite_values.values()):
        raise ValueError("All numeric simulation and step settings must be finite.")
    if cfg.physics_dt <= 0.0:
        raise ValueError("Physics timestep must be positive.")
    if cfg.step_period <= cfg.physics_dt:
        raise ValueError(f"Step period must be greater than physics dt ({cfg.physics_dt:g} s).")
    if cfg.telemetry_hz <= 0.0:
        raise ValueError("Telemetry rate must be positive.")
    if cfg.velocity_safety_threshold <= 0.0 or cfg.saturation_safety_duration <= 0.0:
        raise ValueError("Safety thresholds must be positive.")
    if cfg.effort_limit_override is not None:
        override = float(cfg.effort_limit_override)
        if not math.isfinite(override) or override <= 0.0:
            raise ValueError("Effort-limit override must be finite and greater than zero.")


def _initial_joint_state(robot: Any, mode: str) -> tuple[Any, Any]:
    """Build a one-environment initial state without robot-specific assumptions."""

    import torch

    position = robot.data.default_joint_pos.clone()
    velocity = torch.zeros_like(robot.data.default_joint_vel)
    if mode == "zeros":
        position.zero_()
    elif mode == "joint_limit_midpoint":
        lower = robot.data.joint_pos_limits[..., 0]
        upper = robot.data.joint_pos_limits[..., 1]
        finite = torch.isfinite(lower) & torch.isfinite(upper)
        midpoint = 0.5 * (lower + upper)
        position = torch.where(finite, midpoint, position)
    elif mode != "asset_default":
        raise ValueError(f"Unknown initial pose mode: {mode!r}")
    position = torch.max(torch.min(position, robot.data.joint_pos_limits[..., 1]), robot.data.joint_pos_limits[..., 0])
    return position, velocity


def _reset_robot(robot: Any, initial_position: Any, initial_velocity: Any) -> None:
    """Reset physical state, internal buffers, and position targets consistently."""

    robot.write_joint_state_to_sim(initial_position, initial_velocity)
    robot.set_joint_position_target(initial_position)
    robot.set_joint_velocity_target(initial_velocity)
    robot.reset()


def _metadata(robot: Any, adapter: Any, cfg: StartConfig) -> dict[str, Any]:
    fixed_base = bool(robot.is_fixed_base)
    return {
        "articulation_name": Path(cfg.asset_file).stem,
        "prim_path": str(robot.cfg.prim_path),
        "asset_file": cfg.asset_file,
        "asset_cfg_name": cfg.asset_cfg_name,
        "project_root": cfg.project_root,
        "device": str(robot.device),
        "physics_dt": cfg.physics_dt,
        "rendering": bool(cfg.render and not cfg.headless),
        "fixed_base": fixed_base,
        "joint_names": list(robot.joint_names),
        "tunable_joint_names": adapter.list_tunable_joints(),
        "joints": [item.to_dict() for item in adapter.joint_infos],
        "actuator_groups": [item.to_dict() for item in adapter.actuator_groups],
        "warnings": list(adapter.warnings),
        "effort_signals": {
            "computed_effort": (
                "Isaac Lab implicit-actuator PD estimate: Kp*(q_target-q) + "
                "Kd*(qd_target-qd) + feed-forward effort"
            ),
            "applied_effort": "The computed effort clipped by the actuator effort limit",
            "measured_joint_effort": None,
        },
    }


def _find_joint_index(robot: Any, adapter: Any, requested: str | None) -> int:
    tunable = adapter.list_tunable_joints()
    if not tunable:
        raise RuntimeError("The articulation has no joints tunable by the implicit-actuator adapter.")
    name = requested if requested in tunable else tunable[0]
    return list(robot.joint_names).index(name)


def _apply_gain_config(adapter: Any, robot: Any, path: str, event_queue: Any) -> None:
    document = load_json(path)
    joints = document.get("joints", {})
    if not isinstance(joints, dict):
        raise ValueError("Gain config 'joints' must be a JSON object.")
    name_to_index = {name: index for index, name in enumerate(robot.joint_names)}
    for name, values in joints.items():
        if name not in name_to_index:
            _send_event(event_queue, EventKind.WARNING, {"message": f"Gain config joint not found: {name}"})
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Gain config entry for {name!r} must be an object.")
        acknowledgements = adapter.apply_gains(
            name_to_index[name],
            values["stiffness"],
            values["damping"],
            values["effort_limit"],
        )
        for acknowledgement in acknowledgements:
            _send_event(event_queue, EventKind.GAIN_APPLIED, acknowledgement.to_dict())


def run_simulator_process(
    start_config: StartConfig,
    control_queue: Any,
    telemetry_queue: Any,
    event_queue: Any,
    shutdown_event: Any,
) -> None:
    """Run one Isaac Sim application and articulation until receiving ``STOP``.

    All Isaac Sim/Isaac Lab imports happen after ``AppLauncher`` starts inside
    this spawned child.  Qt is never imported in this process.
    """

    # multiprocessing.Queue otherwise waits for its feeder thread to flush a
    # telemetry backlog during child shutdown. Telemetry is explicitly lossy;
    # clean SimulationApp shutdown must take priority over stale samples.
    try:
        telemetry_queue.cancel_join_thread()
    except (AttributeError, OSError):
        pass
    simulation_app = None
    sim = None
    try:
        cfg = start_config.normalized()
        _validate_start_config(cfg)
        _send_event(event_queue, EventKind.STATE, {"state": "starting"})
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher({"headless": cfg.headless, "device": cfg.device})
        simulation_app = app_launcher.app

        # Isaac Lab and asset modules may import Omniverse APIs, so they belong after AppLauncher.
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationContext

        from .adapters.implicit_actuator import IsaacLabImplicitActuatorAdapter
        from .adapters.version_compat import collect_version_info
        from .asset_loader import resolve_asset_cfg

        sim = SimulationContext(
            sim_utils.SimulationCfg(
                device=cfg.device,
                dt=cfg.physics_dt,
                render_interval=1,
            )
        )
        sim.set_camera_view([2.0, 2.0, 2.0], [0.0, 0.0, 0.5])
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

        source_asset_cfg = resolve_asset_cfg(cfg.asset_file, cfg.asset_cfg_name, cfg.project_root)
        robot_cfg = source_asset_cfg.replace(prim_path="/World/Robot")
        robot = robot_cfg.class_type(robot_cfg)
        sim.reset()
        initial_position, initial_velocity = _initial_joint_state(robot, cfg.initial_pose_mode)
        _reset_robot(robot, initial_position, initial_velocity)
        robot.write_data_to_sim()
        sim.step(render=cfg.render and not cfg.headless)
        robot.update(cfg.physics_dt)

        adapter = IsaacLabImplicitActuatorAdapter(robot, robot_cfg)
        metadata = _metadata(robot, adapter, cfg)
        if not metadata["fixed_base"]:
            message = (
                "The articulation has a floating base. PD step responses can include base motion; "
                "use a fixed-base asset or physically constrain it for valid joint tuning."
            )
            metadata["warnings"].append(message)
        _send_event(event_queue, EventKind.VERSION_INFO, collect_version_info())
        _send_event(event_queue, EventKind.MODEL_METADATA, metadata)
        for warning in metadata["warnings"]:
            _send_event(event_queue, EventKind.WARNING, {"message": warning})

        if cfg.effort_limit_override is not None:
            for info in adapter.joint_infos:
                if info.tunable:
                    current = adapter.read_gains(info.index)
                    acknowledgements = adapter.apply_gains(
                        info.index,
                        current.stiffness,
                        current.damping,
                        cfg.effort_limit_override,
                    )
                    for acknowledgement in acknowledgements:
                        _send_event(event_queue, EventKind.GAIN_APPLIED, acknowledgement.to_dict())
        if cfg.gain_config:
            _apply_gain_config(adapter, robot, cfg.gain_config, event_queue)

        selected_index = _find_joint_index(robot, adapter, cfg.selected_joint)
        target_positions = initial_position.clone()
        target_velocities = torch.zeros_like(initial_velocity)
        step_signal = PeriodicStepSignal(
            amplitude=cfg.step_amplitude,
            period=cfg.step_period,
            initial_delay=cfg.initial_delay,
            direction=cfg.step_direction,
            repeat=cfg.repeat_step,
        )
        simulation_time = 0.0
        step_signal.restart(simulation_time, float(robot.data.joint_pos[0, selected_index].item()), active=False)
        tracker = StepResponseTracker(cfg.step_amplitude)
        last_target = step_signal.q0
        paused = False
        saturation_started: float | None = None
        saturation_warning_sent = False
        position_limit_warning_active = False
        telemetry_interval = 1.0 / min(max(cfg.telemetry_hz, 1.0), 500.0)
        next_telemetry = 0.0
        state = "running"
        _send_event(
            event_queue,
            EventKind.JOINT_SELECTED,
            {
                "joint": adapter.joint_infos[selected_index].to_dict(),
                "gains": adapter.read_gains(selected_index).to_dict(),
                "q0": step_signal.q0,
            },
        )
        _send_event(event_queue, EventKind.STATE, {"state": state})

        running = True
        next_wall_step = time.perf_counter()
        while running and not shutdown_event.is_set() and simulation_app.is_running():
            while True:
                try:
                    command: ControlCommand = control_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    if command.kind == CommandKind.STOP:
                        running = False
                        _send_event(event_queue, EventKind.STATE, {"state": "stopping"})
                        break
                    if command.kind == CommandKind.PAUSE_SIMULATION:
                        paused = True
                        step_signal.pause()
                        state = "paused"
                        _send_event(event_queue, EventKind.STATE, {"state": state})
                    elif command.kind == CommandKind.RESUME_SIMULATION:
                        paused = False
                        state = "running"
                        next_wall_step = time.perf_counter()
                        _send_event(event_queue, EventKind.STATE, {"state": state})
                    elif command.kind == CommandKind.SELECT_JOINT:
                        name = str(command.payload["joint_name"])
                        if name not in adapter.list_tunable_joints():
                            raise ValueError(f"Joint is not tunable: {name}")
                        step_signal.pause()
                        selected_index = list(robot.joint_names).index(name)
                        q0 = float(robot.data.joint_pos[0, selected_index].item())
                        target_positions[0, selected_index] = q0
                        step_signal.restart(simulation_time, q0, active=False)
                        tracker.reset()
                        last_target = q0
                        saturation_started = None
                        saturation_warning_sent = False
                        position_limit_warning_active = False
                        _send_event(
                            event_queue,
                            EventKind.JOINT_SELECTED,
                            {
                                "joint": adapter.joint_infos[selected_index].to_dict(),
                                "gains": adapter.read_gains(selected_index).to_dict(),
                                "q0": q0,
                                "clear_graph": True,
                            },
                        )
                    elif command.kind in (CommandKind.APPLY_GAINS, CommandKind.APPLY_GAINS_TO_GROUP):
                        values = command.payload
                        current = adapter.read_gains(selected_index)
                        large = []
                        for key, requested, original in (
                            ("stiffness", float(values["stiffness"]), current.stiffness),
                            ("damping", float(values["damping"]), current.damping),
                            ("effort_limit", float(values["effort_limit"]), current.effort_limit),
                        ):
                            if requested > max(abs(original) * 100.0, 1.0e3):
                                large.append(f"{key}={requested:g}")
                        if large:
                            _send_event(
                                event_queue,
                                EventKind.WARNING,
                                {"message": "Unusually large gain request: " + ", ".join(large)},
                            )
                        acknowledgements = adapter.apply_gains(
                            selected_index,
                            float(values["stiffness"]),
                            float(values["damping"]),
                            float(values["effort_limit"]),
                            apply_to_group=command.kind == CommandKind.APPLY_GAINS_TO_GROUP,
                        )
                        for acknowledgement in acknowledgements:
                            _send_event(event_queue, EventKind.GAIN_APPLIED, acknowledgement.to_dict())
                    elif command.kind == CommandKind.RESTORE_ORIGINAL_GAINS:
                        acknowledgements = adapter.restore_original_gains(
                            selected_index,
                            apply_to_group=bool(command.payload.get("apply_to_group", False)),
                        )
                        for acknowledgement in acknowledgements:
                            _send_event(event_queue, EventKind.GAIN_APPLIED, acknowledgement.to_dict())
                    elif command.kind == CommandKind.CONFIGURE_STEP:
                        values = command.payload
                        period = float(values["period"])
                        if period <= cfg.physics_dt:
                            raise ValueError(f"Step period must be greater than physics dt ({cfg.physics_dt:g} s).")
                        step_signal.configure(
                            float(values["amplitude"]),
                            period,
                            float(values["initial_delay"]),
                            int(values["direction"]),
                            bool(values["repeat"]),
                        )
                        tracker.commanded_amplitude = step_signal.amplitude
                    elif command.kind == CommandKind.START_STEP:
                        q0 = float(robot.data.joint_pos[0, selected_index].item())
                        step_signal.restart(simulation_time, q0, active=True)
                        tracker.reset()
                        last_target = q0
                    elif command.kind == CommandKind.PAUSE_STEP:
                        step_signal.pause()
                    elif command.kind == CommandKind.RESTART_STEP:
                        q0 = float(robot.data.joint_pos[0, selected_index].item())
                        step_signal.restart(simulation_time, q0, active=True)
                        tracker.reset()
                        last_target = q0
                        _send_event(event_queue, EventKind.JOINT_SELECTED, {"q0": q0, "clear_graph": True})
                    elif command.kind == CommandKind.RESET_SELECTED_JOINT:
                        step_signal.pause()
                        target = initial_position[:, selected_index : selected_index + 1]
                        velocity = initial_velocity[:, selected_index : selected_index + 1]
                        robot.write_joint_state_to_sim(target, velocity, joint_ids=[selected_index])
                        target_positions[0, selected_index] = target[0, 0]
                        robot.reset()
                        q0 = float(target[0, 0].item())
                        step_signal.restart(simulation_time, q0, active=False)
                        tracker.reset()
                        last_target = q0
                    elif command.kind == CommandKind.RESET_ALL_JOINTS:
                        step_signal.pause()
                        _reset_robot(robot, initial_position, initial_velocity)
                        target_positions.copy_(initial_position)
                        q0 = float(initial_position[0, selected_index].item())
                        step_signal.restart(simulation_time, q0, active=False)
                        tracker.reset()
                        last_target = q0
                except Exception as exc:
                    _send_event(
                        event_queue,
                        EventKind.ERROR,
                        {"message": f"Command {command.kind.value} failed: {exc}", "recoverable": True},
                    )

            if not running:
                break
            if paused:
                if cfg.render and not cfg.headless:
                    sim.render()
                else:
                    time.sleep(0.005)
                continue

            joint_info = adapter.joint_infos[selected_index]
            sample = step_signal.sample(simulation_time, joint_info.lower_limit, joint_info.upper_limit)
            if not math.isclose(sample.applied_target, last_target, rel_tol=0.0, abs_tol=1.0e-9):
                completed = tracker.begin_transition(
                    simulation_time,
                    float(robot.data.joint_pos[0, selected_index].item()),
                    sample.applied_target,
                )
                if completed is not None:
                    _send_event(event_queue, EventKind.STEP_METRICS, {"last_completed": completed.to_dict()})
                last_target = sample.applied_target
            target_positions[0, selected_index] = sample.applied_target
            robot.set_joint_position_target(target_positions)
            robot.set_joint_velocity_target(target_velocities)
            robot.write_data_to_sim()
            sim.step(render=cfg.render and not cfg.headless)
            simulation_time += cfg.physics_dt
            robot.update(cfg.physics_dt)

            actual = float(robot.data.joint_pos[0, selected_index].item())
            velocity = float(robot.data.joint_vel[0, selected_index].item())
            effort = adapter.read_effort_signals(selected_index)
            gains = adapter.read_gains(selected_index)
            effort_for_saturation = effort.applied_effort
            saturated = (
                effort_for_saturation is not None
                and gains.effort_limit > 0.0
                and abs(effort_for_saturation) >= 0.98 * gains.effort_limit
            )
            tracker.update(simulation_time, actual, velocity, effort_for_saturation, saturated)

            safety_message = None
            if not all(
                (
                    math.isfinite(actual),
                    math.isfinite(velocity),
                    _finite_or_none(effort.computed_effort),
                    _finite_or_none(effort.applied_effort),
                )
            ):
                safety_message = "Non-finite joint state or effort detected."
            elif abs(velocity) > cfg.velocity_safety_threshold:
                safety_message = (
                    f"Velocity safety threshold exceeded: {velocity:.3f} rad/s > "
                    f"{cfg.velocity_safety_threshold:.3f} rad/s."
                )

            position_limit_exceeded = (
                actual < joint_info.lower_limit - 1.0e-4
                or actual > joint_info.upper_limit + 1.0e-4
            )
            if position_limit_exceeded and not position_limit_warning_active:
                overrun = max(joint_info.lower_limit - actual, actual - joint_info.upper_limit, 0.0)
                _send_event(
                    event_queue,
                    EventKind.WARNING,
                    {
                        "message": (
                            f"Joint {joint_info.name} exceeded its position limits during manual tuning: "
                            f"actual={actual:.6f} rad, limits=[{joint_info.lower_limit:.6f}, "
                            f"{joint_info.upper_limit:.6f}] rad, overrun={overrun:.6g} rad. "
                            "Manual tuning continues."
                        )
                    },
                )
            position_limit_warning_active = position_limit_exceeded

            if saturated:
                saturation_started = saturation_started or simulation_time
                if (
                    simulation_time - saturation_started >= cfg.saturation_safety_duration
                    and not saturation_warning_sent
                ):
                    _send_event(
                        event_queue,
                        EventKind.WARNING,
                        {
                            "message": (
                                f"Effort remained at or above 98% of the limit for "
                                f"{cfg.saturation_safety_duration:.2f} s during manual tuning. "
                                "Manual tuning continues."
                            )
                        },
                    )
                    saturation_warning_sent = True
            else:
                saturation_started = None
                saturation_warning_sent = False
            if safety_message:
                step_signal.pause()
                paused = True
                state = "error"
                _send_event(event_queue, EventKind.ERROR, {"message": safety_message, "recoverable": True})
                _send_event(event_queue, EventKind.STATE, {"state": state})

            if simulation_time + 1.0e-12 >= next_telemetry:
                current_metrics = tracker.snapshot(simulation_time)
                packet = TelemetryPacket(
                    simulation_time=simulation_time,
                    joint_name=robot.joint_names[selected_index],
                    joint_index=selected_index,
                    target_position=sample.applied_target,
                    requested_target_position=sample.requested_target,
                    actual_position=actual,
                    joint_velocity=velocity,
                    position_error=sample.applied_target - actual,
                    computed_effort=effort.computed_effort,
                    applied_effort=effort.applied_effort,
                    measured_joint_effort=effort.measured_joint_effort,
                    effort_limit=gains.effort_limit,
                    stiffness=gains.stiffness,
                    damping=gains.damping,
                    saturated=saturated,
                    clamp_status=sample.clamped,
                    step_phase=sample.phase,
                    simulation_state=state,
                    current_metrics=current_metrics.to_dict() if current_metrics else {},
                    wall_time_sent=time.time(),
                )
                _put_latest(telemetry_queue, packet)
                next_telemetry = simulation_time + telemetry_interval

            # Pace the rendered child to wall time; headless mode intentionally runs as fast as possible.
            if cfg.render and not cfg.headless:
                next_wall_step += cfg.physics_dt
                remaining = next_wall_step - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(min(remaining, cfg.physics_dt))
                elif remaining < -0.5:
                    next_wall_step = time.perf_counter()

        _send_event(event_queue, EventKind.STATE, {"state": "stopped"})
    except BaseException as exc:
        _send_event(
            event_queue,
            EventKind.ERROR,
            {
                "message": f"Simulation process failed: {type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "recoverable": False,
            },
        )
        _send_event(event_queue, EventKind.STATE, {"state": "error"})
    finally:
        if sim is not None:
            try:
                # Release Python callbacks and the singleton before Kit owns
                # the final timeline/stage shutdown. Calling sim.stop() here
                # can block in _timeline.stop() in a spawned Isaac Sim 5.1
                # child, so shutdown is delegated to SimulationApp.close().
                sim.clear_all_callbacks()
                sim.clear_instance()
            except Exception:
                traceback.print_exc()
        if simulation_app is not None:
            try:
                # Release Kit immediately: the full global plugin cleanup path
                # can wait indefinitely in a spawned child.
                simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
            except Exception:
                traceback.print_exc()
