"""Single user-facing launcher for GUI and headless operation."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
from typing import Any

from .data_buffer import CsvStreamLogger
from .gain_io import build_gain_document, save_json
from .messages import CommandKind, ControlCommand, EventKind, EventPacket, StartConfig, TelemetryPacket
from .simulator_process import run_simulator_process


def build_parser() -> argparse.ArgumentParser:
    """Build the portable command-line interface."""

    parser = argparse.ArgumentParser(description="External-GUI Isaac Lab actuator PD tuning tool")
    parser.add_argument(
        "--project-root",
        default=os.environ.get("PD_TUNER_PROJECT_ROOT"),
        help="Optional import root for the selected asset module (or PD_TUNER_PROJECT_ROOT)",
    )
    parser.add_argument("--asset-directory", help="Initial asset directory shown in the GUI")
    parser.add_argument("--asset-file", help="Python file containing one or more ArticulationCfg objects")
    parser.add_argument("--asset-cfg-name", help="Name of the ArticulationCfg object to spawn")
    parser.add_argument("--joint", help="Initial joint name")
    parser.add_argument("--effort-limit", type=float, help="Optional initial global effort-limit override")
    parser.add_argument("--step-amplitude", type=float, default=0.2, help="Step magnitude in joint units")
    parser.add_argument("--step-period", type=float, default=2.0, help="Full low/high waveform period [s]")
    parser.add_argument("--initial-delay", type=float, default=0.25, help="Delay before the first transition [s]")
    parser.add_argument("--history-seconds", type=float, default=10.0, help="Visible graph history [s]")
    parser.add_argument("--gain-config", help="Optional saved gain JSON loaded after spawn")
    parser.add_argument("--device", default="cuda:0", help="Isaac Lab simulation device")
    parser.add_argument("--physics-dt", type=float, default=1.0 / 120.0, help="Physics timestep [s]")
    parser.add_argument("--initial-pose", default="asset_default", choices=("asset_default", "zeros", "joint_limit_midpoint"))
    parser.add_argument("--session", help="GUI session JSON")
    parser.add_argument("--headless", action="store_true", help="Run without Qt or rendering")
    parser.add_argument("--duration", type=float, default=10.0, help="Headless simulation duration [s]")
    parser.add_argument("--output-directory", help="Output root (default: tuner outputs directory)")
    parser.add_argument("--velocity-safety-threshold", type=float, default=50.0)
    parser.add_argument("--saturation-safety-duration", type=float, default=2.0)
    return parser


def _dependency_error() -> str | None:
    missing = [name for name in ("PySide6", "pyqtgraph") if importlib.util.find_spec(name) is None]
    if not missing:
        return None
    return (
        "Missing external GUI dependencies: "
        + ", ".join(missing)
        + "\nRun ./install_gui_dependencies.sh from the tuner directory."
    )


def _headless_config(args: argparse.Namespace) -> StartConfig:
    if not args.asset_file or not args.asset_cfg_name or not args.joint:
        raise SystemExit("--headless requires --asset-file, --asset-cfg-name, and --joint.")
    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive.")
    return StartConfig(
        project_root=args.project_root,
        asset_file=args.asset_file,
        asset_cfg_name=args.asset_cfg_name,
        device=args.device,
        physics_dt=args.physics_dt,
        render=False,
        headless=True,
        initial_pose_mode=args.initial_pose,
        effort_limit_override=args.effort_limit,
        selected_joint=args.joint,
        step_amplitude=args.step_amplitude,
        step_period=args.step_period,
        initial_delay=args.initial_delay,
        velocity_safety_threshold=args.velocity_safety_threshold,
        saturation_safety_duration=args.saturation_safety_duration,
        gain_config=args.gain_config,
    )


def run_headless(args: argparse.Namespace, package_root: Path) -> int:
    """Run the same child loop without importing Qt, logging telemetry to CSV."""

    config = _headless_config(args)
    output_root = Path(args.output_directory).expanduser().resolve() if args.output_directory else package_root / "outputs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_root / "sessions" / f"pd_tuning_headless_{stamp}.csv"
    gain_path = output_root / "gains" / f"pd_tuning_headless_{stamp}.json"
    logger = CsvStreamLogger(csv_path)
    context = mp.get_context("spawn")
    control_queue = context.Queue(maxsize=256)
    telemetry_queue = context.Queue(maxsize=4096)
    event_queue = context.Queue(maxsize=256)
    shutdown_event = context.Event()
    process = context.Process(
        target=run_simulator_process,
        args=(config, control_queue, telemetry_queue, event_queue, shutdown_event),
        name="pd-tuner-isaac-sim",
    )
    process.start()
    metadata: dict[str, Any] = {}
    gains: dict[str, dict[str, float]] = {}
    step_started = False
    latest_simulation_time = 0.0
    fatal_error = False
    forced_termination = False
    try:
        while process.is_alive():
            for _ in range(200):
                try:
                    event: EventPacket = event_queue.get_nowait()
                except queue.Empty:
                    break
                if event.kind == EventKind.MODEL_METADATA:
                    metadata = event.payload
                    gains = {
                        item["name"]: dict(item["original_gain"])
                        for item in metadata["joints"]
                        if item["tunable"]
                    }
                elif event.kind == EventKind.GAIN_APPLIED:
                    gains[event.payload["joint_name"]] = dict(event.payload["applied"])
                elif event.kind in (EventKind.WARNING, EventKind.ERROR):
                    print(f"[{event.kind.value.upper()}] {event.payload.get('message', event.payload)}", flush=True)
                    # Headless mode has no operator controls with which to recover from a safety pause.
                    fatal_error |= event.kind == EventKind.ERROR
                elif event.kind == EventKind.STATE:
                    print(f"[STATE] {event.payload.get('state')}", flush=True)
                    if event.payload.get("state") == "running" and not step_started:
                        control_queue.put(ControlCommand(kind=CommandKind.START_STEP, wall_time_sent=time.time()))
                        step_started = True
            for _ in range(4000):
                try:
                    packet: TelemetryPacket = telemetry_queue.get_nowait()
                except queue.Empty:
                    break
                logger.append(packet)
                latest_simulation_time = packet.simulation_time
            if fatal_error or latest_simulation_time >= args.duration:
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping headless tuning session…", flush=True)
    finally:
        shutdown_event.set()
        try:
            control_queue.put_nowait(ControlCommand(kind=CommandKind.STOP, wall_time_sent=time.time()))
        except queue.Full:
            pass
        shutdown_deadline = time.monotonic() + 30.0
        while process.is_alive() and time.monotonic() < shutdown_deadline:
            # Keep the pipes flowing while the child closes SimulationApp.
            for _ in range(4000):
                try:
                    packet = telemetry_queue.get_nowait()
                except queue.Empty:
                    break
                logger.append(packet)
            for _ in range(200):
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    break
                if event.kind == EventKind.STATE:
                    print(f"[STATE] {event.payload.get('state')}", flush=True)
            process.join(timeout=0.1)
        if process.is_alive():
            print("Simulation cleanup timed out after 30 s; terminating child.", file=sys.stderr)
            forced_termination = True
            process.terminate()
            process.join(timeout=5.0)
        logger.close()
    if metadata:
        save_json(
            gain_path,
            build_gain_document(metadata["asset_file"], metadata["asset_cfg_name"], metadata["physics_dt"], gains),
        )
    print(f"CSV: {csv_path}")
    if metadata:
        print(f"Gains: {gain_path}")
    return 1 if fatal_error or forced_termination or process.exitcode not in (0, None) else 0


def main(argv: list[str] | None = None) -> int:
    """Launch one external GUI or one headless tuning session."""

    args = build_parser().parse_args(argv)
    package_root = Path(__file__).resolve().parents[1]
    if args.headless:
        return run_headless(args, package_root)
    error = _dependency_error()
    if error:
        print(error, file=sys.stderr)
        return 2
    from .gui_app import run_gui

    return run_gui(args, package_root)
