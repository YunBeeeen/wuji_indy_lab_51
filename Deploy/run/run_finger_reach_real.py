# [run/실물] 중지 reach 정책을 실물 손에서 실행. 4관절만 enable.
"""Run the middle-finger reach policy on the physical Wuji Hand.

Same contract as the MuJoCo path -- 15D observation, 4D residual action, the
same ONNX graph -- with the simulator-only parts replaced:

    MuJoCo                          Real hand
    reset(q) teleport               read current q, then ramp to the start pose
    64 physics substeps hold        90 Hz command republish
    installed Kp/Kd                 firmware servo (gains UNVERIFIED)
    fingertip FK for the log        no Cartesian measurement exists

Bring-up is staged so the hand never runs an untested step:

    --read-only            connect and print 20 joints; no motor is enabled
    --hold-middle          enable the middle finger, hold the present pose
    --test-middle-joint N  move one joint by --delta and confirm the mapping
    --zero-policy          full 15D loop with action forced to zeros
    --policy P.onnx        the real thing

Everything that can fail without hardware -- ONNX shape, scenario contents,
target reachability -- is checked before any motor is enabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from ..policy.finger_reach import (
    FINGER_REACH_RESET_JOINT_POSITIONS,
    FingerReachObservationAdapter,
    MIDDLE_JOINT_NAMES,
    MIDDLE_POLICY_INDICES,
    MiddleFingerReachRunner,
    REACH_ACTION_DIM,
    REACH_OBSERVATION_DIM,
    REACH_RANGE_M,
    finger_major_grid,
)
from ..common.policy_contract import (
    COMMAND_TARGET_LIMITS,
    POLICY_DT,
    POLICY_JOINT_NAMES,
    soft_command_limits,
)
from ..backends.real_wuji import RealWujiHand, middle_enable_mask
from ..backends.real_wuji_scheduler import DEFAULT_COMMAND_HZ, POLICY_HZ, RealWujiScheduler

# No tip_palm_* or error_norm: the real hand has no Cartesian fingertip
# measurement, and filling those columns from FK would label a model output as
# a measurement.  q(t), action(t) and q_target(t) are what the three backends
# actually share.
CSV_COLUMNS = (
    ["time", "wall_time", "policy_tick", "command_tick", "target_index"]
    + ["target_palm_x", "target_palm_y", "target_palm_z"]
    + [f"q_prev_{k}" for k in range(1, 5)]
    + [f"q_curr_{k}" for k in range(1, 5)]
    + [f"action_{k}" for k in range(1, 5)]
    + [f"q_target_{k}" for k in range(1, 5)]
    + ["policy_inference_ms"]
)


class ZeroPolicy:
    def infer(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (REACH_OBSERVATION_DIM,):
            raise ValueError(f"Expected {REACH_OBSERVATION_DIM}D observation.")
        return np.zeros(REACH_ACTION_DIM, dtype=np.float32)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # --read-only is NOT in the exclusive group: combined with --policy it dry
    # runs the whole pipeline on real encoder values and enables nothing.
    parser.add_argument("--read-only", action="store_true",
                        help="Never enable a motor. With --policy, still builds the "
                             "observation, runs inference and decodes q_target so the "
                             "numbers can be inspected before anything moves.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--measure-timing", action="store_true",
                      help="Time the SDK round trips with motors OFF, to check the "
                           "90 Hz command / 30 Hz read loop is actually sustainable.")
    mode.add_argument("--hold-middle", action="store_true")
    mode.add_argument("--test-middle-joint", type=int, choices=(1, 2, 3, 4))
    mode.add_argument("--zero-policy", action="store_true")
    mode.add_argument("--policy", type=Path)

    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--target-duration", type=float, default=None,
                        help="Override the scenario's per-target dwell. Passed through to "
                             "--parallel-mujoco too, so both backends always use the same "
                             "value; letting them differ would silently void the comparison.")
    parser.add_argument("--delta", type=float, default=0.03,
                        help="--test-middle-joint step, radians.")
    parser.add_argument("--command-hz", type=float, default=DEFAULT_COMMAND_HZ)
    parser.add_argument("--lowpass-hz", type=float, default=0.5,
                        help="wujihandpy filter.LowPass(cutoff_freq=...). Lower is gentler "
                             "but adds lag the simulators do not have.")
    # Start-pose arrival.  These are HARDWARE TUNING parameters: the tolerance
    # that a real servo can hold depends on this hand's stiction and gains, and
    # has not been characterised.  Start loose and tighten from measurement.
    parser.add_argument("--start-tolerance-rad", type=float, default=0.03)
    parser.add_argument("--start-stable-seconds", type=float, default=0.5)
    parser.add_argument("--start-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--read-source", choices=("controller", "hand"), default="controller")
    parser.add_argument("--enable-upstream", dest="enable_upstream", action="store_true",
                        default=True)
    parser.add_argument("--no-enable-upstream", dest="enable_upstream", action="store_false")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--current-limit", type=float, default=None,
                        help="Per-joint current limit in AMPERES, applied before enabling. "
                             "Left alone if omitted; the present values are printed either way.")
    parser.add_argument("--max-step-rad", type=float, default=None,
                        help="Reject any commanded target step larger than this (default: the "
                             "reach action scale, so a correct policy never triggers it).")
    parser.add_argument("--limit-margin", type=float, default=0.95, metavar="FRACTION",
                        help="Command only this fraction of each joint's range, measured as "
                             "centre +- FRACTION*half_range (Isaac's soft_joint_pos_limit_factor "
                             "definition).  1.0 is the trained action space and lets the policy "
                             "drive into the mechanical stop; the default 0.95 holds it off by "
                             "about 3 deg on the middle finger's Joint3/Joint4.  Numerical IK "
                             "reaches the whole target box at 1.00, 0.95 and 0.90 alike, so the "
                             "margin costs no reach.  Applies to the policy clamp, the held "
                             "joints and both glides.")
    parser.add_argument("--start-seconds", type=float, default=3.0,
                        help="How long the move to the start pose takes. Like the return, the "
                             "target is walked over this time so the commanded rate is "
                             "displacement/seconds. A fixed target instead starts at "
                             "displacement/tau, which is 21 rad/s if the finger happens to be "
                             "left flexed at 1.68 rad with a 2 Hz filter.")
    parser.add_argument("--return-seconds", type=float, default=3.0,
                        help="How long the return takes. The target is walked over this "
                             "time, so the commanded rate is displacement/seconds -- unlike "
                             "the outbound move, whose exponential approach is fastest at "
                             "its very first tick.")
    parser.add_argument("--return-to-start", action="store_true",
                        help="After the run, drive back to the start pose before disabling. "
                             "Without it the motors simply cut and the finger relaxes wherever "
                             "it ended -- fine for reach, not for anything holding an object.")
    parser.add_argument("--parallel-mujoco", action="store_true",
                        help="Run MuJoCo alongside on its OWN policy loop, viewer open. It is "
                             "not a state mirror: MuJoCo builds its own observation, runs its "
                             "own inference and integrates its own physics. Only the policy "
                             "file and the target command are shared, so what you are watching "
                             "is how differently the two plants follow the same command. "
                             "Started when the real policy starts so the phases line up; the "
                             "exact comparison is still the CSVs, not the eye.")
    parser.add_argument("--mujoco-python", type=Path,
                        default=Path("/home/lsc/anaconda3/envs/wuji_mujoco/bin/python"),
                        help="Interpreter that has mujoco installed (wujihandpy does not).")
    parser.add_argument("--log-commands", action="store_true",
                        help="Also log joint state at the full command rate to a second CSV. "
                             "Diagnostics only: the policy still samples at 30 Hz, because its "
                             "observation history is defined as two samples one policy step "
                             "apart. A read costs 0.009 ms, so this is close to free and shows "
                             "what happens between policy steps.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    return parser


def launch_parallel_mujoco(args, duration_s: float):
    """Start an INDEPENDENT MuJoCo run in the background.

    This does not send the hand's joint angles anywhere.  MuJoCo executes the
    same policy file against the same target sequence with its own observation,
    its own inference and its own physics, so the two runs are two plants
    following one command -- which is the thing worth watching side by side.

    The two environments are disjoint (wujihandpy lives in wuji_hw, mujoco in
    wuji_mujoco), so this shells out to the other interpreter rather than
    importing anything.  Its failure must never reach the hardware path.
    """

    import subprocess

    if not args.mujoco_python.is_file():
        print(f"[PARALLEL] skipped: {args.mujoco_python} not found")
        return None
    command = [
        str(args.mujoco_python), "-m", "Deploy.run.run_finger_reach",
        "--policy", str(args.policy),
        "--viewer", "--realtime",
        "--print-interval", "30",
    ]
    if args.scenario is not None:
        command += ["--scenario", str(args.scenario)]
    else:
        command += ["--target", *[str(v) for v in args.target]]
    # The dwell must match on both sides or the two logs describe different runs.
    command += ["--target-duration", str(duration_s / max(1, len(load_targets(args)[0])))]
    # The margin decides where the finger is allowed to stop, so an unmatched
    # pair would be two different action spaces, not two plants.
    command += ["--limit-margin", str(args.limit_margin)]
    if args.csv is not None:
        command += ["--csv", str(args.csv.with_name(args.csv.stem + "_mujoco.csv"))]
    try:
        process = subprocess.Popen(command, cwd=str(Path(__file__).resolve().parents[2]))
        print(f"[PARALLEL] MuJoCo launched (pid {process.pid}) running its own policy; "
              f"~0.5 s of startup means the phases align only roughly")
        return process
    except Exception as exc:  # pragma: no cover - the hand must not care
        print(f"[PARALLEL] failed to launch, continuing without it: {type(exc).__name__}: {exc}")
        return None


def confirm(prompt: str, skip: bool) -> None:
    if skip:
        print(f"{prompt}  [--yes]")
        return
    input(f"{prompt}\n준비됐으면 Enter >>> ")


def print_state(label: str, q_all: np.ndarray) -> None:
    print(f"\n===== {label} [rad] =====")
    for finger, row in enumerate(finger_major_grid(q_all), start=1):
        marker = "  <- policy" if finger - 1 in {int(i) // 4 for i in MIDDLE_POLICY_INDICES} else ""
        print(f"  finger{finger}: [" + ", ".join(f"{v:+.6f}" for v in row) + f"]{marker}")


def load_targets(args) -> tuple[list[np.ndarray], float]:
    if args.scenario is not None:
        scenario = json.loads(args.scenario.read_text())
        controlled = scenario.get("controlled_joints")
        if controlled is not None and list(controlled) != list(MIDDLE_JOINT_NAMES):
            raise ValueError(
                f"Scenario controls {controlled}; this run controls {list(MIDDLE_JOINT_NAMES)}."
            )
        targets = [np.asarray(t, dtype=np.float32) for t in scenario["targets_palm_m"]]
        return targets, float(args.target_duration or scenario["target_duration_s"])
    if args.target is not None:
        return [np.asarray(args.target, dtype=np.float32)], float(
            args.target_duration or args.seconds
        )
    centre = [sum(REACH_RANGE_M[a]) / 2.0 for a in ("x", "y", "z")]
    print(f"[TARGET]   none given; using the sampling-box centre {np.round(centre, 4).tolist()}")
    return [np.asarray(centre, dtype=np.float32)], float(
        args.target_duration or args.seconds
    )


def main() -> int:
    args = build_argument_parser().parse_args()

    # ---- everything checkable without hardware, before touching the hand ----
    policy = None
    targets: list[np.ndarray] = []
    duration = float(args.seconds)
    running_policy = args.zero_policy or args.policy is not None
    if running_policy:
        targets, duration = load_targets(args)
        for target in targets:
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError(f"Target must be a finite palm-frame xyz, got {target}.")
        if args.policy is not None:
            from ..policy.onnx_policy import OnnxPolicy

            policy = OnnxPolicy(args.policy, REACH_OBSERVATION_DIM, REACH_ACTION_DIM)
            print(f"[POLICY]   {policy.path}")
            print(f"           {policy.input.shape} -> {policy.output.shape}")
        else:
            policy = ZeroPolicy()
            print("[POLICY]   zero-action plumbing check (q_target == q_current)")

    backend = RealWujiHand(read_source=args.read_source, max_step_rad=args.max_step_rad)
    print(f"[HAND]     {backend.describe()}")
    q_now = backend.read_joint_positions()
    print_state("현재 관절 위치", q_now)

    lower, upper = backend.read_hardware_limits()
    out_of_range = np.flatnonzero(
        (FINGER_REACH_RESET_JOINT_POSITIONS < lower) | (FINGER_REACH_RESET_JOINT_POSITIONS > upper)
    )
    if out_of_range.size:
        raise RuntimeError(
            "The reach start pose is outside the hand's reported limits at "
            f"{[POLICY_JOINT_NAMES[i] for i in out_of_range]}."
        )

    print(f"[GUARD]    max target step {backend.max_step_rad:.4f} rad/policy step")
    try:
        limits = backend.read_current_limits()
        print(f"[CURRENT]  present limit {limits.min():.4g} ~ {limits.max():.4g} A")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[CURRENT]  could not read limits: {type(exc).__name__}: {exc}")

    if args.measure_timing:
        print("\n[TIMING]   measuring SDK round trips, motors OFF...")
        stats = backend.measure_io_timing(lowpass_hz=args.lowpass_hz)
        print(f"  controller read : mean {stats['read_mean_ms']:.3f} ms, "
              f"p95 {stats['read_p95_ms']:.3f}, max {stats['read_max_ms']:.3f}")
        print(f"  controller write: mean {stats['write_mean_ms']:.3f} ms, "
              f"p95 {stats['write_p95_ms']:.3f}, max {stats['write_max_ms']:.3f}")
        print(f"  (blocking SDO read for reference: {stats['blocking_read_ms']:.3f} ms -- "
              "not used by the loop)")
        command_budget_ms = 1000.0 / args.command_hz
        policy_budget_ms = 1000.0 * POLICY_DT
        print(f"\n  command tick budget {command_budget_ms:.3f} ms "
              f"(write p95 {stats['write_p95_ms']:.3f}) -> "
              f"{'OK' if stats['write_p95_ms'] < command_budget_ms else 'TOO SLOW'}")
        print(f"  policy tick budget  {policy_budget_ms:.3f} ms "
              f"(read p95 + write p95 = "
              f"{stats['read_p95_ms'] + stats['write_p95_ms']:.3f}) -> "
              f"{'OK' if stats['read_p95_ms'] + stats['write_p95_ms'] < policy_budget_ms else 'TOO SLOW'}")
        print(f"  joint drift while measuring: {stats['q_drift_rad']:.6f} rad")

        # Our numbers above are call rates. The vendor test is the only view of
        # the transport, so run it and report verbatim rather than guessing.
        print("\n[LATENCY]  running the SDK's own latency test for 3 s...")
        try:
            for line in backend.run_vendor_latency_test(3.0):
                print(f"    {line}")
        except Exception as exc:
            print(f"    unavailable: {type(exc).__name__}: {exc}")

        print("\n[TIMING]   no motor was enabled.")
        return 0

    if args.read_only:
        if not running_policy:
            print("\n[READ-ONLY] no motor was enabled.")
            return 0

        # Dry run: everything except transmission.  runner.command() only
        # *stores* a target in the backend, and publish_latest_target is never
        # called, so nothing reaches the hand.  No motor is enabled either.
        print("\n[DRY RUN]  building observations and decoding targets from the REAL "
              "encoder values.\n           Nothing is transmitted and no motor is enabled.")
        runner = MiddleFingerReachRunner(backend, policy, FingerReachObservationAdapter(),
                                         limit_fraction=args.limit_margin)
        runner.seed_from_current_state(backend.read_joint_positions())
        steps = max(1, int(round(args.seconds / POLICY_DT))) if args.seconds else 1
        steps_per_target = max(1, int(round(duration / POLICY_DT)))
        for step in range(steps):
            index = min(step // steps_per_target, len(targets) - 1)
            runner.set_target(targets[index])
            # One read per step, same as the real loop: two reads inside a
            # single tick differ by encoder noise, which showed up as a printed
            # delta of 0.101 rad against a 0.1 rad limit that was never exceeded.
            q_before = backend.read_joint_positions()
            if step > 0:
                runner.observe_after_hold(q_before)
            observation = runner.observations.build()
            decoded = runner.command(q_before)
            if step and step % 30 and step != steps - 1:
                continue
            middle = MIDDLE_POLICY_INDICES
            print(f"\n--- policy step {step}  (target #{index} "
                  f"{np.round(targets[index], 4).tolist()}) ---")
            print(f"  obs[ 0: 4] q_prev  norm  {np.round(observation[0:4], 5)}")
            print(f"  obs[ 4: 8] q_curr  norm  {np.round(observation[4:8], 5)}")
            print(f"  obs[ 8:11] target_palm   {np.round(observation[8:11], 5)}")
            print(f"  obs[11:15] last_action   {np.round(observation[11:15], 5)}")
            print(f"  raw action               {np.round(decoded.raw_action, 5)}")
            print(f"  clipped action           {np.round(decoded.clipped_action, 5)}")
            print(f"  q_current      [rad]     {np.round(q_before[middle], 5)}")
            print(f"  q_target       [rad]     {np.round(decoded.position_target, 5)}")
            print(f"  delta          [rad]     "
                  f"{np.round(decoded.position_target - q_before[middle], 5)}")
            clamped = ~np.isclose(decoded.position_target, decoded.unclamped_target)
            if clamped.any():
                names = [POLICY_JOINT_NAMES[middle[i]] for i in np.flatnonzero(clamped)]
                print(f"  command clamp hit        {names}")
        print("\n[READ-ONLY] finished. No motor was enabled, nothing was transmitted.")
        return 0

    scheduler = RealWujiScheduler(backend, command_hz=args.command_hz)
    if args.test_middle_joint is not None:
        mask = middle_enable_mask(args.test_middle_joint)
        label = f"finger3_joint{args.test_middle_joint} only, delta {args.delta:+.4f} rad"
    else:
        mask = middle_enable_mask()
        label = "middle finger (4 joints)"

    print(f"\n[ENABLE]   {label}")
    print(mask.astype(int))
    print(f"[RATES]    command {args.command_hz:.1f} Hz, policy {POLICY_HZ:.1f} Hz, "
          f"divider {scheduler.divider}")
    print(f"[FILTER]   LowPass cutoff {args.lowpass_hz:.2f} Hz")
    _soft = soft_command_limits(args.limit_margin)[MIDDLE_POLICY_INDICES]
    _hard = COMMAND_TARGET_LIMITS[MIDDLE_POLICY_INDICES]
    print(f"[LIMITS]   command margin {args.limit_margin:.3f} of range (centre +- f*half)"
          + ("  <- trained action space, stops reachable" if args.limit_margin >= 1.0 else ""))
    for _j, _n in enumerate(("J1", "J2", "J3", "J4")):
        print(f"             {_n} [{_soft[_j, 0]:+.4f}, {_soft[_j, 1]:+.4f}]  "
              f"upper held off by {1000 * (_hard[_j, 1] - _soft[_j, 1]):5.1f} mrad "
              f"({np.degrees(_hard[_j, 1] - _soft[_j, 1]):.2f} deg)")
    print(f"[START]    glide {args.start_seconds:.1f} s, tolerance {args.start_tolerance_rad:.4f} rad, "
          f"stable {args.start_stable_seconds:.2f} s, timeout {args.start_timeout_seconds:.1f} s")
    print(f"[READ]     joint state from the {args.read_source}"
          f" (upstream={'on' if args.enable_upstream else 'off'})")
    print(f"[CONTRACT] obs {REACH_OBSERVATION_DIM}D, action {REACH_ACTION_DIM}D, "
          f"joints {list(MIDDLE_JOINT_NAMES)} = canonical {MIDDLE_POLICY_INDICES.tolist()}")
    print("[GAINS]    firmware servo gains are UNVERIFIED; none are set from here.")
    if running_policy:
        print(f"[TARGETS]  {len(targets)} x {duration:.2f}s")
        for index, target in enumerate(targets):
            print(f"             [{index}] {np.round(target, 4).tolist()}")

    rows: list[list] = []
    command_rows: list[list] = []
    controller = None
    mirror = None
    try:
        if args.current_limit is not None:
            backend.write_current_limit(args.current_limit)
            applied = backend.read_current_limits()
            print(f"[CURRENT]  set to {args.current_limit:.4g} A "
                  f"(readback {applied.min():.4g} ~ {applied.max():.4g} A)")
        backend.prime_target_to_current()
        confirm("\n모터를 ENABLE 합니다. 이상하면 Ctrl+C.", args.yes)
        backend.enable(mask)
        print("[ENABLE]   done")

        with backend.realtime_controller(args.lowpass_hz, args.enable_upstream) as controller:
            backend.controller = controller

            if args.hold_middle:
                print(f"\n[HOLD]     holding the present pose for {args.seconds:.1f}s")
                scheduler.run(args.seconds, controller)

            elif args.test_middle_joint is not None:
                index = int(MIDDLE_POLICY_INDICES[args.test_middle_joint - 1])
                goal = backend.latest_target.copy()
                proposed = float(goal[index] + args.delta)
                low, high = COMMAND_TARGET_LIMITS[index]
                if not low <= proposed <= high:
                    safe = min(high, max(low, proposed)) - goal[index]
                    raise RuntimeError(
                        f"{POLICY_JOINT_NAMES[index]} would reach {proposed:+.4f} rad, outside "
                        f"[{low:+.4f}, {high:+.4f}]. Largest safe delta from here: {safe:+.4f} rad."
                    )
                goal[index] = proposed
                print(f"\n[TEST]     {POLICY_JOINT_NAMES[index]} "
                      f"{backend.latest_target[index]:+.6f} -> {proposed:+.6f} rad")
                # Diagnostic, separate from the start-pose move: a small fixed
                # step from the present q, held until the encoders confirm it.
                #
                # The tolerance must be well inside the step, or arrival is
                # satisfied before anything moves -- with --delta equal to the
                # default 0.03 rad tolerance the check passed at t=0 and the
                # joint only travelled as far as the stable window allowed.
                diagnostic_tolerance = min(args.start_tolerance_rad, abs(args.delta) / 4.0)
                print(f"[TEST]     arrival tolerance {diagnostic_tolerance:.5f} rad "
                      f"(a quarter of the {abs(args.delta):.4f} rad step)")
                scheduler.move_to_start_pose(
                    goal, controller,
                    joint_indices=MIDDLE_POLICY_INDICES,
                    tolerance_rad=diagnostic_tolerance,
                    stable_seconds=args.start_stable_seconds,
                    timeout_seconds=args.start_timeout_seconds,
                    limit_fraction=args.limit_margin,
                )
                after = backend.read_joint_positions()
                moved = after - q_now
                print_state("이동 후", after)
                biggest = int(np.argmax(np.abs(moved)))
                print(f"\n[TEST]     largest motion: {POLICY_JOINT_NAMES[biggest]} "
                      f"{moved[biggest]:+.6f} rad "
                      f"({'MATCHES' if biggest == index else 'DOES NOT MATCH'} the commanded joint)")

            else:
                runner = MiddleFingerReachRunner(backend, policy, FingerReachObservationAdapter(),
                                         limit_fraction=args.limit_margin)

                # Only the commanded finger is moved to the simulation start
                # pose.  The other sixteen keep the angles they are actually at:
                # they are not enabled, so overwriting their targets with the
                # simulation pose would command joints nobody is driving.
                q_now = backend.read_joint_positions()
                q_start_target = q_now.copy()
                q_start_target[MIDDLE_POLICY_INDICES] = FINGER_REACH_RESET_JOINT_POSITIONS[
                    MIDDLE_POLICY_INDICES
                ]

                print("\n[START TARGET]")
                print(f"  middle current  {np.round(q_now[MIDDLE_POLICY_INDICES], 6).tolist()}")
                print(f"  middle desired  "
                      f"{np.round(q_start_target[MIDDLE_POLICY_INDICES], 6).tolist()}")
                print(f"  difference      "
                      f"{np.round((q_start_target - q_now)[MIDDLE_POLICY_INDICES], 6).tolist()}")
                print("\n[START MOVE]")
                print(f"  command rate    {args.command_hz:.1f} Hz")
                print(f"  glide           {args.start_seconds:.1f} s, target walked linearly")
                print(f"  LowPass         {args.lowpass_hz:.2f} Hz")
                print(f"  tolerance       {args.start_tolerance_rad:.4f} rad")
                print(f"  stable          {args.start_stable_seconds:.2f} s")
                print(f"  timeout         {args.start_timeout_seconds:.1f} s")
                print("  no inference, no residual: this is a plain position command.")

                def report_move(elapsed, actual, error, max_error):
                    print(f"  t={elapsed:5.2f}s q={np.round(actual, 5).tolist()} "
                          f"err={np.round(error, 5).tolist()} max={max_error:.5f}")

                # Rate limited, same as the return.  A fixed target plus the filter
                # is an exponential: fastest at its very first tick, at
                # displacement/tau.  That is gentle when the finger is already
                # near the start pose, but the finger can be left anywhere -- a
                # run that ended flexed leaves it at 1.68 rad, and starting from
                # there would snap it open.  Walking the target keeps the
                # commanded rate at displacement/seconds regardless.
                elapsed, settled_all, error = scheduler.glide_to_pose(
                    q_start_target, controller,
                    joint_indices=MIDDLE_POLICY_INDICES,
                    seconds=args.start_seconds,
                    tolerance_rad=args.start_tolerance_rad,
                    stable_seconds=args.start_stable_seconds,
                    timeout_seconds=args.start_timeout_seconds,
                    report=report_move,
                    limit_fraction=args.limit_margin,
                )
                print("\n[START REACHED]")
                print(f"  elapsed         {elapsed:.2f} s")
                print(f"  actual          "
                      f"{np.round(settled_all[MIDDLE_POLICY_INDICES], 6).tolist()}")
                print(f"  desired         "
                      f"{np.round(q_start_target[MIDDLE_POLICY_INDICES], 6).tolist()}")
                print(f"  error           {np.round(error, 6).tolist()}")

                # Re-read: seed from what the encoders say, never from the pose
                # that was commanded.
                settled = backend.read_joint_positions()
                runner.seed_from_current_state(settled)
                print("\n[POLICY SEED]")
                print(f"  q_prev          {np.round(runner.observations.q_previous, 6).tolist()}")
                print(f"  q_curr          {np.round(runner.observations.q_current, 6).tolist()}")
                print(f"  last_action     {[0.0] * REACH_ACTION_DIM}")

                steps_per_target = max(1, int(round(duration / POLICY_DT)))
                state = {"target_index": -1}

                def on_policy_tick(policy_index: int, elapsed: float) -> None:
                    index = min(policy_index // steps_per_target, len(targets) - 1)
                    if index != state["target_index"]:
                        state["target_index"] = index
                        runner.set_target(targets[index])
                        print(f"[TARGET]   #{index} {np.round(targets[index], 4).tolist()} "
                              f"at t={elapsed:6.2f}s")

                    began = time.monotonic()
                    # ONE read per policy tick, used for both the history and the
                    # residual base.  Two reads would differ by encoder noise.
                    q_all = backend.read_joint_positions()

                    # Advance the history FIRST.  This q is the result of the
                    # previous target having been published for a full policy
                    # step; MuJoCo gets the same effect from its 64 physics
                    # substeps between command() and observe_after_hold().
                    # Doing it after command() instead would set
                    # q_previous == q_current and the policy would see no motion.
                    if policy_index > 0:
                        runner.observe_after_hold(q_all)

                    q_prev = runner.observations.q_previous
                    decoded = runner.command(q_all)
                    inference_ms = (time.monotonic() - began) * 1000.0
                    rows.append(
                        [elapsed, time.time(), policy_index, policy_index * scheduler.divider,
                         index, *targets[index]]
                        + list(q_prev) + list(runner.observations.q_current)
                        + list(decoded.clipped_action) + list(decoded.position_target)
                        + [inference_ms]
                    )
                    if policy_index % 30 == 0:
                        q = runner.observations.q_current
                        print(f"  t={elapsed:6.2f}s tgt#{index} "
                              f"q=[{q.min():+.3f},{q.max():+.3f}] "
                              f"a=[{decoded.clipped_action.min():+.3f},"
                              f"{decoded.clipped_action.max():+.3f}] "
                              f"({inference_ms:.2f} ms)")

                command_rows: list[list] = []

                def on_command_tick(tick: int, elapsed: float) -> None:
                    q = backend.read_joint_positions()
                    command_rows.append(
                        [elapsed, tick, state["target_index"]]
                        + list(q[MIDDLE_POLICY_INDICES])
                        + list(backend.latest_target[MIDDLE_POLICY_INDICES])
                    )

                total = duration * len(targets)
                print(f"\n[RUN]      {total:.1f}s, {len(targets)} target(s)")
                if args.parallel_mujoco and args.policy is not None:
                    mirror = launch_parallel_mujoco(args, total)
                scheduler.run(
                    total, controller, on_policy_tick,
                    on_command_tick=on_command_tick if args.log_commands else None,
                )

                # A failed return must not cost the run's data: by this point the
                # policy rollout is finished and its logs are the reason the run
                # happened.  Report it and fall through to the CSV write.
                if args.return_to_start:
                    # Rate limited, unlike the outbound move.  The return can
                    # span the whole flexion the policy produced, and a fixed
                    # target over that distance starts at displacement/tau --
                    # measured at 3.7 rad/s for a 1.18 rad return, faster than
                    # the policy's own ceiling.
                    print(f"\n[RETURN]   gliding back to the start pose over "
                          f"{args.return_seconds:.1f}s before disabling")
                    q_end = backend.read_joint_positions()
                    q_home = q_end.copy()
                    q_home[MIDDLE_POLICY_INDICES] = q_start_target[MIDDLE_POLICY_INDICES]
                    try:
                        elapsed, back, error = scheduler.glide_to_pose(
                            q_home, controller,
                            joint_indices=MIDDLE_POLICY_INDICES,
                            seconds=args.return_seconds,
                            tolerance_rad=args.start_tolerance_rad,
                            stable_seconds=args.start_stable_seconds,
                            timeout_seconds=args.start_timeout_seconds,
                            report=lambda e, a, err, m: print(
                                f"  t={e:5.2f}s q={np.round(a, 5).tolist()} max={m:.5f}"
                            ),
                            limit_fraction=args.limit_margin,
                        )
                        print(f"[RETURNED] {elapsed:.2f} s, "
                              f"q={np.round(back[MIDDLE_POLICY_INDICES], 6).tolist()}, "
                              f"err={np.round(error, 6).tolist()}")
                    except Exception as exc:
                        print(f"[RETURN]   FAILED: {type(exc).__name__}: {exc}")
                        print("[RETURN]   motors will still be disabled; logs are kept.")

        print(f"\n[TIMING]   {scheduler.timing.summary()}")

    except KeyboardInterrupt:
        print("\n\nCtrl+C 감지")
    finally:
        if mirror is not None and mirror.poll() is None:
            # It started ~1 s later than the hand, so it is still mid-run when
            # the hand finishes.  Killing it here loses its CSV, which is the
            # whole reason for running it.  Give it the offset back, then stop.
            print("[PARALLEL] waiting for MuJoCo to finish and write its log...")
            try:
                mirror.wait(timeout=30.0)
                print("[PARALLEL] MuJoCo finished")
            except Exception:
                mirror.terminate()
                print("[PARALLEL] MuJoCo did not finish in 30 s; terminated")
        # Leave the realtime controller first (the with-block above has already
        # exited by here), then cut every motor.  This runs on every path.
        backend.controller = None
        try:
            backend.disable()
            print("[DISABLE]  all joints off")
        except Exception as exc:  # pragma: no cover - hardware teardown
            print(f"[DISABLE]  FAILED: {exc}")

    if command_rows and args.csv is not None:
        path = args.csv.with_name(args.csv.stem + "_90hz.csv")
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time", "command_tick", "target_index"]
                + [f"q_curr_{k}" for k in range(1, 5)]
                + [f"q_target_{k}" for k in range(1, 5)]
            )
            writer.writerows(command_rows)
        print(f"wrote {len(command_rows)} command-rate rows -> {path}")

    if rows and args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
