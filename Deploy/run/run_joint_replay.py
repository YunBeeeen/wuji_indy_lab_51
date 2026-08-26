#!/usr/bin/env python3
# [run/양쪽] 녹화한 Isaac 관절 목표를 정책 없이 재생. MuJoCo와 실물이 같은 코드.
"""Replay a logged Isaac joint-target trajectory on MuJoCo or the real hand.

The input is a CSV written by ``play.py``'s ``M`` key (see ``HandJointRecorder``).
Its ``qt_*`` columns are the PD targets the trained policy commanded, one row
per 30 Hz policy step.  Those - not the ``q_*`` measurements - are what a
position-controlled hand can be given: the target sits ahead of the measurement
by the contact preload, and it is that gap times Kp that holds the sticks.
Replaying measured angles would command zero PD error, i.e. no grip.

Two things are deliberately decoupled:

* **Command rate** stays at ``--command-hz`` (90 Hz by default) on hardware,
  matching ``real_wuji_scheduler``.  It is a transmission rate, not a speed.
* **Playback speed** is set by ``--max-joint-speed`` (or ``--speed``).  Slowing
  down stretches the time each logged row occupies and linearly interpolates
  between rows, exactly as ``/home/lsc/wuji_test/move_all.py`` walks a target
  rather than stepping it.  The recorded trajectory peaks at 0.94 rad/s, about
  4.7x the 0.20 rad/s that script uses, so it is played slower by default.
  Time-scaling preserves the grip: the preload is a *position* offset, so it
  survives unchanged while velocities and impacts shrink.

Nothing here enables a motor without an explicit confirmation, and
``--read-only`` transmits nothing at all.
"""

from __future__ import annotations

import argparse
import csv
import json
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..common.policy_contract import (
    COMMAND_TARGET_LIMITS,
    POLICY_DT,
    POLICY_JOINT_NAMES,
    soft_command_limits,
)

HAND_JOINTS = len(POLICY_JOINT_NAMES)
# The speed /home/lsc/wuji_test/move_all.py walks its target at, and the only
# whole-hand rate that has actually been run on this hardware.
VALIDATED_JOINT_SPEED_RAD_S = 0.20

PREGRASP_JOINT_POSITIONS = np.array(
    [
        0.5377866626,
        0.8436813951,
        0.0377136655,
        -0.0000001810,

        0.7017297745,
        0.0553143807,
        1.1822255850,
        1.4215219021,

        0.4649881423,
        -0.0292181600,
        1.6272000000,
        1.1032750607,

        0.9151425958,
        -0.0129909236,
        1.3248542547,
        0.3182539344,

        0.7154092789,
        0.0788998753,
        1.6272000000,
        0.2546040118,
    ],
    dtype=np.float32,
)
# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class Trajectory:
    """A logged 20-joint target sequence, validated against the contract."""

    def __init__(self, csv_path: Path, limit_fraction: float, segment: int | None = None):
        self.csv_path = Path(csv_path).expanduser().resolve()
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"CSV does not exist: {self.csv_path}")
        rows = list(csv.DictReader(self.csv_path.open(encoding="utf-8")))
        if not rows:
            raise ValueError(f"CSV has no rows: {self.csv_path}")

        # Joint identity comes from the file, never from column position.  The
        # recorder resolves Isaac's USD DOF order (which is joint-major) by
        # name; this is where that resolution is checked to still hold.
        meta_path = self.csv_path.with_suffix(".meta.json")
        if meta_path.is_file():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            names = list(self.meta["joint_names"])
        else:
            # A session killed before the M key stopped it has no meta; fall
            # back to the header, which carries the same names.
            self.meta = {}
            names = [k[3:] for k in rows[0] if k.startswith("qt_")]
        if names != list(POLICY_JOINT_NAMES):
            raise ValueError(
                "Recorded joint order does not match POLICY_JOINT_NAMES.\n"
                f"  csv      {names}\n  contract {list(POLICY_JOINT_NAMES)}"
            )
        self.joint_names = names

        if "qt_" + names[0] not in rows[0]:
            raise ValueError(
                f"{self.csv_path.name} has no qt_* columns, so it holds measured angles "
                "only.  Re-record with a play.py that logs joint_pos_target: replaying "
                "measurements would command zero PD error and no grip."
            )

        if segment is not None:
            rows = [r for r in rows if int(r["segment"]) == segment]
            if not rows:
                raise ValueError(f"CSV has no segment {segment}.")
        self.segment = segment
        self.modes = [r["mode"] for r in rows]
        self.segments = np.array([int(r["segment"]) for r in rows])

        raw = np.array(
            [[float(r["qt_" + n]) for n in names] for r in rows], dtype=np.float32
        )
        if not np.isfinite(raw).all():
            raise ValueError("Recorded targets contain non-finite values.")
        self.raw = raw

        # Clamp the trajectory itself, once, before anything is transmitted.
        # Every interpolated point is then in range by construction.  This is
        # not silent clipping of a live policy output: the clamped amount is
        # reported below so a margin that changes the motion is visible.
        limits = soft_command_limits(limit_fraction)
        self.limit_fraction = float(limit_fraction)
        self.targets = np.clip(raw, limits[:, 0], limits[:, 1]).astype(np.float32)
        self.clamped = np.abs(self.targets - raw)

    @property
    def rows(self) -> int:
        return len(self.targets)

    @property
    def duration_s(self) -> float:
        return self.rows * POLICY_DT

    def peak_joint_speed(self) -> float:
        if self.rows < 2:
            return 0.0
        return float(np.abs(np.diff(self.targets, axis=0)).max() / POLICY_DT)

    def clamp_report(self) -> str:
        hit = np.flatnonzero(self.clamped.max(0) > 1e-9)
        if hit.size == 0:
            return f"  no joint was clamped at margin {self.limit_fraction:.3f}"
        lines = []
        for i in hit:
            n = int((self.clamped[:, i] > 1e-9).sum())
            worst = float(self.clamped[:, i].max())
            lines.append(
                f"  {self.joint_names[i]:<16} {n:4d}/{self.rows} rows, "
                f"up to {worst * 1000:6.2f} mrad ({np.degrees(worst):5.2f} deg)"
            )
        return "\n".join(lines)

    def segment_report(self) -> str:
        lines = []
        for s in sorted(set(self.segments.tolist())):
            m = self.segments == s
            lines.append(
                f"  segment {s}  {self.modes[int(np.flatnonzero(m)[0])]:<5} "
                f"{int(m.sum()):4d} rows  {int(m.sum()) * POLICY_DT:5.2f}s"
            )
        return "\n".join(lines)


def resolve_speed(trajectory: Trajectory, speed, max_joint_speed) -> float:
    """Pick the playback factor, from an explicit speed or a rad/s ceiling."""
    if speed is not None and max_joint_speed is not None:
        raise ValueError("Pass --speed or --max-joint-speed, not both.")
    if speed is not None:
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError(f"--speed must be positive and finite, got {speed}.")
        return float(speed)
    ceiling = VALIDATED_JOINT_SPEED_RAD_S if max_joint_speed is None else float(max_joint_speed)
    if not np.isfinite(ceiling) or ceiling <= 0.0:
        raise ValueError(f"--max-joint-speed must be positive and finite, got {ceiling}.")
    peak = trajectory.peak_joint_speed()
    if peak <= 0.0:
        return 1.0
    # Never speed the trajectory UP to meet a generous ceiling; the recorded
    # cadence is the fastest this should ever run.
    return float(min(1.0, ceiling / peak))


def command_schedule(trajectory: Trajectory, speed: float, command_hz: float):
    """Return the interpolated command-rate target stream.

    One logged row spans ``command_hz * POLICY_DT / speed`` commands.  The
    result starts at row 0 and ends exactly on the final row.
    """
    per_row = command_hz * POLICY_DT / speed
    if per_row < 1.0:
        raise ValueError(
            f"speed {speed:.3f} at {command_hz:.1f} Hz would need {per_row:.2f} commands "
            "per logged row; lower the speed or raise --command-hz."
        )
    n = int(round(per_row))
    targets = trajectory.targets
    if len(targets) < 2:
        return targets.copy(), n
    total = (len(targets) - 1) * n + 1
    position = np.arange(total, dtype=np.float64) / n
    lower = np.clip(np.floor(position).astype(int), 0, len(targets) - 2)
    alpha = (position - lower)[:, None]
    stream = (1.0 - alpha) * targets[lower] + alpha * targets[lower + 1]
    return stream.astype(np.float32), n

def timed_glide_real(
    backend,
    controller,
    q_from,
    q_to,
    command_hz,
    seconds,
    max_step_rad,
    label,
):
    """Actual convergence를 요구하지 않고 target을 선형으로 이동시킨다."""

    q_from_raw = np.asarray(q_from, dtype=np.float32)
    q_to_raw = np.asarray(q_to, dtype=np.float32)
    lower = COMMAND_TARGET_LIMITS[:, 0]
    upper = COMMAND_TARGET_LIMITS[:, 1]

    # Measured q can legitimately sit just outside the deployment envelope
    # while the motors are off.  The trajectory CSV is clamped when loaded,
    # but this bring-up/return helper also receives raw encoder poses.  Starting
    # interpolation from that raw value leaves the first few targets outside
    # COMMAND_TARGET_LIMITS and the backend correctly refuses them (measured:
    # finger1_joint2 -0.214788 against the -0.213203 lower bound).  A command
    # trajectory cannot reproduce the out-of-envelope part of a measured pose,
    # so clamp both endpoints explicitly and keep every blended sample inside
    # the same one hardware envelope used by all other real-hand commands.
    q_from = np.clip(q_from_raw, lower, upper).astype(np.float32)
    q_to = np.clip(q_to_raw, lower, upper).astype(np.float32)

    for endpoint, raw, clipped in (
        ("start", q_from_raw, q_from),
        ("goal", q_to_raw, q_to),
    ):
        hit = np.flatnonzero(np.abs(clipped - raw) > 1e-9)
        if hit.size:
            details = ", ".join(
                f"{POLICY_JOINT_NAMES[j]} {raw[j]:+.6f}->{clipped[j]:+.6f}"
                for j in hit
            )
            print(f"[{label}] {endpoint} pose command-limit clamp: {details}")

    dt = 1.0 / command_hz
    ticks = max(1, int(np.ceil(seconds * command_hz)))

    delta = q_to - q_from
    biggest = float(np.max(np.abs(delta)))

    print(
        f"[{label}] max move {biggest:.4f} rad over {seconds:.2f}s "
        f"-> {biggest / seconds:.4f} rad/s"
    )

    start = time.monotonic()

    for i in range(ticks):
        alpha = (i + 1) / ticks

        target = np.clip(
            (1.0 - alpha) * q_from
            + alpha * q_to,
            lower,
            upper,
        ).astype(np.float32)

        backend.write_joint_position_targets(
            target,
            max_step_rad=max_step_rad,
        )
        backend.publish_latest_target(controller)

        deadline = start + (i + 1) * dt
        wait = deadline - time.monotonic()

        if wait > 0:
            time.sleep(wait)

    return seconds

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay a logged Isaac joint-target trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", type=Path, required=True,
                   help="joint_record_*.csv written by play.py's M key.")
    p.add_argument("--backend", choices=("mujoco", "real"), default="mujoco")
    p.add_argument("--segment", type=int, default=None,
                   help="Replay only this OPEN/CLOSE segment (default: the whole file).")
    p.add_argument("--keyboard-open-close", action="store_true",
                   help="Real backend: use 1=OPEN, 2=CLOSE, q=quit with segments from this CSV.")
    p.add_argument("--open-segment", type=int, default=None,
                   help="OPEN segment for keyboard mode (default: last OPEN segment).")
    p.add_argument("--close-segment", type=int, default=None,
                   help="CLOSE segment for keyboard mode (default: last CLOSE segment).")

    p.add_argument("--limit-margin", type=float, default=0.95, metavar="FRACTION",
                   help="Command only this fraction of each joint's range, as "
                        "centre +- f*half. 1.0 is the space the policy trained in.")
    speed = p.add_mutually_exclusive_group()
    speed.add_argument("--speed", type=float, default=None,
                       help="Playback factor. 1.0 replays at the recorded 30 Hz.")
    speed.add_argument("--max-joint-speed", type=float, default=None, metavar="RAD_S",
                       help=f"Derive the factor from this ceiling "
                            f"(default {VALIDATED_JOINT_SPEED_RAD_S}, the rate move_all.py uses).")
    p.add_argument("--command-hz", type=float, default=90.0)

    p.add_argument("--start-seconds", type=float, default=3.0,
                   help="Glide from the present pose to the first logged target over this long.")
    p.add_argument("--settle-seconds", type=float, default=2.0,
                   help="Hold the first target before replay starts, so the filter catches up.")
    p.add_argument("--hold-seconds", type=float, default=2.0,
                   help="Hold the final target after replay ends.")

    p.add_argument("--dry-run", action="store_true",
                   help="Validate, print the plan, and exit without touching any backend.")
    p.add_argument("--out", type=Path, default=None,
                   help="Write the measured response to this CSV.")

    mj = p.add_argument_group("mujoco")
    mj.add_argument("--gains", choices=("vendor", "isaac_tuned"), default="isaac_tuned",
                    help="isaac_tuned reproduces the Kp the policy was trained against.")
    mj.add_argument(
        "--hold-sticks",
        type=str,
        default="glide",
        choices=("glide", "always", "never"),
        metavar="WHEN",
        help=(
            "Pin the sticks at their reset pose while the hand closes onto "
            "them -- the simulator's stand-in for a person holding the "
            "chopsticks. 'glide' (default) pins during the approach and settle, "
            "then lets go when replay starts. 'always' never lets go, which "
            "measures joint tracking with the grasp taken out of the picture. "
            "'never' is the old behaviour: free from step one, so the hand "
            "closes on objects that are already falling."
        ),
    )
    mj.add_argument("--viewer", action="store_true")

    rl = p.add_argument_group("real")
    rl.add_argument("--read-only", action="store_true",
                    help="Connect and read, but enable no motor and transmit nothing.")
    rl.add_argument("--yes", action="store_true", help="Skip the ENABLE confirmation.")
    rl.add_argument("--lowpass-hz", type=float, default=0.5)
    rl.add_argument("--current-limit", type=float, default=None, metavar="AMPERES")
    rl.add_argument("--max-step-rad", type=float, default=0.05,
                    help="Reject a target this far from the measured position.")
    rl.add_argument("--read-source", choices=("controller", "hand"), default="controller")
    rl.add_argument("--enable-upstream", dest="enable_upstream", action="store_true", default=True)
    rl.add_argument("--no-enable-upstream", dest="enable_upstream", action="store_false")
    rl.add_argument("--return-to-start", action="store_true",
                    help="Glide back to the pose the run started from before disabling.")
    rl.add_argument("--no-read-during-replay", dest="read_during_replay",
                    action="store_false", default=True,
                    help="Do not read encoders inside the realtime loop.  Reading there is "
                         "not a vendor-demonstrated pattern; drop it if the loop runs late.")
    return p


def describe_plan(traj: Trajectory, speed: float, per_row: int, stream_len: int, args) -> None:
    peak = traj.peak_joint_speed()
    print(f"[CSV]      {traj.csv_path}")
    if traj.meta:
        print(f"           task={traj.meta.get('task')}  "
              f"recorded={traj.meta.get('recorded_at')}  "
              f"stopped_by={traj.meta.get('stopped_by')}")
    print(f"           {traj.rows} rows = {traj.duration_s:.2f}s at {1 / POLICY_DT:.0f} Hz")
    print(traj.segment_report())
    print(f"[ORDER]    joint names match POLICY_JOINT_NAMES")
    print(f"[LIMITS]   margin {traj.limit_fraction:.3f} of range (centre +- f*half)"
          + ("  <- trained action space, stops reachable" if traj.limit_fraction >= 1.0 else ""))
    print(traj.clamp_report())
    print(f"[SPEED]    factor {speed:.3f}: peak joint speed "
          f"{peak:.3f} -> {peak * speed:.3f} rad/s "
          f"({np.degrees(peak * speed):.1f} deg/s)")
    print(f"           replay {traj.duration_s:.2f}s -> {traj.duration_s / speed:.1f}s, "
          f"{per_row} commands per logged row at {args.command_hz:.1f} Hz "
          f"({stream_len} commands total)")
    print(f"[PHASES]   glide {args.start_seconds:.1f}s -> settle {args.settle_seconds:.1f}s -> "
          f"replay {traj.duration_s / speed:.1f}s -> hold {args.hold_seconds:.1f}s")


def write_response(path: Path, names, stream, actual, extra=None) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["tick", "time_s", *(f"cmd_{n}" for n in names), *(f"act_{n}" for n in names)]
        if extra is not None:
            header += [f"stick{i}_{c}" for i in (1, 2)
                       for c in ("px", "py", "pz", "qw", "qx", "qy", "qz")]
        writer.writerow(header)
        for i, (c, a) in enumerate(zip(stream, actual)):
            row = [i, f"{i * (1.0 / write_response.command_hz):.6f}",
                   *(f"{v:.9f}" for v in c), *(f"{v:.9f}" for v in a)]
            if extra is not None:
                row += [f"{v:.9f}" for v in np.asarray(extra[i]).reshape(-1)]
            writer.writerow(row)
    print(f"[OUT]      {path}")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def run_mujoco(traj: Trajectory, stream, args) -> int:
    from ..backends.mujoco_wuji import MujocoWujiHand

    backend = MujocoWujiHand(controller_gains=args.gains)
    print(f"[MUJOCO]   {backend.model_summary()}")
    kp_lo, kp_hi, kv_lo, kv_hi = backend.gain_summary()
    print(f"[GAINS]    {args.gains}: kp {kp_lo:.3f}~{kp_hi:.3f}, kv {kv_lo:.4f}~{kv_hi:.4f}")

    backend.reset()
    has_sticks = getattr(backend, "has_sticks", False)
    if has_sticks:
        print("[SCENE]    sticks present; palm-frame poses will be logged")

    # One command tick's worth of physics, derived rather than assumed: the
    # model's timestep already spans one policy step in `physics_substeps`.
    steps_per_command = max(1, int(round((1.0 / args.command_hz) / backend.model.opt.timestep)))
    print(f"[TIMING]   {backend.model.opt.timestep * 1000:.4f} ms physics step, "
          f"{steps_per_command} per command tick")

    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(backend.model, backend.data)

    # 스틱을 리셋 자세에 붙들어 두는 장치.
    #
    # 실물에서는 사람이 젓가락을 쥐고 있다가 손이 닫히면 놓는다. MuJoCo 에는
    # 그 손이 없으므로, 접근·정착 구간 동안 매 물리 스텝마다 스틱 상태를 리셋
    # 자세로 되돌려 같은 역할을 시킨다.  ``set_stick_poses_in_palm`` 이 속도까지
    # 0 으로 만들고 ``mj_forward`` 를 부르므로 그대로 쓴다.
    #
    # 겹침은 누적되지 않는다. 매번 정확히 같은 자세로 되돌리므로 관통량은 리셋
    # 시점 값(palm<->stick2 -3.66 mm)에 머무르고, 그게 곧 파지 preload 다.
    #
    # 왜 필요한가: 이 모델은 놓아두면 스틱을 잡지 못한다. 2026-08-20 실측으로
    # 목표를 pregrasp 에 고정해도 Isaac 은 Stick1 을 0.65 초에 놓치고 MuJoCo 는
    # Stick2 를 5 초에 놓친다. 파지는 정책이 매 스텝 붙잡아서 유지되는 것이라,
    # 열린 루프 재생만으로는 처음부터 쥐고 있는 상태를 만들 수 없다.
    reset_stick_poses = (
        backend.get_stick_poses_in_palm().reshape(-1, 7).copy() if has_sticks else None
    )
    if reset_stick_poses is not None and args.hold_sticks != "never":
        print("[STICKS]   hold=" + args.hold_sticks + ": "
              + ("접근·정착 동안 고정하고 재생 시작에 놓는다"
                 if args.hold_sticks == "glide" else "끝까지 고정한다 (파지를 재지 않음)"))

    def drive(target, ticks, pin_sticks=False):
        pin = pin_sticks and reset_stick_poses is not None
        for _ in range(ticks):
            backend.write_joint_position_targets(target)
            if pin:
                for _ in range(steps_per_command):
                    backend.step(1)
                    backend.set_stick_poses_in_palm(reset_stick_poses)
            else:
                backend.step(steps_per_command)
            if viewer is not None:
                viewer.sync()

    try:
        q_start = backend.read_joint_positions()
        glide_ticks = max(1, int(round(args.start_seconds * args.command_hz)))
        print(f"[GLIDE]    {float(np.abs(stream[0] - q_start).max()):.4f} rad max "
              f"over {args.start_seconds:.1f}s")
        pin_now = args.hold_sticks in ("glide", "always")
        for tick in range(glide_ticks):
            alpha = (tick + 1) / glide_ticks
            drive(((1.0 - alpha) * q_start + alpha * stream[0]).astype(np.float32), 1,
                  pin_sticks=pin_now)
        drive(stream[0], max(1, int(round(args.settle_seconds * args.command_hz))),
              pin_sticks=pin_now)
        if args.hold_sticks == "glide" and reset_stick_poses is not None:
            print("[STICKS]   놓는다 - 여기서부터 스틱은 자유다")

        actual = np.empty_like(stream)
        sticks = [] if has_sticks else None
        
        print(f"[REPLAY]   {len(stream)} commands")

        dt = 1.0 / args.command_hz
        start = time.monotonic()

        # 'always' 는 재생 내내 고정한다. 파지를 빼고 관절 추종만 재는 모드다.
        pin_replay = args.hold_sticks == "always"

        for i, target in enumerate(stream):
            drive(target, 1, pin_sticks=pin_replay)
            actual[i] = backend.read_joint_positions()

            # 한 번만 기록한다. 예전 코드는 sleep 앞뒤로 두 번 append 해서
            # 행 수가 stream 의 2배가 됐고, CSV 헤더(스틱 14열)와 어긋났다.
            if sticks is not None:
                sticks.append(backend.get_stick_poses_in_palm())

            deadline = start + (i + 1) * dt
            wait = deadline - time.monotonic()

            if wait > 0:
                time.sleep(wait)

        drive(stream[-1], max(1, int(round(args.hold_seconds * args.command_hz))),
              pin_sticks=pin_replay)

        error = np.abs(actual - stream)
        print(f"[TRACK]    |cmd-act| mean {error.mean():.5f} rad "
              f"({np.degrees(error.mean()):.3f} deg), max {error.max():.5f} rad "
              f"at {POLICY_JOINT_NAMES[int(error.max(0).argmax())]}")
        if sticks is not None:
            s = np.asarray(sticks)
            drift = np.linalg.norm(s[:, :, :3] - s[0, :, :3], axis=-1)
            print(f"[STICKS]   palm-frame drift from the first command: "
                  f"stick1 {drift[-1, 0] * 1000:.2f} mm (max {drift[:, 0].max() * 1000:.2f}), "
                  f"stick2 {drift[-1, 1] * 1000:.2f} mm (max {drift[:, 1].max() * 1000:.2f})")
            if drift.max() > 0.05:
                print("[STICKS]   WARNING: over 50 mm of drift; the grasp did not survive.")
        if args.out is not None:
            write_response.command_hz = args.command_hz
            write_response(args.out, traj.joint_names, stream, actual,
                           extra=np.asarray(sticks).reshape(len(stream), -1) if sticks else None)
    finally:
        if viewer is not None:
            # 그냥 close() 만 하면 종료 시 segfault 가 난다. MuJoCo 의
            # Handle.__exit__ 는 비동기 종료를 요청할 뿐 리눅스 뷰어 스레드를
            # join 하지 않아서, 그 스레드가 GLFW 를 정리하는 사이 파이썬이
            # MjModel/MjData 를 파괴해 버린다. run_policy.py 가 같은 이유로
            # _close_viewer_and_wait 를 쓰며, 여기서도 그걸 그대로 쓴다.
            from .run_policy import _close_viewer_and_wait

            _close_viewer_and_wait(viewer)
    return 0


def confirm(prompt: str, skip: bool) -> None:
    if skip:
        print(f"{prompt}  [--yes]")
        return
    if input(f"{prompt}  계속하려면 'yes': ").strip().lower() != "yes":
        raise SystemExit("중단했습니다.")


def _keyboard_segments(csv_path: Path, limit_fraction: float, open_segment, close_segment):
    """Resolve one OPEN and one CLOSE segment from a single recorder CSV."""
    whole = Trajectory(csv_path, limit_fraction)
    mode_by_segment = {}
    for segment, mode in zip(whole.segments.tolist(), whole.modes):
        previous = mode_by_segment.setdefault(int(segment), mode)
        if previous != mode:
            raise ValueError(f"segment {segment} contains both {previous} and {mode} modes")

    def pick(requested, mode):
        candidates = [s for s, recorded_mode in mode_by_segment.items() if recorded_mode == mode]
        if not candidates:
            raise ValueError(f"CSV has no {mode} segment.")
        selected = max(candidates) if requested is None else int(requested)
        if mode_by_segment.get(selected) != mode:
            raise ValueError(
                f"segment {selected} is {mode_by_segment.get(selected, 'missing')}, not {mode}."
            )
        return selected

    open_id = pick(open_segment, "OPEN")
    close_id = pick(close_segment, "CLOSE")
    return (
        Trajectory(csv_path, limit_fraction, open_id),
        Trajectory(csv_path, limit_fraction, close_id),
        open_id,
        close_id,
    )


def _read_replay_key() -> str:
    """Read one key immediately from a terminal, while still allowing Ctrl+C."""
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def run_real(traj: Trajectory, stream, args, keyboard_streams=None) -> int:
    from ..backends.real_wuji import RealWujiHand
    from ..backends.real_wuji_scheduler import RealWujiScheduler

    backend = RealWujiHand(read_source=args.read_source, max_step_rad=args.max_step_rad, step_guard_reference="command",)
    print(f"[HAND]     {backend.describe()}")
    q_now = backend.read_joint_positions()
    print(f"[STATE]    현재 자세 min {q_now.min():+.4f} max {q_now.max():+.4f} rad")

    # PREGRASP도 실기 command margin을 동일하게 적용한다.
    soft_limits = soft_command_limits(args.limit_margin)

    pregrasp = np.clip(
        PREGRASP_JOINT_POSITIONS,
        soft_limits[:, 0],
        soft_limits[:, 1],
    ).astype(np.float32)

    pregrasp_clamp = np.abs(
        pregrasp - PREGRASP_JOINT_POSITIONS
    )

    hit = np.flatnonzero(pregrasp_clamp > 1e-9)

    if hit.size:
        print("[PREGRASP]  limit-margin 때문에 수정된 관절:")
        for i in hit:
            print(
                f"            {POLICY_JOINT_NAMES[i]:<16} "
                f"{PREGRASP_JOINT_POSITIONS[i]:+.6f} "
                f"-> {pregrasp[i]:+.6f}"
            )

    lower, upper = backend.read_hardware_limits()
    outside = np.flatnonzero((stream.min(0) < lower) | (stream.max(0) > upper))
    if outside.size:
        raise RuntimeError(
            "Trajectory leaves the hand's own reported limits at "
            f"{[POLICY_JOINT_NAMES[i] for i in outside]}."
        )
    print("[CHECK]    전 구간이 하드웨어 보고 한계 안에 있습니다")
    print(f"[GUARD]    max target step {backend.max_step_rad:.4f} rad")
    print(f"[GLIDE]    {float(np.abs(pregrasp - q_now).max()):.4f} rad max "
          f"over {args.start_seconds:.1f}s")

    if args.read_only:
        print("\n[READ-ONLY] 모터를 켜지 않고 아무것도 전송하지 않았습니다.")
        return 0

    scheduler = RealWujiScheduler(backend, command_hz=args.command_hz)
    print(f"[RATES]    command {args.command_hz:.1f} Hz, divider {scheduler.divider}")
    print(f"[FILTER]   LowPass cutoff {args.lowpass_hz:.2f} Hz "
          f"(tau {1000 / (2 * np.pi * args.lowpass_hz):.0f} ms)")
    print("[GAINS]    firmware servo gains are UNVERIFIED; none are set from here.")

    mask = np.ones((5, 4), dtype=bool)
    actual = np.empty_like(stream)
    controller = None
    try:
        if args.current_limit is not None:
            backend.write_current_limit(args.current_limit)
            applied = backend.read_current_limits()
            print(f"[CURRENT]  set to {args.current_limit:.4g} A "
                  f"(readback {applied.min():.4g} ~ {applied.max():.4g} A)")
        backend.prime_target_to_current()
        confirm("\n20관절 전체를 ENABLE 합니다. 이상하면 Ctrl+C.", args.yes)
        backend.enable(mask)
        print("[ENABLE]   done")

        with backend.realtime_controller(
            args.lowpass_hz,
            args.enable_upstream
        ) as controller:

            backend.controller = controller

            # ---------------------------------------------------------
            # 1. 현재 실제 자세 -> PREGRASP
            #
            # actual == target convergence를 요구하지 않는다.
            # move_all.py와 동일하게 target 자체를 천천히 이동시킨다.
            # ---------------------------------------------------------

            delta_to_pregrasp = float(
                np.max(np.abs(pregrasp - q_now))
            )

            pregrasp_seconds = max(
                args.start_seconds,
                delta_to_pregrasp / VALIDATED_JOINT_SPEED_RAD_S,
            )

            timed_glide_real(
                backend=backend,
                controller=controller,
                q_from=q_now,
                q_to=pregrasp,
                command_hz=args.command_hz,
                seconds=pregrasp_seconds,
                max_step_rad=args.max_step_rad,
                label="GLIDE -> PREGRASP",
            )

            print("[GLIDE]    PREGRASP command 도달")


            # ---------------------------------------------------------
            # 2. PREGRASP 잠깐 유지
            # ---------------------------------------------------------

            dt = 1.0 / args.command_hz

            settle_ticks = max(
                1,
                int(round(args.settle_seconds * args.command_hz))
            )

            print(
                f"[SETTLE]   PREGRASP {args.settle_seconds:.1f}s 유지"
            )

            for _ in range(settle_ticks):
                backend.write_joint_position_targets(
                    pregrasp,
                    max_step_rad=args.max_step_rad,
                )
                backend.publish_latest_target(controller)
                time.sleep(dt)

            if keyboard_streams is not None:
                open_stream, close_stream, open_id, close_id = keyboard_streams

                # The recorded OPEN transition starts from a closed grasp.  Establish
                # that state once, then only execute a transition when its opposite
                # state is currently active.
                initial_target = close_stream[-1]
                initial_delta = float(np.max(np.abs(initial_target - pregrasp)))
                timed_glide_real(
                    backend=backend,
                    controller=controller,
                    q_from=pregrasp,
                    q_to=initial_target,
                    command_hz=args.command_hz,
                    seconds=max(0.5, initial_delta / VALIDATED_JOINT_SPEED_RAD_S),
                    max_step_rad=args.max_step_rad,
                    label="BRIDGE PREGRASP -> CLOSE",
                )
                current_mode = "CLOSE"
                current_target = initial_target
                print(
                    f"[KEYBOARD] segment {open_id}=OPEN, {close_id}=CLOSE\n"
                    "           1=OPEN  2=CLOSE  q=종료  (Enter 불필요)"
                )

                while True:
                    key = _read_replay_key()
                    if key == "q":
                        print("\n[KEYBOARD] 종료")
                        break
                    requested = {"1": "OPEN", "2": "CLOSE"}.get(key)
                    if requested is None:
                        continue
                    if requested == current_mode:
                        print(f"\n[KEYBOARD] 이미 {current_mode}; 무시")
                        continue

                    selected = open_stream if requested == "OPEN" else close_stream
                    bridge_delta = float(np.max(np.abs(selected[0] - current_target)))
                    if bridge_delta > 1e-6:
                        timed_glide_real(
                            backend=backend,
                            controller=controller,
                            q_from=current_target,
                            q_to=selected[0],
                            command_hz=args.command_hz,
                            seconds=max(0.25, bridge_delta / VALIDATED_JOINT_SPEED_RAD_S),
                            max_step_rad=args.max_step_rad,
                            label=f"BRIDGE -> {requested}",
                        )

                    print(f"[{requested}]  {len(selected)} commands")
                    replay_start = time.monotonic()
                    for i, target in enumerate(selected):
                        backend.write_joint_position_targets(
                            target, max_step_rad=args.max_step_rad
                        )
                        backend.publish_latest_target(controller)
                        deadline = replay_start + (i + 1) * dt
                        wait = deadline - time.monotonic()
                        if wait > 0:
                            time.sleep(wait)
                        else:
                            scheduler.timing.late_ticks += 1
                    current_target = selected[-1]
                    current_mode = requested
                    print(f"[{requested}]  완료")

                if args.return_to_start:
                    return_delta = float(np.max(np.abs(current_target - q_now)))
                    timed_glide_real(
                        backend=backend,
                        controller=controller,
                        q_from=current_target,
                        q_to=q_now,
                        command_hz=args.command_hz,
                        seconds=max(args.start_seconds,
                                    return_delta / VALIDATED_JOINT_SPEED_RAD_S),
                        max_step_rad=args.max_step_rad,
                        label="RETURN -> START",
                    )
                    print("[RETURN]   시작 자세 command 도달")
                print(f"[TIMING]   {scheduler.timing.summary()}")
                return 0


            # ---------------------------------------------------------
            # 3. PREGRASP -> CSV 첫 qt
            #
            # 여기부터 qt는 "실제 관절이 반드시 도달해야 할 pose"가 아니라
            # PD target trajectory이므로 actual convergence를 검사하지 않는다.
            # ---------------------------------------------------------

            bridge_delta = float(
                np.max(np.abs(stream[0] - pregrasp))
            )

            bridge_seconds = max(
                0.5,
                bridge_delta / VALIDATED_JOINT_SPEED_RAD_S,
            )

            timed_glide_real(
                backend=backend,
                controller=controller,
                q_from=pregrasp,
                q_to=stream[0],
                command_hz=args.command_hz,
                seconds=bridge_seconds,
                max_step_rad=args.max_step_rad,
                label="BRIDGE PREGRASP -> CSV[0]",
            )

            print("[BRIDGE]   CSV 첫 target 연결 완료")

            dt = 1.0 / args.command_hz
            # Read back at the 30 Hz policy rate, not at every command tick.
            # Reading inside the realtime loop is not a pattern any vendor
            # example demonstrates, so it is kept to the lowest rate that still
            # records the response - and can be switched off entirely.
            read_every = scheduler.divider if args.read_during_replay else 0
            print(f"[REPLAY]   {len(stream)} commands = {len(stream) * dt:.1f}s"
                  + (f", reading back every {read_every} ticks" if read_every
                     else ", no encoder read inside the loop"))
            start = time.monotonic()
            last_read = q_now
            for i, target in enumerate(stream):
                backend.write_joint_position_targets(target, max_step_rad=args.max_step_rad)
                backend.publish_latest_target(controller)
                if read_every and i % read_every == 0:
                    last_read = backend.read_joint_positions()
                actual[i] = last_read
                deadline = start + (i + 1) * dt
                wait = deadline - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                else:
                    scheduler.timing.late_ticks += 1
            print(f"[REPLAY]   done in {time.monotonic() - start:.2f}s")

            hold_ticks = max(1, int(round(args.hold_seconds * args.command_hz)))
            for _ in range(hold_ticks):
                backend.publish_latest_target(controller)
                time.sleep(dt)

            if args.return_to_start:

                return_delta = float(
                    np.max(np.abs(stream[-1] - q_now))
                )

                return_seconds = max(
                    args.start_seconds,
                    return_delta / VALIDATED_JOINT_SPEED_RAD_S,
                )

                timed_glide_real(
                    backend=backend,
                    controller=controller,
                    q_from=stream[-1],
                    q_to=q_now,
                    command_hz=args.command_hz,
                    seconds=return_seconds,
                    max_step_rad=args.max_step_rad,
                    label="RETURN -> START",
                )

                print("[RETURN]   시작 자세 command 도달")
                
    finally:
        try:
            backend.disable()
            print("[DISABLE]  done")
        except Exception as exc:  # noqa: BLE001 - shutdown diagnostics only
            print(f"[DISABLE]  실패: {type(exc).__name__}: {exc}")

    error = np.abs(actual - stream)
    print(f"[TRACK]    |cmd-act| mean {error.mean():.5f} rad, max {error.max():.5f} rad")
    print(f"[TIMING]   {scheduler.timing.summary()}")
    if args.out is not None:
        write_response.command_hz = args.command_hz
        write_response(args.out, traj.joint_names, stream, actual)
    return 0


def main() -> int:
    args = build_argument_parser().parse_args()
    keyboard_streams = None
    if args.keyboard_open_close:
        if args.backend != "real":
            raise ValueError("--keyboard-open-close currently supports --backend real only.")
        if args.segment is not None:
            raise ValueError("Do not combine --segment with --keyboard-open-close.")
        if args.out is not None:
            raise ValueError("--out is not supported for an unbounded keyboard session.")
        open_traj, close_traj, open_id, close_id = _keyboard_segments(
            args.csv, args.limit_margin, args.open_segment, args.close_segment
        )
        open_speed = resolve_speed(open_traj, args.speed, args.max_joint_speed)
        close_speed = resolve_speed(close_traj, args.speed, args.max_joint_speed)
        open_stream, open_per_row = command_schedule(open_traj, open_speed, args.command_hz)
        close_stream, close_per_row = command_schedule(close_traj, close_speed, args.command_hz)
        print(f"[KEYBOARD] selected OPEN segment {open_id}, CLOSE segment {close_id}")
        describe_plan(open_traj, open_speed, open_per_row, len(open_stream), args)
        describe_plan(close_traj, close_speed, close_per_row, len(close_stream), args)
        traj, stream = close_traj, close_stream
        keyboard_streams = (open_stream, close_stream, open_id, close_id)
    else:
        if args.open_segment is not None or args.close_segment is not None:
            raise ValueError("--open-segment/--close-segment require --keyboard-open-close.")
        traj = Trajectory(args.csv, args.limit_margin, args.segment)
        speed = resolve_speed(traj, args.speed, args.max_joint_speed)
        stream, per_row = command_schedule(traj, speed, args.command_hz)
        describe_plan(traj, speed, per_row, len(stream), args)

    limits = soft_command_limits(args.limit_margin)
    assert (stream >= limits[:, 0] - 1e-6).all() and (stream <= limits[:, 1] + 1e-6).all()
    hard = COMMAND_TARGET_LIMITS
    assert (stream >= hard[:, 0]).all() and (stream <= hard[:, 1]).all()
    print("[CHECK]    보간된 전 구간이 명령 한계 안에 있습니다")

    if args.dry_run:
        print("\n[DRY RUN]  백엔드를 열지 않고 종료합니다.")
        return 0
    return (run_mujoco(traj, stream, args) if args.backend == "mujoco"
            else run_real(traj, stream, args, keyboard_streams=keyboard_streams))


if __name__ == "__main__":
    sys.exit(main())
