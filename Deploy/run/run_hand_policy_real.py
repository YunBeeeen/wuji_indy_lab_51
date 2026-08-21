"""Drive a 105D grasp policy on the physical Wuji Hand.

The hardware twin of ``run_mujoco_policy.py``.  Same phases, same order, same
names, so the two logs are readings of one procedure:

    [STATE]    read where the hand actually is -- there is no reset
    [GLIDE]    walk the target to the pregrasp pose
    [INSERT]   the operator places the chopsticks   <- MuJoCo's stick pinning
    [SEED]     build the first observation from the settled state
    [RUN]      policy at 30 Hz, published at the command rate
    [RETURN]   glide home, then disable in finally

The one phase with no simulator counterpart is [INSERT], and it is where the
pinning analogy becomes literal: in MuJoCo the sticks are held by re-applying
their pose every substep, and here they are held by a person.

Safety properties this file is responsible for
----------------------------------------------
* Every motor is enabled, not four.  A bug can now move the whole hand.
* The pregrasp pose stalls the fingers against each other when the chopsticks
  are absent.  Measured 2026-08-19: 1.5 A saturated for 96 s, finger1_joint2 at
  88.4 C.  [INSERT] therefore has a hard timeout and watches temperature.
* ``safe_stop`` freezes the command; it does not disable and does not release
  the grip preload.  Holding a frozen target requires that something keep
  publishing it -- the firmware's behaviour when commands stop is unverified --
  so the abort handler here runs that loop rather than trusting the hardware.
* ``finally`` always disables, including on Ctrl+C.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from ..policy.observation_adapter import PolicyObservationAdapter
from ..common.policy_contract import (
    ACTION_DIM,
    COMMAND_TARGET_LIMITS,
    OBSERVATION_DIM,
    POLICY_DT,
    POLICY_JOINT_NAMES,
    soft_command_limits,
)
from ..policy.policy_runner import PolicyRunner
from ..backends.real_wuji import RealWujiHand, full_enable_mask
from ..backends.real_wuji_backend import pending_validation_report
from ..backends.real_wuji_scheduler import POLICY_HZ, RealWujiScheduler
from ..common.isaac_reset import ISAAC_PREGRASP_JOINT_POSITIONS_RAD


#: Refuse to hold the unloaded pregrasp pose longer than this.  The thermal
#: incident took 96 s to reach 88.4 C, so the window is well inside it.
DEFAULT_INSERT_TIMEOUT_S = 45.0
#: Abort the insert wait immediately if any motor passes this.
DEFAULT_TEMPERATURE_LIMIT_C = 60.0


class ZeroPolicy:
    """Plumbing check.  A zero action decodes to ``target = q_current``."""

    def infer(self, observation):
        return np.zeros(ACTION_DIM, dtype=np.float32)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a 105D grasp policy on the real Wuji Hand.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", type=Path, default=None,
                        help="105D ONNX actor. Omit for the zero-action plumbing check.")
    parser.add_argument("--mode", choices=("open", "close"), default="close")
    parser.add_argument("--switch-at", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--seconds", type=float, default=10.0)

    safety = parser.add_argument_group("safety")
    safety.add_argument("--read-only", action="store_true",
                        help="Build observations and decode targets from the REAL "
                             "encoders, transmit nothing, enable no motor.")
    safety.add_argument("--yes", action="store_true", help="Skip the ENABLE prompt.")
    safety.add_argument("--max-step-rad", type=float, default=0.05,
                        help="Reject a target this far from the measured position.")
    safety.add_argument("--current-limit", type=float, default=None, metavar="AMPERES")
    safety.add_argument("--insert-timeout", type=float, default=DEFAULT_INSERT_TIMEOUT_S,
                        metavar="SECONDS",
                        help="Abort if the chopsticks are not placed within this. The "
                             "unloaded pregrasp pose stalls the fingers.")
    safety.add_argument("--temperature-limit", type=float, default=DEFAULT_TEMPERATURE_LIMIT_C,
                        metavar="CELSIUS", help="Abort the insert wait above this.")
    safety.add_argument("--temperature-interval", type=float, default=1.0, metavar="SECONDS",
                        help="Seconds between temperature checks during the insert wait. "
                             "Each check is a blocking SDO read, so this is a cost/latency "
                             "trade -- it is never done while the policy runs.")
    safety.add_argument("--allow-synthetic-sticks", action="store_true",
                        help="Permit DRIVING with a constant stick pose. Without a real "
                             "tracker the policy is acting on a fiction; this exists so "
                             "that has to be typed out.")

    phase = parser.add_argument_group("bring-up")
    phase.add_argument("--glide-seconds", type=float, default=5.0)
    phase.add_argument("--settle-seconds", type=float, default=2.0)
    phase.add_argument("--start-tolerance-rad", type=float, default=0.03)
    phase.add_argument("--start-stable-seconds", type=float, default=0.5)
    phase.add_argument("--start-timeout-seconds", type=float, default=20.0)
    phase.add_argument("--return-seconds", type=float, default=5.0)
    phase.add_argument("--no-return", dest="return_to_start", action="store_false",
                       default=True, help="Do not glide home before disabling.")
    phase.add_argument("--limit-margin", type=float, default=1.0, metavar="FRACTION")

    io = parser.add_argument_group("io")
    io.add_argument("--command-hz", type=float, default=90.0)
    io.add_argument("--lowpass-hz", type=float, default=0.5)
    io.add_argument("--read-source", choices=("controller", "hand"), default="controller")
    io.add_argument("--enable-upstream", dest="enable_upstream",
                    action="store_true", default=True)
    io.add_argument("--no-enable-upstream", dest="enable_upstream", action="store_false")
    io.add_argument("--stick-provider", choices=("synthetic",), default="synthetic",
                    help="Only 'synthetic' exists until the dual-camera tracker is bridged.")
    io.add_argument("--print-interval", type=int, default=30)
    io.add_argument("--out", type=Path, default=None)
    return parser


def confirm(prompt: str, skip: bool) -> None:
    if skip:
        print(f"{prompt}  [--yes]")
        return
    if input(f"{prompt}  계속하려면 'yes': ").strip().lower() != "yes":
        raise SystemExit("사용자가 중단했습니다.")


def print_state(label: str, q: np.ndarray) -> None:
    print(f"[{label}]")
    for finger in range(5):
        row = q[finger * 4:(finger + 1) * 4]
        print(f"  finger{finger + 1}  " + "  ".join(f"{v:+.5f}" for v in row))


def build_provider(name: str):
    if name == "synthetic":
        from ..common.perception import SyntheticStickPoseProvider

        return SyntheticStickPoseProvider()
    raise ValueError(f"Unknown stick provider {name!r}.")


def wait_for_chopsticks(backend: RealWujiHand, controller, args) -> None:
    """Hold the pregrasp pose while a person places the chopsticks.

    This is the phase MuJoCo does with ``pin_sticks``.  It is bounded in time
    and watched by temperature because the unloaded pose is a stall: the
    fingers press on each other, current saturates, and nothing in a position
    command makes that self-limiting.

    Publishing is done here rather than through ``RealWujiScheduler.run`` for
    two reasons.  The target is constant, so this is a held pose rather than a
    paced control loop; and ``read_joint_temperatures`` is a BLOCKING SDO read
    that will overrun a command tick, which would pollute the scheduler's
    timing statistics for the policy run that follows.  A late publish while
    holding a constant target costs nothing.
    """

    import select
    import sys

    period = 1.0 / float(args.command_hz)
    deadline = time.monotonic() + float(args.insert_timeout)
    print(f"\n[INSERT]   젓가락을 끼우고 Enter. "
          f"제한 {args.insert_timeout:.0f}s / {args.temperature_limit:.0f}C")

    began = time.monotonic()
    temperatures = backend.read_joint_temperatures()
    read_cost_ms = (time.monotonic() - began) * 1000.0
    print(f"           온도 읽기 {read_cost_ms:.1f} ms (블로킹 SDO). "
          f"명령 주기 {period * 1000:.1f} ms 보다 크면 이 구간에서만 늦게 나갑니다 "
          f"-- 목표가 고정이라 무해하고, 정책 구간에서는 읽지 않습니다.")
    print(f"           시작 온도 최고 {temperatures.max():5.1f}C "
          f"({POLICY_JOINT_NAMES[int(np.argmax(temperatures))]})")

    last_report = 0.0
    while True:
        backend.publish_latest_target(controller)
        now = time.monotonic()

        if select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.readline()
            print("[INSERT]   확인됨")
            return

        if now - last_report >= float(args.temperature_interval):
            last_report = now
            temperatures = backend.read_joint_temperatures()
            hottest = int(np.argmax(temperatures))
            try:
                peak_current = float(np.max(np.abs(backend.read_joint_efforts())))
                current_text = f"최대전류 {peak_current:.3f}A"
            except RuntimeError:
                current_text = "전류 n/a"
            print(f"  {deadline - now:5.1f}s 남음  최고온도 "
                  f"{temperatures[hottest]:5.1f}C ({POLICY_JOINT_NAMES[hottest]})  {current_text}")
            if float(temperatures[hottest]) >= args.temperature_limit:
                raise RuntimeError(
                    f"{POLICY_JOINT_NAMES[hottest]} reached "
                    f"{temperatures[hottest]:.1f} C during the insert wait."
                )

        if now >= deadline:
            raise RuntimeError(
                f"Chopsticks were not placed within {args.insert_timeout:.0f}s. "
                "Refusing to keep holding an unloaded pregrasp pose."
            )

        wait = period - (time.monotonic() - now)
        if wait > 0:
            time.sleep(wait)


def main() -> int:
    args = build_argument_parser().parse_args()
    pregrasp = np.asarray(ISAAC_PREGRASP_JOINT_POSITIONS_RAD, dtype=np.float32)

    # ---- everything checkable before touching the hand ----
    if args.policy is not None:
        from ..policy.onnx_policy import OnnxPolicy

        policy = OnnxPolicy(args.policy, OBSERVATION_DIM, ACTION_DIM)
        print(f"[POLICY]   {policy.path}")
        print(f"           {policy.input.shape} -> {policy.output.shape}")
    else:
        policy = ZeroPolicy()
        print("[POLICY]   zero-action plumbing check -- NOT a grasp test")

    driving = not args.read_only
    if driving and args.stick_provider == "synthetic" and not args.allow_synthetic_sticks:
        raise SystemExit(
            "거부: --stick-provider synthetic 으로는 주행하지 않습니다.\n"
            "  스틱 포즈가 상수라 정책은 '젓가락이 리셋 위치에 가만히 있다'는\n"
            "  거짓을 보고 20개 관절을 움직입니다. 배선 확인은 --read-only 로,\n"
            "  그래도 돌리려면 --allow-synthetic-sticks 를 명시하세요."
        )

    print()
    print(pending_validation_report())

    backend = RealWujiHand(read_source=args.read_source, max_step_rad=args.max_step_rad)
    print(f"\n[HAND]     {backend.describe()}")
    q_now = backend.read_joint_positions()
    print_state("현재 관절 위치", q_now)

    lower, upper = backend.read_hardware_limits()
    out_of_range = np.flatnonzero((pregrasp < lower) | (pregrasp > upper))
    if out_of_range.size:
        raise RuntimeError(
            "Pregrasp is outside the hand's reported limits at "
            f"{[POLICY_JOINT_NAMES[i] for i in out_of_range]}."
        )
    margin = np.minimum(pregrasp - lower, upper - pregrasp)
    print(f"[PREGRASP] 전 관절 한계 안. 최소 여유 {margin.min():.4f} rad "
          f"({POLICY_JOINT_NAMES[int(np.argmin(margin))]})")

    soft = soft_command_limits(args.limit_margin)
    clipped = np.flatnonzero((pregrasp < soft[:, 0]) | (pregrasp > soft[:, 1]))
    if clipped.size:
        print(f"[LIMITS]   margin {args.limit_margin:.3f} 가 pregrasp 를 자릅니다: "
              f"{[POLICY_JOINT_NAMES[i] for i in clipped]}")

    provider = build_provider(args.stick_provider)
    adapter = PolicyObservationAdapter(mode=args.mode, stick_provider=provider)
    runner = PolicyRunner(backend, policy, adapter)
    print(f"[CONTRACT] obs {OBSERVATION_DIM}D, action {ACTION_DIM}D, "
          f"stick source {type(provider).__name__}")

    # ---------------- READ-ONLY ----------------
    if args.read_only:
        print("\n[DRY RUN]  실측 엔코더로 관측을 만들고 목표를 디코드합니다.\n"
              "           전송 없음, 모터 enable 없음.")
        steps = max(1, int(round(args.seconds / POLICY_DT)))
        q_all = backend.read_joint_positions()
        runner.reset(q_all)
        for step in range(steps):
            q_all = backend.read_joint_positions()
            if step > 0:
                runner.observe_after_hold(q_all)
            observation = runner.observations.build()
            decoded = runner.command()
            if step % args.print_interval and step != steps - 1:
                continue
            print(f"\n--- policy step {step} ---")
            print(f"  obs[  0: 20] q_prev  norm  {np.round(observation[0:20:5], 4)} ...")
            print(f"  obs[ 40: 55] fingertips  {np.round(observation[40:46], 4)} ...")
            print(f"  obs[103:105] mode        {observation[103:105]}")
            print(f"  action                   [{decoded.action_manager_action.min():+.4f}, "
                  f"{decoded.action_manager_action.max():+.4f}]")
            print(f"  |target - q|  max        "
                  f"{np.abs(decoded.position_target - q_all).max():.5f} rad")
            clamp = np.flatnonzero(decoded.target_was_clamped)
            if clamp.size:
                print(f"  command clamp            {[POLICY_JOINT_NAMES[i] for i in clamp]}")
        print("\n[READ-ONLY] 끝. 모터를 켜지 않았고 아무것도 전송하지 않았습니다.")
        return 0

    # ---------------- DRIVE ----------------
    scheduler = RealWujiScheduler(backend, command_hz=args.command_hz)
    mask = full_enable_mask()
    print(f"\n[ENABLE]   전 관절 {int(mask.sum())}개")
    print(f"[RATES]    command {args.command_hz:.1f} Hz, policy {POLICY_HZ:.1f} Hz, "
          f"divider {scheduler.divider}")
    print(f"[FILTER]   LowPass {args.lowpass_hz:.2f} Hz")
    print(f"[GUARD]    max target step {backend.max_step_rad:.4f} rad/policy step")

    rows: list[list] = []
    controller = None
    failure: BaseException | None = None
    try:
        if args.current_limit is not None:
            backend.write_current_limit(args.current_limit)
            applied = backend.read_current_limits()
            print(f"[CURRENT]  set {args.current_limit:.4g} A "
                  f"(readback {applied.min():.4g} ~ {applied.max():.4g} A)")
        backend.prime_target_to_current()
        confirm("\n전 관절 모터를 ENABLE 합니다. 이상하면 Ctrl+C.", args.yes)
        backend.enable(mask)
        print("[ENABLE]   done")

        with backend.realtime_controller(args.lowpass_hz, args.enable_upstream) as controller:
            backend.controller = controller

            # ---------------- GLIDE ----------------
            print(f"\n[GLIDE]    현재 자세 -> pregrasp, {args.glide_seconds:.1f}s 선형")
            travel = float(np.abs(pregrasp - backend.read_joint_positions()).max())
            print(f"           최대 변위 {travel:.4f} rad "
                  f"({travel / max(args.glide_seconds, 1e-9):.3f} rad/s)")

            def report_move(elapsed, actual, error, max_error):
                print(f"  t={elapsed:5.2f}s max|err|={max_error:.5f}")

            elapsed, settled_all, error = scheduler.glide_to_pose(
                pregrasp, controller,
                joint_indices=np.arange(ACTION_DIM),
                seconds=args.glide_seconds,
                tolerance_rad=args.start_tolerance_rad,
                stable_seconds=args.start_stable_seconds,
                timeout_seconds=args.start_timeout_seconds,
                report=report_move,
                limit_fraction=args.limit_margin,
            )
            print(f"[GLIDE]    {elapsed:.2f}s, max |err| {np.abs(error).max():.5f} rad")

            # ---------------- INSERT ----------------
            wait_for_chopsticks(backend, controller, args)

            # ---------------- SETTLE ----------------
            print(f"\n[SETTLE]   {args.settle_seconds:.1f}s")
            scheduler.run(args.settle_seconds, controller)

            # ---------------- SEED ----------------
            q_all = backend.read_joint_positions()
            runner.reset(q_all)
            print(f"[SEED]     mode {args.mode}, "
                  f"preload |qt-q| max {np.abs(backend.latest_target - q_all).max():.5f} rad")

            # ---------------- RUN ----------------
            steps = max(1, int(round(args.seconds / POLICY_DT)))
            switch_step = (
                None if args.switch_at is None
                else max(0, int(round(args.switch_at / POLICY_DT)))
            )
            state = {"aborted": None}

            def on_policy_tick(policy_index: int, elapsed: float) -> None:
                if switch_step is not None and policy_index == switch_step:
                    runner.set_mode("open" if args.mode == "close" else "close")
                    print(f"  [MODE]  -> {runner.observations.mode} at t={elapsed:.2f}s")
                began = time.monotonic()
                # ONE read per policy tick, reused by both calls.
                q_all = backend.read_joint_positions()
                if policy_index > 0:
                    runner.observe_after_hold(q_all)
                decoded = runner.command()
                inference_ms = (time.monotonic() - began) * 1000.0
                efforts = backend.read_joint_efforts()
                rows.append(
                    [elapsed, policy_index, runner.observations.mode,
                     runner.observations.perception_state.value,
                     int(decoded.target_was_clamped.sum()), inference_ms,
                     float(np.max(np.abs(efforts)))]
                    + list(q_all) + list(decoded.position_target)
                    + list(decoded.action_manager_action)
                )
                if policy_index % args.print_interval == 0:
                    print(f"  t={elapsed:6.2f}s {runner.observations.mode:5s} "
                          f"|qt-q|={np.abs(decoded.position_target - q_all).max():.4f} "
                          f"A={np.max(np.abs(efforts)):.3f} "
                          f"clamp={int(decoded.target_was_clamped.sum()):2d} "
                          f"({inference_ms:.2f} ms)")

            print(f"\n[RUN]      {steps} policy steps ({args.seconds:.1f}s)")
            try:
                scheduler.run(args.seconds, controller, on_policy_tick)
            except RuntimeError as exc:
                # safe_stop has frozen the command.  Keep publishing it: whether
                # the firmware holds a target after commands stop is unverified,
                # so the hold is this loop's job, not the hardware's.
                state["aborted"] = exc
                print(f"\n[SAFE STOP] {exc}")
                print(f"[HOLD]     얼린 목표를 계속 전송합니다. "
                      f"Ctrl+C 로 종료하면 복귀 후 disable 합니다.")
                try:
                    scheduler.run(3600.0, controller)
                except KeyboardInterrupt:
                    print("\n[HOLD]     사용자 종료")

            # ---------------- RETURN ----------------
            if args.return_to_start:
                print(f"\n[RETURN]   시작 자세로 {args.return_seconds:.1f}s 복귀")
                q_home = q_now.copy()
                try:
                    scheduler.glide_to_pose(
                        q_home, controller,
                        joint_indices=np.arange(ACTION_DIM),
                        seconds=args.return_seconds,
                        tolerance_rad=args.start_tolerance_rad,
                        stable_seconds=args.start_stable_seconds,
                        timeout_seconds=args.start_timeout_seconds,
                        limit_fraction=args.limit_margin,
                    )
                except Exception as exc:
                    print(f"[RETURN]   실패: {type(exc).__name__}: {exc}")
                    print("           move_home.py 로 복구하세요.")
            if state["aborted"] is not None:
                failure = state["aborted"]
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        failure = exc
        print(f"\n[ABORT]    {type(exc).__name__}: {exc}")
        if rows:
            print(f"           {len(rows)} policy steps 완료 ({rows[-1][0]:.2f}s)")
    finally:
        backend.controller = None
        backend.disable()
        print("[DISABLE]  전 관절 모터 OFF")
        health = backend.health()
        print(f"[HEALTH]   {'ok' if health.ok else 'NOT OK'} -- {health.message}")

    if args.out is not None and rows:
        _write_csv(args.out, rows)
        print(f"[CSV]      {args.out}  ({len(rows)} rows)")
    print(f"[TIMING]   {scheduler.timing.summary()}")
    return 1 if failure is not None else 0


def _write_csv(path: Path, rows) -> None:
    header = (
        ["t_s", "policy_step", "mode", "perception", "targets_clamped",
         "policy_inference_ms", "max_current_a"]
        + [f"q_{n}" for n in POLICY_JOINT_NAMES]
        + [f"qt_{n}" for n in POLICY_JOINT_NAMES]
        + [f"a_{n}" for n in POLICY_JOINT_NAMES]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
