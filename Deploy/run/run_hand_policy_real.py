# [run/실물] 지원되는 grasp 정책을 실물 손에서 실행. ONNX 폭으로 105D/legacy 101D 자동 선택.
"""실물 Wuji Hand 파지 정책 단계별 실행.
비전 이상·온도·예외 감시. 종료 시 모든 모터 끔."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from ..common.policy_contract import (
    ACTION_DIM,
    COMMAND_TARGET_LIMITS,
    POLICY_DT,
    COMMAND_LIMIT_RATIO,
    POLICY_JOINT_NAMES,
    soft_command_limits,
)
from ..common.action_report import (
    action_summary,
    action_verdict,
    finger_row,
    print_action_detail,
)
from ..common.timing import StageTimer
from ..policy.grasp_policy_contract import (
    CURRENT_105D_CONTRACT,
    load_grasp_policy,
)
from ..policy.policy_runner import PolicyRunner
from ..backends.real_wuji import RealWujiHand, full_enable_mask
from ..backends.real_wuji_backend import pending_validation_report
from ..backends.real_wuji_scheduler import POLICY_HZ, RealWujiScheduler
from ..common.isaac_reset import (
    ISAAC_PREGRASP_JOINT_POSITIONS_RAD,
    read_pregrasp_from_env_yaml,
)


#: Refuse to hold the unloaded pregrasp pose longer than this.  The thermal
#: incident took 96 s to reach 88.4 C, so the window is well inside it.
DEFAULT_INSERT_TIMEOUT_S = 45.0
#: Current at which a joint is judged to be pushing rather than moving.  The
#: hand's own limit is what it saturates AT, so this is just a shade below it.
#: Default saturation threshold, for the factory 1.5 A limit.  Only a fallback:
#: --current-limit moves the ceiling, and a fixed 1.49 would then report 0%
#: saturation for a hand pinned at 1.200 A.  saturation_threshold() derives it
#: from the limit actually in force.
SATURATION_A = 1.49


def saturation_threshold(backend) -> float:
    """Current at which a joint counts as saturated, from the APPLIED limit.

    Read back from the device rather than taken from ``--current-limit``: the
    flag may be absent (factory value), and a write that did not take must not
    silently move the threshold.  Uses the smallest per-joint limit so no joint
    is measured against a ceiling higher than its own, and sits just under it
    because the servo dithers around the clamp (measured 1.15~1.20 A while
    pinned at a 1.2 A limit).
    """

    try:
        applied = np.asarray(backend.read_current_limits(), dtype=np.float64)
        if applied.size and np.all(np.isfinite(applied)) and applied.min() > 0.0:
            return float(applied.min()) * 0.99
    except Exception:
        pass
    return SATURATION_A

#: Abort the insert wait immediately if any motor passes this.
#: 90 C on the operator's judgement that this hand is comfortable to 85 C.
#: For scale, the 2026-08-19 stall incident peaked at 88.4 C after 96 s of
#: saturated current, so this is a real ceiling rather than a wide margin --
#: the reading that matters is how fast it is CLIMBING, not the absolute value.
DEFAULT_TEMPERATURE_LIMIT_C = 90.0


class ZeroPolicy:
    """Plumbing check.  A zero action decodes to ``target = q_current``."""

    def infer(self, observation):
        return np.zeros(ACTION_DIM, dtype=np.float32)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a supported 105D or legacy 101D grasp policy on the real Wuji Hand.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", type=Path, default=None,
                        help="ONNX actor; 105D/legacy 101D is auto-detected. "
                             "Omit for the zero-action plumbing check.")
    parser.add_argument("--mode", choices=("open", "close", "neutral", "schedule"),
                        default="schedule",
                        help="schedule: Isaac 의 에피소드 스크립트를 재생한다 -- "
                             "--open-lead-seconds 동안 OPEN, 이후 --segment-seconds "
                             "간격으로 교대. 정책이 학습한 입력이 이것이다. "
                             "open/close 로 고정하면 15초 내내 같은 값이라 쥐고만 있는다.")
    parser.add_argument("--first-segment", choices=("close", "open"), default="close",
                        help="교대 첫 구간. Isaac 은 에피소드마다 샘플하므로 둘 다 "
                             "학습 분포 안이다.")
    parser.add_argument("--open-lead-seconds", type=float, default=5.0, metavar="S",
                        help="교대 시작 전 OPEN 유지 시간. Isaac 은 5초 "
                             "(초기 유지 2 + 회전 2 + 정착 1).")
    parser.add_argument("--segment-seconds", type=float, default=2.0, metavar="S",
                        help="교대 구간 길이. Isaac 은 2초.")
    parser.add_argument("--switch-at", type=float, default=None, metavar="SECONDS",
                        help="이 시각에 반대 모드로 한 번 전환.")
    parser.add_argument("--no-keyboard", dest="keyboard", action="store_false",
                        default=True,
                        help="주행 중 키보드로 OPEN/CLOSE 를 못 바꾸게 한다. "
                             "기본은 켜짐: o+Enter 로 OPEN, c+Enter 로 CLOSE.")
    parser.add_argument("--seconds", type=float, default=10.0)

    safety = parser.add_argument_group("safety")
    safety.add_argument("--read-only", action="store_true",
                        help="Build observations and decode targets from the REAL "
                             "encoders, transmit nothing, enable no motor.")
    safety.add_argument("--yes", action="store_true", help="Skip the ENABLE prompt.")
    safety.add_argument("--max-step-rad", type=float, default=None, metavar="RAD",
                        help="Reject a target this far from the measured position. "
                             "Default is the auto-detected actor contract's own "
                             "action scale (105D: 0.1/0.1/0.2/0.15 rad; legacy "
                             "101D: uniform 0.1 rad), which makes the guard a no-op "
                             "for legal policy output while still catching a corrupt "
                             "graph or a decode bug.")
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
    safety.add_argument("--no-controller-read", dest="controller_read",
                        action="store_false", default=True,
                        help="read-only 에서 realtime controller 를 열지 않는다. "
                             "열면 q 읽기가 주행과 같은 업스트림 스트림 경로를 타서 "
                             "타이밍이 주행을 대표한다(블로킹 SDO 18ms -> 0.01ms). "
                             "모터 enable 도 목표 전송도 하지 않으므로 손은 움직이지 않는다.")
    safety.add_argument("--allow-synthetic-sticks", action="store_true",
                        help="Permit DRIVING with a constant stick pose. Without a real "
                             "tracker the policy is acting on a fiction; this exists so "
                             "that has to be typed out.")

    phase = parser.add_argument_group("bring-up")
    phase.add_argument("--glide-seconds", type=float, default=5.0)
    phase.add_argument("--settle-seconds", type=float, default=2.0)
    phase.add_argument("--start-tolerance-rad", type=float, default=0.03,
                       help="이 안에 들어오면 도착으로 본다.")
    phase.add_argument("--start-abort-rad", type=float, default=0.10, metavar="RAD",
                       help="도착 판정에 실패해도 오차가 이 아래면 경고만 하고 진행한다. "
                            "실물 위치 서보는 무부하에서도 정상상태 오차가 남는다 "
                            "(2026-08-21 실측: finger1_joint2 0.0446 rad). "
                            "이 값을 넘으면 뭔가 걸린 것이므로 중단한다.")
    phase.add_argument("--start-stable-seconds", type=float, default=0.5)
    phase.add_argument("--start-timeout-seconds", type=float, default=20.0)
    phase.add_argument("--return-seconds", type=float, default=5.0)
    phase.add_argument("--no-return", dest="return_to_start", action="store_false",
                       default=True, help="Do not glide home before disabling.")
    phase.add_argument("--limit-margin", type=float, default=1.0, metavar="FRACTION")
    phase.add_argument("--pregrasp-from", type=Path, default=None, metavar="RUN",
                       help="이 학습 런의 params/env.yaml 에서 pregrasp 를 읽어 "
                            "GLIDE 목표로 쓴다. 체크포인트마다 리셋 자세가 다르므로 "
                            "다른 런을 배포할 때 필요하다 (2026-08-21_20-37-48 은 "
                            "기본 상수와 78 mrad 차이). 런 폴더 / params/env.yaml / "
                            "policy.onnx 아무거나. 생략하면 "
                            "ISAAC_PREGRASP_JOINT_POSITIONS_RAD 상수를 쓴다.")

    io = parser.add_argument_group("io")
    io.add_argument("--command-hz", type=float, default=90.0)
    io.add_argument("--lowpass-hz", type=float, default=0.5)
    io.add_argument("--read-source", choices=("controller", "hand"), default="controller")
    io.add_argument("--enable-upstream", dest="enable_upstream",
                    action="store_true", default=True)
    io.add_argument("--no-enable-upstream", dest="enable_upstream", action="store_false")
    io.add_argument("--show-cameras", action="store_true",
                    help="트래커가 그린 카메라 화면을 띄운다. 검출이 왜 안 되는지 "
                         "눈으로 보려면 이것. waitKey 비용은 [TIMING vision] 의 "
                         "show 단계로 찍힌다.")
    io.add_argument("--resync", action="store_true",
                    help="두 스틱이 서로 다른 카메라에서 올 때 히스토리의 최근접 쌍으로 "
                         "재중재한다. 기본 꺼짐 -- 실측 대조에서 켜면 51스텝만에 STALE, "
                         "끄면 540스텝 완주였다 (2026-08-22).")
    io.add_argument("--camera-scale", type=float, default=0.5, metavar="FACTOR",
                    help="표시 배율. 1280x720 창 2개는 21.5ms(p95) 로 11.1ms 명령 틱을 "
                         "넘긴다. 0.5 면 5.9ms. 비용은 그리는 픽셀 수에 붙는다.")
    io.add_argument("--camera-every", type=int, default=15, metavar="TICKS",
                    help="몇 정책 틱마다 화면을 갱신할지. 기본 15 = 2Hz. "
                         "합성 이미지 벤치는 6틱에 4.5ms 였지만 실제로는 카메라 "
                         "스레드와 CPU 를 다투어 p95 11.4ms / 최대 73ms 였다 "
                         "(2026-08-22, late ticks 31/6971). 벤치를 믿지 말고 "
                         "[TIMING vision] 의 show 단계를 볼 것.")
    io.add_argument("--tracker-log", action="store_true",
                    help="트래커 자체 진단 출력을 다시 켠다. 기본은 끔 -- 프레임마다 "
                         "여러 줄을 찍어서 30Hz 에서는 터미널이 흘러가 키 입력이 안 된다.")
    io.add_argument("--stick-provider", choices=("synthetic", "vision"), default="vision",
                    help="vision = the MAIN/SIDE ArUco tracker. synthetic feeds a "
                         "CONSTANT stick pose and is a plumbing check only.")
    io.add_argument("--q6-deg", type=float, default=None, metavar="DEGREES",
                    help="Indy7 joint-6 angle the arm is parked at. Required for "
                         "--stick-provider vision: the palm frame rotates with it, "
                         "and so does every stick pose the policy sees. Measure it; "
                         "do not copy the tracker's typed-in value on faith.")
    io.add_argument("--acknowledge-candidate-geometry", action="store_true",
                    help="[RIG] 미검증 기하 경고를 끈다. 2026-08-23 이전에는 이게 "
                         "없으면 실행을 거부했는데, 매 런 우회해야 하는 거부는 "
                         "신호가 아니라 잡음이라 경고로 바꿨다. 기하 계산은 "
                         "어느 쪽이든 동일하다.")
    io.add_argument("--hold-after-ms", type=float, default=None, metavar="MS",
                    help="Both cameras may lose a stick this long before the pose "
                         "goes stale and the policy is stopped.")
    io.add_argument("--print-interval", type=int, default=30)
    io.add_argument("--no-vectors", dest="print_vectors", action="store_false",
                    default=True,
                    help="관절별 action / 목표 / 현재값 3줄 덤프를 끈다. 기본 켜짐 -- "
                         "정책이 실제로 액션을 내고 있는지는 요약 한 줄로는 알 수 없다.")
    io.add_argument("--out", type=Path, default=None)
    return parser


_NULL_TIMER = StageTimer(name="none")


def _start_provider(provider) -> None:
    """Open the perception source, if it has one to open.

    The camera-backed provider defers opening to ``start()`` so that its
    refusals (missing q6, unverified geometry) can fire with nothing connected.
    That deferral is only safe if something actually calls this -- forgetting to
    was worth a crash on the first hardware run.
    """

    start = getattr(provider, "start", None)
    if start is None:
        return
    began = time.monotonic()
    start()
    print(f"[CAMERAS]  {type(provider).__name__} 기동 "
          f"{(time.monotonic() - began) * 1000:.0f} ms")


def _stop_provider(provider) -> None:
    stop = getattr(provider, "stop", None)
    if stop is not None:
        stop()


def _provider_timing(provider) -> StageTimer:
    """Stick sources vary; only some of them are instrumented."""

    return getattr(provider, "timing", _NULL_TIMER)


def _stick_line(provider) -> str:
    """One line of what the tracker currently sees, in palm mm.

    Shown while the operator is placing the chopsticks: this is the moment to
    find out that a marker is hidden, not after the run has been committed to.
    Never raises -- a tracker that cannot place a stick yet is the normal state
    here, and saying so is the useful output.
    """

    if provider is None or not hasattr(provider, "sample"):
        return ""
    try:
        pair = provider.sample()
    except RuntimeError as exc:
        return f"스틱: 아직 못 잡음 ({str(exc).splitlines()[0]})"
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"스틱: 읽기 실패 {type(exc).__name__}"
    return (f"스틱1 {np.round(pair.stick1[:3] * 1e3, 1)}mm  "
            f"스틱2 {np.round(pair.stick2[:3] * 1e3, 1)}mm  "
            f"[{pair.state.value} {_stick_source(provider)}]")


def _stick_source(provider) -> str:
    """MAIN or SIDE, per stick.  Recorded because the two resolve poses
    differently, so a SIDE stretch must be identifiable afterwards."""

    sources = getattr(provider, "sources", None)
    if sources is None:
        return "n/a"
    return f"{sources.stick1_source}/{sources.stick2_source}"


def scheduled_mode(elapsed_s: float, args) -> str:
    """Isaac's OPEN/CLOSE script at a given time into the run.

    OPEN for ``--open-lead-seconds``, then segments of ``--segment-seconds``
    alternating from ``--first-segment``.  Isaac samples that first mode per
    episode, so either choice is inside the training distribution; the
    alternation after it is not optional.

    A constant mode is what the policy was NOT trained on: it saw each value for
    2 s at a time, never for the length of a run, and never without the
    transitions.  Held at CLOSE it simply squeezes -- measured 2026-08-22,
    |qt-q| sat at 0.026 rad for fifteen seconds and the hand barely moved.
    """

    if elapsed_s < args.open_lead_seconds:
        return "open"
    index = int((elapsed_s - args.open_lead_seconds) // max(args.segment_seconds, 1e-6))
    other = "open" if args.first_segment == "close" else "close"
    return args.first_segment if index % 2 == 0 else other


def read_key():
    """Non-blocking single-letter command from stdin, or None.

    Line buffered on purpose -- 'o' then Enter -- so the terminal is never put
    into raw mode.  A run that dies with the tty in raw mode leaves the shell
    unusable, which is a poor trade for saving one keystroke.

    Costs a ``select`` with a zero timeout, so it is free even at the 90 Hz
    command rate.
    """

    import select
    import sys

    if not select.select([sys.stdin], [], [], 0.0)[0]:
        return None
    line = sys.stdin.readline().strip().lower()
    return line[0] if line else None


def read_mode_key():
    """Non-blocking OPEN/CLOSE keypress, or None.

    Line buffered on purpose -- 'o' then Enter -- so the terminal is never put
    into raw mode.  A run that dies with the tty in raw mode leaves the shell
    unusable, which is a poor trade for saving one keystroke.

    Costs a ``select`` with a zero timeout, so it is free at 30 Hz.  The policy
    was trained against a schedule that alternates every 2 s (Isaac's
    ``HandMoveScheduleCfg``: OPEN until 5 s, then 5 x 2 s); a constant mode is
    an input it never saw for that long, and it simply holds.
    """

    key = read_key()
    # 'n' is NEUTRAL [0, 0], the third command state.  It is not a "no command":
    # the 5 s grasp/setting curriculum (hand_real2) trains on it exclusively, so
    # a keyboard that only offers OPEN/CLOSE cannot drive those checkpoints at
    # all.  See mode_one_hot().
    return {"o": "open", "c": "close", "n": "neutral"}.get(key)


def settle_verdict(backend, goal, args, label: str) -> bool:
    """Judge a glide that timed out by SIZE, not by pass/fail.

    Arriving at a commanded position is not guaranteed on a real hand: a
    position servo leaves a steady-state offset under stiction and load, and it
    does not shrink with time.  Measured 2026-08-21 -- glide to pregrasp,
    finger1_joint2 held 0.045 rad for 25 s; return home, finger2_joint2 held
    0.035 rad against a 0.030 tolerance.  Every other joint was inside 0.007.

    ``move_all.py`` -- the routine that has actually driven this hand -- does not
    check arrival at all; it holds and moves on.  Failing a run because a joint
    is a quarter of a degree short is the wrong call.  A jammed joint still has
    to stop it, so the line is drawn at ``--start-abort-rad`` instead.

    Returns True when the run may continue; raises when something is stuck.
    """

    error = backend.read_joint_positions() - np.asarray(goal, dtype=np.float32)
    worst = int(np.argmax(np.abs(error)))
    if abs(float(error[worst])) > args.start_abort_rad:
        raise RuntimeError(
            f"{POLICY_JOINT_NAMES[worst]} 가 목표에서 {error[worst]:+.4f} rad "
            f"떨어져 멈췄습니다 (한계 {args.start_abort_rad:.3f}). "
            "뭔가 걸린 것으로 보고 중단합니다."
        )
    offenders = [f"{POLICY_JOINT_NAMES[i]} {error[i]:+.4f}"
                 for i in np.flatnonzero(np.abs(error) > args.start_tolerance_rad)]
    print(f"[{label}]    도착 판정 실패했지만 진행합니다 "
          f"(최대 {error[worst]:+.4f} rad < 중단선 {args.start_abort_rad:.3f})")
    print(f"           허용치 초과 관절: {offenders}")
    return True


def confirm(prompt: str, skip: bool) -> None:
    if skip:
        print(f"{prompt}  [--yes]")
        return
    if input(f"{prompt}  계속하려면 'yes': ").strip().lower() != "yes":
        raise SystemExit("사용자가 중단했습니다.")


def confirm_with_preview(prompt: str, args, provider) -> None:
    """Ask for the ENABLE confirmation while the cameras run.

    Deliberately UNBOUNDED, unlike ``wait_for_chopsticks``.  That one holds an
    unloaded pregrasp pose, which is a stall -- current saturates and the hand
    heats, so it has a timeout and a temperature guard.  Here the motors are
    still off and nothing is being held, so a person can take as long as they
    want to look.  This is the last moment at which "the tracker cannot see
    stick2" costs nothing; after ENABLE it costs a bring-up cycle.

    The display is pumped from ``sample()``, so the loop has to keep sampling
    for the windows to refresh.  ``camera_every`` is forced to 1 for the
    duration: it exists to protect the 30 Hz policy budget, and there is no
    such budget here.
    """

    import select
    import sys

    if provider is None or not hasattr(provider, "sample"):
        confirm(prompt, args.yes)
        return

    every = getattr(provider, "camera_every", None)
    if every is not None:
        provider.camera_every = 1

    print(prompt)
    if args.yes:
        print(f"           [--yes] {_stick_line(provider)}")
        if every is not None:
            provider.camera_every = every
        return

    print("           카메라를 보면서 기다립니다. 준비되면 'yes' + Enter, "
          "중단은 Ctrl+C.")
    last_report = 0.0
    last_source = None
    try:
        while True:
            try:
                provider.sample()
            except RuntimeError:
                pass  # _stick_line reports it; a stick not yet placed is normal
            now = time.monotonic()
            source = _stick_source(provider)
            if source != last_source or now - last_report >= 1.0:
                last_report, last_source = now, source
                line = _stick_line(provider)
                if line:
                    print(f"           {line}")
            if select.select([sys.stdin], [], [], 0.0)[0]:
                answer = sys.stdin.readline().strip().lower()
                if answer == "yes":
                    return
                print(f"           'yes' 를 입력해야 진행합니다 (받은 값: {answer!r}).")
            time.sleep(0.05)
    except KeyboardInterrupt:
        raise SystemExit("사용자가 중단했습니다.")
    finally:
        if every is not None:
            provider.camera_every = every


def print_state(label: str, q: np.ndarray) -> None:
    print(f"[{label}]")
    for finger in range(5):
        row = q[finger * 4:(finger + 1) * 4]
        print(f"  finger{finger + 1}  " + "  ".join(f"{v:+.5f}" for v in row))


def build_provider(args):
    """Pick the stick-pose source.  The only place a perception backend is chosen.

    Constructing it does NOT open the cameras -- ``provider.start()`` does that,
    once the run is actually committed.  What happens here is the refusals.
    """

    if args.stick_provider == "synthetic":
        from ..common.perception import SyntheticStickPoseProvider

        return SyntheticStickPoseProvider()

    from ..vision.provider import DualCameraStickPoseProvider

    if args.q6_deg is None:
        raise SystemExit(
            "--q6-deg 가 필요합니다.\n"
            "  palm 프레임이 Indy7 6축에 매달려 있어서, 이 각도가 틀리면\n"
            "  정책이 보는 젓가락 포즈 전체가 그만큼 회전합니다. 트래커에\n"
            "  적힌 25.000097 을 그대로 믿지 말고 실제 값을 확인해 주세요."
        )
    extra = {}
    if args.hold_after_ms is not None:
        extra["hold_after_ms"] = args.hold_after_ms
    return DualCameraStickPoseProvider(
        args.q6_deg,
        acknowledge_candidate_geometry=args.acknowledge_candidate_geometry,
        prefer_synchronised_pair=args.resync,
        quiet_tracker=not args.tracker_log,
        show_cameras=args.show_cameras,
        camera_scale=args.camera_scale,
        camera_every=args.camera_every,
        **extra,
    )


def wait_for_chopsticks(backend: RealWujiHand, controller, args, provider=None) -> None:
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
            sticks = _stick_line(provider)
            print(f"  {deadline - now:5.1f}s 남음  최고온도 "
                  f"{temperatures[hottest]:5.1f}C ({POLICY_JOINT_NAMES[hottest]})  {current_text}")
            if sticks:
                print(f"           {sticks}")
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
    if args.show_cameras:
        # Qt prints a two-line font warning per window creation.  Harmless, and
        # it buries the run's own output.
        import os

        os.environ.setdefault("QT_LOGGING_RULES", "*=false")

    # ---- everything checkable before touching the hand ----
    # Select by the graph itself, never by a user-provided --legacy flag.  A
    # mismatched switch would be a silent policy-input corruption; a fixed ONNX
    # width is unambiguous.
    if args.policy is not None:
        policy, policy_contract = load_grasp_policy(args.policy)
        print(f"[POLICY]   {policy.path}")
        print(f"           {policy.input.shape} -> {policy.output.shape}")
        print(f"[ADAPTER]  {policy_contract.key} (ONNX input width auto-detected)")
    else:
        policy = ZeroPolicy()
        policy_contract = CURRENT_105D_CONTRACT
        print("[POLICY]   zero-action plumbing check -- NOT a grasp test")
    if args.mode != "schedule" and args.mode not in policy_contract.supported_modes:
        raise SystemExit(
            f"{policy_contract.key} does not support --mode {args.mode!r}; "
            f"expected one of {policy_contract.supported_modes} or 'schedule'."
        )

    if args.pregrasp_from is not None:
        pregrasp = read_pregrasp_from_env_yaml(args.pregrasp_from)
        delta = pregrasp - np.asarray(ISAAC_PREGRASP_JOINT_POSITIONS_RAD, dtype=np.float32)
        worst = int(np.argmax(np.abs(delta)))
        print(f"[PREGRASP] {args.pregrasp_from} 의 env.yaml 에서 읽음. "
              f"기본 상수 대비 최대 {1000.0 * delta[worst]:+.1f} mrad "
              f"({POLICY_JOINT_NAMES[worst]})")
    elif policy_contract.default_pregrasp is not None:
        # The legacy adapter owns the reset that was saved with that actor.
        # This is what lets the old CLI keep working without a new mandatory
        # --pregrasp-from argument.
        pregrasp = np.asarray(policy_contract.default_pregrasp, dtype=np.float32).copy()
        print(f"[PREGRASP] {policy_contract.key} 에 저장된 전용 reset pose 자동 선택")
    else:
        pregrasp = np.asarray(ISAAC_PREGRASP_JOINT_POSITIONS_RAD, dtype=np.float32)

    driving = not args.read_only
    if driving and args.stick_provider == "synthetic" and not args.allow_synthetic_sticks:
        raise SystemExit(
            "거부: --stick-provider synthetic 으로는 주행하지 않습니다.\n"
            "  스틱 포즈가 상수라 정책은 '젓가락이 리셋 위치에 가만히 있다'는\n"
            "  거짓을 보고 20개 관절을 움직입니다. 배선 확인은 --read-only 로,\n"
            "  그래도 돌리려면 --allow-synthetic-sticks 를 명시하세요."
        )

    # Built BEFORE the hand is opened.  Its guards -- a missing q6, unverified
    # rig geometry -- are refusals, and a refusal that arrives after a hand has
    # been connected has already done half of what it was meant to prevent.
    # Opening the cameras is deferred to provider.start(), further down.
    provider = build_provider(args)
    print(f"[STICKS]   {type(provider).__name__}"
          + (f", q6 = {args.q6_deg:.6f} deg" if args.q6_deg is not None else ""))

    print()
    print(pending_validation_report())

    # Per-joint by default: the twenty-joint contract's legal step differs by
    # joint, so one scalar cannot both permit joint3's 0.2 rad and stay tight on
    # joint1's 0.1 rad.
    step_limit = (
        policy_contract.action_scale_rad
        if args.max_step_rad is None
        else args.max_step_rad
    )
    backend = RealWujiHand(read_source=args.read_source, max_step_rad=step_limit)
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

    # COMMAND_LIMIT_RATIO narrows every command below the factory stop, and the
    # Isaac pregrasp is trained AT the stop on two joints -- so the start pose
    # itself no longer fits.  Clamp it here rather than letting the scheduler's
    # first target be rejected: the backend refuses out-of-range targets by
    # design, so an unclamped pregrasp would kill the run at the glide.
    ratio_lower = COMMAND_TARGET_LIMITS[:, 0]
    ratio_upper = COMMAND_TARGET_LIMITS[:, 1]
    outside = np.flatnonzero((pregrasp < ratio_lower) | (pregrasp > ratio_upper))
    if outside.size:
        print(f"[LIMITS]   명령 한계 비율 {COMMAND_LIMIT_RATIO:.2f} 가 pregrasp 를 "
              f"{outside.size}개 관절에서 자릅니다:")
        for i in outside:
            target = float(np.clip(pregrasp[i], ratio_lower[i], ratio_upper[i]))
            print(f"           {POLICY_JOINT_NAMES[i]:16s} "
                  f"{pregrasp[i]:+.4f} -> {target:+.4f} rad "
                  f"({1000.0 * (target - pregrasp[i]):+.1f} mrad)")
        print("           학습된 pregrasp 와 그만큼 다른 자세에서 시작합니다.")
        pregrasp = np.clip(pregrasp, ratio_lower, ratio_upper).astype(np.float32)

    soft = soft_command_limits(args.limit_margin)
    clipped = np.flatnonzero((pregrasp < soft[:, 0]) | (pregrasp > soft[:, 1]))
    if clipped.size:
        print(f"[LIMITS]   margin {args.limit_margin:.3f} 가 pregrasp 를 자릅니다: "
              f"{[POLICY_JOINT_NAMES[i] for i in clipped]}")

    if args.mode == "neutral":
        # A neutral-trained checkpoint (episode_length_s == open_close_start_time_s
        # with neutral_before_open_close) saw [0, 0] and nothing else.  Holding
        # it IS the run; there is no alternation to schedule.
        print("[MODE]     neutral [0,0] 고정 -- OPEN/CLOSE 를 학습하지 않은 "
              "체크포인트용입니다 (5초 grasp/setting 커리큘럼).")
    initial_mode = (scheduled_mode(0.0, args) if args.mode == "schedule" else args.mode)
    adapter = policy_contract.make_observation_adapter(
        mode=initial_mode, stick_provider=provider
    )
    runner = PolicyRunner(
        backend, policy, adapter, action_decoder=policy_contract.action_decoder
    )
    print(f"[CONTRACT] {policy_contract.key}: obs {policy_contract.observation_dim}D, "
          f"action {policy_contract.action_dim}D, "
          f"stick source {type(provider).__name__}")

    # ---------------- READ-ONLY ----------------
    if args.read_only:
        print("\n[DRY RUN]  실측 엔코더로 관측을 만들고 목표를 디코드합니다.\n"
              "           전송 없음, 모터 enable 없음.")
        steps = max(1, int(round(args.seconds / POLICY_DT)))
        ro_timer = StageTimer(budget_ms=1000.0 * POLICY_DT, name="tick")
        _start_provider(provider)
        try:
            if args.controller_read:
                # Open the realtime controller WITHOUT enabling a motor and
                # without ever publishing a target.  Two separate reasons:
                #
                # * read_joint_positions() is one function that branches on
                #   whether a controller exists.  With one open it takes the
                #   same path the drive loop takes, so the timing here means
                #   something about the drive loop.  Without one it falls to a
                #   blocking SDO read -- measured at 18.3 ms of a 21.7 ms tick,
                #   a cost the drive path never pays.
                # * "reading q inside realtime_controller with
                #   enable_upstream=True" is on the NOT-MEASURED list.  With
                #   every motor off is the cheapest place to exercise it; the
                #   alternative is meeting it for the first time mid-grasp.
                #
                # Nothing moves: enable() is never called and
                # publish_latest_target() is never called, so no target ever
                # reaches the hand.
                print(f"[CONTROLLER] realtime controller 를 엽니다 "
                      f"(LowPass {args.lowpass_hz:.2f} Hz, "
                      f"upstream={'on' if args.enable_upstream else 'off'}).\n"
                      f"             모터 enable 없음, 목표 전송 없음 -- 손은 움직이지 않습니다.")
                with backend.realtime_controller(args.lowpass_hz, args.enable_upstream) as controller:
                    backend.controller = controller
                    try:
                        _read_only_loop(runner, backend, provider, args, steps, ro_timer)
                    finally:
                        backend.controller = None
            else:
                print("[CONTROLLER] 열지 않음 -- q 읽기가 블로킹 SDO 경로로 갑니다 "
                      "(주행 타이밍을 대표하지 않음).")
                _read_only_loop(runner, backend, provider, args, steps, ro_timer)
        finally:
            _stop_provider(provider)
            print()
            print(ro_timer.report())
            # Printed even if the loop raised: a run that failed on perception
            # is the run whose perception timing you want.
            print()
            print(runner.timing.report())
            print(runner.observations.timing.report())
            vision_timer = _provider_timing(provider)
            if vision_timer.labels:
                print(vision_timer.report())
        print("\n[READ-ONLY] 끝. 모터를 켜지 않았고 아무것도 전송하지 않았습니다.")
        return 0

    # ---------------- DRIVE ----------------
    scheduler = RealWujiScheduler(backend, command_hz=args.command_hz)
    mask = full_enable_mask()
    print(f"\n[ENABLE]   전 관절 {int(mask.sum())}개")
    print(f"[RATES]    command {args.command_hz:.1f} Hz, policy {POLICY_HZ:.1f} Hz, "
          f"divider {scheduler.divider}")
    print(f"[FILTER]   LowPass {args.lowpass_hz:.2f} Hz")
    guard = np.atleast_1d(backend.max_step_rad)
    print(f"[GUARD]    max target step {guard.min():.3f}~{guard.max():.3f} rad/policy step"
          + ("  (계약의 관절별 액션 스케일)" if args.max_step_rad is None else "  (수동 설정)"))

    rows: list[list] = []
    # The inner finally needs to know whether we are unwinding from Ctrl+C
    # before the outer except has had a chance to record it.
    _ctrl_c_pending = [False]
    controller = None
    failure: BaseException | None = None
    # Created before the try, because the CSV write outside it references this.
    # A run that aborts partway is exactly the one whose timing matters, so the
    # timer must survive every path out of the block.
    tick_timer = StageTimer(budget_ms=1000.0 * POLICY_DT, name="tick")
    try:
        if args.current_limit is not None:
            backend.write_current_limit(args.current_limit)
            applied = backend.read_current_limits()
            print(f"[CURRENT]  set {args.current_limit:.4g} A "
                  f"(readback {applied.min():.4g} ~ {applied.max():.4g} A)")
        start_temperatures = None
        # Read temperature BEFORE enabling.  The glide that follows presses the
        # thumb for the whole of --glide-seconds, and a hand that started warm
        # from a previous run has no chance to shed that.
        try:
            temperatures = backend.read_joint_temperatures()
            hottest = int(np.argmax(temperatures))
            start_temperatures = temperatures
            print(f"[TEMP]     시작 온도 최고 {temperatures[hottest]:.1f}C "
                  f"({POLICY_JOINT_NAMES[hottest]}), 한계 {args.temperature_limit:.0f}C")
            if float(temperatures[hottest]) >= args.temperature_limit:
                raise RuntimeError(
                    f"{POLICY_JOINT_NAMES[hottest]} 가 이미 "
                    f"{temperatures[hottest]:.1f}C 입니다. 식힌 뒤에 하세요."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            print(f"[TEMP]     읽기 실패: {type(exc).__name__}: {exc}")

        # Cameras open BEFORE the motors, not at SEED.  Two reasons: the 1.2 s
        # warm-up stops sitting between the settle and the first policy step,
        # and -- the point of it -- the INSERT wait can then show whether the
        # tracker actually picked up the chopsticks that were just placed.
        # Deciding that after committing to the run is too late.
        _start_provider(provider)

        backend.prime_target_to_current()
        confirm_with_preview(
            "\n전 관절 모터를 ENABLE 합니다. 이상하면 Ctrl+C.", args, provider)
        backend.enable(mask)
        print("[ENABLE]   done")

        with backend.realtime_controller(args.lowpass_hz, args.enable_upstream) as controller:
            backend.controller = controller
            try:
                # ---------------- GLIDE ----------------
                print(f"\n[GLIDE]    현재 자세 -> pregrasp, {args.glide_seconds:.1f}s 선형")
                travel = float(np.abs(pregrasp - backend.read_joint_positions()).max())
                print(f"           최대 변위 {travel:.4f} rad "
                      f"({travel / max(args.glide_seconds, 1e-9):.3f} rad/s)")

                # Saturation is not itself a fault: the pregrasp pose with no
                # chopsticks in it IS a stall -- the fingers press on each other
                # and no current can reach the target.  What varies is how long
                # it is endured.  1.5 A for 2 s is nothing; the same 1.5 A for
                # 96 s reached 88.4 C (2026-08-19), and 25 s of it reached 71 C
                # here.  Time-at-saturation is the damage variable, so measure
                # that rather than the current, which is a foregone conclusion.
                saturated = {"seconds": 0.0, "last": None}
                saturation_a = saturation_threshold(backend)

                def elapsed_total():
                    return saturated["last"] or 0.0

                def report_move(elapsed, actual, error, max_error):
                    # Effort comes off the controller's upstream stream, so it
                    # is cheap enough for this callback -- unlike temperature,
                    # which is a 27.6 ms blocking SDO read.  A joint that cannot
                    # reach its target saturates here long before the heat shows
                    # up, which is what happened to finger1_joint2.
                    try:
                        efforts = np.abs(backend.read_joint_efforts())
                        peak, at = float(efforts.max()), int(np.argmax(efforts))
                        if saturated["last"] is not None and peak >= saturation_a:
                            saturated["seconds"] += elapsed - saturated["last"]
                        saturated["last"] = elapsed
                        hot = int(np.sum(efforts >= saturation_a))
                        note = (f"  A={peak:.3f} ({POLICY_JOINT_NAMES[at]})"
                                + (f" +{hot - 1}개" if hot > 1 else ""))
                        if peak >= saturation_a:
                            note += f"  포화 누적 {saturated['seconds']:.1f}s"
                    except Exception:
                        note = ""
                    print(f"  t={elapsed:5.2f}s max|err|={max_error:.5f}{note}")

                try:
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
                except RuntimeError as exc:
                    from ..backends.real_wuji_scheduler import PlateauedError

                    if isinstance(exc, PlateauedError):
                        print(f"[GLIDE]    {exc}")
                    settle_verdict(backend, pregrasp, args, "GLIDE")

                if saturated["seconds"] > 0.0:
                    print(f"[GLIDE]    전류 포화 누적 {saturated['seconds']:.1f}s "
                          f"({100.0 * saturated['seconds'] / max(elapsed_total(), 1e-9):.0f}% 구간)")
                    print(f"           포화 자체는 정상입니다 -- 젓가락 없는 pregrasp 는 "
                          f"손가락끼리 미는 스톨이라 목표에 닿을 수 없습니다.")
                    print(f"           온도를 줄이는 유일한 방법은 이 시간을 줄이는 것입니다 "
                          f"(--glide-seconds ↓, --lowpass-hz ↑).")

                # ---------------- INSERT ----------------
                wait_for_chopsticks(backend, controller, args, provider)

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
                # HOLD rides through silently by design: both cameras may miss a
                # stick for up to hold_after_ms and the policy keeps running on
                # the last good pose.  That is correct, and it is also exactly
                # what you want counted afterwards -- "did tracking ever drop
                # out" cannot be answered from "the run did not abort", which
                # only rules out STALE/LOST.
                perception = {}
                sources = {}
                # "did the policy command anything" must survive an abort, so
                # accumulate it here rather than re-deriving it from the CSV.
                action_peak = {"sum": 0.0, "max": 0.0, "n": 0}
                # Per-joint current, accumulated for the same reason: the run
                # that aborts is the one whose load you want to read.
                current_sum = np.zeros(ACTION_DIM, dtype=np.float64)
                current_hits = np.zeros(ACTION_DIM, dtype=np.int64)

                def on_policy_tick(policy_index: int, elapsed: float) -> None:
                    if args.mode == "schedule":
                        want = scheduled_mode(elapsed, args)
                        if want != runner.observations.mode:
                            runner.set_mode(want)
                            print(f"  [MODE]  -> {want:5s} at t={elapsed:5.2f}s")
                    elif args.keyboard:
                        key = read_mode_key()
                        if key is not None and key != runner.observations.mode:
                            runner.set_mode(key)
                            print(f"  [MODE]  -> {key} at t={elapsed:.2f}s  (키보드)")
                    if switch_step is not None and policy_index == switch_step:
                        runner.set_mode("open" if args.mode == "close" else "close")
                        print(f"  [MODE]  -> {runner.observations.mode} at t={elapsed:.2f}s")
                    with tick_timer.stage("total"):
                        # ONE read per policy tick, reused by both calls -- and on
                        # the first tick that read already happened, at SEED.
                        with tick_timer.stage("joint_read"):
                            if policy_index == 0:
                                q_all = runner.q_current
                            else:
                                q_all = backend.read_joint_positions()
                        if policy_index > 0:
                            with tick_timer.stage("observe"):
                                runner.observe_after_hold(q_all)
                        with tick_timer.stage("command"):
                            decoded = runner.command()
                        with tick_timer.stage("effort_read"):
                            efforts = backend.read_joint_efforts()
                    seen = runner.observations.perception_state.value
                    perception[seen] = perception.get(seen, 0) + 1
                    src = _stick_source(provider)
                    sources[src] = sources.get(src, 0) + 1
                    total_ms = tick_timer.last["total"]

                    rows.append(
                        [elapsed, policy_index, runner.observations.mode,
                         runner.observations.perception_state.value,
                         int(decoded.target_was_clamped.sum()), total_ms,
                         float(np.max(np.abs(efforts))), _stick_source(provider)]
                        + list(q_all) + list(decoded.position_target)
                        + list(decoded.action_manager_action)
                        + list(runner.last_observation)
                        # Per joint, not just the peak.  "max_current_a" is the
                        # maximum over all twenty, so it cannot answer which
                        # joint is drawing it -- and that is exactly the
                        # question COMMAND_LIMIT_RATIO raises (2026-08-22: the
                        # railed joint moved off its stop while the peak stayed
                        # at 1.500 A, which the peak alone could not explain).
                        + list(efforts)
                        + list(tick_timer.csv_row())
                        + list(runner.timing.csv_row())
                        + list(runner.observations.timing.csv_row())
                        + list(_provider_timing(provider).csv_row())
                    )
                    magnitude = np.abs(efforts)
                    # In-place slice assignment, NOT `+=`.  Inside this nested
                    # function a bare `current_sum += ...` rebinds the name, so
                    # Python makes it local and the first tick dies with
                    # UnboundLocalError -- which is what killed the 2026-08-23
                    # run before step 0.  `action_peak[...] += ...` above is
                    # safe only because mutating a dict is not a rebinding.
                    current_sum[:] += magnitude
                    current_hits[:] += (magnitude >= saturation_a)
                    action_peak["sum"] += float(np.max(np.abs(decoded.onnx_action)))
                    action_peak["n"] += 1
                    action_peak["max"] = max(
                        action_peak["max"], float(np.max(np.abs(decoded.onnx_action))))
                    if policy_index % args.print_interval == 0:
                        over = "  <== 예산 초과" if total_ms > 1000.0 * POLICY_DT else ""
                        print(f"  t={elapsed:6.2f}s {runner.observations.mode:5s} "
                              f"|qt-q|={np.abs(decoded.position_target - q_all).max():.4f} "
                              f"{action_summary(decoded)} "
                              f"A={np.max(np.abs(efforts)):.3f} "
                              f"clamp={int(decoded.target_was_clamped.sum()):2d} "
                              f"{total_ms:5.1f}ms{over}")
                        if args.print_vectors:
                            print_action_detail(decoded, q_all)
                            print("           A  " + finger_row(np.abs(efforts), "+.2f"))

                print(f"\n[RUN]      {steps} policy steps ({args.seconds:.1f}s)")
                if args.mode == "schedule":
                    print(f"           스케줄: {args.open_lead_seconds:.0f}초 OPEN -> "
                          f"{args.segment_seconds:.0f}초씩 교대 "
                          f"(첫 구간 {args.first_segment})")
                elif args.keyboard:
                    print(f"           키보드 제어: o+Enter = OPEN, "
                          f"c+Enter = CLOSE, n+Enter = NEUTRAL")
                    print(f"           시작 모드 {runner.observations.mode} "
                          f"-- 입력이 없으면 계속 유지합니다.")
                try:
                    scheduler.run(args.seconds, controller, on_policy_tick)
                except RuntimeError as exc:
                    # safe_stop has frozen the command.  Keep publishing it: whether
                    # the firmware holds a target after commands stop is unverified,
                    # so the hold is this loop's job, not the hardware's.
                    state["aborted"] = exc
                    # The aborting tick raised inside command(), before the
                    # counter below it ran, so without this the [PERCEPTION]
                    # histogram shows only "valid/hold" for a run that STALE
                    # actually stopped -- and then claims STALE would have
                    # aborted it.  Count the state that did.
                    seen = runner.observations.perception_state.value
                    perception[seen] = perception.get(seen, 0) + 1
                    print(f"\n[SAFE STOP] {exc}")
                    print(f"[HOLD]     얼린 목표를 계속 전송합니다 -- 파지는 유지됩니다.")
                    print(f"           q+Enter 로 정상 종료(복귀 후 disable), "
                          f"Ctrl+C 는 즉시 disable.")
                    hold = {"quit": False, "last_state": None,
                            "began": time.monotonic()}

                    def on_hold_tick(tick: int, elapsed: float) -> None:
                        # The policy loop has exited, so the mode keys read in
                        # on_policy_tick are no longer polled -- which is why
                        # keys pressed here did nothing on 2026-08-22.  Read
                        # them from the command loop instead.
                        if read_key() == "q":
                            hold["quit"] = True
                            return
                        if tick % 30:
                            return
                        # scheduler.run() is called in one-second slices, so its
                        # own `elapsed` restarts every slice.  Time the hold
                        # from here or every line reads t=0.0s.
                        held = time.monotonic() - hold["began"]
                        try:
                            provider.sample()
                            now = provider.sources
                            line = f"{now.stick1_source}/{now.stick2_source}"
                        except Exception:
                            line = "포즈 없음"
                        if line != hold["last_state"]:
                            hold["last_state"] = line
                            ok = "NONE" not in line and line != "포즈 없음"
                            print(f"  [HOLD] t={held:5.1f}s  인지 {line}"
                                  + ("  (추적 복구)" if ok else "  (추적 끊김)"))
                            if not ok:
                                # The tracker's own verdict, not our inference.
                                why = getattr(provider, "sources", None)
                                if why is not None and hasattr(why, "why"):
                                    for i in (0, 1):
                                        print(f"           스틱{i+1}  {why.why(i)}")

                    try:
                        while not hold["quit"]:
                            # Short slices so a q lands within ~0.2 s instead of
                            # waiting out a long run() call.
                            scheduler.run(0.2, controller, on_command_tick=on_hold_tick)
                        held = time.monotonic() - hold["began"]
                        print(f"[HOLD]     사용자 종료 (q) -- {held:.1f}s 유지했습니다")
                    except KeyboardInterrupt:
                        print("\n[HOLD]     사용자 종료 (Ctrl+C)")
                        _ctrl_c_pending[0] = True

                # ---------------- 지연 보고 ----------------
                # Printed even after an abort: a run that was stopped by a stale
                # pose is exactly the run whose timing you want to read.
                if action_peak["n"]:
                    print()
                    for line in action_verdict(
                            action_peak["sum"] / action_peak["n"],
                            action_peak["max"], action_peak["n"]):
                        print(line)

                # Temperature is not read DURING the policy loop -- the SDO
                # blocks ~27 ms against an 11.1 ms command period, and a late
                # command matters here in a way it does not while holding a
                # fixed pose. Reading it the moment the loop ends costs
                # nothing and is the only measurement of how much the run
                # actually heated the hand, which is what bounds --seconds.
                try:
                    after = backend.read_joint_temperatures()
                    hot = int(np.argmax(after))
                    print(f"\n[TEMP]     종료 온도 최고 {after[hot]:.1f}C "
                          f"({POLICY_JOINT_NAMES[hot]}), 한계 "
                          f"{args.temperature_limit:.0f}C")
                    if start_temperatures is not None:
                        rise = float(after[hot]) - float(start_temperatures[hot])
                        seconds = max(action_peak["n"] * POLICY_DT, 1e-9)
                        print(f"           같은 관절 상승 {rise:+.1f}C / "
                              f"{seconds:.1f}s = {rise / seconds:+.2f} C/s")
                        if rise > 0.0:
                            budget = (args.temperature_limit - float(after[hot]))
                            print(f"           이 속도면 90C 까지 "
                                  f"{budget / (rise / seconds):.0f}s 남았습니다.")
                except Exception as exc:
                    print(f"\n[TEMP]     종료 온도 읽기 실패: {type(exc).__name__}")

                if action_peak["n"]:
                    n = action_peak["n"]
                    mean_a = current_sum / n
                    duty = current_hits / n
                    ranked = np.argsort(-duty * 1e3 - mean_a)[:5]
                    print(f"\n[CURRENT]    관절별 전류 (포화 {saturation_a:.2f}A 기준, "
                          f"{n} 스텝)")
                    for j in ranked:
                        note = ""
                        if duty[j] >= 0.99:
                            note = "  <== 상시 포화"
                        elif duty[j] > 0.0:
                            note = f"  포화 {duty[j] * n * POLICY_DT:.1f}s"
                        print(f"             {POLICY_JOINT_NAMES[j]:16s} "
                              f"평균 {mean_a[j]:.3f}A  포화율 {100.0 * duty[j]:5.1f}%{note}")
                    hot = int(np.argmax(mean_a))
                    if duty.sum() == 0.0:
                        print("             포화한 관절 없음.")
                    print(f"             최대 부하 {POLICY_JOINT_NAMES[hot]} "
                          f"평균 {mean_a[hot]:.3f}A -- 화면의 A= 는 20개 최댓값이라 "
                          f"이 관절을 특정할 수 없습니다.")

                if perception:
                    total_steps = sum(perception.values())
                    parts = "  ".join(f"{k} {v}" for k, v in sorted(perception.items()))
                    print(f"\n[PERCEPTION] {parts}   ({total_steps} 스텝)")
                    held = perception.get("hold", 0)
                    stale = perception.get("stale", 0) + perception.get("lost", 0)
                    if held and not stale:
                        print(f"             HOLD {held}회 -- 두 카메라가 잠깐 놓쳤고 "
                              f"마지막 포즈로 버텼습니다. STALE 까지 갔으면 중단됐을 것입니다.")
                    elif stale:
                        print(f"             HOLD {held}회를 버티다 STALE 로 중단됐습니다 "
                              f"({total_steps}/{steps} 스텝, "
                              f"{total_steps * POLICY_DT:.1f}s 지점).")
                    else:
                        print("             끊긴 적 없음 -- 매 스텝 새 포즈가 들어왔습니다.")
                    print(f"             소스: " + "  ".join(f"{k} {v}" for k, v in sorted(sources.items())))

                print()
                print(tick_timer.report())
                print(runner.timing.report())
                print(runner.observations.timing.report())
                vision_timer = _provider_timing(provider)
                if vision_timer.labels:
                    print(vision_timer.report())

                # ---------------- RETURN ----------------
                if state["aborted"] is not None:
                    failure = state["aborted"]
            except KeyboardInterrupt:
                # Recorded before the finally runs, so the return knows this is
                # an emergency stop rather than a tidy end of run.
                _ctrl_c_pending[0] = True
                raise
            finally:
                # Home return lives HERE, inside the controller context.  Put in
                # the outer finally it ran after ``with`` had closed the
                # controller and died with "Controller is closed." -- on the very
                # path it existed for.  Ctrl+C still skips it: that is an
                # emergency stop, not a tidy shutdown.
                if args.return_to_start and not _ctrl_c_pending[0]:
                    try:
                        print(f"\n[RETURN]   시작 자세로 {args.return_seconds:.1f}s 복귀")
                        scheduler.glide_to_pose(
                            q_now, controller,
                            joint_indices=np.arange(ACTION_DIM),
                            seconds=args.return_seconds,
                            tolerance_rad=args.start_tolerance_rad,
                            stable_seconds=args.start_stable_seconds,
                            timeout_seconds=args.start_timeout_seconds,
                            limit_fraction=args.limit_margin,
                        )
                        print("[RETURN]   완료")
                    except RuntimeError:
                        # Same steady-state offset as the outbound glide.  The
                        # hand IS home; only the assertion failed, and pointing
                        # the operator at move_home.py for a quarter of a degree
                        # is misleading.
                        settle_verdict(backend, q_now, args, "RETURN")
                    except KeyboardInterrupt:
                        print("\n[RETURN]   사용자가 중단 -- 바로 끕니다")
                    except Exception as exc:
                        # Never hide the original failure, never skip disable().
                        print(f"[RETURN]   실패: {type(exc).__name__}: {exc}")
                        print("           move_home.py 로 복구하세요.")

    except KeyboardInterrupt as exc:
        failure = exc
        print(f"\n[ABORT]    사용자 중단 (Ctrl+C) -- 복귀 없이 바로 끕니다")
    except (RuntimeError, ValueError) as exc:
        failure = exc
        print(f"\n[ABORT]    {type(exc).__name__}: {exc}")
        if rows:
            print(f"           {len(rows)} policy steps 완료 ({rows[-1][0]:.2f}s)")
    finally:
        _stop_provider(provider)
        backend.controller = None
        backend.disable()
        print("[DISABLE]  전 관절 모터 OFF")
        health = backend.health()
        print(f"[HEALTH]   {'ok' if health.ok else 'NOT OK'} -- {health.message}")

    if args.out is not None and rows:
        _write_csv(
            args.out,
            rows,
            observation_columns=policy_contract.observation_csv_columns(),
            timers=(
                ("tick", tick_timer), ("policy", runner.timing),
                ("obs", runner.observations.timing),
                ("vision", _provider_timing(provider)),
            ),
        )
        print(f"[CSV]      {args.out}  ({len(rows)} rows)")
    print(f"[TIMING]   {scheduler.timing.summary()}")
    return 1 if failure is not None else 0


def _read_only_loop(runner, backend, provider, args, steps: int, timer) -> None:
    """Build observations and decode targets from the REAL encoders, at 30 Hz.

    Nothing is transmitted and no motor is enabled: ``runner.command()`` only
    stores a target in the backend, and ``publish_latest_target`` is never
    called.

    Paced against an absolute deadline, the same discipline the drive schedulers
    use.  Free-running, this loop answers "how long does one step take" -- which
    it did, at about 3 ms -- but NOT "does the policy hold 30 Hz", because 30 Hz
    was never attempted.  Reporting a budget the loop does not keep is how a
    latency figure gets read as a rate problem.
    """

    q_all = backend.read_joint_positions()
    runner.reset(q_all)
    began = time.monotonic()
    for step in range(steps):
        with timer.stage("total"):
            # Staged to the same granularity as the drive path, so "total" has
            # no remainder.  Without this the tick reported 21.6 ms against
            # 2.9 ms of measured parts, and 18.7 ms was going somewhere nobody
            # was looking -- the exact failure this instrumentation exists for.
            with timer.stage("joint_read"):
                if step == 0:
                    # Reuse the sample reset() was seeded with.  Reading again
                    # here would decode against the seed while the backend's
                    # slew guard checks the newer read, and the drift between
                    # them adds to a legal step -- 2.35 mrad was enough to have
                    # a 0.150 rad joint4 move refused.
                    q_all = runner.q_current
                else:
                    q_all = backend.read_joint_positions()
            if step > 0:
                with timer.stage("observe"):
                    runner.observe_after_hold(q_all)
            with timer.stage("build"):
                observation = runner.observations.build()
            with timer.stage("command"):
                decoded = runner.command()

        deadline = began + (step + 1) * POLICY_DT
        wait = deadline - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        else:
            timer.gauge("late_by_ms", -wait * 1000.0)

        if step % args.print_interval and step != steps - 1:
            continue
        blocks = runner.observations.debug_slices()
        slices = runner.observations.observation_slices

        def obs_label(name: str) -> str:
            term = slices[name]
            return f"obs[{term.start:3d}:{term.stop:3d}]"

        print(f"\n--- policy step {step} ---")
        print(f"  {obs_label('joint_previous')} q_prev  norm  "
              f"{np.round(blocks['joint_previous'][::5], 4)} ...")
        print(f"  {obs_label('fingertips')} fingertips  "
              f"{np.round(blocks['fingertips'][:6], 4)} ...")
        print(f"  {obs_label('mode')} mode        {blocks['mode']}")
        print(f"  action                   [{decoded.action_manager_action.min():+.4f}, "
              f"{decoded.action_manager_action.max():+.4f}]  {action_summary(decoded)}")
        if args.print_vectors:
            print_action_detail(decoded, q_all)
        print(f"  |target - q|  max        "
              f"{np.abs(decoded.position_target - q_all).max():.5f} rad")
        clamp = np.flatnonzero(decoded.target_was_clamped)
        if clamp.size:
            print(f"  command clamp            {[POLICY_JOINT_NAMES[i] for i in clamp]}")
        print(f"  perception               {runner.observations.perception_state.value}"
              f"  source {_stick_source(provider)}")
        print(f"  {obs_label('stick1_previous')} stick1 prev "
              f"{np.round(blocks['stick1_previous'], 4)}")
        print(f"  {obs_label('stick2_previous')} stick2 prev "
              f"{np.round(blocks['stick2_previous'], 4)}")


def _write_csv(path: Path, rows, observation_columns, timers=()) -> None:
    """Per-step CSV.  The timing columns are per-STEP, not summaries.

    The printed report gives p95 and the worst stage; this gives the trace.
    A bottleneck that only appears on 3 ticks out of 300 never shows up in an
    aggregate, and those are the ticks worth finding.
    """

    header = (
        ["t_s", "policy_step", "mode", "perception", "targets_clamped",
         "tick_total_ms", "max_current_a", "stick_source"]
        + [f"q_{n}" for n in POLICY_JOINT_NAMES]
        + [f"qt_{n}" for n in POLICY_JOINT_NAMES]
        + [f"a_{n}" for n in POLICY_JOINT_NAMES]
        + list(observation_columns)
        + [f"i_{n}" for n in POLICY_JOINT_NAMES]
    )
    for prefix, timer in timers:
        header += [f"{prefix}_{c}" for c in timer.csv_columns()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
