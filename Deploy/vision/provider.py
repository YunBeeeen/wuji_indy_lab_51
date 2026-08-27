# [vision] 듀얼 카메라 트래커를 StickPoseProvider로 감싸는 브리지. 원본 유지, 조립만 수행.
"""듀얼 카메라 추적기와 정책의 스틱 포즈 인터페이스 연결.
q6 기반 hand frame 변환과 시간 기준 HOLD/STALE/LOST 상태 관리."""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..common.perception import PoseState, StickPosePair7D
from ..common.stick_pose import canonicalize_square_stick_quaternion
from ..common.timing import StageTimer
from . import deploy_rig
from .sim_aruco import STICK_REFERENCE_QUATERNIONS_PALM_WXYZ


_VISION_DIR = Path(__file__).resolve().parent
TRACKER_PATH = _VISION_DIR / "run_dual_camera_hand_stick_final.py"

#: How long BOTH cameras may fail to place a stick before the policy is stopped.
#:
#: Reached only when MAIN sees neither of that stick's markers AND the SIDE
#: fallback is also invalid -- one dropped frame, or one camera losing sight,
#: never gets here.  Below HOLD the last good pose is reused and the policy
#: keeps running; past STALE, ``PolicyRunner`` calls ``safe_stop``, which
#: freezes the command without releasing the grip.
#:
#: 100 ms is three frames at 30 Hz.  These are starting values, NOT measured
#: ones: nobody has yet observed how long this grasp tolerates a frozen stick
#: pose.  Tune them against a real run rather than trusting them.
HOLD_AFTER_MS = 100.0
STALE_AFTER_MS = 250.0

#: Per-camera freshness, from the tracker's own constant.
CAMERA_STALE_MS = deploy_rig.CAMERA_STALE_MS


def load_tracker(path: Path | str = TRACKER_PATH):
    """추적기 ``main()`` 실행 없이 스크립트를 경로로 로드."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tracker script not found: {path}")
    name = f"_wuji_tracker_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class StickSourceReport:
    """각 스틱에 사용된 카메라와 선택 이유를 진단용으로 기록."""

    stick1_source: str = "NONE"
    stick2_source: str = "NONE"
    stick1_reason: str = ""
    stick2_reason: str = ""
    #: The tracker's OWN per-camera verdict, not the cross-camera arbitration.
    #: Read it instead of inferring: on 2026-08-22 "no pose while the markers
    #: are clearly visible" was blamed on the workspace prior, which does not
    #: reject anything (WORKSPACE_PRIOR_FALLBACK_TO_REPROJ is True).  The real
    #: verdict is here -- ``choose_final_source`` returns NO_VALID_ESTIMATOR
    #: only when DUAL is unaccepted AND neither single survives.
    detail: dict = field(default_factory=dict)

    def why(self, index: int) -> str:
        """One line of why this stick has no pose, in the tracker's own words."""

        parts = []
        for camera in ("MAIN", "SIDE"):
            info = self.detail.get((camera, index))
            if info is None:
                parts.append(f"{camera}:프레임없음")
                continue
            seen = info.get("detected_markers") or []
            parts.append(f"{camera}:{info.get('reason', '?')}"
                         f" 마커{seen or '없음'}"
                         f"{'' if info.get('raw_valid') else ' raw무효'}")
        return "  ".join(parts)

    def degraded(self) -> bool:
        return "SIDE" in (self.stick1_source, self.stick2_source)


class DualCameraStickPoseProvider:
    """``StickPoseProvider`` backed by the MAIN/SIDE ArUco tracker.

    ``q6_source`` is either a float (the arm is parked, and this is the angle it
    is parked at -- verify it, do not assume) or a zero-argument callable
    returning the live joint-6 angle in degrees.
    """

    representation = "StickPose7D"

    def __init__(
        self,
        q6_source,
        *,
        quiet_tracker: bool = True,
        show_cameras: bool = False,
        camera_scale: float = 0.5,
        camera_every: int = 6,
        tracker_path: Path | str = TRACKER_PATH,
        hold_after_ms: float = HOLD_AFTER_MS,
        stale_after_ms: float = STALE_AFTER_MS,
        acknowledge_candidate_geometry: bool = False,
        prefer_synchronised_pair: bool = False,
    ):
        if q6_source is None:
            raise ValueError(
                "q6_source is required: the palm frame rotates with Indy7 joint 6, "
                "and every stick pose the policy sees is expressed in that frame."
            )
        # The rig still rests on a coarsely-verified yaw sum; make the caller say
        # so out loud rather than discovering it later from a 6 mm offset.
        deploy_rig.assert_deployable(
            acknowledge_candidates=acknowledge_candidate_geometry
        )
        if float(hold_after_ms) >= float(stale_after_ms):
            raise ValueError("hold_after_ms must be below stale_after_ms.")

        self._q6_source = q6_source
        #: OFF, and measured to belong that way.
        #:
        #: The tracker's per-stick rule -- MAIN wins if it sees any of that
        #: stick's markers, SIDE only when MAIN misses both -- runs unmodified
        #: either way.  What this switch changes is what that rule is SHOWN:
        #: ``latest()`` from each camera, which is the tracker's own path, or a
        #: closest-in-time pair pulled from history.
        #:
        #: Feeding it history was my addition, meant to cut the ~49 ms
        #: cross-camera skew worth about 1 mm of stick1-to-stick2 error.  It ran
        #: on 70 % of samples and handed the rule frames whose DETECTIONS were
        #: older, because ``find_nearest_timestamp_pair`` minimises |dt| and
        #: nothing else.  Measured back-to-back on hardware 2026-08-22:
        #:
        #:     on   51 steps before STALE,  8-11 HOLD
        #:     off  540 steps,  4 HOLD, the whole OPEN/CLOSE schedule
        #:
        #: Leaving a function untouched is not the same as leaving its inputs
        #: untouched, and 1 mm of skew is not worth a tracking dropout.
        self.prefer_synchronised_pair = bool(prefer_synchronised_pair)
        self.hold_after_ms = float(hold_after_ms)
        self.stale_after_ms = float(stale_after_ms)
        # Loaded in start(), not here.  Constructing this object is the guard
        # stage -- it must work in an environment with no camera stack at all,
        # so the freshness ladder can be exercised without RealSense installed.
        self._tracker_path = Path(tracker_path)
        #: Silence the tracker scripts' own diagnostics.  They print several
        #: lines per frame from their worker threads, which at 30 Hz buries the
        #: run's output and -- measured on hardware -- makes the terminal
        #: unusable for the o/c keypresses this file reads from stdin.  Their
        #: content is not lost: the per-stick reason is already surfaced as
        #: PoseState and StickSourceReport.
        self.quiet_tracker = bool(quiet_tracker)
        #: Show each camera's annotated frame.  The tracker already draws the
        #: stick axes into ``ProcessedFrame.vis``; this only displays it.
        #:
        #: Scale and rate both matter, and not for the reason that looks
        #: obvious.  Measured 2026-08-22: two 1280x720 windows cost 21.5 ms p95,
        #: which blows the 11.1 ms command tick -- and ``pollKey`` barely helps
        #: (18.7 ms), because the cost is Qt's event loop pushing pixels, not
        #: the wait.  Halving each side drops it to 5.9 ms; refreshing every
        #: sixth policy step drops it to 4.5 ms.  A view of the markers does not
        #: need full resolution or 30 Hz.
        self.show_cameras = bool(show_cameras)
        self.camera_scale = float(camera_scale)
        self.camera_every = max(1, int(camera_every))
        self._shown = 0
        self.tracker = None
        self.sources = StickSourceReport()
        #: Where a slow sample() went, plus the gauges no duration can express:
        #: how OLD the frame being used was, and how far apart MAIN and SIDE
        #: arrived.  A 3 ms sample() that consumed a 90 ms-old frame is fast and
        #: wrong at the same time, and only the gauges say so.
        self.timing = StageTimer(name="vision")

        self._modules = None
        self._streams = None
        self._workers = None
        self._last_pose = [None, None]
        self._last_valid_ms = [None, None]
        self._last_quaternion = [None, None]

    # -- lifecycle --------------------------------------------------------
    def q6_deg(self) -> float:
        value = self._q6_source() if callable(self._q6_source) else self._q6_source
        return float(value)

    #: How long start() waits for each camera to deliver its first PROCESSED
    #: frame.  Opening a RealSense pipeline returns before any frame arrives,
    #: and the tracker threads then need a frame plus one detection pass.
    #: Sampling before that gives ``latest() is None``, which the arbitration
    #: reads as "saw nothing" -- indistinguishable from markers out of view.
    FIRST_FRAME_TIMEOUT_S = 5.0

    def start(self, wait_first_frame_s: float | None = None) -> None:
        """Open both cameras, start their threads, and wait for a first frame."""

        if self._workers is not None:
            return
        if self.tracker is None:
            self.tracker = load_tracker(self._tracker_path)
        t = self.tracker
        stick1 = t.load_module("wuji_stick1", t.STICK1_MODULE_PATH)
        stick2 = t.load_module("wuji_stick2", t.resolve_stick2_module_path())
        self._modules = (stick1, stick2)
        if self.quiet_tracker:
            # Python resolves a bare ``print`` in module globals before
            # builtins, so binding a no-op there silences exactly these three
            # modules and nothing else -- including the worker threads, which a
            # sys.stdout redirect in sample() would never have reached.
            for module in (stick1, stick2, t):
                module.print = lambda *a, **k: None
        target_ids = set(stick1.TARGET_IDS) | set(stick2.TARGET_IDS)

        width, height, fps = int(stick1.WIDTH), int(stick1.HEIGHT), int(stick1.FPS)
        self._streams, self._workers = [], []
        for label, serial, extrinsic, prior in (
            ("MAIN", t.MAIN_CAMERA_SERIAL, deploy_rig.T_BASE_CAMERA_MAIN, True),
            ("SIDE", t.SIDE_CAMERA_SERIAL, deploy_rig.T_BASE_CAMERA_SIDE, False),
        ):
            stream = t.CameraStream(label, serial, width, height, fps)
            stream.start()
            worker = t.CameraProcessingWorker(
                label, stream,
                t.StickTrackerState(stick1, f"{label}-stick1", prior),
                t.StickTrackerState(stick2, f"{label}-stick2", prior),
                stick1, stick2, target_ids,
                extrinsic, deploy_rig.invert(extrinsic),
            )
            worker.start()
            self._streams.append(stream)
            self._workers.append(worker)

        self.wait_for_first_frames(
            self.FIRST_FRAME_TIMEOUT_S if wait_first_frame_s is None
            else wait_first_frame_s
        )

    def wait_for_first_frames(self, timeout_s: float) -> dict[str, float]:
        """Block until BOTH cameras have produced a processed frame.

        Returns each camera's warm-up time in ms.  Raises rather than letting
        the first ``sample()`` report "no markers": a camera that never delivers
        a frame is a different fault from one whose markers are out of view, and
        conflating them sends the operator to move the sticks when the problem
        is the camera.
        """

        deadline = time.monotonic() + float(timeout_s)
        began = time.monotonic()
        warmup = {}
        while len(warmup) < len(self._workers):
            for worker in self._workers:
                if worker.name in warmup:
                    continue
                if worker.latest() is not None:
                    warmup[worker.name] = (time.monotonic() - began) * 1000.0
            if len(warmup) == len(self._workers):
                break
            if time.monotonic() > deadline:
                silent = [w.name for w in self._workers if w.name not in warmup]
                raise RuntimeError(
                    f"{', '.join(silent)} 카메라가 {timeout_s:.1f}초 안에 프레임을 "
                    "내놓지 않았습니다. 마커가 아니라 카메라 문제입니다 -- "
                    "연결/시리얼/다른 프로세스 점유를 확인하세요 "
                    f"(MAIN={self.tracker.MAIN_CAMERA_SERIAL}, "
                    f"SIDE={self.tracker.SIDE_CAMERA_SERIAL})."
                )
            time.sleep(0.005)
        for name, ms in warmup.items():
            self.timing.gauge(f"{name.lower()}_warmup_ms", ms)
        return warmup

    def stop(self) -> None:
        if self.show_cameras:
            try:
                import cv2

                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception:
                pass
        for worker in self._workers or []:
            worker.stop()
        for stream in self._streams or []:
            stream.stop()
        self._workers = self._streams = None

    def reset(self) -> None:
        """Clear tracker history, then wait for a fresh frame.  Cameras stay open.

        The wait is not optional.  ``CameraProcessingWorker.reset()`` sets its
        latest frame to ``None`` along with the tracker history, and the caller
        that resets is ``PolicyObservationAdapter.reset()``, whose very next line
        samples.  Without re-waiting, that sample sees no frame from either
        camera and reports "no markers" -- which is what a hardware run actually
        did, sending the diagnosis at the cameras when the frames had just been
        thrown away here.
        """

        self.timing.reset()
        for worker in self._workers or []:
            worker.reset()
        self._last_pose = [None, None]
        self._last_valid_ms = [None, None]
        self._last_quaternion = [None, None]
        self.sources = StickSourceReport()
        if self._workers:
            self.wait_for_first_frames(self.FIRST_FRAME_TIMEOUT_S)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # -- sampling ---------------------------------------------------------
    def sample(self) -> StickPosePair7D:
        if self._workers is None:
            raise RuntimeError("Call start() (or use the context manager) first.")
        t = self.tracker
        with self.timing.stage("frame_fetch"):
            main_frame = self._workers[0].latest()
            side_frame = self._workers[1].latest()
        now_ms = time.monotonic() * 1000.0

        # Gauges, not durations.  Frame age is the latency the policy actually
        # acts on -- the tracker can be fast while the pose it hands over is
        # already three frames old.
        for label, frame in (("main", main_frame), ("side", side_frame)):
            if frame is not None:
                self.timing.gauge(f"{label}_frame_age_ms",
                                  now_ms - frame.host_timestamp_ms)
                self.timing.gauge(f"{label}_tracker_ms", frame.processing_ms)
        if main_frame is not None and side_frame is not None:
            # NOT a fusion error: the arbitration picks ONE camera per stick, so
            # these two frames are never combined.  It matters only when Stick1
            # and Stick2 end up on different cameras -- then the two halves of
            # one observation describe instants this far apart.
            #
            # The tracker prints the same quantity as "Latest timestamp |dt|"
            # and it swings the same way (8-58 ms measured).  Its steadier
            # "Nearest sync pair |dt|" is a different number: a history search
            # used only for MAIN-vs-SIDE agreement diagnostics, never for the
            # pose that is handed onward.
            self.timing.gauge("latest_pair_dt_ms",
                              abs(main_frame.host_timestamp_ms
                                  - side_frame.host_timestamp_ms))

        with self.timing.stage("hand_fk"):
            t_hand_base = deploy_rig.invert(deploy_rig.t_base_hand(self.q6_deg()))

        # First pass on the newest frame from each camera.
        selections = [
            self._select(t, index, main_frame, side_frame, t_hand_base, now_ms)
            for index in range(2)
        ]

        # If the two sticks came from DIFFERENT cameras, the two halves of one
        # observation describe instants up to a frame apart -- measured p95
        # 49 ms, which at the sticks' own 21-24 mm/s is about 1 mm of error in
        # the stick1-to-stick2 geometry the grasp actually depends on.  The
        # tracker already computes the closest-in-time pair from its history
        # (it prints it as "Nearest sync pair", 7-8.5 ms); it just never feeds
        # it to the pose path, because a single-camera display has no use for
        # it.  Building ONE observation out of both cameras does.
        #
        # Only when the sources differ: with both sticks on MAIN, a
        # nearest-pair frame is simply older for no benefit.
        used_main, used_side = main_frame, side_frame
        if self._cross_camera(selections) and self.prefer_synchronised_pair:
            with self.timing.stage("resync"):
                paired = self._nearest_pair(t, now_ms)
                if paired is not None:
                    pair_main, pair_side, pair_dt = paired
                    self.timing.gauge("paired_dt_ms", pair_dt)
                    self.timing.gauge(
                        "paired_age_ms",
                        max(now_ms - pair_main.host_timestamp_ms,
                            now_ms - pair_side.host_timestamp_ms),
                    )
                    used_main, used_side = pair_main, pair_side
                    selections = [
                        self._select(t, index, pair_main, pair_side,
                                     t_hand_base, now_ms)
                        for index in range(2)
                    ]

        if self.show_cameras:
            self._shown += 1
            if self._shown % self.camera_every == 0:
                with self.timing.stage("show"):
                    self._show(main_frame, side_frame)

        poses, states = [], []
        report = StickSourceReport()
        for index in range(2):
            with self.timing.stage(f"select_stick{index + 1}"):
                selection = selections[index]
                # The frames the arbitration ACTUALLY used, which after a
                # resync are not the newest ones.  Reporting latest() here made
                # the diagnostic contradict the verdict: it showed a valid MAIN
                # pose next to a NONE result, because the two were reading
                # different frames.
                report.detail.update(self._detail(index, used_main, used_side))
            setattr(report, f"stick{index + 1}_source", selection["source"])
            setattr(report, f"stick{index + 1}_reason", selection["reason"])
            with self.timing.stage(f"advance_stick{index + 1}"):
                pose, state = self._advance(index, selection, now_ms)
            poses.append(pose)
            states.append(state)
        self.sources = report

        if any(p is None for p in poses):
            # No pose has EVER been produced for this stick, so there is nothing
            # to hold -- distinct from having lost one that was held before.
            raise RuntimeError(self._first_pose_failure(main_frame, side_frame, report))

        severity = {PoseState.VALID: 0, PoseState.REINIT: 0, PoseState.HOLD: 1,
                    PoseState.STALE: 2, PoseState.LOST: 3}
        state = max(states, key=severity.__getitem__)
        return StickPosePair7D(
            poses[0].astype(np.float32), poses[1].astype(np.float32),
            now_ms / 1000.0, state, state in (PoseState.VALID, PoseState.REINIT),
        )

    def _first_pose_failure(self, main_frame, side_frame, report) -> str:
        """Say which of the three distinct faults this is.

        "No pose" has three causes that need different actions, and one message
        for all of them sends the operator to the wrong one:
          * no frame at all        -> camera / connection
          * frames but no markers  -> aim, occlusion, lighting
          * markers but no pose    -> the tracker rejected every IPPE branch
        """

        lines = ["젓가락 포즈를 아직 한 번도 못 만들었습니다."]
        for label, frame in (("MAIN", main_frame), ("SIDE", side_frame)):
            if frame is None:
                lines.append(f"  {label}: 프레임 없음 -- 카메라 쪽 문제입니다.")
            else:
                seen = sorted(frame.detected) if frame.detected else []
                lines.append(
                    f"  {label}: 프레임 있음, 검출된 마커 {seen or '없음'}"
                    + ("  (마커가 안 보입니다 -- 조준/가림/조명)" if not seen else "")
                )
        for index, stick in enumerate(("stick1", "stick2"), start=1):
            reason = getattr(report, f"{stick}_reason")
            ids = tuple(self._modules[index - 1].TARGET_IDS)
            lines.append(f"  Stick{index}: {reason}  (필요한 마커 {list(ids)})")
        lines.append("  두 스틱 모두 포즈가 있어야 105D 관측이 만들어집니다.")
        return "\n".join(lines)

    # -- internals --------------------------------------------------------
    def _show(self, main_frame, side_frame) -> None:
        """Display the tracker's own annotated frames.  Diagnostics only.

        Called from the policy thread, which is where cv2's GUI has to live.
        ``waitKey(1)`` is what actually pumps the window, and it is not free --
        the ``show`` stage measures it, so its cost is visible rather than
        hidden inside the tick.
        """

        try:
            import cv2
        except ImportError:
            self.show_cameras = False
            return
        for label, frame in (("MAIN", main_frame), ("SIDE", side_frame)):
            image = getattr(frame, "vis", None) if frame is not None else None
            if image is None:
                continue
            if self.camera_scale != 1.0:
                image = cv2.resize(image, None, fx=self.camera_scale,
                                   fy=self.camera_scale)
            cv2.imshow(f"wuji {label}", image)
        # pollKey over waitKey(1): same cost within noise, but it does not claim
        # to be a keypress reader.  Mode keys are read from stdin, not here --
        # these windows may not even have focus.
        cv2.pollKey()

    @staticmethod
    def _detail(index, main_frame, side_frame) -> dict:
        """Per-camera verdict for one stick, straight from the tracker."""

        out = {}
        for camera, frame in (("MAIN", main_frame), ("SIDE", side_frame)):
            result = None if frame is None else (
                frame.result1 if index == 0 else frame.result2
            )
            if result is None:
                continue
            out[(camera, index)] = {
                "source": result.get("source"),
                "reason": result.get("reason"),
                "raw_valid": result.get("raw_valid"),
                "detected_markers": result.get("detected_markers"),
            }
        return out

    @staticmethod
    def _cross_camera(selections) -> bool:
        """True when the two sticks resolved to different cameras."""

        sources = {s["source"] for s in selections}
        return "MAIN" in sources and "SIDE" in sources

    #: A resynced pair may be no older than this.  ``find_nearest_timestamp_pair``
    #: minimises |dt| and nothing else, so it will happily return the two OLDEST
    #: frames in an eight-deep history if those happen to line up best -- and
    #: with them, whatever the tracker decided about that older frame.  Trading
    #: 40 ms of skew for 200 ms of staleness is not a trade worth making.
    MAX_PAIRED_AGE_MS = 120.0

    def _nearest_pair(self, t, now_ms: float):
        """Closest-in-time MAIN/SIDE frames, subject to an age ceiling."""

        main_history = self._workers[0].history_snapshot()
        side_history = self._workers[1].history_snapshot()

        def fresh(history):
            return [f for f in history
                    if now_ms - f.host_timestamp_ms <= self.MAX_PAIRED_AGE_MS]

        pair_main, pair_side, dt_ms = t.find_nearest_timestamp_pair(
            fresh(main_history), fresh(side_history)
        )
        if pair_main is None or pair_side is None:
            return None
        return pair_main, pair_side, dt_ms

    def _select(self, t, index, main_frame, side_frame, t_hand_base, now_ms):
        """Run the tracker's own per-stick MAIN/SIDE arbitration."""

        module = self._modules[index]

        def parts(frame):
            if frame is None:
                return {}, None, None, None
            result = frame.result1 if index == 0 else frame.result2
            hand = (
                t.compute_hand_relative_pose(result, module, t_hand_base,
                                             self._last_quaternion[index])
                if result is not None else None
            )
            return frame.detected, result, hand, frame.host_timestamp_ms

        main_detected, main_result, main_hand, main_ms = parts(main_frame)
        _, side_result, side_hand, side_ms = parts(side_frame)
        return t.select_final_pose_for_stick(
            f"Stick{index + 1}",
            tuple(module.TARGET_IDS),
            main_detected, main_result, main_hand, main_ms,
            side_result, side_hand, side_ms,
            now_ms, self._last_quaternion[index],
        )

    def _advance(self, index, selection, now_ms):
        """Fold a per-frame selection into a held pose plus a freshness state."""

        pose = selection.get("pose")
        if pose is not None:
            first = self._last_valid_ms[index] is None
            quaternion = canonicalize_square_stick_quaternion(
                np.asarray(pose["quaternion"], dtype=np.float64),
                STICK_REFERENCE_QUATERNIONS_PALM_WXYZ[index],
            )
            self._last_quaternion[index] = quaternion
            self._last_pose[index] = np.concatenate(
                (np.asarray(pose["position"], dtype=np.float64), quaternion)
            )
            self._last_valid_ms[index] = now_ms
            return self._last_pose[index], (PoseState.REINIT if first else PoseState.VALID)

        if self._last_valid_ms[index] is None:
            return None, PoseState.LOST
        blind_ms = now_ms - self._last_valid_ms[index]
        if blind_ms <= self.hold_after_ms:
            return self._last_pose[index], PoseState.HOLD
        if blind_ms <= self.stale_after_ms:
            return self._last_pose[index], PoseState.STALE
        return self._last_pose[index], PoseState.LOST
