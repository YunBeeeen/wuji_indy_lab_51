#!/usr/bin/env python3
from __future__ import annotations

"""
Two-D435 / two-stick / Indy7->Wuji Hand integration verification.

Transform convention
--------------------
T_A_B maps coordinates expressed in B into A.

MAIN:
    T_HAND_STICK_MAIN
    = T_HAND_BASE @ T_BASE_CAMERA_MAIN @ T_MAIN_STICK

SIDE:
    T_HAND_STICK_SIDE
    = T_HAND_BASE @ T_BASE_CAMERA_SIDE @ T_SIDE_STICK

Final source policy (PER STICK)
-------------------------------
Stick1 uses MAIN whenever MAIN sees ID0 OR ID1.
SIDE is allowed only when MAIN sees NEITHER ID0 NOR ID1.

Stick2 uses MAIN whenever MAIN sees ID2 OR ID3.
SIDE is allowed only when MAIN sees NEITHER ID2 NOR ID3.

Important:
- SIDE trackers run continuously in the background even while MAIN owns output.
  This keeps SIDE history/DUAL correction warm for handoff.
- MAIN uses the existing MAIN-camera workspace prior.
- SIDE intentionally does NOT use the MAIN-camera workspace prior.
  On a fresh SIDE single-marker start it falls back to the best reprojection
  branch. DUAL/history still have priority when available.
- The cameras are not hardware synchronized. Each frame gets a host monotonic
  timestamp at acquisition; MAIN/SIDE comparison prints |dt| in ms.
"""

import importlib.util
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# PATHS / CAMERA CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STICK1_MODULE_PATH = BASE_DIR / "run_stick1_dual.py"

STICK2_MODULE_CANDIDATES = [
    BASE_DIR / "run_stick2_dual.py",
    BASE_DIR / "run_stick2_dual(1).py",
]

MAIN_CAMERA_SERIAL = "814412070582"
SIDE_CAMERA_SERIAL = "342222074358"

PRINT_INTERVAL_SEC = 1.0
PERF_INTERVAL_SEC = 5.0
STICK_AXIS_LENGTH_M = 0.030

# Host-arrival timestamp is used for cross-camera comparison.
# This flag does NOT reject poses; it only labels whether the comparison
# was temporally close enough to trust strongly.
SYNC_COMPARE_GOOD_MS = 20.0

# Prevent a disconnected/frozen camera from supplying an old pose forever.
CAMERA_STALE_MS = 150.0


# ============================================================
# CAMERA -> INDY7 BASE EXTRINSICS
# ============================================================

# MAIN D435, calibrated with fixed base markers.
T_BASE_CAMERA_MAIN = np.array(
    [
        [ 0.011009927,  0.713786390, -0.700276924,  0.937431906],
        [ 0.999937040, -0.009377283,  0.006163071, -0.111095107],
        [-0.002167578, -0.700300689, -0.713844693,  0.435281684],
        [ 0.0,          0.0,          0.0,          1.0        ],
    ],
    dtype=np.float64,
)

# SIDE D435, serial 342222074358.
# Chosen from the calibration run with the lowest reported RMS (0.950 mm).
T_BASE_CAMERA_SIDE = np.array(
    [
        [ 0.999989882, -0.003139647,  0.003221657,  0.654231507],
        [-0.003071812,  0.046616270,  0.998908148, -0.372806389],
        [-0.003286401, -0.998907937,  0.046606154,  0.153432702],
        [ 0.0,          0.0,          0.0,          1.0        ],
    ],
    dtype=np.float64,
)


# ============================================================
# INDY7 -> WUJI HAND
# ============================================================

# q1~q5 fixed:
# [10.044318, -64.03332, -131.97517, 9.914346, 103.22368] deg
#
# BASE <- J6, immediately before variable q6 rotation.
T_BASE_J6 = np.array(
    [
        [-0.044204390, -0.008842933,  0.998983370,  0.473385519],
        [-0.047828501,  0.998832922,  0.006725220, -0.131376150],
        [-0.997876950, -0.047482593, -0.044575744,  0.161305068],
        [ 0.0,          0.0,          0.0,          1.0        ],
    ],
    dtype=np.float64,
)

# q6 remains a VARIABLE.
# Later replace the manual value with the live Indy7 joint-6 encoder.
Q6_INITIAL_DEG = 25.000097
Q6_STEP_DEG = 1.0

# link6 +Z and Wuji Hand +Z are physically shared.
# Translation along shared +Z:
#   60 mm final robot section + 17 mm bracket + 30 mm mount = 107 mm
HAND_OFFSET_Z_M = 0.107

# Current fixed mounting-yaw candidate.
# Keep this explicit so it can be changed after physical Hand-axis verification.
HAND_MOUNT_YAW_OFFSET_DEG = 155.0


# ============================================================
# BASIC TRANSFORM HELPERS
# ============================================================

def rotz_transform(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))

    return np.array(
        [
            [c, -s, 0.0, 0.0],
            [s,  c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def canonicalize_quaternion(q, previous=None):
    q = np.asarray(q, dtype=np.float64).reshape(4).copy()
    q /= np.linalg.norm(q)

    if previous is not None:
        previous = np.asarray(previous, dtype=np.float64).reshape(4)
        previous /= np.linalg.norm(previous)

        if np.dot(previous, q) < 0.0:
            q = -q
    elif q[0] < 0.0:
        q = -q

    return q


def build_T_LINK6_HAND() -> np.ndarray:
    T = rotz_transform(np.deg2rad(HAND_MOUNT_YAW_OFFSET_DEG))
    T[:3, 3] = np.array(
        [0.0, 0.0, HAND_OFFSET_Z_M],
        dtype=np.float64,
    )
    return T


T_LINK6_HAND = build_T_LINK6_HAND()


def get_T_BASE_HAND(q6_deg: float) -> np.ndarray:
    return (
        T_BASE_J6
        @ rotz_transform(np.deg2rad(float(q6_deg)))
        @ T_LINK6_HAND
    )


# ============================================================
# MODULE LOADING
# ============================================================

def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Tracker module not found: {path}")

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_stick2_module_path() -> Path:
    for path in STICK2_MODULE_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Stick2 tracker module not found. Tried:\n  "
        + "\n  ".join(str(p) for p in STICK2_MODULE_CANDIDATES)
    )


# ============================================================
# CAMERA ACQUISITION
# ============================================================

@dataclass
class CameraPacket:
    frame_number: int
    host_timestamp_ms: float
    device_timestamp_ms: float
    image: np.ndarray


class CameraStream:
    """
    One independent RealSense pipeline + acquisition thread.

    host_timestamp_ms:
        time.monotonic_ns() stamped immediately after the color frame arrives.
        This is used for MAIN/SIDE software timing comparison.

    device_timestamp_ms:
        RealSense's own frame timestamp, kept only for diagnostics because
        separate devices are not assumed to share the same device clock epoch.
    """

    def __init__(
        self,
        name: str,
        serial: str,
        width: int,
        height: int,
        fps: int,
    ):
        self.name = name
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_device(self.serial)
        self.config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )

        self.profile = None
        self.K = None
        self.dist = None

        self._lock = threading.Lock()
        self._latest = None
        self._stop = threading.Event()
        self._thread = None
        self._started = False

        # Acquisition diagnostics.  These counters are updated in the
        # camera thread, independently of ArUco processing in the main loop.
        self._capture_count = 0
        self._capture_skipped = 0
        self._last_capture_frame_number = None

    def start(self):
        self.profile = self.pipeline.start(self.config)
        self._started = True

        color_profile = (
            self.profile
            .get_stream(rs.stream.color)
            .as_video_stream_profile()
        )
        intr = color_profile.get_intrinsics()

        self.K = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.dist = np.asarray(intr.coeffs, dtype=np.float64)

        self._thread = threading.Thread(
            target=self._worker,
            name=f"{self.name}_camera_thread",
            daemon=True,
        )
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames(1000)
            except RuntimeError:
                if self._stop.is_set():
                    break
                continue

            color = frames.get_color_frame()
            if not color:
                continue

            # Stamp on host immediately after frame arrival.
            host_timestamp_ms = time.monotonic_ns() * 1e-6

            frame_number = int(color.get_frame_number())

            packet = CameraPacket(
                frame_number=frame_number,
                host_timestamp_ms=float(host_timestamp_ms),
                device_timestamp_ms=float(color.get_timestamp()),
                image=np.asanyarray(color.get_data()).copy(),
            )

            with self._lock:
                if self._last_capture_frame_number is not None:
                    gap = frame_number - self._last_capture_frame_number
                    if gap > 1:
                        self._capture_skipped += gap - 1

                self._last_capture_frame_number = frame_number
                self._capture_count += 1
                self._latest = packet

    def latest(self):
        with self._lock:
            return self._latest

    def stats_snapshot(self):
        with self._lock:
            return {
                "capture_count": int(self._capture_count),
                "capture_skipped": int(self._capture_skipped),
                "last_capture_frame_number": self._last_capture_frame_number,
            }

    def stop(self):
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=1.2)

        if self._started:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass

        self._started = False


# ============================================================
# TRACKER STATE
# ============================================================

class StickTrackerState:
    """
    Stateful wrapper around one existing single-stick tracker module.

    use_workspace_prior=True:
        Normal MAIN-camera behavior.

    use_workspace_prior=False:
        SIDE-camera behavior.
        - DUAL reference still used.
        - HISTORY reference still used.
        - Fresh single-marker start does NOT use MAIN-camera workspace CSV.
        - Fresh start picks the valid IPPE candidate with smallest reprojection.
    """

    def __init__(
        self,
        module,
        name: str,
        use_workspace_prior: bool,
    ):
        self.m = module
        self.name = name
        self.use_workspace_prior = bool(use_workspace_prior)

        self.marker_a = int(module.MARKER_A_ID)
        self.marker_b = int(module.MARKER_B_ID)
        self.marker_ids = (self.marker_a, self.marker_b)

        if self.use_workspace_prior:
            self.workspace_refs = (
                module.load_workspace_orientation_references(
                    module.WORKSPACE_REFERENCE_CSV
                )
            )
        else:
            self.workspace_refs = {
                self.marker_a: [],
                self.marker_b: [],
            }

        self.T_MA_MB, self.marker_to_stick = (
            module.build_marker_to_stick_transforms()
        )
        self.dual_object_a, self.dual_object_b = (
            module.build_dual_object_points(
                self.marker_to_stick
            )
        )

        self.reset()

    def reset(self):
        m = self.m

        self.previous_single_position = {
            marker_id: None
            for marker_id in self.marker_ids
        }
        self.previous_single_quaternion = {
            marker_id: None
            for marker_id in self.marker_ids
        }
        self.marker_miss_count = {
            marker_id: 0
            for marker_id in self.marker_ids
        }
        self.single_reject_streak = {
            marker_id: 0
            for marker_id in self.marker_ids
        }

        self.previous_marker_quaternion_camera = {
            marker_id: None
            for marker_id in self.marker_ids
        }

        self.previous_dual_position = None
        self.previous_dual_quaternion = None
        self.dual_miss_count = 0
        self.dual_reject_streak = 0

        self.correction_state = {
            marker_id: m.make_correction_state()
            for marker_id in self.marker_ids
        }

        self.final_filtered_position = None
        self.final_filtered_quaternion = None
        self.final_invalid_streak = 0

    def _select_physical_branch(
        self,
        marker_id,
        candidates,
        T_BASE_CAMERA,
        T_CAMERA_BASE,
        dual,
    ):
        """
        MAIN:
            Delegate to original verified selector.

        SIDE:
            DUAL/HISTORY still use original verified selector.
            Fresh start skips workspace prior and uses min reprojection.
        """
        m = self.m

        if self.use_workspace_prior:
            return m.select_physical_branch(
                marker_id,
                candidates,
                T_BASE_CAMERA,
                self.marker_to_stick[marker_id],
                T_CAMERA_BASE,
                dual,
                self.previous_marker_quaternion_camera[marker_id],
                self.workspace_refs,
            )

        # SIDE: no candidates.
        if candidates is None or len(candidates) == 0:
            return None

        # SIDE: if DUAL or HISTORY exists, reuse the original physical selector.
        if (
            (dual is not None and dual.get("accepted", False))
            or self.previous_marker_quaternion_camera[marker_id] is not None
        ):
            return m.select_physical_branch(
                marker_id,
                candidates,
                T_BASE_CAMERA,
                self.marker_to_stick[marker_id],
                T_CAMERA_BASE,
                dual,
                self.previous_marker_quaternion_camera[marker_id],
                self.workspace_refs,
            )

        # SIDE fresh single-marker start:
        # no workspace restriction, only basic reprojection acceptance.
        valid = [
            c
            for c in candidates
            if c["reproj"] <= m.SINGLE_MAX_REPROJ_ERROR_PX
        ]

        if not valid:
            return None

        chosen = min(
            valid,
            key=lambda c: c["reproj"],
        )

        return m.build_single_from_specific_candidate(
            chosen,
            T_BASE_CAMERA,
            self.marker_to_stick[marker_id],
            None,
            "SIDE_FRESH_REPROJ",
        )

    def update(
        self,
        detected,
        K,
        dist_coeffs,
        T_BASE_CAMERA,
        T_CAMERA_BASE,
        now,
    ):
        m = self.m

        # ----------------------------------------------------
        # Miss handling
        # ----------------------------------------------------
        for marker_id in self.marker_ids:
            if marker_id in detected:
                if (
                    self.marker_miss_count[marker_id]
                    >= m.SINGLE_HISTORY_RESET_MISSES
                ):
                    self.previous_single_position[marker_id] = None
                    self.previous_single_quaternion[marker_id] = None
                    self.previous_marker_quaternion_camera[marker_id] = None

                self.marker_miss_count[marker_id] = 0
            else:
                self.marker_miss_count[marker_id] += 1

        # ----------------------------------------------------
        # SINGLE candidate generation
        # ----------------------------------------------------
        single_raw = {}
        candidate_lists = {}

        for marker_id in self.marker_ids:
            if marker_id not in detected:
                continue

            candidates = m.get_ippe_candidates(
                detected[marker_id],
                K,
                dist_coeffs,
            )
            candidate_lists[marker_id] = candidates

            selected = m.select_single_candidate(
                candidates,
                T_BASE_CAMERA,
                self.marker_to_stick[marker_id],
                self.previous_single_position[marker_id],
                self.previous_single_quaternion[marker_id],
            )

            if selected is None:
                self.single_reject_streak[marker_id] += 1

                if (
                    self.single_reject_streak[marker_id]
                    >= m.SINGLE_REJECT_RESET_FRAMES
                ):
                    self.previous_single_position[marker_id] = None
                    self.previous_single_quaternion[marker_id] = None
                    self.previous_marker_quaternion_camera[marker_id] = None
                    self.single_reject_streak[marker_id] = 0

                    selected = m.select_single_candidate(
                        candidates,
                        T_BASE_CAMERA,
                        self.marker_to_stick[marker_id],
                        None,
                        None,
                    )
            else:
                self.single_reject_streak[marker_id] = 0

            if selected is None:
                continue

            self.previous_single_position[marker_id] = (
                selected["position"].copy()
            )
            self.previous_single_quaternion[marker_id] = (
                selected["quaternion"].copy()
            )
            single_raw[marker_id] = selected

        # ----------------------------------------------------
        # DUAL
        # ----------------------------------------------------
        dual = None

        both_detected = (
            self.marker_a in detected
            and self.marker_b in detected
        )

        if both_detected:
            self.dual_miss_count = 0

            dual = m.estimate_dual_pose(
                detected[self.marker_a],
                detected[self.marker_b],
                self.dual_object_a,
                self.dual_object_b,
                K,
                dist_coeffs,
                T_BASE_CAMERA,
                T_CAMERA_BASE,
                self.previous_dual_position,
                self.previous_dual_quaternion,
            )

            if dual is not None and dual.get("accepted", False):
                self.previous_dual_position = dual["position"].copy()
                self.previous_dual_quaternion = dual["quaternion"].copy()
                self.dual_reject_streak = 0
            else:
                self.dual_reject_streak += 1

                if (
                    self.dual_reject_streak
                    >= m.DUAL_REJECT_RESET_FRAMES
                ):
                    self.previous_dual_position = None
                    self.previous_dual_quaternion = None
                    self.dual_reject_streak = 0

        else:
            self.dual_miss_count += 1

            if (
                self.dual_miss_count
                >= m.DUAL_HISTORY_RESET_MISSES
            ):
                self.previous_dual_position = None
                self.previous_dual_quaternion = None

        # ----------------------------------------------------
        # Physical IPPE branch selection
        # ----------------------------------------------------
        branch_info = {}
        marker_debug = {}

        for marker_id in self.marker_ids:
            physical = self._select_physical_branch(
                marker_id,
                candidate_lists.get(marker_id),
                T_BASE_CAMERA,
                T_CAMERA_BASE,
                dual,
            )

            if physical is None:
                if marker_id in candidate_lists:
                    single_raw.pop(marker_id, None)
                continue

            single_raw[marker_id] = physical

            self.previous_marker_quaternion_camera[marker_id] = (
                physical["marker_quaternion_camera"].copy()
            )
            self.previous_single_position[marker_id] = (
                physical["position"].copy()
            )
            self.previous_single_quaternion[marker_id] = (
                physical["quaternion"].copy()
            )

            branch_info[marker_id] = {
                "branch": physical.get("branch"),
                "reference": physical.get(
                    "marker_reference_source",
                    "NONE",
                ),
                "dR_ref_deg": physical.get(
                    "marker_reference_rotation_deg"
                ),
            }

            # Marker-level debug in BASE, useful for MAIN/SIDE verification.
            T_BASE_MARKER = physical.get("T_BASE_MARKER")
            T_CAMERA_MARKER = physical.get("T_CAMERA_MARKER")

            if (
                T_BASE_MARKER is not None
                and T_CAMERA_MARKER is not None
            ):
                q_base_marker = (
                    m.rotation_matrix_to_quaternion_wxyz(
                        T_BASE_MARKER[:3, :3]
                    )
                )
                q_camera_marker = (
                    m.rotation_matrix_to_quaternion_wxyz(
                        T_CAMERA_MARKER[:3, :3]
                    )
                )

                marker_debug[marker_id] = {
                    "T_BASE_MARKER": T_BASE_MARKER.copy(),
                    "T_CAMERA_MARKER": T_CAMERA_MARKER.copy(),
                    "base_position": T_BASE_MARKER[:3, 3].copy(),
                    "base_quaternion": q_base_marker.copy(),
                    "camera_position": T_CAMERA_MARKER[:3, 3].copy(),
                    "camera_quaternion": q_camera_marker.copy(),
                    "branch": physical.get("branch"),
                    "reference": physical.get(
                        "marker_reference_source",
                        "NONE",
                    ),
                    "reproj": physical.get("reproj"),
                }

        # ----------------------------------------------------
        # Online SINGLE -> DUAL correction
        # ----------------------------------------------------
        if dual is not None and dual.get("accepted", False):
            for marker_id in self.marker_ids:
                if marker_id in single_raw:
                    m.update_online_correction(
                        self.correction_state[marker_id],
                        single_raw[marker_id],
                        dual,
                        now,
                    )

        single_corrected = {
            marker_id: m.apply_online_correction(
                estimator,
                self.correction_state[marker_id],
            )
            for marker_id, estimator in single_raw.items()
        }

        # ----------------------------------------------------
        # Final source inside THIS camera
        # ----------------------------------------------------
        actual_source, reason, selected_estimator = (
            m.choose_final_source(
                single_corrected,
                dual,
            )
        )

        if selected_estimator is None:
            raw_position = None
            raw_quaternion = None
        else:
            raw_position = selected_estimator["position"].copy()
            raw_quaternion = selected_estimator["quaternion"].copy()

        raw_valid = raw_position is not None

        # ----------------------------------------------------
        # Final filter inside THIS camera
        # ----------------------------------------------------
        if raw_valid:
            self.final_invalid_streak = 0

            if self.final_filtered_position is None:
                self.final_filtered_position = raw_position.copy()
            else:
                self.final_filtered_position = (
                    m.POS_ALPHA * raw_position
                    + (1.0 - m.POS_ALPHA)
                    * self.final_filtered_position
                )

            if self.final_filtered_quaternion is None:
                self.final_filtered_quaternion = raw_quaternion.copy()
            else:
                if (
                    np.dot(
                        self.final_filtered_quaternion,
                        raw_quaternion,
                    )
                    < 0.0
                ):
                    raw_quaternion = -raw_quaternion

                self.final_filtered_quaternion = (
                    m.quaternion_slerp(
                        self.final_filtered_quaternion,
                        raw_quaternion,
                        m.ROT_ALPHA,
                    )
                )

        else:
            self.final_invalid_streak += 1

            if (
                self.final_invalid_streak
                >= m.FINAL_FILTER_RESET_MISSES
            ):
                self.final_filtered_position = None
                self.final_filtered_quaternion = None

        filtered_available = (
            self.final_filtered_position is not None
        )

        return {
            "name": self.name,
            "source": actual_source,
            "reason": reason,
            "raw_valid": raw_valid,
            "filtered_available": filtered_available,
            "position": (
                None
                if self.final_filtered_position is None
                else self.final_filtered_position.copy()
            ),
            "quaternion": (
                None
                if self.final_filtered_quaternion is None
                else self.final_filtered_quaternion.copy()
            ),
            "dual": dual,
            "branch_info": branch_info,
            "marker_debug": marker_debug,
            "detected_markers": [
                marker_id
                for marker_id in self.marker_ids
                if marker_id in detected
            ],
        }


# ============================================================
# ARUCO / POSE HELPERS
# ============================================================

def build_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_SUBPIX
    )

    return cv2.aruco.ArucoDetector(
        dictionary,
        params,
    )


def detect_target_markers(
    frame,
    detector,
    target_ids,
):
    vis = frame.copy()
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    corners, ids, _ = detector.detectMarkers(gray)
    detected = {}

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(
            vis,
            corners,
            ids,
        )

        for corner, marker_id in zip(
            corners,
            ids.flatten(),
        ):
            marker_id = int(marker_id)

            if marker_id not in target_ids:
                continue

            detected[marker_id] = (
                np.asarray(
                    corner,
                    dtype=np.float64,
                )
                .reshape(4, 2)
            )

    return detected, vis


def result_to_base_transform(result, module):
    if (
        result is None
        or result["position"] is None
        or result["quaternion"] is None
    ):
        return None

    return module.make_transform(
        module.quaternion_wxyz_to_rotation_matrix(
            result["quaternion"]
        ),
        result["position"],
    )


def compute_hand_relative_pose(
    result,
    module,
    T_HAND_BASE,
    previous_quaternion=None,
):
    # Do not expose a held filtered pose as a currently valid measurement.
    # The tracker may retain its internal filter state for reacquisition, but
    # GUI / comparison / final output should become NO POSE on the first
    # processed frame for which raw tracking is invalid.
    if result is None or not result.get("raw_valid", False):
        return {
            "available": False,
            "T_BASE_STICK": None,
            "T_HAND_STICK": None,
            "base_position": None,
            "base_quaternion": None,
            "position": None,
            "quaternion": None,
        }

    T_BASE_STICK = result_to_base_transform(
        result,
        module,
    )

    if T_BASE_STICK is None:
        return {
            "available": False,
            "T_BASE_STICK": None,
            "T_HAND_STICK": None,
            "base_position": None,
            "base_quaternion": None,
            "position": None,
            "quaternion": None,
        }

    T_HAND_STICK = (
        T_HAND_BASE
        @ T_BASE_STICK
    )

    q_hand = (
        module.rotation_matrix_to_quaternion_wxyz(
            T_HAND_STICK[:3, :3]
        )
    )
    q_hand = canonicalize_quaternion(
        q_hand,
        previous_quaternion,
    )

    q_base = (
        module.rotation_matrix_to_quaternion_wxyz(
            T_BASE_STICK[:3, :3]
        )
    )

    return {
        "available": True,
        "T_BASE_STICK": T_BASE_STICK,
        "T_HAND_STICK": T_HAND_STICK,
        "base_position": T_BASE_STICK[:3, 3].copy(),
        "base_quaternion": q_base.copy(),
        "position": T_HAND_STICK[:3, 3].copy(),
        "quaternion": q_hand.copy(),
    }


def draw_stick_pose(
    vis,
    result,
    module,
    K,
    dist,
    T_CAMERA_BASE,
):
    # Do not draw a stale/held filtered axis after raw detection is lost.
    if result is None or not result.get("raw_valid", False):
        return

    T_BASE_STICK = result_to_base_transform(
        result,
        module,
    )

    if T_BASE_STICK is None:
        return

    T_CAMERA_STICK = (
        T_CAMERA_BASE
        @ T_BASE_STICK
    )

    if T_CAMERA_STICK[2, 3] <= 0.0:
        return

    rvec, _ = cv2.Rodrigues(
        T_CAMERA_STICK[:3, :3]
    )
    tvec = (
        T_CAMERA_STICK[:3, 3]
        .reshape(3, 1)
    )

    cv2.drawFrameAxes(
        vis,
        K,
        dist,
        rvec,
        tvec,
        STICK_AXIS_LENGTH_M,
        2,
    )


def pose_difference(
    hand_a,
    hand_b,
    module,
):
    if (
        hand_a is None
        or hand_b is None
        or not hand_a["available"]
        or not hand_b["available"]
    ):
        return None

    dp_mm = float(
        np.linalg.norm(
            hand_a["position"]
            - hand_b["position"]
        )
        * 1000.0
    )

    dr_deg = float(
        module.quaternion_angle_deg(
            hand_a["quaternion"],
            hand_b["quaternion"],
        )
    )

    return dp_mm, dr_deg


def format_hand_pose(hand_result):
    if (
        hand_result is None
        or not hand_result["available"]
    ):
        return "NO POSE"

    p = hand_result["position"] * 1000.0
    q = hand_result["quaternion"]

    return (
        f"XYZ[{p[0]:+.1f},{p[1]:+.1f},{p[2]:+.1f}]mm "
        f"Q[{q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f}]"
    )


def format_base_pose(hand_result):
    if (
        hand_result is None
        or not hand_result["available"]
    ):
        return "NO POSE"

    p = hand_result["base_position"] * 1000.0
    q = hand_result["base_quaternion"]

    return (
        f"XYZ[{p[0]:+.1f},{p[1]:+.1f},{p[2]:+.1f}]mm "
        f"Q[{q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f}]"
    )


# ============================================================
# FINAL MAIN-vs-SIDE SOURCE POLICY
# ============================================================

def select_final_pose_for_stick(
    stick_name,
    marker_ids,
    main_detected,
    main_result,
    main_hand,
    main_timestamp_ms,
    side_result,
    side_hand,
    side_timestamp_ms,
    now_ms,
    previous_final_quaternion,
):
    """
    Exact requested policy:

    If MAIN sees at least one marker of this stick:
        MAIN owns this stick.
        SIDE is NOT allowed to replace it.

    Only if MAIN sees neither marker:
        SIDE may provide the pose.
    """

    main_has_any_marker = any(
        marker_id in main_detected
        for marker_id in marker_ids
    )

    main_fresh = (
        main_timestamp_ms is not None
        and now_ms - main_timestamp_ms <= CAMERA_STALE_MS
    )
    side_fresh = (
        side_timestamp_ms is not None
        and now_ms - side_timestamp_ms <= CAMERA_STALE_MS
    )

    selected = None
    source = "NONE"
    reason = "NO_VALID_POSE"

    if main_fresh and main_has_any_marker:
        if (
            main_result is not None
            and main_result["raw_valid"]
            and main_hand is not None
            and main_hand["available"]
        ):
            selected = dict(main_hand)
            source = "MAIN"
            reason = "MAIN_SEES_AT_LEAST_ONE_MARKER"
        else:
            # Intentional: do NOT use SIDE here.
            reason = "MAIN_VISIBLE_BUT_INVALID_SIDE_BLOCKED"

    else:
        if (
            side_fresh
            and side_result is not None
            and side_result["raw_valid"]
            and side_hand is not None
            and side_hand["available"]
        ):
            selected = dict(side_hand)
            source = "SIDE"
            reason = "MAIN_MISSED_BOTH_MARKERS"
        else:
            reason = "MAIN_MISSED_BOTH_SIDE_INVALID"

    if selected is not None:
        selected["quaternion"] = canonicalize_quaternion(
            selected["quaternion"],
            previous_final_quaternion,
        )

    return {
        "stick": stick_name,
        "source": source,
        "reason": reason,
        "pose": selected,
        "main_has_any_marker": main_has_any_marker,
    }


# ============================================================
# MARKER-LEVEL SNAPSHOT
# ============================================================

def print_marker_snapshot(
    main_results,
    side_results,
    modules,
):
    print()
    print("=" * 96)
    print("MARKER-LEVEL MAIN vs SIDE SNAPSHOT (BASE frame)")
    print("=" * 96)

    for stick_index, (
        main_result,
        side_result,
        module,
    ) in enumerate(
        zip(main_results, side_results, modules),
        start=1,
    ):
        marker_ids = sorted(
            set(main_result["marker_debug"].keys())
            | set(side_result["marker_debug"].keys())
        )

        print(f"\nSTICK{stick_index}")

        if not marker_ids:
            print("  No physical marker pose available.")
            continue

        for marker_id in marker_ids:
            m = main_result["marker_debug"].get(marker_id)
            s = side_result["marker_debug"].get(marker_id)

            print(f"  ID{marker_id}")

            if m is not None:
                p = m["base_position"] * 1000.0
                q = m["base_quaternion"]
                print(
                    "    MAIN BASE "
                    f"XYZ[{p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}]mm "
                    f"Q[{q[0]:+.5f},{q[1]:+.5f},{q[2]:+.5f},{q[3]:+.5f}] "
                    f"b={m['branch']} ref={m['reference']}"
                )
            else:
                print("    MAIN BASE NO POSE")

            if s is not None:
                p = s["base_position"] * 1000.0
                q = s["base_quaternion"]
                print(
                    "    SIDE BASE "
                    f"XYZ[{p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}]mm "
                    f"Q[{q[0]:+.5f},{q[1]:+.5f},{q[2]:+.5f},{q[3]:+.5f}] "
                    f"b={s['branch']} ref={s['reference']}"
                )
            else:
                print("    SIDE BASE NO POSE")

            if m is not None and s is not None:
                dp_mm = float(
                    np.linalg.norm(
                        m["base_position"]
                        - s["base_position"]
                    )
                    * 1000.0
                )
                dr_deg = float(
                    module.quaternion_angle_deg(
                        m["base_quaternion"],
                        s["base_quaternion"],
                    )
                )
                print(
                    f"    DIFF dP={dp_mm:.3f} mm "
                    f"dR={dr_deg:.3f} deg"
                )

    print("=" * 96)


# ============================================================
# MAIN
# ============================================================


# ============================================================
# PARALLEL CAMERA PROCESSING
# ============================================================

from collections import deque


@dataclass
class ProcessedFrame:
    frame_number: int
    host_timestamp_ms: float
    device_timestamp_ms: float
    detected: dict
    result1: dict | None
    result2: dict | None
    vis: np.ndarray
    processing_ms: float


@dataclass
class PoseSnapshot:
    frame_number: int
    host_timestamp_ms: float
    detected: dict
    result1: dict | None
    result2: dict | None


class CameraProcessingWorker:
    """
    One processing thread per camera.

    The camera acquisition thread stamps each RGB frame with the common host
    monotonic clock.  This worker carries THAT SAME acquisition timestamp
    through ArUco + Stick1/Stick2 processing.  Processing completion time is
    never used for camera synchronization.

    Only the newest camera frame is processed.  If processing falls behind,
    intermediate frames are intentionally dropped rather than queued, keeping
    latency bounded.
    """

    def __init__(
        self,
        name: str,
        camera: CameraStream,
        tracker1: StickTrackerState,
        tracker2: StickTrackerState,
        module1,
        module2,
        target_ids,
        T_BASE_CAMERA: np.ndarray,
        T_CAMERA_BASE: np.ndarray,
        history_size: int = 8,
    ):
        self.name = str(name)
        self.camera = camera
        self.tracker1 = tracker1
        self.tracker2 = tracker2
        self.module1 = module1
        self.module2 = module2
        self.target_ids = set(target_ids)
        self.T_BASE_CAMERA = np.asarray(T_BASE_CAMERA, dtype=np.float64)
        self.T_CAMERA_BASE = np.asarray(T_CAMERA_BASE, dtype=np.float64)

        # One detector per processing thread.  Do not share an ArUcoDetector
        # instance between MAIN and SIDE threads.
        self.detector = build_aruco_detector()

        self._lock = threading.Lock()
        self._tracker_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        self._latest: ProcessedFrame | None = None
        self._history = deque(maxlen=int(history_size))

        self._processed_count = 0
        self._processed_skipped = 0
        self._last_processed_frame_number = None
        self._processing_ms_sum = 0.0
        self._processing_ms_count = 0
        self._processing_ms_last = 0.0

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker,
            name=f"{self.name}_vision_thread",
            daemon=True,
        )
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            packet = self.camera.latest()

            if packet is None:
                time.sleep(0.0005)
                continue

            with self._lock:
                already_processed = (
                    self._last_processed_frame_number == packet.frame_number
                )

            if already_processed:
                time.sleep(0.0005)
                continue

            t0 = time.perf_counter()

            # All stateful tracker mutations for this camera live in this
            # thread.  reset() uses the same lock so reset cannot interleave
            # halfway through a tracker update.
            with self._tracker_lock:
                detected, vis = detect_target_markers(
                    packet.image,
                    self.detector,
                    self.target_ids,
                )

                now_sec = packet.host_timestamp_ms * 1e-3

                result1 = self.tracker1.update(
                    detected,
                    self.camera.K,
                    self.camera.dist,
                    self.T_BASE_CAMERA,
                    self.T_CAMERA_BASE,
                    now_sec,
                )
                result2 = self.tracker2.update(
                    detected,
                    self.camera.K,
                    self.camera.dist,
                    self.T_BASE_CAMERA,
                    self.T_CAMERA_BASE,
                    now_sec,
                )

                # Draw axes in the worker.  This does not depend on q6/Hand FK.
                draw_stick_pose(
                    vis,
                    result1,
                    self.module1,
                    self.camera.K,
                    self.camera.dist,
                    self.T_CAMERA_BASE,
                )
                draw_stick_pose(
                    vis,
                    result2,
                    self.module2,
                    self.camera.K,
                    self.camera.dist,
                    self.T_CAMERA_BASE,
                )

            processing_ms = (time.perf_counter() - t0) * 1000.0

            processed = ProcessedFrame(
                frame_number=packet.frame_number,
                host_timestamp_ms=packet.host_timestamp_ms,
                device_timestamp_ms=packet.device_timestamp_ms,
                detected=dict(detected),
                result1=result1,
                result2=result2,
                vis=vis,
                processing_ms=processing_ms,
            )

            snapshot = PoseSnapshot(
                frame_number=packet.frame_number,
                host_timestamp_ms=packet.host_timestamp_ms,
                detected=dict(detected),
                result1=result1,
                result2=result2,
            )

            with self._lock:
                previous = self._last_processed_frame_number
                if previous is not None:
                    gap = packet.frame_number - previous
                    if gap > 1:
                        self._processed_skipped += gap - 1

                self._last_processed_frame_number = packet.frame_number
                self._processed_count += 1
                self._processing_ms_sum += processing_ms
                self._processing_ms_count += 1
                self._processing_ms_last = processing_ms
                self._latest = processed
                self._history.append(snapshot)

    def latest(self):
        with self._lock:
            return self._latest

    def history_snapshot(self):
        with self._lock:
            return list(self._history)

    def stats_snapshot(self):
        with self._lock:
            mean_ms = (
                self._processing_ms_sum / self._processing_ms_count
                if self._processing_ms_count > 0
                else 0.0
            )
            return {
                "processed_count": int(self._processed_count),
                "processed_skipped": int(self._processed_skipped),
                "last_processed_frame_number": self._last_processed_frame_number,
                "processing_ms_mean": float(mean_ms),
                "processing_ms_last": float(self._processing_ms_last),
            }

    def reset(self):
        with self._tracker_lock:
            self.tracker1.reset()
            self.tracker2.reset()

        with self._lock:
            self._latest = None
            self._history.clear()
            # Keep performance counters cumulative; only tracking state/history
            # is reset here.

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        self._thread = None


def find_nearest_timestamp_pair(main_history, side_history):
    """
    Find the closest acquisition-time pair among recent processed results.

    Used only for MAIN-vs-SIDE agreement diagnostics.  The final per-stick
    selector still uses each camera's latest fresh result independently.
    """
    if not main_history or not side_history:
        return None, None, None

    best_main = None
    best_side = None
    best_dt_ms = None

    for m in main_history:
        for s in side_history:
            dt_ms = abs(m.host_timestamp_ms - s.host_timestamp_ms)
            if best_dt_ms is None or dt_ms < best_dt_ms:
                best_dt_ms = dt_ms
                best_main = m
                best_side = s

    return best_main, best_side, best_dt_ms


def _compute_hand_pair_from_processed(
    processed,
    s1,
    s2,
    T_HAND_BASE,
    previous_hand_quaternion,
    prefix,
):
    if processed is None:
        return None, None

    hand1 = compute_hand_relative_pose(
        processed.result1,
        s1,
        T_HAND_BASE,
        previous_hand_quaternion[f"{prefix}_STICK1"],
    )
    hand2 = compute_hand_relative_pose(
        processed.result2,
        s2,
        T_HAND_BASE,
        previous_hand_quaternion[f"{prefix}_STICK2"],
    )

    if hand1["available"]:
        previous_hand_quaternion[f"{prefix}_STICK1"] = hand1["quaternion"].copy()
    if hand2["available"]:
        previous_hand_quaternion[f"{prefix}_STICK2"] = hand2["quaternion"].copy()

    return hand1, hand2


def _snapshot_to_hand_pair(snapshot, s1, s2, T_HAND_BASE):
    if snapshot is None:
        return None, None
    hand1 = compute_hand_relative_pose(snapshot.result1, s1, T_HAND_BASE, None)
    hand2 = compute_hand_relative_pose(snapshot.result2, s2, T_HAND_BASE, None)
    return hand1, hand2


# ============================================================
# MAIN -- two parallel vision processing threads
# ============================================================

def main():
    np.set_printoptions(precision=6, suppress=True)

    # --------------------------------------------------------
    # Tracker modules
    # --------------------------------------------------------
    stick2_module_path = resolve_stick2_module_path()

    s1 = load_module("stick1_tracker_module", STICK1_MODULE_PATH)
    s2 = load_module("stick2_tracker_module", stick2_module_path)

    for attr in ("WIDTH", "HEIGHT", "FPS", "MARKER_SIZE_M"):
        if getattr(s1, attr) != getattr(s2, attr):
            raise RuntimeError(
                f"Stick1/Stick2 config mismatch for {attr}: "
                f"{getattr(s1, attr)} vs {getattr(s2, attr)}"
            )

    width = int(s1.WIDTH)
    height = int(s1.HEIGHT)
    fps = int(s1.FPS)

    T_CAMERA_BASE_MAIN = invert_transform(T_BASE_CAMERA_MAIN)
    T_CAMERA_BASE_SIDE = invert_transform(T_BASE_CAMERA_SIDE)

    # Four independent tracker histories.  Each pair is touched only by its
    # own camera processing worker.
    main_tracker1 = StickTrackerState(s1, "STICK1_MAIN", use_workspace_prior=True)
    main_tracker2 = StickTrackerState(s2, "STICK2_MAIN", use_workspace_prior=True)
    side_tracker1 = StickTrackerState(s1, "STICK1_SIDE", use_workspace_prior=False)
    side_tracker2 = StickTrackerState(s2, "STICK2_SIDE", use_workspace_prior=False)

    target_ids = set(main_tracker1.marker_ids) | set(main_tracker2.marker_ids)
    if target_ids != {0, 1, 2, 3}:
        raise RuntimeError(f"Unexpected marker IDs: {sorted(target_ids)}")

    # --------------------------------------------------------
    # Camera acquisition threads
    # --------------------------------------------------------
    main_cam = CameraStream(
        "MAIN", MAIN_CAMERA_SERIAL, width, height, fps
    )
    side_cam = CameraStream(
        "SIDE", SIDE_CAMERA_SERIAL, width, height, fps
    )

    # Processing workers are constructed now but started only AFTER the camera
    # streams have started, so K/dist are already available.
    main_worker = CameraProcessingWorker(
        "MAIN",
        main_cam,
        main_tracker1,
        main_tracker2,
        s1,
        s2,
        target_ids,
        T_BASE_CAMERA_MAIN,
        T_CAMERA_BASE_MAIN,
    )
    side_worker = CameraProcessingWorker(
        "SIDE",
        side_cam,
        side_tracker1,
        side_tracker2,
        s1,
        s2,
        target_ids,
        T_BASE_CAMERA_SIDE,
        T_CAMERA_BASE_SIDE,
    )

    q6_deg = float(Q6_INITIAL_DEG)

    previous_hand_quaternion = {
        "MAIN_STICK1": None,
        "MAIN_STICK2": None,
        "SIDE_STICK1": None,
        "SIDE_STICK2": None,
    }
    previous_final_quaternion = {
        "STICK1": None,
        "STICK2": None,
    }

    last_consumed_frame = {"MAIN": None, "SIDE": None}

    detected_main = {}
    detected_side = {}
    result_main1 = None
    result_main2 = None
    result_side1 = None
    result_side2 = None
    hand_main1 = None
    hand_main2 = None
    hand_side1 = None
    hand_side2 = None
    main_ts_ms = None
    side_ts_ms = None
    main_vis = None
    side_vis = None
    final1 = None
    final2 = None

    latest_perf = {
        "MAIN": {"capture_fps": 0.0, "processed_fps": 0.0},
        "SIDE": {"capture_fps": 0.0, "processed_fps": 0.0},
    }

    last_print = 0.0
    perf_last_time = None
    perf_prev_capture_count = {"MAIN": 0, "SIDE": 0}
    perf_prev_capture_skipped = {"MAIN": 0, "SIDE": 0}
    perf_prev_processed_count = {"MAIN": 0, "SIDE": 0}
    perf_prev_processed_skipped = {"MAIN": 0, "SIDE": 0}

    print()
    print("=" * 96)
    print("TWO-CAMERA / TWO-STICK / PARALLEL 30-HZ VISION")
    print("=" * 96)
    print(f"MAIN serial : {MAIN_CAMERA_SERIAL}")
    print(f"SIDE serial : {SIDE_CAMERA_SERIAL}")
    print(f"RGB         : {width}x{height}@{fps}")
    print("Processing  : MAIN and SIDE each have their own vision thread")
    print("Sync clock  : host time.monotonic_ns() stamped at acquisition")
    print("Sync pair   : nearest acquisition timestamps from recent history")
    print(f"Perf report : every {PERF_INTERVAL_SEC:.1f} s")
    print("MAIN prior  : ENABLED")
    print("SIDE prior  : DISABLED")
    print("Fallback    : per stick; SIDE only if MAIN misses both markers")
    print(f"Initial q6  : {q6_deg:.6f} deg")
    print(f"Mount yaw   : {HAND_MOUNT_YAW_OFFSET_DEG:.3f} deg")
    print()
    print("Keys")
    print("  q : quit")
    print("  x : reset all 4 tracker histories")
    print("  [ : q6 -1 deg")
    print("  ] : q6 +1 deg")
    print("  p : print current transforms")
    print("  m : print marker-level MAIN vs SIDE BASE poses")
    print("=" * 96)

    try:
        main_cam.start()
        side_cam.start()
        main_worker.start()
        side_worker.start()
        print("[CAMERA] MAIN + SIDE acquisition started.")
        print("[VISION] MAIN + SIDE processing threads started.")

        perf_last_time = time.monotonic()
        for camera_name, cam in (("MAIN", main_cam), ("SIDE", side_cam)):
            stats = cam.stats_snapshot()
            perf_prev_capture_count[camera_name] = stats["capture_count"]
            perf_prev_capture_skipped[camera_name] = stats["capture_skipped"]

        GUI_FPS = 30.0
        GUI_PERIOD_SEC = 1.0 / GUI_FPS
        last_gui_time = 0.0
        
        while True:
            T_BASE_HAND = get_T_BASE_HAND(q6_deg)
            T_HAND_BASE = invert_transform(T_BASE_HAND)

            latest_main = main_worker.latest()
            latest_side = side_worker.latest()
            any_new = False

            # -------------------------------------------------
            # Consume latest MAIN result (already processed in MAIN thread)
            # -------------------------------------------------
            if (
                latest_main is not None
                and latest_main.frame_number != last_consumed_frame["MAIN"]
            ):
                last_consumed_frame["MAIN"] = latest_main.frame_number
                any_new = True
                detected_main = latest_main.detected
                result_main1 = latest_main.result1
                result_main2 = latest_main.result2
                main_ts_ms = latest_main.host_timestamp_ms
                main_vis = latest_main.vis.copy()
                hand_main1, hand_main2 = _compute_hand_pair_from_processed(
                    latest_main,
                    s1,
                    s2,
                    T_HAND_BASE,
                    previous_hand_quaternion,
                    "MAIN",
                )

            # -------------------------------------------------
            # Consume latest SIDE result (already processed in SIDE thread)
            # -------------------------------------------------
            if (
                latest_side is not None
                and latest_side.frame_number != last_consumed_frame["SIDE"]
            ):
                last_consumed_frame["SIDE"] = latest_side.frame_number
                any_new = True
                detected_side = latest_side.detected
                result_side1 = latest_side.result1
                result_side2 = latest_side.result2
                side_ts_ms = latest_side.host_timestamp_ms
                side_vis = latest_side.vis.copy()
                hand_side1, hand_side2 = _compute_hand_pair_from_processed(
                    latest_side,
                    s1,
                    s2,
                    T_HAND_BASE,
                    previous_hand_quaternion,
                    "SIDE",
                )

            now_ms = time.monotonic_ns() * 1e-6

            # -------------------------------------------------
            # Final per-stick selector uses latest fresh results.
            # It does NOT wait for a synchronized camera pair.
            # -------------------------------------------------
            if any_new:
                final1 = select_final_pose_for_stick(
                    "STICK1",
                    main_tracker1.marker_ids,
                    detected_main,
                    result_main1,
                    hand_main1,
                    main_ts_ms,
                    result_side1,
                    hand_side1,
                    side_ts_ms,
                    now_ms,
                    previous_final_quaternion["STICK1"],
                )
                final2 = select_final_pose_for_stick(
                    "STICK2",
                    main_tracker2.marker_ids,
                    detected_main,
                    result_main2,
                    hand_main2,
                    main_ts_ms,
                    result_side2,
                    hand_side2,
                    side_ts_ms,
                    now_ms,
                    previous_final_quaternion["STICK2"],
                )

                if final1["pose"] is not None:
                    previous_final_quaternion["STICK1"] = (
                        final1["pose"]["quaternion"].copy()
                    )
                if final2["pose"] is not None:
                    previous_final_quaternion["STICK2"] = (
                        final2["pose"]["quaternion"].copy()
                    )

            # -------------------------------------------------
            # Software sync diagnostics: nearest recent acquisition pair.
            # This pairing is ONLY for MAIN-vs-SIDE agreement measurement.
            # -------------------------------------------------
            pair_main, pair_side, pair_dt_ms = find_nearest_timestamp_pair(
                main_worker.history_snapshot(),
                side_worker.history_snapshot(),
            )

            diff1 = None
            diff2 = None
            if pair_main is not None and pair_side is not None:
                paired_main_h1, paired_main_h2 = _snapshot_to_hand_pair(
                    pair_main, s1, s2, T_HAND_BASE
                )
                paired_side_h1, paired_side_h2 = _snapshot_to_hand_pair(
                    pair_side, s1, s2, T_HAND_BASE
                )
                diff1 = pose_difference(paired_main_h1, paired_side_h1, s1)
                diff2 = pose_difference(paired_main_h2, paired_side_h2, s2)

            latest_dt_ms = None
            if main_ts_ms is not None and side_ts_ms is not None:
                latest_dt_ms = abs(main_ts_ms - side_ts_ms)

            # -------------------------------------------------
            # Performance report
            # -------------------------------------------------
            perf_now = time.monotonic()
            perf_elapsed = perf_now - perf_last_time

            if perf_elapsed >= PERF_INTERVAL_SEC:
                print()
                print("-" * 96)
                print(
                    f"PERFORMANCE over {perf_elapsed:.2f} s | requested={fps} Hz"
                )

                for camera_name, cam, worker in (
                    ("MAIN", main_cam, main_worker),
                    ("SIDE", side_cam, side_worker),
                ):
                    cam_stats = cam.stats_snapshot()
                    worker_stats = worker.stats_snapshot()

                    capture_delta = (
                        cam_stats["capture_count"]
                        - perf_prev_capture_count[camera_name]
                    )
                    capture_skip_delta = (
                        cam_stats["capture_skipped"]
                        - perf_prev_capture_skipped[camera_name]
                    )
                    processed_delta = (
                        worker_stats["processed_count"]
                        - perf_prev_processed_count[camera_name]
                    )
                    processed_skip_delta = (
                        worker_stats["processed_skipped"]
                        - perf_prev_processed_skipped[camera_name]
                    )

                    capture_fps = capture_delta / perf_elapsed
                    processed_fps = processed_delta / perf_elapsed
                    latest_perf[camera_name]["capture_fps"] = capture_fps
                    latest_perf[camera_name]["processed_fps"] = processed_fps

                    print(
                        f"{camera_name:4s} capture   : {capture_fps:6.2f} Hz "
                        f"| device/frame gaps={capture_skip_delta:4d} "
                        f"| total={cam_stats['capture_count']}"
                    )
                    print(
                        f"{camera_name:4s} processed : {processed_fps:6.2f} Hz "
                        f"| skipped-by-processing={processed_skip_delta:4d} "
                        f"| proc_mean={worker_stats['processing_ms_mean']:5.1f} ms "
                        f"| proc_last={worker_stats['processing_ms_last']:5.1f} ms "
                        f"| total={worker_stats['processed_count']}"
                    )

                    perf_prev_capture_count[camera_name] = cam_stats["capture_count"]
                    perf_prev_capture_skipped[camera_name] = cam_stats["capture_skipped"]
                    perf_prev_processed_count[camera_name] = worker_stats["processed_count"]
                    perf_prev_processed_skipped[camera_name] = worker_stats["processed_skipped"]

                if pair_dt_ms is None:
                    print("SYNC nearest pair : unavailable")
                else:
                    print(f"SYNC nearest pair : |dt|={pair_dt_ms:.3f} ms")
                print("-" * 96)
                perf_last_time = perf_now

            # -------------------------------------------------
            # GUI overlays
            # -------------------------------------------------
            if main_vis is not None:
                cv2.putText(
                    main_vis,
                    f"MAIN IDs={sorted(detected_main.keys())} q6={q6_deg:+.3f}",
                    (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    main_vis,
                    f"cap={latest_perf['MAIN']['capture_fps']:.1f}Hz proc={latest_perf['MAIN']['processed_fps']:.1f}Hz",
                    (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    main_vis,
                    f"S1 {format_hand_pose(hand_main1)}",
                    (20, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    main_vis,
                    f"S2 {format_hand_pose(hand_main2)}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if side_vis is not None:
                cv2.putText(
                    side_vis,
                    f"SIDE IDs={sorted(detected_side.keys())} q6={q6_deg:+.3f}",
                    (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    side_vis,
                    f"cap={latest_perf['SIDE']['capture_fps']:.1f}Hz proc={latest_perf['SIDE']['processed_fps']:.1f}Hz",
                    (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    side_vis,
                    f"S1 {format_hand_pose(hand_side1)}",
                    (20, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    side_vis,
                    f"S2 {format_hand_pose(hand_side2)}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                gui_now = time.monotonic()

                if gui_now - last_gui_time >= GUI_PERIOD_SEC:
                    if main_vis is not None:
                        cv2.imshow("MAIN D435", main_vis)

                    if side_vis is not None:
                        cv2.imshow("SIDE D435", side_vis)

                    last_gui_time = gui_now

            # -------------------------------------------------
            # Terminal state report
            # -------------------------------------------------
            now_sec = time.monotonic()
            if now_sec - last_print >= PRINT_INTERVAL_SEC:
                print()
                print("=" * 96)
                print(f"TWO CAMERA LIVE | q6={q6_deg:+.6f} deg")
                print(
                    "MAIN IDs:", sorted(detected_main.keys()),
                    "| SIDE IDs:", sorted(detected_side.keys()),
                )

                if latest_dt_ms is None:
                    print("Latest timestamp |dt|: unavailable")
                else:
                    print(f"Latest timestamp |dt| = {latest_dt_ms:.3f} ms")

                if pair_dt_ms is None:
                    print("Nearest sync pair |dt|: unavailable")
                else:
                    sync_label = (
                        "SYNC_CLOSE"
                        if pair_dt_ms <= SYNC_COMPARE_GOOD_MS
                        else "ASYNC"
                    )
                    print(
                        f"Nearest sync pair |dt| = {pair_dt_ms:.3f} ms [{sync_label}]"
                    )

                print()
                print("STICK1")
                print("  MAIN BASE :", format_base_pose(hand_main1))
                print("  MAIN HAND :", format_hand_pose(hand_main1))
                print("  SIDE BASE :", format_base_pose(hand_side1))
                print("  SIDE HAND :", format_hand_pose(hand_side1))
                if diff1 is not None:
                    print(f"  M/S DIFF  : dP={diff1[0]:.3f} mm dR={diff1[1]:.3f} deg")
                else:
                    print("  M/S DIFF  : unavailable")
                if final1 is None:
                    print("  FINAL     : not initialized")
                else:
                    print(f"  FINAL     : {final1['source']} | {final1['reason']}")
                    print(
                        "             ",
                        "NO FINAL POSE" if final1["pose"] is None else format_hand_pose(final1["pose"]),
                    )

                print()
                print("STICK2")
                print("  MAIN BASE :", format_base_pose(hand_main2))
                print("  MAIN HAND :", format_hand_pose(hand_main2))
                print("  SIDE BASE :", format_base_pose(hand_side2))
                print("  SIDE HAND :", format_hand_pose(hand_side2))
                if diff2 is not None:
                    print(f"  M/S DIFF  : dP={diff2[0]:.3f} mm dR={diff2[1]:.3f} deg")
                else:
                    print("  M/S DIFF  : unavailable")
                if final2 is None:
                    print("  FINAL     : not initialized")
                else:
                    print(f"  FINAL     : {final2['source']} | {final2['reason']}")
                    print(
                        "             ",
                        "NO FINAL POSE" if final2["pose"] is None else format_hand_pose(final2["pose"]),
                    )
                print("=" * 96)
                last_print = now_sec

            # -------------------------------------------------
            # Keyboard
            # -------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("x"):
                main_worker.reset()
                side_worker.reset()
                last_consumed_frame = {"MAIN": None, "SIDE": None}
                detected_main = {}
                detected_side = {}
                result_main1 = result_main2 = None
                result_side1 = result_side2 = None
                hand_main1 = hand_main2 = None
                hand_side1 = hand_side2 = None
                main_ts_ms = side_ts_ms = None
                for k in previous_hand_quaternion:
                    previous_hand_quaternion[k] = None
                previous_final_quaternion["STICK1"] = None
                previous_final_quaternion["STICK2"] = None
                print("[RESET] MAIN/SIDE tracker histories cleared.")
            elif key == ord("["):
                q6_deg -= Q6_STEP_DEG
                print(f"[q6] {q6_deg:+.6f} deg")
            elif key == ord("]"):
                q6_deg += Q6_STEP_DEG
                print(f"[q6] {q6_deg:+.6f} deg")
            elif key == ord("p"):
                T_BASE_HAND = get_T_BASE_HAND(q6_deg)
                T_HAND_BASE = invert_transform(T_BASE_HAND)
                print()
                print("=" * 96)
                print("CURRENT FIXED / FK TRANSFORMS")
                print("=" * 96)
                print("\nT_BASE_CAMERA_MAIN =")
                print(T_BASE_CAMERA_MAIN)
                print("\nT_BASE_CAMERA_SIDE =")
                print(T_BASE_CAMERA_SIDE)
                print("\nT_BASE_J6 =")
                print(T_BASE_J6)
                print("\nT_LINK6_HAND =")
                print(T_LINK6_HAND)
                print(f"\nq6 = {q6_deg:+.6f} deg")
                print("\nT_BASE_HAND =")
                print(T_BASE_HAND)
                print("\nT_HAND_BASE =")
                print(T_HAND_BASE)
                if final1 is not None and final1["pose"] is not None:
                    print("\nFINAL T_HAND_STICK1 =")
                    print(final1["pose"]["T_HAND_STICK"])
                else:
                    print("\nFINAL T_HAND_STICK1 = NO POSE")
                if final2 is not None and final2["pose"] is not None:
                    print("\nFINAL T_HAND_STICK2 =")
                    print(final2["pose"]["T_HAND_STICK"])
                else:
                    print("\nFINAL T_HAND_STICK2 = NO POSE")
                print("=" * 96)
            elif key == ord("m"):
                if (
                    result_main1 is None
                    or result_main2 is None
                    or result_side1 is None
                    or result_side2 is None
                ):
                    print("[MARKER SNAPSHOT] Need at least one processed frame from both cameras.")
                else:
                    print_marker_snapshot(
                        [result_main1, result_main2],
                        [result_side1, result_side2],
                        [s1, s2],
                    )

            # Do not busy-spin the GUI/main selector thread.
            time.sleep(0.0005)

    finally:
        # Stop processors first so they no longer ask cameras for new frames.
        main_worker.stop()
        side_worker.stop()
        main_cam.stop()
        side_cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
