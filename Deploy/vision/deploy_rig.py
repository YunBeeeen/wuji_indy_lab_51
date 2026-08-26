# [vision] 실험실 실측 리그 — D435 2대 외부파라미터, Indy7 J6에 달린 palm 프레임. 값마다 측정/후보 출처 표시.
"""The physical installation, with each number's provenance attached.

Not to be confused with ``scene_contract.py``.  That file describes the
SIMULATED scene: one rendered camera and an invented marker layout, which exist
so the ArUco detection and PnP maths can be validated inside MuJoCo against a
pose that is known exactly.  It is a test fixture and its numbers are labelled
CANDIDATE in its own comments.

This file describes the rig that actually exists in the lab: two calibrated
D435s, the Indy7 that carries the hand, and the mounting between them.  The
numbers are transcribed from ``/home/lsc/Vision`` -- the tracker that produced
and uses them -- rather than re-derived, so there is one place they can drift
and it is visible.

Only the SUM of the two yaws exists
-----------------------------------
``HAND_MOUNT_YAW_OFFSET_DEG`` and q6 are rotations about the SAME axis, and the
mount translation runs along that axis too, so::

    Rz(q6) @ Rz(mount_yaw) == Rz(q6 + mount_yaw)      (agree to 4.6e-16)

155/25 and 100/80 produce identical geometry.  The tracker's note about
verifying the mount yaw separately therefore asks for something no measurement
can deliver: what is identifiable is ``TOTAL_YAW_DEG``, and the split is
bookkeeping.  Keep the split only so the number matches the tracker's source.

What the sum is verified against
--------------------------------
At q6 = 25.000097 deg the operator observes the palm facing the sky, and that
IS a constraint on the sum: the rotation axis (hand +Z) lies nearly horizontal
in Base, so turning about it swings the palm normal strongly -- 42.8 deg off
vertical at a sum of 140, 3.7 deg at 180, 37.4 deg at 220.

Two frame facts make that check meaningful, both established by geometry rather
than by comment:

* ``palm_link`` +X is the palm plate normal.  The plate's bounding box is
  31.9 x 81.1 x 104.8 mm -- X is the thin axis -- and 99.77 % of the mesh's
  area-weighted facet normal lies on X.
* **Base +Z is up here.**  ``scene_contract`` documents a DIFFERENT frame also
  called Base, in which +X is up; that one belongs to the simulated MuJoCo
  scene.  Mixing them makes the palm look 87.5 deg away from vertical instead
  of 3.7, which is how this was first misread.

Residual
--------
The sum that maximises palm-up is 182.724 deg, not 180.000.  Using 180 leaves
the palm normal 3.73 deg off vertical, which displaces a stick 100 mm from the
palm origin by 6.5 mm.  The eye cannot separate these: every sum from 178.4 to
187.0 deg looks "within 5 deg of level".  So the observation confirms the frame
is not grossly wrong; it does not calibrate it, and the residual could equally
be an error in ``T_BASE_J6`` (i.e. in the fixed q1..q5).  If tracked stick poses
look displaced by roughly half a centimetre, start here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MEASURED = "measured"      # calibrated or physically measured
COARSE = "coarse"          # constrained by observation, but only loosely
CANDIDATE = "candidate"    # typed in; no measurement behind it
BOOKKEEPING = "bookkeeping"  # not separately identifiable; carried for readability


@dataclass(frozen=True)
class Provenance:
    status: str
    note: str


PROVENANCE: dict[str, Provenance] = {
    "T_BASE_CAMERA_MAIN": Provenance(
        MEASURED, "D435 814412070582, calibrated against fixed base markers"
    ),
    "T_BASE_CAMERA_SIDE": Provenance(
        MEASURED, "D435 342222074358, lowest-RMS calibration run (0.950 mm)"
    ),
    "T_BASE_J6": Provenance(
        MEASURED,
        "Base<-J6 with q1..q5 fixed at "
        "[10.044318, -64.03332, -131.97517, 9.914346, 103.22368] deg",
    ),
    "HAND_OFFSET_Z_M": Provenance(
        MEASURED, "60 mm final robot section + 17 mm bracket + 30 mm mount"
    ),
    "HAND_MOUNT_YAW_OFFSET_DEG": Provenance(
        BOOKKEEPING,
        "not separately identifiable: only q6 + mount yaw exists. Kept at the "
        "tracker's value so the two sources read alike.",
    ),
    "TOTAL_YAW_DEG": Provenance(
        COARSE,
        "confirmed by palm-up observation at q6=25.000097, which pins it only "
        "to about 178.4-187.0 deg. The palm-up optimum is 182.724, so the "
        "value in use leaves 3.73 deg (6.5 mm at 100 mm) on the table.",
    ),
    "q6_deg": Provenance(
        CANDIDATE,
        "typed in and nudged by keyboard in the tracker, not read from the "
        "arm. The palm frame turns with it, so every stick observation does too.",
    ),
}

#: The identifiable quantity.  Recomputed from the parts so the two cannot drift.
NOMINAL_Q6_DEG = 25.000097
TOTAL_YAW_DEG = NOMINAL_Q6_DEG + 155.0
#: Sum that would put the palm normal closest to vertical, from a 70001-point
#: sweep against the palm_link +X plate normal.  Not adopted: see the module
#: docstring -- the residual may belong to T_BASE_J6 instead.
PALM_UP_OPTIMUM_TOTAL_YAW_DEG = 182.724
PALM_UP_RESIDUAL_DEG = 3.73


# --------------------------------------------------------------------------- #
# Cameras.  T_A_B maps coordinates expressed in B into A.
# --------------------------------------------------------------------------- #
T_BASE_CAMERA_MAIN = np.asarray(
    [[0.011009927, 0.713786390, -0.700276924, 0.937431906],
     [0.999937040, -0.009377283, 0.006163071, -0.111095107],
     [-0.002167578, -0.700300689, -0.713844693, 0.435281684],
     [0.0, 0.0, 0.0, 1.0]], dtype=np.float64,
)
T_BASE_CAMERA_SIDE = np.asarray(
    [[0.999989882, -0.003139647, 0.003221657, 0.654231507],
     [-0.003071812, 0.046616270, 0.998908148, -0.372806389],
     [-0.003286401, -0.998907937, 0.046606154, 0.153432702],
     [0.0, 0.0, 0.0, 1.0]], dtype=np.float64,
)
CAMERA_SERIALS = {"main": "814412070582", "side": "342222074358"}
T_BASE_CAMERA = {"main": T_BASE_CAMERA_MAIN, "side": T_BASE_CAMERA_SIDE}

#: The tracker's per-camera freshness gate.
CAMERA_STALE_MS = 150.0


# --------------------------------------------------------------------------- #
# Indy7 -> Wuji Hand.  The palm frame rides on joint 6.
# --------------------------------------------------------------------------- #
T_BASE_J6 = np.asarray(
    [[-0.044204390, -0.008842933, 0.998983370, 0.473385519],
     [-0.047828501, 0.998832922, 0.006725220, -0.131376150],
     [-0.997876950, -0.047482593, -0.044575744, 0.161305068],
     [0.0, 0.0, 0.0, 1.0]], dtype=np.float64,
)
HAND_OFFSET_Z_M = 0.107
HAND_MOUNT_YAW_OFFSET_DEG = 155.0


def rotation_z(angle_rad: float) -> np.ndarray:
    cosine, sine = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.asarray(
        [[cosine, -sine, 0.0, 0.0],
         [sine, cosine, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float64,
    )


def invert(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def t_link6_hand() -> np.ndarray:
    """link6 -> hand.  link6 +Z and Wuji Hand +Z are physically shared."""

    transform = rotation_z(np.deg2rad(HAND_MOUNT_YAW_OFFSET_DEG))
    transform[:3, 3] = [0.0, 0.0, HAND_OFFSET_Z_M]
    return transform


def t_base_hand(q6_deg: float) -> np.ndarray:
    """Return Base<-Hand for a given Indy7 joint-6 angle.

    ``q6_deg`` is required and has no default on purpose.  The tracker keeps a
    typed-in starting value; making a guess importable here is how it would
    quietly become "the" hand pose.  Pass the angle the arm is actually at.
    """

    if q6_deg is None or not np.isfinite(float(q6_deg)):
        raise ValueError(
            "q6_deg must be the Indy7 joint-6 angle in degrees. There is no "
            "default: the palm frame rotates with it, and so does every stick "
            "observation the policy sees."
        )
    return T_BASE_J6 @ rotation_z(np.deg2rad(float(q6_deg))) @ t_link6_hand()


def t_hand_camera(camera: str, q6_deg: float) -> np.ndarray:
    """Return Hand<-Camera, the transform a palm-frame stick pose needs."""

    if camera not in T_BASE_CAMERA:
        raise ValueError(f"camera must be one of {sorted(T_BASE_CAMERA)}, got {camera!r}.")
    return invert(t_base_hand(q6_deg)) @ T_BASE_CAMERA[camera]


def unverified_inputs() -> tuple[str, ...]:
    """Names whose values are candidates rather than measurements."""

    return tuple(
        name for name, item in PROVENANCE.items()
        if item.status in (CANDIDATE, COARSE)
    )


def assert_deployable(acknowledge_candidates: bool = False) -> None:
    """Report unverified rig geometry.  Warns; does not refuse.

    Until 2026-08-23 this raised unless the caller passed
    ``acknowledge_candidates``.  The intent was to keep two un-measured
    rotations visible, but a refusal that every single run has to opt out of
    stops being a signal -- the flag just becomes part of the command line and
    nobody reads the reason again.  Printing it every run keeps the information
    where it is actually seen, and costs no keystrokes.

    Nothing about the geometry changes either way: the same TOTAL_YAW_DEG and
    q6 are used whether this warns or is silenced.  ``acknowledge_candidates``
    now only suppresses the message, and is kept so existing commands still
    parse.
    """

    pending = unverified_inputs()
    if pending and not acknowledge_candidates:
        detail = "\n".join(f"           ! {name}: {PROVENANCE[name].note}"
                            for name in pending)
        print("[RIG]      미검증 기하 위에서 동작합니다 "
              "(--acknowledge-candidate-geometry 로 이 경고를 끕니다):\n"
              f"{detail}\n"
              "           둘 다 palm 프레임을 돌리므로 스틱 관측 전체가 함께 "
              "움직입니다. 스틱이 수 mm 어긋나 보이면 여기부터 볼 것.")


def provenance_report() -> str:
    lines = ["[DEPLOY RIG]"]
    for name, item in PROVENANCE.items():
        marker = {MEASURED: "+", BOOKKEEPING: "=", COARSE: "~", CANDIDATE: "!"}[item.status]
        lines.append(f"  {marker} {name:26s} {item.status:9s} {item.note}")
    return "\n".join(lines)


for _array in (T_BASE_CAMERA_MAIN, T_BASE_CAMERA_SIDE, T_BASE_J6):
    _array.setflags(write=False)
