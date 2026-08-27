# [common] 20관절 정책 계약 — 관절 이름·순서, 세 한계 테이블, obs 슬라이스, 액션 스케일, kp/kd/effort.
"""모든 백엔드가 공유하는 Wuji Hand 정책 계약.
관절 순서·한계·관측 슬라이스·액션 스케일 고정."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


WUJI_DESCRIPTION_REVISION = "06e5f14cdd1d5fad0a666ca463a668bf609f9534"
WUJI_DESCRIPTION_RELEASE = "v2026.8.14"
PALM_FRAME_NAME = "right_palm_link"
STICK_REPRESENTATION = "StickPose7D: palm xyz + quaternion wxyz"

# --- 하드웨어 공통값: task와 무관한 손 자체의 정보 -----------------------
# 손 관절 수와 정책 출력 차원은 별도 상수로 관리.
HAND_JOINT_COUNT = 20

# --- 젓가락 파지 정책 계약: hand_real / hand_final ------------------------
# 아래 action·observation 값은 해당 정책에만 적용.
ACTION_DIM = 20
ACTION_CLIP = np.float32(1.0)
JOINT4_POLICY_INDICES = np.asarray([3, 7, 11, 15, 19], dtype=np.int32)

POLICY_DT = 1.0 / 30.0


@dataclass(frozen=True)
class CanonicalJoint:
    """Backend-independent policy joint identity."""

    policy_index: int
    canonical_name: str


CANONICAL_JOINTS: tuple[CanonicalJoint, ...] = tuple(
    CanonicalJoint(
        policy_index=4 * (finger - 1) + (joint - 1),
        canonical_name=f"finger{finger}_joint{joint}",
    )
    for finger in range(1, 6)
    for joint in range(1, 5)
)
POLICY_JOINT_NAMES = tuple(joint.canonical_name for joint in CANONICAL_JOINTS)


# Isaac ``HAND_REAL_ACTION_SCALE``과 같은 관절별 잔차 스케일.
# ``hand_move``의 전 관절 0.1 rad 계약과 혼용하지 않는다.
_ACTION_SCALE_BY_JOINT_SUFFIX = {
    "joint1": 0.1,
    "joint2": 0.1,
    "joint3": 0.2,
    "joint4": 0.15,
}
ACTION_SCALE_RAD = np.asarray(
    [
        _ACTION_SCALE_BY_JOINT_SUFFIX[name.split("_", 1)[1]]
        for name in POLICY_JOINT_NAMES
    ],
    dtype=np.float32,
)
ACTION_SCALE_RAD.setflags(write=False)


# A. Nominal mechanism/articulation range from the pinned official Hand 1 URDF.
# This is explicitly not the connected hand's factory-calibrated range.
OFFICIAL_NOMINAL_PHYSICAL_LIMITS = np.asarray(
    [
        (0.0475, 1.6033), (-0.1387, 0.9324), (-0.4642, 1.5623), (-0.4699, 1.5568),
        (-0.1585, 1.5604), (-0.3700, 0.3700), (-0.4777, 1.5485), (-0.4683, 1.5753),
        (-0.1644, 1.5516), (-0.3700, 0.3700), (-0.4739, 1.5512), (-0.4684, 1.5745),
        (-0.1554, 1.5585), (-0.3700, 0.3700), (-0.4765, 1.5487), (-0.4777, 1.5634),
        (-0.1626, 1.5585), (-0.3700, 0.3700), (-0.4768, 1.5490), (-0.4683, 1.5735),
    ],
    dtype=np.float32,
)

# A2. The CONNECTED hand's own factory limits, read from the device on
# 2026-08-18 (SDK v1.7.0, firmware 1.2.1).  The vendor description above is
# narrower on all 20 of 20 joints, so it cannot serve as the observation
# contract: normalization is the affine map 2*(q-center)/(upper-lower), and a
# real encoder near its true limit normalizes to |q_norm| > 1 against the
# narrower table -- an input no policy ever saw in training.  Worst measured
# deviation: finger3_joint2, 8% of the full -1..+1 span.
#
# This is THIS unit's calibration frozen into the contract, NOT a model
# constant.  It is deliberately a hard-coded table rather than a runtime SDK
# read: a policy input must not change because a different hand was plugged in.
# Compare a second hand with validate_factory_limits(), which reports without
# adopting.  Isaac carries the same numbers via the USD override in
# assets/model/urdf/wuji_right/wuji_right/wuji_right_filtered.usda.
REAL_HAND_FACTORY_LIMITS = np.asarray(
    [
        (-0.08703218, 1.69273507), (-0.22442448, 0.99236929),
        (-0.54381580, 1.67827984), (-0.53936029, 1.67382433),
        (-0.29236769, 1.60136471), (-0.42065594, 0.42065594),
        (-0.54539557, 1.67985970), (-0.54855444, 1.68301829),
        (-0.29632606, 1.60532307), (-0.41939202, 0.41939202),
        (-0.54558273, 1.68004658), (-0.54936740, 1.68383135),
        (-0.29673620, 1.60573315), (-0.40010168, 0.40010173),
        (-0.54702421, 1.68148825), (-0.55082101, 1.68528500),
        (-0.29750600, 1.60650295), (-0.40155294, 0.40155294),
        (-0.54067725, 1.67514129), (-0.54903634, 1.68350029),
    ],
    dtype=np.float32,
)

# B. Fixed policy preprocessing range.  A separate allocation from the tables it
# is derived from, and never replaced from a runtime hardware read.
OBSERVATION_NORMALIZATION_LIMITS = REAL_HAND_FACTORY_LIMITS.copy()

# C. Policy-command range.  Identical to the observation range as of
# 2026-08-18: the five Joint4 floors at 0 rad were lifted on the Isaac side
# (``hand_move_env_cfg.py``, ``joint_position_lower_overrides=None``) and this
# table has to mirror the action space the policy was trained in.
#
# The floors were cosmetic -- distal hyperextension looks unnatural for a human
# hand -- but play traces showed ``finger1_joint4`` requesting a negative
# residual on every sampled step while clamped to exactly 0.000, producing
# 0.000 N.m from a joint that carries one of the six functional contacts.
# Keeping the floor here and not in Isaac would re-freeze that joint on the
# real hand only.
#
# 2026-08-22: scaled by COMMAND_LIMIT_RATIO on the user's instruction, to keep
# the hand off its mechanical stops.  This is a DELIBERATE narrowing of the
# trained action space, not a correction -- see the constant below.

#: Fraction of each factory joint limit the policy is allowed to command.
#:
#: The trained action space was the full range (Isaac's
#: ``soft_joint_pos_limit_factor`` is 1.0, so the policy WAS allowed to command
#: the hard stop, and measurements on 2026-08-22 showed it doing exactly that:
#: ``finger5_joint3`` railed on 100% of steps, asking for 179 mrad/tick past
#: the stop).  Narrowing here therefore changes what the policy can express --
#: it was chosen for hardware safety with that cost accepted.
#:
#: The rule is ``limit * ratio``, NOT ``centre +- ratio * half_range``
#: (``soft_command_limits`` below, which is Isaac's definition and is used for
#: the bring-up glides).  The two give different numbers -- for
#: ``finger5_joint3`` the upper bound is 1.5914 here versus 1.6197 there -- so
#: never substitute one for the other.  ``limit * ratio`` only shrinks a range
#: when it straddles zero, which is asserted at import.
COMMAND_LIMIT_RATIO = 0.95

COMMAND_TARGET_LIMITS = (REAL_HAND_FACTORY_LIMITS * COMMAND_LIMIT_RATIO).astype(np.float32)
if not (
    np.all(REAL_HAND_FACTORY_LIMITS[:, 0] <= 0.0)
    and np.all(REAL_HAND_FACTORY_LIMITS[:, 1] >= 0.0)
):
    # Multiplying a range that does not contain zero moves one bound AWAY from
    # the centre -- a "safety" ratio would then widen the very side it was
    # meant to protect.  All twenty joints straddle zero today; if a future
    # table does not, this must become an explicit shrink-toward-centre.
    raise RuntimeError(
        "COMMAND_LIMIT_RATIO assumes every joint range contains 0; "
        "scaling a one-sided range would widen it.")


def soft_command_limits(fraction: float, limits=None):
    """Shrink a command range toward its centre, Isaac's ``soft_joint_pos_limit_factor``.

    ``centre +- fraction * half_range`` -- the same definition Isaac's
    articulation uses, so a number chosen here means the same thing if it is
    ever moved into ``wuji.py`` (which is at ``1.0`` today, i.e. the trained
    policy WAS allowed to command the hard stop).

    This deliberately returns a NEW table instead of narrowing
    ``COMMAND_TARGET_LIMITS``: that constant records the action space the policy
    was trained in and the ``validate_contract`` equality with the observation
    range depends on it.  A margin is a deployment decision layered on top, and
    keeping it separate is what makes the resulting sim-to-real mismatch
    visible rather than silent.

    Measured motivation (2026-08-19, ``finger_reach_real_5.csv``): the reach
    policy parked ``finger3_joint3`` within 5 mrad of its upper stop for 36.5 %
    of a 20 s run while still commanding ``a3 ~ +0.45`` into it.  Numerical IK
    over the whole ``REACH_RANGE`` box reaches every point to <2 mm at
    ``fraction`` 1.00, 0.95 and 0.90 alike, so the margin costs no reach.
    """

    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Limit fraction must lie in (0, 1], got {fraction!r}.")
    table = np.asarray(COMMAND_TARGET_LIMITS if limits is None else limits, dtype=np.float32)
    if table.ndim != 2 or table.shape[1] != 2 or np.any(table[:, 0] >= table[:, 1]):
        raise ValueError("Limits must be an (N, 2) array with lower < upper.")
    if fraction == 1.0:
        # Bit-exact passthrough, not "close enough".  Reconstructing the same
        # edge as centre +- half in float32 lands one ULP off, which would make
        # the no-margin default fail to reproduce an existing log byte for byte.
        soft = table.copy()
    else:
        wide = table.astype(np.float64)
        centre = (wide[:, 0] + wide[:, 1]) / 2.0
        half = (wide[:, 1] - wide[:, 0]) / 2.0 * fraction
        soft = np.stack([centre - half, centre + half], axis=1).astype(np.float32)
    soft.setflags(write=False)
    return soft


for _limits in (
    OFFICIAL_NOMINAL_PHYSICAL_LIMITS,
    REAL_HAND_FACTORY_LIMITS,
    OBSERVATION_NORMALIZATION_LIMITS,
    COMMAND_TARGET_LIMITS,
):
    _limits.setflags(write=False)

# D. Deployed joint-controller gains, in canonical policy order.
#
# These are NOT from the vendor description.  The pinned official MJCF carries
# its own identified position-servo gains (kp 0.18~0.69, kv 0.008~0.031), which
# target roughly 4.7 Hz / zeta 0.67 per joint.  The values below were tuned by
# the user directly in Isaac Sim with joint step inputs, and they are the gains
# the deployed policy was actually trained against:
#
#     source: hand_real_env_cfg.py  HAND_REAL_STIFFNESS / HAND_REAL_DAMPING
#     verified against the saved params/env.yaml of hand_real runs
#     2026-08-18_00-50-53, 2026-08-18_09-50-02 and 2026-08-18_11-01-15
#
# Sim-to-sim isolates one variable - the physics engine - so MuJoCo must run
# the same commanded controller the policy was trained with.  The official MJCF
# gains stay reachable through ``MujocoWujiHand(controller_gains="official")``
# as a deliberate plant-mismatch robustness case, not as the default.
#
# Whether the real Hand 1 firmware exposes settable Kp/Kd is still unresolved
# (PENDING_REAL_VALIDATION).  If it does not, these gains describe a hand that
# does not exist and the sim-to-real gap lands here.
DEPLOY_STIFFNESS_NM_PER_RAD = np.asarray(
    [
        1.70, 2.70, 0.75, 1.00,
        2.40, 0.70, 0.75, 1.30,
        2.40, 0.70, 0.75, 1.30,
        2.40, 0.70, 0.75, 1.30,
        2.40, 0.70, 0.75, 1.15,
    ],
    dtype=np.float64,
)
DEPLOY_DAMPING_NMS_PER_RAD = np.asarray(
    [
        0.0400, 0.0500, 0.0015, 0.0015,
        0.0550, 0.0200, 0.0015, 0.0005,
        0.0550, 0.0200, 0.0015, 0.0001,
        0.0550, 0.0200, 0.0015, 0.0001,
        0.0550, 0.0200, 0.0015, 0.0001,
    ],
    dtype=np.float64,
)
# Effort limits are the official per-joint URDF ``effort`` values.  The pinned
# MJCF ``forcerange`` and Isaac's ``HAND_REAL_EFFORT_LIMITS`` are byte-identical
# to these, so applying them explicitly is a cross-source consistency check
# rather than a change.
DEPLOY_EFFORT_LIMITS_NM = np.asarray(
    [
        0.4452, 0.4259, 0.1888, 0.1468,
        0.6188, 0.1822, 0.2251, 0.2170,
        0.6494, 0.1827, 0.2078, 0.2018,
        0.6389, 0.1832, 0.2249, 0.2044,
        0.6441, 0.1798, 0.2384, 0.1866,
    ],
    dtype=np.float64,
)

for _gains in (
    DEPLOY_STIFFNESS_NM_PER_RAD,
    DEPLOY_DAMPING_NMS_PER_RAD,
    DEPLOY_EFFORT_LIMITS_NM,
):
    _gains.setflags(write=False)


DEFAULT_RESET_JOINT_POSITIONS = np.clip(
    np.zeros(ACTION_DIM, dtype=np.float32),
    REAL_HAND_FACTORY_LIMITS[:, 0],
    REAL_HAND_FACTORY_LIMITS[:, 1],
).astype(np.float32)
DEFAULT_RESET_JOINT_POSITIONS.setflags(write=False)


@dataclass(frozen=True)
class ObservationSlice:
    start: int
    stop: int
    description: str

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def dim(self) -> int:
        return self.stop - self.start


OBSERVATION_SLICES: dict[str, ObservationSlice] = {
    "joint_previous": ObservationSlice(0, 20, "previous normalized canonical q"),
    "joint_current": ObservationSlice(20, 40, "current normalized canonical q"),
    "fingertips": ObservationSlice(40, 55, "current tip-site xyz in canonical palm"),
    "stick1_previous": ObservationSlice(55, 62, "previous canonical Stick1 Pose7D"),
    "stick1_current": ObservationSlice(62, 69, "current canonical Stick1 Pose7D"),
    "stick2_previous": ObservationSlice(69, 76, "previous canonical Stick2 Pose7D"),
    "stick2_current": ObservationSlice(76, 83, "current canonical Stick2 Pose7D"),
    "last_action": ObservationSlice(83, 103, "last applied clipped policy action"),
    "mode": ObservationSlice(103, 105, "OPEN/CLOSE one-hot policy command"),
}
# Grasp-task observation width.  finger_reach.REACH_OBSERVATION_DIM is the
# reach probe's equivalent; neither is "the" observation dimension.
OBSERVATION_DIM = max(term.stop for term in OBSERVATION_SLICES.values())


def observation_csv_columns() -> list[str]:
    """One CSV column name per observation element, named by its block.

    Derived from ``OBSERVATION_SLICES`` rather than written out, so a layout
    change cannot leave the logs mislabelled -- the failure mode where a column
    called ``stick1`` quietly holds fingertips is unrecoverable after the fact.
    The index inside the name is the index within the block, which is what you
    want when reading e.g. ``obs_stick1_current_03`` as a quaternion component.
    """

    columns = ["" for _ in range(OBSERVATION_DIM)]
    for name, term in OBSERVATION_SLICES.items():
        for offset, index in enumerate(range(term.start, term.stop)):
            columns[index] = f"obs_{name}_{offset:02d}"
    if any(not column for column in columns):
        raise RuntimeError("OBSERVATION_SLICES leaves a gap; columns would be unnamed.")
    return columns


def normalize_joint_positions(joint_positions: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Normalize canonical actual q without clipping.

    Wuji joints have materially different ranges, so this fixed affine map
    gives policy joint-position features a consistent scale.  It is observation
    preprocessing only; residual actions always use unnormalized radians.
    """

    q = _finite_vector(joint_positions, ACTION_DIM, "canonical joint positions")
    lower = OBSERVATION_NORMALIZATION_LIMITS[:, 0]
    upper = OBSERVATION_NORMALIZATION_LIMITS[:, 1]
    center = (lower + upper) * np.float32(0.5)
    return (np.float32(2.0) * (q - center) / (upper - lower)).astype(np.float32)


def mode_one_hot(mode: str) -> npt.NDArray[np.float32]:
    """OPEN/CLOSE/NEUTRAL as the two-element command in ``obs[103:105]``.

    NEUTRAL is ``[0, 0]`` -- not a missing command but a THIRD state the task
    can train on.  ``HandMoveOpenCloseCommandCfg.neutral_before_open_close``
    emits it before ``open_close_start_time_s``, and a task whose
    ``episode_length_s`` equals that boundary (the 5 s grasp/setting curriculum
    stage, e.g. ``hand_real2``) therefore trains on NEUTRAL and nothing else.
    Such a checkpoint has never seen ``[1, 0]`` or ``[0, 1]``; driving it with
    the OPEN/CLOSE schedule feeds it an input outside its training set.
    """

    normalized = mode.strip().lower()
    if normalized == "open":
        return np.asarray([1.0, 0.0], dtype=np.float32)
    if normalized == "close":
        return np.asarray([0.0, 1.0], dtype=np.float32)
    if normalized == "neutral":
        return np.asarray([0.0, 0.0], dtype=np.float32)
    raise ValueError(
        f"Unknown hand mode {mode!r}; expected 'open', 'close' or 'neutral'.")


def validate_factory_limits(
    lower: npt.ArrayLike, upper: npt.ArrayLike
) -> dict[str, npt.NDArray[np.float32]]:
    """Validate, but never adopt, one real hand's calibrated limits."""

    lo = _finite_vector(lower, ACTION_DIM, "factory lower limits")
    hi = _finite_vector(upper, ACTION_DIM, "factory upper limits")
    if np.any(lo >= hi):
        raise ValueError("Every factory lower limit must be below its upper limit.")
    return {
        "factory_lower": lo.copy(),
        "factory_upper": hi.copy(),
        "lower_minus_nominal": lo - OFFICIAL_NOMINAL_PHYSICAL_LIMITS[:, 0],
        "upper_minus_nominal": hi - OFFICIAL_NOMINAL_PHYSICAL_LIMITS[:, 1],
        "lower_minus_contract": lo - REAL_HAND_FACTORY_LIMITS[:, 0],
        "upper_minus_contract": hi - REAL_HAND_FACTORY_LIMITS[:, 1],
    }


def contract_summary() -> str:
    return "\n".join(
        [
            "[CANONICAL WUJI HAND 1 CONTRACT]",
            f"description revision: {WUJI_DESCRIPTION_REVISION} ({WUJI_DESCRIPTION_RELEASE})",
            f"joints: {ACTION_DIM} {POLICY_JOINT_NAMES}",
            f"palm frame: {PALM_FRAME_NAME}",
            f"stick representation: {STICK_REPRESENTATION}",
            f"observation: {OBSERVATION_DIM}D, oldest-to-newest history",
            f"action: clip +/-{ACTION_CLIP:g}, q + per-joint scale*a",
            "action scale [rad]: "
            + ", ".join(
                f"{suffix}={value:g}"
                for suffix, value in _ACTION_SCALE_BY_JOINT_SUFFIX.items()
            ),
            f"Joint4 command floors: none (lifted 2026-08-18; indices "
            f"{JOINT4_POLICY_INDICES.tolist()} now reach the articulation lower "
            f"limit)",
            f"policy frequency: {1/POLICY_DT:.1f} Hz",
            "controller gains: Isaac-tuned (hand_real HAND_REAL_STIFFNESS/DAMPING), "
            f"Kp {DEPLOY_STIFFNESS_NM_PER_RAD.min():g}~{DEPLOY_STIFFNESS_NM_PER_RAD.max():g}, "
            f"Kd {DEPLOY_DAMPING_NMS_PER_RAD.min():g}~{DEPLOY_DAMPING_NMS_PER_RAD.max():g} "
            "(NOT the official MJCF identification)",
        ]
    )


def _finite_vector(value: npt.ArrayLike, size: int, label: str) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{label} must have shape {(size,)}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite.")
    return array


def validate_contract() -> None:
    if HAND_JOINT_COUNT != ACTION_DIM:
        raise RuntimeError(
            "The grasp policy is expected to drive every hand joint; if that "
            "changes, split the two constants at their use sites."
        )
    if tuple(j.policy_index for j in CANONICAL_JOINTS) != tuple(range(ACTION_DIM)):
        raise RuntimeError("Canonical policy indices must be exactly 0..19.")
    if len(set(POLICY_JOINT_NAMES)) != ACTION_DIM:
        raise RuntimeError("Canonical joint names must be unique.")
    for limits in (
        OFFICIAL_NOMINAL_PHYSICAL_LIMITS,
        REAL_HAND_FACTORY_LIMITS,
        OBSERVATION_NORMALIZATION_LIMITS,
        COMMAND_TARGET_LIMITS,
    ):
        if limits.shape != (ACTION_DIM, 2) or np.any(limits[:, 0] >= limits[:, 1]):
            raise RuntimeError("Every limit table must be a valid independent (20,2) array.")
    if np.shares_memory(REAL_HAND_FACTORY_LIMITS, OBSERVATION_NORMALIZATION_LIMITS):
        raise RuntimeError("Factory and observation limits must not alias.")
    if np.any(OFFICIAL_NOMINAL_PHYSICAL_LIMITS[:, 0] < REAL_HAND_FACTORY_LIMITS[:, 0]) or np.any(
        OFFICIAL_NOMINAL_PHYSICAL_LIMITS[:, 1] > REAL_HAND_FACTORY_LIMITS[:, 1]
    ):
        raise RuntimeError(
            "The vendor description is expected to be contained in the measured "
            "factory range; re-check the device read if this fires."
        )
    if np.shares_memory(OBSERVATION_NORMALIZATION_LIMITS, COMMAND_TARGET_LIMITS):
        raise RuntimeError("Observation and command limits must not alias.")
    # Until 2026-08-22 these two had to be EQUAL, because Isaac trains with a
    # single clamp derived from the articulation limits.  COMMAND_LIMIT_RATIO
    # deliberately breaks that equality to keep the hand off its stops, so the
    # invariant is now containment plus the exact scaling rule.  Normalization
    # must stay on the unscaled table: it defines what the network's inputs
    # mean, and rescaling it would feed the policy numbers it never trained on.
    if not np.array_equal(OBSERVATION_NORMALIZATION_LIMITS, REAL_HAND_FACTORY_LIMITS):
        raise RuntimeError(
            "Observation normalization must stay on the unscaled factory table."
        )
    if not np.allclose(COMMAND_TARGET_LIMITS,
                       REAL_HAND_FACTORY_LIMITS * COMMAND_LIMIT_RATIO,
                       atol=1e-7):
        raise RuntimeError(
            "COMMAND_TARGET_LIMITS must be REAL_HAND_FACTORY_LIMITS * COMMAND_LIMIT_RATIO."
        )
    if np.any(COMMAND_TARGET_LIMITS[:, 0] < OBSERVATION_NORMALIZATION_LIMITS[:, 0]) or np.any(
        COMMAND_TARGET_LIMITS[:, 1] > OBSERVATION_NORMALIZATION_LIMITS[:, 1]
    ):
        raise RuntimeError(
            "The command range must sit inside the normalization range; a "
            "commandable target the policy cannot express is a wiring error."
        )
    if OBSERVATION_DIM != 105:
        raise RuntimeError(f"StickPose7D contract must be 105D, got {OBSERVATION_DIM}.")
    if ACTION_SCALE_RAD.shape != (ACTION_DIM,) or ACTION_SCALE_RAD.flags.writeable:
        raise RuntimeError("Action scale must be a read-only per-joint (20,) array.")
    if not np.all(ACTION_SCALE_RAD > 0.0):
        raise RuntimeError("Every per-joint action scale must be positive.")
    for _name, _table, _strict_positive in (
        ("stiffness", DEPLOY_STIFFNESS_NM_PER_RAD, True),
        ("damping", DEPLOY_DAMPING_NMS_PER_RAD, False),
        ("effort limits", DEPLOY_EFFORT_LIMITS_NM, True),
    ):
        if _table.shape != (ACTION_DIM,) or _table.flags.writeable:
            raise RuntimeError(f"Deploy {_name} must be a read-only (20,) array.")
        if _strict_positive and not np.all(_table > 0.0):
            raise RuntimeError(f"Every deploy {_name} value must be positive.")
        if not _strict_positive and np.any(_table < 0.0):
            raise RuntimeError(f"Deploy {_name} must not be negative.")


validate_contract()

# Temporary compatibility aliases for external imports.  Their names make the
# old ambiguity visible and new code must use the three explicit limit tables.
ACTION_SCALE = ACTION_SCALE_RAD
ACTION_TARGET_LOWER_LIMITS = COMMAND_TARGET_LIMITS[:, 0]
ACTION_TARGET_UPPER_LIMITS = COMMAND_TARGET_LIMITS[:, 1]
