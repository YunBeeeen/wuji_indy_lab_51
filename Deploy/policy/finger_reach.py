# [policy] 중지 4관절 reach 계약(15D->4D)과 러너. 20관절 파지 계약과 섞이면 안 되는 별개 계약.
"""Middle-finger reach diagnostic: the MuJoCo half of the Sim-to-Sim probe.

This is deliberately NOT the 105D ``hand_real`` stack.  The reach task exists to
answer one question -- "given the same policy, the same initial joint state and
the same target command, do Isaac / MuJoCo / the real hand move the same way?"
-- so its contract is the smallest thing that can carry that comparison:

    observation 15D   q_prev(4) + q_curr(4) + target_palm(3) + last_action(4)
    action       4D   middle-finger residual

No vision, no sticks, no fingertip FK in the policy path.  ``target_palm`` is a
*command* the operator supplies, not a sensor reading, so this same observation
can be built on hardware from encoders alone.  Fingertip FK is computed for
logging only, which keeps a kinematics mismatch measurable and separate from a
policy-input mismatch.

Mirrors ``isaac_neuromeka/tasks/manipulation/hand_grasp/finger_reach_env_cfg.py``.

Reset is a training concept, not a deployment one
-------------------------------------------------
Training resets by teleporting joints to ``q_reset`` and resampling the target,
which a real hand cannot do.  A validation run therefore resets ONCE and then
runs continuously, changing only the target -- which is what
``MiddleFingerReachRunner`` does: ``reset()`` is called before the loop and the
loop calls ``set_target()`` only.  Making the Isaac episode long enough not to
auto-reset is half of the equivalence; not resetting between targets is the
other half.

On hardware that single reset becomes a ramp:

    1. read the current 20-joint state;
    2. interpolate from it to the start pose over seconds, publishing at the
       hardware rate -- never command the start pose in one frame, the servo
       would slam toward it;
    3. hold and let it settle;
    4. seed the observation history from the SETTLED state, both slots equal
       and ``last_action`` zero, exactly as ``reset()`` does here;
    5. run the policy continuously, moving the target on a schedule;
    6. ramp down and disable.

Only step 4 is this module's business.  Steps 1-3 and 5-6 belong to the real
backend, which is still refused construction until its SDK contract is measured.
A waypoint target also steps discontinuously, so a short dwell makes the policy
command a large residual the instant it changes; prefer several seconds per
target on hardware, and keep Isaac and MuJoCo on the same dwell so the three
logs stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..common.policy_contract import (
    REAL_HAND_FACTORY_LIMITS,
    soft_command_limits,
    OBSERVATION_NORMALIZATION_LIMITS,
    POLICY_JOINT_NAMES,
)


REACH_FINGER_INDEX = 3
MIDDLE_JOINT_NAMES: tuple[str, ...] = tuple(
    f"finger{REACH_FINGER_INDEX}_joint{k}" for k in range(1, 5)
)
# Resolved by NAME against the canonical 20-joint order -- never hard-coded,
# for the same reason the backend maps MuJoCo storage by name.
MIDDLE_POLICY_INDICES = np.asarray(
    [POLICY_JOINT_NAMES.index(name) for name in MIDDLE_JOINT_NAMES], dtype=np.int32
)
MIDDLE_POLICY_INDICES.setflags(write=False)

REACH_ACTION_DIM = 4
REACH_OBSERVATION_DIM = 15
# Isaac's FingerReachActionsCfg uses a single scalar 0.1 for all four joints.
# This is NOT the per-joint HAND_REAL_ACTION_SCALE (0.1/0.1/0.2/0.15); that
# table belongs to the twenty-joint grasp contract and must not leak in here.
REACH_ACTION_SCALE_RAD = np.float32(0.1)
REACH_ACTION_CLIP = np.float32(1.0)

REACH_OBSERVATION_SLICES = {
    "q_previous": slice(0, 4),
    "q_current": slice(4, 8),
    "target_palm": slice(8, 11),
    "last_action": slice(11, 15),
}

# Palm-frame sampling box used by Isaac's FingerTipReachCommand.  Reproduced
# here so a scenario generator and a range check share one definition.
# Mirrors Isaac's REACH_RANGE_X/Y/Z.  y is a single value in the live run, so
# targets vary in the palm plane spanned by x and z only.
REACH_RANGE_M = {
    "x": (0.025, 0.050),
    "y": (0.010, 0.010),
    "z": (0.085, 0.115),
}

# The real hand streams and accepts 20 joints in FINGER-MAJOR order
# (finger1..finger5 x joint1..joint4) -- stated outright in the vendor's
# subscribe example -- which is exactly CANONICAL_JOINTS.  The 5x4 grid below
# is therefore a reshape, never a reordering, and ``finger_major_grid`` exists
# so logs and hardware frames share the vendor's shape.
FINGERS = 5
JOINTS_PER_FINGER = 4


def finger_major_grid(values20: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Reshape a canonical 20-vector into the SDK's ``(5, 4)`` hand layout."""

    flat = _finite(values20, FINGERS * JOINTS_PER_FINGER, "20-joint vector")
    return flat.reshape(FINGERS, JOINTS_PER_FINGER)

# (5,4) -> (20,) 반환해주는 함수
def from_finger_major_grid(grid: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Flatten an SDK-shaped ``(5, 4)`` hand layout back to canonical order."""

    array = np.asarray(grid, dtype=np.float32)
    if array.shape != (FINGERS, JOINTS_PER_FINGER) or not np.isfinite(array).all():
        raise ValueError(f"Grid must be a finite {(FINGERS, JOINTS_PER_FINGER)} array.")
    return np.ascontiguousarray(array.reshape(-1))


# Reach starts from an OPEN hand, not from the grasp pregrasp.
# ``MujocoWujiHand.reset()`` defaults to ISAAC_PREGRASP_JOINT_POSITIONS_RAD --
# the curled pose_005 chopstick grasp -- which belongs to the hand_grasp family.
# Isaac's finger_reach inherits HandGraspSceneCfg's articulation init_state
# instead and has no reset_pregrasp event, so its reset is
# ``finger1_joint1=0.05`` with the other nineteen at zero (verified against the
# live run's params/env.yaml).  Starting MuJoCo from the grasp pose instead
# would put every joint up to 1.63 rad away from where Isaac starts.
FINGER_REACH_RESET_JOINT_POSITIONS = np.zeros(len(POLICY_JOINT_NAMES), dtype=np.float32)
FINGER_REACH_RESET_JOINT_POSITIONS[POLICY_JOINT_NAMES.index("finger1_joint1")] = 0.05
FINGER_REACH_RESET_JOINT_POSITIONS.setflags(write=False)


_MIDDLE_NORMALIZATION = OBSERVATION_NORMALIZATION_LIMITS[MIDDLE_POLICY_INDICES]

# The reach command range is the connected hand's own factory range -- no
# distal floor.  Isaac's finger_reach dropped its joint4 override on 2026-08-18
# for the same reason the twenty-joint tasks did, and the floor was measurably
# inert for model_500 (joint4 never went below +0.186 rad; MuJoCo trajectories
# with and without it matched to 0.000 mrad over 120 policy steps).
#
# Derived from the UNSCALED factory table, not from COMMAND_TARGET_LIMITS.
# The grasp task's COMMAND_LIMIT_RATIO (0.95) narrows that table, and the reach
# CLIs already apply their own 0.95 through ``middle_soft_command_limits``
# (``--limit-margin``, default 0.95) -- taking the scaled table here would
# stack the two into an effective 0.90 and silently move a validated task.
MIDDLE_COMMAND_TARGET_LIMITS = REAL_HAND_FACTORY_LIMITS[MIDDLE_POLICY_INDICES]
_MIDDLE_COMMAND = MIDDLE_COMMAND_TARGET_LIMITS


def normalize_middle_joints(q_middle: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Same affine map Isaac's ``joint_pos_limit_normalized`` applies, no clip."""

    q = _finite(q_middle, REACH_ACTION_DIM, "middle joint positions")
    lower = _MIDDLE_NORMALIZATION[:, 0]
    upper = _MIDDLE_NORMALIZATION[:, 1]
    center = (lower + upper) * np.float32(0.5)
    return (np.float32(2.0) * (q - center) / (upper - lower)).astype(np.float32)


@dataclass(frozen=True)
class DecodedReachAction:
    raw_action: npt.NDArray[np.float32]
    clipped_action: npt.NDArray[np.float32]
    unclamped_target: npt.NDArray[np.float32]
    position_target: npt.NDArray[np.float32]


def decode_reach_action(
    q_middle_current: npt.ArrayLike,
    raw_action: npt.ArrayLike,
    command_limits: npt.ArrayLike | None = None,
) -> DecodedReachAction:
    """``clip -> q_current + 0.1*a -> clamp`` , matching CustomResidualJointPositionAction.

    ``command_limits`` defaults to the trained action space.  Passing a
    narrowed table (see ``soft_command_limits``) keeps the finger off its
    mechanical stops; the clamp is the only stage that changes, so the raw and
    unclamped fields still record what the policy actually asked for.
    """

    q = _finite(q_middle_current, REACH_ACTION_DIM, "middle q")
    raw = _finite(raw_action, REACH_ACTION_DIM, "raw action")
    limits = _MIDDLE_COMMAND if command_limits is None else np.asarray(command_limits, np.float32)
    if limits.shape != (REACH_ACTION_DIM, 2):
        raise ValueError(f"Command limits must have shape {(REACH_ACTION_DIM, 2)}.")
    clipped = np.clip(raw, -REACH_ACTION_CLIP, REACH_ACTION_CLIP).astype(np.float32)
    unclamped = (q + REACH_ACTION_SCALE_RAD * clipped).astype(np.float32)
    target = np.clip(unclamped, limits[:, 0], limits[:, 1]).astype(np.float32)
    return DecodedReachAction(raw.copy(), clipped, unclamped, target)


def middle_soft_command_limits(fraction: float) -> npt.NDArray[np.float32]:
    """The four middle-finger command limits shrunk toward their centres."""

    return soft_command_limits(fraction, MIDDLE_COMMAND_TARGET_LIMITS)


@dataclass
class FingerReachObservationAdapter:
    """Two-sample ``[previous, current]`` history, sampled once per policy step."""

    target_palm: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )

    def __post_init__(self) -> None:
        self._initialized = False
        self._q_previous = np.zeros(REACH_ACTION_DIM, dtype=np.float32)
        self._q_current = np.zeros(REACH_ACTION_DIM, dtype=np.float32)
        self._last_action = np.zeros(REACH_ACTION_DIM, dtype=np.float32)
        self.set_target(self.target_palm)

    def set_target(self, target_palm: npt.ArrayLike) -> None:
        self.target_palm = _finite(target_palm, 3, "target_palm")

    def reset(self, q_middle: npt.ArrayLike) -> None:
        q = _finite(q_middle, REACH_ACTION_DIM, "middle q")
        # Isaac's history buffer fills every slot with the first sample, so both
        # slots equal the reset pose and the policy infers zero motion.
        self._q_previous = q.copy()
        self._q_current = q.copy()
        self._last_action.fill(0.0)
        self._initialized = True

    def advance(self, q_middle: npt.ArrayLike, last_action: npt.ArrayLike) -> None:
        if not self._initialized:
            raise RuntimeError("Call reset() before advance().")
        self._q_previous = self._q_current.copy()
        self._q_current = _finite(q_middle, REACH_ACTION_DIM, "middle q")
        self._last_action = _finite(last_action, REACH_ACTION_DIM, "last_action")

    @property
    def q_previous(self) -> npt.NDArray[np.float32]:
        """Previous middle-joint sample in RADIANS (the observation is normalized)."""

        return self._q_previous.copy()

    @property
    def q_current(self) -> npt.NDArray[np.float32]:
        """Current middle-joint sample in RADIANS."""

        return self._q_current.copy()

    def build(self) -> npt.NDArray[np.float32]:
        if not self._initialized:
            raise RuntimeError("Call reset() before build().")
        observation = np.empty(REACH_OBSERVATION_DIM, dtype=np.float32)
        observation[REACH_OBSERVATION_SLICES["q_previous"]] = normalize_middle_joints(self._q_previous)
        observation[REACH_OBSERVATION_SLICES["q_current"]] = normalize_middle_joints(self._q_current)
        observation[REACH_OBSERVATION_SLICES["target_palm"]] = self.target_palm
        observation[REACH_OBSERVATION_SLICES["last_action"]] = self._last_action
        if not np.isfinite(observation).all():
            raise AssertionError("Reach observation must be finite float32.")
        return np.ascontiguousarray(observation)


class MiddleFingerReachRunner:
    """Drive four joints, hold the other sixteen, log everything else.

    The sixteen uncontrolled joints keep the target captured at ``reset()``.
    They are deliberately NOT re-commanded to ``q_current`` every step: that
    would let drift or an external force redefine the hold, which is exactly
    what Isaac's ``hold_joints_at_default`` reset event avoids.
    """

    def __init__(
        self,
        backend,
        policy,
        observation_adapter: FingerReachObservationAdapter,
        limit_fraction: float = 1.0,
    ):
        self.backend = backend
        self.policy = policy
        self.observations = observation_adapter
        # 1.0 reproduces the trained clamp exactly; below 1.0 holds the four
        # policy joints off their stops.  The sixteen held joints are clamped
        # with the same table so a hand parked past a soft edge is walked in
        # rather than commanded to stay there.
        self.limit_fraction = float(limit_fraction)
        self.command_limits = middle_soft_command_limits(self.limit_fraction)
        self.hold_limits = soft_command_limits(self.limit_fraction)
        self._hold_targets = np.zeros(len(POLICY_JOINT_NAMES), dtype=np.float32)
        self.last_decoded: DecodedReachAction | None = None

    def seed_from_current_state(
        self, q_all: npt.ArrayLike, write_hold: bool = True
    ) -> npt.NDArray[np.float32]:
        """Initialize from wherever the hand already is -- no teleport.

        A real hand cannot be reset: it can only be moved somewhere and then
        observed.  This takes the 20 joint angles as they actually are, freezes
        the sixteen non-policy ones as the hold, and seeds the history with
        ``q_previous = q_current`` and ``last_action = 0`` -- the same state
        Isaac's reset leaves behind.  It commands no motion of its own.

        ``reset()`` below is the simulator path and reuses this, so the two
        backends cannot drift apart in how they start.
        """

        q_all = np.asarray(q_all, dtype=np.float32)
        if q_all.shape != (len(POLICY_JOINT_NAMES),) or not np.isfinite(q_all).all():
            raise ValueError(
                f"q_all must be a finite ({len(POLICY_JOINT_NAMES)},) vector, got {q_all.shape}."
            )
        self._hold_targets = np.clip(
            q_all, self.hold_limits[:, 0], self.hold_limits[:, 1]
        ).astype(np.float32)
        if write_hold:
            self.backend.write_joint_position_targets(self._hold_targets)
        self.observations.reset(q_all[MIDDLE_POLICY_INDICES])
        self.last_decoded = None
        return self.observations.build()

    def reset(self, q_reset_all: npt.ArrayLike | None = None) -> npt.NDArray[np.float32]:
        """Simulator-only: teleport to the reach pose, then seed from it.

        Never call this on hardware -- ``backend.reset()`` is a state jump.
        Use ``seed_from_current_state()`` after ramping instead.
        """

        # Default to the reach reset, never the backend's grasp pregrasp.
        if q_reset_all is None:
            q_reset_all = FINGER_REACH_RESET_JOINT_POSITIONS
        self.backend.reset(np.asarray(q_reset_all, dtype=np.float32))
        return self.seed_from_current_state(self.backend.read_joint_positions())

    def set_target(self, target_palm: npt.ArrayLike) -> None:
        self.observations.set_target(target_palm)

    def command(self, q_all: npt.ArrayLike | None = None) -> DecodedReachAction:
        """Infer and store the next target.

        ``q_all`` lets a caller supply a measurement it already took.  MuJoCo
        passes nothing and reads from the backend, exactly as before.  Hardware
        passes one reading per policy tick so the observation and the residual
        base are the same sample -- two separate SDK reads inside one tick would
        differ by encoder noise, making the policy see a q it was not decoded
        against.
        """

        raw = np.asarray(self.policy.infer(self.observations.build()), dtype=np.float32)
        q_all = (
            self.backend.read_joint_positions()
            if q_all is None
            else np.asarray(q_all, dtype=np.float32)
        )
        decoded = decode_reach_action(q_all[MIDDLE_POLICY_INDICES], raw, self.command_limits)
        targets = self._hold_targets.copy()
        targets[MIDDLE_POLICY_INDICES] = decoded.position_target
        self.backend.write_joint_position_targets(targets)
        self.last_decoded = decoded
        return decoded

    def observe_after_hold(
        self, q_all: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float32]:
        """Advance the history AFTER the plant has run for one policy step.

        The name is literal: it must be called once the target has actually been
        applied for 1/30 s, never immediately after ``command()``.  Calling it
        before the plant moves makes ``q_previous == q_current``, so the policy
        reads zero motion every step and the history carries no information.
        """

        if self.last_decoded is None:
            raise RuntimeError("command() must precede observe_after_hold().")
        q_all = (
            self.backend.read_joint_positions()
            if q_all is None
            else np.asarray(q_all, dtype=np.float32)
        )
        self.observations.advance(q_all[MIDDLE_POLICY_INDICES], self.last_decoded.clipped_action)
        return self.observations.build()

    def joint_command_grid(self) -> npt.NDArray[np.float32]:
        """Current full-hand position command as the SDK's ``(5, 4)`` grid.

        This is the frame a real publisher sends: twenty ``JointCommand``
        entries in finger-major order, of which only row 2 (the middle finger)
        is policy-driven and the rest are the reset hold.  Exposing it in the
        vendor's shape keeps the MuJoCo and hardware paths textually similar.
        """

        targets = self._hold_targets.copy()
        if self.last_decoded is not None:
            targets[MIDDLE_POLICY_INDICES] = self.last_decoded.position_target
        return finger_major_grid(targets)

    def joint_state_grid(self) -> npt.NDArray[np.float32]:
        """Measured joint positions as the SDK's ``(5, 4)`` grid."""

        return finger_major_grid(self.backend.read_joint_positions())

    def middle_fingertip_in_palm(self) -> npt.NDArray[np.float32]:
        """Logging only -- never an input to the policy."""

        tips = self.backend.get_fingertip_positions_in_palm().reshape(5, 3)
        return tips[REACH_FINGER_INDEX - 1].astype(np.float32)


def _finite(value: npt.ArrayLike, size: int, label: str) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ValueError(f"{label} must have shape {(size,)}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite.")
    return array
