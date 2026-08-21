"""Backend-neutral policy interface and two-phase control step."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from .action_adapter import DecodedAction, decode_policy_action
from ..common.backend_protocol import WujiBackend
from .observation_adapter import PolicyObservationAdapter
from ..common.perception import PoseState


class Policy(Protocol):
    def infer(self, observation: npt.ArrayLike) -> npt.NDArray[np.float32]: ...


class PolicyRunner:
    """Share observation/action plumbing between MuJoCo and Real.

    A backend-specific scheduler owns the interval between ``command`` and
    ``observe_after_hold``: a fixed number of physics substeps in MuJoCo, and a
    separately validated I/O/control schedule on hardware.  The pacing differs;
    the contract below must not, which is why this class is shared rather than
    duplicated per backend.

    **One joint reading per policy step.**  The observation and the residual
    base are required to be the SAME sample: ``q_target = q_current + scale *
    action`` is only meaningful if ``q_current`` is the q the policy was shown.
    This class therefore reads once and caches, instead of reading again inside
    ``command()``.

    That used to be two reads.  In MuJoCo they returned bit-identical values
    (measured 2026-08-21: max delta 0.000e+00 rad over 60 steps) because no
    physics runs between ``observe_after_hold`` and the next ``command``, so the
    redundancy was invisible.  On hardware the same two reads differ by encoder
    noise -- finger_reach saw a 0.101 rad delta reported against a 0.100 rad
    guard that was never actually exceeded.
    """

    def __init__(
        self,
        backend: WujiBackend,
        policy: Policy,
        observation_adapter: PolicyObservationAdapter,
    ) -> None:
        self.backend = backend
        self.policy = policy
        # Perception is supplied through the independently composed adapter;
        # the runner does not select a synthetic, vision, or backend provider.
        self.observations = observation_adapter
        self.last_decoded_action: DecodedAction | None = None
        # The single joint sample this policy step is built on.  Set by reset()
        # and observe_after_hold(); read by command().
        self._q_current: npt.NDArray[np.float32] | None = None

    def reset(self, q_current: npt.ArrayLike | None = None) -> npt.NDArray[np.float32]:
        """Seed the history from the present joint state.

        ``q_current`` lets a caller supply a reading it has already taken --
        hardware takes exactly one per policy step and reuses it.  MuJoCo passes
        nothing and this reads from the backend, as before.

        Fingertips are solved from q inside the adapter, in the trained tip
        frames.  The backend is never asked for them: its own model's tips are
        not the ones the policy learned.
        """

        self._q_current = self._sample(q_current)
        self.observations.reset(self._q_current)
        self.last_decoded_action = None
        return self.observations.build()

    def set_mode(self, mode: str) -> None:
        self.observations.set_mode(mode)

    def command(self) -> DecodedAction:
        state = self.observations.perception_state
        if state in (PoseState.STALE, PoseState.LOST):
            # HOLD is deliberately NOT here: a couple of dropped frames must not
            # freeze a grasp.  With the dual-camera tracker this point is only
            # reached once BOTH cameras have missed the stick for long enough --
            # MAIN sees neither marker AND the SIDE fallback is also invalid.
            #
            # safe_stop freezes the command; it does not disable and does not
            # release the grip preload.  Holding it requires that the caller
            # keep publishing, which is why this raises into an abort handler
            # rather than simply returning.
            self.backend.safe_stop(f"{state.value} perception")
            raise RuntimeError(f"Policy command blocked by {state.value} perception.")
        if self._q_current is None:
            raise RuntimeError("reset() must precede command().")
        raw_action = self.policy.infer(self.observations.build())
        # The cached sample, not a fresh read: see the class docstring.
        decoded = decode_policy_action(self._q_current, raw_action)
        self.backend.write_joint_position_targets(decoded.position_target)
        self.last_decoded_action = decoded
        return decoded

    def observe_after_hold(
        self, q_current: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float32]:
        """Advance the history AFTER the target has been held for one step.

        The name is literal.  Calling it before the plant has moved makes
        ``q_previous == q_current``, so the policy reads zero motion every step
        and the history carries no information.

        The sample taken here becomes the residual base for the next
        ``command()``, which is what keeps the pair consistent.
        """

        if self.last_decoded_action is None:
            raise RuntimeError("command() must precede observe_after_hold().")
        self._q_current = self._sample(q_current)
        self.observations.advance(
            self._q_current,
            self.last_decoded_action.action_manager_action,
        )
        return self.observations.build()

    def _sample(self, q_current: npt.ArrayLike | None) -> npt.NDArray[np.float32]:
        if q_current is None:
            return self.backend.read_joint_positions()
        return np.asarray(q_current, dtype=np.float32).copy()
