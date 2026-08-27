# [policy] command/observe_after_hold 2상 스텝. 정책 스텝당 q를 한 번만 읽어 관측과 잔차 기준을 같은 표본으로 유지.
"""MuJoCo와 실물에서 공유하는 2단계 정책 실행 인터페이스."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from .action_adapter import DecodedAction, decode_policy_action
from ..common.backend_protocol import WujiBackend
from ..common.perception import PoseState
from ..common.policy_contract import POLICY_DT
from ..common.timing import StageTimer


class Policy(Protocol):
    def infer(self, observation: npt.ArrayLike) -> npt.NDArray[np.float32]: ...


class ObservationAdapter(Protocol):
    mode: str
    timing: StageTimer

    def reset(self, q_current: npt.ArrayLike) -> None: ...
    def advance(self, q_current: npt.ArrayLike, last_action: npt.ArrayLike) -> None: ...
    def build(self) -> npt.NDArray[np.float32]: ...
    def set_mode(self, mode: str) -> None: ...

    @property
    def perception_state(self) -> PoseState: ...


class ActionDecoder(Protocol):
    def __call__(
        self, q_current_policy_order: npt.ArrayLike, raw_policy_action: npt.ArrayLike
    ) -> DecodedAction: ...


class PolicyRunner:
    """백엔드 독립 관측 조립과 액션 적용.

    정책 틱마다 관절을 한 번 읽고 관측과 잔차 목표 계산에 같은 표본 사용.
    """

    def __init__(
        self,
        backend: WujiBackend,
        policy: Policy,
        observation_adapter: ObservationAdapter,
        action_decoder: ActionDecoder = decode_policy_action,
    ) -> None:
        self.backend = backend
        self.policy = policy
        # Perception is supplied through the independently composed adapter;
        # the runner does not select a synthetic, vision, or backend provider.
        self.observations = observation_adapter
        # The decoder is part of the checkpoint contract.  The default retains
        # the active 105D path byte-for-byte; legacy policies inject their
        # isolated uniform-0.1 decoder at the composition root.
        self.action_decoder = action_decoder
        self.last_decoded_action: DecodedAction | None = None
        #: Per-stage latency inside one policy tick.  Always on: timing you have
        #: to switch on is timing you do not have when the bad run happens.
        #: ``observation_build`` includes the stick-pose sample, so a slow camera
        #: shows up here as well as in the provider's own timer.
        self.timing = StageTimer(budget_ms=1000.0 * POLICY_DT, name="policy")
        # The single joint sample this policy step is built on.  Set by reset()
        # and observe_after_hold(); read by command().
        self._q_current: npt.NDArray[np.float32] | None = None
        #: The observation the most recent command() ran on.  See command().
        self.last_observation: npt.NDArray[np.float32] | None = None

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
        self.last_observation = None
        return self.observations.build()

    @property
    def q_current(self):
        """The single joint sample this policy step is built on.

        The first tick after ``reset()`` must decode against THIS, not against a
        fresh read.  Reading again splits the tick across two samples: the
        observation and the residual come from one, the backend's slew guard
        checks the other, and the drift between them lands on top of a legal
        step.  Measured 2026-08-21 on an unpowered hand settling: 2.35 mrad,
        enough to push a 0.150 rad joint4 step to 0.15235 and be refused.
        """

        if self._q_current is None:
            raise RuntimeError("Call reset() before reading q_current.")
        return self._q_current.copy()

    def set_mode(self, mode: str) -> None:
        self.observations.set_mode(mode)

    def command(self) -> DecodedAction:
        with self.timing.stage("observation_build"):
            observation = self.observations.build()
        # Kept so the caller can log the exact input this action came from.
        # Without it a run is not reproducible: rerunning the ONNX on recorded
        # q alone cannot rebuild obs, which also carries the stick history the
        # cameras happened to see.  It is the only way to separate "the policy
        # is wired wrong" from "the policy saw a different world".
        self.last_observation = observation
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
        with self.timing.stage("policy_infer"):
            raw_action = self.policy.infer(observation)
        # The cached sample, not a fresh read: see the class docstring.
        with self.timing.stage("action_decode"):
            decoded = self.action_decoder(self._q_current, raw_action)
        with self.timing.stage("target_write"):
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
        with self.timing.stage("joint_read"):
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
