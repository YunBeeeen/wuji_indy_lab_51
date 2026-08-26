"""Physics-step Auto Tune state machine used by the Isaac Sim child."""

from __future__ import annotations

import math
from typing import Any, Callable

from ..messages import EventKind
from .config import AutoTuneTorquePolicy, ResolvedAutoTuneConfig
from .controller import AutoTuneController
from .metrics import TrialMetrics, TrialSample, evaluate_trial, hard_failure_reasons_for_sample
from .result import AutoTuneOutcome, CandidateResult, CandidateSpec, CandidateStatus


EventSender = Callable[[EventKind, dict[str, Any]], None]


class AutoTuneRuntime:
    """Advance independent candidate trials without blocking GUI or physics IPC."""

    def __init__(
        self,
        *,
        robot: Any,
        adapter: Any,
        config: ResolvedAutoTuneConfig,
        target_positions: Any,
        target_velocities: Any,
        event_sender: EventSender,
    ) -> None:
        self.robot = robot
        self.adapter = adapter
        self.config = config
        self.target_positions = target_positions
        self.target_velocities = target_velocities
        self.send = event_sender
        self.controller = AutoTuneController(config)
        self.base_positions = robot.data.joint_pos.clone()
        self.base_velocities = robot.data.joint_vel.clone().zero_()
        self.original_gain = adapter.read_gains(config.joint_index)
        self.active = True
        self.paused = False
        self.cancelled = False
        self.complete = False
        self.phase = "starting"
        self.phase_start_time = 0.0
        self.current_candidate: CandidateSpec | None = None
        self.current_trials: list[TrialMetrics] = []
        self.trial_plan = [
            ("positive" if sign > 0 else "negative", sign, repeat)
            for sign in config.direction.signs
            for repeat in range(config.repeats)
        ]
        self.trial_plan_index = 0
        self.current_samples: list[TrialSample] = []
        self.trial_start_position = config.q0
        self.measurement_start_time = 0.0
        self.abort_candidate_after_return = False
        self.last_progress_time = -math.inf
        self.outcome: AutoTuneOutcome | None = None

    @property
    def blocks_manual_waveform(self) -> bool:
        return self.active and not self.complete

    def start(self, simulation_time: float) -> None:
        """Start the first coarse point and publish the resolved configuration."""

        signals = self.adapter.read_effort_signals(self.config.joint_index)
        if signals.computed_effort is None or signals.applied_effort is None:
            raise RuntimeError(
                "Auto Tune requires both computed_effort and applied_effort to measure actuator clipping."
            )
        self.send(
            EventKind.AUTOTUNE_STARTED,
            {
                "resolved_configuration": self.config.to_dict(),
                "search_strategy": self.controller.strategy.name,
                "search_budget": self.config.search_budget,
                "original_gains": self.original_gain.to_dict(),
            },
        )
        self._start_next_candidate(simulation_time)

    def pause(self) -> None:
        if self.active and not self.complete:
            self.paused = True
            self.send(EventKind.AUTOTUNE_PROGRESS, {"state": "paused", **self.progress_payload()})

    def resume(self) -> None:
        if self.active and not self.complete:
            self.paused = False
            self.send(EventKind.AUTOTUNE_PROGRESS, {"state": "running", **self.progress_payload()})

    def cancel(self, simulation_time: float) -> None:
        if not self.active or self.complete:
            return
        self.cancelled = True
        if self.current_candidate is not None:
            cancelled = CandidateResult(
                candidate=self.current_candidate,
                direction=self.config.direction.value,
                trials=list(self.current_trials),
                status=CandidateStatus.CANCELLED,
            )
            self.controller.add_result(cancelled)
        self.outcome = self.controller.finalize()
        self.outcome.cancelled = True
        self.outcome.selected = None
        self.outcome.selection_reason = "Auto Tune was cancelled by the user."
        self._restore_start_state(simulation_time)
        self._finish(self.outcome)

    def _reset_physics_state(self) -> None:
        """Reset actual q/qd, actuator buffers, and all held targets between trials."""

        self.robot.write_joint_state_to_sim(self.base_positions, self.base_velocities)
        self.target_positions.copy_(self.base_positions)
        self.target_velocities.zero_()
        self.robot.set_joint_position_target(self.target_positions)
        self.robot.set_joint_velocity_target(self.target_velocities)
        self.robot.reset()

    def _apply_candidate(self, candidate: CandidateSpec) -> list[str]:
        failures: list[str] = []
        acknowledgements = self.adapter.apply_gains(
            self.config.joint_index,
            candidate.kp,
            candidate.kd,
            self.config.effort_limit,
        )
        for acknowledgement in acknowledgements:
            self.send(EventKind.GAIN_APPLIED, acknowledgement.to_dict())
        applied = self.adapter.read_gains(self.config.joint_index)
        for name, requested, confirmed in (
            ("Kp", candidate.kp, applied.stiffness),
            ("Kd", candidate.kd, applied.damping),
            ("effort_limit", self.config.effort_limit, applied.effort_limit),
        ):
            tolerance = max(1.0e-8, abs(requested) * 1.0e-6)
            if not math.isclose(requested, confirmed, rel_tol=0.0, abs_tol=tolerance):
                failures.append(
                    f"gain readback mismatch for {name}: requested {requested:.9g}, applied {confirmed:.9g}"
                )
        return failures

    def _start_next_candidate(self, simulation_time: float) -> None:
        candidate = self.controller.next_candidate()
        if candidate is None:
            outcome = self.controller.finalize()
            self._restore_start_state(simulation_time)
            self._finish(outcome)
            return
        self.current_candidate = candidate
        self.current_trials = []
        self.trial_plan_index = 0
        self.abort_candidate_after_return = False
        failures = self._apply_candidate(candidate)
        self.send(
            EventKind.AUTOTUNE_CANDIDATE_STARTED,
            {
                "candidate": candidate.to_dict(),
                "candidate_number": self.controller.tested_count + 1,
                "search_budget": self.config.search_budget,
                "search_stage": self.controller.search_stage,
            },
        )
        if failures:
            result = CandidateResult(
                candidate=candidate,
                direction=self.config.direction.value,
                status=CandidateStatus.HARD_CONSTRAINT_FAILED,
                hard_constraint_passed=False,
                hard_failure_reasons=failures,
            )
            self._complete_candidate(result, simulation_time)
            return
        self._prepare_trial(simulation_time)

    def _prepare_trial(self, simulation_time: float) -> None:
        self._reset_physics_state()
        self.current_samples = []
        self.phase = "stabilizing"
        self.phase_start_time = simulation_time
        self.trial_start_position = self.config.q0

    def _current_trial(self) -> tuple[str, int, int]:
        return self.trial_plan[self.trial_plan_index]

    def before_physics_step(self, simulation_time: float) -> None:
        """Set all position targets immediately before the normal write/sim step."""

        if not self.active or self.complete or self.paused:
            return
        self.target_positions.copy_(self.base_positions)
        direction_name, _sign, _repeat = self._current_trial()
        if self.phase == "stabilizing":
            if simulation_time - self.phase_start_time + 1.0e-12 >= self.config.stabilization_duration:
                actual = float(self.robot.data.joint_pos[0, self.config.joint_index].item())
                velocity = float(self.robot.data.joint_vel[0, self.config.joint_index].item())
                reset_position_tolerance = max(self.config.settling_tolerance, 1.0e-4)
                reset_velocity_tolerance = max(0.01, 0.01 * self.config.maximum_velocity)
                reset_ready = (
                    math.isfinite(actual)
                    and math.isfinite(velocity)
                    and abs(actual - self.config.q0) <= reset_position_tolerance
                    and abs(velocity) <= reset_velocity_tolerance
                )
                if reset_ready:
                    self.phase = "measuring"
                    self.measurement_start_time = simulation_time
                    self.trial_start_position = actual
                    self.current_samples = []
                elif simulation_time - self.phase_start_time >= max(
                    1.0, 4.0 * self.config.stabilization_duration
                ):
                    assert self.current_candidate is not None
                    result = CandidateResult(
                        candidate=self.current_candidate,
                        direction=self.config.direction.value,
                        status=CandidateStatus.HARD_CONSTRAINT_FAILED,
                        hard_constraint_passed=False,
                        hard_failure_reasons=[
                            "candidate-independent q0/zero-velocity reset did not stabilize: "
                            f"position error={abs(actual - self.config.q0):.6g} rad, "
                            f"velocity={abs(velocity):.6g} rad/s"
                        ],
                    )
                    self._complete_candidate(result, simulation_time)
                    return
            else:
                self.target_positions[0, self.config.joint_index] = self.config.q0
        if self.phase == "measuring":
            self.target_positions[0, self.config.joint_index] = self.config.applied_targets[direction_name]
        elif self.phase == "returning":
            self.target_positions[0, self.config.joint_index] = self.config.q0

    def after_physics_step(
        self,
        simulation_time: float,
        *,
        actual_position: float,
        joint_velocity: float,
        computed_effort: float | None,
        applied_effort: float | None,
    ) -> None:
        """Collect the outward response and advance trial/candidate phases."""

        if not self.active or self.complete or self.paused:
            return
        direction_name, _sign, repeat_index = self._current_trial()
        if self.phase == "measuring":
            elapsed = max(simulation_time - self.measurement_start_time, 0.0)
            sample = TrialSample(
                elapsed_time=elapsed,
                target_position=self.config.applied_targets[direction_name],
                actual_position=actual_position,
                joint_velocity=joint_velocity,
                computed_effort=computed_effort,
                applied_effort=applied_effort,
            )
            self.current_samples.append(sample)
            immediate_failures = hard_failure_reasons_for_sample(
                sample,
                effort_limit=self.config.effort_limit,
                torque_match_tolerance=self.config.torque_match_tolerance,
                maximum_velocity=self.config.maximum_velocity,
                lower_limit=self.config.lower_limit,
                upper_limit=self.config.upper_limit,
                allow_torque_saturation=(
                    self.config.torque_policy is AutoTuneTorquePolicy.ALLOW_CLIPPING
                ),
            )
            if immediate_failures or elapsed + 1.0e-12 >= self.config.hold_duration:
                trial = evaluate_trial(
                    self.current_samples,
                    direction=direction_name,
                    repeat_index=repeat_index,
                    start_position=self.trial_start_position,
                    requested_target=self.config.requested_targets[direction_name],
                    applied_target=self.config.applied_targets[direction_name],
                    target_clamped=self.config.target_clamped[direction_name],
                    settling_tolerance=self.config.settling_tolerance,
                    settling_hold_time=self.config.settling_hold_time,
                    effort_limit=self.config.effort_limit,
                    torque_match_tolerance=self.config.torque_match_tolerance,
                    maximum_velocity=self.config.maximum_velocity,
                    lower_limit=self.config.lower_limit,
                    upper_limit=self.config.upper_limit,
                    allow_torque_saturation=(
                        self.config.torque_policy is AutoTuneTorquePolicy.ALLOW_CLIPPING
                    ),
                )
                self.current_trials.append(trial)
                self.abort_candidate_after_return = bool(immediate_failures)
                self.phase = "returning"
                self.phase_start_time = simulation_time
        elif self.phase == "returning":
            if simulation_time - self.phase_start_time + 1.0e-12 >= self.config.return_duration:
                if self.abort_candidate_after_return:
                    self._finish_current_candidate(simulation_time)
                else:
                    self.trial_plan_index += 1
                    if self.trial_plan_index >= len(self.trial_plan):
                        self._finish_current_candidate(simulation_time)
                    else:
                        self._prepare_trial(simulation_time)

        if simulation_time - self.last_progress_time >= 0.10:
            self.last_progress_time = simulation_time
            self.send(EventKind.AUTOTUNE_PROGRESS, self.progress_payload(simulation_time))

    def _finish_current_candidate(self, simulation_time: float) -> None:
        assert self.current_candidate is not None
        result = CandidateResult.aggregate(
            self.current_candidate,
            self.config.direction.value,
            self.current_trials,
        )
        self._complete_candidate(result, simulation_time)

    def _complete_candidate(self, result: CandidateResult, simulation_time: float) -> None:
        self.controller.add_result(result)
        provisional = self.controller.finalize()
        rejected = sum(not item.hard_constraint_passed for item in provisional.candidates)
        self.send(
            EventKind.AUTOTUNE_CANDIDATE_RESULT,
            {
                "result": result.to_dict(),
                "tested_count": self.controller.tested_count,
                "search_budget": self.config.search_budget,
                "best_feasible_candidate_id": (
                    provisional.best_feasible.candidate.candidate_id
                    if provisional.best_feasible
                    else None
                ),
                "best_fallback_candidate_id": (
                    provisional.best_fallback.candidate.candidate_id
                    if provisional.best_fallback
                    else None
                ),
                "rejected_candidate_count": rejected,
            },
        )
        self._start_next_candidate(simulation_time)

    def _restore_start_state(self, simulation_time: float) -> None:
        acknowledgements = self.adapter.apply_gains(
            self.config.joint_index,
            self.original_gain.stiffness,
            self.original_gain.damping,
            self.original_gain.effort_limit,
        )
        for acknowledgement in acknowledgements:
            self.send(EventKind.GAIN_APPLIED, acknowledgement.to_dict())
        self._reset_physics_state()
        self.phase_start_time = simulation_time

    def _finish(self, outcome: AutoTuneOutcome) -> None:
        self.outcome = outcome
        self.active = False
        self.complete = True
        self.phase = "cancelled" if outcome.cancelled else "complete"
        self.send(
            EventKind.AUTOTUNE_COMPLETE,
            {
                "resolved_configuration": self.config.to_dict(),
                "original_gains": self.original_gain.to_dict(),
                "outcome": outcome.to_dict(),
            },
        )

    def progress_payload(self, simulation_time: float | None = None) -> dict[str, Any]:
        direction_name, _sign, repeat_index = self._current_trial()
        provisional = self.controller.finalize() if self.controller.results else None
        return {
            "state": "paused" if self.paused else self.phase,
            "candidate_number": self.controller.tested_count + 1,
            "search_budget": self.config.search_budget,
            "search_stage": self.controller.search_stage,
            "current_candidate": self.current_candidate.to_dict() if self.current_candidate else None,
            "direction": direction_name,
            "repeat": repeat_index + 1,
            "repeats": self.config.repeats,
            "elapsed_simulation_time": simulation_time,
            "best_feasible_candidate_id": (
                provisional.best_feasible.candidate.candidate_id
                if provisional and provisional.best_feasible
                else None
            ),
            "best_fallback_candidate_id": (
                provisional.best_fallback.candidate.candidate_id
                if provisional and provisional.best_fallback
                else None
            ),
            "rejected_candidate_count": sum(
                not result.hard_constraint_passed for result in self.controller.results
            ),
        }
