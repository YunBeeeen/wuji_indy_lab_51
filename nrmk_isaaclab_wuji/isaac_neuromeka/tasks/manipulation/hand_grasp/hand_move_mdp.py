"""Task-local MDP terms for ``hand_move`` (2026-08-05).

Everything here is additive: no ``hand_grasp`` file, function or class is
modified.  ``hand_move`` starts from the validated ``hand_grasp`` functional
grasp, rotates the floating hand root to a random goal orientation, and then
performs the OPEN/CLOSE sequence at that new orientation.

Responsibility split (kept strictly separate on purpose):

1. :class:`HandMoveRootOrientationCommand` - *trajectory generator*.  It owns
   ``q_start`` / ``q_goal`` / ``q_cmd`` and computes ``q_cmd`` from the episode
   elapsed time.  It never writes to the simulation.
2. ``HandRootHoldAction`` (``hand_move_root_actions.py``) - *PD controller*.
   Every physics step it reads the latest ``q_cmd`` and the captured
   ``p_start`` and produces a wrench.  It runs for the whole episode.
3. The same action term is the *wrench writer*: it writes the PD output into
   the PhysX permanent wrench buffer on every physics step, including when the
   output is zero.

Only the SLERP interpolation finishes at ``t = 4 s``.  ``q_cmd`` is then pinned
to ``q_goal`` and the controller keeps running until the episode ends.

Manager lifecycle facts this module relies on (verified in the installed
Isaac Lab, not assumed):

* ``ManagerBasedRLEnv._reset_idx`` applies ``reset``-mode events *before*
  ``command_manager.reset``.  Capturing the root pose inside
  ``_resample_command`` therefore sees the post-``reset_to_functional_pregrasp``
  state, never a stale one.
* ``scene.reset`` (which runs before the events) calls
  ``Articulation.reset`` -> ``permanent_wrench_composer.reset``, so the
  persistent external wrench is already zeroed on reset.
* ``CommandManager`` is built *before* ``ActionManager``
  (``rl_task_custom_env.py:47-55``), so the action term can resolve this
  command term at construction time, but not the other way round.
* ``command_manager.compute`` runs at the *end* of ``step`` (after
  ``episode_length_buf += 1`` and after ``_reset_idx``).  The ``q_cmd``
  computed at the end of step ``N`` is the target used by the physics loop of
  step ``N+1``; that is one control period (1/30 s) of lag, which the PD
  controller absorbs.

Quaternion convention (verified against ``isaaclab.utils.math``): ``(w, x, y,
z)``.  ``quat_from_euler_xyz(roll, pitch, yaw)`` builds
``q = q_z(yaw) * q_y(pitch) * q_x(roll)``.
"""

# 학습순서
# 1. grasp $ open-close -> 1000 iter
# 2. rotate
# 3. translate
# 4. noise

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import (
    CommandTerm,
    CommandTermCfg,
    ManagerTermBase,
    SceneEntityCfg,
)
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)

from .mdp import _group_forces

# OPEN/CLOSE one-hot encoding inherited from hand_grasp's OpenCloseModeCommand:
# index 0 = OPEN ([1, 0]), index 1 = CLOSE ([0, 1]).
OPEN_INDEX = 0
CLOSE_INDEX = 1


@configclass
class HandMoveScheduleCfg:
    """Single place to tune the whole ``hand_move`` episode script.

    Both ``hand_move`` command terms take their configuration from the module
    level :data:`HAND_MOVE_SCHEDULE`, so the timing used by the orientation
    trajectory and by the OPEN/CLOSE scheduler comes from one source.

    Note that ``configclass`` deep-copies mutable defaults into each config
    instance, so the two terms end up holding *equal copies* rather than the
    same object.  Edit :data:`HAND_MOVE_SCHEDULE` in this module (or override
    both terms) to retune; changing one term's copy at runtime will not
    propagate to the other.  ``validate()`` is called on both copies at
    construction, which catches an inconsistent timing decomposition.
    """

    # -- Relative Euler goal range, in radians, about the *hand's own* axes.
    #    x = relative roll, y = relative pitch, z = relative yaw.
    #    Reduce these to make the task easier, e.g. pitch +-5 deg:
    #        range_x = (0.0, 0.0)
    #        range_y = (-0.0872665, 0.0872665)
    #        range_z = (0.0, 0.0)
    #    NOTE: sampling the three Euler angles independently is *not* a uniform
    #    distribution over SO(3) (it concentrates near the gimbal axis).  That
    #    is accepted here on purpose: per-axis difficulty control matters more
    #    at this stage than distribution uniformity.  Uniform quaternion
    #    sampling is intentionally not implemented.

    range_x: tuple[float, float] = (0, 0)
    range_y: tuple[float, float] = (0, 0)
    range_z: tuple[float, float] = (0, 0)

    #range_x: tuple[float, float] = (0, 0)
    #range_y: tuple[float, float] = (0, math.pi/4)
    #range_z: tuple[float, float] = (-math.pi/4, 0)

    # -- Timing, in seconds.  Phases are derived from these at runtime using
    #    the environment's own ``step_dt``; no step counts are hard-coded.
    initial_hold_time_s: float = 2.0
    rotation_interpolation_time_s: float = 2.0
    rotation_settling_time_s: float = 1.0
    open_close_start_time_s: float = 5.0

    open_close_segment_time_s: float = 2.0
    num_open_close_segments: int = 5
# curriculum learning : epi_length = 5.0 -> grasp 학습 / 15.0 & range(0,0) -> close-open 학습 / 15.0 & range -> move 학습
    episode_length_s: float = 15.0

    # -- SLERP
    use_slerp: bool = True
    use_smoothstep: bool = True

    # -- OPEN/CLOSE
    first_mode_open_probability: float = 0.5

    # -- Diagnostics only.  Never a reward, a termination or a phase gate.
    orientation_convergence_tolerance: float = 0.0872664626  # 5 deg
    angular_speed_convergence_tolerance: float = 0.1

    # -- Visualization
    debug_vis: bool = True
    visualize_current_frame: bool = True
    visualize_goal_frame: bool = True
    # The command frame only differs from the goal frame during the 2 s SLERP
    # window; from the settling phase onwards the two coincide.  Turn it on when
    # the question is "is the controller lagging or is the trajectory still
    # moving?", i.e. when tuning the root PD gains.
    visualize_command_frame: bool = False
    visualize_start_frame: bool = False
    debug_env_ids: tuple[int, ...] = (0,)

    # Axis length of the *current* frame marker, in metres.  The Wuji hand
    # spans roughly 0.15 m, so Isaac Lab's ``FRAME_MARKER_CFG`` default of 0.5
    # (a half-metre triad) buries it completely.  The other frames are drawn at
    # a fraction of this, see ``_FRAME_SCALE_FACTORS``.
    frame_marker_scale: float = 0.04
    # Vertical spacing between the frames, in metres.  Translation only: the
    # drawn orientations are never modified.
    #
    # Do not set this to 0.0 with the size factors below: the frame_prim axes
    # are opaque meshes radiating from the origin, so a smaller triad sharing
    # the origin with a larger one is simply buried inside it and cannot be
    # seen.  A few centimetres of separation is what makes the goal frame
    # readable next to the current one.
    frame_marker_offset_step: float = 0.03

    def validate(self) -> None:
        """Fail loudly on an inconsistent schedule instead of at 3 a.m."""
        for name in ("range_x", "range_y", "range_z"):
            bounds = getattr(self, name)
            if len(bounds) != 2:
                raise ValueError(f"HandMoveScheduleCfg.{name} must have exactly 2 entries, got {bounds!r}.")
            if bounds[0] > bounds[1]:
                raise ValueError(f"HandMoveScheduleCfg.{name} minimum exceeds maximum: {bounds!r}.")

        expected_start = (
            self.initial_hold_time_s
            + self.rotation_interpolation_time_s
            + self.rotation_settling_time_s
        )
        if abs(self.open_close_start_time_s - expected_start) > 1.0e-9:
            raise ValueError(
                "HandMoveScheduleCfg: open_close_start_time_s must equal "
                "initial_hold_time_s + rotation_interpolation_time_s + rotation_settling_time_s "
                f"({self.open_close_start_time_s} != {expected_start})."
            )

        expected_length = (
            self.open_close_start_time_s
            + self.open_close_segment_time_s * self.num_open_close_segments
        )
        if abs(self.episode_length_s - expected_length) > 1.0e-9:
            raise ValueError(
                "HandMoveScheduleCfg: episode_length_s must equal open_close_start_time_s + "
                "open_close_segment_time_s * num_open_close_segments "
                f"({self.episode_length_s} != {expected_length})."
            )

        if self.rotation_interpolation_time_s <= 0.0:
            raise ValueError("HandMoveScheduleCfg.rotation_interpolation_time_s must be positive.")
        if self.num_open_close_segments < 1:
            raise ValueError("HandMoveScheduleCfg.num_open_close_segments must be at least 1.")

    """
    Derived phase boundaries.
    """

    @property
    def slerp_end_time_s(self) -> float:
        return self.initial_hold_time_s + self.rotation_interpolation_time_s


# Shared schedule instance.  Edit this one object to retune the task.
HAND_MOVE_SCHEDULE = HandMoveScheduleCfg()
HAND_MOVE_SCHEDULE.validate()

# Per-frame marker size, relative to ``frame_marker_scale``.  Different sizes
# keep the frames tellable apart when they coincide.  The dict order is also the
# stacking order: enabled frames are offset upwards by consecutive multiples of
# ``frame_marker_offset_step``, so disabling one closes the gap instead of
# leaving a hole.
_FRAME_SCALE_FACTORS = {"current": 1.0, "goal": 0.85, "command": 0.6, "start": 0.4}


def quat_slerp_batch(
    quat_start: torch.Tensor,
    quat_end: torch.Tensor,
    alpha: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Batched shortest-path SLERP between two ``(w, x, y, z)`` quaternions.

    Isaac Lab ships ``isaaclab.utils.math.quat_slerp``, but its docstring
    states it "does not support batch processing", it takes a Python float for
    ``tau`` and it mutates its second argument in place.  This task-local
    implementation is therefore used instead.

    Args:
        quat_start: Start quaternions, shape ``(N, 4)``.
        quat_end: End quaternions, shape ``(N, 4)``.
        alpha: Interpolation coefficient in ``[0, 1]``, shape ``(N, 1)``.
        eps: Threshold below which the normalized linear interpolation
            fallback is used instead of dividing by ``sin(theta)``.

    Returns:
        Interpolated unit quaternions, shape ``(N, 4)``.
    """
    quat_start = _normalize_quat(quat_start)
    quat_end = _normalize_quat(quat_end)

    dot = torch.sum(quat_start * quat_end, dim=-1, keepdim=True)
    # Quaternion double cover: q and -q are the same rotation.  Flip one input
    # so the interpolation always takes the short way round.
    quat_end = torch.where(dot < 0.0, -quat_end, quat_end)
    dot = dot.abs().clamp(max=1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    # Guard the division; the result is discarded wherever ``nearly_parallel``.
    safe_sin_theta = sin_theta.clamp(min=eps)

    weight_start = torch.sin((1.0 - alpha) * theta) / safe_sin_theta
    weight_end = torch.sin(alpha * theta) / safe_sin_theta

    # Near-parallel (or identical) quaternions: fall back to NLERP, which is
    # numerically stable and indistinguishable from SLERP at small angles.
    nearly_parallel = sin_theta < eps
    weight_start = torch.where(nearly_parallel, 1.0 - alpha, weight_start)
    weight_end = torch.where(nearly_parallel, alpha, weight_end)

    return _normalize_quat(weight_start * quat_start + weight_end * quat_end)


def _normalize_quat(quat: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp(min=eps)


def _quat_geodesic_angle(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
    """Shortest-path angle between two quaternions, in radians. Shape ``(N,)``."""
    delta = quat_mul(quat_a, quat_conjugate(quat_b))
    return 2.0 * torch.acos(delta[:, 0].abs().clamp(max=1.0))


class HandMoveRootOrientationCommand(CommandTerm):
    """Scripted world-frame orientation target for the floating hand root.

    Timeline (``t`` is the per-environment episode elapsed time):

    ==================  ===========================================
    ``t``               ``q_cmd``
    ==================  ===========================================
    ``[0, 2)``          ``q_start``
    ``[2, 4)``          ``SLERP(q_start, q_goal, smoothstep(alpha))``
    ``[4, 15)``         ``q_goal``
    ==================  ===========================================

    With ``use_slerp = True`` this falls out of a single expression: ``alpha``
    is clamped to ``[0, 1]``, so it is exactly ``0`` before ``t = 2`` and
    exactly ``1`` from ``t = 4`` onwards.  ``q_cmd`` is therefore never
    "switched off" - it stays pinned at ``q_goal`` for the rest of the episode,
    including the whole OPEN/CLOSE sequence.

    The term only *computes* a target.  It never writes root poses or
    velocities into the simulation.
    """

    cfg: HandMoveRootOrientationCommandCfg

    def __init__(self, cfg: HandMoveRootOrientationCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.cfg.schedule.validate()
        self._robot: Articulation = env.scene[cfg.asset_name]

        num_envs, device = self.num_envs, self.device
        identity = torch.zeros((num_envs, 4), device=device)
        identity[:, 0] = 1.0

        self._quat_start = identity.clone()
        self._quat_goal = identity.clone()
        self._quat_command = identity.clone()
        self._position_start = torch.zeros((num_envs, 3), device=device)
        self._delta_euler = torch.zeros((num_envs, 3), device=device)

        self._alpha = torch.zeros((num_envs, 1), device=device)
        self._phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        # -1 keeps the first update from spuriously latching a phase boundary.
        self._previous_elapsed = torch.full((num_envs,), -1.0, device=device)

        for name in (
            "goal_delta_roll",
            "goal_delta_pitch",
            "goal_delta_yaw",
            "goal_geodesic_angle",
            "orientation_error_current",
            "orientation_error_at_slerp_end",
            "orientation_error_at_open_close_start",
            "root_angular_speed_at_slerp_end",
            "root_angular_speed_at_open_close_start",
            "reached_slerp_end_fraction",
            "reached_open_close_start_fraction",
            "converged_by_open_close_start",
            "command_tracking_error",
            "drop_during_initial_hold",
            "drop_during_slerp",
            "drop_during_settling",
            "drop_during_open_close",
        ):
            self.metrics[name] = torch.zeros(num_envs, device=device)

        self._debug_env_ids = torch.as_tensor(
            list(cfg.schedule.debug_env_ids), device=device, dtype=torch.long
        ).clamp(max=max(num_envs - 1, 0))

        # Manual play only; training never touches this.
        self._manual_override = False

    def __str__(self) -> str:
        schedule = self.cfg.schedule
        return (
            "HandMoveRootOrientationCommand:\n"
            f"\thold {schedule.initial_hold_time_s}s"
            f" -> slerp {schedule.rotation_interpolation_time_s}s"
            f" -> settle {schedule.rotation_settling_time_s}s"
            f" -> open/close {schedule.open_close_segment_time_s}s"
            f" x {schedule.num_open_close_segments}\n"
            f"\tuse_slerp={schedule.use_slerp}, use_smoothstep={schedule.use_smoothstep}\n"
            f"\trelative euler range x={schedule.range_x} y={schedule.range_y} z={schedule.range_z}"
        )

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """World-frame orientation target ``q_cmd``. Shape ``(N, 4)``.

        This is *not* part of the policy observation; the observation group
        only consumes the 2D OPEN/CLOSE one-hot.
        """
        return self._quat_command

    @property
    def target_quat_w(self) -> torch.Tensor:
        """Alias used by the root action term."""
        return self._quat_command

    @property
    def start_quat_w(self) -> torch.Tensor:
        return self._quat_start

    @property
    def goal_quat_w(self) -> torch.Tensor:
        return self._quat_goal

    @property
    def start_pos_w(self) -> torch.Tensor:
        return self._position_start

    @property
    def delta_euler(self) -> torch.Tensor:
        """Sampled relative ``(roll, pitch, yaw)`` in radians. Shape ``(N, 3)``."""
        return self._delta_euler

    @property
    def alpha(self) -> torch.Tensor:
        """Current interpolation coefficient (after smoothstep). Shape ``(N, 1)``."""
        return self._alpha

    @property
    def phase(self) -> torch.Tensor:
        """0 = initial hold, 1 = SLERP, 2 = settling, 3 = OPEN/CLOSE."""
        return self._phase

    @property
    def elapsed_time(self) -> torch.Tensor:
        return self._env.episode_length_buf.float() * self._env.step_dt

    """
    Manual play override.
    """

    @property
    def manual_override(self) -> bool:
        return self._manual_override

    def enable_manual_override(self, enabled: bool = True) -> None:
        """Hand ``q_cmd`` over to an external controller (keyboard play).

        While enabled, ``_resample_command`` samples no goal and
        ``_update_command`` does nothing, so nothing in the environment writes
        ``_quat_command`` any more.  ``HandRootHoldAction`` keeps pulling that
        same buffer every physics step, so the PD controller and the persistent
        wrench update carry on exactly as in training.

        Training never calls this: the flag defaults to False and is only
        flipped by the manual play controller.
        """
        self._manual_override = bool(enabled)
        if self._manual_override:
            self._quat_command[:] = _normalize_quat(
                self._robot.data.root_link_quat_w.clone()
            )
            self._quat_goal[:] = self._quat_command

    def set_manual_target_quat(
        self, quat_w: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set the manual world-frame orientation target (normalized)."""
        ids = slice(None) if env_ids is None else env_ids
        quat = torch.as_tensor(quat_w, device=self.device, dtype=self._quat_command.dtype)
        if quat.ndim == 1:
            quat = quat.unsqueeze(0)
        self._quat_command[ids] = _normalize_quat(quat)
        self._quat_goal[ids] = self._quat_command[ids]

    def apply_manual_local_rotation(
        self, delta_axis_angle: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Rotate the manual target by an increment in its own (palm) frame.

        ``q_new = q_old (x) dq_local``: right-multiplying applies the increment
        in the target's local frame, so the roll/pitch/yaw keys always act on
        the hand's own x/y/z axes no matter where it has been rotated to.
        Built as an incremental quaternion rather than by accumulating Euler
        angles, so there is nothing to wrap and no gimbal lock.
        """
        ids = slice(None) if env_ids is None else env_ids
        delta = torch.as_tensor(
            delta_axis_angle, device=self.device, dtype=self._quat_command.dtype
        )
        if delta.ndim == 1:
            delta = delta.unsqueeze(0)
        half = 0.5 * delta
        angle = torch.linalg.vector_norm(half, dim=-1, keepdim=True)
        scale = torch.where(
            angle > 1.0e-9,
            torch.sin(angle) / angle.clamp(min=1.0e-9),
            torch.ones_like(angle),
        )
        delta_quat = torch.cat((torch.cos(angle), scale * half), dim=-1)
        self._quat_command[ids] = _normalize_quat(
            quat_mul(self._quat_command[ids], delta_quat)
        )
        self._quat_goal[ids] = self._quat_command[ids]

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Attribute stick drops to a phase, then run the standard reset.

        ``CommandTerm.reset`` logs the mean of every metric and zeroes it, so
        the drop attribution has to be written *before* delegating.  The
        termination flags computed earlier in this same step are still valid
        here: ``_reset_idx`` runs after ``termination_manager.compute``.
        """
        self._record_drop_phase(env_ids)
        return super().reset(env_ids)

    def _record_drop_phase(self, env_ids: Sequence[int] | None) -> None:
        termination_manager = getattr(self._env, "termination_manager", None)
        if termination_manager is None:
            return
        try:
            dropped = termination_manager.get_term("stick1_dropped") | termination_manager.get_term(
                "stick2_dropped"
            )
        except (KeyError, AttributeError):
            return

        ids = slice(None) if env_ids is None else env_ids
        phase = self._phase
        for index, name in enumerate(
            ("drop_during_initial_hold", "drop_during_slerp", "drop_during_settling", "drop_during_open_close")
        ):
            self.metrics[name][ids] = (dropped & (phase == index)).float()[ids]

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Capture the post-reset root pose and sample one relative goal.

        Called from ``CommandTerm.reset`` which ``_reset_idx`` invokes after
        the ``reset``-mode events, so ``root_link_pos_w`` / ``root_link_quat_w``
        already reflect ``reset_scene_to_default`` +
        ``reset_to_functional_pregrasp``.  ``resampling_time_range`` is set to
        an effectively infinite interval, so this only ever runs on reset.
        """
        count = len(env_ids)
        if count == 0:
            return
        schedule = self.cfg.schedule

        self._position_start[env_ids] = self._robot.data.root_link_pos_w[env_ids].clone()
        quat_start = _normalize_quat(self._robot.data.root_link_quat_w[env_ids].clone())
        self._quat_start[env_ids] = quat_start

        if self._manual_override:
            # Manual play: no goal is sampled and no trajectory is generated.
            # The command is re-seeded to the freshly reset pose so the operator
            # starts from the functional grasp again after every reset, and
            # ``_update_command`` then leaves it alone.
            self._quat_goal[env_ids] = quat_start
            self._quat_command[env_ids] = quat_start
            self._delta_euler[env_ids] = 0.0
            self._alpha[env_ids] = 0.0
            self._phase[env_ids] = 0
            self._previous_elapsed[env_ids] = -1.0
            return

        delta = torch.empty((count, 3), device=self.device)
        for axis, bounds in enumerate((schedule.range_x, schedule.range_y, schedule.range_z)):
            delta[:, axis].uniform_(float(bounds[0]), float(bounds[1]))
        self._delta_euler[env_ids] = delta

        quat_delta = quat_from_euler_xyz(delta[:, 0], delta[:, 1], delta[:, 2])
        # q_goal = q_start (x) q_delta  <=>  R_goal = R_start @ R_delta, i.e. the
        # sampled rotation is applied in the hand's own frame, not the world's.
        quat_goal = _normalize_quat(quat_mul(quat_start, quat_delta))
        self._quat_goal[env_ids] = quat_goal
        self._quat_command[env_ids] = quat_start

        self._alpha[env_ids] = 0.0
        self._phase[env_ids] = 0
        self._previous_elapsed[env_ids] = -1.0

        self.metrics["goal_delta_roll"][env_ids] = delta[:, 0]
        self.metrics["goal_delta_pitch"][env_ids] = delta[:, 1]
        self.metrics["goal_delta_yaw"][env_ids] = delta[:, 2]
        self.metrics["goal_geodesic_angle"][env_ids] = _quat_geodesic_angle(quat_goal, quat_start)

    def _update_command(self) -> None:
        """Recompute ``q_cmd`` for every environment from the elapsed time."""
        if self._manual_override:
            # The keyboard owns ``_quat_command`` in manual play.  Returning
            # here is what stops the scripted SLERP from overwriting it on the
            # next step; ``HandRootHoldAction`` keeps reading the same buffer,
            # so the PD controller and the wrench write are untouched.
            self._phase[:] = 0
            return
        schedule = self.cfg.schedule
        elapsed = self.elapsed_time

        alpha_raw = torch.clamp(
            (elapsed - schedule.initial_hold_time_s) / schedule.rotation_interpolation_time_s,
            min=0.0,
            max=1.0,
        )
        if schedule.use_smoothstep:
            alpha = alpha_raw * alpha_raw * (3.0 - 2.0 * alpha_raw)
        else:
            alpha = alpha_raw
        self._alpha[:] = alpha.unsqueeze(-1)

        if schedule.use_slerp:
            # alpha is exactly 0 before the hold ends and exactly 1 from the
            # SLERP end onwards, so this single expression already produces
            # q_start / interpolation / q_goal for the three phases.
            self._quat_command[:] = quat_slerp_batch(
                self._quat_start, self._quat_goal, self._alpha
            )
        else:
            use_goal = (elapsed >= schedule.initial_hold_time_s).unsqueeze(-1)
            self._quat_command[:] = torch.where(use_goal, self._quat_goal, self._quat_start)

        self._phase[:] = self._compute_phase(elapsed)

    def _compute_phase(self, elapsed: torch.Tensor) -> torch.Tensor:
        schedule = self.cfg.schedule
        phase = torch.zeros_like(elapsed, dtype=torch.long)
        phase = torch.where(elapsed >= schedule.initial_hold_time_s, torch.ones_like(phase), phase)
        phase = torch.where(
            elapsed >= schedule.slerp_end_time_s, torch.full_like(phase, 2), phase
        )
        phase = torch.where(
            elapsed >= schedule.open_close_start_time_s, torch.full_like(phase, 3), phase
        )
        return phase

    def _update_metrics(self) -> None:
        schedule = self.cfg.schedule
        elapsed = self.elapsed_time

        root_quat = self._robot.data.root_link_quat_w
        root_angular_speed = torch.linalg.vector_norm(
            self._robot.data.root_link_ang_vel_w, dim=-1
        )
        error_to_goal = _quat_geodesic_angle(self._quat_goal, root_quat)
        error_to_command = _quat_geodesic_angle(self._quat_command, root_quat)

        self.metrics["orientation_error_current"][:] = error_to_goal
        self.metrics["command_tracking_error"][:] = error_to_command

        converged = (error_to_goal < schedule.orientation_convergence_tolerance) & (
            root_angular_speed < schedule.angular_speed_convergence_tolerance
        )

        # Latch the diagnostic snapshots the first time each boundary is crossed.
        for boundary, error_key, speed_key, reached_key in (
            (
                schedule.slerp_end_time_s,
                "orientation_error_at_slerp_end",
                "root_angular_speed_at_slerp_end",
                "reached_slerp_end_fraction",
            ),
            (
                schedule.open_close_start_time_s,
                "orientation_error_at_open_close_start",
                "root_angular_speed_at_open_close_start",
                "reached_open_close_start_fraction",
            ),
        ):
            crossing = (self._previous_elapsed < boundary) & (elapsed >= boundary)
            if not bool(crossing.any()):
                continue
            self.metrics[error_key][crossing] = error_to_goal[crossing]
            self.metrics[speed_key][crossing] = root_angular_speed[crossing]
            self.metrics[reached_key][crossing] = 1.0
            if reached_key == "reached_open_close_start_fraction":
                self.metrics["converged_by_open_close_start"][crossing] = converged[crossing].float()

        self._previous_elapsed[:] = elapsed

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        schedule = self.cfg.schedule
        if debug_vis:
            if not hasattr(self, "_frame_visualizers"):
                self._frame_visualizers = {}
                self._frame_offsets = {}
                enabled_frames = {
                    "current": schedule.visualize_current_frame,
                    "goal": schedule.visualize_goal_frame,
                    "command": schedule.visualize_command_frame,
                    "start": schedule.visualize_start_frame,
                }
                slot = 0
                for name, factor in _FRAME_SCALE_FACTORS.items():
                    if not enabled_frames[name]:
                        continue
                    scale = schedule.frame_marker_scale * factor
                    marker_cfg = FRAME_MARKER_CFG.copy()
                    marker_cfg.prim_path = f"/Visuals/hand_move/{name}_frame"
                    # Keep only the axis triad; the default cfg also carries a
                    # 1 m "connecting_line" cylinder that is not used here.
                    marker_cfg.markers = {"frame": marker_cfg.markers["frame"]}
                    marker_cfg.markers["frame"].scale = (scale, scale, scale)
                    self._frame_visualizers[name] = VisualizationMarkers(marker_cfg)
                    self._frame_offsets[name] = slot * schedule.frame_marker_offset_step
                    slot += 1
            for visualizer in self._frame_visualizers.values():
                visualizer.set_visibility(True)
        elif hasattr(self, "_frame_visualizers"):
            for visualizer in self._frame_visualizers.values():
                visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        # ``CommandTerm.__init__`` registers this callback before the subclass
        # buffers exist, so every attribute is probed defensively.
        robot = getattr(self, "_robot", None)
        env_ids = getattr(self, "_debug_env_ids", None)
        if robot is None or env_ids is None or not hasattr(self, "_frame_visualizers"):
            return
        if not robot.is_initialized or env_ids.numel() == 0:
            return

        root_pos = robot.data.root_link_pos_w[env_ids]
        # Translation-only offsets.  The quaternions themselves are never
        # altered, so what is drawn is exactly the orientation being tracked.
        quaternions = {
            "current": robot.data.root_link_quat_w[env_ids],
            "goal": self._quat_goal[env_ids],
            "command": self._quat_command[env_ids],
            "start": self._quat_start[env_ids],
        }
        for name, visualizer in self._frame_visualizers.items():
            translations = root_pos + self._z_offset(self._frame_offsets[name])
            visualizer.visualize(
                translations=translations, orientations=quaternions[name]
            )

    def _z_offset(self, height: float) -> torch.Tensor:
        offset = torch.zeros((1, 3), device=self.device)
        offset[0, 2] = height
        return offset


@configclass
class HandMoveRootOrientationCommandCfg(CommandTermCfg):
    """Configuration for :class:`HandMoveRootOrientationCommand`."""

    class_type: type = HandMoveRootOrientationCommand

    asset_name: str = "robot"
    schedule: HandMoveScheduleCfg = HAND_MOVE_SCHEDULE

    # The goal is sampled once per episode, on reset only.  An effectively
    # infinite resampling interval is the project's existing idiom for this.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = HAND_MOVE_SCHEDULE.debug_vis


class HandMoveOpenCloseCommand(CommandTerm):
    """Scripted OPEN/CLOSE schedule for ``hand_move``.

    Unlike ``hand_grasp``'s ``OpenCloseModeCommand`` (which resamples on a
    timer), this term is a deterministic function of the elapsed time:

    * ``t < open_close_start_time_s``: OPEN by default.  A task-local config
      may instead emit neutral ``[0, 0]`` for a reference/contact-only stage.
    * ``t >= open_close_start_time_s``: ``num_open_close_segments`` segments of
      ``open_close_segment_time_s`` each.  The first segment's mode is sampled
      per episode; every following segment is the opposite of the previous one.

    The switch at ``open_close_start_time_s`` is unconditional: orientation
    convergence is *never* used as a gate.
    """

    cfg: HandMoveOpenCloseCommandCfg

    def __init__(self, cfg: HandMoveOpenCloseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.cfg.schedule.validate()
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self._command[:, OPEN_INDEX] = 1.0
        self._first_mode_open = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._segment = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        for name in (
            "open_fraction",
            "close_fraction",
            "neutral_fraction",
            "first_mode_open_fraction",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

        # Manual play only; training never touches this.
        self._manual_override = False
        self._manual_mode_index = CLOSE_INDEX

    def __str__(self) -> str:
        schedule = self.cfg.schedule
        return (
            "HandMoveOpenCloseCommand:\n"
            f"\tOPEN until {schedule.open_close_start_time_s}s, then "
            f"{schedule.num_open_close_segments} x {schedule.open_close_segment_time_s}s alternating"
        )

    @property
    def command(self) -> torch.Tensor:
        """OPEN/CLOSE one-hot, ``[1, 0]`` = OPEN and ``[0, 1]`` = CLOSE."""
        return self._command

    @property
    def segment(self) -> torch.Tensor:
        return self._segment

    @property
    def first_mode_open(self) -> torch.Tensor:
        return self._first_mode_open

    """
    Manual play override.
    """

    @property
    def manual_override(self) -> bool:
        return self._manual_override

    @property
    def manual_mode_index(self) -> int:
        return self._manual_mode_index

    def enable_manual_override(self, enabled: bool = True, mode_index: int = CLOSE_INDEX) -> None:
        """Let the keyboard pick OPEN/CLOSE instead of the 2 s schedule.

        The one-hot encoding and the command name are unchanged, so the policy
        observation, the reward terms and ``OpenCloseModeHeld`` all bind to it
        exactly as in training.  Only *who decides the mode* changes.
        """
        self._manual_override = bool(enabled)
        if self._manual_override:
            self.set_manual_mode(mode_index)

    def set_manual_mode(self, mode_index: int) -> None:
        """Latch a mode; it is held until the next key press."""
        self._manual_mode_index = OPEN_INDEX if int(mode_index) == OPEN_INDEX else CLOSE_INDEX
        self._command[:, OPEN_INDEX] = float(self._manual_mode_index == OPEN_INDEX)
        self._command[:, CLOSE_INDEX] = float(self._manual_mode_index == CLOSE_INDEX)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        count = len(env_ids)
        if count == 0:
            return
        if self._manual_override:
            # Keep the operator's selection across resets instead of forcing
            # OPEN and re-sampling a first mode.
            self._command[env_ids, OPEN_INDEX] = float(self._manual_mode_index == OPEN_INDEX)
            self._command[env_ids, CLOSE_INDEX] = float(self._manual_mode_index == CLOSE_INDEX)
            self._segment[env_ids] = 0
            return
        first_open = (
            torch.rand(count, device=self.device) < self.cfg.schedule.first_mode_open_probability
        )
        self._first_mode_open[env_ids] = first_open
        # The sampled mode is used only after the rotation/settling window.
        # hand_real uses neutral [0, 0] before that boundary so its 5-second
        # curriculum learns reference/contact preservation without an OPEN
        # gap objective.  Other tasks retain the historical OPEN pre-phase.
        prephase_open = not self.cfg.neutral_before_open_close
        self._command[env_ids, OPEN_INDEX] = float(prephase_open)
        self._command[env_ids, CLOSE_INDEX] = 0.0
        self._segment[env_ids] = 0
        self.metrics["first_mode_open_fraction"][env_ids] = first_open.float()

    def _update_command(self) -> None:
        if self._manual_override:
            # The scripted 2 s alternation is what would otherwise overwrite the
            # keyboard selection on the very next step, so it is skipped entirely.
            return
        schedule = self.cfg.schedule
        elapsed = self._env.episode_length_buf.float() * self._env.step_dt

        in_open_close = elapsed >= schedule.open_close_start_time_s
        segment = torch.floor(
            (elapsed - schedule.open_close_start_time_s) / schedule.open_close_segment_time_s
        ).long()
        segment = segment.clamp(min=0, max=schedule.num_open_close_segments - 1)
        self._segment[:] = torch.where(in_open_close, segment, torch.zeros_like(segment))

        # Even segments keep the sampled first mode, odd segments invert it.
        alternated = self._first_mode_open ^ (self._segment % 2 == 1)
        # Before alternation, either keep the historical OPEN mode or emit the
        # explicit neutral [0, 0] requested by a task-local config.
        mode_open = (in_open_close & alternated) | (
            (~in_open_close) & (not self.cfg.neutral_before_open_close)
        )
        mode_close = in_open_close & (~alternated)

        self._command[:, OPEN_INDEX] = mode_open.float()
        self._command[:, CLOSE_INDEX] = mode_close.float()

    def _update_metrics(self) -> None:
        self.metrics["open_fraction"][:] = self._command[:, OPEN_INDEX]
        self.metrics["close_fraction"][:] = self._command[:, CLOSE_INDEX]
        self.metrics["neutral_fraction"][:] = 1.0 - torch.clamp(
            self._command.sum(dim=-1), min=0.0, max=1.0
        )


@configclass
class HandMoveOpenCloseCommandCfg(CommandTermCfg):
    """Configuration for :class:`HandMoveOpenCloseCommand`."""

    class_type: type = HandMoveOpenCloseCommand

    schedule: HandMoveScheduleCfg = HAND_MOVE_SCHEDULE

    # If true, emit [0, 0] before ``open_close_start_time_s``.  This is a
    # deliberate third semantic state, not a malformed one-hot: mode-specific
    # rewards and success are gated off while reference/contact rewards remain.
    neutral_before_open_close: bool = False

    # Deterministic function of elapsed time; the only sampling happens on
    # reset, so the timer must never fire.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False


def _euler_xyz_from_quat(quat: torch.Tensor) -> tuple[float, float, float]:
    """Inverse of ``quat_from_euler_xyz``: returns intrinsic (roll, pitch, yaw).

    ``quat_from_euler_xyz(r, p, y)`` builds ``q_z(y) * q_y(p) * q_x(r)``, so the
    decomposition below is the matching one.  Written out rather than imported
    because Isaac Lab's ``euler_xyz_from_quat`` returns angles wrapped to
    ``[0, 2pi)``, which prints a -5 deg rotation as +355 and is unusable as a
    number to paste into a config.
    """
    w, x, y, z = (float(v) for v in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def geometry_report(
    env: ManagerBasedRLEnv,
    stick1_cfg,
    stick2_cfg,
    tip_offset_o: tuple[float, float, float],
    orientation_command_name: str = "root_orientation",
    env_id: int = 0,
) -> str:
    """Root pose and stick-tip geometry for a manual play session.

    ``play.py --manual_root`` already prints the root position and quaternion
    every second, but two numbers that matter for placing an object cannot be
    read off that line by eye:

    * **the relative rotation from the reset pose, about the hand's own z.**
      That is the quantity the scripted trajectories take
      (``q_goal = q_start (x) q_delta``), so a calibrated angle has to be
      expressed in it - not as an absolute world quaternion.
    * **the distal tip midpoint.**  With the root held at its reset position by
      the PD controller and the joints at ``pose_005``, this is a deterministic
      function of that relative rotation, so once an angle has been chosen the
      place to put an object follows from it - nothing further to measure.

    Works in any task that has the two sticks.  ``hand_object`` prints a longer
    block that adds the cube, the support and the contact forces.
    """
    from isaaclab.utils.math import quat_apply

    index = int(env_id)
    origin = env.scene.env_origins[index]
    robot: Articulation = env.scene["robot"]

    stick1 = env.scene[stick1_cfg.name]
    stick2 = env.scene[stick2_cfg.name]
    offset = torch.as_tensor(
        tip_offset_o,
        dtype=stick1.data.root_pos_w.dtype,
        device=stick1.data.root_pos_w.device,
    ).expand(env.num_envs, -1)
    tip1 = stick1.data.root_pos_w + quat_apply(stick1.data.root_quat_w, offset)
    tip2 = stick2.data.root_pos_w + quat_apply(stick2.data.root_quat_w, offset)
    midpoint = 0.5 * (tip1[index] + tip2[index])
    delta = tip1[index] - tip2[index]
    gap = float(torch.linalg.vector_norm(delta))
    axis = delta / max(gap, 1.0e-8)

    orientation = env.command_manager.get_term(orientation_command_name)
    quat_start = orientation.start_quat_w[index]
    quat_now = robot.data.root_link_quat_w[index]
    relative = quat_mul(quat_conjugate(quat_start.unsqueeze(0)), quat_now.unsqueeze(0))[0]
    angle = 2.0 * math.acos(min(1.0, abs(float(relative[0]))))
    axis_len = float(torch.linalg.vector_norm(relative[1:]))
    relative_axis = relative[1:] / max(axis_len, 1.0e-8)
    # Signed about local z. Only meaningful when the rotation axis really is
    # local z, which is why the axis itself is printed right above it.
    signed_yaw = angle * (1.0 if float(relative_axis[2]) >= 0.0 else -1.0)

    root_local = robot.data.root_link_pos_w[index] - origin
    # The scripted trajectory takes the relative rotation as an intrinsic
    # (roll, pitch, yaw) triple, so decompose it the same way rather than
    # reporting only the single-axis angle - the operator is free to use all
    # three keys, and a pure-yaw readout would silently drop the rest.
    euler = _euler_xyz_from_quat(relative)

    def vec(tensor, sub=None) -> str:
        values = tensor - sub if sub is not None else tensor
        return "(" + ", ".join(f"{float(v):+.4f}" for v in values) + ")"

    return "\n".join(
        [
            f"--- hand geometry (env {index}, positions are env-local) " + "-" * 14,
            f"  root position            : {vec(robot.data.root_link_pos_w[index], origin)}",
            f"  root quaternion          : {vec(quat_now)}",
            f"  reset quaternion         : {vec(quat_start)}",
            f"  relative rotation        : {angle:.4f} rad ({math.degrees(angle):+.2f} deg)",
            (
                f"  relative rotation axis   : {vec(relative_axis)}"
                "   (pure local-z yaw 이면 ~(0,0,+-1))"
                if angle > 1.0e-4
                else "  relative rotation axis   : (회전 0 — 축 미정의)"
            ),
            f"  local-z signed yaw       : {signed_yaw:+.4f} rad "
            f"({math.degrees(signed_yaw):+.2f} deg)",
            "",
            "  === 여기서 멈췄다면 아래 두 줄을 hand_object_mdp.py 에 붙여넣기 ===",
            f"  HAND_OBJECT_TARGET_ROOT_POS_E = ({float(root_local[0]):.4f}, "
            f"{float(root_local[1]):.4f}, {float(root_local[2]):.4f})",
            f"  HAND_OBJECT_TARGET_EULER_RAD  = ({float(euler[0]):.4f}, "
            f"{float(euler[1]):.4f}, {float(euler[2]):.4f})"
            f"   # deg ({math.degrees(float(euler[0])):+.2f}, "
            f"{math.degrees(float(euler[1])):+.2f}, {math.degrees(float(euler[2])):+.2f})",
            "",
            f"  Stick1 tip               : {vec(tip1[index], origin)}",
            f"  Stick2 tip               : {vec(tip2[index], origin)}",
            f"  tip midpoint             : {vec(midpoint, origin)}"
            "   (여기가 큐브에 닿아야 함)",
            f"  tip gap                  : {gap * 1000:.2f} mm",
            f"  closing axis (2 -> 1)    : {vec(axis)}"
            f"   z성분 {float(axis[2]):+.3f} (0 에 가까울수록 수평)",
            "-" * 68,
        ]
    )


# ---------------------------------------------------------------------------
# Chopstick disturbance (2026-08-07).
#
# Purpose: the restoring rewards (``stick1_pivot`` w10, ``stick2_reference_pose``
# w15) already pay for "the stick is back where it belongs", and the policy can
# already *see* the error (14D palm-frame stick poses + 12D relative velocities).
# What training never produced was the **state** - nothing displaces a stick, so
# the recovery behaviour has no gradient to flow through.  This term creates
# those states and changes nothing else.
# ---------------------------------------------------------------------------


class StickDisturbance(ManagerTermBase):
    """One short transverse force pulse per episode on one randomly chosen stick.

    Why an ``interval`` event that fires every step
    ----------------------------------------------
    A pulse has a *duration*, so something has to run on every policy step to
    start it, hold it and stop it.  Of the event modes only ``interval`` runs
    per step; ``reset`` runs once and ``startup`` once ever.  Setting
    ``interval_range_s`` to (0, 0) makes ``EventManager.apply`` fire this term on
    every step for every environment (``event_manager.py:210-232``: the per-env
    ``time_left`` is decremented by ``dt`` and any environment at ``< 1e-6``
    fires, then resamples 0 again).

    The repeated firing is *only* a per-step hook - it is not the disturbance
    schedule.  Whether a pulse is running is decided by this term's own
    per-environment state against the episode clock, so the "at most once per
    episode" guarantee does not depend on the interval mechanism at all.  That is
    exactly what makes plain ``mode="interval"`` scheduling unusable here: it
    would resample a fresh interval after every firing and perturb repeatedly.

    External wrench lifecycle (verified against the installed Isaac Lab)
    -------------------------------------------------------------------
    * ``RigidObject.permanent_wrench_composer`` is a **persistent** buffer.
      ``write_data_to_sim`` re-applies whatever is in it before *every*
      simulation step (``rigid_object.py:134-152``), so a force written once
      keeps pushing until it is overwritten.  Nothing decays it.
    * ``set_external_force_and_torque`` is deprecated in this build - it logs a
      warning on every call and forwards to
      ``permanent_wrench_composer.set_forces_and_torques``
      (``rigid_object.py:442-467``).  This term calls the composer directly:
      the public property exists (``rigid_object.py:113``) and calling the
      deprecated wrapper once per step would flood the log.
    * Because the buffer persists, **the full force tensor is rewritten every
      step**, zeros included.  Ending a pulse is therefore not a separate code
      path that could be missed - the state machine simply stops marking the
      environment active and the next write puts zeros there.
    * ``RigidObject.reset(env_ids)`` also clears the composer
      (``rigid_object.py:126-132``), so a reset cannot leave a force behind
      either.  Both mechanisms are in place; neither relies on the other.

    Ordering note: interval events run *after* the physics decimation of the
    current policy step (``manager_based_rl_env.py:233-235``), so a force
    written on step N first acts during step N+1.  One policy step of latency,
    identical for every environment, and irrelevant at these time scales.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self._stick1: RigidObject = env.scene[params["stick1_cfg"].name]
        self._stick2: RigidObject = env.scene[params["stick2_cfg"].name]
        self._sensor_groups = params["sensor_groups"]
        self._contact_threshold = float(params["contact_threshold"])
        self._time_range = tuple(params["time_range_s"])
        self._duration = float(params["duration_s"])
        self._force_range = tuple(params["force_range_n"])
        self._probability = float(params["probability"])

        if self._time_range[0] > self._time_range[1]:
            raise ValueError(
                "StickDisturbance.time_range_s must be ordered (low, high); got "
                f"{self._time_range}."
            )
        if self._duration <= 0.0:
            raise ValueError("StickDisturbance.duration_s must be positive.")
        if self._force_range[0] < 0.0 or self._force_range[0] > self._force_range[1]:
            raise ValueError(
                "StickDisturbance.force_range_n must be ordered and non-negative; "
                f"got {self._force_range}."
            )

        num_envs, device = env.num_envs, env.device
        zeros = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
        falses = lambda: torch.zeros(num_envs, dtype=torch.bool, device=device)  # noqa: E731

        # -- schedule, sampled once per episode in reset()
        self._scheduled = falses()          # this episode gets a pulse at all
        self._start_time = zeros()
        self._target_is_stick1 = falses()
        self._magnitude = zeros()
        self._random_dir = torch.zeros(num_envs, 3, device=device)

        # -- run state
        self._fired = falses()              # pulse has already started
        self._active = falses()             # pulse is running right now
        self._force_w = torch.zeros(num_envs, 3, device=device)

        # -- diagnostics (episode-latched, never used by reward/termination)
        self._contacts_before = zeros()
        self._min_contacts_after = zeros()
        self._recovered = zeros()
        self._recovery_time = zeros()

        self._local_y = torch.tensor((0.0, 1.0, 0.0), device=device).expand(num_envs, -1)
        self._zero_wrench = torch.zeros(num_envs, 1, 3, device=device)
        self._contact_count = zeros()       # previous step's count, for "before"

        # Guards the sim-side write in reset().  Manager terms are constructed
        # while the managers are being built; touching an asset's wrench buffer
        # then is not something the lifecycle promises is safe.  The first real
        # reset (the wrapper calls env.reset() right after construction) flips
        # this and does the write.
        self._ready = False
        self.reset()
        self._ready = True

    """
    Diagnostics, read by the metric block in ``isaac_neuromeka/env/managers.py``.
    """

    @property
    def applied(self) -> torch.Tensor:
        """1.0 where this episode's pulse has already started. Shape ``(N,)``."""
        return self._fired.float()

    @property
    def force_magnitude(self) -> torch.Tensor:
        return self._magnitude

    @property
    def target_is_stick1(self) -> torch.Tensor:
        return self._target_is_stick1.float()

    @property
    def contacts_before(self) -> torch.Tensor:
        return self._contacts_before

    @property
    def min_contacts_after(self) -> torch.Tensor:
        return self._min_contacts_after

    @property
    def recovered(self) -> torch.Tensor:
        return self._recovered

    @property
    def recovery_time_s(self) -> torch.Tensor:
        return self._recovery_time

    """
    Lifecycle.
    """

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Resample the schedule and guarantee no force survives into the episode."""
        if env_ids is None:
            env_ids = slice(None)
            count = self.num_envs
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device).long()
            count = int(env_ids.numel())
            if count == 0:
                return

        device = self.device
        rand = lambda: torch.rand(count, device=device)  # noqa: E731

        low, high = self._time_range
        self._scheduled[env_ids] = rand() < self._probability
        self._start_time[env_ids] = low + rand() * (high - low)
        self._target_is_stick1[env_ids] = rand() < 0.5
        f_low, f_high = self._force_range
        self._magnitude[env_ids] = f_low + rand() * (f_high - f_low)
        # Uniform on the sphere; the shaft component is projected out later, at
        # the instant the pulse starts, against the stick's pose *then*.
        direction = torch.randn(count, 3, device=device)
        self._random_dir[env_ids] = direction / torch.clamp(
            torch.linalg.vector_norm(direction, dim=-1, keepdim=True), min=1.0e-8
        )

        self._fired[env_ids] = False
        self._active[env_ids] = False
        self._force_w[env_ids] = 0.0

        self._contacts_before[env_ids] = 0.0
        self._min_contacts_after[env_ids] = float(len(self._sensor_groups))
        self._recovered[env_ids] = 0.0
        self._recovery_time[env_ids] = 0.0
        self._contact_count[env_ids] = 0.0

        # Clear the sim-side buffer for these environments as well.  The asset's
        # own reset already does this, but this term must not depend on the order
        # the two run in.
        if self._ready:
            self._write_wrenches()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        stick1_cfg: SceneEntityCfg,
        stick2_cfg: SceneEntityCfg,
        sensor_groups: tuple[tuple[str, ...], ...],
        contact_threshold: float,
        time_range_s: tuple[float, float],
        duration_s: float,
        force_range_n: tuple[float, float],
        probability: float,
    ) -> None:
        """Advance the per-environment pulse state machine by one policy step.

        ``env_ids`` is ignored on purpose.  The interval mechanism fires this
        term for whichever environments happened to hit zero, but the state
        machine is driven by each environment's own ``episode_length_buf`` and is
        cheap to evaluate for all of them.  Deriving the state from the clock
        rather than from *which call* arrived keeps the behaviour identical no
        matter how the event manager batches the firing.
        """
        del env_ids, stick1_cfg, stick2_cfg, sensor_groups, contact_threshold
        del time_range_s, duration_s, force_range_n, probability

        elapsed = env.episode_length_buf.float() * env.step_dt
        starting = self._scheduled & (~self._fired) & (elapsed >= self._start_time)
        if bool(starting.any()):
            self._begin_pulse(starting)

        self._active = (
            self._fired
            & (elapsed >= self._start_time)
            & (elapsed < self._start_time + self._duration)
        )
        self._write_wrenches()
        self._update_diagnostics(elapsed, starting)

    """
    Internal.
    """

    def _begin_pulse(self, starting: torch.Tensor) -> None:
        """Freeze the force vector for the environments whose pulse starts now.

        The shaft component is removed **here**, against the stick's orientation
        at this instant, so the pulse is transverse to where the shaft actually
        is rather than to where it was at reset.  A force along the shaft would
        pull the stick out of the hand lengthwise, which is a different failure
        from the slip/tilt this experiment is about.
        """
        quat = torch.where(
            self._target_is_stick1.unsqueeze(-1),
            self._stick1.data.root_quat_w,
            self._stick2.data.root_quat_w,
        )
        shaft = quat_apply(quat, self._local_y)

        direction = self._random_dir
        transverse = direction - (
            torch.sum(direction * shaft, dim=-1, keepdim=True) * shaft
        )
        norm = torch.linalg.vector_norm(transverse, dim=-1, keepdim=True)

        # Degenerate only when the sampled direction is (anti)parallel to the
        # shaft.  Probability ~0 for a uniform sphere sample, but a zero vector
        # here would normalise to NaN, so fall back to an explicit perpendicular:
        # cross with whichever world axis the shaft is least aligned to.
        world_z = torch.zeros_like(shaft)
        world_z[:, 2] = 1.0
        world_x = torch.zeros_like(shaft)
        world_x[:, 0] = 1.0
        alt = torch.where(shaft[:, 2:3].abs() < 0.9, world_z, world_x)
        fallback = torch.linalg.cross(shaft, alt, dim=-1)
        transverse = torch.where(norm < 1.0e-6, fallback, transverse)
        norm = torch.linalg.vector_norm(transverse, dim=-1, keepdim=True)
        transverse = transverse / torch.clamp(norm, min=1.0e-8)

        force = self._magnitude.unsqueeze(-1) * transverse
        self._force_w = torch.where(starting.unsqueeze(-1), force, self._force_w)
        self._fired = self._fired | starting

    def _write_wrenches(self) -> None:
        """Push the current force into both sticks' persistent buffers.

        Written every step for *all* environments, zeros included.  The composer
        keeps whatever it was last given and ``write_data_to_sim`` re-applies it
        before each physics step, so "stop pushing" has to be an explicit write -
        there is no expiry.  Rewriting unconditionally means the pulse can never
        outlive its window, and no separate teardown path exists to get wrong.

        ``is_global=True``: the force was built in world coordinates from the
        world-frame shaft axis.
        """
        active = self._active.unsqueeze(-1)
        force = torch.where(active, self._force_w, torch.zeros_like(self._force_w))
        on_stick1 = self._target_is_stick1.unsqueeze(-1)
        force1 = torch.where(on_stick1, force, torch.zeros_like(force)).unsqueeze(1)
        force2 = torch.where(on_stick1, torch.zeros_like(force), force).unsqueeze(1)

        for asset, wrench in ((self._stick1, force1), (self._stick2, force2)):
            asset.permanent_wrench_composer.set_forces_and_torques(
                forces=wrench,
                # Explicit zeros, not None: this experiment is translation only
                # (torque perturbation is a separate follow-up), and passing the
                # zeros makes sure no torque can be left over from anywhere else.
                torques=self._zero_wrench,
                env_ids=None,
                is_global=True,
            )

    def _update_diagnostics(self, elapsed: torch.Tensor, starting: torch.Tensor) -> None:
        """Latch the recovery statistics.  Never read by reward or termination."""
        forces = _group_forces(self._env, self._sensor_groups)
        count = (forces > self._contact_threshold).float().sum(dim=-1)
        full = float(len(self._sensor_groups))

        # "before" = the count carried over from the previous step.  Interval
        # events run after the physics of this step but the pulse only starts
        # acting on the *next* one, so either reading is pre-disturbance; the
        # previous step's is the more conservative of the two.
        self._contacts_before = torch.where(
            starting, self._contact_count, self._contacts_before
        )
        self._min_contacts_after = torch.where(
            starting, count, self._min_contacts_after
        )

        after = self._fired & (~starting)
        self._min_contacts_after = torch.where(
            after, torch.minimum(self._min_contacts_after, count), self._min_contacts_after
        )
        newly_recovered = after & (self._recovered < 0.5) & (count >= full - 1.0e-6)
        self._recovery_time = torch.where(
            newly_recovered, elapsed - self._start_time, self._recovery_time
        )
        self._recovered = torch.where(
            newly_recovered, torch.ones_like(self._recovered), self._recovered
        )

        self._contact_count = count
