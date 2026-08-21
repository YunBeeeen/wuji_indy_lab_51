"""Floating-root controller for the ``hand_move`` task (2026-08-05).

This module is additive: ``hand_grasp`` is a validated baseline and no existing
file or shared action term is modified to support ``hand_move``.  It lives in
the ``hand_grasp`` package next to ``hand_move_env_cfg.py``, mirroring how
``hand_setting`` is organised.

The existing project-wide floating-base term
``isaac_neuromeka.mdp.actions.base_actions.FloatingBaseVelocityAction`` was
inspected first and cannot be reused here for three reasons:

1. its action layout is a mobile-base ``[Vx, Wz]`` pair, not a 3D rotational
   command with translation held;
2. it levels roll/pitch toward zero, which would fight any commanded hand
   rotation;
3. it calls ``set_joint_position_target(...)`` on every physics step, which
   would overwrite the 20D finger residual action of ``hand_move``.

The wrench-based control strategy (external force/torque on the root body, no
per-step ``write_root_pose_to_sim`` teleport) is taken from that class, so this
term follows the project's existing precedent.

Frames and conventions
----------------------
``palm_link`` is the URDF root link of the Wuji hand (verified: it is the only
link that is never a joint child).  Therefore the articulation root pose *is*
the palm pose and no extra offset is needed.

Control law, evaluated every physics step:

    p_err = p_target - p_root                      (world)
    F_w   = m * (Kp_pos * p_err - Kd_pos * v_root)

    q_target <- dq(omega_cmd * dt) * q_target      (world-frame integration)
    e_w   = axis_angle(q_target * conj(q_root))    (world)
    T_w   = I * (Kp_rot * e_w - Kd_rot * omega_root)

The gains are specified as accelerations (``1/s^2`` and ``1/s``) and are
multiplied by the asset's mass / effective inertia, so they stay meaningful if
the asset changes.  Both wrenches are rotated into the root link frame and
submitted with ``is_global=False``; that avoids IsaacLab's wrench composer
caching the link pose of the *first* call and reusing it for a rotating body.

Attitude target modes
---------------------
``rotation_action_dim`` stays at ``0`` for this task: the root is never driven
by the policy, and the action vector must remain 20D.  The attitude target
comes from one of two sources:

* ``orientation_command_name`` set (what ``hand_move`` uses): the target is the
  scripted ``q_cmd`` published by :class:`HandMoveRootOrientationCommand`.  No
  angular-velocity integration happens in this mode, so nothing can accumulate
  on top of the SLERP trajectory and drift away from it.
* ``orientation_command_name`` unset: the original integrating behaviour,
  where ``debug_angular_velocity`` (or a 3D policy action, if
  ``rotation_action_dim == 3``) is integrated into the target.  Kept for
  diagnostics and for a possible future policy-controlled root.

Either way the PD controller and the wrench write run on **every physics step
for the whole episode**.  There is no phase-dependent early return: when the
root has converged the computed wrench is simply near zero, and that near-zero
value is still written into the persistent buffer.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
)


class HandRootHoldAction(ActionTerm):
    """Hold the floating hand root in place while tracking an angular command."""

    cfg: HandRootHoldActionCfg

    def __init__(self, cfg: HandRootHoldActionCfg, env):
        super().__init__(cfg, env)
        self.cfg = cfg
        self._robot: Articulation = env.scene[cfg.asset_name]

        if cfg.rotation_action_dim not in (0, 3):
            raise ValueError(
                "HandRootHoldActionCfg.rotation_action_dim must be 0 (hold only)"
                f" or 3 (policy-controlled), received {cfg.rotation_action_dim}."
            )

        self._physics_dt = float(env.physics_dt)

        # Total articulation mass, read from the simulation rather than the URDF
        # so the controller stays correct if the asset is retuned.
        self._total_mass = float(
            self._robot.root_physx_view.get_masses()[0].sum().item()
        )

        self._raw_actions = torch.zeros(
            (self.num_envs, cfg.rotation_action_dim), device=self.device
        )
        # Commanded world-frame angular velocity actually applied this step.
        self._angular_velocity_command = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        # Owned buffer (not an expanded view) so a diagnostic script can write
        # into it at runtime via ``set_debug_angular_velocity``.
        self._debug_angular_velocity = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._debug_angular_velocity[:] = torch.tensor(
            cfg.debug_angular_velocity, device=self.device, dtype=torch.float
        )

        # Hold targets, (re)captured at reset from the post-event root state.
        self._target_root_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_root_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._target_root_quat_w[:, 0] = 1.0
        # Pose captured at the last reset.  Manual keyboard control is bounded
        # relative to this, and "R" restores it.
        self._start_root_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._start_root_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._start_root_quat_w[:, 0] = 1.0

        # Resolve the root link index instead of assuming 0.  The floating-base
        # overlay moves PhysicsArticulationRootAPI onto ``palm_link``, so the
        # link ordering is worth reading rather than guessing.
        root_body_index = self._robot.find_bodies([cfg.root_body_name])[0][0]
        self._root_body_ids = torch.tensor(
            [root_body_index], device=self.device, dtype=torch.long
        )

        # Scripted-target mode: the attitude target comes from a command term
        # instead of being integrated from an angular velocity.  The command
        # manager is built *before* the action manager
        # (``rl_task_custom_env.py:47-55``), so this resolves at construction
        # time; the lazy branch in ``_scripted_target_quat`` only exists as a
        # guard for a future reordering.
        # Opt-in physics-rate diagnostic buffer.  Costs nothing when disabled.
        self._trace_count = 0
        self._trace = torch.zeros(
            (cfg.trace_capacity if cfg.trace_enabled else 0, len(self.TRACE_COLUMNS)),
            device=self.device,
        )

        # Optional scripted *position* target, mirroring the orientation one.
        # Unset (hand_move) keeps the original behaviour exactly: the position
        # target is captured at reset and then only the keyboard moves it.
        self._position_command_term = None
        self._orientation_command_term = None
        if cfg.orientation_command_name is not None:
            command_manager = getattr(env, "command_manager", None)
            if command_manager is not None:
                self._orientation_command_term = command_manager.get_term(
                    cfg.orientation_command_name
                )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self.cfg.rotation_action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """World-frame angular velocity command in rad/s."""
        return self._angular_velocity_command

    @property
    def target_root_pos_w(self) -> torch.Tensor:
        """Position the controller is holding the root at (diagnostics)."""
        return self._target_root_pos_w

    @property
    def target_root_quat_w(self) -> torch.Tensor:
        """Integrated target orientation of the root (diagnostics)."""
        return self._target_root_quat_w

    def set_debug_angular_velocity(self, angular_velocity: Sequence[float]) -> None:
        """Overwrite the open-loop angular command used when the action is inert.

        Diagnostics only.  Has no effect once ``rotation_action_dim == 3``,
        where the command comes from the policy instead.
        """
        self._debug_angular_velocity[:] = torch.as_tensor(
            angular_velocity, device=self.device, dtype=self._debug_angular_velocity.dtype
        )

    """
    Operations.
    """

    def _resolve_env_ids(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None
    ) -> torch.Tensor | slice:
        if env_ids is None or isinstance(env_ids, slice):
            return slice(None)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Capture the reset root pose as the hold target.

        ``ManagerBasedRLEnv._reset_idx`` applies the ``reset``-mode events before
        calling ``action_manager.reset``, so the root pose read here is already
        the one written by ``reset_scene_to_default``.
        """
        env_ids = self._resolve_env_ids(env_ids)
        self._raw_actions[env_ids] = 0.0
        self._angular_velocity_command[env_ids] = 0.0
        self._target_root_pos_w[env_ids] = self._robot.data.root_link_pos_w[env_ids].clone()
        self._target_root_quat_w[env_ids] = self._robot.data.root_link_quat_w[env_ids].clone()
        self._start_root_pos_w[env_ids] = self._target_root_pos_w[env_ids]
        self._start_root_quat_w[env_ids] = self._target_root_quat_w[env_ids]

    def process_actions(self, actions: torch.Tensor) -> None:
        """Convert the policy output (if any) into an angular velocity command."""
        if self.cfg.rotation_action_dim == 3:
            self._raw_actions[:] = actions
            command = torch.clamp(actions, -1.0, 1.0) * self.cfg.max_angular_velocity
        else:
            command = self._debug_angular_velocity
        self._angular_velocity_command[:] = command

    @property
    def start_root_pos_w(self) -> torch.Tensor:
        """Root position captured at the last reset."""
        return self._start_root_pos_w

    @property
    def start_root_quat_w(self) -> torch.Tensor:
        """Root orientation captured at the last reset."""
        return self._start_root_quat_w

    def set_target_position(
        self,
        target_pos_w: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Set the world-frame position target directly.

        Environments not listed in ``env_ids`` keep their current target, so
        this is safe under partial reset.  Nothing else writes this buffer
        except :meth:`reset`, so a manual target survives until it is changed
        again or the episode ends.
        """
        env_ids = self._resolve_env_ids(env_ids)
        position = torch.as_tensor(
            target_pos_w, device=self.device, dtype=self._target_root_pos_w.dtype
        )
        if position.ndim == 1:
            position = position.unsqueeze(0)
        self._target_root_pos_w[env_ids] = position

    def add_target_position_delta(
        self,
        delta_pos_w: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
        max_distance_from_start: float | None = None,
    ) -> None:
        """Move the position target by a world-frame increment.

        This is what keyboard translation drives: it never touches the actual
        root pose, only the PD target.  ``max_distance_from_start`` bounds the
        target to a sphere around the pose captured at reset so a held key
        cannot walk the target away indefinitely.
        """
        env_ids = self._resolve_env_ids(env_ids)
        delta = torch.as_tensor(
            delta_pos_w, device=self.device, dtype=self._target_root_pos_w.dtype
        )
        if delta.ndim == 1:
            delta = delta.unsqueeze(0)
        updated = self._target_root_pos_w[env_ids] + delta
        if max_distance_from_start is not None:
            offset = updated - self._start_root_pos_w[env_ids]
            offset = self._clamp_norm(offset, max_distance_from_start)
            updated = self._start_root_pos_w[env_ids] + offset
        self._target_root_pos_w[env_ids] = updated

    def set_target_orientation(
        self,
        target_quat_w: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Set the world-frame attitude target directly.

        Batched, per-environment and normalizing.  Environments not listed in
        ``env_ids`` keep their current target, which makes this safe under
        partial reset.  Used by diagnostics; in normal operation the target is
        pulled from the orientation command term every physics step.
        """
        env_ids = self._resolve_env_ids(env_ids)
        quat = torch.as_tensor(
            target_quat_w, device=self.device, dtype=self._target_root_quat_w.dtype
        )
        if quat.ndim == 1:
            quat = quat.unsqueeze(0)
        norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp(min=1.0e-9)
        self._target_root_quat_w[env_ids] = quat / norm

    def apply_actions(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None = None
    ) -> None:
        """Refresh the attitude target and apply the hold wrench.

        Runs on every physics step for the whole episode.  There is no early
        return anywhere in this method: the PD output - zero included - is
        written into the persistent wrench buffer every single time, so a stale
        non-zero wrench can never be left behind.
        """
        del env_ids  # the controller always runs on every environment

        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        root_lin_vel_w = self._robot.data.root_link_lin_vel_w
        root_ang_vel_w = self._robot.data.root_link_ang_vel_w

        self._refresh_target_orientation()
        self._refresh_target_position()

        # -- position hold
        position_error = self._target_root_pos_w - root_pos_w
        position_error = self._clamp_norm(position_error, self.cfg.max_position_error)
        force_w = self._total_mass * (
            self.cfg.position_kp * position_error - self.cfg.position_kd * root_lin_vel_w
        )
        force_w = self._clamp_norm(force_w, self.cfg.max_force)

        # -- attitude tracking
        orientation_error = self._orientation_error(root_quat_w)
        orientation_error = self._clamp_norm(
            orientation_error, self.cfg.max_orientation_error
        )
        torque_w = self.cfg.effective_inertia * (
            self.cfg.orientation_kp * orientation_error
            - self.cfg.orientation_kd * root_ang_vel_w
        )
        torque_w = self._clamp_norm(torque_w, self.cfg.max_torque)

        # The wrench composer caches link poses, so submit link-frame values.
        force_b = quat_apply_inverse(root_quat_w, force_w).unsqueeze(1)
        torque_b = quat_apply_inverse(root_quat_w, torque_w).unsqueeze(1)
        self._write_root_wrench(force_b, torque_b)

        if self.cfg.trace_enabled:
            self._record_trace(
                orientation_error, root_ang_vel_w, torque_w, position_error, root_lin_vel_w
            )

    """
    Physics-rate trace (diagnostics only).
    """

    # Column layout of the trace buffer.  Kept as a class attribute so the
    # probe never has to guess indices.
    TRACE_COLUMNS = (
        "orientation_error_x", "orientation_error_y", "orientation_error_z",
        "ang_vel_x", "ang_vel_y", "ang_vel_z",
        "torque_x", "torque_y", "torque_z",
        "position_error_x", "position_error_y", "position_error_z",
        "lin_vel_x", "lin_vel_y", "lin_vel_z",
        "finger_joint_vel_norm",
        # World-frame stick angular velocities.  Recorded so a probe can
        # reconstruct exactly what ``object_pair_angular_speed_excess_l2``
        # sees, ``|w_stick - w_palm|``, and measure how much of that penalty
        # is palm jitter rather than the sticks actually moving in the hand.
        "stick1_ang_vel_x", "stick1_ang_vel_y", "stick1_ang_vel_z",
        "stick2_ang_vel_x", "stick2_ang_vel_y", "stick2_ang_vel_z",
    )

    @property
    def trace(self) -> torch.Tensor:
        """Recorded samples so far, shape ``(n, len(TRACE_COLUMNS))``.

        One row per **physics** step, not per policy step.  That distinction is
        the whole point: an instability that flips sign every physics step is
        invisible at the 30 Hz policy rate because of aliasing.
        """
        return self._trace[: self._trace_count]

    def reset_trace(self) -> None:
        self._trace_count = 0

    def _record_trace(
        self,
        orientation_error: torch.Tensor,
        ang_vel: torch.Tensor,
        torque: torch.Tensor,
        position_error: torch.Tensor,
        lin_vel: torch.Tensor,
    ) -> None:
        if self._trace_count >= self._trace.shape[0]:
            return
        env_id = self.cfg.trace_env_id
        row = self._trace[self._trace_count]
        row[0:3] = orientation_error[env_id]
        row[3:6] = ang_vel[env_id]
        row[6:9] = torque[env_id]
        row[9:12] = position_error[env_id]
        row[12:15] = lin_vel[env_id]
        # Lets the probe tell "the palm is being shaken by the fingers" apart
        # from "the root controller is shaking the palm".
        row[15] = torch.linalg.vector_norm(self._robot.data.joint_vel[env_id])
        scene = self._env.scene
        row[16:19] = scene["stick1"].data.root_ang_vel_w[env_id]
        row[19:22] = scene["stick2"].data.root_ang_vel_w[env_id]
        self._trace_count += 1

    """
    Internals.
    """

    def _refresh_target_orientation(self) -> None:
        """Update the attitude target for this physics step.

        Two mutually exclusive modes:

        * ``orientation_command_name`` set - *scripted target*.  The target is
          copied from the command term's ``q_cmd``.  No angular-velocity
          integration is performed at all, so nothing can accumulate on top of
          the scripted trajectory and drift away from it.
        * ``orientation_command_name`` unset - the original behaviour: the
          target is integrated from an angular-velocity command.
        """
        scripted_target = self._scripted_target_quat()
        if scripted_target is not None:
            self._target_root_quat_w[:] = scripted_target
            return
        self._integrate_target_orientation()

    def _refresh_target_position(self) -> None:
        """Copy the command term's scripted position target, if there is one.

        Without ``position_command_name`` this does nothing at all, so
        ``hand_move`` - which holds the root where it was reset and lets only
        the keyboard move it - is bit-for-bit unaffected.

        ``hand_object`` sets it, because reaching a cube at a fixed place needs
        the hand to translate as well as rotate; holding the position would
        leave the tips swinging on an arc that never arrives.
        """
        if self.cfg.position_command_name is None:
            return
        if self._position_command_term is None:
            command_manager = getattr(self._env, "command_manager", None)
            if command_manager is None:
                return
            self._position_command_term = command_manager.get_term(
                self.cfg.position_command_name
            )
        target = getattr(self._position_command_term, "target_pos_w", None)
        if target is not None:
            self._target_root_pos_w[:] = target

    def _scripted_target_quat(self) -> torch.Tensor | None:
        """Return the command term's ``q_cmd``, or ``None`` in integrate mode."""
        if self.cfg.orientation_command_name is None:
            return None
        if self._orientation_command_term is None:
            command_manager = getattr(self._env, "command_manager", None)
            if command_manager is None:
                return None
            self._orientation_command_term = command_manager.get_term(
                self.cfg.orientation_command_name
            )
        return self._orientation_command_term.target_quat_w

    def _integrate_target_orientation(self) -> None:
        """Advance the attitude target by the commanded world angular velocity."""
        half_angle = 0.5 * self._angular_velocity_command * self._physics_dt
        angle = torch.linalg.vector_norm(half_angle, dim=-1, keepdim=True)
        # sinc-style expansion keeps the zero-command case exact and gradient-free.
        scale = torch.where(
            angle > 1.0e-9,
            torch.sin(angle) / angle.clamp(min=1.0e-9),
            torch.ones_like(angle),
        )
        delta_quat = torch.cat((torch.cos(angle), scale * half_angle), dim=-1)
        target = quat_mul(delta_quat, self._target_root_quat_w)
        self._target_root_quat_w[:] = target / torch.linalg.vector_norm(
            target, dim=-1, keepdim=True
        ).clamp(min=1.0e-9)

    def _orientation_error(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        """World-frame rotation vector taking the root onto its target."""
        error_quat = quat_mul(self._target_root_quat_w, quat_conjugate(root_quat_w))
        # ``axis_angle_from_quat`` already resolves the quaternion double cover,
        # so the returned rotation vector is always the short way round.
        return axis_angle_from_quat(error_quat)

    @staticmethod
    def _clamp_norm(vectors: torch.Tensor, limit: float) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        scale = torch.where(
            norm > limit, limit / norm.clamp(min=1.0e-9), torch.ones_like(norm)
        )
        return vectors * scale

    def _write_root_wrench(
        self, force_b: torch.Tensor, torque_b: torch.Tensor
    ) -> None:
        """Submit the root wrench through the non-deprecated composer if present."""
        composer = getattr(self._robot, "permanent_wrench_composer", None)
        if composer is not None:
            composer.set_forces_and_torques(
                forces=force_b,
                torques=torque_b,
                body_ids=self._root_body_ids,
                is_global=False,
            )
            return
        # Older Isaac Lab releases only expose the deprecated setter.
        self._robot.set_external_force_and_torque(
            force_b, torque_b, body_ids=self._root_body_ids.tolist(), is_global=False
        )


@configclass
class HandRootHoldActionCfg(ActionTermCfg):
    """Configuration for :class:`HandRootHoldAction`.

    The default configuration is deliberately inert with respect to the policy:
    ``rotation_action_dim=0`` keeps the ``hand_move`` action vector identical to
    ``hand_grasp`` (20D) while still floating the root under closed-loop hold.
    """

    class_type: type[ActionTerm] = HandRootHoldAction

    asset_name: str = "robot"

    # Root link of the articulation.  ``palm_link`` is the URDF root: it is the
    # only link that never appears as a joint child.
    root_body_name: str = "palm_link"

    # 0 -> hold only, no policy action consumed.  ``hand_move`` must keep this
    # at 0: the root is driven by a scripted trajectory, not by the policy, and
    # the action vector has to stay 20D.
    # 3 -> policy commands a world-frame angular velocity.  Unused by this task.
    rotation_action_dim: int = 0

    # Name of the command term supplying the scripted world-frame attitude
    # target.  When set, the term tracks that quaternion and performs no
    # angular-velocity integration.  ``None`` restores the integrating mode.
    orientation_command_name: str | None = None

    # Name of a command term exposing ``target_pos_w``; when set, the position
    # target follows that scripted trajectory instead of being held at the pose
    # captured on reset.  ``hand_move`` leaves this None.
    position_command_name: str | None = None

    # Commanded angular velocity at |action| = 1.  Deliberately small: the first
    # target motion is a ~5 deg step, not a fast slew.
    max_angular_velocity: float = 0.3

    # Open-loop world-frame angular velocity used when rotation_action_dim == 0.
    # Non-zero values are for diagnostics only.
    debug_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Position hold gains, in acceleration units (1/s^2 and 1/s).  400/40 is a
    # critically damped 20 rad/s response; the hand spawns with gravity disabled
    # so the only steady load is the ~0.2 N weight of the two 10 g sticks.
    position_kp: float = 400.0
    position_kd: float = 10.0

    # Attitude gains, same units, multiplied by ``effective_inertia``.
    #
    # kd was 40 for runs up to 2026-08-06_00-12-28, which made the explicit
    # damping term numerically unstable: the applied torque flipped sign on
    # nearly every physics step.  Measured on that checkpoint over the
    # post-convergence window (t = 4..15 s, 1320 physics samples), kd only:
    #
    #                          kd = 40      kd = 10
    #     torque sign-flip       67.7 %       0.5 %
    #     torque lag-1 r        -0.762      -0.140
    #     palm |w| rms / peak  0.51/2.34   0.36/1.59  rad/s
    #     orientation error     0.0162      0.0101    rad rms
    #
    # Lowering kd removed the alternating component *and* improved tracking,
    # because the error was dominated by the oscillation itself.  Keep
    # ``effective_inertia * orientation_kd`` at or below about 0.03; the
    # analytic limit derived from the palm-only inertia predicted the x axis
    # would still be stable at 0.1 and was wrong, so trust that measured bound
    # rather than recomputing one.  Tracking authority is set by kp, not kd.
    #
    # What remains after this change is *not* the controller: the torque is
    # smooth (flip 0.5 %) while the palm angular velocity still alternates
    # (flip 47 %, ~28 Hz), i.e. finger motion and stick contact are shaking a
    # wrist-less floating hand.  Separate problem, roughly half the amplitude.
    orientation_kp: float = 400.0
    orientation_kd: float = 10.0

    # Effective rotational inertia of the hand about the palm root [kg m^2].
    # URDF estimate: palm izz/iyy/ixx ~ 1e-4..2e-4 plus the finger links at a
    # 5-10 cm radius over the remaining 0.35 kg.  Treat as a first value.
    effective_inertia: float = 2.5e-3

    # Saturations.  The error clamps keep a large disturbance from producing an
    # impulsive correction; the wrench clamps bound what the root can exert.
    max_position_error: float = 0.05
    max_orientation_error: float = 0.5
    max_force: float = 20.0
    max_torque: float = 2.0

    # Physics-rate diagnostic trace.  Off by default; see
    # ``scripts/debug/hand_move_root_vibration_probe.py``.
    trace_enabled: bool = False
    trace_env_id: int = 0
    trace_capacity: int = 8192
