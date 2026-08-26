"""Teacher-driven behavior distillation for the Wuji hand.

RSL-RL's stock :class:`Distillation` lets the untrained student drive the
environment.  That is appropriate for many locomotion tasks, but it destroys
the narrow chopstick grasp before useful 105D observations can be collected.
This variant keeps the frozen 103D teacher in control of the rollout while a
105D student learns the teacher's *physical residual command*.

The two actors do not use the same action scale.  The old teacher used a
uniform 0.1 rad residual; the deployable student uses the current per-joint
0.10/0.20/0.15 rad contract.  Teacher outputs are therefore converted before
they are both applied to the environment and stored as supervised targets::

    student_action = teacher_action * teacher_scale / student_scale

The resulting student checkpoint also exposes its weights under
``actor_state_dict``.  This is intentional: a later PPO run can load the 105D
student with the ordinary ``--init_checkpoint --load_actor_only`` path while
the distillation-only teacher/optimizer state remains available for resume.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation


class TeacherDrivenDistillation(Distillation):
    """Distillation whose frozen teacher, rather than the student, drives rollout."""

    def __init__(
        self,
        *args,
        teacher_action_scale_rad: float | Sequence[float],
        student_action_scale_rad: float | Sequence[float],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._teacher_action_scale = self._action_scale_tensor(
            teacher_action_scale_rad, "teacher_action_scale_rad"
        )
        self._student_action_scale = self._action_scale_tensor(
            student_action_scale_rad, "student_action_scale_rad"
        )
        if self._teacher_action_scale.shape != self._student_action_scale.shape:
            raise ValueError(
                "teacher and student action-scale vectors must have the same shape: "
                f"{tuple(self._teacher_action_scale.shape)} != "
                f"{tuple(self._student_action_scale.shape)}"
            )
        if self._teacher_action_scale.numel() != self.storage.actions_shape[0]:
            raise ValueError(
                "action-scale width does not match the environment action dimension: "
                f"{self._teacher_action_scale.numel()} != {self.storage.actions_shape[0]}"
            )
        self._teacher_to_student = (
            self._teacher_action_scale / self._student_action_scale
        )

    def _action_scale_tensor(
        self,
        value: float | Sequence[float],
        name: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == 0:
            tensor = tensor.repeat(self.storage.actions_shape[0])
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be a scalar or 1D sequence, got {tensor.shape}.")
        if not torch.all(torch.isfinite(tensor)) or torch.any(tensor <= 0.0):
            raise ValueError(f"{name} must contain only finite positive values.")
        return tensor

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Store student/teacher pairs but execute the mapped teacher mean action."""
        # Sampling is storage-only: it initializes the student's distribution
        # state for RSL-RL's action-std logger, but never reaches the env.
        student_action = self.student(obs, stochastic_output=True).detach()
        teacher_action = self.teacher(obs).detach()

        # The old PPO actor was trained behind RslRlVecEnvWrapper(clip_actions=1).
        # Reproduce that clip before changing units.  All mapping factors are
        # <= 1 for the current contract, so the mapped command remains legal.
        teacher_action = torch.clamp(teacher_action, min=-1.0, max=1.0)
        mapped_teacher_action = teacher_action * self._teacher_to_student

        self.transition.actions = student_action
        self.transition.privileged_actions = mapped_teacher_action
        self.transition.observations = obs
        return mapped_teacher_action

    def update(self) -> dict[str, float]:
        """Run stock behavior cloning and add action-error diagnostics."""
        with torch.inference_mode():
            student_actions = self.student(self.storage.observations.flatten(0, 1))
            teacher_actions = self.storage.privileged_actions.flatten(0, 1)
            error = student_actions - teacher_actions
            action_rmse = torch.sqrt(torch.mean(torch.square(error)))
            action_abs_max = torch.max(torch.abs(error))
            physical_error = error * self._student_action_scale
            physical_rmse_rad = torch.sqrt(torch.mean(torch.square(physical_error)))

        loss_dict = super().update()
        loss_dict.update(
            {
                "action_rmse": action_rmse.item(),
                "action_abs_max": action_abs_max.item(),
                "physical_residual_rmse_rad": physical_rmse_rad.item(),
            }
        )
        return loss_dict

    def save(self) -> dict:
        """Save a resumable distillation checkpoint and a PPO-compatible actor alias."""
        saved_dict = super().save()
        saved_dict["actor_state_dict"] = self.student.state_dict()
        saved_dict["distillation_contract"] = {
            "rollout_driver": "teacher",
            "teacher_action_scale_rad": self._teacher_action_scale.detach().cpu(),
            "student_action_scale_rad": self._student_action_scale.detach().cpu(),
        }
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Distinguish a true distillation resume from a PPO teacher import."""
        if load_cfg is None and "student_state_dict" in loaded_dict:
            load_cfg = {
                "student": True,
                "teacher": True,
                "optimizer": True,
                "iteration": True,
            }
        return super().load(loaded_dict, load_cfg, strict)
