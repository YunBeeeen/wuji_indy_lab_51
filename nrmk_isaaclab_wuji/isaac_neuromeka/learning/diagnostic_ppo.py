"""PPO diagnostics that do not change the optimization objective.

RSL-RL 3.0.1 computes KL internally for its adaptive learning-rate schedule,
but its stock logger only exposes the losses, learning rate, and policy std.
This subclass evaluates the completed update against the rollout policy and
adds collapse-oriented diagnostics to the returned loss dictionary.
"""

from __future__ import annotations

import torch

from rsl_rl.algorithms import PPO


class DiagnosticPPO(PPO):
    """Stock RSL-RL PPO with diagnostic-only TensorBoard scalars."""

    _ACTION_CLIP_LIMIT = 1.0
    _ACTION_NEAR_CLIP_LIMIT = 0.95

    @staticmethod
    def _explained_variance(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        targets = targets.float().reshape(-1)
        predictions = predictions.float().reshape(-1)
        target_variance = torch.var(targets, unbiased=False)
        residual_variance = torch.var(targets - predictions, unbiased=False)
        return torch.where(
            target_variance > 1.0e-8,
            1.0 - residual_variance / target_variance,
            torch.zeros_like(target_variance),
        )

    def update(self) -> dict[str, float]:
        """Run the unchanged PPO update, then report collapse diagnostics."""
        storage = self.storage

        # Returns and rollout values describe critic quality before this update.
        explained_variance_before = self._explained_variance(storage.returns, storage.values)

        # Actions are the raw Gaussian samples stored before the VecEnv wrapper
        # clips them to [-1, 1].  This therefore measures actual policy-output
        # clipping, rather than actuator torque saturation.
        actions = storage.actions
        abs_actions = torch.abs(actions)
        action_clip_by_joint = torch.mean(
            (abs_actions >= self._ACTION_CLIP_LIMIT).float(), dim=(0, 1)
        )
        action_near_clip_fraction = torch.mean(
            (abs_actions >= self._ACTION_NEAR_CLIP_LIMIT).float()
        )
        action_clip_fraction = torch.mean(
            (abs_actions >= self._ACTION_CLIP_LIMIT).float()
        )
        action_abs_mean = torch.mean(abs_actions)
        action_abs_max = torch.max(abs_actions)

        loss_dict = super().update()

        # Compare the fully updated policy with the behavior policy that made
        # this rollout.  The stock adaptive schedule observes per-minibatch KL;
        # this post-update value answers the more useful collapse question:
        # "how far did the whole update move the policy?"
        with torch.inference_mode():
            observations = storage.observations.flatten(0, 1)
            rollout_actions = storage.actions.flatten(0, 1)
            old_log_prob = storage.actions_log_prob.flatten(0, 1).squeeze(-1)
            old_distribution_params = tuple(
                parameter.flatten(0, 1) for parameter in storage.distribution_params
            )

            self.actor(observations, stochastic_output=True)
            new_log_prob = self.actor.get_output_log_prob(rollout_actions).squeeze(-1)
            new_distribution_params = self.actor.output_distribution_params
            ratio = torch.exp(new_log_prob - old_log_prob)
            approx_kl = torch.mean(
                self.actor.get_kl_divergence(
                    old_distribution_params,
                    new_distribution_params,
                )
            )
            clip_fraction = torch.mean(
                (torch.abs(ratio - 1.0) > self.clip_param).float()
            )
            ratio_mean = torch.mean(ratio)
            ratio_abs_max_deviation = torch.max(torch.abs(ratio - 1.0))

            updated_values = self.critic(observations)
            explained_variance_after = self._explained_variance(
                storage.returns.flatten(0, 1), updated_values
            )

        loss_dict.update(
            {
                "approx_kl": approx_kl.item(),
                "clip_fraction": clip_fraction.item(),
                "policy_ratio_mean": ratio_mean.item(),
                "policy_ratio_abs_max_deviation": ratio_abs_max_deviation.item(),
                "explained_variance_before": explained_variance_before.item(),
                "explained_variance_after": explained_variance_after.item(),
                "action_clip_fraction": action_clip_fraction.item(),
                "action_near_clip_fraction": action_near_clip_fraction.item(),
                "action_abs_mean": action_abs_mean.item(),
                "action_abs_max": action_abs_max.item(),
                "action_clip_fraction_max_joint": torch.max(action_clip_by_joint).item(),
            }
        )
        for joint_index, fraction in enumerate(action_clip_by_joint):
            loss_dict[f"action_clip_fraction_joint_{joint_index:02d}"] = fraction.item()

        return loss_dict
