from collections.abc import Sequence

import torch

# from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv  # , ManagerBasedRLEnvCfg
from isaaclab.managers import (  # noqa: F401
    ActionManager,
    CommandManager,
    CurriculumManager,
    EventManager,
    ManagerTermBase,
    ManagerTermBaseCfg,
    ObservationManager,
    RewardManager,
    SceneEntityCfg,
    TerminationManager,
)
from prettytable import PrettyTable

from isaac_neuromeka.utils.running_stats import TorchRunningStats


class CustomObservationManager(ObservationManager):
    def compute_group(self, group_name: str) -> torch.Tensor | dict[str, torch.Tensor]:
        # check ig group name is valid
        if group_name not in self._group_obs_term_names:
            raise ValueError(
                f"Unable to find the group '{group_name}' in the observation manager."
                f" Available groups are: {list(self._group_obs_term_names.keys())}"
            )
        # iterate over all the terms in each group
        group_term_names = self._group_obs_term_names[group_name]
        # buffer to store obs per group
        self.group_obs = dict.fromkeys(group_term_names, None)
        # read attributes for each term
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])
        # evaluate terms: compute, add noise, clip, scale.
        for name, term_cfg in obs_terms:
            # compute term's value
            obs: torch.Tensor = term_cfg.func(self._env, **term_cfg.params).clone()
            # apply post-processing
            if term_cfg.noise:
                obs = term_cfg.noise.func(obs, term_cfg.noise)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale:
                obs = obs.mul_(term_cfg.scale)
            # TODO: Introduce delay and filtering models.
            # Ref: https://robosuite.ai/docs/modules/sensors.html#observables
            # add value to list
            self.group_obs[name] = obs
        # concatenate all observations in the group together
        if self._group_obs_concatenate[group_name]:
            return torch.cat(list(self.group_obs.values()), dim=-1)
        else:
            return self.group_obs


class CustomActionManager(ActionManager):
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._prevprev_action = torch.zeros_like(self._action)

    @property
    def prevprev_action(self) -> torch.Tensor:
        return self._prevprev_action

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the action history
        self._prevprev_action[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0
        # reset all action terms
        for term in self._terms.values():
            term.reset(env_ids=env_ids)
        # nothing to log here
        return {}

    def apply_action(self, env_ids: Sequence[int] | None = None) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms.values():
            term.apply_actions(env_ids)

    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prevprev_action[:] = self._prev_action
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device)

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim


class CustomRewardManager(RewardManager):
    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._episode_stats = dict()
        for term_name in self._term_names:
            self._episode_stats[term_name] = TorchRunningStats(dim=self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # store information
            # r_1 + r_2 + ... + r_n
            episodic_sum_avg = torch.mean(self._episode_sums[name][env_ids])
            extras["Episode Reward/" + name] = episodic_sum_avg / self._env.max_episode_length_s
            extras["Episode Reward/Mean_wo_coeff/" + name] = (episodic_sum_avg / self._env.max_episode_length_s) / abs(
                term_cfg.weight
            )
            extras["Episode Reward/Std/" + name] = torch.mean(self._episode_stats[name].standard_deviation()[env_ids])
            # reset episodic sum
            self._episode_sums[name][env_ids] = 0.0
            self._episode_stats[name].reset(env_ids)

        # reset all the reward terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self, dt: float) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[:] = 0.0
        # iterate over all the reward terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value
            self._episode_stats[name].update(value)

        return self._reward_buf


class CostManager(RewardManager):
    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._episode_stats = dict()
        self.num_cost_terms = len(self._term_names)
        for term_name in self._term_names:
            self._episode_stats[term_name] = TorchRunningStats(dim=self.num_envs, device=self.device)

        self._reward_buf = None
        self._cost_buf = torch.zeros((self.num_envs, self.num_cost_terms), dtype=torch.float, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # store information
            # r_1 + r_2 + ... + r_n
            episodic_sum_avg = torch.mean(self._episode_sums[name][env_ids])
            extras["Episode Cost/" + name] = episodic_sum_avg / self._env.max_episode_length_s
            extras["Episode Cost/Mean_wo_coeff/" + name] = (episodic_sum_avg / self._env.max_episode_length_s) / abs(
                term_cfg.weight
            )
            extras["Episode Cost/Std/" + name] = torch.mean(self._episode_stats[name].standard_deviation()[env_ids])
            # reset episodic sum
            self._episode_sums[name][env_ids] = 0.0
            self._episode_stats[name].reset(env_ids)

        # reset all the reward terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self, dt: float) -> torch.Tensor:
        # reset computation
        self._cost_buf = torch.zeros((self.num_envs, self.num_cost_terms), dtype=torch.float, device=self.device)

        # iterate over all the reward terms
        cost_id = 0
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            self._cost_buf[:, cost_id] = value

            # update episodic sum
            self._episode_sums[name] += value
            self._episode_stats[name].update(value)

            cost_id += 1

        return self._cost_buf

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        msg = f"<CostManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Cost Terms"
        table.field_names = ["Index", "Name", "Weight"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, term_cfg.weight])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg
