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
        self._episode_raw_sums = dict()
        for term_name in self._term_names:
            self._episode_stats[term_name] = TorchRunningStats(dim=self.num_envs, device=self.device)
            self._episode_raw_sums[term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._configure_cube_metrics()

    def _configure_cube_metrics(self):
        self._cube_metric_enabled = False
        self._cube_metric_body_names = [
            "palm_link",
            "finger1_tip_link",
            "finger2_tip_link",
            "finger3_tip_link",
            "finger4_tip_link",
            "finger5_tip_link",
        ]
        # Bodies spanning the cage: [thumb_tip, *opposing]. Must mirror CAGE_BODIES in the reward
        # cfg exactly, or the logged diagnostics describe different points than the reward optimises.
        self._cage_body_names = [
            "finger1_tip_link",  # thumb tip: the anchor every line starts from
            "finger2_tip_link",
            "finger2_link3",
            "finger3_tip_link",
            "finger3_link3",
        ]
        self._cage_fractions = torch.tensor([0.25, 0.50, 0.75], dtype=torch.float, device=self.device)
        self._cube_metric_names = [
            # distances to the cube CENTER
            "palm_distance",
            "thumb_distance",
            "index_distance",
            "middle_distance",
            "ring_distance",
            "little_distance",
            "finger_mean_distance",
            "non_thumb_mean_distance",
            "finger_weighted_mean_distance",
            # distances to the cube SURFACE (center distances are not comparable across bodies
            # because the cube is 6 cm wide; the surface gap is what "did it touch" depends on)
            "palm_surface",
            "thumb_surface",
            "index_surface",
            "middle_surface",
            "ring_surface",
            "little_surface",
            # cage diagnostics: how deep the 6 virtual points sit in the cube
            "cage_sdf_mean",
            "cage_sdf_min",
            "cage_inside_frac",
            # Is the cube actually *between* the fingers? +1 = thumb and that finger on opposite
            # sides of it, -1 = same side. cage_inside_frac alone cannot see this: a segment merely
            # clipping the cube's edge still puts some points inside. Logged per finger because a
            # thumb-middle-only cage let the index cross over the middle while still scoring well.
            "thumb_index_opposition",
            "thumb_middle_opposition",
            # Thumb tip to fingertip distances. Shrink as the hand closes; a grasp on a 6 cm cube
            # should end near the cube width, not at the hand's open span.
            "cage_span_index",
            "cage_span",
            # did the hand actually disturb the cube, and did it come off the ground
            "cube_displacement",
            "cube_lift",
        ]
        self._cube_metric_body_ids = []
        self._cage_body_ids = []
        self._cube_metric_finger_weights = torch.tensor(
            [3.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float, device=self.device
        )

        def zeros():
            return {
                name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for name in self._cube_metric_names
            }

        # An episode mean cannot tell "stayed far all episode" apart from "swung away, then
        # converged at the end", so track the final/min/max of every metric alongside the mean.
        self._cube_metric_sums = zeros()
        self._cube_metric_last = zeros()
        self._cube_metric_min = {k: torch.full_like(v, float("inf")) for k, v in zeros().items()}
        self._cube_metric_max = {k: torch.full_like(v, float("-inf")) for k, v in zeros().items()}
        self._cube_init_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self._cube_half_extent = torch.full((3,), 0.03, dtype=torch.float, device=self.device)

        try:
            if "robot" not in self._env.scene.articulations or "cube" not in self._env.scene.rigid_objects:
                return
            robot = self._env.scene["robot"]
            self._cube_metric_body_ids = [robot.find_bodies(n)[0][0] for n in self._cube_metric_body_names]
            self._cage_body_ids = [robot.find_bodies(n)[0][0] for n in self._cage_body_names]
            size = self._env.cfg.scene.cube.spawn.size
            self._cube_half_extent = torch.tensor(size, dtype=torch.float, device=self.device) / 2.0
        except Exception:
            return

        self._cube_metric_enabled = True

    def _cube_signed_distance(self, points_w: torch.Tensor) -> torch.Tensor:
        """Signed distance from (N, P, 3) world points to the cube surface. Negative inside."""
        from isaaclab.utils.math import quat_apply_inverse

        cube = self._env.scene["cube"]
        rel = points_w - cube.data.root_pos_w.unsqueeze(1)
        quat = cube.data.root_quat_w.unsqueeze(1).expand(-1, rel.shape[1], -1)
        q = quat_apply_inverse(quat, rel).abs() - self._cube_half_extent
        return torch.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0)

    def _compute_cube_distance_metrics(self) -> dict[str, torch.Tensor]:
        robot = self._env.scene["robot"]
        cube = self._env.scene["cube"]

        body_pos_w = robot.data.body_state_w[:, self._cube_metric_body_ids, :3]
        cube_pos_w = cube.data.root_pos_w.unsqueeze(1)
        distances = torch.norm(body_pos_w - cube_pos_w, dim=-1)
        finger_distances = distances[:, 1:]
        surface = self._cube_signed_distance(body_pos_w)

        # rebuild the cage points exactly as the finger_cage_hold reward does
        cage_pos_w = robot.data.body_state_w[:, self._cage_body_ids, :3]
        thumb, opposing = cage_pos_w[:, 0], cage_pos_w[:, 1:]
        span = opposing - thumb.unsqueeze(1)
        points = thumb[:, None, None, :] + span.unsqueeze(2) * self._cage_fractions.view(1, 1, -1, 1)
        cage_sdf = self._cube_signed_distance(points.reshape(thumb.shape[0], -1, 3))

        # thumb vs each fingertip seen from the cube: +1 when they sit on opposite sides of it
        index_tip, middle_tip = cage_pos_w[:, 1], cage_pos_w[:, 3]

        def _unit_from_cube(p):
            v = p - cube.data.root_pos_w
            return v / torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-6)

        u_thumb = _unit_from_cube(thumb)
        index_opposition = -torch.sum(u_thumb * _unit_from_cube(index_tip), dim=-1)
        middle_opposition = -torch.sum(u_thumb * _unit_from_cube(middle_tip), dim=-1)

        cube_offset = cube.data.root_pos_w - self._cube_init_pos

        return {
            "palm_distance": distances[:, 0],
            "thumb_distance": distances[:, 1],
            "index_distance": distances[:, 2],
            "middle_distance": distances[:, 3],
            "ring_distance": distances[:, 4],
            "little_distance": distances[:, 5],
            "finger_mean_distance": torch.mean(finger_distances, dim=1),
            "non_thumb_mean_distance": torch.mean(distances[:, 2:6], dim=1),
            "finger_weighted_mean_distance": torch.sum(
                finger_distances * self._cube_metric_finger_weights.unsqueeze(0), dim=1
            )
            / torch.sum(self._cube_metric_finger_weights),
            "palm_surface": surface[:, 0],
            "thumb_surface": surface[:, 1],
            "index_surface": surface[:, 2],
            "middle_surface": surface[:, 3],
            "ring_surface": surface[:, 4],
            "little_surface": surface[:, 5],
            "cage_sdf_mean": torch.mean(cage_sdf, dim=1),
            "cage_sdf_min": torch.min(cage_sdf, dim=1).values,
            "cage_inside_frac": (cage_sdf < 0.0).float().mean(dim=1),
            "thumb_index_opposition": index_opposition,
            "thumb_middle_opposition": middle_opposition,
            "cage_span_index": torch.norm(index_tip - thumb, dim=-1),
            "cage_span": torch.norm(middle_tip - thumb, dim=-1),
            "cube_displacement": torch.norm(cube_offset, dim=1),
            "cube_lift": cube_offset[:, 2],
        }

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
            episodic_raw_sum_avg = torch.mean(self._episode_raw_sums[name][env_ids])
            extras["Episode_Reward/" + name] = episodic_sum_avg / self._env.max_episode_length_s
            extras["Episode_Reward_Raw/" + name] = episodic_raw_sum_avg / self._env.max_episode_length_s
            extras["Episode_Reward_Std/" + name] = torch.mean(self._episode_stats[name].standard_deviation()[env_ids])
            # reset episodic sum
            self._episode_sums[name][env_ids] = 0.0
            self._episode_raw_sums[name][env_ids] = 0.0
            self._episode_stats[name].reset(env_ids)

        if self._cube_metric_enabled:
            for name in self._cube_metric_names:
                # Metrics/cube is the episode MEAN. It is dominated by the travel phase (the first
                # few steps start ~0.7 m out), so it says almost nothing about the pose the policy
                # actually settles into. Metrics/cube_final is the one to read for that.
                extras["Metrics/cube/" + name] = (
                    torch.mean(self._cube_metric_sums[name][env_ids]) / self._env.max_episode_length_s
                )
                extras["Metrics/cube_final/" + name] = torch.mean(self._cube_metric_last[name][env_ids])
                # The startup reset runs before compute() ever has, so min/max are still +/-inf.
                extras["Metrics/cube_min/" + name] = torch.mean(
                    torch.nan_to_num(self._cube_metric_min[name][env_ids], posinf=0.0)
                )
                extras["Metrics/cube_max/" + name] = torch.mean(
                    torch.nan_to_num(self._cube_metric_max[name][env_ids], neginf=0.0)
                )
                self._cube_metric_sums[name][env_ids] = 0.0
                self._cube_metric_last[name][env_ids] = 0.0
                self._cube_metric_min[name][env_ids] = float("inf")
                self._cube_metric_max[name][env_ids] = float("-inf")

            # Events (including reset_cube_position) already ran, so this is the cube's pose for the
            # episode that is about to start — the baseline cube_displacement/cube_lift measure from.
            self._cube_init_pos[env_ids] = self._env.scene["cube"].data.root_pos_w[env_ids]

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
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            raw_value = term_cfg.func(self._env, **term_cfg.params)
            value = raw_value * term_cfg.weight * dt
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value
            self._episode_raw_sums[name] += raw_value * dt
            self._episode_stats[name].update(value)
            self._step_reward[:, term_idx] = value / dt

        if self._cube_metric_enabled:
            for name, metric in self._compute_cube_distance_metrics().items():
                self._cube_metric_sums[name] += metric * dt
                self._cube_metric_last[name][:] = metric
                self._cube_metric_min[name] = torch.minimum(self._cube_metric_min[name], metric)
                self._cube_metric_max[name] = torch.maximum(self._cube_metric_max[name], metric)

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
