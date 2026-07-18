from collections.abc import Sequence

import isaaclab.utils.math as math_utils
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

from isaac_neuromeka.mdp.rewards import square_prism_ori_error as _square_prism_ori_error
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
        # cage 가상점을 만드는 body들. reward cfg의 CAGE_BODIES와 반드시 동일해야 함.
        # 한쪽만 바꾸면 metric이 reward와 다른 점을 측정하게 됨.
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
            # 큐브 "표면"까지의 거리. 큐브가 6cm라 중심까지의 거리로는 접촉 여부를 알 수 없음.
            "palm_surface",
            "thumb_surface",
            "index_surface",
            "middle_surface",
            "ring_surface",
            "little_surface",
            # cage 진단: 가상점들이 큐브 안으로 얼마나 파고들었나
            "cage_sdf_mean",
            "cage_sdf_min",
            "cage_inside_frac",
            # 큐브가 정말 손가락 "사이"에 있나? +1이면 엄지와 그 손가락이 큐브 양쪽, -1이면 같은 쪽.
            # cage_inside_frac만으론 못 봄 (선분이 모서리만 스쳐도 일부 점은 내부에 들어감).
            # 손가락별로 따로 기록함: 중지만 보면 검지가 교차해도 알 수 없기 때문.
            "thumb_index_opposition",
            "thumb_middle_opposition",
            # 엄지끝-손끝 거리. 오므리면 줄어듦. 6cm 큐브를 쥐면 큐브 폭 근처여야 함.
            "cage_span_index",
            "cage_span",
            # 물체가 손바닥 앞에 있나? 손가락은 손바닥 쪽으로 굽으므로 손바닥 뒤의 물체는 못 감쌈.
            # cage 지표는 이걸 못 봄 (선분이 손 방향과 무관하게 큐브를 관통하기 때문).
            "palm_facing",
            # palm_link 기준 arm 6축의 sqrt(det(J Jt)). 0에 가까우면 팔이 특이점으로 접힌 것.
            # 기준: 초기 자세 약 0.064, 무작위 최대 약 0.113. 0.02 아래면 특이점 근처.
            "arm_manipulability",
            # 물체 자세 오차 [rad]: 월드 정렬 대비, 정사각 단면 프리즘 대칭 8개 중 최소각.
            # orientation v1 success(ori_limit)의 판독용. 정육면체(큐브 태스크)에서는 값이
            # 과대평가될 수 있음 (대칭 24개 중 8개만 고려) — box 태스크 기준 지표.
            "box_ori_error",
            # 손이 큐브를 실제로 건드렸나, 그리고 바닥에서 떴나
            "cube_displacement",
            # 중심 높이. lift 신호가 "아님": 바닥의 큐브를 짜면 모서리로 세워져 중심만 몇 mm 올라감.
            # cube_clearance와 나란히 두어 그 편법이 보이게 하려고 남겨둠.
            "cube_lift",
            # 큐브의 "최하 꼭짓점"이 바닥에서 뜬 높이. 이게 진짜 lift 지표.
            "cube_clearance",
            # 팔이 튀는 원인을 "정책이 시킨 것"과 "물리가 이긴 것"으로 가르는 두 지표.
            # action은 절대 관절 목표임 (target = default + 0.2 * action).
            #   track_err 작음(<0.1) + delta 큼(>0.3) -> 팔이 명령대로 발광. 학습/보상 문제
            #   track_err 큼  (>0.3)                  -> 물리가 명령을 이김. dt/decimation 문제
            "action_track_err",  # |관절목표 - 관절실제| 최대 [rad]
            "action_delta",  # |a_t - a_{t-1}| 평균. action 범위가 [-1,1]임
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

        # 에피소드 평균으론 "계속 멀리 있었다"와 "나갔다가 마지막에 붙었다"를 구분 못 함.
        # 그래서 평균과 함께 final/min/max를 전부 기록함.
        self._cube_metric_sums = zeros()
        self._cube_metric_last = zeros()
        self._cube_metric_min = {k: torch.full_like(v, float("inf")) for k, v in zeros().items()}
        self._cube_metric_max = {k: torch.full_like(v, float("-inf")) for k, v in zeros().items()}
        self._cube_init_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self._cube_half_extent = torch.full((3,), 0.03, dtype=torch.float, device=self.device)
        self._cube_surface_z = 0.0
        # 손바닥 법선 (palm_link 로컬 +x). 손가락을 오므릴 때 손끝이 이동하는 방향으로 실측함.
        # palm_facing reward의 palm_normal_b와 반드시 동일해야 함.
        self._palm_normal_b = torch.tensor([0.19, 0.28, 0.94], dtype=torch.float, device=self.device)
        self._palm_normal_b /= torch.norm(self._palm_normal_b)
        self._cube_corner_signs = torch.tensor(
            [
                [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0],
            ],
            dtype=torch.float,
            device=self.device,
        )

        try:
            if "robot" not in self._env.scene.articulations or "cube" not in self._env.scene.rigid_objects:
                return
            robot = self._env.scene["robot"]
            self._cube_metric_body_ids = [robot.find_bodies(n)[0][0] for n in self._cube_metric_body_names]
            self._cage_body_ids = [robot.find_bodies(n)[0][0] for n in self._cage_body_names]
            self._arm_joint_ids, _ = robot.find_joints(["joint[0-5]"])
            size = self._env.cfg.scene.cube.spawn.size
            self._cube_half_extent = torch.tensor(size, dtype=torch.float, device=self.device) / 2.0
            rewards_cfg = getattr(self._env.cfg, "rewards", None)
            cube_lift_cfg = getattr(rewards_cfg, "cube_lift", None)
            if cube_lift_cfg is not None:
                self._cube_surface_z = float(cube_lift_cfg.params.get("surface_z", 0.0))
        except Exception:
            return

        self._cube_metric_enabled = True

    def _cube_signed_distance(self, points_w: torch.Tensor) -> torch.Tensor:
        """Signed distance from (N, P, 3) world points to the cube surface. Negative inside."""
        from isaaclab.utils.math import quat_apply_inverse

        cube = self._env.scene["cube"]
        rel = points_w - cube.data.root_pos_w.unsqueeze(1)
        quat = cube.data.root_quat_w.unsqueeze(1).expand(-1, rel.shape[1], -1)
        # env별 치수 버퍼(Box-Transport) 우선, 없으면 상수 (큐브 태스크 경로 불변)
        half = getattr(self._env, "box_half_extents", None)
        if half is None:
            half = self._cube_half_extent
        half = half.unsqueeze(1) if half.dim() == 2 else half
        q = quat_apply_inverse(quat, rel).abs() - half
        return torch.norm(torch.clamp(q, min=0.0), dim=-1) + torch.clamp(q.max(dim=-1).values, max=0.0)

    def _compute_cube_distance_metrics(self) -> dict[str, torch.Tensor]:
        robot = self._env.scene["robot"]
        cube = self._env.scene["cube"]

        body_pos_w = robot.data.body_state_w[:, self._cube_metric_body_ids, :3]
        cube_pos_w = cube.data.root_pos_w.unsqueeze(1)
        distances = torch.norm(body_pos_w - cube_pos_w, dim=-1)
        finger_distances = distances[:, 1:]
        surface = self._cube_signed_distance(body_pos_w)

        # finger_cage_hold reward와 똑같이 cage 가상점을 재구성함
        cage_pos_w = robot.data.body_state_w[:, self._cage_body_ids, :3]
        thumb, opposing = cage_pos_w[:, 0], cage_pos_w[:, 1:]
        span = opposing - thumb.unsqueeze(1)
        points = thumb[:, None, None, :] + span.unsqueeze(2) * self._cage_fractions.view(1, 1, -1, 1)
        cage_sdf = self._cube_signed_distance(points.reshape(thumb.shape[0], -1, 3))

        # 큐브에서 본 엄지 vs 각 손끝 방향: +1이면 큐브 양쪽에 있음
        index_tip, middle_tip = cage_pos_w[:, 1], cage_pos_w[:, 3]

        def _unit_from_cube(p):
            v = p - cube.data.root_pos_w
            return v / torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-6)

        u_thumb = _unit_from_cube(thumb)
        index_opposition = -torch.sum(u_thumb * _unit_from_cube(index_tip), dim=-1)
        middle_opposition = -torch.sum(u_thumb * _unit_from_cube(middle_tip), dim=-1)

        # 손바닥 법선(실측 +x) vs 큐브 방향
        palm_id = self._cube_metric_body_ids[0]  # palm_link is first in _cube_metric_body_names
        palm_pos_w = robot.data.body_state_w[:, palm_id, :3]
        palm_quat_w = robot.data.body_state_w[:, palm_id, 3:7]
        normal_w = math_utils.quat_apply(palm_quat_w, self._palm_normal_b.expand(self.num_envs, 3))
        to_cube = cube.data.root_pos_w - palm_pos_w
        to_cube = to_cube / torch.clamp(torch.norm(to_cube, dim=-1, keepdim=True), min=1e-6)
        palm_facing = torch.clamp(torch.sum(normal_w * to_cube, dim=-1), 0.0, 1.0)

        # 팔이 얼마나 접혔나? 0에 가까우면 특이점 -> 손을 자유롭게 못 움직임
        jac = robot.root_physx_view.get_jacobians()[:, palm_id - 1, :, :][:, :, self._arm_joint_ids]
        manipulability = torch.sqrt(torch.clamp(torch.det(jac @ jac.transpose(1, 2)), min=0.0))

        # 큐브 8개 꼭짓점 중 최저점의 지면 대비 높이
        half = getattr(self._env, "box_half_extents", None)
        if half is None:
            half = self._cube_half_extent
        if half.dim() == 1:
            corners_b = (self._cube_corner_signs * half).unsqueeze(0).expand(self.num_envs, -1, -1)
        else:
            corners_b = self._cube_corner_signs.unsqueeze(0) * half.unsqueeze(1)  # (N, 8, 3)
        quat = cube.data.root_quat_w.unsqueeze(1).expand(-1, 8, -1)
        corners_w = cube.data.root_pos_w.unsqueeze(1) + math_utils.quat_apply(quat, corners_b)
        clearance = corners_w[..., 2].min(dim=1).values - self._env.scene.env_origins[:, 2] - self._cube_surface_z

        cube_offset = cube.data.root_pos_w - self._cube_init_pos

        # action은 절대 관절 목표임 (processed = raw * scale + default_joint_pos).
        # 목표를 잘 따라가면(track_err 작음) 팔이 튄 건 정책이 그렇게 시킨 것 = 학습 문제.
        # 목표를 못 따라가면(track_err 큼) 물리가 명령을 이긴 것 = dt/decimation 문제.
        action_term = self._env.action_manager.get_term("arm_action")
        target = action_term.processed_actions
        actual = robot.data.joint_pos[:, action_term._joint_ids]
        track_err = (target - actual).abs().amax(dim=-1)
        delta = (self._env.action_manager.action - self._env.action_manager.prev_action).abs().mean(dim=-1)

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
            "palm_facing": palm_facing,
            "arm_manipulability": manipulability,
            "box_ori_error": _square_prism_ori_error(cube.data.root_quat_w),
            "cube_displacement": torch.norm(cube_offset, dim=1),
            "cube_lift": cube_offset[:, 2],
            "cube_clearance": clearance,
            "action_track_err": track_err,
            "action_delta": delta,
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
                # Metrics/cube는 에피소드 "평균"임. 앞 몇 step(0.7m 밖에서 출발)이 지배하므로
                # 정책이 실제로 정착한 자세에 대해선 거의 아무것도 알려주지 않음.
                # 정착 자세는 Metrics/cube_final을 볼 것.
                extras["Metrics/cube/" + name] = (
                    torch.mean(self._cube_metric_sums[name][env_ids]) / self._env.max_episode_length_s
                )
                extras["Metrics/cube_final/" + name] = torch.mean(self._cube_metric_last[name][env_ids])
                # 최초 startup reset은 compute() 이전에 호출되므로 min/max가 아직 +/-inf임.
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

            # 이 시점엔 event(reset_cube_position)가 이미 실행됨 -> 새 에피소드의 큐브 위치.
            # cube_displacement / cube_lift의 기준선이 됨.
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
