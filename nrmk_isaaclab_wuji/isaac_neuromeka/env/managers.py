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
from isaac_neuromeka.mdp.rewards import _SQUARE_PRISM_Y_SYMS_TIP as _TIP_SYMS
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

    def apply_action(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms.values():
            # Match Isaac Lab's ActionManager interface. Standard action terms such as
            # JointPositionToLimitsAction do not accept an env_ids argument, while the
            # local action terms keep it optional and therefore also work with no argument.
            term.apply_actions()

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
        self._configure_hand_grasp_metrics()
        self._configure_hand_setting_metrics()

    def _configure_cube_metrics(self):
        self._cube_metric_enabled = False
        self._tripod_metric_params = None
        self._functional_grasp_metric_params = None
        self._functional_grasp_term = None
        self._cube_metric_body_names = [
            "palm_link",
            "finger1_tip_link",
            "finger2_tip_link",
            "finger3_tip_link",
            "finger4_tip_link",
            "finger5_tip_link",
        ]
        # Fallback for configurations without finger_cage_hold. Normally body ids and fractions are
        # read from the resolved reward term below so metrics always measure the active cage.
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
            # 물체 자세 오차 [rad]: command quaternion 대비, 정사각 단면 프리즘 대칭 8개 중 최소각.
            # orientation v1 success(ori_limit)의 판독용. 정육면체(큐브 태스크)에서는 값이
            # 과대평가될 수 있음 (대칭 24개 중 8개만 고려) — box 태스크 기준 지표.
            "box_ori_error",
            # position < activation_distance + cage gate 조건을 한 번 통과했는지 (0/1 latch).
            "orientation_stage_active",
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

        # The one-stick task has no command, but its tool_ready termination contains all
        # constraint definitions needed to reproduce the current-state success metrics.
        try:
            tool_ready_cfg = self._env.termination_manager.get_term_cfg("tool_ready")
        except ValueError:
            tool_ready_cfg = None
        if tool_ready_cfg is not None:
            required_params = {
                "palm_cfg",
                "index_cfg",
                "thumb_cfg",
                "index_cage_cfg",
                "middle_cage_cfg",
                "object_cfg",
            }
            if required_params.issubset(tool_ready_cfg.params):
                self._functional_grasp_metric_params = tool_ready_cfg.params
                self._functional_grasp_term = tool_ready_cfg.func
                self._cube_metric_names.extend(
                    [
                        "tripod_index_gate",
                        "tripod_middle_gate",
                        "tripod_gate",
                        "index_grip_error",
                        "thumb_grip_error",
                        "hand_stick_orientation_error",
                        "tool_ready_valid",
                        "tool_ready_stable_steps",
                    ]
                )
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

            # Keep diagnostic cage metrics synchronized with the active reward configuration.
            # This supports both the cube task and Box-Transport cage experiments without another
            # hard-coded body list drifting out of sync.
            try:
                hold_cfg = self.get_term_cfg("finger_cage_hold")
            except ValueError:
                hold_cfg = None

            def resolved_body_ids(entity_cfg):
                body_ids = entity_cfg.body_ids
                if isinstance(body_ids, slice):
                    body_names = entity_cfg.body_names or self._cage_body_names
                    if isinstance(body_names, str):
                        body_names = [body_names]
                    return [robot.find_bodies(name)[0][0] for name in body_names]
                return list(body_ids)

            if hold_cfg is None:
                self._cage_body_ids = [robot.find_bodies(n)[0][0] for n in self._cage_body_names]
            elif "asset_cfg" in hold_cfg.params:
                self._cage_body_ids = resolved_body_ids(hold_cfg.params["asset_cfg"])
            elif {"index_cage_cfg", "middle_cage_cfg"}.issubset(hold_cfg.params):
                index_ids = resolved_body_ids(hold_cfg.params["index_cage_cfg"])
                middle_ids = resolved_body_ids(hold_cfg.params["middle_cage_cfg"])
                # Generic cage SDF metrics pool both tripod groups. The exact active reward
                # remains visible separately through tripod_index/middle/gate below.
                self._cage_body_ids = index_ids + [body_id for body_id in middle_ids if body_id not in index_ids]
                self._tripod_metric_params = hold_cfg.params
            else:
                self._cage_body_ids = [robot.find_bodies(n)[0][0] for n in self._cage_body_names]

            if hold_cfg is not None:
                point_fractions = hold_cfg.params.get("point_fractions")
                if point_fractions is None:
                    num_points = int(hold_cfg.params.get("num_points", 3))
                    point_fractions = tuple(i / (num_points + 1) for i in range(1, num_points + 1))
                self._cage_fractions = torch.tensor(point_fractions, dtype=torch.float, device=self.device)

            self._arm_joint_ids, _ = robot.find_joints(["joint[0-5]"])
            size = self._env.cfg.scene.cube.spawn.size
            self._cube_half_extent = torch.tensor(size, dtype=torch.float, device=self.device) / 2.0
            rewards_cfg = getattr(self._env.cfg, "rewards", None)
            cube_lift_cfg = getattr(rewards_cfg, "cube_lift", None)
            if cube_lift_cfg is not None:
                self._cube_surface_z = float(cube_lift_cfg.params.get("surface_z", 0.0))
        except Exception as exc:
            print(f"[WARNING] CustomRewardManager object metrics disabled: {type(exc).__name__}: {exc}")
            return

        self._cube_metric_enabled = True

    def _configure_hand_grasp_metrics(self):
        """Enable contact and OPEN/CLOSE diagnostics only for hand_grasp."""
        self._hand_grasp_metric_enabled = False
        self._hand_grasp_sensor_names = (
            "thumb_distal_stick1",
            "index_tip_stick1",
            "middle_tip_stick1",
            "palm_stick2",
            "thumb_mid_stick2",
            "ring_tip_stick2",
        )
        self._hand_grasp_metric_names = [
            *[f"{name}_force" for name in self._hand_grasp_sensor_names],
            "ring_support_force",
            "min_functional_force",
            "functional_contact_count",
            "functional_contact_fraction",
            "full_contact",
            "quiet_valid",
            "stick1_linear_speed",
            "stick2_linear_speed",
            "max_linear_speed",
            "stick1_angular_speed",
            "stick2_angular_speed",
            "max_angular_speed",
            "mode_open",
            "mode_close",
            "tip_surface_gap",
            "tip_lateral_error",
            "tip_axial_offset",
            "tip_axial_error",
            "target_tip_gap",
            "tip_gap_error",
            "stick1_pivot_error",
            "stick2_position_error",
            "stick2_orientation_error",
            "mode_geometry_valid",
            "success_stable_steps",
        ]

        try:
            success_cfg = self._env.termination_manager.get_term_cfg("success")
        except ValueError:
            return
        required_params = {
            "command_name",
            "sensor_groups",
            "palm_cfg",
            "stick1_cfg",
            "stick2_cfg",
            "stick1_pivot_offset_o",
            "stick1_tip_offset_o",
            "stick2_tip_offset_o",
            "stick_thickness",
            "open_target_gap",
            "close_target_gap",
            "stick1_reference_position_p",
            "stick1_reference_quaternion_p",
            "stick2_reference_position_p",
            "stick2_reference_quaternion_p",
            "reference_separation_direction_stick2",
            "contact_threshold",
            "pivot_error_limit",
            "tip_gap_error_limit",
            "lateral_error_limit",
            "stick2_position_error_limit",
            "stick2_orientation_error_limit",
            "linear_speed_limit",
            "angular_speed_limit",
        }
        if not required_params.issubset(success_cfg.params):
            return
        if any(name not in self._env.scene.sensors for name in self._hand_grasp_sensor_names):
            return

        stick1_name = success_cfg.params["stick1_cfg"].name
        stick2_name = success_cfg.params["stick2_cfg"].name
        if stick1_name not in self._env.scene.rigid_objects or stick2_name not in self._env.scene.rigid_objects:
            return

        self._hand_grasp_stick1_name = stick1_name
        self._hand_grasp_stick2_name = stick2_name
        self._hand_grasp_palm_cfg = success_cfg.params["palm_cfg"]
        self._hand_grasp_success_term = success_cfg.func
        self._hand_grasp_command_name = success_cfg.params["command_name"]
        self._hand_grasp_contact_threshold = float(success_cfg.params["contact_threshold"])
        self._hand_grasp_pivot_error_limit = float(
            success_cfg.params["pivot_error_limit"]
        )
        self._hand_grasp_tip_gap_error_limit = float(
            success_cfg.params["tip_gap_error_limit"]
        )
        self._hand_grasp_lateral_error_limit = float(
            success_cfg.params["lateral_error_limit"]
        )
        self._hand_grasp_stick2_position_error_limit = float(
            success_cfg.params["stick2_position_error_limit"]
        )
        self._hand_grasp_stick2_orientation_error_limit = float(
            success_cfg.params["stick2_orientation_error_limit"]
        )
        self._hand_grasp_linear_speed_limit = float(success_cfg.params["linear_speed_limit"])
        self._hand_grasp_angular_speed_limit = float(success_cfg.params["angular_speed_limit"])
        stick1_position_reference = torch.tensor(
            success_cfg.params["stick1_reference_position_p"],
            dtype=torch.float,
            device=self.device,
        )
        stick1_quaternion_reference = torch.tensor(
            success_cfg.params["stick1_reference_quaternion_p"],
            dtype=torch.float,
            device=self.device,
        )
        stick1_pivot_offset = torch.tensor(
            success_cfg.params["stick1_pivot_offset_o"],
            dtype=torch.float,
            device=self.device,
        )
        self._hand_grasp_stick1_pivot_reference = (
            stick1_position_reference
            + math_utils.quat_apply(
                stick1_quaternion_reference.unsqueeze(0),
                stick1_pivot_offset.unsqueeze(0),
            )[0]
        )
        self._hand_grasp_stick1_pivot_offset = stick1_pivot_offset
        self._hand_grasp_stick1_tip_offset = torch.tensor(
            success_cfg.params["stick1_tip_offset_o"],
            dtype=torch.float,
            device=self.device,
        )
        self._hand_grasp_stick2_tip_offset = torch.tensor(
            success_cfg.params["stick2_tip_offset_o"],
            dtype=torch.float,
            device=self.device,
        )
        self._hand_grasp_stick_thickness = float(
            success_cfg.params["stick_thickness"]
        )
        self._hand_grasp_open_target_gap = float(
            success_cfg.params["open_target_gap"]
        )
        self._hand_grasp_close_target_gap = float(
            success_cfg.params["close_target_gap"]
        )
        self._hand_grasp_stick2_position_reference = torch.tensor(
            success_cfg.params["stick2_reference_position_p"],
            dtype=torch.float,
            device=self.device,
        )
        self._hand_grasp_stick2_quaternion_reference = torch.tensor(
            success_cfg.params["stick2_reference_quaternion_p"],
            dtype=torch.float,
            device=self.device,
        )
        self._hand_grasp_tip_separation_direction_stick2 = torch.tensor(
            success_cfg.params["reference_separation_direction_stick2"],
            dtype=torch.float,
            device=self.device,
        )
        # Resolve the task-specific geometry helper once; compute() runs every
        # policy step and should not repeat the import lookup.
        from isaac_neuromeka.tasks.manipulation.hand_grasp.mdp import (
            _tip_geometry_from_palm_poses,
        )

        self._hand_grasp_tip_geometry = _tip_geometry_from_palm_poses
        _, _, reference_axial_offset = self._hand_grasp_tip_geometry(
            stick1_position_reference.unsqueeze(0),
            stick1_quaternion_reference.unsqueeze(0),
            self._hand_grasp_stick2_position_reference.unsqueeze(0),
            self._hand_grasp_stick2_quaternion_reference.unsqueeze(0),
            self._hand_grasp_stick1_tip_offset,
            self._hand_grasp_stick2_tip_offset,
            self._hand_grasp_stick_thickness,
        )
        self._hand_grasp_tip_axial_reference = reference_axial_offset[0]

        def zeros():
            return {
                name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for name in self._hand_grasp_metric_names
            }

        self._hand_grasp_metric_sums = zeros()
        self._hand_grasp_metric_last = zeros()
        self._hand_grasp_metric_min = {
            name: torch.full_like(value, float("inf"))
            for name, value in zeros().items()
        }
        self._hand_grasp_metric_max = {
            name: torch.full_like(value, float("-inf"))
            for name, value in zeros().items()
        }
        self._hand_grasp_metric_enabled = True

    def _compute_hand_grasp_metrics(self) -> dict[str, torch.Tensor]:
        """Compute per-contact forces and the mode-conditioned success gate."""
        sensor_forces = {}
        for name in self._hand_grasp_sensor_names:
            force_matrix = self._env.scene.sensors[name].data.force_matrix_w
            if force_matrix is None:
                force = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            else:
                force = torch.linalg.vector_norm(force_matrix, dim=-1).sum(dim=(-1, -2))
            sensor_forces[name] = force

        ring_support = sensor_forces["ring_tip_stick2"]
        functional_forces = torch.stack(
            [
                sensor_forces["thumb_distal_stick1"],
                sensor_forces["index_tip_stick1"],
                sensor_forces["middle_tip_stick1"],
                sensor_forces["palm_stick2"],
                sensor_forces["thumb_mid_stick2"],
                ring_support,
            ],
            dim=-1,
        )
        contacts_valid = functional_forces >= self._hand_grasp_contact_threshold

        stick1 = self._env.scene[self._hand_grasp_stick1_name]
        stick2 = self._env.scene[self._hand_grasp_stick2_name]
        robot = self._env.scene[self._hand_grasp_palm_cfg.name]
        palm_id = self._hand_grasp_palm_cfg.body_ids[0]
        palm_position = robot.data.body_pos_w[:, palm_id]
        palm_quaternion = robot.data.body_quat_w[:, palm_id]
        palm_linear_velocity = robot.data.body_lin_vel_w[:, palm_id]
        palm_angular_velocity = robot.data.body_ang_vel_w[:, palm_id]

        stick1_offset = stick1.data.root_pos_w - palm_position
        stick2_offset = stick2.data.root_pos_w - palm_position
        stick1_relative_linear_velocity = (
            stick1.data.root_lin_vel_w
            - palm_linear_velocity
            - torch.cross(
                palm_angular_velocity,
                stick1_offset,
                dim=-1,
            )
        )
        stick2_relative_linear_velocity = (
            stick2.data.root_lin_vel_w
            - palm_linear_velocity
            - torch.cross(
                palm_angular_velocity,
                stick2_offset,
                dim=-1,
            )
        )
        stick1_relative_angular_velocity = (
            stick1.data.root_ang_vel_w - palm_angular_velocity
        )
        stick2_relative_angular_velocity = (
            stick2.data.root_ang_vel_w - palm_angular_velocity
        )
        stick1_linear_speed = torch.linalg.vector_norm(
            stick1_relative_linear_velocity,
            dim=-1,
        )
        stick2_linear_speed = torch.linalg.vector_norm(
            stick2_relative_linear_velocity,
            dim=-1,
        )
        max_linear_speed = torch.maximum(
            stick1_linear_speed,
            stick2_linear_speed,
        )
        stick1_angular_speed = torch.linalg.vector_norm(
            stick1_relative_angular_velocity,
            dim=-1,
        )
        stick2_angular_speed = torch.linalg.vector_norm(
            stick2_relative_angular_velocity,
            dim=-1,
        )
        max_angular_speed = torch.maximum(
            stick1_angular_speed,
            stick2_angular_speed,
        )

        stick1_position_p, stick1_quaternion_p = math_utils.subtract_frame_transforms(
            palm_position,
            palm_quaternion,
            stick1.data.root_pos_w,
            stick1.data.root_quat_w,
        )
        stick2_position_p, stick2_quaternion_p = math_utils.subtract_frame_transforms(
            palm_position,
            palm_quaternion,
            stick2.data.root_pos_w,
            stick2.data.root_quat_w,
        )
        stick1_pivot = (
            stick1_position_p
            + math_utils.quat_apply(
                stick1_quaternion_p,
                self._hand_grasp_stick1_pivot_offset.expand(
                    self.num_envs,
                    -1,
                ),
            )
        )
        stick1_pivot_error = torch.linalg.vector_norm(
            stick1_pivot - self._hand_grasp_stick1_pivot_reference,
            dim=-1,
        )
        stick2_position_error = torch.linalg.vector_norm(
            stick2_position_p - self._hand_grasp_stick2_position_reference,
            dim=-1,
        )
        stick2_orientation_error = 2.0 * torch.acos(
            torch.clamp(
                torch.abs(
                    torch.sum(
                        stick2_quaternion_p
                        * self._hand_grasp_stick2_quaternion_reference,
                        dim=-1,
                    )
                ),
                min=0.0,
                max=1.0,
            )
        )
        # Use exactly the same transverse square-section geometry as the
        # OPEN/CLOSE rewards and success term.
        (
            tip_surface_gap,
            tip_lateral_error,
            tip_axial_offset,
        ) = self._hand_grasp_tip_geometry(
            stick1_position_p,
            stick1_quaternion_p,
            stick2_position_p,
            stick2_quaternion_p,
            self._hand_grasp_stick1_tip_offset,
            self._hand_grasp_stick2_tip_offset,
            self._hand_grasp_stick_thickness,
            self._hand_grasp_tip_separation_direction_stick2,
        )
        tip_axial_error = torch.abs(
            tip_axial_offset - self._hand_grasp_tip_axial_reference
        )
        mode = self._env.command_manager.get_command(
            self._hand_grasp_command_name
        )
        target_tip_gap = (
            mode[:, 0] * self._hand_grasp_open_target_gap
            + mode[:, 1] * self._hand_grasp_close_target_gap
        )
        tip_gap_error = torch.abs(tip_surface_gap - target_tip_gap)
        mode_geometry_valid = (
            (stick1_pivot_error <= self._hand_grasp_pivot_error_limit)
            & (
                stick2_position_error
                <= self._hand_grasp_stick2_position_error_limit
            )
            & (
                stick2_orientation_error
                <= self._hand_grasp_stick2_orientation_error_limit
            )
            & (tip_gap_error <= self._hand_grasp_tip_gap_error_limit)
            & (
                tip_lateral_error
                <= self._hand_grasp_lateral_error_limit
            )
        )
        full_contact = torch.all(contacts_valid, dim=-1)
        quiet_valid = (
            full_contact
            & mode_geometry_valid
            & (max_linear_speed <= self._hand_grasp_linear_speed_limit)
            & (max_angular_speed <= self._hand_grasp_angular_speed_limit)
        )
        stable_steps = getattr(self._hand_grasp_success_term, "_stable_steps", None)
        if stable_steps is None:
            stable_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        else:
            stable_steps = stable_steps.float()

        metrics = {
            f"{name}_force": force
            for name, force in sensor_forces.items()
        }
        metrics.update(
            {
                "ring_support_force": ring_support,
                "min_functional_force": torch.min(functional_forces, dim=-1).values,
                "functional_contact_count": contacts_valid.float().sum(dim=-1),
                "functional_contact_fraction": contacts_valid.float().mean(dim=-1),
                "full_contact": full_contact.float(),
                "quiet_valid": quiet_valid.float(),
                "stick1_linear_speed": stick1_linear_speed,
                "stick2_linear_speed": stick2_linear_speed,
                "max_linear_speed": max_linear_speed,
                "stick1_angular_speed": stick1_angular_speed,
                "stick2_angular_speed": stick2_angular_speed,
                "max_angular_speed": max_angular_speed,
                "mode_open": mode[:, 0],
                "mode_close": mode[:, 1],
                "tip_surface_gap": tip_surface_gap,
                "tip_lateral_error": tip_lateral_error,
                "tip_axial_offset": tip_axial_offset,
                "tip_axial_error": tip_axial_error,
                "target_tip_gap": target_tip_gap,
                "tip_gap_error": tip_gap_error,
                "stick1_pivot_error": stick1_pivot_error,
                "stick2_position_error": stick2_position_error,
                "stick2_orientation_error": stick2_orientation_error,
                "mode_geometry_valid": mode_geometry_valid.float(),
                "success_stable_steps": stable_steps,
            }
        )
        return metrics

    def _configure_hand_setting_metrics(self):
        """Enable open-hand-to-functional-setting diagnostics for hand_setting."""
        self._hand_setting_metric_enabled = False
        self._hand_setting_sensor_names = (
            "thumb_distal_stick1",
            "index_tip_stick1",
            "middle_tip_stick1",
            "palm_stick2",
            "thumb_mid_stick2",
            "ring_tip_stick2",
        )
        self._hand_setting_region_term_names = (
            "thumb_distal_region",
            "index_tip_region",
            "middle_tip_region",
            "ring_tip_region",
        )
        self._hand_setting_metric_names = [
            *[f"{name}_force" for name in self._hand_setting_sensor_names],
            "min_functional_force",
            "functional_contact_count",
            "functional_contact_fraction",
            "full_contact",
            *[
                f"{name}_score"
                for name in self._hand_setting_region_term_names
            ],
            "shaft_region_count",
            "shaft_region_fraction",
            "all_shaft_regions_valid",
            "stick1_position_error",
            "stick1_orientation_error",
            "stick2_position_error",
            "stick2_orientation_error",
            "stick1_pose_valid",
            "stick2_pose_valid",
            "pose_valid",
            "stick1_linear_speed",
            "stick2_linear_speed",
            "max_linear_speed",
            "stick1_angular_speed",
            "stick2_angular_speed",
            "max_angular_speed",
            "setting_valid",
            "success_stable_steps",
        ]

        try:
            success_cfg = self._env.termination_manager.get_term_cfg("success")
        except ValueError:
            return
        required_params = {
            "sensor_groups",
            "palm_cfg",
            "stick1_cfg",
            "stick2_cfg",
            "thumb_distal_cfg",
            "index_tip_cfg",
            "middle_tip_cfg",
            "ring_tip_cfg",
            "stick1_reference_position_p",
            "stick1_reference_quaternion_p",
            "stick2_reference_position_p",
            "stick2_reference_quaternion_p",
            "contact_threshold",
            "stick1_position_error_limit",
            "stick1_orientation_error_limit",
            "stick2_position_error_limit",
            "stick2_orientation_error_limit",
            "linear_speed_limit",
            "angular_speed_limit",
            "long_axis",
            "axial_half_length",
        }
        if not required_params.issubset(success_cfg.params):
            return
        if any(
            name not in self._env.scene.sensors
            for name in self._hand_setting_sensor_names
        ):
            return

        try:
            region_cfgs = {
                name: self.get_term_cfg(name)
                for name in self._hand_setting_region_term_names
            }
        except ValueError:
            return

        self._hand_setting_params = success_cfg.params
        self._hand_setting_success_term = success_cfg.func
        self._hand_setting_region_cfgs = region_cfgs

        from isaac_neuromeka.tasks.manipulation.hand_grasp.mdp import (
            _body_in_box_shaft_region,
            _group_forces,
            _object_pair_speeds_relative_to_palm,
            _setting_geometry,
        )

        self._hand_setting_body_in_region = _body_in_box_shaft_region
        self._hand_setting_group_forces = _group_forces
        self._hand_setting_object_pair_speeds = (
            _object_pair_speeds_relative_to_palm
        )
        self._hand_setting_geometry = _setting_geometry

        def zeros():
            return {
                name: torch.zeros(
                    self.num_envs,
                    dtype=torch.float,
                    device=self.device,
                )
                for name in self._hand_setting_metric_names
            }

        self._hand_setting_metric_sums = zeros()
        self._hand_setting_metric_last = zeros()
        self._hand_setting_metric_min = {
            name: torch.full_like(value, float("inf"))
            for name, value in zeros().items()
        }
        self._hand_setting_metric_max = {
            name: torch.full_like(value, float("-inf"))
            for name, value in zeros().items()
        }
        self._hand_setting_metric_enabled = True

    def _compute_hand_setting_metrics(self) -> dict[str, torch.Tensor]:
        """Compute region, contact, pose, and stability progress for hand_setting."""
        params = self._hand_setting_params
        functional_forces = self._hand_setting_group_forces(
            self._env,
            params["sensor_groups"],
        )
        contacts_valid = (
            functional_forces >= float(params["contact_threshold"])
        )
        full_contact = torch.all(contacts_valid, dim=-1)

        region_scores = {
            name: cfg.func(self._env, **cfg.params)
            for name, cfg in self._hand_setting_region_cfgs.items()
        }
        region_specs = (
            (
                "thumb_distal_region",
                params["thumb_distal_cfg"],
                params["stick1_cfg"],
            ),
            (
                "index_tip_region",
                params["index_tip_cfg"],
                params["stick1_cfg"],
            ),
            (
                "middle_tip_region",
                params["middle_tip_cfg"],
                params["stick1_cfg"],
            ),
            (
                "ring_tip_region",
                params["ring_tip_cfg"],
                params["stick2_cfg"],
            ),
        )
        region_valid = torch.stack(
            [
                self._hand_setting_body_in_region(
                    self._env,
                    body_cfg,
                    object_cfg,
                    int(params["long_axis"]),
                    float(params["axial_half_length"]),
                )
                for _, body_cfg, object_cfg in region_specs
            ],
            dim=-1,
        )
        all_regions_valid = torch.all(region_valid, dim=-1)

        (
            stick1_position_error,
            stick1_orientation_error,
            stick2_position_error,
            stick2_orientation_error,
            geometry_region_valid,
        ) = self._hand_setting_geometry(
            self._env,
            params["palm_cfg"],
            params["stick1_cfg"],
            params["stick2_cfg"],
            params["thumb_distal_cfg"],
            params["index_tip_cfg"],
            params["middle_tip_cfg"],
            params["ring_tip_cfg"],
            params["stick1_reference_position_p"],
            params["stick1_reference_quaternion_p"],
            params["stick2_reference_position_p"],
            params["stick2_reference_quaternion_p"],
            int(params["long_axis"]),
            float(params["axial_half_length"]),
        )
        stick1_pose_valid = (
            (
                stick1_position_error
                <= float(params["stick1_position_error_limit"])
            )
            & (
                stick1_orientation_error
                <= float(params["stick1_orientation_error_limit"])
            )
        )
        stick2_pose_valid = (
            (
                stick2_position_error
                <= float(params["stick2_position_error_limit"])
            )
            & (
                stick2_orientation_error
                <= float(params["stick2_orientation_error_limit"])
            )
        )
        pose_valid = stick1_pose_valid & stick2_pose_valid

        (
            stick1_linear_speed,
            stick2_linear_speed,
            stick1_angular_speed,
            stick2_angular_speed,
        ) = self._hand_setting_object_pair_speeds(
            self._env,
            params["palm_cfg"],
            params["stick1_cfg"],
            params["stick2_cfg"],
        )
        max_linear_speed = torch.maximum(
            stick1_linear_speed,
            stick2_linear_speed,
        )
        max_angular_speed = torch.maximum(
            stick1_angular_speed,
            stick2_angular_speed,
        )
        setting_valid = (
            full_contact
            & geometry_region_valid
            & pose_valid
            & (
                max_linear_speed
                <= float(params["linear_speed_limit"])
            )
            & (
                max_angular_speed
                <= float(params["angular_speed_limit"])
            )
        )

        stable_steps = getattr(
            self._hand_setting_success_term,
            "_stable_steps",
            None,
        )
        if stable_steps is None:
            stable_steps = torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
            )
        else:
            stable_steps = stable_steps.float()

        metrics = {
            f"{name}_force": functional_forces[:, index]
            for index, name in enumerate(self._hand_setting_sensor_names)
        }
        metrics.update(
            {
                "min_functional_force": torch.min(
                    functional_forces,
                    dim=-1,
                ).values,
                "functional_contact_count": contacts_valid.float().sum(dim=-1),
                "functional_contact_fraction": contacts_valid.float().mean(dim=-1),
                "full_contact": full_contact.float(),
                **{
                    f"{name}_score": score
                    for name, score in region_scores.items()
                },
                "shaft_region_count": region_valid.float().sum(dim=-1),
                "shaft_region_fraction": region_valid.float().mean(dim=-1),
                "all_shaft_regions_valid": all_regions_valid.float(),
                "stick1_position_error": stick1_position_error,
                "stick1_orientation_error": stick1_orientation_error,
                "stick2_position_error": stick2_position_error,
                "stick2_orientation_error": stick2_orientation_error,
                "stick1_pose_valid": stick1_pose_valid.float(),
                "stick2_pose_valid": stick2_pose_valid.float(),
                "pose_valid": pose_valid.float(),
                "stick1_linear_speed": stick1_linear_speed,
                "stick2_linear_speed": stick2_linear_speed,
                "max_linear_speed": max_linear_speed,
                "stick1_angular_speed": stick1_angular_speed,
                "stick2_angular_speed": stick2_angular_speed,
                "max_angular_speed": max_angular_speed,
                "setting_valid": setting_valid.float(),
                "success_stable_steps": stable_steps,
            }
        )
        return metrics

    def _compute_functional_grasp_metrics(self, clearance: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute one-stick constraints with the same definitions used by tool_ready."""
        if self._functional_grasp_metric_params is None:
            return {}

        from isaac_neuromeka.mdp.rewards import object_in_finger_cage
        from isaac_neuromeka.tasks.manipulation.functional_grasp.mdp.target_grasp import (
            hand_tool_orientation_error,
            index_grip_error,
            thumb_grip_error,
        )

        p = self._functional_grasp_metric_params
        object_half_extent = p.get("object_half_extent", (0.01, 0.09, 0.01))
        cage_kwargs = {
            "object_cfg": p["object_cfg"],
            "object_half_extent": object_half_extent,
            "num_points": p.get("num_points", 3),
            "sphere_radius": p.get("sphere_radius", 0.005),
            "depth_max": p.get("depth_max", 0.005),
            "point_fractions": p.get("point_fractions"),
        }
        index_gate = object_in_finger_cage(self._env, p["index_cage_cfg"], **cage_kwargs)
        middle_gate = object_in_finger_cage(self._env, p["middle_cage_cfg"], **cage_kwargs)
        tripod_gate = torch.minimum(index_gate, middle_gate)

        index_error = index_grip_error(
            self._env,
            p["palm_cfg"],
            p["index_cfg"],
            p["object_cfg"],
            object_half_extent=object_half_extent,
            **(p.get("index_target") or {}),
        )
        thumb_error = thumb_grip_error(
            self._env,
            p["palm_cfg"],
            p["thumb_cfg"],
            p["object_cfg"],
            object_half_extent=object_half_extent,
            **(p.get("thumb_target") or {}),
        )
        orientation_error = hand_tool_orientation_error(
            self._env,
            p["palm_cfg"],
            p["object_cfg"],
            target_buffer_name=p.get("target_buffer_name", "chopstick_target_palm_quat_o"),
            fallback_quat_o=p.get("fallback_quat_o", (1.0, 0.0, 0.0, 0.0)),
        )
        valid = (
            (index_error < p.get("index_error_limit", 0.02))
            & (thumb_error < p.get("thumb_error_limit", 0.02))
            & (orientation_error < p.get("orientation_error_limit", 0.2617993877991494))
            & (tripod_gate > p.get("gate_threshold", 0.3))
            & (clearance > p.get("clearance_threshold", 0.05))
        )
        stable_steps = getattr(self._functional_grasp_term, "_stable_steps", None)
        if stable_steps is None:
            stable_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        else:
            stable_steps = stable_steps.float()

        return {
            "tripod_index_gate": index_gate,
            "tripod_middle_gate": middle_gate,
            "tripod_gate": tripod_gate,
            "index_grip_error": index_error,
            "thumb_grip_error": thumb_error,
            "hand_stick_orientation_error": orientation_error,
            "tool_ready_valid": valid.float(),
            "tool_ready_stable_steps": stable_steps,
        }

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

        # 큐브에서 본 엄지 vs 각 손끝 방향: +1이면 큐브 양쪽에 있음.
        # Use the fixed fingertip metric bodies instead of cage indices: cage layouts can include
        # arbitrary intermediate links or tip-only bodies.
        thumb, index_tip, middle_tip = body_pos_w[:, 1], body_pos_w[:, 2], body_pos_w[:, 3]

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
        # 잔차(residual) 액션은 processed_actions가 절대 목표가 아니라 "증분"이고 절대 목표는
        # joint_pos_target에 있음. 절대형(processed_actions = 목표)과 둘 다 지원 (2026-07-23).
        target = getattr(action_term, "joint_pos_target", None)
        if target is None:
            target = action_term.processed_actions
        actual = robot.data.joint_pos[:, action_term._joint_ids]
        track_err = (target - actual).abs().amax(dim=-1)
        delta = (self._env.action_manager.action - self._env.action_manager.prev_action).abs().mean(dim=-1)

        goal_quat_w = None
        try:
            command = self._env.command_manager.get_command("cube_goal")
            if command.shape[-1] >= 7:
                goal_quat_w = command[:, 3:7]
        except (KeyError, ValueError):
            pass

        # orientation 항 이름 감지: box=box_orientation(8-대칭), chopstick=stick_orientation
        # (4-대칭, tip/tail 구분). 감지된 태스크에 맞춰 stage_active와 box_ori_error 대칭을 고름.
        orientation_stage_active = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        ori_tip_only = False
        for _cand, _tip in (("box_orientation", False), ("stick_orientation", True)):
            try:
                _ori_cfg = self.get_term_cfg(_cand)
            except ValueError:
                continue
            ori_tip_only = _tip
            _active = getattr(_ori_cfg.func, "active", None)
            if _active is not None:
                orientation_stage_active = _active.float()
            break

        metrics = {
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
            "box_ori_error": _square_prism_ori_error(
                cube.data.root_quat_w, goal_quat_w, syms=_TIP_SYMS if ori_tip_only else None
            ),
            "orientation_stage_active": orientation_stage_active,
            "cube_displacement": torch.norm(cube_offset, dim=1),
            "cube_lift": cube_offset[:, 2],
            "cube_clearance": clearance,
            "action_track_err": track_err,
            "action_delta": delta,
        }
        metrics.update(self._compute_functional_grasp_metrics(clearance))
        return metrics

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

        if self._hand_grasp_metric_enabled:
            episode_duration = torch.clamp(
                self._env.episode_length_buf[env_ids].float() * self._env.step_dt,
                min=self._env.step_dt,
            )
            for name in self._hand_grasp_metric_names:
                extras["Metrics/hand_grasp/" + name] = torch.mean(
                    self._hand_grasp_metric_sums[name][env_ids] / episode_duration
                )
                extras["Metrics/hand_grasp_final/" + name] = torch.mean(
                    self._hand_grasp_metric_last[name][env_ids]
                )
                extras["Metrics/hand_grasp_min/" + name] = torch.mean(
                    torch.nan_to_num(
                        self._hand_grasp_metric_min[name][env_ids],
                        posinf=0.0,
                    )
                )
                extras["Metrics/hand_grasp_max/" + name] = torch.mean(
                    torch.nan_to_num(
                        self._hand_grasp_metric_max[name][env_ids],
                        neginf=0.0,
                    )
                )
                self._hand_grasp_metric_sums[name][env_ids] = 0.0
                self._hand_grasp_metric_last[name][env_ids] = 0.0
                self._hand_grasp_metric_min[name][env_ids] = float("inf")
                self._hand_grasp_metric_max[name][env_ids] = float("-inf")

        if self._hand_setting_metric_enabled:
            episode_duration = torch.clamp(
                self._env.episode_length_buf[env_ids].float()
                * self._env.step_dt,
                min=self._env.step_dt,
            )
            for name in self._hand_setting_metric_names:
                extras["Metrics/hand_setting/" + name] = torch.mean(
                    self._hand_setting_metric_sums[name][env_ids]
                    / episode_duration
                )
                extras["Metrics/hand_setting_final/" + name] = torch.mean(
                    self._hand_setting_metric_last[name][env_ids]
                )
                extras["Metrics/hand_setting_min/" + name] = torch.mean(
                    torch.nan_to_num(
                        self._hand_setting_metric_min[name][env_ids],
                        posinf=0.0,
                    )
                )
                extras["Metrics/hand_setting_max/" + name] = torch.mean(
                    torch.nan_to_num(
                        self._hand_setting_metric_max[name][env_ids],
                        neginf=0.0,
                    )
                )
                self._hand_setting_metric_sums[name][env_ids] = 0.0
                self._hand_setting_metric_last[name][env_ids] = 0.0
                self._hand_setting_metric_min[name][env_ids] = float("inf")
                self._hand_setting_metric_max[name][env_ids] = float("-inf")

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

        if self._hand_grasp_metric_enabled:
            for name, metric in self._compute_hand_grasp_metrics().items():
                self._hand_grasp_metric_sums[name] += metric * dt
                self._hand_grasp_metric_last[name][:] = metric
                self._hand_grasp_metric_min[name] = torch.minimum(
                    self._hand_grasp_metric_min[name],
                    metric,
                )
                self._hand_grasp_metric_max[name] = torch.maximum(
                    self._hand_grasp_metric_max[name],
                    metric,
                )

        if self._hand_setting_metric_enabled:
            for name, metric in self._compute_hand_setting_metrics().items():
                self._hand_setting_metric_sums[name] += metric * dt
                self._hand_setting_metric_last[name][:] = metric
                self._hand_setting_metric_min[name] = torch.minimum(
                    self._hand_setting_metric_min[name],
                    metric,
                )
                self._hand_setting_metric_max[name] = torch.maximum(
                    self._hand_setting_metric_max[name],
                    metric,
                )

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
