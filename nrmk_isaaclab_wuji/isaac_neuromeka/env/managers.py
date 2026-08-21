import copy
import math
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
        self._configure_hand_object_metrics()
        self._configure_disturbance_metrics()

    def _configure_disturbance_metrics(self):
        """Enable the chopstick-disturbance diagnostics when that event exists.

        Diagnostics only.  Nothing here is read by a reward, a termination or a
        phase gate - the point of the experiment is that the *only* change to
        the task is the perturbation itself.

        The values are already per-episode scalars latched by
        ``StickDisturbance`` (not time series), so they need none of the
        sum/min/max machinery the other metric blocks use; the episode value is
        simply averaged over the environments that reset.
        """
        self._disturbance_metric_enabled = False
        self._disturbance_term = None
        try:
            term_cfg = self._env.event_manager.get_term_cfg("stick_disturbance")
        except (AttributeError, ValueError):
            return
        # For class-form terms the manager replaces ``func`` with the instance
        # (manager_base.py:347-353), which is what carries the state.
        term = getattr(term_cfg, "func", None)
        if not hasattr(term, "applied"):
            return
        self._disturbance_term = term
        self._disturbance_metric_enabled = True

    def _log_disturbance_metrics(self, env_ids, extras: dict) -> None:
        """Write the disturbance diagnostics for the environments that reset.

        The recovery statistics are conditioned on a pulse having actually
        fired.  Averaging them over every environment would mix in the
        untouched ones and drag ``minimum_contacts_after`` back towards six,
        which is exactly the number the calibration has to read.
        """
        term = self._disturbance_term
        applied = term.applied[env_ids]
        extras["Metrics/disturbance/disturbance_applied_fraction"] = torch.mean(applied)

        fired = applied > 0.5
        count = torch.clamp(fired.float().sum(), min=1.0)

        def conditional(value: torch.Tensor) -> torch.Tensor:
            return torch.sum(torch.where(fired, value, torch.zeros_like(value))) / count

        extras["Metrics/disturbance/disturbance_force_magnitude"] = conditional(
            term.force_magnitude[env_ids]
        )
        extras["Metrics/disturbance/disturbance_stick1_fraction"] = conditional(
            term.target_is_stick1[env_ids]
        )
        extras["Metrics/disturbance/contacts_before_disturbance"] = conditional(
            term.contacts_before[env_ids]
        )
        extras["Metrics/disturbance/minimum_contacts_after_disturbance"] = conditional(
            term.min_contacts_after[env_ids]
        )
        extras["Metrics/disturbance/recovered_to_six_contacts"] = conditional(
            term.recovered[env_ids]
        )
        # Only meaningful where recovery actually happened; an episode that never
        # got back to six would otherwise contribute a 0 s "instant recovery".
        recovered = term.recovered[env_ids] > 0.5
        recovered_count = torch.clamp(recovered.float().sum(), min=1.0)
        recovery_time = term.recovery_time_s[env_ids]
        extras["Metrics/disturbance/recovery_time_s"] = (
            torch.sum(torch.where(recovered, recovery_time, torch.zeros_like(recovery_time)))
            / recovered_count
        )

    def _configure_hand_object_metrics(self):
        """Enable cube grasp-force diagnostics only for hand_object.

        These exist mainly to calibrate ``HAND_OBJECT_FORCE_SATURATION_N``.
        The number that matters is ``min(inward1, inward2)`` — the bilateral
        force — and what has to be compared is its range on episodes that held
        the cube against ones that dropped it. A per-episode mean alone cannot
        show that, which is why the shared min/max/final aggregation below is
        the point rather than an extra.
        """
        self._hand_object_metric_enabled = False
        self._hand_object_sensor_names = ("stick1_cube_contact", "stick2_cube_contact")
        self._hand_object_metric_names = [
            "inward_force1",
            "inward_force2",
            "bilateral_force",
            "force1_score",
            "force2_score",
            "bilateral_force_score",
            "raw_force1_magnitude",
            "raw_force2_magnitude",
            "functional_grasp_gate",
            "cube_distance_to_stick2_tip",
            "cube_linear_speed_rel",
            "cube_angular_speed_rel",
            "cube_height",
            "support_retract_progress",
            "support_retracted",
            "holding",
            # 2026-08-07: cube_hold 의 stability 인자를 "실제로 물고 있는 순간"에만
            # 재기 위한 게이트 버전.
            #
            # 게이트 없는 cube_distance / *_speed_rel 은 접근·회전 구간(0.5~2.0 s)을
            # 포함한다.  그 구간엔 큐브가 기둥 위에 가만히 있어도 Stick2 가 움직이므로
            # **상대** 각속도가 크게 잡힌다 (실측 평균 9.98 rad/s, 에피소드의 57% 가
            # retract 이전).  그 값으로 sigma 를 판단하면 오진한다.
            #
            # 게이트는 holding( = retracted x 양쪽 힘 > 0 )을 그대로 쓴다.  cube_hold 가
            # 요구하는 조건과 같아서, 여기 나오는 값이 곧 그 보상이 보는 값이다.
            # stability = exp(-d/0.01 - v/0.05 - w/1.0) 로 아래 셋에서 바로 계산된다.
            "hold_distance",
            "hold_linear_speed",
            "hold_angular_speed",
        ]

        # {메트릭: 게이트 메트릭}.  게이트로 마스킹해 누적하므로 에피소드 전체 시간이
        # 아니라 "게이트가 켜져 있던 시간"으로 나눠야 조건부 평균이 된다.
        # hand_grasp 쪽 _hand_grasp_conditional_metrics 와 같은 방식.
        self._hand_object_conditional_metrics = {
            "hold_distance": "holding",
            "hold_linear_speed": "holding",
            "hold_angular_speed": "holding",
        }

        if any(
            name not in self._env.scene.sensors for name in self._hand_object_sensor_names
        ):
            return
        try:
            from isaac_neuromeka.tasks.manipulation.hand_grasp import (
                hand_object_env_cfg as cfg_module,
                hand_object_mdp as mdp_module,
            )
        except Exception:  # noqa: BLE001 - task package may not be importable
            return
        if "support" not in self._env.command_manager.active_terms:
            return

        self._hand_object_cfg_module = cfg_module
        self._hand_object_mdp_module = mdp_module

        # CLOSE-only actuator diagnostics.  J3/J4 are the ten distal joints
        # whose shared hand_object effort limits were substantially larger than
        # hand_final's real-hand limits; they are therefore the first place to
        # distinguish policy failure from torque saturation.  Resolve by name
        # and map physical DOF ids back to hand_action columns explicitly.
        try:
            robot = self._env.scene["robot"]
            action_term = self._env.action_manager.get_term("hand_action")
            distal_names = [
                f"finger{finger}_joint{joint}"
                for finger in range(1, 6)
                for joint in (3, 4)
            ]
            distal_ids, _ = robot.find_joints(distal_names, preserve_order=True)
            distal_ids = distal_ids.tolist() if hasattr(distal_ids, "tolist") else list(distal_ids)
            action_ids = action_term._joint_ids
            if isinstance(action_ids, slice):
                action_columns = list(distal_ids)
            else:
                action_ids = action_ids.tolist() if hasattr(action_ids, "tolist") else list(action_ids)
                action_columns = [action_ids.index(joint_id) for joint_id in distal_ids]
            self._hand_object_distal_joint_diagnostics = tuple(
                (name, int(joint_id), int(action_column))
                for name, joint_id, action_column in zip(
                    distal_names, distal_ids, action_columns, strict=True
                )
            )
        except Exception:  # noqa: BLE001 - diagnostics must never block a run
            self._hand_object_distal_joint_diagnostics = ()

        actuator_metric_names = [
            "close_gate",
            "close_hand_tracking_error_max",
            "close_computed_effort_ratio_max",
            "close_applied_effort_ratio_max",
            "close_effort_saturation_fraction",
        ]
        for joint_name, _, _ in self._hand_object_distal_joint_diagnostics:
            actuator_metric_names.extend(
                [
                    f"close_{joint_name}_target",
                    f"close_{joint_name}_actual",
                    f"close_{joint_name}_tracking_error",
                    f"close_{joint_name}_computed_torque",
                    f"close_{joint_name}_applied_torque",
                    f"close_{joint_name}_effort_ratio",
                    f"close_{joint_name}_effort_saturated",
                ]
            )
        self._hand_object_metric_names.extend(actuator_metric_names)
        self._hand_object_conditional_metrics.update(
            {
                name: "close_gate"
                for name in actuator_metric_names
                if name != "close_gate"
            }
        )

        def zeros():
            return {
                name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for name in self._hand_object_metric_names
            }

        self._hand_object_metric_sums = zeros()
        self._hand_object_metric_last = zeros()
        self._hand_object_metric_min = {
            name: torch.full((self.num_envs,), float("inf"), device=self.device)
            for name in self._hand_object_metric_names
        }
        self._hand_object_metric_max = {
            name: torch.full((self.num_envs,), float("-inf"), device=self.device)
            for name in self._hand_object_metric_names
        }
        self._hand_object_metric_enabled = True

    def _compute_hand_object_metrics(self) -> dict[str, torch.Tensor]:
        """Cube grasp forces and relative state, reusing the reward helpers.

        Everything here calls the same functions the rewards do, so a metric
        can never quietly disagree with the reward it is supposed to explain.
        """
        cfg_module = self._hand_object_cfg_module
        mdp_module = self._hand_object_mdp_module
        env = self._env

        inward1, inward2, _, raw1, raw2 = mdp_module.cube_inward_forces(
            env,
            cfg_module.STICK_1,
            cfg_module.STICK_2,
            cfg_module.STICK_TIP_OFFSET_O,
            cfg_module.STICK1_CUBE_SENSOR,
            cfg_module.STICK2_CUBE_SENSOR,
        )
        score1, score2, bilateral_score = mdp_module.bilateral_force_score(
            inward1, inward2, mdp_module.HAND_OBJECT_FORCE_SATURATION_N
        )
        distance, linear_speed, angular_speed = mdp_module.cube_relative_state(
            env, cfg_module.OBJECT, cfg_module.STICK_2, cfg_module.STICK_TIP_OFFSET_O
        )
        functional_grasp_gate = mdp_module.functional_grasp_gate(
            env,
            cfg_module.FUNCTIONAL_CONTACT_GROUPS,
            mdp_module.HAND_OBJECT_FUNCTIONAL_FORCE_SCALE_N,
        )

        support = env.command_manager.get_term("support")
        retracted = support.retracted_gate
        cube = env.scene[cfg_module.OBJECT.name]
        height = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

        # "Holding" = the support is gone and both sticks are still loaded.
        # This is the fraction that answers "did the grasp survive?", and its
        # episode mean is the hold-success rate.
        holding = retracted * (torch.minimum(inward1, inward2) > 0.0).float()

        robot = env.scene["robot"]
        hand_action = env.action_manager.get_term("hand_action")
        close_gate = env.command_manager.get_command("open_close")[:, 1]
        action_joint_ids = hand_action._joint_ids
        actual_all = robot.data.joint_pos[:, action_joint_ids]
        target_all = hand_action.joint_pos_target
        tracking_all = torch.abs(target_all - actual_all)
        effort_all = torch.clamp(
            torch.abs(robot.data.joint_effort_limits[:, action_joint_ids]),
            min=1.0e-8,
        )
        computed_all = torch.abs(robot.data.computed_torque[:, action_joint_ids])
        applied_all = torch.abs(robot.data.applied_torque[:, action_joint_ids])
        computed_ratio_all = computed_all / effort_all
        applied_ratio_all = applied_all / effort_all
        actuator_metrics = {
            "close_gate": close_gate,
            "close_hand_tracking_error_max": close_gate * torch.max(tracking_all, dim=-1).values,
            "close_computed_effort_ratio_max": close_gate
            * torch.max(computed_ratio_all, dim=-1).values,
            "close_applied_effort_ratio_max": close_gate
            * torch.max(applied_ratio_all, dim=-1).values,
            "close_effort_saturation_fraction": close_gate
            * torch.mean((computed_ratio_all > 1.0 + 1.0e-6).float(), dim=-1),
        }
        for joint_name, joint_id, action_column in self._hand_object_distal_joint_diagnostics:
            effort = torch.clamp(
                torch.abs(robot.data.joint_effort_limits[:, joint_id]), min=1.0e-8
            )
            computed = torch.abs(robot.data.computed_torque[:, joint_id])
            applied = torch.abs(robot.data.applied_torque[:, joint_id])
            target = hand_action.joint_pos_target[:, action_column]
            actual = robot.data.joint_pos[:, joint_id]
            prefix = f"close_{joint_name}"
            actuator_metrics.update(
                {
                    f"{prefix}_target": close_gate * target,
                    f"{prefix}_actual": close_gate * actual,
                    f"{prefix}_tracking_error": close_gate * torch.abs(target - actual),
                    f"{prefix}_computed_torque": close_gate * computed,
                    f"{prefix}_applied_torque": close_gate * applied,
                    f"{prefix}_effort_ratio": close_gate * (applied / effort),
                    f"{prefix}_effort_saturated": close_gate
                    * (computed > effort + 1.0e-6).float(),
                }
            )

        metrics = {
            "inward_force1": inward1,
            "inward_force2": inward2,
            "bilateral_force": torch.minimum(inward1, inward2),
            "force1_score": score1,
            "force2_score": score2,
            "bilateral_force_score": bilateral_score,
            "raw_force1_magnitude": torch.linalg.vector_norm(raw1, dim=-1),
            "raw_force2_magnitude": torch.linalg.vector_norm(raw2, dim=-1),
            "functional_grasp_gate": functional_grasp_gate,
            "cube_distance_to_stick2_tip": distance,
            "cube_linear_speed_rel": linear_speed,
            "cube_angular_speed_rel": angular_speed,
            "cube_height": height,
            "support_retract_progress": support.retract_progress,
            "support_retracted": retracted,
            "holding": holding,
            # holding 은 0/1 이라 곱하면 안 잡고 있는 스텝이 정확히 0 이 된다.
            "hold_distance": holding * distance,
            "hold_linear_speed": holding * linear_speed,
            "hold_angular_speed": holding * angular_speed,
        }
        metrics.update(actuator_metrics)
        return metrics

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
            # 손바닥이 하늘을 보나(flip). palm 로컬 +x(=팜 평면 법선, 실측 0.965,-0.008,0.262)의
            # world z성분 ∈[-1,1]. 1=완전 팜업. PalmUpProgressReward.cos_up과 동일 축이라 flip
            # 진행을 직접 봄. palm_facing(개구부 축·스틱 방향)과는 다른 량이니 혼동 주의.
            "palm_up_cos",
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
            # palm-상대 스틱 선속도 [m/s] (측정용, 2026-07-31). 던짐(스틱이 손 기준 확 튐) vs
            # lift(손과 같이 움직여 ≈0) 분포를 봐서 속도 페널티 임계값 v0를 정하려는 것. 페널티 아님.
            "stick_palm_rel_speed",
            # 팔이 튀는 원인을 "정책이 시킨 것"과 "물리가 이긴 것"으로 가르는 두 지표.
            # action은 절대 관절 목표임 (target = default + 0.2 * action).
            #   track_err 작음(<0.1) + delta 큼(>0.3) -> 팔이 명령대로 발광. 학습/보상 문제
            #   track_err 큼  (>0.3)                  -> 물리가 명령을 이김. dt/decimation 문제
            "action_track_err",  # |관절목표 - 관절실제| 최대 [rad]
            "action_delta",  # |a_t - a_{t-1}| 평균. action 범위가 [-1,1]임
            # arm(joint0-5) 관절 명령(절대 target, rad) per-joint (2026-08-03). 팔이 어떤 자세를
            # 명령받는지 = 손 위치가 왜 목표에서 벗어나는지. cube_final/min/max로 끝 자세·범위 관찰.
            # Indy7 joint0 감소 = 손 하강 방향. track_err 작으면 명령≈실제라 이게 실제 자세.
            "arm_j0_cmd", "arm_j1_cmd", "arm_j2_cmd",
            "arm_j3_cmd", "arm_j4_cmd", "arm_j5_cmd",
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
            # 2026-08-06: OPEN/CLOSE 를 섞어 보면 어느 쪽이 안 되는지 알 수 없어 분리.
            #   누적은 gate 로 마스킹해서 하고(비활성 스텝은 0), 로그 시점에 그 모드가
            #   켜져 있던 시간으로 다시 나눈다 -> _hand_grasp_conditional_metrics 참고.
            "open_tip_gap",
            "close_tip_gap",
            "open_tip_gap_error",
            "close_tip_gap_error",
            # 2026-08-18: 위 네 개는 세그먼트 *전체* 평균이라 "명령이 끝날 때
            #   실제로 도달했나"를 못 읽는다.  OPEN 세그먼트는 2 s 인데 닫힌
            #   상태에서 출발하므로, 끝에 가서야 벌어져도 평균은 중간값이 나온다.
            #   실제로 2026-08-18_02-02-33 의 iteration 2200 은 open_tip_gap
            #   11.99 mm / close_tip_gap -0.63 mm 로 읽혔지만 play 에서
            #   CLOSE -> OPEN 이 되지 않았다.
            #   아래 두 개는 모드가 바뀌는 순간 "직전 스텝의 갭"을 latch 해서
            #   들고 있는다.  ``Metrics/hand_grasp_final/*`` 로 읽으면 그
            #   에피소드의 마지막 OPEN/CLOSE 세그먼트가 끝났을 때의 갭이 되고,
            #   그게 회전 커리큘럼을 켜도 되는지 판단할 값이다.
            "open_segment_end_gap",
            "close_segment_end_gap",
            # 2026-08-07: CLOSE 에서 "평행하게 맞물리지 않는다"를 진단하기 위한 3종.
            #   lateral/axial 은 이미 tip_lateral_error / tip_axial_error 로 있지만
            #   에피소드 전체 평균이라 OPEN 이 섞인다.  gap 과 같은 방식으로 CLOSE 만
            #   따로 뽑아야 "닫을 때" 얼마나 어긋나는지 읽을 수 있다.
            "close_lateral_error",
            "close_axial_error",
            # 두 젓가락 샤프트 축(local +y) 사이 각도 [rad] 와 그 분해.
            #   보상에는 이 양을 재는 항이 하나도 없다 - mode_tip_gap_tracking /
            #   mode_grasp_stability 는 전부 "팁 점 두 개의 상대 위치"만 보고
            #   stick1 의 축 방향(stick1_y)은 계산조차 하지 않는다.  그래서 이건
            #   보상을 설명하는 지표가 아니라 "보이지 않는 자유도가 실제로 얼마나
            #   벌어져 있나"를 재는 지표다.
            #
            #   총각도만 보면 안 된다.  젓가락은 평행하면 개폐가 아예 안 되므로
            #   축 사이 각도 자체는 **개폐 자유도**다.  문제가 되는 건 그게 아니라
            #   두 축이 같은 평면을 벗어나는 성분(skew)이고, 그래야 팁이 서로
            #   지나치지 않고 맞물린다.  그래서 셋을 같이 본다:
            #     stick_axis_angle          총 각도 (개폐 + skew 가 섞임)
            #     stick_axis_skew_angle     면외 성분  <- 0 이어야 하는 값
            #     stick_axis_inplane_angle  면내 성분(부호 있음) <- 개폐로 움직여야 정상
            #     close_axis_skew_angle     CLOSE 구간의 skew  <- 맞물림에 직결
            #     close_axis_inplane_angle  CLOSE 구간의 면내각 <- "닫으면 평행인가"
            #     reset_axis_skew_angle     리셋(pose_005) 의 skew <- 목표 0 이 맞나
            "stick_axis_angle",
            "stick_axis_skew_angle",
            "stick_axis_inplane_angle",
            "close_axis_skew_angle",
            "close_axis_inplane_angle",
            "reset_axis_skew_angle",
            # 단면 roll [rad].  스틱이 등단면 정사각이라 "딱 맞물리는가"는 축이
            # 아니라 roll 문제다.  0 = 면이 정면으로 마주봄, pi/4(45deg) = 모서리.
            #   *_roll_face_angle    각 스틱이 상대에게 면을 내미나 모서리를 내미나
            #   stick_roll_mismatch  두 단면의 방위 차 (90deg 주기라 0~45deg 로 접음)
            #   close_roll_mismatch  CLOSE 구간만
            #   reset_roll_mismatch  리셋(pose_005) 값 <- 목표 0 이 맞나 (skew 교훈)
            "stick1_roll_face_angle",
            "stick2_roll_face_angle",
            "stick_roll_mismatch",
            "close_roll_mismatch",
            "reset_roll_mismatch",
            "stick1_pivot_error",
            "stick2_position_error",
            "stick2_orientation_error",
            "mode_geometry_valid",
            "success_stable_steps",
        ]

        # {메트릭 이름: 게이트 메트릭 이름}.  값이 gate 로 마스킹돼 누적되므로
        # 에피소드 전체 시간이 아니라 "그 모드가 켜져 있던 시간"으로 나눠야
        # 조건부 평균이 된다.  min/final 은 마스킹 때문에 의미가 없어 로그에서 뺀다.
        self._hand_grasp_conditional_metrics = {
            "open_tip_gap": "mode_open",
            "open_tip_gap_error": "mode_open",
            "close_tip_gap": "mode_close",
            "close_tip_gap_error": "mode_close",
            "close_lateral_error": "mode_close",
            "close_axial_error": "mode_close",
            "close_axis_skew_angle": "mode_close",
            "close_axis_inplane_angle": "mode_close",
            "close_roll_mismatch": "mode_close",
        }
        # 위 조건부 메트릭 중 **부호가 있는** 것.  나머지는 전부 오차(>= 0)라
        # 마스킹된 0 이 max 를 건드리지 않지만, 부호 있는 값은 CLOSE 구간이 전부
        # 음수여도 OPEN 의 마스킹 0 이 max 로 잡혀 "최댓값 0" 이라는 거짓을 만든다.
        # 그래서 mean 만 남기고 max 도 지운다.
        self._hand_grasp_signed_conditional_metrics = {
            "close_axis_inplane_angle",
        }

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
        self._hand_grasp_orientation_error_mode = success_cfg.params.get(
            "orientation_error_mode",
            "quaternion",
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
        # reset_axis_skew_angle 용 latch.  에피소드가 시작된 뒤 처음 계산되는
        # skew 를 붙잡아 그 에피소드 내내 같은 값을 내보낸다 -> mean/min/max/final
        # 이 전부 같은 값이 되어 어느 집계를 봐도 "리셋 자세의 skew"가 읽힌다.
        self._hand_grasp_reset_skew = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._hand_grasp_reset_roll_mismatch = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # 두 값이 같은 순간에 latch 되므로 플래그는 하나로 공유한다.
        self._hand_grasp_reset_skew_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # open/close_segment_end_gap 용 상태.  전환을 감지하려면 직전 스텝의
        # 모드와 갭이 둘 다 필요하다: 전환이 보이는 스텝의 갭은 이미 *새* 모드의
        # 첫 샘플이라, 끝난 세그먼트의 값은 한 스텝 전 것이다.
        self._hand_grasp_prev_mode_close = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._hand_grasp_prev_gap = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._hand_grasp_prev_mode_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._hand_grasp_segment_end_gap = {
            "open_segment_end_gap": torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            ),
            "close_segment_end_gap": torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            ),
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
        if self._hand_grasp_orientation_error_mode == "quaternion":
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
        elif self._hand_grasp_orientation_error_mode == "directed_axis":
            metric_local_y = torch.tensor(
                (0.0, 1.0, 0.0),
                dtype=stick2_quaternion_p.dtype,
                device=stick2_quaternion_p.device,
            ).expand(self.num_envs, -1)
            stick2_axis_p = math_utils.quat_apply(
                stick2_quaternion_p,
                metric_local_y,
            )
            reference_axis_p = math_utils.quat_apply(
                self._hand_grasp_stick2_quaternion_reference.expand_as(
                    stick2_quaternion_p
                ),
                metric_local_y,
            )
            stick2_orientation_error = torch.acos(
                torch.clamp(
                    torch.sum(stick2_axis_p * reference_axis_p, dim=-1),
                    min=-1.0,
                    max=1.0,
                )
            )
        else:
            raise ValueError(
                "Unsupported hand-grasp orientation_error_mode: "
                f"{self._hand_grasp_orientation_error_mode!r}"
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
            # 2026-08-07: 진단은 부호 있는 값으로 본다.
            #
            # clamp 를 켜면 "옆으로 지나가는" 상태가 gap=0 으로 보고돼, 교차하고
            # 있는데 tip_surface_gap 이 완벽하다고 나온다.  hand_move 는 보상도
            # clamp 를 껐으므로 켜두면 지표와 보상이 서로 다른 값을 보게 된다.
            #
            # 메트릭은 학습에 영향이 없으므로 세 태스크 모두 부호 있는 값으로
            # 통일한다.  tip_surface_gap 이 음수면 "두 단면이 반두께 합보다
            # 가까운데 충돌하지 않았다" = 교차 중이라는 뜻이다.
            clamp_gap=False,
        )
        tip_axial_error = torch.abs(
            tip_axial_offset - self._hand_grasp_tip_axial_reference
        )
        # Angle between the two shaft axes, both taken as local +y (the semantic
        # distal direction used everywhere else in this file).  Computed in the
        # palm frame, which is the frame the stick poses already arrive in, so
        # hand motion cannot show up as a change here.
        #
        # Deliberately unsigned and not abs()'d on the cosine: 0 rad means the
        # two sticks point the same way, and a value near pi would mean one of
        # them is pointing backwards - a distinction worth being able to see.
        local_y = torch.tensor(
            (0.0, 1.0, 0.0),
            dtype=stick1_quaternion_p.dtype,
            device=stick1_quaternion_p.device,
        ).expand(self.num_envs, -1)
        stick1_axis_p = math_utils.quat_apply(stick1_quaternion_p, local_y)
        stick2_axis_p = math_utils.quat_apply(stick2_quaternion_p, local_y)
        stick_axis_angle = torch.acos(
            torch.clamp(
                torch.sum(stick1_axis_p * stick2_axis_p, dim=-1),
                min=-1.0,
                max=1.0,
            )
        )

        # Decompose that angle against the opening/closing plane.
        #
        # Chopsticks must NOT be parallel - they pivot near the base so the tips
        # can separate, and the angle between the shafts *is* the open/close
        # degree of freedom.  Driving ``stick_axis_angle`` to zero would fight
        # the mode command directly.  What "meeting flush" actually requires is
        # that the two shafts stay **coplanar**: skew axes pass each other
        # instead of closing onto a shared line.
        #
        # The plane is spanned by Stick2's shaft (+y) and the validated
        # separation direction, so its normal in Stick2-local coordinates is the
        # in-cross-section direction perpendicular to that reference - exactly
        # the direction ``lateral_error`` already projects the tip delta onto.
        # Using the same normal keeps "lateral" meaning one thing whether it is
        # measured on the tips or on the shafts.
        reference_xz = self._hand_grasp_tip_separation_direction_stick2[[0, 2]]
        reference_xz = reference_xz / torch.clamp(
            torch.linalg.vector_norm(reference_xz), min=1.0e-8
        )
        stick1_axis_s2 = math_utils.quat_apply_inverse(
            stick2_quaternion_p, stick1_axis_p
        )
        # Plane normal (-ref_z, 0, ref_x); Stick2's own axis (0, 1, 0) lies in
        # the plane by construction, so only Stick1 can leave it.
        out_of_plane = (
            -reference_xz[1] * stick1_axis_s2[:, 0]
            + reference_xz[0] * stick1_axis_s2[:, 2]
        )
        # Both axes are unit length, so the normal component is the sine of the
        # angle between Stick1 and the plane.
        stick_axis_skew_angle = torch.asin(
            torch.clamp(torch.abs(out_of_plane), min=0.0, max=1.0)
        )
        # Signed in-plane angle, for reference: this one is *supposed* to move
        # with OPEN/CLOSE.  Watching it confirms the decomposition is picking
        # the right plane - if this stays flat while the mode switches, the
        # reference direction is wrong.
        in_plane_separation = (
            reference_xz[0] * stick1_axis_s2[:, 0]
            + reference_xz[1] * stick1_axis_s2[:, 2]
        )
        stick_axis_inplane_angle = torch.atan2(
            in_plane_separation, stick1_axis_s2[:, 1]
        )
        # 리셋 자세(pose_005)의 skew 를 에피소드마다 한 번 붙잡아 둔다.
        #
        # 왜 필요한가: 기준 평면을 만드는 TIP_SEPARATION_DIRECTION_STICK2 는
        # pose_005 의 **팁 위치 차이**에서 뽑은 상수인데, 여기서는 그걸 **축 방향**에
        # 적용하고 있다.  검증된 파지의 축이 그 평면 안에 있다는 보장은 어디에도
        # 없으므로, "skew = 0" 이 옳은 목표인지부터 확인해야 한다.
        #   리셋 skew ~ 0  -> 기준이 맞고 5.28 deg 는 정책이 만든 이탈. 0 을 목표로 OK.
        #   리셋 skew ~ 5 deg -> 검증된 파지 자체가 평면 밖. 0 으로 밀면 파지가 깨진다.
        #                        목표를 이 값으로 두거나 평면 정의를 다시 뽑아야 한다.
        #
        # 주의: latch 는 리셋 "직후 첫 계산" 시점이라 액션 한 스텝이 이미 적용된
        # 뒤다.  pose_005 그 자체가 아니라 그 +1 스텝이므로, 소수점 단위로 정확한
        # 값이 필요하면 probe 로 따로 재야 한다.
        # ------------------------------------------------------------------
        # Cross-section roll: is each stick presenting a FACE or an EDGE to the
        # other one?
        #
        # The sticks are constant square section (7 x 7 mm), so "meeting flush"
        # is a roll question, not an axis question.  Nothing in the reward set
        # measures roll - and worse, ``surface_gap`` mildly *prefers* corner-on:
        #     support = half_thickness * (|sep.x| + |sep.z|)
        # is half_thickness at face-on but 1.41x that at corner-on, so rolling
        # to a corner shrinks the reported gap and pays better at CLOSE target 0.
        #
        # ``sep`` is the *instantaneous* transverse separation direction, the
        # same one the reward's support term uses - a reference direction would
        # measure something the reward does not react to.
        # ------------------------------------------------------------------
        tip1_p = stick1_position_p + math_utils.quat_apply(
            stick1_quaternion_p,
            self._hand_grasp_stick1_tip_offset.expand(self.num_envs, -1),
        )
        tip2_p = stick2_position_p + math_utils.quat_apply(
            stick2_quaternion_p,
            self._hand_grasp_stick2_tip_offset.expand(self.num_envs, -1),
        )
        tip_delta_p = tip1_p - tip2_p
        transverse_p = tip_delta_p - (
            torch.sum(tip_delta_p * stick2_axis_p, dim=-1, keepdim=True) * stick2_axis_p
        )
        separation_p = transverse_p / torch.clamp(
            torch.linalg.vector_norm(transverse_p, dim=-1, keepdim=True), min=1.0e-8
        )

        def _roll_phase(quaternion_p, axis_p):
            """Cross-section phase of ``separation_p`` in one stick's section.

            ``separation_p`` is perpendicular to Stick2's axis by construction
            but not to Stick1's, so it is re-projected into each stick's own
            cross-section before the angle is taken; otherwise the ~7 deg axis
            difference would leak into the roll reading.
            """
            local_x = torch.tensor(
                (1.0, 0.0, 0.0),
                dtype=quaternion_p.dtype,
                device=quaternion_p.device,
            ).expand(self.num_envs, -1)
            local_z = torch.tensor(
                (0.0, 0.0, 1.0),
                dtype=quaternion_p.dtype,
                device=quaternion_p.device,
            ).expand(self.num_envs, -1)
            axis_x = math_utils.quat_apply(quaternion_p, local_x)
            axis_z = math_utils.quat_apply(quaternion_p, local_z)
            in_section = separation_p - (
                torch.sum(separation_p * axis_p, dim=-1, keepdim=True) * axis_p
            )
            in_section = in_section / torch.clamp(
                torch.linalg.vector_norm(in_section, dim=-1, keepdim=True), min=1.0e-8
            )
            component_x = torch.sum(in_section * axis_x, dim=-1)
            component_z = torch.sum(in_section * axis_z, dim=-1)
            # 0 rad = a face squarely across the gap, pi/4 = a corner leading.
            face_angle = torch.atan2(
                torch.minimum(component_x.abs(), component_z.abs()),
                torch.maximum(component_x.abs(), component_z.abs()),
            )
            return face_angle, torch.atan2(component_z, component_x)

        stick1_roll_face_angle, phase1 = _roll_phase(stick1_quaternion_p, stick1_axis_p)
        stick2_roll_face_angle, phase2 = _roll_phase(stick2_quaternion_p, stick2_axis_p)
        # Relative roll, folded into [0, pi/4]: a square section repeats every
        # 90 deg, so 0 and 90 deg of relative roll are the same physical state.
        # 0 means the two sections are aligned, i.e. whatever each presents, they
        # present it to each other; pi/4 means one shows a face to the other's
        # corner.
        quarter = 0.5 * math.pi
        roll_offset = torch.remainder(phase1 - phase2, quarter)
        stick_roll_mismatch = torch.minimum(roll_offset, quarter - roll_offset)

        # Latch the reset-pose values once per episode.  The skew reading taught
        # the lesson: 5.65 deg at pose_005 meant "target zero" would have pushed
        # the policy away from the validated grasp.  Roll gets the same check
        # before any term is built on it.
        newly_reset = ~self._hand_grasp_reset_skew_latched
        if bool(newly_reset.any()):
            self._hand_grasp_reset_skew = torch.where(
                newly_reset, stick_axis_skew_angle, self._hand_grasp_reset_skew
            )
            self._hand_grasp_reset_roll_mismatch = torch.where(
                newly_reset, stick_roll_mismatch, self._hand_grasp_reset_roll_mismatch
            )
            self._hand_grasp_reset_skew_latched |= newly_reset
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

        # OPEN/CLOSE 세그먼트가 *끝났을 때*의 갭을 latch 한다.  전환이 보이는
        # 스텝의 gap 은 이미 새 모드의 첫 샘플이므로 직전 스텝 값을 쓴다.
        # 에피소드 첫 스텝은 비교할 이전 모드가 없어 건너뛴다.
        mode_close_now = mode[:, 1] > 0.5
        changed = (mode_close_now != self._hand_grasp_prev_mode_close)
        changed &= self._hand_grasp_prev_mode_valid
        ended_open = changed & ~self._hand_grasp_prev_mode_close
        ended_close = changed & self._hand_grasp_prev_mode_close
        self._hand_grasp_segment_end_gap["open_segment_end_gap"] = torch.where(
            ended_open,
            self._hand_grasp_prev_gap,
            self._hand_grasp_segment_end_gap["open_segment_end_gap"],
        )
        self._hand_grasp_segment_end_gap["close_segment_end_gap"] = torch.where(
            ended_close,
            self._hand_grasp_prev_gap,
            self._hand_grasp_segment_end_gap["close_segment_end_gap"],
        )
        self._hand_grasp_prev_mode_close = mode_close_now
        self._hand_grasp_prev_gap = tip_surface_gap.clone()
        self._hand_grasp_prev_mode_valid = torch.ones_like(
            self._hand_grasp_prev_mode_valid
        )

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
                # mode[:, 0]/[:, 1] 는 one-hot 이라 곱하면 비활성 모드가 0 이 된다.
                "open_tip_gap": mode[:, 0] * tip_surface_gap,
                "close_tip_gap": mode[:, 1] * tip_surface_gap,
                "open_tip_gap_error": mode[:, 0] * tip_gap_error,
                "close_tip_gap_error": mode[:, 1] * tip_gap_error,
                # 마스킹하지 않는다.  latch 된 값을 매 스텝 그대로 내보내므로
                # ``_final`` 집계가 "마지막으로 끝난 세그먼트의 갭"이 된다.
                "open_segment_end_gap": self._hand_grasp_segment_end_gap[
                    "open_segment_end_gap"
                ],
                "close_segment_end_gap": self._hand_grasp_segment_end_gap[
                    "close_segment_end_gap"
                ],
                "close_lateral_error": mode[:, 1] * tip_lateral_error,
                "close_axial_error": mode[:, 1] * tip_axial_error,
                "close_axis_skew_angle": mode[:, 1] * stick_axis_skew_angle,
                "close_axis_inplane_angle": mode[:, 1] * stick_axis_inplane_angle,
                "stick_axis_angle": stick_axis_angle,
                "stick_axis_skew_angle": stick_axis_skew_angle,
                "stick_axis_inplane_angle": stick_axis_inplane_angle,
                "reset_axis_skew_angle": self._hand_grasp_reset_skew,
                "stick1_roll_face_angle": stick1_roll_face_angle,
                "stick2_roll_face_angle": stick2_roll_face_angle,
                "stick_roll_mismatch": stick_roll_mismatch,
                "close_roll_mismatch": mode[:, 1] * stick_roll_mismatch,
                "reset_roll_mismatch": self._hand_grasp_reset_roll_mismatch,
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
        distal_joint_specs = (
            # 2026-08-04: 엄지 opposition/reach 진단 — joint1(opposition 회전)·joint3 추가.
            #   "엄지가 스틱 위로 못 넘어가는 게 joint1이 action 받고도 안 움직여서인지" 판별용.
            #   action 큼+position 정체+tracking_error 큼 → 막힘(limit/self-collision), action~0 → 미명령.
            ("thumb_joint1", "finger1_joint1"),
            ("thumb_joint3", "finger1_joint3"),
            ("thumb_joint4", "finger1_joint4"),
            ("index_joint3", "finger2_joint3"),
            ("index_joint4", "finger2_joint4"),
            ("middle_joint3", "finger3_joint3"),
            ("middle_joint4", "finger3_joint4"),
            ("ring_joint3", "finger4_joint3"),
            ("ring_joint4", "finger4_joint4"),
            ("little_joint4", "finger5_joint4"),
        )
        missing_joint_score_labels = tuple(
            f"{finger_name}_joint{joint_index}"
            for finger_name in ("index", "middle", "ring", "little")
            for joint_index in range(1, 5)
        )
        self._hand_setting_metric_names = [
            *[f"{name}_force" for name in self._hand_setting_sensor_names],
            "min_functional_force",
            "functional_contact_count",
            "functional_contact_fraction",
            "full_contact",
            "shaft_region_count",
            "shaft_region_fraction",
            "all_shaft_regions_valid",
            "stick1_position_error",
            "stick1_orientation_error",
            "stick2_position_error",
            "stick2_orientation_error",
            "stick1_pose_valid",
            "stick2_pose_valid",
            "stick2_valley_pose_valid",
            "stick2_in_valley",
            "stick2_seated",
            "stage1_pair_score",
            "thumb_pivot_distance",
            "thumb_pivot_score",
            "stage1_ready",
            "stage1_unlocked",
            "missing_joint_best_score",
            "semantic_approach_mean_score",
            "semantic_approach_min_score",
            "semantic_approach_score",
            "index_between_coordinate",
            "index_between_slab_score",
            "index_between_stick1_shaft_score",
            "index_between_score",
            "index_between_progress",
            "index_between_ready",
            "index_between_handoff_scale",
            "index_stick1_upper_surface_coordinate",
            "index_stick1_upper_surface_error",
            "index_stick1_upper_surface_score",
            "index_wrong_stick2_force",
            "index_wrong_stick2_contact",
            "index_wrong_stick2_penalty_score",
            *[
                f"{label}_{score_name}"
                for label in missing_joint_score_labels
                for score_name in (
                    "current_linear_score",
                    "best_linear_score",
                )
            ],
            "all_joint_reference_rmse",
            "all_joint_reference_max_error",
            "all_joint_within_5deg",
            "stage2_joint_ready",
            "stage2_ready",
            "stage2_contact_progress",
            "thumb_joint2_position",
            "thumb_joint2_target",
            "thumb_joint2_action",
            "thumb_joint_reference_rmse",
            "index_joint_reference_score",
            "middle_joint_reference_score",
            "ring_joint_reference_score",
            "little_joint_reference_score",
            "index_joint_reference_linear_score",
            "middle_joint_reference_linear_score",
            "ring_joint_reference_linear_score",
            "little_joint_reference_linear_score",
            "index_joint_reference_rmse",
            "middle_joint_reference_rmse",
            "ring_joint_reference_rmse",
            "little_joint_reference_rmse",
            "thumb_joint_reference_max_error",
            "index_joint_reference_max_error",
            "middle_joint_reference_max_error",
            "ring_joint_reference_max_error",
            "little_joint_reference_max_error",
            *[
                f"{label}_{field}"
                for label, _ in distal_joint_specs
                for field in (
                    "position",
                    "target",
                    "action",
                    "tracking_error",
                    "reference_error",
                )
            ],
            "index_tip_stick1_surface_distance",
            "middle_tip_stick1_surface_distance",
            "ring_tip_stick2_surface_distance",
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
            success_params = success_cfg.params
            success_term = success_cfg.func
        except ValueError:
            # hand_setting may deliberately park success termination while
            # retaining the exact same contact/pose contract for diagnostics.
            success_cfg = None
            success_params = getattr(
                self._env.cfg,
                "hand_setting_metric_params",
                None,
            )
            success_term = None
            if success_params is None:
                return
            # Unlike params owned by a Reward/Termination term, this task-level
            # diagnostics dictionary is not resolved automatically by an Isaac
            # Lab manager.  Resolve a private copy once so named bodies such as
            # the four semantic fingertips do not retain ``slice(None)`` IDs.
            success_params = copy.deepcopy(success_params)
            for value in success_params.values():
                if isinstance(value, SceneEntityCfg):
                    value.resolve(self._env.scene)

        try:
            thumb_pivot_cfg = self.get_term_cfg(
                "reference_thumb_pivot_min"
            )
            stage1_cfg = self.get_term_cfg("stage1_joint_reference")
            missing_joint_cfg = self.get_term_cfg(
                "stage1_missing_joint_reference"
            )
            try:
                linear_missing_joint_cfg = self.get_term_cfg(
                    "stage1_missing_joint_best_so_far"
                )
            except ValueError:
                # Rollback compatibility with the parked per-step annuity.
                linear_missing_joint_cfg = self.get_term_cfg(
                    "stage1_missing_joint_linear_reference"
                )
        except ValueError:
            return
        try:
            semantic_approach_cfg = self.get_term_cfg(
                "stage1_semantic_surface_approach"
            )
        except ValueError:
            # Keep all pre-existing hand_setting metrics alive when this
            # optional A/B term is disabled or absent.
            semantic_approach_cfg = None
        try:
            index_between_cfg = self.get_term_cfg("stage1_index_between")
        except ValueError:
            index_between_cfg = None
        try:
            index_upper_surface_cfg = self.get_term_cfg(
                "stage1_index_stick1_upper_surface"
            )
        except ValueError:
            index_upper_surface_cfg = None
        try:
            index_wrong_contact_cfg = self.get_term_cfg(
                "index_wrong_stick2_contact"
            )
        except ValueError:
            index_wrong_contact_cfg = None
        try:
            stage2_contact_cfg = self.get_term_cfg("stage2_contact_mean")
        except ValueError:
            # Historical runs used the strict Stage-2 metric without an active
            # contact-shaping term.
            stage2_contact_cfg = None
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
        if not required_params.issubset(success_params):
            return
        required_thumb_pivot_params = {
            "stick1_cfg",
            "thumb_cfg",
            "stick1_half_extent",
            "long_axis",
            "pivot_station",
            "thumb_sigma",
        }
        if not required_thumb_pivot_params.issubset(
            thumb_pivot_cfg.params
        ):
            return
        required_stage1_params = {
            "asset_cfg",
            "reference_joint_positions",
            "stage2_joint_error_threshold",
            "position_sigma",
            "orientation_sigma",
            "pair_score_threshold",
            "thumb_score_threshold",
        }
        if not required_stage1_params.issubset(stage1_cfg.params):
            return
        required_missing_joint_params = {
            "asset_cfg",
            "reference_joint_positions",
            "joint_sigma",
        }
        if not required_missing_joint_params.issubset(
            missing_joint_cfg.params
        ):
            return
        required_linear_missing_joint_params = {
            "asset_cfg",
            "reference_joint_positions",
            "joint_linear_range",
        }
        if not required_linear_missing_joint_params.issubset(
            linear_missing_joint_cfg.params
        ):
            return
        if any(
            name not in self._env.scene.sensors
            for name in self._hand_setting_sensor_names
        ):
            return

        try:
            robot = self._env.scene["robot"]
            thumb_joint_ids, _ = robot.find_joints(["finger1_joint2"])
            hand_action_term = self._env.action_manager.get_term("hand_action")
            thumb_action_index = list(
                hand_action_term._joint_names
            ).index("finger1_joint2")
            missing_asset_cfg = missing_joint_cfg.params["asset_cfg"]
            missing_joint_ids = list(missing_asset_cfg.joint_ids)
            missing_joint_reference = torch.as_tensor(
                missing_joint_cfg.params["reference_joint_positions"],
                dtype=robot.data.joint_pos.dtype,
                device=self.device,
            )
            full_asset_cfg = stage1_cfg.params["asset_cfg"]
            full_joint_ids = list(full_asset_cfg.joint_ids)
            full_joint_reference = torch.as_tensor(
                stage1_cfg.params["reference_joint_positions"],
                dtype=robot.data.joint_pos.dtype,
                device=self.device,
            )
            if len(full_joint_ids) % 5 != 0:
                return
            full_joint_group_size = len(full_joint_ids) // 5
            stage2_joint_error_threshold = float(
                stage1_cfg.params["stage2_joint_error_threshold"]
            )
            # The active setting guide scores the sixteen non-thumb joints
            # individually.  Derive four four-joint finger groups locally.
            missing_finger_count = 4
            if len(missing_joint_ids) % missing_finger_count != 0:
                return
            missing_joint_group_size = (
                len(missing_joint_ids) // missing_finger_count
            )
            missing_joint_sigma = float(
                missing_joint_cfg.params["joint_sigma"]
            )
            missing_joint_linear_range = float(
                linear_missing_joint_cfg.params["joint_linear_range"]
            )
            linear_asset_cfg = linear_missing_joint_cfg.params["asset_cfg"]
            linear_joint_ids = list(linear_asset_cfg.joint_ids)
            linear_joint_reference = torch.as_tensor(
                linear_missing_joint_cfg.params[
                    "reference_joint_positions"
                ],
                dtype=robot.data.joint_pos.dtype,
                device=self.device,
            )
            full_reference_by_id = {
                int(joint_id): full_joint_reference[index]
                for index, joint_id in enumerate(full_joint_ids)
            }
            action_joint_names = list(hand_action_term._joint_names)
            distal_joint_diagnostics = []
            for label, joint_name in distal_joint_specs:
                joint_ids, _ = robot.find_joints([joint_name])
                if len(joint_ids) != 1:
                    raise ValueError(
                        f"Expected one joint for {joint_name}, got {joint_ids}"
                    )
                joint_id = int(joint_ids[0])
                distal_joint_diagnostics.append(
                    (
                        label,
                        joint_id,
                        action_joint_names.index(joint_name),
                        full_reference_by_id[joint_id],
                    )
                )
            if (
                missing_joint_group_size <= 0
                or missing_joint_reference.numel() != len(missing_joint_ids)
                or linear_joint_ids != missing_joint_ids
                or not torch.equal(
                    linear_joint_reference,
                    missing_joint_reference,
                )
                or full_joint_reference.numel() != len(full_joint_ids)
                or missing_joint_sigma <= 0.0
                or missing_joint_linear_range <= 0.0
                or full_joint_group_size <= 0
                or stage2_joint_error_threshold <= 0.0
            ):
                return
        except (KeyError, TypeError, ValueError, IndexError):
            return

        self._hand_setting_params = success_params
        self._hand_setting_thumb_pivot_params = thumb_pivot_cfg.params
        self._hand_setting_stage1_params = stage1_cfg.params
        self._hand_setting_success_term = success_term
        self._hand_setting_success_stable_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._hand_setting_thumb_joint2_id = int(thumb_joint_ids[0])
        self._hand_setting_thumb_action_index = thumb_action_index
        self._hand_setting_missing_joint_ids = missing_joint_ids
        self._hand_setting_missing_joint_reference = missing_joint_reference
        self._hand_setting_missing_finger_count = missing_finger_count
        self._hand_setting_missing_joint_group_size = missing_joint_group_size
        self._hand_setting_missing_joint_sigma = missing_joint_sigma
        self._hand_setting_missing_joint_linear_range = (
            missing_joint_linear_range
        )
        self._hand_setting_best_so_far_term = (
            linear_missing_joint_cfg.func
            if hasattr(linear_missing_joint_cfg.func, "best_score")
            else None
        )
        self._hand_setting_full_joint_ids = full_joint_ids
        self._hand_setting_full_joint_reference = full_joint_reference
        self._hand_setting_full_joint_group_size = full_joint_group_size
        self._hand_setting_stage2_joint_error_threshold = (
            stage2_joint_error_threshold
        )
        self._hand_setting_stage2_contact_start_rmse = (
            float(stage2_contact_cfg.params["joint_error_start_threshold"])
            if stage2_contact_cfg is not None
            and stage2_contact_cfg.params.get("joint_error_start_threshold")
            is not None
            else None
        )
        self._hand_setting_stage2_contact_term = (
            stage2_contact_cfg.func
            if stage2_contact_cfg is not None
            and hasattr(stage2_contact_cfg.func, "contact_progress")
            else None
        )
        if (
            self._hand_setting_stage2_contact_start_rmse is not None
            and self._hand_setting_stage2_contact_start_rmse
            <= stage2_joint_error_threshold
        ):
            return
        self._hand_setting_distal_joint_diagnostics = (
            distal_joint_diagnostics
        )
        self._hand_setting_missing_joint_score_labels = (
            missing_joint_score_labels
        )
        self._hand_setting_semantic_approach_range = (
            float(semantic_approach_cfg.params["approach_range"])
            if semantic_approach_cfg is not None
            and float(semantic_approach_cfg.params.get("approach_range", 0.0))
            > 0.0
            else None
        )
        self._hand_setting_index_between_params = (
            index_between_cfg.params
            if index_between_cfg is not None
            else None
        )
        self._hand_setting_index_between_term = (
            index_between_cfg.func
            if index_between_cfg is not None
            and hasattr(index_between_cfg.func, "handoff_scale")
            else None
        )
        self._hand_setting_index_upper_surface_params = (
            index_upper_surface_cfg.params
            if index_upper_surface_cfg is not None
            else None
        )
        self._hand_setting_index_between_progress_start = (
            float(stage2_contact_cfg.params["index_between_progress_start"])
            if stage2_contact_cfg is not None
            and stage2_contact_cfg.params.get("index_cfg") is not None
            else None
        )
        self._hand_setting_index_between_ready_threshold = (
            float(stage2_contact_cfg.params["index_between_ready_threshold"])
            if stage2_contact_cfg is not None
            and stage2_contact_cfg.params.get("index_cfg") is not None
            else None
        )
        self._hand_setting_wrong_index_sensor_names = tuple(
            name
            for name in (
                "index_link1_stick2_wrong",
                "index_link2_stick2_wrong",
                "index_link3_stick2_wrong",
                "index_link4_stick2_wrong",
                "index_tip_stick2_wrong",
            )
            if name in self._env.scene.sensors
        )
        self._hand_setting_wrong_index_force_scale = (
            float(index_wrong_contact_cfg.params["force_scale"])
            if index_wrong_contact_cfg is not None
            else 0.10
        )

        from isaac_neuromeka.tasks.manipulation.hand_grasp.mdp import (
            _body_in_box_shaft_region,
            _group_forces,
            _object_pair_speeds_relative_to_palm,
            _setting_geometry,
            body_box_axial_station_distance,
            body_box_surface_region_geometry,
            body_box_surface_distance,
            index_between_sticks_geometry,
        )

        self._hand_setting_body_in_region = _body_in_box_shaft_region
        self._hand_setting_group_forces = _group_forces
        self._hand_setting_object_pair_speeds = (
            _object_pair_speeds_relative_to_palm
        )
        self._hand_setting_geometry = _setting_geometry
        self._hand_setting_thumb_pivot_distance = (
            body_box_axial_station_distance
        )
        self._hand_setting_body_surface_distance = body_box_surface_distance
        self._hand_setting_body_surface_region_geometry = (
            body_box_surface_region_geometry
        )
        self._hand_setting_index_between_geometry = (
            index_between_sticks_geometry
        )

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
        # The stable hand_grasp Stick2 pose is the active semantic valley.
        # This is intentionally identical to the strict Stick2 pose-valid
        # condition used by the final setting validator.
        stick2_valley_pose_valid = stick2_pose_valid
        # Physical in-valley validation requires the strict reference pose and
        # reciprocal palm/thumb-middle support.
        stick2_in_valley = (
            stick2_valley_pose_valid
            & contacts_valid[:, 3]
            & contacts_valid[:, 4]
        )
        stick2_seated = stick2_in_valley & contacts_valid[:, 5]
        pose_valid = stick1_pose_valid & stick2_pose_valid
        pivot_params = self._hand_setting_thumb_pivot_params
        thumb_pivot_distance = self._hand_setting_thumb_pivot_distance(
            self._env,
            pivot_params["thumb_cfg"],
            pivot_params["stick1_cfg"],
            pivot_params["stick1_half_extent"],
            int(pivot_params["long_axis"]),
            float(pivot_params["pivot_station"]),
        )
        thumb_pivot_score = torch.exp(
            -thumb_pivot_distance / float(pivot_params["thumb_sigma"])
        )
        stage1_params = self._hand_setting_stage1_params
        stage1_stick1_score = torch.exp(
            -stick1_position_error / float(stage1_params["position_sigma"])
            -stick1_orientation_error
            / float(stage1_params["orientation_sigma"])
        )
        stage1_stick2_score = torch.exp(
            -stick2_position_error / float(stage1_params["position_sigma"])
            -stick2_orientation_error
            / float(stage1_params["orientation_sigma"])
        )
        stage1_pair_score = torch.minimum(
            stage1_stick1_score,
            stage1_stick2_score,
        )
        stage1_ready = (
            (
                stage1_pair_score
                >= float(stage1_params["pair_score_threshold"])
            )
            & (
                thumb_pivot_score
                >= float(stage1_params["thumb_score_threshold"])
            )
        )

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
        robot = self._env.scene["robot"]
        hand_action_term = self._env.action_manager.get_term("hand_action")
        thumb_joint2_position = robot.data.joint_pos[
            :,
            self._hand_setting_thumb_joint2_id,
        ]
        thumb_joint2_target = hand_action_term.joint_pos_target[
            :,
            self._hand_setting_thumb_action_index,
        ]
        thumb_joint2_action = hand_action_term.raw_actions[
            :,
            self._hand_setting_thumb_action_index,
        ]
        full_joint_error = (
            robot.data.joint_pos[
                :,
                self._hand_setting_full_joint_ids,
            ]
            - self._hand_setting_full_joint_reference
        )
        full_joint_abs_error = torch.abs(full_joint_error)
        all_joint_reference_rmse = torch.sqrt(
            torch.mean(torch.square(full_joint_error), dim=-1)
        )
        all_joint_reference_max_error = torch.max(
            full_joint_abs_error,
            dim=-1,
        ).values
        all_joint_within_5deg = (
            all_joint_reference_max_error
            <= self._hand_setting_stage2_joint_error_threshold
        )
        stage2_joint_ready = stage1_ready & all_joint_within_5deg
        index_between_params = self._hand_setting_index_between_params
        if index_between_params is None:
            index_between_coordinate = torch.zeros_like(stage1_pair_score)
            index_between_slab_score = torch.zeros_like(stage1_pair_score)
            index_between_stick1_shaft_score = torch.zeros_like(
                stage1_pair_score
            )
            index_between_score = torch.zeros_like(stage1_pair_score)
        else:
            (
                index_between_coordinate,
                index_between_slab_score,
                index_between_stick1_shaft_score,
                index_between_score,
            ) = self._hand_setting_index_between_geometry(
                self._env,
                index_between_params["index_cfg"],
                index_between_params["stick1_cfg"],
                index_between_params["stick2_cfg"],
                index_between_params["stick_half_extent"],
                int(index_between_params["long_axis"]),
                float(index_between_params["axial_half_length"]),
                float(index_between_params["between_margin_fraction"]),
                float(index_between_params["stick1_proximity_sigma"]),
            )
        index_upper_surface_params = (
            self._hand_setting_index_upper_surface_params
        )
        if index_upper_surface_params is None:
            index_stick1_upper_surface_coordinate = torch.zeros_like(
                stage1_pair_score
            )
            index_stick1_upper_surface_error = torch.zeros_like(
                stage1_pair_score
            )
            index_stick1_upper_surface_score = torch.zeros_like(
                stage1_pair_score
            )
        else:
            (
                index_stick1_upper_surface_coordinate,
                index_stick1_upper_surface_error,
                index_stick1_upper_surface_score,
            ) = self._hand_setting_body_surface_region_geometry(
                self._env,
                index_upper_surface_params["index_cfg"],
                index_upper_surface_params["stick1_cfg"],
                index_upper_surface_params["object_half_extent"],
                int(index_upper_surface_params["long_axis"]),
                float(index_upper_surface_params["axial_half_length"]),
                int(index_upper_surface_params["surface_axis"]),
                float(index_upper_surface_params["surface_sign"]),
                float(index_upper_surface_params["tangent_margin"]),
                float(index_upper_surface_params["region_sigma"]),
            )
        if (
            self._hand_setting_index_between_progress_start is None
            or self._hand_setting_index_between_ready_threshold is None
        ):
            # Report the parked feature as zero, but use neutral multiplicative
            # gates so the historical Stage-2 logic remains unchanged.
            index_between_progress = torch.zeros_like(stage1_pair_score)
            index_between_ready = torch.zeros_like(
                stage1_ready,
                dtype=torch.bool,
            )
            index_between_contact_gate = torch.ones_like(stage1_pair_score)
            index_between_stage2_gate = torch.ones_like(
                stage1_ready,
                dtype=torch.bool,
            )
        else:
            between_start = self._hand_setting_index_between_progress_start
            between_ready = self._hand_setting_index_between_ready_threshold
            index_between_progress = torch.clamp(
                (index_between_score - between_start)
                / (between_ready - between_start),
                min=0.0,
                max=1.0,
            )
            index_between_ready = index_between_score >= between_ready
            index_between_contact_gate = index_between_progress
            index_between_stage2_gate = index_between_ready
        if self._hand_setting_index_between_term is None:
            index_between_handoff_scale = torch.zeros_like(
                stage1_pair_score
            )
        else:
            index_between_handoff_scale = (
                self._hand_setting_index_between_term.handoff_scale
            )
        stage2_ready = stage2_joint_ready & index_between_stage2_gate
        if self._hand_setting_stage2_contact_term is not None:
            # RewardManager has already evaluated the stateful contact term for
            # this step, so this is exactly the multiplier used by the reward.
            stage2_contact_progress = (
                self._hand_setting_stage2_contact_term.contact_progress
            )
        elif self._hand_setting_stage2_contact_start_rmse is None:
            stage2_contact_progress = torch.zeros_like(
                all_joint_reference_rmse
            )
        else:
            stage2_contact_progress = stage1_ready.float() * torch.clamp(
                (
                    self._hand_setting_stage2_contact_start_rmse
                    - all_joint_reference_rmse
                )
                / (
                    self._hand_setting_stage2_contact_start_rmse
                    - self._hand_setting_stage2_joint_error_threshold
                ),
                min=0.0,
                max=1.0,
            )
            stage2_contact_progress *= index_between_contact_gate
        if self._hand_setting_wrong_index_sensor_names:
            index_wrong_stick2_force = self._hand_setting_group_forces(
                self._env,
                (self._hand_setting_wrong_index_sensor_names,),
            )[:, 0]
        else:
            index_wrong_stick2_force = torch.zeros_like(stage1_pair_score)
        index_wrong_stick2_contact = (
            index_wrong_stick2_force >= float(params["contact_threshold"])
        )
        index_wrong_stick2_penalty_score = torch.clamp(
            index_wrong_stick2_force
            / self._hand_setting_wrong_index_force_scale,
            min=0.0,
            max=1.0,
        ) * (1.0 - index_between_score)
        best_so_far_term = self._hand_setting_best_so_far_term
        if best_so_far_term is None:
            stage1_unlocked = torch.zeros_like(stage1_ready)
            missing_joint_best_score = torch.zeros_like(
                all_joint_reference_rmse
            )
            missing_joint_best_scores = torch.zeros(
                (
                    self.num_envs,
                    len(self._hand_setting_missing_joint_score_labels),
                ),
                dtype=all_joint_reference_rmse.dtype,
                device=self.device,
            )
            missing_joint_current_scores = torch.zeros_like(
                missing_joint_best_scores
            )
        else:
            stage1_unlocked = best_so_far_term.stage1_unlocked
            missing_joint_best_score = best_so_far_term.best_score
            missing_joint_best_scores = best_so_far_term.best_joint_scores
            missing_joint_current_scores = (
                best_so_far_term.current_joint_scores
            )
        full_joint_abs_error_by_finger = full_joint_abs_error.reshape(
            self.num_envs,
            5,
            self._hand_setting_full_joint_group_size,
        )
        full_joint_rmse_by_finger = torch.sqrt(
            torch.mean(
                torch.square(full_joint_error).reshape(
                    self.num_envs,
                    5,
                    self._hand_setting_full_joint_group_size,
                ),
                dim=-1,
            )
        )
        full_joint_max_error_by_finger = torch.max(
            full_joint_abs_error_by_finger,
            dim=-1,
        ).values
        missing_joint_error = (
            robot.data.joint_pos[
                :,
                self._hand_setting_missing_joint_ids,
            ]
            - self._hand_setting_missing_joint_reference
        ).reshape(
            self.num_envs,
            self._hand_setting_missing_finger_count,
            self._hand_setting_missing_joint_group_size,
        )
        missing_joint_abs_error = torch.abs(missing_joint_error)
        missing_joint_squared_error = torch.square(missing_joint_error)
        missing_joint_mse = torch.mean(missing_joint_squared_error, dim=-1)
        missing_joint_rmse = torch.sqrt(missing_joint_mse)
        # Preserve the historical per-finger exponential score and expose the
        # added long-range linear score separately for TensorBoard diagnosis.
        missing_joint_score = torch.exp(
            -missing_joint_mse
            / (self._hand_setting_missing_joint_sigma**2)
        )
        missing_joint_linear_score = torch.mean(
            torch.clamp(
                1.0
                - missing_joint_abs_error
                / self._hand_setting_missing_joint_linear_range,
                min=0.0,
                max=1.0,
            ),
            dim=-1,
        )
        distal_joint_metrics = {}
        for (
            label,
            joint_id,
            action_index,
            reference_position,
        ) in self._hand_setting_distal_joint_diagnostics:
            joint_position = robot.data.joint_pos[:, joint_id]
            joint_target = hand_action_term.joint_pos_target[:, action_index]
            distal_joint_metrics.update(
                {
                    f"{label}_position": joint_position,
                    f"{label}_target": joint_target,
                    f"{label}_action": hand_action_term.raw_actions[
                        :,
                        action_index,
                    ],
                    f"{label}_tracking_error": torch.abs(
                        joint_target - joint_position
                    ),
                    f"{label}_reference_error": torch.abs(
                        joint_position - reference_position
                    ),
                }
            )
        stick_half_extent = pivot_params["stick1_half_extent"]
        missing_tip_surface_metrics = {
            "index_tip_stick1_surface_distance": (
                self._hand_setting_body_surface_distance(
                    self._env,
                    params["index_tip_cfg"],
                    params["stick1_cfg"],
                    stick_half_extent,
                )
            ),
            "middle_tip_stick1_surface_distance": (
                self._hand_setting_body_surface_distance(
                    self._env,
                    params["middle_tip_cfg"],
                    params["stick1_cfg"],
                    stick_half_extent,
                )
            ),
            "ring_tip_stick2_surface_distance": (
                self._hand_setting_body_surface_distance(
                    self._env,
                    params["ring_tip_cfg"],
                    params["stick2_cfg"],
                    stick_half_extent,
                )
            ),
        }
        if self._hand_setting_semantic_approach_range is None:
            semantic_approach_mean_score = torch.zeros_like(stage1_pair_score)
            semantic_approach_min_score = torch.zeros_like(stage1_pair_score)
            semantic_approach_score = torch.zeros_like(stage1_pair_score)
        else:
            semantic_approach_scores = torch.stack(
                tuple(
                    torch.clamp(
                        1.0
                        - missing_tip_surface_metrics[name]
                        / self._hand_setting_semantic_approach_range,
                        min=0.0,
                        max=1.0,
                    )
                    for name in (
                        "index_tip_stick1_surface_distance",
                        "middle_tip_stick1_surface_distance",
                        "ring_tip_stick2_surface_distance",
                    )
                ),
                dim=-1,
            )
            semantic_approach_mean_score = torch.mean(
                semantic_approach_scores,
                dim=-1,
            )
            semantic_approach_min_score = torch.min(
                semantic_approach_scores,
                dim=-1,
            ).values
            semantic_approach_score = stage1_unlocked.float() * (
                0.5 * semantic_approach_mean_score
                + 0.5 * semantic_approach_min_score
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
            # When success termination is parked, diagnostics own the same
            # consecutive-valid counter instead of silently reporting zero.
            self._hand_setting_success_stable_steps = torch.where(
                setting_valid,
                self._hand_setting_success_stable_steps + 1,
                torch.zeros_like(self._hand_setting_success_stable_steps),
            )
            stable_steps = self._hand_setting_success_stable_steps.float()
        else:
            stable_steps = stable_steps.float()

        metrics = {
            f"{name}_force": functional_forces[:, index]
            for index, name in enumerate(self._hand_setting_sensor_names)
        }
        for index, label in enumerate(
            self._hand_setting_missing_joint_score_labels
        ):
            metrics[f"{label}_current_linear_score"] = (
                missing_joint_current_scores[:, index]
            )
            metrics[f"{label}_best_linear_score"] = (
                missing_joint_best_scores[:, index]
            )
        metrics.update(
            {
                "min_functional_force": torch.min(
                    functional_forces,
                    dim=-1,
                ).values,
                "functional_contact_count": contacts_valid.float().sum(dim=-1),
                "functional_contact_fraction": contacts_valid.float().mean(dim=-1),
                "full_contact": full_contact.float(),
                "shaft_region_count": region_valid.float().sum(dim=-1),
                "shaft_region_fraction": region_valid.float().mean(dim=-1),
                "all_shaft_regions_valid": all_regions_valid.float(),
                "stick1_position_error": stick1_position_error,
                "stick1_orientation_error": stick1_orientation_error,
                "stick2_position_error": stick2_position_error,
                "stick2_orientation_error": stick2_orientation_error,
                "stick1_pose_valid": stick1_pose_valid.float(),
                "stick2_pose_valid": stick2_pose_valid.float(),
                "stick2_valley_pose_valid":
                    stick2_valley_pose_valid.float(),
                "stick2_in_valley": stick2_in_valley.float(),
                "stick2_seated": stick2_seated.float(),
                "stage1_pair_score": stage1_pair_score,
                "thumb_pivot_distance": thumb_pivot_distance,
                "thumb_pivot_score": thumb_pivot_score,
                "stage1_ready": stage1_ready.float(),
                "stage1_unlocked": stage1_unlocked.float(),
                "missing_joint_best_score": missing_joint_best_score,
                "semantic_approach_mean_score": (
                    semantic_approach_mean_score
                ),
                "semantic_approach_min_score": semantic_approach_min_score,
                "semantic_approach_score": semantic_approach_score,
                "index_between_coordinate": index_between_coordinate,
                "index_between_slab_score": index_between_slab_score,
                "index_between_stick1_shaft_score": (
                    index_between_stick1_shaft_score
                ),
                "index_between_score": index_between_score,
                "index_between_progress": index_between_progress,
                "index_between_ready": index_between_ready.float(),
                "index_between_handoff_scale": (
                    index_between_handoff_scale
                ),
                "index_stick1_upper_surface_coordinate": (
                    index_stick1_upper_surface_coordinate
                ),
                "index_stick1_upper_surface_error": (
                    index_stick1_upper_surface_error
                ),
                "index_stick1_upper_surface_score": (
                    index_stick1_upper_surface_score
                ),
                "index_wrong_stick2_force": index_wrong_stick2_force,
                "index_wrong_stick2_contact": (
                    index_wrong_stick2_contact.float()
                ),
                "index_wrong_stick2_penalty_score": (
                    index_wrong_stick2_penalty_score
                ),
                "all_joint_reference_rmse": all_joint_reference_rmse,
                "all_joint_reference_max_error": (
                    all_joint_reference_max_error
                ),
                "all_joint_within_5deg": all_joint_within_5deg.float(),
                "stage2_joint_ready": stage2_joint_ready.float(),
                "stage2_ready": stage2_ready.float(),
                "stage2_contact_progress": stage2_contact_progress,
                "thumb_joint2_position": thumb_joint2_position,
                "thumb_joint2_target": thumb_joint2_target,
                "thumb_joint2_action": thumb_joint2_action,
                "thumb_joint_reference_rmse": (
                    full_joint_rmse_by_finger[:, 0]
                ),
                "index_joint_reference_score": missing_joint_score[:, 0],
                "middle_joint_reference_score": missing_joint_score[:, 1],
                "ring_joint_reference_score": missing_joint_score[:, 2],
                "little_joint_reference_score": missing_joint_score[:, 3],
                "index_joint_reference_linear_score": (
                    missing_joint_linear_score[:, 0]
                ),
                "middle_joint_reference_linear_score": (
                    missing_joint_linear_score[:, 1]
                ),
                "ring_joint_reference_linear_score": (
                    missing_joint_linear_score[:, 2]
                ),
                "little_joint_reference_linear_score": (
                    missing_joint_linear_score[:, 3]
                ),
                "index_joint_reference_rmse": missing_joint_rmse[:, 0],
                "middle_joint_reference_rmse": missing_joint_rmse[:, 1],
                "ring_joint_reference_rmse": missing_joint_rmse[:, 2],
                "little_joint_reference_rmse": (
                    missing_joint_rmse[:, 3]
                ),
                "thumb_joint_reference_max_error": (
                    full_joint_max_error_by_finger[:, 0]
                ),
                "index_joint_reference_max_error": (
                    full_joint_max_error_by_finger[:, 1]
                ),
                "middle_joint_reference_max_error": (
                    full_joint_max_error_by_finger[:, 2]
                ),
                "ring_joint_reference_max_error": (
                    full_joint_max_error_by_finger[:, 3]
                ),
                "little_joint_reference_max_error": (
                    full_joint_max_error_by_finger[:, 4]
                ),
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
        metrics.update(distal_joint_metrics)
        metrics.update(missing_tip_surface_metrics)
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

        # flip 지표: 손바닥 평면 법선(로컬 +x)의 world z성분. PalmUpProgressReward.cos_up과 동일.
        # palm_facing(개구부 축·스틱 방향)이 아니라 "손바닥이 하늘을 보나"를 재는 별개 량.
        palm_x_axis_b = torch.zeros_like(palm_pos_w)
        palm_x_axis_b[:, 0] = 1.0
        palm_up_cos = math_utils.quat_apply(palm_quat_w, palm_x_axis_b)[:, 2]

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

        # palm-상대 스틱 선속도 (측정용, 2026-07-31): 던짐 vs lift 구분. body_state_w[7:10]=선속도(월드).
        # 잡고 들면 스틱이 손과 같이 움직여 ≈0, 던지면 손 기준 확 튀어 큼 → 페널티 임계값 v0 산정용.
        palm_lin_vel_w = robot.data.body_state_w[:, palm_id, 7:10]
        stick_palm_rel_speed = torch.norm(cube.data.root_lin_vel_w - palm_lin_vel_w, dim=-1)

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

        # arm(joint0-5) 절대 관절목표를 per-joint로 (2026-08-03). target은 action_order라
        # action_term._joint_ids에서 arm DOF 위치를 찾아 컬럼 인덱싱(한 번만 계산해 캐시).
        # ⚠ _joint_ids는 전(全) 관절 제어 시 slice(None)이라 list()로 못 품(TypeError로 학습 크래시한 이력,
        #   2026-08-03) → slice 분기. 진단 메트릭이 학습을 죽이면 안 되므로 어떤 실패든 zeros 폴백.
        try:
            if not hasattr(self, "_arm_action_cols"):
                _act_ids = action_term._joint_ids
                _arm_ids = self._arm_joint_ids
                _arm_ids = _arm_ids.tolist() if hasattr(_arm_ids, "tolist") else list(_arm_ids)
                if isinstance(_act_ids, slice):
                    # slice(None): action 컬럼 순서 = DOF 순서 → arm DOF id가 곧 컬럼 인덱스
                    self._arm_action_cols = list(_arm_ids)
                else:
                    _act_ids = _act_ids.tolist() if hasattr(_act_ids, "tolist") else list(_act_ids)
                    self._arm_action_cols = [_act_ids.index(j) for j in _arm_ids if j in _act_ids]
            arm_cmd = target[:, self._arm_action_cols]
            if arm_cmd.shape[1] != 6:
                arm_cmd = torch.zeros(self.num_envs, 6, device=self.device)
        except Exception:
            arm_cmd = torch.zeros(self.num_envs, 6, device=self.device)

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
            "palm_up_cos": palm_up_cos,
            "arm_manipulability": manipulability,
            "box_ori_error": _square_prism_ori_error(
                cube.data.root_quat_w, goal_quat_w, syms=_TIP_SYMS if ori_tip_only else None
            ),
            "orientation_stage_active": orientation_stage_active,
            "cube_displacement": torch.norm(cube_offset, dim=1),
            "cube_lift": cube_offset[:, 2],
            "cube_clearance": clearance,
            "stick_palm_rel_speed": stick_palm_rel_speed,
            "action_track_err": track_err,
            "action_delta": delta,
            "arm_j0_cmd": arm_cmd[:, 0],
            "arm_j1_cmd": arm_cmd[:, 1],
            "arm_j2_cmd": arm_cmd[:, 2],
            "arm_j3_cmd": arm_cmd[:, 3],
            "arm_j4_cmd": arm_cmd[:, 4],
            "arm_j5_cmd": arm_cmd[:, 5],
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
            # 아래 일반 루프가 각 이름을 처리한 직후 sums 를 0 으로 지우므로,
            # 조건부 정규화에 쓸 분자/분모는 루프 전에 미리 떠 둔다.
            conditional_sums = {}
            for name, gate_name in self._hand_grasp_conditional_metrics.items():
                for key in (name, gate_name):
                    if key not in conditional_sums:
                        conditional_sums[key] = (
                            self._hand_grasp_metric_sums[key][env_ids].clone()
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
            # 다음 에피소드의 리셋 skew 를 새로 붙잡도록 latch 해제.  이 시점은
            # reset event 가 이미 실행된 뒤라(위 _cube_init_pos 와 같은 위치)
            # 다음 스텝의 계산이 새 에피소드의 자세를 본다.
            self._hand_grasp_reset_skew_latched[env_ids] = False
            # 세그먼트 latch 도 에피소드마다 새로 시작한다.  prev_mode_valid 를
            # 내려두지 않으면 새 에피소드의 첫 스텝이 이전 에피소드의 모드와
            # 비교돼 있지도 않은 전환을 하나 만들어 낸다.
            self._hand_grasp_prev_mode_valid[env_ids] = False
            for buffer in self._hand_grasp_segment_end_gap.values():
                buffer[env_ids] = 0.0

            # 조건부 메트릭 보정.  일반 루프가 넣어 둔 값은 에피소드 전체 시간으로
            # 나뉘어 희석돼 있으므로, 해당 모드가 켜져 있던 시간으로 다시 나눈 값으로
            # 덮어쓴다.  min 은 마스킹된 0 이, final 은 마지막 스텝의 모드가 무엇이냐가
            # 잡혀 둘 다 의미가 없으므로 로그에서 제거한다.  max 는 오차가 음수가 될 수
            # 없어 마스킹 후에도 "활성 스텝 중 최대"가 그대로 남는다.
            # 진단 메트릭 하나 때문에 학습이 죽으면 안 되므로 통째로 감싼다.
            try:
                for name, gate_name in self._hand_grasp_conditional_metrics.items():
                    active_time = torch.clamp(
                        conditional_sums[gate_name],
                        min=self._env.step_dt,
                    )
                    extras["Metrics/hand_grasp/" + name] = torch.mean(
                        conditional_sums[name] / active_time
                    )
                    extras.pop("Metrics/hand_grasp_min/" + name, None)
                    extras.pop("Metrics/hand_grasp_final/" + name, None)
                    if name in self._hand_grasp_signed_conditional_metrics:
                        extras.pop("Metrics/hand_grasp_max/" + name, None)
            except Exception:  # noqa: BLE001 - diagnostics must never kill a run
                pass

        if self._disturbance_metric_enabled:
            # 진단 메트릭 하나 때문에 4096-env 런이 죽으면 안 된다.
            try:
                self._log_disturbance_metrics(env_ids, extras)
            except Exception:  # noqa: BLE001 - diagnostics must never kill a run
                pass

        if self._hand_object_metric_enabled:
            episode_duration = torch.clamp(
                self._env.episode_length_buf[env_ids].float() * self._env.step_dt,
                min=self._env.step_dt,
            )
            # 아래 루프가 각 이름을 처리한 직후 sums 를 0 으로 지우므로, 조건부
            # 정규화에 쓸 분자/분모는 루프 전에 미리 떠 둔다.
            hold_sums = {}
            for name, gate_name in self._hand_object_conditional_metrics.items():
                for key in (name, gate_name):
                    if key not in hold_sums:
                        hold_sums[key] = self._hand_object_metric_sums[key][env_ids].clone()
            for name in self._hand_object_metric_names:
                extras["Metrics/hand_object/" + name] = torch.mean(
                    self._hand_object_metric_sums[name][env_ids] / episode_duration
                )
                extras["Metrics/hand_object_final/" + name] = torch.mean(
                    self._hand_object_metric_last[name][env_ids]
                )
                extras["Metrics/hand_object_min/" + name] = torch.mean(
                    torch.nan_to_num(self._hand_object_metric_min[name][env_ids], posinf=0.0)
                )
                extras["Metrics/hand_object_max/" + name] = torch.mean(
                    torch.nan_to_num(self._hand_object_metric_max[name][env_ids], neginf=0.0)
                )
                self._hand_object_metric_sums[name][env_ids] = 0.0
                self._hand_object_metric_last[name][env_ids] = 0.0
                self._hand_object_metric_min[name][env_ids] = float("inf")
                self._hand_object_metric_max[name][env_ids] = float("-inf")

            # 조건부 보정.  일반 루프가 넣은 값은 에피소드 전체 시간으로 나뉘어
            # 희석돼 있으므로 "holding 이 켜져 있던 시간"으로 다시 나눈다.  min 은
            # 마스킹된 0 이, final 은 마지막 스텝이 holding 이었냐가 잡혀 둘 다
            # 의미가 없으므로 뺀다.  max 는 거리/속도가 음수가 될 수 없어 마스킹
            # 후에도 "물고 있던 스텝 중 최대"가 그대로 남는다.
            # 진단 메트릭 하나 때문에 학습이 죽으면 안 되므로 통째로 감싼다.
            try:
                for name, gate_name in self._hand_object_conditional_metrics.items():
                    hold_time = torch.clamp(hold_sums[gate_name], min=self._env.step_dt)
                    extras["Metrics/hand_object/" + name] = torch.mean(
                        hold_sums[name] / hold_time
                    )
                    extras.pop("Metrics/hand_object_min/" + name, None)
                    extras.pop("Metrics/hand_object_final/" + name, None)
            except Exception:  # noqa: BLE001 - diagnostics must never kill a run
                pass

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
            self._hand_setting_success_stable_steps[env_ids] = 0

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
            # 진단 메트릭 하나 때문에 4096-env 런이 죽으면 안 된다.  로그 쪽 조건부
            # 보정 블록과 같은 방침으로, 계산 자체도 실패하면 그 스텝을 건너뛰고
            # 경고를 한 번만 낸다 (매 스텝 출력은 로그를 못 쓰게 만든다).
            try:
                hand_grasp_metrics = self._compute_hand_grasp_metrics()
            except Exception as error:  # noqa: BLE001
                if not getattr(self, "_hand_grasp_metric_warned", False):
                    self._hand_grasp_metric_warned = True
                    print(
                        "[WARN] hand_grasp 진단 메트릭 계산 실패, 이후 스텝은 건너뜀: "
                        f"{error}"
                    )
                hand_grasp_metrics = {}
            for name, metric in hand_grasp_metrics.items():
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

        if self._hand_object_metric_enabled:
            # 진단 메트릭 하나 때문에 4096-env 런이 죽으면 안 된다.
            try:
                for name, metric in self._compute_hand_object_metrics().items():
                    self._hand_object_metric_sums[name] += metric * dt
                    self._hand_object_metric_last[name][:] = metric
                    self._hand_object_metric_min[name] = torch.minimum(
                        self._hand_object_metric_min[name], metric
                    )
                    self._hand_object_metric_max[name] = torch.maximum(
                        self._hand_object_metric_max[name], metric
                    )
            except Exception:  # noqa: BLE001 - diagnostics must never kill a run
                pass

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
