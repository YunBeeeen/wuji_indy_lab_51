# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import queue
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--print_action",
    action="store_true",
    default=False,
    help="step마다 action / 관절목표-실제 추종오차 / 손 최저높이를 출력. 팔이 튀는 원인이 정책인지 물리인지 가름.",
)
parser.add_argument(
    "--print_action_interval",
    type=int,
    default=1,
    help="--print_action 사용 시 몇 policy step마다 출력할지.",
)
parser.add_argument(
    "--print_action_detail",
    action="store_true",
    default=False,
    help="--print_action 사용 시 joint별 raw/applied/target/actual/error를 표로 출력.",
)
parser.add_argument(
    "--print_hand_joint4",
    action="store_true",
    default=False,
    help="hand_action의 finger1~5_joint4만 raw/target/actual/torque로 실시간 출력.",
)
parser.add_argument(
    "--print_diagnostics",
    action="store_true",
    default=False,
    help="play 화면을 띄우면서 joint별 torque/velocity/reward/cube metric까지 같이 출력.",
)
parser.add_argument(
    "--print_pose_diagnostics",
    action="store_true",
    default=False,
    help="box transport의 goal/자세/keypoint potential/gate/추종오차를 가볍게 출력.",
)
parser.add_argument(
    "--print_hand_mode",
    action="store_true",
    default=False,
    help="hand_grasp의 현재 OPEN/CLOSE 모드와 목표/실제 tip gap을 출력.",
)
parser.add_argument(
    "--debug_close_response",
    action="store_true",
    default=False,
    help=(
        "OPEN/CLOSE 명령 전환마다 응답시간과 20관절 raw/clipped action, target, actual, "
        "computed/applied effort를 출력. 두 방향은 필요한 토크가 다르므로 한쪽만 재면 "
        "'느린 명령'과 '도달 불가능한 명령'을 구분할 수 없어 양방향을 모두 측정함. "
        "기본 play 동작과 policy는 변경하지 않음."
    ),
)
parser.add_argument(
    "--legacy_obs_101d",
    action="store_true",
    default=False,
    help=(
        "2026-08-12_18-32-55 체크포인트를 현재 실물 하드웨어 설정 위에서 play. "
        "obs 105D->101D(directed axis) 와 action scale 균일 0.1 만 복원하며, "
        "effort limit / Kp / Kd 는 현재 실물 값을 그대로 유지함. "
        "play 전용이며 학습 설정 파일은 건드리지 않음."
    ),
)
parser.add_argument(
    "--close_response_interval",
    type=int,
    default=5,
    help="--debug_close_response joint table 출력 간격 [policy step]. 기본 5(30 Hz에서 약 0.167 s).",
)
parser.add_argument(
    "--close_response_timeout_s",
    type=float,
    default=3.0,
    help="OPEN/CLOSE 응답 진단을 한 방향당 유지할 최대 simulation time [s].",
)
parser.add_argument(
    "--close_response_gap_tolerance_mm",
    type=float,
    default=None,
    help=(
        "OPEN/CLOSE first-hit/hold 판정 gap error tolerance [mm]. 생략하면 task의 "
        "OpenCloseModeHeld tip_gap_error_limit(현재 0.5 mm)을 사용."
    ),
)
parser.add_argument(
    "--hand_mode",
    type=str.lower,
    choices=("open", "close"),
    default=None,
    help="hand_grasp play 명령을 OPEN 또는 CLOSE로 고정. reset 뒤에도 선택한 모드를 유지.",
)
parser.add_argument(
    "--keyboard_hand_mode",
    action="store_true",
    default=False,
    help="hand_grasp play 중 숫자 1=OPEN, 2=CLOSE로 command를 실시간 전환.",
)
parser.add_argument(
    "--alternate_hand_mode",
    action="store_true",
    default=False,
    help="hand_grasp play에서 reset 없이 OPEN/CLOSE 명령을 주기적으로 교대.",
)
parser.add_argument(
    "--manual_root",
    action="store_true",
    default=False,
    help="hand_move 전용. floating hand root의 PD 목표 위치/자세를 키보드로 조작."
    " 스크립트 SLERP와 자동 OPEN/CLOSE 스케줄을 끄고 1/2 키로 mode를 고른다."
    " 학습된 finger policy는 그대로 계속 추론한다.",
)
parser.add_argument(
    "--manual_translation_speed",
    type=float,
    default=0.05,
    help="--manual_root 이동 속도 [m/s].",
)
parser.add_argument(
    "--manual_rotation_speed_deg",
    type=float,
    default=30.0,
    help="--manual_root 회전 속도 [deg/s].",
)
parser.add_argument(
    "--manual_max_translation",
    type=float,
    default=0.30,
    help="--manual_root 목표 위치를 reset 시작점에서 최대 몇 m까지 허용할지.",
)
parser.add_argument(
    "--hand_mode_interval_s",
    type=float,
    default=3.0,
    help="--alternate_hand_mode 사용 시 한 모드를 유지하는 시간(초).",
)
parser.add_argument(
    "--print_contact",
    action="store_true",
    default=False,
    help="--print_diagnostics 사용 시 thumb/index/middle/palm과 cube contact force도 출력.",
)
parser.add_argument(
    "--load_experiment",
    type=str,
    default=None,
    help=(
        "--load_run 을 어느 experiment 폴더에서 찾을지. 기본은 현재 task 의"
        " experiment_name 이라 아직 학습한 적 없는 task 는 폴더가 없어 실패한다."
        " 예: hand_object 를 hand_move 체크포인트로 play 할 때"
        " '--load_experiment hand_move --load_run 2026-08-06_00-12-28'."
        " run 이름은 접두사만 줘도 되고(정규식 prefix 매칭), 그 안에서 가장 최신"
        " 체크포인트가 선택된다."
    ),
)
parser.add_argument(
    "--latest_run",
    action="store_true",
    default=False,
    help="logs/rsl_rl/<experiment_name> 아래 가장 최근 run을 load_run으로 사용.",
)
parser.add_argument(
    "--render_interval",
    type=int,
    default=2,
    help="렌더 프레임 사이의 physics step 수. 낮을수록 부드러움. task의 decimation을 주면 학습 때 값 그대로.",
)
parser.add_argument(
    "--joint_record_dir",
    type=str,
    default="logs/joint_records",
    help=(
        "--manual_root 세션에서 M 키로 저장하는 관절각 CSV 의 출력 폴더."
        " 매 policy step(=30Hz) 마다 측정 관절각 20개를 한 줄씩 기록한다."
    ),
)
parser.add_argument(
    "--show_stick_frames",
    action="store_true",
    default=False,
    help=(
        "env 0의 stick1/stick2 local 좌표계(강체 root 프레임)를 RGB=XYZ 삼각축으로 표시."
        " 스틱은 180mm 큐보이드라 원점은 기하 중심이고 +Y(초록)가 tail->tip 샤프트 축."
    ),
)
parser.add_argument(
    "--stick_frame_scale",
    type=float,
    default=0.03,
    help="--show_stick_frames 삼각축 길이(m).",
)
parser.add_argument(
    "--show_palm_vectors",
    action="store_true",
    default=False,
    help="env 0의 palm_link에서 손바닥 평면 법선, 파지 개구부 축, cube 방향을 화살표로 표시.",
)
parser.add_argument(
    "--show_hand_contact_forces",
    action="store_true",
    default=False,
    help=(
        "Robot의 palm/finger link에 작용하는 net contact force를 env 0 viewport에 "
        "화살표로 표시하고 링크별 크기[N]를 주기적으로 출력."
    ),
)
parser.add_argument(
    "--plot_hand_contact_forces",
    action="store_true",
    default=False,
    help=(
        "env 0의 palm/finger link별 net contact-force 크기[N]를 별도 "
        "PySide6+pyqtgraph 창에 실시간 표시. GUI는 play.py와 다른 subprocess에서 실행됨."
    ),
)
parser.add_argument(
    "--hand_contact_plot_history",
    type=float,
    default=10.0,
    help="외부 contact-force 그래프에 보존할 최근 simulation time [s].",
)
parser.add_argument(
    "--hand_contact_plot_hz",
    type=float,
    default=20.0,
    help="외부 contact-force 그래프로 보낼 최대 telemetry rate [Hz].",
)
parser.add_argument(
    "--hand_contact_force_scale",
    type=float,
    default=0.20,
    help="--show_hand_contact_forces 화살표 길이 환산값 [m/N].",
)
parser.add_argument(
    "--hand_contact_force_max_length",
    type=float,
    default=0.12,
    help="contact-force 화살표 최대 길이 [m].",
)
parser.add_argument(
    "--hand_contact_force_threshold",
    type=float,
    default=0.01,
    help="이 값보다 작은 contact force는 화살표/출력에서 숨김 [N].",
)
parser.add_argument(
    "--hand_contact_print_interval",
    type=int,
    default=10,
    help="contact force 숫자를 몇 policy step마다 출력할지.",
)
parser.add_argument(
    "--palm_vector_length",
    type=float,
    default=0.12,
    help="--show_palm_vectors 화살표 길이(m).",
)
parser.add_argument(
    "--resample_goal_on_success",
    action="store_true",
    default=False,
    help="play에서 transport_success가 발생하면 환경 상태는 유지하고 cube_goal만 즉시 재샘플.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.print_diagnostics:
    args_cli.print_action = True
    args_cli.print_action_detail = True
if args_cli.print_hand_joint4:
    args_cli.print_action = True
    args_cli.print_action_detail = True
if args_cli.print_pose_diagnostics:
    args_cli.print_action = True
if args_cli.print_contact:
    args_cli.print_action = True
    args_cli.print_diagnostics = True
if args_cli.debug_close_response:
    args_cli.print_hand_mode = True
    if args_cli.close_response_interval < 1:
        parser.error("--close_response_interval must be >= 1.")
    if args_cli.close_response_timeout_s <= 0.0:
        parser.error("--close_response_timeout_s must be > 0.")
    if (
        args_cli.close_response_gap_tolerance_mm is not None
        and args_cli.close_response_gap_tolerance_mm <= 0.0
    ):
        parser.error("--close_response_gap_tolerance_mm must be > 0 when specified.")
hand_mode_selector_count = sum(
    (
        args_cli.hand_mode is not None,
        args_cli.keyboard_hand_mode,
        args_cli.alternate_hand_mode,
    )
)
if hand_mode_selector_count > 1:
    parser.error(
        "--hand_mode, --keyboard_hand_mode, and --alternate_hand_mode "
        "cannot be used together."
    )
if args_cli.keyboard_hand_mode and args_cli.headless:
    parser.error("--keyboard_hand_mode requires the Isaac Sim GUI; remove --headless.")
if args_cli.show_stick_frames and args_cli.headless:
    parser.error("--show_stick_frames requires the Isaac Sim GUI; remove --headless.")
if args_cli.show_hand_contact_forces and args_cli.headless:
    parser.error("--show_hand_contact_forces requires the Isaac Sim GUI; remove --headless.")
if args_cli.plot_hand_contact_forces and args_cli.headless:
    parser.error("--plot_hand_contact_forces requires a desktop session; remove --headless.")
if args_cli.manual_root:
    if args_cli.headless:
        parser.error("--manual_root requires the Isaac Sim GUI; remove --headless.")
    if hand_mode_selector_count > 0:
        parser.error(
            "--manual_root already provides 1=OPEN / 2=CLOSE; do not combine it with"
            " --hand_mode / --keyboard_hand_mode / --alternate_hand_mode."
        )
if hand_mode_selector_count > 0:
    args_cli.print_hand_mode = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import importlib.metadata as metadata
import json
import math
import os
import time
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from packaging import version

installed_version = metadata.version("rsl-rl-lib")
import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.sensors import ContactSensorCfg
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

# Neuromeka tasks
import isaac_neuromeka.mdp as mdp
import isaac_neuromeka.tasks  # noqa: F401


CONTACT_BODIES = {
    "thumb_tip": "finger1_tip_link",
    "index_tip": "finger2_tip_link",
    "index_mid": "finger2_link3",
    "middle_tip": "finger3_tip_link",
    "middle_mid": "finger3_link3",
    "palm": "palm_link",
}

HAND_NET_CONTACT_SENSOR = "diag_hand_net_contact"


def _latest_run_name(log_root_path: str) -> str:
    root = Path(log_root_path)
    candidates = [path for path in root.glob("20*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No timestamped run directory found under: {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime).name


def _success_env_ids(env) -> torch.Tensor:
    """Return env indices whose transport-success reward fired on the latest step."""
    manager = env.unwrapped.reward_manager
    try:
        term_idx = manager._term_names.index("transport_success")
    except ValueError as exc:
        raise ValueError(
            "--resample_goal_on_success requires a transport_success reward term."
        ) from exc
    return (manager._step_reward[:, term_idx] > 0.0).nonzero(as_tuple=False).flatten()


# 2026-08-12_18-32-55 체크포인트를 "현재 실물 하드웨어 설정 위에서" 돌리기 위한
# 최소 복원.  되돌리는 것은 policy contract 뿐이다:
#   - observation func (105D quaternion -> 101D directed axis).  actor 입력 폭이
#     달라 이걸 안 되돌리면 checkpoint 로드 자체가 실패한다.
#   - action scale (per-joint -> 균일 0.1).  scale 은 정책이 학습된 출력 단위라
#     정책을 고정한 채 바꾸면 그 정책의 액션이 다른 크기로 해석된다.
# effort limit, Kp, Kd 는 의도적으로 건드리지 않는다.  이 실험의 질문이 "실물
# 토크 한계에서 그 정책이 CLOSE->OPEN 을 하는가" 이므로, 하드웨어 값을 옛날
# 값으로 되돌리면 질문 자체가 사라진다.
def _apply_legacy_101d_contract(env_cfg) -> None:
    """Restore only the 2026-08-12 policy contract; keep real-hand actuators."""
    from isaac_neuromeka.tasks.manipulation.hand_grasp import hand_real_mdp

    policy_cfg = env_cfg.observations.policy
    for term_name in ("stick1_pose_history", "stick2_pose_history"):
        term = getattr(policy_cfg, term_name, None)
        if term is None:
            raise ValueError(
                f"--legacy_obs_101d requires the hand_real observation term {term_name}."
            )
        term.func = hand_real_mdp.object_position_and_directed_axis_in_palm
        # The 101D function has no reference branch; leaving the parameter in
        # place would reach the callable as an unexpected keyword argument.
        term.params.pop("reference_quaternion_p", None)

    env_cfg.actions.hand_action.scale = 0.1

    finger_actuator = env_cfg.scene.robot.actuators["fingers"]
    effort = finger_actuator.effort_limit_sim
    print(
        "[INFO] --legacy_obs_101d: obs 101D(directed axis) + action scale 0.1 만 복원. "
        "effort limit / Kp / Kd 는 현재 실물 값을 유지함 "
        f"(예: finger1_joint4 effort={effort['finger1_joint4']}, "
        f"finger1_joint2 Kp={finger_actuator.stiffness['finger1_joint2']}).",
        flush=True,
    )


def _add_contact_sensors(env_cfg) -> list[str]:
    sensor_names = []
    for label, body_name in CONTACT_BODIES.items():
        name = f"diag_{label}_cube_contact"
        setattr(
            env_cfg.scene,
            name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
                update_period=0.0,
                history_length=1,
                debug_vis=False,
                track_pose=True,
            ),
        )
        sensor_names.append(name)
    return sensor_names


def _add_hand_net_contact_sensor(env_cfg) -> str:
    """Attach a play-only unfiltered contact sensor to every Robot body.

    The single multi-body sensor reports ``net_forces_w`` separately for each
    resolved rigid body.  Drawing filters that list to palm/finger links, so
    the same diagnostic remains safe if a future task's Robot also has an arm.
    No filtered force matrix is requested: this is the total normal contact
    force from sticks, objects, furniture, the ground, and self-contact.
    """
    setattr(
        env_cfg.scene,
        HAND_NET_CONTACT_SENSOR,
        ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            track_pose=True,
        ),
    )
    return HAND_NET_CONTACT_SENSOR


def _contact_force(env, sensor_name: str) -> torch.Tensor:
    sensor = env.unwrapped.scene.sensors[sensor_name]
    force_w = getattr(sensor.data, "force_matrix_w", None)
    if force_w is None:
        force_w = getattr(sensor.data, "net_forces_w", None)
    if force_w is None:
        return torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    force = torch.linalg.norm(force_w, dim=-1)
    while force.dim() > 1:
        force = force.sum(dim=-1)
    return force


def _ratio(value: torch.Tensor, limit: torch.Tensor) -> float:
    limit_value = float(torch.abs(limit).item())
    if limit_value <= 1.0e-8:
        return 0.0
    return abs(float(value.item())) / limit_value


def _reward_snapshot(env) -> dict[str, tuple[float, float]]:
    manager = env.unwrapped.reward_manager
    names = list(getattr(manager, "_term_names", []))
    cfgs = list(getattr(manager, "_term_cfgs", []))
    step_reward = getattr(manager, "_step_reward", None)
    if step_reward is None:
        return {}
    out = {}
    for idx, (name, cfg) in enumerate(zip(names, cfgs)):
        weighted = float(step_reward[0, idx].item())
        weight = float(cfg.weight)
        raw = weighted / weight if abs(weight) > 1.0e-8 else 0.0
        out[name] = (weighted, raw)
    return out


def _cube_snapshot(env) -> dict[str, float]:
    unwrapped = env.unwrapped
    out = {}
    if "cube" not in unwrapped.scene.rigid_objects:
        return out
    cube = unwrapped.scene["cube"]
    out["cube_z"] = float(cube.data.root_pos_w[0, 2].item())
    out["cube_speed"] = float(torch.linalg.norm(cube.data.root_lin_vel_w[0]).item())
    out["cube_ang_speed"] = float(torch.linalg.norm(cube.data.root_ang_vel_w[0]).item())

    rewards_cfg = getattr(unwrapped.cfg, "rewards", None)
    lift_cfg = getattr(rewards_cfg, "cube_lift", None)
    if lift_cfg is not None:
        params = lift_cfg.params
        clearance = mdp.box_ground_clearance(
            unwrapped,
            params["object_cfg"],
            params.get("object_half_extent", (0.03, 0.03, 0.03)),
            params.get("surface_z", 0.0),
        )
        out["clearance"] = float(clearance[0].item())

    reward_manager = unwrapped.reward_manager
    if hasattr(reward_manager, "_compute_cube_distance_metrics"):
        try:
            metrics = reward_manager._compute_cube_distance_metrics()
        except Exception:
            metrics = {}
        for key in (
            "palm_facing",
            "cage_inside_frac",
            "cage_sdf_mean",
            "cage_sdf_min",
            "cage_span",
            "thumb_index_opposition",
            "thumb_middle_opposition",
            "arm_manipulability",
            "action_track_err",
            "action_delta",
            "box_ori_error",
        ):
            if key in metrics:
                out[key] = float(metrics[key][0].item())
    return out


def _print_reward_cube_diagnostics(env) -> None:
    rewards = _reward_snapshot(env)
    cube = _cube_snapshot(env)
    reach_raw = rewards.get("finger_cage_reach", (0.0, 0.0))[1]
    hold_raw = rewards.get("finger_cage_hold", (0.0, 0.0))[1]
    lift_raw = rewards.get("cube_lift", (0.0, 0.0))[1]
    facing_raw = rewards.get("palm_facing", (0.0, 0.0))[1]
    print(
        "      reward/raw "
        f"reach={reach_raw:+.4f} hold={hold_raw:+.4f} lift={lift_raw:+.4f} facing={facing_raw:+.4f} | "
        f"clearance={cube.get('clearance', float('nan')):+.4f}m "
        f"cage={cube.get('cage_inside_frac', float('nan')):.3f} "
        f"span={cube.get('cage_span', float('nan')):.3f} "
        f"opp_i/m={cube.get('thumb_index_opposition', float('nan')):+.3f}/"
        f"{cube.get('thumb_middle_opposition', float('nan')):+.3f} "
        f"cube_speed={cube.get('cube_speed', float('nan')):.3f}",
        flush=True,
    )
    _print_hand_pose(env, cube)


def _print_hand_mode_diagnostics(env, episode_reset: bool = False) -> None:
    """Print the active hand-grasp mode and its current tip-gap error."""
    try:
        snapshot = _hand_mode_gap_snapshot(env)
        mode_name = snapshot["mode_name"]
        target_gap = snapshot["target_gap"]
        actual_gap = snapshot["actual_gap"]
        reset_label = " NEW_EPISODE" if episode_reset else ""
        print(
            f"      hand_mode={mode_name}{reset_label} "
            f"target={target_gap * 1000.0:.1f}mm "
            f"gap={actual_gap * 1000.0:.1f}mm "
            f"error={(actual_gap - target_gap) * 1000.0:+.1f}mm",
            flush=True,
        )
    except (AttributeError, KeyError, ValueError) as exc:
        print(f"      hand mode diagnostic unavailable: {exc}", flush=True)


def _hand_mode_gap_snapshot(env) -> dict[str, float | int | str]:
    """Read env-0 OPEN/CLOSE command and the exact reward-side tip gap."""
    from isaac_neuromeka.tasks.manipulation.hand_grasp import mdp as hand_grasp_mdp

    unwrapped = env.unwrapped
    open_cfg = unwrapped.reward_manager.get_term_cfg("open_tip_gap")
    close_cfg = unwrapped.reward_manager.get_term_cfg("close_tip_gap")
    command_name = open_cfg.params["command_name"]
    command = unwrapped.command_manager.get_command(command_name)
    mode_index = int(torch.argmax(command[0]).item())
    mode_cfg = open_cfg if mode_index == 0 else close_cfg
    params = mode_cfg.params
    gap, lateral_error = hand_grasp_mdp._tip_surface_gap_and_lateral_error(
        unwrapped,
        params["palm_cfg"],
        params["stick1_cfg"],
        params["stick2_cfg"],
        params["stick1_tip_offset_o"],
        params["stick2_tip_offset_o"],
        params["stick_thickness"],
        params["reference_separation_direction_stick2"],
        clamp_gap=bool(params.get("clamp_gap", True)),
    )
    target_gap = float(params["target_gap"])
    actual_gap = float(gap[0].item())
    return {
        "mode_index": mode_index,
        "mode_name": "OPEN" if mode_index == 0 else "CLOSE",
        "target_gap": target_gap,
        "actual_gap": actual_gap,
        "error": actual_gap - target_gap,
        "lateral_error": float(lateral_error[0].item()),
    }


def _print_mode_response_joint_table(
    *,
    actions: torch.Tensor,
    applied: torch.Tensor,
    target: torch.Tensor,
    actual: torch.Tensor,
    joint_vel: torch.Tensor,
    robot,
    action_term,
    joint_names: list[str],
) -> None:
    """Print the signals that separate policy hesitation from drive saturation.

    ``use`` is ``|applied| / effort_limit``.  The binary ``sat`` column only
    trips on exact-equality with the cap, so a joint pinned near its limit --
    the state that actually blocks a command -- shows up in ``use`` first.
    """
    print(
        "      joint                  raw clipped    target    actual       err"
        "       vel   computed   applied     limit   use   sat",
        flush=True,
    )
    joint_ids = action_term._joint_ids
    for index, name in enumerate(joint_names):
        joint_id = index if isinstance(joint_ids, slice) else joint_ids[index]
        error = target[0, index] - actual[0, index]
        computed = robot.data.computed_torque[0, joint_id]
        applied_torque = robot.data.applied_torque[0, joint_id]
        effort_limit = robot.data.joint_effort_limits[0, joint_id]
        tolerance = torch.maximum(
            effort_limit.abs() * 1.0e-4,
            torch.as_tensor(1.0e-6, device=effort_limit.device),
        )
        saturated = bool(
            (
                (computed.abs() > effort_limit.abs() + tolerance)
                | ((computed - applied_torque).abs() > tolerance)
            ).item()
        )
        effort_use = float(
            applied_torque.abs() / effort_limit.abs().clamp(min=1.0e-9)
        )
        print(
            f"      {name:<20}"
            f"{actions[0, index]:>8.3f}{applied[0, index]:>8.3f}"
            f"{target[0, index]:>10.3f}{actual[0, index]:>10.3f}"
            f"{error:>10.3f}{joint_vel[0, index]:>10.3f}"
            f"{computed:>11.3f}{applied_torque:>10.3f}{effort_limit:>10.3f}"
            f"{effort_use * 100.0:>5.0f}%"
            f"{' YES' if saturated else ' no':>6}",
            flush=True,
        )


# 2026-07-24: hand_stick_orientation 목표 자세(q_O_H) 측정용. 실제 파지 순간의 손-스틱 상대자세를
# env 0에서 찍는다. "엄지·검지 접촉 + opposition>0 + 들림" 조건이 맞을 때의 q_O_H를 목표로 쓸 것.
def _print_hand_pose(env, cube) -> None:
    try:
        from isaac_neuromeka.tasks.manipulation.functional_grasp import mdp as fg_mdp
        from isaaclab.managers import SceneEntityCfg

        u = env.unwrapped
        palm_cfg = SceneEntityCfg("robot", body_names=["palm_link"])
        obj_cfg = SceneEntityCfg("cube")
        palm_cfg.resolve(u.scene)
        obj_cfg.resolve(u.scene)
        q = fg_mdp.hand_orientation_in_object(u, palm_cfg, obj_cfg)[0]
        opp_i = cube.get("thumb_index_opposition", float("nan"))
        opp_m = cube.get("thumb_middle_opposition", float("nan"))
        clr = cube.get("clearance", float("nan"))
        good = (opp_i > 0.0) and (opp_m > 0.0) and (clr > 0.03)
        print(
            f"      q_O_H(w,x,y,z)=[{q[0]:.4f},{q[1]:.4f},{q[2]:.4f},{q[3]:.4f}]  "
            f"{'★목표후보(파지+들림+opp>0)' if good else '(조건 미달)'}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"      q_O_H unavailable: {exc}", flush=True)


def _print_pose_diagnostics(env) -> None:
    """Print the exact pose potential inputs without advancing or mutating the environment."""
    unwrapped = env.unwrapped
    manager = unwrapped.reward_manager
    try:
        transport_cfg = manager.get_term_cfg("cube_transport")
    except ValueError:
        print("      pose diagnostic unavailable: cube_transport reward is not configured", flush=True)
        return

    params = transport_cfg.params
    command_name = params["command_name"]
    object_cfg = params["object_cfg"]
    cube = unwrapped.scene[object_cfg.name]
    command = unwrapped.command_manager.get_command(command_name)
    goal_w = unwrapped.scene.env_origins + command[:, :3]
    goal_quat_w = command[:, 3:7]
    pos_error = torch.linalg.norm(cube.data.root_pos_w - goal_w, dim=1)
    ori_error = mdp.square_prism_ori_error(cube.data.root_quat_w, goal_quat_w)
    keypoint_error = mdp.square_prism_keypoint_goal_distance(
        unwrapped,
        command_name,
        object_cfg,
        params.get("object_half_extent", (0.03, 0.03, 0.03)),
        params.get("symmetry", "square_prism_y"),
    )
    potential_eps = float(params.get("potential_eps", 0.05))
    phi = potential_eps / (potential_eps + pos_error)
    best_phi_value = getattr(transport_cfg.func, "_best_phi", None)
    best_phi = float(best_phi_value[0].item()) if best_phi_value is not None else float("nan")

    gate = mdp.object_in_finger_cage(
        unwrapped,
        params["asset_cfg"],
        object_cfg,
        params.get("object_half_extent", (0.03, 0.03, 0.03)),
        params.get("num_points", 3),
        params.get("sphere_radius", 0.005),
        params.get("depth_max", 0.005),
        params.get("point_fractions"),
    )
    rewards = _reward_snapshot(env)
    transport_weighted, transport_raw = rewards.get("cube_transport", (0.0, 0.0))
    orientation_weighted, orientation_raw = rewards.get("box_orientation", (0.0, 0.0))
    cube_metrics = _cube_snapshot(env)

    orientation_active = False
    try:
        orientation_cfg = manager.get_term_cfg("box_orientation")
        active = getattr(orientation_cfg.func, "active", None)
        if active is not None:
            orientation_active = bool(active[0].item())
    except ValueError:
        pass

    roll, pitch, yaw = math_utils.euler_xyz_from_quat(cube.data.root_quat_w)
    rpy_deg = torch.rad2deg(torch.stack((roll, pitch, yaw), dim=-1))[0]

    arm_track_error = float("nan")
    finger_track_error = float("nan")
    try:
        robot = unwrapped.scene["robot"]
        action_term = unwrapped.action_manager.get_term("arm_action")
        target = action_term.processed_actions
        actual = robot.data.joint_pos[:, action_term._joint_ids]
        error = torch.abs(target - actual)[0]
        arm_mask = torch.tensor(
            [name.startswith("joint") for name in action_term._joint_names], device=error.device
        )
        if torch.any(arm_mask):
            arm_track_error = float(error[arm_mask].max().item())
        if torch.any(~arm_mask):
            finger_track_error = float(error[~arm_mask].max().item())
    except (AttributeError, KeyError, ValueError):
        pass

    goal_radius = float("nan")
    gate_threshold = float("nan")
    ori_limit = float("nan")
    hold_count = 0
    hold_steps = 0
    try:
        # 2026-07-24: chopstick success가 termination→reward(GoalReachedBonus)로 분리됨.
        #   success termination이 없으면 KeyError로 조용히 넘어감(아래 except). 진단이 필요하면
        #   reward_manager.get_term_cfg("transport_success")의 GoalReachedBonus._count를 읽도록 교체.
        success_cfg = unwrapped.termination_manager.get_term_cfg("success")
        success_params = success_cfg.params
        goal_radius = float(success_params.get("goal_radius", float("nan")))
        gate_threshold = float(success_params.get("gate_threshold", float("nan")))
        ori_limit_value = success_params.get("ori_limit")
        if ori_limit_value is not None:
            ori_limit = float(ori_limit_value)
        hold_steps = int(success_params.get("hold_steps", 0))
        count = getattr(success_cfg.func, "_count", None)
        if count is not None:
            hold_count = int(count[0].item())
    except (AttributeError, KeyError, ValueError):
        pass

    pos_ok = pos_error[0].item() < goal_radius
    gate_ok = gate[0].item() > gate_threshold
    ori_ok = ori_error[0].item() < ori_limit
    print(
        "      pose "
        f"goal={pos_error[0].item():.4f}m ori={torch.rad2deg(ori_error[0]).item():.1f}deg "
        f"rpy={rpy_deg[0].item():+.1f}/{rpy_deg[1].item():+.1f}/{rpy_deg[2].item():+.1f}deg "
        f"kp={keypoint_error[0].item():.4f}m",
        flush=True,
    )
    print(
        "      pose/reward "
        f"phi={phi[0].item():.4f} best={best_phi:.4f} lost={max(best_phi - phi[0].item(), 0.0):.4f} "
        f"gate={gate[0].item():.3f} ori_active={int(orientation_active)} "
        f"transport(raw/weighted)={transport_raw:+.6f}/{transport_weighted:+.3f} "
        f"orientation(raw/weighted)={orientation_raw:+.6f}/{orientation_weighted:+.3f}",
        flush=True,
    )
    print(
        "      pose/state "
        f"ok(pos/gate/ori)={int(pos_ok)}/{int(gate_ok)}/{int(ori_ok)} "
        f"stable={hold_count}/{hold_steps} "
        f"v/w={cube_metrics.get('cube_speed', float('nan')):.3f}m/s/"
        f"{cube_metrics.get('cube_ang_speed', float('nan')):.3f}rad/s "
        f"clearance={cube_metrics.get('clearance', float('nan')):+.4f}m "
        f"track(arm/finger)={arm_track_error:.3f}/{finger_track_error:.3f}rad",
        flush=True,
    )


def _print_contacts(env, sensor_names: list[str]) -> None:
    if not sensor_names:
        return
    values = []
    for sensor_name in sensor_names:
        label = sensor_name.removeprefix("diag_").removesuffix("_cube_contact")
        values.append(f"{label}={_contact_force(env, sensor_name)[0].item():.3f}N")
    print("      contact " + " ".join(values), flush=True)


def _is_hand_contact_body(body_name: str) -> bool:
    """Return whether a Robot contact-sensor body belongs to the hand."""
    return body_name == "palm_link" or body_name.startswith("finger")


def _print_hand_contact_forces(env, sensor_name: str, threshold_n: float) -> None:
    """Print exact env-0 per-link net contact-force magnitudes in newtons."""
    sensor = env.unwrapped.scene.sensors[sensor_name]
    magnitudes = torch.linalg.norm(sensor.data.net_forces_w[0], dim=-1)
    active = [
        (body_name, float(magnitudes[body_id].item()))
        for body_id, body_name in enumerate(sensor.body_names)
        if _is_hand_contact_body(body_name)
        and float(magnitudes[body_id].item()) >= threshold_n
    ]
    active.sort(key=lambda item: item[1], reverse=True)
    if not active:
        print("      hand_contact no force above threshold", flush=True)
        return
    force_text = " ".join(f"{name}={force:.3f}N" for name, force in active)
    summed_magnitudes = sum(force for _, force in active)
    print(
        f"      hand_contact {force_text} | sum_link_norm={summed_magnitudes:.3f}N",
        flush=True,
    )


def _draw_hand_contact_forces(
    env,
    draw_interface,
    sensor_name: str,
    force_scale_m_per_n: float,
    max_length_m: float,
    threshold_n: float,
) -> None:
    """Draw env-0 hand-link net contact forces at each link origin.

    Arrow direction is the world-frame force acting on the sensed link.  Arrow
    length is ``min(|F| * scale, max_length)``.  Color encodes the absolute
    force band: green < 0.1 N, yellow < 0.5 N, red >= 0.5 N.
    """
    sensor = env.unwrapped.scene.sensors[sensor_name]
    positions_w = sensor.data.pos_w[0]
    forces_w = sensor.data.net_forces_w[0]

    starts = []
    ends = []
    colors = []
    thicknesses = []
    for body_id, body_name in enumerate(sensor.body_names):
        if not _is_hand_contact_body(body_name):
            continue
        force = forces_w[body_id]
        magnitude = float(torch.linalg.norm(force).item())
        if not math.isfinite(magnitude) or magnitude < threshold_n:
            continue

        direction = force / max(magnitude, 1.0e-12)
        length = min(magnitude * force_scale_m_per_n, max_length_m)
        if length <= 0.0:
            continue
        origin = positions_w[body_id]
        tip = origin + direction * length
        if magnitude < 0.1:
            color = (0.1, 1.0, 0.1, 1.0)
        elif magnitude < 0.5:
            color = (1.0, 0.85, 0.1, 1.0)
        else:
            color = (1.0, 0.1, 0.1, 1.0)

        reference = torch.tensor((0.0, 0.0, 1.0), dtype=force.dtype, device=force.device)
        if torch.abs(torch.dot(direction, reference)) > 0.9:
            reference = torch.tensor((0.0, 1.0, 0.0), dtype=force.dtype, device=force.device)
        side = torch.linalg.cross(direction, reference)
        side /= torch.clamp(torch.linalg.norm(side), min=1.0e-6)
        head_length = min(0.02, max(0.002, length * 0.25))
        head_width = min(0.01, max(0.001, length * 0.12))
        head_base = tip - direction * head_length
        head_a = head_base + side * head_width
        head_b = head_base - side * head_width

        starts.extend((origin.tolist(), tip.tolist(), tip.tolist()))
        ends.extend((tip.tolist(), head_a.tolist(), head_b.tolist()))
        colors.extend((color, color, color))
        thicknesses.extend((5.0, 5.0, 5.0))

    if starts:
        draw_interface.draw_lines(starts, ends, colors, thicknesses)


def _draw_palm_vectors(
    env,
    draw_interface,
    palm_body_id: int,
    grasp_opening_b_value: tuple[float, float, float],
    length: float,
) -> None:
    """Draw palm-local directions in world coordinates for env 0."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    cube = unwrapped.scene["cube"]

    palm_pos_w = robot.data.body_pos_w[0, palm_body_id]
    palm_quat_w = robot.data.body_quat_w[0, palm_body_id]
    dtype = palm_pos_w.dtype
    device = palm_pos_w.device

    # 손가락 뿌리 5점의 평면에서 측정한 palm_link 로컬 법선.
    # 부호는 손가락이 오므라드는 쪽을 향하도록 선택됨. local +x는 이 값의 근사일 뿐임.
    palm_normal_b = torch.tensor((0.965, -0.008, 0.262), dtype=dtype, device=device)
    palm_normal_b /= torch.linalg.norm(palm_normal_b)
    grasp_opening_b = torch.tensor(grasp_opening_b_value, dtype=dtype, device=device)
    grasp_opening_b /= torch.linalg.norm(grasp_opening_b)

    palm_normal_w = math_utils.quat_apply(palm_quat_w, palm_normal_b)
    grasp_opening_w = math_utils.quat_apply(palm_quat_w, grasp_opening_b)
    to_cube_w = cube.data.root_pos_w[0] - palm_pos_w
    to_cube_w /= torch.clamp(torch.linalg.norm(to_cube_w), min=1.0e-6)

    directions = (palm_normal_w, grasp_opening_w, to_cube_w)
    colors = (
        (1.0, 0.1, 0.1, 1.0),  # red: measured physical palm-plane normal
        (0.1, 0.5, 1.0, 1.0),  # blue: grasp-opening axis used by palm_facing
        (0.1, 1.0, 0.1, 1.0),  # green: palm-to-cube direction
    )

    starts = []
    ends = []
    line_colors = []
    thicknesses = []
    head_length = min(0.025, length * 0.25)
    head_width = min(0.012, length * 0.12)
    for direction, color in zip(directions, colors):
        tip = palm_pos_w + direction * length
        reference = torch.tensor((0.0, 0.0, 1.0), dtype=dtype, device=device)
        if torch.abs(torch.dot(direction, reference)) > 0.9:
            reference = torch.tensor((0.0, 1.0, 0.0), dtype=dtype, device=device)
        side = torch.linalg.cross(direction, reference)
        side /= torch.clamp(torch.linalg.norm(side), min=1.0e-6)
        head_base = tip - direction * head_length
        head_a = head_base + side * head_width
        head_b = head_base - side * head_width

        starts.extend((palm_pos_w.tolist(), tip.tolist(), tip.tolist()))
        ends.extend((tip.tolist(), head_a.tolist(), head_b.tolist()))
        line_colors.extend((color, color, color))
        thicknesses.extend((5.0, 5.0, 5.0))

    draw_interface.draw_lines(starts, ends, line_colors, thicknesses)


# Local frames drawn by --show_stick_frames.  Both proxies are the same 180 mm
# cuboid (``_stick_cfg`` in hand_move_env_cfg.py), so the rigid-body root frame
# *is* the stick local frame: origin at the geometric centre, local +Y from tail
# to distal tip (``STICK_TIP_OFFSET_O``), X/Z spanning the 7 mm square section.
STICK_FRAME_OBJECTS = ("stick1", "stick2")


def _create_stick_frame_visualizer(env, scale: float):
    """Build the stick local-frame triad markers, or None if the scene has no sticks.

    Returns ``(visualizer, names)``.  ``names`` is the subset of
    ``STICK_FRAME_OBJECTS`` actually present, in marker-instance order.
    """
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG

    rigid_objects = env.unwrapped.scene.rigid_objects
    names = [name for name in STICK_FRAME_OBJECTS if name in rigid_objects]
    if not names:
        return None, []

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.prim_path = "/Visuals/play/stick_frames"
    # Keep only the axis triad; the default cfg also carries a 1 m
    # "connecting_line" cylinder that is not used here.
    marker_cfg.markers = {"frame": marker_cfg.markers["frame"]}
    marker_cfg.markers["frame"].scale = (scale, scale, scale)
    return VisualizationMarkers(marker_cfg), names


def _draw_stick_frames(env, visualizer, names: list[str]) -> None:
    """Draw env 0's stick local frames at their true poses.

    No offset and no re-orientation: what is drawn is exactly the rigid-body
    root frame, so the green axis is the physical shaft direction.
    """
    scene = env.unwrapped.scene
    translations = torch.stack([scene[name].data.root_link_pos_w[0] for name in names])
    orientations = torch.stack([scene[name].data.root_link_quat_w[0] for name in names])
    visualizer.visualize(translations=translations, orientations=orientations)


class HandJointRecorder:
    """Log the *measured* hand joint angles, one row per policy step.

    Sampling period is not configurable and does not need to be: one turn of
    the play loop is exactly one ``env.step``, i.e. ``sim.dt * decimation``
    = 1/120 * 4 = 1/30 s for every hand task here.  So recording once per turn
    already yields the policy rate, and it is the *simulated* interval that is
    exact - wall-clock pacing in the GUI is irrelevant and deliberately not
    used.  30 Hz is also what the deploy path consumes: ``real_wuji_scheduler``
    runs 90 Hz commands with ``policy_divider(90) == 3``, so one recorded row
    is exactly three command ticks.

    Both the measured angle (``q_*``) and the PD target (``qt_*``) are logged.
    They are not interchangeable: this task's action is a residual,
    ``q_target = q_current + scale * action``, so the target sits ahead of the
    measurement and that gap times Kp *is* the grip force.  Replaying measured
    angles as targets would command zero PD error, i.e. no grip.  The same
    distinction the pregrasp constants already make
    (``PREGRASP_JOINT_POSITIONS`` vs ``PREGRASP_JOINT_TARGETS``).

    Columns are named after the physical joints rather than positional, because
    Isaac's raw ``joint_pos`` columns follow USD DOF order while every consumer
    downstream (``Deploy.contract.policy_contract.POLICY_JOINT_NAMES``) is
    finger-major.  They happen to agree today; the header is what keeps a
    future divergence from being silent.
    """

    def __init__(self, env, output_dir, task: str, checkpoint: str | None = None):
        robot = env.unwrapped.scene["robot"]
        # Resolve by name with preserve_order=True rather than reusing the
        # module-level HAND_JOINTS SceneEntityCfg: resolve() mutates the cfg,
        # and that object is shared with the managers that already resolved it.
        from isaac_neuromeka.tasks.manipulation.hand_grasp.hand_grasp_env_cfg import (
            HAND_JOINT_NAMES,
        )

        self._joint_ids, self._joint_names = robot.find_joints(
            HAND_JOINT_NAMES, preserve_order=True
        )
        if list(self._joint_names) != list(HAND_JOINT_NAMES):
            raise ValueError(
                "Resolved hand joints do not match HAND_JOINT_NAMES order:\n"
                f"  wanted {list(HAND_JOINT_NAMES)}\n  got    {list(self._joint_names)}"
            )
        self._robot = robot
        self._env = env
        self._output_dir = Path(output_dir)
        self._task = task
        self._checkpoint = checkpoint
        self._dt = float(env.unwrapped.step_dt)
        self._handle = None
        self._writer = None
        self._path = None
        self._rows = 0
        self._episode = 0
        self._started_at = None
        # OPEN/CLOSE segmentation.  The operator's 1/2 keys are what separates
        # one replayable motion from the next, so the transition is recorded
        # rather than left to be re-derived from the mode column downstream.
        self._segment = -1
        self._previous_mode = None
        self._segments = []

    @property
    def is_recording(self) -> bool:
        return self._handle is not None

    def toggle(self) -> None:
        """Key ``M``: start a new file, or close the open one."""
        if self.is_recording:
            self.stop("M key")
        else:
            self.start()

    def start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        self._path = self._output_dir / f"joint_record_{stamp}.csv"
        self._handle = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(
            [
                "step",
                "sim_time_s",
                "episode",
                "segment",
                "mode",
                "mode_changed",
                *(f"q_{name}" for name in self._joint_names),
                *(f"qt_{name}" for name in self._joint_names),
            ]
        )
        self._rows = 0
        self._episode = 0
        self._started_at = stamp
        self._segment = -1
        self._previous_mode = None
        self._segments = []
        print(
            f"[REC] recording hand joint angles at {1.0 / self._dt:.1f} Hz -> {self._path}\n"
            f"[REC] press M again to stop; an episode reset stops it automatically.",
            flush=True,
        )

    def capture(self) -> None:
        """Append one row.  Call once per ``env.step``; a no-op while stopped."""
        if not self.is_recording:
            return
        q = self._robot.data.joint_pos[0, self._joint_ids]
        q_target = self._robot.data.joint_pos_target[0, self._joint_ids]
        command = self._env.unwrapped.command_manager.get_command("open_close")
        mode = "CLOSE" if int(torch.argmax(command[0]).item()) == 1 else "OPEN"
        # A new segment starts on the first row and at every OPEN<->CLOSE
        # change.  The flagged row is the *first* of the new segment, i.e. the
        # first state the changed command produced.
        changed = mode != self._previous_mode
        if changed:
            self._close_open_segment()
            self._segment += 1
            self._segments.append(
                {
                    "index": self._segment,
                    "mode": mode,
                    "start_row": self._rows,
                    "start_time_s": round(self._rows * self._dt, 6),
                    "rows": 0,
                    "duration_s": 0.0,
                }
            )
            self._previous_mode = mode
        self._writer.writerow(
            [
                self._rows,
                f"{self._rows * self._dt:.6f}",
                self._episode,
                self._segment,
                mode,
                int(changed),
                *(f"{float(v):.9f}" for v in q),
                *(f"{float(v):.9f}" for v in q_target),
            ]
        )
        self._rows += 1
        # 30 Hz of small rows; flushing every one keeps the file readable while
        # the session is still open, which is the point of recording it.
        self._handle.flush()

    def _close_open_segment(self) -> None:
        """Fill in the row count of the segment that just ended."""
        if not self._segments:
            return
        last = self._segments[-1]
        last["rows"] = self._rows - last["start_row"]
        last["duration_s"] = round(last["rows"] * self._dt, 6)

    def stop(self, reason: str) -> None:
        if not self.is_recording:
            return
        self._close_open_segment()
        path, rows = self._path, self._rows
        self._handle.close()
        self._handle = None
        self._writer = None
        meta = {
            "task": self._task,
            "checkpoint": self._checkpoint,
            "recorded_at": self._started_at,
            "stopped_by": reason,
            "columns": {
                "q_*": "measured joint position (Articulation.data.joint_pos), radians",
                "qt_*": (
                    "commanded PD target (Articulation.data.joint_pos_target), radians."
                    " This is the quantity to send to MuJoCo/the real hand; qt-q is the"
                    " grip preload."
                ),
            },
            "policy_dt_s": self._dt,
            "policy_hz": 1.0 / self._dt,
            "rows": rows,
            "duration_s": rows * self._dt,
            "joint_names": list(self._joint_names),
            "env_index": 0,
            "segments": self._segments,
        }
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        summary = ", ".join(
            f"#{seg['index']} {seg['mode']} {seg['duration_s']:.2f}s"
            for seg in self._segments
        )
        print(
            f"[REC] stopped ({reason}): {rows} rows = {rows * self._dt:.2f}s, "
            f"{len(self._segments)} segment(s)\n"
            f"[REC]   {summary}\n"
            f"[REC]   {path}\n[REC]   {meta_path}",
            flush=True,
        )


def _acquire_debug_draw_interface():
    """Load debug draw lazily so normal/headless play does not depend on its extension."""
    import omni.kit.app

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_name = "isaacsim.util.debug_draw"
    if not extension_manager.is_extension_enabled(extension_name):
        extension_manager.set_extension_enabled_immediate(extension_name, True)

    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

    return omni_debug_draw.acquire_debug_draw_interface()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Play is used to inspect whether a policy can recover a weakened grasp.
    # Keep physical drop, timeout, and success terminations intact, but do not
    # reset merely because the functional-contact latch is lost.
    terminations_cfg = getattr(env_cfg, "terminations", None)
    if (
        terminations_cfg is not None
        and getattr(terminations_cfg, "functional_contact_lost", None) is not None
    ):
        terminations_cfg.functional_contact_lost = None
        print(
            "[INFO] Play override: disabled functional_contact_lost termination; "
            "other terminations remain active."
        )

    if args_cli.hand_mode is not None or args_cli.keyboard_hand_mode:
        mode_cfg = getattr(getattr(env_cfg, "commands", None), "open_close", None)
        if mode_cfg is None:
            raise ValueError(
                "--hand_mode/--keyboard_hand_mode requires an open_close command term."
            )
        # Keep the command manager from changing the requested mode between
        # resets.  The runtime override below also restores it immediately
        # after a drop reset, before the next policy observation is used.
        mode_cfg.resampling_time_range = (1.0e6, 1.0e6)
    if args_cli.alternate_hand_mode:
        # Play-only continuous OPEN/CLOSE demonstration: mode success and the
        # normal eight-second timeout must not reset the in-hand state.
        env_cfg.episode_length_s = 1.0e6
        success_cfg = getattr(getattr(env_cfg, "terminations", None), "success", None)
        if success_cfg is None:
            raise ValueError("--alternate_hand_mode requires a success termination term.")
        success_cfg.params["hold_steps"] = 1_000_000
        mode_cfg = getattr(getattr(env_cfg, "commands", None), "open_close", None)
        if mode_cfg is None:
            raise ValueError("--alternate_hand_mode requires an open_close command term.")
        mode_interval = max(float(args_cli.hand_mode_interval_s), 1.0e-3)
        mode_cfg.resampling_time_range = (mode_interval, mode_interval)

    # Must run before gym.make: it changes the actor input width, so the
    # checkpoint load below would otherwise fail the strict shape check.
    if args_cli.legacy_obs_101d:
        _apply_legacy_101d_contract(env_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # task cfg가 render_interval = decimation이라 policy step당 1프레임만 그림 (2.5 Hz).
    # 학습에는 맞는 설정이지만 play에서는 뷰포트가 끊기는 것처럼 보임.
    if args_cli.render_interval is not None:
        env_cfg.sim.render_interval = args_cli.render_interval

    # play 는 학습이 아니므로 보정 가드를 끈다.
    #
    # 그 가드의 목적은 "아무도 검증 안 한 기하로 학습이 돌아가는 것"을 막는 것이고,
    # play 는 오히려 그 기하를 "측정하러" 들어오는 경로다. 켜둔 채로 두면 보정
    # 세션을 시작할 때마다 env.require_calibration=false 를 붙여야 해서, 정작
    # 보정을 하려는 사람만 계속 막힌다. train.py 쪽 가드는 그대로 살아 있다.
    if hasattr(env_cfg, "require_calibration") and env_cfg.require_calibration:
        env_cfg.require_calibration = False
        print(
            "[INFO] play: require_calibration 을 자동으로 껐음"
            " (보정값이 없으면 스크립트 궤적은 스폰 자세를 유지)."
        )

    # 수동 조작 전용 태스크(hand_play)는 --manual_root 없이는 의미가 없다.
    # 그 씬은 캘리브레이션된 목표 자세가 테이블 상판 아래라, 스크립트 궤적을 그대로
    # 돌리면 손이 가구를 뚫고 들어간다.  플래그가 없는 기존 태스크는 전부 무영향.
    if getattr(env_cfg, "require_manual_root", False) and not args_cli.manual_root:
        raise SystemExit(
            f"[ERROR] --task {args_cli.task} 는 수동 조작 전용입니다. --manual_root 를"
            " 붙여 주세요.\n"
            "        이 씬은 캘리브레이션된 목표 자세(팜 z=0.365)가 테이블 상판"
            " (0.404) 아래라 스크립트 궤적으로는 가구를 통과합니다."
        )

    # specify directory for logging experiments
    #
    # --load_experiment lets a task load a checkpoint trained under a *different*
    # experiment name.  Needed because get_checkpoint_path only ever looks inside
    # logs/rsl_rl/<experiment_name>/, so a task that has never been trained (e.g.
    # hand_object starting from a hand_move policy) has no folder to search.
    experiment_name = args_cli.load_experiment or agent_cfg.experiment_name
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    if args_cli.load_experiment:
        print(
            f"[INFO] --load_experiment: resolving --load_run under '{experiment_name}'"
            f" instead of '{agent_cfg.experiment_name}'."
        )
    elif not os.path.isdir(log_root_path):
        # A task that logs under its own experiment name has no folder until it
        # has been trained once, so --load_run would fail on a task that is only
        # ever played from some other task's checkpoint. The env cfg names where
        # to look instead.
        fallback = getattr(env_cfg, "checkpoint_source_experiment", None)
        if fallback:
            fallback_path = os.path.abspath(os.path.join("logs", "rsl_rl", fallback))
            if os.path.isdir(fallback_path):
                print(
                    f"[INFO] '{experiment_name}' has no runs yet; resolving"
                    f" --load_run under '{fallback}' instead."
                    " Pass --load_experiment to choose explicitly."
                )
                experiment_name = fallback
                log_root_path = fallback_path
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.latest_run or agent_cfg.load_run in ("latest", "last"):
        agent_cfg.load_run = _latest_run_name(log_root_path)
        print(f"[INFO] Resolved latest run: {agent_cfg.load_run}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        # Preserve the standard explicit path/URI behavior, while also allowing
        # the convenient ``--load_run <run> --checkpoint model_300.pt`` form.
        # A bare filename is resolved inside the selected experiment/run;
        # paths containing a directory component remain explicit paths.
        checkpoint_arg = args_cli.checkpoint
        checkpoint_path = Path(checkpoint_arg).expanduser()
        is_bare_filename = (
            checkpoint_arg == checkpoint_path.name and not checkpoint_path.is_absolute()
        )
        if is_bare_filename:
            resume_path = get_checkpoint_path(
                log_root_path,
                agent_cfg.load_run,
                rf"^{re.escape(checkpoint_arg)}$",
            )
            print(
                f"[INFO] Resolved checkpoint filename '{checkpoint_arg}' inside"
                f" run '{agent_cfg.load_run}'."
            )
        else:
            resume_path = retrieve_file_path(str(checkpoint_path))
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    contact_sensor_names = _add_contact_sensors(env_cfg) if args_cli.print_contact else []
    hand_net_contact_sensor_name = (
        _add_hand_net_contact_sensor(env_cfg)
        if args_cli.show_hand_contact_forces or args_cli.plot_hand_contact_forces
        else None
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if version.parse(installed_version) >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    goal_command_term = None
    if args_cli.resample_goal_on_success:
        goal_command_term = env.unwrapped.command_manager.get_term("cube_goal")
        print("[INFO] Play-only goal resampling on transport success is enabled.")

    hand_mode_command_term = None
    previous_hand_mode = None
    requested_hand_mode = None
    hand_mode_key_queue = None
    hand_mode_input_interface = None
    hand_mode_keyboard = None
    hand_mode_keyboard_subscription = None
    if args_cli.hand_mode is not None:
        hand_mode_command_term = env.unwrapped.command_manager.get_term("open_close")
        requested_hand_mode = 0 if args_cli.hand_mode == "open" else 1
        hand_mode_command_term._command[:, 0] = float(requested_hand_mode == 0)
        hand_mode_command_term._command[:, 1] = float(requested_hand_mode == 1)
        hand_mode_command_term.time_left[:] = 1.0e6
        previous_hand_mode = requested_hand_mode
        obs = env.get_observations()
        print(
            f"[INFO] Fixed hand mode play: {args_cli.hand_mode.upper()}; "
            "the selected command is restored after every reset.",
            flush=True,
        )
    elif args_cli.keyboard_hand_mode:
        import carb.input
        import omni.appwindow

        hand_mode_command_term = env.unwrapped.command_manager.get_term("open_close")
        requested_hand_mode = 0
        hand_mode_command_term._command[:, 0] = 1.0
        hand_mode_command_term._command[:, 1] = 0.0
        hand_mode_command_term.time_left[:] = 1.0e6
        previous_hand_mode = requested_hand_mode
        obs = env.get_observations()

        hand_mode_key_queue = queue.SimpleQueue()

        def _on_hand_mode_key_press(event) -> bool:
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if event.input == carb.input.KeyboardInput.KEY_1:
                    hand_mode_key_queue.put(0)
                elif event.input == carb.input.KeyboardInput.KEY_2:
                    hand_mode_key_queue.put(1)
            return True

        hand_mode_input_interface = carb.input.acquire_input_interface()
        hand_mode_keyboard = (
            omni.appwindow.get_default_app_window().get_keyboard()
        )
        hand_mode_keyboard_subscription = (
            hand_mode_input_interface.subscribe_to_keyboard_events(
                hand_mode_keyboard,
                _on_hand_mode_key_press,
            )
        )
        print(
            "[INFO] Keyboard hand mode play: 1=OPEN, 2=CLOSE; starting in OPEN. "
            "The selected mode is restored after every reset.",
            flush=True,
        )
    elif args_cli.alternate_hand_mode:
        hand_mode_command_term = env.unwrapped.command_manager.get_term("open_close")
        hand_mode_command_term._command[:, 0] = 1.0
        hand_mode_command_term._command[:, 1] = 0.0
        hand_mode_command_term.time_left[:] = max(args_cli.hand_mode_interval_s, dt)
        previous_hand_mode = 0
        obs = env.get_observations()
        print(
            f"[INFO] Continuous hand mode play: OPEN first, switching every "
            f"{max(args_cli.hand_mode_interval_s, dt):.2f}s; "
            "success/time-out reset disabled, drop reset remains enabled.",
            flush=True,
        )

    close_response_command_term = None
    close_response_state = None
    close_response_tolerance_m = None
    close_response_hold_steps = 30
    if args_cli.debug_close_response:
        try:
            close_response_command_term = env.unwrapped.command_manager.get_term("open_close")
            success_cfg = env.unwrapped.reward_manager.get_term_cfg("success")
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "--debug_close_response requires open_close command plus success reward terms."
            ) from exc
        if args_cli.close_response_gap_tolerance_mm is None:
            close_response_tolerance_m = float(
                success_cfg.params.get("tip_gap_error_limit", 5.0e-4)
            )
        else:
            close_response_tolerance_m = (
                float(args_cli.close_response_gap_tolerance_mm) * 1.0e-3
            )
        close_response_hold_steps = int(success_cfg.params.get("hold_steps", 30))
        close_response_state = {
            "previous_mode": None,
            # Label of the mode currently being measured.  Every report line is
            # tagged with it so OPEN and CLOSE segments stay separable in a log.
            "mode_name": "OPEN",
            "active": False,
            "start_step": 0,
            "start_gap": float("nan"),
            "best_abs_error": float("inf"),
            "first_hit_s": None,
            "valid_steps": 0,
            "held": False,
        }
        print(
            "[INFO] OPEN/CLOSE response diagnostic enabled: "
            f"interval={args_cli.close_response_interval} steps, "
            f"timeout={args_cli.close_response_timeout_s:.2f}s, "
            f"gap_tolerance={close_response_tolerance_m * 1000.0:.3f}mm, "
            f"hold={close_response_hold_steps} steps.",
            flush=True,
        )

    manual_root_controller = None
    joint_recorder = None
    if args_cli.manual_root:
        from isaac_neuromeka.tasks.manipulation.hand_grasp.hand_move_manual_control import (
            HandMoveManualRootController,
        )

        for term_name in ("root_orientation", "open_close"):
            if term_name not in env.unwrapped.command_manager.active_terms:
                raise ValueError(
                    f"--manual_root requires a '{term_name}' command term"
                    " (hand_move); this task does not have one."
                )
        if "root_action" not in env.unwrapped.action_manager.active_terms:
            raise ValueError(
                "--manual_root requires a 'root_action' action term (hand_move)."
            )
        manual_root_controller = HandMoveManualRootController(
            env.unwrapped,
            translation_speed=args_cli.manual_translation_speed,
            rotation_speed=math.radians(args_cli.manual_rotation_speed_deg),
            max_translation_from_start=args_cli.manual_max_translation,
            # Manual tasks may choose the command that is valid at their reset
            # distribution.  hand_play starts OPEN like hand_object training;
            # existing hand_move/hand_object manual sessions retain CLOSE.
            initial_mode_index=int(
                getattr(env_cfg, "manual_root_initial_mode_index", 1)
            ),
        )
        # Bind the 'P' key to a geometry readout.  Two levels, chosen from what
        # the scene actually has rather than from the task name, so a renamed or
        # derived task keeps working:
        #   hand_move   -> root pose, relative local-z yaw, stick tips, midpoint
        #   hand_object -> the same plus the cube, the support and the forces
        #
        # The hand_object readout is the one to use: it adds the cube, so the
        # operator can see how far the tips still are from it, and it is what the
        # force saturation has to be measured with.  The plain hand_move readout
        # exists for flying that task around without a cube in the scene.
        from isaac_neuromeka.tasks.manipulation.hand_grasp import (  # noqa: E402
            hand_grasp_env_cfg as _hg_cfg,
            hand_move_mdp as _hm_mdp,
        )

        if "stick1" in env.unwrapped.scene.rigid_objects:
            if "stick1_cube_contact" in env.unwrapped.scene.sensors:
                from isaac_neuromeka.tasks.manipulation.hand_grasp import (
                    hand_object_env_cfg as _ho_cfg,
                    hand_object_mdp as _ho_mdp,
                )

                def _calibration_report() -> str:
                    return _ho_mdp.calibration_report(
                        env.unwrapped,
                        stick1_cfg=_ho_cfg.STICK_1,
                        stick2_cfg=_ho_cfg.STICK_2,
                        cube_cfg=_ho_cfg.OBJECT,
                        # hand_play swaps the retracting column for a table, so
                        # the support prim and its command term are absent
                        # there.  Pass None and the readout drops those rows.
                        support_cfg=(
                            _ho_cfg.OBJECT_SUPPORT
                            if "object_support" in env.unwrapped.scene.rigid_objects
                            else None
                        ),
                        tip_offset_o=_ho_cfg.STICK_TIP_OFFSET_O,
                        stick1_sensor_name=_ho_cfg.STICK1_CUBE_SENSOR,
                        stick2_sensor_name=_ho_cfg.STICK2_CUBE_SENSOR,
                    )

            else:

                def _calibration_report() -> str:
                    return _hm_mdp.geometry_report(
                        env.unwrapped,
                        stick1_cfg=_hg_cfg.STICK_1,
                        stick2_cfg=_hg_cfg.STICK_2,
                        tip_offset_o=_hg_cfg.STICK_TIP_OFFSET_O,
                    )

            manual_root_controller.calibration_reporter = _calibration_report

        # The 'M' key.  Built here rather than lazily on the first press so a
        # bad --joint_record_dir fails before the operator has flown the hand
        # into position, not after.
        if "open_close" in env.unwrapped.command_manager.active_terms:
            joint_recorder = HandJointRecorder(
                env,
                output_dir=args_cli.joint_record_dir,
                task=args_cli.task,
                checkpoint=resume_path,
            )
            manual_root_controller.record_toggle = joint_recorder.toggle
            print(
                f"[INFO] Joint recording available: press M "
                f"(output dir {Path(args_cli.joint_record_dir).resolve()})."
            )
        manual_root_controller.attach()
        obs = env.get_observations()

    debug_draw_interface = None
    hand_contact_plot_publisher = None
    hand_contact_body_ids = []
    hand_contact_body_names = []
    palm_body_id = None
    grasp_opening_b = (0.19, 0.28, 0.94)
    stick_frame_visualizer = None
    stick_frame_names: list[str] = []
    if args_cli.show_stick_frames:
        stick_frame_scale = max(float(args_cli.stick_frame_scale), 0.001)
        stick_frame_visualizer, stick_frame_names = _create_stick_frame_visualizer(
            env, stick_frame_scale
        )
        if stick_frame_visualizer is None:
            raise ValueError(
                "--show_stick_frames requires a scene rigid object named "
                f"{' or '.join(STICK_FRAME_OBJECTS)}; --task {args_cli.task} has neither."
            )
        print(
            f"[INFO] Stick local frames (env 0): {', '.join(stick_frame_names)}; "
            f"axis length={stick_frame_scale:.3f}m, RED=+X / GREEN=+Y (tail->tip shaft) / BLUE=+Z. "
            "Origin is the 180mm cuboid geometric centre."
        )
    if args_cli.show_palm_vectors:
        if "cube" not in env.unwrapped.scene.rigid_objects:
            raise ValueError("--show_palm_vectors requires a scene rigid object named 'cube'.")
        palm_ids, _ = env.unwrapped.scene["robot"].find_bodies(["palm_link"])
        if len(palm_ids) != 1:
            raise ValueError(f"Expected one palm_link body, found {len(palm_ids)}.")
        palm_body_id = palm_ids[0]
        rewards_cfg = getattr(env.unwrapped.cfg, "rewards", None)
        for term_name in ("palm_facing", "finger_cage_reach"):
            term_cfg = getattr(rewards_cfg, term_name, None)
            term_params = getattr(term_cfg, "params", {}) or {}
            if "palm_normal_b" in term_params:
                grasp_opening_b = tuple(term_params["palm_normal_b"])
                break
        print(
            "[INFO] Palm vector legend: RED=palm-plane normal local=(0.965, -0.008, 0.262), "
            f"BLUE=grasp opening axis {grasp_opening_b}, GREEN=palm-to-cube"
        )
    if hand_net_contact_sensor_name is not None:
        if hand_net_contact_sensor_name not in env.unwrapped.scene.sensors:
            raise ValueError(
                f"Play contact sensor '{hand_net_contact_sensor_name}' was not created."
            )
        hand_sensor = env.unwrapped.scene.sensors[hand_net_contact_sensor_name]
        hand_contact_body_ids = [
            body_id
            for body_id, body_name in enumerate(hand_sensor.body_names)
            if _is_hand_contact_body(body_name)
        ]
        hand_contact_body_names = [
            hand_sensor.body_names[body_id] for body_id in hand_contact_body_ids
        ]
        if not hand_contact_body_ids:
            raise ValueError(
                "Hand contact-force diagnostic found no palm/finger rigid bodies "
                f"among: {hand_sensor.body_names}"
            )
    if args_cli.show_hand_contact_forces:
        print(
            "[INFO] Hand contact-force legend: direction=world force on link, "
            f"length=min(F*{max(float(args_cli.hand_contact_force_scale), 0.0):.3f}m/N, "
            f"{max(float(args_cli.hand_contact_force_max_length), 0.001):.3f}m), "
            "GREEN<0.1N, YELLOW<0.5N, RED>=0.5N. "
            "Values are net normal contact force at each link origin, not actuator torque."
        )
    hand_contact_plot_interval = 1
    if args_cli.plot_hand_contact_forces:
        plot_history_s = float(args_cli.hand_contact_plot_history)
        plot_rate_hz = float(args_cli.hand_contact_plot_hz)
        if not math.isfinite(plot_history_s) or plot_history_s <= 0.0:
            raise ValueError("--hand_contact_plot_history must be finite and > 0.")
        if not math.isfinite(plot_rate_hz) or plot_rate_hz <= 0.0:
            raise ValueError("--hand_contact_plot_hz must be finite and > 0.")
        try:
            from hand_contact_force_plot import HandContactPlotPublisher

            hand_contact_plot_publisher = HandContactPlotPublisher(
                history_seconds=max(plot_history_s, 1.0),
                force_threshold_n=max(float(args_cli.hand_contact_force_threshold), 0.0),
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                "Failed to start the external hand contact-force plot."
            ) from exc
        print(
            "[INFO] External hand contact-force graph started in a separate process: "
            f"{len(hand_contact_body_names)} hand links, "
            f"history={max(plot_history_s, 1.0):.1f}s, "
            f"telemetry<={plot_rate_hz:.1f}Hz."
        )
        hand_contact_plot_interval = max(
            1,
            int(round(1.0 / (plot_rate_hz * float(dt)))),
        )
    if args_cli.show_palm_vectors or args_cli.show_hand_contact_forces:
        debug_draw_interface = _acquire_debug_draw_interface()

    # Action term 종류와 무관하게 정책 출력, 관절 target, 실제 상태를 함께 비교한다.
    # 추종오차가 작으면 정책/보상 문제이고, 크면 drive/관절 물리 문제를 우선 의심한다.
    print_interval = max(args_cli.print_action_interval, 1)
    action_debug_enabled = args_cli.print_action or args_cli.debug_close_response
    if action_debug_enabled:
        robot = env.unwrapped.scene["robot"]
        action_manager = env.unwrapped.action_manager
        action_term_names = list(getattr(action_manager, "_term_names", []))
        if "arm_action" in action_term_names:
            action_term_name = "arm_action"
        elif "hand_action" in action_term_names:
            action_term_name = "hand_action"
        else:
            raise ValueError(
                "--print_action requires an arm_action or hand_action term; "
                f"available terms: {action_term_names}"
            )
        action_term = action_manager.get_term(action_term_name)
        joint_names = list(action_term._joint_names)
        hand_ids = [i for i, n in enumerate(robot.body_names) if "finger" in n or "palm" in n]
        prev_action = torch.zeros_like(policy(obs))
        clipped_limit = agent_cfg.clip_actions
        detail_scope = "finger*_joint4" if args_cli.print_hand_joint4 else "all controlled joints"
        if args_cli.print_action:
            print(
                f"[INFO] Action diagnostics: term={action_term_name}, detail={detail_scope}, "
                f"interval={print_interval} policy steps."
            )
            print(
                f"\n{'step':>5}{'|raw|':>9}{'|applied|':>10}{'clip%':>8}"
                f"{'|Δa|평균':>10}{'|Δa|최대':>10}{'추종오차(rad)':>14}{'관절속도':>9}{'손최저z(cm)':>12}"
            )

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if hand_mode_key_queue is not None:
                keyboard_mode = None
                while True:
                    try:
                        keyboard_mode = hand_mode_key_queue.get_nowait()
                    except queue.Empty:
                        break
                if keyboard_mode is not None:
                    requested_hand_mode = keyboard_mode
                    hand_mode_command_term._command[:, 0] = float(
                        requested_hand_mode == 0
                    )
                    hand_mode_command_term._command[:, 1] = float(
                        requested_hand_mode == 1
                    )
                    hand_mode_command_term.time_left[:] = 1.0e6
                    previous_hand_mode = requested_hand_mode
                    obs = env.get_observations()
                    mode_name = "OPEN" if requested_hand_mode == 0 else "CLOSE"
                    print(
                        f"[INFO] keyboard hand mode selected: {mode_name}.",
                        flush=True,
                    )
            if manual_root_controller is not None:
                # Integrate the keyboard into the root PD *targets* before the
                # policy runs.  The policy still sees the full 103D observation
                # and still produces the 20D finger action; only the wrist
                # target moved.
                if manual_root_controller.update(env.unwrapped.step_dt):
                    obs = env.get_observations()
                if manual_root_controller.consume_env_reset_request():
                    # The 'R' key. Done here, between steps, because env.reset()
                    # rebuilds the very buffers the controller integrates.
                    if joint_recorder is not None:
                        joint_recorder.stop("env reset by 'R'")
                    env.unwrapped.reset()
                    manual_root_controller.on_env_reset()
                    obs = env.get_observations()
                    print("[INFO] env reset by keyboard.", flush=True)
            if close_response_command_term is not None:
                # Detect the command represented in the observation that is
                # about to enter the policy.  Doing this before inference makes
                # the measured latency start at the first mode-conditioned
                # policy action, rather than one step before/after it.
                close_snapshot = _hand_mode_gap_snapshot(env)
                close_mode = int(close_snapshot["mode_index"])
                previous_mode = close_response_state["previous_mode"]
                if previous_mode is None or close_mode != previous_mode:
                    # Measure both directions.  Separating the tips costs more
                    # distal torque than letting them meet, so a CLOSE-only
                    # probe cannot distinguish a slow command from one the
                    # current effort limits make unreachable.
                    if close_response_state["active"]:
                        first_hit = close_response_state["first_hit_s"]
                        print(
                            f"[{close_response_state['mode_name']} RESPONSE END] "
                            "command changed before timeout: "
                            f"best_error={close_response_state['best_abs_error'] * 1000.0:.3f}mm "
                            f"first_hit={'never' if first_hit is None else f'{first_hit:.3f}s'} "
                            f"held={close_response_state['held']}",
                            flush=True,
                        )
                        close_response_state["active"] = False
                    close_response_state.update(
                        {
                            "active": True,
                            "mode_name": str(close_snapshot["mode_name"]),
                            "start_step": timestep,
                            "start_gap": float(close_snapshot["actual_gap"]),
                            "best_abs_error": abs(float(close_snapshot["error"])),
                            "first_hit_s": None,
                            "valid_steps": 0,
                            "held": False,
                        }
                    )
                    print(
                        f"[{close_response_state['mode_name']} RESPONSE START] "
                        f"policy_step={timestep} "
                        f"gap={float(close_snapshot['actual_gap']) * 1000.0:.3f}mm "
                        f"target={float(close_snapshot['target_gap']) * 1000.0:.3f}mm",
                        flush=True,
                    )
                    close_response_state["previous_mode"] = close_mode
            # agent stepping
            actions = policy(obs)
            if goal_command_term is not None:
                command_counter_before = goal_command_term.command_counter.clone()
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if joint_recorder is not None:
                # One row per env.step == one policy step == 1/30 s.  Taken
                # after the step so the logged angles are the state the next
                # observation is built from, and before the reset check below
                # so the final pre-reset pose is not lost.
                joint_recorder.capture()
                if bool(dones.any().item()):
                    joint_recorder.stop("episode reset")
            if manual_root_controller is not None and bool(dones.any().item()):
                # The action term and the command term have already re-captured
                # the fresh functional-grasp pose; drop stale key state so a key
                # held across the reset does not keep integrating.
                manual_root_controller.on_env_reset()
                obs = env.get_observations()
            if hand_mode_command_term is not None:
                current_hand_mode = int(
                    torch.argmax(hand_mode_command_term._command[0]).item()
                )
                if requested_hand_mode is not None:
                    mode_changed = bool(
                        torch.any(
                            torch.argmax(hand_mode_command_term._command, dim=-1)
                            != requested_hand_mode
                        ).item()
                    )
                    hand_mode_command_term._command[:, 0] = float(
                        requested_hand_mode == 0
                    )
                    hand_mode_command_term._command[:, 1] = float(
                        requested_hand_mode == 1
                    )
                    hand_mode_command_term.time_left[:] = 1.0e6
                    if mode_changed:
                        current_hand_mode = requested_hand_mode
                        obs = env.get_observations()
                if current_hand_mode != previous_hand_mode:
                    previous_hand_mode = current_hand_mode
                    mode_name = "OPEN" if current_hand_mode == 0 else "CLOSE"
                    transition_kind = (
                        "after drop reset"
                        if bool(dones[0].item())
                        else "without reset"
                    )
                    print(
                        f"[INFO] hand mode switched to {mode_name} {transition_kind}.",
                        flush=True,
                    )
            if goal_command_term is not None:
                success_env_ids = _success_env_ids(env)
                if len(success_env_ids) > 0:
                    still_active = dones[success_env_ids] == 0
                    timer_resampled = (
                        goal_command_term.command_counter[success_env_ids]
                        != command_counter_before[success_env_ids]
                    )
                    success_env_ids = success_env_ids[still_active & (~timer_resampled)]
                    if len(success_env_ids) > 0:
                        goal_command_term._resample(success_env_ids)
                        obs = env.get_observations()
                        print(
                            f"[INFO] Success: resampled cube_goal for envs {success_env_ids.tolist()}.",
                            flush=True,
                        )

            if debug_draw_interface is not None:
                # This interface owns one shared line buffer. Clear exactly
                # once, then append every enabled diagnostic for this frame.
                debug_draw_interface.clear_lines()
            if stick_frame_visualizer is not None:
                _draw_stick_frames(env, stick_frame_visualizer, stick_frame_names)
            if args_cli.show_palm_vectors:
                _draw_palm_vectors(
                    env,
                    debug_draw_interface,
                    palm_body_id,
                    grasp_opening_b,
                    max(float(args_cli.palm_vector_length), 0.01),
                )
            if args_cli.show_hand_contact_forces:
                force_threshold_n = max(
                    float(args_cli.hand_contact_force_threshold), 0.0
                )
                _draw_hand_contact_forces(
                    env,
                    debug_draw_interface,
                    hand_net_contact_sensor_name,
                    max(float(args_cli.hand_contact_force_scale), 0.0),
                    max(float(args_cli.hand_contact_force_max_length), 0.001),
                    force_threshold_n,
                )
                if timestep % max(args_cli.hand_contact_print_interval, 1) == 0:
                    _print_hand_contact_forces(
                        env,
                        hand_net_contact_sensor_name,
                        force_threshold_n,
                    )
            if (
                hand_contact_plot_publisher is not None
                and hand_contact_plot_publisher.running
                and timestep % hand_contact_plot_interval == 0
            ):
                hand_sensor = env.unwrapped.scene.sensors[
                    hand_net_contact_sensor_name
                ]
                force_magnitudes = torch.linalg.norm(
                    hand_sensor.data.net_forces_w[0, hand_contact_body_ids],
                    dim=-1,
                )
                hand_contact_plot_publisher.publish(
                    simulation_time=float(timestep) * float(dt),
                    body_names=hand_contact_body_names,
                    force_magnitudes_n=force_magnitudes.detach().cpu().tolist(),
                )

            if args_cli.print_hand_mode and timestep % print_interval == 0:
                _print_hand_mode_diagnostics(env, episode_reset=bool(dones[0].item()))

            if action_debug_enabled:
                applied = env.unwrapped.action_manager.action
                # 잔차 액션은 절대 목표가 joint_pos_target에 있음 (절대형은 processed_actions)
                target = getattr(action_term, "joint_pos_target", None)
                if target is None:
                    target = action_term.processed_actions
                actual = robot.data.joint_pos[:, action_term._joint_ids]
                joint_vel = robot.data.joint_vel[:, action_term._joint_ids]
                if args_cli.print_action:
                    delta = (applied - prev_action).abs()
                    prev_action = applied.clone()

                if args_cli.print_action and timestep % print_interval == 0:
                    if clipped_limit is None:
                        clip_ratio = torch.zeros((), device=actions.device)
                    else:
                        clip_ratio = (actions.abs() > float(clipped_limit)).float().mean() * 100.0
                    print(
                        f"{timestep:>5}{actions.abs().mean():>9.3f}{applied.abs().mean():>10.3f}"
                        f"{clip_ratio:>8.1f}{delta.mean():>10.3f}{delta.max():>10.3f}"
                        f"{(target - actual).abs().max():>14.3f}"
                        f"{joint_vel.abs().max():>9.2f}"
                        f"{robot.data.body_pos_w[:, hand_ids, 2].min() * 100:>12.2f}"
                    )

                    if args_cli.print_action_detail:
                        print(
                            "      joint                  raw   applied    target    actual       err"
                            "       vel    v%   torque   tq%   comp"
                        )
                        # 잔차/전관절 액션은 _joint_ids가 slice(None) → [i] 인덱싱 불가. slice면 i 그대로.
                        _jids = action_term._joint_ids
                        for i, name in enumerate(joint_names):
                            if args_cli.print_hand_joint4 and not name.endswith("_joint4"):
                                continue
                            joint_id = i if isinstance(_jids, slice) else _jids[i]
                            err = target[0, i] - actual[0, i]
                            vel = joint_vel[0, i]
                            torque = robot.data.applied_torque[0, joint_id]
                            comp_torque = robot.data.computed_torque[0, joint_id]
                            effort_limit = robot.data.joint_effort_limits[0, joint_id]
                            vel_limit = robot.data.joint_vel_limits[0, joint_id]
                            vel_pct = _ratio(vel, vel_limit) * 100.0
                            torque_pct = _ratio(torque, effort_limit) * 100.0
                            print(
                                f"      {name:<20}"
                                f"{actions[0, i]:>8.3f}{applied[0, i]:>10.3f}"
                                f"{target[0, i]:>10.3f}{actual[0, i]:>10.3f}"
                                f"{err:>10.3f}{vel:>10.3f}{vel_pct:>6.0f}"
                                f"{torque:>9.2f}{torque_pct:>6.0f}{comp_torque:>8.2f}"
                            )
                    if args_cli.print_diagnostics:
                        _print_reward_cube_diagnostics(env)
                        _print_contacts(env, contact_sensor_names)
                    if args_cli.print_pose_diagnostics:
                        _print_pose_diagnostics(env)

                if args_cli.debug_close_response and close_response_state["active"]:
                    response_mode = close_response_state["mode_name"]
                    if bool(dones[0].item()):
                        print(
                            f"[{response_mode} RESPONSE RESET] "
                            f"env 0 reset before {response_mode} completed.",
                            flush=True,
                        )
                        close_response_state["active"] = False
                        close_response_state["previous_mode"] = None
                    else:
                        close_snapshot = _hand_mode_gap_snapshot(env)
                        elapsed_s = (
                            timestep - int(close_response_state["start_step"]) + 1
                        ) * float(dt)
                        abs_error = abs(float(close_snapshot["error"]))
                        close_response_state["best_abs_error"] = min(
                            float(close_response_state["best_abs_error"]),
                            abs_error,
                        )
                        within_tolerance = abs_error <= float(close_response_tolerance_m)
                        if within_tolerance:
                            close_response_state["valid_steps"] += 1
                            if close_response_state["first_hit_s"] is None:
                                close_response_state["first_hit_s"] = elapsed_s
                                print(
                                    f"[{response_mode} RESPONSE FIRST HIT] "
                                    f"latency={elapsed_s:.3f}s "
                                    f"error={abs_error * 1000.0:.3f}mm",
                                    flush=True,
                                )
                        else:
                            close_response_state["valid_steps"] = 0
                        if (
                            not close_response_state["held"]
                            and close_response_state["valid_steps"]
                            >= close_response_hold_steps
                        ):
                            close_response_state["held"] = True
                            print(
                                f"[{response_mode} RESPONSE HELD] "
                                f"latency={elapsed_s:.3f}s; "
                                f"within {close_response_tolerance_m * 1000.0:.3f}mm for "
                                f"{close_response_hold_steps} consecutive policy steps.",
                                flush=True,
                            )
                        if timestep % args_cli.close_response_interval == 0:
                            computed = robot.data.computed_torque[
                                0, action_term._joint_ids
                            ]
                            applied_torque = robot.data.applied_torque[
                                0, action_term._joint_ids
                            ]
                            effort_limit = robot.data.joint_effort_limits[
                                0, action_term._joint_ids
                            ].abs()
                            torque_tolerance = torch.maximum(
                                effort_limit * 1.0e-4,
                                torch.full_like(effort_limit, 1.0e-6),
                            )
                            saturation_count = int(
                                torch.count_nonzero(
                                    (computed.abs() > effort_limit + torque_tolerance)
                                    | (
                                        (computed - applied_torque).abs()
                                        > torque_tolerance
                                    )
                                ).item()
                            )
                            action_clip_count = 0
                            if agent_cfg.clip_actions is not None:
                                action_clip_count = int(
                                    torch.count_nonzero(
                                        actions[0].abs()
                                        > float(agent_cfg.clip_actions)
                                    ).item()
                                )
                            # Effort headroom, not just the binary saturation
                            # count: a joint sitting at 95% of its limit is
                            # already the wall even though it never trips the
                            # exact-equality test above.
                            effort_use = applied_torque.abs() / effort_limit.clamp(
                                min=1.0e-9
                            )
                            worst_joint = int(torch.argmax(effort_use).item())
                            print(
                                f"[{response_mode} RESPONSE] "
                                f"t={elapsed_s:.3f}s "
                                f"gap={float(close_snapshot['actual_gap']) * 1000.0:.3f}mm "
                                f"error={float(close_snapshot['error']) * 1000.0:+.3f}mm "
                                f"lateral={float(close_snapshot['lateral_error']) * 1000.0:.3f}mm "
                                f"best={close_response_state['best_abs_error'] * 1000.0:.3f}mm "
                                f"valid={close_response_state['valid_steps']}/{close_response_hold_steps} "
                                f"action_clip={action_clip_count} "
                                f"tracking_max={(target - actual).abs().max().item():.3f}rad "
                                f"torque_sat={saturation_count}/{len(joint_names)} "
                                f"effort_max={effort_use[worst_joint].item() * 100.0:.0f}%"
                                f"({joint_names[worst_joint]})",
                                flush=True,
                            )
                            _print_mode_response_joint_table(
                                actions=actions,
                                applied=applied,
                                target=target,
                                actual=actual,
                                joint_vel=joint_vel,
                                robot=robot,
                                action_term=action_term,
                                joint_names=joint_names,
                            )
                        if elapsed_s >= float(args_cli.close_response_timeout_s):
                            first_hit = close_response_state["first_hit_s"]
                            print(
                                f"[{response_mode} RESPONSE TIMEOUT] "
                                f"elapsed={elapsed_s:.3f}s "
                                f"best_error={close_response_state['best_abs_error'] * 1000.0:.3f}mm "
                                f"first_hit={'never' if first_hit is None else f'{first_hit:.3f}s'} "
                                f"held={close_response_state['held']}",
                                flush=True,
                            )
                            close_response_state["active"] = False
            timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if joint_recorder is not None:
        # Flushed every row already, but this is what writes the .meta.json for
        # a session closed with the window still recording.
        joint_recorder.stop("session end")
    if hand_contact_plot_publisher is not None:
        hand_contact_plot_publisher.close()
    if debug_draw_interface is not None:
        debug_draw_interface.clear_lines()
    if manual_root_controller is not None:
        manual_root_controller.detach()
    if hand_mode_keyboard_subscription is not None:
        hand_mode_input_interface.unsubscribe_to_keyboard_events(
            hand_mode_keyboard,
            hand_mode_keyboard_subscription,
        )
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
