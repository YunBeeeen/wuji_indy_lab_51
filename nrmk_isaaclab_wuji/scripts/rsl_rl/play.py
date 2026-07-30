# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import queue
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
    "--show_palm_vectors",
    action="store_true",
    default=False,
    help="env 0의 palm_link에서 손바닥 평면 법선, 파지 개구부 축, cube 방향을 화살표로 표시.",
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
if args_cli.print_pose_diagnostics:
    args_cli.print_action = True
if args_cli.print_contact:
    args_cli.print_action = True
    args_cli.print_diagnostics = True
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
if hand_mode_selector_count > 0:
    args_cli.print_hand_mode = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import os
import time

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
        from isaac_neuromeka.tasks.manipulation.hand_grasp import mdp as hand_grasp_mdp

        unwrapped = env.unwrapped
        open_cfg = unwrapped.reward_manager.get_term_cfg("open_tip_gap")
        close_cfg = unwrapped.reward_manager.get_term_cfg("close_tip_gap")
        command_name = open_cfg.params["command_name"]
        command = unwrapped.command_manager.get_command(command_name)
        mode_index = int(torch.argmax(command[0]).item())
        mode_name = "OPEN" if mode_index == 0 else "CLOSE"
        mode_cfg = open_cfg if mode_index == 0 else close_cfg
        params = mode_cfg.params
        gap = hand_grasp_mdp._tip_surface_gap(
            unwrapped,
            params["palm_cfg"],
            params["stick1_cfg"],
            params["stick2_cfg"],
            params["stick1_tip_offset_o"],
            params["stick2_tip_offset_o"],
            params["stick_thickness"],
        )
        target_gap = float(params["target_gap"])
        actual_gap = float(gap[0].item())
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

    draw_interface.clear_lines()
    draw_interface.draw_lines(starts, ends, line_colors, thicknesses)


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

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # task cfg가 render_interval = decimation이라 policy step당 1프레임만 그림 (2.5 Hz).
    # 학습에는 맞는 설정이지만 play에서는 뷰포트가 끊기는 것처럼 보임.
    if args_cli.render_interval is not None:
        env_cfg.sim.render_interval = args_cli.render_interval

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
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
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    contact_sensor_names = _add_contact_sensors(env_cfg) if args_cli.print_contact else []

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

    palm_draw_interface = None
    palm_body_id = None
    grasp_opening_b = (0.19, 0.28, 0.94)
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
        palm_draw_interface = _acquire_debug_draw_interface()
        print(
            "[INFO] Palm vector legend: RED=palm-plane normal local=(0.965, -0.008, 0.262), "
            f"BLUE=grasp opening axis {grasp_opening_b}, GREEN=palm-to-cube"
        )

    # action은 절대 관절 목표임 (target = default_joint_pos + scale * action).
    # 팔이 튈 때 그게 "정책이 시킨 것"인지 "물리가 명령을 이긴 것"인지 가르려면 목표와 실제를 비교해야 함.
    #   추종오차 작음(<0.1) + |Δa| 큼(>0.3) -> 팔이 명령대로 발광. 학습/보상 문제
    #   추종오차 큼  (>0.3)                 -> 물리가 명령을 이김. dt/decimation 문제
    print_interval = max(args_cli.print_action_interval, 1)
    if args_cli.print_action:
        robot = env.unwrapped.scene["robot"]
        action_term = env.unwrapped.action_manager.get_term("arm_action")
        joint_names = list(action_term._joint_names)
        arm_ids, _ = robot.find_joints(["joint[0-5]"])
        hand_ids = [i for i, n in enumerate(robot.body_names) if "finger" in n or "palm" in n]
        prev_action = torch.zeros_like(policy(obs))
        clipped_limit = agent_cfg.clip_actions
        print(
            f"\n{'step':>5}{'|raw|':>9}{'|applied|':>10}{'clip%':>8}"
            f"{'|Δa|평균':>10}{'|Δa|최대':>10}{'추종오차(rad)':>14}{'팔속도':>9}{'손최저z(cm)':>12}"
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
            # agent stepping
            actions = policy(obs)
            if goal_command_term is not None:
                command_counter_before = goal_command_term.command_counter.clone()
            # env stepping
            obs, _, dones, _ = env.step(actions)
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

            if palm_draw_interface is not None:
                _draw_palm_vectors(
                    env,
                    palm_draw_interface,
                    palm_body_id,
                    grasp_opening_b,
                    max(float(args_cli.palm_vector_length), 0.01),
                )

            if args_cli.print_hand_mode and timestep % print_interval == 0:
                _print_hand_mode_diagnostics(env, episode_reset=bool(dones[0].item()))

            if args_cli.print_action:
                applied = env.unwrapped.action_manager.action
                # 잔차 액션은 절대 목표가 joint_pos_target에 있음 (절대형은 processed_actions)
                target = getattr(action_term, "joint_pos_target", None)
                if target is None:
                    target = action_term.processed_actions
                actual = robot.data.joint_pos[:, action_term._joint_ids]
                joint_vel = robot.data.joint_vel[:, action_term._joint_ids]
                delta = (applied - prev_action).abs()
                prev_action = applied.clone()

                if timestep % print_interval == 0:
                    if clipped_limit is None:
                        clip_ratio = torch.zeros((), device=actions.device)
                    else:
                        clip_ratio = (actions.abs() > float(clipped_limit)).float().mean() * 100.0
                    print(
                        f"{timestep:>5}{actions.abs().mean():>9.3f}{applied.abs().mean():>10.3f}"
                        f"{clip_ratio:>8.1f}{delta.mean():>10.3f}{delta.max():>10.3f}"
                        f"{(target - actual).abs().max():>14.3f}"
                        f"{robot.data.joint_vel[:, arm_ids].abs().max():>9.2f}"
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
    if palm_draw_interface is not None:
        palm_draw_interface.clear_lines()
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
