# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
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
if args_cli.print_contact:
    args_cli.print_action = True
    args_cli.print_diagnostics = True

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
    if args_cli.print_action:
        robot = env.unwrapped.scene["robot"]
        action_term = env.unwrapped.action_manager.get_term("arm_action")
        joint_names = list(action_term._joint_names)
        arm_ids, _ = robot.find_joints(["joint[0-5]"])
        hand_ids = [i for i, n in enumerate(robot.body_names) if "finger" in n or "palm" in n]
        prev_action = torch.zeros_like(policy(obs))
        print_interval = max(args_cli.print_action_interval, 1)
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
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            if palm_draw_interface is not None:
                _draw_palm_vectors(
                    env,
                    palm_draw_interface,
                    palm_body_id,
                    grasp_opening_b,
                    max(float(args_cli.palm_vector_length), 0.01),
                )

            if args_cli.print_action:
                applied = env.unwrapped.action_manager.action
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
                        for i, name in enumerate(joint_names):
                            joint_id = action_term._joint_ids[i]
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
                timestep += 1
        if args_cli.video:
            timestep += 1
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
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
