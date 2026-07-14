# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

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
    "--render_interval",
    type=int,
    default=2,
    help="렌더 프레임 사이의 physics step 수. 낮을수록 부드러움. task의 decimation을 주면 학습 때 값 그대로.",
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
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
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
import isaac_neuromeka.tasks  # noqa: F401


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
                        print("      joint                  raw   applied    target    actual       err       vel")
                        for i, name in enumerate(joint_names):
                            err = target[0, i] - actual[0, i]
                            print(
                                f"      {name:<20}"
                                f"{actions[0, i]:>8.3f}{applied[0, i]:>10.3f}"
                                f"{target[0, i]:>10.3f}{actual[0, i]:>10.3f}"
                                f"{err:>10.3f}{joint_vel[0, i]:>10.3f}"
                            )
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
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
