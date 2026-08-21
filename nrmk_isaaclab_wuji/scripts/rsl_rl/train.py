# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

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
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--render_interval",
    type=int,
    default=None,
    help=(
        "렌더 프레임 사이의 physics step 수. GUI나 RTX 센서가 켜져 있을 때만 효과가 있으므로"
        " --headless 학습 속도에는 영향이 없음. 기본값은 task cfg(= decimation)이며 policy step마다"
        " 1프레임만 그림. GUI로 학습을 볼 때는 4 정도로 낮출 것."
    ),
)
parser.add_argument(
    "--reset_policy_std",
    type=float,
    default=None,
    help="checkpoint resume 직후 Gaussian policy std를 이 값으로 재설정하고 해당 optimizer state를 초기화.",
)
parser.add_argument(
    "--load_actor_only",
    action="store_true",
    default=False,
    help="resume checkpoint에서 actor만 불러오고 critic/optimizer/iteration은 새 run으로 초기화.",
)
parser.add_argument(
    "--init_checkpoint",
    type=str,
    default=None,
    help=(
        "다른 experiment의 체크포인트 .pt 경로를 명시적으로 지정해 fine-tuning 시작."
        " --resume/--load_run 은 logs/rsl_rl/<현재 experiment_name>/ 안에서만 찾기 때문에"
        " (train.py:168, get_checkpoint_path) experiment_name 이 다른 런에서 이어받을 수 없다."
        " 예: hand_move 정책으로 hand_object 를 fine-tuning 하는 경우."
        " 'latest' 를 암묵적으로 고르지 않고 지정한 파일만 사용한다."
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import os
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import omni
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# Neuromeka tasks
import isaac_neuromeka.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # 수동 조작 전용 씬(hand_play)은 학습 대상이 아니다.  그 씬은 캘리브레이션된 목표
    # 자세가 테이블 상판 아래라 스크립트 궤적이 가구를 통과하고, 그 상태로 몇 시간을
    # 돌린 뒤에야 이상함을 눈치채게 된다.  플래그가 없는 기존 태스크는 전부 무영향.
    if getattr(env_cfg, "require_manual_root", False):
        raise SystemExit(
            f"[ERROR] --task {args_cli.task} 는 수동 play 전용이라 학습할 수 없습니다.\n"
            "        스크립트 궤적의 목표 자세(팜 z=0.365)가 테이블 상판(0.404)"
            " 아래라 손이 가구를 통과합니다.\n"
            "        학습은 --task hand_object 로 하세요."
        )

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # task cfg가 render_interval = decimation이라 policy step당 1프레임만 그림.
    # decimation 24 + physics 1/60초면 시뮬 1초당 2.5프레임 -> GUI로 보면 슬라이드쇼처럼 보임.
    # GUI / RTX 센서가 있을 때만 영향을 줌 (headless 학습 속도와 무관).
    if args_cli.render_interval is not None:
        env_cfg.sim.render_interval = args_cli.render_interval

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if args_cli.init_checkpoint is not None:
        # 명시 경로 fine-tuning. get_checkpoint_path 는 현재 experiment_name 폴더
        # 안에서만 찾으므로 다른 태스크의 체크포인트는 이 경로로만 받을 수 있다.
        resume_path = os.path.abspath(args_cli.init_checkpoint)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(
                f"--init_checkpoint 파일이 없음: {resume_path}"
            )
        if agent_cfg.resume:
            raise ValueError(
                "--init_checkpoint 와 --resume 은 같이 쓸 수 없음."
                " 전자는 다른 experiment 에서 가중치만 받아오는 것이고,"
                " 후자는 같은 experiment 의 런을 이어가는 것이라 대상이 다름."
            )
    elif agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if (
        args_cli.init_checkpoint is not None
        or agent_cfg.resume
        or agent_cfg.algorithm.class_name == "Distillation"
    ):
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        load_cfg = None
        if args_cli.load_actor_only:
            load_cfg = {
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            }
        elif args_cli.init_checkpoint is not None:
            # 다른 태스크에서 넘어오는 fine-tuning: actor/critic 가중치는 받되
            # optimizer state 와 iteration 은 새로 시작한다. 보상 구성이 달라져서
            # 옛 Adam moment 와 value 스케일이 새 목적함수와 맞지 않고,
            # iteration 을 이어받으면 로그의 x축이 새 런과 어긋난다.
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            }
        runner.load(resume_path, load_cfg=load_cfg)
        if args_cli.load_actor_only:
            print("[INFO]: Loaded actor only; critic, optimizer, and iteration start fresh.")
        elif args_cli.init_checkpoint is not None:
            print(
                "[INFO]: Fine-tuning from an explicit checkpoint."
                " Loaded actor and critic weights; optimizer state and iteration"
                " counter start fresh."
            )
        # 실제로 로드됐음을 shape 로 확인해 남긴다. load_state_dict 가 strict=True 라
        # obs/action 차원이 다르면 위에서 이미 예외가 났을 것이고, 여기까지 왔다면
        # 두 태스크의 policy interface 가 동일하다는 뜻이다.
        try:
            actor_params = list(runner.alg.actor.parameters())
            print(
                "[INFO]: policy interface verified -"
                f" actor input {actor_params[0].shape[1]},"
                f" actor output {actor_params[-1].shape[0]}"
            )
        except Exception as exc:  # noqa: BLE001 - 로그용, 학습을 막지 않는다
            print(f"[WARN]: could not report policy shapes: {type(exc).__name__}: {exc}")
    if args_cli.reset_policy_std is not None:
        reset_std = float(args_cli.reset_policy_std)
        if reset_std <= 0.0:
            raise ValueError("--reset_policy_std must be positive.")
        actor = runner.alg.actor
        distribution = getattr(actor, "distribution", None)
        if distribution is None:
            raise ValueError("The loaded policy does not expose a Gaussian distribution.")
        with torch.no_grad():
            if hasattr(distribution, "std_param"):
                std_parameter = distribution.std_param
                std_parameter.fill_(reset_std)
            elif hasattr(distribution, "log_std_param"):
                std_parameter = distribution.log_std_param
                std_parameter.fill_(torch.log(torch.tensor(reset_std, device=std_parameter.device)))
            else:
                raise ValueError("The loaded policy distribution has no resettable std parameter.")
        # Adam moments from the old, diverged std would immediately push the
        # reset value back toward it.  Reinitialize only this parameter's state.
        runner.alg.optimizer.state.pop(std_parameter, None)
        print(f"[INFO]: Reset policy std to {reset_std:g} and cleared its optimizer state.")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
