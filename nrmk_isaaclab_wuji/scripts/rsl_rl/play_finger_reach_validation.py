"""Replay a deterministic finger-reach scenario in Isaac and log it for comparison.

PLAY ONLY.  Nothing here touches the training path: the environment config, the
command term and ``play.py`` are all left alone.  ``FingerReachCommandsCfg`` sets
``resampling_time_range`` to 1e9, so the command term never resamples on its
own after the initial reset -- which means this script can simply own the target
buffers and drive them from a file.

Why this exists: in training every episode draws a random target, so an Isaac
run and a MuJoCo run aim at different points and their errors cannot be laid
side by side.  Feeding both the same target sequence makes any remaining
difference attributable to the simulator rather than to the command.

The CSV columns are identical to ``Deploy.run.run_finger_reach``, so the two
files can be compared column by column.

Example:
    python scripts/rsl_rl/play_finger_reach_validation.py \
        --load_run 2026-08-18_15-06-02 --checkpoint model_500.pt \
        --scenario ../Deploy/validation_scenario.json \
        --csv reach_isaac.csv --headless
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", type=str, required=True,
                    help="validation_scenario.json shared with the MuJoCo backend.")
parser.add_argument("--csv", type=str, default="reach_isaac.csv")
parser.add_argument("--task", type=str, default="finger_reach")
parser.add_argument("--load_run", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default="model_500.pt")
parser.add_argument("--target-duration", type=float, default=None,
                    help="Override the scenario dwell. Must match the MuJoCo run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- everything below needs the simulation app to exist ---------------------
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
from importlib import metadata  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401,E402  (registers "finger_reach")
from isaac_neuromeka.tasks.manipulation.hand_grasp.finger_reach_env_cfg import (  # noqa: E402
    MIDDLE_JOINT_NAMES,
    PALM_BODY_NAME,
    REACH_TIP_LINK,
)

CSV_COLUMNS = (
    ["time", "target_index", "target_palm_x", "target_palm_y", "target_palm_z"]
    + [f"q_prev_{k}" for k in range(1, 5)]
    + [f"q_curr_{k}" for k in range(1, 5)]
    + [f"action_{k}" for k in range(1, 5)]
    + [f"q_target_{k}" for k in range(1, 5)]
    + ["tip_palm_x", "tip_palm_y", "tip_palm_z"]
    + ["error_palm_x", "error_palm_y", "error_palm_z", "error_norm"]
)


def main() -> int:
    scenario = json.loads(Path(args_cli.scenario).read_text())
    controlled = scenario.get("controlled_joints")
    if controlled is not None and list(controlled) != list(MIDDLE_JOINT_NAMES):
        raise ValueError(
            f"Scenario controls {controlled}; this task controls {list(MIDDLE_JOINT_NAMES)}."
        )
    targets = [np.asarray(t, dtype=np.float32) for t in scenario["targets_palm_m"]]
    duration = float(args_cli.target_duration or scenario["target_duration_s"])

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # One continuous rollout: an auto reset mid-scenario would teleport the hand
    # and reseed the observation history, which is exactly what a real hand
    # cannot do and what makes the two logs incomparable.
    total_seconds = duration * len(targets)
    env_cfg.episode_length_s = total_seconds + 10.0

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    # This project's runner cfgs are written against the OLD rsl_rl API (a
    # single ``policy`` block).  The installed rsl_rl expects separate
    # ``actor``/``critic`` blocks each carrying ``class_name``, and
    # OnPolicyRunner fails with KeyError: 'class_name' without the translation.
    # train.py and play.py both apply this shim; so must anything that builds a
    # runner by hand.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.load_run = args_cli.load_run
    agent_cfg.load_checkpoint = args_cli.checkpoint

    log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
    if not log_root.is_dir():
        raise FileNotFoundError(
            f"{log_root.resolve()} does not exist. Run this from the "
            "nrmk_isaaclab_wuji directory: the checkpoint root is relative."
        )
    # get_checkpoint_path treats both arguments as regexes and picks the newest
    # match, which silently loads a different run when a name is mistyped.  Both
    # are explicit here, so resolve them directly and say exactly what is
    # available when the file is not there.
    direct = log_root / args_cli.load_run / args_cli.checkpoint
    if direct.is_file():
        resume_path = str(direct)
    else:
        available_runs = sorted(d.name for d in log_root.iterdir() if d.is_dir())
        run_dir = log_root / args_cli.load_run
        detail = (
            f"runs under {log_root}: {available_runs}"
            if not run_dir.is_dir()
            else f"checkpoints in {run_dir}: "
                 f"{sorted(f.name for f in run_dir.glob('*.pt'))}"
        )
        raise FileNotFoundError(f"{direct} not found.\n  {detail}")
    print(f"[CHECKPOINT] {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    device = unwrapped.device
    command_term = unwrapped.command_manager.get_term("finger_target")
    action_term = unwrapped.action_manager.get_term("hand_action")
    middle_ids = robot.find_joints(list(MIDDLE_JOINT_NAMES), preserve_order=True)[0]
    palm_id = robot.find_bodies(PALM_BODY_NAME)[0][0]
    tip_id = robot.find_bodies(REACH_TIP_LINK)[0][0]

    print(f"[CONTRACT] middle joints {list(MIDDLE_JOINT_NAMES)} -> ids {middle_ids}")
    print(f"[TIMING]   policy {1.0 / (unwrapped.step_dt):.1f} Hz, "
          f"sim dt {unwrapped.physics_dt:.6f}s, decimation {env_cfg.decimation}")
    print(f"[SCENARIO] {len(targets)} targets x {duration:.2f}s = {total_seconds:.2f}s, "
          f"episode_length_s={env_cfg.episode_length_s}")

    def install_target(index: int) -> None:
        """Own the command buffers directly instead of letting them resample."""

        palm_target = torch.tensor(
            targets[min(index, len(targets) - 1)], dtype=torch.float32, device=device
        ).unsqueeze(0)
        palm_pos = robot.data.body_pos_w[:, palm_id]
        palm_quat = robot.data.body_quat_w[:, palm_id]
        target_w = palm_pos + quat_apply(palm_quat, palm_target)
        command_term.target_palm[:] = palm_target
        command_term.target_e[:] = target_w - unwrapped.scene.env_origins

    steps_per_target = max(1, int(round(duration / unwrapped.step_dt)))
    total_steps = steps_per_target * len(targets)

    # The wrapper already reset during construction, and reset() lets the command
    # term resample a random target.  So: reset, overwrite the target, then
    # recompute the observation.  get_observations() calls
    # ObservationManager.compute(update_history=False), so this refresh does NOT
    # advance the two-sample history -- both slots still hold the reset pose,
    # exactly as MuJoCo's adapter.reset() leaves them.
    env.reset()
    install_target(0)
    obs = env.get_observations()

    rows = []
    q_previous = robot.data.joint_pos[0, middle_ids].detach().cpu().numpy().copy()
    with torch.inference_mode():
        for step in range(total_steps):
            target_index = step // steps_per_target
            target = targets[target_index]

            action = policy(obs)

            # Install the NEXT step's target before stepping, because the
            # observation this step returns is what the next action will see.
            # MuJoCo's runner has no such lag, so aligning here keeps the two
            # logs sample-for-sample comparable.
            install_target((step + 1) // steps_per_target)
            obs, _, _, _ = env.step(action)

            q_current = robot.data.joint_pos[0, middle_ids].detach().cpu().numpy().copy()
            q_target = action_term.joint_pos_target[0].detach().cpu().numpy().copy()
            # Log the POST-clip action ActionManager actually accepted, not the
            # network's raw output.  RslRlVecEnvWrapper clips to [-1,1] inside
            # step(), so policy(obs) can exceed the range -- MuJoCo logs the
            # clipped value, and the contract's obs[11:15] is the clipped one
            # too.  Logging the raw output here made an identical policy look
            # like a 1.7-unit action mismatch.
            raw_action = (
                unwrapped.action_manager.action[0].detach().cpu().numpy().copy()
            )

            palm_pos = robot.data.body_pos_w[:, palm_id]
            palm_quat = robot.data.body_quat_w[:, palm_id]
            tip = quat_apply_inverse(
                palm_quat, robot.data.body_pos_w[:, tip_id] - palm_pos
            )[0].detach().cpu().numpy()
            error = target - tip

            rows.append(
                [(step + 1) * unwrapped.step_dt, target_index, *target]
                + list(q_previous) + list(q_current)
                + list(raw_action) + list(q_target)
                + list(tip) + list(error) + [float(np.linalg.norm(error))]
            )
            q_previous = q_current

            if step % 30 == 0 or step == total_steps - 1:
                print(f"step={step + 1:04d} t={(step + 1) * unwrapped.step_dt:6.3f}s "
                      f"tgt#{target_index} q=[{q_current.min():+.3f},{q_current.max():+.3f}] "
                      f"err={np.linalg.norm(error) * 1000:7.2f}mm")

    destination = Path(args_cli.csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {destination}")

    env.close()
    return 0


if __name__ == "__main__":
    # ``simulation_app.close()`` tears the process down hard, which swallows any
    # traceback raised from main() if it runs in a bare finally.  Print the
    # traceback and flush BEFORE closing, otherwise a failure looks like the
    # script simply exiting with no output at all.
    import sys
    import traceback

    exit_code = 1
    try:
        exit_code = main() or 0
    except BaseException:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    sys.exit(exit_code)
