# [tool/Isaac] obs[40:55] 손끝이 배포 FK 와 같은 프레임인지 대조. 정책 불필요, 1스텝만.
"""Check that Isaac's fingertip observation matches the deploy-side FK.

``obs[40:55]`` is solved on the deploy side from joint angles against a URDF.
Isaac produces it from the USD's ``finger*_tip_link`` body poses.  The USD was
imported from ``model/urdf/wuji_right/wuji_right.urdf`` and the deploy FK reads
that same file, so the two SHOULD agree exactly -- but "was imported from" is
not proof, and nothing has ever compared them.

It matters because the two Wuji URDFs in this repository put ``finger1_tip_link``
3.0 mm apart.  Picking the wrong one silently biases the thumb by 23 % of its
whole range of motion (2026-08-21).  This closes the last leg: URDF FK has been
checked against MuJoCo (6.7e-08 m), and against the other URDF; only
"USD == the URDF it came from" is untested.

Run it AFTER training finishes -- it opens Isaac Sim, and running that beside a
4096-env training run has frozen this machine before.

    python scripts/rsl_rl/verify_tip_frames_isaac.py --task hand_real
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="hand_real")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--tolerance_mm", type=float, default=0.05,
                    help="Same-file kinematics should agree far below this.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import isaac_neuromeka.tasks  # noqa: F401  (task registration)
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Deploy.common.fingertip_fk import (
    ISAAC_URDF,
    OFFICIAL_URDF,
    POLICY_TIP_FRAME_URDF,
    WujiHand1FingertipFK,
)
from Deploy.common.policy_contract import OBSERVATION_SLICES, POLICY_JOINT_NAMES


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    observation, _ = env.reset()
    # One physics step: buffers right after reset may not reflect the reset yet
    # (measure_stick_hold_isaac.py hit exactly that).
    observation, *_ = env.step(torch.zeros(env.num_envs, env.action_space.shape[-1],
                                           device=env.device))
    obs = observation["policy"][0].detach().cpu().numpy()

    tips_slice = OBSERVATION_SLICES["fingertips"].slice
    isaac_tips = obs[tips_slice].reshape(5, 3)

    robot = env.scene["robot"]
    joint_ids = [robot.joint_names.index(n) for n in POLICY_JOINT_NAMES]
    q = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().astype(np.float32)

    print("\n" + "=" * 74)
    print(f"obs[{tips_slice.start}:{tips_slice.stop}] vs deploy FK   task={args_cli.task}")
    print("=" * 74)
    print(f"{'finger':10s} {'|Isaac - ISAAC_URDF|':>22s} {'|Isaac - OFFICIAL_URDF|':>25s}")
    print("-" * 74)

    worst = {}
    for label, source in (("isaac", ISAAC_URDF), ("official", OFFICIAL_URDF)):
        tips = WujiHand1FingertipFK(source).fingertip_positions_in_palm(q).reshape(5, 3)
        worst[label] = np.linalg.norm(isaac_tips - tips, axis=1) * 1.0e3

    for finger in range(5):
        print(f"finger{finger + 1:<4d} {worst['isaac'][finger]:19.4f} mm "
              f"{worst['official'][finger]:22.4f} mm")
    print("-" * 74)
    print(f"{'max':10s} {worst['isaac'].max():19.4f} mm "
          f"{worst['official'].max():22.4f} mm")

    ok = float(worst["isaac"].max()) <= args_cli.tolerance_mm
    print()
    print(f"POLICY_TIP_FRAME_URDF = {POLICY_TIP_FRAME_URDF.label}")
    if ok:
        print(f"PASS -- USD 가 임포트 원본 URDF 와 일치합니다 "
              f"(<= {args_cli.tolerance_mm} mm). obs[40:55] 배선이 맞습니다.")
    else:
        print(f"FAIL -- USD 와 로컬 URDF 가 {worst['isaac'].max():.4f} mm 어긋납니다.")
        print("  같은 파일에서 나온 기하가 이만큼 다를 이유가 없습니다.")
        print("  USD 가 다른 URDF 에서 임포트됐거나 이후에 편집됐다는 뜻입니다.")
        print("  official 쪽 수치가 더 작다면 POLICY_TIP_FRAME_URDF 선택이 틀린 것입니다.")

    env.close()
    simulation_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
