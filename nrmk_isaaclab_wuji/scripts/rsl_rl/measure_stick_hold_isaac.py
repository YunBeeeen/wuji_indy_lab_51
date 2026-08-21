"""Isaac 에서 pregrasp 자세를 고정하고 젓가락이 얼마나 미끄러지는지만 잰다.

PLAY ONLY, 정책 없음, 체크포인트 불필요.  학습 경로는 건드리지 않는다.

MuJoCo 의 ``run_policy.py --hold-pose`` 와 **같은 조건**을 만드는 것이 목적이다.

    리셋 -> 관절 목표를 pregrasp 에 고정 -> N 초 유지 -> 스틱 변위 기록

정책도, 잔차도, 액션 매니저도 쓰지 않는다.  ``env.step()`` 을 부르면 액션
매니저가 목표를 덮어쓰므로 여기서는 물리를 직접 돌린다:

    robot.set_joint_position_target(target)
    robot.write_data_to_sim()
    sim.step()
    scene.update(dt)

왜 이 측정이 필요한가: MuJoCo 에서 Stick2 는 5 초에 떨어지고 Stick1 은 10 초
동안 47.7 mm 를 계속 미끄러진다.  그런데 **몇 mm 면 정상인지 기준이 없다.**
같은 자세에서 Isaac 이 몇 mm 인지 알아야 MuJoCo 를 무엇에 맞출지 정해진다.
Isaac 이 0 mm 에 가까우면 두 시뮬의 접촉/마찰 차이를 좁히면 되고, Isaac 도
비슷하게 미끄러지면 이 자세 자체가 원래 불안정한 것이므로 방향이 완전히 달라진다.

예:
    python scripts/rsl_rl/measure_stick_hold_isaac.py --headless
    python scripts/rsl_rl/measure_stick_hold_isaac.py --seconds 10 --task hand_grasp
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="hand_real",
                    help="스틱과 pregrasp 리셋을 가진 태스크.")
parser.add_argument("--seconds", type=float, default=10.0,
                    help="목표를 유지하는 시간. MuJoCo 쪽과 맞출 것.")
parser.add_argument("--report-hz", type=float, default=1.0,
                    help="몇 초마다 한 줄 찍을지.")
parser.add_argument("--csv", type=str, default=None,
                    help="주면 정책 스텝(30 Hz)마다 한 줄씩 기록한다.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- 아래는 simulation app 이 있어야 import 된다 -----------------------------
import csv as csv_module  # noqa: E402
import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401,E402
from isaac_neuromeka.tasks.manipulation.hand_grasp.hand_grasp_env_cfg import (  # noqa: E402
    HAND_JOINT_NAMES,
    PREGRASP_JOINT_POSITIONS,
)

PALM_BODY_NAME = "palm_link"
STICK_NAMES = ("stick1", "stick2")


def stick_poses_in_palm(scene, palm_index):
    """스틱 중심을 palm 프레임 xyz 로.  MuJoCo 의 get_stick_poses_in_palm 과 같은 양."""

    robot = scene["robot"]
    palm_position = robot.data.body_link_pos_w[0, palm_index]
    palm_quaternion = robot.data.body_link_quat_w[0, palm_index]

    out = []
    for name in STICK_NAMES:
        world = scene[name].data.root_link_pos_w[0]
        out.append(quat_apply_inverse(palm_quaternion, world - palm_position))
    return torch.stack(out)


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # 에피소드가 도중에 끝나 리셋이 끼면 측정이 끊긴다.
    env_cfg.episode_length_s = args_cli.seconds + 60.0

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    # 떠 있는 루트를 쓰는 태스크(hand_move/hand_real/hand_object/hand_final)는
    # HandRootHoldAction 이 루트를 붙잡는다.  이 스크립트는 env.step() 을
    # 우회하므로 그 액션 term 이 실행되지 않아 손 전체가 떠내려간다.
    # 그러면 palm 프레임 기준 변위가 손의 운동에 오염된다.
    if not bool(getattr(env.cfg.scene.robot.spawn.articulation_props, "fix_root_link", True)):
        print("  [경고] 이 태스크는 루트가 떠 있고, 루트를 잡는 것은 액션 term 이다.")
        print("         이 스크립트는 env.step() 을 쓰지 않아 그 term 이 돌지 않는다.")
        print("         손 전체가 떠내려가므로 변위 숫자를 믿지 말 것.")
        print("         팜이 고정된 --task hand_grasp 로 재는 것이 MuJoCo 와 같은 조건이다.")

    scene = env.scene
    robot = scene["robot"]
    sim = env.sim

    joint_ids, joint_names = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
    if list(joint_names) != list(HAND_JOINT_NAMES):
        raise ValueError(
            "손 관절 순서가 HAND_JOINT_NAMES 와 다르다:\n"
            f"  기대 {list(HAND_JOINT_NAMES)}\n  실제 {list(joint_names)}"
        )
    palm_index = robot.find_bodies(PALM_BODY_NAME)[0][0]

    # 관절 한계로 clamp.  MuJoCo 쪽도 COMMAND_TARGET_LIMITS 로 clamp 한 뒤 넣는다.
    limits = robot.data.soft_joint_pos_limits[0, joint_ids]
    target = torch.tensor(PREGRASP_JOINT_POSITIONS, device=env.device, dtype=torch.float32)
    target = torch.clamp(target, limits[:, 0], limits[:, 1]).unsqueeze(0)

    physics_dt = float(sim.get_physics_dt())
    total_steps = int(round(args_cli.seconds / physics_dt))
    report_every = max(1, int(round(args_cli.report_hz / physics_dt)))
    sample_every = max(1, int(round((1.0 / 30.0) / physics_dt)))

    # 기준점은 물리를 한 스텝 돌린 뒤에 잡는다.
    # env.reset() 직후의 data 버퍼는 아직 리셋 값을 반영하지 않은 경우가 있고,
    # 그걸 기준으로 삼으면 "움직이지 않는데 수백 mm 어긋난" 상수가 나온다.
    # 2026-08-20 실제로 그렇게 읽었다: Stick1 이 554.2 mm 로 10 초 내내 상수.
    robot.set_joint_position_target(target, joint_ids=joint_ids)
    robot.write_data_to_sim()
    sim.step(render=not args_cli.headless)
    scene.update(physics_dt)

    start = stick_poses_in_palm(scene, palm_index).clone()
    joints_at_start = robot.data.joint_pos[0, joint_ids].clone()

    # 기준점이 맞는지 눈으로 확인할 수 있게 절대 위치도 찍는다.
    print("  기준점 (물리 1 스텝 후, palm 프레임 mm)")
    for name, row in zip(STICK_NAMES, start):
        print(f"    {name}: {np.round(row.detach().cpu().numpy() * 1000.0, 2)}")

    print("[HOLD POSE - ISAAC]", flush=True)
    print(f"  task {args_cli.task}   physics dt {physics_dt:.6f} s"
          f"   {total_steps} steps = {args_cli.seconds:.1f} s")
    print("  목표는 pregrasp 에 고정. 액션 매니저/정책은 쓰지 않는다.")
    print(f"  {'t[s]':>6}{'Stick1':>12}{'Stick2':>12}{'관절 이탈':>12}", flush=True)

    writer = handle = None
    if args_cli.csv:
        handle = open(args_cli.csv, "w", newline="", encoding="utf-8")
        writer = csv_module.writer(handle)
        writer.writerow(["time_s", "stick1_mm", "stick2_mm", "joint_drift_mrad"])

    try:
        for step in range(total_steps):
            robot.set_joint_position_target(target, joint_ids=joint_ids)
            robot.write_data_to_sim()
            sim.step(render=not args_cli.headless)
            scene.update(physics_dt)

            if (step + 1) % sample_every and (step + 1) % report_every:
                continue

            now = stick_poses_in_palm(scene, palm_index)
            moved = torch.linalg.norm(now - start, dim=1) * 1000.0
            drift = (robot.data.joint_pos[0, joint_ids] - joints_at_start).abs().max() * 1000.0
            seconds = (step + 1) * physics_dt

            if writer is not None and not (step + 1) % sample_every:
                writer.writerow([f"{seconds:.6f}",
                                 f"{float(moved[0]):.4f}", f"{float(moved[1]):.4f}",
                                 f"{float(drift):.4f}"])

            if not (step + 1) % report_every:
                print(f"  {seconds:6.2f}{float(moved[0]):9.1f}mm"
                      f"{float(moved[1]):9.1f}mm{float(drift):9.1f}mrad", flush=True)

        final = stick_poses_in_palm(scene, palm_index)
        worst = float((torch.linalg.norm(final - start, dim=1) * 1000.0).max())
        print(f"\n  최대 이동 {worst:.1f} mm")
        print("  MuJoCo 같은 조건 실측: Stick1 47.7 mm / Stick2 208.0 mm (5초에 낙하)")
        print("  두 값을 나란히 놓고 무엇을 맞출지 정한다.")
        return 0
    finally:
        if handle is not None:
            handle.close()
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        # simulation_app.close() 가 예외를 삼켜서 아무 흔적 없이 죽는 이력이 있다.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
