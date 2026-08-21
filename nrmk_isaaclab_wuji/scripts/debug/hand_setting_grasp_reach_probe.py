"""hand_setting grasp-reachability probe (2026-08-04).

**정책과 동일한 잔차 액션**(target = 현재각 + action*scale, action∈[-1,1])으로 손을 open에서
PREGRASP(=잡은 자세) 쪽으로 구동하며, 엄지 tip이 grasp 기준점 대비 어디로 가는지 + PREGRASP
도달 여부(best|q-PREGRASP|) + 리셋(스틱 낙하 등)을 출력한다. 보상·gate는 무시.

목적: "정책이 낼 수 있는 잔차 힘(kp*scale)으로 grasp 자세·조작이 되는가"를 학습과 분리해 검증.
  (※ 직접 절대목표는 effort까지 밀어 정책보다 세므로 안 씀 — 잔차로 공정하게.)
  - best|q-PREGRASP|≈0 + 리셋 0 → 정책 힘으로 자세 도달·유지 가능 → 문제는 보상/학습.
  - best|q-PREGRASP| 큼 → 잔차 힘으로 못 감(충돌/힘부족). 리셋↑ → 잡기 전 스틱 낙하.

실행 (학습 안 도는 동안):
    python scripts/debug/hand_setting_grasp_reach_probe.py --num_envs 1
    (--headless 로 GUI 생략 — 눈으로 보려면 빼고 실행)
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="hand_setting")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--ramp_steps", type=int, default=150, help="open→PREGRASP 램프 물리스텝 수")
parser.add_argument("--settle_steps", type=int, default=80, help="PREGRASP 유지 물리스텝 수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402
from isaac_neuromeka.tasks.manipulation.hand_grasp.hand_grasp_env_cfg import (  # noqa: E402
    HAND_JOINT_NAMES,
    PREGRASP_JOINT_POSITIONS,
    PREGRASP_STICK1_POSITION_P,
    PREGRASP_STICK2_POSITION_P,
)


def fmt(v):
    return [round(float(x), 3) for x in v]


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    device = env.device
    robot = env.scene["robot"]
    stick1 = env.scene["stick1"]
    stick2 = env.scene["stick2"]

    tip_id = robot.find_bodies(["finger1_tip_link"])[0][0]
    palm_id = robot.find_bodies(["palm_link"])[0][0]
    hand_ids, _ = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
    thumb_ids, _ = robot.find_joints(
        ["finger1_joint1", "finger1_joint2", "finger1_joint3", "finger1_joint4"],
        preserve_order=True,
    )

    pregrasp = (
        torch.tensor(PREGRASP_JOINT_POSITIONS, device=device, dtype=torch.float)
        .unsqueeze(0)
        .repeat(args_cli.num_envs, 1)
    )
    # grasp 기준점(palm-local): 엄지는 stick1을 잡으므로 stick1 reference가 주 비교 대상.
    ref1 = torch.tensor(PREGRASP_STICK1_POSITION_P, device=device, dtype=torch.float)
    ref2 = torch.tensor(PREGRASP_STICK2_POSITION_P, device=device, dtype=torch.float)

    print("scene sensors:", list(env.scene.sensors.keys()))

    def report(tag):
        tip = robot.data.body_pos_w[:, tip_id]
        palm_p = robot.data.body_pos_w[:, palm_id]
        palm_q = robot.data.body_quat_w[:, palm_id]
        tip_l = quat_apply_inverse(palm_q, tip - palm_p)
        s1_l = quat_apply_inverse(palm_q, stick1.data.root_pos_w - palm_p)
        s2_l = quat_apply_inverse(palm_q, stick2.data.root_pos_w - palm_p)
        tq = robot.data.joint_pos[:, thumb_ids]
        e = 0
        print(f"\n[{tag}]")
        print(f"  thumb q(j1..4)      = {fmt(tq[e])}   (opposition = j1 ↑ / grasp target j1={round(float(pregrasp[e,0]),3)})")
        print(f"  thumb_tip (palm)    = {fmt(tip_l[e])}")
        print(f"  stick1 REF (palm)   = {fmt(ref1)}   ← 엄지가 가야 할 grasp 지점")
        print(f"  stick2 REF (palm)   = {fmt(ref2)}")
        print(f"  tip - stick1REF     = {fmt(tip_l[e] - ref1)}   |d|={float((tip_l[e]-ref1).norm()):.3f}")
        print(f"  live stick1 (palm)  = {fmt(s1_l[e])}   (REF서 멀면 스틱이 떨어진 것)")
        print(f"  live stick2 (palm)  = {fmt(s2_l[e])}")

    report("RESET (open hand)")

    # 정책과 동일한 잔차 액션으로 구동: target = 현재각 + action*scale, action ∈ [-1,1].
    # (직접 절대목표는 effort까지 밀어 정책보다 세므로 공정하지 않음 — 잔차 힘 kp*scale로 봄.)
    scale = 0.1  # HandSettingActionsCfg 잔차 scale (uniform)

    def drive(steps, tag):
        resets = 0
        best_q_err = float("inf")
        for step in range(steps):
            current = robot.data.joint_pos[:, hand_ids]
            action = torch.clamp((pregrasp - current) / scale, -1.0, 1.0)
            _, _, terminated, truncated, _ = env.step(action)
            q_err = float((robot.data.joint_pos[:, hand_ids] - pregrasp).abs().max())
            best_q_err = min(best_q_err, q_err)
            if bool((terminated | truncated).any()):
                resets += 1
        print(f"  [{tag}] best|q-PREGRASP|={best_q_err:.3f}  resets(스틱낙하/timeout 등)={resets}")

    drive(args_cli.ramp_steps, "ramp(잔차)")
    report("after ramp (residual action)")
    drive(args_cli.settle_steps, "settle(잔차)")
    report("after settle (residual action)")

    reached = robot.data.joint_pos[:, hand_ids]
    err = (reached - pregrasp).abs()
    j1_err = (robot.data.joint_pos[:, thumb_ids[0]] - pregrasp[:, 0]).abs()
    print(f"\n  |q - PREGRASP| max = {float(err.max()):.3f}   (≈0 잡은 자세 도달, 크면 물리가 막음)")
    print(f"  thumb j1 err       = {float(j1_err.mean()):.3f}   (opposition 도달 여부)")
    print("\n판정 (잔차=정책 힘 기준):")
    print("  · best|q-PREGRASP|≈0 + 리셋 0 → 정책 잔차 힘으로 자세 도달·유지 O → 문제는 보상/학습.")
    print("  · best|q-PREGRASP| 큼 → 잔차 힘(kp*scale)으로 자세 못 감(충돌/힘부족) → kp나 scale 필요.")
    print("  · 리셋(스틱 낙하)↑ → 잔차 그립이 약해 잡기 전에 스틱을 놓침 → 힘 문제.")
    print("  · live stick이 REF서 멀면 → 스틱을 reference로 못 옮김(조작 힘 부족).")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
