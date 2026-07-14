"""이 손이 큐브를 '들 수 있는 손'인가만 측정한다. 정책/보상/접근은 전부 뺀다.

큐브를 cage 중심에 고정한 채 손가락을 천천히 오므리고(= 완벽하게 쥐여주기), 놓아주고, 팔을 든다.
env마다 큐브 질량을 다르게 줘서 "버틸 수 있는 최대 질량"을 한 번에 뽑는다.

못 들면 정책이 아니라 액추에이터(effort_limit)/질량/마찰이 원인이므로,
시작 자세를 아무리 바꿔도 cube_lift는 계속 0이다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Indy-Wuji-Cube-Grasp")
parser.add_argument("--close_frac", type=float, default=0.50,
                    help="오므림 목표 = 관절상한 x 이 비율. 큐브 접촉은 0.30 부근이라 0.50이면 꽉 쥠")
parser.add_argument("--lift_rad", type=float, default=0.5, help="팔을 들어올릴 joint1 변화량(rad)")
parser.add_argument("--out_cm", type=float, default=2.5,
                    help="배치를 손바닥->손끝 방향으로 미는 오프셋. tip_link 원점이 패드보다 2~3cm 안쪽이라 보정")
parser.add_argument("--gui", action="store_true",
                    help="GUI로 보면서 실행 (env 1개, 느린 오므림). 큐브가 어디로 왜 빠지는지 눈으로 확인용")
parser.add_argument("--vendor_gains", action="store_true",
                    help="제조사 right.xml의 kp/kd로 교체 (kp 2/2/1/0.8, kd 0.05). 현 sim은 그 10~25배")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = not args_cli.gui

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaac_neuromeka.tasks  # noqa: F401
from isaac_neuromeka.mdp.rewards import cage_points
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

# 오므림량 스윕. 간격=큐브(6cm)는 30% 지점. 50%는 지나쳐 조여서 큐브를 짜냈다.
# 질량은 학습 설정 그대로 0.30 kg 고정.
FRACS = [0.32, 0.34, 0.36, 0.38, 0.40, 0.44, 0.50, 0.60]
MASS = 0.30
REPEAT = 4  # 값당 env 수 (접촉 난수 평균)
NUM_ENVS = len(FRACS) * REPEAT

import sys
if "--gui" in sys.argv:  # GUI: env 1개만. 0.40 = 스윕에서 4/4 성공한 오므림량
    FRACS, REPEAT, NUM_ENVS = [0.40], 1, 1

P = lambda *a: print(*a, flush=True)  # noqa: E731


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=NUM_ENVS)
    env_cfg.episode_length_s = 1e9  # RL step을 안 쓰지만 혹시 모를 리셋 차단
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    scene, sim = env.scene, env.sim
    robot, cube = scene["robot"], scene["cube"]
    dt = sim.get_physics_dt()

    env.reset()

    # ── 질량 주입 ────────────────────────────────────────────────
    masses = torch.full((NUM_ENVS, 1), MASS)
    cube.root_physx_view.set_masses(masses, torch.arange(NUM_ENVS))
    fracs = torch.tensor(FRACS, dtype=torch.float32).repeat_interleave(REPEAT).to(env.device)
    P(f"\n오므림량 스윕: {FRACS}  (질량 {MASS}kg 고정, 각 {REPEAT}개 env)\n")

    # ── 관절 인덱스 ──────────────────────────────────────────────
    fing_cfg = SceneEntityCfg("robot", joint_names=["finger[1-3]_joint[1-4]"])
    arm_cfg = SceneEntityCfg("robot", joint_names=["joint[0-5]"])
    palm_cfg = SceneEntityCfg("robot", body_names=["palm_link"])
    for c in (fing_cfg, arm_cfg, palm_cfg):
        c.resolve(scene)
    fing_ids, arm_ids, palm_id = fing_cfg.joint_ids, arm_cfg.joint_ids, palm_cfg.body_ids[0]

    # 제조사 gain 적용 (right.xml position actuator: kp 2/2/1/0.8, joint damping 0.05)
    if args_cli.vendor_gains:
        KP = {"joint1": 2.0, "joint2": 2.0, "joint3": 1.0, "joint4": 0.8}
        kp = torch.zeros(NUM_ENVS, len(fing_ids), device=env.device)
        for k, j in enumerate(fing_ids):
            kp[:, k] = KP[robot.joint_names[j].split("_")[1]]
        robot.write_joint_stiffness_to_sim(kp, joint_ids=fing_ids)
        robot.write_joint_damping_to_sim(torch.full_like(kp, 0.05), joint_ids=fing_ids)
        P("★ 제조사 gain 적용: kp 2/2/1/0.8, kd 0.05 (기존 20/0.5)")

    lim = robot.data.joint_pos_limits[0]
    P("오므림 방향 확인 (finger 관절 limit):")
    for i in fing_ids[:4]:
        P(f"  {robot.joint_names[i]:22s} [{lim[i,0]:+.2f}, {lim[i,1]:+.2f}]")

    cage_cfg = SceneEntityCfg("robot", body_names=[
        "finger1_tip_link", "finger2_tip_link", "finger2_link3",
        "finger3_tip_link", "finger3_link3"], preserve_order=True)
    cage_cfg.resolve(scene)

    target = robot.data.default_joint_pos.clone()
    hold_pose = target.clone()  # 팔은 리셋 자세 유지

    # ── 사전탐색: 큐브가 '실제로 들어갈 수 있는' 지점을 찾는다 ───
    # cage 점들의 평균을 쓰면 링크 내부라서 큐브가 관통해 폭발한다 (step0에 속도 1.3 m/s).
    # hand_geometry.py 측정: 엄지-중지 간격이 오므림 30%에서 5.8cm = 큐브(6cm)와 같아짐.
    # 그 지점의 '엄지끝-중지끝 중점'이 큐브가 딱 물리는 빈 공간이다.
    tip_cfg = SceneEntityCfg("robot", body_names=["finger1_tip_link", "finger3_tip_link"],
                             preserve_order=True)
    tip_cfg.resolve(scene)
    thumb_id, mid_id = tip_cfg.body_ids

    away = cube.data.default_root_state.clone()
    away[:, 2] = -5.0  # 사전탐색 동안 큐브를 치운다
    cube.write_root_state_to_sim(away)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt)

    FIT = 0.30  # 간격이 큐브 크기와 같아지는 오므림 비율
    probe = hold_pose.clone()
    probe[:, fing_ids] = lim[fing_ids, 1] * FIT  # 상한(최대굴곡)의 30%
    robot.write_joint_state_to_sim(probe, torch.zeros_like(probe))
    robot.set_joint_position_target(probe)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt)

    thumb, mid = robot.data.body_pos_w[:, thumb_id], robot.data.body_pos_w[:, mid_id]
    palm_now = robot.data.body_pos_w[:, palm_id]
    tip_mid = 0.5 * (thumb + mid)
    # tip_link 원점은 패드보다 2~3cm 손바닥 쪽 -> 원점 중점은 손바닥에 치우침(전에 큐브가
    # 손바닥/새끼에 얹힌 원인). 손바닥->중점 방향으로 밀어 진짜 집게 주머니에 놓는다.
    out_dir = tip_mid - palm_now
    out_dir = out_dir / torch.clamp(torch.norm(out_dir, dim=-1, keepdim=True), min=1e-6)
    grip_center = tip_mid + args_cli.out_cm * 0.01 * out_dir
    P(f"\n파지 지점: 오므림 {FIT:.0%}에서 엄지끝-중지끝 간격 "
      f"{torch.norm(thumb - mid, dim=-1).mean()*100:.1f} cm (큐브 6.0cm)")

    # 손을 다시 편 상태로 (큐브를 넣을 공간 확보)
    robot.write_joint_state_to_sim(hold_pose, torch.zeros_like(hold_pose))
    robot.set_joint_position_target(hold_pose)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt)
    target = hold_pose.clone()

    N_CLOSE, N_SETTLE, N_LIFT = 90, 60, 150  # 1.5s 오므림 / 1.0s 버티기 / 2.5s 들기
    if args_cli.gui:  # 눈으로 따라갈 수 있게 4배 느리게
        N_CLOSE, N_SETTLE, N_LIFT = 360, 240, 600

    # ── 큐브를 파지 지점에 '한 번만' 놓고 중력을 끈다 ────────────
    # 이전 버전은 매 step 위치를 강제로 되돌렸는데, 손가락이 밀어낸 만큼 관통이 누적되어
    # PhysX가 손가락을 관절 한계 밖으로 폭발시켰다 (오차 1.5 rad = 기하적으로 불가능한 값).
    # 중력만 꺼두면 큐브가 제자리에 떠 있고, 손가락은 자연스러운 접촉으로 감싼다.
    cube_state = cube.data.default_root_state.clone()
    cube_state[:, :3] = grip_center
    cube_state[:, 3:] = 0.0
    cube_state[:, 3] = 1.0  # quat w
    cube.write_root_state_to_sim(cube_state)

    all_idx = torch.arange(NUM_ENVS)
    cube.root_physx_view.set_disable_gravities(
        torch.ones(NUM_ENVS, dtype=torch.uint8), all_idx)

    z0 = grip_center[:, 2].clone()
    released_z = None
    zero_vel = torch.zeros((NUM_ENVS, 6), device=env.device)

    for step in range(N_CLOSE + N_SETTLE + N_LIFT):
        # 손가락: 0 -> close_target 으로 선형 오므림
        a = min(step / N_CLOSE, 1.0) * fracs  # env마다 다른 오므림량
        target[:, fing_ids] = lim[fing_ids, 1].unsqueeze(0) * a.unsqueeze(1)

        # 팔: settle 이후 joint1을 천천히 돌려 손을 든다.
        # ★ 부호 주의: joint1 감소(-0.45 -> -0.95)는 GUI 실측 결과 손이 '내려가는' 방향이었음
        if step >= N_CLOSE + N_SETTLE:
            b = (step - N_CLOSE - N_SETTLE) / N_LIFT
            target[:, arm_ids[1]] = hold_pose[:, arm_ids[1]] + b * args_cli.lift_rad

        robot.set_joint_position_target(target)

        # 오므리는 동안: 속도만 매 step 0으로 (밀려도 튀지 않게. 위치는 물리에 맡겨 관통 없음).
        # 다 오므리면: 중력을 되돌리고 완전히 놓아준다.
        if step < N_CLOSE:
            cube.write_root_velocity_to_sim(zero_vel)
        elif released_z is None:
            cube.root_physx_view.set_disable_gravities(
                torch.zeros(NUM_ENVS, dtype=torch.uint8), all_idx)
            released_z = cube.data.root_pos_w[:, 2].clone()
            drift = torch.norm(cube.data.root_pos_w - grip_center, dim=-1)
            P(f"\n[{step:3d}] 중력 복원 + 놓음. 오므림 동안 밀린 거리 평균 {drift.mean()*100:.1f}cm "
              f"(3cm 넘으면 손 밖으로 빠진 것)\n")

        scene.write_data_to_sim()
        sim.step(render=args_cli.gui)
        scene.update(dt)

        if step % 30 == 0 or step == N_CLOSE + N_SETTLE + N_LIFT - 1:
            cz = cube.data.root_pos_w[:, 2]
            cv = torch.norm(cube.data.root_lin_vel_w, dim=-1)
            err = (target[:, fing_ids] - robot.data.joint_pos[:, fing_ids]).abs().max()
            tq = robot.data.applied_torque[:, fing_ids].abs().max()
            phase = "오므림" if step < N_CLOSE else ("버티기" if step < N_CLOSE + N_SETTLE else "들기")
            pz = robot.data.body_pos_w[:, palm_id, 2]
            P(f"[{step:3d}] {phase:6s} 큐브z={cz.mean()*100:5.1f}cm 손z={pz.mean()*100:5.1f}cm "
              f"속도={cv.mean():4.2f} 손가락오차={err:.3f} 최대토크={tq:.3f}/0.6")

    # ── 관절별 실측: 오차 1.5가 기하적으로 불가능해서 인덱싱을 검증한다 ──
    P("\n═══ 손가락 관절별 (env0, 마지막 step) ═══")
    P(f"{'관절':>22} {'목표':>7} {'실제':>7} {'오차':>7} {'하한':>7} {'상한':>7} {'토크':>7}")
    for j in fing_ids:
        t, q = target[0, j].item(), robot.data.joint_pos[0, j].item()
        P(f"{robot.joint_names[j]:>22} {t:>7.3f} {q:>7.3f} {t-q:>7.3f} "
          f"{lim[j,0]:>7.2f} {lim[j,1]:>7.2f} {robot.data.applied_torque[0,j]:>7.3f}")

    # 큐브가 손끝 사이에 실제로 있는가?
    th, md = robot.data.body_pos_w[:, thumb_id], robot.data.body_pos_w[:, mid_id]
    cb = cube.data.root_pos_w
    P(f"\n큐브-엄지끝 {torch.norm(cb-th,dim=-1).mean()*100:5.1f}cm   "
      f"큐브-중지끝 {torch.norm(cb-md,dim=-1).mean()*100:5.1f}cm   "
      f"엄지끝-중지끝 {torch.norm(th-md,dim=-1).mean()*100:5.1f}cm")
    P("  (큐브 반지름 3cm. 손끝이 3cm 안쪽이어야 접촉 중인 것)")

    # ── 판정: "손이 올라간 만큼 큐브도 올라왔는가" ───────────────
    # 이전 기준(z0 유지)은 팔 이동 방향과 무관해서, 손이 내려가면 물고 있어도 실패 판정이 났음.
    palm_w = robot.data.body_pos_w[:, palm_id]
    cube_w = cube.data.root_pos_w
    dist = torch.norm(cube_w - palm_w, dim=-1)
    fell = cube_w[:, 2] < 0.05                       # 바닥에 떨어짐
    cube_rise = cube_w[:, 2] - released_z            # 놓은 뒤 큐브 상승량
    held = (~fell) & (dist < 0.20) & (cube_rise > 0.05)

    P("\n" + "═" * 64)
    P(f"{'오므림':>9} {'잡음/시도':>10} {'최종 큐브z':>11} {'palm거리':>9}")
    P("─" * 64)
    for i, m in enumerate(FRACS):
        s = slice(i * REPEAT, (i + 1) * REPEAT)
        n = int(held[s].sum())
        mark = "OK" if n == REPEAT else ("일부" if n else "실패")
        P(f"{m:>9.2f} {n:>5d}/{REPEAT:<4d} {cube_w[s,2].mean()*100:>10.1f}cm {dist[s].mean()*100:>8.1f}cm  {mark}")
    P("═" * 64)

    # ★ '집게 성립' 판정: 엄지와 중지가 둘 다 큐브 표면에 붙어 있어야 함 (새끼에 걸린 건 제외)
    tips_all = SceneEntityCfg("robot", body_names=[
        "finger1_tip_link", "finger3_tip_link", "finger4_tip_link", "finger5_tip_link"],
        preserve_order=True)
    tips_all.resolve(scene)
    tp = robot.data.body_pos_w[:, tips_all.body_ids]  # (N,4,3) thumb, middle, ring, little
    d_th = torch.norm(cube_w.unsqueeze(1) - tp, dim=-1)  # (N,4)
    CONTACT = 0.055  # 중심-tip원점: 반지름3cm + 패드~2.5cm
    pinch = held & (d_th[:, 0] < CONTACT) & (d_th[:, 1] < CONTACT)
    P(f"\n{'오므림':>7} {'잡음':>6} {'집게(엄지+중지)':>14} {'엄지(cm)':>9} {'중지(cm)':>9} {'약지/새끼(cm)':>12}")
    P("─" * 64)
    for i, m in enumerate(FRACS):
        sl = slice(i * REPEAT, (i + 1) * REPEAT)
        P(f"{m:>7.2f} {int(held[sl].sum()):>4d}/{REPEAT:<2d} {int(pinch[sl].sum()):>8d}/{REPEAT:<2d} "
          f"{d_th[sl,0].mean()*100:>9.1f} {d_th[sl,1].mean()*100:>9.1f} "
          f"{d_th[sl,2:].min(dim=-1).values.mean()*100:>12.1f}")
    P("─" * 64)
    P("  '잡음'인데 '집게' 아님 = 약지/새끼에 걸렸거나 손바닥에 얹힌 것")

    ok = [m for i, m in enumerate(FRACS) if held[i*REPEAT:(i+1)*REPEAT].any()]
    if not ok:
        P("\n★ 어떤 오므림량으로도 0.30kg을 못 들었다.")
        P("  '물체 크기에서 멈추기'로도 안 되면 접촉 자체(마찰/접점 방향)가 문제다.")
    else:
        P(f"\n★ 드는 오므림량 존재: {ok}  (0.30kg)")
        P("  -> 손은 물리적으로 파지 가능. 정책이 '적당히 멈추는 오므림'을 배워야 한다.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
