"""손이 6cm 큐브를 물리적으로 감쌀 수 있는 손인가? 물리/정책 없이 기하로만 답한다.

핵심 질문: 손가락을 오므릴 때 엄지끝과 중지끝이 6cm 이내로 모이는가?
못 모이면 effort_limit도 마찰도 시작자세도 전부 무의미하다.

관절을 0% -> 100% 오므리며 손끝 간격을 잰다. 접촉이 없어야 하므로 큐브는 치워둔다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Indy-Wuji-Cube-Grasp")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym
import isaac_neuromeka.tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

P = lambda *a: print(*a, flush=True)  # noqa: E731
CUBE = 0.06  # 큐브 한 변


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    scene, sim = env.scene, env.sim
    robot, cube = scene["robot"], scene["cube"]
    dt = sim.get_physics_dt()
    env.reset()

    # 큐브를 치워서 접촉이 기하 측정을 방해하지 않게
    away = cube.data.default_root_state.clone()
    away[:, 2] = -5.0
    cube.write_root_state_to_sim(away)

    fing = SceneEntityCfg("robot", joint_names=["finger[1-3]_joint[1-4]"])
    tips = SceneEntityCfg("robot", body_names=[
        "finger1_tip_link", "finger2_tip_link", "finger3_tip_link"], preserve_order=True)
    for c in (fing, tips):
        c.resolve(scene)
    fid, tid = fing.joint_ids, tips.body_ids
    lim = robot.data.joint_pos_limits[0]
    dflt = robot.data.default_joint_pos[0]

    P("\n═══ 손가락 관절: 기본값과 가동범위 ═══")
    for j in fid:
        P(f"  {robot.joint_names[j]:22s} default={dflt[j]:+.2f}  limit=[{lim[j,0]:+.2f}, {lim[j,1]:+.2f}]")

    # 오므림 = 각 관절을 자기 상한(굴곡 방향)까지. 상한이 곧 '최대로 오므린' 자세.
    open_pose = robot.data.default_joint_pos.clone()
    closed = open_pose.clone()
    closed[:, fid] = lim[fid, 1]  # 상한 = 최대 굴곡

    P("\n═══ 오므림에 따른 손끝 간격 ═══")
    P(f"{'오므림%':>7} {'엄지-중지':>10} {'엄지-검지':>10} {'검지-중지':>10}   (큐브 6.0cm)")
    P("─" * 52)

    best = 1e9
    for pct in range(0, 101, 10):
        pose = open_pose.clone()
        pose[:, fid] = open_pose[:, fid] + (closed[:, fid] - open_pose[:, fid]) * (pct / 100.0)
        # 물리 없이 자세만 강제로 박고 FK 갱신
        robot.write_joint_state_to_sim(pose, torch.zeros_like(pose))
        robot.set_joint_position_target(pose)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)

        p = robot.data.body_pos_w[0, tid]  # (3,3): thumb, index, middle
        d_tm = torch.norm(p[0] - p[2]).item() * 100  # 엄지-중지  ★ 논문의 cage
        d_ti = torch.norm(p[0] - p[1]).item() * 100  # 엄지-검지
        d_im = torch.norm(p[1] - p[2]).item() * 100  # 검지-중지
        best = min(best, d_tm)
        flag = "  <- 큐브 통과 가능" if d_tm < CUBE * 100 else ""
        P(f"{pct:>6}% {d_tm:>9.1f}cm {d_ti:>9.1f}cm {d_im:>9.1f}cm{flag}")

    P("─" * 52)
    P(f"\n엄지-중지 최소 간격 = {best:.1f} cm   (큐브 6.0 cm)")
    if best > CUBE * 100:
        P("\n★ 손을 최대로 오므려도 엄지-중지가 큐브보다 넓게 벌어져 있다.")
        P("  이 손은 6cm 큐브를 '집게'로 못 쥔다. 감싸기(power grasp)만 가능하거나,")
        P("  큐브를 더 크게 하거나, 손가락 관절 범위/방향 설정이 잘못된 것이다.")
    else:
        P("\n★ 엄지-중지가 큐브보다 좁게 모인다 -> 집게 파지가 기하적으로 가능하다.")
        P("  파지 실패는 기하가 아니라 힘(effort_limit)/마찰/정책 문제다.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
