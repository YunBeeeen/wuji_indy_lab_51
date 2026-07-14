"""파지 실패를 눈으로 본다. 손에 큐브를 쥐여주고 오므리는 과정을 손 근처에서 촬영한다.

숫자로만 보면 "손가락이 저항 없이 3.4cm까지 닫혔다 / 큐브는 30cm 밖"까지만 알 수 있고,
큐브가 '어디로' 어떻게 빠져나가는지는 알 수 없다. 그림이 답을 준다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Indy-Wuji-Cube-Grasp")
parser.add_argument("--close_frac", type=float, default=0.50)
parser.add_argument("--out", type=str, default="/tmp/grip")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import imageio
import numpy as np
import torch

import isaac_neuromeka.tasks  # noqa: F401
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab_tasks.utils import parse_env_cfg

P = lambda *a: print(*a, flush=True)  # noqa: E731


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, num_envs=1)
    env_cfg.episode_length_s = 1e9
    env_cfg.scene.gripcam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/gripcam",
        update_period=0.0,
        height=600, width=800,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 20.0)),
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    scene, sim = env.scene, env.sim
    robot, cube = scene["robot"], scene["cube"]
    dt = sim.get_physics_dt()
    env.reset()

    fing = SceneEntityCfg("robot", joint_names=["finger[1-3]_joint[1-4]"])
    tips = SceneEntityCfg("robot", body_names=["finger1_tip_link", "finger3_tip_link"],
                          preserve_order=True)
    for c in (fing, tips):
        c.resolve(scene)
    fid = fing.joint_ids
    thumb_id, mid_id = tips.body_ids
    lim = robot.data.joint_pos_limits[0]
    hold = robot.data.default_joint_pos.clone()

    # 파지 지점 = 오므림 30%에서의 엄지끝-중지끝 중점 (간격이 큐브 6cm와 같아지는 곳)
    away = cube.data.default_root_state.clone()
    away[:, 2] = -5.0
    cube.write_root_state_to_sim(away)
    probe = hold.clone()
    probe[:, fid] = lim[fid, 1] * 0.30
    robot.write_joint_state_to_sim(probe, torch.zeros_like(probe))
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt)
    grip = 0.5 * (robot.data.body_pos_w[:, thumb_id] + robot.data.body_pos_w[:, mid_id])

    robot.write_joint_state_to_sim(hold, torch.zeros_like(hold))
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt)

    # 손 옆 30cm에서 파지 지점을 바라보게 카메라를 옮긴다
    cam = scene["gripcam"]
    g = grip[0].cpu().numpy()
    cam.set_world_poses_from_view(
        eyes=torch.tensor([[g[0] + 0.05, g[1] - 0.30, g[2] + 0.12]], device=env.device),
        targets=torch.tensor([g], device=env.device),
    )

    cube_state = cube.data.default_root_state.clone()
    cube_state[:, :3] = grip
    cube_state[:, 3:] = 0.0
    cube_state[:, 3] = 1.0

    target = hold.clone()
    N_CLOSE, N_HOLD = 90, 90
    shots = {0: "01_open", 45: "02_closing", 89: "03_closed_pinned",
             95: "04_just_released", 120: "05_after_1s", 179: "06_end"}

    for step in range(N_CLOSE + N_HOLD):
        a = min(step / N_CLOSE, 1.0) * args_cli.close_frac
        target[:, fid] = lim[fid, 1] * a
        robot.set_joint_position_target(target)
        if step < N_CLOSE:
            cube.write_root_state_to_sim(cube_state)  # 오므리는 동안 손에 고정

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(dt)
        # 카메라는 scene.update가 갱신함

        if step in shots:
            th = robot.data.body_pos_w[0, thumb_id]
            md = robot.data.body_pos_w[0, mid_id]
            cb = cube.data.root_pos_w[0]
            gap = torch.norm(th - md).item() * 100
            d_th = torch.norm(cb - th).item() * 100
            d_md = torch.norm(cb - md).item() * 100
            P(f"[{step:3d}] {shots[step]:18s} 엄지-중지간격={gap:5.1f}cm  "
              f"큐브-엄지={d_th:5.1f}cm  큐브-중지={d_md:5.1f}cm  큐브z={cb[2]*100:5.1f}cm")
            img = cam.data.output["rgb"][0].cpu().numpy()
            imageio.imwrite(f"{args_cli.out}/{shots[step]}.png", img[..., :3].astype(np.uint8))

    P(f"\n이미지: {args_cli.out}/")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
