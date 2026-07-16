"""Box-Transport 치수 랜덤화 end-to-end 검증: 버퍼 vs USD scale vs 정착 높이."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
app = AppLauncher(args)
simulation_app = app.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg("Indy-Wuji-Box-Transport", num_envs=4)
env = gym.make("Indy-Wuji-Box-Transport", cfg=cfg).unwrapped
env.reset()
# 몇 스텝 정착
for _ in range(30):
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))

stage = get_current_stage()
cube = env.scene["cube"]
he = env.box_half_extents
print("=== env별 검증 (버퍼 half | USD scale | 정착 z − (상판+half_z)) ===", flush=True)
for i in range(env.num_envs):
    prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Cube")
    scale = prim.GetAttribute("xformOp:scale").Get()
    z = cube.data.root_pos_w[i, 2].item() - env.scene.env_origins[i, 2].item()
    resid = z - (0.25 + he[i, 2].item())
    print(f"env{i}: half={he[i].tolist()}  scale={list(scale)}  settle_resid={resid*1000:+.1f}mm", flush=True)

env.close()
simulation_app.close()
