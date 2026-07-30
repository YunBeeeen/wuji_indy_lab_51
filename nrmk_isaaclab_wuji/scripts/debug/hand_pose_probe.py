"""학습된 정책으로 실제 파지 순간의 손-스틱 자세(q_O_H) 측정 probe (2026-07-24, v2).

목적: hand_stick_orientation 보상의 목표 자세(q_O_H)를, **리셋 우연 자세가 아니라 잘 잡는
정책이 실제로 만든 파지 자세**에서 측정한다. 개별 grip region은 유지한 채 "엄지 vs palm 위아래"
손 전체 자세를 그 값으로 잡아주기 위함(capture_hand_tool_target의 target_quat_o).

v1(손가락만 scripted 오므림)은 팔을 안 움직여 손이 스틱 40cm 밖이라 무의미했음 → 정책 로드로 교체.

방식:
  1. --load_run 정책 로드 (예: 09-42-28 성공 런).
  2. 정책으로 N스텝 굴려 스틱을 실제로 잡게 함.
  3. "잘 잡은 스텝"(세 손끝 다 스틱 근처 + opposition>0 + clearance>0)에서 q_O_H 측정·집계.
     - 여러 env·여러 스텝의 조건 만족 샘플을 모아 q_O_H 평균(대표 파지 자세)과 분포를 냄.

⚠ Isaac Sim을 띄우므로 학습 중 실행 금지. 학습 멈춘 뒤.
  사용: python scripts/debug/hand_pose_probe.py --task Indy-Wuji-Chopsticks-Grasp \
          --num_envs 64 --load_run "2026-07-24_09-42-28(성공)" --steps 200
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure q_O_H at real grasp from a trained policy.")
parser.add_argument("--task", type=str, default="Indy-Wuji-Chopsticks-Grasp")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200, help="정책으로 굴릴 스텝 수(파지 형성 대기).")
parser.add_argument("--collect_from", type=int, default=60, help="이 스텝 이후부터 파지 샘플 수집.")
parser.add_argument("--surface_max", type=float, default=0.06, help="손끝-스틱중심 거리 이하를 '접촉'으로.")
parser.add_argument("--clearance_min", type=float, default=0.03, help="이 clearance 이상을 '들림'으로.")
# rsl_rl load 인자 (--load_run, --checkpoint 등). cli_args는 scripts/rsl_rl/에 있어 경로 추가 필요.
import os as _os  # noqa: E402
import sys as _sys  # noqa: E402

_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "rsl_rl"))
import cli_args  # noqa: E402  # isort: skip

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402
from isaac_neuromeka.tasks.manipulation.functional_grasp import mdp as fg_mdp  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import quat_from_euler_xyz  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def main():
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    # agent cfg (load_run 해석용)
    from isaac_neuromeka.tasks.manipulation.functional_grasp.indy_wuji.learning.rsl_rl_cfg import (
        ChopstickAcquirePPORunnerCfg,
    )

    agent_cfg = ChopstickAcquirePPORunnerCfg()
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)

    # --checkpoint로 직접 지정하면 그걸 우선 사용 (폴더명에 괄호/한글 있으면 get_checkpoint_path의
    # 정규식 매칭이 실패하므로, 그런 run은 --checkpoint <경로>로 넘길 것).
    if getattr(args, "checkpoint", None):
        resume_path = os.path.abspath(args.checkpoint)
    else:
        log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[probe] load: {resume_path}", flush=True)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    u = env.unwrapped
    robot = u.scene["robot"]
    obj = u.scene["cube"]
    palm_cfg = SceneEntityCfg("robot", body_names=["palm_link"])
    thumb_cfg = SceneEntityCfg("robot", body_names=["finger1_tip_link"])
    index_cfg = SceneEntityCfg("robot", body_names=["finger2_tip_link"])
    middle_cfg = SceneEntityCfg("robot", body_names=["finger3_tip_link"])
    obj_cfg = SceneEntityCfg("cube")
    for c in (palm_cfg, thumb_cfg, index_cfg, middle_cfg, obj_cfg):
        c.resolve(u.scene)

    def dist(cfg):
        fid = cfg.body_ids[0]
        return torch.norm(robot.data.body_state_w[:, fid, :3] - obj.data.root_pos_w, dim=-1)

    def unit(cfg):
        fid = cfg.body_ids[0]
        v = robot.data.body_state_w[:, fid, :3] - obj.data.root_pos_w
        return v / torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-6)

    # BASE_Z: env_cfg에서 스틱 표면 z 기준
    base_z = 0.25  # ChopsticksGraspEnvCfg.BASE_Z

    collected = []  # 조건 만족 q_O_H 모음
    obs, _ = env.get_observations()
    with torch.inference_mode():
        for t in range(args.steps):
            act = policy(obs)
            obs, _, _, _ = env.step(act)
            if t < args.collect_from:
                continue
            u_th = unit(thumb_cfg)
            opp_i = -torch.sum(u_th * unit(index_cfg), dim=-1)
            opp_m = -torch.sum(u_th * unit(middle_cfg), dim=-1)
            clr = obj.data.root_pos_w[:, 2] - base_z
            good = (
                (dist(thumb_cfg) < args.surface_max)
                & (dist(index_cfg) < args.surface_max)
                & (opp_i > 0.0)
                & (opp_m > 0.0)
                & (clr > args.clearance_min)
            )
            if good.any():
                q = fg_mdp.hand_orientation_in_object(u, palm_cfg, obj_cfg)
                collected.append(q[good].detach().cpu())

    print("\n" + "=" * 72)
    print(f"[hand_pose_probe v2] task={args.task} run={agent_cfg.load_run} steps={args.steps}")
    print("=" * 72)
    if not collected:
        print("조건 만족(엄지·검지 접촉 + opposition>0 + 들림) 샘플 0개.")
        print(f"  surface_max={args.surface_max}, clearance_min={args.clearance_min} 완화하거나 정책 확인.")
        simulation_app.close()
        return
    allq = torch.cat(collected, dim=0)  # (M,4)
    # 쿼터니안 반구 정렬 후 평균 (부호 모호성: 첫 샘플과 dot<0이면 뒤집기)
    ref = allq[0]
    signs = torch.where((allq * ref).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    aligned = allq * signs
    mean_q = aligned.mean(0)
    mean_q = mean_q / torch.clamp(torch.norm(mean_q), min=1e-6)
    std = aligned.std(0)
    print(f"수집 샘플: {allq.shape[0]}개 (엄지·검지 접촉 + opposition>0 + clearance>{args.clearance_min})")
    print(f"q_O_H 평균 (w,x,y,z) = {[round(v,5) for v in mean_q.tolist()]}")
    print(f"q_O_H 표준편차       = {[round(v,4) for v in std.tolist()]}  (작으면 일관된 파지 자세)")
    print("\n→ 이 평균 q_O_H를 capture_hand_tool_target(target_quat_o=...)에 넣고 hand_stick_orientation weight를 켤 것.")
    print("  std가 크면(>0.2) 파지 자세가 제각각이라 단일 목표로 부적합 — 파지부터 안정화 필요.")
    simulation_app.close()


if __name__ == "__main__":
    main()
