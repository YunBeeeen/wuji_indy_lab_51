"""Gym registrations for the isolated Wuji-hand chopstick tasks.

The three task IDs intentionally represent different experiment boundaries:

* ``hand_grasp`` learns OPEN/CLOSE motion from an already established grasp.
* ``hand_setting`` learns the preceding open-hand-to-functional-grasp transition.
* ``hand_real`` keeps hand_move physics but replaces simulator velocity observations
  with one-step joint/stick pose history for deployment.
* ``hand_object`` fine-tunes hand_move to actually pinch and hold a 1 cm cube;
  the same ID serves training and (with --manual_root) play/calibration.
* ``hand_object_105_distill`` keeps the successful 103D hand_object actor as a
  frozen rollout teacher and trains a deployable 105D student.
* ``hand_final`` applies the same cube objective to the deployable 105D
  hand_real observation and actuator contract.
* ``hand_final_play`` runs that 105D policy in hand_play's table-and-plates
  scene without changing the legacy 103D hand_play interface.
* ``hand_grasp_object`` is the legacy ID for the same config (the cfg classes
  were renamed and aliased), kept so existing commands keep working.

Keeping separate task IDs prevents checkpoints with different observation
dimensions or success definitions from being loaded into the wrong MDP.
"""

import gymnasium as gym

from . import learning
from .finger_reach_env_cfg import FingerReachEnvCfg
from .hand_final_env_cfg import HandFinalEnvCfg
from .hand_grasp_env_cfg import HandGraspEnvCfg
from .hand_real_env_cfg import HandRealEnvCfg
from .hand_real2_env_cfg import HandReal2EnvCfg
from .hand_object_env_cfg import HandObjectGraspEnvCfg
from .hand_move_env_cfg import HandMoveEnvCfg
from .hand_object_env_cfg import HandObjectEnvCfg
from .hand_object_distill_env_cfg import HandObjectDistillationEnvCfg
from .hand_play_env_cfg import HandFinalPlayEnvCfg, HandPlayEnvCfg
from .hand_setting_env_cfg import HandSettingEnvCfg


gym.register(
    id="hand_grasp",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandGraspEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:HandGraspPPORunnerCfg",
    },
)


gym.register(
    id="hand_grasp_object",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandObjectGraspEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:HandObjectGraspPPORunnerCfg"
        ),
    },
)


gym.register(
    id="hand_setting",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandSettingEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:HandSettingPPORunnerCfg"
        ),
    },
)


# 진단 전용(2026-08-04): 손끝 하나가 랜덤 목표점을 따라가는 mini-reach. PD/action이 손끝을
# 임의 점에 갖다놓을 수 있나 격리 검증. 핑거 교체는 finger_reach_env_cfg.REACH_TIP_LINK.
gym.register(
    id="finger_reach",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FingerReachEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:FingerReachPPORunnerCfg"
        ),
    },
)


# floating-root 기반(2026-08-05): `hand_grasp`와 동작이 같은 복제본인데 손 root만
# world에 고정되지 않고 wrench 컨트롤러가 제자리에 붙잡는다. 현재는 root 회전 액션이
# 비활성(action 20D / obs 103D = hand_grasp와 동일)이고, 이후 목표 손 자세 command와
# 3D 회전 액션을 여기에 얹을 예정. 별도 task ID라 root 구성이 다른 체크포인트가
# 서로 섞여 로드될 수 없다.
gym.register(
    id="hand_move",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandMoveEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:HandMovePPORunnerCfg"
        ),
    },
)


# Sim-to-real sibling of hand_move.  All dynamics/objectives are inherited,
# but its policy interface is 105D and intentionally checkpoint-incompatible
# with the simulator-velocity-based 103D hand_move policy.
gym.register(
    id="hand_real",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandRealEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:HandRealPPORunnerCfg"
        ),
    },
)


# Acquisition A/B sibling.  It shares hand_real's 105D policy and actuator
# contract but writes to logs/rsl_rl/hand_real2 and starts non-thumb pads
# 15 mm outside the reference grasp.
gym.register(
    id="hand_real2",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandReal2EnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:HandReal2PPORunnerCfg"
        ),
    },
)


# 2026-08-06: hand_move의 검증된 손/스틱 spawn·floating root·PD 컨트롤러를 그대로
# 상속하고 1cm dynamic cube + 지지 기둥 + 스틱별 cube 접촉센서만 얹은 환경.
# obs 103D / action 20D가 hand_move와 동일해서 체크포인트가 shape 변경 없이 로드된다.
#
# task ID를 train/play로 나누지 않은 것은 hand_move와 같은 이유다 — 세 스크립트
# 커맨드(open_close·root_orientation·support)가 전부 manual override를 갖고 있어
# play.py --manual_root 하나로 수동 모드가 되고 calibration도 같은 경로를 쓴다.
#   학습:   train.py --task hand_object --headless
#   수동:   play.py  --task hand_object --num_envs 1 --manual_root \\
#              env.require_calibration=false env.episode_length_s=300.0
gym.register(
    id="hand_object",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandObjectEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:HandObjectPPORunnerCfg"
        ),
    },
)


# Compatibility bridge only: the 103D model_300 teacher drives the successful
# hand_object rollout while a separate 105D actor learns its physical residual
# commands.  A resulting student checkpoint is then fine-tuned/played through
# hand_real or hand_final, not through this two-observation training task.
gym.register(
    id="hand_object_105_distill",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandObjectDistillationEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:HandObject105DistillationRunnerCfg"
        ),
    },
)


# Final sim-to-real fine-tuning boundary: cube scene/objective from hand_object,
# but actor observation and task-local Kp/Kd from hand_real.  It intentionally
# logs separately so 105D hand_final checkpoints cannot be confused with the
# 103D hand_object lineage.
gym.register(
    id="hand_final",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandFinalEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:HandFinalPPORunnerCfg"
        ),
    },
)

# Play-only sibling of ``hand_object``: same policy interface, but the
# retracting column is replaced by a table with two plates and the cube sits in
# the left one.  Deliberately shares ``HandObjectPPORunnerCfg`` so its
# ``experiment_name`` stays ``hand_object`` and ``--load_run <run>`` resolves
# against the folder the checkpoints actually live in.
#
# Training this ID is refused (``HandPlayEnvCfg.require_manual_root``): the
# calibrated goal pose is below the tabletop, so the scripted trajectory would
# drive the hand through the furniture.
gym.register(
    id="hand_play",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandPlayEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:HandObjectPPORunnerCfg"
        ),
    },
)


# 105D sibling of hand_play for checkpoints trained under hand_final.  Keeping
# a separate task ID prevents a 105D actor from being loaded into hand_play's
# established 103D hand_object observation contract (and vice versa).
gym.register(
    id="hand_final_play",
    entry_point="isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HandFinalPlayEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{learning.__name__}.rsl_rl_cfg:HandFinalPPORunnerCfg"
        ),
    },
)
