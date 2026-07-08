# study.md

## 전체 요약

- 목표는 Isaac Lab에서 Indy7 + Wuji hand 기반 end-effector tracking RL 환경을 구성하는 것임.
- 현재 구현은 Neuromeka `Indy-Reach` 스타일을 따라간 `Indy-Wuji-Reach` task임.
- 현재는 hand 조작이 아니라 arm 6축으로 `palm_link`를 target pose에 tracking하는 단계임.
- 현재 action은 arm joint 6개만 사용함.
- 현재 hand joint는 articulation에는 있지만 policy action/observation에서는 제외함.
- 현재 active USD는 `indy7_wuji_right_simplified.usd`임.
- 현재 tracking body는 `tcp`가 아니라 `palm_link`임.

## 코드 흐름

- 실행은 `scripts/rsl_rl/train.py`에서 시작됨.
- `train.py`가 IsaacLab `AppLauncher`로 Isaac Sim을 먼저 띄움.
- 그 다음 `isaac_neuromeka.tasks`를 import함.
- task import 과정에서 `Indy-Wuji-Reach`가 gym에 등록됨.
- `gym.make("Indy-Wuji-Reach")`가 `ManagerBasedRLEnv`를 생성함.
- `Indy7WujiReachEnvCfg`가 env 설정으로 들어감.
- `ReachEnvCfg` 공통 구조가 먼저 만들어짐.
- `Indy7WujiReachEnvCfg.__post_init__()`에서 robot, body name, action, observation 범위를 덮어씀.
- RSL-RL runner가 observation/action shape를 읽고 actor/critic model을 만듦.
- 이후 PPO 학습 loop가 돌아감.

## 주요 파일

- task 등록 파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/reach/indy_wuji/__init__.py
```

- Indy-Wuji env override 파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/reach/indy_wuji/env_cfg.py
```

- Neuromeka 공통 reach env 파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/reach/reach_env_cfg.py
```

- 공통 command/action/observation/reward 설정 파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py
```

- robot asset 설정 파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/assets/indy.py
```

- custom MDP 함수 폴더임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/mdp/
```

- 학습 스크립트임.

```text
nrmk_isaaclab_wuji/scripts/rsl_rl/train.py
```

- play 스크립트임.

```text
nrmk_isaaclab_wuji/scripts/rsl_rl/play.py
```

## 환경 구성 방식

- `Indy-Wuji-Reach`는 `ManagerBasedRLEnv` 기반임.
- Direct task처럼 env class 안에 reward/observation/action을 직접 쓰는 방식이 아님.
- command, action, observation, reward, termination을 manager config로 나눠서 구성함.
- 공통 reach 구조는 `ReachEnvCfg`에 있음.
- robot-specific 설정은 `Indy7WujiReachEnvCfg`에서 override함.
- robot asset은 `INDY7_WUJI_RIGHT_CFG`로 지정함.
- USD 경로는 `assets/indy.py` 안의 `INDY7_WUJI_RIGHT_CFG.spawn.usd_path`가 결정함.
- end-effector command는 `ee_pose` 하나임.
- reward는 `palm_link`와 `ee_pose` command 간 position/orientation tracking 기준임.

## 현재 Indy-Wuji 설정

- task id는 `Indy-Wuji-Reach`임.
- env cfg는 `Indy7WujiReachEnvCfg`임.
- robot cfg는 `INDY7_WUJI_RIGHT_CFG`임.
- active USD는 `indy7_wuji_right_simplified.usd`임.
- tracking body는 `palm_link`임.
- action joint는 `joint[0-5]`임.
- action dimension은 6임.
- policy observation dimension은 55임.
- `joint_pos` observation은 arm 6축만 봄.
- `joint_vel` observation은 arm 6축만 봄.
- `joint_vel` penalty도 arm 6축만 봄.
- hand joint는 나중에 hand action 확장 시 다시 포함할 수 있음.

## Franka 예제와 차이

- IsaacLab 공식 Franka Reach는 공식 표준 구조 확인용임.
- Franka Reach는 IsaacLab core `mdp`를 많이 재사용함.
- Neuromeka Indy는 `isaac_neuromeka.mdp` wrapper와 custom 함수가 더 많음.
- Franka와 Indy-Wuji 모두 ManagerBased 구조임.
- 파일 배치와 custom wrapper가 다를 뿐 흐름은 비슷함.
- 지금 구현은 Franka 스타일보다 Neuromeka `Indy-Reach` 스타일을 따라가는 것이 맞음.

## 치명적인 주의사항

- `tcp`를 reward/command body로 쓰면 안 됨.
- 현재 USD articulation에서 `tcp`는 실제 rigid body로 쓰기 어려움.
- tracking body는 `palm_link`를 써야 함.
- `SceneEntityCfg` 객체 하나를 여러 term에 재사용하면 안 됨.
- IsaacLab이 resolve 과정에서 `joint_ids`를 내부에 채우므로 term마다 새 `SceneEntityCfg`를 만들어야 함.
- `isaac_neuromeka` import는 IsaacLab `AppLauncher` 이후에 하는 흐름이 안전함.
- plain `python -c "import isaac_neuromeka"`로 검증하려 하면 `pxr` 문제로 실패할 수 있음.
- `train.py`와 `play.py`의 rsl-rl config migration 코드를 제거하면 안 됨.
- rsl-rl 5 계열에서는 deprecated `policy` config를 actor/critic으로 변환해야 함.
- Wuji collision은 URDF importer만 믿으면 안 됨.
- 현재 fidelity USD는 collision STL 26개를 직접 USD Mesh collider로 넣어둔 상태임.
- Isaac Sim 5.1 URDF importer의 on-disk export path는 geometry가 빈 USD를 만들 수 있음.
- URDF 재import가 필요하면 in-memory import 후 `Save As` 방식이 안전함.
- hand action을 추가하기 전에는 observation/reward penalty에 hand joint를 섞지 않는 것이 안전함.

## 학습 확인 기준

- smoke test는 `num_envs=1`, `max_iterations=1`부터 봄.
- 그 다음 `128/20`, `512/100` 순서로 키움.
- crash가 없어야 함.
- NaN이 없어야 함.
- PhysX explosion이 없어야 함.
- action shape가 6인지 봄.
- observation shape가 55인지 봄.
- position error가 계속 커지기만 하지 않는지 봄.
- orientation error가 3.14 근처에 고정되지 않는지 봄.
- GUI에서 로봇이 발산하거나 심하게 떨지 않는지 봄.

## 다음 방향

- `512 env / 100 iter` 결과를 봄.
- 결과가 안정적이면 `512 env / 500 iter`로 확장함.
- tracking이 잘 안 되면 command range 조정함.
- orientation이 부담되면 orientation reward weight 조정함.
- 움직임이 너무 느리면 decimation 조정 검토함.
- 실제 TCP 기준 tracking이 필요하면 `palm_link` 기준 offset frame 추가함.
- hand 제어가 필요해지면 hand action/observation/reward를 별도 단계로 확장함.
