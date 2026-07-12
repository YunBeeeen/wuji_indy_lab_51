# study.md

- 이 문서는 Indy7 + Wuji Isaac Lab 환경 구조와 핵심 개념을 공부하기 위한 요약 문서임.

## 전체 요약

- 목표는 Isaac Lab에서 Indy7 + Wuji hand 기반 end-effector tracking RL 환경을 구성하는 것임.
- 현재 구현은 Neuromeka `Indy-Reach` 스타일을 따라간 `Indy-Wuji-Reach` task임.
- 현재는 hand 조작이 아니라 arm 6축으로 `link6` arm flange를 target pose에 tracking하는 단계임.
- 현재 action은 arm joint 6개만 사용함.
- 현재 hand joint는 articulation에는 있지만 policy action/observation에서는 제외함.
- 현재 active USD는 `indy7_wuji_right_simplified.usd`임.
- 현재 tracking body는 `link6`임.
- virtual EE offset은 실험 후 보류하고, `link6` baseline long-run을 먼저 보기로 함.

## 코드 흐름

- 실행은 `scripts/rsl_rl/train.py`에서 시작됨.
- `train.py`가 IsaacLab `AppLauncher`로 Isaac Sim을 먼저 띄움.
- 그 다음 `isaac_neuromeka.tasks`를 import함.
- task import 과정에서 `Indy-Wuji-Reach`가 gym에 등록됨.
- `gym.make("Indy-Wuji-Reach")`가 `ManagerBasedRLEnv`를 생성함.
- `Indy7WujiReachEnvCfg`가 env 설정으로 들어감.
- `ReachEnvCfg` 공통 구조가 먼저 만들어짐.
- `Indy7WujiReachEnvCfg.__post_init__()`에서 robot, tracking body, action, observation 범위를 덮어씀.
- RSL-RL runner가 observation/action shape를 읽고 actor/critic model을 만듦.
- 이후 PPO 학습 loop가 돌아감.

## 주요 파일

- 실행 흐름 상세 정리 문서임.

```text
flow_study.md
```

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
- command는 공통 `UniformPoseCommandCfg` 기준임.
- reward는 `link6`와 `ee_pose` command 간 tracking 기준임.

## 현재 Indy-Wuji 설정

- task id는 `Indy-Wuji-Reach`임.
- env cfg는 `Indy7WujiReachEnvCfg`임.
- robot cfg는 `INDY7_WUJI_RIGHT_CFG`임.
- active USD는 `indy7_wuji_right_simplified.usd`임.
- tracking body는 `link6`임.
- virtual EE offset은 현재 적용하지 않음.
- action joint는 `joint[0-5]`임.
- action dimension은 6임.
- policy observation dimension은 55임.
- `joint_pos` observation은 arm 6축만 봄.
- `joint_vel` observation은 arm 6축만 봄.
- `joint_vel` penalty도 arm 6축만 봄.
- hand joint는 나중에 hand action 확장 시 다시 포함할 수 있음.

## Reward 구조

- reward term 설정은 아래 파일에 있음.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py
```

- reward 계산 함수는 아래 파일에 있음.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/mdp/rewards.py
```

- `end_effector_position_tracking`은 position distance 기반임.
- `weight=0.2`임.
- `distance_max=0.5`임.
- 계산은 `1 - clamp(distance, 0, distance_max) / distance_max`임.
- `end_effector_orientation_tracking`은 quaternion orientation error 기반임.
- `weight=0.1`임.
- `distance_max=0.25`임.
- 계산은 `1 - clamp(orientation_error, 0, 3.14) / 3.14`임.
- 단, position distance가 `0.25m`보다 크면 orientation reward는 0이 됨.
- `end_effector_speed`, `action_rate`, `joint_vel`은 penalty임.
- 현재 이 reward들은 `link6` 기준으로 적용됨.

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
- 현재 reach baseline에서는 tracking 기준 rigid body를 `link6`로 둠.
- Wuji hand 기준 tracking으로 돌아갈 때는 `palm_link` 또는 virtual EE frame을 다시 검토함.
- raw `palm_link` orientation은 frame mismatch 가능성이 있음.
- 현재는 `link6` long-run 결과를 먼저 봄.
- offset은 long-run 결과 확인 후 다시 검토함.
- Allegro는 `tcp`와 hand base를 fixed joint/offset으로 분리하는 구조임.
- Wuji는 현재 `tcp`를 body로 못 쓰므로 offset을 쓰게 되면 code-level virtual frame 후보가 됨.
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
- 그 다음 `128/20`, `512/500`, `4000/50000` 순서로 키움.
- crash가 없어야 함.
- NaN이 없어야 함.
- PhysX explosion이 없어야 함.
- action shape가 6인지 봄.
- observation shape가 55인지 봄.
- position error가 계속 커지기만 하지 않는지 봄.
- orientation error가 3.14 근처에 고정되지 않는지 봄.
- GUI에서 로봇이 발산하거나 심하게 떨지 않는지 봄.

## 다음 방향

- `4000 env / 50000 iter` 학습을 돌림.
- TensorBoard로 reward/error 추세를 봄.
- tracking이 잘 안 되면 command range 조정함.
- orientation이 부담되면 orientation reward weight 조정함.
- 움직임이 너무 느리면 decimation 조정 검토함.
- 실제 TCP 기준 tracking이 필요하면 virtual EE offset을 다시 검증/수정함.
- hand 제어가 필요해지면 hand action/observation/reward를 별도 단계로 확장함.

## 2026-07-10 Cube Grasp 구현 요약

- `Indy-Wuji-Cube-Grasp` task skeleton을 추가함.
- cube grasp는 reach task를 덮어쓴 것이 아니라 별도 `manipulation/grasp` package로 분리함.
- cube grasp 공통 cfg는 `cube_grasp_env_cfg.py`에 둠.
- Indy/Wuji 전용 override는 `grasp/indy_wuji/env_cfg.py`에 둠.
- cube는 `{ENV_REGEX_NS}/Cube`에 `RigidObjectCfg`로 추가함.
- cube size는 `0.06 m`임.
- cube mass는 `0.08 kg`임.
- cube initial position은 `(0.45, -0.18, 0.03)`임.
- cube collision은 켜져 있음.
- 4096 env 학습에서 PhysX patch buffer overflow가 발생함.
- cube grasp cfg에서 `gpu_max_rigid_patch_count`를 `2**19`로 올림.
- `2**18`은 4096 env resume run에서 약 `263k` patch 요구치에 살짝 부족했음.
- cube 위치는 reset 때 default 위치 기준 `x ±0.06`, `y ±0.08` 범위로 randomization함.
- cube randomization은 functional grasp proxy가 고정 위치에만 과적합하지 않게 하는 목적임.

## Cube Grasp Action

- 최신 cube grasp action은 18D임.
- arm 6축과 thumb/index/middle 12축을 같이 제어함.
- controlled joint pattern은 `joint[0-5]`, `finger[1-3]_joint[1-4]`임.
- `finger1`은 thumb임.
- `finger2`는 index임.
- `finger3`은 middle임.
- `finger4`는 ring이지만 현재 action에서는 고정/비제어 보조 finger로 둠.
- `finger5`는 little이지만 현재 action에서는 고정/비제어 보조 finger로 둠.

## Cube Grasp Observation

- 최신 cube grasp policy observation은 57D임.
- controlled joint position 18D를 봄.
- `palm_link` 기준 cube relative position 3D를 봄.
- thumb/index/middle/ring/little fingertip 기준 cube relative position 15D를 봄.
- cube target relative position 3D를 봄.
- previous action 18D를 봄.
- joint velocity는 아직 observation에서 뺌.
- fingertip velocity도 아직 observation에서 뺌.
- contact state도 아직 observation에 넣지 않음.
- contact/lift는 이후 reward/success 계산에 먼저 쓰는 방향임.

## Cube Grasp Reward

- 현재 reward는 완성된 grasp success reward가 아님.
- 현재 reward는 functional grasp/chopstick grasp로 가기 위한 proxy shaping reward임.
- 최종 목표는 젓가락을 기능적으로 잡고 젓가락질이 가능한 파지 상태를 만드는 것임.
- `Dexterous Pre-grasp.pdf` 계열 functional grasp/pre-grasp 논문 흐름이 주 기준임.
- DexPoint는 구현 목표가 아니라 contact/lift gate 설계 참고임.
- active reward term은 현재 `finger_cube_reach`, `functional_hold`, `action_rate`임.
- `arm_cube_reach`는 `palm_link`와 cube root position 거리 기반임.
- `arm_cube_reach` weight는 `0.05`임.
- `arm_cube_reach`는 현재 palm over-guidance를 줄이기 위해 비활성화함.
- `finger_cube_reach`는 thumb/index/middle tip과 cube root position 사이 weighted distance의 progress 기반임.
- `finger_cube_reach`는 현재 거리 자체가 아니라 episode best distance가 줄어드는지 봄.
- 현재 cfg는 `mode="best"`임.
- 계산 흐름은 `progress = previous_best_distance - current_distance`임.
- `finger_cube_reach` body weight는 thumb/index/middle = `3.0/1.0/1.0`임.
- `finger_cube_reach` term weight는 `0.3`임.
- `finger_cube_reach` distance_max는 progress 정규화 scale로 `0.03`임.
- 여기서 `distance_max`는 실제 최대 거리가 아니라 한 step progress를 reward로 키우는 scale임.
- `functional_hold`는 functional grasp/pre-grasp 논문의 hold/cage 아이디어를 현재 cube task에 맞춘 reward임.
- `functional_hold`는 cube가 thumb/index/middle tip들이 만드는 grasp region 안쪽에 들어오는지 봄.
- `functional_hold`는 weighted fingertip-cube closeness, uniform cube-grasp-center closeness, thumb-opposition을 조합함.
- `functional_hold` body weight는 thumb/index/middle = `3.0/1.0/1.0`임.
- 이 body weight는 fingertip-cube distance bonus에만 사용함.
- grasp center 계산은 thumb 쪽으로 끌리지 않도록 uniform weight를 사용함.
- `functional_hold`는 곱셈 gate만 쓰지 않고 additive 구조를 사용함.
- 현재 raw는 `0.4 * center_bonus + 0.4 * weighted_distance_bonus + 0.2 * center_bonus * opposition`임.
- `functional_hold` term weight는 `0.2`임.
- `functional_hold` distance_max는 `0.18`임.
- `functional_hold` center_distance_max는 `0.18`임.
- random policy smoke test에서 `functional_hold Raw`가 0이어도 정상임.
- `cube_lift`는 cube root z height 기반 bounded reward임.
- `cube_lift` weight는 `0.05`임.
- `cube_goal_tracking`은 cube z가 `0.08 m` 이상일 때만 켜지는 target position reward임.
- cube goal position은 `(0.55, -0.05, 0.12)`임.
- `cube_goal_tracking` weight는 `0.2`임.
- `cube_lift`와 `cube_goal_tracking`은 contact-gated grasp가 검증될 때까지 비활성화함.
- `action_rate` weight는 `-0.0003`임.
- raw reward는 각 reach term 기준 최대 1임.
- active weighted positive max는 현재 대략 `0.5`임.
- TensorBoard에서는 `Episode_Reward/*`가 weighted reward임.
- TensorBoard에서는 `Episode_Reward_Raw/*`가 raw reward임.
- TensorBoard에서는 `Metrics/cube/*`가 실제 거리 error임.
- `Metrics/cube/palm_distance`는 palm-cube 거리임.
- `Metrics/cube/thumb_distance`, `index_distance`, `middle_distance`, `ring_distance`, `little_distance`는 fingertip-cube 거리임.
- `Metrics/cube/finger_mean_distance`는 five-finger 거리 평균임.
- `Metrics/cube/non_thumb_mean_distance`는 index/middle/ring/little 거리 평균임.
- `Metrics/cube/finger_weighted_mean_distance`는 thumb을 크게 본 five-fingertip 평균 거리임.
- `Metrics/cube/*`는 reward가 아니라 학습 상태 확인용 metric임.

## Cube Grasp 학습 해석

- `finger_cube_reach Raw`가 0에 가깝다는 것은 이번 step에서 최단거리 기록을 새로 갱신하지 못했다는 뜻임.
- `finger_cube_reach Raw`는 현재 fingertip-cube 절대 거리 자체가 아님.
- `Metrics/cube/finger_mean_distance`가 내려가면 실제 five fingertip들이 cube에 가까워지는 뜻임.
- 실제 접근 여부는 `Metrics/cube/thumb_distance`, `index_distance`, `middle_distance`, `finger_mean_distance`를 봄.
- `distance_max=0.03`은 progress reward scale임.
- `distance_max`를 줄이면 같은 거리 개선량에 대한 reward scale이 커짐.
- progress reward만으로는 일정 거리까지 간 뒤 reward가 희소해질 수 있음.
- progress reward만으로 hand가 cube 근처에 머무르지 않으면 absolute closeness reward를 같이 둠.
- 현재 cube grasp는 `finger_cube_reach`로 접근 방향을 주고, `finger_cube_closeness`로 가까운 상태 유지를 보상함.
- velocity observation/reward는 아직 넣지 않음.
- contact/lift 단계에서는 contact-gated reward와 success metric이 같이 필요함.
- 현재는 contact 조건이 없으므로 cube를 안정적으로 잡는 task는 아직 아님.
- 현재 단계는 arm+finger가 cube에 접근하고, lift 후 target으로 옮기는 구조를 처음 연결한 baseline임.
- 다음 단계는 contact group reward임.
- 그 다음 단계는 contact-gated lift/goal reward임.
