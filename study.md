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

## Cube Grasp Reward 개념 (2026-07-12 기준)

### 핵심 아이디어: 가상점 cage

- 엄지끝에서 검지/중지로 선분을 긋고, 그 위에 **가상점 12개**를 찍음.
- 그 점들이 물체 **내부**로 파고들면 보상함.
- **손을 오므리면 점들이 서로 가까워지며 물체 안으로 들어감.** 따라서 "오므리기"가 직접 보상됨.
- **접촉센서가 필요 없음.** 논문도 "expensive contact information"을 명시적으로 피함.
- 물체 SDF는 큐브라 해석식임. CAD 불필요함.

### 왜 "거리 reward"가 아니라 "cage"인가 (가장 중요한 개념)

```text
거리 reward :  물체를 만지면 밀려남 -> 거리가 늘어남 -> 감점
               => 접촉이 "손해"  => 정책은 hover만 함

cage reward :  가상점이 물체를 파고들어야 점수가 남
               => 접촉이 "이득"  => 정책이 파지를 함
```

- **부호가 반대임.** 이 차이가 파지 학습의 핵심임.
- 움직이는 물체에 대한 거리 reward는 **구조적으로 접촉을 처벌함.**

### 4개 항의 역할 분담

- `finger_cage_reach` (0.3): 파지 간극을 물체 위로. (물체가 손가락 사이에 오는가)
- `finger_cage_hold` (1.0): 오므려라. (손가락 사이 공간이 물체를 파고드는가)
- `palm_facing` (0.5): 손바닥을 물체 쪽으로. (오므리는 것이 물리적으로 가능한 자세인가)
- `cube_lift` (3.0): 들어라. (하중을 견디는가 = 진짜 파지인가)

### 희소(sparse) vs 조밀(dense) 보상

```text
희소 :  성공했을 때만 값이 있음.  gradient 없음.  우연히 성공해야 학습 시작.  해킹 불가
조밀 :  상태에 따라 연속적으로 변함.  gradient 있음.  학습됨.  해킹당함
```

- 논문 처방: **조밀은 유도용, 희소는 성공 정의용.** 둘 다 필요함.
- 우리 `cube_lift`는 조밀형임. 희소형이면 영원히 `0`이라 학습이 시작조차 안 됨.
- **이 프로젝트에서 겪은 거의 모든 문제가 조밀 보상의 부작용이었음.**

### 겪은 reward hacking 목록 (전부 조밀 보상의 부작용)

- **dawdle**: progress reward의 `distance_max`가 작아 상시 포화 -> "천천히 접근하면 step마다 만점" -> 도착 안 함.
- **swing-out**: `mode="best"` + `clamp(min=0)` -> 후퇴가 공짜이고 첫 step에 멀어지면 기준선이 부풀려짐 -> 팔을 크게 휘두름.
- **엄지만 박기**: "손끝 -> 물체 중심" 거리 + 엄지 가중치 3배 -> 엄지 하나만 넣는 것이 최적해.
- **hover**: 거리 reward가 접촉을 처벌 -> 만지지 않는 법을 학습.
- **기울이기**: `cube_lift`가 중심 높이를 봄 -> 큐브를 모서리로 세워 중심만 올림 (최하 모서리는 바닥).
- **손바닥 하늘**: cage 선분이 손 방향과 무관하게 물체를 관통 -> 손바닥이 하늘을 봐도 만점.

- **공통점: 정책이 reward를 정직하게 최대화했는데 그게 우리가 원한 게 아니었음.**

### orientation 판단 기준

```text
큐브   :  orientation 목표 없음.  기능 요구가 없으므로 어떤 목표 회전도 자의적임
젓가락 :  orientation 목표 있음.  기능이 손 회전을 물리적으로 결정함
```

- **기준은 "물체가 대칭이냐"가 아니라 "기능 요구가 있느냐"임.**
- 논문이 `r_hr`(목표 손 회전)을 주는 이유는 드릴을 **"트리거를 당길 수 있게"** 쥐어야 하기 때문임.
- 그냥 집어 올리기만 할 거라면 `g_r`이 필요 없음.
- **목표 회전은 "잡기"가 아니라 "쓰기"에서 나옴.**

### `palm_facing`은 왜 자의적이지 않은가

- **손가락은 손바닥 쪽으로 굽음. 손바닥 뒤에 있는 물체는 오므려도 안 감싸짐.**
- 즉 "이렇게 잡아라"가 아니라 **"잡는 것이 가능한 자세여야 한다"**임. 스타일이 아니라 **전제조건**임.
- 손바닥 법선 축만 정렬하고 **roll은 자유**임. 대칭 물체의 파지 방식을 고르지 않음.
- 논문 `r_hr`(회전 3자유도 전부)에서 **물리적으로 반드시 필요한 성분 1개**만 남긴 것임.
- 젓가락에서는 `r_hr`이 상위호환 교체함. **버려지는 것이 아니라 승격됨.**

## 치명적인 주의사항 (2026-07-12 추가)

- **`Metrics/cube/*`(에피소드 평균)로 성능을 판단하지 말 것.** 앞 4 step(이동)이 평균의 77%를 지배함. `Metrics/cube_final/*`을 볼 것.
- **과거 run 분석 시 `logs/rsl_rl/<exp>/<run>/params/env.yaml`을 읽을 것.** 현재 코드의 파라미터를 대입하면 틀린 결론이 나옴.
- **reward가 오르는데 실제 지표가 나빠지면 즉시 reward hacking을 의심할 것.**
- **지표가 좋은데 GUI가 이상하면 지표를 의심할 것.** 손바닥 하늘과 기울이기 편법 모두 지표상 정상이었고 육안으로 잡혔음.
- **reward를 넣기 전에 "물리적으로 가능한가"부터 확인할 것.** 못 하는 것을 요구하면 물리와 싸울 뿐임.
- **높이 기반 reward는 중심이 아니라 최하점 기준으로.** 중심 높이는 기울이기로 올릴 수 있음.
- **action scale은 도달 반경임.** `target = default + scale * action`인 절대 위치 명령이라, jitter를 줄이려고 낮추면 팔이 물체에 못 닿음. jitter는 `action_rate` weight로 잡을 것.
- **progress reward는 `mode="previous"` + `clamp(min=-1)` + `reset()` seeding, 셋을 다 해야 함.** 하나라도 빠지면 swing-out이 남음.
- **`managers.py`의 `_cage_body_names`/`_palm_normal_b`는 reward cfg와 반드시 동일해야 함.**
- **정책이 이상하게 행동하면 정책이 틀린 것이 아니라 reward가 틀린 것임.**

## Reward 형태: 차분(differential) vs 절대(absolute) — 2026-07-13 확립

### 핵심 원칙

```text
절대 양수 + 유지가 쉬움    ->  반드시 farming당함
절대 양수 + 유지가 어려움  ->  안전. 유지가 곧 과제의 목표
절대 페널티 (<=0)          ->  안전. 최대가 0이라 쌓을 것이 없음
차분                       ->  안전. 가만히 있으면 0
```

- **새 reward를 넣을 때 "이건 유도인가 유지인가", "가장 싼 만족 방법이 뭔가"를 먼저 물을 것.**

### 논문은 거의 전부 차분형임

- `r_hp`(Eq.9), `r_hr`(Eq.10, 손 회전), `r_hj`(Eq.11), `r_reach`(Eq.14), `r_orient`(Eq.16): **전부 차분.**
- `r_hold`(Eq.15): **절대. 논문에서 유일함.**
- 논문: "Wide use of differential distances in our reward, instead of directly using the velocities, naturally avoids learning overshooting behaviors."
- **차분형은 "가만히 있으면 0"이라 farming이 불가능함.** 그래서 논문은 weight `1.0`을 줘도 안전함.

### 실패 사례

- `palm_facing`을 **절대형**으로 넣었더니 (weight `0.5`) 전체 보상의 **`98.6%`**를 차지함.
- 정책이 큐브 **31cm 밖에서 손바닥만 겨누고 정지**함. 접근이 순손실이었기 때문임.
- 게다가 **팔을 접어서 손목만 돌리는 것**이 가장 싼 만족 방법이라, manipulability가 `13%`까지 추락함.
- **가중치를 낮추는 것이 아니라 형태를 바꾸는 것이 정답이었음.**

## 논문의 가중치와 우리 방침

```text
논문:  r = r_grasp + r_reach + 25*r_hold + 500*r_orient + r_MP + 5000*r_T
              1        1         25          500          1       5000
       approach(1) -> hold(25) -> orient(500) -> grasp(5000)   (각 단계 약 20배)
```

- 논문: "The exact values do not affect learning significantly, **as long as the overall proportions reflect the logical sequence**."
- **절대값이 아니라 비율만 중요함. scale은 우리 보상값의 실제 크기에 맞춰 역산할 것.**
- **차분형은 telescoping되어 1회만 지급되고, 절대형은 20 step 누적됨.** 규모가 근본적으로 다르므로 **에피소드당 기여량**으로 환산해서 비교해야 함.
- 논문 경고: "the policy is greedily maximizing reward through, for example, **holding an object, rather than exploring more difficult and failure-prone orienting** of the object if it were yielding less reward."

## Curriculum이 필수인 이유 — 2026-07-13

### 과제가 사슬임

```text
접근 -> 손바닥 정렬 -> 감싸기 -> 오므리기 -> 들기
                                              ^
                                    앞의 4개를 다 해야만 보상이 나옴
```

- 무작위 탐색으로 사슬 전체를 완성할 확률이 사실상 `0`임.
- 마지막 보상을 **한 번도 경험하지 못하면** 그 가치가 value function에 전파되지 않음.
- **정책은 자기가 도달할 수 있는 가장 뒤쪽 링크에서 멈춤.**
- 지금까지 관찰한 모든 정체가 이것임 (hover -> 엄지 찌르기 -> cage -> 서서 겨누기).

### Curriculum은 사슬을 거꾸로 배움

```text
close-start:  물체가 이미 손 안/옆에 있음 -> 마지막 보상이 도달 가능해짐 -> "드는 것이 가치있다"를 학습
Stage 2:  물체를 멀리 둠              -> 접근에 목적이 생김           -> value가 뒤에서 앞으로 전파
```

- 논문 데이터: curriculum 없으면 성공률 약 `50%`에 편차 폭발, 있으면 **`97%`** (wall-clock 동일).

### 결정적 사실: `cube_lift` 보상은 한 번도 나온 적이 없음

- 전 학습 기간 동안 `Episode_Reward_Raw/cube_lift` = **정확히 `0`**. 큐브가 바닥에서 떨어진 적이 없음.
- **`0`인 보상은 가중치를 아무리 올려도 `0`임.**
- 원인: 최하 모서리 기준으로 바꿔 기울이기 편법을 막았더니 **사실상 희소해짐.** 정직해진 대신 도달 불가능해짐.
- **가중치를 조정하기 전에 "그 보상이 실제로 발생한 적이 있는지"부터 확인할 것.**

### 보상 단계화도 curriculum임

```text
논문:   환경을 단계화   물체를 손 옆에 놓음   -> 파지가 도달 가능해짐
대안:   보상을 단계화   접근/파지를 먼저 학습 -> 그 상태에서 lift가 도달 가능해짐
```

- 보상 단계화가 더 저렴함. 환경을 안 건드리므로 warm-start가 깔끔함.
- **성립 조건**: Phase 1 수렴 자세에서 lift가 **탐색으로 도달 가능**해야 함. 아니면 `0 x W = 0`임.
- 넘어가기 전에 **"강제로 오므리고 팔을 올리면 큐브가 따라오는가"**를 반드시 검증할 것.

---

## action space 설계 — `scale`과 `clip_actions`는 한 쌍임 (2026-07-13)

### 왜 정책 출력에 상한이 없나
PPO 정책은 가우시안 `N(mu, sigma)`이고, `mu`는 MLP의 마지막 `nn.Linear` 출력임.
**tanh도 sigmoid도 없으므로 수학적으로 상한이 없음.** 학습하다 보면 `mu`가 1.5, 5, 10도 됨.
(`rsl_rl/modules/distribution.py:169-179`)

### `scale`이 곧 "도달 반경"임
```
관절 목표 = default_joint_pos + scale x action     ← 절대 위치 명령
```
`scale`은 "부드럽게 하는 손잡이"가 아님. **`|a|=1`일 때 관절이 얼마나 움직이는가**임.

### 그래서 둘은 한 쌍임
```
scale 작음 + clip 없음   ->  정책이 |a|를 키워서 보상하려 함  ->  발산      (2026-07-13 이전)
scale 작음 + clip 있음   ->  관절이 조금밖에 못 움직임        ->  도달 불가
scale 큼   + clip 없음   ->  여전히 상한 없음                ->  발산
scale 큼   + clip 있음   ->  정상                                          (현재)
```

### 설계 절차
1. **과제가 요구하는 관절 이동량을 먼저 측정한다.**
   (우리: `default ± s` 범위에서 샘플링 -> `s=0.5`면 큐브에 3.63cm까지밖에 못 감. `s=1.0`이면 0.20cm)
2. `scale = s` 로 잡는다.
3. `clip_actions = 1.0` 을 건다.
4. 이제 `|a| ∈ [-1,1]`이 의미를 가지고, 관절 목표는 `default ± scale`에 갇힌다.

### 증상으로 알아보는 법
```
Episode_Reward_Raw/action_rate      = sum (a_t - a_{t-1})^2.  이미 찍히고 있음 (공짜 진단기)
   167 (20 step) -> |Δa| ~ 0.68  ->  action 범위가 [-1,1]인데 매 step 0.68씩 튐 = 발산
```
**`|a| > 1`이 상시로 나오면 무조건 뭔가 잘못된 것.** action은 `[-1,1]` 안에서 의미를 가져야 함.

---

## 논문의 순서 강제 방식 — **게이팅이 아니라 가중치 스케일링** (2026-07-13, 원문 확인)

### 논문 원문
> "For best performance, we scale the rewards according to their position in the sequence of
> interconnected sub-tasks: **r_T >> r_orient >> r_hold >> r_reach**. **This reduces the probability
> that the policy gets stuck in the local minima, created by accumulating rewards for actions that
> are easier to achieve compared to the following more complex sub-tasks.**"

**우리가 겪은 문제를 그 문장이 정확히 서술함:** "쉬운 앞 단계 보상을 긁어모으다 local minima에 갇힘."

### 실제 가중치
```
r_reach   x 1
r_hold    x 25          <- reach의 25배
r_orient  x 500
r_T       x 5,000
```
> "Both scaling factors are chosen to be one order of magnitude less than the final reward and the
> corresponding reward component with higher scaling."  (단계마다 약 20배씩)

### 방향은 `r_hr`로 직접 보상 — **논문은 목표 파지 자세를 안다**
```
r_grasp = r_hp + r_hr + lambda * r_hj          (Eq. 8)
           ^      ^       ^
        손 위치  손 회전  손 관절   <- 전부 "주어진 목표 파지 g"를 향한 차분 보상
```
논문은 `g = (hp, hr, hj)`를 외부(oracle/데이터셋)에서 받음. `r_hr`이 "그 목표 회전으로 돌려라"를
**직접** 보상함. **게이팅이 필요 없음 — 답을 알고 있으니까.**

**우리는 목표 파지 자세가 없음. 이게 본질적 차이이고, 우리 과제가 더 어려움.**

### 커리큘럼 (Section IV-D)
> "In the first stage, we place the objects in poses where target functional grasps can be reached
> directly. Objects are positioned on the table in their nominal poses **5 cm away from the inner
> side of the hand**. **The arm is set to a neutral configuration with a high manipulability score.**
> We **disable the r_man** reward term during the first stage. The first stage continues until at
> least a **50% success rate** is achieved."

**close-start에서 `r_man`(= r_reach + r_hold + r_orient)을 통째로 끔.** 물체가 이미 손 안에 있으니
접근할 필요가 없고, `r_grasp`(목표 자세로 가기)와 `r_T`(도달 성공)만 학습함.

### 가상점(cage) — 논문은 **엄지-중지만, 6점**
> "For simplicity, we use only the points between the thumb and the middle-finger in this work,
> **yielding six points**. The intuition is that **if an object is contained between the middle finger
> and the thumb, it is also contained between the index and ring fingers, as defined by the hand
> topology.** While it is straightforward to utilize several finger pairs at the same time, we
> observed in practice that using only the middle-thumb line was sufficient."

우리는 엄지-검지/중지 4선분 12점. **논문도 "여러 쌍 써도 된다"고 명시하므로 해롭지 않음.**
단 논문의 위상 논리는 "손가락이 교차하지 않음"을 전제함 (우리 Wuji 손은 교차했었음).

### 엄지의 지분은 줄일 수 없음 (구조적, 버그 아님)
```
point = thumb + (opposing - thumb) x f,   f = 0.25, 0.50, 0.75

d(point)/d(thumb)    = 1-f = 0.75, 0.50, 0.25   -> 평균 0.5
d(point)/d(opposing) =   f = 0.25, 0.50, 0.75   -> 평균 0.5      (대칭!)

엄지는 4개 선분 전부의 시작점 -> 12개 점 전부에 영향 -> 총 6.0
손가락 하나는 자기 선분에만   ->  3개 점에만 영향   -> 총 1.5
=> 엄지 6.0 : 나머지 4개 합 6.0 = 50:50
```
**논문(엄지-중지 6점)도 엄지 3.0 : 나머지 3.0 = 50:50으로 동일. 선분을 줄여도 안 바뀜.**
엄지가 파지의 절반인 건 설계임. (`hold`는 엄지만으로 절대 0 — 해석적으로 검증함)

---

## 우리의 선택: **게이팅 + 가중치 (B안)** — 2026-07-13

### 왜 게이팅을 추가했나
`reach`가 차분형이라 **속도와 무관하게 총액이 같은데**, `hold`/`lift`는 절대형이라 **일찍 도착할수록
오래 쌓임** -> **정책이 서두르는 게 합리적임.** 그런데 서둘러 도착하면 방향이 틀린 채로 큐브에
막혀서 손을 못 돌림 (실측). **속도 페널티로는 못 고침 — 방향이 아니라 속도만 줄어들 뿐.**

```python
reach = facing x delta(cage_sdf)      # 방향이 안 맞으면 접근해도 보상 0
```
**속도를 제한하는 게 아니라 "잘못된 순서"를 무가치하게 만듦.**
논문의 `r_hr`(목표 회전으로 돌리기)을 우리는 쓸 수 없으므로(목표가 없으니) **그 자리의 대용품임.**

### 왜 논문의 25배를 그대로 안 쓰나
```
논문:  r_hp (손을 "아는" 목표 위치로) x 1   <- 별도의 강한 접근 신호가 있음
       r_hold x 25
       + close-start에서 물체를 손 5cm 앞에 스폰 (warm start)

우리:  접근 신호가 reach 하나뿐.  목표 위치도 모름.  커리큘럼도 없음
```
**reach를 4로 낮추면 접근 신호가 거의 사라지는데 hold는 가까이 가야만 켜짐 -> 닭-달걀.**
논문은 커리큘럼으로 회피함. 우리는 없으므로 **한 단계씩**: `hold/reach` 4.0 -> 10 (논문 25 방향).

### 현재 가중치 (최대 획득량 = Episode_Reward_Raw = 에피소드 합 / 8초)
```
항목                형태     weight    최대 획득량
palm_facing         차분      8.0        0.033
finger_cage_reach   차분     10.0        0.050    x facing 게이트
finger_cage_hold    절대      1.0        0.500
cube_lift           절대      3.0        0.750
arm_manipulability  페널티    1.0
hand_floor          페널티    0.5
action_rate                 -0.005

순서: palm(0.033) < reach(0.050) < hold(0.500) < lift(0.750)   <- 논문 논리
hold / reach = 10.0     (이전 4.0,  논문 25)
```

**규칙: `palm 최대 < reach 최대`.** 넘기면 겨누기가 접근보다 비싸져 standoff
(weight 60에서 palm 0.24 > reach 0.13 -> cage_inside가 0.20 -> 0.000으로 붕괴함).

### 다음 단계
- **젓가락 과제에선 논문대로 갈 것** — 젓가락은 목표 파지 자세가 정의되므로 `r_hp`/`r_hr`/`r_hj`/`r_T`를
  전부 쓸 수 있고, 그러면 게이팅 없이 논문의 25/500/5000 스케일이 그대로 성립함.
- 지금 과제(큐브)는 목표 파지가 없으므로 게이팅으로 대체.

---

## 계획 B (게이팅이 실패하면): **논문을 제대로 구현한다** — 2026-07-13 확정

### 사용자 판단
> "이번에 안 되면 어차피 젓가락 할 때는 회전 고려해서 논문처럼 할 거니까 그냥 그렇게 하자.
>  cube도 직육면체로 만들어서 orientation까지 고려하고, 학습도 커리큘럼 러닝으로 가자."

### 왜 옳은가: 지금 겪는 문제가 **전부 "목표 파지 자세가 없어서"** 생긴 것임

| 지금 겪는 문제 | 논문에선 왜 안 생기나 |
|---|---|
| 방향을 안 잡고 달려듦 | **`r_hr`**이 "목표 회전으로 돌려라"를 직접 보상 |
| palm_facing 축을 뭘로 잡을지 헤맴 | **목표 회전이 주어짐.** 축을 추측할 필요가 없음 |
| 게이팅이라는 임시방편 | **필요 없음.** `r_grasp`가 처음부터 방향을 끌고 감 |
| hold가 안 켜져 닭-달걀 | **커리큘럼** (물체를 손 5cm 앞에 스폰) |
| 가중치 줄타기 (palm < reach < hold) | **논문 값 그대로** (1 / 25 / 500 / 5000) |

**대칭인 정육면체에서는 목표 파지 회전을 정의할 수 없었음.** 직육면체로 바꾸면 해석적으로 정의됨
(짧은 축을 가로질러 잡고, 손바닥이 긴 면을 향하게).

**그리고 `r_orient`가 진짜 의미를 가짐.** 직육면체가 누워 있으면 그대로는 못 잡으니 **먼저 세워야** 함
— **그게 논문 제목의 "pre-grasp manipulation"임.** 정육면체로는 논문의 절반을 못 쓰고 있었음.

### 만들 것
```
1. 물체        직육면체 (예: 0.04 x 0.06 x 0.10)
2. 목표 파지 g  = (hp, hr, hj)          <- 박스 치수에서 해석적으로 계산
3. r_grasp     = r_hp + r_hr + lambda*r_hj      (Eq. 8~12)
4. r_orient    물체를 nominal 자세로             (Eq. 16)
5. r_T         목표 파지 도달 성공 (희소)          (Eq. 18)
6. r_lift                                       (Eq. 20)
7. 커리큘럼     close-start: 물체를 손 5cm 앞 + 팔은 manip 높은 중립 자세 + r_man 끔
                       -> 성공률 50%까지
               stage 2: 전체 난이도
8. 가중치       r_reach 1 / r_hold 25 / r_orient 500 / r_T 5000    <- 논문 그대로
9. 게이팅 제거   r_hr이 대신함
```

**이미 있는 것:** `r_reach`(cage 차분), `r_hold`(cage), `r_MP`, `r_lift`, cage 가상점, box SDF, 지표 전부
**새로 필요한 것:** `r_grasp` 3항(`r_hp`/`r_hr`/`r_hj`), `r_orient`, `r_T`, 목표 파지 정의, 커리큘럼

### 젓가락과의 연결
**젓가락은 목표 파지 자세가 본질적으로 정의됨** (검지를 특정 위치에, EE 회전을 특정 방향으로).
논문의 constraint-based 표현이 정확히 그것임:
> "we represent the target grasp as a 3D target position of the index fingertip and the end-effector
> rotation relative to the object."

**즉 직육면체 파지는 젓가락으로 가는 정석 경로임.** 지금 큐브에서 하는 임시방편(게이팅, palm_facing
축 추측)은 젓가락에선 전부 버려짐.

## 2026-07-24 — 젓가락질 전체 Phase 로드맵 (연구 방향)

> 큐브→박스→1스틱 functional grasp를 거쳐 확정한 최종 연구 방향. 상세·논문 근거는
> `ACTIVITY_2026-07-23.md`, 실험 이력은 `worklog.md`.

**Phase 1 — 젓가락 획득 + functional grasp 형성** (현재 단계)
- 완료 조건: 두 젓가락 획득 → 손바닥 위 정렬 → 각 손가락이 지정 젓가락의 지정 영역 접촉 →
  아래 stick 안정 유지 → 위 stick 개폐 → 반복 open-close에도 안 미끄러지고 안 떨어짐.
- topology 가설: 엄지·검지·중지→Stick1, 약지→Stick2, 새끼 보조(스타일 A/B/C).
- 내부 3단계로 분리(한 reward sum 금지): STATE A 획득+staging / B standard grip 형성 / C open-close.
  각 단계 성공을 latch(0.5~1s 연속)로 전환, 이후 이전 reward 끄고 성공 bonus 1회(파밍 차단).
- 구현: **방식 2(단계별 별도 환경/curriculum) 먼저 → 나중에 FSM 결합.** FSM 한 번에는
  horizon·reward 충돌·파밍 위험. 단계 하나가 아직 취약(액션 구조 하나로 파지 성패가 갈린 실측).
- BO(논문 C 축소): surface_axis/sign 고정, 손가락별 **axial center만** sweep → 유망 범위서 BO.

**Phase 2 — 젓가락으로 물체 집기** (나중)
- tip으로 물체 파지 → 들기 → 이동 → **큰 목표 공간(바구니)에 놓기**.
- ⚠ **정밀 pick&place 아님.** 자세 정렬 불필요 → 목표항은 위치 φ만으로 충분.
  keypoint/쿼터니안 자세매칭은 Phase 1 staging에서나 필요. Phase 2 병목은 tip 파지+open-close 유지.

**목표항 방식 정리 (성공 전제 품질 비교)**
- keypoint(pos·ori 융합): 정밀·단일항·튜닝 편함. 단 목표 대칭이 물체마다 다르면 over-constrain.
- 분리형(위치 φ + 자세 게이팅): 필요한 축만·진단성·물체별 유연. 항이 많아 밸런싱 어려움.
- → 접근/운반엔 keypoint 성향(선형 견인), 말단 정밀엔 분리형 성향. Phase 1 staging은 keypoint 적합.

**진행 순서**: 1-stick 고정 pose 수렴 → 랜덤 pose 도달성 검증(2-stick 전제) → STATE A → B → C.

**참고 논문**: SimToolReal(keypoint goal-pose, best-so-far, I_grasped 전후 gate 분리),
Dexterous Pre-grasp Manipulation(functional pre-grasp 상태로 도구 이동),
Learning to Use Chopsticks in Diverse Gripping Styles 2205.14313
(gripping style로 손가락-젓가락 담당 지정, axial 접촉위치 BO 탐색, open-close 평가, r_contact=−w·Σdᵢ).

## 2026-07-25 — Phase 1 재정의(주먹 파지) + 성공-래치 커리큘럼 수식

> 위 로드맵의 Phase 1(2-stick functional grasp)을 1-stick 관점으로 구체화. A/B·파지 진단으로
> 확정한 실사용 계획. 상세 실험 이력은 `ACTIVITY_2026-07-25.md`, `worklog.md`.

**Phase 1 종료 상태(확정)**: 주먹(5-finger wrap) 파지 → goal로 transport → **palm-up 자세 매칭**.
- wrap은 Phase 1 목표 자체 (tool grasp는 Phase 2). 나중에 tool로 가도 "다시 배우기"가 아니라
  성공-래치로 보상만 전환 → unlearning 최소.

**A/B 결론(정정) — keypoint는 고정 pose에서 검증됨, 방법 우열은 미결**:
- ⚠ 앞서 "랜덤 pose에서 keypoint/쿼터니안 우열"로 판단하려던 시도는 **confound 투성이라 폐기**.
  - chopstick 쿼터니안 "34° 벽"은 방법이 아니라 **grip 오설정 아티팩트**였음. 엄지·중지 x부호를
    창발 파지에 맞추자(엄지−x/중지+x) 같은 쿼터니안이 ori min 4°까지 내려감(08-47-53).
  - box keypoint "정체"도 방법이 아니라 **랜덤 pose(1.57~3.14)** 때문. **고정 pose(09-43-43)에선
    keypoint가 err_pos 4cm·ori 3°로 수렴, success 상승** — 잘 됨. ("kp_raw 0.0003=신호죽음"은
    오판; 잘 된 09-43-43도 동일 raw. best-so-far 텔레스코핑이라 raw 작은 게 정상.)
- **핵심**: 랜덤 자세매칭은 **두 Phase 어디도 요구 안 함**(Phase 1=palm-up 고정, Phase 2=위치만).
  즉 실제 과제는 keypoint가 강한 고정-pose 영역. 랜덤 우열 비교는 요구 밖이라 의미 축소.
- 방향: **keypoint(box)를 주력 라인으로**, 파지를 주먹(wrap)으로. chopstick(쿼터니안+tripod)은
  known-good 상태로 park. (2026-07-25 코드: box에 penta wrap 적용, chopstick wrap 철회·복원.)

**파지 자세 진단 (왜 자세가 안 다듬어지나)**:
- grip 보상(FingertipGripProgressReward, progress형)은 에피소드 예산 ~1.3점, cube_lift(gate형)는
  240~800점. **grip 완성 보상이 lift의 ~1/500** → 자세를 독립적으로 당기는 신호가 사실상 없음.
  자세는 오직 `lift×gate` 결합으로만 형성됨. 위험 = "느슨-lift 국소최적"(gate 0.3에서 안주).
- 개입 기준: **느슨-lift가 실제 드롭(stick_dropped)을 유발할 때만** gate를 sharp하게(`gate^k`).
  드롭 안 나면 자세 완벽화는 과잉공학.

**성공-래치 커리큘럼 수식 (Phase 전환, fresh-run 규칙 준수)**:
- env i마다 래치 `L[i]∈{0,1}`: `s1[i]`=Phase1 성공조건, `L ← max(L, s1)` (에피소드 리셋 때 0).
- `R[i] = (1−L[i])·R_phase1[i] + L[i]·R_phase2[i]`.
- **리워드 함수가 하나로 연속** → resume 아님(fresh-run 규칙 안 어김), critic 충격은 env별 성공
  시점이 staggered라 배치 평균이 매끄러움. 하드 래치 걱정되면 `L ← clamp(L + s1/K, 0, 1)` 램프.
- `GoalReachedBonus._awarded` 래치 패턴 재사용 가능.

**진행**: 1-stick wrap+palm-up 수렴 → (성공-래치로) Phase 2 tool grasp 보상 전환.
