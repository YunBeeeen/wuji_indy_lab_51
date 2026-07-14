# code_write.md

- 이 문서는 코드 변경 의도와 구현 방향을 날짜별로 간단히 남기는 코드 작성 메모 문서임.

## 2026-07-10

- 목표는 `Indy-Wuji-Reach` baseline을 보존하면서 cube grasp용 task skeleton을 추가하는 것임.
- 초기 reward는 arm-to-cube reach baseline에서 시작함.
- 최신 reward는 reach -> lift -> lifted object-goal tracking 구조까지 연결함.
- contact 기반 grasp reward는 아직 보류함.
- 먼저 robot + cube + 초기 cube reach reward가 같은 task에서 도는 smoke test를 목표로 함.

## Added Task Structure

- 새 grasp package를 추가함.
- 경로는 `nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/grasp/`임.
- `cube_grasp_env_cfg.py`를 추가함.
- `grasp/indy_wuji/env_cfg.py`를 추가함.
- `grasp/indy_wuji/__init__.py`에서 `Indy-Wuji-Cube-Grasp` task id를 등록함.
- `tasks/manipulation/__init__.py`에서 `grasp` package도 import되도록 연결함.

## CubeGrasp Base

- `CubeGraspSceneCfg`는 기존 `ReachSceneCfg`를 상속함.
- 기존 ground, robot placeholder, contact sensor, light 구조를 재사용함.
- `CubeGraspSceneCfg`에는 cube만 추가함.
- cube는 `RigidObjectCfg`로 추가함.
- cube prim path는 `{ENV_REGEX_NS}/Cube`임.
- 병렬 env마다 cube가 하나씩 생기도록 `{ENV_REGEX_NS}` 아래에 둠.

## Cube Parameters

- cube size는 `0.06 m` 정육면체로 정함.
- 기존 `0.05 m`보다 초기 grasp/contact smoke test에 조금 더 관대함.
- Wuji finger 길이가 대략 10 cm급이라 6 cm cube가 초기 cube grasp baseline에 더 적절하다고 판단함.
- cube mass는 `0.08 kg`으로 정함.
- cube initial position은 `(0.45, -0.18, 0.03)`으로 정함.
- `z=0.03`은 cube half-size와 맞춰 ground 위에 놓기 위한 값임.
- `x=0.45`, `y=-0.18`은 기존 Indy reach workspace 중심에 가깝게 둔 값임.
- 4096 env long run에서 PhysX patch buffer overflow가 발생해 `gpu_max_rigid_patch_count`를 `2**19`로 올림.
- `2**18`은 2026-07-10 resume run의 약 `263k` patch 요구치에 살짝 부족했음.

## Indy Wuji Cube Grasp Env

- `Indy7WujiCubeGraspEnvCfg`는 `CubeGraspEnvCfg`를 상속함.
- robot asset은 `INDY7_WUJI_RIGHT_CFG`를 사용함.
- 현재 action은 arm 6축 + thumb/index/middle 12축을 사용함.
- action joint는 `joint[0-5]`, `finger[1-3]_joint[1-4]`임.
- action dim은 18임.
- active command는 없음.
- policy observation은 controlled joint position 18D, cube-palm relative position 3D, five-fingertip relative position 15D, cube-goal relative position 3D, previous action 18D임.
- policy observation dim은 57임.
- cube relative position은 `palm_link` 기준 cube root position임.
- cube-fingertip relative position은 thumb/index/middle/ring/little tip 기준 cube root position임.
- cube-goal relative position은 fixed target `(0.55, -0.05, 0.12)` 기준임.
- 초기 reward는 `palm_link`와 cube root position 거리 기반 `arm_cube_reach`임.
- fingertip reach reward는 `finger_cube_reach`임.
- `finger_cube_reach`는 controllable thumb/index/middle tip을 쓰고 thumb weight를 더 크게 둠.
- `finger_cube_reach`는 논문식 progress reward로 바꿈.
- `finger_cube_reach`는 current distance absolute reward가 아니라 best-so-far distance 개선량을 봄.
- `finger_cube_reach` body weight는 `3.0/1.0/1.0`임.
- `finger_cube_reach` weight는 초기 학습 확인용으로 `0.3`임.
- `finger_cube_reach` distance_max는 progress 정규화 scale로 `0.5`임.
- `functional_hold`를 추가함.
- `functional_hold`는 Dexterous Functional Grasp 참고용 hold/cage reward임.
- `functional_hold`는 cube가 thumb/index/middle fingertip region에 들어오는지 봄.
- `functional_hold` weight는 `0.2`임.
- `arm_cube_reach`는 coarse guide로 남기고 weight를 낮춤.
- `cube_lift`는 cube root z height 기반 bounded reward임.
- `cube_goal_tracking`은 cube z가 `0.08 m` 이상일 때만 켜지는 lifted-gated target reward임.
- `cube_goal_tracking` target은 `(0.55, -0.05, 0.12)`임.
- `action_rate` penalty weight는 초기 학습 확인용으로 `-0.0003`임.
- contact 기반 grasp reward는 아직 넣지 않음.

## Finger Grouping Decision

- Wuji hand는 5개 finger group으로 보는 방향이 적절함.
- 사람이 읽는 이름은 다음 alias로 정리함.
- `finger1`은 `thumb`으로 취급함.
- `finger2`는 `index`로 취급함.
- `finger3`은 `middle`로 취급함.
- `finger4`는 `ring`으로 취급함.
- `finger5`는 `little`로 취급함.
- 코드상의 joint/body 이름은 당장은 URDF/USD 이름인 `finger[1-5]_joint[1-4]`, `finger[1-5]_link[1-4]`, `finger[1-5]_tip_link`를 그대로 사용함.
- reward/contact 설계 단계에서만 thumb/index/middle/ring/little alias를 사용함.

## Smoke Test Command

- GUI smoke test 명령은 다음임.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/train.py --task Indy-Wuji-Cube-Grasp --num_envs 1 --max_iterations 1
```

- headless smoke test 명령은 다음임.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/train.py --task Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1
```

## Verification

- `indy.py`에 잘못 붙은 finger actuator 조각 때문에 syntax error가 났음.
- 잘못 끼어든 조각을 제거함.
- 실제 finger actuator 정의는 이미 `INDY7_WUJI_RIGHT_CFG.actuators["fingers"]`에 있음.
- `py_compile`로 다음 파일 문법 확인함.
- `isaac_neuromeka/assets/indy.py` 확인함.
- `grasp/cube_grasp_env_cfg.py` 확인함.
- `grasp/indy_wuji/env_cfg.py` 확인함.
- `grasp/indy_wuji/__init__.py` 확인함.
- `grasp/indy_wuji/learning/rsl_rl_cfg.py` 확인함.
- `mdp/observations.py` 확인함.
- `mdp/rewards.py` 확인함.
- `env_cfg_common.py` 확인함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- smoke test에서 active command term 0개 확인함.
- smoke test에서 action shape 6 확인함.
- smoke test에서 policy observation shape 15 확인함.
- smoke test에서 observation term은 `joint_pos`, `cube_pos`, `action_history`로 확인함.
- smoke test에서 reward term은 `arm_cube_reach`, `action_rate`로 확인함.
- 중간에 26D action 구조까지 확장했으나 논문식 구현 우선순위에 맞춰 18D action으로 되돌림.
- 18D action + 57D observation + lift/goal reward 구조 smoke test를 실행함.
- reward 완화 후 `env_cfg_common.py` 문법 확인은 통과함.
- PhysX patch buffer 변경 후 `cube_grasp_env_cfg.py` 문법 확인은 통과함.
- TensorBoard cube distance error metric을 `CustomRewardManager`에 추가함.
- 추가 metric은 `Metrics/cube/palm_distance`, `thumb_distance`, `index_distance`, `middle_distance`, `ring_distance`, `little_distance`, `finger_mean_distance`, `non_thumb_mean_distance`, `finger_weighted_mean_distance`임.
- metric은 reward 계산에는 들어가지 않고 logging만 함.
- metric 추가 후 `managers.py` 문법 확인은 통과함.
- 최신 smoke test에서 active command term 0개 확인함.
- 최신 smoke test에서 action shape 18 확인함.
- 최신 smoke test에서 policy observation shape 57 확인함.
- progress reward 단계 smoke test에서 reward term은 `arm_cube_reach`, `finger_cube_reach`, `cube_lift`, `cube_goal_tracking`, `action_rate`로 확인함.
- 최신 smoke test log dir은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_21-17-55`임.
- `finger_cube_reach`를 논문식 best-so-far progress reward로 변경함.
- `mdp.BodiesToObjectProgressReward`를 추가함.
- `BodiesToObjectProgressReward`는 per-env previous/best distance buffer를 가짐.
- 현재 cfg는 `mode="best"`를 사용함.
- reset/first step에서 best distance를 현재 distance로 초기화함.
- progress reward 변경 후 `py_compile` 확인 통과함.
- progress reward 변경 후 smoke test 통과함.
- progress reward smoke test log dir은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_21-31-26`임.
- Dexterous Functional Grasp 참고용 `functional_hold` reward를 추가함.
- `functional_hold` 추가 후 smoke test에서 reward term 6개 확인함.
- `functional_hold` 추가 후 smoke test log dir은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_21-41-01`임.

## Next

- GUI smoke test로 robot과 cube가 동시에 보이는지 확인함.
- cube가 너무 작거나 너무 멀면 scene 위치를 다시 조정함.
- contact group reward를 추가함.
- contact-gated lift/goal reward로 바꿈.
- 이후 object pose/keypoint reward와 chopstick/tool-use trajectory reward로 확장함.

## 2026-07-11

- cube grasp reward를 Dexterous Pre-grasp Manipulation 논문 방식으로 전면 재설계함.

### 왜 갈아엎었는가

- 기존 reward는 전부 "손끝 -> 큐브 **중심**" 거리였음.
- 큐브 중심은 표면에서 `0.03 m` 안쪽이라 **손끝이 물리적으로 도달할 수 없는 목표**임.
- gradient가 항상 큐브 속을 향하고, `body_weights=(3,1,1)`로 엄지가 가중평균의 60%를 차지함.
- 따라서 가장 싼 해법이 **"엄지 하나만 큐브에 박고 나머지는 방치"**가 됨. 실측으로 확인함.
- 거리 reward는 접촉도 처벌함. 만지면 큐브가 밀려나 거리가 늘어남.
- 즉 **파지할수록 손해인 구조**였음.

### 무엇으로 바꿨는가

- 목표를 큐브 **표면(SDF)**으로 바꿈. 도달 가능한 목표가 됨.
- 손끝 개별 점이 아니라 **엄지-손가락 사이 파지 간극**을 대상으로 함.
- `object_in_finger_cage` (Eq.15): 가상점이 큐브 **내부**로 파고든 깊이를 보상함. **오므리기가 직접 보상됨.**
- `ObjectCageProgressReward` (Eq.14): **같은 가상점**의 표면까지 SDF의 차분. 파지 간극을 큐브 위로 끌어옴.
- **부호가 반대임.** 거리 reward는 접촉을 처벌하고, cage reward는 접촉을 보상함. 이것이 핵심임.

### progress reward 규칙 (셋을 다 해야 함)

- `mode="previous"` + `clamp(min=-1)` + `reset()`에서 기준선 seeding.
- `clamp(min=0)`이면 후퇴가 공짜이고, 기준선을 첫 `__call__`에서 잡으면 첫 액션이 기준선을 공짜로 부풀림 (swing-out).
- 셋을 다 하면 총합이 `d(reset) - d(final)`로 telescoping됨.

### 삭제한 함수 (전부 미참조 확인 후)

- `bodies_to_object_position_tracking_bounded`, `body_to_object_position_tracking_bounded`.
- `object_in_functional_grasp_region`, `BodiesToObjectProgressReward`.
- `CubeGraspRewardsCfg` docstring에 삭제 사유를 남김. 같은 실수 반복 방지용임.

## 2026-07-12

### 가상점 6 -> 12 (검지 추가)

- 6점 cage로 학습한 결과 지표는 완벽했으나 GUI에서 **검지·중지가 교차하고 손바닥이 하늘을 향함**.
- 원인: cage가 "엄지끝 ↔ 중지"만 봄. **검지는 reward에 등장조차 하지 않았음.**
- 논문은 6점으로 충분했으나, 논문에는 `r_grasp`(`r_hr` + `r_hj`)가 손 회전과 손가락 관절각을 붙잡고 있었음.
- 큐브는 목표 파지 자세가 없어 `r_grasp`를 못 씀. 그래서 검지가 완전히 자유였음.
- `_cage_points` -> `cage_points`로 일반화함. 정확히 3개 body 요구 -> `[thumb, *opposing]` N개 허용.
- `CAGE_BODIES`에 `finger2_tip_link`, `finger2_link3` 추가. 대향 body 4개 x 3점 = 12점.
- **reward 수식은 변경 없음.** 점 개수만 늘고 평균을 냄.
- 엄지+검지+중지는 젓가락 그립과 동일하므로 임시방편이 아님.

### `cube_lift`를 최하 모서리 기준으로 (기울이기 편법 차단)

- 첫 버전은 큐브 **중심 높이**를 봤음.
- 정책이 **손가락 2개로 큐브를 눌러 모서리로 세우는** 편법을 학습함.
- 실측: 중심 `+4.28 mm` 올라갔는데 **최하 모서리는 `-0.04 mm`로 바닥에 붙어 있었음.**
- `box_ground_clearance` 추가함. 큐브 8개 꼭짓점을 world로 변환해 최소 z를 구함.
- 기울이면 한 모서리가 바닥이므로 최소 z = 0 -> 보상 0. **편법이 원천 차단됨.**
- metric에 `cube_clearance`(진짜 lift)를 추가하고 기존 `cube_lift`(중심)도 유지함.
- **두 값이 벌어지면 기울이기 재발 신호임.**

### `palm_facing` 도입

- 12점 cage로도 손바닥 방향을 못 잡음. 실측 `palm_facing = 0.182`인데 `cage_inside_frac = 0.753`.
- 원인: **엄지끝-손가락 선분은 손 방향과 무관하게 큐브를 관통할 수 있음.**
- 그래서 손바닥이 하늘을 봐도 cage가 만점이 나옴.
- **자의적 제약이 아니라 물리적 필요조건임.** 손가락은 손바닥 쪽으로 굽으므로, 손바닥 뒤의 물체는 오므려도 안 감싸짐.
- 손바닥 법선 축만 정렬하고 **roll은 자유**임. 대칭 물체의 파지 방식을 고르지 않음.
- 손바닥 법선은 `palm_link` 로컬 `+x`임. **손가락을 오므릴 때 손끝이 이동하는 방향으로 실측함.** 추측이 아님.
- **넣기 전에 kinematic 도달성부터 확인함.** 팔 관절 40만 개 샘플링 결과 손바닥 정면(`+1.000`)이 도달 가능함.
- 다만 `0.14%`로 드물어 명시적 보상 없이는 무작위 탐색으로 못 찾음.

### 최종 reward

- `finger_cage_reach` (0.3), `finger_cage_hold` (1.0), `cube_lift` (3.0), `palm_facing` (0.5), `action_rate` (-0.0003).
- action shape 18, observation shape 57 불변.

### 이번 작업에서 배운 것

- **정책이 이상하게 행동하면 정책이 아니라 reward가 틀린 것임.** 모든 문제가 그러했음.
- **지표가 좋은데 GUI가 이상하면 지표를 의심할 것.** 손바닥 하늘과 기울이기 편법 모두 지표상 정상이었고 육안으로 잡혔음.
- **reward를 넣기 전에 "물리적으로 가능한가"부터 확인할 것.** 못 하는 것을 요구하면 물리와 싸울 뿐임.
- **높이 기반 reward는 중심이 아니라 최하점 기준으로.** 중심 높이는 기울이기로 올릴 수 있음.

## 2026-07-13

### `arm_manipulability` 추가 (논문 Eq.17, weight `1.0`)

- `palm_facing`(절대형) 도입 후 정책이 큐브 31cm 밖에서 손바닥만 겨누고 정지함.
- 사용자 GUI 관찰로 **팔이 접혀 특이점에 빠진 것**을 발견함.
- manipulability 실측: 초기 자세 `0.0645` (`57%`) -> 수렴 자세 **`0.0144` (`13%`)**. `joint2`가 한계의 `81%`까지 접힘.
- 인과: `palm_facing`을 만족시키는 **가장 싼 방법이 "팔을 접어서 손목만 돌리기"**였음. 팔을 뻗는 것보다 훨씬 쉬움.
- `r = 1 - 2 / (1 + (min(|J|, j_max)/j_max)^3)`, 범위 `[-1, 0]`, `j_max = 0.02`.
- `|J| = sqrt(det(J J^T))`, `palm_link` 기준 arm 6축 Jacobian.
- 신규 metric `arm_manipulability` 추가. 기준: 초기 `0.064`, 무작위 최대 `0.113`, `0.02` 아래면 특이점.

### `palm_facing`을 차분형으로 교체 (`PalmFacingProgressReward`)

- 사용자가 "논문과 가중치 비율이 다르다"고 지적한 것에서 시작함.
- 확인 결과 **가중치가 아니라 "형태"가 달랐음.**
- **논문의 거의 모든 항이 차분형임**: `r_hp`(Eq.9), `r_hr`(Eq.10), `r_hj`(Eq.11), `r_reach`(Eq.14), `r_orient`(Eq.16).
- **절대형은 `r_hold`(Eq.15) 하나뿐임.**
- 논문: "Wide use of differential distances in our reward... naturally avoids learning overshooting behaviors."
- **차분형은 "가만히 있으면 0"이라 farming이 불가능함.** 그래서 논문은 weight `1.0`을 줘도 안전함.
- 우리 `palm_facing`은 절대형이라 정렬만 유지해도 매 step 보상 -> weight `0.5`에서도 전체의 `98.6%`를 먹음.
- **가중치를 낮추는 것이 아니라 형태를 바꾸는 것이 정답이었음.**
- `r(t) = facing(t) - facing(t-1)`, `reset()`에서 기준선 seeding, `clamp(-1, 1)`.
- `palm_facing_object`는 절대형 그대로 두고 **metric 전용**으로 씀.

### reward 형태 선택 원칙 (확립)

```text
절대 양수 + 유지가 쉬움    ->  반드시 farming당함
절대 양수 + 유지가 어려움  ->  안전. 유지가 곧 목표
절대 페널티 (<=0)          ->  안전. 최대가 0이라 쌓을 것이 없음
차분                       ->  안전. 가만히 있으면 0
```

- **새 reward를 넣을 때 "이건 유도인가 유지인가", "가장 싼 만족 방법이 뭔가"를 먼저 물을 것.**

### 결정적 발견: `cube_lift` 보상이 한 번도 나온 적이 없음

- 전 학습 기간 동안 `Episode_Reward_Raw/cube_lift` = **정확히 `0`**.
- 큐브가 단 한 번도 바닥에서 떨어진 적이 없음 (`cube_clearance` 최대 `60um` = 노이즈).
- **`0`인 보상은 가중치가 `3`이든 `500`이든 무의미함.**
- 원인: `cube_lift`를 **최하 모서리** 기준으로 바꿔 기울이기 편법을 막았더니 **사실상 희소해짐.**
- 정직해진 대신 **도달 불가능**해짐. 희소 보상은 curriculum 없이 학습 불가능함.
- **가중치를 조정하기 전에 "그 보상이 실제로 발생한 적이 있는지"부터 확인할 것.**

### 다음 단계 (보상 단계화 curriculum)

- **Phase 1** (진행 중): 접근 + 손바닥 정렬 + 감싸기 학습. `cube_lift`는 `0`이라 사실상 없는 것과 같음.
- **수렴 후 검증**: 그 자세에서 손가락을 강제로 오므리고 팔을 올렸을 때 큐브가 따라오는가?
- 따라오면 -> lift가 탐색으로 도달 가능 -> **Phase 2** (`cube_lift` 가중치 상향 + `--resume`).
- 안 따라오면 -> 환경 curriculum (Stage-1 arm 자세) 필요.
- 가중치는 **논문 숫자가 아니라 우리 raw 값 기준으로 역산**할 것. 논문도 "exact values do not affect learning"이라 명시함.

### `Indy-Wuji-Cube-Grasp-Easy` 추가

- hard task `Indy-Wuji-Cube-Grasp`는 유지함.
- easy curriculum task `Indy-Wuji-Cube-Grasp-Easy`를 추가함.
- 구현 위치: `isaac_neuromeka/tasks/manipulation/grasp/indy_wuji/env_cfg.py`.
- gym 등록 위치: `isaac_neuromeka/tasks/manipulation/grasp/indy_wuji/__init__.py`.
- easy는 hard cfg를 상속하고 cube 초기/reset 위치만 바꿈.
- hard/easy 모두 action shape `18`, observation shape `57`, reward/model shape 동일함.
- easy cube 위치는 `(0.74, -0.18, 0.03)`임.
- easy reset randomization은 `x/y ±0.015`, `z=0`임.
- 목적은 `finger_cage_hold`, `cage_inside_frac`, `cube_lift`가 켜질 수 있는 가까운 초기 상태를 먼저 학습하는 것임.

### hand/cube contact response 완화

- ring/little을 접은 뒤에도 palm/hand contact에서 arm이 튀는 영상이 나옴.
- active `INDY7_WUJI_RIGHT_CFG`의 contact response를 완화한 상태를 확인함.
- `max_depenetration_velocity`: `1000.0` -> `5.0`.
- `max_contact_impulse`: `1e32` -> `100.0`.
- 작은 관통을 PhysX가 과격하게 해소하면서 arm 전체가 튀는 상황을 줄이려는 변경임.
- action/observation/reward/model shape 불변임.
- checkpoint load/resume 가능함. 단 physics가 바뀌므로 최종 평가는 재학습 또는 resume adaptation 후 판단함.
