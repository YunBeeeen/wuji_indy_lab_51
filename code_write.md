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
