# WORKLOG.md

- 이 문서는 현재 프로젝트 상태, 변경 이력, 실행 결과, 다음 할 일을 남기는 작업 로그 문서임.

## Current Status

- 현재 목표는 Indy7 + Wuji hand asset으로 arm end-effector tracking RL env 구성임.
- 현재 active workspace는 `~/wuji_indy_lab_51`임.
- 현재 active code dir은 `~/wuji_indy_lab_51/nrmk_isaaclab_wuji`임.
- 현재 active conda env는 `env_isaaclab`임.
- Isaac Sim 5.1 환경 사용함.
- IsaacLab 2.3 계열 환경 사용함.
- `Indy-Wuji-Reach` task 생성됨.
- Neuromeka `Indy-Reach` 스타일로 구현됨.
- 현재 task는 headless 학습과 GUI 실행 모두 확인됨.
- 현재는 full training 전 arm-only tracking 안정성 확인 단계임.
- 현재는 `link6` arm flange 기준 position-only reach baseline으로 long-run 확인하는 단계임.
- 최종 조작 목표는 젓가락을 기능적으로 잡고 젓가락질이 가능한 파지 상태를 학습하는 것임.
- `Dexterous Pre-grasp.pdf` 계열 functional grasp/pre-grasp 논문 흐름을 주 목표로 봄.
- 현재 policy observation shape는 15임.
- 현재 action shape는 6임.
- 현재 active reward는 position tracking과 action rate penalty임.
- virtual EE offset 코드는 실험 후 되돌리고 보류함.
- cube grasp active task는 `Indy-Wuji-Cube-Grasp`임.
- cube grasp는 최종 목표가 아니라 functional grasp/chopstick grasp로 가기 위한 proxy task임.
- cube grasp는 functional hold/cage -> contact condition -> contact-gated lift -> object goal/keypoint tracking 흐름으로 정리 중임.
- cube grasp action shape는 18임.
- cube grasp policy observation shape는 57임.

## Important Decisions

- Isaac Sim 4.5 / IsaacLab 2.2.1 맞추기 실험 중단함.
- `chop_ws/chop_rl` 기반 실험 폐기함.
- Neuromeka public/main branch 스타일 우선함.
- DirectRLEnv from scratch보다 Neuromeka ManagerBasedRLEnv 스타일 우선함.
- 기존 `Indy-Reach`를 덮어쓰지 않고 `indy_wuji` 새 task로 분리함.
- 초기 학습은 arm 6축만 action으로 사용함.
- hand joint는 articulation에 남기되 policy action에는 넣지 않음.
- arm tracking 단계에서는 hand joint observation 제외함.
- 현재 tracking 기준 rigid body는 `link6` 사용함.
- `palm_link` tracking은 frame mismatch 확인 후 보류함.
- `palm_link`는 실제 articulation rigid body로 사용 가능함.
- `tcp`는 USD articulation body로 tracking에 쓰기 부적합하다고 판단함.
- `link6` baseline을 먼저 길게 학습시켜 보고 Wuji hand frame offset/reward 수정 여부를 판단함.
- virtual EE offset 방식은 후보로 보류함.

## Asset Status

- active robot cfg는 `INDY7_WUJI_RIGHT_CFG`임.
- active USD는 `isaac_neuromeka/assets/model/usd/indy7_wuji_right/indy7_wuji_right_simplified.usd`임.
- `indy7_wuji_right_simplified.usd`에 Wuji hand collision mesh 26개 적용함.
- Wuji collision mesh는 `*_collision.STL`에서 가져옴.
- 직접 USD Mesh collider로 삽입함.
- collider prim에 `PhysicsCollisionAPI` 등 적용함.
- active nested collision leftover는 제거/비활성 처리함.
- validation 결과 hand collision mesh 26개 확인함.
- GUI에서 collision visualization 관련 초록/빨간 표시 차이는 보기 모드/선택 상태 이슈로 판단함.
- Isaac Sim Fabric GPU 모드는 정상으로 판단함.
- `Properties not updated`는 Fabric 모드의 GUI property sync 안내 수준으로 판단함.

## Code Status

- `isaac_neuromeka/assets/indy.py` 수정됨.
- `INDY7_WUJI_RIGHT_CFG` 추가/사용됨.
- `Indy-Wuji-Reach` 등록됨.
- `indy_wuji/env_cfg.py` 구현됨.
- `indy_wuji/learning/rsl_rl_cfg.py` 구성됨.
- `train.py`, `play.py`는 rsl-rl 5 계열 config migration 대응됨.
- `indy_wuji`는 base/teacher/student/CMDP cfg 골자 갖춤.
- 현재 active registration은 `Indy-Wuji-Reach` 하나임.
- teacher/student/CMDP registration은 future block으로 둠.
- arm-only observation 적용됨.
- `policy` observation shape는 175 -> 55 -> 19 -> 15로 감소함.
- actor/critic input feature는 현재 15임.
- action shape는 6임.
- command는 공통 `UniformPoseCommandCfg` 기준임.
- command 자체는 7D pose command지만 policy observation에는 xyz position 3D만 넣음.
- reward는 현재 position tracking과 action rate penalty만 사용함.
- 현재 reward body는 `link6`임.
- offset command/reward 코드는 제거함.
- 실행 흐름 공부 문서는 root `flow_study.md`에 정리함.
- `Indy-Wuji-Reach` entry point를 `CustomManagerBasedRLEnv`로 바꿈.
- TensorBoard에 weighted reward와 raw unweighted reward를 같이 기록하도록 `CustomRewardManager`를 적용함.
- orientation tracking, end-effector speed, joint velocity reward term은 현재 position-only baseline에서 제거함.

## Frame Status

- raw `palm_link` orientation tracking은 error가 약 2 rad 근처로 남았음.
- `link6` tracking은 orientation error가 약 0.96 수준으로 확인됨.
- `palm_link` command range를 임시로 바꾸며 fixed offset 후보를 확인함.
- `roll=-pi/2`, `pitch=-pi/2`, `yaw=(-3.14, 3.14)` 조합에서 orientation error가 약 0.18까지 낮아짐.
- 이 결과로 Wuji palm frame과 Indy reach command frame 사이 고정 회전 offset이 있다고 판단함.
- 이후 virtual EE frame offset을 구현해 smoke test까지 확인함.
- 그러나 long-run baseline을 먼저 보기 위해 offset 구현은 되돌리고 보류함.
- 이후 약 2000 iteration 학습에서 orientation error가 2점대에서 약 0.8까지 하락함.
- `palm_link`와 `link6` 모두 학습이 진행되면 orientation error가 유사하게 내려가는 것으로 봄.
- 따라서 URDF/offset 문제가 주 원인이라고 단정하지 않기로 함.
- 현재는 학습 시간과 reward 구조 영향이 컸던 것으로 판단함.
- 현재 command range는 공통 reach 기본값인 `roll=0`, `pitch=pi`, `yaw=(-3.14, 3.14)` 기준임.

## Verified Runs

- `--num_envs 1 --max_iterations 1` smoke test 통과함.
- `--num_envs 32 --max_iterations 5` headless test 통과함.
- GUI 실행 확인함.
- GUI에서 robot asset 보임.
- GUI에서 Fabric GPU로 실행됨.
- `--num_envs 1 --max_iterations 1` arm-only observation 변경 후 재검증함.
- arm-only observation 재검증 결과 action shape 6 유지됨.
- arm-only observation 재검증 결과 policy observation shape 55 확인됨.
- 사용자가 `--num_envs 128 --max_iterations 20` 실행함.
- `512 env / 100 iterations` run에서 `model_99.pt` checkpoint 생성됨.
- `play.py`로 checkpoint 재생 확인됨.
- position-only observation/reward 변경 후 `--num_envs 1 --max_iterations 1` smoke test 통과함.
- position-only smoke test 결과 actor/critic input 15 확인함.
- position-only smoke test 결과 active reward log는 position tracking과 action rate만 확인함.
- TensorBoard raw reward log `Episode_Reward_Raw/*` 확인함.

## Latest Smoke Test Result

- 최신 position-only smoke test 결과임.
- 사용자 실행 command는 `--num_envs 1 --max_iterations 1` 기준임.
- actor model input feature는 15로 확인함.
- critic model input feature는 15로 확인함.
- action output은 6으로 확인함.
- `Episode_Reward/end_effector_position_tracking` 확인함.
- `Episode_Reward_Raw/end_effector_position_tracking` 확인함.
- `Episode_Reward/action_rate` 확인함.
- `Episode_Reward_Raw/action_rate` 확인함.
- `Metrics/ee_pose/position_error` 확인함.
- orientation error metric은 command manager metric으로 남아 있지만 reward에는 사용하지 않음.
- 1 PPO iteration 완료함.

## Historical Smoke Test Result

- command 사용함.

```bash
conda run -n env_isaaclab python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 1 --max_iterations 1
```

- action manager shape 6 확인함.
- observation manager policy shape 55 확인함.
- `joint_pos` shape 18 확인함.
- `joint_vel` shape 18 확인함.
- `pose_command` shape 7 확인함.
- `action_history` shape 12 확인함.
- actor model input 55 확인함.
- critic model input 55 확인함.
- 1 PPO iteration 완료함.
- virtual EE frame 적용 후 smoke test 재실행함.
- command term type은 `OffsetUniformPoseCommand`로 확인됨.
- action shape는 6 유지됨.
- policy observation shape는 55 유지됨.
- `Metrics/ee_pose/orientation_error`는 0.5433으로 확인됨.
- 1 PPO iteration 완료함.
- 이후 offset 구현을 제거하고 raw `palm_link` baseline으로 되돌림.
- raw `palm_link` baseline 복귀 후 smoke test 재실행함.
- command term type은 `UniformPoseCommand`로 확인됨.
- action shape는 6 유지됨.
- policy observation shape는 55 유지됨.
- `Metrics/ee_pose/orientation_error`는 0.9658로 확인됨.
- 1 PPO iteration 완료함.
- 이후 reach 예제 학습 성공 확인을 위해 tracking body를 `link6`로 변경함.
- `link6` 기준 smoke test 재실행함.
- command term type은 `UniformPoseCommand`로 확인됨.
- action shape는 6 유지됨.
- policy observation shape는 55 유지됨.
- `Metrics/ee_pose/position_error`는 0.1382로 확인됨.
- `Metrics/ee_pose/orientation_error`는 0.9658로 확인됨.
- 1 PPO iteration 완료함.

## Known Warnings

- rsl-rl `policy` config deprecated warning 있음.
- migration handler로 actor/critic 자동 유도됨.
- `distribution_cfg` warning 있음.
- `rsl_rl` package git repo 못 찾는 warning 있음.
- 학습에는 치명적이지 않음.
- Fabric point instancer warning 있음.
- command visualization 관련 warning으로 판단함.
- PhysX actuator `effort_limit`/`velocity_limit` deprecation warning 있음.
- 추후 `effort_limit_sim`, `velocity_limit_sim`로 정리 가능함.

## Next Steps

- `4096 env / 50000 iterations` 학습 실행함.
- TensorBoard로 학습 추세 확인함.
- `Metrics/ee_pose/position_error` 확인함.
- `Episode_Reward/end_effector_position_tracking` 확인함.
- `Episode_Reward_Raw/end_effector_position_tracking` 확인함.
- `Episode_Reward/action_rate` 확인함.
- `Episode_Reward_Raw/action_rate` 확인함.
- `Mean reward` 확인함.
- NaN/PhysX error 여부 확인함.
- GUI play에서 `link6` reach 움직임 확인함.
- `link6` position-only baseline이 안정되면 Wuji `palm_link` tracking 또는 virtual EE offset을 다시 검토함.

## Change Log

- Initial handoff docs 생성함.
- Neuromeka Indy-Reach 기반 구조 확인함.
- Indy-Wuji 새 task 구성함.
- Wuji hand collision 26개 복구함.
- active USD를 `indy7_wuji_right_simplified.usd`로 설정함.
- `tcp` 대신 `palm_link` tracking 적용함.
- RSL-RL train/play config migration 대응함.
- GUI 실행 확인함.
- arm-only observation 적용함.
- `WORKLOG.md`, `AGENTS.md`, `study.md` 최신화함.
- `play.py`의 pretrained checkpoint import 경로를 IsaacLab 2.3.2 환경에 맞게 호환 수정함.
- 최신 확인 checkpoint는 `logs/rsl_rl/indy_wuji_reach/2026-07-08_18-16-06/model_99.pt`임.
- `play.py`의 rsl-rl 5 policy export 경로도 수정함.
- rsl-rl 5에서는 `runner.alg.policy` / `actor_critic` 대신 `runner.export_policy_to_jit()` / `runner.export_policy_to_onnx()` 사용함.
- 2 step headless play test 통과함.
- `CLI.md` 생성함.
- 학습/GUI/play/resume 명령을 일반화해서 정리함.
- `CLI.md`를 shell 변수 없이 바로 복붙 실행 가능한 명령 형식으로 변경함.
- Wuji EE orientation frame mismatch 원인 확인함.
- `palm_link` 기반 virtual EE frame 구현함.
- `OffsetUniformPoseCommandCfg` 추가함.
- offset-aware reward 함수 추가함.
- Indy-Wuji task가 virtual EE frame 기준 command/reward를 쓰도록 수정함.
- virtual EE frame 적용 후 `--num_envs 1 --max_iterations 1` smoke test 통과함.
- 사용자 판단에 따라 virtual EE offset 코드를 되돌림.
- raw `palm_link` baseline으로 장시간 학습 후 offset/reward 수정 여부를 판단하기로 함.
- raw `palm_link` baseline 복귀 후 `--num_envs 1 --max_iterations 1` smoke test 통과함.
- reach 구조 확인을 위해 `Indy-Wuji-Reach` command/reward body를 `link6`로 변경함.
- `link6` 기준 `--num_envs 1 --max_iterations 1` smoke test 통과함.
- `flow_study.md` 생성함.
- 실행 흐름, task registration, env cfg 조립, command/action/observation/reward 연결, 핵심 코드 발췌를 개괄식으로 정리함.
- `CustomRewardManager`가 `Episode_Reward/*`, `Episode_Reward_Raw/*`, `Episode_Reward_Std/*`를 기록하도록 수정함.
- `Indy-Wuji-Reach`가 custom reward manager를 쓰도록 custom env entry point로 변경함.
- policy observation을 15차원으로 축소함.
- command observation을 position xyz만 보도록 변경함.
- orientation reward, end-effector speed reward, joint velocity reward를 제거함.
- position-only smoke test 통과함.
- `CLI.md`의 play 명령에서 고정 `--load_run 2026-07-08_18-16-06` 예시를 제거하고 최신 run 자동 선택 방식으로 변경함.
- 2026-07-09 reward 논문 공부 내용을 정리함.
- 현재 cube grasp 구현은 point cloud/force sensor 없이 oracle state 기반 reward/success부터 시작하기로 판단함.
- oracle state는 sim이 알고 있는 cube pose, fingertip pose, contact, lift height, velocity 같은 정답 상태를 뜻함.
- policy observation에 넣으면 oracle observation이고, reward/success 계산에만 쓰면 oracle reward/success condition으로 구분함.
- `palm_link`-cube 중심 거리만으로 grasp를 정의하는 것은 부족하다고 정리함.
- grasp success는 contact group, lift threshold, object velocity 안정성을 같이 보는 쪽으로 정리함.
- functional grasp/pre-grasp 논문 흐름을 cube grasp/chopstick grasp의 주 목표로 정리함.
- DexPoint는 cube grasp baseline의 구현 목표가 아니라 보조 참고 자료로 정리함.
- DexPoint에서 참고할 핵심은 fingertip reach, contact reward, contact-gated lift reward, lift 이후 target reward, action/velocity/controller penalty임.
- DexPoint 논문식 contact는 thumb contact + other fingers 2개 이상이고, 공개 코드식 완화 조건은 finger/palm contact group count 2개 이상임.
- 초기 `indy_wuji_right` cube grasp에서는 완화 조건으로 시작하고, 학습 후 thumb + non-thumb 조건으로 강화하는 방향을 기록함.
- TriFinger transfer 논문은 grasp보다 object 6-DoF pose tracking reward 참고로 분류함.
- TriFinger object reward는 cube/object 8개 keypoint current-target distance로 position과 orientation을 함께 다룸.
- TriFinger reach reward는 현재 거리 보상이 아니라 `curr_dist - prev_dist` 접근 progress reward이며, 후반 curriculum으로 끄는 구조임.
- TriFinger fingertip velocity penalty는 손가락 움직임 안정화용으로 정리함.
- SimToolReal은 tool-use/chopstick 단계 참고로 분류함.
- SimToolReal reward 구조는 `r = r_smooth + r_grasp + I_grasped * r_goal`이고, lift 이후 object-centric goal pose progress가 주도함.
- 현재 논문별 역할은 Functional/Pre-grasp = 기능적 파지와 object 재배치의 주 기준, DexPoint = contact/lift gate 보조 참고, TriFinger = 잡은 뒤 object pose reward 참고, SimToolReal = tool-use trajectory 참고로 정리함.
- 추천 구현 순서는 functional hold/cage baseline, contact condition, contact-gated lift, object keypoint/goal reward, chopstick/tool trajectory reward 순서임.
- 2026-07-10 Dexterous Pre-grasp Manipulation 논문 reward 내용을 정리함.
- 이 논문은 functional grasp 전 object를 reposition/reorient/regrasp하는 pre-grasp manipulation 주 목표 논문으로 분류함.
- cube grasp는 이 논문 흐름을 젓가락 파지로 확장하기 전의 proxy task로 기록함.
- 핵심 reward는 `r_grasp + r_man + r_MP + r_T`이고, constraint-based 방식에서는 `r_lift`가 추가됨.
- `r_man = r_reach + r_hold + r_orient`가 가장 중요한 manipulation reward로 정리함.
- `r_hold`는 object가 thumb-finger 사이 공간에 들어오도록 유도하는 cage-like reward라서 Wuji hand grasp에 유용한 아이디어로 기록함.
- explicit target grasp는 object 기준 EE pose와 hand joint target을 주는 방식이고, 성능은 높지만 물체별 target 정의 부담이 큼.
- constraint-based target grasp는 index fingertip target position과 EE orientation 같은 기능 조건을 쓰며, fake success 방지를 위해 lift 조건이 필요함.
- curriculum은 nominal pose/direct grasp에서 시작해 다양한 object pose의 full pre-grasp manipulation으로 확장하는 구조로 정리함.
- 업데이트된 논문별 역할은 Pre-grasp/Functional grasp = 주 목표, DexPoint = contact/lift gate 보조 참고, TriFinger = object pose control 참고, SimToolReal = tool-use trajectory 확장 참고임.
- 업데이트된 구현 순서는 Pre-grasp식 hold/cage reward, contact condition, contact-gated lift, TriFinger식 keypoint object-goal reward, SimToolReal식 tool trajectory reward 순서임.
- 논문 내용만 따로 모은 root `thesis.md`를 생성함.
- `thesis.md`에는 DexPoint, TriFinger transfer, Dexterous Pre-grasp Manipulation, SimToolReal의 목적/reward/takeaway를 정리함.
- cube grasp debug 확장보다 논문식 구현 흐름을 우선하기로 정리함.
- cube grasp action을 five-finger 26D에서 arm + thumb/index/middle 18D로 되돌림.
- ring/little은 당장 action/reward 핵심에서 제외하고 observation/metric 보조로 남김.
- `cube_to_goal` observation을 추가함.
- 최신 cube grasp policy observation은 joint position 18D, cube relative position 3D, five-fingertip relative position 15D, cube goal relative position 3D, previous action 18D로 총 57D임.
- `finger_cube_reach` reward를 controllable thumb/index/middle 기준으로 다시 좁힘.
- 최신 `finger_cube_reach` body weight는 `(3.0, 1.0, 1.0)`임.
- `cube_lift` reward를 추가함.
- `cube_goal_tracking` reward를 추가함.
- `cube_goal_tracking`은 cube z가 `0.08 m` 이상일 때만 켜지는 lifted-gated target reward임.
- 현재 cube target position은 `(0.55, -0.05, 0.12)`임.
- 당시 active reward term은 `arm_cube_reach`, `finger_cube_reach`, `cube_lift`, `cube_goal_tracking`, `action_rate`임.
- `py_compile` 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- 당시 smoke test에서 action shape 18, policy observation shape 57, reward term 5개 확인함.
- smoke test log dir은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_21-17-55`임.
- `thesis.md`에는 프로젝트 구현 상태와 실행 명령을 제외하고 논문별 reward 내용만 남김.
- 2026-07-10 cube grasp task skeleton을 추가함.
- 새 task id는 `Indy-Wuji-Cube-Grasp`임.
- `CubeGraspSceneCfg`는 기존 `ReachSceneCfg`를 상속하고 cube만 추가함.
- cube는 `{ENV_REGEX_NS}/Cube`에 `RigidObjectCfg`로 추가함.
- cube size는 `0.06 m`, mass는 `0.08 kg`, initial position은 `(0.45, -0.18, 0.03)`으로 정함.
- cube grasp RSL-RL experiment name을 `indy_wuji_cube_grasp`로 분리함.
- Wuji finger alias를 `finger1=thumb`, `finger2=index`, `finger3=middle`, `finger4=ring`, `finger5=little`로 정리함.
- `code_write.md`를 생성해 2026-07-10 코드 작성 내용을 개괄식으로 기록함.
- `indy.py`에 잘못 끼어든 finger actuator 조각을 제거하고 syntax error를 복구함.
- cube grasp용 초기 observation/reward 계산 함수를 연결함.
- `mdp.object_position_relative`를 추가해 `palm_link` 기준 cube relative position을 observation으로 넣음.
- `mdp.body_to_object_position_tracking_bounded`를 추가해 body-cube 거리 기반 bounded reward를 계산함.
- `CubeGraspObservationsCfg` policy group은 `joint_pos`, `cube_pos`, `action_history`로 구성함.
- `CubeGraspRewardsCfg`는 `arm_cube_reach`와 `action_rate`만 active로 둠.
- `Indy-Wuji-Cube-Grasp`는 active command 없이 cube state 기반 reach baseline으로 시작함.
- `py_compile`로 cube grasp 관련 cfg와 mdp 파일 문법 확인함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- smoke test에서 action shape 6, policy observation shape 15 확인함.
- smoke test에서 active reward term은 `arm_cube_reach`, `action_rate`로 확인함.
- smoke test log dir은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_16-00-02`임.
- cube grasp action을 arm 6축 + thumb/index/middle 12축으로 확장함.
- cube grasp action dim은 18이 될 예정임.
- controlled joint regex는 `joint[0-5]`, `finger[1-3]_joint[1-4]`임.
- `mdp.object_position_relative_to_bodies`를 추가해 cube 위치를 thumb/index/middle fingertip 기준 상대 벡터 9D로 observation에 넣음.
- `mdp.bodies_to_object_position_tracking_bounded`를 추가해 여러 body-cube 거리 reward를 한 term으로 계산함.
- `CubeGraspObservationsCfg`에 `cube_in_fingertips` observation term을 추가함.
- `CubeGraspRewardsCfg`에 `finger_cube_reach` reward term을 추가함.
- `finger_cube_reach`는 thumb/index/middle tip을 사용하고 body weight는 `(2.0, 1.0, 1.0)`임.
- `arm_cube_reach` weight는 coarse guide 용도로 `0.05`로 낮춤.
- 당시 18D 구조의 예상 policy observation dim은 48이었음.
- 당시 18D 구조의 Isaac smoke test는 사용자가 실행 예정이었음.
- `py_compile`로 수정 파일 문법 확인함.
- `finger_cube_reach` raw reward가 너무 낮아 초기 학습 signal 확보용으로 reward를 완화함.
- `finger_cube_reach` weight를 `0.2`에서 `0.3`으로 올림.
- `finger_cube_reach` `distance_max`를 `0.3`에서 `0.5`로 넓힘.
- `action_rate` weight를 `-0.001`에서 `-0.0003`으로 낮춤.
- 완화 후 `env_cfg_common.py` 문법 확인 통과함.
- `4096 env / 50000 iter` cube grasp run 중 PhysX patch buffer overflow가 발생함.
- 로그 요구치는 약 `171k` patch count였고 기본값은 부족했음.
- `CubeGraspEnvCfg.__post_init__`에서 `self.sim.physx.gpu_max_rigid_patch_count = 2**18`로 설정함.
- patch buffer 변경 후 `cube_grasp_env_cfg.py` 문법 확인 통과함.
- TensorBoard에 cube 실제 거리 error metric을 추가함.
- `CustomRewardManager`에서 scene에 `robot`과 `cube`가 있으면 cube metric logging을 켬.
- 추가 metric은 `Metrics/cube/palm_distance`, `thumb_distance`, `index_distance`, `middle_distance`, `finger_mean_distance`임.
- 이 metric들은 reward가 아니라 palm/fingertip과 cube root 사이 실제 거리 확인용임.
- `managers.py` 문법 확인 통과함.
- `distance_max=0.15` resume run 중 PhysX patch buffer 요구치가 약 `263k`까지 올라감.
- `2**18 = 262144`로는 부족해 `gpu_max_rigid_patch_count`를 `2**19`로 올림.
- 이 변경은 새 train/resume 실행부터 반영됨.
- 중간 실험으로 cube grasp control을 arm + five fingers로 확장함.
- 당시 controlled joint regex는 `joint[0-5]`, `finger[1-5]_joint[1-4]`였음.
- 당시 action dim은 26이었음.
- 당시 `cube_in_fingertips` observation은 five-fingertip 기준 15D였음.
- 당시 policy observation dim은 `26 + 3 + 15 + 26 = 70`이었음.
- `finger_cube_reach` body_names를 finger1~5 tip으로 확장함.
- `finger_cube_reach` body weight는 `(3.0, 1.0, 1.0, 1.0, 1.0)`임.
- TensorBoard cube metric을 five-finger 기준으로 확장함.
- 추가 metric은 `ring_distance`, `little_distance`, `non_thumb_mean_distance`, `finger_weighted_mean_distance`임.
- metric은 `CustomRewardManager`에서 episode 평균 거리로 기록함.
- cube grasp `finger_cube_reach`를 논문식 progress reward로 변경함.
- 새 reward class는 `mdp.BodiesToObjectProgressReward`임.
- `BodiesToObjectProgressReward`는 per-env `previous_distance`, `best_distance`를 유지함.
- 현재 cfg는 `mode="best"`를 사용함.
- reward 계산은 `progress = previous_best_distance - current_distance` 흐름임.
- `current_distance`는 thumb/index/middle fingertip과 cube root 거리의 weighted average임.
- `finger_cube_reach` body weight는 `(3.0, 1.0, 1.0)` 유지함.
- `distance_max=0.5`는 progress 정규화 scale로 사용함.
- 첫 step/reset 직후에는 best distance를 현재 거리로 초기화해서 raw reward가 0임.
- `Episode_Reward_Raw/finger_cube_reach`는 현재 거리 자체가 아니라 최단거리 갱신량임.
- 실제 접근 거리는 `Metrics/cube/*`로 확인함.
- `py_compile`로 `mdp/rewards.py`, `env_cfg_common.py` 문법 확인함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- progress reward smoke test log dir은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_21-31-26`임.
- Dexterous Functional Grasp 참고용 `functional_hold` reward를 추가함.
- `functional_hold`는 cube가 thumb/index/middle fingertip region 안쪽에 들어오는지 보는 hold/cage reward임.
- `functional_hold`는 weighted fingertip-cube closeness, thumb-opposition, cube-grasp-center closeness를 조합함.
- `functional_hold` body weight는 `(3.0, 1.0, 1.0)`임.
- `functional_hold` weight는 `0.2`임.
- `functional_hold` distance_max와 center_distance_max는 `0.18`임.
- `functional_hold` 추가 후 active cube grasp reward term은 6개임.
- active term은 `arm_cube_reach`, `finger_cube_reach`, `functional_hold`, `cube_lift`, `cube_goal_tracking`, `action_rate`임.
- `py_compile`로 `mdp/rewards.py`, `env_cfg_common.py` 문법 확인함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- smoke test에서 action shape 18, policy observation shape 57, reward term 6개 확인함.
- random policy smoke test에서 `functional_hold Raw`는 0이었고, 이는 cube가 아직 grasp region 안에 들어오지 않았다는 뜻임.
- `functional_hold` smoke test log dir은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_21-41-01`임.
- cube grasp long run 전 lift shortcut 방지를 위해 `cube_lift`와 `cube_goal_tracking` reward를 비활성화함.
- observation/action shape 유지를 위해 `cube_to_goal` observation은 남겨둠.
- 비활성화 후 active reward term은 `arm_cube_reach`, `finger_cube_reach`, `functional_hold`, `action_rate` 4개임.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- smoke test에서 action shape 18, policy observation shape 57, reward term 4개 확인함.
- lift 제거 smoke test log dir은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_22-25-03`임.
- cube reset event에 `reset_cube_position`을 추가함.
- cube는 default `(0.45, -0.18, 0.03)` 기준 `x ±0.06`, `y ±0.08`, `z 0` 범위에서 randomize됨.
- cube grasp task에서만 robot initial arm posture를 살짝 높임.
- override joint 값은 `joint1=-0.45`, `joint2=-1.85`, `joint4=1.20`임.
- 이 변경은 action/observation shape는 유지하지만 action offset 의미를 바꾸므로 long run은 fresh run 권장임.
- `py_compile`로 `env_cfg_common.py`, `grasp/indy_wuji/env_cfg.py` 문법 확인 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- smoke test에서 active reset event는 `reset_all`, `reset_cube_position` 2개로 확인함.
- smoke test에서 action shape 18, policy observation shape 57, reward term 4개 유지 확인함.
- cube random / higher arm smoke test log dir은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_22-30-24`임.
- Wuji hand actuator gain이 전체 finger 공통으로 `stiffness=20.0`, `damping=0.5`, `friction=0.02`, `effort_limit=0.6` (2026-07-12에 stiffness를 `8.0`에서 올림. damping은 한때 `2.5`였으나 **최대 폐합 속도 = effort_limit/damping = 0.24 rad/s로 손가락이 5배 느려져** `0.5`로 되돌림) 상태임을 확인함.
- 이 설정은 ring/little finger passive 흔들림을 줄이기 위한 안정화 설정임.
- `py_compile`로 `assets/indy.py` 문법 확인 통과함.
- cube grasp action scale을 `0.2`에서 `0.1`로 낮춘 상태를 확인함.
- `arm_cube_reach`를 비활성화해 palm over-guidance를 줄임.
- active reward term은 `finger_cube_reach`, `functional_hold`, `action_rate` 3개임.
- `py_compile`로 `env_cfg_common.py`, `grasp/indy_wuji/env_cfg.py` 문법 확인 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- smoke test에서 action shape 18, policy observation shape 57, reward term 3개 확인함.
- arm reach 제거 smoke test log dir은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-11_13-11-03`임.

## 2026-07-11 Progress Scale And Functional Hold Update

- `finger_cube_reach` progress reward scale을 수정함.
- `finger_cube_reach`의 `distance_max`를 `0.5`에서 `0.03`으로 낮춤.
- 여기서 `distance_max`는 실제 최대 거리라기보다 한 step progress를 reward로 키우는 정규화 scale임.
- `functional_hold`의 grasp center 계산을 thumb-weighted average에서 uniform average로 변경함.
- thumb weight `(3.0, 1.0, 1.0)`는 fingertip-cube distance bonus에만 사용함.
- grasp center가 thumb 쪽으로 과하게 끌리는 문제를 줄이기 위한 변경임.
- `functional_hold` return을 곱셈 gate 구조에서 additive shaping 구조로 변경함.
- 새 raw 수식은 `0.4 * center_bonus + 0.4 * weighted_distance_bonus + 0.2 * center_bonus * opposition`임.
- 이전 `center_bonus * (...)` 구조는 cube가 center 근처에 없으면 distance/opposition 신호까지 거의 죽는 gate처럼 작동했음.
- 새 구조는 초반에도 fingertip 접근, cube center 정렬, thumb opposition이 각각 학습 신호를 줌.
- `py_compile`로 `mdp/rewards.py`, `env_cfg_common.py` 문법 확인 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- smoke test에서 action shape 18, policy observation shape 57, reward term 3개 확인함.
- smoke test log dir은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-11_13-38-25`임.

## 2026-07-11 Cube Grasp Closeness Reward

- cube grasp GUI play에서 hand가 cube 근처로 수렴하지 않고 위아래로 움직이는 현상 확인함.
- 원인은 velocity 문제가 아니라 progress reward만으로는 가까운 상태 유지 압력이 약한 구조로 판단함.
- velocity observation/reward는 이번 변경에서 제외함.
- action/observation dimension은 유지함.
- `finger_cube_closeness` reward를 추가함.
- `finger_cube_closeness`는 thumb/index/middle fingertip과 cube root 사이 absolute bounded distance reward임.
- `finger_cube_closeness` weight는 `0.2`임.
- `finger_cube_closeness` distance_max는 `0.7`임.
- active positive reward는 `finger_cube_reach`, `finger_cube_closeness`, `functional_hold` 구조임.
- 의미는 progress reach, absolute closeness 유지, functional hold/cage 순서임.
- `py_compile`로 `env_cfg_common.py`, `rewards.py` 문법 확인 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- smoke test에서 action shape 18, policy observation shape 57 유지 확인함.
- smoke test에서 active reward term은 `finger_cube_reach`, `finger_cube_closeness`, `functional_hold`, `action_rate` 4개로 확인함.
- smoke test에서 `Episode_Reward_Raw/finger_cube_closeness`가 0이 아닌 값으로 기록됨을 확인함.
- smoke test log dir은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-11_14-08-08`임.

## 2026-07-11 Reward 전면 재설계 (cage 방식 도입)

- 상세 일지는 `ACTIVITY_2026-07-11.md`, 세부 기록은 `nrmk_isaaclab_wuji/worklog.md`임.
- 문제: 접근은 하는데 큐브를 쥐지 못함. 원인은 물리도 학습 부족도 아니고 **reward 설계가 파지를 처벌하고 있었던 것**임.

### 지표 문제

- `Metrics/cube/*`가 에피소드 20 step **평균**뿐이었음. 앞 4 step(이동 + swing-out)이 평균의 `77%`를 지배함.
- "계속 멀리 있었다"와 "밖으로 나갔다가 마지막에 붙었다"를 구분할 수 없어 정착 자세를 볼 수 없었음.
- 손가락별 거리도 큐브 **중심**까지만 있어서 접촉 여부를 알 수 없었음 (큐브가 `0.06 m`임).
- `Metrics/cube_final/*`, `cube_min/*`, `cube_max/*`와 표면거리(`*_surface`)를 추가함. 평가는 `cube_final`로 할 것.
- 과거 run 분석 시 `logs/rsl_rl/<exp>/<run>/params/env.yaml`을 읽을 것. 현재 코드 값을 대입하면 틀린 결론이 나옴.

### 제거한 reward와 그 이유

- `finger_cube_reach`, `finger_cube_closeness`: 손끝에서 큐브 **중심**까지의 거리를 씀.
- 큐브 중심은 표면에서 `0.03 m` 안쪽이라 손끝이 **도달 불가능한 목표**임. gradient가 항상 큐브 속을 향함.
- `body_weights=(3,1,1)`로 엄지가 가중평균의 `60%`라 **"엄지 하나만 박고 나머지 방치"가 최적해**가 됨.
- 실측: thumb 표면까지 `0.017 m`, index `0.072`, middle `0.078`.
- 그 자세에서 엄지-중지 선분이 큐브를 관통하지 않아, 오므리면 cage 가상점이 큐브 밖으로 빠져나감.
- 강제 오므림 시 `cage_inside_frac`이 `0.47` -> `0.40`으로 **하락**함. 정책이 오므리기를 거부한 것이 합리적이었음.
- 거리 reward는 접촉도 처벌함. 만지면 큐브가 밀려나 거리가 늘어남. `cube_displacement`가 `0.146` -> `0.037`로 감소함.
- `functional_hold`: 손끝 중심거리 + 대향 내적만 봄. 손가락을 편 채로도 만족되어 **오므림을 보상하지 않았음**.

### 도입한 reward (Dexterous Pre-grasp Manipulation)

- 엄지끝과 중지 사이에 가상점 6점. 비율 `[0.25, 0.50, 0.75]`, 선분 A(엄지끝->중지끝, 핀치), 선분 B(엄지끝->중지 중간마디, 파워).
- `finger_cage_hold` (Eq.15, `weight=1.0`): 가상점이 큐브 **내부**로 파고든 깊이를 보상함. **오므리기가 직접 보상됨.** 접촉센서 불필요함.
- `finger_cage_reach` (Eq.14, `weight=0.3`): **같은 6점**의 큐브 표면까지 SDF의 차분. 파지 간극을 큐브 위로 끌어옴.
- 거리 reward는 접촉을 **처벌**하고, cage reward는 접촉을 **보상**함. 부호가 반대임. 이것이 파지 학습의 핵심임.
- 가중치는 `hold(1.0) >> reach(0.3)`. 논문의 `r_T >> r_orient >> r_hold >> r_reach` 순서임. 기존은 역순이었고 그것이 국소최적의 원인이었음.

### 기타 수정

- action scale `0.1` -> `0.2` 복구. 절대 위치 명령이라 scale이 곧 도달 반경임. jitter는 `action_rate`로 잡을 것.
- progress reward를 `mode="previous"` + `clamp(min=-1)` + `reset()`에서 기준선 seeding으로 변경. swing-out 제거함.
- `train.py --render_interval` 추가. headless 학습에는 영향 없음.
- `gpu_max_rigid_patch_count` `2**19` -> `2**20`. overflow는 크래시가 아니라 접촉을 조용히 버림.
- 진단 metric `thumb_middle_opposition`, `cage_span` 추가.

### 미해결

- 손등이 바닥에 닿는 구조. 큐브가 맨바닥 위에 있고 테이블이 없음.
- 실측: palm `z=0.082`, 검지끝 `z=0.030`, 중지끝 `z=0.018`, `thumb_middle_opposition=-0.98`.
- 다만 테이블이 답인지 미확정임. `thumb_middle_opposition`으로 판정할 것.
- `cube_lift`는 아직 미도입. 강제로 완전히 오므려도 lift가 `0.0003 m`라 희소 보상이 발화하지 않음.

## 2026-07-12 Cage 12점 확장 + cube_lift 도입

- 상세 일지는 `ACTIVITY_2026-07-12.md`, 세부 기록은 `nrmk_isaaclab_wuji/worklog.md`임.

### 문제: cage는 성공했으나 자세가 기괴함

- run `2026-07-11_19-50-45` (1583 iter)에서 `opposition -0.965` -> `+0.922`, `cage_inside_frac 0` -> `0.837`, `cage_span 0.157` -> `0.111`로 수렴함.
- 지표상 완벽한 파지였으나 GUI에서 **검지와 중지가 교차하고 손바닥이 하늘을 향함**을 확인함.
- 원인: cage가 "엄지끝 ↔ 중지" 선분만 봄. **검지는 reward에 등장조차 하지 않았고 손바닥 방향도 제약 없었음.**
- 논문에는 `r_grasp`(`r_hr` + `r_hj`)가 손 회전과 손가락 관절각을 붙잡고 있어 이 문제가 없었음. 우리는 큐브에 목표 파지 자세가 없어 `r_grasp`를 구현하지 못했음.

### 수정 1: 가상점 6 -> 12 (검지 추가)

- `CAGE_BODIES = [finger1_tip_link(엄지끝, 기준점), finger2_tip_link, finger2_link3, finger3_tip_link, finger3_link3]`.
- 대향 body 4개 x 등간격 3점 = 12점. 논문이 명시한 확장임.
- **엄지+검지+중지는 젓가락 그립과 동일하므로 임시방편이 아님.**
- metric 신규: `thumb_index_opposition`, `cage_span_index`. 중지만 보면 검지 교차를 놓침.

### 수정 2: `palm_facing` 철회

- 손바닥 방향 reward를 검토했으나 **자의적**이라 철회함.
- 논문이 `r_hr`을 주는 이유는 **기능** 때문임 (드릴을 "트리거를 당길 수 있게" 쥐어야 함).
- 큐브에는 기능 요구가 없으므로 목표 회전이 필요 없는 것이 맞음. **목표 회전은 "잡기"가 아니라 "쓰기"에서 나옴.**
- 참고: `palm_link`의 손바닥 법선은 실측 결과 로컬 `+x`축임.

### 수정 3: `cube_lift` (진짜 선별 기준)

- `r = cage_gate * clamp(height/0.08, 0, 1)`, `weight=3.0`.
- **들지 못하는 자세는 파지가 아님.** 하중을 견디는지만 물으면 자세를 지정할 필요가 없음. **물리가 자세를 결정함.**
- 논문도 fake success 방지용으로 `r_lift`를 넣음.
- gate가 없으면 "파지 없이 큐브를 튕겨 올리는" 편법이 가능함.
- 조밀형이라 현재의 `2 mm` 상승에도 gradient가 있음. 희소형이면 영원히 `0`이라 학습 불가함.

### 역할 분담 (중요)

- **cage는 자세를 유도하지 않음.** "물체가 손가락 사이에 있는가"만 봄. 6점 cage가 손바닥 하늘로 수렴한 것이 반증임.
- 자세를 선별하는 것은 `cube_lift`, 정확히는 **물리**임.

### 최종 reward

- `finger_cage_reach` (0.3), `finger_cage_hold` (1.0), `cube_lift` (3.0), `action_rate` (-0.0003).
- 논문의 `r_T >> r_hold >> r_reach` 순서를 따름.
- action shape 18, observation shape 57 불변. smoke test 통과.

### 실패 시 용의자 순서

- 1순위: 손가락 actuator (`stiffness=20.0`, `damping=0.5`)가 물러서 `0.08 kg`을 못 쥘 수 있음.
- 2순위: 큐브가 맨바닥에 있어 팔이 `z=0.03`까지 굽혀 내려가야 함. manipulability 최악임.
- 3순위: 제어 주파수 `2.5 Hz`가 손가락 폐합에 거침. 논문은 `30 Hz`임.

### 젓가락 로드맵

- 젓가락에는 **기능 요구가 있으므로 `r_hr`(손 회전)이 반드시 필요함.** 자의적이지 않음.
- constraint-based 방식 사용: `g = [g_ifp(검지 끝 목표 위치), g_r(손 회전)]`. 손가락 관절각은 주지 않음.
- 넘어가는 것: cage 기계 전체, 검지 포함 가상점, `cube_lift`.
- 추가되는 것: `r_hr`, `r_hp`. **버려지는 것 없음.**
- 순서: 큐브 파지 -> 직육면체 + 랜덤 방향 -> 젓가락 pre-grasp(논문 완성) -> 젓가락 조작(논문 밖).

## 2026-07-13 palm_facing 차분형 + arm_manipulability + cube_lift가 0이라는 발견

- 상세 일지는 `ACTIVITY_2026-07-13.md`, 세부 기록은 `nrmk_isaaclab_wuji/worklog.md`임.

### 문제: `palm_facing`(절대형)이 국소최적을 만듦

- run `2026-07-12_20-11-57` (1803 iter): `palm_facing`이 `0.994`까지 올랐으나 손이 큐브 **31cm 밖에서 정지**함.
- 보상 분해 결과 `palm_facing`이 전체 `mean_reward`의 **`98.6%`**를 차지함. 나머지 항은 전부 `0`이었음.
- **거리와 무관하게 지급**되므로 (단위벡터라 거리가 사라짐) 멀리서 겨눠도 만점이었음.
- 접근하면 `palm_facing`이 떨어져 **순손실**이었음. 정책은 합리적으로 행동한 것임.
- 사용자 GUI 관찰로 **팔이 접혀 특이점에 빠진 것**을 발견함.
- manipulability 실측: 초기 자세 `0.0645` (`57%`) -> 수렴 자세 **`0.0144` (`13%`)**.

### 수정 1: `arm_manipulability` (논문 Eq.17, weight `1.0`)

- `r = 1 - 2 / (1 + (min(|J|, j_max)/j_max)^3)`, 범위 `[-1, 0]`, `j_max = 0.02`.
- 논문: "penalizes coming close to singularities and leads to learning more intuitive motions."

### 수정 2: `palm_facing`을 차분형으로 (`PalmFacingProgressReward`, weight `1.0`)

- **논문의 거의 모든 항이 차분형임** (`r_hp`, `r_hr`, `r_hj`, `r_reach`, `r_orient`). **절대형은 `r_hold` 하나뿐임.**
- **차분형은 "가만히 있으면 0"이라 farming이 불가능함.** 그래서 논문은 weight `1.0`을 줘도 안전함.
- 우리 `palm_facing`은 절대형이라 정렬만 유지해도 매 step 보상이 나왔음.
- **가중치가 아니라 형태가 문제였음.**

### 형태 선택 원칙 (확립)

- 절대 양수 + 유지가 쉬움 -> **반드시 farming당함**.
- 절대 양수 + 유지가 어려움 -> 안전 (`finger_cage_hold`, `cube_lift`).
- 절대 페널티 (`<=0`) -> 안전 (최대가 `0`이라 쌓을 것이 없음).
- 차분 -> 안전.

### 논문의 실제 가중치 (9쪽)

- `r = r_grasp + r_reach + 25*r_hold + 500*r_orient + r_MP + 5000*r_T`.
- approach(1) -> hold(25) -> orient(500) -> grasp(5000). **각 단계마다 약 20배씩.**
- "The exact values do not affect learning significantly, **as long as the overall proportions reflect the logical sequence**."
- **절대값이 아니라 비율만 중요하고, scale은 각자의 보상값 크기에 맞춰야 함.**

### 결정적 발견: `cube_lift` 보상이 한 번도 나온 적이 없음

- 전 학습 기간 동안 `Episode_Reward_Raw/cube_lift` = **정확히 `0`**. 큐브가 단 한 번도 바닥에서 떨어진 적이 없음.
- **`0`인 보상은 가중치가 `3`이든 `500`이든 무의미함.** 정책은 "들기" gradient를 받은 적이 없음.
- 원인: `cube_lift`를 **최하 모서리** 기준으로 바꿔 기울이기 편법을 막았더니 **사실상 희소해짐.**
- 희소 보상은 curriculum 없이 학습 불가능함. 논문의 `r_T`(sparse)가 curriculum과 세트인 이유임.

### Curriculum이 필수인 이유

- 과제가 **사슬**임. 마지막 보상은 앞의 모든 링크를 통과해야 나옴.
- 마지막 보상을 한 번도 경험 못 하면 value function에 전파되지 않고, **정책은 도달 가능한 가장 뒤쪽 링크에서 멈춤.**
- Curriculum은 사슬을 **거꾸로** 배움. close-start에서 마지막 보상을 도달 가능하게 만들고, 후속 단계에서 앞 단계에 목적을 부여함.
- 논문 데이터: curriculum 없으면 성공률 약 `50%`에 편차 폭발, 있으면 **`97%`** (wall-clock 동일).

### 사용자 통찰: 보상 단계화도 curriculum임

- "접근/파지를 먼저 학습시키고 lift를 넣어 다시 학습"도 curriculum임. **환경이 아니라 보상을 단계화**하는 것임.
- 더 저렴함. 환경을 안 건드리므로 warm-start가 깔끔함.
- **성립 조건**: Phase 1이 끝났을 때 그 자세에서 lift가 **탐색으로 도달 가능**해야 함. 아니면 `0 x W = 0`임.

### 현재 reward (6개)

- `finger_cage_reach` (차분, 0.3), `palm_facing` (차분, 1.0), `finger_cage_hold` (절대, 1.0).
- `cube_lift` (절대, 3.0), `arm_manipulability` (절대 페널티, 1.0), `action_rate` (-0.0003).
- action shape 18, observation shape 57 불변.

### 계획

- Phase 1 (진행 중 `2026-07-13_10-26-32`): 접근 + 손바닥 정렬 + 감싸기 학습. `cube_lift`는 사실상 `0`이라 없는 것과 같음.
- 수렴 후 (1) 실측 raw 값으로 가중치 역산, (2) **"들 수 있는 자세"인지 검증**.
- Phase 2: `cube_lift` 가중치 상향 + `--resume`.

## 2026-07-13 `Indy-Wuji-Cube-Grasp-Easy` curriculum task 추가

- `Indy-Wuji-Cube-Grasp-Easy` task를 추가함.
- 목적은 reward를 더 만지는 것이 아니라 **현재 reward가 켜질 수 있는 쉬운 초기 상태**를 만드는 것임.
- 기존 hard task `Indy-Wuji-Cube-Grasp`는 유지함.
- Easy task는 `Indy7WujiCubeGraspEnvCfg`를 상속함.
- action dim, observation dim, reward term, runner/model 구조는 hard task와 동일하게 유지함.
- cube 초기 위치만 손 가까운 grasp aperture 안쪽으로 옮김.
- Easy cube 위치는 `(0.74, -0.18, 0.03)`임.
- Easy reset random range는 `x/y = +/-0.015`, `z = 0`으로 좁힘.
- gym id는 `Indy-Wuji-Cube-Grasp-Easy`임.
- 의도는 easy에서 `finger_cage_hold > 0`을 먼저 경험시키고, 이후 hard task로 checkpoint resume하는 것임.

### 변경 파일

- `nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/grasp/indy_wuji/env_cfg.py`
- `nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/grasp/indy_wuji/__init__.py`
- `CLI.md`

### 실행 흐름

```bash
python scripts/rsl_rl/train.py \
  --task Indy-Wuji-Cube-Grasp-Easy \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name easy_close_start
```

### 다음 확인

- smoke test에서 task 등록이 되는지 확인함. 완료함.
- Action Manager shape가 hard와 같은 `18`인지 확인함. 완료함.
- Observation shape가 hard와 같은 `57`인지 확인함. 완료함.
- Reward Manager active term 6개 확인함.
- `Episode_Reward_Raw/finger_cage_hold`가 `0`에서 벗어나는지 확인함.
- `Metrics/cube_final/cage_inside_frac`가 `0`에서 벗어나는지 확인함.

## 2026-07-13 hand/cube contact response 완화 확인

- 영상에서 ring/little을 접은 뒤에도 palm/hand가 cube와 닿을 때 arm이 뒤로 튀는 현상을 확인함.
- 원인 후보를 ring/little 단독 collision에서 palm/hand contact impulse 쪽으로 갱신함.
- active `INDY7_WUJI_RIGHT_CFG`에서 `RigidBodyPropertiesCfg` 값이 완화된 것을 확인함.
- 현재 active 값은 `max_depenetration_velocity=5.0`, `max_contact_impulse=100.0`임.
- 이전 active 값은 `max_depenetration_velocity=1000.0`, `max_contact_impulse=1e32`였고, 관통 보정 속도/impulse가 사실상 무제한이라 contact 순간 arm이 튈 수 있었음.
- cube 쪽은 아직 `CollisionPropertiesCfg(collision_enabled=True)`만 있고 `contact_offset/rest_offset`는 따로 주지 않음.
- 이 변경은 action/observation/reward/model shape를 바꾸지 않음.
- 기존 checkpoint로 `play` 비교 가능함.
- 학습은 기존 checkpoint에서 `resume` 가능함.
- 단, physics가 바뀐 것이므로 최종 성능 평가는 resume adaptation 또는 fresh/easy curriculum run 후 판단함.

## 2026-07-14 close-start 높이/받침면 수정

- close-start이 실제로는 큐브 `x/y`만 손 파지 중심 근처였고 `z`는 바닥 `0.03m`라 손 높이와 맞지 않는 문제를 확인함.
- `BASE_Z=0.50` 받침면을 추가함.
- `{ENV_REGEX_NS}/Support` kinematic cuboid를 `CubeGraspSceneCfg`에 추가함.
- cube 중심 높이는 `BASE_Z + CUBE_HALF = 0.53m`로 맞춤.
- close-start cube 위치는 `(0.704, -0.279, 0.530)`임.
- hard cube 위치는 `(0.850, 0.000, 0.530)`임.
- `cube_lift` reward에 `surface_z` 파라미터를 추가해 월드 바닥이 아니라 받침면 기준 lift를 보게 함.
- TensorBoard `cube_clearance` metric도 같은 `surface_z` 기준으로 보정함.
- `hand_floor` penalty도 같은 `surface_z` 기준으로 보정함.
- `python -m py_compile` 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- 확인된 shape는 action `18`, policy observation `57`임.

## 2026-07-14 close-start 파지 간극/방향 검증 정정

- 사용자 지적대로 `palm_facing=0`과 `joint5` action 반경 축소는 초기 손-큐브 방향 정렬이 검증돼야만 성립함.
- `/tmp/cube_grasp_probe.py`로 close-start reset 상태를 수치 확인함.
- 기존 close-start cube `(0.704, -0.279, 0.430)`는 cage 중심에서 y로 약 `9cm` 벗어났음.
- 기존 위치는 zero action 30 step만 해도 cube가 `(0.758, -0.220, 0.430)`으로 밀리고 `palm_facing`이 `0.838 -> 0.260`으로 깨짐.
- close-start cube를 `(0.692, -0.369, 0.430)`으로 수정함.
- 수정 후 reset `palm_facing=0.987`, zero action 30 step 뒤 `0.997`임.
- 수정 후 `cage_center_to_cube` xy 오차는 reset 약 `0.5mm`, zero action 30 step 뒤 약 `3mm`임.
- 따라서 **현재 close-start 배치에서만** `finger_cage_reach=0`, `palm_facing=0`, `joint5` scale 축소가 타당함.
- cube 배치나 arm reset pose를 바꾸면 이 probe를 다시 해야 함.
- `python -m py_compile` 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.

## 2026-07-14 close-start action 출력 보강

- action이 잘못 나오는지 직접 확인하기 위해 `scripts/rsl_rl/play.py`의 `--print_action` 출력을 보강함.
- 기존 출력은 전체 action 요약과 `actions[0,5]` 하나만 보여서 원인 분리에 부족했음.
- 새 출력은 `raw policy action`, `clip 후 applied action`, `target joint position`, `actual joint position`, `target-actual error`, `joint velocity`를 joint 이름별로 출력함.
- `clip%`도 출력해 policy가 `clip_actions=1.0` 밖으로 자주 나가는지 확인 가능함.
- 판정 기준은 다음과 같음.
- `err`가 작고 action/Delta action이 크면 policy가 그렇게 시킨 것임.
- `err`가 크면 물리/PD/접촉/decimation이 목표를 못 따라가는 것임.
- `CLI.md`에 close-start action 확인 명령을 추가함.
- `python -m py_compile nrmk_isaaclab_wuji/scripts/rsl_rl/play.py` 통과함.

## 2026-07-14 Cube Grasp 단일 task 정리

- 사용자 판단대로 `close-start`/`Hard`를 나눠 쓰지 않기로 함.
- `Indy-Wuji-Cube-Grasp` 본체가 검증된 close-start nominal grasp 배치를 사용하도록 정리함.
- 별도 호환 alias는 남기지 않음. task id는 `Indy-Wuji-Cube-Grasp` 하나임.
- 새 학습/play/smoke test 명령은 전부 `Indy-Wuji-Cube-Grasp`를 사용함.
- 현재 cube 위치는 `(0.692, -0.369, 0.430)`임.
- cube reset randomization은 꺼둠. `x/y/z = 0`.
- 현재 reward override는 `finger_cage_reach=0`, `palm_facing=0`, `finger_cage_hold=1`, `cube_lift=10`, `arm_manipulability=1`, `hand_floor=0.5`, `action_rate=-0.0003`임.
- `CLI.md`, `AGENTS.md`, `nrmk_isaaclab_wuji/agent.md`를 단일 task 기준으로 정리함.
- 같은 experiment 안에 과거 smoke/hard/easy run이 섞여 있으므로 최신 run 자동 선택은 위험함.
- `python -m py_compile` 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- smoke test에서 action `18`, observation `57`, reward 7개를 확인함.
- reward table은 `finger_cage_reach=0`, `finger_cage_hold=1`, `cube_lift=10`, `palm_facing=0`, `arm_manipulability=1`, `hand_floor=0.5`, `action_rate=-0.0003`임.
- `/tmp/cube_grasp_probe.py --task Indy-Wuji-Cube-Grasp --headless --num-envs 1` 통과함.
- probe reset 값은 cube `(0.692, -0.369, 0.430)`, cage xy error 약 `0.5mm`, `palm_facing=0.986780`, `cage_hold=0.210871`임.
- zero action 30 step 뒤 cage xy error 약 `3mm`, `palm_facing=0.996594`임.

## 2026-07-14 close-start alias 제거 최종 정리

- 예전 curriculum alias와 관련 class/register를 제거함.
- cube grasp 실행 이름은 `Indy-Wuji-Cube-Grasp` 하나만 남김.
- `CLOSE_START_CUBE`는 가까운 nominal grasp 배치 상수로 유지함.
- 과거 hard 배치 상수는 `LEGACY_HARD_CUBE`로 이름만 남겨 공통 cfg 기본값과 구분함.
- RSL-RL `run_name`은 `cube_grasp_close_start`로 정리함.
- 예전 probe helper는 `/tmp/cube_grasp_probe.py`로 이름을 바꿈.
- functional_grasp/chopsticks skeleton의 예전 alias class/register도 제거함.
- repo 전체에서 예전 alias/문자열을 검색해 남은 직접 참조가 없음을 확인함.
- `python -m py_compile` 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- `/tmp/cube_grasp_probe.py --task Indy-Wuji-Cube-Grasp --headless --num-envs 1` probe 통과함.

## 2026-07-14 hand-only cube grasp 재정리

- cube grasp를 다시 "Wuji hand가 물체를 감쌀 수 있는지"부터 확인하는 hand-only 진단 모드로 바꿈.
- policy action은 `finger[1-3]_joint[1-4]` 12축만 사용함.
- arm 6축은 `FixedJointPositionAction` 0D term을 새로 만들어 default joint target을 매 step 유지하게 함.
- 이 수정 전에는 arm action을 빼면서 arm joint target도 사라져 zero action 30 step만에 arm이 무너지고 cube가 날아갔음.
- `CubeGraspActionsCfg`에 `arm_hold_action` 슬롯을 추가함.
- `Indy-Wuji-Cube-Grasp` Action Manager는 `arm_action=12`, `arm_hold_action=0`, total action shape `12`로 확인함.
- policy observation shape는 `42`임.
- `cube_to_goal` observation은 hand-only 진단에서 제거함.
- active reward는 `finger_cage_hold=1`, `hand_floor=0.5`, `action_rate=-0.0003` 중심임.
- `finger_cage_reach`, `palm_facing`, `cube_lift`, `arm_manipulability` weight는 현재 `0`임.
- RSL-RL run name은 `cube_grasp_hand_only`로 바꿈.
- `python -m py_compile` 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- `/tmp/cube_grasp_probe.py --task Indy-Wuji-Cube-Grasp --headless --num-envs 1 --close-action 1.0 --close-steps 60` 통과함.
- probe에서 zero action 30 step 후 arm collapse/cube ejection이 사라짐.
- close action 60 step 후 `cage_hold=0.427465`, `cage_inside_frac=0.666667`로 증가함.
- `--num_envs 128 --max_iterations 20` 짧은 학습 통과함.
- 짧은 학습에서 `finger_cage_hold`는 초반 `0.003`에서 `0.13~0.18`까지 증가함.
- 같은 학습에서 `hand_floor` penalty가 커지고 mean reward는 음수로 유지됨.
- 다음 판단은 GUI에서 손이 받침면을 긁는지 확인한 뒤 `hand_floor`/초기 높이/손가락 action scale을 조정하는 것임.

## 2026-07-14 grasp+lift active 재정리

- `Indy-Wuji-Cube-Grasp`를 다시 arm 6축 + thumb/index/middle 12축, 총 18D action으로 돌림.
- observation은 joint position 18D, palm 기준 cube 3D, five-fingertip 기준 cube 15D, previous action 18D로 총 54D임.
- `cube_to_goal` observation은 현재 grasp+lift baseline에서 제거함.
- active reward는 `finger_cage_reach=3`, `finger_cage_hold=5`, `cube_lift=50`, `cube_support=2`, `palm_facing=0`, `arm_manipulability=0`, `hand_floor=0.2`, `action_rate=-0.0003`임.
- `cube_support`를 추가해 cube 최하 모서리가 받침면 아래로 내려가면 벌점을 주게 함.
- `--num_envs 128 --max_iterations 20` 짧은 학습에서는 hold는 켜졌지만 lift는 거의 0이고, cube를 받침 아래로 누르는 실패 모드가 보임.
- `/tmp/cube_lift_probe.py`를 만들어 scripted feasibility를 확인함.
- 기본 설정에서 close action은 `cage_hold`를 약 `0.40`까지 올리지만 `cube_clearance`는 0 근처이고 `cube_lift_reward_raw=0`임.
- `joint0~joint5` 단일축 ±1 lift 후보를 모두 넣어도 양의 cube clearance가 나오지 않음.
- 강한 손/가벼운 큐브 probe(`finger_effort=3`, `stiffness=40`, `cube_mass=0.03`, `friction=2`)에서도 lift는 0임.
- 결론: 긴 RL 전에 실제로 cube를 들어올리는 scripted arm/hand sequence 또는 초기 자세를 먼저 찾아야 함.

## 2026-07-14 contact/lift 확인 스크립트 추가

- `scripts/debug/check_cube_contact_lift.py` 추가함.
- policy 없이 `reset -> zero settle -> finger close -> optional arm lift`를 실행하게 함.
- thumb/index/middle/palm 링크별 cube contact force를 출력하게 함.
- cube clearance와 `GOOD_CONTACT thumb+middle`, `LIFT_SUCCESS` 판정을 출력하게 함.
- cube 자체에는 contact reporter가 없어 `Cube -> Support` contact sensor는 제거함.
- 손 링크 기준 cube contact sensor만 사용함.
- 짧은 headless 검증 통과함.
- close-only 확인 결과 `after_close`에서 `index_mid=5.5961N`, `thumb_tip=0.0593N`, middle contact 없음.
- `after_lift`에서 `thumb_tip=0.4257N`, `middle_mid=0.7081N`로 `GOOD_CONTACT=True`가 늦게 켜짐.
- 같은 run에서 `max_clearance=0.0003m`라 lift는 실패함.
- 결론: contact 자체는 생기지만 현재 scripted close-only는 cube를 들어 올리지 못함.
- 긴 학습 전에 `--sweep-lift` 또는 cube pose sweep으로 `GOOD_CONTACT=True`와 `max_clearance>0.005m`가 나오는 조건을 먼저 찾아야 함.

## 2026-07-14 contact/lift probe finger sweep 추가

- `scripts/debug/check_cube_contact_lift.py`에 `--finger-action` 옵션을 추가함.
- `--finger-action`은 thumb/index/middle 3개 그룹 값 또는 finger action 12개 값을 직접 받음.
- `--sweep-fingers` 옵션을 추가함.
- `--sweep-fingers`는 cube를 고정하고 thumb/index/middle close 값 조합을 훑음.
- `--contact-mode` 옵션을 추가함.
- contact 판정은 `thumb_middle`, `thumb_index`, `thumb_any`, `tripod` 중 선택 가능함.
- 짧은 검증 명령 `--finger-action 1 0 1 --contact-mode thumb_middle` 실행 통과함.
- 해당 짧은 검증에서는 thumb과 middle이 같은 시점에 안정적으로 물지 못해 `lift_success=False`임.
- 다음 확인은 cube를 움직이기보다 `--sweep-fingers`로 손가락 joint 명령 조합부터 찾는 것이 우선임.

## 2026-07-14 play diagnostics 추가 및 현재 정책 판정

- `scripts/rsl_rl/play.py`에 `--latest_run`, `--load_run latest/last`, `--print_diagnostics`, `--print_contact`를 추가함.
- 별도 상세 진단 스크립트 `scripts/debug/policy_joint_diagnostics.py`도 추가함.
- `--print_diagnostics --print_contact --print_action_interval 1`은 렉이 심함. 평소에는 interval 10~20 권장.
- 사용자 play 로그 판정:
  - 안정 구간에서 `|raw| ~= 2.5`, `|applied| ~= 0.83`, clip 약 `66.7%`.
  - arm torque 부족 아님. `joint1` torque 약 `3~4%`, err 약 `0.14rad`.
  - finger는 여러 관절이 effort limit에 붙음 (`tq%=100`).
  - `finger_cage_hold` raw는 약 `0.46~0.48`이지만 `cube_lift`와 `clearance`는 0 근처.
  - `cage_inside_frac ~= 0.58`, `cage_span ~= 0.11m`, `thumb_middle_opposition ~= 0.52`, `thumb_index_opposition ~= 0~0.08`.
- 결론: 현재 정책은 "잡는 듯한 cage/hold 점수"는 먹지만 실제 하중 지지/lift는 못 하는 local optimum임.
- 다음 레버는 arm torque가 아니라 finger action range/scale, finger joint2 처리, negative target 처리, contact/lift/r_T 계층임.
