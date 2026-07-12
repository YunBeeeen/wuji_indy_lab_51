# AGENTS.md

- 이 문서는 Codex/Claude가 프로젝트 상태와 작업 규칙을 공유하기 위한 인수인계 문서임.

## Project Context

- 목표는 Isaac Lab 기반 Indy7 + Wuji hand end-effector tracking RL 환경 구성임.
- 최종 조작 목표는 젓가락을 기능적으로 잡고 젓가락질이 가능한 파지 상태를 학습하는 것임.
- `Dexterous Pre-grasp.pdf` 계열 functional grasp/pre-grasp 아이디어가 최종 연구 방향의 주 기준임.
- 현재 workspace는 `~/wuji_indy_lab_51`임.
- 현재 실제 코드 디렉터리는 `~/wuji_indy_lab_51/nrmk_isaaclab_wuji`임.
- 초기 문서에는 repo가 `~/wuji_indy_lab_51/nrmk_isaaclab_public`로 기록돼 있었음.
- 현재 작업은 `nrmk_isaaclab_wuji` 기준으로 진행 중임.
- 기존 `chop_ws/chop_rl`, Isaac Sim 4.5, IsaacLab 2.2.1 실험은 폐기함.
- 현재는 잘 켜지는 Isaac Sim 5.1, IsaacLab 2.3 계열, `env_isaaclab` 환경 사용함.
- Neuromeka public/main branch 스타일을 우선 사용함.
- 목표는 full training 성능이 아니라 env 구조 이해와 arm end-effector tracking 구성임.
- 현재 reach baseline task는 `Indy-Wuji-Reach`임.
- 현재 cube grasp 구현 task는 `Indy-Wuji-Cube-Grasp`임.
- 현재 active USD는 `indy7_wuji_right_simplified.usd`임.
- 초기 후보였던 `indy7_allegro_hand_right_simplified.usd`는 참고/비교용임.
- 현재 tracking body는 `link6`임.
- 현재는 Wuji hand frame 문제를 분리하고 Indy arm flange 기준 reach baseline을 확인하는 단계임.
- cube grasp는 최종 목표가 아니라 functional grasp/chopstick grasp로 가기 위한 중간 proxy task임.
- `tcp`는 현재 USD articulation rigid body로 쓰기 부적합하다고 판단함.
- virtual EE offset 방식은 실험 후 보류함.
- 현재 action은 arm 6축만 사용함.
- hand joint 20축은 articulation에는 남아 있음.
- hand action은 아직 policy action에 넣지 않음.
- hand joint는 초기 arm tracking 단계에서는 고정 또는 최소 취급함.
- 이후 필요하면 hand action까지 확장함.

## Current Implementation

- `Indy-Wuji-Reach`는 Neuromeka `Indy-Reach` 스타일로 구현됨.
- 공통 reach 구조는 `isaac_neuromeka/tasks/manipulation/reach/reach_env_cfg.py` 기반임.
- 공통 MDP 설정은 `isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py` 기반임.
- task override는 `isaac_neuromeka/tasks/manipulation/reach/indy_wuji/env_cfg.py`에 있음.
- gym registration은 `isaac_neuromeka/tasks/manipulation/reach/indy_wuji/__init__.py`에 있음.
- RSL-RL config는 `isaac_neuromeka/tasks/manipulation/reach/indy_wuji/learning/rsl_rl_cfg.py`에 있음.
- robot asset config는 `isaac_neuromeka/assets/indy.py`의 `INDY7_WUJI_RIGHT_CFG`임.
- arm action dim은 6임.
- policy observation dim은 현재 15임.
- observation은 arm 6축 joint position, target position command, previous action만 사용함.
- `joint_vel` observation은 제거함.
- observation history와 observation noise는 현재 policy group에서 제거함.
- hand joint는 observation에서 제외됨.
- `sim.render_interval = decimation` 적용됨.
- command는 공통 `UniformPoseCommandCfg`를 사용함.
- command manager의 `ee_pose` 자체는 7D pose command임.
- policy observation에는 `ee_pose` 중 position xyz 3D만 넣음.
- active reward는 position tracking과 action rate penalty만 사용함.
- orientation tracking, end-effector speed, joint velocity reward term은 현재 제거함.
- command/reward body는 `link6`임.
- `Indy-Wuji-Reach` entry point는 `CustomManagerBasedRLEnv`임.
- TensorBoard에는 weighted reward `Episode_Reward/*`와 raw unweighted reward `Episode_Reward_Raw/*`를 같이 기록함.

## Cube Grasp Current State

- cube grasp용 새 task skeleton은 `Indy-Wuji-Cube-Grasp`임.
- cube grasp package 경로는 `isaac_neuromeka/tasks/manipulation/grasp/`임.
- cube grasp 공통 cfg는 `isaac_neuromeka/tasks/manipulation/grasp/cube_grasp_env_cfg.py`임.
- Indy/Wuji 전용 override는 `isaac_neuromeka/tasks/manipulation/grasp/indy_wuji/env_cfg.py`임.
- `CubeGraspSceneCfg`는 기존 `ReachSceneCfg`를 상속하고 cube만 추가함.
- cube는 `RigidObjectCfg`로 `{ENV_REGEX_NS}/Cube`에 생성함.
- 현재 cube size는 `0.06 m`임.
- 현재 cube mass는 `0.08 kg`임.
- 현재 cube initial position은 `(0.45, -0.18, 0.03)`임.
- 4096 env long run에서 PhysX patch buffer overflow가 발생해 `gpu_max_rigid_patch_count`를 `2**19`로 올림.
- 2026-07-10 resume run에서 요구 patch count가 약 `263k`까지 올라가 `2**18`로는 부족했음.
- cube grasp RSL-RL experiment name은 `indy_wuji_cube_grasp`임.
- 현재 cube grasp task는 arm 6축 + thumb/index/middle 12축 action을 사용함.
- 현재 cube grasp action dim은 18임.
- 현재 cube grasp action scale은 `0.2`임 (2026-07-11에 `0.1`에서 복구함).
- **action은 `target = default_joint_pos + scale * raw_action`인 절대 위치 명령임.** 증분이 아니므로 과거 action이 누적되지 않고, `scale`이 곧 default 자세 중심의 **도달 반경**임. jitter를 줄이려고 낮추면 팔이 큐브에 못 닿음. jitter는 `action_rate` weight로 잡을 것.
- 현재 cube grasp policy observation dim은 57임.
- 현재 cube grasp controlled joints는 `joint[0-5]`, `finger[1-3]_joint[1-4]`임.
- 현재 cube grasp observation은 controlled joint position 18D, `palm_link` 기준 cube relative position 3D, five-fingertip 기준 cube relative position 15D, cube goal relative position 3D, previous action 18D임.
- 현재 cube grasp command manager는 active command 없이 시작함.
- cube reset 때 default `(0.45, -0.18, 0.03)` 기준 `x ±0.06`, `y ±0.08` 범위로 randomize함.
- cube grasp task에서만 Indy arm initial joint를 살짝 높은 pre-grasp 자세로 override함.
- cube grasp initial arm override는 `joint1=-0.45`, `joint2=-1.85`, `joint4=1.20`임.
- 위 arm override는 action offset도 바꾸므로 이전 checkpoint resume은 가능하지만 fresh run이 더 깔끔함.
- Wuji hand actuator는 현재 전체 finger 공통으로 `stiffness=8.0`, `damping=0.5`, `friction=0.02`임.
- 이 값은 ring/little finger 떨림을 줄이기 위한 안정화 설정임.
- 현재 cube grasp reward는 pre-grasp/functional-hold baseline으로 구성함.
- cube grasp의 주 목표는 DexPoint 재현이 아니라 functional grasp 논문 흐름을 Wuji/cube task에 맞게 구현하고 검증하는 것임.
- DexPoint는 grasp reward 구현 목표가 아니라 reach/contact/lift gate 설계 참고 자료임.
### Active reward (2026-07-12 전면 재설계)

- active reward는 `finger_cage_reach` (`0.3`), `finger_cage_hold` (`1.0`), `cube_lift` (`3.0`), `action_rate` (`-0.0003`) 4개뿐임.
- 가중치는 논문의 `r_T >> r_orient >> r_hold >> r_reach` 순서를 따름.
- 세 항 모두 **같은 12개 가상점** 위에서 동작함. `CAGE_BODIES` 상수를 공유함.
- action shape `18`, observation shape `57` 불변임.

### 가상점 12개 (`CAGE_BODIES`)

- `finger1_tip_link` (엄지끝, 모든 선분의 기준점).
- `finger2_tip_link`, `finger2_link3` (검지 끝, 검지 중간마디).
- `finger3_tip_link`, `finger3_link3` (중지 끝, 중지 중간마디).
- 엄지끝에서 대향 body 4개로 선분을 긋고 각 등간격 3점 = **12점**.
- `SceneEntityCfg`에 `preserve_order=True` 필수임. 기본값 `False`면 body_ids가 정렬되어 엄지가 기준점 자리에서 밀려남.
- `managers.py`의 `_cage_body_names`와 **반드시 동일해야 함.** 한쪽만 바꾸면 metric이 reward와 다른 점을 측정함.
- 논문은 엄지↔중지만 써서 6점이지만, 논문에는 `r_grasp`(`r_hr` + `r_hj`)가 손 회전과 손가락 관절각을 붙잡고 있음. 큐브는 목표 파지 자세가 없어 `r_grasp`를 못 쓰는데, 6점만 쓰면 **검지가 완전히 자유가 되어 "손바닥이 하늘을 보고 검지·중지가 교차한" 자세로도 만점**이 나옴 (2026-07-11 실측).
- 엄지+검지+중지는 젓가락 그립과 동일하므로 임시방편이 아님.

### `finger_cage_hold` (논문 Eq.15, weight `1.0`)

- 각 가상점을 반지름 `sphere_radius`의 구로 보고, 그 구가 큐브를 **파고든 깊이**를 보상함.
- `r = clamp((sphere_radius - sdf) / (sphere_radius + depth_max), 0, 1)`의 12점 평균임.
- `sdf`는 큐브 **표면**까지의 signed distance. 내부이면 음수임.
- **손을 오므리면 점들이 큐브 안으로 들어가므로 "오므리기"가 직접 보상됨.** 접촉센서 불필요함.
- 큐브 SDF는 해석식(`_box_signed_distance`)임. CAD나 사전계산 SDF 불필요함.
- 파라미터는 실측 튜닝됨: `sphere_radius=0.005`, `depth_max=0.02`.
- `sphere_radius`가 크면 손가락을 벌린 채 큐브가 사이에 있기만 해도 점수가 나와 대비가 죽음.

### `finger_cage_reach` (논문 Eq.14, weight `0.3`)

- **같은 12개 가상점**의 큐브 표면까지 SDF 평균의 **차분**임. `mdp.ObjectCageProgressReward`.
- "파지 간극을 큐브 위로" 끌어당겨 큐브가 손가락 **사이**에 놓이게 만듦.
- `mode="previous"` + `clamp(min=-1)` + `reset()`에서 기준선 seeding, 셋을 다 해야 함.
- `clamp(min=0)`이면 후퇴가 공짜이고, 기준선을 첫 `__call__`에서 잡으면 첫 액션이 기준선을 공짜로 부풀림 (swing-out).
- 셋을 다 하면 총합이 `d(reset) - d(final)`로 telescoping되어 페이스 조작과 swing-out이 모두 무의미해짐.
- `distance_max=0.5`. 실제 step당 최대 개선량보다 충분히 커야 함. 포화되면 "천천히 접근하기"를 보상하게 됨.

### `cube_lift` (논문 `r_lift`, weight `3.0`)

- `mdp.object_lift_in_cage`. `r = cage_gate * clamp(height / 0.08, 0, 1)`.
- **어떤 자세를 "진짜 파지"로 인정할지 결정하는 항임.**
- **들지 못하는 자세는 파지가 아니므로, 자세를 지정할 필요 없이 하중을 견디는지만 물으면 됨. 물리가 자세를 결정함.**
- `cage_gate`가 없으면 "파지 없이 큐브를 튕겨 올리는" 편법이 가능함.
- 조밀형(dense)이라 `2 mm` 상승에도 gradient가 있음. 희소형이면 영원히 `0`이라 학습 불가함.
- 논문도 fake success 방지용으로 `r_lift`를 넣음. 2026-07-11 run이 정확히 그 상태였음 (`opposition +0.92`, `inside_frac 0.84`인데 `cube_lift 0.002 m`).

### 절대 다시 넣지 말 것: "손끝 -> 큐브 중심" 거리 reward

- 큐브 중심은 표면에서 `0.03 m` 안쪽이라 **손끝이 도달 불가능한 목표**임. gradient가 항상 큐브 속을 향함.
- `body_weights=(3,1,1)`이면 엄지가 가중평균의 `60%`라 **"엄지 하나만 박고 나머지 방치"가 최적해**가 됨.
- 거리 reward는 접촉도 처벌함. 만지면 큐브가 밀려나 거리가 늘어남.
- cage reward는 반대로 물체를 파고들어야 점수가 남. **접촉이 이득**임. 이 부호 차이가 파지 학습의 핵심임.
- 이 이유로 `finger_cube_reach`, `finger_cube_closeness`, `functional_hold`, `arm_cube_reach`와 그 함수들을 전부 삭제함.

### 손바닥 방향 reward (`palm_facing`)는 검토 후 철회함

- 논문이 `r_hr`(목표 손 회전)을 주는 이유는 **기능(functional grasp)** 때문임. 드릴을 "트리거를 당길 수 있게" 쥐어야 함.
- **큐브에는 기능 요구가 없으므로 목표 회전이 필요 없는 것이 맞음.** 사람이 정하면 자의적이고, 젓가락에서 `r_hr`로 교체되며 버려짐.
- **목표 회전은 "잡기"가 아니라 "쓰기"에서 나옴.** 젓가락에는 반드시 필요함.
- 참고: `palm_link`의 손바닥 법선은 실측 결과 로컬 **`+x`**축임 (손가락이 오므라들 때 손끝이 이동하는 방향).

### 역할 분담 (중요)

- **cage는 자세를 유도하지 않음.** "물체가 손가락 사이에 있는가"만 봄. 6점 cage가 손바닥 하늘로 수렴한 것이 반증임.
- `finger_cage_reach` -> 파지 간극을 물체 위로.
- `finger_cage_hold` -> 오므려라.
- `cube_lift` -> 들어라. **자세를 결정하는 것은 이것, 정확히는 물리임.**

### TensorBoard 판정 지표

- `Metrics/cube/*`는 에피소드 **평균**이라 성능 지표가 아님. 앞 4 step이 평균의 `77%`를 지배함.
- 평가는 `Metrics/cube_final/*` (마지막 step, 정착 자세)로 할 것. `cube_min/*`, `cube_max/*`도 있음.
- 핵심: `cube_final/cube_lift` (**이것이 전부임**), `thumb_index_opposition`, `thumb_middle_opposition`, `cage_inside_frac`, `cage_span`, `*_surface` (음수=관통).
- **`opposition`만으로 자세를 판정하지 말 것.** 손바닥 방향을 보지 못하므로 `+0.5`를 넘어도 손바닥이 하늘일 수 있음.
- **검지 opposition을 따로 볼 것.** 중지만 보면 검지 교차를 놓침.
- TensorBoard cube distance metric은 `Metrics/cube/*`로 기록함.
- `Metrics/cube/palm_distance`는 `palm_link`와 cube root 사이 실제 거리임.
- `Metrics/cube/thumb_distance`, `index_distance`, `middle_distance`, `ring_distance`, `little_distance`는 각 fingertip과 cube root 사이 실제 거리임.
- `Metrics/cube/finger_mean_distance`는 five-fingertip 거리 평균임.
- `Metrics/cube/non_thumb_mean_distance`는 index/middle/ring/little 거리 평균임.
- `Metrics/cube/finger_weighted_mean_distance`는 thumb을 크게 본 five-fingertip 거리 평균임.
- cube distance metric은 reward가 아니라 error 확인용 logging임.
- contact 기반 grasp reward는 아직 구현하지 않음.
- lift reward와 lifted-gated cube goal reward는 구현됐지만 현재 active reward에서는 비활성화됨.
- 6D arm-only `Indy-Wuji-Cube-Grasp` headless smoke test는 통과함.
- 최신 18D action + 57D observation + active reward 3개 smoke test는 통과함.

## Wuji Finger Naming

- Wuji hand는 5개 finger group으로 정리함.
- `finger1`은 `thumb`으로 취급함.
- `finger2`는 `index`로 취급함.
- `finger3`은 `middle`로 취급함.
- `finger4`는 `ring`으로 취급함.
- `finger5`는 `little`로 취급함.
- 코드에서는 당분간 USD/URDF 이름인 `finger[1-5]_joint[1-4]`, `finger[1-5]_link[1-4]`, `finger[1-5]_tip_link`를 유지함.
- reward/contact 문서와 설계에서는 thumb/index/middle/ring/little alias를 사용함.
- 현재 cube grasp active finger는 `finger1`, `finger2`, `finger3`만 사용함.

## Frame Notes

- `palm_link`는 물리 hand body임.
- `palm_link`를 그대로 task EE orientation으로 쓰면 frame mismatch가 생김.
- raw `palm_link` tracking에서는 orientation error가 약 2 rad 근처로 크게 남았음.
- `link6` tracking은 arm flange 기준 reach baseline으로 사용함.
- command orientation을 임시로 `roll=-pi/2`, `pitch=-pi/2`, `yaw` 자유로 두면 orientation error가 약 0.18까지 떨어졌음.
- 위 결과로 Wuji hand frame과 Indy reach EE command frame 사이에 고정 회전 offset이 있다고 판단함.
- Allegro 쪽도 손 base를 직접 tracking하기보다 `tcp` frame과 hand base를 fixed joint/offset으로 분리하는 구조임.
- 다만 약 2000 iteration 학습 후 orientation error가 2점대에서 약 0.8까지 내려감.
- `palm_link`와 `link6` 모두 충분히 학습하면 orientation error가 유사하게 내려가는 것으로 봄.
- 따라서 현재는 URDF/offset 문제로 단정하지 않음.
- 학습 시간과 reward 구조 영향이 컸던 것으로 보고 long-run 결과를 먼저 확인함.

## Reward Notes

- reward term 설정은 `isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py`에 있음.
- reward 계산 함수는 `isaac_neuromeka/mdp/rewards.py`에 있음.
- position reward는 target과 body position 거리 기반 bounded reward임.
- orientation reward는 position-only baseline에서 제거함.
- end-effector speed penalty와 joint velocity penalty도 현재 제거함.
- action rate penalty는 남김.
- 현재 `Indy-Wuji-Reach`에서는 position reward가 `link6` 기준으로 적용됨.
- TensorBoard에서 weighted reward는 `Episode_Reward/*`, raw reward는 `Episode_Reward_Raw/*`로 확인함.

### Cube Grasp Reward (2026-07-11 전면 재설계)

- active reward는 `finger_cage_reach` (`0.3`), `finger_cage_hold` (`1.0`), `action_rate` (`-0.0003`) 3개뿐임.
- 둘 다 Dexterous Pre-grasp Manipulation 논문 방식이며, **같은 6개 가상점** 위에서 동작함.
- 가상점은 엄지끝(`finger1_tip_link`)과 중지(`finger3_tip_link`, `finger3_link3`) 사이에 비율 `[0.25, 0.50, 0.75]`로 배치함. 선분 A는 핀치 파지, 선분 B는 파워 파지 위치임.
- `finger_cage_hold` (Eq.15): 가상점이 큐브 **내부**로 파고든 깊이를 보상함. **오므리기가 직접 보상됨.** 접촉센서 불필요함. 큐브 SDF는 해석식임.
- `finger_cage_reach` (Eq.14): 같은 6점의 큐브 **표면**까지 SDF의 차분. 파지 간극을 큐브 위로 끌어옴.
- 가중치는 `hold(1.0) >> reach(0.3)`. 논문의 `r_T >> r_orient >> r_hold >> r_reach` 순서임.

### 절대 다시 넣지 말 것: "손끝 -> 큐브 중심" 거리 reward

- 큐브 중심은 표면에서 `0.03 m` 안쪽이라 **손끝이 도달 불가능한 목표**임. gradient가 항상 큐브 속을 향함.
- `body_weights=(3,1,1)`이면 엄지가 가중평균의 `60%`라 **"엄지 하나만 박고 나머지 방치"가 최적해**가 됨.
- 그 자세에서는 엄지-중지 선분이 큐브를 관통하지 않아 오므리면 가상점이 큐브 밖으로 빠져나감. 즉 **파지가 손해**가 됨.
- 거리 reward는 접촉도 처벌함. 만지면 큐브가 밀려나 거리가 늘어남.
- cage reward는 반대로 물체를 파고들어야 점수가 남. **접촉이 이득**임. 이 부호 차이가 파지 학습의 핵심임.
- 이 이유로 `bodies_to_object_position_tracking_bounded`, `object_in_functional_grasp_region`, `BodiesToObjectProgressReward`를 삭제함.

### Progress reward 규칙

- `mode="previous"` + `clamp(min=-1)` + `reset()`에서 기준선 seeding, 셋을 모두 해야 함.
- `clamp(min=0)`이면 후퇴가 공짜이고, 기준선을 첫 `__call__`에서 잡으면 첫 액션이 기준선을 공짜로 부풀림 (swing-out).
- `distance_max`는 실제 step당 최대 개선량보다 충분히 커야 함. 포화되면 "천천히 접근하기"를 보상하게 됨.

### TensorBoard 지표

- `Metrics/cube/*`는 에피소드 **평균**이라 성능 지표가 아님. 앞 4 step이 평균의 `77%`를 지배함.
- 평가는 `Metrics/cube_final/*` (마지막 step, 정착 자세)로 할 것. `cube_min/*`, `cube_max/*`도 있음.
- 핵심 지표는 `cube_final/thumb_middle_opposition` (`+1`=큐브 양쪽, `-1`=같은 쪽), `cage_inside_frac`, `cage_span`, `*_surface` (음수=관통), `cube_lift`임.

## Reward Study Notes

- 다음 단계 cube grasp baseline은 먼저 oracle state 기반으로 구성하는 쪽이 적절함.
- 여기서 oracle state는 simulator가 직접 알고 있는 cube pose, fingertip pose, contact, lift height, velocity 같은 정답 상태를 뜻함.
- oracle 정보를 policy observation에 넣으면 oracle observation policy이고, reward/success 계산에만 쓰면 oracle reward/success condition임.
- 지금은 point cloud, force/tactile sensor부터 넣지 않음.
- 처음에는 cube pose/fingertip pose/contact/lift 같은 sim state를 reward 계산에 쓰고, point cloud/force는 real transfer나 젓가락 미세 접촉 단계에서 검토함.
- grasp 판정은 `palm_link`-cube 거리만으로 정의하면 약함.
- grasp success는 contact + lift + 안정성 기준을 같이 봐야 함.
- 후보 구조는 thumb 또는 palm 쪽 contact, non-thumb finger contact 1~2개 이상, cube lift threshold, cube velocity 안정성임.
- 현재 cube grasp의 주 논문 목표는 functional grasp/pre-grasp 계열 아이디어를 Wuji hand에 맞게 검증하는 것임.
- DexPoint는 보조 참고 자료임. DexPoint 전체 구현이나 재현이 목표가 아님.
- DexPoint에서 가져갈 것은 fingertip-object reach, contact group, contact-gated lift, action/velocity penalty 같은 안정적인 grasp-shaping 패턴임.
- DexPoint 논문식 contact는 thumb contact + other finger 2개 이상이지만, 공개 코드식 완화 조건은 finger/palm contact group count 2개 이상임.
- `indy_wuji_right` 초기 구현에서는 완화 조건으로 시작하고, 학습이 되면 thumb + non-thumb finger 조건으로 강화하는 방향이 좋음.
- DexPoint lift reward는 contact가 성립했을 때만 켜는 gate 구조가 중요함.
- TriFinger transfer 논문은 grasp 자체보다 object 6-DoF pose goal tracking reward 참고용임.
- TriFinger 핵심은 cube/object의 8개 keypoint current-target distance로 position과 orientation을 함께 보는 object-goal reward임.
- TriFinger reach reward는 현재 fingertip-object 거리 자체가 아니라 `curr_dist - prev_dist`에 음수 weight를 곱하는 접근 progress reward임.
- TriFinger reach reward는 초반 exploration용이고, 후반에는 curriculum으로 꺼서 regrasp/finger gaiting을 방해하지 않게 함.
- TriFinger fingertip velocity penalty는 손가락이 너무 빠르게 튀는 움직임을 줄이는 용도임.
- SimToolReal은 tool-use/chopstick 단계 참고용임.
- SimToolReal reward는 `r = r_smooth + r_grasp + I_grasped * r_goal` 구조임.
- SimToolReal의 `r_grasp = r_approach + (1 - I_grasped) * r_lift`이고, lift 이후에는 object-centric goal pose progress reward가 주도함.
- SimToolReal은 tool의 grasp bounding box, object keypoint, goal pose trajectory를 써서 잡은 물체를 목표 pose sequence로 움직이는 관점이 중요함.
- Dexterous Pre-grasp Manipulation 논문은 functional grasp 전 object reposition/reorient/regrasp reward의 주 기준임.
- Pre-grasp 논문 흐름은 cube proxy 단계부터 염두에 두고, 이후 hand 20축 action과 functional grasp/젓가락 파지로 확장함.
- Pre-grasp 논문의 핵심 reward는 `r_man = r_reach + r_hold + r_orient`임.
- Pre-grasp의 `r_hold`는 단순 fingertip 거리나 contact count가 아니라 object가 thumb-finger 사이 공간에 들어왔는지를 보상하는 cage-like reward라서 Wuji hand grasp에 유용함.
- Pre-grasp는 explicit target grasp와 constraint-based target grasp를 비교함.
- explicit target grasp는 object 기준 EE pose와 hand joint target을 직접 주는 방식이고, 성능은 높지만 물체별 target grasp 정의 부담이 큼.
- constraint-based target grasp는 index fingertip target position과 EE orientation 같은 기능 조건만 주고, fake success를 막기 위해 lift reward/condition을 추가함.
- Pre-grasp는 curriculum이 중요함. 먼저 가까운 nominal pose에서 grasp를 배우고, 이후 다양한 object pose에서 pre-grasp manipulation을 학습함.
- 현재 해석은 Functional/Pre-grasp 논문이 "functional grasp가 가능하도록 object를 손 안/손 앞에서 정렬하고 유지하는 reward"의 주 기준이고, DexPoint는 "contact/lift gate를 안정적으로 넣는 보조 참고", TriFinger는 "잡은 뒤 object pose를 맞추는 reward 참고", SimToolReal은 "object pose tracking을 tool-use trajectory로 확장한 참고"임.
- cube grasp 구현 권장 순서는 functional grasp hold/cage baseline, contact condition, contact-gated lift, 이후 object goal/keypoint tracking 순서임.

## Asset Notes

- Wuji collision 문제는 `indy7_wuji_right_simplified.usd` 기준으로 post-process 처리함.
- 26개 Wuji hand collision STL을 USD Mesh collider로 삽입함.
- 직접 삽입한 collision mesh prim에 `PhysicsCollisionAPI` 등을 적용함.
- active hand collision mesh 수는 26개로 검증함.
- arm collision은 simplified collision 사용함.
- hand collision은 Wuji `*_collision.STL` convex hull 기반임.
- `indy7_wuji_right_all_simplified.usd`는 fallback/debug용이었으나 현재 git status에서는 삭제 상태로 보임.
- `indy7_wuji_right.usd`는 full mesh/reference baseline 성격임.

## Study Order

- Direct Cartpole 봄.
- Isaac-Ant-v0 봄.
- Neuromeka Indy-Reach 우선 봄.
- Neuromeka Indy-Wuji-Reach 현재 구현 봄.
- IsaacLab Franka Reach는 공식 ManagerBased 구조 비교용으로 봄.
- KUKA Allegro/Dexsuite는 hand 확장할 때 봄.
- 실행 흐름과 핵심 코드 연결은 root `flow_study.md`에 정리함.
- cube grasp reward 설계용으로 functional grasp/pre-grasp 논문을 주 기준으로 보고, DexPoint, TriFinger transfer, SimToolReal reward 구조는 보조 참고로 정리함.

## Working Rules

- 관련 없는 IsaacLab core 파일은 수정하지 않음.
- 기존 예제를 직접 덮어쓰기보다 새 task/env로 구성함.
- 변경 전 `git status` 확인함.
- 큰 변경 전 계획 요약함.
- 작업 후 `WORKLOG.md` 기록함.
- active repo 내부 작업 기록은 `nrmk_isaaclab_wuji/worklog.md`에도 남김.
- 실행/학습 테스트는 `env_isaaclab` 기준임.
- 코드 수정 후 작은 테스트부터 실행함.
- 학습 smoke test는 `--num_envs 1 --max_iterations 1`부터 함.
- 이후 `128/20`, `512/500`, `4096/50000` 순서로 키움.
- commit은 사용자가 확인 후 진행함.
- 임의 commit 하지 않음.
- 사용자가 남긴 변경은 되돌리지 않음.

## Useful Commands

- 일반화된 실행/학습/play 명령은 root `CLI.md`에 정리함.
- 실행 흐름 공부용 문서는 root `flow_study.md`에 정리함.

- env 활성화함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
```

- 1회 smoke test 실행함.

```bash
python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 1 --max_iterations 1
```

- 중간 테스트 실행함.

```bash
python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 128 --max_iterations 20
```

- 긴 학습 실행함.

```bash
python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 4096 --max_iterations 50000
```

- TensorBoard 실행함.

```bash
tensorboard --logdir logs/rsl_rl/indy_wuji_reach --port 6006 --reload_interval 5
```

- GUI 확인 실행함.

```bash
python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --num_envs 1 --max_iterations 1
```

- cube grasp smoke test 실행함.

```bash
python scripts/rsl_rl/train.py --task Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1
```
