# Indy7 + Wuji 파지·운반 강화학습 중간점검

- 정리 범위: 2026-07-08 ~ 2026-07-20
- 기준 자료: `ACTIVITY_2026-07-08.md` ~ `ACTIVITY_2026-07-19.md`, 현재 코드, 최신 run의 `params/env.yaml`
- 현재 진행 중인 실험: `Indy-Wuji-Box-Transport`, run `2026-07-20_10-07-55`

## 1. 연구 목표와 현재 위치

최종 목표는 Indy7 로봇 팔과 Wuji hand가 테이블 위의 젓가락을 기능적으로 파지하고, 이후 젓가락으로 물체를 집어 운반할 수 있는 정책을 만드는 것이다. 현재까지는 이 문제를 바로 풀기보다 다음 순서로 난도를 분해했다.

1. Indy7 팔의 기본 목표점 도달 능력 검증
2. Wuji hand로 단일 큐브 접근·파지·들기 검증
3. 잡은 물체를 공중 목표 위치까지 운반
4. 크기가 다른 직육면체로 일반화
5. 목표 위치뿐 아니라 목표 orientation까지 맞추기
6. 이후 직육면체를 젓가락 프록시로 확장하고 tool-ready state를 학습

현재까지의 핵심 결과는 다음과 같다.

- `Indy-Wuji-Reach`에서 팔 6축 position reach baseline을 구축했다.
- `Indy-Wuji-Cube-Grasp`에서 접근, SDF cage 파지, 실제 lift, 목표 위치 운반까지 연결했다.
- 큐브 운반은 run `2026-07-16_16-05-23`에서 성공률 89.4%, 0.1 kg 비교 run `2026-07-17_23-06-15`에서 98.2%까지 확인했다.
- `Indy-Wuji-Box-Transport`를 별도 태스크로 만들고, 병렬 환경마다 서로 다른 직육면체 크기를 물리에 실제 반영했다.
- 위치 운반만으로는 직육면체가 약 70~90도 기울어진 채 매달리는 문제가 남아, 현재는 position과 orientation을 분리한 2단계 운반 보상을 학습 중이다.
- 2026-07-20 현재 cage는 엄지-검지, 엄지-중지의 tip-only 6점 구조이며, 고정된 45도 yaw 목표와 15도 성공 기준을 사용한다.

중간 결론은 단순히 reward weight를 키우는 것으로 파지가 해결되지 않았다는 것이다. 실제 성능을 결정한 것은 장면 기하, action 정의, 관절 구동 가능 범위, 물체 표면 기준 reward, 파지 gate, 성공 종료, 그리고 이를 확인하는 metric을 함께 맞춘 과정이었다.

### 주요 참고 논문과 적용 범위

| 표기 | 논문 | 이 프로젝트에서 참고한 부분 |
|---|---|---|
| `[P1]` | *Dexterous Pre-grasp Manipulation for Human-like Functional Categorical Grasping: Deep Reinforcement Learning and Grasp Representations* (Pavlichenko and Behnke, arXiv:2307.16752v2) | 엄지-대향 손가락 사이 가상점과 SDF 기반 `reach`/`hold`, 차분형 유도 보상, lift를 포함한 안정 파지 조건, `reach < hold < orient < terminal`의 단계적 보상 구조 |
| `[P2]` | *DexPoint: Generalizable Point Cloud Reinforcement Learning for Sim-to-Real Dexterous Manipulation* (Qin et al., arXiv:2211.09423v2) | 손가락 접근 이후 contact가 성립해야 lift가 열리는 gate 구조와 action·velocity·controller penalty 설계 |
| `[P3]` | *Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger* (Allshire et al., arXiv:2108.09779v2) | 물체 8개 keypoint로 위치와 orientation을 함께 나타내는 6D pose reward |
| `[P4]` | *SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation* (Kedia et al., arXiv:2602.16863v2) | grasp 이후 물체·도구의 goal pose trajectory를 추종하는 object-centric 정책 구조 |
| `[P5]` | *Learning to Use Chopsticks in Diverse Gripping Styles* (Yang, Yin, and Liu, arXiv:2205.14313v3) | 안정적인 젓가락 파지 자세를 먼저 확보하고 이후 물체 relocation controller로 연결하는 최종 정책 구상 |

현재 코드는 위 논문들의 완전한 재현이 아니다. `[P1]`의 가상점·SDF가 파지 reward의 주 기반이고, `[P2]`와 `[P3]`는 각각 gate와 물체 자세 제어를 보완하는 참고 자료다. `[P4]`, `[P5]`는 아직 active reward가 아니라 젓가락 단계의 설계 근거다. env별 box 크기, 최저 꼭짓점 clearance, best-so-far potential, tip-only 6점, position 이후 orientation stage는 이 프로젝트의 물리 조건과 실패 사례에 맞춘 변형이다.

## 2. 전체 진행 과정

| 시기 | 질문 | 봉착한 문제 | 해결 및 검증 | 현재 결과물에 남은 내용 |
|---|---|---|---|---|
| 07-08 | 현재 Isaac 환경에서 Indy7+Wuji를 정상 구동할 수 있는가 | 이전 Isaac Sim 4.5/IsaacLab 2.2.1 실험과 현재 환경이 섞였고, Wuji hand frame 해석도 불명확했음 | Isaac Sim 5.1, IsaacLab 2.3, `env_isaaclab`로 기준을 통일하고 `link6` 기준 reach를 먼저 검증 | 현재 active workspace와 `Indy-Wuji-Reach` baseline |
| 07-09 | RL 환경에서 command, observation, action, reward가 어떻게 연결되는가 | 관측과 reward가 많아 어떤 신호로 학습되는지 해석하기 어려웠음 | reach를 arm joint 6D + target xyz 3D + previous action 6D의 15D 관측으로 축소하고 position reward만 남김 | 최소 reach baseline, raw/weighted TensorBoard logging |
| 07-10 | 손가락을 포함해 cube를 잡을 수 있는가 | fingertip-to-center reward가 접근은 만들었지만 파지를 만들지 못함. 4096 env에서 PhysX patch overflow도 발생 | cube grasp 태스크를 별도 등록하고 GPU rigid patch buffer를 `2**20`으로 확대 | `Indy-Wuji-Cube-Grasp`, 대규모 병렬 학습 설정 |
| 07-11 | 왜 손가락이 큐브에 가까워져도 잡지 않는가 | cube 중심은 손끝이 도달할 수 없는 목표이고, thumb 가중치가 큰 거리 평균은 엄지 하나만 넣는 정책을 유도 | fingertip center-distance reward를 폐기하고 논문의 가상점 + box SDF 기반 reach/hold로 교체 | `_box_signed_distance`, `cage_points`, `ObjectCageProgressReward`, `object_in_finger_cage` |
| 07-12 | cage 점수가 높은데 왜 자세가 이상한가 | 엄지-중지 6점만으로는 검지가 자유롭고, 손바닥이 위를 보는 자세도 점수를 받을 수 있었음 | 검지까지 포함한 12점 cage와 opposition/span/final metric을 도입 | 다지점 cage 실험 기반과 상세 파지 metric |
| 07-12 | cube 중심 높이가 올랐는데 실제 lift인가 | 큐브를 기울여 중심만 올리고 한 모서리는 바닥에 둔 reward hacking | 8개 꼭짓점의 최저 높이인 `cube_clearance`로 lift를 재정의하고 cage gate를 곱함 | `box_ground_clearance`, `object_lift_in_cage` |
| 07-13 | 손이 cube를 향하지만 접근하지 않는 이유는 무엇인가 | 절대형 `palm_facing`이 전체 보상의 98.6%를 차지해 멀리서 방향만 맞추는 local optimum 발생 | facing을 progress형으로 바꾸고, 유도 보상은 차분형·유지 보상은 절대형으로 구분 | reward 형태 선택 원칙과 `PalmFacingProgressReward` |
| 07-13 | 접촉 순간 손과 팔이 튀는 원인은 물리인가 정책인가 | 상태만 봐서는 충돌 폭발과 잘못된 action을 구분할 수 없었음 | raw/applied action, target, actual, tracking error, velocity, torque를 출력해 정책 raw action 발산을 확인 | `play.py` action diagnostics와 `action_track_err`, `action_delta` metric |
| 07-13 | 왜 정책 action이 발산했는가 | `clip_actions`가 없고 action scale이 작아 정책이 큰 raw 값을 출력해 도달 범위를 보상하려 했음 | `clip_actions=1.0`, action scale과 도달 범위를 함께 재설계하고 action-rate penalty 강화 | 현재 clipped absolute joint target 구조 |
| 07-13~14 | 왜 접촉과 hold가 켜지지 않는가 | 정책이 제어하지 않는 약지·새끼가 펴진 채 cube를 쳐냈고, 일부 관절은 목표를 물리적으로 추종하지 못했음 | 약지·새끼를 중지 action에 mimic coupling하고, scripted contact/lift probe와 grip-capacity test로 물리 가능성을 분리 검증 | 18D policy action을 유지하면서 5지 목표를 만드는 `MimicJointActionCfg` |
| 07-14~15 | 손이 바닥 아래로 내려가거나 cube 높이 기준이 틀리는 이유는 무엇인가 | cube와 손의 시작 높이가 맞지 않았고, lift가 월드 바닥 기준이면 테이블 위 스폰부터 잘못된 값을 가짐 | 받침 테이블을 추가하고 `surface_z=BASE_Z`를 lift, floor, success, metric에 공통 적용 | 현재 table scene과 surface-relative clearance |
| 07-15 | 순간 lift가 아니라 안정적인 성공을 어떻게 정의할 것인가 | hold/lift 연금을 오래 받거나 목표를 순간 통과하는 정책 가능 | goal 반경, cage gate, 연속 유지 시간을 모두 만족하면 terminal reward를 한 번 주고 즉시 종료 | `ObjectAtGoalHeld`, `transport_success`, `hold_steps=15` |
| 07-15~16 | lift 다음에 목표 위치까지 어떻게 운반시킬 것인가 | 선형 거리 progress는 먼 곳과 가까운 곳의 1 cm를 같게 평가해 목표를 지나치는 현상 발생 | `phi(d)=eps/(eps+d)`의 best-so-far position progress를 사용해 근거리 개선을 크게 평가 | `ObjectToGoalProgressReward` |
| 07-16 | 하나의 cube에서 다른 크기 물체로 어떻게 확장할 것인가 | 모든 병렬 환경이 같은 크기면 일반화 여부를 확인할 수 없음 | env별 prestartup scaling, 실제 half extent buffer, size/quat observation, `replicate_physics=False` 적용 | `Indy-Wuji-Box-Transport`, obs 64에서 시작한 random box 구조 |
| 07-17~18 | lift reward를 없애도 transport만으로 들고 있을 수 있는가 | lift weight 0에서도 목표 근처까지 갔지만, progress를 받은 뒤 내려놓고 hold를 유지해 success 0% | lift가 공중 유지 사다리 역할을 한다고 판정하고 weight 50을 유지 | 현재 box `cube_lift=50`, 포화 높이 20 cm |
| 07-18 | success에 orientation 조건만 추가하면 자세가 맞는가 | orientation 15도 조건만 넣자 success가 0이었고, 자세 오차는 72~93도로 악화 | orientation을 판정만 하지 말고 dense shaping과 observation을 함께 제공해야 한다고 판정 | symmetry-aware orientation error, orientation observation/reward |
| 07-18~19 | keypoint pose reward를 주면 orientation이 해결되는가 | 유인은 생겼지만 손끝 2점 파지는 긴 박스를 경첩처럼 매달아 회전시킬 물리적 수단이 부족 | WRAP 30점 cage로 접촉 분산을 늘리는 실험을 수행하고 파지 기하 문제를 분리 | WRAP cage 정의를 비교용 코드로 보존 |
| 07-19 | 얇은 스틱을 왜 잡지 못하는가 | 파지 지점이 hand-floor penalty 영역과 겹치고, 엄지 단독 부분점수 및 말단 관절 토크 포화가 발생 | floor clearance/weight를 완화하고 torque/action 진단으로 reward 충돌과 구동 한계를 구분 | `hand_floor` clearance 1 cm, weight 0.2, 스틱 복귀 조건 카드 |
| 07-20 | orientation과 tip pinch를 단순한 기준선에서 다시 검증할 수 있는가 | WRAP은 점 평균 희석과 파지 목적 혼합으로 원인 해석이 복잡했음 | tip-only 6점으로 되돌리고 position transport와 orientation progress를 분리. 양쪽 선분 참여를 요구하도록 gate를 0.6으로 강화 | 현재 run `2026-07-20_10-07-55` |

## 3. 핵심 문제와 해결 논리

### 3.1 물체 중심 거리 reward를 폐기한 이유

초기에는 각 fingertip과 cube 중심의 거리를 줄이는 reward를 사용했다. 그러나 6 cm cube의 중심은 표면에서 3 cm 안쪽이므로 손끝이 물리적으로 도달할 수 없는 목표다. 접촉해 cube가 밀리면 중심 거리가 다시 커져 접촉 자체가 손해가 되는 문제도 있었다.

현재는 물체 중심이 아니라 손가락 사이에 만든 가상점과 box 표면의 signed distance를 사용한다.

```text
cage point = thumb_tip + fraction * (opposing_tip - thumb_tip)
sdf < 0  : 가상점이 box 내부
sdf = 0  : box 표면
sdf > 0  : box 바깥
```

이 변경으로 “손가락 하나가 중심에 가까움”이 아니라 “물체가 엄지와 대향 손가락 사이에 들어옴”을 직접 평가할 수 있게 됐다.

이 가상점·SDF 표현은 `[P1]`의 manipulation reach/hold 식을 기반으로 하며, Wuji hand에서는 엄지-검지와 엄지-중지 선분을 사용하도록 확장했다.

### 3.2 reach와 hold를 분리한 이유

파지에는 서로 다른 두 동작이 필요하다.

- `finger_cage_reach`: 파지 간극 전체를 물체 쪽으로 이동
- `finger_cage_hold`: 물체가 손가락 사이에 들어온 뒤 손가락을 오므려 유지

reach는 이전 step보다 SDF 평균이 줄어든 만큼만 주는 차분형이다.

```text
r_reach = clamp((sdf_prev - sdf_now) / distance_max, -1, 1)
```

hold는 각 가상점이 물체 안으로 들어간 정도의 절대값 평균이다.

```text
penetration_i = sphere_radius - sdf_i
r_hold = mean(clamp(penetration_i / (sphere_radius + depth_max), 0, 1))
```

유도 과정인 reach를 절대형으로 주면 가까이 있기만 하면서 보상을 계속 받을 수 있다. 반대로 hold는 유지 자체가 목표이므로 절대형 보상이 필요하다. 이 구분은 이후 facing, transport, orientation reward를 설계하는 기준이 됐다.

차분형 `reach`와 절대형 `hold`의 역할 구분 및 보상 단계의 상대적 크기는 `[P1]`을 참고했다. 다만 현재의 tip-only 6점 배치와 파라미터는 Wuji hand와 직육면체 실험에서 다시 정한 값이다.

### 3.3 실제 lift와 기울이기 편법을 구분한 방법

초기 lift는 cube 중심 z 증가량을 사용했다. 이 방식에서는 cube를 모서리로 세우기만 해도 중심이 올라가므로 실제 lift처럼 보였다.

현재 lift는 회전된 직육면체 8개 꼭짓점 중 가장 낮은 점과 테이블 표면의 거리를 사용한다.

```text
clearance = min(corner_z) - surface_z
r_lift = cage_gate * clamp(clearance / lift_height, 0, 1)
```

따라서 한 모서리라도 테이블에 닿아 있으면 clearance는 0에 가깝고, 손가락으로 잡지 않은 채 물체를 튕겨 올리는 경우에는 cage gate가 reward를 차단한다.

파지 조건을 만족해야 lift 신호를 여는 발상은 `[P1]`의 constraint-based lift와 `[P2]`의 contact-gated lift를 참고했다. 8개 꼭짓점의 최저 높이로 기울이기 편법을 막는 `clearance` 계산은 본 프로젝트에서 추가한 보정이다.

### 3.4 reward 값이 0일 때 weight를 키우지 않은 이유

한동안 `Episode_Reward_Raw/cube_lift`가 전체 학습에서 정확히 0이었다. 이때 weight를 3에서 50 또는 500으로 키워도 결과는 계속 0이다. 먼저 scripted probe로 실제 contact와 lift가 가능한지 확인한 뒤 학습 신호를 조정해야 했다.

이 원칙 때문에 다음 디버그 도구를 별도로 만들었다.

- `scripts/debug/check_cube_contact_lift.py`: contact와 clearance 확인
- `scripts/debug/grip_capacity.py`: 정책을 제외하고 물리적 파지·lift 가능성 확인
- `scripts/debug/policy_joint_diagnostics.py`: 관절별 target/actual/error/velocity/torque 확인
- `scripts/debug/box_dims_probe.py`: env별 직육면체 실제 치수 확인
- `scripts/debug/box_obs_probe.py`: box pose, size, orientation observation 배선 확인

### 3.5 action과 물리 문제를 분리한 방법

팔이 튀거나 손가락이 따라가지 못할 때 처음에는 충돌·solver 문제로만 해석하기 쉬웠다. 그러나 action을 직접 출력해 다음 두 경우를 분리했다.

```text
tracking error 작음 + action delta 큼 -> 정책이 큰 변화를 명령함
tracking error 큼                    -> actuator/접촉/물리가 목표를 못 따라감
```

실측 결과 정책 raw action이 정의 범위를 크게 벗어나고 있었으며 `clip_actions`가 없었다. 현재는 `clip_actions=1.0`을 사용하며 action은 다음 절대 관절 목표로 변환된다.

```text
target_joint_pos = default_joint_pos + scale * clipped_action
```

현재 Box Transport의 scale은 1.0이다. 약지와 새끼는 action dimension을 추가하지 않고 중지 목표를 mimic한다. 이로써 policy는 arm 6축과 thumb/index/middle 12축, 총 18D만 출력하지만 실제로는 5개 손가락 모두 매 step 목표를 받는다.

action·velocity·controller penalty를 함께 확인하는 관점은 `[P2]`를 참고했지만, 현재의 clipped absolute joint target과 mimic coupling은 프로젝트 고유의 제어 구성이다. `[P1]`의 arm pose delta + IK 구조를 그대로 사용한 것은 아니다.

### 3.6 scene geometry를 reward보다 먼저 맞춘 이유

초기 close-start는 cube x/y만 손에 가까웠고 z는 바닥에 있어 실제로는 손과 높이가 맞지 않았다. 로봇은 cube를 잡기 위해 크게 웅크렸고, 테이블이 없는 장면에서는 floor 관련 reward도 실제 작업과 다른 기하를 학습시켰다.

현재는 다음 기준을 공통으로 사용한다.

- table surface: `BASE_Z=0.25 m`
- box spawn z: `BASE_Z + env별 box half-height`
- lift clearance 기준: `surface_z=BASE_Z`
- hand-floor penalty 기준: `surface_z=BASE_Z`
- drop termination 기준: `BASE_Z - 0.05 m`
- goal z: `BASE_Z + 0.20 m = 0.45 m`

이 배선은 시각적 배치뿐 아니라 reward와 termination이 같은 좌표 기준을 보도록 만든다.

### 3.7 transport reward와 terminal success를 나눈 이유

목표 위치까지의 progress만 주면 목표를 잠깐 통과하거나, 한 번 progress reward를 받은 뒤 다시 내려놓을 수 있다. 반대로 목표 근처 절대 reward를 계속 주면 캠핑과 reward farming이 생긴다.

현재 position transport는 best-so-far 역수 potential progress다.

```text
phi(d) = 0.05 / (0.05 + d)
r_transport = max(phi_now - phi_best, 0) * cage_gate
```

목표에 가까울수록 같은 거리 개선의 가치가 커지고, 이미 받은 progress는 왕복해도 다시 받을 수 없다. 최종 성공은 별도 조건으로 판정한다.

```text
position error < 0.05 m
AND cage gate > 0.6
AND orientation error < 15 deg
AND 위 조건을 15 step 연속 유지
```

성공한 step에는 terminal reward가 한 번 지급되고 episode가 즉시 종료된다. 이는 이미 성공한 상태에서 hold/lift 보상을 계속 누적하는 것을 막는다.

성공 시 큰 terminal reward를 한 번 지급하고 episode를 끝내는 구조는 `[P1]`의 `r_T`를, `reach -> grasp/contact -> lift -> target` 순서의 gate는 `[P2]`를 참고했다. 역수 potential과 best-so-far 적립 방식은 목표 통과 후 왕복하는 현상을 줄이기 위해 본 프로젝트에서 선택했다.

### 3.8 orientation을 별도 단계로 분리한 이유

성공 조건에 orientation 15도만 추가한 v1은 success가 0이었다. 기존 정책이 상자를 72~93도 기울여 운반했고, 이를 줄이는 dense 신호가 없었기 때문이다. 위치와 자세를 8개 꼭짓점 keypoint 거리로 합친 v1.1도 파지 기하가 불안정해 자세가 약 70도에서 정체했다.

현재는 위치와 orientation을 분리했다.

1. 먼저 position transport로 목표 10 cm 안에 접근한다.
2. cage gate가 0.6을 넘으면 orientation stage를 latch한다.
3. 이후 최근접 대칭 orientation error가 줄면 보상, 늘면 감점을 준다.

```text
r_ori = cage_gate * clamp((theta_prev - theta_now) / 45deg, -1, 1)
```

직육면체 단면이 정사각형이므로 길이축 기준 90도 회전과 뒤집기 대칭 8개를 같은 자세로 인정한다. policy에는 현재 quaternion만 주는 대신, 목표의 최근접 등가 자세까지 필요한 회전을 axis-angle 3D로 직접 제공한다.

8개 keypoint로 물체 6D pose를 표현하는 출발점은 `[P3]`이다. 현재의 직육면체 대칭 처리, position 선행 조건, orientation stage latch, signed angular progress, nearest-symmetry axis-angle observation은 두 단계의 원인을 분리하기 위해 추가한 프로젝트 변형이다.

## 4. 현재 최종 구현 구조

### 4.1 태스크 구성

| 태스크 | 역할 | 현재 의미 |
|---|---|---|
| `Indy-Wuji-Reach` | 팔 6축 reach baseline | 프레임과 ManagerBased RL 흐름을 검증한 최소 기준선 |
| `Indy-Wuji-Cube-Grasp` | 정육면체 파지·lift·고정 goal 운반 | reward 사다리와 terminal success를 검증한 테스트베드 |
| `Indy-Wuji-Box-Transport` | env별 랜덤 직육면체 파지·운반·자세 정렬 | 현재 주 실험, 젓가락 프록시로 가기 위한 일반화 단계 |

### 4.2 현재 Box Transport action

- action dimension: 18
- arm: `joint0`~`joint5`, 6D
- active hand: `finger1`~`finger3`의 joint1~4, 12D
- finger4/5: middle finger target을 mimic
- command type: absolute joint position target
- scale: 1.0
- action clip: `[-1, 1]`
- action-rate penalty: `-0.005`

### 4.3 현재 Box Transport observation

| observation | 차원 | 목적 |
|---|---:|---|
| controlled joint position | 18 | 현재 arm/hand 관절 상태 |
| box position relative to palm | 3 | 손과 물체의 상대 위치 |
| box position relative to five fingertips | 15 | 손가락별 물체 위치 |
| box-to-goal position error | 3 | 목표 이동 방향 |
| box size | 3 | env별 랜덤 크기 |
| box quaternion | 4 | 현재 물체 자세 |
| nearest-symmetry orientation error | 3 | 목표 자세까지의 signed axis-angle |
| previous action | 18 | 제어 이력과 진동 억제 |
| 합계 | 67 | 현재 policy input |

`box_size`, `box_quat`, `box_ori_to_target`은 직육면체마다 파지 방향과 회전 응답이 다르기 때문에 추가됐다. `box_obs_probe.py`에서 실제 scene state와 observation을 직접 비교해 오차 0으로 확인했다.

### 4.4 현재 Box Transport command와 랜덤화

- command shape: position xyz + quaternion wxyz = 7D
- 현재 목표 position: `(0.62, -0.20, 0.45)`
- 현재 목표 orientation: roll 0도, pitch 0도, yaw 45도
- command는 episode 안에서 고정
- 기본 box 크기: 단면 3~6 cm, 길이비 1.5~3, 길이축 y
- 각 env는 prestartup에서 서로 다른 크기를 가지며 `env.box_half_extents`에 실제 반크기를 저장
- SDF, clearance, observation은 모두 이 env별 half extent를 사용
- 독립 길이 범위를 사용하면 단면 1.5~6 cm, 길이 최대 20 cm까지 확장 가능

### 4.5 현재 Box Transport reward

RewardManager의 실제 step reward는 다음과 같다.

```text
total_reward_t = sum(raw_term_i * weight_i * dt)
dt = 1/30 s
```

| 순서 | TensorBoard 이름 | 형태 | weight | 역할과 gate |
|---:|---|---|---:|---|
| 1 | `finger_cage_reach` | signed progress | 8 | tip-only cage SDF 감소. 양수 progress에만 facing gate 적용 |
| 2 | `finger_cage_hold` | absolute | 15 | 6개 가상점의 box 침투·물림 정도 |
| 3 | `cube_lift` | absolute | 50 | `cage_gate * clearance/0.20`, 목표 높이까지 공중 유지 |
| 4 | `cube_transport` | best-so-far progress | 4000 | `cage_gate * Delta phi(position_error)` |
| 5 | `box_orientation` | signed progress | 4000 | position < 10 cm, gate > 0.6 이후 orientation error 감소 |
| 6 | `goal_proximity` | absolute | 0 | lift/transport 통합 대안으로 보존된 비활성 실험 항 |
| 7 | `transport_success` | terminal | 30000 | success termination을 한 번의 보상으로 변환 |
| 8 | `drop_penalty` | terminal penalty | 0 | 현재 비활성, 초기 탐색 회피 방지 |
| 9 | `palm_facing` | signed progress | 4 | 파지 개구부가 물체를 향하도록 유도 |
| 10 | `arm_manipulability` | penalty | 1 | 팔 특이점 근처 자세 억제 |
| 11 | `hand_floor` | penalty | 0.2 | 손이 table 아래로 내려가는 것 억제 |
| 12 | `action_rate` | penalty | -0.005 | action 변화와 진동 억제 |

현재 cage는 다음 6개 가상점이다.

```text
thumb_tip -> index_tip  : fractions 0.1, 0.5, 0.9
thumb_tip -> middle_tip : fractions 0.1, 0.5, 0.9
```

WRAP 30점 코드는 삭제하지 않고 비교·복구용으로 주석 보존했다. 현재 실험은 파지 목표를 단순화해 fingertip pinch 자체와 orientation 제어 가능성을 먼저 확인하는 기준선이다.

## 5. 정량 결과와 판정

| 실험 | 결과 | 판정 |
|---|---|---|
| Reach baseline | arm 6D, obs 15D 학습 및 play 확인 | 환경·프레임·학습 흐름 기준선 확보 |
| Cube transport `2026-07-16_16-05-23` | success 89.4%, drop 3.0%, position error 4.6 cm | fixed-goal grasp-lift-transport 사다리 성립 |
| Cube mass 0.1 kg `2026-07-17_23-06-15` | success 98.2%, drop 1.3%, position error 2.8 cm | 0.2 kg보다 초기 학습과 최종 성공률 우세 |
| Box transport `2026-07-16_16-33-21` | success 43.5% 이상, drop 15.9%까지 감소 | 랜덤 직육면체에도 기본 운반 가능 |
| Box lift-off `2026-07-17_23-15-16` | lift reward 0에서 운반은 했지만 success 0% | lift는 공중 정착을 유지하는 필수 층 |
| Orientation v1 `2026-07-18_15-37-42` | success 0, orientation error 72~93도 | 판정 조건만 추가해서는 학습 신호 없음 |
| Keypoint v1.1 `2026-07-18_22-48-57` | success 0, orientation error 약 70도 | 자세 유인만으로는 불안정한 파지 기하를 해결하지 못함 |
| Long-stick `2026-07-18_20-56-53` | final lift 약 7.9 cm, clearance 약 4.6 cm, success 약 24%, drop 약 32% | 일부 파지·lift 가능하지만 얇은 물체 일반화는 아직 불안정 |
| WRAP orientation `2026-07-19_22-44-57` | success 0, final orientation error 약 42도, median clearance 약 2.9 mm | WRAP만으로 안정 lift·orientation success를 만들지 못함 |
| Tip-only orientation `2026-07-20_10-07-55` | 학습 진행 중 | 45도 goal, 15도 success, gate 0.6의 현재 기준선 |

## 6. 최종 결과물에 적용된 연구 원칙

### 6.1 먼저 raw 신호가 존재하는지 확인한다

reward weight를 조정하기 전에 raw reward, contact, clearance가 한 번이라도 발생하는지 본다. raw가 0이면 reward 문제가 아니라 도달 가능성, scene, action, actuator 문제를 먼저 찾는다.

### 6.2 상태와 제어 입력을 같이 본다

물체 pose와 reward만 보면 충돌로 튄 것인지 정책이 튀라고 명령한 것인지 구분할 수 없다. action raw/applied, target/actual, tracking error, velocity, torque를 함께 기록한다.

### 6.3 reward는 유도, 유지, 종료 역할을 분리한다

- 유도: signed progress 또는 best-so-far progress
- 유지: 실제로 유지가 어려운 hold/lift만 absolute reward
- 종료: 성공 조건을 일정 시간 만족했을 때 한 번 지급하고 즉시 reset

이 구조로 쉬운 항을 오래 유지해 reward를 farming하는 문제를 줄였다.

### 6.4 물리적으로 의미 있는 기준을 사용한다

- 물체 중심 거리 대신 표면 SDF
- 중심 높이 대신 최하 꼭짓점 clearance
- 월드 바닥 대신 실제 table `surface_z`
- 단일 손가락 접근 대신 cage gate
- quaternion 단순 오차 대신 직육면체 대칭을 고려한 geodesic error

### 6.5 episode 평균 하나로 성공을 판단하지 않는다

TensorBoard의 `Metrics/cube/*`는 episode 평균이라 초기 멀리 있는 구간의 영향을 크게 받는다. 현재는 다음을 같이 본다.

- `Metrics/cube_final/*`: episode 마지막 상태
- `Metrics/cube_min/*`, `Metrics/cube_max/*`: 순간 최선·최악 상태
- `cube_clearance`: 실제 lift
- `cage_inside_frac`: 가상점 포획 비율
- `thumb_index_opposition`, `thumb_middle_opposition`: 양쪽 파지 여부
- `box_ori_error`: 목표 orientation 오차
- `orientation_stage_active`: orientation stage 진입 여부
- `action_track_err`, `action_delta`: 제어와 물리 추종 상태

### 6.6 checkpoint 호환성은 shape뿐 아니라 의미까지 본다

observation/action dimension이 같아도 action scale, reward 정의, 물리 파라미터가 바뀌면 기존 정책의 의미가 달라진다. 특히 obs 64에서 67로 바뀐 orientation 실험은 fresh run이 필수다. 각 run은 코드 기억이 아니라 `params/env.yaml`로 확인한다.

## 7. 현재 한계

1. 현재 orientation 정책은 아직 성공이 검증되지 않았다. 45도 목표를 향해 실제로 회전하고 15도 안에서 안정적으로 유지하는지 확인해야 한다.
2. tip-only cage는 단순하고 해석하기 쉽지만, 긴 물체의 중력 토크를 버티기에는 접촉 분산이 부족할 수 있다.
3. WRAP cage는 접촉 분산을 늘렸지만 가상점 평균 희석과 부분점수 문제가 있어 그대로 최종안으로 쓰기 어렵다.
4. 얇은 스틱은 table 근접 파지, 말단 관절 torque saturation, 엄지 단독 부분점수 문제가 동시에 존재한다.
5. 현재 goal position과 orientation은 고정이다. random position과 random orientation generalization은 아직 다음 단계다.
6. 현재 정책은 관절공간 absolute action이다. 최종 젓가락 task에서 논문처럼 arm IK delta + finger joint command 구조로 전환할지는 별도 비교가 필요하다.
7. 현재 성공은 world-frame box pose와 cage 유지 기준이다. 최종 젓가락 정책에 필요한 hand-tool 상대 pose와 slip 안정성은 아직 success 조건에 포함되지 않았다.

## 8. 다음 단계

### 즉시 판정할 항목

현재 run `2026-07-20_10-07-55`에서 다음 순서로 본다.

1. `finger_cage_hold`와 `cage_inside_frac`가 tip-only 6점 기준으로 상승하는가
2. `cube_clearance`가 일시적 max가 아니라 final에서도 유지되는가
3. `orientation_stage_active`가 충분한 환경에서 1이 되는가
4. stage 진입 후 `box_ori_error`가 45도 목표 대비 15도 아래로 감소하는가
5. `transport_success`가 발생하고 drop/time-out 비율이 안정되는가
6. play에서 tip pinch가 실제로 물체를 고정하며 회전시키는가

### 이후 확장 순서

1. tip-only 성공 여부 판정
2. 실패 원인을 orientation 유인 부족과 파지 토크 부족으로 분리
3. 필요하면 tip-only와 WRAP 사이의 최소 contact geometry를 설계
4. 고정 45도 목표 성공 후 yaw goal 범위를 랜덤화
5. goal position 범위를 랜덤화
6. 얇고 긴 직육면체 분포로 다시 확장
7. 직육면체 keypoint를 젓가락의 tip/tail/side semantic keypoint로 교체
8. 최종 성공을 world pose뿐 아니라 hand-tool relative pose, slip, 일정 시간 유지까지 확장
9. 정책 A의 성공 상태를 정책 B의 초기 상태 분포로 사용해 젓가락 물체 조작으로 연결

마지막 두 단계는 `[P4]`의 object-centric tool goal trajectory와 `[P5]`의 안정 젓가락 파지/물체 조작 분리 관점을 참고한다. 즉 현재 Box Transport는 최종 젓가락질 정책이 아니라, 다음 정책에 넘길 안정적인 `tool-ready state`를 만드는 전단계로 해석한다.

## 9. 중간발표용 요약

본 연구는 처음부터 젓가락질 전체를 학습하지 않고, reach, grasp, lift, transport, orientation의 연쇄 과제로 분해했다. 초기에는 손끝과 물체 중심의 거리 reward를 사용했지만, 이 목표가 물체 내부에 있어 접촉과 파지를 오히려 방해한다는 문제를 발견했다. 이를 해결하기 위해 `[P1]` Dexterous Pre-grasp의 가상점·SDF 방식을 도입해 손가락 사이 공간과 물체 표면의 관계를 직접 평가했다. lift gate는 `[P1]`, `[P2]`, 6D orientation 표현은 `[P3]`을 참고하되 Wuji hand와 직육면체 물리에 맞게 변형했다.

학습 과정에서는 높은 reward가 곧 올바른 파지를 의미하지 않았다. 물체를 기울여 중심만 올리는 편법, 멀리서 방향만 맞추는 facing reward farming, 제어하지 않는 손가락의 충돌, unclipped action 발산, table 높이와 reward 기준면 불일치가 차례로 나타났다. 각 문제는 TensorBoard 값만으로 판단하지 않고 GUI, scripted probe, contact, clearance, 관절 target/actual/torque 진단을 함께 사용해 원인을 분리했다.

그 결과 cube에서는 접근-파지-lift-운반-성공 종료가 연결된 정책을 확보했고, 이를 env별 크기가 다른 직육면체 task로 확장했다. 현재는 위치 운반과 orientation 정렬을 분리한 2단계 reward, 대칭을 고려한 orientation error, 목표 상대 axis-angle observation, tip-only cage를 이용해 45도 목표 자세 정렬을 학습 중이다. 이 단계가 통과되면 랜덤 pose와 얇은 직육면체로 확장하고, 최종적으로 젓가락의 tip/tail/opening을 표현하는 tool-ready state로 전환할 계획이다.
