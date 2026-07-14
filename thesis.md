# Dexterous Manipulation Reward Study

- 이 문서는 dexterous manipulation 관련 논문 reward 구조만 따로 모은 공부 문서임.

이 문서는 논문 내용만 따로 모은 정리임.
프로젝트 구현 상태, 실행 명령, git/worklog 내용은 제외함.

## Overall Map

| 논문 | 핵심 역할 | reward 관점 |
| --- | --- | --- |
| Dexterous Pre-grasp Manipulation | functional grasp 주 목표 | reach-hold-orient manipulation reward |
| DexPoint | 보조 grasp acquisition 참고 | 손가락 접근, contact, lift gate |
| TriFinger transfer | object 6-DoF pose control | keypoint 기반 object-goal reward |
| SimToolReal | tool-use trajectory | object-centric goal pose progress |

읽는 순서는 다음처럼 잡는 게 자연스러움.

```text
Dexterous Pre-grasp Manipulation
-> DexPoint
-> TriFinger transfer
-> SimToolReal
```

의미는 다음과 같음.

```text
Pre-grasp:
  functional grasp가 가능하도록 물체를 손 안/손 앞에서 정렬하는 주 목표 reward

DexPoint:
  contact/lift gate와 접근 shaping을 보조로 참고하는 reward

TriFinger:
  잡은 물체를 목표 6-DoF pose로 맞추는 reward

SimToolReal:
  object pose tracking을 tool-use trajectory로 확장하는 reward
```

## DexPoint

### Purpose

DexPoint는 dexterous hand가 object를 grasp하고 들어올리는 정책을 학습하는 논문임.

핵심은 한 번에 "잡아라"라고 보상하는 게 아니라, 다음 순서로 reward를 열어주는 구조임.

```text
reach -> contact -> lift -> target
```

### Overall Reward

논문 reward는 다음 weighted sum 구조임.

```text
R =
  w_reach   * r_reach
+ w_contact * r_contact
+ w_lift    * r_lift
+ w_penalty * r_penalty
```

부록 기준 weight는 다음과 같음.

```text
w_reach   = 1
w_contact = 0.5
w_lift    = 10
w_penalty = 0.01
```

### Reach Reward

각 fingertip과 object 사이 거리를 사용함.

```text
r_reach = sum_finger 1 / (eps + d(x_finger, x_obj))
```

의미:

- fingertip이 object 근처로 오게 함.
- grasp 전 exploration을 쉽게 함.
- palm이나 손등이 아니라 finger 중심 접근을 유도함.

공개 코드에서는 Allegro Hand의 4개 fingertip을 사용하고, 거리 d를 0.03-0.8 m로 clamp한 뒤 다음 형태로 계산함.

```text
reward = sum 1 / (0.06 + finger_object_dist) * finger_reward_scale
```

thumb tip에 해당하는 finger는 scale을 더 크게 줌.

### Contact Reward

논문 수식은 강한 조건임.

```text
r_contact =
  IsContact(thumb, object)
  AND
  sum IsContact(other_finger, object) >= 2
```

즉:

```text
thumb contact
+ other finger contact 2개 이상
-> contact reward
```

공개 코드 구현은 더 완화되어 있음.

```text
finger/palm contact group count >= 2
```

이때 같은 finger의 여러 link가 닿아도 group contact는 1개로 계산함.

### Lift Reward

lift reward는 contact로 gate됨.

```text
r_lift = r_contact * Lift(x_obj, x_target)
```

구현상 핵심은 다음과 같음.

```text
if is_contact:
    reward += contact_bonus
    reward += 10 * clamp(current_object_z - initial_object_z, 0, 0.2)
```

의미:

- contact가 없으면 lift reward도 없음.
- 물체를 치거나 밀어서 날리는 reward hacking을 줄임.
- contact 이후 object height가 올라갈 때만 lift reward를 줌.

### Target Reward

target reward는 바로 켜지지 않고, object가 일정 높이 이상 들린 뒤 켜짐.

```text
if lift > 0.02:
    reward += target_distance_reward
```

의미:

- 먼저 grasp/lift를 만들게 함.
- object를 잡기 전에 target으로 밀어버리는 행동을 줄임.

### Penalty

논문은 action L2 penalty를 설명함.

```text
r_penalty = -||a||^2
```

공개 코드에서는 joint velocity penalty와 Cartesian controller error penalty도 사용함.

### Takeaway

DexPoint에서 가져갈 핵심은 gate 구조임.

```text
reach는 항상 열려 있음
contact가 되면 lift reward가 열림
lift가 충분히 되면 target reward가 열림
```

현재 프로젝트에서는 DexPoint식 reward가 목표가 아니라 functional grasp 목표를 안정화하는 gate 참고로만 씀.

## TriFinger Transfer

### Purpose

TriFinger transfer 논문은 grasp 자체보다 object를 목표 6-DoF pose로 맞추는 reward가 핵심임.

reward에서 가져갈 내용은 크게 4개임.

```text
1. keypoint 기반 object-goal reward
2. finger reaching reward
3. fingertip velocity penalty
4. reaching reward curriculum
```

### Overall Reward

전체 reward는 3개 term으로 구성됨.

```text
R(s, a) =
  w_f * r_f * I(t <= N_v)
+ w_v * r_v
+ w_o * r_o
```

각 항은 다음 의미임.

```text
r_f: finger reaching reward
r_v: fingertip velocity penalty
r_o: object displacement reward
```

weight는 다음과 같음.

```text
w_f = -750
w_v = -0.5
w_o = 40
```

`r_f`는 초반 `N_v = 5e7` timesteps까지만 사용하고 이후 꺼짐.

### Object Displacement Reward

핵심 reward는 object keypoint 기반 reward임.

```text
r_o = sum_i K(||k_i_current - k_i_target||)
```

여기서:

```text
k_i_current: 현재 object의 i번째 keypoint
k_i_target : target pose에서 object의 i번째 keypoint
i = 1 ... 8
```

즉, cube/object의 현재 8개 꼭짓점과 목표 8개 꼭짓점 사이 거리를 사용함.

이 방식의 의미:

- position과 orientation을 따로 섞지 않음.
- quaternion error scale 튜닝 문제를 줄임.
- pose error를 3D keypoint 거리 문제로 바꿈.

### Kernel

거리 error를 그대로 쓰지 않고 bounded logistic kernel로 바꿈.

의미:

- 거리 error가 작을수록 reward가 커짐.
- reward가 무한히 커지지 않게 함.

초기 구현에서는 꼭 같은 kernel을 쓸 필요는 없고, 다음 같은 형태로 시작할 수 있음.

```text
r_obj = exp(-k * keypoint_error)
```

또는:

```text
r_obj = 1 / (eps + keypoint_error)
```

### Finger Reaching Reward

TriFinger의 reach reward는 현재 거리가 작으면 계속 보상을 주는 방식이 아님.

현재 fingertip-object 거리와 이전 timestep 거리를 비교함.

```text
r_f = sum_i Delta_i
Delta_i =
  ||f_i,t   - p_t^C||_2
- ||f_i,t-1 - p_t-1^C||_2
```

손가락이 object에 가까워지면:

```text
current distance - previous distance < 0
```

그런데 weight가 음수이므로:

```text
negative weight * negative progress = positive reward
```

의미:

- 손가락이 object 쪽으로 접근할 때만 보상.
- 가까이 있기만 하는 static proximity reward를 줄임.
- 초반 exploration shaping에 적합함.

### Reach Curriculum

이 논문의 중요한 포인트는 reaching reward를 후반에 끄는 것임.

이유:

- 초반에는 손가락이 object에 접근해야 하므로 reach reward가 필요함.
- 후반에는 reorientation을 위해 ungrasping, regrasping, finger gaiting이 필요함.
- reach reward가 계속 켜져 있으면 손가락이 object에 붙어 있으려 해서 오히려 방해됨.

### Fingertip Velocity Penalty

손가락 끝 속도에 penalty를 줌.

```text
r_v = sum_i ||f_dot_i||^2
w_v = -0.5
```

의미:

- 손가락의 빠르고 튀는 움직임을 줄임.
- in-hand manipulation을 안정화함.

### Alternative Reward

논문은 position error reward + orientation error reward도 비교함.

결과적으로 이 방식은 object position 목표는 어느 정도 맞추지만, orientation 학습이 느렸음.

결론:

```text
position + quaternion error를 따로 섞는 reward보다
keypoint distance reward가 6-DoF pose 목표에 더 적합함.
```

### Takeaway

TriFinger는 grasp acquisition보다 object pose control 참고용임.

핵심은 다음 두 가지임.

```text
1. object pose reward는 8개 keypoint distance로 구성
2. reach reward는 초반 progress shaping으로 쓰고 나중에 끔
```

## Dexterous Pre-grasp Manipulation

### Purpose

이 논문은 functional grasp를 하기 전에 필요한 pre-grasp manipulation을 RL로 학습하는 연구임.

Functional grasp는 단순히 물체를 잡는 것이 아니라, 물체를 실제 기능에 맞게 잡는 것임.

예시:

- drill: index finger가 trigger 위치에 있어야 함.
- spray bottle: 분사 버튼을 누를 수 있는 자세여야 함.
- mug: 손잡이를 잡아야 함.

문제는 물체가 처음부터 이런 자세로 놓여 있지 않을 수 있다는 점임.
따라서 로봇은 먼저 물체를 밀고, 돌리고, 세우고, 다시 잡는 pre-grasp manipulation을 해야 함.

### Robot And Setup

사용 로봇:

```text
UR5e 6-DoF arm
+ Schunk SIH hand
```

시뮬레이터와 학습:

```text
Isaac Gym
PPO + RL-Games
```

학습은 image나 point cloud를 직접 쓰지 않고, perception이 끝났다고 가정한 high-level state를 사용함.

입력 정보는 대략 다음과 같음.

```text
hand pose
hand joint positions
object pose
object bounding box
finger-object distance
object category
target grasp representation
```

대상 object category:

```text
drill
spray bottle
mug
```

### Target Grasp Representation

논문은 목표 functional grasp를 표현하는 방법을 두 가지로 비교함.

#### Explicit Target Grasp

목표 grasp를 직접 줌.

```text
end-effector pose relative to object
+ finger joint positions
```

장점:

- 목표가 명확함.
- 학습이 빠르고 안정적임.

단점:

- 각 물체마다 target grasp를 사람이 정의해야 함.
- 새로운 object category로 확장하기 어려움.

#### Constraint-Based Target Grasp

목표 grasp 전체가 아니라 기능 조건만 줌.

논문에서는 대략 다음 조건을 사용함.

```text
index fingertip target position
+ end-effector orientation
```

장점:

- 목표 정의가 간단함.
- 정확한 finger joint target이 필요 없음.
- policy가 여러 grasp configuration을 탐색할 수 있음.

단점:

- 실제로 안정적인 grasp인지 보장하기 어려움.
- fake success를 막기 위해 object lift 조건이 필요함.

### Explicit Grasp Reward

전체 reward는 다음 구조임.

```text
r(t) =
  r_grasp
+ r_man
+ r_MP
+ r_T
```

각 항의 의미:

```text
r_grasp: 목표 grasp pose에 가까워지도록 유도
r_man  : object를 조작해서 grasp 가능한 상태로 만들도록 유도
r_MP   : manipulability가 낮은 자세 penalty
r_T    : 목표 grasp 성공 sparse reward
```

### r_grasp

```text
r_grasp = r_hp + r_hr + lambda * r_hj
```

각 항:

```text
r_hp: hand position이 target grasp position에 가까워지는 보상
r_hr: hand rotation이 target grasp rotation에 가까워지는 보상
r_hj: hand joint가 target grasp joint position에 가까워지는 보상
```

중요한 점은 많은 reward가 현재 거리 자체보다:

```text
이전 step보다 목표에 가까워졌는가
```

를 기준으로 계산된다는 점임.

### r_man

가장 중요한 manipulation reward임.

```text
r_man = r_reach + r_hold + r_orient
```

각 항:

```text
r_reach : 손이 object에 접근하도록 유도
r_hold  : object가 손가락 사이에 들어오도록 유도
r_orient: object를 nominal orientation으로 돌리도록 유도
```

이 reward가 다음 흐름을 만듦.

```text
Reach -> Hold -> Orient
```

#### r_reach

단순히 한 fingertip이 object에 가까워지는 것이 아니라, thumb과 다른 finger 사이의 가상점들이 object 표면에 가까워지는 것을 보상함.

의미:

```text
object가 손가락 사이 공간으로 들어오게 유도
```

#### r_hold

object가 손가락 사이에 잘 들어왔는지를 보상함.

논문은 thumb tip과 middle finger 사이에 여러 점을 만들고, 이 점들이 object 내부 또는 표면 근처에 있을 때 hold reward가 커지게 설계함.

가상점 구성 (원문 확인, arXiv:2307.16752):

- "Each direction thumb-tip→fingertip and thumb-tip→finger-middle has **three equidistant points**."
- 선분 2개(엄지끝→중지끝, 엄지끝→중지 중간마디) × 등간격 내부점 3개 = 총 6점. 끝점(손끝 자체)은 미포함.
- 중앙점의 관용은 의도된 설계임: "This ensures a positive response when an object is positioned
  between the thumb and other fingers **imperfectly**." — 불완전한 중간 상태에도 gradient를 주기 위함.
- 이 관용이 만드는 "가만히 새장만 유지해도 매 스텝 적립" 문제의 마개는 hold가 아니라 r_T임 (아래 참고).

의미:

- contact force나 tactile sensor 없이도 grasp-like behavior를 유도함.
- 단순 fingertip distance나 contact count보다 cage-like 상태를 직접 유도함.

#### r_orient

object를 functional grasp가 쉬운 nominal orientation으로 돌리는 보상임.

논문에서는 nominal orientation을 대략 다음처럼 둠.

```text
object z-axis: upward
object x-axis/tool-tip direction: hand에서 멀어지는 방향
```

### r_MP

manipulability penalty임.

팔이 singularity 근처에 가거나 움직이기 어려운 자세로 가는 것을 막기 위한 penalty임.

### r_T

최종 성공 sparse reward임.

Explicit grasp 성공 조건 (Eq. 18):

```text
hand position error < 1 cm
hand rotation error < 0.15 rad
hand joint error < 0.1 rad
```

크기와 지급 방식 (원문 확인):

- r_T = **5000**. 서열은 r_T ≫ r_orient(500) ≫ r_hold(25) ≫ r_reach(1).
- **일회성이며 성공 시 에피소드 즉시 종료함**: "The episode ends when reaching the target grasp."
- 이 두 가지가 hold 연금의 마개임: 성공하면 에피소드가 끝나므로 "성공 후 계속 벌기"가 불가능하고,
  hold만 farming하는 정책은 5000을 영영 못 받음. 앉아서 버는 총액 상한 < 성공 한 방.

Constraint-based 성공 조건 (Eq. 20):

```text
index fingertip target error < 1 cm
hand rotation error < 0.15 rad
object z > table + 15 cm
```

- lift 조건의 목적 (원문): "By accepting only grasps that enable an object to be lifted off the
  table, we introduce an implicit grasp stability constraint" — 손 모양만 흉내내는 fake success 방지.
- 이 조건을 향한 gradient용 보조 lifting reward가 따로 있음 (Eq. 22).

출처: arXiv:2307.16752 (Pavlichenko & Behnke), https://arxiv.org/html/2307.16752v2

### Constraint-Based Reward

Constraint-based 방식에서는 reward가 다음처럼 바뀜.

```text
r(t) =
  r_grasp
+ r_lift
+ r_man
+ r_MP
+ r_T
```

`r_lift`가 추가되는 이유는 fake success 때문임.
index fingertip target과 end-effector orientation만 맞추고 실제로 물체를 안정적으로 잡지 않는 경우가 가능함.

따라서 성공 조건에 object lift를 넣음.

성공 조건:

```text
index fingertip target position 만족
end-effector orientation 만족
object가 table에서 충분히 들어올려짐
```

### Curriculum

논문은 curriculum이 중요하다고 봄.

Explicit grasp curriculum:

```text
close-start:
  object를 nominal pose로 손 가까이에 둠
  target grasp가 바로 가능함
  r_man을 끄고 grasp reaching부터 학습

Stage 2:
  object를 다양한 pose로 둠
  full reward 사용
  pre-grasp manipulation 학습
```

Constraint-based curriculum:

```text
close-start:
  target constraint만 만족하도록 학습
  object lift는 요구하지 않음

Stage 2:
  target constraint + object lift 요구

Stage 3:
  object를 다양한 roll/yaw pose로 초기화
  full pre-grasp manipulation 학습
```

### Results

Explicit grasp test success:

```text
overall      94.1%
drill        94.3%
spray bottle 92.3%
mug          95.6%
```

Constraint-based grasp test success:

```text
overall      90.1%
drill        90.3%
spray bottle 88.6%
mug          91.1%
```

### Ablation

비교 항목:

```text
full reward
no r_reach
no r_hold
no r_orient
no r_man
```

결과:

- `r_man` 전체를 제거하면 성능이 크게 떨어짐.
- 특히 explicit grasp에서는 `r_hold`가 중요했음.
- target grasp로 가는 reward만으로는 부족함.
- object를 손 안에 넣고 조작하게 만드는 reward가 필요함.

### Limitation

한계:

- real-world 실험이 없음.
- 정확한 6D object pose 추정이 필요함.
- 손에 가려진 object pose estimation이 어려움.
- finger-object surface distance 같은 privileged information이 필요함.
- sim-to-real gap이 남아 있음.

### Takeaway

이 논문은 cube를 처음 잡게 만드는 논문이라기보다, functional grasp가 가능하도록 object를 손 안/손 앞에서 재배치하는 reward를 제안한 논문임.

가장 중요한 reward는 다음임.

```text
r_man = r_reach + r_hold + r_orient
```

특히 `r_hold`는 object가 thumb-finger 사이 공간에 들어왔는지를 보상하는 구조라서 dexterous hand grasp에 중요함.

## SimToolReal

### Purpose

SimToolReal은 single object-centric RL policy를 simulation에서 학습하고, real-world tool manipulation에 zero-shot transfer하는 논문임.

핵심 관점:

```text
tool-use task = object/tool을 goal pose sequence로 움직이는 문제
```

즉, task-specific reward를 매번 설계하기보다, object pose trajectory를 따라가게 하는 일반 policy를 학습함.

### Policy Input

policy는 다음 정보를 사용함.

```text
robot proprioception
current object/tool 6D pose
grasp-region bounding box
goal pose
```

grasp bounding box는 hammer handle 같은 graspable region을 나타냄.

### Overall Reward

reward 구조:

```text
r = r_smooth + r_grasp + I_grasped * r_goal
```

초반에는 grasp/lift가 중요하고, object가 grasped/lifted 상태가 되면 goal pose reaching reward가 메인이 됨.

### Grasp Reward

```text
r_grasp = r_approach + (1 - I_grasped) * r_lift
```

의미:

- grasp 전에는 hand-object 접근과 lift를 보상함.
- object가 lifted threshold를 넘으면 `I_grasped = 1`.
- 이후 lift reward는 꺼지고 goal reward가 주도함.

### r_approach

현재 fingertip-object mean distance가 episode 중 best distance보다 좋아졌을 때 보상함.

```text
r_approach = lambda_approach * max(d_ft_best - d_ft_current, 0)
```

의미:

- 가까이 유지하는 것보다 접근 progress를 보상함.
- static proximity exploitation을 줄임.

### r_lift

```text
r_lift =
  lambda_lift * max(z - z_init, 0)
+ I[z >= z_lifted] * B_lifted
```

`I_grasped`는 object height가 lifted threshold를 넘으면 true가 됨.

### Goal Reward

object가 grasped 된 뒤에는 goal pose reward가 주도함.

```text
r_goal =
  max(d_best - d(current_object_pose, goal_pose), 0)
+ B_succ * I[d(current_object_pose, goal_pose) < eps]
```

핵심:

- current goal에 대해 지금까지의 best distance보다 좋아졌을 때만 dense reward를 줌.
- goal에 도달하면 sparse success bonus를 줌.
- goal에 계속 가까이 머무는 static reward hacking을 줄임.

### Pose Distance

goal pose distance는 object-frame keypoint로 계산함.

```text
d(o_t, g) = max_i ||o_t,i - g_i||
```

논문에서는 reward용 fixed keypoint offset을 사용함.

도구는 길쭉한 경우가 많으므로, x scale을 y/z보다 크게 둬서 pitch/yaw error에 민감하게 함.

### Smoothness Reward

joint velocity L1 penalty를 사용함.

```text
r_smooth =
  -lambda_arm  * ||qdot_arm||_1
  -lambda_hand * ||qdot_hand||_1
```

### Termination

reset 조건:

- object가 table 아래로 떨어짐.
- grasp 이후 object가 다시 table height로 내려감.
- hand가 object에서 너무 멀어짐.
- table force가 너무 큼.
- timeout.
- max consecutive successes 달성.

### Domain Randomization

sim-to-real을 위해 다음 randomization을 사용함.

- object pose noise.
- object pose delay.
- action/observation delay.
- joint velocity observation noise.
- external force/torque perturbation.
- table height randomization.
- grasp bounding box perturbation.

### Takeaway

SimToolReal은 grasp reward 논문이라기보다, grasp 이후 object-centric goal pose trajectory를 따라가는 논문임.

핵심 구조는:

```text
approach/lift
-> I_grasped
-> keypoint-based goal pose progress
-> next goal pose
```

tool-use나 chopstick trajectory로 확장할 때 중요함.

## Cross-Paper Reward Lessons

### 1. Grasp Acquisition

Functional grasp 목표가 우선임.
DexPoint는 contact/lift gate 참고로만 씀.

```text
fingertip reach
contact group
contact-gated lift
lift-gated target
```

### 2. Hold Or Cage

Pre-grasp 논문의 `r_hold`가 중요함.

단순히:

```text
fingertip-object distance
contact count
```

만 보는 것이 아니라:

```text
object가 thumb-finger 사이 공간에 들어왔는가
```

를 보상함.

### 3. Object Pose Control

TriFinger와 SimToolReal은 keypoint distance를 사용함.

장점:

- position과 orientation을 하나의 3D 거리 문제로 통합함.
- quaternion scale tuning 문제를 줄임.
- cube/tool pose tracking에 적합함.

### 4. Progress Reward

TriFinger, Pre-grasp, SimToolReal 모두 progress 형태의 reward를 중요하게 사용함.

```text
previous보다 좋아졌는가
episode best보다 좋아졌는가
```

이 방식은 static proximity reward hacking을 줄이는 데 도움이 됨.

### 5. Gating

여러 논문에서 reward를 단계적으로 열어줌.

```text
contact 이후 lift
lift 이후 target
grasped 이후 goal pose
```

이 구조는 dense reward를 주면서도 reward hacking을 줄이는 방법임.

### 6. Curriculum

TriFinger와 Pre-grasp에서 curriculum이 중요함.

```text
초반:
  접근, 직접 grasp, 쉬운 pose

후반:
  regrasp, finger gaiting, 다양한 pose, full manipulation
```

특히 reach reward는 후반에는 줄이거나 끄는 것이 필요할 수 있음.

## Implementation Order Suggested By Papers

논문들만 기준으로 보면 자연스러운 구현 순서는 다음과 같음.

```text
1. Pre-grasp-style hold/cage reward
   object inside thumb-finger region

2. Contact condition and contact-gated lift
   DexPoint-style gate as helper reference

3. TriFinger-style object keypoint reward
   object/cube 6-DoF pose tracking

4. SimToolReal-style trajectory reward
   object-centric goal pose sequence for tool use
```
