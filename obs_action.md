# Observation / Action 정리

이 문서는 현재 코드 기준 `Indy-Wuji-Cube-Grasp`의 observation과 action 구조를 정리한 것이다.

기준 파일:

- obs/action 설정: `nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py`
- Indy/Wuji override: `nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/grasp/indy_wuji/env_cfg.py`
- observation 함수: `nrmk_isaaclab_wuji/isaac_neuromeka/mdp/observations.py`
- action 함수: `nrmk_isaaclab_wuji/isaac_neuromeka/mdp/actions/joint_actions.py`
- RSL-RL action clip: `nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/grasp/indy_wuji/learning/rsl_rl_cfg.py`

## 전체 구조

현재 active task는 `Indy-Wuji-Cube-Grasp`이다.

정책 observation shape:

```text
57D
```

정책 action shape:

```text
18D
```

ActionManager에는 active action term이 하나다.

```text
arm_action: 18D
```

이름은 `arm_action`이지만 실제로는 arm 6축 + thumb/index/middle 12축을 같이 제어한다.

## Observation 구성

현재 policy observation은 하나의 group인 `policy`에 모두 concatenate된다.

| Obs term | Dim | 함수 | 의미 |
| --- | ---: | --- | --- |
| `joint_pos` | 18 | `mdp.joint_pos` | 제어 대상 joint들의 현재 관절 위치 |
| `cube_pos` | 3 | `mdp.object_position_relative` | `palm_link` 기준 cube 상대 위치 |
| `cube_in_fingertips` | 15 | `mdp.object_position_relative_to_bodies` | 5개 fingertip 기준 cube 상대 위치 |
| `cube_to_goal` | 3 | `mdp.object_position_error_to_command` | cube에서 goal까지의 상대 벡터 |
| `action_history` | 18 | `mdp.action_history` | 직전 policy action |

합계:

```text
18 + 3 + 15 + 3 + 18 = 57
```

## `joint_pos` 18D

`joint_pos`는 robot 전체 joint가 아니라 현재 controlled joints만 본다.

현재 controlled joint regex:

```python
["joint[0-5]", "finger[1-3]_joint[1-4]"]
```

즉:

- Indy arm 6축
- thumb/index/middle finger 12축

만 policy joint observation에 들어간다.

약지 `finger4`와 새끼 `finger5` 관절 위치는 `joint_pos` obs에는 직접 들어가지 않는다.

## `cube_pos` 3D

`cube_pos`는 cube world 좌표 자체가 아니다.

계산:

```python
cube_pos = cube.root_pos_w - palm_link.pos_w
```

즉 policy가 보는 것은:

```text
palm_link 기준으로 cube가 어디에 있는가
```

이다.

이 구조의 의미:

- arm base 기준 absolute cube pose를 외우는 것보다 낫다.
- 손과 cube 사이의 상대 관계를 직접 보게 한다.
- grasp에서는 world 좌표보다 hand-relative coordinate가 더 중요하다.

## `cube_in_fingertips` 15D

5개 fingertip 각각에 대해 cube 상대 위치를 본다.

body list:

```python
[
    "finger1_tip_link",
    "finger2_tip_link",
    "finger3_tip_link",
    "finger4_tip_link",
    "finger5_tip_link",
]
```

각 body마다:

```python
cube.root_pos_w - fingertip.pos_w
```

을 계산한다.

따라서:

```text
5 fingertips * xyz 3D = 15D
```

여기서 중요한 점:

- policy action은 thumb/index/middle 3개 finger만 직접 제어한다.
- 하지만 obs에는 ring/little fingertip 상대 위치도 들어간다.
- ring/little은 직접 action에는 없지만 mimic으로 움직이므로, 그 결과는 fingertip 위치 obs로 policy가 볼 수 있다.

## `cube_to_goal` 3D

`cube_to_goal`은 현재 transport task용 goal observation이다.

계산:

```python
goal_w = env_origin + cube_goal_command
cube_to_goal = goal_w - cube.root_pos_w
```

즉 policy가 보는 것은:

```text
cube에서 goal까지 어느 방향으로 얼마나 남았는가
```

이다.

현재 goal은 `UniformCubeGoalCommandCfg`로 command manager가 들고 있다. 다만 현재 curriculum stage에서는 `cube_grasp_env_cfg.py`에서 고정점으로 override한다.

현재 고정 goal:

```text
x = CUBE_POS[0]
y = CUBE_POS[1]
z = BASE_Z + 0.20
```

즉 cube spawn 위치 위쪽 약 20cm 지점이다.

## `action_history` 18D

`action_history`는 직전 action만 반환한다.

```python
return env.action_manager.prev_action
```

현재 action history는:

- 현재 action 아님
- 두 step history 아님
- `prev_action` 하나만

이다.

이 obs가 들어가는 이유:

- position command action은 절대 목표라서, 현재 명령이 얼마나 바뀌었는지 policy가 직접 알기 어렵다.
- 직전 action을 보면 급격한 변화나 유지 상태를 policy가 구분할 수 있다.
- `action_rate` penalty와도 맞물린다.

## Observation에 없는 것

현재 policy observation에는 다음이 직접 들어가지 않는다.

- joint velocity
- contact force
- cube orientation
- cube angular velocity
- fingertip velocity
- absolute cube world position
- raw contact sensor 값
- ring/little joint position
- command goal absolute coordinate

현재 구조는 최소한의 oracle state 기반이다.

즉, policy는 sim에서 직접 아는 cube/fingertip 위치를 보지만, contact force나 point cloud는 아직 쓰지 않는다.

## Action 구성

현재 action은 `MimicJointActionCfg`를 사용한다.

설정:

```python
self.actions.arm_action = MimicJointActionCfg(
    asset_name="robot",
    joint_names=["joint[0-5]", "finger[1-3]_joint[1-4]"],
    scale=1.0,
    use_default_offset=True,
    mimic={...},
)
```

핵심:

```text
target = default_joint_pos + scale * clipped_action
```

현재 `scale = 1.0`이고 RSL-RL config에서:

```python
clip_actions = 1.0
```

이므로 policy raw action은 최종적으로 대략 `[-1, 1]`로 clip된다.

따라서 각 action dim은 기본 관절각 기준:

```text
default_joint_pos ± 1.0 rad
```

범위의 절대 위치 target을 만든다.

중요:

```text
action은 증분 명령이 아니다.
```

즉, 매 step action이 누적되는 구조가 아니다.

## Action order 18D

현재 action order는 실측 기준 다음과 같다.

| Index | Joint |
| ---: | --- |
| 0 | `joint0` |
| 1 | `joint1` |
| 2 | `joint2` |
| 3 | `joint3` |
| 4 | `joint4` |
| 5 | `joint5` |
| 6 | `finger1_joint1` |
| 7 | `finger2_joint1` |
| 8 | `finger3_joint1` |
| 9 | `finger1_joint2` |
| 10 | `finger2_joint2` |
| 11 | `finger3_joint2` |
| 12 | `finger1_joint3` |
| 13 | `finger2_joint3` |
| 14 | `finger3_joint3` |
| 15 | `finger1_joint4` |
| 16 | `finger2_joint4` |
| 17 | `finger3_joint4` |

finger alias:

- `finger1` = thumb
- `finger2` = index
- `finger3` = middle
- `finger4` = ring
- `finger5` = little

## Arm action 6D

Action index `0~5`는 Indy arm 관절이다.

```text
joint0 ~ joint5
```

이 6개도 policy가 직접 제어한다.

현재는 IK action이 아니다. 논문처럼 EE IK + hand joint command 구조가 아니라, arm도 joint position action이다.

즉 현재 구조:

```text
policy action -> arm joint target + hand joint target
```

논문식 구조:

```text
policy action -> EE delta/pose command -> IK -> arm joint target
             -> hand joint target
```

현재 task는 아직 논문식 IK action 구조가 아니라 joint-space baseline이다.

## Finger action 12D

Action index `6~17`은 thumb/index/middle finger 12축이다.

직접 제어하는 finger:

- thumb: `finger1_joint1~4`
- index: `finger2_joint1~4`
- middle: `finger3_joint1~4`

직접 제어하지 않는 finger:

- ring: `finger4_joint1~4`
- little: `finger5_joint1~4`

## Ring/Little Mimic 구조

현재 ring/little은 action dim에 없지만, 중지 목표를 따라간다.

Mimic mapping:

```python
"finger4_joint1": "finger3_joint1"
"finger4_joint2": "finger3_joint2"
"finger4_joint3": "finger3_joint3"
"finger4_joint4": "finger3_joint4"
"finger5_joint1": "finger3_joint1"
"finger5_joint2": "finger3_joint2"
"finger5_joint3": "finger3_joint3"
"finger5_joint4": "finger3_joint4"
```

즉:

```text
ring/little target = middle target + default offset
```

이 구조의 의미:

- policy action dim은 18D로 유지한다.
- 하지만 실제 손은 5개 손가락이 같이 움직인다.
- ring/little이 받침 역할을 하도록 하되, action 공간을 26D로 늘리지는 않는다.

주의:

`finger4/5`는 policy action에는 없지만 실제 articulation target은 매 step 들어간다.

## Action과 Reward의 관계

현재 reward와 action의 관계는 다음처럼 연결된다.

### Arm action

arm action은 손 전체를 cube와 goal 근처로 움직인다.

주로 영향을 주는 reward:

- `finger_cage_reach`
- `palm_facing`
- `cube_transport`
- `hand_floor`
- `arm_manipulability`
- `action_rate`

arm이 cube를 향해 손을 가져가면 `finger_cage_reach`가 좋아진다.

arm이 cube를 들고 goal 방향으로 움직이면 `cube_transport`가 좋아진다.

하지만 arm이 접혀 특이점에 가까워지면 `arm_manipulability` penalty를 받는다.

손이 table 아래로 내려가면 `hand_floor` penalty를 받는다.

### Finger action

finger action은 cube를 감싸고 유지하는 역할이다.

주로 영향을 주는 reward:

- `finger_cage_hold`
- `cube_lift`
- `cube_transport`
- `transport_success`
- `action_rate`

손가락을 오므려 cage 가상점이 cube 안/표면 근처로 들어가면 `finger_cage_hold`가 오른다.

이 hold가 좋아야 `cube_lift`, `cube_transport`, `success` gate가 열린다.

즉 finger action은 단순히 hold reward만 먹는 것이 아니라, 상위 reward들이 켜지는 조건을 만든다.

## Action 해석 시 주의점

### 1. raw action과 applied action은 다르다

policy network가 큰 값을 내도 `clip_actions = 1.0` 때문에 applied action은 `[-1, 1]`로 잘린다.

따라서 debug에서:

```text
raw = 4.0
applied = 1.0
```

처럼 보이면 정책이 이미 saturation된 것이다.

### 2. applied action은 관절 목표 자체가 아니다

실제 target은:

```text
target = default_joint_pos + applied_action * scale
```

현재 scale이 1.0이므로:

```text
target = default_joint_pos + applied_action
```

이다.

### 3. actual joint가 target을 못 따라갈 수 있다

특히 손가락은 effort limit이 낮고 cube/contact가 있으므로 target과 actual이 크게 벌어질 수 있다.

이때는 policy가 잘못했다기보다:

- actuator torque 부족
- contact가 관절을 막음
- target이 물리적으로 불가능함
- timestep/decimation 문제

일 수 있다.

그래서 play debug에서 봐야 하는 값은:

- raw action
- applied action
- target joint position
- actual joint position
- tracking error
- joint velocity
- torque saturation

이다.

### 4. action_history는 target history가 아니다

`action_history`는 직전 policy action이다.

즉, 실제 joint target이나 actual joint position history가 아니다.

정책은:

- 현재 joint_pos
- 직전 action
- cube 상대 위치
- goal error

를 조합해서 다음 action을 결정한다.

## 현재 구조의 한계

현재 구조는 joint-space baseline이다.

한계:

- arm이 EE pose command/IK로 움직이지 않는다.
- cube orientation을 obs로 보지 않는다.
- contact force를 obs로 보지 않는다.
- finger velocity를 obs로 보지 않는다.
- ring/little joint state는 직접 obs에 없다.

장점:

- 구조가 단순하다.
- action/obs dim이 명확하다.
- cage/lift/transport reward가 제대로 작동하는지 먼저 확인하기 좋다.
- chopsticks task 전 cube grasp proxy로 디버깅하기 쉽다.

## 핵심 요약

현재 policy가 보는 것:

```text
제어 joint 현재 위치
손 기준 cube 위치
각 fingertip 기준 cube 위치
cube 기준 goal 방향
직전 action
```

현재 policy가 내는 것:

```text
arm 6축 + thumb/index/middle 12축의 절대 joint position target
```

실제로 함께 움직이는 것:

```text
arm 6축
thumb/index/middle 12축
ring/little 8축은 middle finger target을 mimic
```

현재 action의 가장 중요한 해석:

```text
action은 누적 delta가 아니라 default 기준 절대 목표다.
```

따라서 action scale을 바꾸면 "한 step 움직임 크기"만 바뀌는 것이 아니라, policy가 도달 가능한 joint target 범위 자체가 바뀐다.
