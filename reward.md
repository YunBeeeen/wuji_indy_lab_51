# Reward 정리

이 문서는 현재 코드 기준 `Indy-Wuji-Cube-Grasp` reward 구조를 정리한 것이다.

기준 파일:

- reward 설정: `nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py`
- reward 함수: `nrmk_isaaclab_wuji/isaac_neuromeka/mdp/rewards.py`
- cube/table/goal 높이 override: `nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/grasp/cube_grasp_env_cfg.py`

## 전체 계산 방식

각 reward term은 매 environment step마다 다음 방식으로 합산된다.

```python
total_reward += raw_reward * weight * dt
```

현재 cube grasp 환경은 대략 `dt = 1 / 30` 기준으로 동작한다.

TensorBoard에서:

- `Episode_Reward/<term>`: weight가 적용된 reward
- `Episode_Reward_Raw/<term>`: weight가 적용되지 않은 raw reward
- 실제 PPO return에는 `raw_reward * weight * dt`가 들어간다.

## Cage 가상점

현재 grasp reward의 중심은 contact sensor가 아니라 cage 가상점이다.

`CAGE_BODIES`:

- `finger1_tip_link`: thumb tip, 모든 선분의 기준점
- `finger2_tip_link`: index tip
- `finger2_link3`: index middle link
- `finger3_tip_link`: middle tip
- `finger3_link3`: middle middle link

엄지끝에서 나머지 4개 body로 선분을 만들고, 각 선분 위에 `point_fractions = (0.1, 0.5, 0.9)` 위치의 점을 찍는다.

따라서 총 가상점 수는:

```text
4 opposing bodies * 3 points = 12 cage points
```

중요한 점은 `finger_cage_reach`, `finger_cage_hold`, `cube_lift`, `cube_transport`, `success`가 모두 같은 cage 구조를 기준으로 한다는 것이다.

`drop_penalty`는 cage가 아니라 `cube_dropped` termination을 본다.

## Active Reward Terms

표 순서는 gate를 만드는 term이 gate를 타는 term보다 먼저 오도록 정리한다.

| Term | Weight | 구조 | 보상 받는 경우 | 깎이는 경우 / Gate |
| --- | ---: | --- | --- | --- |
| `palm_facing` | `4.0` | 손 파지 방향의 best-so-far 양수 progress | 에피소드 최고 facing을 새로 갱신함 | 악화/유지는 0. 이전 최고점 재방문도 0 |
| `finger_cage_reach` | `8.0` | cage 가상점들의 cube surface SDF progress, 양수 progress는 `palm_facing` gate 적용 | cage 가상점들이 cube 표면/내부 쪽으로 가까워지고 손 방향도 cube를 향함 | 멀어지면 음수. 양수 reward만 `palm_facing` gate를 탐 |
| `finger_cage_hold` | `15.0` | cage 가상점이 cube 표면 근처 또는 내부에 들어간 정도 | 손가락을 오므려 cube가 thumb-index/middle 사이에 들어감 | 음수는 없음. 멀면 0 |
| `cube_lift` | `50.0` | `cage_gate * cube_clearance / 0.08` | cage가 된 상태로 cube 최하 꼭짓점이 받침면 위로 뜸 | cage가 약하거나 lift가 없으면 거의 0 |
| `cube_transport` | `4000.0` | 역수 포텐셜의 best-so-far 양수 progress, cage gate 적용 | cube를 잡은 상태로 에피소드 최단 goal 거리를 새로 갱신함 | 후퇴/왕복은 0원. 잡지 않으면 gate 때문에 거의 0 |
| `transport_success` | `30000.0` | `success` termination 발생 여부 | goal 반경 안에서 cage 상태를 15 step 유지 | 성공 step에 한 번 지급되고 즉시 종료 |
| `drop_penalty` | `0.0` | `cube_dropped` termination 발생 여부 | 현재 비활성 | curriculum에서 음수 weight로 켤 예정 |
| `arm_manipulability` | `1.0` | arm singularity penalty | 좋은 arm posture면 0 | 특이점에 가까우면 음수 |
| `hand_floor` | `1.0` | 손/손가락 table penetration penalty | 손이 받침면 위에 있으면 0 | 손 body가 받침면 근처 아래로 내려가면 음수 |
| `action_rate` | `-0.005` | action 변화량 L2 penalty | action 변화가 작으면 덜 깎임 | action이 흔들리면 음수 증가 |

## Reward별 의미

### `palm_facing`

목적은 손의 파지 개구부 방향이 cube를 향하도록 유도하는 것이다.

현재는 절대 orientation reward가 아니라 best-so-far progress reward다.

```text
facing = clamp(dot(palm_opening_direction_w, unit(cube_pos_w - palm_pos_w)), 0, 1)
raw = max(facing - best_facing, 0)
best_facing = max(best_facing, facing)
```

따라서 에피소드 최고 정렬값을 갱신할 때만 양수이고, 나빠지거나 유지하거나 이전 최고점으로 돌아오면 0이다.

이 구조는 "방향만 맞추고 가만히 있기" farming을 줄이기 위한 것이다.

### `finger_cage_reach`

목적은 손가락 끝 하나를 cube center로 보내는 것이 아니라, 파지 공간 전체를 cube 쪽으로 가져가는 것이다.

계산은 cage 가상점들의 cube box surface까지 signed distance 평균을 보고, 이전 step보다 가까워졌는지를 progress로 계산한다.

```text
progress = clamp((previous_distance - current_distance) / distance_max, -1, 1)
facing_gate = palm_facing_object(...)
raw = progress * facing_gate    if progress > 0
raw = progress                  if progress <= 0
```

가까워지면 양수, 멀어지면 음수다.

단, 양수 reward만 `palm_facing` gate를 탄다. 손 방향이 cube를 향하지 않은 상태에서 cage만 가까워지는 편법을 줄이기 위한 구조다.

반대로 멀어지는 음수 progress는 gate 없이 그대로 들어간다. 손 방향이 나쁘다는 이유로 후퇴가 공짜가 되면 안 되기 때문이다.

### `finger_cage_hold`

목적은 손가락을 오므려 cube가 thumb-index/middle 사이 cage 안으로 들어오게 하는 것이다.

각 cage 가상점을 작은 sphere처럼 보고, 그 점이 cube 표면 근처 또는 내부로 얼마나 들어갔는지를 본다.

```text
penetration = sphere_radius - sdf
raw = clamp(penetration / (sphere_radius + depth_max), 0, 1)
```

이 reward는 절대형 positive reward다. 즉, hold 상태를 유지하면 계속 보상을 받는다.

다만 이 term만 있으면 "잡은 것처럼 보이는 자세"에 머물 수 있으므로, 실제 lift/transport reward와 함께 봐야 한다.

### `cube_lift`

목적은 cube를 table surface에서 실제로 띄우는 것이다.

현재 lift는 cube center 높이가 아니라 cube 8개 꼭짓점 중 최저점 기준 clearance를 사용한다. 그래서 cube가 살짝 기울어져 center만 올라가는 편법을 줄인다.

```text
lift = clamp(cube_clearance / 0.08, 0, 1)
raw = cage_gate * lift
```

즉, 잡지 않고 튕겨서 cube가 뜨는 것은 큰 보상이 되지 않는다. cage가 약하면 lift reward도 약하다.

`surface_z`는 월드 바닥이 아니라 현재 support/table 높이 기준이다.

### `cube_transport`

목적은 잡은 cube를 `cube_goal`로 옮기는 것이다.

현재는 거리 자체의 선형 차분이 아니라 역수 포텐셜을 사용한다. 에피소드 안에서 가장 가까웠던 goal 거리보다 더 가까워져 포텐셜 최고값을 갱신한 양만 보상한다.

즉 best-so-far 방식이다.

```text
phi = potential_eps / (potential_eps + current_distance)
progress = clamp(phi - best_phi, min=0)
cage_gate = object_in_finger_cage(...)
raw = progress * cage_gate
```

이 항은 양수 전용이다.

따라서:

- goal 최단거리를 새로 갱신하면 보상
- 이미 갔던 거리로 다시 돌아오면 0
- goal에서 멀어져도 이 항에서는 0
- 잡지 않고 밀거나 던져서 가까워지면 cage gate 때문에 거의 0

후퇴나 낙하의 비용은 `cube_transport` 안에 섞지 않고, 별도 `drop_penalty`가 담당한다.

### `transport_success`

이 term은 직접 물리량을 계산하는 reward가 아니라, `success` termination이 켜졌는지를 reward로 바꾼 것이다.

현재 success 조건은 다음과 같다.

- cube가 `cube_goal` 반경 `0.05 m` 안에 있음
- cage gate가 `0.3`보다 큼
- 위 상태를 `15 step` 유지함

성공하면 `raw = 1`이고, weight가 `30000`이라 큰 terminal reward가 들어간다.

대략 `dt = 1 / 30`이면 성공 step의 PPO reward 기여는:

```text
30000 * 1/30 = 1000
```

이 성공 신호는 두 곳에서 동시에 사용된다.

```text
ObjectAtGoalHeld가 success=True
  -> transport_success가 raw=1을 읽어 +1000을 한 번 지급
  -> success termination이 같은 step에 episode를 종료
```

즉 0.5초마다 반복 지급되는 reward가 아니다. 15번째 연속 성공 step에서 한 번 지급하고 바로
리셋하여, goal 밖으로 나갔다가 다시 들어와 terminal reward를 반복 적립하는 것을 막는다.

성공 후 장기 유지 성능을 play에서 관찰할 때는 termination을 삭제하지 않고 다음처럼 판정
시간만 timeout보다 크게 덮어쓴다.

```bash
env.terminations.success.params.hold_steps=1000000
```

이 override는 정책이나 grasp reward를 바꾸지 않는다. 성공 reward와 성공 종료만 8초 timeout
뒤로 미루고, 기존 `time_out`과 `cube_dropped`는 그대로 유지한다. `success=null`만 사용하면
`transport_success`의 `term_keys="success"` 참조가 끊겨 reward manager 생성이 실패한다.

### `drop_penalty`

이 term은 `cube_dropped` termination이 켜졌는지를 penalty로 바꾼 것이다.

현재 `cube_dropped`는 cube root height가 minimum height 아래로 내려가면 켜진다. `cube_grasp_env_cfg.py`에서 이 minimum height는 table 상판 기준으로 override된다.

현재 weight는 `0`이라 비활성이다. 이후 curriculum에서 weight를 `-3000`으로 켜면 다음처럼
동작하도록 준비된 항이다.

```text
raw = 1    if cube_dropped
raw = 0    otherwise
weighted = raw * (-3000) * dt
낙하 순간 PPO reward = -3000 * 1/30 = -100
```

이 항은 `cube_transport`를 양수 전용으로 바꾸면서 생긴 별도 벌금이다.

즉:

- 운반 progress는 `cube_transport`에서 양수만 지급
- 놓치거나 떨어뜨린 비용은 `drop_penalty`에서 정액으로 감점

이렇게 보상과 벌금을 분리한 구조다.

### `arm_manipulability`

목적은 arm이 특이점 근처로 접히는 것을 막는 것이다.

manipulability가 충분하면 0이고, 낮아질수록 음수 penalty가 들어간다.

이 term은 task progress reward가 아니라 posture safety penalty다.

### `hand_floor`

목적은 손 또는 손가락이 table/support 아래로 파고드는 것을 막는 것이다.

손 body들이 `surface_z + clearance` 아래로 내려가면 음수 penalty가 들어간다.

이 term도 reward를 주는 항이 아니라 침투를 막는 penalty다.

### `action_rate`

목적은 action이 심하게 흔들리는 것을 줄이는 것이다.

IsaacLab 기본 action rate penalty처럼 action 변화량 제곱합을 계산한다.

raw는 양수지만 weight가 `-0.005`라서 실제 reward는 음수다.

## Reward 간 관계

현재 reward 구조는 대략 다음 순서로 설계되어 있다.

```text
palm_facing
  -> finger_cage_reach
  -> finger_cage_hold
  -> cube_lift
  -> cube_transport
  -> transport_success
cube_dropped
  -> drop_penalty
```

관계는 다음과 같다.

1. `palm_facing`은 손 방향을 맞추는 progress reward다.
2. `palm_facing`은 `finger_cage_reach`의 양수 보상을 gate한다.
3. `finger_cage_reach`는 손의 파지 공간을 cube 쪽으로 가져온다.
4. `finger_cage_hold`는 cube가 실제로 손가락 사이 cage 안에 들어왔는지 본다.
5. `finger_cage_hold`의 cage 값은 `cube_lift`의 gate로 쓰인다.
6. 같은 cage 값은 `cube_transport`의 best-so-far 양수 progress gate로도 쓰인다.
7. 같은 cage 값은 `success` termination 조건에도 쓰인다.
8. `transport_success`는 `success` termination이 켜졌을 때만 큰 보상을 준다.
9. `drop_penalty`는 `cube_dropped` termination이 켜졌을 때만 큰 음수 보상을 준다.

즉, 현재 구조에서 가장 중요한 gate는 `finger_cage_hold` 계열 cage gate다.

잡지 않으면:

- lift 보상이 거의 안 나옴
- transport 양수 progress가 약해짐
- success가 켜지지 않음

떨어뜨리면:

- `cube_dropped` termination이 켜짐
- `drop_penalty`가 한 번 들어감
- episode가 끝남

## 현재 구조에서 봐야 할 핵심 지표

TensorBoard에서는 평균 metric보다 final/min/max를 같이 봐야 한다.

중요한 것은:

- `Episode_Reward_Raw/finger_cage_hold`
- `Episode_Reward_Raw/cube_lift`
- `Episode_Reward_Raw/cube_transport`
- `Episode_Reward_Raw/transport_success`
- `Episode_Reward/drop_penalty`
- `Episode_Reward_Raw/drop_penalty`
- `Metrics/cube_final/cage_inside_frac`
- `Metrics/cube_final/cube_clearance`
- `Metrics/cube_final/cube_lift`
- `Metrics/cube_final/thumb_index_opposition`
- `Metrics/cube_final/thumb_middle_opposition`
- `Metrics/cube_final/action_delta`
- `Metrics/cube_final/action_track_err`

특히 `cube_lift`나 `cube_transport` raw가 계속 0이면 weight를 올려도 의미가 없다.

먼저 raw reward가 실제로 발생하는지 확인해야 한다.

반대로 `Episode_Reward_Raw/drop_penalty`가 올라가면 낙하 종료가 발생한다는 뜻이다. 이 raw 값은 event indicator라 양수로 보이고, 실제 감점은 `Episode_Reward/drop_penalty`에서 음수로 봐야 한다.

## 주의할 점

`finger_cage_hold`는 손가락 사이에 cube가 들어온 것만 본다. 이것만으로 좋은 grasp posture를 보장하지 않는다.

좋은 파지는 결국:

- cage가 유지되고
- cube가 table에서 뜨고
- goal 쪽으로 안정적으로 이동하고
- success 조건을 만족하는지

로 판단해야 한다.

따라서 현재 구조에서는 `hold`만 높고 `lift/transport/success`가 낮으면 아직 "잡은 것처럼 보이는 상태"일 가능성이 크다.
