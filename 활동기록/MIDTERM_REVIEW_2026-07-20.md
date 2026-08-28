# Indy7 + Wuji 젓가락 조작 연구 중간 리뷰

- 정리 범위: 2026-07-08 ~ 2026-07-31.
- 기준 자료: 날짜별 활동 기록, 당시 코드, run별 `params` 사용.
- 중간 결론: 팔 도달부터 젓가락 pregrasp 획득까지 과제를 단계적으로 분해.

## 1. 연구 목표

Indy7과 Wuji Hand를 이용한 젓가락 파지·개폐·물체 조작 정책 구축.

```text
최종 목표
  |
  +-- 팔 도달
  +-- 물체 접근·파지
  +-- 들어 올리기
  +-- 목표 위치 운반
  +-- 목표 자세 정렬
  +-- 젓가락 기능 파지
  +-- OPEN/CLOSE
  `-- 젓가락을 이용한 물체 조작
```

한 번에 전체 동작을 학습하지 않고 `ACQUIRE → TOOL_READY → USE`로 분해.

```text
+------------------+     +------------------+     +------------------+
| ACQUIRE          |     | TOOL_READY       |     | USE              |
| 젓가락 획득      +---->| 6-contact 파지   +---->| 개폐·운반·조작   |
| hand_setting     |     | hand_grasp       |     | 후속 정책        |
+------------------+     +------------------+     +------------------+
```

## 2. 연구 진행 흐름

```text
Indy Reach
    |
    v
Cube Reach -> SDF Cage -> Lift -> Transport -> Success
                                          |
                                          v
Random Box -> Orientation -> Stick Proxy
                                 |
                                 v
                    Semantic Contact Topology
                                 |
                                 v
                      hand_grasp -> hand_setting
```

| 단계 | 기간 | 핵심 질문 | 확보 내용 |
|---|---|---|---|
| 팔 기준선 | 07-08~09 | 프레임과 RL 배선 정상 여부 | 6축 reach와 15D 최소 관측 확보. |
| 큐브 파지 | 07-10~14 | 손가락 접근을 실제 파지로 전환 가능 여부 | SDF cage, action 진단, contact probe 확보. |
| 운반 | 07-15~17 | 안정 lift 후 목표 위치 이동 가능 여부 | terminal success와 best-so-far transport 구축. |
| 자세 정렬 | 07-18~20 | 위치와 orientation 동시 만족 가능 여부 | 대칭 자세 오차와 단계형 reward 구축. |
| 젓가락 프록시 | 07-20~26 | 긴 물체를 기능적으로 잡아 운반 가능 여부 | balanced tripod와 semantic region 구축. |
| 파지 획득 | 07-27~31 | 열린 손에서 6-contact pregrasp 생성 가능 여부 | `pose_005`, `hand_grasp`, `hand_setting` 구축. |

## 3. 시스템 구성

```text
+-------------------+      +-------------------+      +-------------------+
| Isaac Lab Scene   |      | ManagerBased RL   |      | RSL-RL PPO        |
| Indy7 + Wuji      +----->| obs/action/reward +----->| 병렬 환경 학습    |
| table + object    |      | termination       |      | checkpoint 저장   |
+---------+---------+      +---------+---------+      +---------+---------+
          |                          |                          |
          v                          v                          v
+---------+---------+      +---------+---------+      +---------+---------+
| Contact·SDF Probe |      | TensorBoard       |      | play 진단         |
| 물리 가능성 확인  |      | final/min/max     |      | action·torque     |
+-------------------+      +-------------------+      +-------------------+
```

- 물리 가능성 확인 후 reward 조정.
- Raw reward, contact, clearance, action, torque 동시 확인.
- Run 재현은 코드 기억이 아닌 `params/env.yaml` 기준 사용.

## 4. 핵심 설계 변화

### 4.1 물체 중심에서 표면 SDF로 변경

물체 중심은 손끝이 도달할 수 없는 목표. 접촉 후 물체가 이동하면 접근 보상이 감소하는 문제 발생.

```text
thumb tip o------o index/middle tip
           x x x             가상 cage point
         +-------+
         | object|            box surface SDF 계산
         +-------+
```

- `SDF > 0`: 물체 바깥.
- `SDF = 0`: 물체 표면.
- `SDF < 0`: 물체 내부.
- 손가락 사이에 물체가 들어왔는지 직접 평가.

### 4.2 유도·유지·종료 보상 분리

```text
접근 progress
    -> cage hold
        -> clearance lift
            -> goal transport
                -> orientation
                    -> terminal success
```

| 역할 | 형태 | 목적 |
|---|---|---|
| 유도 | signed 또는 best-so-far progress | 왕복 farming 억제. |
| 유지 | bounded absolute reward | 실제 파지와 lift 유지. |
| 종료 | 연속 조건 만족 후 1회 지급 | 성공 상태 연금 방지. |

### 4.3 중심 높이에서 최저 꼭짓점 clearance로 변경

중심 높이만 사용하면 물체를 기울여 lift 보상 획득 가능.

```text
clearance = min(rotated_corner_z) - table_surface_z
lift      = cage_gate * clip(clearance / target_height, 0, 1)
```

한 모서리라도 테이블에 닿으면 lift 미인정.

### 4.4 Action과 물리 문제 분리

```text
tracking error 작음 + action 변화 큼 = 정책 명령 문제.
tracking error 큼                    = actuator·접촉·물리 추종 문제.
```

- `clip_actions=1.0` 적용.
- Raw/applied action, target/actual, torque 지표 추가.
- 제어하지 않는 약지·새끼는 mimic 또는 별도 task로 분리.

### 4.5 위치와 자세 정렬 분리

Position success에 orientation 조건만 추가하면 자세 신호 부재.
Keypoint 융합 reward는 random pose에서 유리하나 파지 도달성에 영향.

```text
goal 접근
   |
   +-- position error 감소
   `-- grasp gate 유지 + orientation error 감소
```

- Box: 정사각 단면 8-대칭 사용.
- Chopstick: tip/tail을 구분하는 4-대칭 사용.
- 평균 orientation error보다 `15° 이내 비율`과 success 우선 확인.

### 4.6 접촉 개수에서 semantic topology로 변경

단순 접촉 개수는 잘못된 손가락·스틱 조합도 성공으로 인정.

```text
Stick1: thumb distal + index + middle
Stick2: palm + thumb middle + ring
보조:   ring tip + little tip 안정 지지
```

- 정확한 link–stick pair 기준 사용.
- 독립 접촉 reward의 contact 교환 문제 확인.
- Minimum·coupled gate로 일부 접촉 파밍 차단.

## 5. 날짜별 핵심 전환

| 날짜 | 전환 | 판단 |
|---|---|---|
| 07-08 | 현재 Isaac Sim·Isaac Lab 환경 통일. | Reach 기준선부터 재구성. |
| 07-11 | 중심 거리 폐기, 가상점·SDF 도입. | 파지 공간을 직접 평가. |
| 07-12 | 최저 꼭짓점 clearance 도입. | 기울이기 편법 차단. |
| 07-13 | Action clip과 관절 진단 추가. | 정책 발산과 물리 한계 분리. |
| 07-15 | 15-step terminal success 도입. | 순간 통과와 reward farming 차단. |
| 07-16 | Env별 크기가 다른 Box-Transport 신설. | 물체 크기 일반화 시작. |
| 07-18 | Orientation dense shaping 추가. | 판정 조건만으로 학습 불가 확인. |
| 07-20 | `ACQUIRE → TOOL_READY → USE` 확정. | 젓가락 연구 구조 분해. |
| 07-22 | Chopstick 4-대칭과 coupled orientation 적용. | Tip/tail 의미 보존. |
| 07-23 | Index·middle·thumb semantic surface 구성. | Functional tripod 명시. |
| 07-26 | Penta에서 quad로 축소. | 한 손가락이 전체 신호 차단하는 문제 제거. |
| 07-27 | `hand_grasp`와 CEM seed 탐색 신설. | 운반과 파지 획득 분리. |
| 07-28 | `pose_005`와 6-contact topology 확정. | 실제 collider 기준 pregrasp 생성. |
| 07-29 | OPEN/CLOSE 103D·20D 구성. | 반복 개폐 학습 시작. |
| 07-30 | 성공 run 보존, `hand_setting` 101D·20D 신설. | 편 손에서 pregrasp 획득 시작. |
| 07-31 | Pair reference와 thumb pivot gate 구성. | Geometry 선행 후 contact 활성. |

## 6. 정량 결과

| 실험 | 핵심 결과 | 판정 |
|---|---|---|
| Cube `2026-07-16_16-05-23` | Success 89.4%, drop 3.0%, position error 4.6 cm. | Grasp–lift–transport 사다리 성립. |
| Cube 0.1 kg `2026-07-17_23-06-15` | Success 98.2%, drop 1.3%, position error 2.8 cm. | 큐브 기준선 확보. |
| Box `2026-07-16_16-33-21` | Success 43.5% 이상, drop 15.9%까지 감소. | Random box 기본 운반 가능. |
| Orientation v1 `2026-07-18_15-37-42` | Success 0, orientation error 72~93°. | 판정 조건만 추가한 방식 실패. |
| Keypoint v1.1 `2026-07-18_22-48-57` | Success 0, orientation error 약 70°. | 파지 기하 병목 확인. |
| Long stick `2026-07-18_20-56-53` | Lift 약 7.9 cm, success 약 24%, drop 약 32%. | 얇은 물체 부분 가능. |
| Chopsticks `2026-07-21_23-01-25` | Position·orientation 성공 기준선 보존. | 후속 이식 기준 run 사용. |
| `hand_grasp` 07-28 첫 run | Final contact 4.73/6. | 5/6 seed에서 6-contact 학습 가능성 확인. |
| `hand_grasp` `2026-07-29_21-05-32` | Iter 1500에서 5.97/6, 이후 약 3/6로 붕괴. | 최신보다 best checkpoint 우선. |
| `hand_grasp` `2026-07-30_12-21-21` | OPEN/CLOSE 성공 기준선. | 전체 run 별도 보존. |
| `hand_setting` 07-30~31 | Contact-only 약 3/6, geometry-first 약 1.65/6. | 접촉 전 reference 신호 필요. |

## 7. 7월 31일 기준 Task 구조

| Task | 역할 | 관측·액션 | 상태 |
|---|---|---|---|
| `Indy-Wuji-Reach` | 팔 reach 기준선. | 15D·6D. | 완료. |
| `Indy-Wuji-Cube-Grasp` | 파지·lift·운반 테스트베드. | 당시 task 계약 사용. | 성공 기준선 확보. |
| `Indy-Wuji-Box-Transport` | 크기 일반화와 pose 운반. | 당시 67D 계열·18D. | 자세 정렬 실험 진행. |
| `Indy-Wuji-Chopsticks-Grasp` | One-stick tool-ready 운반. | 72~75D·18D A/B. | Semantic tripod 구축. |
| `hand_grasp` | Two-stick pregrasp·OPEN/CLOSE. | 103D·20D. | 성공 run 확보. |
| `hand_setting` | 편 손에서 two-stick pregrasp 획득. | 101D·20D. | Stage gate 실험 진행. |

관측 차원은 해당 날짜의 active run 기준. 이후 계약과 혼용 금지.

## 8. 주요 실패와 교훈

| 실패 | 원인 | 교정 |
|---|---|---|
| 손끝이 물체를 밀기만 함. | 물체 중심이 도달 불가능한 목표. | 표면 SDF와 cage 사용. |
| 멀리서 손바닥 방향만 맞춤. | 절대형 facing reward farming. | Progress형으로 변경. |
| Lift처럼 보이나 바닥 접촉 유지. | 중심 높이 사용. | 최저 꼭짓점 clearance 사용. |
| 접촉 순간 손과 팔이 튐. | Unclipped raw action 발산. | Action clip과 관절 진단 사용. |
| 목표를 통과한 뒤 내려놓음. | Progress와 success 분리 부족. | Hold 조건과 terminal reward 사용. |
| 자세 조건 추가 후 success 0. | Dense orientation 신호 부재. | 대칭 오차와 coupled reward 사용. |
| 접촉 수는 높지만 잘못된 파지. | Semantic pair 미정의. | 정확한 link–stick topology 사용. |
| 좋은 정책이 후반에 붕괴. | PPO가 좋은 해를 계속 갱신. | Best checkpoint 별도 보존. |
| Contact-only 3/6 정체. | 접촉 전 방향 신호 부재. | Geometry/reference gate 선행. |

## 9. 참고 이론 적용

| 참고 | 적용 내용 |
|---|---|
| Dexterous Pre-grasp | 가상점·SDF, reach/hold 분리, 단계형 보상 참고. |
| DexPoint | Contact-gated lift와 제어 penalty 참고. |
| TriFinger Transfer | Keypoint 기반 6D pose 표현 참고. |
| SimToolReal | Object-centric goal pose 추종 방향 참고. |
| Learning to Use Chopsticks | 안정 파지 정책과 사용 정책 분리 참고. |

논문 구조의 완전 재현이 아닌 Wuji Hand와 젓가락 기하에 맞춘 변형 사용.

## 10. 중간 판정

### 확보

- Reach–grasp–lift–transport 학습 사다리 확보.
- 큐브 운반 success 98.2% 기준선 확보.
- Random box와 orientation 실험 기반 확보.
- Stick tip/tail 의미를 보존하는 4-대칭 정의 확보.
- 실제 contact probe 기반 `pose_005` 확보.
- Two-stick OPEN/CLOSE 성공 run 확보.
- 편 손 획득용 `hand_setting` 환경 확보.

### 미해결

- 긴 물체의 안정 orientation 유지.
- OPEN/CLOSE 중 6-contact 장기 유지.
- 열린 손에서 Stage-1 geometry를 지나 6-contact까지 수렴.
- Random stick pose와 부분 접촉 손실 복구.
- 실물 관측·action 계약과 안전 배포.

## 11. 다음 단계

```text
hand_setting 안정화
    -> 6-contact pregrasp 획득
        -> hand_grasp OPEN/CLOSE
            -> 회전·외란 강건화
                -> sim-to-real 관측 정합
                    -> 실물 Wuji Hand 배포
```

1. Pair reference와 thumb pivot gate 진입률 확인.
2. Joint reference와 semantic contact의 단계 전환 검증.
3. 6-contact best checkpoint 보존과 후반 collapse 감시.
4. Stick pose noise와 partial-contact reset 추가.
5. Simulator velocity를 제외한 실물 관측 계약 설계.
6. ONNX·MuJoCo·실물 backend 연결.

## 12. 중간 결론

핵심 성과는 높은 reward 자체가 아니라 실패 원인을 물리·관측·action·reward·success로 분리한 구조 확립.

물체 중심 대신 표면, 중심 높이 대신 clearance, 접촉 수 대신 semantic topology 사용.
큐브 운반 성공을 젓가락 기능 파지 문제로 확장하고 `hand_grasp`와 `hand_setting`으로 역할 분리.
7월 31일 기준 다음 병목은 열린 손에서 안정적인 two-stick pregrasp를 획득하는 과정.
