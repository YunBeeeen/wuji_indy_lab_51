# Indy7 + Wuji 젓가락 조작 연구 최종 리뷰

- 정리 범위: 2026-07-08 ~ 2026-08-28.
- 대상: Isaac Lab 학습, MuJoCo 검증, 듀얼 D435 인식, Wuji Hand 실물 배포.
- 현재 상태: Two-stick 파지·OPEN/CLOSE·외란 복구·실물 정책 실행까지 연결. 안정적인 획득과 장시간 유지 보완 진행.

## 1. 전체 목표

Indy7과 Wuji Hand를 이용해 젓가락을 획득하고, 기능 파지를 유지하며, 개폐·회전·물체 조작을 수행하는 정책 구축.

```text
물체 접근
   -> 젓가락 획득
      -> 6-contact 기능 파지
         -> OPEN/CLOSE
            -> 회전·외란 복구
               -> 젓가락 물체 조작
                  -> 실물 배포
```

## 2. 전체 시스템 구성

```text
                              +----------------------+
                              | Isaac Lab + RSL-RL   |
                              | 병렬 PPO 학습        |
                              +----------+-----------+
                                         |
                              checkpoint | ONNX
                                         v
+-------------------+       +------------+------------+       +-------------------+
| Dual D435 Vision  |------>| 105D Observation       |------>| PolicyRunner      |
| ArUco Stick1/2    | pose  | joint·tip·stick·action |       | 20D residual      |
+-------------------+       +------------+------------+       +---------+---------+
                                         |                              |
                                         |                              v
                              +----------+-----------+       +----------+---------+
                              | Contract Verifier    |       | Safety Decoder     |
                              | dim·order·limit·scale|       | clip·clamp·slew   |
                              +----------------------+       +----------+---------+
                                                                        |
                                                +-----------------------+------------------+
                                                |                                          |
                                                v                                          v
                                     +----------+---------+                     +----------+---------+
                                     | MuJoCo Backend    |                     | Real Wuji Backend |
                                     | 120 Hz simulation |                     | 90 Hz command    |
                                     +-------------------+                     +-------------------+
```

## 3. 연구 아키텍처

```text
학습 계층
  |
  +-- Reach·Cube·Box
  |     `-- 접근·파지·lift·운반·orientation 원리 검증
  |
  +-- hand_grasp
  |     `-- pose_005 기반 6-contact·OPEN/CLOSE
  |
  +-- hand_setting
  |     `-- 편 손에서 pregrasp 획득
  |
  +-- hand_move / hand_real / hand_real2
  |     `-- 회전·외란·sim-to-real 관측
  |
  +-- hand_object / hand_final
  |     `-- 젓가락으로 cube 접촉·유지
  |
  `-- Deploy
        `-- ONNX·MuJoCo·Vision·실물 Wuji 실행
```

## 4. 단계별 진행 요약

| 단계 | 기간 | 핵심 결과 |
|---|---|---|
| 기본 조작 | 07-08~17 | Reach, SDF cage, lift, transport 구축. Cube success 98.2% 확보. |
| 자세·젓가락 프록시 | 07-18~26 | Box orientation, keypoint, 4-대칭, semantic tripod 구축. |
| Two-stick 파지 | 07-27~31 | `pose_005`, 6-contact topology, OPEN/CLOSE, `hand_setting` 구축. |
| 파지 강건화 | 08-02~09 | Reward gate 교정, cube hold, 외란 복구, Stage 구조 구축. |
| Sim-to-real 계약 | 08-10~16 | Simulator velocity 없는 105D 관측과 20D residual 확정. |
| 배포 구축 | 08-17~24 | MuJoCo, 듀얼 D435, real backend, 20초 실물 실행 완료. |
| 최종 보완 | 08-25~28 | Box raw 69D, tip force, contact face, hand-setting 재설계. |

## 5. 핵심 정책 계약

### 5.1 실물 주력 105D Observation

```text
joint history        40 = q(t-1), q(t)
fingertip position   15 = 5 fingertips x palm-frame xyz
Stick1 pose history  14 = previous/current xyz+wxyz
Stick2 pose history  14 = previous/current xyz+wxyz
last applied action  20 = 현재 상태를 만든 직전 action
mode                   2 = OPEN/CLOSE 또는 neutral
-------------------------------------------------------
total                105
```

- Joint position: factory joint limit 기준 정규화.
- Stick pose: palm-frame `xyz+wxyz` 사용.
- Quaternion: reference-nearest 4-대칭 선택 후 `w≥0` canonical 적용.
- History: oldest-to-newest 사용.
- Reset: previous/current에 같은 sample 입력.
- Simulator-only joint·rigid-body velocity 제거.

### 5.2 20D Action

```text
q_target = q_current + scale_joint * clip(action, -1, 1)
```

| 관절 | Residual scale |
|---|---:|
| Joint1 | 0.10 rad |
| Joint2 | 0.10 rad |
| Joint3 | 0.20 rad |
| Joint4 | 0.15 rad |

- 정책 추론 30 Hz 사용.
- 실물 명령 90 Hz 사용.
- MuJoCo 명령 120 Hz 사용.
- 실물 command limit에 0.95 margin 적용.
- Per-joint slew guard와 current limit 적용.

## 6. 현재 주요 Task

| Task | 역할 | 정책 계약 | 비고 |
|---|---|---|---|
| `Indy-Wuji-Box-Transport` | 물체 pose 운반. | 69D·26D. | Raw world pose 관측 사용. |
| `hand_grasp` | Two-stick 파지·OPEN/CLOSE. | 103D·20D. | Sim velocity 포함 legacy 학습 구조. |
| `hand_setting` | 편 손에서 pregrasp 획득. | 105D·20D. | Mode `[0,0]` neutral 사용. |
| `hand_move` | OPEN/CLOSE·회전·외란. | 103D·20D. | Hand-real의 학습 기반. |
| `hand_real` | 실물 배포용 파지·개폐. | 105D·20D. | Quaternion history 사용. |
| `hand_real2` | 획득 전단·contact A/B. | 105D·20D. | Task-local reset·reward override 사용. |
| `hand_object` | 젓가락 cube 파지. | 103D·20D. | Cube 상태는 reward-only 사용. |
| `hand_final` | 105D cube 파지 fine-tuning. | 105D·20D. | Hand-real 관측과 hand-object 목표 결합. |
| `finger_reach` | 단일 손가락 실물 검증. | 15D·4D. | Joint·FK·limit 점검용. |

Checkpoint load 전 관측 차원뿐 아니라 term 순서, normalization, action scale, reset, command 의미 확인.

## 7. Reward 설계 결론

```text
Geometry 접근
    -> 관절 target 접근
        -> semantic contact 형성
            -> contact minimum 유지
                -> OPEN/CLOSE·회전
                    -> 외란 복구
                        -> 물체 사용
```

| 원칙 | 적용 |
|---|---|
| 도달 가능한 목표 사용. | 물체 중심 대신 표면 SDF 사용. |
| 실제 lift 판정. | 중심 높이 대신 최저 꼭짓점 clearance 사용. |
| 일부 파밍 방지. | 평균보다 minimum·coupled gate 사용. |
| 기능 파지 판정. | 정확한 link–stick pair 사용. |
| 원거리 gradient 확보. | Linear/coarse 접근과 near-target finishing 분리. |
| 성공 연금 방지. | 연속 hold 후 terminal reward 1회 지급. |
| 회복 행동 학습. | Reset 실패 상태뿐 아니라 명시적 force pulse 사용. |
| 목표 면 구분. | Force 크기와 signed surface coordinate 함께 사용. |

## 8. 주요 성과

### 8.1 시뮬레이션

- Cube 운반 success 98.2% 확보.
- Random box 운반과 orientation 실험 기반 확보.
- Stick tip/tail을 보존하는 4-대칭 정의 확보.
- 실제 collider probe 기반 6-contact `pose_005` 확보.
- Two-stick OPEN/CLOSE 성공 기준선 확보.
- Hand-move 외란 학습 후 full contact 0.335→0.948 개선.
- Minimum functional force 0.107→0.449 N 개선.
- Hand-object 최초 cube hold 0→0.4375 확인.

### 8.2 배포

- Official fixed-base Wuji MuJoCo 모델과 name-based joint mapping 구성.
- Isaac–MuJoCo observation/action 정적 대조 구성.
- D435 듀얼 카메라 Stick1/2 pose provider 구성.
- 105D common adapter와 20D decoder 구성.
- Read-only 150-step 관측·추론 완주.
- 실물 600/600 step, 20초 정책 실행 완료.
- 동일 observation 입력에서 sim-to-real action 일치 확인.
- Checkpoint 계약 검사와 CSV provenance 추적 구성.

## 9. 보존 기준 모델

| 목적 | Run·checkpoint | 의미 |
|---|---|---|
| Cube 운반 | `2026-07-17_23-06-15` | Success 98.2% 기준선. |
| Chopsticks 운반 | `2026-07-21_23-01-25` | Position·orientation 성공 기준선. |
| OPEN/CLOSE | `2026-07-30_12-21-21` | Hand-grasp 성공 run. |
| Hand-move 외란 | `2026-08-08_00-55-42(3)/model_3450.pt` | 파지 회복 기준선. |
| Hand-object | `2026-08-08_20-39-52/model_300.pt` | Cube hold 기준선. |
| Hand-real geometry | `2026-08-13_17-54-50/model_900.pt` | 5 mm A/B의 6-contact 정점. |
| Hand-real 실물 후보 | `2026-08-18_23-57-25/model_4500.pt` | 105D 후속 curriculum source. |
| Hand-real 배포 | `2026-08-24_09-29-55/model_2400.pt` | 실물 실행 검증 모델. |
| Hand-real tip force | `2026-08-26_23-04-15/model_250.pt` | Tip force fine-tuning 저장점. |

Best checkpoint는 latest checkpoint와 다를 수 있음. Run 전체 지표와 play 결과를 함께 사용.

## 10. 반복해서 확인된 실패 유형

| 현상 | 실제 원인 | 대응 |
|---|---|---|
| Reward 상승, 파지 실패. | 쉬운 항 farming. | Term별 raw·final·min·max 확인. |
| 물체를 기울여 lift 획득. | 중심 높이 사용. | 최저 꼭짓점 clearance 적용. |
| 일부 손가락만 접촉. | Mean reward 희석. | Minimum과 semantic pair 적용. |
| 좋은 모델이 후반 붕괴. | PPO가 이미 좋은 해를 계속 갱신. | 주기별 checkpoint 보존과 best 선정. |
| Resume 후 성능 급락. | Reward·reset·obs 의미 변경. | Resume 대신 init checkpoint 또는 fresh 사용. |
| MuJoCo에서 파지 실패. | Isaac과 다른 hand asset 사용. | Asset·joint frame·limit 계약 대조. |
| 실물에서 glide 거부. | Limit margin 또는 float ULP 초과. | Target 재클램프와 사전 contract 검사. |
| 비전 STALE. | Tracker 재중재·marker 가림. | History resync OFF, hold 후 safe stop 사용. |
| 손 온도 급상승. | 상시 current saturation. | Action·PD·current·temperature 동시 확인. |
| Contact force는 있으나 잘못된 면. | Force만으로 위치 판정 불가. | Face axis와 signed surface coordinate 사용. |

## 11. Sim-to-real 검증 흐름

```text
checkpoint + env.yaml + agent.yaml
               |
               v
      contract verifier
               |
       +-------+-------+
       |               |
       v               v
  MuJoCo play      Real read-only
       |               |
       +-------+-------+
               v
      same-observation action 비교
               |
               v
      짧은 motor-enabled 실행
               |
               v
      current·temperature 확인
               |
               v
         실행 시간 확대
```

Trajectory가 다르다는 사실만으로 정책 불일치 판정 금지. 동일 observation에 대한 action 직접 비교.

## 12. 안전 구성

- Motor enable 전 camera preview 사용.
- 최초 실행 `--read-only` 사용.
- Pregrasp glide에 속도·slew·tracking guard 사용.
- Joint target을 실물 command limit 안으로 clamp.
- Current cap과 saturation duty 기록.
- 온도는 control loop를 막지 않는 저주기 monitor 사용.
- Perception hold 한도 초과 시 stale safe stop 사용.
- 예외·중단 시 motor disable과 CSV 저장 실행.
- 장시간 demo 전 관절별 온도 상승률 확인.

## 13. 현재 한계

1. `hand_setting`의 편 손 시작에서 안정적인 6-contact 획득률 부족.
2. OPEN/CLOSE와 회전 중 functional contact 장기 유지 미완.
3. Reset stick pose noise와 partial-contact-loss recovery 범위 제한.
4. Ring·little support와 각 stick semantic contact의 trade-off 잔존.
5. Force를 높일수록 실물 전류 포화와 온도 상승 발생.
6. ArUco pose agreement의 position p90 21.78 mm long-tail 잔존.
7. 카메라 marker 가림 시 stale perception 가능.
8. Isaac·MuJoCo·실물 asset 차이에 대한 완전한 물리 정합 미완.
9. 젓가락을 이용한 최종 물체 relocation success는 후속 검증 필요.

## 14. 후속 연구 순서

```text
hand_setting 획득 안정화
    -> functional contact 장기 유지
        -> OPEN/CLOSE
            -> root rotation
                -> disturbance + reset noise
                    -> partial-contact recovery
                        -> cube grasp·relocation
                            -> 장시간 실물 demo
```

1. 한 번에 reward·reset 변수 하나만 변경.
2. Neutral 5초 파지 안정화 후 OPEN/CLOSE 연결.
3. OPEN/CLOSE 안정화 후 회전 범위 확대.
4. 회전 안정화 후 외란·stick reset noise 결합.
5. Nominal, clearance, partial loss, perturbed pose mixture reset 적용.
6. Tip force와 current·temperature 사이 안전 영역 측정.
7. Camera long-tail과 marker 가림 대응 개선.
8. Best model의 PT·ONNX·params를 한 세트로 보존.

## 15. 최종 결론

연구의 가장 큰 성과는 하나의 거대한 젓가락질 문제를 검증 가능한 단계로 분해한 구조 확립.

Reach·cube·box 실험으로 reward와 물리 조건을 검증하고, `hand_grasp`에서 기능 파지와 개폐를 확보.
`hand_setting`에서 파지 획득을 분리하고, `hand_move`·`hand_real`에서 회전·외란·실물 관측 계약을 연결.
MuJoCo·듀얼 카메라·Wuji backend를 통해 학습 정책을 실제 손에서 실행하는 전체 경로 완성.

최종 병목은 모델 실행 자체가 아니라 편 손에서 정확한 semantic contact를 만들고, 개폐·회전·외란 동안 이를 오래 유지하는 강건성.
후속 연구는 이 획득·유지·복구를 완성한 뒤 젓가락 물체 조작으로 확장.
