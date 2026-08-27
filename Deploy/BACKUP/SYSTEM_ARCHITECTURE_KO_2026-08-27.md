# 젓가락 파지 시스템 구성

학습 정책이 시뮬레이션에서 실물 Wuji Hand까지 전달되는 흐름 정리.
현재 배포 기준: 105D 관측, 20D 관절 잔차 액션, 30 Hz 정책, 90 Hz 명령.

## 전체 흐름

```text
+----------------------+      +----------------------+      +----------------------+
| Isaac Lab 학습       |      | 정책 내보내기        |      | 실물·MuJoCo 실행      |
| hand_real 계열 task  +----->| checkpoint / ONNX    +----->| Deploy/run            |
| RSL-RL PPO           |      | 고정 입출력 계약     |      | 공통 PolicyRunner     |
+----------+-----------+      +----------+-----------+      +----------+-----------+
           |                             |                             |
           v                             v                             v
  reward·reset·physics          obs 105D / action 20D        관측 조립·추론·명령
```

학습 task의 보상 계산에는 접촉력과 물리 상태 사용 가능.
실물 배포 actor 관측에서는 시뮬레이터 속도와 접촉력 제외.

## 실물 실행 구조

```text
                       +----------------------+
                       | 사용자 입력          |
                       | CLI·OPEN/CLOSE 모드  |
                       +----------+-----------+
                                  |
                                  v
+--------------------+   +--------+---------+   +----------------------+
| MAIN / SIDE D435   |   | run_hand_policy |   | Wuji Hand encoder    |
| ArUco ID 0~3       +-->| _real.py         |<--+ 20개 현재 관절       |
+---------+----------+   +--------+---------+   +----------+-----------+
          |                       |                        |
          v                       v                        v
+---------+----------+   +--------+---------+   +----------+-----------+
| StickPoseProvider  |   | 105D Observation |   | RealWujiHand backend |
| hand-frame pose    +-->| Adapter           |<--+ SDK·한계·상태 읽기   |
+--------------------+   +--------+---------+   +----------+-----------+
                                  |                        ^
                                  v                        |
                         +--------+---------+              |
                         | ONNX policy      |              |
                         | 105D -> 20D      |              |
                         +--------+---------+              |
                                  |                        |
                                  v                        |
                         +--------+---------+              |
                         | Action Adapter   |              |
                         | clip·scale·clamp +--------------+
                         +------------------+    90 Hz 관절 목표
```

실물 실행 순서.

1. `[STATE]`: 실측 관절 상태 읽기.
2. `[GLIDE]`: 현재 자세에서 시작 자세까지 저속 이동.
3. `[INSERT]`: 사용자가 젓가락 배치.
4. `[SEED]`: 관절·비전 히스토리 초기화.
5. `[RUN]`: 정책 30 Hz 실행, 동일 목표 90 Hz 전송.
6. `[RETURN]`: 요청 시 시작 자세로 복귀 후 모터 끔.

## 듀얼 카메라 인식

```text
 MAIN D435                              SIDE D435
 ID0·1=Stick1, ID2·3=Stick2             같은 마커를 보조 관측
     |                                      |
     +------------------+-------------------+
                        v
              +---------+----------+
              | 마커별 pose 추정   |
              | jump gate·history  |
              +---------+----------+
                        |
              +---------+----------+
              | 스틱별 source 선택 |
              | MAIN 우선·SIDE 보조|
              +---------+----------+
                        |
              +---------+----------+
              | camera -> base     |
              | -> hand frame      |
              +---------+----------+
                        |
                        v
              Stick1/2 xyz+wxyz
```

- 해당 스틱 마커가 MAIN에서 하나라도 보이면 MAIN 결과 사용.
- MAIN이 두 마커를 모두 놓친 경우에만 SIDE 결과 사용.
- 두 카메라 결과가 모두 무효면 HOLD를 거쳐 STALE/LOST 전환.
- `Vision/`: 캘리브레이션·추적기 개발 원본. `Deploy/vision/`: 배포 런타임 구성.

## 105D 관측 계약

| 순서 | 항목 | 차원 | 설명 |
|---:|---|---:|---|
| 1 | 이전·현재 관절 위치 | 40 | 실물 관절 한계로 정규화한 20D 두 시점 |
| 2 | 현재 손끝 위치 | 15 | 실측 관절값으로 계산한 palm-frame FK |
| 3 | Stick1 이전·현재 pose | 14 | palm-frame `xyz+wxyz` 두 시점 |
| 4 | Stick2 이전·현재 pose | 14 | palm-frame `xyz+wxyz` 두 시점 |
| 5 | 직전 적용 액션 | 20 | 현재 상태를 만든 직전 정책 액션 |
| 6 | OPEN/CLOSE 모드 | 2 | one-hot 명령 |
|  | 합계 | 105 | 현재 `hand_real` 배포 계약 |

Quaternion 순서: `wxyz`. 사각 스틱의 길이축 90도 대칭 중 reference 최근접 표현 선택.
101D `hand_final` 구형 정책과 103D `hand_move` 정책은 별도 adapter 없이 105D 실행기 사용 불가.

## 20D 액션 처리

```text
ONNX 출력
   -> [-1, 1] clip
   -> 현재 q + 관절별 residual scale
   -> 실물 command limit clamp
   -> 90 Hz 위치 목표 전송
```

잔차 스케일: Joint1/2 `0.10 rad`, Joint3 `0.20 rad`, Joint4 `0.15 rad`.
정책 관측과 액션 기준에 같은 현재 관절 표본 재사용.

## 주요 코드 위치

| 경로 | 역할 |
|---|---|
| `Deploy/run/` | 실물·MuJoCo 실행, 진단, 관절 기록 재생 |
| `Deploy/policy/` | 관측 adapter, 액션 adapter, ONNX 추론, 공통 runner |
| `Deploy/backends/` | 실물 SDK와 MuJoCo 차이를 공통 인터페이스로 연결 |
| `Deploy/vision/` | 듀얼 카메라 ArUco 인식과 palm-frame 스틱 pose 공급 |
| `Deploy/common/` | 정책 차원·관절 순서·한계·타이밍 공통 계약 |
| `Vision/` | 카메라 외부 파라미터와 마커 쌍 캘리브레이션, 추적기 검증 |
| `isaac_neuromeka/.../hand_grasp/` | 학습 scene·reward·reset·observation 구성 |
| `isaac_neuromeka/learning/` | PPO 진단과 teacher-driven distillation |

## 안전 동작

- `--read-only`: 모터 enable과 목표 전송 없이 관측·추론만 확인.
- 실물 구동: 전체 20관절 enable 전 사용자 확인.
- 액션 제한: 정책 clip → 관절별 residual → 실물 명령 한계.
- 비전 STALE/LOST: 새 정책 동작 중단 후 safe stop 전환.
- 전류·온도 감시. 예외나 `Ctrl+C` 발생 시 `finally`에서 모터 끔.
- 신규 정책 확인 순서: read-only → 짧은 실행 → 장시간 실행.

## 변경 시 확인 항목

- checkpoint observation/action 차원과 실행 adapter 일치 여부.
- 학습·배포의 관절 순서, 관절 한계, residual scale 일치 여부.
- Stick pose의 frame, 단위, quaternion 순서 일치 여부.
- MAIN/SIDE calibration 파일과 설치 카메라 serial 일치 여부.
- reward·reset 의미 변경 시 기존 checkpoint resume 가능 여부.
