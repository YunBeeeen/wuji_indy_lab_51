# Wuji Hand 젓가락 정책 배포

Isaac Lab에서 학습한 손 정책을 MuJoCo와 실물 Wuji Hand에서 실행하기 위한 배포 패키지.
현재 기준: 105D 관측, 20D 잔차 액션, 30 Hz 정책, 90 Hz 실물 명령.

## 시스템 구성

```text
+----------------------+      +----------------------+      +----------------------+
| Isaac Lab 학습       |      | 정책 내보내기        |      | 배포 실행             |
| hand_real 계열 task  +----->| checkpoint / ONNX    +----->| MuJoCo / 실물 손      |
| RSL-RL PPO           |      | 입출력 계약 고정     |      | 공통 PolicyRunner     |
+----------------------+      +----------------------+      +----------+-----------+
                                                                     |
                                      +------------------------------+------------------+
                                      |                                                 |
                                      v                                                 v
                           +----------+-----------+                          +----------+-----------+
                           | 듀얼 D435 비전      |                          | Wuji Hand backend   |
                           | Stick1/2 hand pose  |                          | encoder·SDK·명령    |
                           +----------+-----------+                          +----------+-----------+
                                      |                                                 ^
                                      v                                                 |
                           +----------+-----------+     20D 액션             +----------+-----------+
                           | 105D 관측 조립      +------------------------->| clip·scale·clamp   |
                           +----------------------+                          +----------------------+
```

## 실물 실행 순서

1. `[STATE]`: 현재 관절 상태 읽기.
2. `[GLIDE]`: 현재 자세에서 시작 자세까지 저속 이동.
3. `[INSERT]`: 사용자 젓가락 배치.
4. `[SEED]`: 관절·비전 히스토리 초기화.
5. `[RUN]`: 정책 30 Hz 실행, 동일 관절 목표 90 Hz 전송.
6. `[RETURN]`: 요청 시 시작 자세 복귀 후 모터 끔.

## 105D 관측 계약

| 순서 | 항목 | 차원 | 내용 |
|---:|---|---:|---|
| 1 | 이전·현재 관절 위치 | 40 | 실물 관절 한계 기준 정규화 |
| 2 | 현재 손끝 위치 | 15 | 실측 관절값 기반 palm-frame FK |
| 3 | Stick1 이전·현재 pose | 14 | palm-frame `xyz+wxyz` |
| 4 | Stick2 이전·현재 pose | 14 | palm-frame `xyz+wxyz` |
| 5 | 직전 적용 액션 | 20 | 현재 상태를 만든 직전 정책 액션 |
| 6 | 동작 모드 | 2 | OPEN/CLOSE one-hot 또는 neutral `[0,0]` |
|  | 합계 | 105 | `hand_real` 배포 계약 |

- Quaternion 순서: `wxyz`.
- 사각 스틱 처리: local +Y축 90도 대칭 중 reference 최근접 표현 선택.
- 히스토리 순서: 이전 값 → 현재 값.
- reset 첫 관측: 같은 표본 두 번 입력.
- 101D·103D 정책: 전용 adapter 없이 105D 실행기 사용 불가.

## 20D 액션 계약

```text
ONNX 출력
  -> [-1, 1] clip
  -> 현재 q + 관절별 residual scale
  -> 실물 command limit clamp
  -> 90 Hz 위치 목표 전송
```

- Joint1/2 scale: `0.10 rad`.
- Joint3 scale: `0.20 rad`.
- Joint4 scale: `0.15 rad`.
- 관절 순서: `finger1_joint1`부터 `finger5_joint4`까지 20개.
- 관측과 액션 기준: 같은 현재 관절 표본 재사용.

## 듀얼 카메라 인식

```text
MAIN D435 ──┐
            ├─> 마커 pose ─> 스틱별 source 선택 ─> Base ─> Hand frame ─> xyz+wxyz
SIDE D435 ──┘
```

- Stick1 마커: ID0·ID1.
- Stick2 마커: ID2·ID3.
- Source 우선순위: MAIN 우선, MAIN이 해당 두 마커를 모두 놓친 경우 SIDE 사용.
- 두 카메라 무효: HOLD → STALE/LOST 전환.
- 정책 보호: STALE/LOST 발생 시 새 동작 중단 후 safe stop.

## 폴더 구성

| 경로 | 역할 |
|---|---|
| `run/` | 실물·MuJoCo 실행, 정책 진단, 관절 기록 재생 |
| `policy/` | ONNX 추론, 105D 관측 조립, 20D 액션 변환 |
| `backends/` | 실물 SDK와 MuJoCo 공통 인터페이스 |
| `vision/` | 듀얼 D435 추적기와 정책용 pose provider |
| `common/` | 관절 순서, 한계, 차원, 타이밍 공통 계약 |
| `assets/` | MuJoCo·카메라·Wuji description 자원 |
| `models/` | 배포 정책 파일 위치 |

## 안전 기준

- 최초 확인: `--read-only` 사용.
- 실물 enable: 사용자 확인 후 실행.
- 관절 목표: policy clip → residual scale → command limit clamp.
- 비전 이상: stale 상태에서 새 액션 차단.
- 부하 감시: 전류·온도 확인.
- 예외 처리: `Ctrl+C`와 오류 발생 시 `finally`에서 모터 끔.
- 검증 순서: read-only → 짧은 실행 → 장시간 실행.

## 기술 문서

- 실행 명령: [CLI.md](CLI.md).
- 검증 결과: [VALIDATION_REPORT.md](VALIDATION_REPORT.md).
- 학습·배포 계약 기록: [common/ISAAC_POLICY_CONTRACT.md](common/ISAAC_POLICY_CONTRACT.md).
- 비전 상세 기록: [vision/vision.md](vision/vision.md).
- 기존 영문 README: [BACKUP/README_before_korean_2026-08-27.md](BACKUP/README_before_korean_2026-08-27.md).

## 변경 시 확인

- checkpoint와 adapter의 observation/action 차원 일치 여부.
- 학습·배포의 관절 순서와 한계 일치 여부.
- residual scale과 policy frequency 일치 여부.
- Stick pose의 frame·단위·quaternion 순서 일치 여부.
- 카메라 serial과 calibration 파일 일치 여부.
- reward·reset·physics 변경 시 checkpoint 재사용 가능 여부.
