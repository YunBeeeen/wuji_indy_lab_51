# 젓가락 ArUco 비전

D435와 ArUco 마커를 이용해 Stick1·Stick2 pose를 추정하는 개발·캘리브레이션 패키지.
배포 런타임 이식 위치: `Deploy/vision/`.

## 데이터 흐름

```text
+------------------+     +------------------+     +------------------+
| MAIN / SIDE D435 |---->| ArUco 검출·PnP   |---->| 마커 쌍 결합     |
+------------------+     +------------------+     +---------+--------+
                                                          |
                                                          v
                                               +----------+-----------+
                                               | Stick pose 선택      |
                                               | DUAL 우선·SINGLE 보조|
                                               +----------+-----------+
                                                          |
                                                          v
                                               Camera -> Base -> Hand
```

## 마커 구성

| 스틱 | 마커 | 추적 파일 |
|---|---|---|
| Stick1 | ID0·ID1 | `run_stick1_dual.py` |
| Stick2 | ID2·ID3 | `run_stick2_dual.py` |

- 출력 pose: `xyz+wxyz`.
- 길이 단위: m.
- Quaternion 순서: `wxyz`.
- 변환 규약: `T_A_B`는 B 좌표를 A 좌표로 변환.

## 주요 파일

| 파일 | 역할 |
|---|---|
| `calibrate_base.py` | Camera → Indy7 Base 외부 파라미터 계산 |
| `calibrate_aruco_pair.py` | Stick1 ID0·1 상대 pose와 handoff 보정 계산 |
| `calibrate_aruco_pair_stick2.py` | Stick2 ID2·3 상대 pose와 handoff 보정 계산 |
| `run_stick1_dual.py` | Stick1 DUAL/SINGLE pose 추정 |
| `run_stick2_dual.py` | Stick2 DUAL/SINGLE pose 추정 |
| `run_dual_camera_hand_stick_final.py` | MAIN/SIDE 통합과 Hand-frame 변환 |
| `references/` | 작업공간 회전 prior |
| `aruco_pair_calibration_*/` | 마커 쌍 캘리브레이션 결과 |
| `indy7_camera_extrinsic_calibration/` | Base-Camera 외부 파라미터 결과 |

## 캘리브레이션 순서

1. 실제 인쇄 마커 크기 확인.
2. `calibrate_base.py`로 각 카메라의 `T_BASE_CAMERA` 계산.
3. Stick1 마커 쌍 캘리브레이션.
4. Stick2 마커 쌍 캘리브레이션.
5. 각 스틱 DUAL/SINGLE 추적 확인.
6. 듀얼 카메라 통합 추적 확인.
7. Hand-frame pose와 reference pose 비교.

## Source 선택

- 스틱별 독립 선택.
- MAIN이 해당 마커를 하나라도 검출한 경우 MAIN 사용.
- MAIN이 해당 두 마커를 모두 놓친 경우 SIDE 사용.
- SIDE 추적기 상시 실행으로 handoff history 유지.
- 비동기 카메라 프레임에 host monotonic timestamp 사용.

## 확인 항목

- 카메라 serial과 calibration 파일 일치 여부.
- 마커 실제 크기와 코드 설정 일치 여부.
- Marker → Stick 변환 방향 일치 여부.
- `xyz` 단위와 `wxyz` 순서 일치 여부.
- MAIN/SIDE pose 차이와 timestamp 차이 확인.
- 축 시각화 경고와 pose 유효성 분리 판단.

## 백업

- 기존 README: [BACKUP/README_before_korean_2026-08-27.md](BACKUP/README_before_korean_2026-08-27.md).
- 과거 실험 코드: `archive/`, `*_backup.py`, `*_before_thread.py`.
