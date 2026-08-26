# Deploy — 작업 중 막힌 지점

MuJoCo / 실물 손 배포 쪽 함정만 모았다. 이 폴더 파일을 건드릴 때 로드된다.
패키지 구조와 층 규칙도 여기 있다.

### 패키지 구조 (2026-08-21 `mujoco_deploy` -> `Deploy` 재편)

- **`Deploy/` 는 층으로 나뉘고 임포트는 한 방향이다**:
  `run -> backends -> policy/vision -> common`. 취향이 아니라 **환경 제약**이다.
  `wuji_mujoco` 엔 wujihandpy 가 없고 `wuji_hw` 엔 mujoco 가 없어서, 역방향이
  생기면 환경 하나가 **임포트 시점에** 깨진다.
  `tests/test_common.py: PackageLayerTests` 가 소스를 읽어 검사하고, 재편 중에
  실제로 두 건을 잡았다 (`policy/finger_reach.make_finger_reach_backend` 가
  MuJoCo 백엔드를 생성 -> `backends/mujoco_wuji.py` 로 이동).
- **`common/` 의 기준은 "공유"가 아니라 "바꾸면 체크포인트가 무효"다.** 이름이
  `common` 이라 잡동사니 서랍이 되기 쉬운데, 판별 기준은 `common/__init__.py` 에
  적어 뒀다: *이걸 바꿔도 이미 학습된 정책이 유효한가?* 유효하면 다른 층이다.
  MuJoCo 관절 저장 순서(`backends/joint_mapping.py`)와 카메라 외부파라미터
  (`vision/deploy_rig.py`)는 실측값이지만 **한쪽에만 존재**하므로 여기가 아니다.
- **Protocol 은 구현이 아니라 계약이다.** `perception.py`(StickPoseProvider)와
  `backend_protocol.py`(WujiBackend)는 이름만 보면 vision/backends 소속 같지만
  **인터페이스 정의**라 `common/` 에 있다. `policy/` 가 이 둘에 의존하는데,
  `policy/` 는 카메라도 시뮬레이터도 없는 환경에서 임포트돼야 한다.
- **옛 `scene_contract.py` 는 진짜와 가짜가 섞여 있었다.** Isaac 리셋 상수(실제)
  -> `common/isaac_reset.py`, 렌더링 ArUco 픽스처(전부 지어낸 값)
  -> `vision/sim_aruco.py`. 섞여 있던 탓에 Base 규약 두 개를 혼동했다(위 항목 참조).
- **경로가 한 단계 깊어졌다.** `Path(__file__).resolve().parent` 로 자산을 찾던
  모듈(`fingertip_fk`, `mujoco_wuji`, `aruco_perception`, `run_policy`)은 전부
  `parents[1]`/`parents[2]` 로 고쳤다. 파일을 옮길 때 **`__file__` 기반 경로를
  같이 볼 것** — 임포트만 고치면 테스트가 FileNotFoundError 로 무더기로 깨진다.
- **`models/` 에 ONNX 를 복사해 둔다.** 배포 기본 경로가
  `nrmk_isaaclab_wuji/logs/rsl_rl/<run>/exported/` 를 직접 가리키면 런을 정리하는
  순간 죽는다 — 실제로 `run_policy` 기본값이 사라진 런을 가리키고 있었다.
  파일명은 `<task>_<run>.onnx` (출처가 곧 이름).
- **`Deploy/` 는 2026-08-21 에야 git 에 들어갔다.** 그 전까지 통째로 untracked
  였다. 이 폴더를 대규모로 손대기 전에 `git status` 로 추적 여부를 확인할 것.

### 배포 리그 프레임 (2026-08-21)

- **"Base" 라는 이름의 프레임이 두 개고 up 방향이 다르다.**
  `scene_contract.py` 의 Base 는 **+X 가 위**(`R_WORLD_BASE` 주석), Vision 의
  Base 는 Indy7 로봇 베이스라 **+Z 가 위**다. 섞으면 팜 법선이 수직에서
  3.7도인 걸 **87.5도**로 읽는다. 2026-08-21 실제로 이걸로 "손바닥이 하늘을
  안 본다"고 오진했고, 사용자가 눈으로 보고 있는 사실과 정면으로 충돌해서야
  잡혔다. **Vision 값으로 계산할 땐 up = Base +Z.**
- **마운트 요각 155도와 q6 는 따로 검증할 수 없다.** 둘 다 같은 축(+Z) 회전이고
  마운트 이동 `[0,0,0.107]` 도 그 축 방향이라
  `Rz(q6) @ Rz(155) == Rz(q6+155)` 이 **4.6e-16** 으로 성립한다.
  155/25 든 100/80 이든 기하가 동일하다. 트래커 주석의 "physical Hand-axis
  verification 후 155 를 고쳐라" 는 **존재하지 않는 양을 재라는 요구**다.
  식별 가능한 건 합 `TOTAL_YAW_DEG = 180.000097` 뿐.
- **`palm_link` +X 가 팜 판 법선이다** — 주석이 아니라 형상으로 확인.
  팜 메시 bbox `31.9 x 81.1 x 104.8 mm` (X 가 얇은 축), 면적가중 면법선의
  99.77% 가 X. 손끝 구름으로 추정하면 `[0.879, 0.057, 0.474]` 로 애매하게
  나오니 **판 메시를 쓸 것**.
- **"손바닥이 하늘을 본다"는 확인이지 캘리브레이션이 아니다.** 회전축(hand +Z)이
  Base 에서 거의 수평이라 팜업이 합에 민감하긴 하다(합 140도→42.8도 기울어짐,
  180도→3.7도, 220도→37.4도). 하지만 **5도 이내로 보이는 합 범위가 178.4~187.0도**
  라 눈으로는 8.6도 폭까지만 좁혀진다. 팜업 최적은 **182.724도**이고 180 을 쓰면
  **3.73도(= palm 원점 100mm 지점에서 6.5mm) 잔차**가 남는다. 스틱 포즈가 5~7mm
  씩 어긋나 보이면 여기부터 의심할 것. 단 잔차가 요각이 아니라 `T_BASE_J6`
  (고정 q1~q5)의 오차일 수도 있어 **최적값을 그냥 채택하지 않고 기록만 해뒀다**
  (`deploy_rig.PALM_UP_OPTIMUM_TOTAL_YAW_DEG`).
- **`scene_contract` 를 실측값으로 덮어쓰지 말 것.** 그 파일의 카메라·마커는
  MuJoCo 안에서 ArUco 를 렌더링해 검증하는 **테스트 픽스처**고
  (`tests/test_vision.py`, `run_policy --validate-aruco`), 갈아치우면 검증
  대상 자체가 바뀐다. 실측 리그는 `deploy_rig.py` 로 분리해 뒀다.

### tip 프레임 검증 도구 (2026-08-21)

- **`nrmk_isaaclab_wuji/scripts/rsl_rl/verify_tip_frames_isaac.py`** — `obs[40:55]` 를
  두 URDF FK 와 대조한다. 정책·체크포인트 불필요, 1스텝. 실측 결과
  `|Isaac − ISAAC_URDF| 0.0001 mm` / `|Isaac − OFFICIAL_URDF| 2.9981 mm`.
  **USD 를 다시 임포트하거나 URDF 를 손대면 이걸 먼저 돌릴 것.**
- 학습 중에는 돌리지 말 것(Isaac Sim 을 띄운다). 도구 docstring 에도 적어 뒀다.

### 남의 코드를 "그대로 쓴다"의 의미 (2026-08-22)

- **함수를 안 고치는 것과 그 함수의 입력을 안 바꾸는 것은 다르다.**
  듀얼 카메라 트래커의 스틱별 규칙(MAIN 이 마커를 하나라도 보면 MAIN, 둘 다 놓칠
  때만 SIDE)을 한 줄도 안 고쳤는데, 그 규칙에 `latest()` 대신 **히스토리 프레임**을
  먹였다. 함수는 같아도 결과가 달라진다. 실측 대조:

  | | 스텝 | HOLD | 결과 |
  |---|---|---|---|
  | 재동기화 ON | 51 | 8~11 | 17초 못 버팀 |
  | 재동기화 OFF | **540** | **4** | 스케줄 전체 완주 |

- **`find_nearest_timestamp_pair` 는 |dt| 만 최소화한다. 나이는 안 본다.**
  히스토리 8프레임에서 "가장 시간이 붙은 쌍"이 제일 오래된 두 프레임일 수 있고,
  그러면 **그 시점의 검출 결과가 딸려온다.** 스큐 49→10 ms 를 사려고 검출을
  잃었다. 원본이 이걸 진단 출력에만 쓴 이유가 있었다 —
  docstring 에 *"Used only for MAIN-vs-SIDE agreement diagnostics"* 라고 적혀 있다.
  **진단용 함수를 제어 경로에 넣을 땐 반환값에 제한을 걸 것.**
- **1 mm 를 줄이려다 추적을 잃지 말 것.** 교차 카메라 스큐가 만드는 오차는
  스틱 속도 21~24 mm/s 기준 약 1 mm 였다. 그걸 없애려다 초당 몇 번씩 NONE 이 났다.
- **진단이 실제 경로와 다른 데이터를 읽으면 진단이 거짓말을 한다.** 중재는 페어
  프레임을 쓰는데 로그는 `latest()` 를 찍어서, "MAIN 에 유효한 포즈가 있는데
  결과는 NONE" 이라는 모순된 로그가 나왔다. 그걸 근거로 또 오진할 뻔했다.
  **로그는 반드시 실제로 사용된 값을 찍을 것.**

### 원본 트래커의 구조적 성질 (제어 루프에서만 드러남)

- **`MAIN_VISIBLE_BUT_INVALID_SIDE_BLOCKED`** — MAIN 이 마커를 봤는데 포즈를 못 풀면
  SIDE 가 멀쩡해도 아무것도 안 내놓는다. 원본 주석에 `# Intentional` 이라고 적혀 있고,
  **화면 표시 용도로는 맞는 선택**이다(캘리브레이션 섞인 값을 보여주느니 비운다).
  그러나 제어 루프에서는 그게 곧 추적 끊김 → HOLD → STALE → safe_stop 이다.
  용도가 바뀌면 설계 결정의 값도 바뀐다. **바꿀지는 사용자 판단** — 대안은
  ① 포즈 실패 시 SIDE 허용 ② HOLD 시간(100 ms)을 늘려 버티기.
- **`raw_valid` 와 필터 출력은 다르다.** `result_to_base_transform` 은 `position`
  (= `final_filtered_position`)을 보는데, `raw_valid` 가 True 면 필터가 그 자리에서
  초기화되므로 둘이 어긋나는 경우는 **없다**. 이 경로는 범인이 아니니 다시 파지 말 것.

### 성능 진단 (2026-08-21, 헛짚음 3회 기록)

- **추측하지 말고 계측을 먼저 볼 것.** 2026-08-21 프레임 나이 48 ms 를 보고 원인을
  세 번 연속 지어냈다: ① onnxruntime 스레드풀이 CPU 를 뺏는다(실측 **2.3%**,
  노이즈) ② 트래커가 30 Hz 를 못 따라간다(실측 **processed 29.8 Hz**, 5초에 스킵 1~3)
  ③ GIL 경합(배경 스레드 2개가 메인 스레드에 **0.05 ms**). 전부 아니었고, 정답은
  계측 구멍 안에 있었다. **계측을 붙여놓고 추측부터 하면 계측이 없는 것과 같다.**
- **`total` 과 부분 합이 안 맞으면 그게 답이다.** 21.67 ms 총합에 측정된 부분이
  2.9 ms 였다 — 18.7 ms 가 아무도 안 보는 데 있었고, 그게 병목이었다. `StageTimer`
  가 `(미계상)` 줄을 찍으니 **가설을 세우기 전에 그 줄부터 볼 것.**
- **read-only 의 `joint_read` 18 ms 는 주행 숫자가 아니다.**
  `RealWujiHand.read_joint_positions` 는 `self.controller` 가 있으면
  `controller.get_joint_actual_position()`(업스트림 스트림), 없으면
  `hand.read_joint_actual_position()`(**블로킹 SDO**) 로 간다. read-only 는
  `realtime_controller` 를 안 열어서 항상 느린 쪽이다. read-only 틱 21.7 ms 중
  18.3 ms 가 이것이고, 주행 추정치는 **≈3.4 ms**. **두 경로의 타이밍을 섞어 보고하지 말 것.**
- **지키지 않는 예산을 보고서에 찍지 말 것.** `_read_only_loop` 에 페이싱이 없던 채로
  `예산 33.3 ms` 를 출력했고, 그 상태에서 "30 Hz 가 안 나온다"는 얘기를 꺼냈다.
  30 Hz 는 시도된 적조차 없었다. 지금은 절대 데드라인으로 페이싱한다.
- **지연(latency)과 처리율(rate)은 다른 축이다.** 프레임 나이 48~55 ms 는
  트래커 처리(27) + 비동기 오프셋(≤ 카메라 주기 33) 이라 **30 Hz 를 완벽히 지켜도
  그대로 남는다.** 나이가 크다고 처리율 문제라고 말하면 엉뚱한 곳을 파게 된다.

### 듀얼 카메라 트래커 (2026-08-21)

- **`find_nearest_timestamp_pair` 는 포즈 경로가 아니다.** docstring 이 명시한다:
  *"Used only for MAIN-vs-SIDE agreement diagnostics. The final per-stick selector
  still uses each camera's latest fresh result independently."* 원본이 출력하는
  안정적인 `Nearest sync pair |dt| = 7~8.5 ms` 와 요동치는
  `Latest timestamp |dt| = 8~58 ms` 는 **다른 양**이고, 브리지가 쓰는 건 후자다.
  이걸 혼동해서 "내 통합이 원본보다 나쁘다"고 오진했다 — 실제로는 동일 동작.
- **`CameraProcessingWorker.reset()` 은 최신 프레임까지 지운다** (`_latest = None`).
  그런데 `PolicyObservationAdapter.reset()` 은 바로 다음 줄에서 샘플한다. 제공자의
  `reset()` 은 지운 뒤 **프레임을 다시 기다려야** 한다. 안 그러면 첫 샘플이
  "마커가 안 보인다"고 보고하는데, 실제로는 방금 자기가 버린 것이다.
- **`CameraStream.start()` 는 프레임이 오기 전에 반환한다.** 곧바로 샘플하면
  `latest()` 가 None 이고, 중재 로직은 그걸 "아무것도 못 봤다"로 읽는다 —
  마커가 시야 밖인 것과 구분이 안 된다. `start()` 에서 두 카메라의 첫 처리 프레임을
  기다릴 것(`wait_for_first_frames`).
- **"포즈 없음"은 원인이 셋이고 조치가 다 다르다**: 프레임 없음(카메라/연결) /
  프레임은 있는데 마커 0개(조준·가림·조명) / 마커는 보이는데 포즈 없음(IPPE 후보 전부 기각).
  한 메시지로 뭉치면 사람을 엉뚱한 곳으로 보낸다.

### 슬루 가드 (2026-08-21)

- **가드는 액션 스케일보다 작으면 안 된다.** `--max-step-rad` 를 0.05 로 두었더니
  계약이 허용하는 `joint3` 0.2 / `joint4` 0.15 를 전부 거부했다. 가드의 목적은
  손상된 ONNX·디코드 버그를 잡는 것이지 액션 스케일을 재심하는 게 아니다.
  기본값은 `ACTION_SCALE_RAD`(관절별) — 합법 출력에 정확히 무동작.

### 명령 한계·소프트 여백 (2026-08-19)

- **기계적 스톱에 눌린 관절은 소프트 한계를 "넘겨서" 읽힌다.** 실측
  `finger3_joint3` = 1.6804522 vs `COMMAND_TARGET_LIMITS` 상한 1.680047 (초과
  0.4 mrad = 0.023°). 측정값에서 궤적 보간을 시작하면 **첫 목표가 이미 범위 밖**이라
  `write_joint_position_targets`가 거부하고 주행이 죽는다. 궤적 끝점은 clamp할 것
  (`real_wuji_scheduler.py`). 단 **관측·잔차에 쓰는 q는 원본을 유지**하고 정책 출력에
  대한 거부 정책도 유지 — clamp는 궤적 생성에만 적용한다. 초과가 mrad가 아니라 수십
  mrad라면 clamp로 덮을 게 아니라 `REAL_HAND_FACTORY_LIMITS`를 다시 읽을 신호다.
- **~~`COMMAND_TARGET_LIMITS`를 줄여서 여백을 만들지 말 것.~~ (2026-08-22 사용자 지시로 철회)**
  이제 `COMMAND_TARGET_LIMITS = REAL_HAND_FACTORY_LIMITS * COMMAND_LIMIT_RATIO`
  (**0.95**)다. 사용자가 안전을 위해 명시적으로 선택했고, 학습 액션 공간을 좁히는
  대가를 알고 결정한 것이다 — **되돌리지 말 것.**
  다만 원래 경고의 핵심은 그대로 유효하다: **`OBSERVATION_NORMALIZATION_LIMITS`는
  절대 같이 줄이지 말 것.** 그건 정책 입력의 의미를 정의하므로 스케일하면 학습한
  정책이 본 적 없는 값을 먹는다. `validate_contract`가 이제 등식 대신
  ① 정규화 표 == 공장 표 ② 명령 표 == 공장 표 × ratio ③ 명령 ⊂ 정규화 를 강제한다.
- **`limit × ratio` 와 `soft_command_limits` 는 다른 값이고 섞이면 안 된다.**
  전자는 사용자 지정 규약(`COMMAND_LIMIT_RATIO`, 정책 명령 경로),
  후자는 Isaac 규약 `중심 ± f × 반범위`(기동 glide, finger_reach).
  `finger5_joint3` 상한이 **1.5914 vs 1.6197** 로 다르다.
  `limit × ratio` 는 **범위가 0을 품을 때만** 양쪽을 좁힌다 — 20관절 전부 그렇고,
  아니게 되면 `policy_contract.py` import 시점에 예외가 난다.
- **`MIDDLE_COMMAND_TARGET_LIMITS`(finger_reach)는 공장 표에서 직접 딴다.**
  `COMMAND_TARGET_LIMITS`를 쓰면 reach CLI 의 `--limit-margin 0.95` 와 겹쳐
  **실효 0.90** 이 되어 검증된 태스크가 조용히 움직인다.
- **`fraction == 1.0`은 재계산하지 말고 원본을 복사할 것.** float32에서
  `centre ± half`로 재구성하면 **1 ULP 어긋나** "기본값은 기존 로그를 그대로 재현한다"가
  깨진다. 2026-08-19 테스트가 실제로 잡았다.
- **`soft_command_limits` 의 여백 정의는 `중심 ± f × 반범위`** (Isaac
  `soft_joint_pos_limit_factor` 규약). 이 함수 안에서 `f × 상한`으로 바꾸지 말 것 —
  범위가 0을 안 품으면 뜻이 뒤집힌다. (정책 명령 경로의 `COMMAND_LIMIT_RATIO` 는
  일부러 `limit × ratio` 이고, 0을 품는다는 전제를 import 시점에 검사한다.)
- **Isaac Wuji는 `soft_joint_pos_limit_factor=1.0`이다** (`assets/wuji.py:95`) —
  즉 **학습된 정책은 하드 스톱까지 명령할 수 있었다.** 배포에서만 0.95를 걸면 의도된
  sim-real 불일치이고, 그래서 계약 테이블과 분리해 이름을 갖게 둔 것이다.
  **2026-08-22 실측으로 이게 이론이 아님이 확인됐다**: `finger5_joint3` 가 시뮬·실물
  양쪽에서 100% 스텝 동안 상한에 붙어 틱당 179 mrad 를 벽 너머로 요구했고
  (`finger2_joint4` 96.7%, `finger1_joint2` 90.8%), OPEN/CLOSE 로 액션이 달라져도
  출력은 같았다. 0.95 적용 후 그 관절 q 가 1.6763 -> 1.6022 로 내려온다.
- **finger_reach에서 여백이 공짜였다고 `hand_real`에 그대로 옮기지 말 것.**
  finger_reach는 4 DoF로 3D 목표를 쫓아 여유자유도가 1개라 0.90까지도 도달 범위
  손실이 0이었다(19만점 격자 IK). 그러나 `hand_real` pregrasp는 **범위 밖**이다 —
  `finger3_joint3`(idx 10)과 `finger5_joint3`(idx 18)이 둘 다 1.6272 인데,
  `soft_command_limits(0.95)` 로는 상한 1.6244 / 1.6197(초과 2.8 / 7.5 mrad),
  현행 `limit × 0.95` 로는 1.5960 / 1.5914(**초과 31.2 / 35.8 mrad**).
  `run_hand_policy_real` 이 pregrasp 를 clamp 하고 그 양을 출력한다 — 안 하면
  glide 첫 목표가 `write_joint_position_targets` 에 거부당해 주행이 죽는다.
- **도달성 판단에 `WujiHand1FingertipFK.fingertip_positions_in_palm`를 루프로 돌리면
  너무 느리다** (20관절 전체 계산, 격자 8만점에 2분 초과). 한 손가락 체인만
  `_chain_to_palm(fk.tip_link_names[i])`로 뽑고 Rodrigues로 배치 계산할 것.
  `wuji_mujoco` 환경에 **scipy가 없어** `least_squares`도 못 쓴다 — 격자 + 국소
  정밀화로 풀었다.
- **`min`/한계 압착 진단은 30 Hz 로그로 부족하다.** "떨고 있나 기대어 있나"는 90 Hz
  명령 로그의 진폭·속도로 갈린다 (실측 J3: 진폭 11.6 mrad, 붙었을 때 |v| 85 mrad/s
  vs 자유 구간 240 → 망치질이 아니라 기대어 있음).

### MuJoCo 젓가락 파지 진단 (2026-08-20, 오진 6회 기록)

**결론부터: "pregrasp 목표를 고정하면 젓가락이 붙어 있어야 한다"는 전제가 틀렸다.**
같은 조건에서 Isaac 실측(`hand_grasp`, 루트 고정, 목표 고정 10초):

| | Isaac | MuJoCo |
|---|---|---|
| Stick1 | **0.65초에 이탈** (554mm 후 정지) | 47.7mm 미끄러짐, 10초 후에도 손에 있음 |
| Stick2 | 0.3mm 유지 | 208mm, 5초에 낙하 |

**어느 쪽도 둘 다 잡지 못하고 서로 반대다.** Stick2는 손바닥에 고정되는 정적 접촉이라
양쪽 다 안정적이지만, **Stick1은 정책이 매 스텝 붙잡는 것**이라 보정을 없애면 Isaac에서도
떨어진다. MuJoCo 충돌 형상·마찰·접촉 그룹 어느 것도 원인이 아니었다.
**파지가 안 된다고 MuJoCo부터 의심하지 말 것 — Isaac에서 같은 측정을 먼저 하라.**
도구는 `nrmk_isaaclab_wuji/scripts/rsl_rl/measure_stick_hold_isaac.py`(정책 불필요) 와
`run_policy.py --hold-pose`.

#### 측정 도구 함정 (이것들 때문에 오진했다)

- **`--smoke-backend`는 파지 테스트가 아니다.** 액션 0이 잔차라
  `target = q_current`가 되어, 손이 **자기 드리프트를 매 스텝 새 목표로 삼는다.**
  파지가 구조적으로 풀린다(10초 실측: 고정 목표 47.7mm vs 액션 0 387926mm, 관절 124° 이동).
  파지 확인은 `--hold-pose`(목표 고정).
- **`env.step()`을 우회하면 액션 term이 안 돈다.** 떠 있는 루트 태스크
  (`hand_move`/`hand_real`/`hand_object`/`hand_final`)는 `HandRootHoldAction`이 루트를
  잡으므로, `sim.step()`을 직접 부르면 **손 전체가 떠내려가** palm 프레임 측정이 오염된다
  (변위가 442→147mm로 *줄어드는* 게 그 증상). `hand_grasp`는 루트 고정이라 안전하고
  MuJoCo(팜 고정)와 조건이 같다.
- **`env.reset()` 직후 `data` 버퍼는 리셋 값을 반영하지 않을 수 있다.** 거기서 기준점을
  잡으면 "10초 내내 상수인데 554mm 어긋난" 값이 나온다. **물리 1스텝 뒤에** 잡을 것.
  검증용 palm 프레임 리셋 기준값: `stick1 [25.07, 24.25, 96.96]`,
  `stick2 [35.60, 16.08, 73.37] mm`.

#### MuJoCo 모델 읽을 때

- **mesh geom 충돌은 convex hull이다.** 판별은 `mesh_graphadr >= 0`.
  **`mesh_facenum`은 렌더용이라 볼록화 여부 판단에 쓰면 안 된다** — 9122 그대로라서
  "볼록화 안 한다"고 오판했다.
- **`mesh_graph`로 hull 정점/면 수를 읽은 내 코드는 틀렸다.** 관계없는 geom을 바꿨는데
  다른 메시의 hull이 377/750 → 18/59로 변한 걸로 드러났다. 그 숫자를 근거로 쓰지 말 것.
- **`geom_pos`/`geom_quat`은 MuJoCo의 메시 재중심화 보정이다.** 0이 아니어도 정상이고,
  적용하면 원본 STL과 0.0000mm로 복원된다. 배치 이상으로 오해하지 말 것.
- **접촉 그룹**: link1 `(0,0)`, link2 `(2,2)`, link3/link4/palm `(1,1)`, stick `(1,1)`.
  `1 & 2 == 0`이라 link2↔stick은 성립하지 않는다. 다만 **link2를 `(1,1)`로 바꿔도
  pregrasp에서 link2↔stick 접촉은 0개**다(거리가 멀다). 원인이 아니었다.
- **Isaac도 `convexHull`이다** (`configuration/wuji_right_physics.usd`).
  convexDecomposition이 아니므로 "PhysX는 분해하고 MuJoCo는 hull"은 틀린 얘기다.
- 손바닥 충돌 메시는 Isaac이 쓰는 원본(`palm_link_collision.STL`)으로 교체해 뒀다
  (`assets/isaac_collision/`). 관통은 3.942 → 3.662mm로 **거의 안 변했다** —
  두 메시 부피 차이가 1%뿐이라 예상됐어야 했다.
- **접촉 강성(solref)을 낮추면 더 나빠진다.** 실측 timeconst 0.02(기본) 40mm →
  0.05 이상은 전부 사출. 강성은 지렛대가 아니다.

#### 자체 제작 기하 검사를 근거로 쓰지 말 것

좌표 변환 방향을 틀려서(`(w-p) @ R` vs `R.T @ (w-p)`) "스틱이 손바닥 바깥에 있다"는
결론을 세 번 반복했다. bbox를 찍어봤으면 x가 100mm까지 뻗은 게 바로 보였다.
**시뮬레이터 자신의 값만 쓸 것** — `mj_geomDistance`, `contact.dist`, `geom_size`,
`geom_margin`. 자작 검사가 꼭 필요하면 **정답을 아는 입력으로 먼저 검증**하고
(무게중심=내부, 1m 밖=외부), bbox를 출력해 축 방향을 눈으로 확인할 것.

### 실물 Wuji Hand (wujihandpy)
- **SDK가 두 종류다. 섞으면 조용히 안 된다.** 구형 `wuji_sdk`(`read_joint_state()`,
  `hand.enable()`, `joint_command().publish()`, `LowPass(cutoff_hz=)`)와 현행
  `wujihandpy`(`read_joint_actual_position()`, `write_joint_enabled()`,
  `realtime_controller(enable_upstream, filter)`, `LowPass(cutoff_freq=)`).
  **인자 이름부터 `cutoff_hz` vs `cutoff_freq`로 다르다.** 설치·실측 검증된 건
  `wujihandpy`(`/home/lsc/wuji_test/move_middle_j1.py`가 실제로 중지 J1을 움직임).
  `hold_current_pose.py`는 구형 API라 참고하면 안 됨.
- **`read_handedness()`는 손 본체의 좌우가 아니다.** 실측 오른손에서 `0`을 반환하고
  `TactileHandedness`는 `0=LEFT`라 왼손으로 오독하기 딱 좋다(2026-08-18 실제로 오독함).
  촉각 글러브용 필드이거나 미설정 기본값. 좌우 계약의 근거는 핀된
  `wuji-description hand/body RIGHT` 모델이다.
- **`read_joint_actual_position()`은 모터 enable 없이도 읽힌다.** enable은 *명령*을 위한 것.
  `move_middle_j1.py`도 enable 전에 먼저 읽는다.
- **realtime 루프 중 q 읽기는 미검증이다.** 벤더 예제 셋 다 루프 안에서 q를 한 번도 안 읽는다
  (같은 target을 반복 전송만). 잔차 정책은 매 policy step마다 읽어야 하므로
  `IController.get_joint_actual_position()` + `enable_upstream=True`를 쓰는데,
  **이 조합은 예제로 검증된 패턴이 아니다.** `--read-source hand` 대안과
  `policy_inference_ms` 로깅을 같이 둘 것.
- **`wujihandpy`는 `wuji_hw` 환경에만 있고 거기엔 mujoco가 없다.** 반대로 `wuji_mujoco`엔
  wujihandpy가 없다. `real_wuji.py`는 mujoco를 import하면 안 되고, 공용 계약 모듈도
  모듈 레벨에서 mujoco를 끌면 안 됨(`make_finger_reach_backend` 안에서만 지연 import).
- **`onnxruntime`이 `wuji_hw`에 기본 설치돼 있지 않다.** `--policy` 쓰려면 별도 설치.

