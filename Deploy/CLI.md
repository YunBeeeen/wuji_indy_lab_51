# `Deploy` CLI 레퍼런스

모든 인자와 기본값을 코드에서 직접 뽑아 정리한 것이다. 전부 **프로젝트 루트**
(`/home/lsc/wuji_indy_lab_51`)에서 `python -m Deploy.run.<진입점>`으로 실행한다.
(2026-08-21 이전에는 패키지 이름이 `mujoco_deploy` 였고 모듈이 전부 최상위에
있었다. 옛 명령은 더 이상 동작하지 않는다.)

## 패키지 구조

```
Deploy/
├── common/     바꾸면 학습된 체크포인트가 무효가 되는 것.  numpy 만
│               policy_contract  fingertip_fk  stick_pose  isaac_reset
│               perception(제공자 인터페이스)  backend_protocol
├── policy/     obs → action.  백엔드 중립
│               observation_adapter  action_adapter  policy_runner
│               onnx_policy  finger_reach
├── vision/     스틱 포즈 소스
│               deploy_rig(실측 리그)  aruco_perception  sim_aruco(시뮬 픽스처)
│               run_stick{1,2}_dual  run_dual_camera_hand_stick_final  references/
├── backends/   플랜트.  mujoco XOR wujihandpy — 절대 둘 다 아님
│               mujoco_*  real_wuji*
├── run/        백엔드를 고르는 유일한 층 = 진입점
├── models/     배포용 ONNX 액터 (학습 로그에서 복사)
└── assets/  tools/  tests/
```

`common/` 의 기준은 "공유되니까" 가 아니라 **"바꾸면 이미 학습된 정책이 무효가 되는가"**
이다. 손의 물리값(관절 한계·kp/kd·effort)과 순수 규약(obs 배치·액션 스케일·
쿼터니언 브랜치)과 인터페이스(Protocol)가 성격은 달라도 전부 그 조건을 만족한다.
반대로 MuJoCo 관절 저장 순서나 카메라 외부파라미터는 실측값이지만 **한쪽에만
존재**하므로 `backends/`·`vision/` 에 있다. 자세한 판별 기준은 `common/__init__.py`.

**임포트는 한 방향으로만 흐른다**: `run → backends → policy/vision → common`.
환경이 셋으로 갈리기 때문에 강제되는 규칙이고
(`wuji_mujoco`엔 wujihandpy 가 없고 `wuji_hw`엔 mujoco 가 없다),
역방향이 생기면 환경 하나가 **임포트 시점에** 깨진다.
`tests/test_common.py: PackageLayerTests` 가 소스를 읽어 검사한다.

| 진입점 | 환경 | 대상 | 계약 |
|---|---|---|---|
| `run_finger_reach` | `wuji_mujoco` | MuJoCo | 15D → 4D (중지) |
| `run_finger_reach_real` | `wuji_hw` | **실물 Wuji Hand** | 15D → 4D (중지) |
| `run_mujoco_policy` | `wuji_mujoco` | MuJoCo | 105D → 20D. **실물과 같은 기동 절차** |
| `run_policy` | `wuji_mujoco` | MuJoCo | 105D → 20D. 계약 검증·진단 도구 모음 |
| `run_joint_replay` | 양쪽 | MuJoCo / **실물** | 정책 없음. 녹화한 관절 목표 재생 |

환경이 갈리는 이유: `wujihandpy`는 `wuji_hw`에만, `mujoco`는 `wuji_mujoco`에만 있다.
**reach와 파지는 계약이 다르다**(관측 차원, 액션 차원, 액션 스케일). 섞지 않는다.

환경을 activate 한 뒤 실행한다.

```bash
conda activate wuji_mujoco     # MuJoCo 두 개
conda activate wuji_hw         # 실물
```

MuJoCo 쪽 둘은 리포지토리 루트의 런처로 실행한다. 인자는 `-m` 형태와 동일하다.

| 런처 | = |
|---|---|
| `python mj_grasp.py` | `python -m Deploy.run.run_policy` |
| `python mj_reach.py` | `python -m Deploy.run.run_finger_reach` |

`python Deploy/run/run_policy.py` 처럼 직접 실행하면 안 된다 — 패키지 내부가
상대 import 라 `attempted relative import with no known parent package` 가 난다.
런처가 리포지토리 루트를 `sys.path` 에 넣어서 그걸 해결한다.

실물과 `tools/*` 는 런처가 없어서 `-m` 을 쓴다.

```bash
python -m Deploy.run.run_finger_reach_real --read-only
python -m Deploy.tools.compare_reach_logs a.csv b.csv
```

---

# 1. `run_finger_reach` — MuJoCo 손가락 reach

```bash
python mj_reach.py --policy PATH --scenario PATH --viewer --realtime
```

### 무엇을 돌릴지

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--policy` | 경로 | `None` | 15→4 ONNX. **생략하면 액션 0**으로 배선만 확인 |
| `--scenario` | 경로 | `None` | 타깃 시퀀스 JSON. 세 백엔드가 공유 |
| `--target X Y Z` | float×3 | `None` | 단일 타깃, **palm frame 미터** |
| `--seconds` | float | `4.0` | `--target` 하나의 유지 시간 |
| `--target-duration` | float | `None` | 시나리오의 dwell을 덮어씀 |

`--policy` 없이도 돌아간다(액션 0). `--scenario`와 `--target`은 둘 중 하나.

### 물리 — 계약이 아니라 수치 손잡이

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--physics-substeps` | int | `64` (= 1/1920 s) | 정책 스텝당 물리 스텝. **유지 시간은 1/30 s 고정** |
| `--controller-gains` | `vendor` \| `isaac_tuned` | **`vendor`** | 아래 주의 참조 |

> **주의 — 기본값이 학습 게인이 아니다.** `vendor`는 제조사 MJCF의 동정값이고,
> 정책이 학습된 게인은 `isaac_tuned`다. `README.md`/`VALIDATION_REPORT.md`는
> 아직 옛 이름(`deploy`/`official`)과 "Isaac 게인이 기본"이라고 적혀 있는데
> **둘 다 낡았다.** 지금까지의 sim-to-sim 비교는 전부 `vendor`로 돌았다.

### 출력

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--csv` | 경로 | `None` | 정책 스텝당 한 줄 |
| `--print-interval` | int | `10` | 몇 스텝마다 찍을지 |
| `--viewer` | 플래그 | 꺼짐 | MuJoCo passive viewer |
| `--realtime` | 플래그 | 꺼짐 | 벽시계 시간에 맞춤 |

`--viewer`만 주면 20초 주행이 순식간에 끝난다. **눈으로 볼 거면 `--realtime`을 같이.**

### 한계 여백

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--limit-margin` | float | **`1.0`** | 아래 3절 공통 설명 참조. MuJoCo만 1.0(기존 로그 재현성) |

---

# 2. `run_finger_reach_real` — 실물 Wuji Hand

```bash
python -m Deploy.run.run_finger_reach_real --policy PATH --scenario PATH \
    --lowpass-hz 2.0 --log-commands --return-to-start --csv out.csv
```

### 모드 — 하나만 선택 (상호 배타)

위에서 아래로 **순서대로** 검증한다.

| 모드 | 타입 | 모터 | 하는 일 |
|---|---|---|---|
| `--read-only` | 플래그 | **끔** | 관절 읽기만. `--policy`와 같이 주면 obs·action·q_target까지 계산하되 **전송 안 함** |
| `--measure-timing` | 플래그 | **끔** | SDK 왕복 지연 측정 → 90/30 Hz 지속 가능한지 판정 |
| `--hold-middle` | 플래그 | 켬 | 중지 4개만 켜고 현재 자세 유지 |
| `--test-middle-joint N` | int, `1~4` | 켬 | 그 관절만 `--delta`만큼 이동. **관절 매핑 검증** |
| `--zero-policy` | 플래그 | 켬 | 전체 루프를 돌되 액션 0 |
| `--policy PATH` | 경로 | 켬 | 실주행 |

`--read-only`만 상호 배타 그룹 **밖**이라 `--policy`와 함께 줄 수 있다(드라이런).

### 타깃 — MuJoCo와 동일

| 인자 | 타입 | 기본값 |
|---|---|---|
| `--scenario` | 경로 | `None` |
| `--target X Y Z` | float×3 | `None` |
| `--seconds` | float | `4.0` |
| `--target-duration` | float | `None` (`--parallel-mujoco`에도 그대로 전달) |
| `--delta` | float | `0.03` | `--test-middle-joint` 스텝, 라디안 |

### 타이밍

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--command-hz` | float | `90.0` | **30의 정수배만 허용** (100 Hz는 거부) |
| `--lowpass-hz` | float | `0.5` | `wujihandpy filter.LowPass(cutoff_freq=)` |

시상수 `τ = 1/(2πf)`. 정책 한 스텝(1/30 s)에 명령이 반영되는 비율:

| cutoff | τ | 한 스텝 반영률 | 실측 |
|---|---|---|---|
| 0.5 Hz | 318 ms | 9.9% | — |
| **2.0 Hz** | 80 ms | 34.2% | J1~J4 **30~32%** |

시뮬에는 이 필터가 없다. 실물에만 있는 플랜트 요소다.

### 명령 한계 여백 (`--limit-margin`)

| 인자 | 타입 | 기본값 |
|---|---|---|
| `--limit-margin` | float, `(0, 1]` | 실물 **`0.95`** / MuJoCo `1.0` |

**중심 ± f × 반범위** (Isaac `soft_joint_pos_limit_factor` 정의). `f × 상한`이 아니다 —
음수 하한에서 뜻이 뒤집힌다.

- `1.0` = 학습한 액션 공간. 정책이 **기계적 스톱까지 밀 수 있다**
- `0.95` = 중지 J3/J4 상한을 약 3.2° 낮춘다

적용 범위: 정책 클램프 / 붙잡아 두는 16관절 / 시작·복귀 glide / 단일 관절 진단.
`--parallel-mujoco`가 실물 쪽 값을 **자식 프로세스에 자동 전달**한다 — 짝지은 비교는
같은 액션 공간이어야 두 *플랜트*의 비교가 된다.

도달성 손실은 없다. 격자 IK로 REACH 박스 전체가 1.00/0.95/0.90 모두 100% 5 mm 이내.
MuJoCo 실측으로 J3 상한 압착 80.4% → 0.0%, 지문 오차 8.80 → 8.17 mm로 오히려 개선.
**단 이건 중지 4관절 태스크 얘기다. `hand_real` pregrasp(idx 10/18 = 1.6272)는
0.95 상한 1.6244/1.6197을 넘는다.**

### 시작 자세 이동 — 시뮬의 reset을 대신한다

실물엔 teleport가 없다.

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--start-seconds` | float | `3.0` | 목표를 이 시간에 걸쳐 **선형**으로 옮긴다 |
| `--start-tolerance-rad` | float | `0.03` | 도착 판정 |
| `--start-stable-seconds` | float | `0.5` | 이만큼 유지돼야 도착 |
| `--start-timeout-seconds` | float | `15.0` | 포기 시각 |

중지 4개만 시뮬 시작 자세로 가고 **나머지 16개는 실측값을 유지**한다.

고정 타깃 + 필터는 지수형이라 초기 속도가 `Δ/τ`다. 주행이 끊겨 손가락이 굽은 채
남으면 1.68 rad에서 출발해 2 Hz 기준 **21 rad/s**가 나온다. 그래서 복귀와 같이
선형으로 간다(3초면 **0.56 rad/s**).

tolerance는 **하드웨어 튜닝 값**이다. 이 손의 고착/게인은 미측정.

### 복귀

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--return-to-start` | 플래그 | 꺼짐 | 끝나고 시작 자세로 돌아간 뒤 disable |
| `--return-seconds` | float | `3.0` | 선형 |

없으면 모터가 그냥 끊겨 손가락이 끝난 자리에서 늘어진다. reach는 괜찮지만
물체를 쥐고 있으면 안 된다.

### 안전

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--current-limit` | float (A) | `None` | 전류 한계. 생략하면 유지, **어느 쪽이든 현재값 출력** |
| `--max-step-rad` | float | `None` → 액션 스케일 `0.1` | 명령이 **측정 q**로부터 이만큼 떨어지면 거부 |
| `--yes` | 플래그 | 꺼짐 | 확인 프롬프트 생략 |

모든 경로에서 `finally`가 전체 모터를 disable한다.
`--max-step-rad`는 목표끼리가 아니라 **측정값 대비**로 잰다(차이는 서보 추종 오차).

### 피드백·진단

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--read-source` | `controller` \| `hand` | `controller` | realtime 중엔 `controller`(0.009 ms). `hand`는 블로킹 SDO(20 ms) |
| `--enable-upstream` / `--no-enable-upstream` | 플래그 | **켜짐** | realtime 상향 피드백. 모터 enable과 무관 |
| `--csv` | 경로 | `None` | 30 Hz, 정책 스텝당 한 줄 |
| `--log-commands` | 플래그 | 꺼짐 | **90 Hz** 관절 상태를 `<csv>_90hz.csv`에 추가 기록 |
| `--parallel-mujoco` | 플래그 | 꺼짐 | MuJoCo를 **독립 정책 루프**로 동시 실행 |
| `--mujoco-python` | 경로 | `~/anaconda3/envs/wuji_mujoco/bin/python` | mujoco가 있는 인터프리터 |

`--log-commands`는 진단 전용이다. **정책은 여전히 30 Hz 샘플만 쓴다** — 관측 이력이
"정책 스텝 하나 간격의 두 샘플"로 정의돼 있기 때문. 읽기가 0.009 ms라 거의 공짜다.

`--parallel-mujoco`는 **상태 복사가 아니다.** MuJoCo가 자기 관측을 만들고 자기
추론을 돌리고 자기 물리를 적분한다. 공유되는 건 정책 파일과 타깃 명령뿐이라,
보고 있는 것은 **같은 명령에 두 플랜트가 얼마나 다르게 반응하는가**다.
기동에 ~0.5 s 걸려 위상은 대략만 맞는다.

---

# 3. `run_joint_replay` — 녹화한 관절 궤적 재생

Isaac 에서 뽑은 관절 목표를 **정책 없이 그대로 되감는다.** 시뮬과 실물이 같은 코드다.

```bash
# MuJoCo (wuji_mujoco)
python -m Deploy.run.run_joint_replay \
    --csv nrmk_isaaclab_wuji/logs/joint_records/joint_record_2026-08-20_00-34-12.csv \
    --viewer

# 실물 (wuji_hw)
python -m Deploy.run.run_joint_replay --csv <같은 파일> --backend real
```

**입력은 `play.py` 의 M 키가 남긴 `joint_record_*.csv`.** 그 파일의 `qt_*`(PD 목표)가
보낼 값이고 `q_*`(실측)는 참고다 — `qt - q` 가 파지 preload 다.

## 무엇을 재생할지

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--csv` | 경로 | **필수** | `joint_record_*.csv` |
| `--backend` | `mujoco` \| `real` | `mujoco` | 실행 대상 |
| `--segment` | int | 전체 | OPEN/CLOSE 구간 하나만. 인덱스는 `.meta.json` 의 `segments` |
| `--dry-run` | 플래그 | 꺼짐 | 검증하고 계획만 찍고 종료. **백엔드를 건드리지 않는다** |

## 속도 — 둘 중 하나만

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--speed` | float | — | 재생 배속. `1.0` 이 녹화된 30 Hz 원속도 |
| `--max-joint-speed` | float | `0.2` rad/s | 이 상한에서 배속을 역산 |

기본은 `--max-joint-speed 0.2`, 즉 `move_all.py` 와 같은 속도다. 원속도로 돌리려면
`--speed 1.0` 을 명시할 것.

## 앞뒤 구간

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--start-seconds` | float | `3.0` | 현재 자세 → 궤적 첫 목표까지 선형 이동 |
| `--settle-seconds` | float | `2.0` | 재생 전 첫 목표 유지 (필터가 따라올 시간) |
| `--hold-seconds` | float | `2.0` | 재생 후 마지막 목표 유지 |
| `--return-to-start` | 플래그 | 꺼짐 | 끝나고 시작 자세로 복귀한 뒤 disable |

## 공통

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--limit-margin` | float | `0.95` | 중심 ± f × 반범위. `1.0` 이 학습한 공간 |
| `--command-hz` | float | `90.0` | 명령 주기 |
| `--out` | 경로 | `None` | 실측 응답을 CSV 로 |
| `--gains` | `isaac_tuned` \| `vendor` | **`isaac_tuned`** | MuJoCo 전용. 기본이 학습 게인이다 |
| `--viewer` | 플래그 | 꺼짐 | MuJoCo 뷰어 |

## 스틱 고정 (`--hold-sticks`, MuJoCo 전용)

실물에서는 사람이 젓가락을 쥐고 있다가 손이 닫히면 놓는다. MuJoCo 에는 그 손이
없으므로, 매 물리 스텝 스틱 상태를 리셋 자세로 되돌려 같은 역할을 시킨다.

| 값 | 동작 |
|---|---|
| `glide` (기본) | 접근·정착 동안 고정하고 **재생 시작에 놓는다** |
| `always` | 끝까지 고정. 파지를 빼고 **관절 추종만** 잰다 |
| `never` | 처음부터 자유. 손이 이미 떨어지는 중인 물체를 잡으러 간다 |

매번 정확히 같은 자세로 되돌리므로 관통이 누적되지 않는다. 관통량은 리셋 시점
값(`palm↔stick2 −3.66 mm`)에 머무르고 그게 곧 파지 preload 다.

**실측 (`joint_record_2026-08-20_00-34-12.csv`, `--speed 1.0`)**

| 모드 | Stick1 | Stick2 | 읽는 법 |
|---|---|---|---|
| `glide` | 212 mm | 210 mm | 놓는 즉시 둘 다 낙하 |
| `always` | 0 mm | 0 mm | 고정이니 당연. 추종 오차만 유효 |
| `never` | 49 mm | 0 mm | 재생 전에 이미 떨어져 있어 0 |

**사람이 잡아주는 걸 완벽히 흉내내도 놓는 순간 MuJoCo 는 둘 다 떨어뜨린다.**
Isaac 은 같은 자세에서 Stick2 를 0.3 mm 로 붙들므로 이건 MuJoCo 쪽 문제이지만
2026-08-20 시점에 원인 미해결이다.

따라서 **지금 이 도구로 검증되는 것은 파지가 아니라 관절 추종**이다.
`--hold-sticks always` 로 파지를 빼고 `[TRACK]` 값을 보는 것이 현재 유효한 사용법이다
(실측 평균 0.51°, 최대 2.68° @ `finger4_joint2`).

## 실물 전용

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--read-only` | 플래그 | 꺼짐 | 연결·읽기만. **모터 enable 안 하고 아무것도 전송 안 함** |
| `--yes` | 플래그 | 꺼짐 | ENABLE 확인 프롬프트 생략 |
| `--lowpass-hz` | float | `0.5` | `filter.LowPass(cutoff_freq=)` |
| `--current-limit` | float (A) | `None` | 생략하면 현재값 유지 |
| `--max-step-rad` | float | `0.05` | 명령이 **측정 q** 로부터 이만큼 떨어지면 거부 |
| `--read-source` | `controller` \| `hand` | `controller` | |
| `--enable-upstream` / `--no-enable-upstream` | 플래그 | 켜짐 | |
| `--no-read-during-replay` | 플래그 | 꺼짐 | 붙이면 realtime 루프 안에서 엔코더를 **안 읽는다**. 코드 기본값은 `read_during_replay=True`(읽음)이고, 루프가 늦을 때 끄는 용도 — 루프 안 읽기가 벤더 예제에 없는 패턴이기 때문 |

## 순서

```bash
# 1. 파일이 제대로 읽히는지 (아무것도 안 건드림)
python -m Deploy.run.run_joint_replay --csv <파일> --dry-run

# 2. MuJoCo 로 눈으로
python -m Deploy.run.run_joint_replay --csv <파일> --viewer --out replay_mujoco.csv

# 3. 실물, 모터 끄고 계산만
python -m Deploy.run.run_joint_replay --csv <파일> --backend real --read-only

# 4. 실물 주행
python -m Deploy.run.run_joint_replay --csv <파일> --backend real \
    --return-to-start --out replay_real.csv
```

> **젓가락에 대해**: 이 재생은 **열린 루프**다. 녹화된 궤적은 정책이 매 스텝
> 보정하면서 만든 것인데 재생에는 그 보정이 없다. 2026-08-20 실측으로 pregrasp
> 목표를 고정하면 **Isaac 에서도 Stick1 이 0.65 초에 떨어진다** (Stick2 는 0.3mm 유지).
> Stick1 은 정적으로 물려 있는 게 아니라 정책이 붙잡고 있는 것이다.
> 재생은 고정이 아니라 변하는 궤적이라 조건이 다르지만, 어긋났을 때 되돌릴 수단이
> 없다는 점은 같다. **Stick2 부터 확인하는 편이 안전하다.**

---

# 4. `run_policy` — 젓가락 파지 (105D 계약)

```bash
python mj_grasp.py --inspect-contract
MUJOCO_GL=egl python mj_grasp.py --validate-aruco
```

### 모드 — 하나만 선택 (상호 배타, 전부 플래그)

| 모드 | 하는 일 |
|---|---|
| `--inspect-contract` | 105D/20D 계약, 세 한계 테이블, 정규화 요약 |
| `--inspect-model` | MJCF 관절·액추에이터 전체 + canonical 매핑 |
| `--inspect-joints` | 이름 기반 qpos/dof/actuator 매핑만 |
| `--validate-fk` | tip site vs URDF FK 대조 |
| `--test-joints` | 20관절 각각 `--small-delta` 단독 명령, 격리·방향·복원 |
| `--smoke-backend` | `--policy-steps` 만큼 닫힌 루프 스모크 |
| `--run-policy` | 정책 주행 |
| `--onnx-only` | 물리 없이 ONNX 입출력만 |
| `--view-scene` | 리셋 적용 후 viewer (**raw MJCF 직접 열지 말 것**) |
| `--view-camera` | 보정된 D435 광학 카메라에 고정된 viewer |
| `--inspect-camera` | 카메라 파라미터 (EGL 필요) |
| `--validate-aruco` | 렌더 + ArUco 검출 대조 (EGL 필요) |

### 공통 인자

| 인자 | 타입 | 기본값 |
|---|---|---|
| `--model` | 경로 | `assets/wuji_description/hand/body/mjcf/right_with_tip_sites.xml` |
| `--policy` | 경로 | `logs/rsl_rl/hand_final/2026-08-13_14-15-09/exported/policy.onnx` |
| `--mode` | `open` \| `close` | `open` |
| `--policy-steps` | int | `300` |
| `--print-interval` | int | `10` |
| `--small-delta` | float | `0.03` |
| `--joint-test-physics-steps` | int | `180` |
| `--controller-gains` | `vendor` \| `isaac_tuned` | **`vendor`** |
| `--physics-substeps` | int | `64` |
| `--integrator` | `euler` \| `rk4` \| `implicit` \| `implicitfast` | `implicitfast` |
| `--stick-provider` | `synthetic` \| `ground-truth` \| `aruco` | `synthetic` |
| `--joint-limit-tolerance` | float | `0.02` |
| `--viewer` / `--realtime` / `--debug-observation` | 플래그 | 꺼짐 |

`synthetic` 스틱 포즈는 **배선 검증용이지 성능 평가가 아니다.**

---

# 4-B. `run_mujoco_policy` — 실물 절차 그대로 MuJoCo 에서

`run_finger_reach_real.py` 와 **같은 단계 이름·같은 순서**로 돕니다. sim-to-sim 이
두 프로그램을 대조하는 게 아니라 **한 절차의 두 로그**를 읽는 게 되도록.

```
[RESET]    park 자세로 리셋 + 스틱을 Isaac 리셋 포즈에 배치
[GLIDE]    시작 자세까지 목표를 선형으로 walk, 스틱 PINNED
[SETTLE]   그 자세로 유지, 스틱 PINNED
[RELEASE]  핀 해제  ← 여기부터가 진짜 파지
[SEED]     정착한 상태에서 첫 관측 생성
[RUN]      30 Hz 정책
[REPORT]   릴리스 시점 대비 스틱 이동량
```

**핀은 "사람이 젓가락을 쥐고 있는 것"의 시뮬레이터 대역**입니다. 정해진 순간에
놓는 게 이 스크립트의 존재 이유고, 그 앞은 전부 준비 과정입니다.
`run_policy --run-policy` 는 텔레포트한 자세에서 이미 놓인 채로 시작하므로
이걸 볼 수 없었습니다.

```bash
python -m Deploy.run.run_mujoco_policy \
    --policy nrmk_isaaclab_wuji/logs/rsl_rl/hand_final/<run>/exported/policy.onnx \
    --seconds 10 --viewer --out mj_policy.csv
```

## 주요 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--policy` | 없음 | 105D ONNX. 생략하면 **zero-action 배선 확인**(파지 테스트 아님) |
| `--mode` | `close` | OPEN/CLOSE one-hot |
| `--switch-at` | 없음 | 이 시각(초)에 반대 모드로 전환 |
| `--seconds` | `10.0` | 기동 단계 이후 정책 구간 길이 |
| `--park-pose` | `reset` | 출발 자세. glide 가 이동할 거리를 만듦 |
| `--start-pose` | `pregrasp` | 도착 자세 |
| `--glide-seconds` | `3.0` | |
| `--settle-seconds` | `2.0` | |
| `--release` | `after-settle` | `never`=끝까지 고정(관절 추종만 측정) / `immediately`=옛 동작 |
| `--limit-margin` | `1.0` | 기동 구간 명령 여백. `1.0` 이 학습한 공간 |
| `--stick-provider` | `ground-truth` | `synthetic` 은 **상수**를 먹임 |
| `--controller-gains` | `isaac_tuned` | 정책이 학습한 Kp |
| `--joint-limit-tolerance` | `0.02` | 이만큼 한계를 넘으면 중단. 정책이 스톱을 향해 명령하면 서보가 조금 넘으므로 0 이 아님 |
| `--viewer` / `--realtime` | 꺼짐 | |
| `--out` | 없음 | 스텝별 CSV |

**실측 (`hand_final/2026-08-21_01-14-36`, CLOSE, 6 s)**

| `--release` | Stick1 | Stick2 | 읽는 법 |
|---|---|---|---|
| `after-settle` | 92.3 mm | 108.8 mm | 놓으면 둘 다 낙하 (기존 결론과 동일) |
| `never` | 0.00 mm | 0.00 mm | 핀이 정확히 동작함을 확인하는 대조군 |

릴리스 기준값이 CLAUDE.md 의 검증값과 일치하는지 매번 확인할 것:
`stick1 [25.07, 24.25, 96.96]`, `stick2 [35.60, 16.08, 73.37] mm`.

> 중단되어도 그때까지의 행은 CSV 로 남습니다. 관절 한계 초과로 죽는 건
> `--joint-limit-tolerance` 를 올려 확인할 것 — 정책이 스톱까지 명령하도록
> 학습됐기 때문입니다(`soft_joint_pos_limit_factor=1.0`).

---

# 5. 도구 (`Deploy.tools.*`)

| 도구 | 인자 | 기본값 |
|---|---|---|
| `make_validation_scenario` | `--output` | **필수** |
| | `--targets` | `4` |
| | `--duration` | `1.0` |
| | `--seed` | `0` |
| | `--max-reach-error-mm` | `3.0` |
| `compare_reach_logs` | `reference` (위치) | 필수 |
| | `others ...` (위치, 1개 이상) | 필수 |
| | `--tolerance-mrad` | `10.0` |
| `analyze_command_log` | `log` (위치) | 필수 — `*_90hz.csv` |
| | `--divider` | `3` (90/30) |
| `sweep_camera2_mount` | `--heights` / `--angles` / `--down-angle` / `--side-y` | 씬 상수 |
| `build_finger_reach_scene` | 없음 | 스틱 없는 reach MJCF 재생성 |
| `build_physical_testbed_scene` | 없음 | 파지 씬 테스트베드 |
| `build_aruco_visual_assets` | 없음 | ArUco 마커 텍스처 |

`make_validation_scenario`는 **실제 도달집합**에 대해 검사하므로 못 가는 점이
시나리오에 들어가지 않는다.
`compare_reach_logs`는 시각이 아니라 **policy step 서수**로 정렬하고, 양쪽에 다
있는 컬럼만 비교한다.

---

# 6. 실물 첫 기동 순서

```bash
python -m Deploy.run.run_finger_reach_real --read-only
python -m Deploy.run.run_finger_reach_real --measure-timing
python -m Deploy.run.run_finger_reach_real --hold-middle --seconds 3
python -m Deploy.run.run_finger_reach_real --test-middle-joint 1 --delta 0.03   # 1~4 각각
python -m Deploy.run.run_finger_reach_real --zero-policy --target 0.035 0.010 0.100 --seconds 3
python -m Deploy.run.run_finger_reach_real --policy PATH --target 0.035 0.010 0.100 --seconds 4
```

`--test-middle-joint`는 **네 번 다 `MATCHES`**가 나와야 canonical `finger3_jointN`과
하드웨어 J(N) 대응이 확인된다.

실물 + MuJoCo 동시 비교:

```bash
python -m Deploy.run.run_finger_reach_real \
    --policy PATH --scenario Deploy/validation_scenario.json \
    --lowpass-hz 2.0 --log-commands --return-to-start \
    --parallel-mujoco --csv finger_reach_real_N.csv
python -m Deploy.tools.compare_reach_logs \
    finger_reach_real_N.csv finger_reach_real_N_mujoco.csv
```

---

# 7. 자주 걸리는 것

- **`No module named 'Deploy'`** — 프로젝트 루트가 아닌 곳에서 실행했다
- **`--policy`가 `wuji_hw`에서 안 됨** — `onnxruntime` 별도 설치 필요
- **CSV 저장 위치** — `--csv`의 상대경로는 현재 작업 디렉터리 기준
- **두 CSV의 `time`이 안 맞음** — 정상이다. MuJoCo는 `(step+1)/30`, 실물은 실측
  경과시간. `compare_reach_logs`가 policy step 서수로 맞춘다
- **`--controller-gains`의 기본이 학습 게인이 아니다** — `vendor`가 기본이고
  학습 게인은 `isaac_tuned`. 옛 이름 `deploy`/`official`은 더 이상 안 받는다
- **`--viewer`만 주면 순식간에 끝난다** — `--realtime`을 같이






# 저장된 타겟값으로 이동
python -m Deploy.run.run_joint_replay \
    --csv nrmk_isaaclab_wuji/logs/joint_records/joint_record_2026-08-20_00-34-12.csv \
    --backend real \
    --limit-margin 0.95 \
    --start-seconds 9 \
    --max-step-rad 0.05 \
    --return-to-start
