# Claude handoff — 2026-08-17

> 기준 시각: 2026-08-17 17:22 KST. 학습 run은 계속 진행될 수 있으므로 아래의
> `latest` 숫자는 TensorBoard event를 다시 읽기 전까지의 snapshot이다.

## 0. 세션을 시작하면 먼저 읽을 문서

```text
AGENTS.md
CLAUDE.md
ACTIVITY_2026-08-17.md
ACTIVITY_2026-08-16.md
mujoco_deploy/README.md
mujoco_deploy/VALIDATION_REPORT.md
mujoco_deploy/ISAAC_POLICY_CONTRACT.md
```

가장 중요한 작업 규칙:

- 사용자가 명시적으로 요청하지 않으면 Isaac Sim, `train.py`, `play.py`, physics probe를
  직접 실행하지 않는다. 저장된 cfg/TensorBoard 판독과 정적 검사만 한다.
- 사용자가 진행 중인 두 학습 run을 종료·삭제·수정하지 않는다.
- observation/action/reward를 바꾸기 전에 현재 run의 `params/*.yaml`과 active source를
  반드시 대조한다.
- Isaac policy, MuJoCo, 향후 실물 Wuji 사이에서 raw 배열 순서가 같다고 가정하지 않고
  항상 physical joint name으로 mapping한다.

## 1. 현재 active `hand_real` 계약

### Observation: 105D

```text
[  0: 20] previous normalized joint q
[ 20: 40] current normalized joint q
[ 40: 55] current fingertip xyz in palm frame
[ 55: 62] previous Stick1 palm xyz+wxyz
[ 62: 69] current  Stick1 palm xyz+wxyz
[ 69: 76] previous Stick2 palm xyz+wxyz
[ 76: 83] current  Stick2 palm xyz+wxyz
[ 83:103] action_manager.action that produced the current state
[103:105] OPEN/CLOSE one-hot
```

- Stick quaternion은 local `+Y` 주위 90도 사각 단면 대칭 4개 중 pose reference와 가장
  가까운 branch를 고르고 `w>=0`으로 canonicalize한다.
- reward/success orientation은 directed local `+Y` shaft axis를 사용하므로 shaft roll을
  특정 값으로 강제하지 않는다.
- history는 policy step 단위 `[previous,current]`; reset에서는 두 슬롯이 같은 sample이다.

### Action: 20D current-joint residual

```text
raw ONNX action
  -> RSL-RL wrapper clip [-1,1]
  -> q_current + scale * clipped_action
  -> command-limit clamp
  -> joint position target
```

- Joint1/2 scale `0.10 rad`
- Joint3 scale `0.20 rad`
- Joint4 scale `0.15 rad`
- 다섯 Joint4 command lower override `0 rad`
- policy/physics cadence `30/120 Hz`이며 target 하나를 physics 4 step 유지한다.

### Task-local actuator state

- `hand_real_env_cfg.py`의 20-joint Kp/Kd table과 official URDF per-joint effort limit을 쓴다.
- 현재 Joint4 Kp(thumb/index/middle/ring/little)는 `1.0/1.3/1.3/1.3/1.15`다.
- Stick1 5 mm axial A/B 상수는 `0.0`; reset/reference/pivot 모두 원래 기하다.
- source:
  `nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/hand_grasp/hand_real_env_cfg.py`

## 2. `hand_real` disturbance curriculum — 현재 가장 중요한 상태

### 공통 source checkpoint

```text
run:   2026-08-16_21-14-45
model: model_3800.pt
force: 0.02~0.10 N
```

`model_3800.pt`의 직전 50-iteration 평균 기준:

```text
reward                  624.13
contact/final           5.69 / 5.31
mode/final mode         0.073 / 0.083
OPEN/CLOSE gap error    7.59 / 2.20 mm
lateral error           5.43 mm
6-contact recovery      98.2%
min contacts after      4.16
```

이 checkpoint에서 optimizer까지 이어받는 두 개의 true-resume branch를 비교 중이다.

### A. Strong disturbance `0.3~1.2 N`

```text
run: 2026-08-17_02-50-15
saved cfg: resume=true, force_range_n=(0.3,1.2), probability=1
latest event step: 9009
```

초반에는 크게 붕괴했다. global 6300 부근에서 6-contact recovery가 0이 됐고 6700에서
contact/final이 약 `3.34/3.30`이었다. 그러나 이후 실제 복구 추세가 나타났다.

최근 50-iteration trailing mean:

| global itr | reward | contact/final | mode/final | OPEN/CLOSE mm | lateral mm | recovery |
|---:|---:|---:|---:|---:|---:|---:|
| 7800 | 276.5 | 4.14 / 3.04 | .005 / .001 | 13.26 / 22.86 | 11.02 | 17.0% |
| 8000 | 500.0 | 5.80 / 5.61 | .010 / .016 | 12.29 / 11.96 | 7.56 | 89.9% |
| 8200 | 511.4 | 5.90 / 5.76 | .015 / .035 | 14.95 / 2.57 | 8.10 | 97.7% |
| 8400 | 492.5 | 5.83 / 5.71 | .043 / .131 | 15.11 / 2.06 | 7.24 | 96.8% |
| 8600 | 580.2 | 5.82 / 5.53 | .115 / .052 | 15.06 / 4.91 | 5.19 | 92.5% |
| 8800 | 454.2 | 5.58 / 5.21 | .050 / .043 | 13.32 / 7.60 | 6.40 | 84.0% |
| 9000 | 488.3 | 5.76 / 5.67 | .024 / .026 | 12.15 / 8.57 | 5.00 | 91.7% |
| 9009 | 486.8 | 5.75 / 5.63 | .020 / .019 | 11.95 / 9.45 | 5.37 | 91.4% |

판독:

- “강한 외란에 완전히 무너졌다”는 현재는 틀리다. 접촉과 recovery는 명백히 복구됐다.
- 그러나 OPEN gap이 계속 약 `12~15 mm`이고 mode geometry valid도 낮다. 접촉 복구와
  OPEN/CLOSE 기하 복구는 별개다.
- contact/recovery만 보면 8200~8400, 평균 mode/lateral까지 보면 8600, 최신 균형은
  9000 부근이 후보지만 play 확인 전 단일 best로 확정하지 않는다.

### B. Medium disturbance `0.1~0.6 N` — 공정 resume

```text
run: 2026-08-17_11-04-35
saved cfg:
  resume=true
  load_run=2026-08-16_21-14-45
  load_checkpoint=model_3800.pt
  force_range_n=(0.1,0.6)
  probability=1
latest event step: 6010
latest checkpoint observed: model_6000.pt
```

최근 50-iteration trailing mean:

| global itr | reward | contact/final | mode/final | OPEN/CLOSE mm | lateral mm | recovery |
|---:|---:|---:|---:|---:|---:|---:|
| 3900 | 508.4 | 5.51 / 4.87 | .005 / .003 | 6.74 / 10.05 | 9.04 | 89.5% |
| 4300 | 472.5 | 5.34 / 4.68 | .028 / .027 | 11.25 / 17.46 | 5.92 | 69.2% |
| 4800 | 283.5 | 4.69 / 3.55 | .008 / .000 | 21.74 / 48.20 | 26.05 | 5.7% |
| 5300 | 537.8 | 4.99 / 4.97 | .040 / .056 | 6.28 / 7.72 | 4.28 | 94.9% |
| 5800 | 565.1 | 5.24 / 4.05 | .049 / .016 | 13.82 / 11.96 | 6.66 | 60.9% |
| 6000 | 740.2 | 5.91 / 5.37 | .094 / .085 | 15.51 / 2.49 | 2.86 | 99.1% |
| 6010 | 733.8 | 5.90 / 5.28 | .091 / .077 | 15.50 / 2.69 | 3.07 | 99.0% |

판독:

- 4800의 큰 붕괴 뒤 5300과 6000에서 복구했다. PPO 추세는 단조가 아니다.
- 6000은 contact/recovery/CLOSE/lateral이 매우 좋지만 OPEN gap `15.5 mm`라 CLOSE 편향이다.
- 같은 global 6000 기준으로는 strong branch보다 훨씬 빨리 복구했지만, 최종 목표는
  “접촉만 유지”가 아니라 OPEN/CLOSE 둘 다 맞추는 것이므로 아직 확정 best는 아니다.

### 폐기된 혼동 방지용 run

`2026-08-17_02-51-28`은 `0.1~0.6 N`이었지만 `resume:false`, local iteration 0부터
시작한 branch여서 위 두 true-resume run과 공정 비교 대상이 아니었다. 현재 폴더도 삭제된
상태다. 그 run의 숫자를 새 medium branch와 섞지 않는다.

### 동일 medium branch 재현 CLI

```bash
cd /home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji
python scripts/rsl_rl/train.py \
  --task hand_real \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000 \
  --resume \
  --load_run 2026-08-16_21-14-45 \
  --checkpoint model_3800.pt \
  'env.events.stick_disturbance.params.force_range_n=[0.1,0.6]'
```

## 3. MuJoCo sim-to-sim — 완료된 범위

- `mujoco_deploy/`에 fixed-base official Wuji Hand RIGHT를 구성했다. Indy7은 없다.
- official description commit은 `06e5f14cdd1d5fad0a666ca463a668bf609f9534`로 pin했다.
- 20 policy joint ↔ MuJoCo qpos/dof/actuator를 이름 기반 strict mapping한다.
- physical/observation/command limit을 분리했고 Joint4 command lower는 `0 rad`다.
- common 105D observation builder, ONNX wrapper, Isaac-equivalent residual action decoder,
  30/120 Hz scheduler를 MuJoCo backend와 분리했다.
- fixed-hand closed loop 외에 두 dynamic stick, approximate testbed scene, D435/ArUco pose
  provider, stale-perception safe-stop까지 구성했다.
- `mujoco_deploy/VALIDATION_REPORT.md` 작성 시점 기준 common+MuJoCo+rendered-vision test
  21개, 20-joint single-command test, scheduler smoke가 통과했다.
- `RealWujiBackend`는 SDK/handedness/joint order·sign·zero/watchdog/firmware controller 의미를
  실측하기 전까지 의도적으로 construction을 거부한다. 아직 실물 명령은 보내지 않는다.

주의:

- 과거 101D ONNX와 현재 105D policy를 섞지 않는다.
- 최종 선택한 `hand_real`/`hand_final` checkpoint의 새 105D ONNX로 input/output shape와
  slice를 다시 검증해야 한다.
- MuJoCo actuator dynamics가 PhysX/실물과 완전히 match됐다는 의미는 아니다. 현재 완료 범위는
  mapping, observation/action plumbing, timing, position-target interface 검증이다.

## 4. ArUco Stick1 tracking — 완료된 범위

외부 경로:

```text
/home/lsc/FoundationPose/run_aruco_test.py
/home/lsc/FoundationPose/calibrate_aruco_pair.py
/home/lsc/FoundationPose/run_aruco_stick_agreement.py
/home/lsc/FoundationPose/run_stick1_dual.py
```

Canonical frame:

```text
Stick origin = 180 mm geometric center
Stick +Y     = tail -> tip
Stick +Z     = Marker0 outward normal
Stick +X     = right-handed completion
p_M0_S       = [0, +0.090, -0.0035] m
R_M0_S       = identity
T_M1_S       = inverse(T_M0_M1) @ T_M0_S
```

따라서 ID1을 이상적인 45도로 손 코딩하지 않고 실제 pair calibration 오차까지 반영한다.

Calibration snapshot:

```text
file: /home/lsc/FoundationPose/aruco_pair_calibration_stick1/calibration_latest.json
samples/inliers: 250/132
p_M0_M1: [3.903, 0.167, -8.899] mm
relative SO(3) magnitude: 131.668 deg
inlier residual mean: 2.467 mm / 1.919 deg
```

596-frame marker별 Stick pose agreement:

```text
position: median 4.84 mm, mean 8.87 mm, p90 21.78 mm, max 41.08 mm
rotation: median 1.89 deg, mean 2.10 deg, p90 3.73 deg, max 6.08 deg
```

회전은 비교적 합의하지만 position long-tail이 아직 크다. `run_stick1_dual.py`는 single-marker
IPPE candidates, calibrated pair consistency, dual 8-corner PnP, reject 후 history/filter reset,
EMA/SLERP를 구현했다. 생산 투입 완료 상태는 아니다.

## 5. 다음 작업 우선순위

1. 두 disturbance branch의 최신 event를 같은 trailing window로 계속 비교한다.
2. 후보 checkpoint는 reward/contact만으로 고르지 말고 OPEN gap, CLOSE gap, lateral,
   mode geometry, recovery를 함께 보고 `play`는 사용자가 실행한다.
3. 강한 branch가 OPEN 기하까지 회복하는지 확인한다. 접촉만 복구된 checkpoint를 최종으로
   오인하지 않는다.
4. 선택한 105D checkpoint를 ONNX로 export한 뒤 MuJoCo에서 exact `105 -> 20` contract,
   joint-name mapping, clip/residual/clamp, 30/120 Hz를 다시 검증한다.
5. ArUco position long-tail을 corner/PnP branch, 가림, 움직임 구간으로 분리하고
   Camera→Palm extrinsic을 측정 파일로 고정한다.
6. 실물 Wuji SDK contract가 확정되기 전에는 `RealWujiBackend` 안전 차단을 풀지 않는다.

## 6. MuJoCo 추가 Camera2 결정 (같은 날 후속)

사용자 설치 요구 때문에 Camera2 하강각은 **0 deg(완전 수평)**로 고정한다. Camera2는
Hand/Palm `+Y` 측에서 `-Base Y`를 보고, full-workspace tracking이 아니라 reset에서 각
stick의 tail marker ID0/ID2를 확실히 확인하는 보조 카메라다.

MuJoCo reset marker center 추출값:

```text
ID0 Base xyz: [0.121009, 0.061799, 0.072649] m
ID2 Base xyz: [0.129380, 0.049104, 0.047509] m
```

현재 simulation candidate optical center는 Base
`[X,Y,Z]=[0.125,0.200,0.060] m`다. 12.5 cm 높이는 ID0/ID2 높이의 반올림 midpoint이며,
reset/Palm-Z 0 deg에서 Camera2 단독 두 marker 동시 검출은 PASS했다. 단 수평 Camera2 단독
5 deg sweep은 `7/19`이므로 full `-90..0 deg` coverage라고 주장하면 안 된다.

비교 실험에서는 60 deg 하향/40 cm가 1 deg sweep `91/91`이었지만 물리 설치가 어려워
선택하지 않았다. 관련 상수와 기본 candidate camera pose는 `scene_contract.py`와
`right_with_tip_sites.xml`, sweep 도구는 `mujoco_deploy/tools/sweep_camera2_mount.py`에 있다.
정확한 Real Camera2 intrinsics/extrinsics는 장착 후 별도 calibration해야 한다.
