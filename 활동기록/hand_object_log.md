# hand_object 태스크 로그

`hand_move` 정책을 fine-tuning해서 **두 젓가락으로 1cm 큐브를 실제 접촉력으로 집고,
손을 움직여도 안 떨어뜨리게** 만드는 태스크. 큐리에이션 로그(요약·현재 상태).
날짜별 상세는 `ACTIVITY_2026-08-*.md`.

---

## 현재 상태 (2026-08-07)

**처음으로 큐브를 잡고 버틴다.** `holding` 0.000 → **0.4375**, 에피소드의 **48.6%**가
9초를 완주(`time_out`). run `2026-08-07_15-32-42`, it 149.

```
[x] 환경·보상·종료·메트릭·calibration 도구 구현
[x] 정적 검증 (AST 37항목) / 런타임 smoke test 20/20
[x] calibration — 목표 root pose / 큐브 / 기둥 확정
[x] 이동 궤적 2단계화 (정렬 -> 수직 하강)
[x] 기둥 제거를 조건 트리거로 (파지 OR 데드라인, latch)
[x] stick2_dropped 임계값 0.40 -> 0.20  <- 이게 전 학습을 막고 있었음
[x] hold 발생 -> force_saturation 실측 가능해짐
[x] hold 게이트 메트릭 3종 신설 -> 접근 구간 오염 11배 확인
[x] cube_hold 를 죽이던 각속도 항 제거 (노이즈였음, 지수의 87%)
[ ] 각속도 제거 효과 확인  <- cube_hold 0.084 -> 139 점 예상
[ ] 각속도가 정말 노이즈인지 확정  <- 순 회전각 메트릭 필요, 지금은 추론
[ ] force_saturation / 거리 기준점 조정 (보류, 근거는 확보)
```

### 2026-08-07 결정적 발견 — 종료 조건이 병목이었다

`HandObjectTerminationsCfg`가 `hand_move`의 `stick2_dropped`(`minimum_height=0.40`)를
그대로 상속했다. `hand_move`는 손이 스폰 높이에 머무르므로 맞는 값이지만,
`hand_object`는 큐브를 잡으러 **의도적으로 z = 0.365까지 하강**한다.

하강이 끝나는 **정확히 2.00초**에 스틱 루트가 0.40을 지나 전 에피소드가 오판 종료됐고
(`Episode_Termination/stick2_dropped` = 1.0000, `mean_episode_length` = 59.96 steps),
CLOSE가 2.5초 시작이라 **압착을 한 번도 시도해보지 못했다.**

큐브 보상 3종이 전부 0이던 것, `force2_score`가 계속 0이던 것, `retract_by_grasp`가
0이던 것이 전부 이 하나로 설명된다. **"큐브가 두 팁 사이에 없다 / 기하 보정을 다시
해야 한다"던 진단은 오진이었다.**

0.20 m로 오버라이드 후 (`hand_move`는 0.40 유지):

```
                       고치기 전      고친 후 (it 149)
stick2_dropped          1.0000         0.0000
mean_episode_length     59.96          229.58  (7.65s / 9.0s)
holding                 0.0000         0.4375
force2_score            0.0000         0.5194   <- Stick2 최초 접촉
bilateral_force [N]     0.0000         0.0681   (max 0.2161)
```

### 조건 트리거 실동작 확인
```
retract_by_grasp        0.9095    <- 90.9% 가 파지 조건으로 발동 (나머지는 데드라인)
retract_trigger_time_s  3.50 s    <- 데드라인 5.5초보다 2초 빠름
```
CLOSE 시작(2.5초) 후 **1초 만에** 파지 조건(양쪽 inward >= 0.025 N, 연속 5스텝)을
만족한다. `조건 OR 데드라인 + latch` 구조가 의도대로 작동함이 처음 확인됐다.

### 보정값
```python
HAND_OBJECT_TARGET_ROOT_POS_E = (0.0780, -0.0470, 0.3700)   # 스폰 대비 15.9cm
HAND_OBJECT_TARGET_EULER_RAD  = (0.0934, 0.3472, -1.0331)   # deg (+5.35, +19.89, -59.19)
HAND_OBJECT_CUBE_POS_E        = (0.1468, 0.0865, 0.4114)    # = 그 자세의 tip 중점
HAND_OBJECT_SUPPORT_POS_E     = (0.1468, 0.0865, 0.2030)    # 기둥 높이 406mm
HAND_OBJECT_FORCE_SATURATION_N = 0.05   # PROVISIONAL — 미확정
```

손 스폰과 큐브가 둘 다 고정이라 **보정 대상은 "손이 날아갈 목표 root pose" 하나**이고,
큐브는 그 자세의 tip 중점으로 따라온다.

smoke test 로 확정된 것: hand_move 체크포인트 로드(actor 103->128->128->20),
센서 shape `(N,1,1,3)`, **force 부호 규약 실측 확인**(압착 642회 전부 양수, 음수 0),
초기 충돌 없음(손 링크 57.7mm), 2단계 궤적 이탈 0.000mm, retract 스텝당 1.333mm,
drop 종료는 항상 retract 완료 이후, env별 독립, NaN 없음.
메트릭 `Metrics/hand_object/*` 15종 × 4집계 = 60키 실동작 확인.

**force_saturation 이 미확정인 이유**: 압착 자체는 `min(f1,f2)` 최대 **0.167 N** 까지 나오는데,
"기둥 내려간 뒤 버티는" 힘은 hold 가 생겨야 잰다. hand_move 정책은 33/33 전부 놓친다.
fine-tuning 후 `Metrics/hand_object_max/bilateral_force` 로 확정할 것.

## 설계 요약

### 정책 인터페이스는 동결
obs **103D**, action **20D** — `hand_move`와 완전 동일. 그래서 hand_move 체크포인트가
shape 변경 없이 로드됨. **큐브·접촉력·phase는 actor가 못 봄** (보상·종료·메트릭 전용).
"기존 proprioceptive 상태만으로 안 보이는 물체를 잡고 있을 수 있나"를 묻는 실험이라 의도적.

### 손·스틱은 안 건드림
spawn, `pose_005` 리셋, 보상 세트 전부 `hand_move` 상속. 기하가 안 맞으면
**큐브·기둥·yaw를 움직이지 검증된 파지를 건드리지 않는다.**

### task ID 하나
`hand_object` 하나로 train/play/calibration 전부. `hand_move`와 같은 방식 —
세 command(`open_close`·`root_orientation`·`support`)가 전부 manual override를 갖고 있어
`play.py --manual_root`가 셋을 동시에 뒤집음.

```bash
# 학습
python scripts/rsl_rl/train.py --task hand_object --headless --num_envs 4096 \
    --init_checkpoint logs/rsl_rl/hand_move/<run>/model_<N>.pt

# 수동 play / calibration
python scripts/rsl_rl/play.py --task hand_object --num_envs 1 --manual_root \
    --load_run <run> env.require_calibration=false env.episode_length_s=300.0
```

### 에피소드 (7초)
```
0.0-0.5  OPEN            0.5-2.0  OPEN + yaw SLERP     2.0-2.5  settle
2.5-3.5  CLOSE 파지      3.5-4.0  기둥 20mm 하강       4.0-7.0  hold ← 평가
```

### 접촉력
스틱마다 ContactSensor 1개, 각각 **Cube만 필터**. closing axis
(`normalize(tip1-tip2)`, Stick2→Stick1)에 투영해 스틱별 inward force를 구하고
**`min(score1, score2)`** 로 bilateral score. mean/sum이 아니라 min인 이유:
한쪽만 세게 눌러도 만점이 나오면 안 됨. `force_saturation`에서 포화 → 과압착 gradient 없음.

부호는 추정이 아니라 isaacsim 테스트에서 유도: `force_matrix_w` = 필터가 센서에 가하는 힘
→ `STICK1_SIGN=+1`, `STICK2_SIGN=-1`. calibration 출력에 raw 벡터와 두 dot을 다 찍어 실측 확인 가능.

### 보상 (전부 provisional)
```
bilateral_cube_force      w=100  CLOSE에서만        450점
cube_relative_stability   w= 50  retract 이후       150점
cube_hold                 w=200  close×retract×force×stability   600점  ← 목표 그 자체
```
`close_tip_gap`은 target 0 그대로, force에 따른 fade-out **없음**.
drop penalty·over-force penalty **없음** (drop은 종료만).

---

## 미측정 값 (파일: `hand_object_mdp.py` 상단)

```python
HAND_OBJECT_TARGET_YAW_RAD     = None   # UNCALIBRATED
HAND_OBJECT_CUBE_POS_E         = None   # UNCALIBRATED
HAND_OBJECT_SUPPORT_POS_E      = None   # UNCALIBRATED
HAND_OBJECT_FORCE_SATURATION_N = 0.05   # PROVISIONAL
```
미보정으로 train을 걸면 `HandObjectRootOrientationCommand.__init__`에서 즉시 예외.

### calibration 키
| 키 | 동작 |
|---|---|
| `A` / `D` | yaw ± (palm-local z, 스크립트와 같은 축) |
| `P` | calibration 블록 출력 |
| `V` | 기둥 하강 |
| `1` / `2` | OPEN / CLOSE |
| `I/K/J/L/U/O` | root 이동 · `Q/E/W/S` roll·pitch · `H/G/R/C` 동기화/복원/리셋/큐브리셋 |

출력의 `local-z signed` → `TARGET_YAW_RAD`, `tip midpoint` → `CUBE_POS_E`,
`min(inward1,inward2)`의 유지 vs 낙하 구간 비교 → `FORCE_SATURATION_N`.

### yaw 탐색 시작점 (측정된 기하)
reset의 압착축은 `(0.82, -0.02, -0.57)`로 **수평에서 34.7° 기울어** 있어 큐브가 축을 타고
미끄러지기 쉬움. **local-z ±90°에서 압착축이 거의 수평**이 되어 마찰로 중력을 받치기 유리
(tip 중점은 151mm 이동). ±90° 근방부터 볼 것.

주의: `hand_move`의 "yaw"(local z)는 **월드 수직축이 아니라 월드 +x축** 회전이다.

---

## 파일

| 파일 | 역할 |
|---|---|
| `.../hand_grasp/hand_object_mdp.py` | 스케줄·보정상수·3 command·force 헬퍼·3 보상·drop 종료·calibration 리포트 |
| `.../hand_grasp/hand_object_env_cfg.py` | Scene/Commands/Rewards/Terminations/Event/Env cfg |
| `.../hand_grasp/hand_move_manual_control.py` | `V`·`P` 키 (hand_move와 공용) |
| `.../hand_grasp/learning/rsl_rl_cfg.py` | `HandObjectPPORunnerCfg` (LR 3e-4) |
| `isaac_neuromeka/env/managers.py` | `hand_object` 메트릭 15종 |
| `scripts/rsl_rl/train.py` | `--init_checkpoint` |

---

## 보류 중 (근거는 확보, 적용은 안 함 — 2026-08-07)

### (a) `HAND_OBJECT_FORCE_SATURATION_N` 0.05 → 0.10~0.15 N

`holding`이 생기면서 처음으로 실측이 됐다.
```
Metrics/hand_object_max/bilateral_force   0.2161 N
Metrics/hand_object/bilateral_force       0.0681 N (평균)
```
설정값 0.05 N을 상시 초과한다. 포화가 자주 걸려 `bilateral_force_score`가 0/1에 가까운
거친 신호가 되고 있다. 과압착 gradient 제거라는 설계 의도에는 부합하나 해상도가 낮다.

참고 — 이론 하한: 큐브 3 g → 무게 0.0294 N, μ=1.0에서 양쪽 각각 **0.015 N** 필요
(압착축 수평 가정, 실제는 34.7° 기울어 있어 더 필요). 수동 calibration 최대치 0.167 N.

### (b) 거리 기준점 0 → 6 mm

`stability = exp(-distance/σ)` 는 **d = 0 에서 최대**라 "이상적인 거리 = 0"을 요구한다.
그런데 스틱 반두께 3.5 + 큐브 반폭 2.5 = **6.0 mm 가 물리적 하한**이라, 완벽한 파지도
`exp(-0.6) = 0.55` 가 천장이다. `TIP_AXIAL_OFFSET_STICK2` 를 일부러 0 이 아닌 실측값으로
둔 것과 같은 상황.

**보류 사유**: 효과가 일률적으로 `exp(0.6) = 1.822배`뿐이라(0.084 → 0.153) 아래 각속도
문제를 안 고치면 무의미했다. 각속도 제거 후에는 거리 항이 `exp(-0.807) = 0.446` 으로
의미 있는 판별자가 되므로 급하지 않다. 실측 8.07 mm 확보.

기준값은 **실측 8.07 이 아니라 기하 이상값 6.0** 을 써야 한다 — 8.07 로 맞추면 현재
자세를 정답으로 고정해 gradient 가 0 이 된다. 6.0 이면 23% 여지가 남고, 그걸 메우려면
roll 을 면-정면으로 돌려야 해서 지금 아무도 안 보는 자유도에 간접 압력이 생긴다.

---

## 2026-08-07 두 번째 발견 — `cube_hold` 를 죽인 건 각속도 노이즈였다

### 게이트 메트릭으로 확인
`cube_distance` / `*_speed_rel` 은 게이트가 없어 접근·회전 구간을 포함한다. `holding`
으로 게이트한 3종을 신설해 재보니:

```
                     게이트 없음    파지 중(게이트)
hold_distance          93.3 mm        8.07 mm     <- 11배 오염돼 있었음
hold_linear_speed      0.085 m/s      0.040 m/s
hold_angular_speed     10.25 rad/s   10.43 rad/s  <- 게이트해도 그대로

stability 지수:  거리 0.81 + 선속도 0.80 + 각속도 10.43 = 12.04  ->  6e-6
```

**각속도가 전체 지수의 87%.** 거리도 선속도도 정상 범위였다.

### 각속도 10.43 rad/s 는 회전이 아니라 솔버 노이즈
- 큐브 5 mm / 3 g → `I = 1.25e-8 kg·m²`. 0.07 N 이 2.5 mm 편심이면 `α = 14,000 rad/s²`,
  **물리 스텝 한 번에 117 rad/s**. 10 rad/s 는 노이즈 바닥.
- 지표가 `norm(ω_cube − ω_stick2)` 이라 **알짜 회전 0 인 진동도 평균 10 으로 찍힌다.**
- `hold_distance` 편차가 0.5 mm 뿐. 초당 1.7 바퀴 돌면 정사각 단면이 면↔모서리로 바뀌며
  팁 간격이 41% 출렁여야 한다.

### 조치
`cube_hold` 와 `cube_relative_stability` **둘 다** 각속도 항 제거. `linear_speed` 유지.

곱셈 항에 **정책이 못 줄이는 양**을 넣으면 σ 오설정보다 나쁘다 — 다른 게 아무리 좋아져도
목표가 0 에 붙는다. 한쪽만 빼면 "큐브가 안정적이다"가 두 보상에서 다른 뜻이 되므로 같이 뺐다.

weight·거리 기준점·σ 는 전부 그대로. 예상: `cube_hold` 0.084 → **139 점**,
`cube_relative_stability` 0.021 → **39 점**. `holding` 이 0.9 로 오르면 323 점.

## 다음 할 일

1. 각속도 제거 효과 확인 — `cube_hold` 0.084 → 100 근처로 뛰는지. 안 뛰면 계산 어딘가가
   틀린 것이니 `hold_distance` / `hold_linear_speed` 를 다시 볼 것
2. 각속도가 정말 노이즈인지 확정 — **순 회전각**(파지 시작 자세 대비 누적) 메트릭 필요.
   지금은 관성모멘트 계산·norm 특성·거리 안정성 셋으로 추론했을 뿐 직접 관측이 아니다
3. 거리 기준점 0 → 6 mm (위 (b)), 효과 1.8배라 급하지 않음
4. `force_saturation` 0.05 → 0.10~0.15 (위 (a))
5. roll 무제약 — 면-정면에서 20.6° 이탈. 모서리로 물리면 압착축 회전이 자유로워 위
   각속도 노이즈의 물리적 배경이 된다
6. `raw_forceN_magnitude` vs `inward_forceN` 비율 — 0.8 이하면 tip-region gate 검토
7. hand_move 3단 커리큘럼 완성 후 그 체크포인트로 재출발

## 후속 검토 대상 (실제 증상이 보이면)
- CLOSE action 포화 / 큐브 튕김 / Stick2 붕괴 → over-force penalty
- 스틱 shaft로 눌러 force farming → tip-region contact gate
- 일부러 빨리 drop해서 종료 → 작은 one-shot drop penalty
- 학습 성공 후: 큐브 위치 ±0.5mm → ±1mm → 크기 9~11mm → mass/friction ±10%
