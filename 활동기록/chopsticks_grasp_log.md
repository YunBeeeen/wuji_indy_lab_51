# chopsticks_grasp_log.md — Indy-Wuji-Chopsticks-Grasp / functional_grasp

> Task별 큐레이션 로그. **원본(수정 금지)**: `nrmk_isaaclab_wuji/worklog.md`(2026-07-13~07-30).
> 07-15~24 운반·keypoint·orientation 기계는 box_transport와 공유 → box_transport_log.md도 참조.

## 개요
- **task**: `Indy-Wuji-Chopsticks-Grasp`(직육면체 = 젓가락/스틱 프록시). 스틱 파지 → world goal pose로 운반·자세맞춤.
  파생: `functional_grasp`(one-stick A1 분리 태스크).
- **상태**: **활성**. 2cm에서 파지+lift 성공(2026-07-30), 현재 **1cm 전환** 진행 중.
- **핵심 산출물**: cube의 cage 체계 + box의 transport/keypoint/orientation 이식, tip/tail 4-대칭 자세,
  잔차(residual) 액션, sim-to-real obs 슬림(84D), 캠핑 탈출용 유지-연금 삭감, region-free 파지.

## 마일스톤 타임라인
- **07-13 생성**: 직육면체 = 젓가락 프록시로 신규 task. cube의 cage 구조 상속.
- **07-15~17**: 로드맵 확정. transport-φ(best-so-far 위치) 설계. run 파라미터 검증, 일반화 순서.
- **07-18**: B안 배선/보류, orientation v1 구현→기각, 크기 확장, 2슬롯 라운드 → **keypoint transport(v1.1) 전환**.
- **07-19**: WRAP 라운드(A 청신호, B 파지 실패 수정), 스틱 보류, 슬롯 A ori 패키지.
- **07-20**: **one-stick A1 functional grasp 분리 태스크** 구현. 젓가락 RL 전체 단계 확정.
  semantic anchor/contact/stability 후보. index/thumb surface-region 적용.
- **07-21~22**: chopstick lift 보상 ↔ Box-Transport 상호 이식. orientation(c안)+goal_proximity,
  **tip/tail 4-대칭** 자세오차, 말단 지수 커널, 페널티 3종, action_second_rate 활성, keypoint A/B.
- **07-23**: middle grip region 추가(사용자). **잔차 액션 전환 + 약지·새끼 커플링 해제 + 가중치 전면 재조정**.
- **07-24**: 밤샘 판독, 재설계·재샘플·success 분리, 재샘플 7D key 버그수정, palm-down 철회.
- **07-25**: A/B 결론 = **keypoint 채택**. Phase 1 wrap(주먹) 파지 적용 → **정정: keypoint 재평가, wrap을 box로 이동**.
- **07-29**: 1cm maintained gate 결합 배선. **obs 경량화**(grip_error 등 sim-to-real oracle 제거).
- **07-30 (이 세션)**: 아래 상세.

## 2026-07-30 세션 상세 (활성)
1. **obs 슬림 검증 준비**: 07-25 2cm 베이스라인 복원 → **불완전 복원 발견**(스틱 스폰 정의는
   `chopsticks_grasp_env_cfg.py`에 별도라 1cm 잔존, 물체↔보상 cage 불일치) → 물체 세트 복원.
   **손가락 residual scale 0.3→0.1 회귀** 발견·복구(env_cfg.py).
2. **캠핑 진단**: play로 파지 자세는 OK인데 lift 안 함 → `finger_cage_hold`·`goal_proximity`가 **위치무관**
   지급이라 "잡고 제자리"가 안정 최적점(캠핑). → **cage_hold 15→5, goal_proximity 75→20** 삭감.
3. **성공**: 2cm에서 grasp+lift 성공(run 13-06-26). **grip obs 없이 됨 → obs 슬림 파지+lift 검증**.
4. **1cm 전환**: STICK_SIZE 1cm + 얇은 물리(contact 0.001·rest 0.0·depenet 0.2·solver_vel 4) +
   STICK_HALF_EXTENT 0.005(물체↔보상 일치).
5. **1cm 튜닝(변수 1개씩)**: cage 치팅(손가락 뭉쳐 비비면 5mm 구가 얇은 스틱에 먹힘) → sphere 0.005→0.002
   (게이트 사망) → **0.004**. grip < hold 예산이라 뭉치기 선택 → **grip 40→150**(ceiling이 hold 이기게).
   현재 1cm 파지 형성 여부 관찰 중.

## 핵심 교훈·함정 (이 task 고유)
- **캠핑 = 위치무관 유지 연금**: cage_hold·goal_proximity가 위치와 무관하게 지급되면 "잡고 앉기"가 최적점.
  파지 확인 후 그 연금을 삭감하면 lift가 유일 수입이 됨.
- **cage 게이트는 "닿으면 포화"**(depth_max에서 saturate, 크러시 방지) → 조임 유도 불가. 조임은 grip 보상/성공으로.
- **grip ceiling > hold ceiling** 이어야 제대로 배치. 아니면 뭉쳐서 cheatable hold만 먹음.
- **sphere_radius**: 크면 얇은 스틱에서 치팅(뭉치기), 작으면 게이트 사망. grip 강화가 치팅을 막아줌.
- **물체 스폰(STICK_SIZE, 물리)은 `chopsticks_grasp_env_cfg.py`, 보상 cage(STICK_HALF_EXTENT)는
  `chopstick_mdp_cfg.py`** — 크기 바꿀 땐 **반드시 동반**(예전 불일치 버그).
- **보상 weight는 항 간 직접 비교 금지** — gate형 2.4W ≈ progress형 W/30 ≈ telescoping Δ×gate×W/30. 예산(점)으로 환산.
- **grip 보상은 privileged(sim-side)** — 실물엔 안 실리니 obs만 sim-to-real 신경 쓰면 됨.
- 잔차 액션: target=현재각+action×scale. 손가락 scale이 유지토크(≈kp×action×scale) 좌우. play엔 학습 오버라이드 복붙.

## 소스 포인터
- `nrmk_isaaclab_wuji/worklog.md`: 2026-07-13~07-30 (검색 "chopstick", "functional_grasp").
- 코드: `tasks/manipulation/functional_grasp/chopstick_mdp_cfg.py`, `chopsticks_grasp_env_cfg.py`,
  `indy_wuji/env_cfg.py`, `mdp/rewards.py`, `mdp/target_grasp.py`.
- 백업: `chopstick_mdp_cfg.py.*backup*`, `maintained_1cm_newobs_2026-07-29` 등.

## 2026-07-30 (추가) — 1cm 포기, 실타깃 전환
- **1cm 단일 프록시 = 인위적으로 어려움**(치팅/게이트 사망 whack-a-mole). 실제 타깃(7mm 젓가락 2개 다발)로 전환.
- **1.4cm(단면 14×14 정사각) 프록시로 2cm 성공 config 리사이즈 → lift 성공(사용자 판정 성공)**.
  1cm 지옥 우회 확인. grip 40·sphere 0.005·cage_hold 5·goal_prox 20·lift 0.05·mass 0.02·2cm물리.
- **통찰**: 1.4cm 정사각은 두께도 14mm라 실제와 다름. 실물은 **14 wide × 7 thick**(각 스틱 7mm). 두께 7mm가
  1cm보다 얇아 진짜 난관. 다음: ① 단일 사각프록시 14×7(두께 7mm 난이도만 격리) 또는 ② 스틱 2개(각 7mm) 실구조.

## 2026-07-31 — 실타깃(7mm) 난관 + "손바닥에 담기" goal 재설계
### 실험 경과
- 1cm 단일 프록시 whack-a-mole → **실타깃 전환**: 1.4cm(14×14 정사각)로 2cm config 리사이즈 → lift 성공.
- 통찰: 1.4cm 정사각은 두께 14mm라 실제와 다름. 실물 = **14(폭)×7(두께) mm** → **단일 14×7 프록시**로 전환
  (STICK_SIZE (0.014,0.18,0.007), half_extent (0.007,0.09,0.0035), 나머지 2cm 성공값).
- 7mm 튜닝: cage 치팅→sphere 0.005→0.002(게이트 사망)→0.004→0.003. grip 40→150→40. ring/pinky grip 15 추가
  (약지·새끼가 물체 눌러 lift 방해 → +x 정렬 유도, wrap 방향). cube_lift 100→300(lift 유도). dt 1/120·decimation 4
  (얇은 물체 접촉 fidelity). **`stick_palm_rel_speed` 메트릭 추가**(managers.py, 던짐 페널티 임계값 측정용, 페널티 아님).
- **play 진단(스크린샷)**: 7mm 납작 스틱이 테이블에 누워 손끝이 **바닥에 닿은 채 옆면만 밀음, 밑면 접근 없음**
  → 보상 문제 아닌 **기하 문제**. 납작 물체를 평면서 집는 건 본질적 난제(FSM phase 0 획득, 논문·hand_grasp가 회피).

### 새 goal 설계 (사용자 발상, 미구현 — "ㄱ" 시 구현)
- **재프레임**: world-goal pose 운반 → **"스틱을 손바닥 안에 담고 안 떨어뜨리기"**. 고정 파지자세 제약 제거 →
  정책이 뒤집든 긁어담든 담기만 하면 됨(emergent 조작으로 납작물체 우회). FSM phase 0 목표와 일치.
- **구조(확정)**:
  - `cage reach + cage hold`(유지): 손가락 접근 + **움켜쥐기**(가상점 침투 연속값=dense). 파지 개구부(grip 영역) 제거.
  - `stick_in_palm` = **telescoping** best-so-far, `d=‖stick_pos_palm − T‖`, palm-local. (exp는 sparse라 telescoping.)
  - `palm_up`: palm 자세를 팜업 quat(HAND_ROOT_ROT)에 매칭, **yaw 자유**. **팜 법선(0.19,0.28,0.94)=파지 개구부라 안 씀.**
  - `throw 페널티`: `gate×(1−exp(−max(0,v_rel−v0)/σ))`, v_rel=palm-상대 스틱 선속도. **던짐 억제 전용**(held 아님).
  - `in_palm_success`: `d<3cm + palm_up cos>0.9 + off_table` 15스텝 유지 → 보너스.
- **목표점 T** = palm-local **(0.02, 0.0, 0.065)** ← hand_setting 월드 (0.065,0,0.5195)를 팜업 자세 기준 변환
  (palm≈root 가정, 몇 cm 오차 가능 → σ 넉넉·나중 정밀화). hand_setting: HAND_ROOT_POS(0,0,0.5)+ROT로 palm +x=world +z(팜업).
- **핵심 설계 근거**: ① "안 떨어짐"은 position(팜-local 목표 이탈)이 직접 잡음(속도 아님 — 낙하 초기 v_rel 작아 lenient v0면 놓침).
  ② 중력에 안 떨어지려면 손가락이 막아야 하고, 팜업 "그릇" 형태면 중력이 오히려 담아줌 → cage가 감싸기 보조.
  ③ clutch=cage hold(dense), 스틱→팜=telescoping(dense)라 안 sparse.
- **제거 예정**: stick_transport·stick_orientation·fine×2·goal_proximity·transport_success·cube_lift. cube_goal 커맨드 미사용.

### 2026-08-02: palm-in 항 cage 분리 + flip metric 신설
- **문제 재판독**: run 07-58-22(반성공)에서 flip이 cos_up~0.92 정체 + `in_palm_success` 전 구간 0.
  원인 = ① `stick_in_palm`·`in_palm_success`가 tripod **cage로 gate**됨 → 뒤집으며 핀치가 풀리면
  cage→0이라 "손바닥에 담기"를 벌줌(cage_inside_frac final 0.047→0.005 붕괴). ② 얇은 스틱은 cage
  최대 ~0.1이라 success의 `gate>0.3`을 영영 못 넘음(구조적 0).
- **사용자 통찰**: "빠르게 돌려서 손바닥에 닿을 정도"(날리진 않게). → cage로 gate하는 지금 구조가 그 전략을 벌줌.
- **수정(fresh 필수)**:
  - `StickInPalmProgressReward` (`chopstick_mdp_cfg.py`): `progress×cage_gate×lift_gate` →
    **`progress×lift_gate×palm_up_factor`** (`palm_up_factor=clamp(cos_up,0,1)`). 핀치 풀려도 담기 보상 유지, 팜다운 헛보상 차단.
  - `StickInPalmSuccessBonus`: `valid`에서 **`gate>gate_threshold`(cage) 조건 삭제** → `d<goal_radius & cos_up>min & clearance`.
    (cage 파라미터·gate_threshold는 RewTerm cfg 호환 위해 시그니처에만 남김, 미사용.)
  - throw_penalty(#3)는 **안 함**(사용자 결정, weight 0 유지).
- **metric 신설** (`managers.py` `_compute_cube_distance_metrics`): **`palm_up_cos`** = 팜 로컬 +x의 world z성분
  (=보상 cos_up과 동일 축). 지금까지 flip을 재는 metric이 없었음 — `palm_facing`은 flip이 아니라
  개구부 축·스틱 방향 내적(파지 정렬)이라 별개 량. 앞으로 flip 진행은 `Metrics/cube*/palm_up_cos`로 직접 관찰.
