# CLAUDE.md

## 기본 행동 강령 (모든 작업에 항상 적용)


- **우선순위**: 안전 규칙 > 아래 `세션 규칙`·`작업 중 막힌 지점` > 일반 관행.
- 이 프로젝트의 실제 정보(명령·구조·컨벤션)는 이 파일과 `AGENTS.md`, `Deploy/CLI.md`를
  기준으로 한다.
- 2026-08-21: `CLAUDE2.md` / `범용_행동강령_B.md` / `CLAUDE_HANDOFF_2026-08-17.md` 의
  `@import` 를 제거했다. 앞의 둘은 프로젝트와 무관한 일반 행동 강령이고, 마지막은 시점이
  지난 상태 보고서다(최신 기록은 `ACTIVITY_*.md` 와 `worklog.md`). 파일 자체는
  `범용_행동강령_B.md` 와 핸드오프만 남겨 두었다.

## 세션 규칙

- 항상 한국어로 답한다.
- 기록 라우팅: **사용자 실험 기록**은 `nrmk_isaaclab_wuji/worklog.md`, `ACTIVITY_YYYY-MM-DD.md`,
  `nrmk_isaaclab_wuji/agent.md`(개괄식). **에이전트가 작업 중 막힌 지점·도구 함정**은 사용자
  기록에 섞지 말고 이 파일과 `AGENTS.md`에만 적는다.
- 학습(train.py)이 도는 동안 Isaac Sim probe/GUI 실행 금지 (RAM 31GB + swap 프리징 이력).
- Claude와 codex를 같은 파일에 동시 작업시키지 않는다.
- 커밋은 사용자 요청 시에만. 멀티라인 메시지는 heredoc(`git commit -m "$(cat <<'EOF' ...)"`) —
  사용자 터미널 복붙은 실패한 이력 있음.

## 작업 중 막힌 지점 (같은 곳에서 다시 막히지 않기 위한 함정 목록)

### 설정/보상 배선
- **`Episode_Reward/*` 는 에피소드 합계가 아니라 "초당 평균"이다** —
  `episodic_sum / max_episode_length_s` (`managers.py:2056`, IsaacLab `reward_manager.py:120`).
  따라서 **한 항의 최대치 = weight x (그 항이 활성인 시간 비율)**, `weight x 활성초`가
  아니다. 2026-08-07 이걸 놓쳐 예산을 **15배(=episode_length_s) 부풀려** 계산했고,
  "파지가 예산의 4%만 번다"(실제 60%)와 "파지를 버리고 벌리는 게 +87.5 이득"(실제
  -5.7 손해) 두 결론이 통째로 뒤집혔다. 예산 얘기를 꺼내기 전에 이 나눗셈부터 확인할 것.
- **지표 한 시점으로 추세를 단정하지 말 것.** `full_contact` 는 학습 중
  **0.03 ~ 0.98 로 크게 진동**한다(이전 보상 런 it 1050~2650 실측). 2026-08-07 저점
  하나(it 454)를 보고 "단조 발산"이라 오진했고, 300 iteration 뒤 스스로 회복했다.
  최소 5~10 지점을 뽑아보고 말할 것. 진동한다는 사실 자체가 "능력은 있는데 유지를
  못 한다"는 진단이기도 하다.
- **외란(external force)은 파지를 약화시키는 게 아니라 강화한다** — `min` 게이트와
  결합할 때. `functional_contact_min` 은 접촉 하나만 끊겨도 20 weight 가 0 이라,
  외란이 그 사건을 자주 만들면 정책이 **여유를 크게 두는 쪽**으로 학습한다.
  2026-08-08 실측(보상 동일, 외란만 다름): `full_contact` 0.335 -> 0.996/0.948,
  `min_functional_force` 0.107 -> 0.264 -> 0.449 N (용량-반응), `stick1_pivot_error`
  10.35 -> 6.86 -> 3.83 mm. **"이미 깨진 상태를 겪으니 외란은 불필요"는 틀렸다.**
- **`clamp(gap, min=0)` + `CLOSE_TIP_GAP = 0` 은 서로 다른 상태를 전부 만점으로 만든다.**
  clamp 가 음수를 전부 0 으로 모으는데 그 0 이 하필 CLOSE 의 목표값이다. 게다가
  `transverse_distance` 가 **norm** 이라 방향을 무시해서, "옆으로 9mm 어긋나 서로
  지나가는" 교차 상태가 `surface_gap = 0` = 완벽한 CLOSE 로 보고된다(스틱 단면 7mm,
  support 합 9.02mm). 2026-08-07 실제 증상. `clamp_gap=False` 는 부분 교차(lateral
  2~7mm)만 잡고 완전 교차는 못 잡는다 — 근본 수정은 `transverse_distance` 를
  ref 성분 투영으로 바꾸는 것이고 **미적용**.
- **공유 지수에서 항을 빼면 "희석"만 없어지는 게 아니라 "복원력"도 없어진다.**
  `mode_tip_gap_tracking` / `mode_grasp_stability` 는 gap/lateral/axial 을 한 지수에
  곱으로 넣는다. lateral 을 빼면 gap gradient 가 안 깎이지만, 동시에 lateral 오차의
  대가가 **90 weight -> 15 weight 로 6배 줄어든다**. 독립 Laplacian 은 `e/sigma > 3`
  에서 gradient 가 죽어 큰 오차에서 복귀가 안 된다. 분리할 때 이 손실을 계산할 것.
- **파지가 무너질 때 "손바닥 + 엄지" 로 후퇴한다.** 엄지 `finger1_joint2` 는 다른
  손가락 `finger*_joint2` 대비 토크 2.34배(0.426 vs 0.182 N.m) / 강성 3.9배(2.7 vs 0.7)
  이고 손바닥은 액추에이터가 아예 없다. 보상이 안 나올 때 제일 싼 자세가 "Stick2 를
  손바닥에 대고 엄지로 누르기" 라, 손끝 셋(index/middle/ring)만 죽는 패턴이 나온다.
  `min` 구조가 이걸 가속한다 — 하나가 0 이면 `functional_contact_min`(20)과
  `mode_grasp_stability`(50, 내부 `contact_gate` 도 min)가 **동시에** 0 이 되어 나머지
  다섯을 붙들 유인이 사라진다. 2026-08-07 hand_object 붕괴 실측.
- **파생 태스크가 "상속받은 종료 조건"과 모순되는 동작을 하는지 먼저 볼 것.**
  2026-08-07 hand_object 가 큐브 보상 3종 전부 정확히 0 이었던 진짜 원인은 보상도
  기하도 아니라 `hand_move` 에서 물려받은 `stick2_dropped`(`minimum_height=0.40`)였다.
  hand_object 는 큐브를 잡으러 **일부러 z=0.365 까지 하강**하는데, 하강이 끝나는
  정확히 2.00초에 스틱 루트가 0.40 을 지나 전 에피소드가 오판 종료됐고 CLOSE(2.5초)를
  한 번도 못 봤다. **진단 순서: `Episode_Termination/*` 와 `mean_episode_length` 를
  보상보다 먼저 볼 것.** `mean_episode_length` 가 스케줄 상의 특정 위상 경계와 정확히
  일치하면 그 위상에서 뭔가 잘못 종료되고 있다는 뜻이다. 나는 이걸 안 보고 "큐브가 두
  팁 사이에 없다, 기하 보정을 다시 하라"고 두 번 오진했다.
- **`min` 기반 보상은 평균이 좋아도 최악 순간이 전부를 결정한다.** `functional_contact_min`
  은 여섯 접촉력의 평균이 전부 포화(0.10N)를 넘는데도(0.47~2.12 N) 예산의 4% 만 번다 —
  벌리는 순간 `index_tip_stick1`/`ring_tip_stick2` 가 떨어져 그 스텝이 0 이 되기 때문
  (`full_contact` 0.62). **가중치를 논하기 전에 `*_min` 집계와 `full_contact` 를 볼 것.**
- **정책은 "지금 받는 액수"가 아니라 "남은 여지"로 움직인다.** 2026-08-07 hand_move 2단계
  에서 파지 계열이 이미 보상의 79% 를 벌고 있는데도 정책이 파지를 버리고 벌렸다.
  OPEN 완성 여지 +99.5 vs 파지 전량 −12.0 → 버리는 게 +87.5 이득이라 합리적 선택이었다.
  **비중이 아니라 `(최대 − 획득)` 을 항목별로 비교할 것.**
- **σ 를 조일지 말지는 `e/σ` 로 판단한다 (Laplacian 커널 기준).** gradient 크기는
  `(1/σ)·exp(−e/σ)` 라 `u = e/σ` 로 두면 `u = 1` 에서 최대다. 즉 **최적 σ = 현재 오차**.
  두 σ 의 gradient 교차점은 `ln(σ₁/σ₂)/(1/σ₂ − 1/σ₁)`. 현재 오차가 그보다 크면 조일수록
  **손해**다 (2026-08-07: skew 3.05° 에서 σ 0.05→0.01 은 gradient 14배 약화, 리셋
  5.21° 에서는 보상 0.0001 로 항이 죽는다). 조이는 건 2단 구성으로: 넓은 σ 로 캡처 →
  오차가 교차점 아래로 내려온 뒤 좁은 σ 로 수렴.
- **한 지수에 여러 오차를 넣으면 σ 튜닝이 거의 제로섬이 된다.** `mode_tip_gap_tracking` 은
  `exp(−gap/σg − lat/σl − ax/σa)` 라 셋이 곱해진다. 한 축을 조이면 그 축 gradient 는
  조금 오르지만 **항 전체 값이 줄어 나머지 축 gradient 가 같은 비율로 깎인다**
  (실측: lateral σ 5→2.65mm 는 lateral +18% / gap −37%). 새 항은 덧셈으로 분리할 것.
- **`ChopsticksGraspRewardsCfg`는 어느 태스크에도 연결 안 된 미사용 클래스.**
  `Indy-Wuji-Cube-Grasp`는 `CubeGraspRewardsCfg`를 쓴다 (`cube_grasp_env_cfg.py:90`).
  Chopsticks 쪽을 고치면 조용히 무시됨 — 2026-07-14 가중치 실험이 실제로 미적용된 채 돌았음.
- **보상 가중치에는 dt(1/30)가 곱해진다** (`isaac_neuromeka/env/managers.py:427`).
  일회성 보상의 로그 스케일 한 방 = weight/30 → 원하는 값의 30배로 설정할 것
  (예: `lift_success` weight 15000 = 한 방 +500).
- **`is_terminated_term`은 isaaclab의 클래스형 reward term** (`ManagerTermBase`).
  IsaacLab step 순서가 종료 계산 → 보상 계산이라 성공 종료 스텝에 같은 스텝 지급이 됨.
  `isaac_neuromeka.mdp`로 와일드카드 재수출되어 바로 참조 가능.
- **리셋은 관절 '상태'만 복원하고 위치 '목표' 버퍼는 안 채운다** — 액션에 없는 관절은
  목표 0을 향해 저절로 움직임. `hold_joints_at_default` 리셋 이벤트로 해결해 둠.
- **`init_state.joint_pos`는 정규식 키가 중복 매칭된다** — `finger[1-5]`가 이미 있으면
  반드시 pop 후 넣을 것 (`grasp/indy_wuji/env_cfg.py` 참고).
- **`object_below_surface_penalty`로 "누르기" 감지 불가** — 바닥이 강체라 관통이
  -0.04mm 수준(실측). 압착 억제는 penetration 페널티가 아니라 r_T 구조(성공 종료)로 해결함.
- **`ObjectToGoalProgressReward` 시그니처 변천** — distance_max(v1) → +window(v2) → potential_eps만
  (v2.1). 옛 파라미터를 cfg에 남기면 env 생성 TypeError. 주석 블록 재활성 시 시그니처 대조 필수.
- **매니저는 params의 "직접" SceneEntityCfg만 resolve한다** (`manager_base.py:398`,
  `isinstance(value, SceneEntityCfg)`) — **리스트/튜플 안에 넣은 SceneEntityCfg는 resolve 안 됨**
  (body_ids None → cage 함수에서 터짐). N개 cage를 넘기고 싶어도 reward term params엔 반드시
  **named 파라미터로 하나씩** 줄 것. 유연화는 공개 함수(named params) → 내부에서 리스트로 묶어
  헬퍼 호출하는 식으로 (`balanced_quad_cage_gate`가 이 패턴). 2026-07-26 quad 게이트에서 확인.
- **hydra CLI 오버라이드는 타입 엄격** — `env.rewards.X.weight=0`은 int로 파싱돼
  "Expected float, Received int" 에러. 반드시 `0.0`, `75.0`처럼 소수점 포함해 쓸 것.
- **`logs/` 정리는 글롭 금지, 정확한 폴더명 rm만.** 2026-07-18 사고: 스모크 정리용
  `rm -rf .../2026-07-18_1[4-9]*`가 사용자가 방금 시작한 라이브 런 폴더까지 매칭 →
  TB writer FileNotFoundError로 학습 크래시. 스모크는 시작 전 폴더명을 정확히 기록해두고
  그 이름만 지울 것. 지우기 전 `ps aux | grep train.py`로 라이브 런 유무도 확인.
- **라이브 런 폴더는 rename도 금지** (rm과 동일한 크래시 — TB writer가 옛 경로로 append).
  2026-07-18 두 번째 사고: 도는 런에 `_noori` 라벨을 붙이는 순간 FileNotFoundError로 죽음.
  라벨링 습관(`(success)` 등)은 **런 종료 후에만**. 사용자가 라벨을 붙이려 하면 먼저
  `ps aux | grep train.py`로 그 런이 끝났는지 확인하도록 안내할 것.
- **play 환경은 체크포인트가 아니라 "현재 코드 기본값 + CLI"로 빌드됨** — CLI 오버라이드로
  학습한 런(크기·ori 등)을 play할 때 같은 오버라이드를 안 넘기면 다른 환경(기본값)에서
  정책이 돌아감. 2026-07-19 실측: 스틱 런 play에 크기 인자 누락 → 뚱뚱한 상자 스폰,
  진단 결과 오독 위험. play 명령에 학습 시 오버라이드를 그대로 복붙할 것 (run의
  params/env.yaml과 대조 가능).
- **gap σ는 세 곳에 흩어져 있고, 제일 센 건 이름이 다른 쪽이다.**
  `open_tip_gap.sigma`(w20) + `close_tip_gap.sigma`(w20) + **`mode_grasp_stability.gap_sigma`(w50)**.
  앞의 둘만 고치면 w50짜리가 여전히 헐거운 σ로 "13mm든 20mm든 비슷하게 좋다"고 말해 효과가 희석됨.
  반대로 이걸 이용해 **앞 둘은 넓게(캡처) + gap_sigma만 좁게(수렴)**로 항 추가 없이 2단 구성 가능
  (2026-08-06 채택: `gap_sigma 0.005→0.001`). 부작용은 리셋 구간 gap 압력의 55~71%가 사라지는 것.
- **`gap_sigma`는 파일마다 따로 있다. 한 곳만 고치면 나머지 태스크는 옛 값으로 돈다.**
  `HandMoveRewardsCfg`는 hand_grasp 보상을 **상속이 아니라 복사**한 것이라
  `mode_grasp_stability`의 params가 자기 파일에 따로 존재한다.
    - `hand_grasp_env_cfg.py:559`  -> `hand_grasp` 태스크만
    - `hand_move_env_cfg.py:520`   -> `hand_move` + `hand_object`(상속)
  2026-08-06 실제로 두 값이 0.001 / 0.003 으로 갈린 채 돌았다. **어느 태스크를 고치는 건지
  확인하고, 실제 적용값은 런의 `params/env.yaml`에서 확인할 것** (`grep gap_sigma`).
  이 "복사본" 구조는 `HandMove*Cfg` 전반에 해당하므로 보상 파라미터를 고칠 때 항상 의심할 것.
- **`mode_grasp_stability` 안의 변수명 `mode_gate`는 misnomer다.** `mode_tip_gap_tracking`의
  `mode_gate`는 커맨드 **one-hot(0/1) 스위치**지만, `mode_grasp_stability`의 동명 변수는
  **σ로 만든 tolerance 커널**이고 one-hot 곱셈이 없다(모드에 따라 `target_gap`만 20mm↔0mm 블렌딩).
  "게이트"라는 이름 때문에 모드 전환 담당으로 오독하기 쉬움.
- **`open/close_tip_gap`은 별개 구현이 아니라 `mode_tip_gap_tracking` 같은 함수를 두 번 등록한 것.**
  `mode_index`/`target_gap`만 다름. "mode_tip_gap_tracking의 σ를 고친다" = "그 두 term의 sigma를 고친다".
- **axial(장축) 항은 OPEN에 사실상 무영향이고 CLOSE만 깎는다** (2026-08-06 실측).
  전 개폐 구간 axial 스윙 폭이 1.2mm뿐인데 σ가 5mm라 제약 역할은 못 하면서 완전 폐합에서만 -22%.
  게다가 리셋에서 정확히 1.0(최댓값)이라 아래 σ 함정과 같은 형태. 손대려면 σ를 1.5mm로 조이거나
  CLOSE 쪽만 해제할 것.
- **`bilateral_cube_force`(hand_object)는 젓가락 파지 유지에 게이트가 없다** — CLOSE 모드만 본다.
  예산 450 vs 파지 방어선 490(`functional_contact_min` 140 + `mode_grasp_stability` 350)이라
  여유가 8%뿐. "젓가락을 흘리면서 큐브만 짓누르기"가 이론상 거의 이득이다.
  `Metrics/hand_grasp/functional_contact_count`가 5.5 아래로 내려가면서
  `Metrics/hand_object/bilateral_force_score`가 오르면 그 거래가 일어나는 중 → weight 100→50.
- **weight를 정확히 `0.0`으로 두면 리워드 항이 아예 계산에서 빠진다**
  (`reward_manager.py:146`, 프로젝트 `managers.py:1947`). 주석 처리와 동작이 같아서
  커리큘럼 on/off에 그대로 쓸 수 있다. 단 hand_setting 메트릭처럼 **보상 term에 하드-커플링된
  메트릭은 같이 꺼지므로** 주의(hand_object 메트릭은 센서에서 직접 계산해 영향 없음).
- **헐거운 pose σ는 리셋(스폰) 상태를 이미 고득점으로 만들어 그 항의 gradient를 죽인다.** 그러면 남은
  sub-reward가 지배하는데, 그게 유클리드 거리 기반이면 **방향 없는 지름길**(opposition 대신 '아래로
  파고들기')을 택한다. hand_setting 획득 실패의 진짜 원인이 이거였음 — two_stick/thumb σ를
  `0.10/1.57(90°)→0.01/0.25(~14°)`(thumb_sigma 0.02→0.06)로 조여 리셋 baseline을 낮추니 실제 정렬을
  강제 → **획득 첫 성공(2026-08-05)**. kp/kd가 원인인 줄 알았으나 σ였음(kp/kd는 악화 요인일 뿐).
  **교훈**: dense pose 보상 σ는 "리셋에서 낮은 점수"가 되게 조일 것 + min/유클리드 sub-term에 방향
  없는 지름길 여지가 없나 점검.

### 액션 파이프라인 (2026-07-23 진단)
- **"clip"이 세 종류다. 섞어 말하지 말 것.** ① `clip_actions=1.0`(rsl_rl cfg) = **actor 출력**을
  ±1로 자름, 적용 지점은 래퍼(`isaaclab_rl/.../vecenv_wrapper.py:153`) — **켜져 있음**.
  ② `ActionTermCfg.clip`(dict) = **관절 target(rad)** clamp — 설정 안 됨. ③ `ClampedJointActionCfg`
  — demo 태스크만 사용. ①이 있어도 target이 관절 limit 안이라는 보장은 없음.
- **`ActionTerm.raw_actions`는 클리핑 "이후" 값이다** (래퍼가 `env.step` 진입 전에 자름).
  클리핑 전 원본은 ActionTerm에서 볼 수 없음. 반면 **PPO storage에는 클리핑 전 값이 저장**됨
  (`rsl_rl/algorithms/ppo.py:144`) — ±1 밖에서도 그래디언트가 흐름.
- **관절 target을 clamp하는 코드가 전 경로에 없다.** `soft_joint_pos_limit_factor`는
  `_data.soft_joint_pos_limits`를 **계산·저장만** 하고(`articulation.py:765,1660`) 어떤 clamp에도
  안 쓰임. 유일한 방어선은 PhysX joint limit이고, target 버퍼엔 limit 밖 값이 그대로 남음.
- **`resolve_matching_names`의 산문 docstring은 뒤집혀 있다**(`isaaclab/utils/string.py:186-191`).
  구현(225-241행)과 바로 아래 예시가 맞음: `preserve_order=False`(기본) → **articulation joint
  순서**, True → 정규식 나열 순서. 액션/관측 순서는 정규식을 어떻게 쓰든 USD DOF 순서를 따름.
- **`ActionTerm._joint_ids`는 전(全) 관절 제어 시 `slice(None)`이다** — Wuji는 26 DOF 전부 제어라
  실제로 slice. torch 인덱싱(`joint_pos[:, _joint_ids]`)은 slice를 그대로 받지만, **파이썬에서
  `list(_joint_ids)`/`.index()`로 다루면 `TypeError: 'slice' object is not iterable`로 터진다.**
  2026-08-03 arm_jN_cmd 메트릭 추가 때 이걸로 사용자 런이 시작 즉시 크래시(events 88바이트, 체크포인트 0).
  slice 분기 필수(slice면 컬럼=DOF id 직접). **그리고 진단 메트릭은 학습을 죽이므로 try/except로 zeros
  폴백을 감쌀 것** — 메트릭 하나 때문에 4096-env 런이 죽으면 안 됨.
- **뉴로메카 예제 `JointResidualAction`은 실행된 적 없는 코드**(어느 태스크도 미참조)이고
  관절 부분집합에서 깨진다: `zeros_like(data.joint_pos)`(N,26)에 (N,18)을 더해 RuntimeError,
  `apply_actions`의 `env_ids.unsqueeze`는 slice에서 AttributeError. 잔차가 필요하면
  `CustomResidualJointPositionAction`(2026-07-23 신설)을 쓸 것.
- **obs 차원을 셀 때 `action_history`를 빼먹지 말 것.** `mdp.action_history`는
  `action_manager.prev_action`(`observations.py:179`)이라 **액션 차원을 그대로 따라간다.**
  액션이 18→26이 되면 `joint_pos`와 `action_history`가 **둘 다** 커져 obs가 75→**91**(83 아님).
  2026-07-23 실제로 두 번 틀림.
- **보상 weight는 항 간에 직접 비교하면 안 된다 — gate형 1 ≈ progress형 300.**
  gate형(`finger_cage_hold`·`cube_lift`·`goal_proximity`)은 240스텝 지급이라 총액 `2.4W`,
  progress형(`*_transport`·`*_orientation`·`keypoint_goal`·`*_grip`)은 best-so-far 텔레스코핑이라
  에피소드당 1회분 예산 `Δ×gate×W/30`. "transport 6000인데 lift 150보다 작다"가 정상.
  가중치를 논하기 전에 **예산(점) 단위로 환산**할 것.
- **`cube_lift`는 W와 W/h가 서로 다른 것을 정한다.** `W`=호버 연금 총액, `W/h`=바닥에서 떼는 기울기.
  h만 올리면 총액은 그대로인데 떼는 힘만 약해져 "테이블에서 비비는" 증상이 난다 (2026-07-23 실측).
- **keypoint 보상의 `d`는 정규화되지 않은 미터**라 φ(0~1)와 스케일이 100배 이상 다르다.
  box에서 keypoint 8000이 예산 13점뿐이라 학습 신호가 아예 없었던 이력(2026-07-23).
- **큰 weight 항에서 gate를 뺄 땐 "대체 gate가 hack 상태에서 정확히 0인지" 확인.** 2026-08-04
  `palm_up`(w20000)을 cage에서 떼고 `goal_gate`만 남겼더니 파지를 아예 포기하고 **빈손으로 팜만
  뒤집는 해킹**(run 23-52-23: palm_up_cos 0.98인데 cage_inside 0, palm_dist 0.31m). 원인: 목표가
  스틱 테이블 높이와 가까워 **테이블에서도 `goal_gate`가 0이 아니라 ~0.033 잔여** → 큰 weight가 증폭.
  `_goal_proximity_gate`(위치)는 물체 정지점에서 잔여가 남아 hack 차단에 부족하고, `_lift_gate`
  (clearance)는 테이블에서 **정확히 0**이라 확실. flip/담기 항은 `lift_gate × goal_gate`로 둘 다 걸 것.
- **잔차 전환 시 `processed_actions`의 의미가 바뀐다**(절대 목표 → 증분).
  `env/managers.py:480`(action_track_err), `scripts/rsl_rl/play.py:687`이 이걸 절대 목표로 읽으므로
  `joint_pos_target`으로 같이 고칠 것. 각 지점에 경고 주석 있음.

### Isaac Lab API 규약 (문서에 안 적혀 있어 매번 다시 파야 하는 것들)
- **`ContactSensorData.force_matrix_w`는 "필터 body가 센서 body에 가하는 힘"이다.**
  docstring에 부호 규약이 없어 추정 금지. 근거는 설치된 isaacsim 테스트
  `isaacsim/core/api/tests/test_rigid_prim_view.py: contact_force_test` —
  `g=-10`, Box(1kg) 위에 TopBox(1kg), `sensor=Box / filter=TopBox`인데
  `assert forces_matrix[:,0,:] == [0,0,-10]`. TopBox가 Box를 **아래로** 누르는데 -10z가
  보고됨 → 센서가 **받는** 힘. shape는 `(N, B, M, 3)` = 센서body × 필터body.
  (기존 프로젝트 코드는 전부 `norm()`만 써서 부호가 필요 없었음 — 2026-08-06 hand_object가 최초.)
- **hydra CLI 오버라이드는 cfg `__post_init__` "이후"에 적용된다**
  (`isaaclab_tasks/utils/hydra.py:80` 인스턴스화 → `:90` `from_dict`).
  따라서 `env.some_flag=false`로 끌 수 있어야 하는 검증을 `__post_init__`에서 raise하면
  **오버라이드가 읽히기 전에 죽어서 영원히 못 끈다.** 그런 검증은 env 생성 시점
  (command term / action term의 `__init__`, 여기선 `env.cfg`가 이미 오버라이드된 상태)으로 옮길 것.
  2026-08-06 hand_object `require_calibration`에서 확인.
- **`get_checkpoint_path`는 `logs/rsl_rl/<현재 experiment_name>/` 안에서만 찾는다**
  (`train.py:168`). experiment_name이 다른 런에서 fine-tuning하려면 `--resume/--load_run`으로
  안 되고 경로를 직접 줘야 함 → 2026-08-06 `--init_checkpoint` 신설.
  `runner.load`는 `strict=True`라 obs/action 차원이 다르면 거기서 예외가 난다(= shape 검증 겸용).
- **`configclass`는 "상속받은" property를 처리하지 못한다.** `_custom_post_init`이 `dir(obj)`를
  돌며 deepcopy할 때 property는 건너뛰는데, 판별을 **`obj.__class__.__dict__`**로만 한다
  (`configclass.py:398`). 부모에만 있는 property는 자식의 `__dict__`에 없어서 property로
  인식되지 않고 setattr을 시도 → `AttributeError: property 'X' ... has no setter`.
  **cfg를 상속할 때 부모의 property를 자식에 재선언할 것** (2026-08-06 `HandObjectScheduleCfg`가
  `slerp_end_time_s`로 터짐).
- **`configclass` 클래스의 필드는 `Cls.field`로 접근할 수 없다**(dataclass 필드로 바뀜).
  `HandMovePPORunnerCfg.algorithm` → AttributeError. 그렇다고 `__post_init__`에서 고치면
  **더 나쁘다**: configclass는 사용자 `__post_init__`을 먼저 실행하고 그 **다음에** 멤버를
  deepcopy하므로(`configclass.py:93`), 아직 **공유 상태인 기본 객체**를 건드려 형제 cfg들까지
  같이 바뀐다. 값 하나만 다르더라도 블록 전체를 명시 선언할 것.
- **커맨드 term이 액션 term의 목표 버퍼를 덮어쓰면 키보드가 조용히 죽는다.** 2026-08-06
  hand_object에 위치 궤적을 넣자 `HandRootHoldAction._refresh_target_position`이 매 물리 스텝
  액션 term의 `_target_root_pos_w`를 커맨드 값으로 덮었는데, 키보드
  `add_target_position_delta`가 **바로 그 버퍼**에 쓴다 → 이동키 무반응.
  **회전키는 커맨드 버퍼에 직접 써서 멀쩡했던 게 진단을 늦췄다.** 수동 모드에서 커맨드가
  `target_pos_w`로 `None`을 반환해 손을 떼게 하는 게 해법(자세 쪽 `_scripted_target_quat`와
  같은 패턴). **"일부 키만 안 먹는다" = 두 경로가 같은 버퍼를 두고 싸우는지 볼 것.**
- **`RslRlVecEnvWrapper.__init__`이 `env.reset()`을 부른다** (`vecenv_wrapper.py:66`).
  play.py에서 `manual_root_controller.attach()`는 그보다 **뒤**라, 커맨드 term의 첫 리셋은
  항상 **스크립트 경로**로 돈다. 수동 모드를 전제로 한 예외/분기를 `_resample_command`에 두면
  수동 세션이 시작조차 못 한다. 2026-08-06 hand_object 보정 세션이 이걸로 막혔음.
- **"학습을 막는 가드"를 play에도 걸지 말 것.** 미보정 기하로 *학습*이 도는 걸 막으려던
  `require_calibration`을 play에도 걸어놨더니, 정작 그 값을 **재러** 들어오는 보정 세션만
  계속 막혔다. play.py가 시작 시 자동으로 끄게 함(train은 유지).
- **command/action term은 모듈 레벨 cfg의 "복사본"을 들고 있다** (configclass deepcopy).
  모듈 객체(`HAND_MOVE_SCHEDULE` 등)만 고치면 이미 만들어진 env엔 반영 안 됨 —
  `env_cfg.commands.<term>.schedule`도 같이 고칠 것.
- **클래스형 manager term은 반드시 `ManagerTermBase`를 상속해야 한다**
  (`manager_base.py:347-352`가 `issubclass` 검사, 아니면 TypeError). 상속하면 `(cfg, env)`로
  자동 인스턴스화되고 `reset(env_ids)`도 불러준다.
- **CommandTerm의 `metrics` dict는 자동으로 TB에 올라간다** — 진단용 스칼라를 managers.py의
  거대한 metric 블록에 넣지 않고 가볍게 붙일 자리. 단 min/max/final 집계는 managers.py 쪽에만 있음.

### 모델·계약 출처 (2026-08-18 sim-to-sim 진단에서 나온 것들)

- **벤더 description은 실물보다 20/20 관절 전부 좁다.** 실물 SDK(v1.7.0/fw 1.2.1) 읽은 값이
  `wuji-description@06e5f14`보다 모든 관절에서 넓다 (예: `finger3_joint3` 상한 실물 1.680047
  vs 벤더 1.5512). **정규화가 아핀 변환이라 포함관계로 넘어갈 수 없다** —
  `q_norm = 2(q−center)/(upper−lower)`이므로 좁은 표로 정규화하면 실물이 한계 근처일 때
  `|q_norm| > 1`이 나온다(학습에서 본 적 없는 입력). 최악 `finger3_joint2` 전범위의 8%.
  2026-08-18에 **실물값으로 통일**: `wuji_right_filtered.usda`(degree) +
  `Deploy/common/policy_contract.py`의 `REAL_HAND_FACTORY_LIMITS`.
  **이건 "이 개체"의 공장 캘리브레이션이다** — 두 번째 손은 `validate_factory_limits()`로
  대조할 것(채택 말고 보고만).
- **`wuji_right_filtered.usda` 하나가 전 태스크에 걸린다.** `hand_grasp`/`hand_setting`/
  `finger_reach`는 직접 로드(`assets/wuji.py:30`), `hand_move`/`hand_object`/`hand_real`/
  `hand_final`은 `wuji_right_floating.usda`(`hand_move_env_cfg.py:170`)가 **이걸 참조**한다.
  한계를 고치면 도는 학습의 resume이 다른 환경이 된다. 고치기 전에 `ps aux | grep train.py`.
- **Isaac과 Deploy는 서로 다른 손 모델을 쓴다.** Isaac은 로컬
  `assets/model/urdf/wuji_right/wuji_right.urdf`에서 임포트한 USD, Deploy는 공식
  description. **부품 CAD는 동일**(무게중심 반경분포 완전 일치, bbox 대각 차이 0.000000mm)이고
  좌표 배치만 다르다 — 단 `finger1_link2`만 삼각형 수가 다름(1852 vs 1636 = 실제 형상 변경).
  물리적 부품 위치 차이는 최대 0.33mm(엄지), **`finger3`는 0.001mm로 사실상 동일**
  (→ 중지가 sim-to-sim 진단에 최적인 이유).
- **공식 description에는 충돌 메시가 없다.** 로컬 URDF는 `*_collision.STL` 26개를 따로 갖는데
  (palm 25KB) 공식은 visual만 있어 **MuJoCo가 456KB visual의 convex hull을 충돌로 쓴다.**
  손바닥 convex hull이 오목부를 메워 리셋에서 `palm↔stick2 −3.94mm 관통` → 스틱 사출.
  **중력을 꺼도 1.5m 밖으로 튕겨나가는 게 증거**(파지력 부족이면 제자리에 떠 있어야 함).
  게인·dt·마찰·접촉그룹 어느 것에도 반응하지 않았던 진짜 원인.
- **MJCF `contype/conaffinity` 그룹 때문에 Isaac 접촉 하나가 MuJoCo에서 재현 불가.**
  link1=(0,0) 충돌없음, link2=(2,2), link3/4/tip/palm=(1,1), stick=(1,1). `1&2==0`이라
  **`finger1_link2 ↔ stick`이 물리적으로 성립 안 함** = Isaac `ANCHOR_CONTACT_GROUPS`의
  `thumb_mid_stick2`. 스틱 geom을 `contype/conaffinity=3`으로 하면 해결. 단 실측상
  이것만으론 파지가 안 살아난다(위 convex hull 문제가 지배적).
- **dt 하나로 결론내지 말 것 — 수렴 검사를 할 것.** 15초 에피소드 Stick1 변위 실측:
  1/120 189.9mm / 1/240 182.7 / **1/480 발산(75.9 m/s)** / **1/960 86.1(이상치)** /
  1/1920 195.9 / 1/3840 196.1. **비단조이고 1/960만 좋다.** 1/1920과 1/3840이 일치하니
  참값은 "놓친다". 2026-08-18 1/480 단일 표본을 보고 "dt가 원인이었다"고 오진했다.
- **Isaac dt와 MuJoCo dt를 같게 맞출 이유가 없다.** PhysX는 1/120에서 스텝당
  `solver_position_iteration_count=16`을 돌지만(`hand_grasp_env_cfg.py:141`) MuJoCo Newton은
  그렇지 않다. 맞춰야 하는 건 **"목표를 1/30초 유지"**이지 substep 수가 아니다.
  `mujoco_scheduler.py`의 하드 assert를 그 불변식으로 바꿔 뒀다.
- **Isaac Kp/Kd는 벤더값이 아니라 사용자가 Isaac Sim에서 스텝 입력으로 튜닝한 값이다.**
  벤더 MJCF의 identified kp/kv(관절별 ω≈4.7Hz/ζ≈0.67 설계)보다 1.5~7.2배 세다.
  그리고 **0.2rad 스텝은 20개 중 6개를 즉시 토크 포화**시킨다(joint4 5개 + finger1_joint2).
  포화 구간 스텝응답으로 동정하면 kp를 과대평가하고 kd는 아무 값이나 되므로,
  joint4의 ζ가 0.003~0.05로 나온 게 그 징후일 수 있다. 선형 구간(0.05rad)에서 재확인할 것.
  effort limit은 URDF=MJCF=Isaac 3중 완전 일치(차이 0.00e+00)라 논쟁 대상이 아니다.
- **`FingerTipReachCommand`는 palm-local로 샘플하지만 `command` 프로퍼티는 env-frame이다**
  (`commands.py`: `target_e = target_w − env_origins`). 주석만 palm-local이라 오독하기 쉽다.
  2026-08-18에 `target_palm`을 따로 저장하게 해서 관측이 그걸 직독한다.
- **액션을 관절 부분집합으로 줄이면 나머지 관절이 목표 0으로 흘러간다.**
  `set_joint_position_target(..., joint_ids=self._joint_ids)`라 액션 term은 자기 관절만 쓴다.
  `hold_joints_at_default` 리셋 이벤트 필수 — **매 스텝 `q_target=q_current` 갱신은 안 됨**
  (외란/드리프트가 새 목표로 굳는다). 2026-08-18 finger_reach 4D 전환에서 확인.

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

### tip_link 프레임 (2026-08-21)

- **Wuji URDF 가 두 벌이고 `*_tip_link` 원점이 다르다.** Isaac 은
  `nrmk_isaaclab_wuji/.../wuji_right/wuji_right.urdf`(`wujihand-right-v1.0.2`),
  MuJoCo/배포는 `Deploy/assets/wuji_description/.../right.urdf`(신 릴리스).
  `*_tip_fixed` origin z 가 **엄지 28.3 vs 31.3 mm, 검지 27.4 vs 26.7 mm**,
  나머지 셋은 동일. 관절 축·중간 프레임은 규약이 달라도 합성하면 전부 상쇄되므로
  **차이는 이 한 숫자뿐**이고 모든 자세에서 상수다(0.04 mm 이내).
- **그런데 물리적 부품은 같은 자리에 있다.** 메시 원점이 반대 방향으로 정확히
  상쇄한다 — palm 좌표에서 부품 centroid 차이 finger2 **0.0003 mm**, 엄지
  **0.33 mm**(= 이미 알려진 값). **"엄지가 3 mm 짧은 손으로 학습했다"는 오진이다.**
  기하·접촉·pregrasp 는 재검토 대상이 아니고, 틀린 건 **관측 배선뿐**이다.
  `obs[40:55]` 는 부품이 아니라 **링크 프레임 원점**을 보고하기 때문.
- **그래서 `obs[40:55]` 는 백엔드에 묻지 말고 q 에서 FK 로 푼다.**
  `observation_adapter` 가 `POLICY_TIP_FRAME_URDF`(= Isaac) 로 계산한다.
  강체라 `site == FK(q)` 가 6.7e-08 m 이므로 잃는 게 없고, 실물은 애초에 이 방법뿐이다.
  `WujiBackend` 프로토콜에서 `get_fingertip_positions_in_palm` 을 뺀 이유가 이것 —
  실물 백엔드를 막고 있던 메서드가 사실 백엔드 책임이 아니었다.
- **`WujiHand1FingertipFK(source)` 는 기본값이 없다.** 이 버그의 원인이
  "암묵적 기본값이 조용히 official 을 뜻한 것" 이라서 일부러 필수 인자로 만들었다.
  `POLICY_TIP_FRAME_URDF`(관측용) / `OFFICIAL_URDF`(벤더↔MuJoCo 검사용) 중 명시할 것.
- **중지만 보면 절대 안 보인다.** `finger3` 는 두 파일이 0.0011 mm 로 일치하고
  finger_reach 는 중지만 쓰는 데다 FK 를 관측에서 아예 뺐다. "중지가 sim-to-sim
  진단에 최적" 이라는 기존 메모는 **같은 이유로 중지가 이 문제를 가린다**는 뜻이기도 하다.
  손끝 관련 검증은 **엄지를 반드시 포함**할 것.
- 회귀 방지: `tests/test_common.py: PolicyTipFrameTests` 가 3.0/0.7 mm 를 못박고,
  `run_policy --validate-fk` 가 `[OFFICIAL vs POLICY TIP FRAMES]` 로 매번 출력한다.

### 명령 한계·소프트 여백 (2026-08-19)

- **기계적 스톱에 눌린 관절은 소프트 한계를 "넘겨서" 읽힌다.** 실측
  `finger3_joint3` = 1.6804522 vs `COMMAND_TARGET_LIMITS` 상한 1.680047 (초과
  0.4 mrad = 0.023°). 측정값에서 궤적 보간을 시작하면 **첫 목표가 이미 범위 밖**이라
  `write_joint_position_targets`가 거부하고 주행이 죽는다. 궤적 끝점은 clamp할 것
  (`real_wuji_scheduler.py`). 단 **관측·잔차에 쓰는 q는 원본을 유지**하고 정책 출력에
  대한 거부 정책도 유지 — clamp는 궤적 생성에만 적용한다. 초과가 mrad가 아니라 수십
  mrad라면 clamp로 덮을 게 아니라 `REAL_HAND_FACTORY_LIMITS`를 다시 읽을 신호다.
- **`COMMAND_TARGET_LIMITS`를 줄여서 여백을 만들지 말 것.**
  `policy_contract.py`의 `validate_contract`가
  `COMMAND_TARGET_LIMITS == OBSERVATION_NORMALIZATION_LIMITS`를 강제하는데, 이건
  "Isaac이 이 범위로 학습했다"는 **사실의 기록**이다. 여백은 `soft_command_limits()`가
  **새 테이블을 반환**해 그 위에 얹는다. 정규화 테이블을 같이 줄이면 정책 입력 매핑이
  바뀌어 학습한 정책이 깨진다.
- **`fraction == 1.0`은 재계산하지 말고 원본을 복사할 것.** float32에서
  `centre ± half`로 재구성하면 **1 ULP 어긋나** "기본값은 기존 로그를 그대로 재현한다"가
  깨진다. 2026-08-19 테스트가 실제로 잡았다.
- **여백의 정의는 `중심 ± f × 반범위`** (Isaac `soft_joint_pos_limit_factor` 규약).
  `f × 상한`으로 하면 음수 하한에서 뜻이 뒤집힌다.
- **Isaac Wuji는 `soft_joint_pos_limit_factor=1.0`이다** (`assets/wuji.py:95`) —
  즉 **학습된 정책은 하드 스톱까지 명령할 수 있었다.** 배포에서만 0.95를 걸면 의도된
  sim-real 불일치이고, 그래서 계약 테이블과 분리해 이름을 갖게 둔 것이다.
- **finger_reach에서 여백이 공짜였다고 `hand_real`에 그대로 옮기지 말 것.**
  finger_reach는 4 DoF로 3D 목표를 쫓아 여유자유도가 1개라 0.90까지도 도달 범위
  손실이 0이었다(19만점 격자 IK). 그러나 `hand_real` pregrasp는 policy idx 10이
  1.6272 vs 0.95 상한 1.6244, idx 18이 1.6272 vs 1.6197로 **범위 밖**이다.
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

### 키보드/입력
- **이 Kit 빌드의 carb는 콜백에 `KeyboardInput` enum이 아니라 "문자열"을 넘긴다.**
  `event.input.name` → `AttributeError: 'str' object has no attribute 'name'`.
  게다가 **carb 구독 안에서 난 예외는 전파되지 않아 키를 누를 때마다 무한 재출력**된다.
  `event.type`도 마찬가지라 enum 비교(`event.type == carb.input.KeyboardEventType.KEY_PRESS`)가
  조용히 항상 False가 되어 **키가 아예 안 먹는 증상**으로도 나타난다.
  → 이름을 정규화해서 문자열 비교할 것(`hand_move_manual_control.normalize_carb_name`),
  콜백 전체를 try/except로 감싸 1회만 경고할 것, 바인딩 없는 키 이름을 찍어 키맵 불일치를 노출할 것.
  **IsaacLab 자체 `devices/keyboard/se3_keyboard.py`도 `event.input.name`을 쓰므로 같은 잠재 버그.**
  `play.py:865`의 `--keyboard_hand_mode` 경로도 미수정 상태(2026-08-06).

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

### 동시 작업/도구
- **같은 문자열이 여러 클래스에 중복된 파일은 클래스 범위로 잘라서 편집할 것.**
  `commands.py`의 `self.metrics["error_pos"] = torch.zeros(...)`는 커맨드 클래스 두 곳에
  똑같이 있어 전역 replace가 실패하거나 엉뚱한 클래스를 고친다. `s.index("class X")`~
  `s.index("class Y")`로 슬라이스한 뒤 그 안에서만 치환하고, 끝나면 `git diff | grep '^-'`로
  **삭제된 줄이 의도한 것뿐인지** 확인할 것.
- **파일이 세션 밖에서 바뀐다** (사용자·codex 동시 작업). Edit 전에 해당 구간을 다시 Read하고
  `git diff`로 상태를 확인할 것. old_string 불일치가 그 신호.
- **`cube_grasp_env_cfg.py`의 `__post_init__` surface_z 배선 블록(★ 표시)은 절대 지우지 말 것.**
  2026-07-15 편집 중 유실됨 — 없으면 상판 큐브의 clearance가 스폰부터 +BASE_Z라 lift 보상이
  만점에서 시작하는 대형 버그. 파일 구간을 재작성할 때 기존 오버라이드 줄을 보존할 것.
- **headless + 카메라 렌더 스크립트는 행 걸림** (grip_snapshot.py 24분 무응답 이력).
  눈으로 확인할 건 `grip_capacity.py --gui` 방식으로.
- **CRLF/멀티라인 XML은 정규식 매칭이 조용히 실패** — 제조사 MuJoCo XML 파싱은 ElementTree로.

### Isaac 측정 함정 (자세한 근거는 nrmk_isaaclab_wuji/agent.md)
- 물체 고정은 매 스텝 teleport 금지 (관통 누적 → PhysX 폭발). `set_disable_gravities`
  + 매 스텝 속도 0으로.
- `finger*_tip_link` 원점은 패드가 아니라 마지막 관절 (패드는 2~3cm 더 앞).
- Indy7 `joint1`은 감소가 손 '하강' 방향.
- 파지 판정에는 "무엇으로 잡았는지"(엄지·중지 접촉 거리)를 반드시 포함 — held만 보면 속음.
- **`Metrics/cube*/palm_facing`는 flip(팜업) 지표가 아니다.** palm_facing = "파지 개구부 축"
  `(0.19,0.28,0.94)`(≠팜 법선, misnomer — `rewards.py:850`)를 **스틱 방향(to_cube)**과 내적한
  **파지 정렬** 값. "손바닥이 하늘을 보나"(supination)는 `PalmUpProgressReward.cos_up`(팜 로컬 +x=팜
  평면 법선, 실측 0.965,-0.008,0.262의 world z성분)이고, 이걸 재는 metric은 2026-08-02에야
  **`palm_up_cos`**로 신설됨(managers.py). flip 얘기하면 `palm_up_cos`를 볼 것. 2026-08-02 실제로
  palm_facing 0.94를 flip 정체로 오독함(사실은 파지 정렬).
