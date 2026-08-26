# nrmk_isaaclab_wuji — 작업 중 막힌 지점

Isaac / IsaacLab 쪽 함정만 모았다. 이 폴더 파일을 건드릴 때 로드된다.
루트 `CLAUDE.md` 에는 어느 항목이 여기 있는지 색인만 남겨 뒀다 — 파일을 열기 전에
보상 예산이나 액션 배선을 논해야 하면 그 색인을 보고 이 파일을 먼저 읽을 것.

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
