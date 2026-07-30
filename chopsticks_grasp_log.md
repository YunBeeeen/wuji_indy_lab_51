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
