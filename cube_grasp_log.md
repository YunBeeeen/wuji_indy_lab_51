# cube_grasp_log.md — Indy-Wuji-Cube-Grasp

> Task별 큐레이션 로그. **원본(수정 금지)**: `nrmk_isaaclab_wuji/worklog.md`(2026-07-08~07-15),
> 루트 `WORKLOG.md`(Cube Grasp 섹션). 상세는 원본 해당 날짜 참조.

## 개요
- **task**: `Indy-Wuji-Cube-Grasp` — Indy7 팔 + Wuji 핸드로 테이블 큐브 파지·리프트.
- **상태**: 단독 task는 **종료/기반화**. 여기서 만든 **cage 보상 체계(Dexterous Pre-grasp Eq.14/15)**
  와 `cube_lift` 선별이 이후 `functional_grasp`(chopsticks)·`hand_grasp`의 공통 토대.
- **핵심 산출물**: 접촉센서 없이 가상점 SDF로 "감싸기"를 보상하는 cage reward, reach(Eq.14)/hold(Eq.15)
  역할 분담, `cube_lift`(진짜 파지만 선별), palm_facing 차분형, arm_manipulability(r_MP).

## 마일스톤 타임라인
- **07-08~10 (셋업)**: 씬/자산, GUI stutter, 체크포인트 mismatch, lift reward 비활성, 큐브 randomization.
- **07-11 (cage 도입)**: reward hacking 회귀(파지 대신 거리항 farming) → 진단 연쇄(per-step 궤적,
  `distance_max`가 hacking 메커니즘, "**파지가 보상에 아예 없음**") → **cage hold(Eq.15) 구현**, reach(Eq.14) 추가.
- **07-12 (cage 완성)**: 가상점 6→**12점**(검지)로 자세 교정. **`cube_lift` 도입**. palm_facing + 기울이기 편법 차단.
- **07-13 (자세·발산)**: palm_facing 국소최적 → **arm_manipulability** → 차분형 복원. **제어 안 되는 약지·새끼가
  큐브를 쳐냄**. **action space 설계 오류로 정책 발산** → 제어주파수/reward 재조정 → **최초 hold 켜짐**.
  게이팅 버그(telescoping 깨짐, 왕복 farming) 교정. **`Indy-Wuji-Chopsticks-Grasp` 신규 생성**.
- **07-14 (단독 큐브 포기 → functional_grasp)**: 근본원인 2가지로 포기·되돌리기.
  **손은 0.30kg 큐브를 들 수 있음 검증**(하드웨어 한계 아님). cage 가상점 손끝 쪽으로, 약지·새끼 커플링(SIH).
- **07-15 (돌파)**: 밤샘 런이 **캠핑 탈출해 리프트 성공** + r_T 구현.

## 핵심 교훈·함정 (이후 task 계승)
- **거리 보상은 파지를 안 만든다** — 만지면 물체가 밀려 거리↑ → 접촉이 손해 → hover만. cage(침투 깊이)가 필요.
- `cube_grasp_env_cfg.py __post_init__`의 **surface_z 배선(★) 삭제 금지** — 없으면 lift가 만점서 시작하는 버그.
- 제어 안 되는 관절(약지·새끼)이 물체를 쳐냄 → 커플링/클램프 필요.
- action space 설계 오류로 정책 발산 → 제어주파수·reward 스케일과 함께 봐야 함.

## 소스 포인터
- `nrmk_isaaclab_wuji/worklog.md`: 2026-07-08~07-15 (검색 "Cube Grasp", "cage", "cube_lift").
- 루트 `WORKLOG.md`: "2026-07-11 Cube Grasp...", "2026-07-12 Cage 12점 확장 + cube_lift 도입".
