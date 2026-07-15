# CLAUDE.md

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

### 동시 작업/도구
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
