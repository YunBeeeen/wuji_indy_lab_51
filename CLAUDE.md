# CLAUDE.md

## 기본 행동 강령 (모든 작업에 항상 적용)

아래 두 문서를 이 프로젝트에서의 기본 행동 강령으로 삼는다. 세션 시작 시 함께 로드된다.

@CLAUDE2.md
@범용_행동강령_B.md

- **우선순위**: 안전 규칙 > 아래 `세션 규칙`·`작업 중 막힌 지점` > 위 두 강령 문서 >
  일반 관행. 충돌하면 이 파일의 프로젝트 규칙이 강령보다 우선한다.
- `CLAUDE2.md`의 "프로젝트 정보 — 직접 채우세요" 절은 빈 템플릿이므로 무시한다. 이 프로젝트의
  실제 정보(명령·구조·컨벤션)는 이 파일과 `AGENTS.md`, `CLI.md`를 기준으로 한다.
- `범용_행동강령_B.md`는 `범용_시스템프롬프트_개발결과.md`의 **B절(완성본)만** 떼어낸
  실사용본이다(플레이스홀더는 이 환경 값으로 치환). 원본의 A·C·D절은 참고용이며 로드하지 않는다.
  강령을 고칠 때는 이 실사용본을 고치고, 원본은 기록으로 남긴다.

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
- **잔차 전환 시 `processed_actions`의 의미가 바뀐다**(절대 목표 → 증분).
  `env/managers.py:480`(action_track_err), `scripts/rsl_rl/play.py:687`이 이걸 절대 목표로 읽으므로
  `joint_pos_target`으로 같이 고칠 것. 각 지점에 경고 주석 있음.

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
