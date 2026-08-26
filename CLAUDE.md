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

### 다른 파일로 옮긴 함정 (색인)

아래 주제가 나오면 **파일을 열기 전에** 해당 파일을 먼저 읽을 것. 하위 폴더
`CLAUDE.md` 는 그 폴더 파일을 건드릴 때만 로드되는데, 나는 파일을 열기 전에
추론하다 틀린 적이 있다 (2026-08-07 보상 예산 15배 오산).

**`nrmk_isaaclab_wuji/CLAUDE.md`** — 보상 가중치·예산 계산, `Episode_Reward` 의 의미,
σ 튜닝, 종료 조건 진단 순서, 액션 파이프라인과 세 종류의 clip, `configclass`/hydra/
manager term 규약, 커맨드 term, 키보드 입력, Isaac 측정 함정.

**`Deploy/CLAUDE.md`** — `Deploy/` 층 구조와 임포트 방향, 실측 리그 프레임과 Base 규약
두 종류, 명령 한계·소프트 여백, MuJoCo 젓가락 파지 진단(오진 6회), wujihandpy 실물 SDK,
**성능 진단(헛짚음 3회 — 계측 구멍을 먼저 볼 것)**, 듀얼 카메라 트래커, 슬루 가드.

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

