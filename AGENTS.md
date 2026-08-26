# AGENTS.md

- 이 문서는 Codex/Claude가 프로젝트 상태와 작업 규칙을 공유하기 위한 인수인계 문서임.
- Codex 자체의 실수/병목 회고는 사용자 연구 기록에 섞지 않고 root `codex.md`에 따로 남김.
- 같은 내용의 핵심 요약은 아래 `Codex 운영 메모` 섹션을 우선 확인함.

## Codex 운영 메모

- **2026-08-25 Box-Transport quaternion orientation reward 복원:** active pose reward를
  8-corner keypoint MAX에서 position/orientation 분리형으로 바꿨다. 검증된 box 성공 run
  `2026-07-21_23-01-25_chopsticks(success)` 기준으로 `cube_transport=8000`, quaternion 내적 기반
  8-대칭 `box_orientation=4000`을 활성화하고 `keypoint_goal=None`으로 active reward
  manager에서 제거했다. 비종료 성공
  보너스도 keypoint 거리 대신 `position<5 cm && orientation<15° && tripod_gate>0.3`을 15 step
  유지하는 `PoseGoalReachedBonus`로 바꿨으며 weight `60000`은 유지했다. grip/lift/action/69D raw
  observation은 불변이다. reward/success 의미 변경이므로 이전 checkpoint resume 금지, fresh 학습만 한다.
- **2026-08-25 Box-Transport raw observation 전환:** `Indy-Wuji-Box-Transport`의 default
  actor observation을 92D engineered 구성에서 **69D raw-pose 구성**으로 바꿨다. 순서는
  `joint_pos(26) + box_pose_w(xyz+wxyz,7) + box_size(3) + goal_pose_w(xyz+wxyz,7)
  + action_history(26)`이다. box/goal은 모두 world frame으로 표기하며 quaternion은 canonical
  `wxyz`다. 병렬 env의 배치 translation은 position에서 제거한다.
  `cube_in_fingertips(15)`,
  `index/thumb/middle_grip_error(9)`, `cube_to_goal(3)`, `box_ori_to_target(3)`은 actor에서
  제거했다. grip reward와 gate는 유지하며 pose reward/success는 위 quaternion 항목처럼 후속
  변경했다. Cube-Grasp는 별도 공용 cfg라
  무영향이다. 이전 default 92D 및 legacy 76D checkpoint는 새 관측과 비호환이므로 반드시 fresh로
  학습한다. `WUJI_LEGACY_ACTION=1`은 이제 action 18 / raw obs 53이며 예전 checkpoint play 복원
  기능이 아니다. 정적 `py_compile`/scoped `git diff --check`만 완료했고 런타임 69D 확인은 사용자 실행이
  필요하다.
- **2026-08-17 MuJoCo Camera2 reset-tail 결정:** 추가 Camera2는 Hand/Palm `+Y` 측에서
  `-Y`를 보는 **하강각 0° 수평 설치**로 고정하고, full-workspace tracker가 아니라 reset의
  tail marker ID0/ID2 확인용 보조 카메라로 범위를 좁혔다. reset marker Base 높이는
  `12.10/12.94 cm`라 candidate optical center를 Base
  `[X,Y,Z]=[0.125,0.200,0.060] m`로 둔다. Camera2 단독 reset ID0+ID2 검출은 PASS지만
  Palm-Z `-90..0°` 5° sweep은 `7/19`이므로 full-range 보장 아님. 비교상 60° 하향·40 cm는
  1° sweep `91/91`이었으나 설치 난이도 때문에 선택하지 않았다. 상세는
  `ACTIVITY_2026-08-17.md`, `Deploy/VALIDATION_REPORT.md` 참고.
- **2026-08-17 최신 Claude 인계:** 먼저 `CLAUDE_HANDOFF_2026-08-17.md`를 읽는다. active
  deploy contract는 105D quaternion history와 Joint1/2 `0.10`, Joint3 `0.20`, Joint4
  `0.15` residual이다. 약한 외란 source `2026-08-16_21-14-45/model_3800.pt`에서 optimizer까지
  이어받은 true-resume A/B를 실행 중이다. strong `0.3~1.2 N` run
  `2026-08-17_02-50-15`는 recovery 0까지 붕괴했다가 latest 9009에서 contact/final
  `5.75/5.63`, recovery `91.4%`로 복구했지만 OPEN/CLOSE `11.95/9.45 mm`다. medium
  `0.1~0.6 N` run `2026-08-17_11-04-35`는 latest 6010에서 contact/final `5.90/5.28`,
  recovery `99.0%`, CLOSE/lateral `2.69/3.07 mm`이나 OPEN `15.50 mm`라 둘 다 기하 판정이
  남았다. MuJoCo는 official fixed-base right hand, name-based mapping, 105D common adapter,
  30/120 Hz scheduler, dynamic sticks, D435/ArUco provider까지 구성됐고 Real backend는 안전상
  비활성이다. ArUco ID0/ID1 pose agreement median은 `4.84 mm/1.89°`, position p90
  `21.78 mm` long-tail이 후속 과제다. 상세는 handoff와 `ACTIVITY_2026-08-17.md` 참고.
- **2026-08-13 `hand_real` Stick1 5 mm A/B:** Stick1 reference/reset을 local `+y`로
  5 mm 이동했으며 pivot local offset `-60→-65 mm`와 axial target 변경은 모두 같은
  기하 이동의 파생값이다. 동일 초기-stage baseline `2026-08-12_10-16-57`과 fresh
  `2026-08-13_17-54-50`은 5 s·고정 회전·외란 OFF·동일 PPO이고 이 5 mm 외에는 active
  설정 차이가 없다. 새 run은 iter 900에서 contact `6/6`, full-contact `1.0`을 달성했으나
  950 이후 geometry가 무너지고 resume `19-38-46`의 iter 1650에는 약 `3.19/6`으로
  붕괴했다. task 불가능이 아니라 좋은 해를 잃은 PPO 후반 붕괴로 판정하며 현재 보존
  후보는 `17-54-50/model_900.pt`다. 상세는 `ACTIVITY_2026-08-13.md`.
- **⚠ 사용자 실행 전용 원칙 (2026-07-31):** 앞으로 Codex는 이 프로젝트에서 Isaac Sim
  smoke test, `train.py`, `play.py`, physics probe를 직접 실행하지 않음. 코드 변경 뒤 검증은
  `py_compile`, 정적 검색/구성 대조, `git diff --check` 같은 **정적 검사까지만** 수행함.
  실제 시뮬레이션·학습·play는 사용자가 실행하고, Codex는 사용자가 전달한 로그와 저장 결과를
  판독함. 사용자가 특정 실행을 이번 작업에서 명시적으로 요청한 경우에만 예외로 함.
- **2026-08-16 `hand_real` 최신 deploy/action 계약:** actor observation은 Stick1/2의
  palm-frame `xyz+wxyz` previous/current를 쓰는 **105D**다. 사각 스틱 local-Y 90° 대칭
  4개 중 pose_005 reference-nearest quaternion을 선택하고 `w>=0`으로 canonicalize하지만,
  reward/success orientation은 계속 `directed_axis`라 특정 shaft roll을 강제하지 않는다.
  task-local current-joint residual scale은 Joint1/2 `0.10`, Joint3 `0.20`, Joint4 `0.15`이며
  `hand_real`/`hand_final`/`hand_play`에 적용되고 공용 `hand_move` all-joint `0.1`은 불변이다.
  Joint4는 사용자가 0.15 rad step 기준으로 조정한 Kp/Kd와 맞춘 값이다. 기존 101D ONNX/
  MuJoCo adapter는 새 105D 모델과 비호환이므로 배포 전에 함께 갱신해야 한다. Stick은 별도
  offset 없는 7×180×7 mm Cuboid라 root/geometric center/CoM이 일치하고 local tip/tail은
  Y축 `+90/-90 mm`; Stick1 5 mm axial A/B는 현재 `0.0`으로 원복됨. 상세는
  `ACTIVITY_2026-08-16.md`.
- **2026-08-10 `hand_setting` Stage-2 contact 최신 상태:** run
  `2026-08-10_18-30-36` iter 1656의 Stage-1 평균/final은 `0.736/0.896`이나
  all-joint max q-ref error 평균/final이 `1.603/1.329 rad`라 strict
  `stage2_ready=(stage1_ready && max error<=5°)`는 0이었다. strict gate는 유지하고
  contact mean/min(weight `5/20`, force scale `0.10 N`)만
  `stage1_ready*clip((0.80-q_RMSE)/(0.80-0.0873),0,1)`로 점진 활성화했다.
  TensorBoard `stage2_contact_progress`를 추가했고 between-sticks 조건은 아직 미적용이다.
  reward 변경이므로 기존 run에 resume하지 말고 필요하면 checkpoint를 `--init_checkpoint`로
  넘긴 새 run을 사용한다. `hand_real`에는 tuner 표의 20관절 Kp/Kd가 task-local로 적용되며
  공용 asset/다른 hand task/effort limit은 불변이다. 상세는 `ACTIVITY_2026-08-10.md`.
- **2026-08-10 `hand_real` sim-to-real 관측 계약:** `hand_move`의 scene/action/reward/
  termination/disturbance를 상속하고 actor observation만 simulator velocity가 없는
  **105D**로 교체했다. 순서는 joint history `[q_(t-1),q_t]` 40 + current fingertip
  palm xyz 15 + Stick1/2 history `[pose_(t-1),pose_t]` 각 14 + 현재 상태를 만든 직전
  action 20 + mode one-hot 2다. Stick pose는 palm-frame `xyz+wxyz`이며 quaternion을
  `w>=0`으로 canonicalize한다. action은 post-step 기준 `action_manager.action`을 쓰며
  `prev_action`은 한 step 더 오래되어 사용하지 않는다. `history_length=2`는 oldest-to-newest,
  reset에서는 두 슬롯이 같은 sample이다. `hand_move` 103D checkpoint와 shape가 달라
  **직접 load 금지/fresh 학습**이다. 실물 bridge는 같은 joint 순서·limit 정규화·30 Hz,
  encoder FK fingertip, vision palm-frame Stick pose를 제공해야 한다. 정적 검사만 완료했고
  런타임 105D/20D 확인은 사용자 실행이 필요하다. 상세는 `ACTIVITY_2026-08-10.md`.
- **2026-08-10 hand action/setting/play 결정:** `hand_move`의 `q_pose005+0.3a` A/B는
  full-contact를 유지했지만 OPEN error 약 16 mm로 CLOSE에 머물러 active action을 다시
  `q_current+0.1a`로 복원했다(고정-reference 블록은 주석 보존). `hand_setting`은 새끼가
  약지 경로를 막지 않게 missing-16 best-so-far에서 새끼 4관절 credit만 `0.25`, 나머지는
  `1.0`으로 두고, Stage-1 unlock 뒤 index/middle→Stick1·ring→Stick2 surface 접근을
  weight 2의 약한 live guide로 추가했다. play는 contact-loss reset만 끄고 물리 drop은
  유지하며, `hand_play`는 OPEN 시작이다. `--plot_hand_contact_forces`는 별도 GUI process에서
  전 손 링크 PhysX net contact-force magnitude를 표시한다.
- **2026-08-09 `hand_setting` 최신 관절별 best-so-far A/B:** obs/action은 `101D/20D`
  그대로다. live gate는
  `stage1_ready=(pair_score>=0.65 && thumb_pivot_score>=0.35)`,
  `stage2_ready=(stage1_ready && all-20 max |q-q_ref|<=5°)`인 memoryless 조건이지만,
  missing-16 best-so-far reward 내부에는 Stage 1 최초 통과를 기억하는 one-way unlock이 있다.
  이전 통합 score(`0.5*mean+0.5*min`)는 한 joint 개선이 worst joint와 16-joint 평균에
  희석됐으므로, 현재는 각 joint에
  `s_j=clip(1-|q_j-q_ref_j|/2.5,0,1)`과 독립 `best_j`를 두고
  `raw=mean(max(s_j-best_j,0))`을 지급한다. unlock 순간은 현재값으로 seed해 0점이다.
  mean reduction이라 weight `3000`과 30 Hz 최대 episode 예산 약 `100`은 유지됨.
  `2026-08-09_22-51-45` 통합-score run은 약 550 iter에서 이전 대비 RMSE
  `1.13→0.96 rad`, max error `2.11→1.83 rad`, middle RMSE `1.21→0.83 rad`로
  개선됐지만 `stage2_ready=0`이라 중단 비교 대상으로만 사용한다. 다음은 reward 변경 때문에
  resume하지 않고 fresh run. Stage-2 contact mean/min은 현재 cfg에서 여전히 주석 상태다.
  TensorBoard에는 기존 aggregate와 함께 16 joint 각각의
  `*_current_linear_score`, `*_best_linear_score` 32개를 기록한다. 상세는
  `ACTIVITY_2026-08-09.md`의 후속 변경 3 참고.
- **2026-08-06 `hand_object` 신설 — 인계 시 반드시 알 것:**
  `hand_move` 정책을 fine-tuning해 1cm 큐브를 실제 접촉력으로 집고 유지하는 태스크.
  task ID는 `hand_object` **하나**이고 train/play/calibration이 전부 여기서 갈린다
  (세 command `open_close`·`root_orientation`·`support`의 manual override + `--manual_root`).
  - **obs 103D / action 20D는 절대 건드리지 말 것.** 이게 유지돼야 hand_move 체크포인트가
    로드된다. 큐브·접촉력·phase를 actor observation에 추가하면 태스크 전제가 깨진다.
  - **손은 이동+회전한다.** 처음엔 yaw 만 돌리고 root 위치는 스폰 고정으로 만들었는데,
    그러면 tip 이 팜 주위로 호만 그려 큐브에 도달하지 못한다. `position_command_name` 으로
    위치도 커맨드 궤적을 따르게 했고, **미설정이 기본이라 `hand_move` 는 무영향**.
    궤적은 2단계다 — 스폰 높이에서 방향·x,y 정렬(앞 50%) → 순수 수직 하강(뒤 50%).
    경유점은 목표 z 만 스폰 z 로 바꾼 파생값이라 상수가 아니다.
  - **보정 완료(2026-08-06)**: `HAND_OBJECT_TARGET_ROOT_POS_E`, `..._TARGET_EULER_RAD`,
    `..._CUBE_POS_E`, `..._SUPPORT_POS_E` 전부 측정값이 들어가 있다.
    다만 **`HAND_OBJECT_FORCE_SATURATION_N=0.05` 는 아직 PROVISIONAL** — 압착은 0.167N 까지
    나오지만 "버티는" 힘은 hold 가 생겨야 잴 수 있어 미확정. 임의로 확정하지 말 것.
  - **런타임 검증 통과**(`scripts/debug/hand_object_probe.py`, 검사 20개).
    hand_move 체크포인트 actor `103->128->128->20` 로드 확인(fine-tuning 호환),
    센서 shape `(N,1,1,3)`, **force 부호 실측 확인**(압착 642회 전부 양수, 음수 0),
    초기 충돌 없음(손 링크 57.7mm), 2단계 궤적 이탈 0.000mm, retract 스텝당 1.333mm.
  - **커리큘럼**: 큐브 보상 3개는 cfg 기본 ON. 1단계(파지·OPEN/CLOSE 안정화)는 CLI 로
    `env.rewards.<term>.weight=0.0` 을 줘서 끈다. 단계 전환은 보상이 바뀌므로 `--resume` 이
    아니라 `--init_checkpoint` 를 쓸 것(★재학습 원칙).
  - **`bilateral_cube_force`(예산 450)는 젓가락 파지 유지에 게이트가 없다.** 파지를 버리고
    큐브만 짓눌러도 손익이 450 vs 490 이라 여유가 8% 뿐. `functional_contact_count` 가
    5.5 아래로 내려가면서 `bilateral_force_score` 가 오르면 그 거래가 일어나는 중이니 100→50.
  - 상세는 `hand_object_log.md`, 날짜별은 `ACTIVITY_2026-08-06.md`, API 함정은 root `CLAUDE.md`.
- Codex 실수/병목 상세 기록은 `codex.md`에 둠. `WORKLOG.md`, `ACTIVITY_*.md`, `study.md`, `thesis.md`에는 Codex 시행착오를 섞지 않음.
- **2026-08-03 현재 `hand_setting` Joint4 A/B (아래 07-31 설정값보다 우선):**
  baseline `2026-08-03_00-13-19`은 전 관절 current-joint residual scale `0.1`이고,
  fresh `09-37-49`는 Joint1~3 `0.1`, 다섯 Joint4만 `0.3`으로 바꿈. reward·obs
  `101D`·action `20D`·episode `8 s`·PPO는 불변임. `10-04-47`은
  `09-37-49/model_200.pt`부터 이어진 resume run이며 두 env cfg는 출력 경로 외 동일함.
  active gate는 `pair_score>=0.65 && thumb_pivot_score>=0.35`이고,
  all-20 `stage1_joint_reference`는 weight `2`/sigma `0.8`, Index/Middle/Ring 12-joint
  `stage1_missing_joint_reference`는 weight `6`/sigma `0.8`임. missing guide는
  `6*(1-min_i clamp(F_i/0.02N,0,1))`로 6개 접촉이 모두 `0.02 N`일 때만 0이 됨.
  모든 손가락 Joint4의 action/target/actual/tracking/reference-error metric을 추가했으며
  이는 학습 의미를 바꾸지 않는 진단-only 변경임. `10-04-47` iteration `1496`에서
  Stage-1 평균/final `0.639/0.874`, contact 평균/final/max `2.675/2.743/2.996`임.
  Index Joint4는 `+0.394 rad`까지 개선됐지만 Middle/Ring Joint4 actual은 거의 0이고
  max contact는 3에 정체했으므로, Joint4 scale 확대는 일부 dead zone만 해소했고
  6-contact 병목은 미해결로 판정함.
- **2026-07-31 `hand_setting` Stage-1 gated functional-contact A/B:** 직전
  Stick2-reference-only A/B를 paired-reference + thumb-pivot으로 확장함.
  `two_stick_reference_min=12`는 `pose_005` 기준 Stick1/2 palm-relative
  full-pose exponential score의 `min`을 사용하고,
  `reference_thumb_pivot_min=8`은 그 pair score와
  `finger1_link3`의 Stick1 local `y=-60 mm` pivot-station 접근 score의
  `min`을 사용함. 최종 Stick pose/contact/speed 성공 조건은 변경하지
  않음. position/orientation/thumb scale은 각각
  `0.10 m/90°/20 mm`임. 따라서 한 stick만 맞추거나 엄지만 pivot을
  추종해서 보상을 파밍할 수 없음. 별도 drop penalty는 넣지 않고 기존
  Stick1/2 world-z `<0.40 m` 즉시 종료와 남은 양의 보상 상실만 사용함.
  TensorBoard `Metrics/hand_setting{,_final,_min,_max}`에는 reward와 같은
  계산을 쓰는 `thumb_pivot_distance`(m)와 `thumb_pivot_score`(0~1)를
  추가함. 이후 memoryless gate를
  `pair_score>=0.50 && thumb_pivot_score>=0.35`로 정의하고 gate 뒤에
  `stage1_joint_reference`는 max weight `8`(sigma `0.80`)이며, 여섯
  semantic force의 weakest-contact progress
  `m=min_i(clamp(F_i/0.02,0,1))`에 따라 effective weight가
  `8*(1-0.75m)`, 즉 **접촉 전 8 → 6개 모두 0.02 N일 때 2**로
  memoryless 감쇠함. 접촉을 잃으면 자동으로 다시 강해짐.
  `stage1_contact_mean=5`, `stage1_contact_min=20`을 활성화함.
  region reward는 사용하지 않음. strict `FunctionalSettingHeld` 성공에는
  `30000` terminal reward를 연결해 성공 종료가 미래 보상만 끊는 역유인이
  되지 않게 함. TensorBoard에 `stage1_pair_score`, `stage1_ready`도 추가함.
  나머지 region/completion/stability/action-rate reward는 cfg에서 주석 처리됨.
  strict `stick2_in_valley` metric/gate는 Stick2 pose error
  `<=15 mm/20°`와 palm/thumb-link2 force 각각 `>=0.02 N`임. 기존
  finite-shaft `-60 mm` 코드는 재현용으로 보존하되 reward/gate/metric에서
  사용하지 않음. TensorBoard는 `stick2_valley_pose_valid`,
  `stick2_in_valley`를 기록함. 비활성 reward에 종속되던 region score와
  loose `stick2_valley_support_ready` tag는 제거했지만 독립적인 force,
  pose error, shaft-region-valid metric은 유지함.
  직전 reference-only smoke는
  `logs/rsl_rl/hand_setting/2026-07-31_11-59-59`이며, 현재 변경은 사용자
  실행 전용 원칙에 따라 `py_compile`, 정적 검색, `git diff --check`만
  통과시킴. objective가 바뀌었으므로 이전 hand_setting checkpoint resume
  금지, fresh run으로 시작함.
- **2026-07-31 `hand_setting` spawn feasibility reset:** 초기 Stick2가 열린 엄지의
  pivot 경로를 막지 않도록 이 task의 절대 reset world position을
  Stick1/2 각각 `(0.075,0,0.5195)` / `(0.055,0,0.5195)`로 둠.
  이는 최초 검증 때 inherited scene spawn을 world `+x`로 `20 mm` 이동한
  결과와 같지만, 현재 코드는 offset 계산 없이 절대 좌표를 직접 기록함. 기존
  `20 mm` pair separation과 orientation은 유지함. 손은 preload 없이
  `finger1_joint2=-0.1659 rad`, 나머지 19관절 `0`에서 시작함.
  1-env 5초 zero-action probe에서 초기 6 semantic force는 모두 `0 N`,
  5초 뒤 Stick1/2 palm position은 약
  `(4.69,0.05,83.97)` / `(11.80,-0.02,55.80) mm`, 최저 world z는
  `0.5046/0.5118 m`로 drop threshold `0.40 m` 위였음. Stick2는 palm에만
  약 `0.095 N`으로 안착하고 thumb-mid force는 `0 N`이었음.
  `hand_grasp` shared spawn은 변경하지 않았고, valley target/reward 정의도 이번
  배치 A/B에서는 변경하지 않음. spawn/reset이 달라졌으므로 이전 hand_setting
  checkpoint에 resume하지 않고 fresh run으로 판정함.
- **2026-07-30 `hand_grasp` play 수동 모드 선택:** 주 사용법은
  `scripts/rsl_rl/play.py --keyboard_hand_mode`이며 실행 중 숫자 `1=OPEN`, `2=CLOSE`임.
  Isaac/Carb viewport callback은 숫자를 thread-safe queue에만 넣고 simulation thread가 command one-hot과
  observation을 갱신하므로 다음 policy action부터 즉시 반영됨. OPEN으로 시작하며 stick drop 등으로
  env가 reset되어도 마지막으로 선택한 모드를 다음 observation 전에 복원함.
  GUI viewport에 포커스를 둔 상태에서 사용하며 `--headless`와 함께 쓰면 CLI error로 막음.
  보조 고정 CLI `--hand_mode {open,close}`와 자동 교대 `--alternate_hand_mode`도 보존하되
  세 방식은 상호 배타적임. 수동/고정/교대 사용 시 mode·target·actual gap을 자동 출력함.
- **2026-07-30 `hand_grasp` 성공 기준선과 axial A/B:** 현재 가장 잘 된 OPEN/CLOSE run은
  `2026-07-30_12-21-21`이며 전체 run을
  `backups/runs/hand_grasp/2026-07-30_12-21-21_success/`에 보존함. 축 보완 전 소스는
  `nrmk_isaaclab_wuji/backups/hand_grasp_pre_axial_2026-07-30/`에 있음.
  이 run은 CLOSE `0 mm`, 공통 gap tolerance `0.5 mm`, episode/dwell `10 s/2 s`,
  obs/action `103D/20D`, entropy `0.001`임. 이후 axial A/B는 별도 penalty를 추가하지 않고
  기존 OPEN/CLOSE 결합 커널을
  `exp(-e_gap/5mm-e_lateral/5mm-e_axial/5mm)`로만 확장함.
  `e_axial=|dot(p_tip1-p_tip2, y_stick2)-(-4.8016 mm)|`이며 성공 run에는 이 항이 없음.
  `tip_axial_offset/error` metric만 추가됐고 obs/action/success hard gate는 불변임.
  새 실험은 성공 run에 resume하지 말고 fresh로 시작함.
- **2026-07-30 Phase 2→3 `hand_setting` 최초 구현:** 별도 subphase/FSM 없이 열린 손에서
  정렬된 두 dynamic stick을 `pose_005` functional grasp로 만드는 단일 단계 task임.
  `HandGraspEnvCfg`의 scene/6개 contact sensor/20D current-joint residual action/physics를
  재사용하되 OPEN/CLOSE command를 제거해 obs는 **101D**임. 당시 reset은 world
  `x=0.055/0.035, y=0, z=0.5195`의 평행 stick과 열린 손(thumb joint2 `-0.1659`)이며
  state와 PD target을 같게 써 preload를 넣지 않음. reward ladder는 broad final-pose/q prior,
  네 link의 central-shaft semantic proximity, contact mean(첫 접촉부터), contact min(6/6),
  pose·region-gated completion/stability 순서임. mean 항은 같은 총 weight 5를 여섯 개
  개별 tag(weight 5/6씩)로 분해해 빠진 접촉을 TensorBoard에서 바로 볼 수 있음.
  성공은 정확한 6 link–stick pair가
  `0.02 N` 이상이고 네 좁은 접촉이 중앙 160 mm shaft 안에 있으며 Stick1/2 palm pose와
  palm-relative speed 조건을 30 policy step 유지하는 것임. 성공 즉시 종료함.
  최종 smoke run `logs/rsl_rl/hand_setting/2026-07-30_17-05-31`에서 101D/20D,
  command 0, reward 18개, termination 4개와 24 step을 확인함.
  `hand_grasp`는 수정하지 않았고
  hand_setting은 기존 103D checkpoint와 호환되지 않으므로 fresh run만 사용함.
- **2026-07-30 `hand_setting` TensorBoard metric:** `CustomRewardManager`가
  `Metrics/hand_setting{,_final,_min,_max}/*`를 기록함. 각 family에는 exact 6-pair force와
  contact count/fraction/full-contact, 네 central-shaft region score와 valid count,
  Stick1/2 palm-frame position/orientation error와 pose-valid, palm-relative
  linear/angular speed, 최종 `setting_valid`, `success_stable_steps`가 포함됨.
  base는 에피소드 중 시간 평균, final은 종료 직전 값, min/max는 에피소드 내 극값임.
  smoke `logs/rsl_rl/hand_setting/2026-07-30_17-26-48`에서 32개 지표 × 4 family =
  TensorBoard scalar 128개가 event 파일에 실제 기록됨을 확인함.
  실행 중이던 Python 학습 프로세스에는 이 변경이 반영되지 않으므로 metric을 보려면 프로세스를
  재시작해야 함. obs/action/reward/PPO는 바뀌지 않았으므로 같은 hand_setting 최신 checkpoint
  resume는 호환되며, metric 비교를 깔끔히 분리하려면 새 run 폴더를 사용함.
- **2026-07-30 `hand_setting` contact-only A/B:** fresh run
  `2026-07-30_17-31-40`은 약 333 iter에서 mean reward `144.7`로 정체했지만
  final functional contact는 약 `1/6`, index/middle/ring force와 full-contact,
  pose-valid, setting-valid, success는 계속 0이었음. policy가 joint/Stick pose reference와
  broad shaft region만 먹는 local optimum으로 판정함. 다음 fresh run은 새 항을 추가하지 않고
  기존 joint reference, Stick1/2 reference, 네 region, pose/region-gated
  completion/stability의 weight를 모두 `0`으로 둠. 여섯 per-contact saturated term
  (각 `5/6`)과 hard contact-min `20`, action-rate 및 기존 strict final success만 유지함.
  따라서 positive dense shaping은 contact-only이며, strict success는 원래 pose/region/speed
  조건을 우연히 함께 만족했는지 검증하는 terminal validator로만 남음. reward가 바뀌었으므로
  `17-31-40` checkpoint resume 금지, fresh run으로 시작함.
- **2026-07-30 `hand_setting` missing-tip approach A/B:** contact-only fresh run
  `2026-07-30_18-14-08`은 744 iter/약 89분에도 final contact `2.85/6`,
  episode max `3/6`, contact-min/full-contact/success 전부 0이었음. 붙은 항은
  thumb-distal–Stick1 약 `1.9 N`, palm–Stick2 약 `0.10 N`,
  thumb-mid–Stick2 약 `0.10 N`; index/middle/ring은 최대도 각각
  `0.0005/0.0009/0 N`이라 contact-only의 pre-contact 방향 신호 부재로 판정함.
  새 항은 만들지 않고 기존 region 중 이미 해결된 thumb는 `0`, 빠진
  index/middle/ring만 기존 `2`의 1/4인 `0.5`씩 켬. joint/Stick reference와
  completion/stability는 계속 `0`, six contact/contact-min/success/action-rate는 유지함.
  objective 변경이므로 `18-14-08`도 resume하지 않고 fresh run으로 비교함.
- **백업 정리 기준:** root `backups/README.md`를 인덱스로 사용함.
  단일 코드 snapshot은 `backups/code/<task>/`, 완전한 학습 run은
  `backups/runs/<task>/`, 여러 source가 한 세트인 snapshot은
  `nrmk_isaaclab_wuji/backups/<task>_<purpose>_<date>/`에 둠.
- **⚠ 2026-07-27 Wuji self-collision 물리 변경:** standalone `WUJI_RIGHT_CFG`와 결합형
  `INDY7_WUJI_RIGHT_CFG` 모두 `palm_link ↔ finger1~5_link2` 다섯 structural pair를 제외하는
  non-destructive USD overlay를 사용함. 이 변경은 Box Transport·Chopstick을 포함한 모든 이후
  Indy+Wuji run의 손가락 도달 범위와 접촉 dynamics를 바꿈. **변경 전 checkpoint를 본 실험에
  resume하지 말고 성능 판정은 fresh run으로 시작함.** 이전 성공 run은 구 물리 baseline으로만 보존함.
- `Stage1`, `Easy`, `Hard`처럼 task/run 이름을 늘리면 checkpoint 선택과 해석이 꼬임. 현재 주 실험은 `Indy-Wuji-Box-Transport`이고 `Indy-Wuji-Cube-Grasp`는 reward 테스트베드임. 새 alias를 늘리지 말고 확인한 run 폴더명을 명시함.
- **2026-07-30 hand_grasp 6-contact 보존:** run `2026-07-29_21-05-32`는
  1500 iter의 `5.97/6`, full-contact `96.9%`에서 3100 iter 이후 약 `3/6`, success 0으로
  전체 정책이 붕괴함. active reward는 `c_i=clip(F_i/0.02,0,1)`의 mean weight 10 +
  minimum weight 40이며, OPEN/CLOSE gap+lateral 보상은 `min(c_i)`로 gate함.
  6/6을 0.02 N 이상 5 step 획득한 뒤 0.01 N 이상 접촉이 4개 이하로 10 step 지속되면
  `functional_contact_lost` 종료함. 5/6에는 복구 기회를 남김. success/entropy/action/obs는
  각각 `30000/0.001/20D/103D` 유지. 변경 전 backup은
  `backups/hand_grasp_pre_contact_gate_2026-07-30/`, smoke는
  `logs/rsl_rl/hand_grasp/2026-07-30_09-32-09`. 다음 판정은 fresh run으로 함.
- **2026-07-30 hand_grasp lateral 분리:** OPEN/CLOSE gap shaping은 이제
  `contact_gate*exp(-gap_error/5mm)`로 gap만 평가함. Lateral은 mode/contact 독립 bounded
  penalty `-5*(1-exp(-lateral_error/5mm))`로 분리해 접촉을 끊어 감점을 지우지 못하게 함.
  `mode_grasp_stability`의 결합 gate와 success의 lateral `<=5mm` hard 조건은 유지함.
  backup은 `backups/hand_grasp_pre_lateral_split_2026-07-30/`, smoke는
  `logs/rsl_rl/hand_grasp/2026-07-30_10-28-30`; 다음 학습은 fresh run으로 함.
- checkpoint load 전에는 action dim, observation dim, reward cfg, env cfg를 먼저 확인함. shape가 다르면 `runner.load()`에서 size mismatch가 남.
- scene 배치는 학습 전에 눈과 probe로 확인함. cube 높이, support/table 높이, hand 시작 높이, palm/cage 방향을 확인하지 않고 reward만 조정하지 않음.
- TensorBoard 평균 `Metrics/cube/*`만으로 판단하지 않음. `Metrics/cube_final/*`, `cube_clearance`, `cage_inside_frac`, contact, `cube_speed`를 같이 봄.
- `palm_facing`은 초기 방향이 맞는지 검증하기 전에는 끄지 않음. 절대형 양수 facing reward는 farming 위험이 크므로 차분형 또는 gate로만 사용함.
- `cube_lift` weight를 키우기 전에 raw lift가 실제로 발생하는지 확인함. raw lift가 0이면 weight를 키워도 신호는 0임.
- 2026-07-20 Box Transport 성공은 goal position error `<0.05m`, cage gate `>0.6`, symmetry-aware orientation error `<15deg`, `hold_steps=15` 기준임. play에서 episode가 빨리 끝나면 time-out이 아니라 `success` termination일 수 있음.
- `success`는 종료 조건인 동시에 `transport_success` reward의 입력임. 현재 weight `30000`, `dt=1/30`이라 성공 순간 `+1000`을 한 번 받고 즉시 종료함. 0.5초 유지 후 이탈/재진입으로 terminal reward를 반복 적립하는 것을 막는 구조임.
- 성공 후 장기 유지 play는 `success=null`로 지우지 말고 `env.terminations.success.params.hold_steps=1000000`으로 덮어씀. 그러면 기존 8초 `time_out`과 `cube_dropped`는 유지됨. `success=null`만 쓰면 `transport_success(term_keys="success")` 참조가 끊겨 env 생성이 실패함.
- 자세가 아쉬운 lift는 height를 더 키우기보다 stable lift 조건을 봄: `cube_speed`, contact group, stricter cage gate, 유지 시간.
- `play.py --print_contact`나 joint detail을 interval 1로 켜면 GUI가 매우 느려짐. 기본 진단은 `--print_diagnostics --print_action_interval 10` 정도로 시작함.
- 긴 학습 전에 scripted probe로 `GOOD_CONTACT`, `max_clearance`, cage/contact 유지 여부를 먼저 확인함.

## 연구 아키텍처와 이어서 작업할 기준 (2026-07-20 — 이 절을 최우선으로 읽을 것)

- 아래 구조가 현재 연구 방향의 canonical handoff임. 아래의 오래된 `현재 상태`, cube-only reward,
  lift/transport A/B 절과 충돌하면 이 절을 우선함. 단, 실행 시점의 실제 값은 항상 현재 코드와 해당
  run의 `params/env.yaml`로 다시 확인함.
- 상세 진행 과정과 문제 해결 근거는 `MIDTERM_REVIEW_2026-07-20.md`, 논문 reward 원문 정리는
  `thesis.md`를 참고함.

> **2026-07-22 업데이트 (chopstick 운반 이식 + orientation 결합형 + 지표 교훈)**
>
> - **`Indy-Wuji-Chopsticks-Grasp`가 이제 acquire 전용이 아니라 world goal pose 운반+자세 매칭까지 함.**
>   box-transport 구조를 이식: `cube_goal`(7D pose, yaw 45°, +20cm), obs `stick_to_goal`+`stick_ori_to_target`
>   (**obs 66→72**), 성공 종료를 `tool_ready`(functional grasp)→`success`(`BalancedObjectAtGoalHeld`, goal 도달)로
>   교체. `cube_lift` weight 4000→150·lift_height 0.08→0.20(4000 연금이 transport 압도해 호버 방지). 상세는
>   `ACTIVITY_2026-07-22.md`.
> - **orientation 보상은 box의 sticky latch가 아니라 per-step 결합형**(`BalancedObjectOrientationCoupledReward`):
>   매 스텝 `pos<3cm & gate>0.3`일 때만 지급 → goal 벗어나 자세만 맞추는 드리프트 0원. **(c안 2026-07-22)**:
>   `best_error`를 goal 안·밖 모두 갱신(지급은 안에서만)해서, **밖에서 회전을 개선해 best에 넣었다 재진입 때
>   인출하는 왕복 파밍(진동)까지 차단**. 이제 개선 회전을 goal 안에 머문 채 해야 지급됨.
> - `goal_proximity`(gate×φ 위치 연금)도 chopstick에 이식, **weight 0**(필요 시 켬). 드리프트를 단일부호로
>   막는 보완 장치(캠핑 상한은 success 종료).
> - **지표 교훈 (매우 중요)**: `box_ori_error` **평균/max는 목표가 아님.** 진짜 목표는 **success rate**와
>   **ori<15°(0.26) 임계를 넘는 env 비율.** 평균을 낮추는 것 ≠ success — 모두 20°로 모여도 임계(15°)를
>   못 넘으면 success 0. 아래꼬리가 임계를 넘나(중앙값/percentile)를 볼 것. mean은 spread에 오염됨.
> - **min/max 큰 스프레드의 원인은 대칭이 아님**(error 메트릭이 이미 최근접 대칭으로 처리). 88°는 스틱
>   **장축이 goal 방향에서 실제로 어긋난 것**(잘못된 yaw / 경첩 기울임) = **파지 방향 복권 + 도달성** 문제.
>   ⚠ "대칭 자유도를 줄인다"는 예전 제안은 오판이라 철회함(대칭 파지는 같은 물체 자세를 만듦).
> - 다음 병목: **잡은 스틱의 장축을 위치 유지한 채 goal 방향으로 둘 수 있나(도달성)**. 안 되면 (c)·연금·
>   커리큘럼 무엇도 max를 못 낮춤 → reachability 디버그(IK 스윕/play)가 결국 필요.
> - **tip/tail 4-대칭 (2026-07-22)**: 젓가락은 tip(+y)≠tail(-y)이라 orientation 대칭을 box의 8개에서
>   **길이축 회전 4개로 좁힘** (chopstick 전용 `stick_tip_ori_error`/`stick_ori_error_nearest_sym`,
>   reward·success·obs 전부 사용). box 공유 함수(square_prism 8-대칭)는 그대로. 이유: 8-대칭은
>   tail-forward를 정답 인정 → 젓가락엔 틀리고 정답이 둘로 갈려 수렴 모호. **`box_ori_error` metric도
>   2026-07-22 4-대칭으로 맞춤**: managers.py가 `stick_orientation` 항 유무로 chopstick을 감지해
>   `_SQUARE_PRISM_Y_SYMS_TIP`(4개)를 씀. box(`box_orientation`)는 8-대칭 유지 — `square_prism_ori_error`에
>   optional `syms` 파라미터 추가(기본 8, box 무영향). `orientation_stage_active`도 stick_orientation을
>   읽게 고쳐 chopstick에서 작동.
> - **[계획] 젓가락 프록시를 사각뿔대(frustum)로**: 현재 균일 box(2×2×18)라 사람이 play에서 tip/tail을
>   구분 못 함. 한쪽 단면이 작은 **사각뿔대**로 바꾸면 tip/tail이 시각·물리적으로 구분돼 4-대칭과 일관되고
>   눈으로 orientation 검증 가능. ⚠ 단 box SDF(`_box_signed_distance`)가 박스 전용이라 cage/grip SDF를
>   뿔대용으로 개편해야 함(비자명). **나중 작업.**
> - **말단 정밀 수렴 지수 커널 (2026-07-22, chopstick, weight 0)**: 선형 보상은 말단 당김이 약해서
>   `exp(-e/σ)`(0에서 기울기 최대)를 얹음. `FineProximityReward`(σ_d=2cm)·`FineOrientationReward`
>   (σ_θ=10°, pos<3cm 게이팅+(c)), best-so-far. **pos·ori 분리**(곱커널 X — 한 축 막혀도 다른 축 refine,
>   진단성). 핵심: **σ<success 임계**(임계 안쪽 refine), 너무 뾰족하면 sparse-like라 넉넉히 시작→축소.
>   coarse(φ+lift) 유지 = 모양의 사다리. 튜닝 후 weight 켤 것.
> - **페널티 3종 (2026-07-22, box·chopstick 양쪽)**: `end_effector_speed`(-0.001, palm 선속도),
>   `action_rate`(-0.005→-0.001), `action_second_rate`(-0.0001, **활성화**). ⚠ box도 reward 변경 → 다음 run fresh.
> - **CustomActionManager 배선 (2026-07-22)**: `action_second_rate_l2`가 쓰는 `prevprev_action`이 표준
>   ActionManager엔 없어 broken이었음 → `rl_task_custom_env.py:load_managers`에서 표준 ActionManager를
>   `CustomActionManager`로 교체(배선). 이제 action 2차 rate 사용 가능. (전 태스크 공통 영향 — env 클래스 변경.)
> - **SimToolReal keypoint를 box에 구현 (2026-07-22) — chopstick과 A/B**: SimToolReal(2602.16863) Eq.2
>   `r=gate×max(d_best−d,0)`, `d=8-corner max 거리`(대칭최소). **max라 모든 꼭짓점이 가까워야(pos·ori 동시,
>   roll 포함)** d 줄어 구조적 결합. box_mdp_cfg에 `KeypointMaxGoalProgressReward`/`KeypointMaxAtGoalHeld` 신설,
>   `keypoint_goal`(w8000) 활성 + cube_transport·box_orientation·**goal_proximity 0** + success→keypoint(eps 0.05).
>   rewards.py `square_prism_keypoint_goal_distance`에 `reduce="max"` + `symmetry="square_prism_y_tip"`(roll 4-대칭,
>   flip 제외) 추가. ⚠ **2점(tail/tip)은 정사각 스틱에 under-constrained(roll 90°만 대칭이라 45° roll 성공처리)라
>   철회 → 8-corner full-pose 사용.** `_stick_tail_tip_pose_error`는 원형 스틱용 보존.
>   **A/B: box=8-corner full-pose keypoint(roll 4-sym) ∥ chopstick=쿼터니안 4-sym+exp.** 둘 다 full pose(roll 포함) 요구.
>   ⚠ standard PPO라 탐색 병목 가능(논문은 SAPG) → 막히면 goal 커리큘럼. box 다음 run fresh, obs 73 불변.

> **2026-07-23 업데이트 (chopstick obs 75, 커널 활성화, A/B 중간 판독)**
>
> - **chopstick policy obs 72 → 75.** 사용자가 중지(middle)에도 semantic grip region을 추가함
>   (엄지 +x의 대향면 local **−x**). `target_grasp.py`에 `middle_grip_error_b`/`middle_grip_error`,
>   `chopstick_mdp_cfg.py`에 `MIDDLE_CFG`·`MIDDLE_GRIP_REGION`·관측 3D·리워드 `middle_grip`(w40).
>   → 엄지·검지·중지 tripod 전부가 명시적 surface region을 가짐. **72D 체크포인트 resume 금지.**
>   ⚠ `ChopstickAcquireObservationsCfg` 독스트링 헤더는 아직 "72"(항목 나열은 75가 맞음).
> - **현재 chopstick 가중치(=라이브 런 12-13-23)**: stick_transport **6000**(eps 0.08),
>   cube_lift **300**(lift_h **0.15**), fine_position **3000**(σ **0.05**), fine_orientation
>   **1500**(σ **15°**), stick_orientation 4000(act_d 0.05), middle/index/thumb_grip 40, hold 15,
>   reach 8, transport_success 30000, 페널티 3종. box는 keypoint_goal 8000 + cube_lift **500**(0.10).
>   ⚠ **지수 커널 σ가 success 임계와 같아짐**(σ_d 0.05 = goal_radius, σ_θ 15° = ori_limit).
>   원 설계는 "σ < 임계"(임계 안쪽 refine)이고 코드 주석도 아직 옛 값(2cm/10°) — 다음 판정 때 확정할 것.
> - **07-23 A/B 중간 판독**: 둘 다 success 0이고 보상 대부분이 `cube_lift`. box(10-07-35, ~2350it)는
>   `keypoint_goal` raw≈0 + error_pos **0.43** = 잘 들지만 goal 아닌 쪽(keypoint 미발화).
>   chopstick(12-13-23, ~1150it)은 **orientation_stage_active final 0.976** = 거의 모든 env가 goal
>   5cm+gate에 한 번은 진입 → **15-11-01의 위치 도달 병목은 뚫림**. 남은 병목은 ori 평균 21.9°
>   (min 1.3°/max 62°)와 진입 후 이탈(error_pos 평균 0.205).
> - **판정 공백 (다음 1순위)**: ori<15° **비율** 지표가 없어 "전부 22°"와 "절반 5°/절반 40°"를 구분
>   못 함. `env/managers.py`에 15°미만 비율 + (pos<5cm & gate & ori<15°) 동시 충족 비율을 추가할 것.
>   학습 중 파일 수정은 이미 로드된 런에 영향 없음. 상세는 `ACTIVITY_2026-07-23.md`.

> **2026-07-23 (저녁) 업데이트 — 액션 잔차 전환 + 가중치 전면 재조정**
>
> - **chopstick 액션이 잔차(residual)로 바뀌고 약지·새끼 커플링이 해제됨.**
>   `CustomResidualJointPositionAction`(`target = 현재 관절각 + action*scale`, 뉴로메카 예제 구조),
>   scale 팔 0.3 / 손가락 0.3, `clamp_to_limits=True`. `joint_names`가 `finger[1-5]` →
>   **action 18 → 26, policy obs 75 → 91.** ⚠ 91인 이유: `joint_pos`(18→26)뿐 아니라
>   **`action_history`(=`prev_action`)도 액션 차원을 따라가서** +8이 두 번 붙음. 83으로 세면 틀림.
>   **07-23 이전 체크포인트 play는 `WUJI_LEGACY_ACTION=1`** (env_cfg가 절대형+mimic으로 분기).
> - **weight 해석 규칙 (매우 중요)**: **gate형 weight 1 ≈ progress형 weight 300.**
>   gate형(`finger_cage_hold`·`cube_lift`·`goal_proximity`)은 240스텝 지급이라 총액 `2.4W`,
>   progress형(`stick_transport`·`stick_orientation`·`keypoint_goal`·`*_grip`)은 best-so-far
>   텔레스코핑이라 에피소드당 **1회분 예산** `Δ×0.3×W/30`. **항 간에 weight 숫자를 직접 비교하지 말 것** —
>   transport 6000(48점)이 lift 150(342점)보다 작았던 게 이 때문.
> - **lift는 W와 W/h가 서로 다른 것을 정한다**: `W` = 호버 연금 총액(`2.4W`),
>   `W/h` = 바닥에서 떼는 기울기(`gate×W/(h·30)`). (150,0.20)=7.5/m에서 "못 들고 테이블에서 비빔"
>   실측. 07-20 A1이 잘 되던 값은 (100,0.08)=12.5/m. → chopstick (100,0.10), box (150,0.20).
> - **현재 가중치**: chopstick lift 100/h0.10, transport **30000**, orientation **30000**,
>   goal_proximity **75**, fine_position 3000, fine_orientation 1500, success **60000(+2000)**.
>   box lift 150/h0.20, keypoint_goal **150000**, success **60000**, `middle_grip` 40 신규(obs 73→**76**).
> - **box keypoint의 `d`는 정규화 안 된 미터** — 초기 d≈0.212, Δd≈0.162 → 예산 `0.00162W`.
>   8000이면 **13점**뿐이라 5750 iter 내내 raw 0.0001이었던 것. 가중치 문제가 아니라 예산 부재였음.
> - `goal_proximity`: chopstick은 **75로 켬**(transport가 best-so-far라 "머무는 힘"이 0인 것 보완),
>   box는 **0 유지**(위치-only 연금이 keypoint의 pos·ori 융합을 깎음).
> - **대향(opposition) gate는 구현 후 철회** — grip 자세는 Skill A 필수가 아니라는 사용자 판단.
>   "얹어 나르기"(thumb_index_opposition −0.28, cage_span 9.7cm vs 스틱 2cm, 손끝이 표면에서 3~5cm)는
>   **미해결**. 잔차로 손가락 dead zone이 사라져 자연 개선될지 `opposition`·`cage_span`으로 볼 것.
> - ⚠ 이번 라운드는 **단일변수 A/B가 아님**(액션·목표항·lift 전부 다름). 패키지 비교로만 읽을 것.
> - ⚠ `stick_orientation` 30000은 progress clamp 상한 1.0 때문에 **한 스텝 최대 300점** 스파이크 가능.
>   mean_reward 급등락 / adaptive LR 급락이 보이면 clamp 상한을 0.2로 줄일 것.

> **2026-07-27 업데이트 — 파지 보상 실험 중단·성공 기준선 복원·`hand_grasp` 시작**
>
> - Box/Chopstick에서 tip-only cage, 좁은 axial region, maintained fingertip proximity를 순차 검증했지만,
>   tip 사이 가상선 false positive와 순간 progress-only 해법이 남아 실제 손끝 밀착 파지를 만들지 못함.
>   상세 수치와 backup은 `ACTIVITY_2026-07-27.md`.
> - 여러 파지 shaping을 더 누적하지 않고 활성 코드를 실제 성공 run 기준으로 리셋함.
>   Chopstick은 `2026-07-25_08-47-53(랜덤성공)`, Box Transport는
>   `2026-07-26_00-04-17(성공)`의 `params/env.yaml`을 기준으로 복원함.
>   실패 실험 코드는 backup과 비활성 함수로 보존됨.
> - 다음 검증은 arm reach/transport를 제거하고 손가락 파지만 격리한 신규 task **`hand_grasp`**에서 진행함.
>   Indy 팔 없이 고정 Wuji right hand만 스폰하고, 손바닥을 하늘로 향하게 한 뒤
>   `7 x 180 x 7 mm` 스틱 두 개를 손바닥 위에 평행 배치함.
> - 손바닥 배치는 `palm_link` 실제 메시의 로컬 `+x` 면을 world `+z`로 맞춘 것임.
>   **기존 파지 개구부 추정 법선, `palm_facing`, cage 방향은 절대 사용하지 않음.**
> - 스틱 장축은 손가락 진행 방향과 직교하는 world `+y`이고, 두 스틱은 손가락 진행 방향
>   world `+x`를 따라 중심 간격 `2 cm`로 배치함. 초기 구현의 finger-parallel 배치는 폐기함.
>   위쪽/world `x=0.055`가 노란색 `stick1`, 아래쪽/world `x=0.035`가 초록색 `stick2`임.
> - `hand_grasp` action은 20개 손가락 관절의 residual 제어임:
>   `target = current_joint_pos + action * 0.3`, `clamp_to_limits=True`.
>   observation은 **101D**:
>   `joint_pos 20 + joint_vel 20 + fingertip_pos 15 + two stick poses 14
>   + two stick relative velocities 12 + previous_action 20`.
> - 공식 ShadowHandOver의 DirectMARLEnv 구조는 사용하지 않음. 배치/관측 설계만 참고하고,
>   실제 task는 `CustomManagerBasedRLEnv` + MDP manager 구성으로 유지함.
> - `hand_grasp`의 최초 scene/action/observation smoke 당시에는 action-rate penalty만 있었음.
>   `2026-07-27_15-09-50` 1-env/0-iteration 생성 smoke에서 action 20D, obs 101D,
>   rigid object 2개, termination 3개 resolve를 확인함. 아래 23시 이후 업데이트부터는 고정
>   pre-grasp STATE B 학습 reward/termination이 active이므로 이 초기 placeholder 설명을 현재 cfg로 읽지 않음.
> - 첫 play의 `JointPositionToLimitsAction.apply_actions()` TypeError는 `CustomActionManager`가 모든 term에
>   `env_ids`를 강제로 넘긴 호환성 문제였음. manager를 upstream과 같은 인자 없는 호출로 수정함.
>   `obs_groups`에도 actor key를 명시해 RSL-RL의 향후 제거 예정 fallback 경고를 없앰.
>   이후 hand action 자체는 joint-limit absolute가 아닌 Wuji residual action으로 확정함.
>   `15-09-50` smoke checkpoint는 변경 전 action/스틱 배치라 resume하지 말고 다음 학습은 fresh.
> - 수정 후 fresh smoke `2026-07-27_15-18-38`에서 1 env / 1 iteration, 실제 24 step을 통과함.
>   action 20D·obs 101D, 두 stick drop 0이며 위 TypeError/actor obs 경고가 재발하지 않음.
> - **`hand_grasp` 기준 파지는 simulation only로 탐색함.** 실물 Wuji 자세·관절값을 가져오지 않고,
>   2022 젓가락 논문의 표준 topology를 시작점으로 contact 후보를 IK로 만든 뒤 PhysX에서 안정성을 평가함.
>   필수 접촉은 `stick1`–엄지/검지/중지 tip, `stick2`–약지 tip,
>   `stick2`–엄지·검지 사이 valley anchor임. 엄지 중간마디–`stick2`,
>   검지 중간마디–`stick1`은 Wuji 형상 의존 보조 접촉이라 우선 metric으로만 기록함.
> - **현재 다음 1순위는 IK가 아니라 collision/valley feasibility probe임.**
>   `stick1`–`stick2` 닫힘 각도 sweep과 `stick2`–`palm_link/finger1_link1/finger2_link1`
>   실제 접촉을 먼저 확인함. tip보다 handle/pivot이 먼저 충돌하거나 penetration·반발·마찰 고착이
>   생기면 IK 해도 물리적으로 무효임. 통과 후 `stick2` valley pose 고정 → fingertip IK →
>   1~2초 PhysX static hold(contact/slip/drop/torque) → `q_ref_closed` 선정 → open-close 순서로 진행함.
>   파지 개구부 법선과 가상 cage는 이 탐색에 사용하지 않음. 상세는 `ACTIVITY_2026-07-27.md`.
> - **20:15 self-collision 진단 및 공통 물리 수정:** candidate 14를 self-collision ON으로 replay하자
>   기존 standalone USD에서 `palm_link ↔ finger2/3/4/5_link2`가 첫 step부터 계속 충돌했고
>   최대 `9.75 kN`, joint tracking 최대 `1.662 rad`였음. 동봉 MuJoCo 모델에는
>   `palm_link ↔ finger1~5_link2` 제외가 있지만 URDF-derived USD에는 pair filter가 없었음.
>   standalone은 `wuji_right_filtered.usda`, Indy+Wuji는
>   `indy7_wuji_right_simplified_filtered.usda` overlay를 사용하도록 변경해 이 다섯 pair만 제외함.
>   corrected probe `2026-07-27_20-15-27`에서 palm pair는 모두 사라지고 tracking 최대가
>   `0.411 rad`로 감소함. 대신 검지–중지 실제 충돌 10쌍이 드러났으므로 candidate 14는 최종 자세가
>   아니며 실제 finger-finger 충돌은 필터링 금지.
> - filter 적용 ON CEM `2026-07-27_20-24-20`은 proximal tracking은 풀었지만 `stable=0/16`.
>   palm/thumb-mid–Stick2는 약 96% 이상 유지된 반면 index/middle/ring contact가 대부분 0~3%였음.
>   CEM 물리 단계가 stick을 밖에 주차하고 analytic 거리만 보던 search/validation 불일치가 원인이라,
>   physics-aware CEM에서도 후보 stick을 실제 배치하고 6개 contact force를 목적함수에 추가함.
>   다음은 로그의 `contacts=N/6`과 dynamic validation contact fraction을 함께 판정함.
> - contact-aware CEM `2026-07-27_20-36-31`은 physics best가 `1/6→5/6`까지 개선됐지만
>   `stable=0/16`. candidate 2는 Stick1 변위 `1.67 mm`, index/middle `100%`,
>   palm/thumb-mid `91.7/97.9%`였고 thumb-tip/ring-tip만 `0.8%`라 가장 가까움.
>   다음 probe는 thumb-tip/ring-tip contact weight를 `1.5/2.0`으로 높이고
>   `--search-iterations 45 --physics-aware-iterations 20`으로 실행함. search contact force와
>   validation `maintained_contact_count`를 JSON/로그에 저장하도록 계측함.
> - extended run `2026-07-27_20-46-24`도 고정 상태 최대 `5/6`, validation 대부분 `2/6`,
>   `stable=0/16`이라 반복 연장은 무의미. search force상 고정된 stick에는 다섯 손 접촉이 실제로
>   생겼지만 고정을 풀면 fingertip contact가 무너지고 palm/thumb-mid만 남았음. 따라서 CEM이
>   kinematic stick에 순간 힘을 거는 해를 이용한 것으로 판정함. physics CEM 내부에 24-step
>   mini-release를 추가해 contact 유지율(`>=50%`)과 두 stick 변위(scale `10 mm`)를 loss에 직접
>   포함함. 다음 로그는 `contacts`, `release`, `release_disp_mm`를 함께 판정함.
> - mini-release CEM run `2026-07-27_21-15-49`는 최종 validation에서 `5/6` 유지 후보 4개가
>   나왔고, 네 후보 모두 유일하게 빠진 항은 `ring_tip_stick2`였음. candidate 5는
>   thumb/index/middle `100%`, palm/thumb-mid `84.2/99.6%`, ring `0.4%`,
>   Stick1/2 release 변위 `1.32/8.28 mm`, torque clipping `0`. 약지는 고정 stick에서
>   `0.386 N`, 24-step mini-release에서 `25%` 접촉했지만 장기 release에서 사라졌으므로
>   센서/전체 topology가 아니라 lower-stick 약지 지지의 단일 동역학 병목임.
> - 위 run의 좋은 후보가 Stick2 palm-normal 기존 `36.3 mm` 상한에 다시 붙어, 다음 탐색은
>   상한을 seed+`30 mm`로 확장하고 ring geometry weight `1→1.5`, ring dynamic contact weight
>   `2→6`, 유지 손실 계수 `0.75→1`, mini-release `24→60 step`으로 변경함. 6/6 조건과
>   최종 stable 기준은 완화하지 않음.
> - ring 강화 run `2026-07-27_21-40-43`은 ring contact 자체는 해결함. candidate 3/14/15에서
>   최종 ring `100%`였지만 palm/thumb-mid가 `0/0%`; 반대로 candidate 5는 palm/thumb-mid
>   `87.1/92.9%`지만 ring `5%`였음. 새 x 상한 `46.3 mm` 부근에서 약지와 valley anchor를
>   서로 교환하는 weighted-sum local mode임.
> - 다음 코드는 static ring weight `6→2`, thumb-mid 기하 weight `0.45→1.0`으로 균형을 되돌리고,
>   mini-release 목적을 `5×missing_contact_count + 1.5×squared_deficit_sum`으로 변경함.
>   gradient-free CEM에서 6/6 contact 수를 먼저 우선하고 같은 수 안에서 유지율을 비교하기 위함임.
>   Stick2 범위는 아직 유지하고 이 목적함수 수정 결과 뒤에 다시 판단함.
> - contact-count run `2026-07-27_21-50-59`는 physics best가 거의 계속 `release=5/6`이고
>   validation 5/6 후보가 6개였지만, 공통 누락은 다시 ring이었음. 원인은 mini-release에서
>   Stick2가 `11~19 mm` 움직인 뒤에도 `_geometric_loss`가 release 전 pinned stick pose를 기준으로
>   계산하던 잔여 불일치였음. physics CEM은 이제 release 후 실제 두 stick pose를 PhysX에서 읽어
>   palm frame으로 변환하고 그 pose에 대해 fingertip geometry를 계산함. 다음 physics
>   `errors_mm`부터 실제 settled object 기준임.
> - settled-pose run `2026-07-27_22-03-24`에서 anchor 유지 5/6 해의 실제 ring error가
>   `23~31 mm`로 드러났고, ring error `0~3 mm` 해는 valley anchor를 잃어 4/6이었음.
>   원인은 pinned stick에 약지까지 먼저 닫은 뒤 Stick2가 release에서 `17~19 mm` valley로
>   이동하는 동시-close 순서임. search/validation을 ring-open 상태의 anchor close → free valley
>   settle → ring-only ramp-close → final hold 순서로 통일함. search는 `60+24+60 step`,
>   validation은 `60+60+240 step`이며 contact/geometry는 마지막 free-stick 상태에서 측정함.
> - 마지막 staged run `2026-07-27_22-45-04`도 최대 5/6, `stable=0/16`이고 공통 누락은
>   ring-tip→Stick2였음. 합의한 기준에 따라 CEM 반복은 종료함. residual RL seed는 candidate 6:
>   thumb/index/middle `100%`, palm/thumb-mid `98.3/99.6%`, ring `0%`,
>   displacement `1.16/14.00 mm`, speed `0.088/0.057 m/s`, pair contact 없음,
>   tracking `0.248 rad`, finger4_joint4 torque clip `11.7%`.
>   `pregrasp_candidate.json`을 이 self-collision ON 5/6 후보로 갱신함. 완성 파지가 아니며,
>   다음 단계는 추가 CEM이 아니라 seed 주변 residual RL로 ring support를 완성하는 것임.
> - 약지 해결 방향 판별용 `--probe-ring-reach`를 추가함. candidate 6의 다른 관절/Stick pose는
>   고정하고 `finger4_joint2`만 soft-limit 2% margin 전 범위로 한 번 병렬 sweep함.
>   ring pad의 Stick2-local xyz와 axial/cross-section excess를 저장하므로, axial miss일 때만
>   젓가락 길이 연장을 검토하고 x/z miss면 길이 변경을 배제할 수 있음. 전체 CEM은 재개하지 않음.
> - joint2 sweep `2026-07-27_23-15-18` 결과는 ring contact `0/512`, 다른 5접촉
>   `440/512`, full 6접촉 `0/512`. 최선 ring pad는 Stick2 local
>   `[19.0, -21.7, 9.7] mm`, axial excess `0`, cross-section excess `16.6 mm`였음.
>   따라서 젓가락 길이 부족과 ring joint2 단독 조정은 배제함.
> - **23시 이후 active `hand_grasp` 학습 구성:** 추가 CEM 없이 canonical candidate 6을 매 reset의
>   고정 pre-grasp로 사용함. 접촉 topology는 Stick1의 thumb/index/middle tip,
>   Stick2의 palm/thumb-middle anchor 5개를 보존하고, Wuji 형상에 맞춰 ring support만
>   `finger4_tip_link OR finger4_link4`로 완화함. obs/action은 101D/20D 그대로임.
>   anchor는 thumb/index/middle tip, palm, thumb-middle을 각각 weight `1`로 기록하며 합계는
>   기존 `5 × mean`과 동일함. ring support `30`, 모든 접촉이 열린 상태의 저속 안정화
>   `50`, success `30000`, action-rate `-0.001`. 6 semantic group이 `0.02 N` 이상이고 두 스틱
>   선속도 `<0.15 m/s`, 각속도 `<3 rad/s`를 30 policy step(1초) 유지하면 success 종료함.
>   TensorBoard에는 개별 reward와 함께 `Metrics/hand_grasp{,_final,_min,_max}/*`로 7개 sensor force,
>   ring OR force, 6-contact count/fraction, full-contact, quiet-valid, stick 속도, stable step을 기록함.
>   이 환경은 STATE B 고정-pose 파지 학습이며 acquisition/FSM/open-close는 아직 섞지 않음.
> - **2026-07-28 `hand_grasp` 5/6 local optimum 수정:** run `2026-07-27_23-43-11`은
>   5166 iter까지 success/quiet/stable-step이 전부 0이고, 최종 접촉력이 thumb/index/palm/thumb-mid/ring
>   `0.37/0.89/0.66/0.60/1.09 N`인 반면 middle→Stick1은 약 `1e-5 N`이었음.
>   ring 연금 약 235점이 mean reward 265의 대부분이라 약지를 붙이는 대신 중지를 버린 접촉 교환임.
>   다음 fresh부터 독립 `ring_support`를 `ring_support_coupled`로 교체해 다른 5 anchor가 모두
>   `0.02 N` 이상일 때만 ring strength×30을 지급함. middle tip→Stick1 oriented-box surface
>   proximity `5`(`exp(-d/0.02)`)를 추가하고, full stability `50`도 6개 모두 `0.02 N` 이상일 때만
>   켬. 두 stick 최대 각속도의 3 rad/s 초과분 제곱에 `-0.1` penalty를 접촉과 독립적으로 적용함.
>   obs/action/success 조건은 101D/20D 및 30-step 그대로이며 기존 run resume 금지.
> - **2026-07-28 reset pose contact-pair 실측:** `pose_005`를 60-step settle 후 240-step
>   전 link pair 스캔한 결과, Stick1의 엄지 접촉은 육안상 tip 아래 마디로 보이지만 PhysX
>   collision pair는 `finger1_link3`였음(240/240, 평균 `1.986 N`). 따라서 활성 semantic
>   `thumb_distal_stick1`은 `finger1_link3↔Stick1`을 측정함. `finger1_link4` 추정은 폐기함.
>   그 밖에 index-tip/Stick1 `2.161 N`, middle-tip/Stick1 `2.632 N`,
>   `finger1_link2`/Stick2 `2.487 N`, palm/Stick2 `2.242 N`, ring-tip/Stick2
>   `0.397 N`이 모두 240/240 유지되어 기존 6 semantic group이 실제 reset topology와 일치함.
>   육안과 달리 palm/Stick1도 `1.637 N`이었으나 이는 별도 필수 group으로 추가하지 않음.
>   ring-tip↔pinky-tip도 240/240, `1.343 N`이지만 물체 직접 접촉이 아니므로 success에는
>   추가하지 않고 q-reference가 보존하게 둠. TensorBoard 엄지 tag는
>   `thumb_distal_stick1_force`를 사용함. 기존 ring `tip OR link4` 완화도 폐기하고, probe에서
>   실제 240/240 유지된 `finger4_tip_link↔Stick2`만 여섯 번째 필수 group으로 사용함.
>   따라서 active contact sensor와 semantic group은 정확히 6개임.
> - **2026-07-28 GUI 검증으로 candidate 6 seed 폐기:** action scale 0으로 기존 reset 자세를
>   직접 확인하자 새끼 외 손가락이 먼저 닫혀 있고 Stick2가 엄지–검지 valley 안에 들어가지 않았음.
>   `palm↔Stick2`와 `thumb_link2↔Stick2` body-pair force만으로는 접촉 위치를 제한할 수 없어
>   바깥쪽 접촉을 valley anchor로 오인한 것임. 위의 candidate 6 canonical/active 설명보다 이 판정을
>   우선하며, 해당 seed에서 reward만 바꾼 학습은 계속하지 않음. 다음은 열린 손에서 Stick1을 격리하고
>   Stick2를 손가락으로 valley에 이동시키는 목표 pose를 먼저 찾음.
>   `scripts/debug/hand_grasp_keyboard.py --task hand_grasp`에서 finger `1~5`, joint `1~4`를
>   선택해 20관절을 직접 조정하고 자세 JSON을 저장할 수 있음. 수동 확인상 기존 stick spawn은
>   손목 쪽으로 낮아 기본값에서 두 stick을 world `+x`/palm `+z`로 20 mm 올렸고, Stick1은
>   그 spawn pose에 고정하며 Stick2만 dynamic으로 둠. CLI offset과 Stick1 mode로 변경 가능함.
>   열린 reset은 수동 확인된 엄지 완전 신전 자세를 쓰며 `finger1_joint2`를 soft lower limit
>   `-0.1659 rad`에 놓고 실제 joint state와 target을 동시에 초기화함.
>   두 stick의 palm-local `xyz`·상대 delta·횡단 간격은 keyboard tool이 기본 5 Hz로 출력하며
>   `T`와 `--stick-print-hz`로 toggle/주기를 조절함.
> - **2026-07-28 수동 pose를 새 `hand_grasp` reset seed로 채택:** 사용자가 두 stick을 dynamic으로
>   둔 keyboard run `2026-07-28_10-43-39`의 `pose_002.json`에서 엄지로 두 stick을 valley 쪽에
>   고정한 자세를 저장함. 엄지는 joint1을 먼저 굽힘 → joint2로 stick을 누름 → joint1을 조금
>   다시 펴는 순서로 만들었고, 검지·중지·약지·새끼는 열린 상태임. active reset은 저장된
>   palm-local Stick1/2 pose를 그대로 복원하며, 관절 **actual state**
>   `(thumb j1/j2/j3≈0.3593/0.8707/0.0603 rad)`와 **PD target**
>   `(0.3491/0.9339/0 rad)`을 구분해 엄지 preload도 보존함. 두 stick은 reset 순간에만 pose/속도를
>   쓰고 이후에는 gravity/collision이 켜진 독립 dynamic rigid body임. candidate 6 reset 설명은
>   더 이상 active가 아니며 reset/reward가 바뀌었으므로 다음 학습은 checkpoint resume 없이 fresh로 함.
> - **2026-07-28 수동 완성 파지 `pose_005`를 STATE B 기준으로 확정:** 열린 손에서 사용자가 직접
>   만든 `logs/debug/hand_grasp_keyboard/2026-07-28_12-39-52/pose_005.json`은 약 12초간 연속
>   저장에서 Stick1/2 drift가 약 `0.02/0.15 mm`였고 두 stick을 모두 유지함. 종료 직전
>   `pose_006`은 actuator target만 open으로 초기화된 스냅샷이라 사용하지 않음. active reset은
>   `pose_005`의 joint actual, 별도 PD target(preload), 두 palm-local stick pose를 그대로 복원함.
>   action은 20D **`pose_005 PD target + residual`**로 두고 scale을 `0.1 rad`로 설정함.
>   current-position residual에서 scale만 0.1로 낮추면 action clip ±1 때문에 수동 preload
>   오차(최대 약 0.36 rad)를 재현하지 못하므로 reference-centered residual을 별도 action term으로
>   구현함. zero action이 수동 preload를 그대로 유지하고 정책은 그 주변만 미세 보정함.
>   이전 candidate-acquisition용 `middle_surface_proximity`와 `ring_support_coupled`는 제거하고,
>   joint reference `2`(약한 prior), Stick1/2 palm-pose reference `10/15`, 6 semantic contact의
>   **minimum** strength `20`, full-contact 저속 stability `50`, success `30000`으로 STATE B
>   유지/복원 보상을 구성함. success는 접촉·속도뿐 아니라 두 stick이 기준 pose에서
>   `2 cm/20°` 이내인지도 30 step 동안 요구함. TensorBoard에 두 stick position/orientation error와
>   `reference_pose_valid`를 추가함. obs/action shape는 101D/20D 그대로지만 scale/reset/reward/물리가
>   달라졌으므로 이전 checkpoint resume 금지.
> - **2026-07-28 `hand_grasp` OPEN/CLOSE 모드 1단계:** 위 STATE B가 수렴한 뒤 같은 `pose_005`
>   reset에서 에피소드별 `OPEN=[1,0]`/`CLOSE=[0,1]` command를 50:50으로 고정 샘플함.
>   mode one-hot을 policy obs에 넣어 **101D→103D**, action은 20D reference residual·scale
>   `0.1 rad`를 유지함. saved pose 기하상 distal tip은 두 stick 모두 local `+y`; 초기 tip-center
>   거리는 `23.3 mm`, 7 mm 두께를 뺀 초기 surface gap은 `16.3 mm`임. OPEN target은 실제
>   개방 동작을 요구하도록 `20 mm`, CLOSE target은 tip 직접 충돌을 피하도록 `3 mm`로 둠.
>   Stick1 full-pose 보상은 open-close를 막으므로 제거하고 local `y=-0.06 m` in-hand pivot만
>   palm 기준으로 유지하며, Stick2 full pose는 기준 rail로 유지함. `open_tip_gap`/`close_tip_gap`
>   reward는 active mode만 켜지고, stability `50`도 **6접촉+저속+active-mode gap** 결합형이라
>   CLOSE 명령에서 OPEN 자세 유지로 연금을 먹을 수 없음. success는 mode gap 오차 `≤3 mm`,
>   pivot `≤15 mm`, Stick2 `≤20 mm/20°`, 6접촉·저속을 30 step 유지할 때 종료함.
>   TensorBoard는 mode, actual/target/error gap, pivot error, mode geometry valid를 추가 기록함.
>   smoke `16-31-57`(resolve)와 `16-32-21`(1 env, 24 step)이 통과함. **103D라 이전 checkpoint
>   resume 금지.** 에피소드 중 OPEN↔CLOSE 전환은 두 고정 mode가 각각 수렴한 뒤 다음 단계임.
> - **2026-07-28 수동 완성 pose의 IK 역할:** keyboard run `2026-07-28_10-57-45`의
>   `pose_017.json`이 새끼까지 굽힌 마지막 완성 후보이며 `pose_018`은 직후 reset 상태임.
>   pose 17은 reset이나 최종 `q_ref`가 아니라 residual RL 직전 nominal IK endpoint로 사용함.
>   `hand_grasp_ik_replay.py`가 현재 엄지-loaded 열린 reset에서 검지→중지→약지→새끼 각각
>   `joint4 말단 close→joint3 close→joint1/2 배치→joint3/4 최종각 release`를 수행하고 엄지는
>   마지막에 조금 펴며, 두 dynamic stick의 최종 pose/contact force를 JSON에 저장함.
>   `env.step()`을 사용하지 않아 termination auto-reset은 없음. 이 replay의 접촉 재현성을 먼저
>   확인한 뒤 RL reward를 연결함.

> **2026-07-29 업데이트 — 연속 OPEN/CLOSE, tip gap 교정, 현재 run**
>
> - 아래 내용이 위의 2026-07-28 “에피소드별 고정 mode” 설명보다 최신임. `hand_grasp` command는
>   reset 때만 OPEN/CLOSE를 50:50으로 뽑고 이후 **3초마다 반드시 반대 mode로 토글**함.
>   `[1,0]=OPEN`, `[0,1]=CLOSE`이며 episode를 `16 s→30 s`로 늘려 한 reset 상태에서 약 10개
>   mode 구간을 경험하게 함. 성공은 종료가 아니라 mode 구간당 1회 bonus이고, 두 stick 중 하나가
>   떨어지거나 30초 timeout일 때만 reset함. 목적은 정책 자신의 누적 파지 오차에서 다음
>   OPEN/CLOSE로 복구하는 능력을 학습하는 것임.
> - active target은 **OPEN `20 mm`, CLOSE `3 mm`**, mode gap 허용오차 `3 mm`, mode 유지
>   `30 policy step=1 s`임. action/obs는 `20D/103D`, action은
>   `pose_005 PD target + action×0.1 rad`, PPO 초기 std `0.3`, entropy coefficient `0.001`임.
>   6 semantic contact, Stick1 pivot, Stick2 palm-relative pose, angular-speed 조건과 reward 구조는
>   유지함.
> - tip gap을 기존
>   `max(||p1_tip-p2_tip||₂ - 7 mm, 0)` 단순식에서 **장축 slip과 사각 단면 회전을 반영하는
>   횡단 표면 간격**으로 교체함. 현재 active 구현은 고정 rail 역할인 Stick2의 순간 local `+y`
>   장축 성분을 두 distal endpoint 차벡터에서 제거해 횡단거리 `d_perp`를 구하고,
>   순간 횡단 분리방향 `n`으로 각 정사각 단면의
>   support radius
>   `r_i=3.5 mm (|n·x_i|+|n·z_i|)`를 계산해
>   `gap=max(d_perp-r_1-r_2,0)`을 사용함. 고정 palm normal/파지 개구부 법선은 사용하지 않음.
>   Stick1도 포함한 평균 공통축은 움직이는 Stick1의 기울기에 측정축이 따라가는 문제가 있어
>   `2026-07-29_11-14-30` baseline까지만 사용하고 폐기함. 이 helper가 mode gap reward,
>   stability, success, TensorBoard diagnostic에 모두 연결됨.
> - `pose_005` 기준 old 식은 tip-center `23.322 mm - 7 mm = 16.322 mm`였음. 새 식은
>   axial offset `5.094 mm`를 제외한 횡단 중심거리 `22.759 mm`에서
>   `r1=4.714 mm`, `r2=4.735 mm`를 빼 **`13.310 mm`**임. obs/action shape는 같아도 reward와
>   success의 의미가 바뀌었으므로 old-gap checkpoint resume 및 절대 reward 직접 비교를 금지함.
> - 자정을 넘어 계속 본 old-gap `2026-07-28_21-54-22(2cm오픈성공)`은 iter 1744에서
>   OPEN/CLOSE raw `0.333/0.380`, final gap error `1.79 mm`, full contact `99.3%`,
>   quiet-valid `40.4%`로 가장 좋았으나 iter 6513에는 reward `977→839`,
>   quiet-valid `40.4→19.9%`로 후반 퇴화함. **old-gap 2 cm 기준선**으로만 보존함.
> - old-gap `2026-07-29_00-04-22(3cm실패)`은 OPEN을 `30 mm`로 늘렸으나 iter 5890에서
>   OPEN/CLOSE raw `0.264/0.056`, gap error `8.90 mm`, geometry-valid `28.2%`,
>   quiet-valid `6.9%`, success pulse `0`이었음. 6접촉은 `99.9%`라 파지 소실이 아니라
>   큰 OPEN을 추구하며 CLOSE와 저속 안정화를 희생한 것으로 판정하고 30 mm를 기각함.
> - corrected-gap fresh run **`2026-07-29_11-14-30`**은 최신 기록 iter `1214`에서
>   mean reward `1606.7`, policy std `0.099`, OPEN/CLOSE raw `0.301/0.287`,
>   stability raw `0.103`임. 전체/final gap error는 `5.46/3.28 mm`,
>   전체/final geometry-valid는 `26.5/62.3%`, full contact `98.0%`, drop `0`임.
>   Stick1 pivot `9.10 mm`, Stick2 position `2.33 mm`, orientation `4.04°`로 anchor는 유지됨.
>   이 run은 평균 공통축 gap의 비교 baseline이며, Stick2-axis gap으로는 resume하지 않고 fresh
>   run을 시작함. OPEN/CLOSE target과 나머지 reward/action/PPO 설정은 그대로 유지해 단일변수로 비교함.
>   success raw가 `1.18e-4`로 처음 비영점 pulse가 보이지만 아직 희소함. 남은 병목은 contact가 아니라
>   max angular speed `4.15 rad/s`와 quiet-valid `9.5%`로 드러나는 전환 후 감속/정착임.
>   play에서도 OPEN은 실제로 나오지만 느리므로 “CLOSE만 유지”로 단정하지 않음.
> - 현재 run은 최소 iter `1500~2000`까지 같은 설정으로 판정함. entropy/action scale을 중간에
>   바꾸지 않음. 그때도 mode 전환 후 목표 도달이 2초 이상 걸리면 다음 fresh에서 dwell을
>   `4~6 s`로 늘린 뒤 다시 3초로 줄이는 curriculum을 검토함. 다음 계측 후보는
>   mode별 actual gap, command switch 뒤 tolerance 진입 시간, mode별 success pulse임.
> - 다음 물체 파지 환경 전에는 현재 두 stick 파지를 유지한 채 **stick이나 물체를 직접 움직이지 않고**
>   hand/root에만 scripted `+z→xyz→제자리 회전→결합 랜덤 pose`를 주어 palm-relative stick pose,
>   6-contact, gap, slip/drop을 검증함. 이후 물체는 tip 사이의 고정된 reset 위치에 dynamic으로
>   스폰하고 Stick1 CLOSE로 파지하게 함. 진짜 성공은 단순 거리나 접촉 sensor가 아니라 hand/stick을
>   움직일 때 물체가 함께 따라오며 상대 pose를 유지하는 것임. 물체 접촉 sensor를 필수 observation이나
>   손가락 힘 proxy로 쓰지 않음.

### Phase 로드맵 (사용자 설계, 2026-07-24) — 이 절이 현재 연구 방향의 상위 계획

> 상세·논문 근거는 `ACTIVITY_2026-07-23.md`의 "Phase 계획", 연구방향 맥락은 `study.md` 2026-07-24 절.

- **Phase 1 (현재)**: 두 젓가락 획득 + functional grasp 형성. 완료조건 = 두 젓가락 획득 →
  손바닥 위 정렬 → 각 손가락이 지정 젓가락의 지정 영역 접촉 → 아래 stick 안정 유지 →
  위 stick 개폐 → 반복 open-close에도 안 미끄러지고 안 떨어짐.
  - topology 가설: 엄지·검지·중지→Stick1, 약지→Stick2, 새끼 보조(스타일 A/B/C).
  - 내부 3단계(한 reward sum 금지): STATE A 획득+staging / B standard grip / C open-close.
    각 단계 latch(0.5~1s 연속)로 전환, 이후 이전 reward 끄고 성공 bonus 1회.
  - 구현: **방식 2(단계별 별도 환경/curriculum) 먼저 → 나중에 FSM 결합** (사용자·에이전트 합의).
  - BO(논문 C 축소): surface_axis/sign 고정, 손가락별 axial center만 sweep.
- **Phase 2 (나중)**: 젓가락으로 물체 집어 **큰 목표 공간(바구니)에 놓기**.
  ⚠ **정밀 pick&place 아님** — 자세 무관, 목표항 위치 φ만. 병목은 tip 파지 + open-close 유지.
- **진행 순서**: 1-stick 고정 pose 수렴 → **랜덤 pose 도달성 검증(2-stick 전제)** → STATE A→B→C.

- **⚠ 2026-07-24 구조 변경 (chopstick, 다음 fresh부터)**:
  - **에피소드 내 goal 재샘플**: `resampling_time_range=(5,10)`. best-so-far 보상(transport·orientation·
    fine 2개)은 `_goal_changed`로 goal 변화를 감지해 기준선(_pending/_seen) 재장전 → 새 goal 따라감.
    안 하면 옛 goal 기준선 고착(무보상 구간/자세 고착). box는 keypoint 1개에 같은 캐시 배선(준비됨).
  - **success가 termination→reward로 분리**: `GoalReachedBonus`(비종료, goal당 1회, `_awarded`로 반복
    파밍 차단). 성공해도 에피소드 유지 → 재샘플로 다음 goal. **`Episode_Termination/success` 지표 없어짐**
    → 성공 빈도는 `Episode_Reward_Raw/transport_success`로 판독. timeout 8초 유지(빼면 무한 에피소드 —
    Manager-Based는 termination_manager가 max_length 자동 체크 안 함, time_out DoneTerm이 유일한 다리).
  - **grip region 재설계**: tip/tail 뒤집음(+y=tail 파지, −y=tip. axial 양수. orientation은 y축 180°
    대칭이 흡수해 무변경) + palm-down 배치(엄지−x/중지+x/검지+z). 좌표계 identity라 로컬=월드, −x=로봇 몸쪽.
    복구용 09-42-28 버전 주석 보존. ⚠ 검지 +z는 palm-up이면 도달 불가 — 근본 해결은 palm-down 강제
    (hand_stick_orientation)와 접촉 보상 강화(grip을 per-step −w·d로). region 배치만으론 정책이 안 따름(weight 40≈1.3점).

- **⚠ obs가 oracle(sim-to-real 불가) — 실물 전이 전 반드시 정리할 것**: 현재 chopstick obs 91 중 error
  계열이 **시뮬 전용**임 — `index/thumb/middle_grip_error`(9), `stick_ori_to_target`(3),
  `hand_stick_orientation_error`(3). 실물 로봇은 "정답까지의 오차"를 직접 못 잰다(스틱 정밀 pose+region
  기하+FK 필요). 정책이 오차를 직접 보고 학습 = 전이 불가. 실측 가능한 건 joint_pos·action_history,
  비전 추정 가능은 stick_pos·stick_in_fingertips·stick_size. **지금은 태스크 성립 확인이라 oracle 유지가
  맞지만**(error 빼면 학습 난도 급상승), 태스크가 풀린 뒤 error obs를 raw로 교체하거나 asymmetric
  actor-critic(critic만 oracle)으로 가야 함. 사수님 지적(2026-07-24).
- ⚠ 현재 코드는 **스틱 1개**(`chopsticks_grasp_env_cfg.py`의 `cube` 1개)뿐. Stick 2·약지-Stick2·
  open-close·palm-up·2스틱 성공조건은 **전부 미구현** — Phase 1 STATE A부터가 신규 태스크 구축임.

### 최종 결론: 도구 획득 정책 + 계층형 젓가락 사용 정책

처음부터 하나의 거대한 정책이 아래 전 과정을 동시에 학습하게 하지 않음.

```text
테이블 접근
-> 젓가락 2개 획득
-> 기능적 파지 형성
-> 젓가락 opening 제어
-> 물체 접근/파지
-> 운반
```

최종 구조는 다음과 같음.

```text
상위 FSM / Skill Manager

Skill A: Chopstick Acquisition & Functional Pre-grasp
         테이블 위 젓가락을 획득해 사용 가능한 tool-ready state를 만듦
         현재 Box Transport의 cage -> hold -> lift -> transport -> success를 확장

Skill B: Chopstick Use
         Low-level  = 손가락으로 젓가락 파지 안정화 + opening 제어
         High-level = arm 이동 + 목표 tool pose/opening 명령 + 물체 파지/운반
```

학습 정책은 크게 A/B 두 개로 분리하고, Skill B 내부만 high/low hierarchy로 구성함. 처음에는
network를 합치지 않고 FSM으로 전환하며, 두 정책이 각각 안정된 뒤에만 전체 rollout fine-tuning이나
distillation을 검토함.

### 현재 연구 범위: Skill A 완성

- 지금 만드는 것은 젓가락 사용 정책이 아니라 **테이블 위 막대 획득 + 안정적인 기능적 파지 자세**임.
- 현재 `Indy-Wuji-Box-Transport`의 랜덤 직육면체는 **젓가락 한 개 프록시**임. 아직 두 개의 느슨한
  젓가락을 순서대로 집는 문제나 opening 제어를 학습하는 단계가 아님.
- 현재 run 기준 시험 목표는 막대를 잡은 채 고정 goal position과 yaw 45도 orientation을 맞추고,
  orientation error 15도 안에서 안정적으로 유지할 수 있는지 확인하는 것임.
- 이 45도 goal은 회전 제어 가능성을 확인하는 기준선이지 최종 functional grasp pose 자체가 아님.

현재 Skill A가 만들어야 하는 흐름:

```text
테이블 위 막대 접근
-> cage 안에 넣기
-> 손가락을 오므려 안정적으로 파지
-> 실제 clearance를 만들며 lift
-> 사용 준비 위치로 transport
-> 목표 tool world orientation 정렬
-> 올바른 hand-tool relative grasp와 낮은 slip 유지
-> 일정 시간 만족하면 tool_ready terminal success
```

`Indy-Wuji-Box-Transport`에는 cage reach/hold, clearance lift, world position transport, world
orientation progress, cage+position+orientation 유지 성공까지 들어 있음. 별도 A1 검증 태스크인
`Indy-Wuji-Chopsticks-Grasp`에는 첫 번째 functional constraint인 stick-relative hand orientation과
index/thumb semantic grip-region target까지 구현됨. 아직 공통으로 완성해야 할 핵심은 손 안에서 막대가
미끄러지는 상대 선속도/각속도 조건임.

### A1 분리 검증 태스크: `Indy-Wuji-Chopsticks-Grasp` (2026-07-20)

- 목적은 얇은 물체에서 cage/hold만 파밍하는 문제와 constraint-based functional grasp가 실제로 도움이
  되는지 A0 Box-Transport와 독립적으로 검증하는 것임.
- scene/MDP/PPO/log를 분리함. `Cube-Grasp`와 `Box-Transport` cfg를 상속하지 않으며 신규 로그는
  `logs/rsl_rl/indy_wuji_chopsticks_grasp`에 기록함. 이름 통일 전 과거 런은
  `logs/rsl_rl/indy_wuji_chopstick_acquire`에 그대로 보존함.
- 현재 물체는 local `+y`가 tip인 고정 `2 x 18 x 2 cm` 직육면체 프록시 한 개임. 두 젓가락이나
  opening 제어는 포함하지 않음.
- policy action은 arm 6 + thumb/index/middle 12의 18D 절대 joint target임. 단, 현재
  `MimicJointActionCfg`가 middle action을 ring/little에도 복제하므로 약지·새끼가 reset 자세에 passive하게
  고정되는 구조는 아님. action 출력 차원만 18D이고 실제 ring/little 목표도 middle을 따라 갱신됨.
- policy observation은 66D임: joint 18, palm-stick position 3, five fingertip-stick position 15,
  stick size 3, palm-frame index region error 3, thumb region error 3, hand-stick relative orientation error 3,
  previous action 18.
- index target은 object local `+z` 윗면의 tail-side axial interval `[-0.60, -0.30]`, thumb target은
  접근 가능한 local `+x` 측면의 interval `[-0.55, -0.25]`임. 각 error는 fingertip에서 해당 rectangular
  surface patch의 최근접점까지의 3D 벡터이며 region 내부는 0임. surface normal 방향 허용 폭은 `5mm`임.
- thumb를 index의 기계적인 반대인 local `-z`에 두지 않음. `-z`는 table-top acquisition에서 바닥에
  막히므로 현재는 index `+z`, thumb `+x`, 기존 middle `-x` cage가 tripod를 이루는 가설을 검증함.
- hand-stick target orientation은 임의 quaternion을 하드코딩하지 않고 `reset_all/reset_stick` 직후의
  설정된 pre-grasp `q_O_H`를 캡처함. startup body pose는 reset 자세가 아니어서 사용하지 않음. 이
  baseline은 고정 stick orientation에서 초기 **pre-grasp** 상대 자세를 보존하는 실험임. 캡처된 값이
  물리적으로 안정적인 canonical grasp라는 뜻은 아니며, 수동/scripted feasibility 검증 뒤 측정한 기능 자세
  quaternion으로 교체해야 함. A2 random orientation에서도 같은 명시 target을 사용해야 함.
- 현재 reward는 cage reach 8, index grip progress 20, thumb grip progress 20, hand-stick orientation 0,
  cage hold 35, clearance lift 100, tool-ready terminal 30000, manip/floor/action penalty로 구성함.
- 2026-07-20 기존 mean cage에서 엄지-검지만 막대에 붙인 채 바닥을 문지르며 hold를 받는 실패를
  확인함. 다음 fresh 검증본은 reach의 기존 12점은 유지하되, hold/lift/tool-ready gate를
  `min(thumb-index cage, thumb-middle cage)`로 변경함. 이 변경만으로도 lift가 나오지 않아 다음 fresh
  실험에서 위 index region과 thumb region을 추가함. hold/lift 가중치, hand-floor, action은 유지함.
- `index_grip`과 `thumb_grip`은 현재 signed previous-step progress이고, `hand_stick_orientation` 구현도 signed지만
  weight 0으로 비활성화함. 파지/lift 검증 뒤 functional target을 확정하고 양수 전용 방식으로 재검토함.
- `tool_ready`는 index region error `<2 cm`, thumb region error `<2 cm`, relative orientation error
  `<15 deg`, balanced cage gate `>0.3`, clearance `>5 cm`를 15 step 유지해야 종료함.
- `scripts/debug/chopstick_functional_probe.py`가 obs source 일치, 두 semantic region bounds, captured `q_O_H`,
  초기 index/thumb/orientation error, 두 cage group, clearance를 출력함.
- 2026-07-20 static compile, manager entity/term resolution, GPU 1-env numeric probe와 1-iteration PPO
  smoke를 통과함. probe에서 observation 63D/action 18D와 각 observation source의 `max_abs_diff=0`을
  확인했고, reset target 캡처 후 첫 zero step relative orientation error는 `3.42 deg`였음. smoke run은
  이름 통일 전 경로인 `indy_wuji_chopstick_acquire/2026-07-20_12-16-28`임. index/thumb region 변경 후에도
  66D numeric probe의 모든 observation source `max_abs_diff=0`과 1-iteration PPO smoke를 통과함. smoke run은
  `indy_wuji_chopsticks_grasp/2026-07-20_17-05-10`이며 성능 판정용 run이 아님.

### Skill A reward 재사용과 확장

| 현재 reward | Skill A에서의 역할 | 처리 방침 |
|---|---|---|
| `finger_cage_reach` | 막대를 엄지-대향 손가락 사이로 유도 | 유지 |
| `finger_cage_hold` | 막대를 cage 안에 넣고 손가락을 오므림 | 유지 |
| `cube_lift` | fake grasp를 배제하고 실제 지지력을 확인 | 유지 |
| `cube_transport` | tool-ready world position으로 이동 | tool pose reward로 확장 |
| `box_orientation` | 잡힌 막대의 world orientation 정렬 | 현재 기준선으로 유지 |
| `transport_success` | 최종 상태를 일정 시간 만족하면 종료 | functional grasp/slip 조건까지 확장 |
| `palm_facing` | 대략적인 접근 방향 shaping | 최종적으로 stick-relative hand rotation으로 교체 |
| `action_rate` | 진동과 과도한 action 변화 억제 | 유지 |

Skill A의 목표 reward 구조:

```text
R_A =
    w_r * r_cage_reach
  + w_h * r_cage_hold
  + w_l * r_lift
  + w_p * r_tool_pose
  + w_g * r_functional_grasp
  + w_T * r_tool_ready
  - w_s * r_slip
  - w_a * r_action_rate
```

가중치의 논리적 순서는 다음을 유지함.

```text
terminal success
>> functional grasp / tool pose
>> lift
>> hold
>> reach
```

이는 쉬운 reach/hold에 머무르는 정책보다 이후 단계를 완료하는 정책이 더 큰 episode return을 받게
하려는 순서임. 논문 숫자를 그대로 복사하지 않고 현재 각 raw term의 episode 기여량을 기준으로 정함.

### 반드시 분리해서 볼 두 orientation

`orientation`이라는 이름으로 다음 두 문제를 섞지 않음.

```text
A. Tool world pose
   T_W_S: 막대가 작업 공간에서 어느 위치와 방향에 있는가

B. Hand-tool relative grasp
   T_H_S = inverse(T_W_H) * T_W_S
   손이 막대의 어느 영역을 어떤 방향으로 잡고 있는가
```

- 현재 `box_orientation`과 `box_ori_to_target`은 A를 학습함. 목표 pose의 주체는 EE가 아니라
  **잡힌 막대/box**임. arm과 finger action은 막대 pose 오차를 줄이는 수단임.
- 최종 functional grasp는 B를 추가로 봐야 함. 같은 막대 world pose라도 손이 끝부분을 잘못 잡거나
  개폐가 불가능한 방향이면 Skill B에 넘길 수 없음.
- `palm_facing`은 물체를 대략 향하는 접근 신호일 뿐 B를 보장하지 않음. 최종적으로는 막대 frame
  기준 hand rotation, index fingertip grip-region target, 필요 시 thumb/middle contact region으로 교체함.
- 전체 hand joint pose를 강제하기보다 constraint-based 목표를 우선함. Wuji hand가 물리를 견디는
  자세를 스스로 찾을 여지를 남김.

한 막대 프록시의 functional grasp 후보:

```text
r_functional_grasp =
    r_hand_rotation_relative_to_stick
  + lambda_p * r_index_fingertip_to_grip_region
  + lambda_c * r_contact_region
```

최종 Skill A 성공 조건 후보:

```text
tool_ready =
    tool_clearance > lift_threshold
    AND tool_position_error < position_threshold
    AND tool_orientation_error < orientation_threshold
    AND hand_tool_grasp_error < grasp_threshold
    AND cage_gate > cage_threshold
    AND hand_tool_slip < slip_threshold
    AND stable_steps >= required_steps
```

`hand_tool_slip`은 `T_H_S`의 시간 변화 또는 hand와 stick 사이 상대 선속도/각속도로 측정함. 처음부터
obs/reward를 모두 늘리지 말고 metric으로 먼저 기록해 성공 rollout의 정상 범위를 측정한 뒤 gate나
reward로 승격함.

### Tool world pose 표현

- **현재 active 구현은 8-corner 통합 reward가 아님.** `cube_transport`의 position best-so-far와
  `box_orientation`의 orientation-error best-so-far 양수 progress를 분리해 사용함. 8-corner
  `KeypointGoalProgressReward`는 `rewards.py`에 비교/재사용 코드로만 남아 있고 active cfg에는 연결되지 않음.
- 현재 직육면체에서는 이 분리형을 먼저 검증하고, 필요하면 8-corner keypoint pose progress와 A/B 비교함.
- 실제 젓가락으로 넘어가면 물리적 의미가 있는 semantic keypoint 인터페이스로 일반화함.

```text
tip  : 물체를 집는 앞쪽 끝
tail : 뒤쪽 끝
grip : 손이 잡아야 하는 영역 중심
side : 단면 roll이 기능적으로 중요할 때만 추가
```

- 원형 젓가락처럼 축 주위 roll이 기능에 영향이 없다면 `side`를 강제하지 않음.
- 기존 8-corner 코드는 버리지 말고 `PoseKeypointProvider`와 같은 공통 인터페이스의 한 구현으로
  보존하는 방향을 우선함.

### Skill A 확장 주의사항 (2026-07-20 검토 완료)

#### World pose 성공을 functional grasp 성공으로 부르지 않음

- 현재 fixed position + yaw 45도 실험의 성공 의미는 “직육면체를 잡아 목표 world pose로 운반”까지임.
- 성공해도 tip 근처를 잘못 잡음, 손 안에서 미끄러짐, 개폐 불가능한 손 배치가 남을 수 있으므로
  “젓가락을 사용할 수 있게 잡음”으로 해석하지 않음.
- 현재 world pose 단계가 통과된 뒤 hand-tool relative grasp를 별도 층으로 추가함. 두 층을 한 fresh
  실험에서 동시에 새로 넣지 않음.

#### Semantic keypoint는 방향성을 보존함

- 직육면체의 8-corner와 실제 젓가락의 semantic point를 같은 provider 인터페이스로 계산함.

```text
current_points = keypoint_provider(object_state)
goal_points    = keypoint_provider(goal_state)
pose_error     = mean_distance(current_points, goal_points)
```

- box provider는 8 corners, one-stick provider는 `tip/tail/grip[/side]`, pair provider는
  `tip1/tail1/tip2/tail2`를 반환하도록 설계함.
- 정사각 단면 box는 축 주위 대칭과 필요 시 앞뒤 대칭을 허용할 수 있지만 실제 젓가락은
  `tip != tail`임. 젓가락 긴 축 오차에 `abs(dot(current_axis, goal_axis))`를 쓰면 앞뒤 반전을
  정답으로 만들므로 금지함. semantic tip/tail로 방향을 보존함.
- 원형 단면에서 기능적으로 무의미한 축 주위 roll은 허용할 수 있지만 tip-tail 반전은 허용하지 않음.

#### Orientation은 절대 양수 연금으로 만들지 않음

- `exp(-k * orientation_error)`처럼 정렬 상태에 매 step 큰 양수를 주면 orientation만 유지하며 다른
  과제를 포기하는 farming이 가능함.
- 현재 보상 설계 원칙은 양수 reward와 음수 penalty를 같은 항에 섞지 않는 것임. orientation shaping은
  best-so-far 양수 progress + terminal을 사용하고, 자세 악화 억제가 필요하면 별도 penalty로 정의함.
- 현재 active `box_orientation`은 에피소드 최저 orientation error를 갱신할 때만 양수이며, 자세 악화와
  이전 최고 자세까지의 복구는 0이라 왕복 진동으로 반복 적립할 수 없음.

#### Lift/clearance 유지 보상을 너무 일찍 제거하지 않음

- A′ 실험에서 lift weight 0이어도 목표 높이까지 운반한 순간은 있었지만, position progress를 받은 뒤
  다시 내려놓고 hold를 farming해 success 0%가 됨.
- Skill A에서는 `lift/clearance 유지 + tool pose progress + terminal tool_ready + 즉시 종료`를 유지함.
- lift 감액/제거는 tool_ready terminal이 충분히 발견되고 공중 유지가 안정된 뒤 단일-run curriculum
  또는 별도 fresh A/B로만 검토함.

#### Functional grasp는 최소 constraint부터 추가함

- 처음부터 엄지/검지/중지 전 관절 목표와 손목 quaternion 전체를 강제하지 않음.
- 현재는 `index grip region + thumb grip region + captured pre-grasp orientation + balanced cage`까지 active임.
- balanced cage 단독 fresh에서 lift가 나오지 않아 다음 **별도 fresh**로 index/thumb region을 함께 추가함.
  middle target/contact region은 아직 넣지 않았으며 현재 변경의 결과를 먼저 판정함.
- 최종 후보는 `stick-relative hand orientation + index grip-region target + thumb/middle contact region +
  기존 cage hold` 정도임.
- 전체 joint target은 위 constraint로 기능적 파지가 나오지 않는다는 실측 근거가 생긴 뒤에만 검토함.

#### A1 semantic anchor와 stability 후속 설계

- 아래는 현재 active 코드가 아님. balanced cage fresh 결과와 canonical grasp feasibility를 확인한 뒤
  항목별 별도 fresh 실험으로 적용함.
- index point를 `s_index in [-0.60, -0.30]`인 local `+z` surface patch까지의 최근접 거리로 바꾸고,
  영역 내부 error를 0으로 두는 구현을 적용함. 기존 palm-frame 3D error 형태는 유지함.
- thumb anchor는 `s_thumb in [-0.55, -0.25]`인 local `+x` surface patch로 적용함. index의 기계적인
  반대면이 아니라 table-top에서 접근 가능한 면을 선택한 가설이며, 영상과 lift 결과로 검증해야 함.
- index+thumb가 같은 axial 구간에서 pinch를 만들어도 긴 막대의 축 방향 토크까지 충분히 막지는 못함.
  그 다음에 middle 또는 palm support anchor를 길이축으로 떨어진 구간에 추가하고, 필요할 때만 contact
  axial span을 metric/reward로 승격함.
- orientation shaping은 hand-relative full quaternion 하나에 영구 고정하지 않고 `long-axis + roll`로
  분해하는 방향을 우선함. tip-tail은 기능적으로 다르므로 axis dot에 `abs()`를 쓰지 않음. square proxy는
  roll을 사용할 수 있고 원형 stick은 roll weight를 0 또는 작은 값으로 둠.
- stability는 best-so-far가 아니라 lifted current-state annuity 후보임. 단순
  `v_stick - v_palm`은 palm 회전에 의한 정상 선속도를 slip으로 오인할 수 있으므로
  `T_H_S(t)=inverse(T_W_H(t))*T_W_S(t)`의 step 간 translation/rotation 변화량으로 relative linear/angular
  speed를 계산함. 먼저 metric으로 정상 분포를 측정한 뒤 scale/threshold를 정함.
- balanced cage는 SDF 가상점 기반 **기하 조건**이지 실제 contact force가 아님. 이후 grouped physical
  contact를 넣을 때는 thumb/index/support contact와 cage를 별도 metric/reward로 유지함.
- 적용 순서는 현재 `index region + thumb accessible region`까지 진행됨. 다음은 결과 판정 후
  `axially separated support anchor -> axis/roll -> relative-pose drift metric -> stability reward/success gate`임.
- grouped contact/span/slip/orientation과 hold weight 감액을 한 번에 넣지 않음. 제안 가중치
  `hold 10~15`, `contact 25`, `stability 20~40` 등은 raw 분포 측정 전에는 채택값이 아님.

#### Cage는 필요조건이지 충분조건이 아님

- `cage_inside_frac`, `cage_span`, `opposition`, `finger_cage_hold`가 높아도 실제 하중 지지나 기능적
  파지를 보장하지 않음. 과거에도 선분만 물체를 관통하는 퇴화 해가 있었음.
- 최종 판정에는 clearance, world pose, hand-tool relative pose, relative slip, stable steps가 함께 필요함.
- 현재 `ObjectAtGoalHeld`는 `position + cage + orientation + hold_steps`만 검사함. goal이 공중이라 lift를
  간접적으로 요구하지만 explicit clearance, relative grasp, slip은 아직 미구현임.

#### 두 개의 느슨한 젓가락을 바로 집게 하지 않음

Skill A 자체도 다음 하위 단계로 나눔.

```text
A0. 직육면체 1개: 목표 world position/orientation
A1. 젓가락 프록시 1개: canonical functional hand-tool grasp + lift
A1.5. 젓가락 1개: Skill B 시작용 world ready pose
A2. 젓가락 1개: spawn/goal position 및 world orientation 강건성
A3. 두 젓가락: 이미 가까운 초기 상태에서 functional pair 형성
A4. 두 젓가락: 테이블에서 실제 획득
```

- A3는 curriculum reset 또는 약하게 연결된 pair proxy로 시작할 수 있음.
- A3 전에는 pair opening, lower/upper stick 역할, 두 stick 상대 slip을 Skill A0/A1에 섞지 않음.

#### Skill A terminal state를 Skill B 자산으로 남김

성공 시 다음 상태를 저장할 수 있는 recorder/export 구조를 Skill B 연결 전에 구현함.

```text
robot joint position/velocity
hand pose
stick pose/velocity
finger-stick relative transforms
두 stick 단계의 opening
contact/cage state
```

- 단일 terminal pose만 저장하지 않고 성공 분포를 저장함.
- Skill B reset에는 실제 terminal state와 작은 position/orientation/opening/contact perturbation을 사용함.

#### 한 실험에서 한 축만 바꿈

- object size, spawn position/orientation, goal position/orientation, functional target, reward weight,
  success 조건, action space를 한 번에 바꾸지 않음.
- reward/task/observation/action 의미가 바뀌면 fresh run. 같은 설정의 학습 연장만 resume함.
- 현재 orientation 기준선이 실패하면 keypoint, functional grasp, 얇은 stick을 동시에 넣지 않고
  `orientation_stage_active`, `box_ori_error`, final clearance, cage gate, action saturation으로 원인을 먼저 나눔.

### Skill B: 계층형 젓가락 사용 정책 (Skill A 이후)

Skill B는 두 젓가락이 손에 안정적으로 잡힌 상태에서 시작함. 초기 구현에서 테이블 획득까지 다시
학습시키지 않음.

#### Low-level: 파지 안정화와 opening 제어

입력 후보:

```text
hand joint state
stick1 pose relative to hand
stick2 pose relative to hand
stick1-stick2 relative pose
current opening
desired opening
previous action
```

출력은 우선 thumb/index/middle finger joint target만 사용하고 arm은 high-level이 담당함.

두 독립 quaternion만 쓰지 말고 pair frame을 정의함.

```text
t1, t2       = 두 젓가락 tip
b1, b2       = 두 젓가락 tail
pair_origin  = (t1 + t2) / 2
long_axis    = normalize((t1 - b1) + (t2 - b2))
closing_axis = normalize(t1 - t2)
opening      = ||t1 - t2||

pair state = tool pose 6D + opening 1D
```

초기 난도를 낮추기 위해 lower stick은 hand-relative pose를 강하게 유지하는 quasi-fixed stick으로,
upper stick은 뒤쪽 지지점을 유지하며 tip opening을 만드는 주 이동 stick으로 취급함.

Low-level reward 후보:

```text
pair parallelism
tip depth alignment
desired opening tracking
lower-stick hand-relative stability
upper-stick anchor/contact stability
hand-tool slip penalty
tool linear/angular velocity penalty
```

#### High-level: 젓가락 물체 파지와 운반

입력은 pair tool state, tool/hand 기준 object pose, target pose, object size/grasp width이며, 출력은
arm Cartesian motion과 desired opening command로 구성함.

현재 reward의 역할을 다음처럼 옮김.

| 현재 직접 손 파지 | Skill B의 젓가락 물체 파지 |
|---|---|
| `finger_cage_reach` | 두 tip을 물체 양쪽으로 접근 |
| `finger_cage_hold` | 물체를 두 tip 사이에 위치 |
| cage gate | bilateral tip contact 또는 양쪽 파지 조건 |
| `cube_lift` | 젓가락으로 물체 lift |
| `cube_transport` | 잡힌 물체의 goal pose progress |
| `transport_success` | 물체 goal pose 안정 성공 |

High-level reward 후보는 `tip_reach`, `closing_axis_alignment`, `bilateral_contact`,
`object_lift`, `object_goal_keypoint`, `object_success`임.

### 정책 연결과 reset distribution

초기 연결은 다음 FSM으로 함.

```text
ACQUIRE   : Skill A 실행
TOOL_READY: tool_ready 조건을 N step 만족
USE       : Skill B 실행
```

Skill B를 완벽한 수동 자세에서만 시작시키지 않음. 그렇게 하면 Skill A가 넘기는 작은 위치·회전·접촉
오차에서 바로 실패함.

연결 학습 순서:

1. Skill A의 성공 terminal state를 여러 개 저장
2. 이 상태들을 Skill B의 reset distribution으로 사용
3. position, orientation, opening, contact에 작은 perturbation 추가
4. Skill B가 실제 handoff 오차를 복구하도록 학습
5. 두 정책이 각각 안정된 뒤 전체 rollout fine-tuning 검토

### 개발 순서 (순서 유지)

현재 A1 즉시 순서는 다음으로 고정함.

1. balanced cage 단일 변경 fresh run에서 middle 참여, raw hold, clearance를 판정
2. RL reward를 더 켜기 전에 수동/scripted canonical one-stick grasp feasibility 확인
3. 실제 안정 파지에서 `q_O_H`를 측정하고 현재 reset-captured pre-grasp target과 비교
4. 도달 가능한 target이 확정된 뒤 orientation metric 분포 확인
5. `HandToolOrientationProgressReward`를 stage-latched best-so-far 양수 progress로 바꾸고 작은 nonzero
   weight로 별도 fresh run
6. one-stick tool-ready 성공 후에만 world ready pose(A1.5), grip region/thumb target, randomization(A2)을
   각각 분리 실험

아래는 그 이후를 포함한 전체 순서임.

1. 현재 active 분리형 reward로 직육면체 fixed position + yaw 45도 orientation 성공 여부 판정
2. 다음 fresh에서 success에 explicit clearance를 추가해 pose+cage+clearance terminal 검증
3. random goal position만 추가
4. random goal orientation만 추가
5. 8-corner/분리형 계산을 `PoseKeypointProvider` 계열 인터페이스로 정리
6. `hand_tool_relative_pose`, relative linear/angular speed를 **metric으로 먼저** 추가
7. 성공 rollout에서 functional grasp와 slip의 정상 범위를 측정
8. one-stick semantic `tip/tail/grip` + stick-relative hand orientation + grip-region constraint 추가
9. 얇고 긴 젓가락 크기에서 A1/A2를 검증
10. `tool_ready` terminal에 world pose + relative grasp + slip + stable steps 통합
11. Skill A terminal state recorder/export 구현
12. 두 젓가락이 이미 가까운 상태에서 functional pair/open-close Skill B-Low 학습
13. 고정 arm에서 젓가락 tip으로 물체 파지/lift
14. Skill B-High에 arm 운반과 object goal pose 추가
15. Skill A terminal distribution에서 Skill B를 시작시켜 FSM 연결

현재 orientation run이 실패하면 즉시 Skill B로 넘어가지 않음. 먼저 `orientation_stage_active`,
`box_ori_error`, final clearance, cage gate, action saturation을 보고 orientation 유인 부족과 파지 토크
부족을 분리함.

### 논문별 역할과 적용 한계

| 우선순위 | 논문 | 이 프로젝트에서의 역할 |
|---|---|---|
| 주 기반 | Pavlichenko and Behnke, *Dexterous Pre-grasp Manipulation for Human-like Functional Categorical Grasping* (arXiv:2307.16752v2) | Skill A의 cage/SDF reach, hold, orient, constraint-based grasp, lift, terminal reward, 단계적 reward scaling |
| 주 기반 | Yang, Yin, and Liu, *Learning to Use Chopsticks in Diverse Gripping Styles* (arXiv:2205.14313v3) | 안정적인 gripping pose, 물체 없는 open-close 검증, 7-DoF chopstick pair 표현, grasp와 이후 사용 단계 분리 근거 |
| 주 기반 | Xu et al., *Hierarchical Reinforcement Learning for Articulated Tool Manipulation with Multifingered Hand* (arXiv:2507.06822v1) | Skill B의 low-level finger/tool configuration과 high-level arm/tool goal 계층 |
| 보조 | Allshire et al., *Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger* (arXiv:2108.09779v2) | 8-keypoint 기반 object 6D pose reward와 observation |
| 보조 | Ke et al., *Grasping with Chopsticks: Combating Covariate Shift in Model-free Imitation Learning for Fine Manipulation* (arXiv:2011.06719v1) | Skill A->B handoff의 오차 누적, 초기 상태 perturbation과 교정 분포 확장 근거 |
| 장기 확장 | Kedia et al., *SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation* (arXiv:2602.16863v2) | tool use를 object-centric goal pose sequence 추종으로 확장 |

적용 시 과장하지 않음.

- Functional Pre-grasp가 현재 reward의 가장 직접적인 기반이지만 현재 코드는 해당 논문의 완전한 재현이 아님.
- Yang et al.은 테이블 위 두 젓가락 획득을 end-to-end로 해결하지 않으며, 안정 gripping pose와 이후
  controller를 나눈 관점이 근거임.
- Xu et al.의 도구는 구조적으로 연결된 tweezer형 articulated tool임. 두 개의 느슨한 젓가락에는
  pair-frame 안정성과 두 stick의 상대 slip 조건을 별도로 추가해야 함.
- TriFinger의 8-keypoint가 pose 표현의 출발점이며, 현재 symmetry 처리와 position->orientation
  stage latch는 프로젝트에서 추가한 방식임.
- 느슨한 두 젓가락을 테이블에서 획득해 사용까지 연결하는 전체 구조가 이 연구의 차별화 후보임.

### 후속 agent 작업 규칙

- 현재 position/orientation reward의 목표 주체를 EE로 바꾸지 않음. **tool pose가 목표**, EE/hand pose는
  functional relative grasp의 별도 조건임.
- world tool pose와 hand-tool relative pose를 하나의 orientation metric으로 섞지 않음.
- relative pose와 slip은 먼저 metric으로 검증하고 정상 범위를 얻은 뒤 reward/obs/success에 추가함.
- reward, observation, action, physics를 한 실험에서 동시에 바꾸지 않음. 귀속 가능한 최소 변경으로 비교함.
- Skill A의 one-stick `tool_ready`가 검증되기 전에는 두 젓가락 획득이나 Skill B hierarchy 구현을 시작하지 않음.
- 새 task alias를 늘리지 않고 기존 `Indy-Wuji-Box-Transport`에서 명확한 run과 commit으로 실험을 구분함.
- action/observation shape 또는 의미가 바뀌면 fresh run을 사용함. 같은 설정의 단순 연장만 resume 허용함.

## 상세 실험 이력과 이전 다음 단계 (2026-07-17 당시 기록)

아래 절은 lift/transport 및 orientation 설계가 발전한 과정의 기록임. 현재 이어서 작업할 때는 위
`연구 아키텍처와 이어서 작업할 기준`을 우선함.

### 현재 달성 상태
- `Indy-Wuji-Box-Transport`에서 랜덤 직육면체를 cage로 잡고, 자세를 크게 무너뜨리지 않은 채
  공중으로 들어 유지하는 단계까지 확인함. 이제 `lift 가능성` 자체보다 goal 운반과 정착 품질을
  비교할 단계임.
- 현재 goal 성공은 box 중심 위치만 봄: `|p_box - p_goal| < 0.05m`, cage gate `> 0.3`,
  15 step(0.5초) 연속 유지. box orientation은 reward와 success에 아직 들어가지 않음.
- 다음 실험의 기준선은 현재 분리형 보상임: lift(공중 파지 유지) + transport(goal 거리 신기록)
  + terminal success. 이 기준선을 보존하고 통합형 후보와 fresh A/B 비교함.

### ★ 재학습 원칙 (2026-07-16 사수님 지시, 전 에이전트 적용)
- **보상/태스크 구조가 바뀌면 resume 금지, fresh run.** resume 결과는 "옛 정책+적응"이라
  설계 검증으로 무효 (귀속 불가 + 옛 습관 오염). resume 허용 = 같은 설정의 순수 연장뿐.
- 보상 단계화가 필요하면 resume이 아니라 **단일 런 내 curriculum manager**
  (mdp.modify_reward_weight)로 스케줄할 것.

### 다음 개발 로드맵 (순서 고정)

#### 1. Lift/transport 분리형과 통합형 A/B 테스트

- **A안(현재 기준선, split)**
  - `r_lift = cage_gate * clamp(clearance / 0.08, 0, 1)`의 절대형 공중 유지 보상.
  - `r_transport = cage_gate * (phi(d_t) - best_phi)^+`,
    `phi(d)=0.05/(0.05+d)`의 goal 거리 신기록 보상.
  - 장점: goal에서 멀어도 lift를 먼저 배울 수 있어 탐색 신호가 강함.
  - 위험: lift가 8cm 위에서 포화 연금이 되어 goal 접근보다 호버링/오버슈트를 선호할 수 있음.
- **B안(통합 후보, merged)**
  - 기존 킵 카드인 `r_goal_proximity = cage_gate * phi(d)` 한 항으로 공중 유지와 goal 접근을
    동시에 지불함. goal 자체가 공중에 있으므로 가까이 머무는 것이 lift와 transport를 함께 뜻함.
  - 적용 실험에서는 `cube_lift`와 `cube_transport` weight를 0으로 두고 통합항만 켬.
  - `cube_lift` cfg 자체는 삭제하지 않음. `CustomRewardManager`의 `surface_z` metric 배선이
    해당 params를 읽으므로 weight 0으로 은퇴시키는 방식으로 비교함.
  - 장점: 8cm 이상에서 높이만으로 받는 연금을 없애고, 잡은 채 goal에 가까이 유지할 이유를 줌.
  - 위험: goal 반경 바로 밖에서 proximity 연금을 받는 boundary camping, 먼 거리/table 위에서의
    작은 연금, 초기 lift 신호 약화 가능성이 있음. terminal success가 goal 안 체류 연금의 상한임.
- A/B에서는 reward 구성 외의 조건을 바꾸지 않음: 같은 box 분포, goal, seed 묶음, env 수,
  총 environment step, PPO cfg, drop penalty weight를 사용함. reward가 바뀌므로 둘 다 fresh run이며
  기존 lift checkpoint에서 resume하지 않음.
- run/task alias를 추가 등록하지 않음. `Indy-Wuji-Box-Transport` 하나를 유지하고, 확인한 코드
  commit과 명시적인 run 폴더명으로 A/B를 구분함.
- **판정 지표**: success rate, time-to-success, `cube_final/cube_clearance`, goal position error,
  cage gate/hold, `cube_speed`, drop rate, action rate, 최대 clearance(오버슈트), box 크기 구간별 성공률.
  play에서는 성공 종료를 늦춰 8초 동안 자세 유지/goal 정착을 확인함.
- **B안 채택 조건**: A안과 같거나 높은 success, 더 낮은 drop/오버슈트, 더 작은 goal error,
  8초 play에서 안정 유지, table/boundary camping 없음. 하나라도 뚜렷이 악화되면 A안을 유지함.

##### 1-a. 선행 실험 A′(lift-off) — 2026-07-17 사용자 시작

- 구성: A안에서 `cube_lift` weight만 0 (term 삭제 금지 — surface_z metric 배선 유지),
  transport 일시불(best-so-far φ)과 나머지는 동일. fresh run.
- **실행 확인 (env.yaml 검증)**: 박스 run `2026-07-17_23-15-16` (질량 0.1, lift 0, transport 4000,
  r_T 30000). 대조군 = 박스 run `2026-07-16_16-33-21` (질량 0.1, lift 50, 그 외 동일 — success
  43.5% 상승 확인본). lift 유무 단일 변수 A/B임.
- 병행: 큐브 질량 비교 run `2026-07-17_23-06-15` (0.1kg, lift 50 — 대조군은 89.4% 수렴본
  `2026-07-16_16-05-23`의 0.2kg). run 파라미터 대장은 `ACTIVITY_2026-07-17.md` 참조.
- 목적: lift 연금 없이 φ의 높이 구배만으로 사다리가 유지되는지 확인. 결과가 B안 필요성까지
  한 번에 판독함 (아래 문제 1·2가 정확히 B안이 고치도록 설계된 문제들이므로).
- 예상 문제와 TB 시그니처 / 대응:
  1. **재도전 무보상** — best-so-far라 낙하 후 재상승 여정이 기록 미달인 동안 0원.
     시그니처: 낙하율이 안 떨어지고 success 정체.
     대응: ① B안 연금 승격 (연금은 재접근도 매 스텝 지급) ② `ObjectToGoalProgressReward`에
     gate 상실 시 φ_best 재시드 옵션 — 단, 고의 놓기-재운반 파밍 루프가 열림 (사이클당 ~+107,
     r_T +1000이 지배해 계산상 비수익이지만 r_T를 모르는 초반 정책에겐 유혹 여지) → 보조 옵션
     ③ lift 감액 부활 (0 대신 10~15).
  2. **중간 처짐 무비용** — 기록 후 처져도 벌 없음 + 재상승 보상 없음. φ는 마지막 5cm에
     65% 집중이라 d≈0.10~0.15가 인센티브 공백.
     시그니처: error_pos 0.10~0.15 고원 + episode length 긺.
     대응: ① B안 연금 (처짐 = 즉시 소득 감소) ② `potential_eps` 0.05→0.10으로 φ 완만화
     (중간 구간 배분↑, 종말 집중 65→50%; fresh 필요). 음수 시간 페널티는 단일 부호 원칙과
     탐색 회피 함정 이력으로 배제.
  3. **초반 시드 저하** — 매 스텝 연금 → 신기록 일시불로 바뀌어 우연 들기의 강화 확률 하락.
     시그니처: 첫 success가 v2.1 기준(~iter 3,000)의 2배(iter 6,000)를 넘도록 없음.
     대응: ① lift 훈련바퀴 커리큘럼 — 단일 런 내 curriculum manager(`mdp.modify_reward_weight`)로
     lift 50 시작 → N iter 후 0 (drop 커리큘럼과 같은 메커니즘, ★재학습 원칙 부합)
     ② iter 6,000까지 무 success면 중단 판정.
- **판정 트리**: 시그니처 1·2 발생 → B안 fresh / 3만 발생 → lift 커리큘럼 fresh /
  무증상 + v2.1급 수렴(65% @ ~4,500) → lift 영구 은퇴 확정 (박스에도 적용, 구성 단순화).
- 비교 기준선 (큐브 v2.1, run 2026-07-16_16-05-23): 첫 success ~iter 3,000 → 65% @4,463 →
  89.4% @7,440, 낙하 3.0%, error_pos 0.046, 오버슈트 없음.
- **★ 판정 (2026-07-18, iter 6,427 대조): A′ 기각 — 시그니처 1+2 혼합형 → B안 fresh 처방.**
  - 실측: success 0% 전 구간 (대조군 35.8%, 첫 success iter 854). 단, **들기·운반은 됨** —
    max clearance 0.200 (목표 높이), transport 수입 3.35 vs 3.57 (동급).
  - 실패 지점 = **정착**: φ 현금화 → 내려놓음 → hold 파밍 (hold 수입 2배, time_out 83%,
    종료 시 clearance −0.009). goal 체류가 테이블 체류보다 수입이 같은데 힘만 들어서.
  - lift의 실제 역할 재해석: 씨앗뿐 아니라 8cm+ 포화 연금이 "공중 체류비"를 지불해
    r_T 발견(우연한 15스텝 유지)을 다리 놓음.
  - ⚠ 지표 함정 2건 기록: ① `cube_final/cube_clearance`만 보면 "들기 전무"로 오판
    (에피소드 끝에 내려놓으면 0) — **`cube_max/` 변형 필수 확인**. ② per-step 평균
    (`Metrics/cube/...`)은 에피소드 구성(성공 조기종료 vs 타임아웃)에 오염 — 런 간 비교는
    `cube_max/cube_final`로. 실제로 파지 자세 가설(이동 인센티브→자세 붕괴)은 per-step
    평균으로는 성립해 보였으나 max 지표로 기각됨 (A′ max 맞물림 0.995, cage 0.873으로
    대조군보다 오히려 깊음).
  - 중간 탐색 산물: iter ~75% 지점 max clearance 0.51/error 0.76 스파이크 = 던지기 시기
    (일시불 구조의 기록 갱신 해킹 시도, 자연 소멸).

#### 2. Goal orientation을 포함한 6D pose 성공 조건

- **적용 시점은 아래 '일반화 순서'(2026-07-17 사용자 확정)를 따름 — random goal position 다음.**
- A/B 승자를 먼저 고정한 다음 orientation을 추가함. reward 통합과 orientation 도입을 같은 run에서
  동시에 바꾸지 않음. 두 효과가 섞이면 실패 원인을 분리할 수 없음.
- current success의 box 중심 거리 조건은 유지하고 goal quaternion/orientation error를 추가함.
  후보 성공식은 `position_error < 0.05m AND orientation_error < 15deg AND cage_gate > 0.3`을
  15 step 유지하는 것임. 15도에서 시작하고 안정화 후 10도로 조이는 것은 별도 실험으로 함.
- orientation error는 quaternion 성분별 차가 아니라 geodesic angle을 사용함:
  `theta = 2*acos(clamp(|dot(q_box, q_goal)|, 0, 1))`.
- 현재 random box는 단면 두 축이 같은 square prism 계열이므로 geometry-only 목표에서는 동일한
  대칭 자세를 오답 처리하지 않도록 symmetry-aware 최소 orientation error를 써야 함. 기능 방향이
  있는 젓가락/tool 단계에서는 대칭 허용 대신 semantic keypoint/정확한 목표 회전을 사용함.
- policy가 목표 회전을 알 수 있도록 goal orientation 또는 box→goal 상대 orientation을 observation에
  추가함. 고정 goal orientation만으로 먼저 검증한 뒤 random goal orientation으로 확장함.
  observation dim이 바뀌므로 이 단계는 반드시 fresh run임.
- orientation shaping은 절대 양수 연금으로 단독 지급하지 않음. cage gate와 position 근접 gate를
  붙인 best-so-far orientation progress, 또는 current/goal 8-keypoint distance를 후보로 비교함.
  성공 terminal reward는 position + orientation + cage를 모두 만족할 때만 지급함.
- goal marker도 구가 아니라 orientation을 볼 수 있는 frame/ghost box로 표시하고,
  `position_error`, symmetry-aware `orientation_error`, pose success rate를 TensorBoard metric으로 추가함.

#### 3. 일반화 순서 (2026-07-17 사용자 확정, 같은 날 보완 반영)

> **2026-07-18 재우선순위 (사용자)**: B안 통합은 **보류** (박스 goal_proximity 배선은 weight 0으로
> 잔류, 큐브 배선은 철회). 0-a(유지력 조이기)와 goal 위치 랜덤화보다 **orientation 성공**과
> **직육면체 크기 경우의 수 확대**를 앞당김. 실행 층:
> - **orientation v1 (구현 완료, 스모크 통과)**: success에 "스폰 자세 대비 대칭 최소각 < 15°"
>   추가 (`ObjectAtGoalHeld.ori_limit=0.2618`, box_mdp_cfg 기본값) = 기울이지 않고 나르기.
>   obs 불변 (goal 자세 상수 + box_quat 기존 보유 — 죽은 채널 회피). shaping 없음 (씨앗 확률
>   높을 것으로 예상; 미이륙 시 quaternion 항이 아니라 **keypoint 거리**로 — TriFinger 실측:
>   "pos+quat 분리 보상은 ori 학습이 느림", thesis.md). 지표: `Metrics/cube*/box_ori_error`.
> - **v2 (조건부)**: constraint-lite — success gate에 엄지-검지/중지 opposition 임계 추가
>   ("무엇으로 잡았는지" 조건, 논문 constraint-based의 염가판). 갈고리 파지(종료 시 맞물림
>   음수 실측)가 ori 제어를 막으면 투입.
> - **v3 (젓가락 브리지)**: 풀 constraint-based target grasp g (검지 끝 target + 손 회전
>   + 들기, 논문 Eq.20). 직육면체 = 젓가락 1개 프록시 관점 (사용자 방향).
> - **병행 슬롯**: 크기 랜덤화 범위 확대 — 확장 방향(얇게/크게/ratio) 사용자 결정 대기.
>
> **2026-07-19 (밤) 현재 상태**: 스틱 라인 **보류** (평면 위 얇은 물체 = 누르기 파밍 +
> 말단 토크 포화 실측; 복귀 카드 = 받침 스폰 커리큘럼 + 맞물림 팩터 + 말단 effort 상향).
> 슬롯 A(WRAP, 매달림 70°→42°) 완성 우선 — **"ori 완성 패키지" 구현됨, 스모크 미실시**:
> lift_height 0.20(호버 급여 제거) + ori 커리큘럼 45→30→15°(curriculums.py 신규, r_T 씨앗)
> + 최근접 대칭 상대회전 obs(+3 → **policy 67**, 구 체크포인트 play 호환 깨짐).
> ⚠ 다음 작업자: 런치 전 4 env 스모크 필수 (obs dim 변경 + 커리큘럼 첫 사용).
> 상세 파일 목록은 ACTIVITY_2026-07-19.md "ori 완성 패키지" 절.
>
> **확정 진행 순서 (2026-07-18 사용자, 런치 후)** — 현 라운드 다음의 계단:
> 1. (도는 중) 슬롯 A `2026-07-18_15-37-42`: **고정 ori + 고정 position** 수렴 확인
>    ∥ 슬롯 B `2026-07-18_15-47-34`: ori 없이 크기 확장(단면 1.5~6cm, 길이≤20cm) —
>    "얇은 것도 잘 집나". 판독 = 크기-버킷 success 분해 (단면 1.5~2.8cm 구간이 핵심:
>    물리 한계 vs 학습 한계).
> 2. **랜덤 position** — 싼 단계: goal 위치 obs가 command 기반이라 command ranges만
>    넓히면 배관 그대로 (obs 불변, 후보 범위는 cube_grasp_env_cfg.py 주석 stage-2 값).
>    fresh 필요하나 코드 몇 줄.
> 3. **랜덤 ori (v2 배관 공사)** — command에 quat 추가(pose command화) + obs에 goal
>    ori(또는 상대 ori) 추가 + square_prism_ori_error를 상대 quat(q_goal⁻¹·q_box)
>    시그니처로 확장 + goal 마커 ghost box화. obs dim 변경 = 무조건 fresh.
> - 2·3은 **순차 투입** (동시 금지 — 실패 원인 분리). 단 2가 싱겁게 통과하면 3에서 합류 가능.
> - v1 목표 ori는 command가 아니라 판정 함수 상수(월드 정렬=스폰 자세)임 — command에는
>   orientation이 아직 없음. 스폰 자세도 고정 (reset pose_range에 x/y만 있음).

0. **(판정 완료 2026-07-18)** 큐브 질량: **0.1kg 승** (98.2% @6,866 vs 0.2kg 88.0% 동일 시점,
   첫 이륙 4배 빠름. 단 0.1kg은 커리큘럼 초반값 — 실물 이관 시 질량 복귀 계획 필요).
   박스 lift: **A′(lift 0) 기각** (위 1-a 판정) → 다음 라운드 = 슬롯 A: B안 fresh
   (CLI 오버라이드, box_mdp_cfg의 goal_proximity 배선 완료) ∥ 슬롯 B: 승자 구성(lift 50)
   + hold_steps 15→30 (0-a 조기 실행). 이 둘이 자연 A/B — 승자가 1번(랜덤화 축)으로 진출.
0-a. **유지력 조이기 편입** — 승자 고정 직후 fresh에 success `hold_steps` 15→30 동반.
   성공 "정의 상향"은 비교 변수가 아니라 요구 조건이라 혼입 허용. 단 `gate_threshold`
   0.3→0.4는 같은 fresh에 넣지 않고 다음 차수로 (실패 시 원인 분리를 위해 한 번에 하나).
1. **랜덤화 축 확장** — 한 묶음으로 계획하되 실제 투입은 축별 순차 여부를 그때 결정
   (한 fresh에 축 3개면 실패 귀속이 어려움):
   - goal position: cube_goal command ranges 고정점 → 범위
     (후보: `cube_grasp_env_cfg.py` 주석의 stage-2 값)
   - 물체 spawn 위치: 현행 ±6~8cm → 확장 (목표 범위 미정)
   - **박스 크기: 현행 단면 3~6cm·비율 1.5~3 → 확장** (2026-07-17 사용자 추가.
     확장 목표치 미정 — 크기-버킷 성공률 분해로 현행 하한 취약점 실측 후 결정)
   - command/이벤트 분포 변경 = fresh.
2. **orientation 결합 fresh** — goal orientation(또는 box→goal 상대 orientation) obs 추가
   + orientation shaping(gate 걸린 best-so-far) + success 판정(position + geodesic/
   symmetry-aware error + cage gate)을 **한 fresh에 동시 투입** (세부는 위 로드맵 2 참조).
   ⚠ obs만 먼저 넣고 보상/판정이 참조하지 않는 학습 run은 금지 — 정책이 그 채널을
   무시하도록 학습됨 (죽은 채널). obs 배관 확인은 학습 전 1 env 스모크로만 함.
   obs dim 변경 = 반드시 fresh.
3. 초기 box yaw 랜덤화(우선 ±30도)를 켜고 다양한 크기/초기 자세에서 성공률을 확인함.
4. 이후 젓가락 진입에서 IK action 전환 여부와 functional target grasp `g`를 정의함.

- (상비 장치 제안, 미확정) drop penalty curriculum — 독립 단계가 아니라 매 fresh에 싣는
  단일 런 스케줄(탐색기 0 → success 등장 후 음수)로 재분류. fresh 시작부터 강한 낙하
  페널티를 켜서 탐색 회피를 만들지 않음.
- (제안, 미확정) 젓가락 직전 얇은 물체 브리지 — 폭 하한 3→2→1cm 단계 확장 또는
  grip_capacity로 커플링 pinch 한계 실측. 크기-버킷에서 얇은 쪽 성공률이 낮으면 필수로 승격.
- (보류, 2026-07-17 사용자) 단계별 통과 게이트 수치화는 미정 — 당장은 버킷별 분해 지표
  (goal 구역/크기/yaw별 success)만 유지하고 숫자 기준은 각 단계 진입 시 결정.

### 큐브 태스크 (Indy-Wuji-Cube-Grasp) 상태
- obs 57 동결 (기존 체크포인트 play 호환). 보상은 v2.1로 갱신됨 (보상 실험 테스트베드 역할):
  reach 8 / hold 15 / lift 50(0~8cm 사다리) / transport 4000(전 구간 역수 φ=0.05/(0.05+d),
  best-so-far, gate 곱, 단일 부호) / r_T 30000(goal ±5cm + gate 0.5s 유지 → +1000·즉시 종료) /
  drop 0 / palm 4 / manip 1 / floor 1. goal = 고정점 (0.62, −0.20, BASE_Z+0.20).
- 릴레이 구조: lift(0~8cm) → φ(연속, 근거리 집중) → r_T(도착·종료). lift_height=0.08은
  상한이 아니라 포화(그 위에서 만점 유지, 증가만 정지).
- **2026-07-17 수렴 판정**: run 2026-07-16_16-05-23 — success 89.4% @iter 7,440, 낙하 3.0%,
  error_pos 0.046, 오버슈트 재발 없음. φ 전 구간 설계 검증 + 로드맵 ①관문(고정 위치 운반) 통과.
- play 관찰: 성공은 하나 파지에 진동이 있고 장기 유지는 약함 — 성공=0.5초 즉시 종료라 그
  이후 구간은 학습 분포 밖(OOD) + gate 0.3의 관대한 합격 기준 + bang-bang 액션(action_rate raw ~2.3).
  유지력 카드(다음 큐브 fresh): hold_steps 15→30~60, gate_threshold 0.3→0.4.
  depth_max 증량은 수박씨 배출 이력(40%+ 오므림 시 간격 2.8cm)으로 배제.

### 킵해둔 카드 (조건부)
- 기존 근접 연금 카드는 위 로드맵의 lift/transport 통합 B안으로 승격됨. A/B 결과 전에는 기본안으로
  확정하지 않음.
- 팔꿈치 자세: arm_floor(팔 링크 높이 페널티) 설계 있음 (git 이력 fe2c6fa 근방) — 필요 시 부활.
- 8-keypoint 물체 표현(위치+회전+크기 통합)은 box orientation shaping 후보로 앞당김.
  box 단계에서 geodesic angle 방식과 비교하고, 젓가락에서는 semantic keypoint로 확장함.

## Project Context

- 목표는 Isaac Lab 기반 Indy7 + Wuji hand end-effector tracking RL 환경 구성임.
- 최종 조작 목표는 젓가락을 기능적으로 잡고 젓가락질이 가능한 파지 상태를 학습하는 것임.
- `Dexterous Pre-grasp.pdf` 계열 functional grasp/pre-grasp 아이디어가 최종 연구 방향의 주 기준임.
- 현재 workspace는 `~/wuji_indy_lab_51`임.
- 현재 실제 코드 디렉터리는 `~/wuji_indy_lab_51/nrmk_isaaclab_wuji`임.
- 초기 문서에는 repo가 `~/wuji_indy_lab_51/nrmk_isaaclab_public`로 기록돼 있었음.
- 현재 작업은 `nrmk_isaaclab_wuji` 기준으로 진행 중임.
- 기존 `chop_ws/chop_rl`, Isaac Sim 4.5, IsaacLab 2.2.1 실험은 폐기함.
- 현재는 잘 켜지는 Isaac Sim 5.1, IsaacLab 2.3 계열, `env_isaaclab` 환경 사용함.
- Neuromeka public/main branch 스타일을 우선 사용함.
- 목표는 full training 성능이 아니라 env 구조 이해와 arm end-effector tracking 구성임.
- 현재 reach baseline task는 `Indy-Wuji-Reach`임.
- 현재 cube grasp task는 `Indy-Wuji-Cube-Grasp`임 (obs 57, 2026-07-16부터 동결 — 체크포인트 play 호환 유지).
- 2026-07-16 신규: `Indy-Wuji-Box-Transport` — env별 랜덤 직육면체(단면 3~6cm x 비율 1.5~3) 파지/운반.
  obs 64 (= 57 + box_size 3 + box_quat 4), experiment `indy_wuji_box_transport` (로그 분리).
  cfg는 grasp/box_mdp_cfg.py + box_transport_env_cfg.py + indy_wuji_box/ (큐브 cfg의 사본 — 서로 반영 안 됨).
  현재 transport 3종(cube_transport/transport_success/success 종료)은 활성 상태이며, 랜덤 box의
  안정 lift까지 확인함. 다음 관문은 분리형/통합형 reward A/B와 orientation-aware goal임.
  replicate_physics=False라 startup이 느림 (정상). env별 치수 검증은 scripts/debug/box_dims_probe.py.
- `Indy-Wuji-Cube-Grasp-Easy`는 이전 실험 이름이며 현재 active registration에는 없음.
- 현재 active USD는 `indy7_wuji_right_simplified.usd`임.
- 초기 후보였던 `indy7_allegro_hand_right_simplified.usd`는 참고/비교용임.
- 현재 tracking body는 `link6`임.
- 현재는 Wuji hand frame 문제를 분리하고 Indy arm flange 기준 reach baseline을 확인하는 단계임.
- cube grasp는 최종 목표가 아니라 functional grasp/chopstick grasp로 가기 위한 중간 proxy task임.
- `tcp`는 현재 USD articulation rigid body로 쓰기 부적합하다고 판단함.
- virtual EE offset 방식은 실험 후 보류함.
- reach baseline action은 arm 6축만 사용함.
- 현재 active cube grasp policy action은 thumb/index/middle 12축만 사용함.
- cube grasp arm 6축은 `FixedJointPositionAction` 0D term으로 default joint target을 매 step 유지함.
- 현재 cube grasp ActionManager total action dim은 12임. `arm_action=12`, `arm_hold_action=0`임.
- hand joint 20축은 articulation에는 남아 있음.
- cube grasp에서는 `finger1~3`의 12축만 policy action에 넣음.
- `finger4~5`는 현재 policy action에 넣지 않고 접어둔 초기 자세/actuator로 처리함.
- 이후 젓가락 task에서 필요하면 hand action을 20축으로 확장함.

## Current Implementation

- `Indy-Wuji-Reach`는 Neuromeka `Indy-Reach` 스타일로 구현됨.
- 공통 reach 구조는 `isaac_neuromeka/tasks/manipulation/reach/reach_env_cfg.py` 기반임.
- 공통 MDP 설정은 `isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py` 기반임.
- task override는 `isaac_neuromeka/tasks/manipulation/reach/indy_wuji/env_cfg.py`에 있음.
- gym registration은 `isaac_neuromeka/tasks/manipulation/reach/indy_wuji/__init__.py`에 있음.
- RSL-RL config는 `isaac_neuromeka/tasks/manipulation/reach/indy_wuji/learning/rsl_rl_cfg.py`에 있음.
- robot asset config는 `isaac_neuromeka/assets/indy.py`의 `INDY7_WUJI_RIGHT_CFG`임.
- arm action dim은 6임.
- policy observation dim은 현재 15임.
- observation은 arm 6축 joint position, target position command, previous action만 사용함.
- `joint_vel` observation은 제거함.
- observation history와 observation noise는 현재 policy group에서 제거함.
- hand joint는 observation에서 제외됨.
- `sim.render_interval = decimation` 적용됨.
- command는 공통 `UniformPoseCommandCfg`를 사용함.
- command manager의 `ee_pose` 자체는 7D pose command임.
- policy observation에는 `ee_pose` 중 position xyz 3D만 넣음.
- active reward는 position tracking과 action rate penalty만 사용함.
- orientation tracking, end-effector speed, joint velocity reward term은 현재 제거함.
- command/reward body는 `link6`임.
- `Indy-Wuji-Reach` entry point는 `CustomManagerBasedRLEnv`임.
- TensorBoard에는 weighted reward `Episode_Reward/*`와 raw unweighted reward `Episode_Reward_Raw/*`를 같이 기록함.

## Cube Grasp Current State

- 2026-07-14 기준 cube grasp는 `Indy-Wuji-Cube-Grasp` 하나만 사용함.
- 별도 curriculum/hard task를 나누지 않음. run/checkpoint 선택이 꼬여서 디버깅 비용이 커졌기 때문임.
- 예전 curriculum alias/class/register는 제거됨. 새 학습/play/smoke test에서는 `Indy-Wuji-Cube-Grasp`만 사용함.
- 현재 main task 자체가 가까운 nominal grasp 배치를 사용함.
- 이전 가까운 배치는 큐브 `x/y`만 손 파지 중심 근처였고 `z`는 바닥 `0.03m`라 손 높이와 맞지 않았음.
- 현재 `BASE_Z=0.40` 받침면을 추가함.
- cube 중심 높이는 `BASE_Z + 0.03 = 0.43m`임.
- `CubeGraspSceneCfg`에 `{ENV_REGEX_NS}/Support` kinematic cuboid를 추가함.
- 현재 cube 위치는 probe로 검증된 `(0.692, -0.369, 0.430)`임.
- reset probe에서 `palm_facing=0.987`, zero action 30 step 뒤 `0.997`임.
- 이전 cube `(0.704, -0.279, 0.430)`는 cage 중심에서 y로 약 `9cm` 벗어나 zero action만으로 cube가 밀려났음.
- `cube_lift`와 `Metrics/cube*/cube_clearance`는 월드 바닥이 아니라 `surface_z=BASE_Z` 기준으로 계산함.
- `hand_floor` penalty도 월드 바닥이 아니라 `surface_z=BASE_Z` 기준으로 계산함.
- cube reset은 고정임. `x/y/z = 0`.
- action shape는 `18`, policy observation shape는 `54`임.
- active reward terms는 8개임.
- 현재 override reward는 `finger_cage_reach=3`, `finger_cage_hold=5`, `cube_lift=50`, `cube_support=2`, `palm_facing=0`, `arm_manipulability=0`, `hand_floor=0.2`, `action_rate=-0.0003`임.
- `cube_support`는 큐브 최하 모서리가 받침면 아래로 내려가면 음수 보상을 줌. hold를 받으려고 큐브를 받침 안으로 누르는 실패 모드를 막기 위해 추가함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- 메인 task 이름으로 close-start probe 확인함.
- probe 결과 reset `cage_center_to_cube=(0.000461, -0.000059, -0.016252)`, `palm_facing=0.986780`, `cage_hold=0.210871`임.
- zero action 30 step 뒤 `cage_center_to_cube=(0.002833, 0.002668, -0.035702)`, `palm_facing=0.996594`임.
- 0D arm hold 추가 후 zero action 30 step에서 arm collapse/cube ejection이 사라짐.
- 0D arm hold 추가 후 close action `1.0` 60 step probe에서 `cage_hold=0.427465`, `cage_inside_frac=0.666667`까지 증가함.
- 같은 probe에서 cube는 `(0.694, -0.368, 0.436)` 근처로 유지됨. 큐브가 날아가지 않음.
- `--num_envs 128 --max_iterations 20` grasp+lift 짧은 학습 통과함.
- 짧은 학습에서 `finger_cage_hold`는 켜졌지만 `cube_lift`는 거의 0이고, `cube_support`가 큐브를 아래로 누르는 실패 모드를 드러냄.
- `/tmp/cube_lift_probe.py` scripted feasibility probe 결과, 손만 닫으면 `cage_hold`는 약 `0.40`까지 오르지만 `cube_clearance`는 거의 0이고 `cube_lift_reward_raw=0`임.
- 같은 probe에서 `joint0~joint5` 단일축 ±1 lift 후보를 모두 넣어도 양의 clearance가 나오지 않음.
- 강한 손/가벼운 큐브 probe(`finger_effort=3`, `stiffness=40`, `cube_mass=0.03`, `friction=2`)에서도 lift는 0임.
- repo 내부 contact/lift 확인 스크립트는 `nrmk_isaaclab_wuji/scripts/debug/check_cube_contact_lift.py`임.
- 이 스크립트는 policy 없이 `reset -> zero settle -> finger close -> optional arm lift`를 실행하고 thumb/middle contact force와 cube clearance를 출력함.
- `--finger-action`으로 thumb/index/middle close 값을 직접 줄 수 있음. 예: `--finger-action 1 0 1`.
- `--sweep-fingers`로 cube를 고정한 채 thumb/index/middle close 값 조합을 먼저 훑음.
- `--contact-mode`는 `thumb_middle`, `thumb_index`, `thumb_any`, `tripod` 중 선택함.
- 2026-07-14 확인에서 close-only는 `thumb+middle GOOD_CONTACT`가 늦게 켜졌지만 `max_clearance=0.0003m`라 lift 실패임.
- 긴 학습 전에는 이 스크립트에서 `GOOD_CONTACT=True`와 `max_clearance > 0.005m`가 먼저 나와야 함.
- 따라서 긴 학습 전에 scripted sequence로 실제 lift가 가능한 arm/hand 조합 또는 초기 자세를 먼저 찾아야 함.
- 더 아래의 오래된 `hard`, `Easy`, `action_rate=-0.005` 기록은 실험 히스토리로 읽고 현재 지침으로 쓰지 않음.
- cube grasp용 새 task skeleton은 `Indy-Wuji-Cube-Grasp`임.
- cube grasp package 경로는 `isaac_neuromeka/tasks/manipulation/grasp/`임.
- cube grasp 공통 cfg는 `isaac_neuromeka/tasks/manipulation/grasp/cube_grasp_env_cfg.py`임.
- Indy/Wuji 전용 override는 `isaac_neuromeka/tasks/manipulation/grasp/indy_wuji/env_cfg.py`임.
- `CubeGraspSceneCfg`는 기존 `ReachSceneCfg`를 상속하고 cube만 추가함.
- cube는 `RigidObjectCfg`로 `{ENV_REGEX_NS}/Cube`에 생성함.
- 현재 cube size는 `0.06 m`임.
- 현재 cube mass는 `0.10 kg`임.
- 현재 cube initial position은 `(0.692, -0.369, 0.430)`임.
- `Indy-Wuji-Cube-Grasp-Easy`는 이전 실험 이름이며 현재 active registration에는 없음.
- 4096 env long run에서 PhysX patch buffer overflow가 발생해 `gpu_max_rigid_patch_count`를 `2**20`으로 올림.
- 2026-07-10 resume run에서 요구 patch count가 약 `263k`까지 올라가 `2**18`로는 부족했음.
- cube grasp RSL-RL experiment name은 `indy_wuji_cube_grasp`임.
- 현재 cube grasp policy action은 arm 6축 + `finger[1-3]_joint[1-4]` 12축, 총 18축임.
- 현재 cube grasp action dim은 18임.
- 현재 cube grasp action scale은 arm `0.25`, finger `0.5`임.
- **action은 `target = default_joint_pos + scale * raw_action`인 절대 위치 명령임.** 증분이 아니므로 과거 action이 누적되지 않음.
- 현재 cube grasp policy observation dim은 54임.
- 현재 cube grasp controlled joints는 `joint[0-5]`, `finger[1-3]_joint[1-4]`임.
- 현재 cube grasp observation은 controlled joint position 18D, `palm_link` 기준 cube relative position 3D, five-fingertip 기준 cube relative position 15D, previous action 18D임.
- `cube_to_goal` observation은 현재 grasp+lift baseline에서 제거됨.
- 현재 cube grasp command manager는 active command 없이 시작함.
- cube reset은 고정임. `x/y/z = 0`.
- 같은 experiment 안에 과거 smoke/hard/easy run이 섞여 있으므로 `--load_run "$(ls -td ... | head -n 1)"` 자동 선택은 위험함.
- play/resume은 가능한 한 확인한 run 폴더명을 직접 지정함.
- cube grasp task에서만 Indy arm initial joint를 살짝 높은 pre-grasp 자세로 override함.
- cube grasp initial arm override는 `joint1=-0.75`, `joint2=-1.85`, `joint3=-1.61`, `joint4=-1.62`, `joint5=2.35`임.
- 위 arm override는 action offset도 바꾸므로 이전 checkpoint resume은 가능하지만 fresh run이 더 깔끔함.
- Wuji hand actuator는 현재 전체 finger 공통으로 `stiffness=20.0`, `damping=0.5`, `friction=0.02`, `effort_limit=0.6` (2026-07-12에 stiffness를 `8.0`에서 올림. damping은 한때 `2.5`였으나 **최대 폐합 속도 = effort_limit/damping = 0.24 rad/s로 손가락이 5배 느려져** `0.5`로 되돌림)임.
- 이 값은 ring/little finger 떨림을 줄이기 위한 안정화 설정임.
- active `INDY7_WUJI_RIGHT_CFG` contact response는 `max_depenetration_velocity=5.0`, `max_contact_impulse=100.0`으로 완화함.
- 이전 값 `max_depenetration_velocity=1000.0`, `max_contact_impulse=1e32`는 palm/hand-cube 접촉에서 관통 보정을 너무 강하게 만들어 arm이 튀는 원인 후보였음.
- 이 변경은 action/observation shape를 바꾸지 않으므로 기존 checkpoint load/resume은 가능함. 다만 physics가 바뀌므로 성능 평가는 재학습 또는 resume adaptation 후 판단함.
- 현재 cube grasp reward는 pre-grasp/functional-hold baseline으로 구성함.
- cube grasp의 주 목표는 DexPoint 재현이 아니라 functional grasp 논문 흐름을 Wuji/cube task에 맞게 구현하고 검증하는 것임.
- DexPoint는 grasp reward 구현 목표가 아니라 reach/contact/lift gate 설계 참고 자료임.
### Active reward (2026-07-12 전면 재설계)

- active reward는 6개임: `finger_cage_reach` (차분, `0.3`), `palm_facing` (차분, `1.0`), `finger_cage_hold` (절대, `1.0`), `cube_lift` (절대, `3.0`), `arm_manipulability` (절대 페널티, `1.0`), `action_rate` (`-0.0003`).

### reward 형태 선택 원칙 (2026-07-13 확립, 매우 중요)

- **절대 양수 + 유지가 쉬움 -> 반드시 farming당함.** `palm_facing`을 절대형으로 넣었더니 전체 보상의 `98.6%`를 먹고 정책이 큐브 31cm 밖에서 손바닥만 겨누며 정지함 (팔은 특이점까지 접힘).
- **절대 양수 + 유지가 어려움 -> 안전.** `finger_cage_hold`, `cube_lift`. 유지가 곧 과제의 목표임.
- **절대 페널티 (`<=0`) -> 안전.** 최대가 `0`이라 쌓을 것이 없음. `arm_manipulability`, `action_rate`.
- **차분 -> 안전.** 가만히 있으면 `0`이라 farming 불가능함.
- **논문의 거의 모든 항이 차분형임** (`r_hp`, `r_hr`, `r_hj`, `r_reach`, `r_orient`). **절대형은 `r_hold` 하나뿐임.** 그래서 논문은 weight `1.0`을 줘도 안전함.
- **새 reward를 넣을 때 "이건 유도인가 유지인가", "가장 싼 만족 방법이 뭔가"를 먼저 물을 것.**

### 논문의 가중치 (9쪽) 와 우리 방침

- `r = r_grasp + r_reach + 25*r_hold + 500*r_orient + r_MP + 5000*r_T`.
- approach(1) -> hold(25) -> orient(500) -> grasp(5000). **각 단계마다 약 20배씩.**
- 논문: "The exact values do not affect learning significantly, **as long as the overall proportions reflect the logical sequence**."
- **절대값이 아니라 비율만 중요함. scale은 우리 보상값의 실제 크기에 맞춰 역산할 것.** 논문 숫자를 그대로 베끼지 말 것.
- 차분형은 telescoping되어 1회만 지급되고 절대형은 20 step 누적되므로 **규모가 근본적으로 다름.** 에피소드당 기여량으로 환산해서 비교할 것.

### `cube_lift`가 한 번도 발생한 적이 없음 (2026-07-13)

- 전 학습 기간 동안 `Episode_Reward_Raw/cube_lift` = **정확히 `0`**. 큐브가 단 한 번도 바닥에서 떨어진 적이 없음.
- **`0`인 보상은 가중치를 아무리 올려도 `0`임.** 가중치 조정 전에 **그 보상이 실제로 발생한 적이 있는지** 먼저 확인할 것.
- 원인: 최하 모서리 기준으로 바꿔 기울이기 편법을 막았더니 **사실상 희소해짐.**
- **희소 보상은 curriculum 없이 학습 불가능함.** 논문의 `r_T`(sparse)가 curriculum과 세트인 이유임.
- 논문 데이터: curriculum 없으면 성공률 약 `50%`에 편차 폭발, 있으면 **`97%`** (wall-clock 동일).
- **보상 단계화도 curriculum임.** 환경이 아니라 보상을 단계화해도 됨 (Phase 1: 접근/파지 -> Phase 2: lift 가중치 상향 + resume).
- 성립 조건: Phase 1 수렴 자세에서 lift가 **탐색으로 도달 가능**해야 함. 넘어가기 전에 반드시 검증할 것.
- 가중치는 논문의 `r_T >> r_orient >> r_hold >> r_reach` 순서를 따름.
- 세 항 모두 **같은 12개 가상점** 위에서 동작함. `CAGE_BODIES` 상수를 공유함.
- action shape `18`, observation shape `57` 불변임.

### 가상점 12개 (`CAGE_BODIES`)

- `finger1_tip_link` (엄지끝, 모든 선분의 기준점).
- `finger2_tip_link`, `finger2_link3` (검지 끝, 검지 중간마디).
- `finger3_tip_link`, `finger3_link3` (중지 끝, 중지 중간마디).
- 엄지끝에서 대향 body 4개로 선분을 긋고 각 등간격 3점 = **12점**.
- `SceneEntityCfg`에 `preserve_order=True` 필수임. 기본값 `False`면 body_ids가 정렬되어 엄지가 기준점 자리에서 밀려남.
- `managers.py`의 `_cage_body_names`와 **반드시 동일해야 함.** 한쪽만 바꾸면 metric이 reward와 다른 점을 측정함.
- 논문은 엄지↔중지만 써서 6점이지만, 논문에는 `r_grasp`(`r_hr` + `r_hj`)가 손 회전과 손가락 관절각을 붙잡고 있음. 큐브는 목표 파지 자세가 없어 `r_grasp`를 못 쓰는데, 6점만 쓰면 **검지가 완전히 자유가 되어 "손바닥이 하늘을 보고 검지·중지가 교차한" 자세로도 만점**이 나옴 (2026-07-11 실측).
- 엄지+검지+중지는 젓가락 그립과 동일하므로 임시방편이 아님.

### `finger_cage_hold` (논문 Eq.15, weight `1.0`)

- 각 가상점을 반지름 `sphere_radius`의 구로 보고, 그 구가 큐브를 **파고든 깊이**를 보상함.
- `r = clamp((sphere_radius - sdf) / (sphere_radius + depth_max), 0, 1)`의 12점 평균임.
- `sdf`는 큐브 **표면**까지의 signed distance. 내부이면 음수임.
- **손을 오므리면 점들이 큐브 안으로 들어가므로 "오므리기"가 직접 보상됨.** 접촉센서 불필요함.
- 큐브 SDF는 해석식(`_box_signed_distance`)임. CAD나 사전계산 SDF 불필요함.
- 파라미터는 실측 튜닝됨: `sphere_radius=0.005`, `depth_max=0.02`.
- `sphere_radius`가 크면 손가락을 벌린 채 큐브가 사이에 있기만 해도 점수가 나와 대비가 죽음.

### `finger_cage_reach` (논문 Eq.14, weight `0.3`)

- **같은 12개 가상점**의 큐브 표면까지 SDF 평균의 **차분**임. `mdp.ObjectCageProgressReward`.
- "파지 간극을 큐브 위로" 끌어당겨 큐브가 손가락 **사이**에 놓이게 만듦.
- `mode="previous"` + `clamp(min=-1)` + `reset()`에서 기준선 seeding, 셋을 다 해야 함.
- `clamp(min=0)`이면 후퇴가 공짜이고, 기준선을 첫 `__call__`에서 잡으면 첫 액션이 기준선을 공짜로 부풀림 (swing-out).
- 셋을 다 하면 총합이 `d(reset) - d(final)`로 telescoping되어 페이스 조작과 swing-out이 모두 무의미해짐.
- `distance_max=0.5`. 실제 step당 최대 개선량보다 충분히 커야 함. 포화되면 "천천히 접근하기"를 보상하게 됨.

### `cube_lift` (논문 `r_lift`, weight `3.0`)

- `mdp.object_lift_in_cage`. `r = cage_gate * clamp(height / 0.08, 0, 1)`.
- **어떤 자세를 "진짜 파지"로 인정할지 결정하는 항임.**
- **들지 못하는 자세는 파지가 아니므로, 자세를 지정할 필요 없이 하중을 견디는지만 물으면 됨. 물리가 자세를 결정함.**
- `cage_gate`가 없으면 "파지 없이 큐브를 튕겨 올리는" 편법이 가능함.
- 조밀형(dense)이라 `2 mm` 상승에도 gradient가 있음. 희소형이면 영원히 `0`이라 학습 불가함.
- 논문도 fake success 방지용으로 `r_lift`를 넣음. 2026-07-11 run이 정확히 그 상태였음 (`opposition +0.92`, `inside_frac 0.84`인데 `cube_lift 0.002 m`).

### 절대 다시 넣지 말 것: "손끝 -> 큐브 중심" 거리 reward

- 큐브 중심은 표면에서 `0.03 m` 안쪽이라 **손끝이 도달 불가능한 목표**임. gradient가 항상 큐브 속을 향함.
- `body_weights=(3,1,1)`이면 엄지가 가중평균의 `60%`라 **"엄지 하나만 박고 나머지 방치"가 최적해**가 됨.
- 거리 reward는 접촉도 처벌함. 만지면 큐브가 밀려나 거리가 늘어남.
- cage reward는 반대로 물체를 파고들어야 점수가 남. **접촉이 이득**임. 이 부호 차이가 파지 학습의 핵심임.
- 이 이유로 `finger_cube_reach`, `finger_cube_closeness`, `functional_hold`, `arm_cube_reach`와 그 함수들을 전부 삭제함.

### 손바닥 방향 reward (`palm_facing`)는 검토 후 철회함

- 논문이 `r_hr`(목표 손 회전)을 주는 이유는 **기능(functional grasp)** 때문임. 드릴을 "트리거를 당길 수 있게" 쥐어야 함.
- **큐브에는 기능 요구가 없으므로 목표 회전이 필요 없는 것이 맞음.** 사람이 정하면 자의적이고, 젓가락에서 `r_hr`로 교체되며 버려짐.
- **목표 회전은 "잡기"가 아니라 "쓰기"에서 나옴.** 젓가락에는 반드시 필요함.
- 참고: `palm_link`의 손바닥 법선은 실측 결과 로컬 **`+x`**축임 (손가락이 오므라들 때 손끝이 이동하는 방향).

### 역할 분담 (중요)

- **cage는 자세를 유도하지 않음.** "물체가 손가락 사이에 있는가"만 봄. 6점 cage가 손바닥 하늘로 수렴한 것이 반증임.
- `finger_cage_reach` -> 파지 간극을 물체 위로.
- `finger_cage_hold` -> 오므려라.
- `cube_lift` -> 들어라. **자세를 결정하는 것은 이것, 정확히는 물리임.**

### TensorBoard 판정 지표

- `Metrics/cube/*`는 에피소드 **평균**이라 성능 지표가 아님. 앞 4 step이 평균의 `77%`를 지배함.
- 평가는 `Metrics/cube_final/*` (마지막 step, 정착 자세)로 할 것. `cube_min/*`, `cube_max/*`도 있음.
- 핵심: `cube_final/cube_lift` (**이것이 전부임**), `thumb_index_opposition`, `thumb_middle_opposition`, `cage_inside_frac`, `cage_span`, `*_surface` (음수=관통).
- **`opposition`만으로 자세를 판정하지 말 것.** 손바닥 방향을 보지 못하므로 `+0.5`를 넘어도 손바닥이 하늘일 수 있음.
- **검지 opposition을 따로 볼 것.** 중지만 보면 검지 교차를 놓침.
- TensorBoard cube distance metric은 `Metrics/cube/*`로 기록함.
- `Metrics/cube/palm_distance`는 `palm_link`와 cube root 사이 실제 거리임.
- `Metrics/cube/thumb_distance`, `index_distance`, `middle_distance`, `ring_distance`, `little_distance`는 각 fingertip과 cube root 사이 실제 거리임.
- `Metrics/cube/finger_mean_distance`는 five-fingertip 거리 평균임.
- `Metrics/cube/non_thumb_mean_distance`는 index/middle/ring/little 거리 평균임.
- `Metrics/cube/finger_weighted_mean_distance`는 thumb을 크게 본 five-fingertip 거리 평균임.
- cube distance metric은 reward가 아니라 error 확인용 logging임.
- contact 기반 grasp reward는 아직 구현하지 않음.
- lift reward와 lifted-gated cube goal reward는 구현됐지만 현재 active reward에서는 비활성화됨.
- 6D arm-only `Indy-Wuji-Cube-Grasp` headless smoke test는 통과함.
- 최신 hand-only 12D action + 42D observation smoke test는 통과함.

## Wuji Finger Naming

- Wuji hand는 5개 finger group으로 정리함.
- `finger1`은 `thumb`으로 취급함.
- `finger2`는 `index`로 취급함.
- `finger3`은 `middle`로 취급함.
- `finger4`는 `ring`으로 취급함.
- `finger5`는 `little`로 취급함.
- 코드에서는 당분간 USD/URDF 이름인 `finger[1-5]_joint[1-4]`, `finger[1-5]_link[1-4]`, `finger[1-5]_tip_link`를 유지함.
- reward/contact 문서와 설계에서는 thumb/index/middle/ring/little alias를 사용함.
- 현재 cube grasp active finger는 `finger1`, `finger2`, `finger3`만 사용함.

## Frame Notes

- `palm_link`는 물리 hand body임.
- `palm_link`를 그대로 task EE orientation으로 쓰면 frame mismatch가 생김.
- raw `palm_link` tracking에서는 orientation error가 약 2 rad 근처로 크게 남았음.
- `link6` tracking은 arm flange 기준 reach baseline으로 사용함.
- command orientation을 임시로 `roll=-pi/2`, `pitch=-pi/2`, `yaw` 자유로 두면 orientation error가 약 0.18까지 떨어졌음.
- 위 결과로 Wuji hand frame과 Indy reach EE command frame 사이에 고정 회전 offset이 있다고 판단함.
- Allegro 쪽도 손 base를 직접 tracking하기보다 `tcp` frame과 hand base를 fixed joint/offset으로 분리하는 구조임.
- 다만 약 2000 iteration 학습 후 orientation error가 2점대에서 약 0.8까지 내려감.
- `palm_link`와 `link6` 모두 충분히 학습하면 orientation error가 유사하게 내려가는 것으로 봄.
- 따라서 현재는 URDF/offset 문제로 단정하지 않음.
- 학습 시간과 reward 구조 영향이 컸던 것으로 보고 long-run 결과를 먼저 확인함.

## Reward Notes

- reward term 설정은 `isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py`에 있음.
- reward 계산 함수는 `isaac_neuromeka/mdp/rewards.py`에 있음.
- position reward는 target과 body position 거리 기반 bounded reward임.
- orientation reward는 position-only baseline에서 제거함.
- end-effector speed penalty와 joint velocity penalty도 현재 제거함.
- action rate penalty는 남김.
- 현재 `Indy-Wuji-Reach`에서는 position reward가 `link6` 기준으로 적용됨.
- TensorBoard에서 weighted reward는 `Episode_Reward/*`, raw reward는 `Episode_Reward_Raw/*`로 확인함.

### Cube Grasp Reward (2026-07-11 전면 재설계)

- 이 섹션은 2026-07-11 reward 재설계 히스토리임.
- 2026-07-14 현재 active override는 `finger_cage_hold=1`, `hand_floor=0.5`, `action_rate=-0.0003` 중심임.
- 현재 `finger_cage_reach`, `palm_facing`, `cube_lift`, `arm_manipulability` weight는 `0`임.
- 2026-07-11 당시 active reward는 `finger_cage_reach` (`0.3`), `finger_cage_hold` (`1.0`), `action_rate` (`-0.0003`) 3개뿐이었음.
- 둘 다 Dexterous Pre-grasp Manipulation 논문 방식이며, **같은 6개 가상점** 위에서 동작함.
- 가상점은 엄지끝(`finger1_tip_link`)과 중지(`finger3_tip_link`, `finger3_link3`) 사이에 비율 `[0.25, 0.50, 0.75]`로 배치함. 선분 A는 핀치 파지, 선분 B는 파워 파지 위치임.
- `finger_cage_hold` (Eq.15): 가상점이 큐브 **내부**로 파고든 깊이를 보상함. **오므리기가 직접 보상됨.** 접촉센서 불필요함. 큐브 SDF는 해석식임.
- `finger_cage_reach` (Eq.14): 같은 6점의 큐브 **표면**까지 SDF의 차분. 파지 간극을 큐브 위로 끌어옴.
- 가중치는 `hold(1.0) >> reach(0.3)`. 논문의 `r_T >> r_orient >> r_hold >> r_reach` 순서임.

### 절대 다시 넣지 말 것: "손끝 -> 큐브 중심" 거리 reward

- 큐브 중심은 표면에서 `0.03 m` 안쪽이라 **손끝이 도달 불가능한 목표**임. gradient가 항상 큐브 속을 향함.
- `body_weights=(3,1,1)`이면 엄지가 가중평균의 `60%`라 **"엄지 하나만 박고 나머지 방치"가 최적해**가 됨.
- 그 자세에서는 엄지-중지 선분이 큐브를 관통하지 않아 오므리면 가상점이 큐브 밖으로 빠져나감. 즉 **파지가 손해**가 됨.
- 거리 reward는 접촉도 처벌함. 만지면 큐브가 밀려나 거리가 늘어남.
- cage reward는 반대로 물체를 파고들어야 점수가 남. **접촉이 이득**임. 이 부호 차이가 파지 학습의 핵심임.
- 이 이유로 `bodies_to_object_position_tracking_bounded`, `object_in_functional_grasp_region`, `BodiesToObjectProgressReward`를 삭제함.

### Progress reward 규칙

- `mode="previous"` + `clamp(min=-1)` + `reset()`에서 기준선 seeding, 셋을 모두 해야 함.
- `clamp(min=0)`이면 후퇴가 공짜이고, 기준선을 첫 `__call__`에서 잡으면 첫 액션이 기준선을 공짜로 부풀림 (swing-out).
- `distance_max`는 실제 step당 최대 개선량보다 충분히 커야 함. 포화되면 "천천히 접근하기"를 보상하게 됨.

### TensorBoard 지표

- `Metrics/cube/*`는 에피소드 **평균**이라 성능 지표가 아님. 앞 4 step이 평균의 `77%`를 지배함.
- 평가는 `Metrics/cube_final/*` (마지막 step, 정착 자세)로 할 것. `cube_min/*`, `cube_max/*`도 있음.
- 핵심 지표는 `cube_final/thumb_middle_opposition` (`+1`=큐브 양쪽, `-1`=같은 쪽), `cage_inside_frac`, `cage_span`, `*_surface` (음수=관통), `cube_lift`임.

## Reward Study Notes

- 다음 단계 cube grasp baseline은 먼저 oracle state 기반으로 구성하는 쪽이 적절함.
- 여기서 oracle state는 simulator가 직접 알고 있는 cube pose, fingertip pose, contact, lift height, velocity 같은 정답 상태를 뜻함.
- oracle 정보를 policy observation에 넣으면 oracle observation policy이고, reward/success 계산에만 쓰면 oracle reward/success condition임.
- 지금은 point cloud, force/tactile sensor부터 넣지 않음.
- 처음에는 cube pose/fingertip pose/contact/lift 같은 sim state를 reward 계산에 쓰고, point cloud/force는 real transfer나 젓가락 미세 접촉 단계에서 검토함.
- grasp 판정은 `palm_link`-cube 거리만으로 정의하면 약함.
- grasp success는 contact + lift + 안정성 기준을 같이 봐야 함.
- 후보 구조는 thumb 또는 palm 쪽 contact, non-thumb finger contact 1~2개 이상, cube lift threshold, cube velocity 안정성임.
- 현재 cube grasp의 주 논문 목표는 functional grasp/pre-grasp 계열 아이디어를 Wuji hand에 맞게 검증하는 것임.
- DexPoint는 보조 참고 자료임. DexPoint 전체 구현이나 재현이 목표가 아님.
- DexPoint에서 가져갈 것은 fingertip-object reach, contact group, contact-gated lift, action/velocity penalty 같은 안정적인 grasp-shaping 패턴임.
- DexPoint 논문식 contact는 thumb contact + other finger 2개 이상이지만, 공개 코드식 완화 조건은 finger/palm contact group count 2개 이상임.
- `indy_wuji_right` 초기 구현에서는 완화 조건으로 시작하고, 학습이 되면 thumb + non-thumb finger 조건으로 강화하는 방향이 좋음.
- DexPoint lift reward는 contact가 성립했을 때만 켜는 gate 구조가 중요함.
- TriFinger transfer 논문은 grasp 자체보다 object 6-DoF pose goal tracking reward 참고용임.
- TriFinger 핵심은 cube/object의 8개 keypoint current-target distance로 position과 orientation을 함께 보는 object-goal reward임.
- TriFinger reach reward는 현재 fingertip-object 거리 자체가 아니라 `curr_dist - prev_dist`에 음수 weight를 곱하는 접근 progress reward임.
- TriFinger reach reward는 초반 exploration용이고, 후반에는 curriculum으로 꺼서 regrasp/finger gaiting을 방해하지 않게 함.
- TriFinger fingertip velocity penalty는 손가락이 너무 빠르게 튀는 움직임을 줄이는 용도임.
- SimToolReal은 tool-use/chopstick 단계 참고용임.
- SimToolReal reward는 `r = r_smooth + r_grasp + I_grasped * r_goal` 구조임.
- SimToolReal의 `r_grasp = r_approach + (1 - I_grasped) * r_lift`이고, lift 이후에는 object-centric goal pose progress reward가 주도함.
- SimToolReal은 tool의 grasp bounding box, object keypoint, goal pose trajectory를 써서 잡은 물체를 목표 pose sequence로 움직이는 관점이 중요함.
- Dexterous Pre-grasp Manipulation 논문은 functional grasp 전 object reposition/reorient/regrasp reward의 주 기준임.
- Pre-grasp 논문 흐름은 cube proxy 단계부터 염두에 두고, 이후 hand 20축 action과 functional grasp/젓가락 파지로 확장함.
- Pre-grasp 논문의 핵심 reward는 `r_man = r_reach + r_hold + r_orient`임.
- Pre-grasp의 `r_hold`는 단순 fingertip 거리나 contact count가 아니라 object가 thumb-finger 사이 공간에 들어왔는지를 보상하는 cage-like reward라서 Wuji hand grasp에 유용함.
- Pre-grasp는 explicit target grasp와 constraint-based target grasp를 비교함.
- explicit target grasp는 object 기준 EE pose와 hand joint target을 직접 주는 방식이고, 성능은 높지만 물체별 target grasp 정의 부담이 큼.
- constraint-based target grasp는 index fingertip target position과 EE orientation 같은 기능 조건만 주고, fake success를 막기 위해 lift reward/condition을 추가함.
- Pre-grasp는 curriculum이 중요함. 먼저 가까운 nominal pose에서 grasp를 배우고, 이후 다양한 object pose에서 pre-grasp manipulation을 학습함.
- 현재 해석은 Functional/Pre-grasp 논문이 "functional grasp가 가능하도록 object를 손 안/손 앞에서 정렬하고 유지하는 reward"의 주 기준이고, DexPoint는 "contact/lift gate를 안정적으로 넣는 보조 참고", TriFinger는 "잡은 뒤 object pose를 맞추는 reward 참고", SimToolReal은 "object pose tracking을 tool-use trajectory로 확장한 참고"임.
- cube grasp 구현 권장 순서는 functional grasp hold/cage baseline, contact condition, contact-gated lift, 이후 object goal/keypoint tracking 순서임.

## Asset Notes

## 2026-07-14 Play Diagnostics Notes

- `scripts/rsl_rl/play.py`는 이제 `--latest_run` 또는 `--load_run latest`로 최신 cube grasp run을 직접 열 수 있음.
- `--print_diagnostics`는 기존 action detail에 joint torque/velocity, reward raw, cube clearance/cage/opposition을 같이 출력함.
- `--print_contact`는 thumb/index/middle/palm contact force도 출력함.
- 이 진단은 출력량이 많아 GUI가 느려짐. 평소에는 `--print_action_interval 10~20`을 쓰고 contact는 필요할 때만 켬.
- 2026-07-14 play 로그 기준 현재 정책은 arm torque 부족이 아님.
  - 안정 구간에서 `joint1` torque는 약 `3~4%`, err는 약 `0.14rad`.
  - finger 관절은 다수 `tq%=100`으로 effort limit에 붙음.
  - `finger_cage_hold` raw는 약 `0.46~0.48`이지만 `cube_lift`/`clearance`는 0 근처.
  - 결론: cage/hold local optimum이며 실제 lift 파지는 아님. 다음 레버는 finger action range/scale, finger joint2/negative target 처리, contact/lift/r_T 계층임.

## Asset Notes

- Wuji collision 문제는 `indy7_wuji_right_simplified.usd` 기준으로 post-process 처리함.
- 26개 Wuji hand collision STL을 USD Mesh collider로 삽입함.
- 직접 삽입한 collision mesh prim에 `PhysicsCollisionAPI` 등을 적용함.
- active hand collision mesh 수는 26개로 검증함.
- arm collision은 simplified collision 사용함.
- hand collision은 Wuji `*_collision.STL` convex hull 기반임.
- `indy7_wuji_right_all_simplified.usd`는 fallback/debug용이었으나 현재 git status에서는 삭제 상태로 보임.
- `indy7_wuji_right.usd`는 full mesh/reference baseline 성격임.

## Study Order

- Direct Cartpole 봄.
- Isaac-Ant-v0 봄.
- Neuromeka Indy-Reach 우선 봄.
- Neuromeka Indy-Wuji-Reach 현재 구현 봄.
- IsaacLab Franka Reach는 공식 ManagerBased 구조 비교용으로 봄.
- KUKA Allegro/Dexsuite는 hand 확장할 때 봄.
- 실행 흐름과 핵심 코드 연결은 root `flow_study.md`에 정리함.
- cube grasp reward 설계용으로 functional grasp/pre-grasp 논문을 주 기준으로 보고, DexPoint, TriFinger transfer, SimToolReal reward 구조는 보조 참고로 정리함.

## Agent Pitfalls (작업 중 막힌 지점, 2026-07-15 정리)

- 사용자 실험 기록(worklog/ACTIVITY/agent.md)에는 에이전트 도구 함정을 섞지 않음. 이 섹션과 root `CLAUDE.md`에만 기록함.
- **2026-07-14의 `ChopsticksGraspRewardsCfg` 미사용 사고는 과거 Cube-Grasp 실험에 해당함.**
  `Indy-Wuji-Cube-Grasp`는 여전히 `CubeGraspRewardsCfg`를 쓰므로 Chopsticks cfg 변경이 Cube에 반영되지는
  않음. 2026-07-20부터 별도 `Indy-Wuji-Chopsticks-Grasp`가 `ChopstickAcquireRewardsCfg`를 실제로 사용함.
- **보상 가중치에는 dt(1/30)가 곱해짐** (`env/managers.py:427`). 일회성 보상의 실제 PPO 기여 = weight/30. 현재 `transport_success` weight 30000 = +1000.
- **`is_terminated_term`은 isaaclab 클래스형 reward term임.** 종료 계산이 보상 계산보다 먼저라 성공 종료 스텝에 같은 스텝 지급됨.
- **리셋은 관절 상태만 복원하고 위치 목표 버퍼는 안 채움.** 액션 밖 관절은 목표 0으로 저절로 이동함. `hold_joints_at_default` 리셋 이벤트로 해결함.
- **`init_state.joint_pos` 정규식 키 중복 매칭 주의.** `finger[1-5]` 키는 pop 후 세분화 키를 넣을 것.
- **`object_below_surface_penalty`는 "누르기" 감지에 못 씀.** 바닥이 강체라 관통 -0.04mm 수준(실측). 압착 억제는 r_T(성공 종료) 구조로 해결함.
- **파일이 세션 밖에서 바뀜** (사용자/다른 에이전트). 수정 전 재확인(`git diff`) 필수.
- **`cube_grasp_env_cfg.py`의 `__post_init__` surface_z 배선 블록(★ 표시)은 절대 지우지 말 것.**
  2026-07-15 편집 중 유실됨 — 없으면 상판 큐브의 clearance가 스폰부터 +BASE_Z라 lift 보상이
  만점에서 시작하는 대형 버그. 파일 구간을 재작성할 때 기존 오버라이드 줄을 보존할 것.
- **headless + 카메라 렌더 스크립트 행 걸림 이력** (grip_snapshot.py 24분). 눈 확인은 GUI 모드로.
- **CRLF/멀티라인 XML은 정규식이 조용히 실패함.** ElementTree로 파싱할 것.
- **물체 고정은 매 스텝 teleport 금지** (관통 누적 → PhysX 폭발). gravity off + 매 스텝 속도 0.
- **`finger*_tip_link` 원점은 마지막 관절임** (패드는 2~3cm 앞). **`joint1`은 감소가 하강임.**
- **git 멀티라인 커밋은 heredoc으로.** 사용자 터미널 복붙은 실패 이력 있음.
- **`ObjectToGoalProgressReward` 시그니처 변천 주의**: distance_max(v1) → potential_eps+window(v2)
  → potential_eps만(v2.1, 2026-07-16). 옛 파라미터를 cfg params에 남기면 env 생성이 TypeError로 죽음.
  box_mdp_cfg의 주석 블록을 살릴 때 현행 시그니처와 대조할 것.

> **2026-07-29 업데이트 — `hand_grasp` 수동 preload 제거 residual A/B**
>
> - episode/dwell은 `10 s/5 s`: reset의 무작위 초기 mode와 반대 mode 전환을 각각 한 번 경험함.
> - active action은 `CustomResidualJointPositionAction`이며
>   `target=current_joint_position+0.1×action`. 이전 target 누적형은 아님.
> - reset은 `pose_005` actual joint position을 실제 state와 PD target에 동일하게 써서 수동 preload를
>   주입하지 않음. 정책이 6개 contact를 유지하려면 nonzero residual로 필요한 PD 오차를 학습해야 함.
> - 수동 `PREGRASP_JOINT_TARGETS`와 `ReferenceResidualJointActionCfg` 설정은 삭제하지 않고 비활성
>   주석으로 보존함. obs/action은 `103D/20D`지만 action 의미가 달라졌으므로 반드시 fresh run을 사용함.

> **2026-07-29 업데이트 — `hand_grasp` 각속도 페널티 복구 + palm-relative 속도**
>
> - 각속도 페널티 제거 run `17-25-14`가 적용 run `16-24-00`보다 뚜렷하게 낫지 않았고,
>   단일 seed 변동도 커서 제거 근거가 없음. 현재 source는 두 stick 중 더 빠른 각속도의
>   `3 rad/s` 초과분에 `-0.1 * clip(excess, 0, 10)^2`를 다시 활성화한 상태임.
> - 향후 floating hand에서 손과 stick이 함께 움직이는 정상 파지를 world 속도로 감점하지 않도록
>   **독립 angular penalty, mode stability, `OpenCloseModeHeld` success 속도 gate,
>   `Metrics/hand_grasp*`의 speed/quiet-valid를 모두 palm-relative로 통일함.**
>   `ω_rel=ω_stick−ω_palm`,
>   `v_rel=v_stick−v_palm−ω_palm×(p_stick−p_palm)`을 사용함.
> - observation의 stick velocity 12D는 원래부터 같은 palm-relative 정의였음. 이제 obs/reward/success/
>   diagnostic의 기준이 일치함.
> - 현재 palm 고정 OPEN/CLOSE에서는 palm 속도가 거의 0이라 world와 relative 수치가 사실상 같음.
>   따라서 `16-24-00`은 고정-palm baseline으로 유효하고 이 변경만으로 폐기할 필요가 없음.
>   이미 실행 중인 process는 old code를 유지하며 새 run부터 변경이 적용됨. OPEN/CLOSE의 Stick1
>   상대 회전도 `3 rad/s`를 넘으면 계속 페널티를 받음.

> **2026-07-29 업데이트 — `hand_grasp` distal tip lateral alignment**
>
> - corrected radial gap만 맞추고 Stick1 distal tip이 Stick2 옆으로 빗겨나는 play 실패를 막기 위해
>   `pose_005` tip separation을 Stick2 local x-z 평면에 투영한 기준 방향
>   `(0.887693, 0, -0.460435)`을 추가함. 이는 검증 자세의 상대 방향이며 palm normal이나
>   파지 개구부 추정 법선을 사용한 것이 아님.
> - 현재 tip delta의 Stick2-local x-z 성분에서 기준 방향에 수직인 크기를 `tip_lateral_error`로
>   계산함. OPEN/CLOSE gap reward와 mode stability에 `exp(-error/0.005)`를 곱하고,
>   mode success에는 `error <= 0.005 m`를 요구함. 독립 lateral 연금은 없음.
> - TensorBoard 태그는
>   `Metrics/hand_grasp{,_final,_min,_max}/tip_lateral_error`임. diagnostic의
>   `tip_surface_gap`도 예전 3D norm 식에서 reward/success와 같은 Stick2-axis
>   square-section helper로 정정함.
> - obs/action은 `103D/20D` 그대로지만 reward/success 의미가 달라 적용 전 checkpoint를
>   resume하지 않고 fresh run으로 비교함.

> **2026-07-29 업데이트 — `hand_grasp_object` 환경 scaffold**
>
> - 기존 OPEN/CLOSE `hand_grasp`와 debug tools는
>   `nrmk_isaaclab_wuji/backups/hand_grasp_pre_object_env_2026-07-29/`에 백업하고,
>   원 task는 유지한 채 실제 물체가 추가된 별도 task `hand_grasp_object`를 등록함.
> - `pose_005` hand-stick 상대 자세 전체를 world x축 약 `-60.16°` 회전해 두 stick 평균 장축을
>   바닥과 평행하게 배치함. hand root는 아직 fixed이며 root action은 다음 단계임.
> - 매 reset에 Stick1/2 local `+y` distal endpoint의 reference midpoint를 계산해
>   `10 mm`, `2 g` dynamic cube를 놓음. 그 아래 `6 x 6 x 500 mm` kinematic post의 top face를
>   cube bottom에 맞춰 CLOSE 전 낙하를 막음. object contact sensor는 사용하지 않음.
> - 이번 scaffold에는 object observation/reward/termination/success를 넣지 않았고 기존
>   OPEN/CLOSE action/obs/reward를 그대로 상속함(`20D/103D`). 따라서 smoke는 scene/reset 배관
>   검증일 뿐 물체 파지 학습 결과가 아님.
> - `2026-07-30_09-16-16` 1-env/1-iteration smoke에서 scene, reset event 3개,
>   24 policy step을 통과함. 다음은 GUI에서 cube/support/stick 무충돌 배치를 눈으로 확인한 뒤
>   object grasp/lift와 hand root motion을 각각 추가함.

> **2026-07-30 업데이트 — `hand_grasp` gap/lateral 분리 + Stick2 anchor 강화**
>
> - 아래 내용이 07-29의 결합형 lateral reward 설명보다 최신임. OPEN/CLOSE gap reward는
>   `contact_gate * exp(-|gap-target|/5mm)`만 계산하고, lateral은 mode/contact와 독립인
>   bounded penalty `-5*(1-exp(-lateral_error/5mm))`로 분리함. 최종
>   `mode_grasp_stability`와 success의 lateral hard 조건은 유지함.
> - fresh run `2026-07-30_10-31-54` 초반(~330 iter)은 ring force가 `0.737 N`까지 생겼지만
>   Stick2 palm-frame pose error가 `9.72 mm / 0.580 rad(33.3°)`, 상대 각속도가
>   `6.45 rad/s`였음. 병목은 약지 접촉력 부족이 아니라 **접촉한 Stick2가 함께 밀리고 도는 것**임.
> - positive `stick2_reference_pose` weight `15`를 제거하고 위치/자세 bounded penalty를 분리함:
>   `-10*(1-exp(-e_pos/5mm))`, `-10*(1-exp(-e_ori/10°))`.
> - OPEN/CLOSE gap reward와 `mode_grasp_stability`에
>   `g_anchor=exp(-e_pos/5mm-e_ori/10°)`를 추가함. 이제 Stick2가 기준 자세를 떠난 채 gap만
>   맞추면 두 mode reward를 충분히 받을 수 없음. Stick2는 물리적으로 fixed가 아니고, 정책이
>   palm/엄지 중간마디/약지 접촉으로 고정해야 하는 dynamic reference rail임.
> - success의 Stick2 hard limit도 `20 mm/20°`에서 `5 mm/10°`로 강화함. command dwell은
>   source 기준 `5 s`, episode은 `10 s`로 한 번 반전함. obs/action/entropy와 6-contact
>   latch는 `103D/20D/0.001` 및 기존 조건 그대로임.
> - 변경 전 backup:
>   `nrmk_isaaclab_wuji/backups/hand_grasp_pre_stick2_anchor_2026-07-30/`.
>   smoke `2026-07-30_11-25-34`에서 reward 13개, termination 5개, 24 step을 통과함.
>   reward 의미가 바뀌었으므로 이전 checkpoint를 resume하지 말고 fresh run으로 판정함.

> **2026-07-30 최신 active — `21-05-32` exact baseline 복원**
>
> - 위 Stick2-anchor 강화 구성은
>   `nrmk_isaaclab_wuji/backups/hand_grasp_stick2_anchor_2026-07-30/`에 보존하고 active에서
>   비활성화함. 현재 active `hand_grasp`는 run `2026-07-29_21-05-32`의 저장
>   `params/env.yaml`·`agent.yaml`과 같은 실행 구성임.
> - 실제 run 값은 episode `10 s`, command dwell **`2 s`**임. 과거 source 주석의 5초 설명은
>   틀렸고 저장 YAML을 우선함. 10초 동안 mode interval이 다섯 번 들어감.
> - active reward 10개:
>   joint reference `2`, Stick1 pivot `10`, positive Stick2 pose `15`,
>   OPEN/CLOSE gap+lateral 결합형 각 `20`, functional contact minimum `20`
>   (force scale `0.10 N`), mode stability `50`, angular excess `-0.1`,
>   success `30000`, action rate `-0.001`.
> - gap reward에는 6-contact gate와 Stick2 anchor gate가 없고, 독립 lateral penalty도 없음.
>   contact mean/min 강화와 `functional_contact_lost` termination도 active가 아님.
>   termination은 time-out, Stick1 drop, Stick2 drop, unreachable success의 4개임.
> - CLOSE는 exact baseline을 위해 `3 mm`, success 허용오차도 공통 `±3 mm`를 유지함.
>   0 mm CLOSE/더 작은 tolerance 제안은 이번 exact reproduction에는 섞지 않음.
> - smoke `2026-07-30_12-13-39`에서 reward 10개, termination 4개, 24 step 통과.
>   생성된 env/agent YAML은 원 run과 `num_envs=1`, `max_iterations=1`, log path만 다르고
>   나머지는 동일함. fresh 4096-env run을 돌리고 contact가 좋던 시점에서 사용자가 수동 중단함.
> - **단일 후속 변경:** reward/termination/PPO 구조는 그대로 두고 CLOSE 목표만 `3→0 mm`,
>   OPEN/CLOSE 공통 success gap tolerance를 `±3→±0.5 mm`로 변경함. dense gap reward의
>   `sigma=5 mm`는 유지함. 따라서 active는 엄밀한 exact baseline이 아니라
>   **`21-05-32 + close 정의 변경`**임.

### 2026-08-04~05 (hand_setting 획득 진단 + chopsticks 해킹)

- **헐거운 pose σ의 지름길 (hand_setting 획득 실패의 진짜 원인).** pair σ가 너무 헐거우면
  (pos 0.10/ori 90°) **리셋(스폰) 스틱 자세가 이미 pair_score 고득점** → 그 항 gradient가 죽고
  정책은 남은 **엄지 유클리드 거리(thumb_pivot)**만 맞추려 함 → 방향 무관이라 opposition(위) 대신
  **스틱 밑으로 파고듦("비비기")**. pair σ를 `pos 0.01/ori 0.25(~14°)`로 조여 리셋 baseline↓ →
  실제 정렬 강제 → **획득 첫 성공(08-05)**. kp/kd가 원인인 줄 알았으나 σ였음. **교훈**: dense pose
  보상 σ는 "리셋에서 낮은 점수"가 되게 조일 것 + min/유클리드 sub-term에 방향 없는 지름길 여지 점검.
- **hand_setting 진단 메트릭은 stage1 reward term에 하드-커플링됨.** metric configure가
  `stage1_joint_reference`/`reference_thumb_pivot_min` 등을 `get_term_cfg`로 찾음 — 없거나 weight가
  **정확히 0.0이면 reward manager가 skip**해서 hand_setting 메트릭 전체가 꺼짐. 끄되 메트릭을 살리려면
  **weight=1e-8**(주석·0.0 금지).
- **잔차 액션의 잡는 힘 = kp×scale**(effort_limit 아님). `target=현재각+action×0.1`이라 오차가 scale로
  상한 → 최대 유지토크 ≈ kp×scale(예 0.7×0.1=0.07 < effort 0.18). 힘 진단 시 이걸 볼 것.
- **reachability probe는 잔차 액션으로 구동할 것.** 직접 절대목표(`set_joint_position_target`)는
  effort까지 밀어 정책보다 세므로 과대평가됨. 정책 힘을 공정 반영하려면 `action=clamp((목표−현재)/scale,±1)`.
- **큰 weight 항에서 gate를 뺄 땐 대체 gate가 hack 상태에서 정확히 0인지 확인**(chopsticks 08-04).
  `goal_gate`는 물체 정지점에서 잔여(~0.033) 남아 hack 차단에 부족, `lift_gate`(clearance)는 테이블에서
  정확히 0이라 확실. flip/담기 항은 `lift_gate × goal_gate`로 둘 다 걸 것.
- **run 저장 폴더에 git 스냅샷이 있음** (`logs/.../<run>/git/*.diff` + base commit) → 옛 코드 복원 가능.
  단 **diff 전체 `git apply`는 worklog.md 한글 멀티바이트 구간에서 `corrupt patch`로 atomic 실패** →
  필요한 파일 섹션만 행 범위로 떼어 apply(16e10af 트리를 `git archive|tar`로 추출한 뒤).
- **hand_grasp(pre-grasped 유지)가 됨 ≠ hand_setting(편 손 획득)이 됨.** 획득은 훨씬 어렵고 힘/탐색
  민감. 새 kp/kd가 hand_grasp에서 잘 나온다고 hand_setting 획득 가능을 보장하지 않음.

### 2026-08-20 (MuJoCo 젓가락 파지 오진 6회)

`--hold-pose`(목표 고정) 실측으로 **Isaac 도 Stick1 을 0.65 초에 놓친다**는 게 확인됐다.
Stick2 만 정적으로 안정(Isaac 0.3mm / MuJoCo 208mm 낙하). Stick1 은 정책이 매 스텝
붙잡는 것이라 보정을 없애면 양쪽 다 떨어진다. **MuJoCo 충돌 형상·마찰·접촉 그룹·접촉
강성 어느 것도 원인이 아니었다.** 파지가 안 될 때 MuJoCo부터 의심하지 말고 Isaac 에서
같은 측정(`scripts/rsl_rl/measure_stick_hold_isaac.py`, 정책 불필요)을 먼저 할 것.

오진의 직접 원인은 전부 **측정 도구** 쪽이었다. 상세는 root `CLAUDE.md` 의
"MuJoCo 젓가락 파지 진단" 절. 요약:

- `--smoke-backend` 는 파지 테스트가 아니다 (액션 0 = `target = q_current`, 파지가
  구조적으로 풀림). `--hold-pose` 신설.
- `env.step()` 우회 시 액션 term 이 안 돈다 → 떠 있는 루트 태스크는 손이 떠내려간다.
- `env.reset()` 직후 data 버퍼는 리셋 값을 반영 안 할 수 있다 → 기준점은 물리 1스텝 뒤.
- `mesh_facenum` 은 렌더용. 볼록화 판별은 `mesh_graphadr >= 0`.
- 자작 기하 검사(좌표 변환 방향)를 세 번 근거로 썼다가 세 번 틀렸다. 시뮬레이터
  자신의 값(`mj_geomDistance`, `contact.dist`)만 쓸 것.

## Working Rules

- 관련 없는 IsaacLab core 파일은 수정하지 않음.
- 기존 예제를 직접 덮어쓰기보다 새 task/env로 구성함.
- 변경 전 `git status` 확인함.
- 큰 변경 전 계획 요약함.
- 작업 후 `WORKLOG.md` 기록함.
- active repo 내부 작업 기록은 `nrmk_isaaclab_wuji/worklog.md`에도 남김.
- 실행/학습 테스트는 `env_isaaclab` 기준임.
- 코드 수정 후 작은 테스트부터 실행함.
- 학습 smoke test는 `--num_envs 1 --max_iterations 1`부터 함.
- 이후 `128/20`, `512/500`, `4096/50000` 순서로 키움.
- commit은 사용자가 확인 후 진행함.
- 임의 commit 하지 않음.
- 사용자가 남긴 변경은 되돌리지 않음.

## Useful Commands

- 일반화된 실행/학습/play 명령은 root `CLI.md`에 정리함.
- 실행 흐름 공부용 문서는 root `flow_study.md`에 정리함.

- env 활성화함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
```

- 1회 smoke test 실행함.

```bash
python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 1 --max_iterations 1
```

- 중간 테스트 실행함.

```bash
python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 128 --max_iterations 20
```

- 긴 학습 실행함.

```bash
python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 4096 --max_iterations 50000
```

- TensorBoard 실행함.

```bash
tensorboard --logdir logs/rsl_rl/indy_wuji_reach --port 6006 --reload_interval 5
```

- GUI 확인 실행함.

```bash
python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --num_envs 1 --max_iterations 1
```

- cube grasp smoke test 실행함.

```bash
python scripts/rsl_rl/train.py --task Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1
```

- cube grasp action 진단 실행함.
- `raw`는 policy 출력, `applied`는 clip 후 env 입력, `target`은 관절 목표, `actual`은 실제 관절각임.

```bash
python scripts/rsl_rl/play.py \
  --task Indy-Wuji-Cube-Grasp \
  --num_envs 1 \
  --latest_run \
  --print_diagnostics \
  --print_action_interval 10
```

- cube grasp contact/lift scripted 확인 실행함.
- `GOOD_CONTACT thumb+middle`과 `max_clearance(m)`를 봄.

```bash
python scripts/debug/check_cube_contact_lift.py \
  --task Indy-Wuji-Cube-Grasp \
  --headless \
  --num-envs 1 \
  --settle-steps 30 \
  --close-steps 60 \
  --lift-steps 30
```

> **2026-07-30 업데이트 — `hand_setting` Stick2-first state gate**
>
> - contact-only run `2026-07-30_18-14-08`은 744 iter에도 최대 `3/6`이었고,
>   thumb-distal/palm/thumb-mid 세 reset-adjacent contact만 유지했음. 이후 빠진
>   index/middle/ring region만 `0.5`로 켰지만, GUI에서 엄지 쪽 pivot이 먼저 닫혀
>   Stick2가 엄지–검지 valley 안으로 진입하지 못하는 순서 문제가 확인됨.
> - active `hand_setting`은 hidden FSM/command/obs를 추가하지 않고 현재 상태 기반
>   `stick2_seated` gate를 사용함. Stick2 palm-relative pose가 `15 mm/20°` 이내이고
>   palm–Stick2와 thumb-middle–Stick2가 각각 `>=0.02 N`일 때만 gate가 1임.
> - gate 전에는 Stick2 reference pose `12`와 두 valley-anchor contact만 활성임.
>   gate 후에만 thumb-distal/index/middle/ring contact, index/middle/ring shaft-region
>   `0.5`, six-contact hard min `20`이 켜짐. 안착이 풀리면 즉시 다시 꺼짐.
> - joint reference, Stick1 reference, thumb region, completion/stability는 계속
>   weight `0`; action/obs는 `20D/101D`, strict success와 drop termination도 불변임.
>   reward term 수는 18개 그대로이며 기존 term의 함수만 gate wrapper로 교체함.
> - TensorBoard 네 family에 `stick2_seated`를 추가해 각 family `33개`,
>   총 `132개` hand-setting metric을 기록함. smoke
>   `hand_setting/2026-07-30_21-44-34`는 1 env/1 iter, 24 policy step을 통과했고
>   reset에서 gate와 후단 region/contact reward가 0인 것을 확인함.
> - reward objective가 달라졌으므로 이전 hand-setting checkpoint에 resume하지 않고
>   다음 비교는 fresh run으로 시작함.

> **2026-07-30 업데이트 — `hand_setting` force gate 폐기**
>
> - 위 Stick2-first force gate run `21-50-01`은 389 iter에도 gate/pose-valid가
>   전 구간 0, final Stick2 error 약 `45 mm/74°`, functional contact `1/6`이었음.
>   지속된 것은 palm–Stick2 하나이고 ring force는 로그상 0임.
> - `0.02 N` anchor force는 reciprocal support가 형성된 뒤 생기는 결과이므로
>   downstream finger reward의 선행 조건에서 제거함.
> - active `stick2_in_valley`는 검증된 `pose_005`의 Stick2 local `y=-60 mm`
>   handle-side centerline point와 directed `+y` axis를 비교하며 `5 mm/10°`
>   이내일 때 force 없이 1임.
> - gate 전에는 centerline/axis tracking `12`, ring region `0.5`, ring contact
>   `5/6`이 Stick2를 이동시킴. gate 후에 다른 다섯 contact와 index/middle region,
>   contact-min `20`이 함께 켜져 상호 지지 force를 동시에 만들 수 있음.
> - `0.02 N`은 strict six-contact success와 diagnostic
>   `stick2_seated=in_valley & palm & thumb-mid`에만 남음.
> - TensorBoard는 valley point/axis error와 in-valley를 포함해
>   `36 metrics × 4 families = 144 tags`. smoke `22-45-33`은 24 step 통과.
> - reward objective 변경으로 모든 이전 hand-setting checkpoint resume 금지,
>   다음 학습은 fresh run으로 시작함.

> **2026-07-31 업데이트 — `hand_setting` force-validated valley + 엄지 probe**
>
> - geometry-only run `2026-07-30_22-59-53`은 4784 iter까지
>   `stick2_in_valley=0`; best point/axis error도 약 `26.1 mm/45.6°`였음.
> - active dense geometry는 point/axis 지수의 `min`을 쓰며, loose
>   `20 mm/30°` corridor에서만 palm/thumb-middle force의 `min`을
>   `valley_anchor_support`로 보상함. strict in-valley는
>   `10 mm/15° + 두 anchor 각각 0.02 N`임. ring은 gate 전, Stick1의
>   나머지 접촉과 six-contact min은 gate 후 활성임.
> - `hand_setting_thumb_action_probe.py`에서 나머지 19 action을 0으로 두고
>   `finger1_joint2`만 `|a|=0.1~1.0`으로 왕복함. 실제 joint2는
>   `-0.1659`에서 최대 `+0.458 rad`까지 정상 이동·재신전했지만 valley
>   error는 전부 약 `46.4 mm/68.7°`, thumb-middle force 최대 `0.0177 N`.
>   **엄지 action scale 부족/음수 차단이 원인이 아님.** 다음 검증은 수동
>   순서인 `joint1 굽힘→joint2 누름→joint1 일부 재신전`이며, 그 전에
>   thumb scale을 키우지 않음. 상세는 `ACTIVITY_2026-07-31.md`.
>
> - 후속 active A/B는 valley reward만 격리함. nonzero weight는
>   `stick2_valley_approach=12`, `valley_anchor_support=12` 두 개뿐이며,
>   Stick1/ring/six-contact/success/action-rate reward는 모두 `0`으로
>   park함. geometry 항은 reset에서 support corridor가 0이라 필요한 선행
>   dense signal임. smoke `2026-07-31_10-05-54` 통과, fresh run 필수.
>
> - **후속 finite-shaft 수정:** 위의 local `y=-60 mm`를 current Stick2의
>   exact marker로 요구하는 설명은 더 이상 active가 아님. `-60 mm`는
>   `pose_005`에서 고정 palm-frame valley target을 복원할 때만 사용하고,
>   active distance는 그 target과 current Stick2의 180 mm finite
>   centerline segment 사이 최단거리임. 따라서 장축 sliding은 허용하되
>   무한 선 연장 false positive는 `±0.09 m` end clamp로 막음. directed
>   axis와 reciprocal anchor-force 조건은 유지하며 metric은
>   `stick2_valley_shaft_distance`. final smoke `2026-07-31_10-32-23` 통과.
