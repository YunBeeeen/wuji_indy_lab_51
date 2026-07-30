# Codex Notes

- 이 문서는 Codex가 반복하지 말아야 할 실수와 병목을 따로 남기는 운영 메모임.
- 사용자 연구 기록(`WORKLOG.md`, `ACTIVITY_*.md`, `study.md`, `thesis.md`)에는 Codex의 시행착오를 섞지 않음.
- 연구/실험 사실을 남겨야 할 때만 별도 요청을 받고 사용자 기록에 정리함.

## 2026-07-15 Codex 실수와 병목

- `Stage1`, `Easy`, `Hard`처럼 task/run 이름을 늘리면 checkpoint 선택과 해석이 꼬임. 현재는 `Indy-Wuji-Cube-Grasp` 하나를 기준으로 보고, run은 자동 최신 선택보다 확인한 run 이름을 우선함.
- checkpoint load 전에는 action dim, observation dim, reward cfg, env cfg를 먼저 확인해야 함. reach checkpoint와 cube grasp checkpoint처럼 shape가 다르면 play가 바로 깨짐.
- scene 배치는 학습 전에 반드시 눈과 probe로 확인해야 함. cube 높이, support/table 높이, hand 시작 높이, palm/cage 방향을 검증하지 않고 reward만 조정하면 원인 분리가 안 됨.
- TensorBoard 평균 `Metrics/cube/*`만 보고 판단하면 오진함. 평가는 `Metrics/cube_final/*`, `cube_clearance`, `cage_inside_frac`, `thumb/index/middle contact`, `cube_speed`를 같이 봐야 함.
- `palm_facing`을 끄려면 초기 방향이 이미 맞는지 먼저 확인해야 함. 반대로 절대형 양수 facing reward는 farming이 쉬우므로 차분형 또는 gate로만 써야 함.
- `cube_lift` weight를 키우기 전에 실제 lift가 발생하는지 먼저 확인해야 함. raw lift가 계속 0이면 weight를 키워도 학습 신호는 0임.
- lift가 발생한 뒤에는 height만 보지 말고 stable lift를 봐야 함. 현재 성공 종료는 `clearance > 0.08`, `gate > 0.3`, `hold_steps=15`임. 자세가 아쉬우면 `cube_speed`, contact group, stricter cage gate 같은 조건을 추가 검토함.
- `play.py --print_contact`나 joint detail을 interval 1로 켜면 GUI가 심하게 느려짐. 기본 진단은 `--print_diagnostics --print_action_interval 10` 정도로 시작함.
- play에서 episode가 빨리 끝나면 time-out이 아니라 `success` termination일 수 있음. `CubeGraspTerminationsCfg.success`와 `ObjectLiftedHeld` 조건을 먼저 확인함.
- `AGENTS.md`와 repo 내부 문서는 오래된 실험 기록이 섞여 있을 수 있음. 답하기 전에 현재 코드 diff와 active cfg를 확인함.
- 긴 학습 전에 scripted probe를 먼저 돌림. 최소 확인값은 `GOOD_CONTACT`, `max_clearance`, `cube_clearance`, cage/contact 유지 여부임.

## 현재 병목 메모

- 2026-07-15 현재 정책은 cube를 몇 cm 이상 들 수 있지만, 파지 자세가 아직 깔끔하지 않음.
- 문제는 "들 수 있느냐"에서 "안정적으로 들고 유지하느냐"로 넘어간 상태임.
- 다음 판단은 단순 height가 아니라 stable lift 기준으로 해야 함:
  - cube 최하점 clearance
  - cage/contact 유지
  - cube linear/angular speed
  - success 전후 자세 유지 시간
  - 엄지/검지/중지 중 실제 하중을 받는 contact group

## 2026-07-26 라이브 학습 로그 판독 부하

- 라이브 학습 중 두 run의 TensorBoard event 파일을 `EventAccumulator.reload()`로 훑어 scalar
  history를 읽으면서 디스크 I/O·CPU·RAM 경합을 유발해 컴퓨터가 멈춘 것처럼 느려졌음.
- 라이브 run은 전체 history 재로딩을 피하고, 기존 TensorBoard 캐시나 텍스트 로그의 최신 구간을
  우선 사용함. event 파일을 꼭 읽어야 하면 한 run씩, 낮은 CPU/I/O 우선순위와 제한된 scalar
  reservoir로 읽고 학습 성능 저하 여부를 즉시 확인함.
- checkpoint 전체 로드, Isaac 환경 생성, GPU 접근은 단순 run 상태 판독에 사용하지 않음.

## 2026-07-27 Isaac cfg 정적 확인

- 일반 Conda Python에서 cfg를 import해 함수 파라미터를 추가 확인하려다 Isaac Sim 전용 `pxr`
  모듈 부재로 import 단계가 실패함. 일반 인터프리터에서는 `py_compile`·소스 기반 검사를 쓰고,
  실제 import가 꼭 필요할 때만 Isaac Sim Python 환경을 사용함.
- `Metrics/cube_*/*_surface`를 물체와 fingertip collider 사이의 실제 표면 간격이라고 설명한 것은
  부정확했음. 구현은 fingertip **link frame origin** 한 점의 box SDF이며, Wuji tip collision mesh는
  원점에서 주로 local -z로 약 1.6~1.7cm 뻗음. 따라서 `surface≈1.5~1.7cm`도 collider 접촉과
  양립할 수 있음. 실제 접촉 판정은 contact sensor 또는 collider-aware pad point가 필요함.

## 2026-07-27 hand_grasp collision probe 판정 오류

- 첫 valley probe에서 성공 anchor를 `palm OR thumb_base OR index_base`로 두고 변위 20 mm,
  속도 0.5 m/s까지 허용해 palm-only 접촉 대부분을 feasible로 잘못 분류함.
- 콘솔에도 feasible 후보만 출력해 사용자가 모든 후보가 YES인 것으로 볼 수 있었음.
- valley는 palm 접촉 자체가 아니라 thumb-side와 index-side의 동시 접촉으로 판정하고,
  모든 후보의 YES/NO를 출력해야 함. 변위·속도 한계도 소형 물체 스케일에 맞춰야 함.
- 접촉/침투 위치에 rigid body를 순간이동시키는 probe는 solver의 depenetration 반발을
  기하학적 불가능과 섞을 수 있음. 후보 pose는 낮은 depenetration 속도에서 먼저 거르고,
  다음 단계에서는 collision-free 접근 또는 관절/물체 pose ramp로 확인해야 함.
- valley scan에서 사용하지 않는 stick을 중력이 켜진 채 공중 `z=1.2 m`에 주차해 매 candidate마다
  하늘에서 떨어지는 잘못된 시각화를 만들었음. unused body는 지면의 격리 위치에 주차하고,
  공중에서 수행해야 하는 pair sweep만 측정 중 일시적으로 gravity를 꺼야 함.
- palm-only false positive를 고친 뒤 반대로 `thumb-side AND index-side`를 필수화해 실제로
  `palm + thumb_mid`에서 안정된 valley 후보까지 전부 NO로 만드는 과도한 조건을 넣었음.
  index가 열린 상태의 valley anchor는 palm+한쪽 side 접촉으로 성립할 수 있고, index 동시 접촉은
  별도 관절 sweep 없이 필수화하면 안 됨. ContactSensor link 소유권과 성공 topology를 구분해야 함.
- isolated stick pair의 world `+x` offset/`z`축 회전을 hand palm frame에 옮기면서 같은 성분 번호를
  그대로 사용해 Stick1을 world `+z` 위에 쌓았음. 현재 hand rotation에서는
  `palm +x -> world +z`, `palm +z -> world +x`이므로, finger progression 간격은 palm `z`,
  닫힘 회전축은 palm `x`로 변환해야 함. frame 변환 뒤에는 world 방향을 수치로 다시 확인해야 함.
- 자동 CEM probe에서 `elite_count=max(4, ...)`로 고정해 1-env GUI 실행 시
  `torch.topk(k=4)`가 범위를 넘어 즉시 실패함. population 기반 코드는 항상
  `1 <= k <= num_envs`로 clamp하고, 소규모 env에서는 CEM이 아니라 약한 sequential search라는
  사실을 경고해야 함.
- IK CEM을 25회 teleported FK 뒤 10회 physics-aware rollout으로 구성하면서 PHYS 전환 시
  이미 joint1 상한 근처로 붕괴한 평균·분산을 그대로 넘겼음. 물리 loss가 커도 10회 동안 다른
  basin을 탐색하지 못했으므로, objective를 바꾸는 stage 경계에서는 이전 archive를 지우는 것뿐
  아니라 평균을 중앙으로 당기고 분산도 다시 넓혀야 함.
- 모든 관절을 동시에 선형 ramp하면 distal joint3/4가 먼저 오므라들어 palm/self-collision에
  걸린 뒤 proximal joint1이 못 움직이는 경로 의존성을 최종 pose 불가능으로 오인할 수 있음.
  hand IK 물리 검증은 proximal joint1/2를 먼저 보낸 뒤 충분히 settle하고, distal joint3/4를 닫은
  뒤에도 다시 settle하는 staged trajectory와 proximal-stage tracking을 함께 기록해야 함. ramp만
  나누고 stage 사이 settle을 생략하면 아직 도달 중인 오차를 기계적 불가능으로 오인할 수 있음.
- self-collision ON/OFF 비교를 한 Isaac 앱 안에서 `env.close()` 후 두 번째 ManagerBasedEnv를
  생성하는 방식으로 만들었더니 첫 ON 측정 후 두 번째 scene 생성 중 Python traceback 없이
  프로세스가 종료됨. Isaac Sim/SimulationContext 재생성을 한 프로세스에서 연속 수행하는 probe를
  만들지 말고, 조건별로 독립 프로세스를 실행해 결과 파일을 비교해야 함.

## 2026-07-28 hand_grasp play CLI 옵션 오기

- pre-grasp 시각화 명령에 존재하지 않는 `--load_checkpoint`를 안내해 Isaac Sim을 불필요하게
  기동한 뒤 Hydra 인자 오류로 종료시킴. 이 저장소의 `scripts/rsl_rl/cli_args.py`가 정의하는
  옵션은 `--checkpoint`임. 실행 CLI를 안내하기 전 프로젝트의 실제 argparse 정의를 확인할 것.
- 같은 명령의 Hydra override에서 `episode_length_s=1000000`을 정수로 전달해 configclass의
  `float` 타입 검사를 실패시킴. 숫자 override도 대상 필드 타입에 맞춰
  `episode_length_s=1000000.0`처럼 명시해야 함.

## 2026-07-28 hand_grasp tip 방향 오판

- collision probe의 임시 `tip=-y` 관례를 active `pose_005`에도 그대로 적용해 stick tip이
  local `-y`라고 먼저 설명함. 저장 pose의 양 끝점을 palm frame에서 수치로 비교했어야 함.
- `pose_005`에서는 local `+y` 끝이 palm에서 더 멀고 두 stick의 해당 endpoint 장축 오차도
  약 `0.4 mm`로 정렬되어 있으므로 active OPEN/CLOSE tip은 local `+y`가 맞음.
- 앞으로 uniform proxy처럼 시각적 tip/tail 구분이 없는 경우 debug 스크립트 관례를 재사용하지
  말고, active saved pose에서 palm 거리·두 endpoint 정렬·접촉 위치를 함께 계산해 정함.
## 2026-07-29 hand_grasp_object smoke

- IsaacLab `configclass`는 scene 항목을 class attribute로 남기지 않으므로
  `HandGraspSceneCfg.robot` 직접 접근은 import에서 실패함. base cfg 인스턴스를 만든 뒤
  `_BASE_HAND_SCENE.robot.replace(...)`로 복제해야 함.
- 일반 sandbox smoke는 GPU/X 접근이 차단되어 실제 env 판정에 사용할 수 없었고,
  승인된 1-env Isaac smoke로 scene/reset/24-step을 확인함.

## 2026-07-30 shell 검색 패턴 quoting

- 문서 line-number 확인용 `rg` 명령의 double-quoted pattern 안에 Markdown backtick을 넣어
  shell command substitution이 발생했고, 불필요한 `hand_setting: command not found`가 한 번 출력됨.
  파일 변경이나 학습 프로세스 영향은 없었음. shell에 넘기는 검색 패턴은 single quote로 감싸거나
  backtick을 제외해 command substitution 가능성을 차단할 것.
- Git 업로드 전 docstring 누락을 찾는 임시 `python -c`에서 f-string 바깥/안쪽에 같은 double quote를
  사용해 8개 병렬 진단이 모두 `SyntaxError`로 끝남. 파일 변경은 없었고, 다음 호출은 문자열 결합과
  서로 다른 quote 층을 사용해 정상 실행함. 짧은 진단도 shell/Python 두 단계 quoting을 먼저 단순화할 것.
