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
