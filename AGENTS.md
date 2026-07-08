# AGENTS.md

## Project Context

- 이 프로젝트의 목표는 Isaac Lab 기반으로 Indy7 + Wuji/Allegro hand end-effector tracking RL 환경을 구성하는 것이다.
- 현재 작업 workspace는 `~/wuji_indy_lab_51`이다.
- 작업 repo는 `~/wuji_indy_lab_51/nrmk_isaaclab_public`로 기록한다.
- 현재 workspace에서 확인되는 코드 디렉터리는 `~/wuji_indy_lab_51/nrmk_isaaclab_wuji`이므로, 실제 작업 전 repo 경로를 다시 확인한다.
- 기존 `chop_ws/chop_rl`, Isaac Sim 4.5 / IsaacLab 2.2.1 실험은 폐기한다.
- 현재는 기존에 잘 켜지는 Isaac Sim 5.1 / IsaacLab 환경을 사용하는 방향이다.
- Neuromeka main branch 기반 코드를 우선 사용한다.
- 처음 목표는 full training 성능이 아니라, env 구조 이해와 end-effector tracking 환경 구성이다.
- 사용할 USD 파일은 `indy7_allegro_hand_right_simplified.usd`이다.
- 우선 arm 6축 기반 end-effector tracking부터 구성하고, hand joint는 처음에는 고정하거나 최소한으로 다룬다.
- 이후 필요하면 hand action까지 확장한다.

## Study Order

1. Direct Cartpole
2. Isaac-Ant-v0
3. Neuromeka Indy-Reach
4. 새 Wuji/Indy Reach 환경 구성

## Working Rules

- 관련 없는 IsaacLab core 파일은 되도록 수정하지 않는다.
- 기존 예제를 직접 덮어쓰기보다 새 task/env를 만드는 방향으로 진행한다.
- 변경 전 `git status`를 확인한다.
- 큰 변경은 하기 전에 계획을 먼저 요약한다.
- 작업 후 `WORKLOG.md`에 변경 사항, 실행 명령어, 결과, 다음 할 일을 기록한다.
- 실행/학습 테스트는 `env_isaaclab` 환경 기준으로 생각한다.
- 코드 수정 후에는 가능한 한 작은 테스트 명령부터 실행한다.
- commit은 사용자가 직접 확인 후 진행하므로 임의로 commit하지 않는다.
