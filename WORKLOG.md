# WORKLOG.md

## Current Status

- 현재 목표는 `indy7_allegro_hand_right_simplified.usd`를 이용해서 end-effector tracking 환경을 구성하는 것
- 기존 4.5/2.2.1 맞추기 작업은 중단하고, 잘 켜지는 Isaac Sim 5.1 / IsaacLab 환경을 사용하기로 함
- 현재는 코드 구현보다 Isaac Lab env 구조 이해와 새 env 설계가 우선임
- `~/wuji_indy_lab_51`와 `~/wuji_indy_lab_51/nrmk_isaaclab_wuji`에서 `git status --short`를 실행했으나, 현재 checkout에는 유효한 `.git` metadata가 없어 git repo로 인식되지 않음
- 문서에 기록된 target repo root는 `~/wuji_indy_lab_51/nrmk_isaaclab_public`이나, 현재 workspace에서 확인되는 코드 디렉터리는 `~/wuji_indy_lab_51/nrmk_isaaclab_wuji`임

## Environment

- Workspace: `~/wuji_indy_lab_51`
- Repo root: `~/wuji_indy_lab_51/nrmk_isaaclab_public`
- Main conda env: `env_isaaclab`
- Isaac Sim: existing working 5.1 environment
- Isaac Lab: existing installed IsaacLab environment
- Neuromeka code: main branch preferred

## Important Decisions

- `chop_ws/chop_rl` 기반 Isaac Sim 4.5 / IsaacLab 2.2.1 실험은 폐기
- Neuromeka main branch 기반으로 진행
- 처음에는 arm 6축만 제어하고 hand joint는 고정 또는 최소 제어
- 새 환경은 기존 Indy-Reach를 바로 덮어쓰기보다 새 task/env로 구성하는 방향
- 초기 목표는 학습 성능이 아니라 env가 정상적으로 생성되고 end-effector tracking reward/action/observation 구조가 돌아가는 것

## Study Plan

1. Direct Cartpole

   - DirectRLEnv 기본 구조 이해
   - action 적용, observation 구성, reward 계산, done/reset 흐름 파악

2. Isaac-Ant-v0

   - articulation robot DirectRLEnv 구조 이해
   - robot asset을 scene에 넣는 방식
   - joint action 적용 방식
   - observation/reward/done/reset 함수 구조 확인

3. Neuromeka Indy-Reach

   - end-effector pose command 구조 확인
   - position/orientation tracking reward 확인
   - arm action 설정 확인
   - observation 구성 확인

4. Wuji/Indy Reach 환경 구성

   - `indy7_allegro_hand_right_simplified.usd` 로드
   - joint/body 이름 확인
   - end-effector body 후보 선정
   - arm 6축 action 구성
   - hand joint 초기 고정
   - end-effector tracking reward 구성
   - headless train 1 iteration 테스트

## Known References

- IsaacLab Direct Cartpole:

  `source/isaaclab_tasks/isaaclab_tasks/direct/cartpole/cartpole_env.py`

- IsaacLab Ant:

  `source/isaaclab_tasks/isaaclab_tasks/direct/ant/ant_env.py`

- Neuromeka Indy-Reach:

  `isaac_neuromeka/tasks/manipulation/reach/indy/env_cfg.py`

  `isaac_neuromeka/tasks/manipulation/reach/indy/learning/rsl_rl_cfg.py`

  `isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py`

  `isaac_neuromeka/mdp/rewards.py`

  `isaac_neuromeka/mdp/observations.py`

  `isaac_neuromeka/mdp/commands.py`

## Next Steps

- `env_isaaclab` 환경에서 IsaacLab 예제 실행 확인
- Direct Cartpole 구조 분석
- Isaac-Ant-v0 구조 분석
- Neuromeka Indy-Reach 구조 분석
- `indy7_allegro_hand_right_simplified.usd`의 joint/body/end-effector 후보 이름 확인
- 새 Wuji/Indy end-effector tracking task 설계

## Change Log

- Initial project handoff documents created.
- Added project goal, current environment direction, study plan, references, and next steps.
- Ran `git status --short` before changes. Result: current workspace does not contain valid git metadata, so git did not recognize it as a repository.
