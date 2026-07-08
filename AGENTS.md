# AGENTS.md

## Project Context

- 목표는 Isaac Lab 기반 Indy7 + Wuji hand end-effector tracking RL 환경 구성임.
- 현재 workspace는 `~/wuji_indy_lab_51`임.
- 현재 실제 코드 디렉터리는 `~/wuji_indy_lab_51/nrmk_isaaclab_wuji`임.
- 초기 문서에는 repo가 `~/wuji_indy_lab_51/nrmk_isaaclab_public`로 기록돼 있었음.
- 현재 작업은 `nrmk_isaaclab_wuji` 기준으로 진행 중임.
- 기존 `chop_ws/chop_rl`, Isaac Sim 4.5, IsaacLab 2.2.1 실험은 폐기함.
- 현재는 잘 켜지는 Isaac Sim 5.1, IsaacLab 2.3 계열, `env_isaaclab` 환경 사용함.
- Neuromeka public/main branch 스타일을 우선 사용함.
- 목표는 full training 성능이 아니라 env 구조 이해와 arm end-effector tracking 구성임.
- 현재 active task는 `Indy-Wuji-Reach`임.
- 현재 active USD는 `indy7_wuji_right_simplified.usd`임.
- 초기 후보였던 `indy7_allegro_hand_right_simplified.usd`는 참고/비교용임.
- 현재 tracking body는 `palm_link`임.
- `tcp`는 USD articulation rigid body로 쓰기 부적합하다고 판단함.
- 현재 action은 arm 6축만 사용함.
- hand joint 20축은 articulation에는 남아 있음.
- hand action은 아직 policy action에 넣지 않음.
- hand joint는 초기 arm tracking 단계에서는 고정 또는 최소 취급함.
- 이후 필요하면 hand action까지 확장함.

## Current Implementation

- `Indy-Wuji-Reach`는 Neuromeka `Indy-Reach` 스타일로 구현됨.
- 공통 reach 구조는 `isaac_neuromeka/tasks/manipulation/reach/reach_env_cfg.py` 기반임.
- 공통 MDP 설정은 `isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py` 기반임.
- task override는 `isaac_neuromeka/tasks/manipulation/reach/indy_wuji/env_cfg.py`에 있음.
- gym registration은 `isaac_neuromeka/tasks/manipulation/reach/indy_wuji/__init__.py`에 있음.
- RSL-RL config는 `isaac_neuromeka/tasks/manipulation/reach/indy_wuji/learning/rsl_rl_cfg.py`에 있음.
- robot asset config는 `isaac_neuromeka/assets/indy.py`의 `INDY7_WUJI_RIGHT_CFG`임.
- arm action dim은 6임.
- policy observation dim은 현재 55임.
- observation은 arm 6축 joint position/velocity history만 사용함.
- hand joint는 observation에서 제외됨.
- `joint_vel` reward penalty도 arm 6축만 적용됨.
- `sim.render_interval = decimation` 적용됨.

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

## Working Rules

- 관련 없는 IsaacLab core 파일은 수정하지 않음.
- 기존 예제를 직접 덮어쓰기보다 새 task/env로 구성함.
- 변경 전 `git status` 확인함.
- 큰 변경 전 계획 요약함.
- 작업 후 `WORKLOG.md` 기록함.
- active repo 내부 작업 기록은 `nrmk_isaaclab_wuji/worklog.md`에도 남김.
- 날짜별 상세 활동 일지는 root의 `ACTIVITY_YYYY-MM-DD.md` 형식으로 남김.
- 2026-07-08 활동 일지는 `ACTIVITY_2026-07-08.md`임.
- 실행/학습 테스트는 `env_isaaclab` 기준임.
- 코드 수정 후 작은 테스트부터 실행함.
- 학습 smoke test는 `--num_envs 1 --max_iterations 1`부터 함.
- 이후 `128/20`, `512/100` 순서로 키움.
- commit은 사용자가 확인 후 진행함.
- 임의 commit 하지 않음.
- 사용자가 남긴 변경은 되돌리지 않음.

## Useful Commands

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

- 다음 권장 테스트 실행함.

```bash
python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 512 --max_iterations 100
```

- GUI 확인 실행함.

```bash
python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --num_envs 1 --max_iterations 1
```
