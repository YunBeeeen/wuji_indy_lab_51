# `train.py` 실행 흐름 추적

- 이 문서는 `scripts/rsl_rl/train.py`가 Isaac Lab task를 실행하는 흐름을 추적한 코드 흐름 분석 문서임.

기준 명령:

```bash
conda activate env_isaaclab
cd /home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji
PYTHONPATH=/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji:$PYTHONPATH \
python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --num_envs 1 --max_iterations 1 --headless
```

## 1. Conda 환경

`env_isaaclab` 안의 Python이 사용된다.

- Python executable: `/home/lsc/anaconda3/envs/env_isaaclab/bin/python`
- Isaac Sim: pip package `isaacsim==5.1.0.0`
- IsaacLab source: `/home/lsc/IsaacLab/source/...`
- 새 extension: `PYTHONPATH`로 `/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji`를 앞에 추가

## 2. CLI parsing

파일:

```bash
scripts/rsl_rl/train.py
```

중요 인자:

- `--task`: Gym task id. 예: `Indy-Wuji-Reach`
- `--num_envs`: 병렬 환경 수
- `--max_iterations`: RSL-RL 학습 반복 수
- `--device`: `cuda`, `cuda:0`, `cpu` 등
- `--video`: 학습 중 영상 저장 여부
- `--headless`: GUI 없이 실행

`cli_args.add_rsl_rl_args(parser)`는 RSL-RL 전용 인자를 추가하고, `AppLauncher.add_app_launcher_args(parser)`는 Isaac Sim 실행 관련 인자를 추가한다.

## 3. AppLauncher

`train.py` 초반에 아래 순서가 나온다.

```python
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

이 단계가 Isaac Sim/Kit runtime을 먼저 연다. `pxr`, `omni`, USD 관련 모듈은 이 이후에 안정적으로 import된다. 그래서 IsaacLab task package를 일반 Python처럼 바로 import하면 `pxr`가 없다는 식의 에러가 날 수 있다.

## 4. Task registration

`train.py`는 IsaacLab 기본 task와 Neuromeka task를 import한다.

```python
import isaaclab_tasks
import isaac_neuromeka.tasks
```

`isaac_neuromeka.tasks.__init__` 내부에서 `import_packages(...)`가 하위 package를 훑고, 각 task package의 `gym.register(...)`가 실행된다.

새 task:

```python
gym.register(
    id="Indy-Wuji-Reach",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg.Indy7WujiReachEnvCfg,
        "rsl_rl_cfg_entry_point": "...ReachPPORunnerCfg",
    },
)
```

## 5. Hydra config loading

`main` 함수는 decorator로 감싸져 있다.

```python
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
```

여기서 `args_cli.task`가 `Indy-Wuji-Reach`이면 Gym registry에서 task kwargs를 찾고, `env_cfg_entry_point`와 `rsl_rl_cfg_entry_point`를 실제 config 객체로 만든다.

결과:

- `env_cfg`: scene, robot, action, observation, reward, termination 설정
- `agent_cfg`: PPO runner, policy network, optimizer, iteration 수 설정

## 6. CLI override

`main` 안에서 CLI 인자가 config를 덮어쓴다.

```python
env_cfg.scene.num_envs = args_cli.num_envs
agent_cfg.max_iterations = args_cli.max_iterations
env_cfg.sim.device = args_cli.device
```

즉 config 파일에 `num_envs=4096`으로 되어 있어도 `--num_envs 1`을 주면 1개 환경만 뜬다.

## 7. Environment creation

```python
env = gym.make(args_cli.task, cfg=env_cfg, render_mode=...)
```

`entry_point="isaaclab.envs:ManagerBasedRLEnv"`이므로 IsaacLab의 manager-based RL environment가 생성된다.

이때 중요한 config 묶음:

- `scene`: robot, terrain, lights, number of envs
- `actions`: policy action을 robot joint command로 바꾸는 규칙
- `observations`: policy input
- `rewards`: 각 reward term과 weight
- `terminations`: episode 종료 조건
- `commands`: 목표 end-effector pose 같은 command generator

## 8. RSL-RL wrapper와 runner

IsaacLab environment는 RSL-RL이 기대하는 interface로 감싼다.

```python
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
```

`agent_cfg.to_dict()` 안에는 PPO algorithm, actor/critic hidden layer, learning rate, batch 설정 등이 들어간다.

## 9. 학습 시작

```python
runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
```

이 한 줄부터 RSL-RL이 rollout 수집, reward 계산, advantage 계산, PPO update를 반복한다.

로그는 보통 아래로 들어간다.

```bash
logs/rsl_rl/<experiment_name>/<timestamp>/
```

`Indy-Wuji-Reach`의 기본 experiment name은 `indy_wuji_reach`로 분리했다.

## 10. 중요한 파라미터 우선순위

처음 공부할 때는 아래 순서로 보면 좋다.

1. `--task`: 어떤 env config와 agent config를 불러오는지 결정
2. `--num_envs`: 병렬 환경 수, GPU 메모리와 학습 속도에 직접 영향
3. `--max_iterations`: 학습 길이
4. `env_cfg.scene.robot`: 어떤 USD asset을 띄우는지
5. `actions.arm_action.scale`: policy output이 joint target으로 얼마나 크게 반영되는지
6. `rewards.*.weight`: agent가 무엇을 잘했다고 보는지
7. `sim.dt`, `decimation`: 물리 step과 RL action 주기
8. `agent_cfg.policy.*hidden_dims`: actor/critic network 크기
9. `agent_cfg.algorithm.learning_rate`: PPO update 크기

## 11. Indy-Wuji 현재 상태

현재 `Indy-Wuji-Reach`는 hand가 붙은 모델로 Indy reach task를 먼저 검증하는 용도다.

- arm action: `joint[0-5]`
- hand joints: asset에는 actuator가 있지만 policy action에는 아직 연결하지 않음
- end-effector body: `tcp`

손가락까지 학습시키려면 다음 단계에서 hand action term, fingertip observation, grasp/contact reward를 추가한다.
