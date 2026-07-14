# flow_study.md

- 이 문서는 Isaac Lab manager-based task 실행 흐름과 코드 연결 관계를 공부하기 위한 흐름 정리 문서임.

## 목적

- `Indy-Wuji-Reach` 실행 흐름을 헷갈릴 때 다시 보기 위한 문서임.
- 파일이 어떤 순서로 import되고 연결되는지 정리함.
- 핵심 코드 발췌와 의미를 같이 정리함.
- 현재 기준 task body는 `link6`임.
- 현재 action은 arm 6축만 사용함.
- 현재 policy observation dim은 55임.

## 큰 그림

```text
train.py
-> isaac_neuromeka.tasks import
-> tasks/__init__.py
-> indy_wuji/__init__.py
-> gym.register("Indy-Wuji-Reach")
-> gym.make("Indy-Wuji-Reach")
-> Indy7WujiReachEnvCfg
-> ReachEnvCfg
-> env_cfg_common.py
-> isaac_neuromeka/mdp/*.py
-> RSL-RL PPO runner
```

- `train.py`는 실행기임.
- `tasks/__init__.py`는 task 패키지를 자동 import함.
- `indy_wuji/__init__.py`는 task 이름을 gym에 등록함.
- `reach_env_cfg.py`는 reach task 공통 조립 틀임.
- `env_cfg_common.py`는 command/action/observation/reward 부품 목록임.
- `indy_wuji/env_cfg.py`는 Indy-Wuji 전용 override임.
- `isaac_neuromeka/mdp/*.py`는 실제 계산 함수 모음임.

## 1. train.py

파일임.

```text
nrmk_isaaclab_wuji/scripts/rsl_rl/train.py
```

핵심 코드임.

```python
import isaac_neuromeka.tasks  # noqa: F401

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
```

의미임.

- `import isaac_neuromeka.tasks`가 task 등록을 시작함.
- `@hydra_task_config(...)`가 task id에 맞는 env cfg와 agent cfg를 가져옴.
- `gym.make(args_cli.task, cfg=env_cfg, ...)`가 실제 env를 생성함.
- `RslRlVecEnvWrapper`가 IsaacLab env를 RSL-RL이 먹을 수 있는 형태로 감쌈.
- 여기서 `args_cli.task`가 `Indy-Wuji-Reach`임.

## 2. tasks 자동 import

파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/tasks/__init__.py
```

핵심 코드임.

```python
from isaaclab_tasks.utils import import_packages

_BLACKLIST_PKGS = ["utils"]
import_packages(__name__, _BLACKLIST_PKGS)
```

의미임.

- `isaac_neuromeka.tasks` 아래 패키지를 자동으로 import함.
- `tasks/manipulation/reach/indy_wuji/__init__.py`도 이 과정에서 실행됨.
- 직접 `indy_wuji`를 import하지 않아도 task 등록이 되는 이유임.

## 3. Indy-Wuji task 등록

파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/reach/indy_wuji/__init__.py
```

핵심 코드임.

```python
gym.register(
    id="Indy-Wuji-Reach",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.Indy7WujiReachEnvCfg,
        "rsl_rl_cfg_entry_point": f"{learning.__name__}.rsl_rl_cfg:ReachPPORunnerCfg",
    },
)
```

의미임.

- `Indy-Wuji-Reach`라는 task 이름을 등록함.
- 실제 env class는 `ManagerBasedRLEnv`임.
- env 설정은 `Indy7WujiReachEnvCfg`임.
- PPO 설정은 `ReachPPORunnerCfg`임.
- 그래서 `train.py --task Indy-Wuji-Reach`가 이 cfg들을 찾아감.

## 4. Reach 공통 조립 틀

파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/reach/reach_env_cfg.py
```

핵심 코드임.

```python
class ReachEnvCfg(NrmkRLEnvCfg):
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg | EmptyCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg | EmptyCfg = EventCfg()
```

의미임.

- `ReachEnvCfg`는 reach task 공통 조립 틀임.
- scene, observation, action, command, reward, termination, event를 한 번에 묶음.
- 여기서 쓰는 `ObservationsCfg`, `ActionsCfg`, `CommandsCfg`, `RewardsCfg`는 `env_cfg_common.py`에서 가져옴.
- 즉 `reach_env_cfg.py`는 부품을 직접 계산하지 않고 부품을 모아 env cfg를 만듦.

기본 시간 설정임.

```python
def __post_init__(self):
    self.decimation = 24
    self.episode_length_s = 8.0
```

의미임.

- physics step 24번마다 policy action 한 번 적용됨.
- episode 길이는 8초임.

## 5. 공통 MDP 부품 설정

파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/common/env_cfg_common.py
```

### Command 부품

핵심 코드임.

```python
ee_pose = mdp.UniformPoseCommandCfg(
    asset_name="robot",
    body_name=MISSING,
    resampling_time_range=(6.0, 10.0),
    debug_vis=True,
    ranges=mdp.UniformPoseCommandCfg.Ranges(
        pos_x=(ConFig.default_ee_pose[0], ConFig.default_ee_pose[0] + 0.3),
        pos_y=(ConFig.default_ee_pose[1] - 0.2, ConFig.default_ee_pose[1] + 0.2),
        pos_z=(ConFig.default_ee_pose[2] - 0.3, ConFig.default_ee_pose[2]),
        roll=(0.0, 0.0),
        pitch=(math.pi, math.pi),
        yaw=(-3.14, 3.14),
    ),
)
```

의미임.

- `ee_pose`라는 target pose command를 만듦.
- target position/orientation 범위를 정함.
- `body_name=MISSING`은 아직 body를 정하지 않았다는 뜻임.
- 실제 body는 `indy_wuji/env_cfg.py`에서 `link6`로 채움.

### Observation 부품

핵심 코드임.

```python
joint_pos = ObsTerm(func=mdp.joint_pos, noise=Gnoise(std=0.01), history_length=3)
joint_vel = ObsTerm(func=mdp.finite_joint_vel, noise=Gnoise(std=0.1), history_length=3)
pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
action_history = ObsTerm(func=mdp.action_history)
```

의미임.

- policy가 볼 observation 목록임.
- `func=mdp.joint_pos`는 실제 함수가 `isaac_neuromeka/mdp/observations.py`에 있다는 뜻임.
- `history_length=3`이라 joint position/velocity가 3 step history로 들어감.
- `pose_command`는 현재 target command를 observation에 넣음.
- `action_history`는 이전 action과 현재 action을 observation에 넣음.

### Reward 부품

핵심 코드임.

```python
end_effector_position_tracking = RewTerm(
    func=mdp.end_effector_position_tracking_bounded,
    weight=0.2,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
        "command_name": "ee_pose",
        "distance_max": 0.5,
    },
)
```

```python
end_effector_orientation_tracking = RewTerm(
    func=mdp.end_effector_orientation_tracking_distance_bounded,
    weight=0.1,
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
        "command_name": "ee_pose",
        "distance_max": 0.25,
    },
)
```

의미임.

- position tracking reward와 orientation tracking reward를 정의함.
- `func=mdp.xxx`는 실제 계산 함수 이름임.
- `weight`는 reward 가중치임.
- `asset_cfg.body_names=MISSING`은 아직 어떤 body를 볼지 모른다는 뜻임.
- 실제 body는 `indy_wuji/env_cfg.py`에서 `link6`로 채움.

## 6. Indy-Wuji 전용 override

파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/reach/indy_wuji/env_cfg.py
```

핵심 코드임.

```python
class Indy7WujiReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = INDY7_WUJI_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.render_interval = self.decimation
```

의미임.

- `ReachEnvCfg` 공통 구조를 먼저 초기화함.
- 그 다음 robot asset을 Indy7 + Wuji asset으로 교체함.
- render interval도 decimation에 맞춤.

현재 tracking body 설정임.

```python
self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link6"]
self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link6"]
self.rewards.end_effector_speed.params["asset_cfg"].body_names = ["link6"]
self.commands.ee_pose.body_name = "link6"
```

의미임.

- command가 추적할 body를 `link6`로 지정함.
- position reward가 볼 body도 `link6`임.
- orientation reward가 볼 body도 `link6`임.
- speed penalty가 볼 body도 `link6`임.
- 지금은 Wuji hand frame 문제를 분리하고 arm flange 기준 reach baseline을 보는 단계임.

현재 arm-only 설정임.

```python
arm_joint_names = ["joint[0-5]"]

def arm_joint_cfg():
    return SceneEntityCfg("robot", joint_names=arm_joint_names)
```

의미임.

- arm 6축만 선택함.
- `SceneEntityCfg`는 이름 기반 설정이고, IsaacLab manager가 나중에 실제 `joint_ids`로 resolve함.

observation override임.

```python
self.observations.policy.joint_pos.params = {"asset_cfg": arm_joint_cfg()}
self.observations.policy.joint_vel.params = {"asset_cfg": arm_joint_cfg()}
```

의미임.

- 공통 observation 함수는 그대로 씀.
- 다만 그 함수가 볼 joint를 arm 6축으로 제한함.

action override임.

```python
self.actions.arm_action = mdp.JointPositionActionCfg(
    class_type=CustomJointPositionAction,
    asset_name="robot",
    joint_names=arm_joint_names,
    scale=0.2,
    use_default_offset=True,
)
```

의미임.

- policy action 6개를 arm joint position target으로 바꿈.
- `scale=0.2`라 action output이 joint target 변화량에 스케일되어 들어감.
- `use_default_offset=True`라 default joint pose 기준 offset action처럼 동작함.

## 7. 실제 reward 계산 함수

파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/mdp/rewards.py
```

position reward임.

```python
des_pos_b = command[:, :3]
des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]

distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
distance_bonus = 1.0 - torch.clamp(distance, 0.0, distance_max) / distance_max
```

의미임.

- command position을 world frame으로 바꿈.
- 현재 body position을 가져옴.
- 둘 사이 거리를 계산함.
- 거리가 작을수록 reward가 큼.
- 현재 `asset_cfg.body_ids[0]`는 `link6`로 resolve됨.

orientation reward임.

```python
des_quat_b = command[:, 3:7]
des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]

distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
orientation_error = quat_error_magnitude(curr_quat_w, des_quat_w)
orientation_bonus = 1.0 - torch.clamp(orientation_error, 0.0, 3.14) / 3.14

bad_indicies = distance > distance_max
total_reward = orientation_bonus
total_reward[bad_indicies] = 0.0
```

의미임.

- command orientation을 world frame으로 바꿈.
- 현재 body orientation을 가져옴.
- quaternion error를 계산함.
- error가 작을수록 reward가 큼.
- position distance가 `distance_max`보다 크면 orientation reward를 0으로 꺼버림.
- 그래서 학습 초반에는 position이 먼저 잡혀야 orientation reward가 잘 들어감.

## 8. 실제 observation 계산 함수

파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/mdp/observations.py
```

joint position observation임.

```python
def joint_pos(env, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]
```

의미임.

- robot joint position을 가져옴.
- 현재 `asset_cfg.joint_ids`는 arm 6축으로 resolve됨.

joint velocity observation임.

```python
def finite_joint_vel(env, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    return asset._finite_joint_vel[:, asset_cfg.joint_ids]
```

의미임.

- finite difference 기반 joint velocity를 가져옴.
- 현재 arm 6축만 observation으로 들어감.

action history observation임.

```python
def action_history(env):
    return torch.cat((env.action_manager.prev_action, env.action_manager.action), dim=-1)
```

의미임.

- 이전 action과 현재 action을 이어붙임.
- 현재 action dim이 6이라 action history는 12임.

## 9. robot asset 설정

파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/assets/indy.py
```

핵심 이름임.

```python
INDY7_WUJI_RIGHT_CFG = FiniteArticulationCfg(...)
```

의미임.

- Indy7 + Wuji robot asset cfg임.
- USD path, actuator, initial state 같은 robot 설정이 들어감.
- `indy_wuji/env_cfg.py`에서 이 cfg를 scene robot으로 넣음.

## 10. PPO 설정

파일임.

```text
nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/reach/indy_wuji/learning/rsl_rl_cfg.py
```

핵심 이름임.

```python
class ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    ...
```

의미임.

- RSL-RL PPO runner 설정임.
- network size, num steps per env, learning rate, checkpoint interval 같은 학습 설정을 담음.
- reward가 흔들리거나 학습이 너무 공격적이면 나중에 여기서 learning rate 등을 봄.

## 11. mdp 폴더 두 개 구분

전역 MDP 함수임.

```text
isaac_neuromeka/mdp/
```

- 현재 `Indy-Wuji-Reach`가 주로 쓰는 함수들이 있음.
- `mdp.joint_pos`, `mdp.finite_joint_vel`, `mdp.action_history`, `mdp.end_effector_position_tracking_bounded` 등이 여기 있음.

manipulation common 보조 MDP임.

```text
isaac_neuromeka/tasks/manipulation/common/mdp/
```

- object position, pointcloud 같은 manipulation task 보조 함수가 있음.
- 현재 `Indy-Wuji-Reach` 핵심 흐름에서는 거의 안 봐도 됨.

## 12. 헷갈릴 때 보는 순서

```text
1. indy_wuji/__init__.py
   - task 이름 등록 확인함.

2. indy_wuji/env_cfg.py
   - 이번 task에서 바꾼 robot/body/joint 확인함.

3. reach_env_cfg.py
   - reach task 공통 조립 구조 확인함.

4. env_cfg_common.py
   - command/observation/reward term 목록 확인함.

5. mdp/rewards.py
   - reward 실제 계산 확인함.

6. mdp/observations.py
   - observation 실제 계산 확인함.

7. assets/indy.py
   - USD asset과 actuator 설정 확인함.

8. learning/rsl_rl_cfg.py
   - PPO 학습 설정 확인함.
```

## 13. 한 줄 요약

- `env_cfg_common.py`는 공통 부품을 정의함.
- `reach_env_cfg.py`는 그 부품들을 조립함.
- `indy_wuji/env_cfg.py`는 이번 robot에 맞게 body/joint/asset을 주입함.
- `isaac_neuromeka/mdp/*.py`는 실제 tensor 계산을 함.
- `train.py`는 등록된 task 이름으로 env를 만들고 PPO 학습을 시작함.

## 14. MDP 공부 전 흐름 정리

현재 이해한 구조임.

```text
rl_task_env_cfg.py
-> 기본 RL 환경 cfg
-> sim, decimation, obs list 같은 공통 설정

env_cfg_common.py
-> command/action/observation/reward/event/termination 부품 정의

reach_env_cfg.py
-> reach task 공통 scene과 MDP 부품 조립

indy_wuji/env_cfg.py
-> Indy7 + Wuji asset에 맞게 robot/body/joint를 override

isaac_neuromeka/mdp/*.py
-> 실제 command/action/observation/reward/event 계산
```

- `reach_env_cfg.py`는 기본 환경 설정을 상속받음.
- `reach_env_cfg.py`는 `ObservationsCfg`, `ActionsCfg`, `RewardsCfg` 같은 부품을 직접 계산하지 않음.
- `reach_env_cfg.py`는 `env_cfg_common.py`에서 정의한 부품들을 가져와서 reach task로 조립함.
- `indy_wuji/env_cfg.py`는 조립된 `ReachEnvCfg`를 다시 상속받음.
- `indy_wuji/env_cfg.py`는 robot asset을 `INDY7_WUJI_RIGHT_CFG`로 바꿈.
- `indy_wuji/env_cfg.py`는 command/reward body를 `link6`로 바꿈.
- `indy_wuji/env_cfg.py`는 action joint와 observation joint를 `joint[0-5]`로 제한함.
- 그래서 현재 task는 `indy7_wuji_right_simplified.usd`에 맞는 arm reach baseline이 됨.

## 15. Command, Observation, Action, Reward 관계

큰 실행 흐름임.

```text
CommandManager
-> target ee_pose 생성

ObservationManager
-> 현재 상태 + target command + action history 구성

Policy / Actor
-> observation을 보고 action 출력

ActionManager
-> action을 joint target으로 변환

Simulation
-> robot state 업데이트

RewardManager
-> 업데이트된 robot state와 command target 비교
```

핵심 구분임.

- command는 목표임.
- observation은 policy가 보는 입력임.
- action은 policy가 내는 제어 입력임.
- reward는 action 자체와 command를 직접 비교해서 나오지 않음.
- reward는 action 적용 후 바뀐 robot state와 command target을 비교해서 나옴.

현재 command임.

```text
ee_pose
= target position + target orientation
= [x, y, z, qw, qx, qy, qz]
```

현재 action임.

```text
joint position action
= joint0~joint5 입력
= [a0, a1, a2, a3, a4, a5]
```

- command는 Cartesian pose임.
- action은 joint 제어 입력임.
- 두 값은 차원도 다르고 의미도 다름.
- 그래서 직접 비교 대상이 아님.

reward 비교 대상임.

```text
현재 link6 position/orientation
vs
command target position/orientation
```

action 관련 reward도 있음.

```text
action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
```

- 이 term은 command와 action을 비교하지 않음.
- 이전 action과 현재 action 차이를 봄.
- action이 너무 급하게 바뀌지 않게 하는 penalty임.
- 현재 action dim이 6이라 arm action 6개 기준으로 계산됨.

## 16. MDP 공부 순서

추천 순서임.

```text
1. mdp/commands.py
2. mdp/observations.py
3. mdp/actions.py
4. mdp/rewards.py
5. mdp/events.py
```

`mdp/commands.py`에서 볼 것임.

- `ee_pose` target이 어떻게 만들어지는지 봄.
- command tensor shape를 봄.
- command가 body frame인지 world frame인지 봄.
- debug marker가 어디서 나오는지 봄.

`mdp/observations.py`에서 볼 것임.

- policy가 실제로 무슨 tensor를 보는지 봄.
- `joint_pos`를 봄.
- `finite_joint_vel`을 봄.
- `generated_commands`를 봄.
- `action_history`를 봄.
- `asset_cfg.joint_ids`가 arm 6축으로 resolve된 뒤 tensor를 자르는 구조를 봄.

`mdp/actions.py`에서 볼 것임.

- policy output 6개가 실제 joint target으로 어떻게 바뀌는지 봄.
- `CustomJointPositionAction`을 봄.
- `scale=0.2` 의미를 봄.
- `use_default_offset=True` 의미를 봄.

`mdp/rewards.py`에서 볼 것임.

- command와 body state를 어떻게 비교하는지 봄.
- `end_effector_position_tracking_bounded`를 봄.
- `end_effector_orientation_tracking_distance_bounded`를 봄.
- `end_effector_speed`를 봄.
- `finite_joint_vel_l2`를 봄.
- `action_rate_l2`를 봄.

`mdp/events.py`에서 볼 것임.

- reset 때 scene과 robot state를 어떻게 초기화하는지 봄.
- 지금은 observation/action/reward를 먼저 본 뒤 봐도 됨.

최종 한 줄임.

- `env_cfg_common.py`는 mdp 함수를 등록함.
- `mdp/*.py`는 등록된 함수가 실제 tensor를 계산함.
- reward는 action과 command를 직접 비교하지 않고, action 이후 robot state와 command target을 비교함.

## 17. Indy-Wuji-Cube-Grasp 흐름

Cube grasp는 reach와 흐름이 조금 다름.

```text
grasp/indy_wuji/__init__.py
-> task id `Indy-Wuji-Cube-Grasp` 등록

grasp/cube_grasp_env_cfg.py
-> cube 포함 scene 구성

grasp/indy_wuji/env_cfg.py
-> Indy7 + Wuji asset 적용
-> controlled joint를 arm + thumb/index/middle로 설정

common/env_cfg_common.py
-> cube grasp observation/reward/action cfg 등록

mdp/observations.py
-> cube relative position / cube goal tensor 계산

mdp/rewards.py
-> palm/fingertip reach, lift, lifted object-goal reward 계산
```

- reach task는 `ee_pose` command를 씀.
- cube grasp task는 현재 active command가 없음.
- cube grasp는 cube root state를 observation/reward에서 직접 씀.
- cube grasp의 목표는 아직 command tracking이 아님.
- cube grasp의 현재 목표는 palm/fingertip 접근, cube lift, lifted 상태에서 cube goal 이동 흐름을 확인하는 것임.

## 18. Cube Grasp Action 흐름

현재 action 구조임.

```text
policy output 18D
-> ActionManager
-> JointPositionActionCfg
-> robot joint position target
```

controlled joints임.

```text
joint[0-5]
finger[1-3]_joint[1-4]
```

- arm은 6축임.
- finger1/thumb은 4축임.
- finger2/index는 4축임.
- finger3/middle은 4축임.
- finger4/ring은 현재 action에서 제외함.
- finger5/little은 현재 action에서 제외함.
- total action dim은 18임.

## 19. Cube Grasp Observation 흐름

현재 policy observation 구조임.

```text
joint_pos
-> controlled joints 18D

cube_pos
-> cube_pos_w - palm_link_pos_w
-> 3D

cube_in_fingertips
-> cube_pos_w - thumb_tip_pos_w
-> cube_pos_w - index_tip_pos_w
-> cube_pos_w - middle_tip_pos_w
-> cube_pos_w - ring_tip_pos_w
-> cube_pos_w - little_tip_pos_w
-> 15D

cube_to_goal
-> cube_goal_pos_w - cube_pos_w
-> 3D

action_history
-> previous action
-> 18D
```

총합임.

```text
18 + 3 + 15 + 3 + 18 = 57D
```

- `cube_pos`는 palm 위치 기준 상대 벡터임.
- `cube_pos`는 palm orientation까지 반영한 local frame은 아님.
- `cube_in_fingertips`도 world axis 기준 상대 벡터임.
- velocity observation은 아직 뺌.
- contact observation도 아직 뺌.

## 20. Cube Grasp Reward 흐름

현재 reward term임.

```text
finger_cube_reach
functional_hold
action_rate
```

`arm_cube_reach`, `cube_lift`, `cube_goal_tracking`은 구현돼 있지만 현재 active reward에서는 비활성화함.

`finger_cube_reach`임.

```text
distance_i = norm(fingertip_i_pos_w - cube_pos_w)
current_distance = weighted_average(distance_i, body_weights=(3.0, 1.0, 1.0))
progress = previous_best_distance - current_distance
raw = clamp(progress / 0.03, 0, 1)
weighted = raw * 0.3
previous_best_distance = min(previous_best_distance, current_distance)
```

`functional_hold`임.

```text
distance_i = norm(fingertip_i_pos_w - cube_pos_w)
distance_bonus_i = 1 - clamp(distance_i, 0, 0.18) / 0.18
weighted_distance_bonus = weighted_average(distance_bonus_i, body_weights=(3.0, 1.0, 1.0))
grasp_center = uniform_average(fingertip_i_pos_w)
center_bonus = 1 - clamp(norm(cube_pos_w - grasp_center), 0, 0.18) / 0.18
opposition = mean(clamp(-dot(unit_thumb_from_cube, unit_other_from_cube), 0, 1))
raw = 0.4 * center_bonus + 0.4 * weighted_distance_bonus + 0.2 * center_bonus * opposition
weighted = raw * 0.2
```

`cube_lift`임.

```text
height = cube_pos_w.z - 0.03
raw = clamp(height, 0, 0.08) / 0.08
weighted = raw * 0.05
```

`cube_goal_tracking`임.

```text
target = (0.55, -0.05, 0.12)
distance = norm(cube_pos_w - target)
raw = 1 - clamp(distance, 0, 0.6) / 0.6
weighted = raw * 0.2, only when cube_pos_w.z >= 0.08
```

`action_rate`임.

```text
raw = action rate L2
weighted = raw * -0.0003
```

- `finger_cube_reach`는 controllable thumb/index/middle만 봄.
- `finger_cube_reach`는 현재 거리 reward가 아니라 best-so-far progress reward임.
- 첫 step/reset 직후에는 best distance를 현재 거리로 초기화해서 raw reward가 0임.
- 실제 fingertip-cube 거리는 `Metrics/cube/*`로 따로 봄.
- `functional_hold`는 functional grasp / Dexterous Pre-grasp 쪽 hold/cage 개념을 cube task에 맞춘 term임.
- `functional_hold`는 cube가 thumb/index/middle 사이에 들어오는지를 봄.
- `functional_hold`는 thumb weight를 distance bonus에만 쓰고, grasp center는 uniform 평균으로 봄.
- `functional_hold`는 완전한 곱셈 gate가 아니라 additive shaping 구조임.
- random policy에서는 `functional_hold`가 0이어도 정상임.
- ring/little은 observation/metric 보조로만 남김.
- reach raw reward는 각 term 기준 최대 1임.
- active positive weighted max는 현재 `0.3 + 0.2 = 0.5`임.
- 이 reward는 아직 contact 기반 grasp success reward가 아님.
- 이 reward는 functional grasp hold/cage를 중심에 두고, DexPoint식 contact/lift gate와 TriFinger/SimToolReal식 object goal 흐름으로 확장하기 위한 shaping reward임.
- contact reward는 다음 단계임.

## 21. Cube Grasp TensorBoard Error

- reward raw 값만 보면 실제 거리가 몇 m인지 바로 안 보임.
- 그래서 `CustomRewardManager`에서 cube 거리 metric을 따로 기록함.
- metric은 reward에 더하지 않음.
- metric은 action/observation shape도 바꾸지 않음.

TensorBoard metric 이름임.

```text
Metrics/cube/palm_distance
Metrics/cube/thumb_distance
Metrics/cube/index_distance
Metrics/cube/middle_distance
Metrics/cube/ring_distance
Metrics/cube/little_distance
Metrics/cube/finger_mean_distance
Metrics/cube/non_thumb_mean_distance
Metrics/cube/finger_weighted_mean_distance
```

계산 흐름임.

```text
body_pos_w = robot body world position
cube_pos_w = cube root world position
distance = norm(body_pos_w - cube_pos_w)
```

- `palm_distance`가 내려가면 arm이 cube 쪽으로 가는 중임.
- `finger_mean_distance`가 내려가면 five fingertip이 cube 쪽으로 가는 중임.
- `non_thumb_mean_distance`는 index/middle/ring/little 쪽 접근 정도임.
- `finger_weighted_mean_distance`는 thumb을 크게 본 five-fingertip 거리임.
- 이 metric은 episode reset 시점에 TensorBoard로 평균 기록됨.
- 새 로그에서 확인해야 함.

## 22. Cube Grasp 다음 단계

- 현재 cube reset randomization 기준으로 학습 추세를 봄.
- `Episode_Reward_Raw/finger_cube_reach`는 최단거리 갱신량이 생기는지 봄.
- `Metrics/cube/finger_mean_distance`가 내려가는지 봄.
- action_rate가 폭주하지 않는지 봄.
- 접근 여부는 reward raw보다 `Metrics/cube/*` 거리 metric으로 판단함.
- 이후 contact group reward를 추가함.
- 이후 contact-gated lift/goal reward로 바꿈.
- 이후 cube randomization을 넓힘.

## 23. Cube Grasp PhysX Buffer

- 4096 env long run에서 PhysX patch buffer overflow가 발생함.
- 에러 메시지는 `Patch buffer overflow detected`임.
- 요구 patch count는 약 `171k`까지 올라감.
- cube grasp cfg에서 `gpu_max_rigid_patch_count`를 `2**19`로 올림.
- `2**18`은 4096 env resume run의 약 `263k` patch 요구치에 살짝 부족했음.
- 이 값은 contact patch buffer 여유를 늘리는 설정임.
- cube, hand collider, 병렬 env 수가 많아지면 patch buffer가 부족해질 수 있음.
- 이 설정은 reward 구조가 아니라 PhysX GPU contact buffer 설정임.

## 12. Cube Grasp reward 흐름 (2026-07-12 기준)

### 전체 연결

```text
env_cfg_common.py                          rewards.py                      managers.py
─────────────────                          ──────────                      ───────────
CAGE_BODIES  ──────────┐
  finger1_tip_link     │
  finger2_tip_link     ├──> cage_points()  ──> _cage_sdf() ──> _box_signed_distance()
  finger2_link3        │         │                                    │
  finger3_tip_link     │         │                                    │
  finger3_link3        │         ├─> object_in_finger_cage()   [hold] │
                       │         ├─> ObjectCageProgressReward  [reach]│
                       └─────────┴─> object_lift_in_cage()      [lift]│
                                                                       │
palm_link ──────────────────────> palm_facing_object()          [palm]│
                                                                       │
                                   _cage_body_names ────────────> _compute_cube_distance_metrics()
                                   _palm_normal_b                      (reward와 동일한 점을 재구성)
```

- **핵심: reward와 metric이 같은 가상점을 쓴다.** `managers.py`의 `_cage_body_names`가 `CAGE_BODIES`와 다르면 metric이 reward와 다른 것을 측정하게 됨.

### 가상점 생성 (`cage_points`)

```text
body_names = [thumb_tip, *opposing]        preserve_order=True 필수
                │           │
             기준점      대향 body N개

각 대향 body마다 엄지끝에서 선분을 긋고, 등간격 3점을 찍음
비율 = [0.25, 0.50, 0.75]  (양 끝점 제외)

현재: 대향 body 4개 (검지끝, 검지중간, 중지끝, 중지중간) x 3점 = 12점
```

- `preserve_order=False`(기본값)이면 `find_bodies`가 body_ids를 **정렬**해서 엄지가 기준점 자리에서 밀려남.
- 논문은 엄지↔중지만 써서 6점임. 우리가 12점인 이유는 `r_grasp`가 없어서 검지가 자유로워지기 때문임.

### SDF (`_box_signed_distance`)

```text
점을 큐브 로컬 좌표로 변환  ->  q = |p_local| - half_extent
sdf = norm(clamp(q, min=0)) + clamp(max(q.x, q.y, q.z), max=0)

음수 = 큐브 내부
```

- 큐브는 박스라 **해석식**임. CAD나 사전계산 SDF 불필요함.
- 젓가락도 원기둥이라 해석식으로 가능함. 임의 메시가 필요해지면 `warp`의 `wp.Mesh`를 씀.

### 4개 reward 항의 역할

```text
finger_cage_reach   가상점 12개의 표면까지 SDF 평균의 "차분"
                    -> 파지 간극을 큐브 위로 끌어옴
                    -> mode="previous" + clamp(min=-1) + reset() seeding

finger_cage_hold    가상점이 큐브 "내부"로 파고든 깊이
                    -> r = clamp((sphere_radius - sdf) / (sphere_radius + depth_max), 0, 1) 의 평균
                    -> 오므리면 점들이 큐브 안으로 들어감. "오므리기"가 직접 보상됨

palm_facing         dot(손바닥 법선(로컬 +x), 단위벡터(큐브 - 손바닥))
                    -> cage 항들은 손 방향을 못 잡음 (선분이 손 방향과 무관하게 큐브를 관통)
                    -> 손가락은 손바닥 쪽으로 굽으므로, 손바닥 뒤 물체는 못 감쌈

cube_lift           cage_gate x clamp(최하 모서리 높이 / lift_height, 0, 1)
                    -> 중심 높이를 쓰면 "기울이기"로 해킹당함
                    -> 최하 모서리를 쓰면 진짜로 띄워야만 보상
```

### 가중치 순서

```text
cube_lift (3.0)  >>  finger_cage_hold (1.0)  >  palm_facing (0.5)  >  finger_cage_reach (0.3)
```

- 논문의 `r_T >> r_orient >> r_hold >> r_reach`를 따름.
- **쉬운 앞 단계 하위과제에 큰 보상을 주면 정책이 거기 눌러앉음** (논문 Sec. IV-C 명시).
- 실제로 `reach(0.3) > hold(0.2)`인 역순이었을 때 국소최적에 갇혔음.

### TensorBoard metric 흐름 (`managers.py`)

```text
compute()  매 step  ->  _compute_cube_distance_metrics()
                        ├─ sums  += metric * dt      (에피소드 평균용)
                        ├─ last   = metric           (정착 자세용)   ★
                        ├─ min    = minimum(...)
                        └─ max    = maximum(...)

reset()             ->  Metrics/cube/<name>        = sums / max_episode_length_s   (평균, 신뢰 금지)
                        Metrics/cube_final/<name>  = last                          (★ 이것을 볼 것)
                        Metrics/cube_min/<name>    = min
                        Metrics/cube_max/<name>    = max
                        그리고 _cube_init_pos를 새 에피소드의 큐브 위치로 갱신
```

- `Metrics/cube/*`(평균)는 **앞 4 step(이동)이 77%를 지배**하므로 성능 지표가 아님.
- `reset()` 시점에는 event(`reset_cube_position`)가 이미 실행된 뒤라 새 에피소드의 큐브 위치가 들어감.

---

# 24. Action 흐름 — 정책 출력이 PhysX까지 가는 전 경로 (2026-07-13)

**2026-07-13에 여기서 정책 발산이 났음.** `clip_actions`가 설정 안 돼 있었고 `scale`이 너무 작았음.

## 전체 경로

```
정책 신경망 (rsl_rl)
   │  a = N(μ, σ).sample()         ← 상한 없음. MLP 마지막 층이 nn.Linear
   ▼
┌───────────────────────────────────────────────────────────────────────┐
│ ①  RslRlVecEnvWrapper.step()                                          │
│    isaaclab_rl/rsl_rl/vecenv_wrapper.py:151-154                       │
│      if self.clip_actions is not None:                                │
│          actions = torch.clamp(actions, -clip_actions, +clip_actions) │
└───────────────────────────────────────────────────────────────────────┘
   │  a ∈ [-1, +1]
   ▼
┌───────────────────────────────────────────────────────────────────────┐
│ ②  ManagerBasedRLEnv.step()                                           │
│    isaaclab/envs/manager_based_rl_env.py:173, 182-188                 │
│      self.action_manager.process_action(action)    ← step당 1번만!    │
│      for _ in range(self.cfg.decimation):          ← 물리 step 반복   │
│          self.action_manager.apply_action()        ← 같은 목표 재전송 │
│          self.scene.write_data_to_sim()                               │
│          self.sim.step(render=False)                                  │
└───────────────────────────────────────────────────────────────────────┘
   ▼
┌───────────────────────────────────────────────────────────────────────┐
│ ③  JointAction.process_actions()          ← scale이 여기서 적용됨     │
│    isaaclab/envs/mdp/actions/joint_actions.py:169-179                 │
│      self._raw_actions[:] = actions                                   │
│      self._processed_actions = self._raw_actions * self._scale        │
│                                 + self._offset     ← default_joint_pos│
│      if self.cfg.clip is not None:                 ← 두 번째 clip     │
│          self._processed_actions = torch.clamp(...)   (우리는 미사용) │
└───────────────────────────────────────────────────────────────────────┘
   │  관절 목표 [rad]
   ▼
④  JointPositionAction.apply_actions()            joint_actions.py:197-199
      self._asset.set_joint_position_target(processed_actions, joint_ids=...)
   ▼
⑤  Articulation.set_joint_position_target()       articulation.py:1079   (내부 버퍼)
   ▼
⑥  Articulation.write_data_to_sim()               articulation.py:218    (PhysX GPU 전송)
   ▼
⑦  PhysX 임플리시트 PD 액추에이터 (물리 60 Hz)
      τ = stiffness x (목표 - 실제) - damping x 속도
        = 100 x (목표 - 실제) - 20 x 속도          ← indy.py arm 액추에이터
```

## 핵심 구조: `process_action`은 1번, `apply_action`은 decimation번

```python
# manager_based_rl_env.py:173, 182-188
self.action_manager.process_action(action)   # step당 딱 1번.  scale 적용
for _ in range(self.cfg.decimation):         # decimation번
    self.action_manager.apply_action()       # 같은 목표를 반복해서 다시 씀
    self.scene.write_data_to_sim()
    self.sim.step(render=False)
```
**정책이 준 관절 목표를 `decimation`번 동안 고정한 채 PD가 밀어붙임.**
`decimation=24`였을 땐 같은 목표를 24번(=0.4초) 밀어붙였음 -> **0.4초간 눈감고 운전.**
바닥에 닿아도 0.4초가 지나야 알아챔 -> 모든 접촉이 슬램이 됨.

## clip이 두 군데인 이유

```
정책 -> a -> [① clip_actions] -> x scale + default -> [② JointActionCfg.clip] -> 로봇
             무차원 [-1,1]                             rad 단위
```
| | 자르는 것 | 단위 | 설정 위치 |
|---|---|---|---|
| ① `clip_actions` | 정책 출력 | 무차원 | `rsl_rl_cfg.py` |
| ② `JointActionCfg.clip` | 관절 목표 | **rad** | action term |

**우리는 ①만 씀.** `|a| <= 1`이 보장되면 목표가 자동으로 `default ± scale`(= ±1.0 rad) 안에 들어오므로
②가 불필요함. **2026-07-13 이전엔 ①도 ②도 없었음** -> `|a|=9.66` -> 목표가 `default ± 1.93 rad`까지 날아감.

## 우리 코드에서 값이 정해지는 곳

| 값 | 파일 | 줄 |
|---|---|---|
| `clip_actions = 1.0` | `tasks/manipulation/grasp/indy_wuji/learning/rsl_rl_cfg.py` | 21 |
| ↳ wrapper로 전달 | `scripts/rsl_rl/train.py` (play.py 동일) | 203 |
| `scale = 1.0` | `tasks/manipulation/grasp/indy_wuji/env_cfg.py` | 64 |
| `decimation = 2` | `tasks/manipulation/grasp/cube_grasp_env_cfg.py` | 102 |
| `stiffness / damping / effort_limit` | `assets/indy.py` | 183~211 |

```python
# scripts/rsl_rl/train.py:203  <- clip_actions가 여기서 wrapper로 들어감
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                                           └── ReachPPORunnerCfg.clip_actions
```

---

# 25. 정책은 왜 가우시안 분포인가 (2026-07-13)

## MLP는 "행동"이 아니라 "행동의 평균"을 뱉음

```python
# rsl_rl/modules/distribution.py:169-179
def update(self, mlp_output):
    mean = mlp_output                       # ← MLP 출력이 그대로 평균 mu
    std  = torch.exp(self.log_std_param)    # ← 별도의 학습 가능 파라미터 sigma
    self._distribution = Normal(mean, std)  # ← 가우시안 N(mu, sigma)

def sample(self):
    return self._distribution.sample()      # ← 실제 action은 여기서 뽑음
```
```
obs -> MLP [64,64] -> mu = [18개]     ← 마지막 층이 nn.Linear. tanh/sigmoid 없음
                                         => 수학적으로 상한이 없음!
       log_std_param -> sigma = [18개]   ← init_noise_std = 1.0
       a ~ N(mu, sigma)
```

## 왜 랜덤하게 뽑나
1. **탐험(exploration).** 항상 같은 행동만 하면 더 좋은 행동을 영원히 발견 못 함
2. **PPO가 확률을 요구함.** 핵심 수식 `ratio = pi_new(a|s) / pi_old(a|s)`.
   결정론적 정책은 확률이 0 또는 1이라 이 비율이 정의되지 않음

## 학습 때와 play 때가 다름
```python
학습 (train.py)  a = distribution.sample()             # mu에서 sigma만큼 흔들어 뽑음
재생 (play.py)   a = deterministic_output(mlp_output)  # = mu. 흔들지 않음
```
**중요:** `play.py --print_action`으로 본 `|a| = 1.5`는 **sigma 노이즈가 아니라 mu 자체임.**
**신경망이 학습해서 "평균 1.5를 뱉도록" 수렴한 것.** 노이즈가 아니라 정책 자체가 발산한 것.

`init_noise_std=1.0`이므로 학습 중엔 sigma=1.0. mu=1.5에 sigma=1.0이면 샘플이 흔히 ±3까지 감.
그래서 `|Δa|`가 9.66까지 나왔음.

## 2026-07-13에 일어난 일
```
clip 없음 (①도 ②도) + scale=0.2가 너무 작아서 |a|=1로는 큐브(55cm 아래)에 못 닿음
   ↓
정책이 보상을 얻으려면 mu를 키워야 함  ->  아무도 안 막음  ->  mu가 1.5까지 커짐
   ↓
학습 중엔 sigma=1.0이 더해져 샘플이 ±3, |Δa| 최대 9.66
   ↓
관절 목표가 한 step에 1.93 rad(110°) 점프  ->  팔이 velocity_limit로 왕복  ->  큐브 67cm 날림
```

## 수정
```python
clip_actions = 1.0   # ① 활성화. |a| <= 1 보장
scale = 1.0          # |a|=1 이 1.0 rad(57°) -> 큐브 도달 가능
```
**둘을 반드시 같이 바꿔야 함.** `clip`만 걸면 `|a|<=1` x `scale 0.2` = 관절이 11°밖에 못 움직여
**큐브에 영영 못 감.** `scale`만 키우면 여전히 상한이 없어 또 발산.

**결과: `|Δa|` 평균 0.04, 최대 0.90 (clip 상한 2.0 이내). 발산 원천 차단됨.**
