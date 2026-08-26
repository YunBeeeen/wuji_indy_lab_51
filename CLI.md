# CLI.md

- 이 문서는 학습, play, TensorBoard, git 작업에 쓰는 명령어를 모아둔 CLI 참고 문서임.

## 기본 전제

- 작업 위치는 `~/wuji_indy_lab_51/nrmk_isaaclab_wuji`임.
- conda env는 `env_isaaclab`임.
- 기본 task는 `Indy-Wuji-Reach`임.
- 현재 action dim은 6임.
- 현재 policy observation dim은 15임.
- cube grasp task는 `Indy-Wuji-Cube-Grasp` 하나만 사용함.
- `Indy-Wuji-Cube-Grasp-Easy`는 이전 실험 이름이라 현재 명령에 쓰지 않음.
- cube grasp action dim은 18임.
- cube grasp policy observation dim은 57임.
- 현재 tracking body는 `link6`임.
- 아래 명령은 변수 없이 바로 복붙 실행하는 형태임.

## Git 상태 확인

- 작업 전 확인함.

```bash
cd ~/wuji_indy_lab_51
git status --short
```

## Repo 이동 / Env 활성화

- 터미널 처음 열었을 때 실행함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
```

## Checkpoint 목록 확인

- 생성된 checkpoint 확인함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
find logs/rsl_rl/indy_wuji_reach -name 'model_*.pt' | sort | tail
```

## Run 폴더 확인

- 최신 run 폴더 확인함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
ls -td logs/rsl_rl/indy_wuji_reach/*
```

## 학습 Smoke Test

- 가장 작은 학습 확인용임.
- 코드 수정 후 가장 먼저 실행함.
- env 생성, asset 로드, action/observation shape 확인용임.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/train.py \
  --task Indy-Wuji-Cube-Grasp \
  --headless \
  --num_envs 1 \
  --max_iterations 1
```

## 학습 중간 테스트

- 빠른 안정성 확인용임.
- reward/error 추세를 대충 봄.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/train.py \
  --task taskname \
  --headless \
  --num_envs 128 \
  --max_iterations 20
```


## 긴 학습 Run

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/train.py \
  --task Indy-Wuji-Cube-Grasp \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000
```

## TensorBoard

- 학습 로그 확인용임.
- 학습 터미널은 그대로 두고 새 터미널에서 실행함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
tensorboard --logdir logs/rsl_rl/indy_wuji_box_transport --port 6006 --reload_interval 5
```

- 브라우저에서 봄.

```text
http://localhost:6006
```

- `6006`이 이미 사용 중이면 다른 포트 사용함.

```bash
tensorboard --logdir logs/rsl_rl/indy_wuji_chopsticks_grasp --port 6007 --reload_interval 5
```

- cube grasp 로그를 볼 때는 logdir를 바꿈.

```bash
tensorboard --logdir logs/rsl_rl/hand_setting --port 6006 --reload_interval 5
```

## GUI 학습 확인

- Isaac Sim GUI에서 robot 움직임 확인용임.
- `--headless` 옵션 없음.
- 학습 성능보다 asset, command marker, 움직임 확인용임.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/train.py \
  --task Indy-Wuji-Cube-Grasp \
  --num_envs 1 \
  --max_iterations 10
```

## Checkpoint Play

- 학습된 policy 재생용임.
- 최신 run을 자동으로 찾아서 재생함.
- 같은 experiment 안에 과거 smoke/hard/easy run이 섞여 있으면 잘못된 run을 잡을 수 있음.
- 중요한 확인은 `ls -td logs/rsl_rl/indy_wuji_cube_grasp/20*`로 run 폴더를 눈으로 보고 직접 지정함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/play.py \
  --task Indy-Wuji-Box-Transport \
  --num_envs 1 \
  --load_run "$(basename "$(ls -td logs/rsl_rl/indy_wuji_box_transport/20* | head -n 1)")"
```

## 성공 후 유지 상태 확인 Play

- 성공 상태를 기본 `0.5초`보다 오래 관찰할 때 사용함.
- `success` termination은 등록된 채로 두고 `hold_steps`만 매우 크게 설정함.
- 기존 `time_out` 8초와 `cube_dropped` 종료는 그대로 동작함.
- `env.terminations.success=null`만 사용하면 `transport_success` reward가 참조할
  termination이 사라져 환경 생성 중 오류가 발생함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/play.py \
  --task Indy-Wuji-Cube-Grasp \
  --num_envs 1 \
  --load_run "$(basename "$(ls -td logs/rsl_rl/indy_wuji_cube_grasp/20* | head -n 1)")" \
  env.terminations.success.params.hold_steps=1000000
```

- 중요한 평가는 자동 최신 run 대신 확인한 run 폴더명을 직접 넣음.

```bash
python scripts/rsl_rl/play.py \
  --task Indy-Wuji-Cube-Grasp \
  --num_envs 1 \
  --load_run 2026-07-15_12-21-00 \
  env.terminations.success.params.hold_steps=1000000
```

## Checkpoint 직접 지정 Play

- checkpoint 파일을 직접 지정함.
- 최신 run의 최신 checkpoint 파일을 자동으로 찾아서 지정함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/play.py \
  --task Indy-Wuji-Cube-Grasp \
  --num_envs 1 \
  --checkpoint "$(find "$(ls -td logs/rsl_rl/indy_wuji_cube_grasp/20* | head -n 1)" -name 'model_*.pt' | sort -V | tail -n 1)"
```

## Headless Play

- GUI 없이 policy load만 확인함.
- checkpoint가 깨졌는지 확인할 때 씀.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/play.py \
  --task Indy-Wuji-Reach \
  --headless \
  --num_envs 1 \
  --load_run "$(basename "$(ls -td logs/rsl_rl/indy_wuji_reach/20* | head -n 1)")"
```

## Video Play

- 짧은 영상 저장 확인용임.
- 영상은 `logs/rsl_rl/indy_wuji_reach/<RUN>/videos/play` 쪽에 저장됨.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/play.py \
  --task Indy-Wuji-Reach \
  --headless \
  --num_envs 1 \
  --load_run "$(basename "$(ls -td logs/rsl_rl/indy_wuji_reach/20* | head -n 1)")" \
  --video \
  --video_length 200
```

## Resume Training

- 기존 run에서 이어서 학습함.
- 아래 예시는 `model_99.pt`에서 이어서 100 iteration 더 학습함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/train.py \
  --task hand_real \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000 \
  --resume \
  --load_run 2026-08-18_21-59-26
```

## 폐기된 Cube Grasp 분기

- `Indy-Wuji-Cube-Grasp-Easy`, Hard resume 구조는 현재 쓰지 않음.
- 2026-07-14 기준 새 학습/play/smoke test는 전부 `Indy-Wuji-Cube-Grasp`로 실행함.
- 과거 checkpoint를 볼 때만 예전 run 이름을 참고함.

## Cube Grasp 확인

- 2026-07-14 기준 `Indy-Wuji-Cube-Grasp` 하나만 사용함.
- 현재 main task는 받침면 `z=0.40` 위에 cube를 놓음.
- cube 중심은 `(0.692, -0.369, 0.430)`임.
- probe 기준 reset `palm_facing=0.987`, zero action 30 step 뒤 `0.997`이라 현재 설정에서는 `palm_facing` reward를 꺼도 되는 배치임.
- smoke test는 headless로 봄.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/train.py \
  --task Indy-Wuji-Cube-Grasp \
  --headless \
  --num_envs 1 \
  --max_iterations 1
```

- 높이/위치를 눈으로 볼 때는 GUI로 봄.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/train.py \
  --task Indy-Wuji-Cube-Grasp \
  --num_envs 1 \
  --max_iterations 1
```

## Cube Grasp Action 확인

- policy가 이상한 action을 내는지 확인함.
- `raw`는 policy network 출력임.
- `applied`는 `clip_actions=1.0` 적용 후 실제 env에 들어간 action임.
- `target`은 `default_joint_pos + scale * applied`로 만들어진 관절 목표임.
- `actual`은 현재 실제 관절각임.
- `err`가 작고 action이 크면 policy/학습 문제임.
- `err`가 크면 물리/PD/접촉/decimation 문제임.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/play.py \
  --task Indy-Wuji-Cube-Grasp \
  --num_envs 1 \
  --load_run "$(basename "$(ls -td logs/rsl_rl/indy_wuji_cube_grasp/20* | head -n 1)")" \
  --print_action \
  --print_action_interval 1 \
  --print_action_detail
```

## Cube Grasp Contact/Lift 확인

- policy 없이 scripted action으로 확인함.
- `GOOD_CONTACT thumb+middle`이 `True`인지 봄.
- `max_clearance(m)`가 `0.005` 이상인지 봄.
- contact가 `True`인데 lift가 `False`면 잡는 게 아니라 누르는 것임.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/debug/check_cube_contact_lift.py \
  --task Indy-Wuji-Cube-Grasp \
  --headless \
  --num-envs 1 \
  --settle-steps 30 \
  --close-steps 60 \
  --lift-steps 30
```

- arm 단일축 lift 후보를 같이 훑음.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/debug/check_cube_contact_lift.py \
  --task Indy-Wuji-Cube-Grasp \
  --headless \
  --num-envs 1 \
  --settle-steps 30 \
  --close-steps 60 \
  --lift-steps 30 \
  --sweep-lift
```

- cube는 고정하고 thumb/index/middle close 값을 훑음.
- 기본 후보는 각 finger별 `0.0`, `0.5`, `1.0` 조합임.
- `--contact-mode`는 `thumb_middle`, `thumb_index`, `thumb_any`, `tripod` 중 선택함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/debug/check_cube_contact_lift.py \
  --task Indy-Wuji-Cube-Grasp \
  --headless \
  --num-envs 1 \
  --settle-steps 30 \
  --close-steps 60 \
  --lift-steps 30 \
  --sweep-fingers \
  --contact-mode thumb_middle
```

- 특정 손가락 조합만 확인함.
- 아래 예시는 thumb/middle만 닫고 index는 열어둠.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/debug/check_cube_contact_lift.py \
  --task Indy-Wuji-Cube-Grasp \
  --headless \
  --num-envs 1 \
  --settle-steps 30 \
  --close-steps 60 \
  --lift-steps 30 \
  --finger-action 1 0 1 \
  --contact-mode thumb_middle
```

## GUI Play

- 학습된 policy를 GUI에서 봄.
- `--headless` 옵션 없음.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/play.py \
  --task Indy-Wuji-Reach \
  --num_envs 1 \
  --load_run "$(basename "$(ls -td logs/rsl_rl/indy_wuji_reach/20* | head -n 1)")"
```

## 최신 Run 확인

- 최신 run 폴더를 눈으로 확인할 때만 사용함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
ls -td logs/rsl_rl/indy_wuji_reach/*
```

python scripts/rsl_rl/play.py   --task Indy-Wuji-Cube-Grasp   --num_envs 1   --load_run "$(basename "$(ls -td logs/rsl_rl/indy_wuji_cube_grasp/20* | head -n 1)")" --print_action   --print_action_detail   --print_action_interval 1



## 로그에서 볼 것

- `Mean reward` 봄.
- `Metrics/ee_pose/position_error` 봄.
- `Episode_Reward/end_effector_position_tracking` 봄.
- `Episode_Reward_Raw/end_effector_position_tracking` 봄.
- `Episode_Reward/action_rate` 봄.
- `Episode_Reward_Raw/action_rate` 봄.
- `Mean action std` 봄.
- `Episode_Termination/time_out` 봄.
- NaN 여부 봄.
- PhysX error 여부 봄.
- GUI에서 발산/떨림 여부 봄.

## 정상으로 보는 기준

- env 생성됨.
- robot asset 로드됨.
- action shape 6 나옴.
- policy observation shape 15 나옴.
- PPO iteration 완료됨.
- checkpoint 생성됨.
- play에서 checkpoint load됨.
- actor input 15 유지됨.
- action output 6 유지됨.

## 현재 자주 쓰는 조합

- 수정 직후 확인은 `1 env / 1 iter`임.
- 짧은 학습 확인은 `128 env / 20 iter`임.
- 중간 학습 확인은 `512 env / 500 iter`임.
- 긴 학습은 `4096 env / 50000 iter`임.
- GUI 확인은 `1 env / 1 iter`임.
- play 확인은 최신 run의 마지막 checkpoint 기준임.

## 옛 체크포인트 play (코드가 앞서갔을 때)

- 차원(obs/action)이 같으면 현재 코드에서 그냥 play 가능함. 되돌리기 불필요.
- 차원이 바뀐 뒤에는 worktree로 그 런의 커밋을 옆에 펴서 play함:

```bash
git worktree add ~/wuji_play_old <그 런을 학습시킨 커밋>
ln -s ~/wuji_indy_lab_51/nrmk_isaaclab_wuji/logs ~/wuji_play_old/nrmk_isaaclab_wuji/logs
cd ~/wuji_play_old/nrmk_isaaclab_wuji && python scripts/rsl_rl/play.py --task ... --load_run ...
# 정리: git worktree remove ~/wuji_play_old
```

- 전제: 런 시작 전 커밋 = 그 런의 코드 스냅샷. 런-커밋 대응은 worklog에 기록함.

## Box-Transport (랜덤 직육면체) 학습/확인

```bash
# 학습 (fresh). 태스크 이름 오타 주의: Transport (Trasnport 아님)
python scripts/rsl_rl/train.py --task Indy-Wuji-Box-Transport --headless --num_envs 4096 --max_iterations 50000

# env별 치수 검증 프로브 (버퍼 = USD scale = 정착 높이 일치 확인)
python scripts/debug/box_dims_probe.py

# TensorBoard (로그 폴더가 큐브 태스크와 분리됨)
tensorboard --logdir logs/rsl_rl/indy_wuji_box_transport --port 6006 --reload_interval 5
```

## Chopsticks A1 (한 막대 Functional Grasp) 검증

- `Cube-Grasp`, `Box-Transport`와 별도 태스크/로그임.
- 현재 action 18D, observation 63D이며 기존 checkpoint와 호환되지 않음. fresh run만 사용함.
- 먼저 probe에서 observation diff가 전부 0인지 확인함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/debug/chopstick_functional_probe.py \
  --num_envs 1 \
  --steps 1 \
  --device cuda:0
```

- 1-env/1-iteration smoke test:

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/train.py \
  --task Indy-Wuji-Chopsticks-Grasp \
  --headless \
  --num_envs 1 \
  --max_iterations 1
```

- 기능 보상 추세를 보는 짧은 fresh 학습:

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/train.py \
  --task Indy-Wuji-Chopsticks-Grasp \
  --headless \
  --num_envs 128 \
  --max_iterations 20
```

- probe와 smoke 통과 후 fresh 학습:

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab
python scripts/rsl_rl/train.py \
  --task Indy-Wuji-Chopsticks-Grasp \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000
```

- TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/indy_wuji_chopsticks_grasp --port 6008 --reload_interval 5
```

- 확인한 run을 명시해서 GUI play:

```bash
python scripts/rsl_rl/play.py \
  --task Indy-Wuji-Chopsticks-Grasp \
  --num_envs 1 \
  --load_run <확인한_RUN_폴더명>
```

## Wuji hand-only two-stick functional pre-grasp

`hand_grasp`는 수동으로 만든 valley 근처 자세에서 시작해 STATE B 접촉 안정화만 학습함.
reset은 `logs/debug/hand_grasp_keyboard/2026-07-28_10-43-39/pose_002.json`의
관절 actual/target과 두 stick palm-local pose를 복원함. reset 뒤 Stick1/2는 둘 다 dynamic임.
기존 checkpoint는 reward/reset이 다르므로 resume하지 않고 fresh로 실행함.

### 손가락 20관절 키보드 조정

기존 CEM candidate는 GUI에서 Stick2가 엄지–검지 valley에 들어가지 않은 자세로 확인되어
더 이상 올바른 학습 seed로 간주하지 않음. 아래 도구는 정책을 로드하지 않고 열린 손에서
시작함. 두 stick은 이전 spawn보다 손가락 방향(world `+x`)으로 2 cm 올리고, 노란 Stick1은
spawn pose에 고정하며 초록 Stick2만 dynamic으로 두어 손가락으로 valley에 이동시킬 수 있음.

```bash
python scripts/debug/hand_grasp_keyboard.py --task hand_grasp
```

- 기본값: `--stick1-mode fixed --stick-forward-offset 0.020`
- Stick1도 움직이려면 `--stick1-mode dynamic`
- Stick1을 치우려면 `--stick1-mode park` 또는 기존 호환 옵션 `--park-stick1`
- 손가락 방향 위치는 `--stick-forward-offset`, palm 위 높이는
  `--stick-height-offset`으로 미터 단위 조정

현재 학습 reset을 두 stick dynamic 상태로 그대로 육안 확인:

```bash
python scripts/debug/hand_grasp_keyboard.py \
  --task hand_grasp \
  --start-pose pregrasp \
  --stick1-mode dynamic
```

저장된 `pose_017`에서 바로 이어서 키보드 미세 조정:

```bash
python scripts/debug/hand_grasp_keyboard.py \
  --task hand_grasp \
  --stick1-mode dynamic \
  --load-pose logs/debug/hand_grasp_keyboard/2026-07-28_10-57-45/pose_017.json
```

`--load-pose`는 저장된 관절 actual/target과 두 stick palm-local pose를 모두 복원함.
`Backspace`도 같은 저장 자세로 돌아가며 새 저장 파일은 새로운 timestamp 폴더에 생성되어
원본 JSON을 덮어쓰지 않음. load 모드에서는 spawn offset 옵션을 적용하지 않음.

수동 완성 후보 `pose_017`까지 단계적으로 joint-space IK replay:

```bash
python scripts/debug/hand_grasp_ik_replay.py \
  --task hand_grasp
```

현재 엄지-loaded reset에서 시작함. 검지→중지→약지→새끼 각각에 대해
`joint4 말단 close → joint3 close → joint1/2 배치 → joint3/4를 pose_017까지 release` 순서를
사용하고 엄지는 마지막에 최종각으로 조금 펴며, 이후 240 physics step 유지함.
두 stick은 전 구간 dynamic이며 task termination을 거치지 않고 PhysX를 직접 step하므로
중간에 자동 reset되지 않음. 결과는
`logs/debug/hand_grasp_ik_replay/<timestamp>/result.json`에 관절 actual/target,
두 stick palm-local pose, 7개 contact force와 활성 sensor 수로 저장됨.

- `1~5`: finger 선택
- `Q/W/E/R`: joint 1~4 선택
- `←/A`, `→/D`: 선택 관절 감소/증가
- `Z/X`: 관절 증분 절반/두 배
- `O/P`: 열린 자세/현재 학습 pre-grasp target 비교
- `Backspace`: scene reset
- `Space`: physics 일시정지/재생
- `V`: 전체 관절각 출력
- `T`: 20개 실제 관절각과 두 stick의 실시간 palm-local 위치 출력 켜기/끄기
- `S`: 관절 target/actual과 두 stick의 palm-local pose를 JSON으로 저장
- `Esc`: 저장 후 종료

저장 위치는 `logs/debug/hand_grasp_keyboard/<timestamp>/pose_*.json`임.
열린 시작 자세와 `O` 자세는 `finger1_joint2`를 soft lower limit
`-0.1659 rad`에 두어 엄지를 완전히 편 상태로 시작함.
20개 실제 관절각과 두 stick 위치는 기본 `5 Hz`로 출력함. 관절각은 finger별
`f1~f5=(j1,j2,j3,j4)` rad 형식이고, stick은 palm-local `xyz`, 두 stick의 `delta`,
장축 `y`를 제외한 횡단 간격을 mm 단위로 표시함. 출력률은
`--stick-print-hz 2`처럼 바꾸고 `0`이면 시작 시 비활성화함.

```bash
cd ~/wuji_indy_lab_51/nrmk_isaaclab_wuji
conda activate env_isaaclab

python scripts/rsl_rl/train.py \
  --task hand_grasp \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000
```

TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/hand_grasp --port 6009 --reload_interval 5
```

우선 볼 항목:

- `Episode_Reward_Raw/success`, `Episode_Termination/success`
- `Episode_Reward_Raw/joint_reference`
- `Episode_Reward_Raw/stick1_reference_pose`, `stick2_reference_pose`
- `Episode_Reward_Raw/functional_contact_min`
- `Episode_Reward_Raw/full_grasp_stability`
- `Episode_Reward_Raw/angular_speed_excess`
- `Metrics/hand_grasp_max/success_stable_steps`
- `Metrics/hand_grasp/quiet_valid`, `Metrics/hand_grasp/full_contact`,
  `Metrics/hand_grasp/reference_pose_valid`
- `Metrics/hand_grasp_final/functional_contact_count`

### 저장된 `hand_grasp` 자세의 실제 contact pair 확인

```bash
cd /home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji
python scripts/debug/hand_grasp_contact_pairs.py \
  --task hand_grasp \
  --pose-file logs/debug/hand_grasp_keyboard/2026-07-28_12-39-52/pose_005.json \
  --headless
```

- `pose_005`를 복원해 0.5초 settle 후 2초 동안 PhysX contact를 측정함.
- 모든 hand-link↔Stick1/Stick2와 hand-link↔hand-link pair의 접촉 유지율·평균력·최대력을 출력함.
- 결과는 `logs/debug/hand_grasp_contact_pairs/<timestamp>/contact_pairs.json`에 저장됨.
- `Metrics/hand_grasp_final/ring_support_force`
- `Metrics/hand_grasp_final/ring_tip_stick2_force`, `ring_distal_stick2_force`
- `Metrics/hand_grasp_final/max_linear_speed`, `max_angular_speed`
- `Metrics/hand_grasp_final/stick1_position_error`, `stick2_position_error`
- `Metrics/hand_grasp_final/stick1_orientation_error`, `stick2_orientation_error`

`Metrics/hand_grasp/*`는 실제 episode 시간 평균이고, `_final/_min/_max`는 episode 마지막/최솟값/최댓값임.
2026-07-28 `pose_005` reset, action scale `0.1`, STATE B reference reward 이전 checkpoint는
resume하지 않고 fresh로 실행함.

# keyboard로 close, open
python scripts/rsl_rl/play.py \
--task hand_grasp \
--num_envs 1 \
--load_run 2026-07-30_12-21-21 \
--keyboard_hand_mode \
--real-time





# hand final play

  # 2026-08-13 legacy 101D model_400: M으로 q/qt CSV 녹화, 1=OPEN, 2=CLOSE
  python scripts/rsl_rl/play.py \
    --task hand_final_play \
    --num_envs 1 \
    --manual_root \
    --legacy_obs_101d \
    --checkpoint '/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji/logs/rsl_rl/hand_final/2026-08-13_14-15-09(최종)/model_400.pt' \
    --real-time

  # CSV: logs/joint_records/joint_record_<timestamp>.csv
  # M을 한 번 누르면 시작, 다시 누르면 종료. qt_*가 replay 입력이다.


  최신 hand_final run의 최신 모델:

  python scripts/rsl_rl/play.py \
    --task hand_final_play \
    --num_envs 1 \
    --manual_root

  특정 날짜의 현재 105D run 최신 모델:

  python scripts/rsl_rl/play.py \
    --task hand_final_play \
    --num_envs 1 \
    --manual_root \
    --load_run <105D_RUN>

  현재 105D run의 특정 모델:

  python scripts/rsl_rl/play.py \
    --task hand_final_play \
    --num_envs 1 \
    --manual_root \
    --load_run <105D_RUN> \
    --checkpoint model_400.pt



# hand final train

cd /home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji
python scripts/rsl_rl/train.py \
  --task hand_final \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000 \
  --init_checkpoint /home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji/logs/rsl_rl/hand_real/2026-08-18_23-57-25/model_4500.pt \
  'env.events.stick_disturbance.params.force_range_n=[0.01,0.3]'




# hand real (1단계 학습은 오버라이드로)
  python scripts/rsl_rl/train.py --task hand_real --headless  env.episode_length_s=5.0



  pose_005 -> prev_target (link4 닿게 j1 더 굽힌 게 curr_target)
  prev_reset -> pregrasp -> pose_005 였음
  curr_reset -> curr_pregrasp (4mm 떨어진 위치)


# hand real2 train

# hand real2 play
    python scripts/rsl_rl/play.py \
      --task hand_real2 \
      --load_run ... \
      --checkpoint model_xxx.pt \
      --mouse_stick_disturbance \
      --mouse_force_base 0.01 \
      --mouse_force_stiffness 1.0 \
      --mouse_force_max 0.30
