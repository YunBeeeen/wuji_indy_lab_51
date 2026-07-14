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
tensorboard --logdir logs/rsl_rl/indy_wuji_cube_grasp --port 6006 --reload_interval 5
```

- 브라우저에서 봄.

```text
http://localhost:6006
```

- `6006`이 이미 사용 중이면 다른 포트 사용함.

```bash
tensorboard --logdir logs/rsl_rl/indy_wuji_reach --port 6007 --reload_interval 5
```

- cube grasp 로그를 볼 때는 logdir를 바꿈.

```bash
tensorboard --logdir logs/rsl_rl/indy_wuji_cube_grasp --port 6006 --reload_interval 5
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
  --task Indy-Wuji-Cube-Grasp \
  --num_envs 1 \
  --load_run "$(basename "$(ls -td logs/rsl_rl/indy_wuji_cube_grasp/20* | head -n 1)")"
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
  --task Indy-Wuji-Cube-Grasp \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000 \
  --resume \
  --load_run "$(basename "$(ls -td logs/rsl_rl/indy_wuji_cube_grasp/20* | head -n 1)")"
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
