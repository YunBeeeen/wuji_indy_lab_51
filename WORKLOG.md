# WORKLOG.md

## Current Status

- 현재 목표는 Indy7 + Wuji hand asset으로 arm end-effector tracking RL env 구성임.
- 현재 active workspace는 `~/wuji_indy_lab_51`임.
- 현재 active code dir은 `~/wuji_indy_lab_51/nrmk_isaaclab_wuji`임.
- 현재 active conda env는 `env_isaaclab`임.
- Isaac Sim 5.1 환경 사용함.
- IsaacLab 2.3 계열 환경 사용함.
- `Indy-Wuji-Reach` task 생성됨.
- Neuromeka `Indy-Reach` 스타일로 구현됨.
- 현재 task는 headless 학습과 GUI 실행 모두 확인됨.
- 현재는 full training 전 arm-only tracking 안정성 확인 단계임.

## Important Decisions

- Isaac Sim 4.5 / IsaacLab 2.2.1 맞추기 실험 중단함.
- `chop_ws/chop_rl` 기반 실험 폐기함.
- Neuromeka public/main branch 스타일 우선함.
- DirectRLEnv from scratch보다 Neuromeka ManagerBasedRLEnv 스타일 우선함.
- 기존 `Indy-Reach`를 덮어쓰지 않고 `indy_wuji` 새 task로 분리함.
- 초기 학습은 arm 6축만 action으로 사용함.
- hand joint는 articulation에 남기되 policy action에는 넣지 않음.
- arm tracking 단계에서는 hand joint observation 제외함.
- tracking body는 `tcp`가 아니라 `palm_link` 사용함.
- `palm_link`는 실제 articulation rigid body로 사용 가능함.
- `tcp`는 USD articulation body로 tracking에 쓰기 부적합하다고 판단함.

## Asset Status

- active robot cfg는 `INDY7_WUJI_RIGHT_CFG`임.
- active USD는 `isaac_neuromeka/assets/model/usd/indy7_wuji_right/indy7_wuji_right_simplified.usd`임.
- `indy7_wuji_right_simplified.usd`에 Wuji hand collision mesh 26개 적용함.
- Wuji collision mesh는 `*_collision.STL`에서 가져옴.
- 직접 USD Mesh collider로 삽입함.
- collider prim에 `PhysicsCollisionAPI` 등 적용함.
- active nested collision leftover는 제거/비활성 처리함.
- validation 결과 hand collision mesh 26개 확인함.
- GUI에서 collision visualization 관련 초록/빨간 표시 차이는 보기 모드/선택 상태 이슈로 판단함.
- Isaac Sim Fabric GPU 모드는 정상으로 판단함.
- `Properties not updated`는 Fabric 모드의 GUI property sync 안내 수준으로 판단함.

## Code Status

- `isaac_neuromeka/assets/indy.py` 수정됨.
- `INDY7_WUJI_RIGHT_CFG` 추가/사용됨.
- `Indy-Wuji-Reach` 등록됨.
- `indy_wuji/env_cfg.py` 구현됨.
- `indy_wuji/learning/rsl_rl_cfg.py` 구성됨.
- `train.py`, `play.py`는 rsl-rl 5 계열 config migration 대응됨.
- `indy_wuji`는 base/teacher/student/CMDP cfg 골자 갖춤.
- 현재 active registration은 `Indy-Wuji-Reach` 하나임.
- teacher/student/CMDP registration은 future block으로 둠.
- arm-only observation 적용됨.
- `policy` observation shape는 175에서 55로 감소함.
- actor/critic input feature는 55임.
- action shape는 6임.

## Verified Runs

- `--num_envs 1 --max_iterations 1` smoke test 통과함.
- `--num_envs 32 --max_iterations 5` headless test 통과함.
- GUI 실행 확인함.
- GUI에서 robot asset 보임.
- GUI에서 Fabric GPU로 실행됨.
- `--num_envs 1 --max_iterations 1` arm-only observation 변경 후 재검증함.
- arm-only observation 재검증 결과 action shape 6 유지됨.
- arm-only observation 재검증 결과 policy observation shape 55 확인됨.
- 사용자가 `--num_envs 128 --max_iterations 20` 실행함.
- `512 env / 100 iterations` run에서 `model_99.pt` checkpoint 생성됨.
- `play.py`로 checkpoint 재생 확인됨.

## Daily Logs

- 2026-07-08 활동 일지는 `ACTIVITY_2026-07-08.md`에 정리함.

## Latest Smoke Test Result

- command 사용함.

```bash
conda run -n env_isaaclab python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 1 --max_iterations 1
```

- action manager shape 6 확인함.
- observation manager policy shape 55 확인함.
- `joint_pos` shape 18 확인함.
- `joint_vel` shape 18 확인함.
- `pose_command` shape 7 확인함.
- `action_history` shape 12 확인함.
- actor model input 55 확인함.
- critic model input 55 확인함.
- 1 PPO iteration 완료함.

## Known Warnings

- rsl-rl `policy` config deprecated warning 있음.
- migration handler로 actor/critic 자동 유도됨.
- `distribution_cfg` warning 있음.
- `rsl_rl` package git repo 못 찾는 warning 있음.
- 학습에는 치명적이지 않음.
- Fabric point instancer warning 있음.
- command visualization 관련 warning으로 판단함.
- PhysX actuator `effort_limit`/`velocity_limit` deprecation warning 있음.
- 추후 `effort_limit_sim`, `velocity_limit_sim`로 정리 가능함.

## Next Steps

- `512 env / 100 iterations` 실행함.
- 마지막 5 iteration 로그 확인함.
- `Mean reward` 확인함.
- `position_error` 확인함.
- `orientation_error` 확인함.
- `joint_vel` penalty 확인함.
- `action std` 확인함.
- NaN/PhysX error 여부 확인함.
- GUI에서 이상 진동/발산 여부 확인함.
- 안정적이면 `512/500` 또는 reward tuning 검토함.
- 성능이 안 좋으면 command range, orientation reward, decimation 조정 검토함.

## Change Log

- Initial handoff docs 생성함.
- Neuromeka Indy-Reach 기반 구조 확인함.
- Indy-Wuji 새 task 구성함.
- Wuji hand collision 26개 복구함.
- active USD를 `indy7_wuji_right_simplified.usd`로 설정함.
- `tcp` 대신 `palm_link` tracking 적용함.
- RSL-RL train/play config migration 대응함.
- GUI 실행 확인함.
- arm-only observation 적용함.
- `WORKLOG.md`, `AGENTS.md`, `study.md` 최신화함.
- `play.py`의 pretrained checkpoint import 경로를 IsaacLab 2.3.2 환경에 맞게 호환 수정함.
- 최신 확인 checkpoint는 `logs/rsl_rl/indy_wuji_reach/2026-07-08_18-16-06/model_99.pt`임.
- `play.py`의 rsl-rl 5 policy export 경로도 수정함.
- rsl-rl 5에서는 `runner.alg.policy` / `actor_critic` 대신 `runner.export_policy_to_jit()` / `runner.export_policy_to_onnx()` 사용함.
- 2 step headless play test 통과함.
- 2026-07-08 활동 일지 `ACTIVITY_2026-07-08.md` 생성함.
