# Agent Notes

- 이 문서는 `nrmk_isaaclab_wuji` repo 내부에서 에이전트가 참고할 작업 환경과 주의사항을 적은 인수인계 문서임.

- Main shared baseline: `/home/lsc/IsaacLab` with `env_isaaclab`, Isaac Sim 5.1, IsaacLab 2.3.x.
- Isaac Sim itself comes from the `isaacsim` pip metapackage (v5.1.0.0) installed inside `env_isaaclab`, not from `/home/lsc/isaacsim_pkg` (that's a separate standalone zip extraction, unused by this workflow). IsaacLab is an editable install pointing at `/home/lsc/IsaacLab/source/isaaclab` (source checkout, not the pip binary release the README's install guide assumes — pre-existing setup, not something to change casually).
- `isaac_neuromeka` (this package, published as `nrmk_isaaclab_public`) must be `pip install -e .`'d into `env_isaaclab` from `nrmk_isaaclab_wuji/` before anything imports it. That install pulled in `pandas>=2,<3`, which downgraded an existing `pandas 3.0.3` to `2.3.3` in the env — worth knowing if something else in `env_isaaclab` expected pandas 3.
- `import isaac_neuromeka` (or anything under `isaaclab`) directly from a plain `python -c` fails with `ModuleNotFoundError: No module named 'pxr'` — this is expected, not a bug. `pxr`/`omni` bindings only become importable after `isaaclab.app.AppLauncher` boots Kit, which is why `scripts/rsl_rl/train.py` always launches the app before importing task modules. Never try to smoke-test task registration without going through `AppLauncher` first.
- New working extension: `/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji`.
- Do not modify shared installs or older user assets directly; copy into the new working extension first.
- Neuromeka public is the IsaacLab/Indy7 extension base. Wuji hand assets came from existing Retargeting/GeoRT-style folders.
- Current target direction: Indy7 + Wuji hand for future chopstick manipulation RL.
- Final manipulation direction is functional chopstick grasp: hold chopsticks in a task-useful configuration so chopstick use becomes possible.
- The primary research target is the functional grasp / Dexterous Pre-grasp style objective, not reproducing DexPoint.
- USD variants:
  - `indy7_wuji_right.usd`: full mesh baseline (arm + hand both convexHull). Reference only, not used for training.
  - `indy7_wuji_right_simplified.usd`: Indy arm simplified (primitives), Wuji hand kept as mesh (`physics:approximation = convexHull`, one hull per link). Fidelity tier for real chopstick-manipulation training.
  - `indy7_wuji_right_all_simplified.usd`: Indy arm simplified, Wuji hand reduced to Cube colliders. Quick-start tier to verify the training pipeline works before chasing fidelity.
- Current active asset in `INDY7_WUJI_RIGHT_CFG` is `indy7_wuji_right_simplified.usd`: Indy arm simplified with Wuji hand collision meshes restored from the 26 `*_collision.STL` files and authored as `MeshCollisionAPI(convexHull)`. `indy7_wuji_right_all_simplified.usd` remains only as a cube-collider fallback/debug tier.
- Known collision caveats (found by opening the USDs directly with a throwaway `usd-core` venv, not the project env):
  - Hand collision in `all_simplified` (cube) is measurably smaller than the visual mesh — worst at fingertips (~58-66% of visual bbox size). Undersized colliders mean sim contact triggers later than the real hand surface would; expect this to bite at sim2real transfer, not during sim-only training.
  - Hand collision in `.usd`/`_simplified.usd` (convexHull) can bulge over concave joint/knuckle dips, so adjacent finger links may start overlapping at tightly curled (chopstick grip) poses even though the source meshes don't — a plausible cause of self-collision jitter at fine-grasp poses.
  - No `contact_offset`/`rest_offset` is authored anywhere (USD or `indy.py`), so PhysX's ~0.02 m default applies to hand parts only ~0.015-0.02 m across. Worth tuning down via `CollisionPropertiesCfg` if contact behavior looks floaty/imprecise once training starts.
- `scripts/rsl_rl/train.py` and `play.py` call `isaaclab_rl.rsl_rl.handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)` right after `cli_args.update_rsl_rl_cfg(...)` (matches upstream IsaacLab's reference scripts). Without it, `rsl-rl-lib==5.0.1` crashes on any task's runner cfg that still uses the deprecated `policy=` field (which `Indy-Reach`/`Dual-Arm-Reach`/`Indy-Wuji-Reach` all do) — don't remove this call, and don't hand-set `actor=`/`critic=` in a task's `RslRlOnPolicyRunnerCfg` as a workaround (it bypasses the migration and leaves deprecated `stochastic`/`init_noise_std` fields dangling, causing `MLPModel.__init__() got an unexpected keyword argument 'stochastic'`). Confirmed working: `Indy-Wuji-Reach` smoke-tested end-to-end (`--headless --num_envs 32 --max_iterations 5`), reward/losses look normal.
- `isaac_neuromeka/assets/model/urdf/wuji_right/wuji_right.urdf` is the standalone hand-only URDF (source for the Wuji hand assets, separate from the combined `indy7_wuji_right.urdf` actually used to build the training `.usd`s). Its `<collision>` blocks used to be commented out on everything except the fingertips — fixed by uncommenting; double-check this file specifically (not just the combined one) if collision ever looks wrong when re-importing the hand alone in Isaac Sim.
- **Isaac Sim 5.1 URDF importer bug**: importing any URDF (reproduced with `wuji_right.urdf`) to an on-disk destination path (the mode that writes `<name>/configuration/{base,physics,robot,sensor}.usd`) silently produces empty `visuals`/`collisions` Xform groups on every link — both visual and collision geometry are missing, not just collision. Importing to an in-memory stage (no destination path) works correctly; the Isaac Sim log prints `"Creating Asset in an in-memory stage, will not create layered structure"` for that working path. Workaround: import with no destination set, then `File → Save As` manually, rather than using the importer's on-disk destination-path option. Don't assume a URDF's geometry is broken just because a disk-exported USD shows empty meshes — reproduce with an in-memory import first.
- `Indy-Wuji-Reach` uses `palm_link` as the real articulation body, not `tcp`. `tcp` is a leftover non-rigid frame under `link6` from the bare indy7 arm (no `RigidBodyAPI` in any of the three usd variants) — the URDF's `tcp -> palm_link` fixed joint got merged away during USD generation. Don't reintroduce `"tcp"` as a body reference in reach/task configs.
- A virtual task EE offset was tested for Wuji (`wuji_ee = palm_link * WUJI_EE_OFFSET`, quaternion `(-0.5, -0.5, 0.5, 0.5)`), and the smoke test worked. This code is now reverted/parked so a long raw `palm_link` baseline can be checked first.
- The current baseline tracks `link6` instead of `palm_link` so the reach pipeline can be validated against a clean Indy arm flange frame before revisiting the Wuji hand frame.
- User reported that after about 2000 training iterations, orientation error dropped from the 2.x range to about 0.8. `palm_link` and `link6` appear similar once training progresses, so do not treat URDF/offset as the confirmed root cause unless later runs plateau again.
- The Allegro reference uses a related concept at the asset level: a `tcp` fixed frame is tracked, and the hand base is attached under it by a fixed joint/offset. For Wuji, revisit a code-level virtual frame only after the `link6` baseline run.
- **GUI viewport stutter is a render-cadence artifact, not a performance problem.** physics `dt = 1/60 s` × `decimation = 24` → environment step `0.4 s` (제어 **2.5 Hz**), and the task cfgs set `sim.render_interval = decimation`, so the viewport redraws only **2.5 times per simulated second** — it looks like dropped frames while the GPU sits at 2-9% utilization. That setting is correct for headless training; don't "fix" it in the task cfg. Before blaming GPU contention for a laggy viewport, check `nvidia-smi` first — if the GPU is idle, it's the render interval, not load.
- **`play.py`와 `train.py` 둘 다 `--render_interval`을 받음.** `play.py`는 기본 `2`, `train.py`는 기본 `None`(= task cfg 값 유지, 기존 동작 불변). GUI로 학습을 볼 때는 `--render_interval 4` (시뮬 1초당 15장) + `--num_envs 64~256`을 권장함.
- **`render_interval`은 `--headless` 학습 속도에 전혀 영향이 없음.** `manager_based_env.py:474,488`에서 `is_rendering = sim.has_gui() or sim.has_rtx_sensors()`일 때만 `sim.render()`를 호출하므로, headless에서는 렌더 자체가 일어나지 않음. IsaacLab이 `render_interval < decimation`일 때 띄우는 경고는 의도된 동작이라 무시해도 됨.
- Isaac Sim 4.5 is not involved in this workflow. Nothing references `/home/lsc/Downloads/isaac-sim-standalone-4.5.0-linux-x86_64` (14 GB, inert); the `4.5.0` lines in `~/.nvidia-omniverse/logs/omni.kit.log` are stale 2026-07-07 telemetry in an append-only log, and the last startups all report `5.1.0`. Don't chase them.
- Cube grasp checkpoints are action/observation-shape specific. `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_20-10-32/model_50.pt` is a 26D-action/70D-obs five-finger checkpoint and cannot load against the current 18D/57D thumb-index-middle code (`runner.load()` raises a `size mismatch` on `mlp.0.weight`).
- **The obs/reward finger asymmetry is intentional — do not "fix" it.** `cube_in_fingertips` (obs) spans all five `finger[1-5]_tip_link`, while the action space and every shaping reward (`finger_cube_reach`, `finger_cube_closeness`, `functional_hold`) cover only thumb/index/middle. The point is to keep the *controlled* dimension low (18D action) for learnability while the policy still perceives the passive ring/little fingers. Ring and little stay in the articulation, driven by the actuator gain (`stiffness=20.0`, `damping=0.5`), not by the policy.
- `scripts/assets/apply_wuji_hand_collision_meshes.py` is the reproducible post-process for the fidelity USD. It de-instances the hand `collisions` Xforms, inserts one collision mesh per Wuji hand link from `isaac_neuromeka/assets/model/urdf/wuji_right/meshes/*_collision.STL`, applies `PhysicsCollisionAPI` + `MeshCollisionAPI(convexHull)`, and blocks nested importer leftovers. Validation target: 26 direct Wuji collision meshes, 0 active nested children, 0 CollisionAPI under hand visuals.
- Root `flow_study.md` summarizes the `Indy-Wuji-Reach` execution flow, task registration, env cfg assembly, MDP function links, and key code snippets for quick review.
- `Indy-Wuji-Reach` now uses `isaac_neuromeka.env.rl_task_custom_env:CustomManagerBasedRLEnv` so TensorBoard logs both weighted reward (`Episode_Reward/*`) and raw unweighted reward (`Episode_Reward_Raw/*`).
- Current position-only baseline uses 15-D policy observations: arm joint_pos 6 + command xyz 3 + previous action 6.
- Current active reward terms are `end_effector_position_tracking` and `action_rate`.
- Orientation tracking, end-effector speed, and joint velocity reward terms are currently removed from the active baseline.

## Latest Handoff Summary

- 아래 `Current Cube Grasp Override (2026-07-14)` 블록이 cube grasp 최신 상태임. 더 아래 오래된 cube grasp 분석 기록은 실험 히스토리로 읽을 것.

## Current Cube Grasp Override (2026-07-14)

- cube grasp는 `Indy-Wuji-Cube-Grasp` 하나만 사용함.
- `Easy`는 이전 실험 이름이며 현재 active registration에는 없음.
- 별도 curriculum/hard task를 나누지 않음. run/checkpoint 선택이 꼬여서 디버깅 비용이 커졌기 때문임.
- 예전 curriculum alias/class/register는 제거됨. 새 학습/play/smoke test에서는 `Indy-Wuji-Cube-Grasp`만 사용함.
- 현재 main task 자체가 가까운 nominal grasp 배치를 사용함.
- cube가 놓이는 받침면은 `BASE_Z=0.40`임.
- cube 위치는 `(0.692, -0.369, 0.430)`임.
- cube 위치는 probe로 검증함. reset `palm_facing=0.987`, zero action 30 step 뒤 `0.997`임.
- 이전 위치 `(0.704, -0.279, 0.430)`는 cage 중심에서 y로 약 `9cm` 벗어나 zero action만으로 cube가 밀려났음.
- `{ENV_REGEX_NS}/Support` kinematic cuboid가 받침면 역할을 함.
- cube reset range는 `x/y/z = 0`임.
- cube mass는 `0.10 kg`임.
- cube size는 `0.06 m`임.
- `cube_lift`와 `cube_clearance`는 월드 바닥이 아니라 받침면 `surface_z=0.40` 기준임.
- `hand_floor` penalty도 월드 바닥이 아니라 받침면 `surface_z=0.40` 기준임.
- `gpu_max_rigid_patch_count`는 `2**20`임.
- active robot contact response는 `max_depenetration_velocity=5.0`, `max_contact_impulse=100.0`임.
- 이전 active 값 `max_depenetration_velocity=1000.0`, `max_contact_impulse=1e32`는 손/palm contact에서 관통 해소 impulse가 너무 커져 arm이 튀는 원인 후보였음.
- 이 변경은 checkpoint shape를 바꾸지 않음. 기존 checkpoint play/resume 가능함.
- 다만 physics가 달라졌으므로 최종 성능 판단은 resume adaptation 또는 fresh/easy run 후 할 것.
- action shape는 `18`임.
- observation shape는 `54`임.
- policy controlled joints는 `joint[0-5]`, `finger[1-3]_joint[1-4]`임.
- arm 6축과 thumb/index/middle 12축을 함께 제어함. `arm_hold_action`은 현재 active task에서 꺼져 있음.
- Action Manager는 `arm_action=18`로 뜸.
- action scale은 arm `0.25`, finger `0.5`임.
- active reward는 8개임: `finger_cage_reach=3`, `finger_cage_hold=5`, `cube_lift=50`, `cube_support=2`, `palm_facing=0`, `arm_manipulability=0`, `hand_floor=0.2`, `action_rate=-0.0003`.
- `cube_support`는 큐브가 받침면 아래로 눌리는 실패 모드를 벌하는 절대 음수 보상임.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- 메인 task 이름으로 close-start probe 확인함.
- probe 결과 reset `cage_center_to_cube=(0.000461, -0.000059, -0.016252)`, `palm_facing=0.986780`, `cage_hold=0.210871`임.
- zero action 30 step 뒤 `cage_center_to_cube=(0.002833, 0.002668, -0.035702)`, `palm_facing=0.996594`임.
- close action `1.0` probe에서 손만 닫으면 `cage_hold`는 약 `0.40`까지 증가하지만 `cube_lift_reward_raw`는 0임.
- `joint0~joint5` 단일축 ±1 lift 후보를 모두 넣어도 양의 cube clearance가 나오지 않음.
- 강한 손/가벼운 큐브 probe(`finger_effort=3`, `stiffness=40`, `cube_mass=0.03`, `friction=2`)에서도 lift는 0임.
- 짧은 `--num_envs 128 --max_iterations 20` grasp+lift 학습에서 hold는 켜지지만 lift는 거의 0이고, cube를 받침 아래로 누르는 실패 모드가 보임.
- 따라서 긴 학습 전에 scripted sequence로 실제 lift가 가능한 arm/hand 조합 또는 초기 자세를 먼저 찾아야 함.
- 더 아래의 오래된 `hard`, `Easy`, `action_rate=-0.005` 기록은 실험 히스토리로 읽고 현재 지침으로 쓰지 않음.
- 같은 experiment 안에 과거 smoke/hard/easy run이 섞여 있으므로 `--load_run "$(ls -td ... | head -n 1)"` 자동 선택은 위험함.
- play/resume은 가능한 한 확인한 run 폴더명을 직접 지정함.
- 목적은 손 높이에 맞는 받침면 위에서 `finger_cage_hold`, `cage_inside_frac`, `cube_lift`가 켜질 수 있는 가까운 초기 상태를 만드는 것임.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1`로 확인함.
- smoke test에서 Action Manager shape `18`, policy observation shape `54`, active reward 8개 확인함.

- reach baseline task는 `Indy-Wuji-Reach`임.
- 현재 active USD는 `indy7_wuji_right_simplified.usd`임.
- 현재 tracking 기준 rigid body는 `link6`임.
- 현재는 Indy arm flange 기준 reach baseline임.
- virtual EE offset 구현은 되돌리고 보류함.
- reach action shape는 6임.
- reach policy observation shape는 15임.
- 현재 observation은 arm 6축 joint position, command xyz, previous action만 봄.
- hand joints는 articulation에 남아 있음.
- reach에서는 hand joints가 policy action/observation에서 제외됨.
- `sim.render_interval = decimation` 적용됨.
- active reward는 position tracking과 action rate penalty만 사용함.
- 다음 긴 학습 후보는 `--num_envs 4096 --max_iterations 50000`임.
- `--num_envs 1 --max_iterations 1` smoke test 통과함.
- 최신 position-only smoke test에서 actor/critic input 15 확인함.
- 최신 position-only smoke test에서 raw/weighted reward log 확인함.
- `--num_envs 32 --max_iterations 5` test 통과함.
- GUI 실행 확인됨.
- 사용자가 `--num_envs 128 --max_iterations 20` 실행함.
- 다음 권장 run은 `--num_envs 512 --max_iterations 100`임.
- root handoff docs는 `/home/lsc/wuji_indy_lab_51/AGENTS.md`, `/home/lsc/wuji_indy_lab_51/WORKLOG.md`, `/home/lsc/wuji_indy_lab_51/study.md`, `/home/lsc/wuji_indy_lab_51/flow_study.md`, `/home/lsc/wuji_indy_lab_51/CLI.md`임.
- 새 cube grasp task skeleton은 `Indy-Wuji-Cube-Grasp`임.
- cube grasp cfg 경로는 `isaac_neuromeka/tasks/manipulation/grasp/cube_grasp_env_cfg.py`임.
- Indy/Wuji cube grasp override 경로는 `isaac_neuromeka/tasks/manipulation/grasp/indy_wuji/env_cfg.py`임.
- cube는 `{ENV_REGEX_NS}/Cube`에 `RigidObjectCfg`로 생성함.
- cube size는 `0.06 m`, mass는 `0.10 kg`, current initial position은 `(0.692, -0.369, 0.430)`임.
- cube grasp RSL-RL experiment name은 `indy_wuji_cube_grasp`임.
- cube grasp long run에서 PhysX patch buffer overflow가 발생해 `gpu_max_rigid_patch_count`를 `2**19`로 올림.
- `2**18`은 2026-07-10 resume run의 약 `263k` patch 요구치에 살짝 부족했음.
- cube grasp는 현재 active command 없이 cube relative position observation을 사용함.
- cube grasp policy observation shape는 최신 구조에서 42임.
- cube grasp action shape는 최신 구조에서 12임.
- cube grasp finger action scale은 최신 구조에서 `0.5`임.
- cube grasp policy controlled joints는 `finger[1-3]_joint[1-4]`임.
- arm 6축은 0D `arm_hold_action`으로 default target 유지함.
- 3-finger 제어는 dimension을 낮추기 위한 의도된 설계임. 비대칭이 아니라 버그가 아님.
- `cube_in_fingertips` observation은 5-finger 전부(`finger[1-5]_tip_link`)를 봄.
- reward와 action은 thumb/index/middle 3개만 다룸.
- ring/little은 policy가 관측만 하고 제어하지 않음. actuator gain으로만 유지됨.
- cube reset randomization은 현재 꺼둠. `x/y/z=0`임.
- cube grasp task에서만 arm initial joint를 `joint1=-0.75`, `joint2=-1.85`, `joint3=-1.61`, `joint4=-1.62`, `joint5=2.35`로 override함.
- arm initial override는 fixed arm target에도 쓰이므로 이전 checkpoint resume은 shape가 맞아도 성능 판단은 fresh run이 더 깔끔함.
- Wuji hand actuator는 전체 finger 공통으로 `stiffness=20.0`, `damping=0.5`, `friction=0.02`, `effort_limit=0.6` (2026-07-12에 stiffness를 `8.0`에서 올림. damping은 한때 `2.5`였으나 **최대 폐합 속도 = effort_limit/damping = 0.24 rad/s로 손가락이 5배 느려져** `0.5`로 되돌림)임.
- 이 값은 ring/little finger 떨림을 줄이기 위한 안정화 설정임.
- **active cube grasp reward는 현재 `finger_cage_hold=1`, `hand_floor=0.5`, `action_rate=-0.0003` 중심임.** `finger_cage_reach`, `palm_facing`, `cube_lift`, `arm_manipulability`은 weight 0으로 꺼둠.
- **"손끝 → 큐브 중심" 거리 reward를 절대 다시 넣지 말 것.** 큐브 중심은 표면에서 `0.03 m` 안쪽이라 **손끝이 도달 불가능한 목표**이고, gradient가 항상 큐브 속을 향함. 게다가 `body_weights=(3,1,1)`이면 엄지가 가중평균의 60%라 **"엄지 하나만 박고 나머지 방치"가 최적해**가 됨. 이것이 2026-07-11 실패의 근본 원인임. 삭제된 함수들(`bodies_to_object_position_tracking_bounded`, `object_in_functional_grasp_region`, `BodiesToObjectProgressReward`)은 이 이유로 제거됨.
- 진단 metric `thumb_middle_opposition` (`+1`=엄지와 중지가 큐브 **양쪽**, `-1`=**같은 쪽**)과 `cage_span` (엄지끝-중지끝 거리)이 추가됨. **`cage_inside_frac`만으로는 "양쪽에서 물었는가"를 볼 수 없음** — 선분이 큐브 모서리만 스쳐도 일부 점은 내부에 들어감.
- **큐브가 맨바닥(ground plane) 위에 있고 테이블이 없음** (`reach_env_cfg.py:47`에 주석 처리). 팔이 `z=0.03`까지 굽혀 내려가야 해서 manipulability가 최악이고 손등이 바닥에 눌림. 실측: palm `z=0.082`, 검지끝 `z=0.030`, **중지끝 `z=0.018`(사실상 바닥)**, `thumb_middle_opposition=-0.98`. 논문은 물체를 테이블 위에 두고 `r_MP`로 낮은 manipulability를 벌함. **`thumb_middle_opposition`이 `-1` 근처에 머물면 테이블 도입이 필요하다는 신호임.**
- cube grasp is a proxy task toward functional chopstick grasp, not the final objective.
- DexPoint is only a helper reference for reach/contact/lift gating patterns.
- Functional grasp / Dexterous Pre-grasp is the main direction for reward design.
- `arm_cube_reach`는 palm over-guidance를 줄이기 위해 비활성화함.
- `cube_lift`와 `cube_goal_tracking`은 contact-gated grasp가 검증될 때까지 비활성화함.
- `arm_cube_reach`는 `palm_link`와 cube root position 거리 기반 bounded reward임.
- `finger_cube_reach`는 thumb/index/middle fingertip과 cube root position 사이 weighted distance의 best-so-far progress reward임.
- `finger_cube_reach`는 현재 거리 자체가 아니라 episode 내 최단거리 갱신량을 reward로 줌.
- `finger_cube_reach` body weight는 thumb/index/middle = `3.0/1.0/1.0`임.
- `finger_cube_reach` weight는 초기 학습 확인용으로 `0.3`임.
- `finger_cube_reach` distance_max는 progress 정규화 scale로 `0.03`임.
- This `distance_max` is a progress scale, not the physical max fingertip-cube distance.
- `finger_cube_reach` cfg는 현재 `mode="best"`임.
- `Episode_Reward_Raw/finger_cube_reach`는 절대 접근도가 아니라 최단거리 개선량임.
- 실제 fingertip-cube 거리는 `Metrics/cube/*`로 봄.
- `finger_cube_closeness`는 progress 이후 손끝이 cube 근처에 머무르게 하는 absolute distance reward임.
- `finger_cube_closeness` weight는 `0.2`, `distance_max`는 `0.7`임.
- `finger_cube_closeness` 추가는 action/observation dimension을 바꾸지 않음.
- velocity observation/reward는 아직 넣지 않음.
- `functional_hold`는 functional grasp / Dexterous Pre-grasp 논문의 hold/cage 아이디어를 현재 cube task에 맞춘 reward임.
- `functional_hold`는 cube가 thumb/index/middle fingertip region 안에 들어왔는지 봄.
- `functional_hold` uses thumb-weighted distance bonus, uniform grasp-center computation, and an additive return to avoid overly sparse gate behavior.
- `functional_hold` raw is `0.4 * center_bonus + 0.4 * weighted_distance_bonus + 0.2 * center_bonus * opposition`.
- `functional_hold` weight는 `0.2`임.
- cube grasp active positive reward는 `finger_cube_reach`, `finger_cube_closeness`, `functional_hold` 순서로 접근, 유지, functional hold를 담당함.
- random policy smoke test에서 `functional_hold`가 0이어도 정상임.
- `action_rate` weight는 `-0.0003`임.
- lift reward와 post-lift cube goal tracking은 현재 active reward에서 제외함.
- cube goal observation은 checkpoint shape 유지를 위해 남겨둠.
- TensorBoard cube distance metric은 `Metrics/cube/*`로 확인함.
- cube distance metric은 reward가 아니라 실제 error 확인용 logging임.
- 현재 metric은 palm, thumb, index, middle, ring, little, five-finger 평균, non-thumb 평균, reward-weighted finger 평균 거리임.
- **핵심 결론 (2026-07-11): 현재 reward에는 "쥐었다"를 보상하는 항이 하나도 없음.** `cube_lift = None`, `cube_goal_tracking = None`, contact sensor 미구현. **grasp이 목적함수에 존재하지 않으므로 `distance_max`를 아무리 튜닝해도 정책은 grasp을 학습할 수 없음.** 지금까지의 reward 논쟁은 전부 접근(reach) 단계 튜닝이었고, 정책은 **이미 접근에 성공했음.** 막힌 지점은 접근 이후임.
- **실측 접촉 테스트: 손은 이미 큐브에 닿고 있고, 밀어내고 있음** (큐브 변위 평균 `32 mm`, 최대 `114 mm`, 73% env에서 `5 mm` 이상 이동; 수직 상승은 `0.6 mm`로 사실상 0). hover가 아님. 거리가 `0.07-0.09 m`에서 정체되는 이유는 **자기가 밀어낸 큐브를 계속 쫓기 때문**임.
- **`functional_hold`는 자세(shape) 보상이지 결과(outcome) 보상이 아님.** 근접도 + 손가락 대향 기하학만 보고 파지 여부를 안 봄. "근처 + 대향" 자세만으로 step당 `0.56`(만점의 56%)을 계속 받음. 더 나쁜 것은 **실제로 쥐려 하면 큐브가 밀려나 거리 reward가 깎이므로 파지에 사실상 페널티가 붙음.** → "가까이 가서 손가락 벌리고 쥐지 않는" 국소최적.
- `Metrics/cube/finger_weighted_mean_distance`는 **정책이 제어하지 않는 ring/little을 포함한 5-finger 가중평균**이라 실제 접근 성능을 과소평가함. 제어 손가락(thumb/index/middle)만 보면 `0.073-0.094 m`임.
- **cage 가상점은 12개임 (2026-07-12).** `CAGE_BODIES = [finger1_tip_link(엄지끝, 기준점), finger2_tip_link, finger2_link3, finger3_tip_link, finger3_link3]` — 엄지끝에서 대향 body 4개로 선분을 긋고 각 3점. 논문은 엄지↔중지만 써서 6점이지만, **논문에는 `r_grasp`(`r_hr`+`r_hj`)가 손 회전과 손가락 관절각을 붙잡고 있음.** 큐브는 대칭이라 목표 파지 자세가 없어 `r_grasp`를 못 쓰는데, 6점만 쓰면 **검지가 완전히 자유가 되어 "손바닥이 하늘을 보고 검지·중지가 교차한" 기괴한 자세로도 만점**이 나옴 (2026-07-11 run에서 실측). 엄지+검지+중지는 젓가락 그립과 동일하므로 임시방편이 아님.
- **`managers.py`의 `_cage_body_names`는 `CAGE_BODIES`와 반드시 동일해야 함.** 한쪽만 바꾸면 metric이 reward와 다른 점을 측정함.
- **`thumb_index_opposition`과 `thumb_middle_opposition`을 둘 다 볼 것.** 중지만 보면 검지가 교차해도 알 수 없음 — 그게 2026-07-11 실패를 놓친 이유임.
- **`finger_cage_hold` (`mdp.object_in_finger_cage`)가 파지를 보상하는 유일한 항임.** 논문 Dexterous Pre-grasp Eq.15 구현. 엄지끝 ↔ 중지(끝/중간마디) 사이에 가상점 6개를 찍고, 각 점이 큐브 **내부**로 파고들수록 보상함. **손을 오므리면 점들이 서로 가까워지며 큐브 안으로 들어가므로 "오므리기"가 직접 보상됨 — 접촉센서 불필요.** 큐브 SDF는 해석식(`_box_signed_distance`)이라 CAD/사전계산 불필요함.
- **거리 reward와 cage reward는 접촉에 대해 부호가 반대임.** 거리 reward는 물체를 만지면 밀려나 거리가 늘고 감점됨 → **접촉이 손해** → hover 국소최적. cage reward는 물체를 파고들어야 점수가 남 → **접촉이 이득**. 이 부호 차이가 파지 학습의 핵심임.
- `object_in_finger_cage`의 `SceneEntityCfg`에는 **`preserve_order=True`가 필수**임. 기본값 `False`면 `find_bodies`가 body_ids를 정렬해서 `[thumb_tip, opposing_tip, opposing_mid]` 순서가 깨짐.
- `finger_cage_hold` 파라미터는 실측 튜닝됨 (`sphere_radius=0.005`, `depth_max=0.02`). **`sphere_radius`가 크면 손가락을 벌린 채 큐브가 사이에 있기만 해도 점수가 나와 대비가 죽음** (`0.02` → hover 0.30/오므림 0.49, 1.6배 / `0.005` → hover 0.19/오므림 0.46, 2.4배). 새 파라미터는 굴곡 kinematic sweep으로 검증할 것.
- **가중치는 반드시 `hold ≫ reach`여야 함** (논문 `r_T >> r_orient >> r_hold >> r_reach`). 쉬운 앞 단계에 큰 보상을 주면 정책이 거기 눌러앉음 — 이것이 2026-07-11 이전 국소최적의 원인이었음 (`reach 0.3 > hold 0.2`로 역순이었음). 현재 `finger_cage_hold(1.0) >> finger_cube_reach(0.3)`.
- **reach와 hold는 반드시 같은 가상점을 공유해야 함 (논문 Eq.14 = Eq.15의 점들).** 2026-07-11에 hold만 논문식(`finger_cage_hold`)으로 바꾸고 reach는 "손끝→큐브중심"으로 두었더니, **엄지가 큐브 중심을 찌르고 나머지 손가락은 뒤에 남는 자세**가 학습됨 (thumb `0.017 m`, index `0.072`, middle `0.078`). 그 자세에서는 엄지-중지 선분이 큐브를 관통하지 않아 **오므릴수록 가상점이 큐브 밖으로 빠져나감** — 강제로 오므리니 `cage_inside_frac`이 `0.47`→`0.40`으로 **하락**했음. 정책은 reward의 최적점에 앉아 있었고, reward가 틀렸던 것임. 현재는 `finger_cage_reach`(`mdp.ObjectCageProgressReward`, Eq.14)가 같은 6점을 써서 **파지 간극을 큐브 위로** 끌어옴.
- **progress reward는 `mode="previous"` + `clamp(min=-1)` + `reset()`에서 기준선 seeding, 이 셋을 다 해야 함.** 하나라도 빠지면 swing-out이 남음: `clamp(min=0)`이면 후퇴가 공짜이고, 기준선을 첫 `__call__`에서 잡으면 **첫 액션이 기준선을 공짜로 부풀림**. 셋을 다 하면 총합이 `d(reset) − d(final)`로 telescoping되어 페이스 조작·swing-out 모두 무의미해짐. 실측: swing-out step이 `0`(무료) → `-0.77`(감점)로 바뀜.
- **강제 오므림 테스트가 진단의 결정타였음** (`extra` rad를 flexion 관절 목표에 더하며 물리와 함께 진행). 이걸로 "큐브가 밀려나서 못 잡는다"(cube_moved는 `8 mm`뿐 → 기각)와 "reward가 오므리기를 처벌한다"(확인)를 갈랐음. 파지가 안 될 때 **가설을 세우지 말고 이 테스트부터 돌릴 것.**
- **2026-07-11 `finger_cage_hold` 첫 학습(run `2026-07-11_16-43-19`, 211 iter): cage는 작동함.** `cage_sdf_mean` `0.597`→`0.010` (가상점이 큐브 표면까지 도달), `cage_inside_frac` `0`→`0.414`, 모든 곡선 아직 상승 중. **큐브가 엄지-중지 파지 간극 안으로 들어옴 — 이전 정책이 못 하던 것임.**
- **주의: `cage_inside_frac`이 높다고 "쥐고 있다"는 뜻이 아님.** 가상점은 엄지-중지 사이 **선분** 위에 있어서, 큐브가 그 선분을 가로지르면 **양쪽 손가락이 안 닿아도** 중간 점들이 큐브 내부에 들어감. 실제 파지 여부는 `*_surface`로 확인할 것 (위 run: thumb `0.026`, index `0.050`, middle `0.060` → 엄지만 접촉, 나머지는 약 `0.02 m` 떠 있음).
- **거리 reward가 cage reward와 직접 충돌함 (실측 확인).** `cube_displacement`가 `0.146`(피크) → `0.037`로 **감소** — 정책이 큐브를 "덜 건드리는" 법을 학습 중임. 만지면 큐브가 밀려나 거리 reward가 깎이기 때문임. **cage가 자리잡은 뒤에는 `finger_cube_closeness`를 약화/제거하는 것을 검토할 것.**
- **`cube_lift` (`mdp.object_lift_in_cage`, weight `3.0`)가 어떤 자세를 "진짜 파지"로 인정할지 결정하는 항임.** `r = cage_gate * clamp(height/0.08, 0, 1)`. **들지 못하는 자세는 파지가 아니므로, 자세를 지정할 필요 없이 하중을 견디는지만 물으면 됨.** gate가 없으면 "파지 없이 큐브를 튕겨 올리는" 편법이 가능함. 조밀형이라 현재의 `2 mm` 상승에도 gradient가 있음 (희소형이면 영원히 `0`이라 학습 불가).
- **`palm_facing` (`mdp.palm_facing_object`, weight `0.5`)는 자의적 제약이 아니라 물리적 필요조건임.** 손가락은 손바닥 쪽으로 굽으므로 **손바닥 뒤에 있는 물체는 오므려도 감싸지지 않음.** cage 항들은 이걸 못 봄 — **엄지끝-손가락 선분은 손 방향과 무관하게 큐브를 관통할 수 있어서**, 손바닥이 하늘을 봐도 cage가 만점임 (2026-07-12 실측: `palm_facing=0.182`인데 `cage_inside_frac=0.753`). 손바닥 법선 축만 정렬하고 **roll은 자유**라 대칭 물체의 파지 방식을 고르지 않음. 논문 `r_hr`에서 **물리적으로 반드시 필요한 성분 1개**만 남긴 것이며, 젓가락에서 `r_hr`이 상위호환 교체함 (버려지는 게 아니라 승격).
- 손바닥 법선은 `palm_link` 로컬 **`+x`**임 (손가락을 오므릴 때 손끝이 이동하는 방향으로 실측). `managers.py`의 `_palm_normal_b`와 reward의 `palm_normal_b`는 반드시 동일해야 함.
- **reward를 넣기 전에 kinematic 도달성부터 확인할 것.** 팔 관절 40만 개 샘플링 결과 손바닥을 큐브에 대고 정면(`+1.000`)으로 돌릴 수 있음 (`0.14%`의 자세). **바닥이 막는 게 아니라 보상이 없어서 안 하는 것이었음.** 다만 `0.14%`라 명시적 보상 없이는 무작위 탐색으로 못 찾음.
- **`cube_lift`는 큐브 중심이 아니라 최하 모서리(`box_ground_clearance`) 기준이어야 함.** 중심 높이를 쓰면 **큐브를 모서리로 세우는 기울이기 편법**에 보상이 나감 (실측: 중심 `+4.28 mm`인데 최하 모서리 `-0.04 mm`로 바닥에 붙어 있었음). metric은 `cube_clearance`(진짜 lift)와 `cube_lift`(중심)를 **둘 다** 로깅하므로, **두 값이 벌어지면 기울이기 재발 신호임.**
- (구) **손바닥 방향 reward(`palm_facing`)는 검토했다가 철회함 — 자의적이기 때문임.** ← **이 판단은 틀렸음.** "`cube_lift`가 자세를 선별할 것"이라 봤으나 `cube_lift`가 기울이기로 해킹당했고 손바닥은 여전히 하늘을 봤음. 논문이 `r_hr`(목표 손 회전)을 주는 이유는 **기능** 때문임 (드릴을 "트리거를 당길 수 있게" 쥐어야 함). 큐브에는 기능 요구가 없으므로 목표 회전이 필요 없는 것이 맞음. **대신 `cube_lift`가 non-arbitrary한 선별 기준임.** 참고로 `palm_link`의 손바닥 법선은 실측 결과 로컬 **`+x`**축임 (나중에 필요해지면 재측정 불필요).
- **`opposition`은 손바닥 방향을 보지 못함.** "엄지와 손가락이 큐브 양쪽에 있는가"만 보므로, `+0.5`를 넘어도 손바닥이 하늘을 볼 수 있음. 자세 판정을 opposition만으로 하지 말 것.
- (구) 아직 미구현: `cube_lift` (sparse 성공 보상). **논문은 fake success 방지용으로 필수라고 명시했고, 2026-07-11 run이 정확히 그 상태였음** — `cube_lift` final `0.0005 m`, max `0.0036 m`로 전혀 못 듦. 그리고 progress reward의 `mode="best"` + `clamp(min=0)` 수정 (swing-out 원인; 위 run에서 `cube_max/finger_weighted_mean_distance`가 `0.824`→`1.153`으로 악화됨).
- **2026-07-10_22-33-40 run은 큐브에서 `0.22 m` 떨어진 local optimum에 완전히 수렴함.** `0.197`까지 접근했다가 `0.222`로 후퇴한 뒤 1200 iteration 동안 고정. `Train/mean_reward`도 `1.24`에서 평평함. 느린 학습이 아니라 갇힌 것임.
- **과거 run을 분석할 때는 반드시 `logs/rsl_rl/<exp>/<run>/params/env.yaml`을 읽을 것.** 현재 코드의 파라미터 값을 과거 run에 대입하면 틀린 결론이 나옴 (2026-07-11에 실제로 이 실수를 함).
- **action은 절대 위치 명령임** (`target = default + scale * raw`, `mdp/actions/joint_actions.py:28,35`). `init_noise_std=1.0`이므로 `scale`이 곧 관절당 도달 반경임 (`scale=0.2` → `±0.2 rad`). **action scale은 도달 범위이므로 jitter를 줄이려고 낮추면 안 됨** — jitter의 직접 레버는 `action_rate` weight임. 2026-07-10 run은 `scale=0.2`로 palm 예산 `0.15-0.20 m` 중 `0.077 m`만 썼으므로 천장은 binding이 아니었음. 이후 `0.1`로 낮춘 것은 non-binding 제약을 binding으로 만드는 방향이라 되돌려야 함.
- **bounded distance reward의 실효 gradient는 `weight / distance_max`임.** PPO는 advantage로 학습하므로 매 step 상수 offset은 value function이 흡수해 상쇄됨 — reward의 절대 크기가 아니라 기울기만 학습 신호가 됨. `distance_max`가 과대하면 실측 거리 구간이 reward 출력의 좁은 상단부에만 매핑되고 나머지는 정보 없는 상수 바닥이 됨. **새 distance reward를 넣을 때 `Metrics/cube/*` 실측 범위부터 보고 `distance_max`를 거기에 맞출 것.** fingertip-cube 실측 범위는 `0.19`(수렴점) ~ `0.40`(에피소드 시작) m임.
- `functional_hold`의 gate(`distance_max`/`center_distance_max` = `0.18`)는 죽은 항이 아니라 **경계 포화** 상태임. `thumb_distance`가 `0.170`(gate 안)까지 갔다가 `0.199`(gate 밖)로 후퇴해 경계에 붙어 앉았고, 가용 reward의 약 `1.4%`만 벌면서 우상향이 멈춤.
- **`mode="best"` progress reward는 구조적으로 착취 가능함. 절대 최대항으로 두면 안 됨.** 에피소드 총합이 대략 `(d_1 - d_min) / distance_max`라서 "어디서 끝나는가"가 아니라 "얼마나 이동했는가"를 지불함 → 첫 step에 물체 반대로 크게 튕겨 `d_1`을 키우는 게 이득이고, 가까이 머무는 데는 한 푼도 안 줌. **절대거리 reward가 반드시 progress reward를 지배해야 함** (경험칙: 기울기 기준 3-4배).
- **`Metrics/cube/*` (TensorBoard)는 에피소드 20 step 전체의 평균이라 성능 지표가 아님.** 앞 4 step(출발 `0.7 m` + swing-out)이 평균의 77%를 지배함. 2026-07-11에 이 평균에 속아 두 번 오진했음.
- **2026-07-11부터 `Metrics/cube_final/*` (마지막 step, ★정착 자세), `Metrics/cube_min/*`, `Metrics/cube_max/*` (swing-out 탐지)가 추가됨. 평가는 `cube_final`로 할 것.** 지표 20종 = 중심거리 9 + **표면거리 6** (`*_surface`, 음수=관통) + **cage 진단 3** (`cage_sdf_mean/min`, `cage_inside_frac`) + **큐브 상태 2** (`cube_displacement`, `cube_lift`). 보상은 `Episode_Reward/*`(가중치 적용)와 `Episode_Reward_Raw/*`(원값) 둘 다 있음.
- **지표는 80개지만 실제로 볼 것은 5개뿐임.** `cube_final/cage_inside_frac` (★★★ 오므리는가, `0`→`0.3+`면 성공), `cube_final/thumb_surface`·`middle_surface` (★★ 닿는가, 음수=접촉), `cube_final/cube_lift` (★★ 드는가), `cube_max/finger_weighted_mean_distance` (★ swing-out, 시작값 `0.7`을 크게 넘으면 팔 휘두르는 중). 나머지 ~72개는 잉여지만 **일부러 유지함** — 계산 비용이 사실상 0이고, 지표가 없어서 오진하면 학습을 다시 돌려야 함. TB 검색창에 `cube_final`만 치면 필터됨.
- 잉여 지표의 성격: `Metrics/cube/*`(평균)는 **오해 유발**이라 성능 판단 금지. `Metrics/cube_min/*`은 수렴 후 머무르므로 `cube_final`과 거의 동일. `ring_*`/`little_*`은 정책이 제어하지 않는 손가락. `*_distance`(중심까지)는 큐브가 `0.06 m`라 body간 비교 불가 — `*_surface`가 상위호환임.
- cage 가상점은 reward(`object_in_finger_cage`)와 metric(`managers.py`) 양쪽에서 **동일하게** 재구성됨 — 비율 `[0.25, 0.50, 0.75]`, 선분 A `finger1_tip_link→finger3_tip_link`(핀치), 선분 B `finger1_tip_link→finger3_link3`(파워). 한쪽만 바꾸면 진단이 어긋남.
- **타이밍 실측: physics `1/60 s`, environment step `0.4 s` (`decimation=24`) → 제어 `2.5 Hz`, 에피소드 `20 step`.** (5 Hz / 40 step이 아님.)
- **손이 큐브에서 `0.72 m` 떨어진 곳에서 시작함.** arm init override가 손을 큐브 근처로 데려오지 못함. **에피소드 20 step 전부가 팔 transit에 소모되고 실제 grasp에 쓸 step이 남지 않음.** 연구 목표는 functional grasp이지 arm transit이 아니므로, 시작 거리를 `0.20-0.30 m`로 줄이는 것이 구조적으로 옳음.
- **action scale = default 자세를 중심으로 한 도달 반경.** `target = default + scale * raw_action`은 **절대** 위치 명령이라 과거 action이 누적되지 않음 — 지금 이 순간의 action 하나가 지금 자세를 전부 결정함. 따라서 도달 가능 집합은 `{default + scale * a}`이고 `scale`이 그 반경임 (`init_noise_std=1.0`이므로 실질 관절당 `±scale` rad). **마우스 감도와 같음: 낮추면 정밀하지만 마우스패드가 모자람.** jitter를 잡으려면 `action_rate` weight를 쓸 것 — 그건 action의 *변화*만 벌하고 도달 범위는 안 건드림.
- **2026-07-10 config가 검증된 baseline임 (iter 50에서 4 step 만에 `0.106 m` 도달, 실측).** 2026-07-11 변경 후 같은 iter 50에서 `0.254 m`로 퇴행함. 신 정책을 scale `0.2`로 replay해도 `0.215 m`에 그치므로 **action scale은 원인의 일부일 뿐이고 reward 변경도 독립적으로 기여함.**
- **2026-07-11에 5가지(action scale, `finger_cube_reach.distance_max`, `arm_cube_reach` 비활성화, `finger_cube_closeness` 추가, `functional_hold` 덧셈형)를 동시에 바꿔서 원인 분리가 불가능해졌음. baseline을 고정하고 한 번에 하나씩 바꿀 것.**
- **`distance_max`는 두 reward 함수에서 의미가 완전히 다름 (naming trap).** 절대거리 reward(`bodies_to_object_position_tracking_bounded`)에서는 reward가 `0`이 되는 **공간적 임계값**임. progress reward(`BodiesToObjectProgressReward`)에서는 `r = clamp(progress/distance_max, 0, 1)`이므로 **"한 step에 몇 m를 좁혀야 만점 `1.0`인가"**를 정하는 정규화 상수임.
- **progress reward의 `distance_max`가 작으면 상시 포화되어 "천천히 접근하기"를 보상함 (dawdle hacking).** 포화가 없으면 `sum_t (best_{t-1} - d_t) = d_start - d_min`으로 telescoping되어 총합이 페이스와 무관해짐 → 착취 불가. **step당 `1.0` clamp이 이 telescoping을 깨뜨려 페이스 조절을 수익화함.** 총 reward = "임계값 이상 신기록을 세운 step 개수"가 되므로, 4 step에 도착하면 4점, 15 step에 걸쳐 질질 끌면 15점 — **도착하면 손해임.**
- **실측: `distance_max=0.03`으로 낮춘 뒤 정책이 `t=5~12` 8 step 연속 포화(step당 개선량 `0.032-0.056 m`, 임계값 `0.03` 바로 위)하며 페이스를 조절하고 끝까지 도착하지 않음.** step당 평균 progress reward는 `0.089`(구, `dmax=0.5`) → `0.489`(신, `dmax=0.03`)로 5.5배 오르고 최종 거리는 `0.106` → `0.254 m`로 2.4배 악화됨. **규칙: progress reward의 `distance_max`는 실제 step당 최대 개선량보다 충분히 커야 함. `0.5`는 미포화, `0.03`은 상시 포화.**
- **`mode="best"` progress reward의 swing-out 해킹이 실측으로 확인됨.** 두 정책 모두 첫 step에 큐브 반대로 튕김 (구 `+0.80 m`, 신 `+0.12 m`) — `(d_1 - d_min)`을 지불하므로 멀리서 시작할수록 이득이기 때문임. 구 정책은 scale이 커서 4 step에 복구해 손해가 없음. 실재하는 결함이지만 퇴행의 주원인은 아님.
- 상세 및 수정 우선순위는 `worklog.md`의 `2026-07-11 Per-Step Trajectory Measurement` 참고 (이전 두 분석을 대체함).
- `action_rate` weight `-0.0003`은 전체 reward의 약 2%로 jitter를 억제하지 못함. 실제로 raw action_rate가 `21` → `95`로 증가함.
- 상세 분석은 `worklog.md`의 `2026-07-11 Cube Grasp Structure Analysis` 참고.
- 6D arm-only `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test는 통과함.
- 최신 hand-only 12D action + 42D observation smoke test는 통과함.
- Wuji finger alias는 `finger1=thumb`, `finger2=index`, `finger3=middle`, `finger4=ring`, `finger5=little`임.
- 코드에서는 USD/URDF의 `finger[1-5]_joint[1-4]` 패턴을 유지함.
- 2026-07-10 코드 작성 기록은 root `code_write.md`에 있음.

## Wuji 손 — 액추에이터 vs 액션 (필독)

- Wuji 손은 **손가락 5개 × 관절 4개 = 20 관절**임 (`finger[1-5]_joint[1-4]`).
  `finger1`=엄지, `finger2`=검지, `finger3`=중지, `finger4`=약지, `finger5`=새끼.
- **액추에이터는 20개 전부에 붙어 있음** (`indy.py`, `stiffness=20.0`).
- **액션은 `finger[1-3]`의 12개만 제어함** (action dim 축소 목적, 총 18D = arm 6 + finger 12).
- **따라서 `finger4`/`finger5`는 "제어 불가 + 액추에이터가 `default_joint_pos`에 뻣뻣하게 고정"임.**
  → **초기 자세를 안 정해주면 편 채로 굳어서 손 앞을 막는 장애물이 됨.**
  실제로 2026-07-13까지 `0.0`(쫙 편 자세)이라 손끝이 검지/중지보다 앞에 있었고
  (tip x: 약지 .841 > 새끼 .838 > 중지 .837 > 검지 .825), 접근할 때마다 큐브를 12cm 밀어냈음.
- **현재 (grasp task)**: 3지 파지 자세로 접어둠 — `finger[4-5]_joint{1,3,4}=1.20`, `joint2=0.0`.

### 함정
`init_state.joint_pos`에 `finger[1-5]_joint[1-4]`(asset 원본 키)와 `finger[4-5]_joint1`이
**동시에 있으면** `ValueError: Multiple matches for 'finger4_joint1'`로 죽음.
→ **override 전에 원본 키를 `pop` 할 것.**

### 손가락 관절 한계 (측정값, rad)
```
finger1_joint1 [-0.045,+1.651]   finger[2-5]_joint1 [-0.327,+1.636]   (굽힘)
finger1_joint2 [-0.166,+0.934]   finger[2-5]_joint2 [-0.495,+0.495]   (벌림)
finger[1-5]_joint3 [-0.493,+1.627]   finger[1-5]_joint4 [-0.493,+1.627]  (굽힘)
```

## Cube Grasp 현재 상태 (2026-07-13)

- reward 6개: `finger_cage_reach`(차분 .3) / `finger_cage_hold`(절대 1.0) / `cube_lift`(절대 3.0) /
  `palm_facing`(**차분** 1.0) / `arm_manipulability`(페널티 1.0) / `action_rate`(-3e-4)
- **`palm_facing`을 절대형→차분형으로 바꾼 게 먹혔음.** 절대형일 땐 1809 iter 동안 31cm 밖에서
  손바닥만 겨눈 채 정체(`cube_displacement`=0.0022, 큐브를 한 번도 안 건드림).
  차분형으로 바꾸니 드디어 큐브에 닿음(`cube_displacement`=0.125).
- **`finger_cage_hold`와 `cube_lift`는 아직 한 번도 0을 벗어난 적 없음.**
  약지/새끼 버그 때문에 물리적으로 불가능했음. 수정 후 재확인 필요.
- **커리큘럼 러닝은 버그 수정 후에 판단.** 판단 기준: `Metrics/cube_final/cage_inside_frac > 0`.
  안 뜨면 그때 도입 (논문: 없으면 ~50%±큰분산, 있으면 97%).

### 다음 run에서 볼 지표 (우선순위 순)
1. `Metrics/cube_final/cube_displacement` → **0.01 이하** (큐브를 안 밀어냄)
2. `Metrics/cube_final/cage_inside_frac` → **0을 벗어나야 함** ← 최우선
3. `Episode_Reward_Raw/finger_cage_hold` → **0을 벗어나야 함**
4. `Metrics/cube_final/palm_facing` → 0.9 이상

### 절대 쓰지 말 것
`Metrics/cube/*` (에피소드 평균) — 초반 4 step이 평균의 77%를 차지해 오진을 유발함.
**반드시 `Metrics/cube_final/*` (정착 자세)를 볼 것.**

## Action space (필독 — 2026-07-13에 여기서 발산이 났음)

```python
# grasp/indy_wuji/env_cfg.py
scale = 1.0            # 관절 목표 = default_joint_pos + scale * action
# learning/rsl_rl_cfg.py
clip_actions = 1.0     # 없으면 None -> 상한 없음 -> 발산함
```

**`scale`과 `clip_actions`는 반드시 같이 설계할 것. 하나만 보면 반드시 틀림.**
- `scale`이 곧 **도달 반경**임 (절대 위치 명령이므로). 작으면 정책이 `|a|`를 키워서 보상하려 함
- `clip`이 없으면 그 키우기가 **발산**으로 이어짐
- 2026-07-13 이전: `scale=0.2` + `clip 없음` -> `|a|` 평균 1.5, `|Δa|` 최대 **9.66**
  -> 관절 목표가 한 step에 110° 점프 -> 팔이 velocity_limit로 왕복 -> 큐브를 67cm 날림
- `clip`만 1.0으로 걸면 관절이 ±11°밖에 못 움직여 **큐브에 영영 못 감**. scale도 같이 키워야 함

**도달성 검증 결과** (4096 x 8회 샘플, `default ± s` 범위, 손끝 3개가 전부 큐브 표면에 닿는 거리):
```
s=0.5 -> 3.63 cm (도달 불가)   s=1.0 -> 0.20 cm (최적)
s=1.5 -> 0.45 cm               s=2.0 -> 1.94 cm (넓힐수록 나빠짐)
```

## 타이밍 / reward 스케일 (필독)

```
sim.dt = 1/60,  decimation = 2  ->  30 Hz,  240 step/episode,  8초
```

### RewardManager가 `raw * weight * dt`로 누적함
```
절대형 (hold, lift, hand_floor, manip)  에피소드 합 = 평균값 x 8초  -> decimation 무관, 불변
차분형 (reach, palm_facing)             에피소드 합 = dt x 총변화량 -> dt에 비례
```
**`decimation`을 바꾸면 차분형 weight를 반드시 같이 바꿀 것.** dt 0.4 -> 1/30이면 12배 약해짐.

### 비용 구조
```
sim.dt      -> GPU 비용에 정확히 비례.  1/60 -> 1/120 이면 물리 계산 2배
decimation  -> 물리 비용과 무관.  정책 forward만 늘어남 (거의 공짜)
```
**physics dt 값을 이미 지불했으면 제어 주파수를 올리는 건 거의 공짜임.**

### iteration 숫자를 run 간에 비교하지 말 것
```
iteration당 시뮬 시간 = num_steps_per_env x decimation x sim.dt
decimation 24 -> 2 이면 iteration당 경험량이 12배 줄어듦 (9.6초 -> 0.8초/iter)
```

### physics dt는 참고 task 중 제일 성김 (의도적, 잠정)
```
Allegro/in-hand 1/120 | ShadowHand 1/120 | Lift-Cube 1/100 | 우리 1/60
```
손끝 0.5m/s면 물리 1 step에 8.3mm 이동 -> 큐브(60mm)를 그만큼 파고든 뒤에야 솔버가 알아챔.
**손가락이 큐브에 실제로 접촉하기 시작한 뒤 침투 깊이를 재보고 올릴 것.**
올릴 땐 decimation도 같이 (예: 1/120 + dec 4 = 30 Hz 유지).

## 진단 도구

### TensorBoard (자동)
```
Metrics/cube_max/action_delta       |a_t - a_{t-1}| 최대.  clip 후 상한 2.0
Metrics/cube_max/action_track_err   |관절목표 - 관절실제| 최대 [rad]
```
**반드시 `cube_max`를 볼 것** — 튀는 건 순간적 사건이라 평균/최종값엔 안 잡힘.

### play.py
```bash
python scripts/rsl_rl/play.py --task Indy-Wuji-Cube-Grasp --num_envs 32 \
  --checkpoint <path>.pt --print_action --print_action_detail
```

- `raw`: policy network 출력.
- `applied`: `clip_actions` 적용 후 실제 env에 들어간 action.
- `target`: `default_joint_pos + scale * applied`.
- `actual`: 현재 실제 관절각.
- `err`: `target - actual`.
- `clip%`가 높으면 policy가 action bound 밖으로 계속 밀고 있다는 뜻임.
- joint별 출력이 필요하면 `--print_action_interval 1 --print_action_detail`을 같이 씀.

### 판정 규칙
```
추종오차 작음(<0.1) + |Δa| 큼(>0.3)  ->  팔이 명령대로 발광. 학습/보상 문제
추종오차 큼(>0.3) 단독                ->  물리가 명령을 이김. dt/decimation 문제
   단, 명령 자체가 도달 불가능해도 추종오차는 커짐 -> |Δa|를 반드시 같이 볼 것
```

**`Episode_Reward_Raw/action_rate`는 공짜 진단 도구임** — 별도 코드 없이 "정책이 명령을 흔드나"를
알려줌. 2026-07-13에 이 값이 167(= |Δa| ~ 0.68)로 발산을 알리고 있었는데 흘려봤음.

## 교훈: 상태(state)가 아니라 제어 입력(action)을 볼 것

손이 87cm 솟구친 것을 보고 "물리 튕김"이라 단정했으나 **정책이 그렇게 명령한 것**이었음.
**결과만 보고 원인을 단정하면 잘못된 진단 위에 헛수고를 쌓게 됨.**

## Action이 PhysX까지 가는 경로 (요약)

```
정책 a = N(mu, sigma).sample()          MLP 마지막이 nn.Linear -> 상한 없음
  -> [clip_actions]                     vecenv_wrapper.py:151-154   ← rsl_rl_cfg.py에서 설정
  -> ManagerBasedRLEnv.step()           manager_based_rl_env.py:173
       process_action(a)                step당 1번.  scale 적용
       for _ in range(decimation):      물리 step마다 반복
           apply_action()               같은 목표를 다시 씀
           write_data_to_sim()
           sim.step()
  -> JointAction.process_actions()      joint_actions.py:169-179
       processed = a * scale + default_joint_pos
       [JointActionCfg.clip]            두 번째 clip (우리는 미사용)
  -> set_joint_position_target()        articulation.py:1079
  -> write_data_to_sim()                articulation.py:218  (PhysX GPU)
  -> PD:  tau = 100*(목표-실제) - 20*속도    indy.py arm 액추에이터
```

**`process_action`은 step당 1번, `apply_action`은 decimation번.**
정책이 준 목표를 decimation번 동안 고정한 채 PD가 밀어붙임 -> 그동안 정책은 눈감고 있음.

### 값이 정해지는 곳
```
clip_actions = 1.0   learning/rsl_rl_cfg.py:21   -> scripts/rsl_rl/train.py:203 에서 wrapper로
scale = 1.0          grasp/indy_wuji/env_cfg.py:64
decimation = 2       grasp/cube_grasp_env_cfg.py:102
stiffness/damping    assets/indy.py:183~211
```

**상세는 `/home/lsc/wuji_indy_lab_51/flow_study.md` §24-25 참고.**

## `palm_facing` — 이름과 달리 "손바닥 법선"이 아님 (2026-07-13)

```python
palm_normal_b = (0.19, 0.28, 0.94)   # env_cfg_common.py + managers.py, 반드시 동일
palm_facing weight = 20.0
```

**이건 손바닥 법선이 아니라 "파지 개구부 방향"임** — 물체가 엄지-손가락 사이로 들어가는 공간.
둘은 palm_link 로컬에서 **65도 어긋나 있음.**

| | 방향 (palm 로컬) | 의미 |
|---|---|---|
| 손바닥 법선 | (0.97, 0.00, 0.26) ~= (1,0,0) | 손바닥이 향하는 쪽, 손가락이 굽는 쪽 |
| **파지 개구부** | **(0.19, 0.28, 0.94)** | 물체가 손가락 사이로 들어가는 공간 |

**파지의 조건은 "손바닥이 향하는가"가 아니라 "물체가 손가락 사이에 들어갈 수 있는가"임.**

### 검증 (무작위 자세 102,400개)
```
축                              hold 상관계수   face>0.9시 hold최대
(1,0,0) 손바닥 법선                  +0.003          0.0000   <- 무상관. weight 올리면 학습이 죽음
(0.19,0.28,0.94) 파지 개구부         +0.105          0.6381
```
`(1,0,0)`으로 weight 60을 줬더니 `cage_inside`가 0.20 -> 0.000으로 붕괴함.

### 고정 벡터로 충분함 (실시간 계산 불필요)
palm_link 로컬 벡터라 팔이 움직여도 안 바뀜 (`quat_apply`로 매 step 월드 변환).
손가락 자세에 따라선 바뀔 수 있으나 **실측상 고정과 실시간이 동일** (상관 +0.105 vs +0.103).

### **약한 신호임 — 과신하지 말 것**
```
hold와의 상관계수 = +0.105
```
**진짜 일은 `hold`가 함.** palm_facing은 초반 shaping 보조. weight를 올려서 뭔가 되길 기대하면 안 됨.

### weight 규칙 (반드시 지킬 것)
```
palm 최대 획득량 < reach 최대 획득량   <- 넘기면 겨누기가 접근보다 비싸져 standoff
weight 60 -> palm 0.24 > reach 0.13  -> standoff (실측)
weight 20 -> palm 0.081 < reach 0.13 -> 안전
```

## 다음 계획 (2026-07-13 확정) — 게이팅이 실패하면 논문을 제대로 구현

**지금 겪는 문제가 전부 "목표 파지 자세가 없어서" 생긴 것임.**
정육면체는 대칭이라 목표 회전을 정의할 수 없음 -> `r_hr`을 못 씀 -> 방향 문제를 게이팅으로 때움.

### 바꿀 것
```
물체:    정육면체 -> 직육면체 (예: 0.04 x 0.06 x 0.10)  -> 목표 파지 회전이 해석적으로 정의됨
보상:    + r_grasp = r_hp + r_hr + lambda*r_hj   (목표 파지 g를 향한 차분)
        + r_orient (물체를 nominal 자세로)  + r_T (도달 성공, 희소)
        - 게이팅 제거 (r_hr이 대신함)
가중치:  논문 그대로  r_reach 1 / r_hold 25 / r_orient 500 / r_T 5000
학습:    커리큘럼 2단계
        close-start: 물체를 손 5cm 앞 + 팔은 manip 높은 중립 자세 + r_man 끔 -> 성공률 50%까지
        후속 단계: 전체 난이도
```

**`r_orient`가 진짜 의미를 가짐:** 직육면체가 누워 있으면 먼저 세워야 잡을 수 있음
= 논문 제목의 **pre-grasp manipulation**. 정육면체로는 논문의 절반을 못 쓰고 있었음.

**젓가락으로 가는 정석 경로임.** 젓가락은 목표 파지가 본질적으로 정의됨
(논문의 constraint-based 표현: "3D target position of the index fingertip + end-effector rotation").
지금의 임시방편(게이팅, palm_facing 축 추측)은 젓가락에선 전부 버려짐.

**상세는 `/home/lsc/wuji_indy_lab_51/study.md` 참고.**

## ★ 2026-07-14: cube grasp 포기. action space가 논문과 다른 게 근본 원인

### 논문의 action space (Section III-B)
```
action = 상대 변위:  손 위치 3 + 손 회전 3(Euler) + 손가락 5  =  11D
팔 관절은 IK로 계산.  관절은 PD 제어.
30 Hz.  v_hp_max = 1 m/s -> step당 최대 3.3cm
로봇: 6 DoF UR5e + 11 DoF Schunk SIH (커플링되어 5 DoF만 제어)
```

### 우리
```
action = 절대 관절각:  팔 6 + 손가락 12  =  18D
IK 없음.  scale x clip_actions로 크기를 수동 제한.
Wuji 손 12 DOF (커플링 없음)
```

### 이 차이가 낳은 것 (전부 2026-07-13~14에 터짐)
- **scale/clip_actions 발산**: 논문 action("손을 3cm 옮겨라")은 물리적으로 유계.
  우리 action("관절을 X rad로")은 scale이 작으면 정책이 |a|를 키움 -> 상한 없으면 발산 (|Δa| 9.66)
- **arm_manipulability 페널티가 불필요**: IK가 팔을 풀어주므로 특이점에 빠질 수 없음
- **palm_facing 축 삽질**: 관절을 명령하면서 손 방향을 보상하려 해서 생긴 문제.
  논문은 손 pose를 명령하고 손 pose를 보상함 (r_hp/r_hr)
- **손가락 12 DOF**: 논문 5 DOF보다 탐색 공간이 훨씬 큼

**-> functional_grasp에서는 `DifferentialInverseKinematicsActionCfg`로 갈 것.**
IsaacLab Lift-Cube에 `ik_abs_env_cfg.py` / `ik_rel_env_cfg.py` 예제 있음.

### 그리고: 정육면체는 목표 파지를 정의할 수 없음
논문은 `g = (hp, hr, hj)`를 알고 `r_hr`이 방향을 직접 보상함.
**대칭인 정육면체는 목표 회전이 정의 불가** -> 대리 지표(palm_facing 축, 게이팅, closability)를
발명했고 전부 뚫림. 직육면체/젓가락에선 이 문제가 없음.

## ★ 검증 규칙 (2026-07-14에 비싸게 배움)

**속일 수 없는 신호는 `cube_clearance`(물체가 실제로 떴는가) 하나뿐이다.**
`hold`, `cage_inside`, `opposition`, `palm_facing` 전부 정책에게 뚫렸다.
(무작위 자세 81,920개 중 `hold>0.1`인 48개를 실제로 오므려보니 **87%가 오므리면 hold가 내려감**
= 손끝이 물체에서 멀어짐 = 퇴화 자세)

```
1. 새 보상/지표를 만들면 최종 목표와의 상관계수부터 재라.  5분이면 됨.
2. "그 상태가 존재하는가"를 샘플링으로 먼저 확인하라.
   (hold>0.1 AND palm_facing>0.7 은 61,440개 중 0개였다 — 존재하지 않는 걸 학습시키고 있었음)
3. "보상은 오르는데 지표는 나빠진다"가 보이면 무조건 허점이 있다.
4. 상태(state)가 아니라 제어 입력(action)을 봐라.  (play.py --print_action)
5. 정책이 지표를 거스르면, 정책이 아니라 지표를 의심하라.
6. 차분형에 곱셈 게이트를 걸 땐 반드시 부호를 나눠라.
   (음수에 곱하면 telescoping이 깨져서 "왕복 farming"이 생김)
```

## 2026-07-14 contact/lift probe

- repo 내부 확인 스크립트는 `scripts/debug/check_cube_contact_lift.py`임.
- policy 없이 `reset -> zero settle -> finger close -> optional arm lift`를 실행함.
- thumb/index/middle/palm 링크별 cube contact force를 출력함.
- `GOOD_CONTACT thumb+middle`과 `max_clearance(m)`를 같이 봄.
- 실행 명령:

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

- 2026-07-14 검증 결과 close-only는 `GOOD_CONTACT=True`가 늦게 켜졌지만 `max_clearance=0.0003m`라 lift 실패임.
- 긴 학습 전에는 `--sweep-lift` 또는 cube pose sweep으로 `GOOD_CONTACT=True`와 `max_clearance>0.005m` 조건을 먼저 찾아야 함.
- cube를 움직이기 전에는 `--sweep-fingers`로 thumb/index/middle close 값 조합부터 확인함.
- 특정 조합은 `--finger-action`으로 직접 확인함. 예: `--finger-action 1 0 1`.
- contact 판정은 `--contact-mode thumb_middle`, `thumb_index`, `thumb_any`, `tripod`으로 바꿈.

## Isaac Sim 측정 시 함정 (2026-07-14 파지 검증에서 실측)

- **물체를 매 step `write_root_state_to_sim`으로 고정하지 말 것.** 접촉 관통이 누적되어
  PhysX가 손가락을 관절 한계 밖으로 폭발시킴. "쥐여주기"는 중력 차단(`set_disable_gravities`)
  + 매 step 속도만 0 (`write_root_velocity_to_sim`)으로. 위치는 물리에 맡김
- **`finger*_tip_link` 원점은 손끝 패드가 아니라 마지막 관절 위치** (패드보다 2~3cm 손바닥 쪽).
  tip 원점 중점에 물체를 놓으면 손바닥/접힌 손가락 위에 얹힘
- **Indy7 joint1은 감소(-0.45→-0.95)가 손 '하강' 방향.** 들기 테스트 부호 주의
- **판정 기준은 조작 방향과 함께 검증할 것.** "z0 유지" 기준은 팔이 내려가면 성공도 실패로 판정
- **불가능한 숫자가 나오면 물리 폭발부터 의심.** 관절 오차가 (목표상한-관절하한)보다 크면
  인덱싱이 아니라 관통 폭발임
- **Wuji 손 기하 (FK 실측):** 엄지-중지 물리는 창은 오므림 30~50%뿐 (40%에서 최소 2.8cm,
  70%에서 6.3cm로 재벌어짐). 60%+ 조이면 6cm 큐브를 짜냄. 검지-중지는 항상 붙어 다님
- 검증 도구: `scripts/debug/hand_geometry.py` (FK 간격 곡선), `scripts/debug/grip_capacity.py`
  (쥐여주기→놓기→들기, `--gui` 지원)

## Wuji 손가락 액추에이터 스펙 (indy7_wuji_right.urdf 실측, 2026-07-14)

- URDF `<limit effort>` 관절별 값: joint1=1.0 (엄지 0.6), joint2=0.2 (엄지 0.6), joint3/4=0.3 N·m
- velocity: 8.2~13.6 rad/s
- sim의 일괄 `effort_limit=0.6`은 이 0.2~1.0 범위의 근사 평균. Allegro(0.7)/LEAP(0.9)와 동급
- **이 스펙(0.6)으로 0.30kg 하중 유지 검증됨 → 힘 부족은 파지 실패의 원인이 아님**
  - 단, 그 "4/4"의 실체는 palm-up 자세의 새끼/손바닥 받침 hold (GUI 확인: 엄지·중지 미접촉)
  - 엄지+중지 집게(pinch)는 미검증. 새 자세(joint3/4=-1.61/-1.62)+편 손가락+집게 판정에선 0/32
  - 판정 코드에 "무엇으로 잡았는지"(엄지·중지 거리)를 반드시 포함할 것 — held만 보면 속음
- effort를 크게 올리면(예: 6.0) 나쁜 자세를 힘으로 버티는 crush 정책이 나올 수 있음 (실기 재현 불가)
- 개선하려면 일괄 0.6 대신 URDF 관절별 값(1.0/0.2/0.3)을 넣는 것이 맞고, 단독 변수로 A/B 할 것
