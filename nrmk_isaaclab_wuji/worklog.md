# Worklog

- 이 문서는 `nrmk_isaaclab_wuji` repo 내부 변경 이력과 실행 결과를 남기는 작업 로그 문서임.

## 2026-07-08

- Confirmed active baseline is existing `/home/lsc/IsaacLab` + `env_isaaclab` + Isaac Sim 5.1.
- Preserved shared `/home/lsc/IsaacLab` and `/home/lsc/isaacsim_pkg`.
- Created new working extension at `/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji`.
- Copied Neuromeka public extension as Indy7/IsaacLab base.
- Added Wuji hand assets from existing Retargeting/GeoRT-style asset folders.
- Built combined URDF: `indy7_wuji_right.urdf`, fixed joint `tcp -> palm_link`.
- Generated and repaired USD assets so stage is `Z-up`, `metersPerUnit=1.0`.
- Added IsaacLab asset config `INDY7_WUJI_RIGHT_CFG` and task registration `Indy-Wuji-Reach`.
- Created three USD collision variants:
  - full mesh: `indy7_wuji_right.usd`
  - arm simplified + hand mesh: `indy7_wuji_right_simplified.usd`
  - arm simplified + reduced hand Cube colliders: `indy7_wuji_right_all_simplified.usd`
- Removed temporary/intermediate wrapper USD backups from the asset folder.
- Notes for chopstick RL: hand collision fidelity matters; start with arm-simplified + hand mesh, then compare all-simplified if mesh collision is unstable.
- Inspected the three USD variants directly with `usd-core` (temp venv, not the project env) to check what "mesh collision" actually means:
  - Earlier inspection suggested `indy7_wuji_right.usd` and `indy7_wuji_right_simplified.usd` had hand convexHull collision, but later Isaac Sim/USD validation showed the hand `collisions` Xforms in `indy7_wuji_right_simplified.usd` were effectively empty/instanced and needed explicit post-processing. See the later fidelity asset fix entry below.
  - Risk found: convex hulls bulge over concave dips (knuckle/joint sockets), so adjacent finger links can start overlapping at tightly curled poses (chopstick grip) even though the source meshes don't overlap there. Likely explanation for the "mesh collision instability" concern.
  - `indy7_wuji_right_all_simplified.usd`: cube colliders measured smaller than the visual mesh bbox on every hand link, worst at fingertips (~58-66% of visual size), better at the palm (~72-81%). Confirmed by bbox comparison script, not just visual inspection.
  - No `contact_offset`/`rest_offset` authored anywhere (USD or `indy.py`), so PhysX default (~0.02 m contact offset) applies to hand parts that are only ~0.015-0.02 m across — likely too large for chopstick-scale contact precision; flagged for later tuning via `CollisionPropertiesCfg`.
- Decided on a two-tier USD strategy for training, following the same active/commented-alternate pattern already used for `INDY7_CFG`:
  - Quick-start tier, initially active for smoke testing: `indy7_wuji_right_all_simplified.usd` — arm + hand both simplified, used to get the RL training pipeline running end-to-end first, precision sacrificed.
  - Fidelity tier, now active after the collision post-process below: `indy7_wuji_right_simplified.usd` — arm collision simplified, hand collision restored from Wuji collision STL meshes as convex hulls.
  - `indy7_wuji_right.usd` (full mesh incl. arm) stays as reference baseline only, not part of the two active tiers.
  - Changed `usd_path` in `isaac_neuromeka/assets/indy.py` (`INDY7_WUJI_RIGHT_CFG`) accordingly.
- Verified `indy7_wuji_right_all_simplified.usd` structurally (articulation root, RigidBodyAPI + mass on every link, joint connectivity/limits) with the same `usd-core` venv — no physics-level blocker found.
- Found and fixed a real training blocker unrelated to collision fidelity: `isaac_neuromeka/tasks/manipulation/reach/indy_wuji/env_cfg.py` had reward/command `body_name(s)` set to `"tcp"`, but `tcp` is only a non-rigid leftover Xform frame under `link6` (no `RigidBodyAPI`) in all three USD variants — the `tcp -> palm_link` fixed joint from the URDF apparently got merged away during USD generation, so no body named `tcp` exists in the articulation at all. This would have crashed env creation regardless of which USD tier was active. Changed all four references (`end_effector_position_tracking`, `end_effector_orientation_tracking`, `end_effector_speed`, `commands.ee_pose.body_name`) to `"palm_link"`, the real attachment body for the wuji hand.
- Cleaned up the throwaway `usd-core` venv used for the inspection above (was only in the scratchpad, never part of the repo).
- Confirmed the actual run setup for training:
  - `env_isaaclab` gets Isaac Sim via the `isaacsim` pip metapackage (5.1.0.0), not `/home/lsc/isaacsim_pkg` (unrelated standalone extraction). `isaaclab` is editable-installed from `/home/lsc/IsaacLab/source/isaaclab`.
  - `isaac_neuromeka` was not yet installed in `env_isaaclab`; ran `pip install -e .` from `nrmk_isaaclab_wuji/` (package name `nrmk_isaaclab_public`) — succeeded, but downgraded `pandas` 3.0.3 -> 2.3.3 in the env to satisfy the package's pin.
  - Verified that a bare `import isaac_neuromeka` fails on `pxr` before `AppLauncher` runs — expected IsaacLab behavior, not an error to chase.
- Ran `python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 32 --max_iterations 5` as a smoke test. Two failures found and fixed along the way, both unrelated to collision fidelity:
  1. `rsl-rl-lib==5.0.1` is installed (matches `isaaclab_rl`'s own pinned extras version), but `scripts/rsl_rl/train.py`/`play.py` never called `isaaclab_rl.rsl_rl.handle_deprecated_rsl_rl_cfg()` — the function that migrates the deprecated `policy=` config field into the `actor`/`critic` `RslRlMLPModelCfg` fields rsl-rl-lib >= 4.0.0 actually reads. Upstream IsaacLab's own reference `scripts/reinforcement_learning/rsl_rl/{train,play}.py` calls it right after `cli_args.update_rsl_rl_cfg(...)`; ours didn't. Added the same call (plus the `installed_version = metadata.version("rsl-rl-lib")` it needs) to both scripts. This is a repo-wide gap — also affects `Indy-Reach` and `Dual-Arm-Reach`, only our two scripts were patched.
  2. (Tried first, reverted): manually setting `actor=`/`critic=` `RslRlMLPModelCfg(...)` directly in `ReachPPORunnerCfg` instead of relying on the migration above. This skips the auto-derivation from `policy=` and leaves the deprecated `stochastic`/`init_noise_std` fields dangling at `MISSING`, which surfaces as `TypeError: MLPModel.__init__() got an unexpected keyword argument 'stochastic'`. Reverted `indy_wuji/learning/rsl_rl_cfg.py` back to `policy=`-only; the migration call in `train.py`/`play.py` derives `actor`/`critic` correctly instead.
  - After both fixes, the smoke test ran clean end-to-end: scene/managers built correctly (5 reward terms incl. the `palm_link`-based ones), 5 PPO iterations completed, no NaNs/crashes, reward ~0.69-0.71, position/orientation error decreasing. This confirms the `all_simplified` USD + current task config is training-ready for a first real run.
- Also installed `pip install -e .` (`nrmk_isaaclab_public`) into `env_isaaclab` earlier in this session; noted the `pandas` 3.0.3 -> 2.3.3 downgrade side effect.
- User reported (via their team lead) that importing `indy7.urdf` in Isaac Sim shows the usual green collision boxes, but importing the standalone `wuji_right/wuji_right.urdf` shows none. Checked the file directly: 21 of its 26 links (`palm_link`, all `finger*_link1-4`) had `<collision>` wrapped in an XML comment (`<!-- <collision>...</collision> -->`); only the 5 `finger*_tip_link` collisions were live. The referenced `*_collision.STL` mesh files all exist on disk (both raw and pre-computed `.convex.stl`), so this was purely a disabled/commented block, not missing geometry.
  - Cross-checked the combined `indy7_wuji_right.urdf` (the file actually used to generate the shipped `.usd` assets) — its collision blocks were never commented out, so this bug never affected the training assets already verified working earlier in this log.
  - Fixed `wuji_right/wuji_right.urdf` by uncommenting all 21 blocks (kept a backup at `/tmp/.../scratchpad/wuji_right.urdf.bak` during the edit, not committed anywhere). Verified with `xml.etree.ElementTree` that the file still parses and all 26 links now have a `<collision>` element.
- After the URDF fix, user re-imported `wuji_right.urdf` in the Isaac Sim GUI with an on-disk destination path, and both visuals AND collisions came back empty for every link (confirmed directly: opened the generated `urdf/wuji_right/wuji_right/configuration/{wuji_right_base,wuji_right_physics}.usd` layers with `usd-core` and found `palm_link/visuals` and `palm_link/collisions` Xform groups present but with 0 children on every link).
  - Root-caused via a headless repro using the same `isaacsim.asset.importer.urdf` commands (`URDFCreateImportConfig` + `URDFParseAndImportFile`) the GUI uses: importing straight into an in-memory stage (`dest_path=""`) populates every mesh correctly (verified point counts: collision meshes all 1500 pts = 500 tri x 3, matching the known 500-tri collision STLs; visual meshes vary correctly per link). The Isaac Sim log explicitly prints `"Creating Asset in an in-memory stage, will not create layered structure"` for that path.
  - Conclusion: this is an **Isaac Sim 5.1 URDF importer bug in the on-disk "layered structure" export path** (the one that writes `configuration/{base,physics,robot,sensor}.usd`), not a problem with `wuji_right.urdf` or with our earlier fix. The URDF itself was independently re-verified clean (all 26 links have correct paired visual/collision STL references).
  - Workaround: import with no destination/output directory set (import into the open stage in-memory), then `File → Save As` to write the USD out, instead of using the importer's on-disk destination-path option directly.
- Fixed the actual training fidelity asset `isaac_neuromeka/assets/model/usd/indy7_wuji_right/indy7_wuji_right_simplified.usd` with a reproducible post-process:
  - Added `scripts/assets/apply_wuji_hand_collision_meshes.py`.
  - Backed up the pre-fix USD as `indy7_wuji_right_simplified.before_hand_collision_fix.usd`.
  - Inserted 26 Wuji hand collision meshes from `isaac_neuromeka/assets/model/urdf/wuji_right/meshes/*_collision.STL`.
  - Applied `PhysicsCollisionAPI`, `PhysxCollisionAPI`, `PhysxConvexHullCollisionAPI`, and `MeshCollisionAPI(approximation="convexHull")` to the direct hand collision mesh prims only.
  - De-instanced the empty hand `collisions` Xforms and blocked active nested importer leftovers under the direct collision mesh prims.
  - Validation result: 26 direct Wuji collision meshes, 0 active nested children under those meshes, 0 CollisionAPI under Wuji visuals; full USD collision list is now 9 Indy arm primitive colliders + 26 Wuji hand mesh colliders.
  - Switched `INDY7_WUJI_RIGHT_CFG` active `usd_path` from `indy7_wuji_right_all_simplified.usd` to `indy7_wuji_right_simplified.usd`; `all_simplified` remains commented as fallback/debug only.
  - Attempted minimal headless check with `conda run -n env_isaaclab env PYTHONPATH=/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 1 --max_iterations 1`. It did not reach env/USD creation in the Codex sandbox because CUDA/GPU access was unavailable and `pynput` could not acquire the X display connection (`DISPLAY=:1`, operation not permitted). Treat this as an execution-environment limitation, not as a USD validation failure.
- Matched the `Indy-Wuji-Reach` config structure to the existing Neuromeka `Indy-Reach` pattern:
  - Added `Indy7WujiReachStudentEnvCfg` and `Indy7WujiReachCMDPEnvCfg` alongside the existing base and teacher configs.
  - Left only `Indy-Wuji-Reach` actively registered; added commented future registration blocks for teacher/student/CMDP variants to mirror the existing `Indy-Reach` layout without changing runtime behavior.
  - Verified syntax with `python -m py_compile isaac_neuromeka/tasks/manipulation/reach/indy_wuji/env_cfg.py isaac_neuromeka/tasks/manipulation/reach/indy_wuji/__init__.py`.
- Reduced `Indy-Wuji-Reach` to arm-only observations/regularization for the initial arm end-effector tracking phase:
  - `policy/joint_pos` and `policy/joint_vel` now use only `joint[0-5]`, while hand joints remain in the articulation but are not exposed to the policy observation.
  - `joint_vel` reward regularization now also applies only to `joint[0-5]`, avoiding penalties on hand joints that are not controlled by the current action term.
  - Future teacher/student proprioception and privileged joint friction/damping terms are also scoped to `joint[0-5]`.
  - Set `sim.render_interval = decimation` in the Indy-Wuji cfg to remove the render interval mismatch for this task.
  - First attempt reused one `SceneEntityCfg` across multiple terms, which failed because IsaacLab resolves and mutates `joint_ids` per term. Fixed by creating a fresh `SceneEntityCfg("robot", joint_names=["joint[0-5]"])` for each term.
  - Verified syntax with `python -m py_compile nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/reach/indy_wuji/env_cfg.py`.
  - Verified smoke run with `conda run -n env_isaaclab python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 1 --max_iterations 1`: action shape stayed 6, policy observation shape changed from 175 to 55 (`joint_pos=18`, `joint_vel=18`, `pose_command=7`, `action_history=12`), actor/critic input features changed to 55, and 1 PPO iteration completed.
- Root handoff/study docs updated:
  - `/home/lsc/wuji_indy_lab_51/AGENTS.md` 최신화함.
  - `/home/lsc/wuji_indy_lab_51/WORKLOG.md` 최신화함.
  - `/home/lsc/wuji_indy_lab_51/study.md` 생성함.
  - 모든 새 문서는 개괄식으로 작성함.
  - `study.md`에는 Direct/ManagerBased 차이, Franka/Neuromeka 차이, Indy-Wuji 구조, observation/reward/command/asset 요약, 실행 명령, 학습 로그 확인 기준을 정리함.
- Fixed `scripts/rsl_rl/play.py` for the installed IsaacLab 2.3.2 environment:
  - `isaaclab.utils.pretrained_checkpoint` is not available in this env.
  - Added fallback import from `isaaclab_rl.utils.pretrained_checkpoint`.
  - Verified syntax with `python -m py_compile scripts/rsl_rl/play.py`.
  - Latest checkpoint found at `logs/rsl_rl/indy_wuji_reach/2026-07-08_18-16-06/model_99.pt`.
- Fixed second `play.py` compatibility issue with `rsl-rl-lib==5.0.1`:
  - Old export code tried `runner.alg.policy` / `runner.alg.actor_critic`, but rsl-rl 5 `PPO` exposes `actor`, `critic`, and `get_policy()` instead.
  - Matched upstream IsaacLab 2.3.2 behavior: for rsl-rl >= 4.0.0, use `runner.export_policy_to_jit()` and `runner.export_policy_to_onnx()`.
  - Verified syntax with `python -m py_compile scripts/rsl_rl/play.py`.
  - Verified short play run with `conda run -n env_isaaclab python scripts/rsl_rl/play.py --task Indy-Wuji-Reach --headless --num_envs 1 --load_run 2026-07-08_18-16-06 --video --video_length 2`.
  - Result: checkpoint `model_99.pt` loaded, actor input stayed 55, action output stayed 6, and play loop completed.
- Added root `CLI.md`:
  - 학습 명령 일반화함.
  - GUI 실행 명령 일반화함.
  - checkpoint play 명령 일반화함.
  - resume train 명령 일반화함.
- Updated root `CLI.md`:
  - shell 변수 사용 제거함.
  - 각 명령을 바로 복붙 실행 가능한 형태로 변경함.
- Diagnosed Wuji EE orientation frame mismatch:
  - Raw `palm_link` tracking left orientation error near ~2 rad.
  - `link6` tracking reduced orientation error to ~0.96, showing the issue was frame alignment, not only learning.
  - Manual command offset tests found `roll=-pi/2`, `pitch=-pi/2`, free yaw reduced orientation error to ~0.18.
  - Conclusion: Wuji palm frame and Indy reach command frame need a fixed task-frame offset.
- Implemented the root task-frame fix without rotating the USD/URDF hand asset:
  - Added `OffsetUniformPoseCommandCfg` / `OffsetUniformPoseCommand` in `isaac_neuromeka/mdp/commands.py`.
  - Added offset-aware position/orientation tracking rewards in `isaac_neuromeka/mdp/rewards.py`.
  - Updated `Indy-Wuji-Reach` to use `wuji_ee = palm_link * WUJI_EE_OFFSET`.
  - Set `WUJI_EE_OFFSET_QUAT = (-0.5, -0.5, 0.5, 0.5)`.
  - Restored command ranges to the normal reach convention: `roll=0`, `pitch=pi`, `yaw=(-3.14, 3.14)`.
- Verified the virtual EE implementation with a minimal headless smoke run:
  - Command: `conda run -n env_isaaclab python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 1 --max_iterations 1`.
  - Command term type: `OffsetUniformPoseCommand`.
  - Action shape: 6.
  - Policy observation shape: 55.
  - `Metrics/ee_pose/orientation_error`: 0.5433.
  - 1 PPO iteration completed.
- Reverted/parked the virtual EE implementation per user decision:
  - Removed `OffsetUniformPoseCommandCfg` / `OffsetUniformPoseCommand`.
  - Removed offset-aware reward functions.
  - Restored `Indy-Wuji-Reach` to the common `UniformPoseCommandCfg` and raw `palm_link` tracking rewards.
  - Current plan is to run a long raw `palm_link` baseline first, then decide whether offset/reward changes are actually needed.
- Verified the raw `palm_link` baseline after the revert:
  - Command: `conda run -n env_isaaclab python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 1 --max_iterations 1`.
  - Command term type: `UniformPoseCommand`.
  - Action shape: 6.
  - Policy observation shape: 55.
  - `Metrics/ee_pose/orientation_error`: 0.9658.
  - 1 PPO iteration completed.
- Switched the active reach baseline from `palm_link` to `link6`:
  - Updated `Indy-Wuji-Reach` command body to `link6`.
  - Updated position/orientation/speed reward body names to `link6`.
  - Kept arm-only action and observation unchanged.
  - Purpose is to validate reach learning on a clean Indy arm flange frame before revisiting Wuji hand frame offsets.
  - Next long-run candidate is `--num_envs 4000 --max_iterations 50000`.
- Verified the `link6` baseline with a minimal headless smoke run:
  - Command: `conda run -n env_isaaclab python scripts/rsl_rl/train.py --task Indy-Wuji-Reach --headless --num_envs 1 --max_iterations 1`.
  - Command term type: `UniformPoseCommand`.
  - Action shape: 6.
  - Policy observation shape: 55.
  - `Metrics/ee_pose/position_error`: 0.1382.
  - `Metrics/ee_pose/orientation_error`: 0.9658.
  - 1 PPO iteration completed.
- User reported a longer training result:
  - Around 2000 iterations, orientation error dropped from the 2.x range to about 0.8.
  - `palm_link` and `link6` looked similar once training progressed.
  - Current interpretation: URDF/offset is not a confirmed root cause; insufficient training time and reward dynamics were likely important.
  - Continue long-run monitoring before adding offset code back.
- Cross-checked the Allegro reference structure:
  - `indy7_allegro_hand_right.urdf` has fixed `tcp` joint under `link6`.
  - `allegro_base_joint` attaches `allegro_base_link` under `tcp` with a fixed offset.
  - Existing Neuromeka `Indy-Reach` tracks `tcp`.
  - This supports the same conceptual split: tracking frame and physical hand body are separate.

## 2026-07-09

- Added root `flow_study.md`.
- 정리 범위는 `Indy-Wuji-Reach` 실행 흐름임.
- `train.py`에서 task import되는 흐름 정리함.
- `gym.register`와 `gym.make` 연결 정리함.
- `ReachEnvCfg`, `env_cfg_common.py`, `indy_wuji/env_cfg.py` 역할 구분함.
- command/action/observation/reward가 어떤 파일과 함수로 연결되는지 정리함.
- 핵심 코드 발췌와 의미를 개괄식으로 정리함.
- Updated `CustomRewardManager` to log weighted reward and raw unweighted reward separately.
- `Episode_Reward/*` is weighted reward.
- `Episode_Reward_Raw/*` is raw reward before weight.
- `Episode_Reward_Std/*` is weighted reward std.
- Switched `Indy-Wuji-Reach` entry point to `CustomManagerBasedRLEnv` so the custom reward manager is active.
- Reduced active policy observation to 15 dimensions.
- Current policy observation is arm joint position 6, command xyz 3, previous action 6.
- Removed joint velocity observation from the active policy group.
- Removed observation history and observation noise from the active policy group.
- Added `generated_position_commands` for position-only command observation.
- Removed active orientation tracking reward.
- Removed active end-effector speed penalty.
- Removed active joint velocity penalty.
- Active reward terms are now position tracking and action rate.
- Latest `--num_envs 1 --max_iterations 1` smoke test passed with actor/critic input 15 and action output 6.
- Next long run target is `--num_envs 4096 --max_iterations 50000`.
- Updated root `CLI.md` play commands to automatically use the latest run instead of hardcoding `--load_run 2026-07-08_18-16-06`.
- Studied reward-design papers for the next cube grasp phase.
- Current plan is to start cube grasp with oracle state reward/success before adding point cloud or force/tactile sensing.
- Oracle state here means simulator-known cube pose, fingertip pose, contacts, lift height, and velocities.
- Distinction recorded: oracle observation means feeding those states to the policy; oracle reward/success means using them only for training signals and metrics.
- `palm_link`-to-cube center distance alone is not sufficient as a grasp definition.
- Grasp success should combine contact groups, lift threshold, and object stability/velocity.
- Functional grasp / Dexterous Pre-grasp is the primary research direction for cube-to-chopstick grasp work.
- DexPoint is a helper reference, not the implementation target.
- DexPoint reward takeaway for this project is limited to fingertip reach, contact groups, contact-gated lift, lift-gated target reward, and action/velocity/controller penalties.
- DexPoint paper contact condition is thumb contact plus at least two other fingers; released-code style is softer, using at least two finger/palm contact groups.
- For `indy_wuji_right`, start with the softer group-count contact reward, then strengthen to thumb plus non-thumb finger contacts once learning works.
- TriFinger transfer paper is categorized as an object 6-DoF pose tracking reference rather than a grasp-acquisition reference.
- TriFinger object-goal reward uses 8 object/cube keypoints so position and orientation are represented by Euclidean keypoint distances.
- TriFinger reach reward is a progress term based on `curr_dist - prev_dist` with a negative weight, not a static closeness reward.
- TriFinger reaching is useful as early exploration shaping and should be reduced/disabled later so it does not block regrasping or finger gaiting.
- TriFinger fingertip velocity penalty is a useful stabilizing reference for later hand-control rewards.
- SimToolReal is categorized as the later tool-use/chopstick reference.
- SimToolReal reward structure is `r = r_smooth + r_grasp + I_grasped * r_goal`, with `r_grasp = r_approach + (1 - I_grasped) * r_lift`.
- SimToolReal's main lesson is object-centric goal-pose progress after lifting, using grasp bounding boxes/keypoints and goal pose trajectories.
- Working interpretation: Functional/Pre-grasp defines the main functional grasp objective, DexPoint provides contact/lift gate patterns, TriFinger teaches object pose control after interaction, and SimToolReal extends object pose control to tool-use trajectories.

## 2026-07-10

- Studied the Dexterous Pre-grasp Manipulation paper from the reward-design perspective.
- Classified it as the main reference for functional grasp preparation and later chopstick grasp.
- Cube grasp is treated as a proxy task for testing this flow before chopstick-specific assets/rewards are added.
- Reward structure noted: explicit grasp uses `r_grasp + r_man + r_MP + r_T`; constraint-based grasp additionally uses lift reward/condition.
- The main transferable reward idea is `r_man = r_reach + r_hold + r_orient`.
- `r_hold` is important because it rewards the object being inside the thumb-finger region, not just fingertip closeness or raw contact count.
- This hold/cage-style reward is now the main shaping idea for Wuji hand cube grasp.
- Explicit target grasp gives object-relative end-effector pose plus hand joint targets; it is easier to learn but needs per-object grasp definitions.
- Constraint-based target grasp gives functional conditions such as index fingertip target position and end-effector orientation; it is easier to define but needs lift to prevent fake success.
- Curriculum takeaway: first learn direct/near-nominal grasping, then expand to varied object poses and full pre-grasp manipulation.
- Updated interpretation: Pre-grasp/Functional grasp is the main target, DexPoint is a contact/lift-gate reference, TriFinger is object pose control, and SimToolReal is object-centric tool-use trajectory following.
- Updated implementation order: functional hold/cage reward, contact condition, contact-gated lift, TriFinger keypoint object-goal reward, then SimToolReal-style tool trajectory reward.
- Added root `thesis.md` as a paper-only reward study document.
- `thesis.md` collects DexPoint, TriFinger transfer, Dexterous Pre-grasp Manipulation, and SimToolReal notes without project implementation details or run commands.
- Added `Indy-Wuji-Cube-Grasp` task skeleton.
- Added `isaac_neuromeka/tasks/manipulation/grasp/cube_grasp_env_cfg.py`.
- `CubeGraspSceneCfg` inherits `ReachSceneCfg` and adds one cube.
- Cube is a `RigidObjectCfg` at `{ENV_REGEX_NS}/Cube`.
- Cube size is `0.06 m`.
- Cube mass is `0.08 kg`.
- Cube initial position is `(0.45, -0.18, 0.03)`.
- Registered cube grasp logs under experiment name `indy_wuji_cube_grasp`.
- Added Wuji finger alias convention: `finger1=thumb`, `finger2=index`, `finger3=middle`, `finger4=ring`, `finger5=little`.
- Kept code-level joint/body names as `finger[1-5]_joint[1-4]`, `finger[1-5]_link[1-4]`, and `finger[1-5]_tip_link`.
- Added root `code_write.md` for 2026-07-10 code writing notes.
- Fixed `indy.py` syntax error caused by misplaced finger actuator fragments.
- Wired the first cube grasp observation/reward baseline.
- Added `mdp.object_position_relative` for cube position relative to `palm_link`.
- Added `mdp.body_to_object_position_tracking_bounded` for bounded body-to-cube distance reward.
- `CubeGraspObservationsCfg` active policy observation is arm joint position, cube relative position, and previous action.
- `CubeGraspRewardsCfg` active rewards are `arm_cube_reach` and `action_rate`.
- `Indy-Wuji-Cube-Grasp` currently has no active command terms.
- `py_compile` passed for cube grasp cfg and related mdp files.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` passed.
- Smoke test confirmed action shape 6 and policy observation shape 15.
- Extended cube grasp control from arm-only 6D to arm + thumb/index/middle 18D.
- Controlled joint regex is `joint[0-5]` plus `finger[1-3]_joint[1-4]`.
- Added `mdp.object_position_relative_to_bodies` for cube-to-fingertip relative vectors.
- Added `mdp.bodies_to_object_position_tracking_bounded` for multi-body cube reach reward.
- Added `cube_in_fingertips` observation term for thumb/index/middle tips.
- Added `finger_cube_reach` reward term with thumb/index/middle weights `2.0/1.0/1.0`.
- Reduced `arm_cube_reach` weight to `0.05` so it acts as a coarse guide.
- Expected latest action shape is 18 and expected policy observation shape is 48.
- Isaac smoke test for the latest 18D action structure is left for the user to run.
- `py_compile` passed for the modified mdp and cube grasp cfg files.
- Relaxed reward parameters for early learning visibility.
- Increased `finger_cube_reach` weight from `0.2` to `0.3`.
- Increased `finger_cube_reach` distance_max from `0.3` to `0.5`.
- Reduced `action_rate` penalty weight from `-0.001` to `-0.0003`.
- `py_compile` passed after the reward-parameter change.
- During a 4096-env cube grasp run, PhysX reported GPU patch buffer overflow and requested roughly 171k patches.
- Increased `CubeGraspEnvCfg.sim.physx.gpu_max_rigid_patch_count` to `2**18`.
- `py_compile` passed for `cube_grasp_env_cfg.py` after the PhysX buffer change.
- Added TensorBoard cube distance error metrics in `CustomRewardManager`.
- New metrics are `Metrics/cube/palm_distance`, `thumb_distance`, `index_distance`, `middle_distance`, and `finger_mean_distance`.
- These metrics log actual palm/fingertip-to-cube distances and do not change reward, observation, or action shape.
- `py_compile` passed for `isaac_neuromeka/env/managers.py`.
- In the resumed 4096-env cube grasp run, PhysX requested roughly 263k rigid patches.
- Increased `CubeGraspEnvCfg.sim.physx.gpu_max_rigid_patch_count` from `2**18` to `2**19`.
- This change applies on the next train/resume launch.
- Earlier five-finger experiment extended cube grasp control from arm + thumb/index/middle 18D to arm + five fingers 26D.
- That experiment used controlled joint regex `joint[0-5]` plus `finger[1-5]_joint[1-4]`.
- The five-finger observation kept `cube_in_fingertips` at five fingertip links with shape 15.
- The five-finger policy observation shape was expected to be 70.
- Reprioritized cube grasp implementation toward the paper-inspired flow instead of debug-only expansion.
- Reverted cube grasp action from five-finger 26D to arm + thumb/index/middle 18D.
- Kept ring/little in observation/metrics as passive context, but removed them from the active fingertip reward.
- Added `cube_to_goal` observation for object-centric target movement.
- Latest policy observation shape is 57:
  - controlled joint position 18D
  - cube relative to `palm_link` 3D
  - cube relative to five fingertips 15D
  - cube target vector 3D
  - previous action 18D
- Changed `finger_cube_reach` back to controllable thumb/index/middle tips.
- Latest `finger_cube_reach` body weights are `(3.0, 1.0, 1.0)`.
- Added `cube_lift` reward.
- Added lifted-gated `cube_goal_tracking` reward.
- Current cube goal position is `(0.55, -0.05, 0.12)`.
- `cube_goal_tracking` is zero until cube root z reaches `0.08 m`.
- At that stage, active cube grasp reward terms were `arm_cube_reach`, `finger_cube_reach`, `cube_lift`, `cube_goal_tracking`, and `action_rate`.
- `py_compile` passed for the modified observation, reward, common cfg, cube grasp cfg, and Indy/Wuji grasp override files.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` passed after escalation for local GPU/display access.
- Smoke test confirmed action shape 18, policy observation shape 57, and reward term count 5.
- Smoke test log dir is `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_21-17-55`.
- Earlier five-finger reward experiment extended `finger_cube_reach` to finger1~5 tip links.
- That experiment used `finger_cube_reach` body weights `3.0/1.0/1.0/1.0/1.0`.
- Extended TensorBoard cube distance metrics to five-finger logging.
- Added `ring_distance`, `little_distance`, `non_thumb_mean_distance`, and `finger_weighted_mean_distance`.
- Changed cube grasp `finger_cube_reach` to a paper-inspired progress reward.
- Added `mdp.BodiesToObjectProgressReward`.
- The reward tracks per-env previous and best weighted fingertip-cube distance.
- Current cfg uses `mode="best"`.
- Reward progress is `previous_best_distance - current_distance`.
- Current distance is the thumb/index/middle weighted fingertip-cube distance.
- `finger_cube_reach` body weights remain `3.0/1.0/1.0`.
- `distance_max=0.5` is now the progress normalization scale.
- Reset/first step initializes the best distance to the current distance.
- `Episode_Reward_Raw/finger_cube_reach` now means best-distance improvement, not absolute closeness.
- Actual fingertip-cube distance is still read from `Metrics/cube/*`.
- `py_compile` passed for `mdp/rewards.py` and `env_cfg_common.py`.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` passed with the progress reward.
- Progress reward smoke test log dir is `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_21-31-26`.
- Added Dexterous Functional Grasp-inspired `functional_hold` reward.
- `functional_hold` rewards the cube entering the thumb/index/middle fingertip region and is the current proxy for the functional grasp hold/cage idea.
- `functional_hold` combines weighted fingertip-cube closeness, thumb opposition, and cube-to-grasp-center closeness.
- `functional_hold` body weights are `3.0/1.0/1.0`.
- `functional_hold` weight is `0.2`.
- `functional_hold` distance_max and center_distance_max are `0.18`.
- Active cube grasp reward terms are now `arm_cube_reach`, `finger_cube_reach`, `functional_hold`, `cube_lift`, `cube_goal_tracking`, and `action_rate`.
- `py_compile` passed for `mdp/rewards.py` and `env_cfg_common.py`.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` passed after adding `functional_hold`.
- Smoke test confirmed action shape 18, policy observation shape 57, and reward term count 6.
- Random-policy smoke test showed `functional_hold Raw = 0`, which is expected before the cube enters the grasp region.
- `functional_hold` smoke test log dir is `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_21-41-01`.

## 2026-07-10 GUI Playback Stutter

- 사용자가 Isaac Sim GUI `play.py` 실행 시 뷰포트가 끊긴다고 보고함.
- 첫 가설은 GPU 경합이었음. 당시 `Indy-Wuji-Cube-Grasp --num_envs 4096` 학습이 GPU 73~74%, 71도로 점유 중이었음.
- 이 가설은 틀렸음.
- 학습 프로세스를 종료한 뒤 `Indy-Wuji-Reach --num_envs 1` GUI play를 20초간 계측함.
- GPU utilization은 2~9%였고 가끔 36~39%로만 튐.
- GPU 온도는 61~62도, GPU clock throttle 없음.
- CPU는 36 스레드 중 1.2 코어만 사용, loadavg 1.59.
- swap in/out은 0으로 swap thrashing 없음.
- 즉 GPU/CPU/메모리 어느 것도 포화되지 않았는데 끊김이 남아 있었음. 처리량 문제가 아니었음.
- 진짜 원인은 render cadence였음.
- `isaac_neuromeka/env/rl_task_env_cfg.py:55`에서 `sim.dt = 1/120`임.
- `reach_env_cfg.py:108`에서 `decimation = 24`임. policy는 5 Hz로 돔.
- `indy_wuji/env_cfg.py`에서 `sim.render_interval = self.decimation`이라 24임.
- 따라서 뷰포트는 물리 24스텝마다 1프레임, 즉 시뮬레이션 시간 기준 초당 5프레임만 그림.
- GPU가 거의 놀고 있던 이유도 같음. 렌더 요청 자체가 드물었음.
- `sim.render_interval = decimation`은 학습에는 맞는 설정임. 불필요한 렌더를 건너뛰기 때문임.
- 다만 GUI play에서는 프레임 드랍처럼 보임.
- 수정은 `scripts/rsl_rl/play.py`에만 적용함. task/training cfg는 건드리지 않음.
- `--render_interval` CLI 인자를 추가함. 기본값은 `2`임.
- `env_cfg.sim.device` 설정 직후 `env_cfg.sim.render_interval = args_cli.render_interval`을 적용함.
- 프레임 수가 12배 늘어남. 물리 24스텝당 1프레임에서 2스텝당 1프레임으로 바뀜.
- 사용자가 뷰포트가 부드러워진 것을 확인함.
- GPU 여유가 충분해서 추가 부담은 없었음.

## 2026-07-10 Ruled Out Diagnostics

- 아래는 조사 후 원인이 아님을 확인한 항목임. 다시 조사하지 않기 위해 기록함.
- Isaac Sim 4.5는 현재 실행에 관여하지 않음.
- 환경변수, `~/.bashrc`, `~/.profile` 어디에도 4.5 경로 참조가 없음.
- `~/.nvidia-omniverse/logs/omni.kit.log`의 최근 3개 startup은 모두 `"appVersion":"5.1.0"`임.
- 같은 로그의 `4.5.0` 항목은 2026-07-07 날짜의 오래된 telemetry임. 이 로그는 append-only라 과거 이력이 남음.
- `/home/lsc/Downloads/isaac-sim-standalone-4.5.0-linux-x86_64`는 14 GB를 차지하지만 아무것도 참조하지 않음. root fs가 91% 사용 중이라 정리 대상 후보임.
- shader compile cache는 정상 동작 중임. `~/.cache/ov`가 2.0 GB이고 `shaders/nv_shadercache`, `shaders/shadercache`가 채워져 있음.
- 첫 실행 시 shader 컴파일로 인한 순간 끊김은 실재하지만, 이번에 보고된 지속적 끊김의 원인은 아님.
- `DISPLAY=:1`은 RTX 3090 위에서 도는 실제 Xorg 서버임. GLX renderer가 `NVIDIA GeForce RTX 3090`으로 보고됨.
- 따라서 software rasterization이나 가상 디스플레이 문제가 아님.

## 2026-07-10 Cube Grasp Checkpoint Mismatch

- cube grasp checkpoint를 GUI로 재생하려다 shape mismatch를 만남.
- `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_20-10-32/model_50.pt`는 action 26D, observation 70D로 학습된 것임.
- 현재 코드는 action 18D, observation 57D를 만듦.
- `runner.load()`에서 `size mismatch for mlp.0.weight: checkpoint [64, 70] vs current [64, 57]` 발생함.
- 원인은 그 run 이후 cube grasp를 five-finger 26D에서 thumb/index/middle 18D 구조로 되돌렸기 때문임.
- 즉 26D로 학습된 checkpoint는 현재 3-finger 코드로는 재생 불가함.
- 재생이 필요하면 코드를 five-finger로 되돌리거나, 현재 3-finger 구조로 새로 학습해야 함.
- 참고로 `env_cfg_common.py`의 `cube_in_fingertips` observation은 아직 `finger[1-5]_tip_link` 5개를 모두 씀.
- 반면 `finger_cube_reach` reward는 thumb/index/middle 3개만 씀.
- observation과 reward의 finger 범위가 서로 다름. 의도된 것인지 확인 필요함.

## 2026-07-10 Lift Reward Disabled

- Disabled `cube_lift` and `cube_goal_tracking` in `CubeGraspRewardsCfg`.
- Kept `cube_to_goal` observation to preserve the 57D policy observation shape.
- Active cube grasp rewards are now `arm_cube_reach`, `finger_cube_reach`, `functional_hold`, and `action_rate`.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` passed.
- Smoke test confirmed action shape 18, policy observation shape 57, and reward term count 4.
- Smoke test log dir is `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_22-25-03`.

## 2026-07-10 Cube Randomization And Higher Arm Start

- Added `reset_cube_position` reset event.
- Cube randomization is centered on default `(0.45, -0.18, 0.03)`.
- Cube randomization range is `x ±0.06`, `y ±0.08`, `z 0`.
- Overrode cube grasp arm initial posture only in `Indy7WujiCubeGraspEnvCfg`.
- Arm initial override is `joint1=-0.45`, `joint2=-1.85`, `joint4=1.20`.
- This keeps action shape 18 and policy observation shape 57 unchanged.
- The arm initial override changes the default action offset, so a fresh long run is cleaner than resuming an older checkpoint.
- `py_compile` passed for `env_cfg_common.py` and `grasp/indy_wuji/env_cfg.py`.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` passed.
- Smoke test confirmed reset events `reset_all` and `reset_cube_position`.
- Smoke test log dir is `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_22-30-24`.

## 2026-07-11 Wuji Finger Actuator Gain

- Confirmed Wuji finger actuator gain is `stiffness=8.0`, `damping=0.5`, `friction=0.02`.
- The gain applies to all `finger[1-5]_joint[1-4]`.
- This is a stabilization setting to reduce passive ring/little finger shaking.
- `py_compile` passed for `assets/indy.py`.

## 2026-07-11 Action Scale And Arm Reach Disabled

- Confirmed cube grasp action scale is `0.1`.
- Disabled `arm_cube_reach` to reduce palm over-guidance.
- Active cube grasp rewards are now `finger_cube_reach`, `functional_hold`, and `action_rate`.
- `py_compile` passed for `env_cfg_common.py` and `grasp/indy_wuji/env_cfg.py`.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` passed.
- Smoke test confirmed action shape 18, policy observation shape 57, and reward term count 3.
- Smoke test log dir is `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-11_13-11-03`.

## 2026-07-11 Progress Scale And Functional Hold Update

- Changed `finger_cube_reach` progress normalization scale.
- Updated `finger_cube_reach` `distance_max` from `0.5` to `0.03`.
- In this reward, `distance_max` is a progress scale, not the physical maximum distance.
- Changed `functional_hold` grasp center from thumb-weighted average to uniform fingertip average.
- Kept thumb weighting only for the fingertip-cube distance bonus.
- This avoids pulling the geometric grasp center too far toward the thumb.
- Changed `functional_hold` from a multiplicative gate to additive shaping.
- New raw form is `0.4 * center_bonus + 0.4 * weighted_distance_bonus + 0.2 * center_bonus * opposition`.
- The old `center_bonus * (...)` form could suppress distance/opposition signals when the cube was not near the center.
- The new form gives early signal from fingertip approach, cube-centering, and thumb opposition separately.
- `py_compile` passed for `mdp/rewards.py` and `env_cfg_common.py`.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` passed.
- Smoke test confirmed action shape 18, policy observation shape 57, and reward term count 3.
- Smoke test log dir is `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-11_13-38-25`.

## 2026-07-11 Cube Grasp Closeness Reward

- Checked GUI playback where the hand moved up and down instead of settling near the cube.
- Interpreted the issue as weak convergence pressure from progress-only reach reward, not a velocity-observation problem.
- Kept velocity observation/reward out for now.
- Kept action and observation dimensions unchanged.
- Added `finger_cube_closeness` reward.
- `finger_cube_closeness` uses absolute bounded distance from thumb/index/middle fingertips to cube root.
- `finger_cube_closeness` weight is `0.2`.
- `finger_cube_closeness` distance_max is `0.7`.
- Active positive cube grasp rewards are now `finger_cube_reach`, `finger_cube_closeness`, and `functional_hold`.
- The intended flow is progress reach, absolute closeness maintenance, then functional hold/cage shaping.
- `py_compile` passed for `env_cfg_common.py` and `rewards.py`.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` passed.
- Smoke test confirmed action shape 18 and policy observation shape 57 are unchanged.
- Smoke test confirmed active reward terms are `finger_cube_reach`, `finger_cube_closeness`, `functional_hold`, and `action_rate`.
- Smoke test confirmed `Episode_Reward_Raw/finger_cube_closeness` logs a non-zero value.
- Smoke test log dir is `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-11_14-08-08`.

## 2026-07-11 Finger Dimension Asymmetry Confirmed Intentional

- `cube_in_fingertips` observation은 5-finger 전부를 보는데 reward/action은 3-finger만 다루는 비대칭을 사용자에게 확인함.
- 사용자가 이 비대칭은 의도된 설계라고 확정함. 목적은 dimension을 낮추는 것임.
- 제어 대상은 `joint[0-5]` + `finger[1-3]_joint[1-4]` = 18D로 유지함.
- observation은 5-finger tip을 모두 포함해 57D를 유지함.
- ring/little finger는 policy 제어 대상이 아니고 actuator gain(`stiffness=8.0`, `damping=0.5`)으로만 안정화함.
- 따라서 obs/reward finger 개수를 맞추는 방향의 수정은 하지 않음.
- 코드 변경 없음. 문서 확정만 반영함.
- `agent.md`의 미확정 항목을 확정 항목으로 갱신함.

## 2026-07-11 Cube Grasp Structure Analysis

- 현재 cube grasp 구조를 코드와 실제 학습 로그로 교차 분석함.
- 분석 대상 run은 `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-10_22-33-40` (1352 iteration)임.
- 이 run은 arm_cube_reach가 살아있고 finger_cube_closeness가 없던 이전 reward 구성임.

### 실측 결과: local optimum 수렴 확인

- `Metrics/cube/finger_weighted_mean_distance`: `0.396` → `0.197` (iter 122) → `0.222`로 후퇴 → 1200 iteration 동안 고정.
- `Metrics/cube/palm_distance`: `0.319` → `0.214` (iter 122) → `0.242`로 후퇴 → 고정.
- `Train/mean_reward`: `-0.02` → `1.24` → 완전 평평.
- 느린 학습이 아니라 완전한 local optimum 수렴임.
- 큐브는 `0.06 m`인데 fingertip이 `0.22 m` 거리에서 멈춤. grasp 불가능한 거리임.
- 한 번 `0.197`까지 접근했다가 `0.222`로 후퇴한 것이 핵심 단서임.

### 원인 1: progress reward에 후퇴 벌칙 없음

- `finger_cube_reach`는 `mode="best"`라 최단거리 갱신 시에만 reward를 줌.
- 최단거리를 한 번 찍으면 그 뒤 물러나도 손해가 없음.
- 이것이 `0.197` → `0.222` 후퇴의 직접 원인임.
- 사용자가 2026-07-11에 추가한 `finger_cube_closeness`가 정확히 이 문제를 겨냥한 것이며 진단은 옳음.

### 원인 2: finger_cube_closeness distance_max가 과대

- 실측 fingertip-cube 거리 범위는 `0.05` ~ `0.40 m`임.
- 현재 `distance_max=0.7`은 reward의 `0.43` ~ `0.93` 구간만 사용함.
- `0.22 m`에서 grasp까지의 gradient가 step당 `0.048`에 불과함.
- `distance_max=0.3`으로 낮추면 동일 구간 gradient가 약 2.4배가 됨.
- dimension 변화 없음. 가장 저렴하고 확실한 수정임.

### 원인 3: functional_hold gate가 도달 범위 밖

- `center_distance_max=0.18`이라 grasp center가 큐브 `0.18 m` 안에 들어와야 reward가 켜짐.
- 정책은 `0.22 m`에 수렴해 있어 gate 밖임.
- `Episode_Reward_Raw/functional_hold`가 `0.56` (40 step 합, step당 `0.014`)로 사실상 0임.
- bootstrap이 불가능한 죽은 reward 항임.
- `0.3` ~ `0.35`로 넓혀야 gradient가 생김.

### 원인 4: action scale이 팔 도달 범위를 물리적으로 제한

- `CustomJointPositionAction`은 `target = default + scale * raw`인 절대 위치 명령임 (`mdp/actions/joint_actions.py:28,35`).
- `init_noise_std=1.0`이므로 자연스러운 action 범위(`|a|~1`)에서 각 관절은 시작 자세 `±0.1 rad` (`±5.7°`)만 이동 가능함.
- 6축 합산 시 palm 이동 가능 거리는 약 `0.07` ~ `0.10 m`임.
- 실측 palm 이동량 `0.077 m`와 정확히 일치함.
- 더 접근하려면 `|a| ~ 3-5`를 출력해야 하는데 entropy_coef `0.01`, desired_kl `0.01`, action_rate 패널티가 모두 이를 억제함.
- 2026-07-11에 action scale을 `0.2` → `0.1`로 낮춘 것은 이 문제를 악화시키는 방향임.

### 부수 관찰: action_rate 패널티가 무력함

- `Episode_Reward_Raw/action_rate`가 `21.6` → `95.6`으로 4.4배 증가함.
- weight `-0.0003`이면 전체 reward `1.24` 중 `-0.029`로 약 2%에 불과함.
- jitter를 전혀 억제하지 못함.
- finger stiffness를 `8.0`으로 올린 것은 같은 증상에 대한 대증요법이었음.

### 구조적 천장 (당장 급하지는 않음)

- contact sensor가 없어 현재 reward의 최적해는 큐브를 손가락 사이에 띄우는 hover-cage임.
- `functional_hold`는 근접도와 대향 기하학만 보고 접촉을 보지 않음.
- 실제 grasp로 넘어가려면 `ContactSensor` 도입이 필수임.
- 제어 주기가 5 Hz (`decimation=24`, `dt=1/120`), 에피소드가 40 step임. 손가락 폐합에는 거침.
- critic이 policy와 동일 obs를 봄. `CubeGraspTeacherObsCfg`가 있으나 미사용임.
- 네트워크가 `64x64`로 57D obs / 18D action 과제에는 작음.
- 테이블이 없음 (`reach_env_cfg.py:47` 주석 처리). 큐브가 ground plane 위에 있음.
- 큐브를 쳐서 날려도 termination이 없어 남은 에피소드를 낭비함.
- `cube_to_goal` observation이 죽어 있음. target이 고정 상수이고 `cube_goal_tracking = None`임.
- "checkpoint shape 유지" 근거는 해당 26D checkpoint가 이미 사용 불가라 더 이상 성립하지 않음.

### 우선순위 제안

- 1순위: `finger_cube_closeness` `distance_max`를 `0.7` → `0.3`으로. dimension 불변.
- 2순위: `functional_hold` `distance_max`/`center_distance_max`를 `0.18` → `0.35`로. dimension 불변.
- 3순위: action scale을 `0.1` → `0.2`~`0.3`으로 올리거나 arm/hand action term을 분리해 각각 다른 scale 적용.
- arm은 큰 scale, finger는 작은 scale이 필요하므로 단일 scale로 양쪽을 만족시키는 것은 원리적으로 불가능함.
- 3순위는 action offset이 바뀌므로 fresh run이 필요함.
- 아직 코드 변경 없음. 분석만 기록함.

## 2026-07-11 Structure Analysis CORRECTION

- 위 `2026-07-11 Cube Grasp Structure Analysis`의 원인 3, 원인 4가 틀렸음. 사용자가 지적해서 정정함.
- 오류 원인: 현재 코드의 파라미터 값을 2026-07-10 run에 그대로 대입해 분석함. 실제 run config를 확인하지 않았음.
- 교훈: 과거 run을 분석할 때는 반드시 `logs/rsl_rl/<exp>/<run>/params/env.yaml`을 읽을 것. 코드 현재 상태를 가정하지 말 것.

### 2026-07-10_22-33-40 run의 실제 config (params/env.yaml)

- `action scale = 0.2` (0.1 아님).
- `use_default_offset = True`, joints는 `joint[0-5]` + `finger[1-3]_joint[1-4]` (이미 18D였음).
- `arm_cube_reach`: `weight=0.05`, `distance_max=0.5`.
- `finger_cube_reach`: `weight=0.3`, `distance_max=0.5`, `mode=best`.
- `functional_hold`: `weight=0.2`, `distance_max=0.18`, `center_distance_max=0.18`.
- `action_rate`: `weight=-0.0003`.
- `decimation=24`, `episode_length_s=8.0`.

### 정정 1: action scale 천장 논리는 성립하지 않음

- 그 run은 `scale=0.2`였으므로 관절당 `±0.2 rad` 가용, palm 이동 예산은 약 `0.15-0.20 m`임.
- 실측 palm 이동량은 `0.077 m`로 예산의 절반도 못 씀.
- 즉 action scale 천장은 애초에 binding 제약이 아니었음. 정체 원인은 reward임.
- 이전 분석의 "원인 4"는 폐기함.

### 그럼에도 action scale은 0.2로 되돌리는 것이 맞음

- 정책이 `0.15-0.20 m` 예산 중 `0.077 m`를 썼는데 예산을 `0.07-0.10 m`로 반토막 내면 non-binding 제약이 binding 제약이 됨.
- codex가 scale 인하를 조언한 것은 action_rate가 `21` → `95`로 증가한 jitter를 보고 나온 것으로 추정됨.
- jitter의 직접적 레버는 `action_rate` weight이지 action scale이 아님.
- scale 인하는 jitter를 줄이는 대신 도달 범위까지 같이 줄이는 부작용이 있음.

### 정정 2: functional_hold는 죽은 항이 아님

- `Episode_Reward_Raw/functional_hold`: `0` → `0.40` (iter 150) → `0.55` → `0.56`으로 계속 우상향함.
- 이전 분석의 "사실상 0 / 죽은 항"은 틀렸음.
- 다만 40 step 만점이 `40.0`인데 `0.56`을 벌고 있음. 가용 reward의 약 `1.4%`임.
- 최근 300 iteration 동안 `0.55` → `0.56`으로 사실상 포화됨.
- 원인: `thumb_distance`가 `0.387` → `0.170` (iter 150, gate 안) → `0.199` (gate 밖)로 후퇴해 gate 경계 `0.18`에 붙어 앉음.
- 정확한 표현은 "죽음"이 아니라 "gate 경계에서 포화"임.
- 처방은 동일함. gate를 넓히면 꼬리 조각이 아니라 정책이 실제 점유하는 `0.19-0.40` 구간 전체에 밀도 있는 신호가 됨.

### gradient 정리 (bounded distance reward)

- `r(d) = 1 - clamp(d, 0, distance_max) / distance_max`. `d < distance_max`에서 직선, 그 밖은 평평한 `0`.
- 실효 기울기는 `weight / distance_max`임.
- `distance_max=0.7`, `w=0.2` → `0.286 /m`.
- `distance_max=0.35`, `w=0.2` → `0.571 /m`.
- `distance_max=0.3`, `w=0.2` → `0.667 /m`.
- PPO는 advantage `A = R - V(s)`로 학습하므로 매 step 상수 offset은 value function이 흡수해 상쇄됨.
- 따라서 reward의 절대 크기가 아니라 기울기만 학습 신호가 됨.
- `distance_max=0.7`이면 실측 거리 `0.19-0.40`이 reward `[0.43, 0.73]`으로 매핑됨. 출력 범위의 43%가 정보 없는 상수 바닥임.
- `distance_max=0.35`면 같은 거리 구간이 `[0, 0.46]`으로 펴져 상수 바닥이 사라짐.

### distance_max를 0.3이 아니라 0.35로 권하는 이유

- 에피소드 시작 fingertip-cube 거리가 약 `0.40 m`임.
- `distance_max=0.3`이면 시작 지점이 clamp 영역에 들어가 초반 gradient가 `0`이 됨.
- `0.35`면 대부분을 덮고, 남는 `0.35-0.40` 구간은 `finger_cube_reach` progress reward가 절대거리와 무관하게 개선량만 보고 커버함.

### 수정된 우선순위

- 1순위: `finger_cube_closeness.distance_max`를 `0.7` → `0.35`로. gradient 2배, 상수 바닥 제거.
- 2순위: `functional_hold`의 `distance_max`/`center_distance_max`를 `0.18` → `0.35`로. gate 경계 포화 해제.
- 3순위: `action_rate.weight`를 `-0.0003` → `-0.002`로. jitter의 직접 레버.
- 4순위: action scale을 `0.1` → `0.2`로 되돌림.
- 1-4 모두 action/observation dimension을 바꾸지 않음.
- 4순위만 action offset이 바뀌어 fresh run이 필요함.
- 아직 코드 변경 없음.

## 2026-07-11 Latest Code Is Reward Hacking (Regression Confirmed)

- 최신 run `logs/rsl_rl/indy_wuji_cube_grasp/2026-07-11_14-17-17` (88 iteration, 현재 코드)을 분석함.
- 결론: 현재 reward 구성은 reward hacking 상태이며 성능이 퇴행함.

### 동일 iteration 직접 비교

- 구코드 run은 `2026-07-10_22-33-40` (action scale `0.2`, `finger_cube_reach.distance_max=0.5`).
- 신코드 run은 `2026-07-11_14-17-17` (action scale `0.1`, `finger_cube_reach.distance_max=0.03`).
- iter 20: 구 `dist=0.402 reward=0.17` / 신 `dist=0.475 reward=0.89`.
- iter 40: 구 `dist=0.298 reward=0.37` / 신 `dist=0.412 reward=1.57`.
- iter 60: 구 `dist=0.236 reward=0.59` / 신 `dist=0.416 reward=1.86`.
- iter 88: 구 `dist=0.202 reward=0.79` / 신 `dist=0.440 reward=2.14`.
- 신코드는 reward가 2.7배 높은데 fingertip-cube 거리는 2.2배 멀음.
- 신코드 거리는 iter 40 이후 계속 악화 중임 (`0.403` → `0.412` → `0.416` → `0.440`).
- reward가 task metric과 역상관임. reward hacking의 정의에 해당함.

### 해킹 메커니즘

- `finger_cube_reach`는 `mode="best"` progress reward임.
- 에피소드 총합은 대략 `(d_1 - d_min) / distance_max`이며 step당 `1.0`으로 clamp됨.
- 즉 "어디서 끝나는가"가 아니라 "얼마나 이동했는가"를 지불함.
- 따라서 첫 step에 큐브 반대 방향으로 크게 움직여 `d_1`을 키우면 이득임.
- 실제로 에피소드 평균거리가 `0.403` → `0.440`으로 상승 중임. 초기 이탈 스윙과 일치함.
- 가까이 머무는 것에는 reward가 없음. 최단거리 갱신이 끝나면 `0`임.

### 근본 원인: 2026-07-11 변경이 착취 가능한 항을 16.7배 증폭함

- `finger_cube_reach` 기울기는 `weight / distance_max`임.
- 구: `0.3 / 0.5 = 0.6 /m`.
- 신: `0.3 / 0.03 = 10.0 /m`. 16.7배 증폭됨.
- 동시에 정직한 절대거리 항이 약화됨. `arm_cube_reach` (`w=0.05`)를 비활성화함.
- 대체재 `finger_cube_closeness`는 `distance_max=0.7` 탓에 기울기가 `0.2 / 0.7 = 0.286 /m`에 불과함.
- 착취항 `10.0 /m` 대 정직항 `0.286 /m`. 35배 차이로 착취항이 압도함.
- 이전 분석에서 `finger_cube_reach`의 `distance_max=0.03`을 "progress scale이라 별개"라며 넘긴 것이 오판이었음.

### 로그 증거

- `Episode_Reward_Raw/finger_cube_reach`: `0.078` → `0.657` 단조 상승. 착취항을 최적화 중임.
- `Episode_Reward_Raw/finger_cube_closeness`: `0.040` → `0.425` peak → `0.396` 하락. 정직항이 밀리는 중임.
- `Episode_Reward_Raw/functional_hold`: 40 step 만점 대비 `0.021`. 손이 `0.44 m`에 있는데 gate가 `0.18`이라 완전히 죽음.
- 참고: 구코드 run에서는 `functional_hold`가 `0.56`까지 올랐음. 신코드에서 오히려 죽은 것임.

### 원칙

- 절대거리 reward가 지배해야 하고, progress reward는 초기 bootstrap 보조일 뿐임.
- progress reward가 최대항이 되면 반드시 착취됨.
- `mode="best"` progress reward는 종료 상태가 아니라 이동량을 지불하므로 구조적으로 착취 가능함.

### 수정안 (dimension 불변, action scale만 fresh run 필요)

- `finger_cube_reach.distance_max`: `0.03` → `0.3`. 기울기 `10/m` → `0.33/m`.
- `finger_cube_reach.weight`: `0.3` → `0.1`. 추가 강등.
- `finger_cube_closeness.distance_max`: `0.7` → `0.35`.
- `finger_cube_closeness.weight`: `0.2` → `0.5`. 기울기 `0.286/m` → `1.43/m`. 주 구동항으로 승격.
- `functional_hold`의 `distance_max`/`center_distance_max`: `0.18` → `0.35`.
- action scale: `0.1` → `0.2`.
- `action_rate.weight`: `-0.0003` → `-0.002`.
- 수정 후 절대거리(`1.43/m`)가 progress(`0.33/m`)를 약 4배 지배함. 착취해도 손해가 되는 구조임.
- 아직 코드 변경 없음.

## 2026-07-11 Per-Step Trajectory Measurement (Supersedes Prior Two Analyses)

- 사용자가 GUI play에서 정책이 잘 수렴한다고 반박함. TensorBoard 지표와 모순되어 직접 측정함.
- 측정 스크립트로 checkpoint를 replay하며 환경의 `_compute_cube_distance_metrics()`를 매 step 호출함.
- 이전 두 분석(`Cube Grasp Structure Analysis`, `Latest Code Is Reward Hacking`)의 결론 일부를 뒤집음.

### 근본 오류: TensorBoard `Metrics/cube/*`는 에피소드 평균임

- `managers.py:255`에서 `_cube_metric_sums += metric * dt`, `managers.py:212`에서 `/ max_episode_length_s`로 정규화함.
- 정규화 자체는 정확함. 값은 진짜 미터 단위임.
- 그러나 20 step 전체의 평균이므로 "계속 멀리 있었다"와 "밖으로 나갔다가 마지막에 붙었다"를 구분하지 못함.
- 이전 분석의 "22cm local optimum 고착", "퇴행" 판정은 모두 이 평균에 속은 것임.
- 교훈: 접근/도달 과제에서 에피소드 평균 거리는 성능 지표가 아님. final/min 거리를 따로 봐야 함.

### 타이밍 실측

- physics step-size는 `1/60 s`임 (`0.01667`).
- environment step-size는 `0.4 s`임 (`decimation=24`).
- 즉 제어 주파수는 `2.5 Hz`이고 에피소드는 `20 step`임.
- 이전 분석에서 5 Hz / 40 step으로 적은 것은 틀렸음.

### 궤적 실측 결과

- 신 정책 `2026-07-11_14-17-17/model_50.pt` (scale `0.1`):
  - `t=0: 0.718` → `t=1: 0.837` (swing-out `+0.119 m`) → 단조 접근 → `t=19: 0.254`.
  - 에피소드 종료 시점에도 여전히 접근 중임. 시간 초과로 강제 종료됨.
  - FINAL `0.254 m`. 큐브는 `0.06 m`이므로 파지 불가 거리임.
- 구 정책 `2026-07-10_22-33-40/model_1350.pt` (scale `0.2`, 자기 학습 scale로 replay):
  - `t=0: 0.714` → `t=1: 1.512` (swing-out `+0.798 m`) → `t=4: 0.109` → `t=5-19: 0.097` 유지.
  - 4 step만에 도달하고 15 step을 유지함.
  - FINAL `0.097 m`.
- 구 정책이 신 정책보다 최종 거리 기준 2.6배 우수함. 퇴행은 실재함.

### 정정: action scale 가설이 옳았음

- 최초 가설(action scale = 도달 범위)을 중간에 철회했으나, 철회가 틀렸음.
- 철회 근거였던 "구 run에서 palm이 `0.077 m`만 이동했다"는 것도 에피소드 평균 artifact였음.
- 실제로 구 정책은 팔을 `0.80 m` 휘둘렀다가 복귀함.
- scale `0.2`는 `0.72 m`를 4 step에 주파함. scale `0.1`은 20 step을 다 써도 도달하지 못함.
- 즉 action scale `0.1`이 퇴행의 직접 원인임.

### swing-out 해킹은 두 정책 모두에 존재함

- 구 정책 `+0.798 m`, 신 정책 `+0.119 m`.
- `mode="best"` progress reward가 `(d_1 - d_min)`을 지불하므로 첫 step에 멀어지는 것이 이득이기 때문임.
- 구 정책은 scale이 커서 4 step에 복구하므로 결과적 손해가 없음.
- 신 정책은 복구할 여력이 없어 손해가 큼.
- progress reward 해킹은 실재하지만 퇴행의 주원인은 아님. 주원인은 action scale임.

### 진짜 구조적 문제: 에피소드 전체가 팔 이동에 소모됨

- 손이 큐브에서 `0.72 m` 떨어진 곳에서 시작함.
- arm init override (`joint1=-0.45`, `joint2=-1.85`, `joint4=1.20`)가 손을 큐브 근처로 데려오지 못함.
- 20 step 전부를 transit에 쓰고 실제 grasp에 쓸 step이 남지 않음.
- 연구 목표는 functional grasp이지 arm transit이 아님.

### 수정 우선순위 (재작성)

- 1순위: action scale `0.1` → `0.2` 복구. 즉효이며 구 정책이 효과를 증명함.
- 2순위: 시작 거리 `0.72 m`를 `0.20-0.30 m`로 줄임. arm init 자세 조정 또는 cube 위치 조정.
- 3순위: `decimation` `24` → `8-12`. 제어 `2.5 Hz` → `5-7.5 Hz`, step `20` → `40-60`.
  - `episode_length_s`를 유지하면 실시간 길이는 불변이고 결정 횟수만 증가함.
  - 물리 step 수가 동일하므로 시뮬 비용 증가는 미미함.
- 4순위: progress reward 해킹 완화. `finger_cube_reach` 강등, `finger_cube_closeness` 승격.
- 최후 수단: `episode_length_s` `8` → `16`. 시뮬 비용 2배이며 근본 원인을 두고 시간만 주는 것임. 1-3으로 해결되면 불필요함.

### 측정 스크립트

- per-step 궤적 측정 스크립트를 사용함. checkpoint를 replay하며 환경 metric을 매 step 기록함.
- `--action_scale`로 학습 당시 scale을 재현할 수 있음. 다른 scale로 replay하면 무의미한 비교가 됨.
- 앞으로 정책 평가 시 TensorBoard 평균이 아니라 이 궤적을 볼 것.

## 2026-07-11 Scale Isolation Test And Methodology Finding

- "action scale이 퇴행 원인"이라는 앞선 결론의 교란변수를 통제함.
- 앞선 비교는 구 정책 iter 1350 vs 신 정책 iter 50으로 학습량이 27배 달랐음.

### 동일 학습량(iter 50) 비교

- 구 config (scale `0.2`, 구 reward), `model_50.pt`: `t=4`에 `0.105 m` 도달, `t=19` FINAL `0.106 m`.
- 신 config (scale `0.1`, 신 reward), `model_50.pt`: `t=19` FINAL `0.254 m`, 종료 시점에도 접근 중.
- 학습량이 같으므로 학습 부족이 아님. config 차이임.

### scale 단독 효과 분리: 신 정책을 scale 0.2로 replay

- 신 정책(scale `0.1`로 학습)을 scale `0.2` 환경에서 재생함.
- FINAL이 `0.254` → `0.215 m`로 개선되는 데 그침. 구 정책의 `0.106 m`에 도달하지 못함.
- swing-out은 `+0.119` → `+0.243 m`로 오히려 증가함.
- 결론: action scale은 원인의 일부일 뿐임. 신 정책은 학습된 행동 자체가 나쁨.
- 근접이 아니라 swing-out과 progress를 최적화하도록 학습된 것으로 보임.
- 즉 reward 변경도 퇴행에 독립적으로 기여함.

### action scale의 역할 (정리)

- `target = default_joint_pos + scale * raw_action` (`mdp/actions/joint_actions.py:28,35`).
- 증분이 아니라 절대 위치 명령임. 과거 action이 누적되지 않음.
- 따라서 팔의 도달 가능 관절 집합은 `{default + scale * a}`이고 `scale`이 그 반경임.
- `init_noise_std=1.0`이므로 실질 반경은 관절당 약 `±scale` rad임.
- clipping이 없어 `|a|`를 키우면 더 갈 수 있으나, entropy bonus와 adaptive KL이 이를 크게 지연시킴.
- jitter를 줄이려면 `action_rate` weight를 쓸 것. scale 인하는 도달 범위까지 함께 줄임.

### 방법론 문제 (가장 중요)

- 2026-07-11에 5가지를 동시에 변경함.
- action scale `0.2` → `0.1`.
- `finger_cube_reach.distance_max` `0.5` → `0.03`.
- `arm_cube_reach` 비활성화.
- `finger_cube_closeness` 신규 추가.
- `functional_hold` 곱셈형 → 덧셈형.
- 결과가 나빠졌으나 어느 변경이 원인인지 분리 불가함.
- 이번 세션에서 세 번 오진한 근본 원인이 여기 있음.
- 앞으로는 baseline을 고정하고 한 번에 하나씩 변경할 것.

### 제안

- 2026-07-10 config를 검증된 baseline으로 삼고 복귀함. iter 50에서 `0.106 m` 도달을 실측으로 확인함.
- 이후 변경은 한 번에 하나씩, 매번 per-step 궤적으로 검증함.
- baseline 복귀 후에도 남는 구조적 병목 두 가지가 있음.
- 시작 거리 `0.72 m`. 20 step 전부가 팔 transit에 소모됨. `0.20-0.30 m`로 줄일 것.
- 제어 주파수 `2.5 Hz` (`decimation=24`, physics `1/60 s`). 손가락 폐합에는 `5-7.5 Hz` (`decimation` `8-12`) 필요함.
- 아직 코드 변경 없음.

## 2026-07-11 finger_cube_reach.distance_max: The Actual Hacking Mechanism

- `BodiesToObjectProgressReward`의 `distance_max` 의미를 정확히 규명함.
- 이것이 2026-07-11 퇴행의 핵심 메커니즘임. 앞선 분석들이 놓쳤던 부분임.

### 같은 이름, 다른 의미 (naming trap)

- `bodies_to_object_position_tracking_bounded` (절대거리): `r = 1 - clamp(d, 0, distance_max) / distance_max`.
  - 여기서 `distance_max`는 공간적 임계값임. reward가 `0`이 되는 거리임.
- `BodiesToObjectProgressReward` (progress): `progress = best_so_far - current`, `r = clamp(progress / distance_max, 0, 1)`.
  - 여기서 `distance_max`는 거리 임계값이 아님.
  - "한 step에 몇 미터를 좁혀야 만점 `1.0`인가"를 정하는 정규화 상수임.

### distance_max=0.03의 효과: 상시 포화

- `distance_max=0.5`: 한 step에 `0.5 m`를 좁혀야 `1.0`. 실제 step당 `0.04-0.05 m` 개선 → `r ~ 0.09`. 포화되지 않음.
- `distance_max=0.03`: `0.03 m`만 좁혀도 `1.0`. 실제 접근 step이 `0.04-0.05 m`씩 좁히므로 항상 포화됨.

### 포화의 두 가지 결과

- (a) gradient 소멸: `0.03 m` 좁혀도 `1.0`, `0.30 m` 좁혀도 `1.0`. 빨리 접근할 유인이 사라짐.
- (b) 지연(dawdle) 보상: 총 reward = "`0.03 m` 이상 신기록을 세운 step의 개수" (step당 상한 `1.0`).
  - 4 step만에 도착 → 약 `4`점.
  - `0.04-0.05 m`씩 15 step에 걸쳐 접근 → 약 `15`점.
  - 느리게 갈수록 이득이며, 도착하면 남은 step에서 벌 것이 없으므로 손해임.

### 실측 궤적이 정확히 이 행동을 보임

- 신 정책 `2026-07-11_14-17-17/model_50.pt`의 step별 신기록 개선량:
- `t=5: +0.046` → `1.00` (포화).
- `t=6: +0.055` → `1.00` (포화).
- `t=7: +0.056` → `1.00` (포화).
- `t=8: +0.055` → `1.00` (포화).
- `t=9: +0.051` → `1.00` (포화).
- `t=10: +0.047` → `1.00` (포화).
- `t=11: +0.039` → `1.00` (포화).
- `t=12: +0.032` → `1.00` (포화). 8 step 연속 포화임.
- 임계값 `0.03` 바로 위에서 페이스를 조절하며 에피소드 끝까지 도착하지 않음.

### TensorBoard 확증

- `Episode_Reward_Raw/finger_cube_reach`는 step당 평균 raw 값임 (`sum(raw*dt)/max_episode_length_s`, `dt=0.4`, 20 step).
- 구 (`distance_max=0.5`) iter 50: step당 평균 `0.089`. 최종 거리 `0.106 m`.
- 신 (`distance_max=0.03`) iter 50: step당 평균 `0.489`. 최종 거리 `0.254 m`.
- 5.5배 더 받으면서 결과는 2.4배 나쁨. reward hacking의 정의에 정확히 부합함.

### 근본 원리: 포화가 telescoping을 깨뜨림

- 포화가 없으면 progress reward는 망원경처럼 접힘: `sum_t (best_{t-1} - d_t) = d_start - d_min`.
- 즉 총합이 "얼마나 가까이 갔는가"에만 의존하고 "어떤 속도/페이스로 갔는가"와 무관해짐.
- 페이스 조절이 이득이 되지 않으므로 착취 불가함.
- step당 `1.0` clamp이 이 telescoping을 깨뜨려 페이스 조절을 수익화함.

### 규칙

- progress reward의 `distance_max`는 실제 step당 최대 개선량보다 충분히 커야 함. 포화되면 안 됨.
- `0.5`는 조건을 만족함. 구 정책의 최대 step 개선량 `0.35 m` → `r = 0.70`으로 미포화임.
- `0.03`은 상시 포화이므로 사용하면 안 됨.
- 되돌릴 값: `finger_cube_reach.distance_max` `0.03` → `0.5`.

## 2026-07-11 Contact Test: The Hand Already Reaches And PUSHES The Cube

- 긴 학습 run `2026-07-10_22-33-40/model_1350.pt` (`distance_max=0.5`, scale `0.2`, 1352 iteration)을 replay하며 큐브 변위를 측정함.
- 목적: 손이 큐브에 실제로 닿는지, 아니면 근처에서 hover만 하는지 판별함.

### 결과: 접촉함. 그리고 밀어냄.

- 큐브 변위 (리셋 위치 대비): 평균 `32.2 mm`, 최대 `113.8 mm`.
- 큐브 수직 변화 (들어올림): 평균 `0.6 mm`. 사실상 `0`임.
- 큐브가 `5 mm` 이상 움직인 env 비율: `73%`.
- 결론: hover가 아님. 손이 큐브에 도달해 수평으로 밀고 있음. 들어올리지는 않음.
- 거리가 `0.07-0.09 m`에서 정체되는 이유는 자기가 밀어낸 큐브를 계속 쫓기 때문임.

### 손가락별 최종 거리 (긴 학습, iter 1350)

- `palm_distance`: `0.588` → `0.132`.
- `thumb_distance`: `0.704` → `0.073` (제어함).
- `index_distance`: `0.757` → `0.091` (제어함).
- `middle_distance`: `0.737` → `0.094` (제어함).
- `ring_distance`: `0.722` → `0.123` (제어 안 함).
- `little_distance`: `0.703` → `0.148` (제어 안 함).
- `finger_weighted_mean_distance`: `0.719` → `0.096`.
- 주의: `finger_weighted_mean_distance`는 정책이 제어하지 않는 ring/little을 포함한 5-finger 가중평균임.
- 실제 제어 손가락(thumb/index/middle)은 `0.073-0.094 m`로 더 가까움.
- 이 지표만 보면 실제 접근 성능을 과소평가하게 됨.

## 2026-07-11 distance_max: Right Instinct, Wrong Function

- 사용자의 이해: "`distance_max`가 크면 그 거리 안에 들어오기만 해도 보너스가 시작되므로 너무 후하다."
- 이 모델은 절대거리 reward에서는 100% 정확함.
- `bodies_to_object_position_tracking_bounded` (= `finger_cube_closeness`, `arm_cube_reach`): `r = 1 - clamp(d,0,dmax)/dmax`.
- 여기서는 `dmax=0.7`이면 실제로 `0.7 m` 안에만 들어와도 보상이 나옴. 후하고 기울기가 죽음. 사용자 직관이 옳음.
- 그러나 `finger_cube_reach`는 progress reward임: `r = clamp((best - current)/dmax, 0, 1)`.
- 여기에는 절대거리가 등장하지 않음. "N m 안에 들어오면"이라는 게이트가 존재하지 않음.
- 큐브에서 `5 m` 떨어져 있어도 이번 step에 `0.05 m` 좁히면 `r = 0.1`을 받음.
- 큐브에 `0.05 m` 붙어 있어도 개선이 없으면 `r = 0`임.
- 즉 `dmax=0.5`는 "`0.5 m` 안에 오면 보너스"가 아니라 "한 step에 `0.5 m`를 좁혀야 만점"이라는 뜻임.
- 결론: 줄였어야 할 파라미터는 `finger_cube_closeness.distance_max` (`0.7` → `0.35`)였음.
- `finger_cube_reach.distance_max`를 `0.03`으로 줄인 것은 상시 포화를 유발해 dawdle 해킹을 만들었음.
- 직관은 옳았으나 적용 대상 파라미터가 틀렸음.

## 2026-07-11 Root Cause: Grasping Is Not In The Reward At All

- 사용자 질문: "functional_hold가 계속 높게 나와서 수렴했다고 오판한 것인가?"
- 실질적으로 맞음.
- `functional_hold`는 자세(shape) 보상이지 결과(outcome) 보상이 아님. 근접도와 손가락 대향 기하학만 봄. 파지 여부를 보지 않음.
- 정책은 "큐브 근처 + 손가락 대향" 자세를 달성하면 step당 약 `0.56` (만점 대비 `56%`)을 계속 받음.
- 거기서 더 개선하려면 실제로 쥐어야 하는데, `functional_hold`는 파지를 거의 보상하지 않음.
- 더 나쁜 것은, 쥐려고 하면 큐브가 밀려나 거리가 멀어지고 거리 reward가 깎임.
- 즉 실제 파지 행동에 사실상 페널티가 붙음.
- 그 결과 "가까이 가서 손가락 벌리고 쥐지 않는" 자세가 국소최적이 됨.
- 접촉 테스트가 이를 확증함. 닿기는 하나 밀어내기만 하고 들어올리지 않음.

### 핵심 결론

- 현재 reward에는 "쥐었다"를 보상하는 항이 하나도 없음.
- `cube_lift = None`, `cube_goal_tracking = None`, contact sensor 미구현.
- grasp이 목적함수에 존재하지 않으므로 `distance_max`를 어떻게 튜닝해도 정책은 grasp을 학습할 수 없음.
- 지금까지의 `distance_max` 논쟁은 전부 접근(reach) 단계 튜닝이었고, 정책은 이미 접근에 성공했음.
- 막힌 지점은 접근 이후임.
- 다음 작업 방향은 reward 미세조정이 아니라 파지 보상 항의 도입임.

## 2026-07-11 TensorBoard Distance Is Dominated By The First 4 Steps

- 사용자가 TensorBoard `Metrics/cube/thumb_distance` 최솟값이 약 `0.16-0.19`인 것을 보고 "엄지가 16cm 떨어져 있는 것 아니냐"고 물음.
- 아님. 그 값은 에피소드 20 step 전체의 평균임.

### 엄지 per-step 궤적 (model_1350, scale 0.2)

- `t=0: 0.703` (출발)
- `t=1: 1.494` (큐브 반대로 `0.79 m` 튕겨나감. swing-out 해킹)
- `t=2: 0.985`
- `t=3: 0.447`
- `t=4: 0.084` (도착)
- `t=5-19: 0.073-0.077` (15 step 내내 유지)
- 에피소드 평균: `0.242` (= TensorBoard 값)

### 핵심

- 앞 4 step(이동 + swing-out)이 평균 `0.242` 중 `0.186`을 차지함. 전체의 `77%`임.
- 20 step 중 4 step이 지표를 지배함.
- 엄지의 실제 정상상태는 큐브 중심까지 `0.073 m`, 표면까지 `0.033 m`임.
- TensorBoard의 `0.16-0.19`는 "출발점이 `0.70 m`이고 첫 step에 `1.49 m`까지 튕겨나간다"를 평균낸 값임.
- 엄지가 큐브에서 16cm 떨어져 있다는 뜻이 전혀 아님.
- `t=1`의 `1.494`가 swing-out 해킹의 육안 증거임. GUI play에서 팔이 한 번 크게 휘두르는 동작이 이것임.

## 2026-07-11 SDF Does Not Require CAD

- 사용자 질문: SDF를 쓰려면 CAD 파일이 필요한 것 아닌가?
- 아님. SDF는 물체 표면까지의 부호 있는 거리이며, 시뮬레이션하는 물체는 이미 기하 정보를 가지고 있음.
- 큐브: 해석식. 3줄. 비용 0. 이미 진단 스크립트 `eval_contact.py`의 `box_sdf`로 구현해 사용함.
- 젓가락(최종 목표): 원기둥/캡슐 근사로 해석식 가능. 비용 0.
- 임의 메시(드릴, 머그 등): 메시에서 voxel SDF 그리드를 오프라인 1회 계산 후 GPU trilinear 조회.
- 논문이 pre-computed SDF를 쓴 이유는 드릴/스프레이/머그 같은 복잡한 메시 때문임. 별도 CAD를 구한 것이 아니라 시뮬에 이미 넣은 메시에서 뽑은 것임.
- 임의 메시가 필요해지면 IsaacLab에 포함된 `warp`의 `wp.Mesh` closest-point 쿼리를 GPU에서 사용 가능함.
- 결론: SDF는 현재 구현의 걸림돌이 아님.

### 큐브 box SDF (전부)

- 큐브 로컬 좌표로 변환 후 `q = |p_local| - half_extent`.
- `sdf = norm(clamp(q, min=0)) + clamp(max(q.x, q.y, q.z), max=0)`.
- 음수이면 큐브 내부임.

## 2026-07-11 Revert To Verified Baseline Plus Closeness

- 사용자 결정: `distance_max`와 action scale을 검증된 baseline 값으로 되돌리고 `finger_cube_closeness`만 얹은 상태로 학습해봄.
- 목적: closeness의 순효과를 격리해서 측정함. 한 번에 하나씩 변경 원칙에 부합함.

### 변경 내용

- `finger_cube_reach.distance_max`: `0.03` → `0.5` (`env_cfg_common.py`).
- action scale: `0.1` → `0.2` (`grasp/indy_wuji/env_cfg.py`).
- `finger_cube_closeness`는 `weight=0.2`, `distance_max=0.7` 그대로 유지함.
- `functional_hold`는 `weight=0.2`, `distance_max=0.18`, `center_distance_max=0.18` 그대로 유지함.
- `py_compile` 통과함.

### 현재 grasp reward 구성

- `finger_cube_reach`: `weight=0.3`, `distance_max=0.5`, `mode=best`.
- `finger_cube_closeness`: `weight=0.2`, `distance_max=0.7`.
- `functional_hold`: `weight=0.2`, `distance_max=0.18`, `center_distance_max=0.18`.
- `action_rate`: `weight=-0.0003`.
- action scale: `0.2`.

### 사전 예측: 이번 학습에서 swing-out은 남아 있을 것임

- `finger_cube_closeness`는 `r = 1 - clamp(d, 0, 0.7) / 0.7`임.
- swing-out 구간의 거리는 `0.70` → `1.49 m`인데 둘 다 `0.7` 이상이라 clamp되어 reward가 동일하게 `0`임.
- 즉 큐브 반대로 `0.79 m` 튕겨나가도 closeness 관점에서 아무 손해가 없음.
- swing-out의 원인은 progress reward의 `mode="best"` + `clamp(min=0.0)` 조합임. 후퇴가 공짜이고 오히려 `d_1`을 키워 이득임.
- 논문은 `d(t-1) - d(t)`를 쓰고 clamp를 하지 않아 후퇴에 감점을 줌.
- 따라서 closeness가 기여하는 것은 "도착 후 머무르기"(`0.7 m` 안쪽 구간)뿐임.
- 이번 run 결과에서 swing-out이 남아 있어도 실패로 해석하지 말 것. 그것은 다음 변수임.

### 평가 방법

- TensorBoard `Metrics/cube/*` 평균으로 판단하지 말 것. 앞 4 step이 평균의 77%를 지배함.
- 반드시 checkpoint를 replay해서 per-step 궤적, 최종 거리, 큐브 변위(접촉 여부)를 볼 것.

## 2026-07-11 Cage Hold Reward Implemented (Dexterous Pre-grasp Eq.15)

### 변경 요약

- `finger_cube_reach.distance_max`: `0.03` → `0.5` (포화 제거, 검증된 baseline 복구).
- action scale: `0.1` → `0.2` (도달 범위 복구).
- `finger_cube_closeness.distance_max`: `0.7` → `0.35` (상수 바닥 제거, 기울기 2배).
- `functional_hold`: `None`으로 비활성화. `finger_cage_hold`가 대체함.
- `finger_cage_hold` 신규 추가. `weight=1.0`.
- action shape 18, policy observation shape 57 불변. 체크포인트 호환성 유지됨.
- `py_compile` 통과, `--num_envs 16 --max_iterations 2` smoke test 통과함.

### 새 reward 함수: `object_in_finger_cage` (`mdp/rewards.py`)

- 논문 Dexterous Pre-grasp Manipulation Eq. 15의 `r_hold` 구현임.
- 엄지끝(`finger1_tip_link`)과 중지 두 지점(`finger3_tip_link`, `finger3_link3`) 사이에 가상점을 찍음.
- 각 선분마다 등간격 3점, 총 6점임 (논문과 동일).
- 각 점에서 큐브 표면까지의 signed distance를 계산함. 큐브 내부이면 음수임.
- 각 점을 반지름 `sphere_radius`의 구로 보고, 그 구가 큐브를 파고든 깊이를 보상함.
- `r = clamp((sphere_radius - sdf) / (sphere_radius + depth_max), 0, 1)`의 6점 평균임.
- 큐브 SDF는 해석식임 (`_box_signed_distance`). CAD나 사전계산 SDF 불필요함.
- `SceneEntityCfg`에 `preserve_order=True` 필수임. 기본값 `False`이면 body_ids가 정렬되어 순서가 깨짐.

### 핵심 원리: 거리 reward와 부호가 반대임

- 거리 reward는 물체를 만지면 물체가 밀려나 거리가 늘고 reward가 깎임. 즉 접촉이 손해임.
- 실제로 구 정책이 큐브를 평균 `32 mm` 밀어내면서 "엄지만 찌르고 나머지는 벌린 채" 머무는 이유가 이것임.
- cage reward는 가상점이 물체 **내부**로 들어갈수록 보상함. 손을 오므리면 점들이 서로 가까워지며 큐브 안으로 들어감.
- 따라서 접촉/감싸기가 이득이 됨. 접촉센서 없이 파지가 보상됨.

### 파라미터 튜닝 (실측 기반)

- 구 정책(`model_1350`)을 큐브 근처에 안착시킨 뒤 손가락 굴곡 관절을 kinematic sweep하여 튜닝함.
- `sphere_radius`가 크면 손가락을 벌린 채 큐브가 사이에 있기만 해도 점수가 나옴.
- `radius=0.020, depth=0.030`: hover `0.299` → 오므림 최대 `0.488`. 비율 `1.6x`. 대비 약함.
- `radius=0.010, depth=0.020`: hover `0.256` → `0.510`. 비율 `2.0x`.
- `radius=0.005, depth=0.020`: hover `0.194` → `0.463`. 비율 `2.4x`. 헤드룸 `+0.27`로 최대임. 채택함.

### 동작 검증 (굴곡 sweep, 채택 파라미터)

- 벌림 `-0.4 rad`: `0.03`.
- 현재 자세 `0.0`: `0.19` (구 정책의 "벌린 채 찌르기").
- 오므림 `+0.2`: `0.39`.
- 오므림 `+0.4`: `0.46` (최대).
- 과다 오므림 `+0.8`: `0.04` (큐브를 짜내면 손해).
- 오므릴수록 단조 증가하다가 과다 지점에서 하락함. 방향이 정확함.

### 가중치 순서

- 논문은 `r_T >> r_orient >> r_hold >> r_reach`로 스케일함.
- 이유: 쉬운 앞 단계 하위과제에 큰 보상을 주면 정책이 거기 눌러앉음 (논문 Sec. IV-C 명시).
- 기존 코드는 `reach(0.3) > hold(0.2)`로 역순이었음. 이것이 국소최적의 원인이었음.
- 현재: `finger_cage_hold(1.0) >> finger_cube_reach(0.3)`. 순서 교정됨.

### 아직 안 넣은 것

- `cube_lift` (sparse 성공 보상). 논문은 fake success 방지용으로 필수라고 명시함.
- cage만 있으면 "큐브를 바닥에 누른 채 오므리기"도 만점 가능함. lift가 성공의 정의임.
- progress reward의 `mode="best"` + `clamp(min=0)` 수정. swing-out과 dawdle 해킹의 원인임.
- 논문은 `d(t-1) - d(t)`를 쓰고 clamp하지 않아 후퇴에 감점을 줌.
- 이번 run에서 swing-out은 여전히 나타날 것임. 실패로 해석하지 말 것.

### 평가 방법

- TensorBoard `Metrics/cube/*` 평균으로 판단 금지. 앞 4 step이 평균의 77%를 지배함.
- checkpoint를 replay해서 per-step 궤적, 최종 거리, 큐브 변위, cage reward 값을 볼 것.

## 2026-07-11 train.py Render Interval Flag

- 사용자가 GUI로 학습을 지켜볼 때 뷰포트가 너무 느리다고 보고함.
- 원인은 `play.py`에서 이미 고친 것과 동일한 render cadence 문제임. `train.py`에는 그 옵션이 없었음.

### 타이밍 실측 (재확인)

- physics step-size는 `1/60 s`임.
- `decimation = 24`이므로 environment step은 `0.4 s`, 제어 주파수는 `2.5 Hz`임.
- task cfg가 `sim.render_interval = decimation`이므로 시뮬레이션 1초당 렌더는 `2.5`장임.
- 이전 기록의 "5 Hz"는 틀렸음. 실제는 `2.5 Hz`임.

### 변경

- `scripts/rsl_rl/train.py`에 `--render_interval` 인자를 추가함. 기본값은 `None`임.
- 기본값이 `None`이므로 지정하지 않으면 task cfg 값(= decimation)을 그대로 사용함. 기존 동작 불변임.
- `env_cfg.sim.device` 설정 직후 `env_cfg.sim.render_interval = args_cli.render_interval`을 적용함.
- `py_compile` 통과함.

### 중요: headless 학습에는 영향이 전혀 없음

- `manager_based_env.py:474`: `is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()`.
- `manager_based_env.py:488`: `if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering: self.sim.render()`.
- 즉 `render_interval`은 GUI 또는 RTX 센서가 활성일 때만 작동함.
- `--headless` 학습에서는 렌더 자체가 일어나지 않으므로 이 값이 무의미함.
- 따라서 값을 낮춰도 headless 본 학습 속도는 저하되지 않음.

### 값 선택 기준

- `24` (기본, = decimation): 시뮬 1초당 `2.5`장. 슬라이드쇼처럼 보임.
- `8`: `7.5`장.
- `4`: `15`장. GUI 관찰용 권장값임.
- `2`: `30`장. 렌더 비용 2배임.
- GUI 학습에서는 env 개수가 많으면 렌더 호출 증가가 실제 속도 저하로 이어짐. `--num_envs`를 `64-256`으로 낮춰서 관찰할 것.
- 본 학습은 `--headless`로 돌릴 것.
- IsaacLab이 `render_interval < decimation`일 때 경고를 출력하나 의도된 동작이므로 무시해도 됨.

### 사용 예

- GUI 관찰: `--num_envs 64 --render_interval 4`.
- headless 본 학습: `--headless --num_envs 4096 --max_iterations 1500` (플래그 불필요).

## 2026-07-11 TensorBoard Metrics Expanded (final/min/max + surface + cage + cube state)

- 사용자가 가중치 유/무 보상과 에러를 모두 TB에 띄워달라고 요청함.
- 기존 `Metrics/cube/*`가 에피소드 평균뿐이라 정착 자세를 판별할 수 없었음. 이번 세션 오진의 근본 원인이었음.
- `isaac_neuromeka/env/managers.py`의 `CustomRewardManager`를 확장함.
- `py_compile` 통과, smoke test 통과, `inf`/`nan` 누수 없음. 총 92개 스칼라 기록됨.

### 보상 (이미 존재하던 것, 가중치 유/무 둘 다 있음)

- `Episode_Reward/<term>`: 가중치 적용값.
- `Episode_Reward_Raw/<term>`: 가중치 미적용 원값.
- `Episode_Reward_Std/<term>`: 표준편차.

### 지표 네임스페이스 (신규 3개 추가)

- `Metrics/cube/<name>`: 에피소드 평균. 기존과 동일. 앞 4 step이 77%를 지배하므로 성능 판단에 쓰지 말 것.
- `Metrics/cube_final/<name>`: 에피소드 마지막 step 값. 정착 자세를 나타냄. 이것을 볼 것.
- `Metrics/cube_min/<name>`: 에피소드 내 최소값.
- `Metrics/cube_max/<name>`: 에피소드 내 최대값. swing-out 탐지용.

### 지표 20종

- 큐브 중심까지 거리 (기존 9종): `palm/thumb/index/middle/ring/little_distance`, `finger_mean_distance` (가중치 없음), `non_thumb_mean_distance`, `finger_weighted_mean_distance` (가중치 `3,1,1,1,1`).
- 큐브 표면까지 거리 (신규 6종): `palm_surface`, `thumb_surface`, `index_surface`, `middle_surface`, `ring_surface`, `little_surface`. 음수이면 관통임.
- 중심 거리는 body마다 비교가 안 됨. 큐브가 `0.06 m`이므로 접촉 여부는 표면 거리로 봐야 함.
- cage 진단 (신규 3종): `cage_sdf_mean`, `cage_sdf_min`, `cage_inside_frac`.
- `cage_inside_frac`는 가상점 6개 중 큐브 내부에 있는 비율임. 오므리기 여부의 직접 지표임.
- 큐브 상태 (신규 2종): `cube_displacement` (리셋 위치 대비 이동량), `cube_lift` (수직 변화량).

### 구현 세부

- `_cube_signed_distance()`: 큐브 box SDF. reward의 `_box_signed_distance`와 동일한 해석식임.
- cage 가상점은 reward와 정확히 동일하게 재구성함. `_cage_body_names = ["finger1_tip_link", "finger3_tip_link", "finger3_link3"]`, `_cage_fractions = [0.25, 0.50, 0.75]`.
- `_cube_init_pos`는 `reset()`에서 갱신함. 이 시점에는 event(`reset_cube_position`)가 이미 실행된 뒤라 새 에피소드의 큐브 위치가 들어감.
- `compute()`에서 매 step `sums += metric*dt`, `last = metric`, `min = minimum(...)`, `max = maximum(...)`를 갱신함.
- 최초 startup reset은 `compute()` 이전에 호출되므로 min/max가 `+/-inf` 상태임. 로깅 시 `torch.nan_to_num`으로 `0` 처리함.

### 가상점 위치 (reward와 metric 공통)

- 등간격 비율은 `[0.25, 0.50, 0.75]`임. 양 끝점은 제외함.
- 선분 A: `finger1_tip_link` -> `finger3_tip_link` (엄지끝 -> 중지끝). 3점. 핀치 파지 위치임.
- 선분 B: `finger1_tip_link` -> `finger3_link3` (엄지끝 -> 중지 중간마디). 3점. 파워 파지 위치임.
- 총 6점. 논문과 동일함.

### 학습 중 볼 지표 (우선순위)

- `Metrics/cube_final/cage_inside_frac`: `0` -> `0.3` 이상으로 오르면 오므리기 시작한 것임. 가장 중요함.
- `Metrics/cube_final/thumb_surface`: 엄지-큐브 표면 간격. 현재 약 `0.03`. 음수이면 접촉임.
- `Metrics/cube_final/middle_surface`: 중지-큐브 표면 간격. 현재 약 `0.056`. `0`에 가까워지면 중지도 붙는 것임.
- `Metrics/cube_final/cube_displacement`: 크면 여전히 밀어내는 중임.
- `Metrics/cube_final/cube_lift`: `0`보다 크면 드는 중임.
- `Metrics/cube_max/finger_weighted_mean_distance`: 시작값(약 `0.7`)보다 크게 초과하면 swing-out 중임.

## 2026-07-11 PhysX Patch Count Raised To 2**20

- `cube_grasp_env_cfg.py:95`: `gpu_max_rigid_patch_count`를 `2**19` (524,288) → `2**20` (1,048,576)으로 올림.
- `py_compile` 통과, `--num_envs 512 --max_iterations 2` smoke test 통과. PhysX patch 경고 없음.

### 근거

- 2026-07-10 run에서 `2**18` (262,144)이 약 `263k` patch 수요에 살짝 부족해 overflow가 발생했음.
- 4096 env 기준 env당 약 `64` patch임.
- `finger_cage_hold`가 손가락을 오므리게 유도하므로 finger-cube 및 finger-finger 접촉이 추가됨.
- env당 `10-20` patch 증가를 가정하면 약 `330k`로 `2**19` (524k) 안에는 들어옴.
- 즉 `2**19`로도 아마 충분했음. 그럼에도 올린 이유는 실패 모드의 비대칭성 때문임.

### 실패 모드가 위험한 이유

- PhysX patch buffer overflow는 크래시가 아님. **접촉을 조용히 버림.**
- 접촉이 버려지면 손가락이 큐브를 통과함.
- 그 결과 `cage_inside_frac`은 오르는데 큐브는 움직이지 않고 `finger_cage_hold`도 오르지 않음.
- 이 증상은 "cage reward가 잘못 설계됨"과 구별이 불가능함.
- 이번 학습은 접촉이 핵심이므로 이 실패 모드가 특히 치명적임.
- 반면 `2**20`으로 올리는 비용은 GPU 메모리 수십 MB에 불과함. RTX 3090 24GB에서 무시 가능함.
- 긴 학습(1500 iteration, 수 시간)에 대한 보험으로 타당함.

## 2026-07-11 Metric Priority (80개 중 실제로 볼 것은 5개)

- TB 지표가 80개(20종 x 4 네임스페이스)로 늘어난 것에 대해 정리함.
- 결정: 그대로 유지함. 계산 비용은 사실상 0이고, TB 검색창에 `cube_final`만 치면 필터됨.
- 이번 세션에서 지표가 없어 세 번 오진했고 그때마다 replay 스크립트를 새로 짜야 했음. 남겨두는 편이 시간을 아낌.

### 반드시 볼 것 (5개)

- `Metrics/cube_final/cage_inside_frac`: 오므리는가. `0` → `0.3` 이상이면 성공. 가장 중요함.
- `Metrics/cube_final/thumb_surface`: 엄지가 닿는가. 음수이면 접촉임.
- `Metrics/cube_final/middle_surface`: 중지가 닿는가. 현재 약 `0.056`이며 `0`으로 가야 함.
- `Metrics/cube_final/cube_lift`: 드는가. `0`보다 크면 드는 중임.
- `Metrics/cube_max/finger_weighted_mean_distance`: swing-out 여부. 시작값 약 `0.7`을 크게 넘으면 팔을 휘두르는 중임.

### 가끔 볼 것 (3개)

- `Metrics/cube_final/cube_displacement`: 여전히 밀어내는 중인가.
- `Metrics/cube_final/cage_sdf_mean`: `cage_inside_frac`의 연속판임.
- `Metrics/cube_final/index_surface`: 검지 접촉 여부.

### 사실상 잉여 (약 72개, 그래도 유지함)

- `Metrics/cube/*` (에피소드 평균 20개): 적극적으로 오해를 유발함. 이번 세션 오진의 직접 원인임. 성능 판단에 쓰지 말 것.
- `Metrics/cube_min/*` (20개): 정책이 수렴해 머무르므로 `cube_final`과 거의 동일함.
- `ring_*`, `little_*`: 정책이 제어하지 않는 손가락임.
- `*_distance` (큐브 중심까지, 9종): 큐브가 `0.06 m`라 중심 거리는 body간 비교가 불가능함. `*_surface`가 상위호환임.
- `finger_mean_distance`와 `non_thumb_mean_distance`: 서로 중복임.

### 정리를 원할 경우 (지금은 안 함)

- `Metrics/cube_min/*`를 통째로 제거하고 중심거리를 `finger_weighted_mean_distance` 하나만 남기면 약 35개로 줄어듦.

## 2026-07-11 Cage Reward Works: Run 2026-07-11_16-43-19 (211 iter, 학습 중)

- `finger_cage_hold` 도입 후 첫 학습 결과임. 211 iteration 시점이며 모든 곡선이 아직 상승 중임.

### cage는 확실히 작동함

- `Metrics/cube_final/cage_sdf_mean`: `0.597` → `0.010`. 가상점이 큐브 표면에서 `0.60 m` → `0.01 m`까지 접근함.
- `Metrics/cube_final/cage_inside_frac`: `0.000` → `0.414`. 계속 상승 중임.
- `Episode_Reward_Raw/finger_cage_hold`: `0.000` → `0.184`. 계속 상승 중임.
- `Train/mean_reward`: `-0.03` → `2.44`. 계속 상승 중임.
- 큐브가 엄지-중지 사이 파지 간극 안으로 들어옴. 이전 정책이 하지 못하던 동작임.

### 그러나 손가락이 큐브를 쥐고 있지는 않음

- `Metrics/cube_final/thumb_surface`: `0.601` → `0.026`.
- `Metrics/cube_final/index_surface`: `0.651` → `0.050`.
- `Metrics/cube_final/middle_surface`: `0.622` → `0.060`.
- `cage_inside_frac = 0.414`는 가상점 6개 중 약 2.5개가 큐브 내부라는 뜻임.
- 가상점은 엄지-중지 사이 선분 위에 있으므로, 큐브가 그 선분을 가로지르면 양쪽 손가락이 닿지 않아도 중간 점들이 큐브 내부에 들어감.
- 즉 현재는 "큐브를 사이에 두고 손가락을 벌린 채 서 있는" 상태에 가까움.
- 링크 원점 오프셋(약 `0.037 m`)을 감안하면 엄지는 접촉했고 검지/중지는 약 `0.02 m` 떠 있음.

### 우려 신호 1: 정책이 접촉을 회피하는 방향으로 학습 중

- `Metrics/cube_final/cube_displacement`: `0.002` → `0.146` (피크) → `0.037`로 계속 감소함.
- 즉 큐브를 덜 건드리는 방향으로 학습하고 있음.
- 원인: 거리 reward(`finger_cube_reach`, `finger_cube_closeness`)가 접촉을 처벌함. 만지면 큐브가 밀려나 거리가 늘고 reward가 깎임.
- 거리 reward가 cage reward와 직접 충돌하고 있음.

### 우려 신호 2: lift 없음

- `Metrics/cube_final/cube_lift`: `0.0005 m`. 사실상 `0`임.
- `Metrics/cube_max/cube_lift`: `0.0036 m`. 최대 `3.6 mm`로 들어올리지 못함.
- 이것이 논문이 말한 fake success 상태임. 조건은 만족하나 실제로 잡지 않음.

### 우려 신호 3: swing-out 악화 (예측대로)

- `Metrics/cube_max/finger_weighted_mean_distance`: `0.824` → `1.153`.
- progress reward의 `mode="best"` + `clamp(min=0)`를 아직 고치지 않았으므로 예상된 결과임.

### 다음 단계 제안

- 1순위: 더 학습시켜서 `cage_inside_frac`이 정체하는 지점을 확인함. 211 iteration은 이름.
- 2순위: `cube_lift` 추가. 논문이 fake success 방지용으로 `r_lift`를 넣은 이유가 정확히 현재 상태임.
- 3순위: `finger_cube_closeness` 약화 또는 제거 검토. cage가 자리잡은 지금 이 항은 접촉을 처벌하는 방향으로만 작용함. `cube_displacement` 감소가 그 증거임.
- 4순위: progress reward를 `mode="previous"` + clamp 제거로 변경해 swing-out 제거.

## 2026-07-11 Root Cause Of "Cages But Never Grips" + Paper-Faithful Reach (Eq.14)

### 결정적 실험: 정책의 손가락을 물리 상태에서 강제로 오므려봄

- checkpoint `2026-07-11_19-02-58/model_200.pt`를 큐브 근처에 안착시킨 뒤, flexion 관절 목표에 `extra` rad를 더하며 물리와 함께 진행함.
- 결과 (`extra=0.80` 행은 12+8=20으로 에피소드 리셋된 것이라 무효):
- `extra=0.00` (안착): `inside_frac 0.456`, thumb `0.0167`, index `0.0718`, middle `0.0777`, cube_moved `0.0000`.
- `extra=0.20`: `inside_frac 0.471` (최대), cube_moved `0.0016`.
- `extra=0.40`: `inside_frac 0.453`, index `0.0537`, cube_moved `0.0038`.
- `extra=0.70`: `inside_frac 0.398`, index `0.0287`, cube_moved `0.0080`, lift `0.0003`.

### 가설 A 기각: 큐브가 밀려나는 것이 아님

- 강제로 오므려도 큐브는 `8 mm`밖에 움직이지 않음. 물리는 정상임.

### 손가락은 물리적으로 잘 닫힘

- 검지가 `0.072` → `0.029 m`까지 접근함. 오므릴 능력 자체는 있음.

### 진짜 원인: cage reward가 오므리면 내려감

- `cage_inside_frac`이 `extra=0.2`에서 `0.471`로 최대, 이후 `extra=0.7`에서 `0.398`로 단조 감소함.
- 즉 정책은 cage reward의 최적점에 정확히 앉아 있으며, 더 오므리면 손해임.
- 정책은 완벽히 합리적으로 행동한 것이고, reward 설계가 잘못된 것임.

### 기하학적 원인

- 안착 자세에서 thumb `0.017 m`, index `0.072`, middle `0.078`임.
- 엄지만 큐브에 붙어 있고 나머지 손가락은 반대편에 없음.
- 가상점은 엄지끝-중지 선분 위에 있는데, 그 선분이 큐브를 제대로 관통하지 않음.
- 손가락을 오므리면 선분이 짧아지며 점들이 큐브 밖으로 빠져나감.
- 논문에서는 엄지와 중지가 물체 양쪽에 있어 선분이 물체를 관통하고, 오므리면 점들이 더 깊이 들어감.

### 근본 원인: hold만 논문식으로 바꾸고 reach는 안 바꿈

- 논문의 `r_reach` (Eq.14)는 `r_hold`와 **같은 6개 가상점**을 사용함.
- 그것이 "파지 간극을 물체 위로" 끌어당겨 물체가 엄지-중지 사이에 놓이게 만듦.
- 기존 `finger_cube_reach`는 "손끝 -> 큐브 중심" 거리였음. 그래서 엄지가 큐브 중심을 찌르는 자세가 만들어짐.
- 그 자세에서는 오므려도 감쌀 수 없음.

### 변경 내용

- `finger_cube_reach`: `None`. (손끝->큐브중심, `mode="best"`, `clamp(min=0)`)
- `finger_cube_closeness`: `None`. (손끝->큐브중심 절대거리. 접촉을 처벌하고 엄지 찌르기를 유도함)
- `finger_cage_reach`: 신규. `mdp.ObjectCageProgressReward`, `weight=0.3`, `distance_max=0.5`.
- `finger_cage_hold`: 유지. `weight=1.0`.
- `mdp/rewards.py`에 `_cage_points()`, `_cage_sdf()` 공용 헬퍼 추가. reward와 metric이 동일한 점을 사용하도록 함.
- action 18D, observation 57D 불변.
- `py_compile` 통과, smoke test 통과.

### ObjectCageProgressReward (논문 Eq.14)

- 가상점 6개의 큐브 **표면**까지 signed distance 평균의 **차분**임.
- `mode="previous"` 방식임. `d(t-1) - d(t)`.
- `clamp(min=-1.0)`임. **후퇴에 감점을 줌.** 기존 `clamp(min=0)`은 후퇴가 공짜였음.
- 기준 거리를 `reset()`에서 **리셋 자세로 seeding**함.
- 이것이 핵심임. 첫 `__call__`에서 seeding하면 첫 액션이 기준선을 공짜로 부풀릴 수 있어 swing-out이 남음.
- 결과적으로 에피소드 총합이 `d(reset) - d(final)`로 telescoping됨. 페이스 조작도 swing-out도 이득이 없음.

### 검증: swing-out이 실제로 처벌됨

- 기존 정책 replay 시 per-step raw 값:
- `t=1`: `cage_sdf` `0.663` -> `1.0465` (`+0.383`, swing-out) -> reach raw `-0.7668`. **감점됨.**
- `t=2`: `-0.538` -> `+0.9938`.
- `t=3`: `-0.409` -> `+0.8186`.
- `t=4`: `-0.083` -> `+0.1667`.
- 에피소드 총합 `+1.2280`.
- swing-out이 총 가용 보상의 약 38%를 물어내게 함. 기존에는 이 자리가 `0`(무료)이었고 오히려 기준선을 부풀려 이득이었음.
- 랜덤 정책 smoke test에서 `Episode_Reward_Raw/finger_cage_reach`가 `-0.0026`, `-0.0035`로 음수임. 후퇴 감점이 정상 작동함.

### cube_lift를 아직 넣지 않는 이유

- 강제로 완전히 오므려도 `cube_lift`가 `0.0003 m`임.
- 손이 큐브를 감싸지 못하므로 들어올릴 수 없음. lift 보상을 발견할 기회가 0임.
- 희소 보상만 추가되고 학습되지 않음.
- reach 교체로 파지 자세가 만들어진 뒤에 넣을 것.

## 2026-07-11 Reward Cleanup + Structure Summary + New Diagnostics

### 삭제한 reward 함수 (`mdp/rewards.py`) - 전부 미참조 확인 후 제거

- `body_to_object_position_tracking_bounded`.
- `bodies_to_object_position_tracking_bounded` (구 `finger_cube_closeness`).
- `object_in_functional_grasp_region` (구 `functional_hold`).
- `BodiesToObjectProgressReward` (구 `finger_cube_reach`).
- `object_lift_bounded`는 다음 단계(`cube_lift`)에 필요하므로 유지함.

### 정리한 reward cfg (`CubeGraspRewardsCfg`)

- 죽은 `= None` 항목(`arm_cube_reach`, `finger_cube_reach`, `finger_cube_closeness`, `functional_hold`, `cube_lift`, `cube_goal_tracking`)과 주석 처리된 블록을 전부 제거함.
- 제거 사유는 클래스 docstring에 요약해 남김. 같은 실수 반복 방지용임.
- `CAGE_BODIES` 상수를 도입해 두 cage term이 동일한 `SceneEntityCfg`를 공유하게 함.
- 최종 구성: `finger_cage_reach` (0.3), `finger_cage_hold` (1.0), `action_rate` (-0.0003). 3개뿐임.

### 전체 구조 요약 (왜 이렇게 되었는가)

- 모든 이전 reward는 "손끝 -> 큐브 **중심**" 거리였음.
- 큐브 중심은 표면에서 `0.03 m` 안쪽임. 손끝이 물리적으로 도달 불가능한 지점임.
- 즉 reward의 최댓값이 도달 불가 지점에 있고, gradient가 항상 큐브 속을 향함.
- 게다가 `body_weights=(3,1,1)`로 엄지가 가중평균의 60%를 차지함.
- 따라서 가장 싼 해법은 "엄지 하나만 큐브에 박고 나머지는 방치"였음.
- 실측 결과가 정확히 그것임. thumb `0.017`, index `0.072`, middle `0.078` (표면까지).
- 그 자세에서는 엄지-중지 선분이 큐브를 관통하지 않음.
- 그래서 손을 오므리면 가상점이 큐브 **밖으로** 빠져나감. 강제 오므림 시 `cage_inside_frac`이 `0.47` -> `0.40`으로 하락함.
- 정책은 reward 최적점에 정확히 앉아 있었고, 오므리기를 거부한 것이 합리적이었음.
- 즉 cage reward 자체는 옳았고, **cage가 작동할 수 없는 자세를 다른 두 항이 만들어놓은 것**이 정확한 인과임.
- 부차적으로 거리 reward는 접촉도 처벌함. 만지면 큐브가 밀려나 거리가 늘어남. `cube_displacement`가 `0.146` -> `0.037`로 감소한 것이 증거임.
- 새 `finger_cage_reach`는 목표가 큐브 **표면**(SDF)이라 도달 가능하고, 엄지 편중이 없으며, 6개 가상점 전체가 큐브에 겹쳐야 만족됨.
- reach와 hold가 같은 6점을 공유하므로 reach가 만든 자세에서 hold가 바로 작동함. 둘이 싸우지 않음.

### 새 진단 metric 2종 추가 (`managers.py`)

- `thumb_middle_opposition`: 큐브에서 본 엄지끝과 중지끝 방향벡터의 내적 부호 반전.
- `+1`이면 두 손가락이 큐브 **양쪽**에 있음. `-1`이면 **같은 쪽**임.
- `cage_inside_frac`만으로는 이것을 볼 수 없음. 선분이 큐브 모서리만 스쳐도 일부 점은 내부에 들어가기 때문임.
- `cage_span`: 엄지끝-중지끝 거리. 오므리면 줄어듦. `0.06 m` 큐브를 쥐면 큐브 폭 근처여야 함.
- smoke test에서 랜덤 정책 기준 `thumb_middle_opposition = -0.978`, `cage_span = 0.166 m`.
- 지표 22종 x 4 네임스페이스 + 보상 3종 x 3 = 총 97개 스칼라임.

## 2026-07-11 손등이 바닥에 닿는 문제 (사용자 관찰, 실측 확인)

- 사용자가 GUI에서 "엄지만 큐브에 올라가고 손등이 바닥에 닿는 구조"라고 관찰함.
- checkpoint `2026-07-11_19-02-58/model_200.pt` 안착 후 world z 실측:
- `palm_link` `0.082 m`.
- `finger1_tip` (엄지) `0.059 m`. 큐브 윗면(`0.061`)에 거의 올라가 있음.
- `finger2_tip` (검지) `0.030 m`. 바닥에서 `3 cm`.
- `finger3_tip` (중지) `0.018 m`. 바닥에서 `1.8 cm`. 사실상 바닥임.
- 큐브 중심 `0.031 m`, 윗면 `0.061 m`.
- `thumb_middle_opposition = -0.98`. 엄지와 중지가 큐브의 같은 쪽에 있음.

### 원인

- 큐브가 맨바닥(ground plane) 위에 있음. 테이블이 없음 (`reach_env_cfg.py:47`에 주석 처리됨).
- 팔이 `z=0.03`까지 완전히 굽혀 내려가야 함.
- 그 자세에서는 손목이 손을 회전시킬 여유가 없고(특이점 근처, manipulability 최악), 손등이 바닥에 눌려 손가락이 큐브를 감쌀 각도가 나오지 않음.
- 논문이 `r_MP` (manipulability penalty)를 넣고 "The arm is set to a neutral configuration with high manipulability"라고 명시한 것이 정확히 이 문제임.
- 논문의 물체들은 모두 테이블 위에 있음.

### 판정 실험 (테이블 도입 전에 먼저 확인할 것)

- 새 `finger_cage_reach`는 "파지 간극을 큐브에 겹쳐라"를 명시적으로 요구하므로 손가락이 큐브를 양쪽에서 물어야 만족됨.
- 옛 reward에는 그런 요구가 없었음.
- 따라서 `Metrics/cube_final/thumb_middle_opposition` 하나로 판정 가능함.
- `-1` 근처에 머물면 바닥이 물리적 병목임. 테이블 도입 필요함.
- `+0.5` 이상 오르면 reward가 해결한 것임. 테이블 불필요함.
- 테이블 도입은 arm 초기자세, 큐브 위치, 충돌 설정을 전부 다시 잡아야 하므로 필요 여부부터 확인할 것.

## 2026-07-12 Cage 12점 확장 (검지 추가)

### 문제: cage가 성공했으나 자세가 기괴함

- run `2026-07-11_19-50-45` (1583 iter)에서 cage reward가 완전히 수렴함.
- `thumb_middle_opposition`: `-0.965` → `+0.922`. 엄지와 중지가 큐브 양쪽에 위치함.
- `cage_inside_frac`: `0.000` → `0.837`. 가상점 6개 중 5개가 큐브 내부임.
- `cage_span`: `0.157` → `0.111`. 손을 오므림.
- `cage_sdf_mean`: `+0.597` → `-0.011`. 점들이 표면 안쪽임.
- orientation 항이나 opposition 항을 전혀 넣지 않고 cage 두 개만으로 달성함.
- 그러나 GUI 확인 결과 **검지와 중지가 서로 교차했고 손바닥이 하늘을 향함**.
- `enabled_self_collisions=True`이므로 물리 설정 문제가 아님. reward 문제임.

### 원인: reward가 검지와 손 자세를 전혀 제약하지 않음

- `finger_cage_reach`/`finger_cage_hold`는 "엄지끝 ↔ 중지" 선분과 큐브의 관계만 봄.
- 검지(`finger2`)는 reward에 등장조차 하지 않음. 완전히 자유였음.
- 손바닥 방향, 손 전체 자세도 제약 없음.
- 따라서 "엄지와 중지가 큐브를 양쪽에서 물기만 하면" 나머지는 무엇을 하든 만점이었음.
- 정책은 reward를 정직하게 최대화했고, metric도 정직하게 만족됨.

### 논문에는 왜 이 문제가 없었는가

- 논문 reward는 `r(t) = r_grasp + r_man + r_MP + r_T`임.
- `r_grasp = r_hp + r_hr + lambda * r_hj`.
- `r_hr`이 손의 회전을, `r_hj`가 손가락 관절각을 목표 파지 자세로 붙잡음.
- 우리가 구현한 것은 `r_man`의 `r_reach` + `r_hold`뿐임.
- 논문이 "엄지-중지 선분 하나로 충분하다"고 한 것은 `r_grasp`가 손 자세를 붙잡은 상태에서의 관찰임.
- 큐브는 대칭이라 목표 파지 자세가 없어 `r_grasp`를 구현할 수 없었고, 그 결과가 현재 자세임.

### 수정: 가상점 6 → 12 (검지 추가)

- `CAGE_BODIES`를 `[thumb_tip, *opposing]` 5개로 확장함.
- `finger1_tip_link` (엄지끝, 기준점).
- `finger2_tip_link`, `finger2_link3` (검지 끝, 검지 중간마디). 신규.
- `finger3_tip_link`, `finger3_link3` (중지 끝, 중지 중간마디). 기존.
- 대향 body 4개 x 등간격 3점 = 12점.
- 논문이 명시적으로 제시한 확장임 ("it is straightforward to utilize several finger pairs at the same time").
- 엄지+검지+중지는 최종 목표인 젓가락 그립과 동일하므로 임시방편이 아님.

### 코드 변경

- `_cage_points` → `cage_points`로 일반화함. 정확히 3개 body 요구 → `[thumb, *opposing]` N개 허용.
- reward 수식(`object_in_finger_cage`, `ObjectCageProgressReward`)은 변경 없음. 점 개수만 늘고 평균을 냄.
- `managers.py`의 `_cage_body_names`를 동일하게 5개로 맞춤. reward와 metric이 같은 점을 보게 함.
- metric 신규 2종: `thumb_index_opposition`, `cage_span_index`.
- 검지 opposition을 따로 로깅해야 함. 중지만 보면 검지가 교차해도 알 수 없기 때문임.
- action shape 18, observation shape 57 불변임.
- `py_compile` 통과, smoke test 통과, 12점 생성 검증함.

### 부작용 (의도된 것)

- 12점 평균이므로 검지가 큐브에 닿지 못하면 `finger_cage_hold` 값이 통째로 낮아짐.
- 기존 "엄지-중지만" 자세는 점수가 약 절반으로 떨어짐. 더 이상 충분하지 않게 만드는 것이 목적임.
- 초반 gradient가 약해질 수 있음.

### 다음 단계

- fresh 학습. `--resume` 금지.
- 판정: `thumb_index_opposition`과 `thumb_middle_opposition`이 **둘 다** `+0.5` 이상으로 올라야 함.
- 손바닥 방향이 여전히 잘못되면 그때 손 자세 정규화(논문 `r_hj`의 축소판) 도입 검토.
- `cube_lift`는 자세가 정상화된 뒤에 넣을 것. 현재 자세(손바닥 위, 손등이 바닥)로는 애초에 들 수 없음.

## 2026-07-12 cube_lift 도입 (palm_facing 제안은 철회)

### 검토했다가 철회한 것: palm_facing reward

- 손바닥이 하늘을 보는 문제를 고치려고 "손바닥이 물체를 향하도록" 하는 reward를 제안했음.
- 손바닥 법선 축을 실측함. `palm_link` 로컬 `+x`임 (손가락이 오므라들 때 손끝이 이동하는 방향, 성분 `+0.92`).
- 그러나 사용자와 논의 중 이 접근이 자의적이라는 결론에 도달해 **철회함**.

### 왜 철회했는가

- 논문이 `r_hr` (목표 손 회전)을 주는 이유는 **기능(functional grasp)** 때문임.
- 드릴은 검지가 트리거에, 스프레이는 분사 버튼에 닿아야 함. "당길 수 있게 쥐어라"는 요구임.
- 드릴을 그냥 집어 올리기만 할 거라면 `g_r`이 필요 없음.
- 큐브에는 기능 요구가 없으므로 목표 회전이 필요 없는 것이 **맞음**.
- 사람이 손바닥 방향을 정해주면 그것이 자의적 제약이 되고, 젓가락에서는 어차피 버려짐.
- `opposition`은 "엄지와 손가락이 큐브 양쪽에 있는가"만 보므로 손바닥 방향과 무관함. `+0.5`를 넘어도 손바닥이 하늘일 수 있음. 사용자 지적이 정확했음.

### 대신 채택: cube_lift가 진짜 선별 기준임

- **들지 못하는 자세는 파지가 아님.** 하중을 견디는지만 물으면 자세를 지정할 필요가 없음.
- 논문도 같은 이유로 `r_lift`를 추가함. "satisfy the constraint without actually stably grasping the object" (fake success) 방지용임.
- 2026-07-11 run이 정확히 그 상태였음. `opposition +0.92`, `inside_frac 0.84`인데 `cube_lift 0.002 m`임.
- 손바닥이 하늘을 보든 무엇을 하든, 큐브를 들 수 있으면 진짜 파지임.

### 구현: `object_lift_in_cage` (`mdp/rewards.py`)

- 기존 `object_lift_bounded`를 교체함.
- `r = cage_gate * clamp(height / lift_height, 0, 1)`.
- `cage_gate`는 `object_in_finger_cage`의 값임 (`0..1`). 감싸지 않으면 lift 보상이 `0`임.
- gate가 없으면 "파지 없이 큐브를 튕겨 올리는" 편법이 가능함.
- 조밀형(dense)임. 현재 정책이 만드는 `2 mm` 상승도 gradient를 가짐.
- 희소형이었다면 영원히 `0`이라 학습이 시작조차 안 됨.
- `initial_height=0.03` (큐브 spawn 높이). reset이 x/y만 랜덤화하므로 유효함.
- `lift_height=0.08`. 이만큼 뜨면 만점임.

### 가중치

- `cube_lift` `3.0` >> `finger_cage_hold` `1.0` >> `finger_cage_reach` `0.3`.
- 논문의 `r_T >> r_orient >> r_hold >> r_reach` 순서를 따름.
- `action_rate` `-0.0003`.
- reward 4개. action shape 18, observation shape 57 불변임.
- `py_compile` 통과, smoke test 통과.

### 이번 학습이 답할 질문

- 12점 cage + lift로 "하중을 견디는 자세"가 나오는가.
- `Metrics/cube_final/cube_lift`가 `0.002`에서 올라가는가.
- `thumb_index_opposition`과 `thumb_middle_opposition`이 둘 다 오르는가.
- lift가 끝내 오르지 않으면 다음 용의자는 손가락 actuator gain임 (`stiffness=8.0`, `damping=0.5`).
- 이 값은 약지/새끼 떨림을 줄이려고 낮춰놓은 것이라 80g 큐브를 쥐기에 물를 수 있음.

### 다음 일반화 단계 (큐브 성공 후)

- 큐브를 직육면체로 교체하고 초기 orientation을 랜덤화함.
- 긴 축은 손의 파지 간극보다 길어 물 수 없으므로, cage가 짧은 축을 자동으로 찾게 됨.
- orientation 항 없이 물체 기하가 스스로 제약함.
- 여기까지 되면 젓가락은 "긴 물체를 특정 방식으로"라는 문제가 되고, 그때 논문의 `r_hr` (진짜 목표 회전)이 등장함.
