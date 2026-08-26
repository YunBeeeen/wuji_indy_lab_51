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
- At that point, active cube grasp reward terms were `arm_cube_reach`, `finger_cube_reach`, `cube_lift`, `cube_goal_tracking`, and `action_rate`.
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

- Confirmed Wuji finger actuator gain is `stiffness=20.0`, `damping=0.5`, `friction=0.02`, `effort_limit=0.6` (2026-07-12에 stiffness를 `8.0`에서 올림. damping은 한때 `2.5`였으나 **최대 폐합 속도 = effort_limit/damping = 0.24 rad/s로 손가락이 5배 느려져** `0.5`로 되돌림).
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
- ring/little finger는 policy 제어 대상이 아니고 actuator gain(`stiffness=20.0`, `damping=0.5`)으로만 안정화함.
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
- lift가 끝내 오르지 않으면 다음 용의자는 손가락 actuator gain임 (`stiffness=20.0`, `damping=0.5`).
- 이 값은 약지/새끼 떨림을 줄이려고 낮춰놓은 것이라 80g 큐브를 쥐기에 물를 수 있음.

### 다음 일반화 단계 (큐브 성공 후)

- 큐브를 직육면체로 교체하고 초기 orientation을 랜덤화함.
- 긴 축은 손의 파지 간극보다 길어 물 수 없으므로, cage가 짧은 축을 자동으로 찾게 됨.
- orientation 항 없이 물체 기하가 스스로 제약함.
- 여기까지 되면 젓가락은 "긴 물체를 특정 방식으로"라는 문제가 되고, 그때 논문의 `r_hr` (진짜 목표 회전)이 등장함.

## 2026-07-12 palm_facing 도입 + cube_lift 기울이기 편법 차단

### 사용자 관찰 (GUI)

- 12점 cage 학습(run `2026-07-12_13-43-04`) 결과 **손바닥이 여전히 하늘을 향함**.
- `cube_lift`가 오른 이유는 파지가 아니라 **손가락 2개가 오므리면서 큐브를 기울여 튕겨 올린 것**임.
- 즉 `cube_lift`도 해킹당하고 있었음.

### 실측 확인 (checkpoint `model_550.pt` replay)

- `cube_lift` (중심 높이): `+4.28 mm`.
- `cube_clearance` (최하 모서리 높이): `-0.04 mm`. **바닥에 붙어 있음.**
- `palm_facing`: `0.182` (`+1`이 정면). 손바닥이 큐브를 거의 보지 않음.
- `cage_inside_frac`: `0.753`. cage는 만족됨.
- 결론: 큐브를 모서리로 세워 중심만 올린 것임. 기울이기 편법 확정됨.

### 원인 1: `cube_lift`가 큐브 **중심 높이**를 봤음

- 큐브를 기울이면 한 모서리가 바닥에 남아도 중심 높이는 올라감.
- `cage_gate`도 만족됨. 가상점이 큐브 안에 있으므로.
- 수정: **최하 모서리 높이(`box_ground_clearance`)** 기준으로 변경함.
- 큐브 8개 꼭짓점을 world로 변환해 최소 z를 구하고 ground plane 대비 높이를 씀.
- 기울이면 한 모서리가 바닥이므로 최소 z = 0 → 보상 0. **편법이 원천 차단됨.**
- `initial_height` 파라미터 제거함. ground 기준이므로 불필요함.

### 원인 2: 손바닥 방향이 여전히 제약되지 않음

- 12점 cage로도 손바닥 방향을 못 잡음.
- 이유: **엄지끝-손가락 선분은 손 방향과 무관하게 큐브를 관통할 수 있음.**
- 따라서 손바닥이 하늘을 봐도 cage가 만점이 나옴.

### 선행 검증: 손바닥을 아래로 돌릴 수 있는가

- reward를 넣기 전에 kinematic 도달성부터 확인함. 팔 관절 40만 개 샘플링.
- 큐브 중심 높이 `0.030 m`.
- 손바닥이 가장 가까이 간 거리 `0.016 m`.
- 20cm 이내에서 최고 `palm_facing` = **`+1.000`** (완벽히 정면).
- `(거리 < 0.20 AND facing > 0.7)` 자세: `561 / 397,312` (`0.14%`).
- **결론: 도달 가능함.** 바닥이 막는 것이 아님. 정책은 할 수 있는데 보상이 없어서 안 하는 것임.
- 다만 `0.14%`로 드물어 **명시적 보상 없이는 무작위 탐색으로 못 찾음.**

### `palm_facing_object` 도입 (weight `0.5`)

- `r = clamp(dot(손바닥 법선, 단위벡터(큐브 - 손바닥)), 0, 1)`.
- 손바닥 법선은 `palm_link` 로컬 `+x`임. 손가락을 오므릴 때 손끝이 이동하는 방향으로 실측함.
- **자의적 제약이 아니라 물리적 필요조건임.** 손가락은 손바닥 쪽으로 굽으므로, 손바닥 뒤에 있는 물체는 오므려도 감싸지지 않음.
- 특정 회전을 지정하지 않음. 손바닥 법선 축만 정렬하고 **roll은 자유**임.
- 대칭 물체에 대해 파지 방식을 고르지 않음. "잡는 것이 가능한 자세"만 요구함.
- 논문 `r_hr`은 목표 회전 3자유도를 전부 맞춤. `palm_facing`은 그중 **물리적으로 반드시 필요한 성분 1개**만 요구함.
- 젓가락에서는 기능이 손 회전을 결정하므로 `r_hr`(진짜 목표 회전)이 이것을 **상위호환 교체**함. 버려지는 것이 아니라 승격됨.

### 이전 판단 정정

- 2026-07-12 초반에 `palm_facing`을 "자의적"이라며 철회하고 "`cube_lift`가 자세를 선별할 것"이라고 판단했음.
- **틀렸음.** `cube_lift`가 기울이기로 해킹당했고, 손바닥은 여전히 하늘을 봄.
- 사용자가 "opposition은 0.5를 넘어도 손바닥이 하늘일 수 있다"고 지적한 것이 옳았음.

### 최종 reward 구성

- `finger_cage_reach`: `0.3`.
- `finger_cage_hold`: `1.0`.
- `cube_lift`: `3.0`. 최하 모서리 기준으로 수정됨.
- `palm_facing`: `0.5`. 신규.
- `action_rate`: `-0.0003`.
- action shape 18, observation shape 57 불변.

### 신규 metric

- `cube_clearance`: 큐브 최하 모서리의 지면 대비 높이. **진짜 lift 지표임.**
- `palm_facing`: 손바닥이 큐브를 향하는 정도.
- 기존 `cube_lift`(중심 높이)도 유지함. **두 값이 벌어지면 기울이기 편법 재발 신호임.**
- `managers.py`의 `_palm_normal_b`는 reward의 `palm_normal_b`와 반드시 동일해야 함.

### 학습 판정 지표

- `Metrics/cube_final/palm_facing`: `0.18` → `0.7` 이상으로 오르는가.
- `Metrics/cube_final/cube_clearance`: `0`에서 오르는가. **진짜 파지 여부임.**
- `Metrics/cube_final/cube_lift`가 `cube_clearance`보다 크게 앞서면 또 기울이는 중임.
- `--resume` 금지. reward가 바뀜.

## 2026-07-13 palm_facing 국소최적 + arm_manipulability (r_MP) 도입

### 문제: 정책이 31cm 밖에서 손바닥만 겨누고 서 있음

- run `2026-07-12_20-11-57` (1803 iter, 완전 수렴).
- `palm_facing`: `0.013` → **`0.994`**. 손바닥은 완벽히 큐브를 향함.
- 그러나 `cage_sdf_mean`: `0.605` → **`0.313`에서 정지**. 손이 31cm 밖임.
- `cage_inside_frac` = `0`, `finger_cage_hold` = `0`, `cube_clearance` = `0`.
- `cube_displacement` = `0.0022`. **큐브를 건드리지도 않음.**

### 원인 1: `palm_facing`이 거리와 무관하게 지급됨

- `palm_facing = clamp(dot(법선, 단위벡터(큐브 - 손바닥)), 0, 1)`.
- **단위벡터라 거리가 사라짐.** 10m 밖에서 겨눠도 만점임.
- 보상 분해: `0.5 x 0.994 x 0.4(dt) x 20(step) = 3.96`. 전체 `mean_reward` `4.015`의 **98.6%**임.
- 나머지 항은 전부 `0`이었음.

### 원인 2: 접근하면 손해였음

- 도달성 테스트에서 "닿으면서 정면"인 자세는 `0.14%`뿐이었음. 가까이 가면 정면 유지가 어려움.
- `31cm -> 1cm` 접근 시 손익 계산:
- 얻는 것: `reach`(차분) = `(0.30 / 0.5) x 0.3 x 0.4 = +0.072` (1회성).
- 잃는 것: `palm_facing`이 `0.99 -> 0.7`로 떨어지면 `(0.29) x 0.5 x 0.4 x 20 = -1.16` (매 step).
- **접근이 순손실 `-1.09`.** 정책은 완벽히 합리적으로 행동한 것임.
- 논문이 경고한 그대로임. "local minima created by accumulating rewards for actions that are **easier to achieve**".
- `palm_facing(0.5) > reach(0.3)`인데 훨씬 쉬움. **가중치 순서가 뒤집혔음.**

### 원인 3 (사용자 GUI 관찰): 팔이 접혀서 특이점에 빠짐

- 사용자가 GUI에서 "arm 자세 때문에 못 가는 것 같다"고 관찰함.
- **manipulability 실측** (`sqrt(det(J J^T))`, palm_link 기준 arm 6축):
- 무작위 자세 최대: `0.1129` (기준 100%).
- 리셋 직후 초기 자세: `0.0645` (`57%`).
- **정책이 수렴한 자세: `0.0144` (`13%`)**. `joint2`가 한계의 `81%`까지 접힘.
- **손을 자유롭게 못 움직이는 자세임.**
- 인과: `palm_facing`을 만족시키는 **가장 싼 방법이 "팔을 접어서 손목만 돌리기"**였음. 팔을 뻗는 것보다 훨씬 쉬움.
- 그 결과 특이점 근처로 가서 접근 자체가 불가능해짐.

### 수정 1: `palm_facing`에 proximity 램프 + 가중치 인하

- `proximity = 1 - clamp(손바닥-큐브 거리, 0, distance_max) / distance_max`.
- `r = facing * proximity`.
- `distance_max = 0.8`. **에피소드 시작 거리(약 `0.7 m`)보다 크게 잡음.**
- 이유: 리셋 직후에도 `proximity`가 `0`이 아니어야 **접근하는 동안 손바닥을 돌림.**
- 하드 게이트면 도착할 때까지 보상이 `0`이라, 큐브 바로 옆에서 손바닥을 급회전시켜 **큐브를 날려버림.**
- 거리별 `proximity`: `0.70 m` -> `0.125`, `0.31 m` -> `0.61`, `0.10 m` -> `0.875`, `0.05 m` -> `0.94`.
- **접근하면 보상이 오름.** 정렬이 다소 깨져도 순이득이라 "서서 겨누기" 국소최적이 사라짐.
- weight `0.5` -> `0.3`. `finger_cage_reach`와 동급이고 `hold(1.0)`, `lift(3.0)`보다 낮음.
- precondition은 가벼워야 함. 논문의 `r_T >> r_orient >> r_hold >> r_reach` 순서를 지킴.

### 수정 2: `arm_manipulability_penalty` (논문 Eq.17, weight `1.0`)

- `r = 1 - 2 / (1 + (min(|J|, j_max) / j_max)^3)`. 범위 `[-1, 0]`.
- `|J|`는 `sqrt(det(J J^T))`. `palm_link` 기준 arm 6축 Jacobian임.
- `j_max = 0.02`. 논문은 "관측 최대치의 15%"라 하며, 실측 최대치가 `0.113`이므로 `0.017`에 해당함.
- 초기 자세(`0.0645`) -> 페널티 `0`. 접힌 자세(`0.0144`) -> 페널티 약 `-0.46`.
- 논문 원문: "This reward penalizes coming close to singularities and leads to learning more intuitive motions."
- 논문은 이 항을 스케일하지 않음 ("We leave the other rewards unscaled"). weight `1.0`.

### 신규 metric

- `arm_manipulability`: `sqrt(det(J J^T))`. 팔이 접히는지 감시함.
- 기준값: 초기 자세 `0.064`, 무작위 최대 `0.113`. **`0.02` 아래로 떨어지면 특이점 근처임.**
- `managers.py`의 `_arm_joint_ids`는 reward의 `joint_names=["joint[0-5]"]`와 동일해야 함.

### 최종 reward (6개)

- `finger_cage_reach` (0.3), `finger_cage_hold` (1.0), `cube_lift` (3.0).
- `palm_facing` (0.3, proximity 램프), `arm_manipulability` (1.0), `action_rate` (-0.0003).
- action shape 18, observation shape 57 불변.
- `py_compile` 통과, smoke test 통과.
- 검증: 초기 자세 `arm_manipulability = 0.0625`(이전 측정 `0.0645`와 일치), 페널티 `0`.
- 검증: `palm_facing` raw `0.0014` (램프 전 `0.017`). 멀리서 겨누기가 억제됨.

### 학습 판정 지표

- `Metrics/cube_final/arm_manipulability`: `0.02` 아래로 떨어지면 팔이 접히는 중임.
- `Metrics/cube_final/cage_sdf_mean`: `0.05` 아래로 내려가는가 (도착).
- `Metrics/cube_final/palm_facing`: 접근하면서 오르는가.
- `Metrics/cube_final/cube_clearance`: `0`에서 오르는가. **진짜 파지 여부임.**
- `Metrics/cube_final/cube_displacement`: 접근 중 큐브가 날아가는가 (급회전 부작용 감시).
- `--resume` 금지.

### 교훈

- **어떤 reward든 "가장 싼 만족 방법"이 무엇인지 먼저 물을 것.** `palm_facing`의 가장 싼 방법은 팔을 접는 것이었음.
- **precondition 성격의 reward는 거리/진행도에 비례해야 함.** 절대값으로 주면 그 자리에서 만족시키고 멈춤.
- **논문의 가중치 순서(`r_T >> r_hold >> r_reach`)는 장식이 아님.** 쉬운 항이 무거우면 반드시 거기 갇힘.
- 이번에 `palm_facing(0.5) > reach(0.3)`으로 넣었다가 정확히 그 함정에 빠짐.

## 2026-07-13 palm_facing을 차분형으로 (논문 원칙 복원)

### 사용자 지적: 논문과 가중치 비율이 다름

- 확인 결과 **가중치가 아니라 "형태"가 달랐음.**

### 논문의 항들은 거의 전부 차분(differential)형임

- `r_hp` (Eq.9): `[Δh_p(t-1) - Δh_p(t)] / Δh_p^max`. 차분.
- `r_hr` (Eq.10): `[Δh_r(t-1) - Δh_r(t)] / Δh_r^max`. 차분. **손 회전.**
- `r_hj` (Eq.11): 차분.
- `r_reach` (Eq.14): 차분.
- `r_orient` (Eq.16): 차분.
- `r_hold` (Eq.15): **절대. 논문에서 유일한 절대형임.**
- 논문 원문: "Wide use of differential distances in our reward, instead of directly using the velocities, naturally avoids learning overshooting behaviors."

### 왜 논문은 `r_grasp`에 weight 1.0을 줘도 안전한가

- **차분형은 "가만히 있으면 0"임.** 정렬해놓고 앉아 있어도 보상이 나오지 않음.
- 따라서 farming이 원천적으로 불가능하고, 큰 weight를 줘도 국소최적이 생기지 않음.
- 우리 `palm_facing`은 **절대형**이라 정렬만 유지해도 매 step 보상이 나왔음.
- 그래서 weight `0.5`에서도 전체 보상의 `98.6%`를 먹고 국소최적을 만들었음.
- **가중치를 낮추는 것이 아니라 형태를 바꾸는 것이 정답이었음.**

### 수정: `PalmFacingProgressReward` (차분형)

- `r(t) = facing(t) - facing(t-1)`. `reset()`에서 기준선 seeding.
- 에피소드 총합이 `facing(final) - facing(reset)`으로 telescoping됨.
- **가만히 있으면 정확히 0.** 정렬이 깨지면 감점.
- `clamp(-1, 1)`. `facing`이 `[0,1]`이므로 차분은 `[-1,1]`임.
- proximity 램프 제거함. 차분형이면 거리 의존성이 불필요함. 거리는 `reach`가 담당함.
- `palm_facing_object`는 절대형 그대로 유지함. **metric 전용**으로 쓰고 reward는 차분 클래스를 씀.

### 가중치는 논문을 그대로 베끼지 않고 우리 구조에 맞춤 (사용자 판단)

- 우리 구조에는 `r_orient`도 `r_T`도 없고, `cube_lift`가 `r_T` 자리를 대신함.
- 에피소드당 실제 기여량을 계산해서 비율을 잡음:
- `finger_cage_reach` (차분, `0.3`): `0.6m / 0.5 x 0.3 x 0.4` = 약 `0.14`.
- `palm_facing` (차분, `1.0`): `1.0 x 1.0 x 0.4` = 약 `0.40`.
- `finger_cage_hold` (절대, `1.0`): `0.6 x 1.0 x 0.4 x 20 step` = 약 `4.8`.
- `cube_lift` (절대, `3.0`): `0.5 x 3.0 x 0.4 x 20 step` = 약 `12.0`.
- `arm_manipulability` (절대 페널티, `1.0`): `0` (정상 자세) ~ `-8` (특이점).
- `action_rate`: 약 `-0.2`.
- **차분형은 한 번만 지급되므로(telescoping) 절대형과 규모가 근본적으로 다름.**
- `reach`와 `palm_facing`이 "유도"(차분, 1회), `hold`와 `cube_lift`가 "유지/달성"(절대, 매 step) 역할임.
- `palm_facing`을 `reach`와 같은 급으로 두고 `hold`/`lift`보다 한 단계 아래로 배치함.

### 최종 reward (6개)

- `finger_cage_reach`: 차분, `0.3`.
- `palm_facing`: **차분**, `1.0`.
- `finger_cage_hold`: 절대, `1.0`.
- `cube_lift`: 절대, `3.0`.
- `arm_manipulability`: 절대 페널티, `1.0`.
- `action_rate`: `-0.0003`.
- action shape 18, observation shape 57 불변.
- `py_compile` 통과, smoke test 통과.
- 검증: `Episode_Reward_Raw/palm_facing = 0.0008` (차분), `Metrics/cube_final/palm_facing = 0.0170` (절대). 분리 확인됨.

### 교훈 (중요)

- **절대형 reward는 "그 상태를 유지하는 것"에 매 step 지급됨. 유지가 쉬우면 반드시 farming당함.**
- **차분형 reward는 "그 상태로 가는 것"에만 지급됨. 도달 후에는 0이므로 farming 불가.**
- 논문이 `r_hold` 하나만 절대형으로 둔 것은 우연이 아님. **hold는 "유지"가 목적이므로 절대형이 맞고, 나머지는 전부 "유도"이므로 차분형이어야 함.**
- 새 reward를 넣을 때 **"이건 유도인가 유지인가"**를 먼저 물을 것. 유도면 차분, 유지면 절대.

## 2026-07-13 작업공간 측정: 큐브 위치가 "가능하지만 찾기 어려운" 지점임

### 사용자 지적

- GUI에서 팔이 접힌 것을 보고 "큐브가 베이스에 너무 가까운 것 아닌가", "x쪽으로 더 빼야 학습이 빠를 것 같다"고 지적함.

### 1차 측정 (최댓값만 봄) — 잘못된 결론

- 바닥 위 위치별로 "손바닥을 아래로 향한 채" 낼 수 있는 **최고** manipulability를 측정함.
- 현재 큐브 위치(베이스로부터 `0.485 m`): 최고 `0.1159`.
- 바닥 전체 최적(`0.62~0.70 m`): 최고 `0.1251`.
- **"93% 수준이니 문제 없다"고 결론냈으나 이것이 틀렸음.**

### 2차 측정 (밀도를 봄) — 사용자가 옳았음

- "가능한가(최댓값)"와 "찾기 쉬운가(평균/좋은자세 비율)"는 **다른 질문**임.

```text
거리            최고 manip    평균 manip    manip>0.08 비율
0.46~0.54 m      0.1159       0.0447          11.9%    <- 현재 큐브
0.54~0.62 m      0.1229       0.0468          14.8%
0.62~0.70 m      0.1251       0.0585          22.6%
0.70~0.78 m      0.1243       0.0666          34.2%    <- 3배 쉬움
```

- **최댓값은 비슷하지만(`0.116` vs `0.125`), 좋은 자세를 만날 확률이 3배 차이남** (`11.9%` -> `34.2%`).
- 평균 manipulability도 `0.045` -> `0.067`로 1.5배임.
- **"가능하지만 찾기 어렵다"**가 정확한 진단임. 사용자 직관이 맞았음.

### 그럼에도 이번 run은 유지하기로 함

- `3배`는 **무작위 탐색 기준**임. 우리는 `arm_manipulability` (r_MP) 페널티가 있어 **경사를 타고 올라감.**
- 실제로 현재 run에서 manipulability가 `0.0527`에 유지됨 (이전 `0.0144`에서 4배 개선). r_MP가 작동 중임.
- `11.9%`는 4096 env x 20 step 기준으로 매 iteration마다 수천 번 마주치는 확률임. 희귀하지 않음.
- 재시작하면 진행 중인 run을 버려야 함. 이번 run의 목적은 `r_MP` + 차분형 `palm_facing`의 효과 확인임.
- **한 번에 하나씩** 원칙 유지.

### 다음 run 권장 사항

- 큐브 위치를 `x`쪽으로 이동: `(0.45, -0.18, 0.03)` -> `(0.62, -0.18, 0.03)`.
- 베이스로부터 `0.485 m` -> `0.646 m`. 좋은 자세 비율 `11.9%` -> `22.6%`.
- **공짜로 밀도 2배를 얻음.** 논문도 "objects are spawned such that at least 75% of their bounding box is in the manipulation workspace"라고 명시함.
- 이번 run이 manipulability 부족으로 정체하면(`0.053`에서 안 오르고 파지 실패) 그때가 옮길 명분임.

### 교훈

- **"가능한가"와 "찾기 쉬운가"는 다른 질문임.** 최댓값만 보면 안 되고 **분포(평균/비율)**를 봐야 함.
- 학습 속도를 논할 때는 **최적해의 존재**가 아니라 **최적해의 밀도**가 중요함.

---

## 2026-07-13 [결정적] 제어 안 되는 약지/새끼가 큐브를 쳐내고 있었음

### 사용자 영상 관찰 (둘 다 정확했음)
- "손이 cube를 밀어버리고 swing out 현상이 벌어짐"
- "충돌 때문에 팔이 뒤로 튕겨져 나갔다가 다시 접근하는 걸 반복. 손바닥이 확실히 더 돌지 않아서
  새끼랑 약지손가락이 자꾸 접촉을 발생시킴"

### 지표 확증 (`2026-07-13_11-49-57`, 181 iter)
```
                   iter 0     60     120     180
cube_displacement  0.003  0.024  0.046  0.125   ← 큐브를 12.5cm 밀어냄
cage_sdf_mean      0.590  0.496  0.455  0.487   ← 120에서 바닥 찍고 다시 멀어짐
MAX cage_sdf_mean  0.784  0.813  0.773  0.944   ← 에피소드 중 94cm까지 벌어짐 (튕겨나감)
cage_inside_frac   0      0      0      0
```

### 원인: 액추에이터 ≠ 액션
```python
액추에이터  finger[1-5]_joint[1-4]  stiffness=20   ← 5개 손가락 전부 구동
초기 자세   finger[1-5]_joint[1-4] = 0.0           ← 전부 쫙 편 상태
액션        finger[1-3]_joint[1-4]                 ← 정책은 3개만 제어
```
`finger4`(약지)/`finger5`(새끼)는 **제어 불가 + 편 자세에 뻣뻣하게 고정 + 콜리전만 켜짐 + cage에도 미포함.**

**초기 자세 손끝 x좌표 (결정적 증거):**
```
palm 0.660 | 검지 0.825 | 중지 0.837 | 약지 0.841(제어X) | 새끼 0.838(제어X)
```
**제어 못 하는 두 손가락이 손에서 제일 앞으로 튀어나와 있었음.** 접근하면 무조건 먼저 큐브를 치고,
stiffness=20으로 버티며 밀어내고, 반작용이 팔을 튕겨냄.

### 수정 — `grasp/indy_wuji/env_cfg.py`
```python
joint_pos = self.scene.robot.init_state.joint_pos
joint_pos.pop("finger[1-5]_joint[1-4]", None)   # 필수: 안 지우면 정규식 중복 매칭 ValueError
joint_pos.update({
    "joint1": -0.45, "joint2": -1.85, "joint4": 1.20,
    "finger[1-3]_joint[1-4]": 0.0,
    "finger[4-5]_joint1": 1.20,   # 한계 +1.636
    "finger[4-5]_joint2": 0.0,    # 벌림 중립
    "finger[4-5]_joint3": 1.20,   # 한계 +1.627
    "finger[4-5]_joint4": 1.20,   # 한계 +1.627
})
```
액션에 없는 관절은 액추에이터가 `default_joint_pos`를 목표로 유지 → **접힌 채 굳음.**
3지 파지 자세이므로 물리적으로 정당하고, **action dim은 18D 그대로.**

### 손가락 관절 한계 (측정값)
```
finger1_joint1 [-0.045,+1.651]  finger[2-5]_joint1 [-0.327,+1.636]  (굽힘)
finger1_joint2 [-0.166,+0.934]  finger[2-5]_joint2 [-0.495,+0.495]  (벌림)
finger[1-5]_joint3 [-0.493,+1.627]   finger[1-5]_joint4 [-0.493,+1.627]  (굽힘)
```

---

## 2026-07-13 두 run 비교 — 차분형 palm_facing 수정은 실제로 먹혔음

| | 07-12_20-11-57 | 07-13_11-49-57 |
|---|---|---|
| palm_facing | **절대형** w=0.5 | **차분형** w=1.0 |
| arm_manipulability | 없음 | w=1.0 |
| 학습량 | **1809 iter** | 181 iter |
| cage_sdf_mean | **0.312 정체** | 0.487 (진행 중) |
| cube_displacement | **0.0022 = 큐브를 한 번도 안 건드림** | 0.125 |
| finger_cage_hold | **0 (내내)** | 0 |

- **이전 run은 1809 iter 동안 큐브 근처에도 못 갔음** (31cm 밖 standoff, palm_facing만 0.995 farming)
- **지금 run의 displacement=0.125는 "드디어 닿았다"는 증거** → 차분형 수정 성공
- 즉 **후퇴가 아니라 전진.** 닿는 순간 약지/새끼 버그가 터질 뿐.

---

## 2026-07-13 커리큘럼 러닝 판단 — "아직 논할 단계가 아님"

**사용자 질문**: "흠 아니면 진짜로 curriculum learning 을 해야되려나"

**답: 아니오. 지금 `hold=0`은 탐색 문제가 아니라 물리적 불가능임.**
- 제어 못 하는 손가락이 큐브를 쳐내는 한 어떤 정책도 cage를 못 만듦
- 논문 커리큘럼(물체를 손 5cm 앞에 스폰)을 지금 적용하면 **더 나빠짐** — 더 가까우니 더 세게 침
- **커리큘럼은 이 버그를 못 고침**

**버그 수정 후 재판단할 것.** 판단 기준: `cage_inside_frac > 0`이 뜨는가.
- 뜨면 → 가중치 재조정(`approach ≪ hold ≪ lift`) 후 Phase 2
- 안 뜨면 → **그때 커리큘럼 도입** (논문: 없으면 ~50%±큰분산, 있으면 97%)

---

## 2026-07-13 교훈 (중요)

- **"제어하지 않는 관절" ≠ "존재하지 않는 관절".**
  액션 dim을 줄이려고 관절을 액션에서 빼도 **액추에이터·콜리전·질량은 그대로 남음.**
  빼려면 **초기 자세도 같이 설계**해야 함 (안 그러면 뻣뻣한 장애물이 됨).
- **지표는 "무엇이" 잘못됐는지 알려주지만 "왜"는 못 알려줌.**
  `cube_displacement` 급증은 지표로 보였지만, 원인(약지/새끼)은 **영상을 본 사용자가 짚었고 그게 맞았음.**
  → **지표 + 영상 둘 다 필요.**

## 2026-07-13 Added `Indy-Wuji-Cube-Grasp-Easy`

- Added a separate curriculum task: `Indy-Wuji-Cube-Grasp-Easy`.
- Kept the original hard task `Indy-Wuji-Cube-Grasp`.
- Implemented the easy task as a subclass of `Indy7WujiCubeGraspEnvCfg`.
- Kept action/observation/reward/model shapes identical to hard task.
- Changed only cube initial/reset distribution for the easy task.
- Easy cube initial position is `(0.74, -0.18, 0.03)`.
- Easy cube reset range is narrow: `x/y = +/-0.015`, `z = 0`.
- Registered the new gym id in `grasp/indy_wuji/__init__.py`.
- Intended flow:
  - train `Indy-Wuji-Cube-Grasp-Easy` first.
  - check that `finger_cage_hold` and `cage_inside_frac` become non-zero.
  - resume the checkpoint on `Indy-Wuji-Cube-Grasp`.
- Use `--run_name easy_close_start` for the easy run so the later hard resume command can target the easy checkpoint explicitly.
- Smoke test passed with `--num_envs 1 --max_iterations 1`.
- Verified Action Manager shape `18`, policy observation shape `57`, and 6 active reward terms.

## 2026-07-13 Softened active hand/cube contact response

- User video showed that after folding ring/little, the arm still bounced when the hand/palm approached the cube.
- Updated/confirmed the root cause hypothesis from ring/little-only collision to excessive contact response on active hand/palm collision.
- Active `INDY7_WUJI_RIGHT_CFG` now uses:
  - `max_depenetration_velocity=5.0`.
  - `max_contact_impulse=100.0`.
- Previous active values were effectively unbounded:
  - `max_depenetration_velocity=1000.0`.
  - `max_contact_impulse=1e32`.
- Those previous values could make PhysX resolve a small hand/cube penetration with a large impulse, kicking the arm away.
- This change does not alter action/observation/reward/model shapes.
- Existing checkpoints can be loaded for play comparison or resumed for adaptation.
- Because the physics changed, final performance should be judged after resume adaptation or a fresh/easy-curriculum run.

---

## 2026-07-13 [진짜 원인] action space 설계 오류 — 정책이 발산하고 있었음

### 사수님 지적이 사건을 해결함
> "튀었던 게 action이 실제로 저렇게 된 거면 학습이 잘못된 거다. 모르는 거니까 action을 찍어봐라"

**앞선 물리 진단(바닥 슬램/튕김)은 전부 틀렸음.** 손의 z 높이·관절 속도·큐브 이동량 같은 **결과**만
보고 "물리 반작용"이라 단정했으나, 그건 가정이었지 측정이 아니었음.

### 측정 (`model_350`, decimation 24 재현)
```
step   |a|평균  |Δa|최대   추종오차   팔속도   손최저z
   2    1.415   3.033     0.600     0.73     1.73 cm
   3    1.601   9.311     0.916     2.77   106.38 cm  ← 106cm로 솟구침
   5    1.090   3.379     0.452     1.35    15.02 cm
   6    1.928   3.066     0.785     0.62     0.42 cm  ← 바닥
  ... 주기 5~6 step 무한 반복. 팔속도가 velocity_limit(2.775)에 딱 붙어 있음
```

### 판정: 물리 튕김이 아니라 **정책이 팔을 휘두른 것**
- `|a|` 평균 1.5 — action 범위가 `[-1,1]`인데 **정의 범위를 벗어남**
- `|Δa|` 최대 9.66 — `scale=0.2`이므로 관절 목표가 한 step에 **1.93 rad(110°) 점프**
- 팔속도가 정확히 velocity_limit → 튕김이면 충돌 후 잦아들어야 하는데 **일정 주기로 최대속도 왕복**
- 추종오차가 큰 것도 물리 탓이 아니라 **도달 불가능한 목표를 명령해서**임
  (0.4초에 팔이 갈 수 있는 최대 1.11 rad < 명령 1.93 rad)

### 근본 원인: `clip_actions` 미설정 + `scale` 과소
```python
# rsl_rl_cfg.py 에 clip_actions 자체가 없었음 -> None -> 상한 없음
# scripts/rsl_rl/{train,play}.py: RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
```
```
scale=0.2, |a|=1  ->  관절이 default에서 0.2 rad(11°)만 움직임
손은 큐브까지 55cm를 내려가야 함  ->  |a|=1로는 절대 도달 불가
-> 정책이 살려고 |a|를 5, 10까지 밀어냄  ->  상한이 없으니 발산
```
**둘 중 하나만 고치면 안 됨:** clip만 1.0으로 걸면 관절이 ±11°만 움직여 큐브에 영영 못 감.
scale만 키우면 여전히 상한이 없어 발산.

### 수정 — 도달성 먼저 검증 (4096 x 8회 샘플링, default ± s 범위)
```
 s (rad)  |  손끝 3개가 전부 큐브 표면에 닿는 최소 거리
    0.5   |   3.63 cm   ← 도달 불가! scale 0.5면 큐브를 못 잡음
    1.0   |   0.20 cm   ← 도달 가능. 최적
    1.5   |   0.45 cm
    2.0   |   1.94 cm   ← 넓힐수록 오히려 나빠짐 (탐색 공간만 커짐)
```
```python
scale = 1.0          # grasp/indy_wuji/env_cfg.py.  관절 목표 = default ± 1.0 rad
clip_actions = 1.0   # learning/rsl_rl_cfg.py.  없었음 — 발산의 직접 원인
action_rate = -0.005 # -0.0003이었음. 발광 비용이 접근 보상의 4%라 사실상 공짜였음
```

---

## 2026-07-13 제어 주파수 / reward 스케일 재조정

### 최종 설정 (재학습 필요, `--resume` 불가)
```
sim.dt = 1/60,  decimation = 2  ->  30 Hz,  240 step/episode,  8초
action: scale 1.0, clip_actions 1.0, 18D

finger_cage_reach   차분   25.0   최대 +1.0     <- dt 보정(x12) + 순서 교정
finger_cage_hold    절대    1.0   최대 +4.0
cube_lift           절대    3.0   최대 +6~12    <- 들고 있던 "시간"에 비례
palm_facing         차분    9.0   최대 +0.29    <- reach보다 싸게
arm_manipulability  페널티  1.0
hand_floor          페널티  0.5   최악 -2.0     <- 신규
action_rate                -0.005               <- 17배 강화

순서:  palm(0.29) < reach(1.0) < hold(4.0) < lift(6~12)   <- 논문 논리
```

### `--resume` 불가인 이유
`scale` 0.2->1.0 (같은 출력이 5배 큰 움직임), `clip_actions` 추가 (기존 |a|~1.5가 전부 잘림),
`decimation` 24->2 (dynamics 변화), reward 재조정 (가치함수 무효화).

### **reward manager가 `raw * weight * dt`로 누적함 — 중대**
```
절대형 (hold, lift, hand_floor, manip)  에피소드 합 = 평균값 x 8초   -> decimation 무관, 불변
차분형 (reach, palm_facing)             에피소드 합 = dt x 총변화량  -> dt에 비례
```
**decimation을 바꾸면 차분형 weight를 반드시 같이 바꿔야 함.** dt 0.4 -> 1/30이면 12배 약해짐.

### iteration 숫자를 이전 run과 비교하지 말 것
```
iteration당 시뮬 시간 = num_steps_per_env x decimation x sim.dt
이전: 24 x 24 / 60 = 9.60초/iter
지금: 24 x  2 / 60 = 0.80초/iter   ← 경험량이 12배 줄어듦
```
5000 iter는 이전 run(1809 iter)의 **23%**에 불과함. `--max_iterations 25000` 이상 권장.

### sim.dt 판단 — 지금은 1/60 유지
```
                   physics dt  decimation  제어 Hz
Allegro / in-hand    1/120         4        30 Hz
ShadowHand           1/120        2~3     40~60 Hz
Lift-Cube-Franka     1/100         2        50 Hz
사수님                1/200        20        10 Hz   (실기 구조: PD 200Hz + 정책 10Hz)
우리                  1/60          2        30 Hz
```
- 제어 30 Hz는 예제와 일치 (Allegro/in-hand와 동일)
- **physics 1/60은 예제 중 제일 성김.** 손끝 0.5m/s면 물리 1 step에 8.3mm 이동 -> 큐브(60mm)를
  8.3mm 파고든 뒤에야 솔버가 알아챔
- **지금 안 올리는 이유**: 팔이 튄 원인이 물리가 아니라 정책 발산이었음이 밝혀졌으므로 근거가 약함.
  실제로 손가락이 큐브에 접촉하기 시작한 뒤 침투 깊이를 재보고 판단할 것.
- 올릴 땐 decimation도 같이 올려야 30 Hz 유지 (예: 1/120 + dec 4)

### 비용 구조 (반드시 기억)
```
sim.dt      -> GPU 비용에 정확히 비례.  1/60 -> 1/120 이면 물리 계산 2배
decimation  -> 물리 비용과 무관.  정책 forward만 늘어남 (MLP라 거의 공짜)
```

### episode_length_s = 8초 판단
참고: Lift-Cube 5초(단, Franka는 테이블 위 큐브 바로 앞에서 시작), ShadowHand/Allegro 10초, in-hand 20초.
우리는 손이 55cm를 내려가야 하므로 더 필요함. 예산: 접근 2~3초 + 정렬 1~2초 + 파지 0.5초 + 들기 1초
= 5~6초, 남는 2~3초가 "들고 버티는 시간". **hold/lift는 절대형이라 그 시간에 비례해 쌓임.**
늘려야 할지는 `Metrics/cube_min/cage_sdf_mean`(에피소드 중 최근접)으로 판단할 것.

---

## 2026-07-13 새 진단 도구

### TensorBoard (모든 run에서 자동)
```
Metrics/cube_max/action_delta       |a_t - a_{t-1}| 최대.  clip 후 상한 2.0 (이전 9.66)
Metrics/cube_max/action_track_err   |관절목표 - 관절실제| 최대 [rad]
```
**`cube_max`를 볼 것** — 튀는 건 순간적 사건이라 평균/최종값엔 안 잡히고 최댓값에만 잡힘.

### play.py
```bash
python scripts/rsl_rl/play.py --task Indy-Wuji-Cube-Grasp --num_envs 32 \
  --checkpoint <path>.pt --print_action
```

### 판정 규칙
```
추종오차 작음(<0.1) + |Δa| 큼(>0.3)  ->  팔이 명령대로 발광. 학습/보상 문제
추종오차 큼(>0.3) 단독                ->  물리가 명령을 이김. dt/decimation 문제
   단, 명령 자체가 도달 불가능하면 추종오차도 커짐 -> |Δa|를 반드시 같이 볼 것
```

---

## 2026-07-13 교훈 (가장 중요)

- **"결과"를 보고 원인을 단정하지 말 것.** 손이 87cm 솟구친 걸 보고 물리 튕김이라 단정했으나
  **정책이 그렇게 명령한 것**이었음. **제어 입력(action)을 안 보고 상태(state)만 봤기 때문.**
- **증거는 이미 화면에 있었음.** `Episode_Reward_Raw/action_rate = 167` (= `|Δa| ≈ 0.68`)이
  TensorBoard에 계속 찍히고 있었음. **정책 발산의 명백한 증거를 보고도 흘려봤음.**
- **`action_rate` reward는 공짜 진단 도구임.** 별도 코드 없이 "정책이 명령을 흔드나"를 알려줌.
- **잘못된 진단 위에 쌓은 수정은 헛수고임.** 약지/새끼 접기, 바닥 페널티, decimation은 전부
  증상을 건드린 것. action space를 고치지 않았으면 무엇도 해결되지 않았을 것.
- **`scale`과 `clip_actions`는 반드시 같이 설계할 것.** scale이 작으면 정책이 |a|를 키워야 하고,
  clip이 없으면 그게 발산으로 이어짐. 하나만 보면 반드시 틀림.

---

## 2026-07-13 action space 수정 후 첫 학습 — 역사상 최초로 hold가 켜짐

### `2026-07-13_17-44-24` (228 iter)
```
                  iter 0      45      91     136     182     228
|Δa| 최대         0.87    1.12    1.08    0.99    0.94    0.90   ← 발산 해결 (이전 9.66)
cage_sdf          0.626   0.171   0.125   0.118   0.101   0.096  ← 손이 드디어 큐브에 도달
cage_inside       0.000   0.0008  0.0066  0.0094  0.0229  0.0599 ← 최초로 0을 벗어남
hold raw          0.000   0.0012  0.0051  0.0167  0.0298  0.0617 ← 최초
opposition       -0.976  -0.731  -0.577  -0.598  -0.527  -0.442  ← 개선 중
mean_reward      -0.122   0.058   0.273   0.432   0.616   0.840
```
**이전 run은 1809 iter 돌고도 cage_sdf 0.31에서 정체. 지금 228 iter만에 0.096.**
**`hold`/`cage_inside`가 이 프로젝트 역사상 처음으로 0이 아님.**

**주의: `decimation` 24 -> 2로 iteration당 경험량이 12배 줄었음. 228 iter = 예전 기준 약 19 iter.**

### 남은 문제 — "감싸기"가 아니라 "엄지 갖다 대기"를 학습 중
사용자 관찰(스크린샷): 손이 큐브 **옆**에 있고 엄지만 큐브를 향함. 나머지 손가락은 반대편으로 말림.
```
thumb_middle_opposition = -0.44   ← 음수 = 엄지와 중지가 큐브의 "같은 쪽"
엄지-큐브표면 = 0.080 m            ← 엄지만 가까움
cage_inside_frac = 0.06            ← 가상점의 6%만 큐브 안
cube_displacement = 0.348          ← 큐브를 35cm 밀어냄. 줄지 않음
```
보상 분해: `reach 47.6%` / `hold 41.6%` / `palm 10.2%` / `action_rate -20.8%` / `hand_floor -16.4%`

**원인**: cage 가상점이 **엄지끝 -> 손가락** 선분 위 `0.25/0.50/0.75` 지점인데, **`0.25` 지점은 엄지 바로 옆**.
엄지가 큐브에 닿으면 그 점들이 **공짜로** 가까워짐. **"나머지 손가락이 큐브 반대편에 있어야 한다"는
보상이 없음.** 그래서 `reach`(47.6%)를 만족시키는 가장 싼 방법이 "엄지만 갖다 대기"가 됨.

### 판단: 더 돌린다 (최소 2000~3000 iter)
지표가 전부 올바른 방향으로 **가속** 중. 228 iter는 매우 초반.

### 유일한 경고 신호
```
cube_displacement   0.25 -> 0.18 -> 0.31 -> 0.31 -> 0.348   ← 안 줄고 있음
```
계속 0.3 이상이면 "밀면서 쫓아가는" 패턴에 갇힌 것.
그때 `opposition`을 reward에 직접 넣을 것 (이미 metric으로 재고 있으므로 3줄).

### 판정 기준
```
thumb_middle_opposition  ->  양수가 되어야 함  (지금 -0.44)
cage_inside_frac         ->  계속 올라야 함     (지금 0.06)
cube_displacement        ->  줄어야 함          (지금 0.35) <- 관건
```

### 부수 확인
- reward는 큐브의 실제 pose를 해석적 SDF로 씀. **인식이 없으므로 손가락을 큐브로 착각할 수 없음.**
- **힘 부족이 아님.** 힘 문제라면 "잡았는데 놓친다"가 나와야 하는데 `cage_inside=0.06`이라 못 잡음.

### `scripts/diag/` 삭제
`play.py --print_action`과 중복이라 제거. action 진단 코드는 `play.py`와 `managers.py` 두 곳뿐.

---

## 2026-07-13 `palm_facing` weight 9 → 60 — 정책이 손바닥 방향을 팔아먹고 있었음

### 사용자 관찰 (스크린샷) + 지표 확증 (`2026-07-13_17-44-24`, 795 iter)
```
                 iter 0     159     318     477     636     795
palm_facing     0.032   0.597   0.538   0.457   0.434   0.408   ← 정점 찍고 역주행!
cage_inside     0.000   0.024   0.137   0.240   0.191   0.203   ← 상승
hold raw        0.000   0.020   0.101   0.125   0.106   0.127   ← 상승
큐브 이동         0.002   0.277   0.415   0.366   0.343   0.452   ← 45cm. 악화
```
**`cage_inside`가 오르는 동안 `palm_facing`이 내려감.** 손을 옆으로 뉘여 엄지-손가락 선분만
큐브에 꽂는 자세가 `hold`를 더 싸게 벌기 때문. 그 과정에서 큐브를 45cm 밀어냄.

### 원인: 절대형 vs 차분형의 구조적 불균형
보상 분해: `hold 48.6%` / `reach 45.0%` / `palm 5.8%` / `hand_floor -16.3%`

**최대 획득량** (Episode_Reward_Raw = 에피소드 합 / 8초):
```
hold  (절대 1.0)  0.50    ← 절대형은 "유지한 시간"만큼 8초 내내 쌓임
reach (차분 25)   0.13
palm  (차분  9)   0.036   ← 차분형은 딱 한 번.  hold의 1/14
```
**weight 9로는 구조적으로 hold를 이길 수 없었음.**

논문도 orientation을 hold의 20배로 둠 (`25·r_hold` vs `500·r_orient`). 우리는 반대로 1/14였음.

### 수정
```python
palm_facing weight 9.0 -> 60.0     # 최대 획득량 0.036 -> 0.24 (hold 0.50의 절반)
```
**farming 위험 없음**: 차분형이라 총액이 `facing(final) - facing(reset)`으로 고정.
weight를 키워도 반복 수확 불가. `hold`(0.50)가 여전히 더 큼.

**절대형 전환은 거부.** 예전 절대형 palm_facing(w=0.5)이 전체 보상의 98%를 먹고
31cm 밖에서 정체시킨 전력이 있음.

**재학습 필요** (`--resume` 불가).

---

## 2026-07-13 `palm_facing`이 파지와 무관한 축(손바닥 법선)을 재고 있었음

### 발단: palm_facing weight 60이 재앙 (`2026-07-13_18-47-02`, 1342 iter)
```
palm_facing  0.032 -> 0.983 (즉시 만점, 유지)
cage_sdf     0.626 -> 0.121 (12cm 정체)
cage_inside  0.000 -> 0.000 (완전 붕괴!  weight 9였을 땐 0.20)
hold raw     0.000 -> 0.000
큐브 이동      0.002 -> 0.003 (큐브를 아예 안 건드림)
```
보상 분해: `palm_facing 70.1%` / `reach 29.9%` / `hold 0.0%`
사용자: "왜 접근을 더 못하지? palm_facing이 막는 건가" -> **정확했음.**

### 1차 진단 (틀렸음): "(1,0,0)이 손바닥 법선이 아니다"
cage 형성 자세들의 큐브 방향이 (0.19, 0.28, 0.94)라 "(1,0,0)은 엉뚱한 축"이라 결론냄.
**-> 사용자 반박: "(0,0,1)이 진짜 손바닥 법선 맞아? 확인해봐"**

### 직접 기하 측정 — **(1,0,0)이 손바닥 법선이 맞았음**
```
손바닥 평면 법선 (손가락 뿌리 5개의 평면) = (+0.965, -0.008, +0.262)  ~= (1,0,0)  O
손가락 오므림 방향                       = (+0.700, -0.482, -0.527)   -> (1,0,0)과 +0.700
```

### 진짜 구조: 손바닥 법선과 파지 개구부는 **서로 다른 축** (65도 어긋남)
손끝의 palm_link 로컬 좌표:
```
              편 상태                      오므린 상태
엄지   (+0.007, +0.123, +0.062)     (+0.059, -0.002, +0.092)
검지   (-0.011, +0.046, +0.195)     (+0.069, +0.013, +0.100)
중지   (-0.020, +0.007, +0.193)     (+0.066, +0.007, +0.099)
```
- 손가락은 **+z로 뻗음**. 오므리면 셋이 **(0.065, 0.006, 0.097) 한 점으로 모임**
- -> 물체가 들어갈 자리는 그 "사이" = 약 (0.25, 0.25, 0.94)
- cage가 형성된 자세들의 큐브 방향 (0.19, 0.28, 0.94)와 **독립적으로 일치**

| | 방향 (palm 로컬) | 의미 |
|---|---|---|
| 손바닥 법선 | (0.97, 0.00, 0.26) | 손바닥이 향하는 쪽, 손가락이 굽는 쪽 |
| **파지 개구부** | **(0.19, 0.28, 0.94)** | **물체가 엄지-손가락 사이로 들어가는 공간** |

**파지의 조건은 "손바닥이 향하는가"가 아니라 "물체가 손가락 사이에 들어갈 수 있는가"임.**

### 검증 (무작위 자세 102,400개)
```
facing 정의                     hold 상관계수   face>0.7&hold>0.1   face>0.9시 hold최대
(1,0,0) 손바닥 법선                  +0.003            0개              0.0000
(0.19,0.28,0.94) 고정 개구부         +0.105           48개              0.6381
실시간 palm->손끝3개 중심              +0.103           48개              0.6381
실시간 palm->파지중심                 +0.104           48개              0.6381
```

### 사용자 질문: "법선을 고정해도 되나? 매번 바뀔 텐데"
1. **팔이 움직여도 안 바뀜** — palm_link 로컬 벡터이고 `quat_apply`로 매 step 월드 변환. 문제없음.
2. **손가락 자세에 따라선 바뀔 수 있으나, 실측 결과 고정과 실시간이 사실상 동일**
   (상관 +0.105 vs +0.103, 양립 48개 동일). **고정 벡터로 충분함.**

### 수정
```python
palm_normal_b:      (1,0,0) -> (0.19, 0.28, 0.94)   # reward + metric 둘 다. 정규화도 추가
palm_facing weight:  60.0   -> 20.0                 # 최대 0.081 < reach 0.13
```

### **`palm_facing`은 약한 신호임 (상관계수 +0.105)**
"손이 큐브 쪽을 향한다"가 "파지 성공"을 거의 예측하지 못함.
- **weight를 올려서 뭔가 되기를 기대하면 안 됨.** 60으로 올렸다가 학습을 죽인 게 증거
- **진짜 일은 `hold`가 함.** palm_facing은 초반 shaping 보조 역할

### weight 규칙
```
palm 최대 획득량 < reach 최대 획득량   <- 넘기면 standoff
weight 60 -> palm 0.24 > reach 0.13  -> standoff (실측)
weight 20 -> palm 0.081 < reach 0.13 -> 안전
```

### 교훈
- **상관관계가 있다고 그게 정의는 아님.** "hold와 상관있는 축"을 찾아놓고 "그게 손바닥 법선"이라
  단정했으나, 손바닥 법선은 따로 있었고 그 축은 "파지 개구부"였음.
  **사용자가 "진짜 법선 맞아?"라고 물어서 잡힘.**
- **기하는 기하로 측정할 것.** 손바닥 법선은 손가락 뿌리 평면의 법선으로, 파지 개구부는 손끝의
  편/오므린 상태 중간으로 — 둘 다 5분이면 직접 잴 수 있었음.
- **"reward가 목표와 상관있는가"를 먼저 잴 것.** palm_facing의 hold 상관이 +0.105로 약하다는 걸
  알았다면 weight를 만지작거리며 시간을 쓰지 않았을 것.

---

## 2026-07-13 접근 순서 강제 — `reach`를 `facing`으로 게이팅 (B안)

### 사용자 관찰
> "손바닥이 하늘을 보고 시작하는데 바로 큐브로 손이 붙고, 손을 돌리려니 큐브에 막혀서 못 돌림"
> "접근 속도가 너무 빨라서 손 돌릴 시간도 없이 접근해버려서 그런 거 같음"

### 왜 서두르나 — 보상이 서두르라고 시킴
```
reach  차분형 -> 속도와 무관하게 총액 동일. 서둘러도 손해 없음
hold   절대형 -> "잡고 있던 시간"에 비례해 쌓임 -> 일찍 도착할수록 이득
```
**속도 페널티로는 못 고침** (틀린 방향으로 "천천히" 도착할 뿐). **속도는 증상, 원인은 순서.**

### 수정 (rewards.py, ObjectCageProgressReward)
```python
reward = clamp(progress / distance_max, -1, 1)
if palm_cfg is not None:
    gate = palm_facing_object(env, palm_cfg, object_cfg, palm_normal_b)
    reward = reward * (gate_floor + (1 - gate_floor) * gate)   # gate_floor=0.0 -> 완전 차단
```
**"잘못된 순서"를 무가치하게 만듦.** 방향 틀린 채 달려들면 reach = 0.

smoke test(15 iter): `palm_facing 0.001 -> 0.323`, `cage_sdf 0.584 -> 0.393`. 의도대로 작동.

---

## 2026-07-13 논문 원문 확인 — 논문은 게이팅을 안 함

### 순서 강제 = 가중치 스케일링
> "we scale the rewards according to their position in the sequence: **r_T >> r_orient >> r_hold >>
> r_reach**. This reduces the probability that the policy gets **stuck in the local minima, created by
> accumulating rewards for actions that are easier to achieve** compared to the following more complex
> sub-tasks."

실제값: `r_reach x1, r_hold x25, r_orient x500, r_T x5000` (단계마다 약 20배)

### 논문은 목표 파지 자세를 앎 — 본질적 차이
```
r_grasp = r_hp + r_hr + lambda*r_hj   <- 전부 "주어진 목표 파지 g"를 향한 차분 보상
```
`r_hr`이 목표 회전으로 돌리기를 **직접** 보상 -> **게이팅 불필요.**
**우리는 목표 파지가 없음. 게이팅이 r_hr의 대용품.**

### 커리큘럼: close-start에서 r_man을 통째로 끔
물체를 **손 5cm 앞**에, 팔은 **manipulability 높은 중립 자세**로. `r_man` 비활성. 성공률 50%까지.

### 가상점: 논문은 엄지-중지만 6점 (사용자 추측 정확)
> "if an object is contained between the middle finger and the thumb, it is also contained between
> the index and ring fingers, **as defined by the hand topology**."

**단 엄지의 50% 지분은 선분을 줄여도 안 바뀜** (엄지는 모든 선분의 시작점).
논문 6점: 엄지 3.0 : 나머지 3.0.  우리 12점: 엄지 6.0 : 나머지 6.0. **동일.**

---

## 2026-07-13 최종 가중치 (논문 방향으로 한 단계)

```
항목                형태     weight   최대 획득량
palm_facing         차분      8.0       0.033      <- 20 -> 8
finger_cage_reach   차분     10.0       0.050      <- 25 -> 10.  x facing 게이트
finger_cage_hold    절대      1.0       0.500
cube_lift           절대      3.0       0.750
arm_manipulability  페널티    1.0
hand_floor          페널티    0.5
action_rate                 -0.005

순서: palm(0.033) < reach(0.050) < hold(0.500) < lift(0.750)
hold / reach = 10.0     (이전 4.0,  논문 25)
```

**논문의 25를 바로 못 쓰는 이유:** 논문은 (a) `r_hp`라는 별도 접근 신호가 있고 (b) close-start 커리큘럼이
있음. 우리는 둘 다 없어서 reach를 죽이면 **접근 신호 소멸 + hold는 가까워야 켜짐 = 닭-달걀.**

**규칙: `palm 최대 < reach 최대`.** 넘기면 standoff (weight 60에서 palm 0.24 > reach 0.13 ->
cage_inside 0.20 -> 0.000 붕괴 실측).

### 다음: 젓가락은 논문대로
젓가락은 목표 파지 자세가 정의되므로 `r_hp`/`r_hr`/`r_hj`/`r_T`를 전부 쓸 수 있음.
그러면 **게이팅 없이 논문의 25/500/5000 스케일이 그대로 성립함.**

---

## 2026-07-13 게이팅 버그 — 차분형의 telescoping을 깨뜨림 (왕복 farming)

### 증상 (`2026-07-13_21-22-25`, 394 iter)
```
iter                0      66     132     199     265     331     394
palm_facing      0.002   0.425   0.819   0.735   0.570   0.454   0.420   ← 정점 후 역주행
palm_facing MAX  0.005   0.515   0.988   0.994   0.990   0.993   0.996   ← 순간적으론 만점
cage_sdf         0.626   0.470   0.129   0.194   0.287   0.364   0.413   ← 바닥 후 역주행
reach raw        0.000   0.001   0.004   0.013   0.028   0.036   0.040   ← 계속 오름
mean_reward     -0.062  -0.374   0.020   0.769   1.915   2.491   2.864   ← 계속 오름
```
**보상은 오르는데 지표는 나빠짐 = 허점을 찾았다는 신호.**

### 원인: 게이트를 음수(후퇴)에도 곱했음
```python
reward = reward * gate      # gate in [0,1].  음수에도 곱하면...
```
```
겨누고 다가감    +0.5 x 1.0 = +0.5
안 겨누고 물러섬 -0.5 x 0.0 =  0     ← 페널티 소멸!
```
**"방향을 버리면 후퇴가 공짜"** -> `겨눔 -> 접근 -> 방향 버림 -> 공짜 후퇴` 왕복으로 reach 무한 수확.
**차분형의 telescoping(총합 = sdf(reset) - sdf(final))이 깨진 것.**

### 수정 (rewards.py, 한 줄)
```python
reward = torch.where(reward > 0.0, reward * gate, reward)
```
- 다가갈 때: 방향이 맞아야 보상 (순서 강제 유지)
- 물러설 때: 방향과 무관하게 페널티 전액 (왕복 farming 차단)
```
왕복 1회:  이전 +0.50 + 0.00 = +0.50   |   수정 +0.50 - 0.50 = 0.00
```

### 교훈
- **차분형에 곱셈 게이트를 걸 땐 반드시 부호를 나눌 것.** 음수에 곱하면 "페널티를 게이트로 지우는"
  편법이 생기고 telescoping이 깨짐.
- **"보상은 오르는데 지표는 나빠진다"가 보이면 무조건 허점이 있음.** 항상 이 신호를 볼 것.
- **`cube_max`와 `cube_final`의 괴리도 신호임** (palm MAX 0.996 vs 최종 0.420 = 잠깐 만족시키고 버림).

---

## 2026-07-13 신규 task `Indy-Wuji-Chopsticks-Grasp` (직육면체 = 젓가락 프록시)

```
tasks/manipulation/functional_grasp/     <- 논문 방식(목표 파지 + orient + 커리큘럼) 폴더
├── chopsticks_grasp_env_cfg.py          # ChopsticksGraspSceneCfg / ChopsticksGraspEnvCfg
└── indy_wuji/
    ├── __init__.py                      # Indy-Wuji-Chopsticks-Grasp / -close-start
    ├── env_cfg.py
    └── learning/rsl_rl_cfg.py           # experiment_name = "indy_wuji_chopsticks_grasp"
```

### 물체
```python
size = (0.03, 0.03, 0.16)        # 3cm x 3cm x 16cm 막대
mass = 0.30 kg
pos  = (0.62, -0.18, 0.015)      # 3cm 면으로 누움
rot  = (0.70711, 0, 0.70711, 0)  # 긴 축을 월드 x로 눕힘
```
**긴 축이 있어서 `r_hr`/`r_orient`가 해석적으로 정의됨** (정육면체는 대칭이라 불가능).
**누워 있으면 바로 못 잡음 -> 먼저 세워야 함 = pre-grasp manipulation.**

### 현재: reward는 아직 CubeGraspRewardsCfg 공유 (half_extent만 override)

### 다음 작업 순서
```
1. ChopsticksGraspRewardsCfg 분리
2. 목표 파지 g = (hp, hr, hj) 정의    <- 나머지가 전부 여기 의존
3. r_grasp = r_hp + r_hr + lambda*r_hj    4. r_orient    5. r_T
6. 가중치 논문 그대로 (1/25/500/5000)      7. 게이팅 제거 (r_hr이 대신함)
8. 커리큘럼 close-start (물체를 손 5cm 앞 + manip 높은 중립 자세 + r_man 끔)
```

### 사고 기록
`functional_grasp`의 `rsl_rl_cfg.py`가 복사본이라 `experiment_name`이 `indy_wuji_cube_grasp`
그대로였음 -> chopsticks smoke test가 **cube 학습 로그에 섞임.**
**복사로 task를 만들 땐 `experiment_name`을 반드시 먼저 바꿀 것.**

---

## 2026-07-14 close-start 높이/받침면 수정

- close-start 구조를 다시 확인함.
- 기존 close-start은 큐브 `x/y`만 손 파지 중심 근처였고 `z`는 바닥 `0.03m`였음.
- 손 최저/파지 중심 높이는 약 `0.53m`라 큐브 높이와 맞지 않았음.
- `BASE_Z=0.50` 받침면을 추가함.
- `{ENV_REGEX_NS}/Support` kinematic cuboid를 scene에 추가함.
- cube 중심 높이는 `0.53m`임.
- close-start cube 위치는 `(0.704, -0.279, 0.530)`임.
- hard cube 위치는 `(0.850, 0.000, 0.530)`임.
- `cube_lift`는 받침면 `surface_z=BASE_Z` 기준으로 계산함.
- `cube_clearance` metric도 같은 기준으로 보정함.
- `hand_floor` penalty도 같은 기준으로 보정함.
- model shape는 변하지 않음. action `18`, observation `57` 유지함.
- `python -m py_compile` 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.

## 2026-07-14 close-start 파지 간극/방향 검증 정정

- `palm_facing=0`과 `joint5` action 반경 축소는 초기 손-큐브 방향 정렬이 검증돼야만 성립함.
- `/tmp/cube_grasp_probe.py`로 close-start reset 상태를 수치 확인함.
- 기존 close-start cube `(0.704, -0.279, 0.430)`는 cage 중심에서 y로 약 `9cm` 벗어났음.
- 기존 위치는 zero action 30 step만 해도 cube가 `(0.758, -0.220, 0.430)`으로 밀리고 `palm_facing`이 `0.838 -> 0.260`으로 깨짐.
- close-start cube를 `(0.692, -0.369, 0.430)`으로 수정함.
- 수정 후 reset `palm_facing=0.987`, zero action 30 step 뒤 `0.997`임.
- 수정 후 `cage_center_to_cube` xy 오차는 reset 약 `0.5mm`, zero action 30 step 뒤 약 `3mm`임.
- 따라서 **현재 close-start 배치에서만** `finger_cage_reach=0`, `palm_facing=0`, `joint5` scale 축소가 타당함.
- cube 배치나 arm reset pose를 바꾸면 이 probe를 다시 해야 함.
- `python -m py_compile` 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.

## 2026-07-14 close-start action 출력 보강

- action이 잘못 나오는지 직접 확인하기 위해 `scripts/rsl_rl/play.py`의 `--print_action` 출력을 보강함.
- 기존 출력은 전체 action 요약과 `actions[0,5]` 하나만 보여서 원인 분리에 부족했음.
- 새 출력은 `raw policy action`, `clip 후 applied action`, `target joint position`, `actual joint position`, `target-actual error`, `joint velocity`를 joint 이름별로 출력함.
- `clip%`도 출력해 policy가 `clip_actions=1.0` 밖으로 자주 나가는지 확인 가능함.
- 판정 기준은 다음과 같음.
- `err`가 작고 action/Delta action이 크면 policy가 그렇게 시킨 것임.
- `err`가 크면 물리/PD/접촉/decimation이 목표를 못 따라가는 것임.
- `CLI.md`에 close-start action 확인 명령을 추가함.
- `python -m py_compile nrmk_isaaclab_wuji/scripts/rsl_rl/play.py` 통과함.

## 2026-07-14 Cube Grasp 단일 task 정리

- `close-start`/`Hard`를 나눠 쓰지 않기로 함.
- `Indy-Wuji-Cube-Grasp` 본체가 검증된 close-start nominal grasp 배치를 사용하도록 정리함.
- 별도 호환 alias는 남기지 않음. task id는 `Indy-Wuji-Cube-Grasp` 하나임.
- 새 학습/play/smoke test 명령은 전부 `Indy-Wuji-Cube-Grasp`를 사용함.
- 현재 cube 위치는 `(0.692, -0.369, 0.430)`임.
- cube reset randomization은 꺼둠. `x/y/z = 0`.
- 현재 reward override는 `finger_cage_reach=0`, `palm_facing=0`, `finger_cage_hold=1`, `cube_lift=10`, `arm_manipulability=1`, `hand_floor=0.5`, `action_rate=-0.0003`임.
- `CLI.md`, `AGENTS.md`, `nrmk_isaaclab_wuji/agent.md`를 단일 task 기준으로 정리함.
- 같은 experiment 안에 과거 smoke/hard/easy run이 섞여 있으므로 최신 run 자동 선택은 위험함.
- `python -m py_compile` 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- smoke test에서 action `18`, observation `57`, reward 7개를 확인함.
- reward table은 `finger_cage_reach=0`, `finger_cage_hold=1`, `cube_lift=10`, `palm_facing=0`, `arm_manipulability=1`, `hand_floor=0.5`, `action_rate=-0.0003`임.
- `/tmp/cube_grasp_probe.py --task Indy-Wuji-Cube-Grasp --headless --num-envs 1` 통과함.
- probe reset 값은 cube `(0.692, -0.369, 0.430)`, cage xy error 약 `0.5mm`, `palm_facing=0.986780`, `cage_hold=0.210871`임.
- zero action 30 step 뒤 cage xy error 약 `3mm`, `palm_facing=0.996594`임.

---

## 2026-07-14 cube grasp 포기 — 근본 원인 두 가지

### ① action space가 논문과 근본적으로 다름  ★ 이게 오늘 사고의 대부분
논문 원문 (Section III-B):
> "An action represents a **relative displacement** in 3D hand position, hand rotation, and hand
>  joint positions. **The arm joint targets are calculated via Inverse Kinematics (IK).**"
> "6 DoF UR5e + 11 DoF Schunk SIH hand. The joints are **coupled, leaving five controllable DoF**.
>  An action is an **11-element vector**: 손위치 변위 3 + 손회전 변위 3(Euler) + 손가락 5."
> "v_hp_max = 1 m/s, dt = 0.0333 s"   -> step당 최대 3.3cm

| | 논문 | 우리 |
|---|---|---|
| action | **상대 변위** (손위치3 + 회전3 + 손가락5 = 11D) | **절대 관절각** (팔6 + 손가락12 = 18D) |
| 팔 제어 | **IK** | 관절 직접 |
| 크기 제한 | `v_max x dt = 3.3cm/step` **내재적** | `scale` x `clip_actions` **수동** |
| 제어 주파수 | 30 Hz | 30 Hz (오늘 맞춤) |

**이 차이가 낳은 결과:**
- **scale/clip_actions 발산**: 논문 action은 "손을 3cm 옮겨라"라 물리적으로 유계. 우리는
  "관절을 X rad로"라서 scale이 작으면 정책이 |a|를 키우고, 상한이 없으면 발산 (|Δa| 9.66).
- **arm_manipulability 페널티가 불필요**: IK가 팔을 풀어주므로 특이점에 빠질 수 없음.
  그 페널티를 만든 이유 자체가 "관절을 직접 명령하기 때문"임.
- **r_hp/r_hr이 action space와 일치**: 정책이 손 pose를 명령하고 보상도 손 pose에 걸림.
  우리는 관절을 명령하면서 손 방향을 보상하려다 palm_facing 축을 추측하는 삽질을 함.
- **손가락 5 DOF vs 12 DOF**: 탐색 공간이 훨씬 작음.

IsaacLab에 `DifferentialInverseKinematicsActionCfg` 있음. Lift-Cube에 `ik_abs/ik_rel_env_cfg.py` 예제.

### ② 정육면체는 "목표 파지"를 정의할 수 없음
논문은 `g = (hp, hr, hj)`를 알고 있고 `r_hr`이 "이 방향으로 돌려라"를 직접 보상함.
**대칭인 정육면체는 목표 회전을 정의할 수 없어서 그걸 못 씀.**
-> 대리 지표(palm_facing 축, 게이팅, closability)를 발명했고 전부 뚫림.

---

## 2026-07-14 오늘의 실패 (반복 방지)

### 실패 1: clip_actions 없음 + scale 과소 -> 정책 발산  ★사수님이 잡음
`|a|` 평균 1.5(범위 [-1,1]), `|Δa|` 최대 9.66 -> 관절 목표가 한 step에 110° 점프
-> 팔이 velocity_limit로 왕복, 큐브를 67cm 날림.
**나는 "손이 87cm 솟구쳤다"는 결과만 보고 "물리 튕김"이라 단정.**
사수님이 "action을 찍어봐라"고 해서 잡힘. `Episode_Reward_Raw/action_rate = 167`이 이미
TensorBoard에 찍혀 있었는데 흘려봤음.

### 실패 2: palm_facing 축을 세 번 틀림
`(1,0,0)`이 hold와 무상관(+0.003)이라 "틀린 축"이라 결론 -> `(0.19,0.28,0.94)`로 바꿈 ->
**사용자가 "진짜 손바닥 법선 맞아?"라고 물어서** 직접 기하 측정 -> **`(1,0,0)`이 맞았음.**
더 파보니 **`hold` 자체가 87% 퇴화**였음 (오므리면 오히려 물체에서 멀어지는 자세).
**썩은 지표와의 상관으로 축을 골라서 퇴화를 최적화하는 축을 고른 것.**

### 실패 3: 게이팅 -> telescoping 파괴 (왕복 farming)
음수(후퇴)에도 게이트를 곱해서 "방향 버리면 후퇴가 공짜"가 됨.
-> `겨눔 -> 접근 -> 방향 버림 -> 공짜 후퇴` 왕복으로 reach 무한 수확.
**보상은 오르는데 지표는 역주행** (mean_reward 2.86인데 cage_sdf 0.129 -> 0.413).
수정: `torch.where(reward > 0, reward * gate, reward)`.

### 실패 4: "리셋 자세가 이미 물체를 잡고 있어야 한다"고 착각  ★사용자가 잡음
논문 close-start: "The arm is set to a **neutral configuration**." **파지는 정책이 학습함.**
나는 테이블 넣고 손목 뒤집고 CEM으로 파지 자세를 찾으려 했음. 전부 헛수고.

---

## 2026-07-14 관통하는 패턴 (진짜 교훈)

**매번 "대리 지표"를 만들고 검증하지 않고 결정했음:**
```
palm_facing 축   <- hold와의 상관으로 고름.  hold가 87% 퇴화라 무의미
게이팅           <- 발명.  telescoping을 깨뜨림
closability      <- 발명.  안 씀
"파지 중심"       <- 손끝 3개의 중심.  엄지가 마주보지 않으면 무의미
손목 자세         <- "파지중심이 palm 아래" + manip.  둘 다 대리 지표
```

**속일 수 없는 신호는 하나뿐: `cube_clearance` (물체가 실제로 떴는가).**
`hold`, `cage_inside`, `opposition`, `palm_facing` 전부 정책에게 뚫렸음.
**그리고 이 프로젝트에서 `cube_lift`는 한 번도 0을 벗어난 적이 없음.**

### 규칙
1. **새 보상/지표를 만들면 최종 목표와의 상관계수부터 재라.** 5분이면 됨.
2. **"그 상태가 존재하는가"를 샘플링으로 먼저 확인하라.**
   (`hold>0.1 AND palm_facing>0.7`은 61,440개 중 **0개**. 존재하지 않는 걸 학습시키고 있었음)
3. **"보상은 오르는데 지표는 나빠진다"가 보이면 무조건 허점이 있다.**
4. **상태(state)가 아니라 제어 입력(action)을 봐라.**
5. **정책이 지표를 거스르면, 정책이 아니라 지표를 의심하라.**

---

## 2026-07-14 코드 되돌리기

### 남긴 것
```
clip_actions = 1.0,  scale = 1.0        rsl_rl_cfg.py:21, indy_wuji/env_cfg.py:76
sim.dt = 1/60,  decimation = 2 (30 Hz)  cube_grasp_env_cfg.py:145-146
play.py --print_action                   진단 도구
사용자/codex 작업 전부                    support 받침면, surface_z, joint5 scale 0.05(진동 해결)
```

### 뺀 것
```
finger_closability                       함수 삭제
reach의 facing 게이팅                     palm_cfg / gate_floor 파라미터 포함
palm_normal_b (0.19,0.28,0.94) -> (1,0,0)   rewards.py + managers.py
가중치 reach 10 / palm 8 / action_rate -0.005 -> 0.3 / 1.0 / -0.0003
별도 curriculum/hard alias
테이블 (넣었다 뺌)
```
**주의: reach/palm_facing weight는 dt=0.4(decimation 24) 시절 값임. 30Hz에선 차분형 에피소드 합이
12배 작아짐. 지금 main task는 둘 다 0으로 꺼둬서 영향 없지만, 켤 때 반드시 재조정할 것.**

---

## 2026-07-14 다음: functional_grasp

**골격 있음:** `functional_grasp/{chopsticks_grasp_env_cfg.py, cfg_skeleton.py, mdp/{target_grasp,rewards}.py}`

**반드시 바꿀 것:**
```
1. action space를 논문처럼: IK(팔) + 관절(손가락), 상대 변위
   -> IsaacLab DifferentialInverseKinematicsActionCfg
   -> scale/clip 참사, arm_manipulability 페널티가 전부 불필요해짐
2. 목표 파지 g = (hp, hr, hj)  -> r_hr이 방향을 직접 보상 -> palm_facing/게이팅 불필요
3. 손가락 DOF 축소 검토 (논문 5, 우리 12)
4. 가중치는 논문 그대로 (r_reach 1 / r_hold 25 / r_orient 500 / r_T 5000)
5. 커리큘럼 (close-start에서 r_man 끔, 물체를 손 5cm 앞, 팔은 manip 높은 중립 자세)
```

## 2026-07-14 close-start alias 제거 최종 정리

- 예전 curriculum alias와 관련 class/register를 제거함.
- cube grasp 실행 이름은 `Indy-Wuji-Cube-Grasp` 하나만 남김.
- `CLOSE_START_CUBE`는 가까운 nominal grasp 배치 상수로 유지함.
- 과거 hard 배치 상수는 `LEGACY_HARD_CUBE`로 이름만 남겨 공통 cfg 기본값과 구분함.
- RSL-RL `run_name`은 `cube_grasp_close_start`로 정리함.
- probe helper는 `/tmp/cube_grasp_probe.py`로 이름을 정리함.
- functional_grasp/chopsticks skeleton의 예전 alias class/register도 제거함.
- repo 전체에서 예전 alias/문자열을 검색해 남은 직접 참조가 없음을 확인함.
- `python -m py_compile` 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- `/tmp/cube_grasp_probe.py --task Indy-Wuji-Cube-Grasp --headless --num-envs 1` probe 통과함.

## 2026-07-14 hand-only cube grasp 재정리

- cube grasp를 다시 "Wuji hand가 물체를 감쌀 수 있는지"부터 확인하는 hand-only 진단 모드로 바꿈.
- policy action은 `finger[1-3]_joint[1-4]` 12축만 사용함.
- arm 6축은 `FixedJointPositionAction` 0D term으로 default joint target을 매 step 유지함.
- `CubeGraspActionsCfg`에 `arm_hold_action` 슬롯을 추가함.
- 이 수정 전에는 arm action을 빼면서 arm joint target도 사라져 zero action 30 step만에 arm이 무너지고 cube가 날아갔음.
- smoke test에서 Action Manager `arm_action=12`, `arm_hold_action=0`, total action shape `12` 확인함.
- policy observation shape는 `42`임.
- `cube_to_goal` observation은 제거함.
- active reward는 `finger_cage_hold=1`, `hand_floor=0.5`, `action_rate=-0.0003` 중심임.
- `finger_cage_reach`, `palm_facing`, `cube_lift`, `arm_manipulability` weight는 현재 `0`임.
- RSL-RL run name은 `cube_grasp_hand_only`임.
- `python -m py_compile` 통과함.
- `Indy-Wuji-Cube-Grasp --headless --num_envs 1 --max_iterations 1` smoke test 통과함.
- probe에서 zero action 30 step 후 arm collapse/cube ejection이 사라짐.
- close action `1.0` 60 step 후 `cage_hold=0.427465`, `cage_inside_frac=0.666667`로 증가함.
- `--num_envs 128 --max_iterations 20` 짧은 학습 통과함.
- 짧은 학습에서 `finger_cage_hold`는 초반 `0.003`에서 `0.13~0.18`까지 증가함.
- 같은 학습에서 `hand_floor` penalty가 커지고 mean reward는 음수로 유지됨.
- 다음 판단은 GUI에서 손이 받침면을 긁는지 확인한 뒤 `hand_floor`/초기 높이/손가락 action scale을 조정하는 것임.

## 2026-07-14 grasp+lift active 재정리

- `Indy-Wuji-Cube-Grasp`를 다시 arm 6축 + thumb/index/middle 12축, 총 18D action으로 돌림.
- observation은 joint position 18D, palm 기준 cube 3D, five-fingertip 기준 cube 15D, previous action 18D로 총 54D임.
- `cube_to_goal` observation은 현재 grasp+lift baseline에서 제거함.
- active reward는 `finger_cage_reach=3`, `finger_cage_hold=5`, `cube_lift=50`, `cube_support=2`, `palm_facing=0`, `arm_manipulability=0`, `hand_floor=0.2`, `action_rate=-0.0003`임.
- `cube_support`를 추가해 cube 최하 모서리가 받침면 아래로 내려가면 벌점을 주게 함.
- `--num_envs 128 --max_iterations 20` 짧은 학습에서는 hold는 켜졌지만 lift는 거의 0이고, cube를 받침 아래로 누르는 실패 모드가 보임.
- `/tmp/cube_lift_probe.py` scripted feasibility probe를 실행함.
- 기본 설정에서 close action은 `cage_hold`를 약 `0.40`까지 올리지만 `cube_clearance`는 0 근처이고 `cube_lift_reward_raw=0`임.
- `joint0~joint5` 단일축 ±1 lift 후보를 모두 넣어도 양의 cube clearance가 나오지 않음.
- 강한 손/가벼운 큐브 probe(`finger_effort=3`, `stiffness=40`, `cube_mass=0.03`, `friction=2`)에서도 lift는 0임.
- 결론: 긴 RL 전에 실제로 cube를 들어올리는 scripted arm/hand sequence 또는 초기 자세를 먼저 찾아야 함.

## 2026-07-14 contact/lift 확인 스크립트 추가

- `scripts/debug/check_cube_contact_lift.py` 추가함.
- policy 없이 `reset -> zero settle -> finger close -> optional arm lift`를 실행하게 함.
- thumb/index/middle/palm 링크별 cube contact force를 출력하게 함.
- cube clearance와 `GOOD_CONTACT thumb+middle`, `LIFT_SUCCESS` 판정을 출력하게 함.
- cube 자체에는 contact reporter가 없어 `Cube -> Support` contact sensor는 제거함.
- 손 링크 기준 cube contact sensor만 사용함.
- 짧은 headless 검증 통과함.
- close-only 확인 결과 `after_close`에서 `index_mid=5.5961N`, `thumb_tip=0.0593N`, middle contact 없음.
- `after_lift`에서 `thumb_tip=0.4257N`, `middle_mid=0.7081N`로 `GOOD_CONTACT=True`가 늦게 켜짐.
- 같은 run에서 `max_clearance=0.0003m`라 lift는 실패함.
- 결론: contact 자체는 생기지만 현재 scripted close-only는 cube를 들어 올리지 못함.
- 긴 학습 전에 `--sweep-lift` 또는 cube pose sweep으로 `GOOD_CONTACT=True`와 `max_clearance>0.005m`가 나오는 조건을 먼저 찾아야 함.

## 2026-07-14 contact/lift probe finger sweep 추가

- `scripts/debug/check_cube_contact_lift.py`에 `--finger-action` 옵션을 추가함.
- `--finger-action`은 thumb/index/middle 3개 그룹 값 또는 finger action 12개 값을 직접 받음.
- `--sweep-fingers` 옵션을 추가함.
- `--sweep-fingers`는 cube를 고정하고 thumb/index/middle close 값 조합을 훑음.
- `--contact-mode` 옵션을 추가함.
- contact 판정은 `thumb_middle`, `thumb_index`, `thumb_any`, `tripod` 중 선택 가능함.
- 짧은 검증 명령 `--finger-action 1 0 1 --contact-mode thumb_middle` 실행 통과함.
- 해당 짧은 검증에서는 thumb과 middle이 같은 시점에 안정적으로 물지 못해 `lift_success=False`임.
- 다음 확인은 cube를 움직이기보다 `--sweep-fingers`로 손가락 joint 명령 조합부터 찾는 것이 우선임.

## 2026-07-14 (오후) — 파지 능력 검증 완료: 손은 0.30 kg 큐브를 들 수 있다

### 결론
- **오므림 40~50%에서 4/4 성공.** 큐브를 문 채 팔로 28cm 들어올림 (62→91cm), palm 거리 9cm 유지
- ※ 정정 (같은 날 저녁, GUI 확인): 이 4/4는 **새끼/손바닥 받침 hold**였음. 엄지·중지는 미접촉.
  "힘은 충분"의 근거로만 유효하고, 엄지+중지 집게(pinch)는 미검증
- 0.32~0.44 대부분 3/4, 0.60은 1/4 (너무 조이면 수박씨처럼 짜냄)
- **cube_lift가 계속 0이었던 건 하드웨어(effort_limit 0.6)가 아니라 정책/보상 문제로 확정**
- 성공 모드: 손바닥 위(palm-up)에서 손가락이 큐브를 물어 고정. 학습 리셋 자세와 같은 방향

### 과정에서 잡은 측정 버그 3개 (모두 "손이 못 잡는다"는 가짜 판정을 만들었음)
1. **매 step 순간이동으로 큐브 고정** → 관통 누적 → PhysX가 손가락을 관절 한계 밖으로 폭발
   (오차 1.5 rad = 기하적으로 불가능한 값이 단서였음)
   → 고침: 중력 차단 + 속도만 매 step 0. 위치는 물리에 맡김
2. **tip_link 원점을 패드로 착각** → 원점은 마지막 관절 위치라 2~3cm 손바닥 쪽
   → 배치가 접힌 약지/새끼 위에 얹힘 (GUI 스크린샷으로 확인. 엄지는 닿지도 않았음)
3. **joint1 감소가 '위'인 줄 알았으나 실측 '아래'** → 물고 있는데 실패 판정
   → 판정 기준을 "놓은 뒤 큐브 상승량 > 5cm"로 교체

### 손 기하 실측 (hand_geometry.py)
- 엄지-중지 간격: 편 17.6cm → 30%에서 5.8cm(=큐브) → 40%에서 최소 2.8cm → 70%에서 6.3cm로 재벌어짐
- **물리는 창은 30~50% 부근 하나뿐. 그 이상 조이면 손끝이 서로를 지나침**
- 검지-중지는 항상 1.5~4.2cm → 사실상 한 손가락 (엄지 vs 검지+중지의 2점 집게)
- 스크린샷의 "손가락 겹침"의 정체: 오므림량이 물리는 창을 지나친 것

### 도구
- `scripts/debug/hand_geometry.py` — 물리 없이 FK로 손끝 간격 vs 오므림 곡선
- `scripts/debug/grip_capacity.py` — 쥐여주기→놓기→들기. `--gui`로 눈 확인 (기본 0.40)

## 2026-07-14 (저녁) — 시작 자세 변경 + 편 손가락 + 재학습 준비

### 시작 자세/자산 변경 (재학습용, 커밋 분리 1)
- 손목: joint3/4 = -1.61/-1.62 (40만 샘플 탐색 최상위. 파지중심이 손바닥 5.7cm 아래, manip 0.0645→0.0803)
- 전 손가락 폄 (finger[1-5]=0.0). 접어두기(1.2) 폐기 — 접힌 손가락끼리 자가 충돌로 씨름
  (유지 오차 1.27rad, 토크 포화 82%)하다 튕기며 파지 지점을 쓸었음. 사용자 결정: "펴는 게 맞고 그래야 안 쳐"
- 손가락 액추에이터를 제조사 값으로: kp 2/2/1/1, kd 0.05, friction 0.01 (right.xml 기준).
  effort 0.6 유지 (URDF 스펙 0.2~1.0의 중간값; 6.0 안(사용자)은 스펙 6~30배라 폐기)
- 목표 버퍼 버그 수정: 리셋은 관절 '상태'만 복원, 위치 '목표'는 0으로 남음 → 액션 밖 관절
  (finger4-5)이 매 에피소드 목표 0으로 저절로 이동. `hold_joints_at_default` 리셋 이벤트 추가

### 집게(pinch) 검증 — 미해결
- grip_capacity.py에 "무엇으로 잡았는지" 판정 추가 (엄지·중지 각각 < 5.5cm여야 pinch)
- 새 자세 + 편 손가락 + out_cm 2.5: **0/32 held, 0/32 pinch.** 오므리는 동안 큐브가 평균 24.4cm
  밀려나 낙하. palm-down 집게는 아직 미입증 — 학습이 스스로 찾는지 관찰이 다음 판단 재료
- 교훈: held만 보면 속음. 접촉 주체(어느 손가락) 판정을 반드시 포함할 것

### 재학습 실험 (사용자 변경, 진행 중)
- 큐브 질량 0.30→0.10 kg, episode 8→6 s
- 가중치: reach 10→8, hold 1→25, lift 3→50, palm_facing 8→4, hand_floor 0.5→1.0
- 관찰: 조금 들려고는 하나 hold 자세가 불완전해 lift가 어려움 → 가상점 개선 논의로 이어짐

## 2026-07-14 (밤) — cage 가상점을 손끝 쪽으로 (커밋 분리 2)

### 문제: "느슨한 hold"로도 고점 → lift gate가 헛것에 열림
- 가상점이 선분 내부 등분 [0.25, 0.5, 0.75]라, 엄지-중지 간격 10cm의 헐렁한 새장에서도
  중앙점이 큐브 깊숙이 박혀 포화 → 손끝이 표면에서 2~3cm 떠도 hold 고점
- cube_lift는 hold를 gate로 곱하므로, 가짜 hold 상태에서 lift 시도 → 못 들고 미끄러짐
  (관찰: "조금 들려고 하는데 hold 자세가 불완전" — 사용자 보고와 일치)

### 변경 (rewards.py + env_cfg_common.py CubeGraspRewardsCfg)
- `cage_points`에 `point_fractions` 파라미터 추가 (0=엄지끝, 1=대향 body). 기본값 None이면
  기존 등분 동작 그대로 (ChopsticksGraspRewardsCfg 등 다른 소비자 무영향)
- reach/hold/lift 셋 다 `point_fractions=(0.1, 0.5, 0.9)` — 반드시 같은 점 공유 (점을 나누면
  "엄지만 박기" 해킹 재발)
- hold/lift `depth_max` 0.02 → 0.005: 끝점 기준 간격 ~6.2cm(접촉 직후)에서 포화.
  0.02면 접촉 후에도 "더 조여라"가 남아 수박씨 짜냄 유도 (물리는 창 30~50% 실측 근거)
- 1.0까지 보내지 않는 이유: tip_link 원점은 패드가 아니라 마지막 관절 (패드는 2~3cm 앞)

### 기대 효과 / 확인할 것
- hold 고점 = 손끝이 표면에 닿은 상태로만 달성 가능 → lift gate가 진짜 파지에서만 열림
- 다음 런에서 Episode_Reward/finger_cage_hold의 절대값이 낮아지는 건 정상 (기준이 엄격해짐)
- 진행 중인 런(가중치 실험)과는 별개 — 이 변경은 다음 재학습부터 적용

## 2026-07-14 (밤 2) — 약지/새끼 커플링 (Schunk SIH 방식)

### 동기 (사용자 제안)
- "새끼손가락까지 grasping에 도움을 주면 잘 들 것 같다" — 근거 있음: palm-up 실측에서
  0.30kg을 버틴 주체가 바로 새끼/손바닥. 받침 능력은 이미 검증됨
- 이 손의 집게는 물리는 창이 좁아(오므림 30~50%) 받침 손가락이 있으면 성공 조건이 느슨해짐

### 구현: 액션/관측 차원 그대로, 약지/새끼가 중지를 따라감
- `MimicJointPositionAction` (mdp/actions/joint_actions.py) + `MimicJointActionCfg` (action_cfgs.py)
  - follower 목표 = source 목표 + (follower 기본 − source 기본). apply마다 복사
  - follower는 action_dim/관측에 안 들어감 → 18D/57D 유지, 논문(Schunk SIH 커플링)과 같은 발상
- env_cfg.py: `mimic={finger4/5_joint[1-4] ← finger3_joint[1-4]}` (8쌍)
- 손가락 게인: cube grasp env에서만 전 손가락 제조사 값(kp 2/2/1/1, kd 0.05)으로 통일.
  asset 기본(finger4-5 kp 20)은 유지 — chopsticks/functional_grasp가 접힘 유지에 필요
- `hold_folded_fingers` 이벤트는 유지 (리셋~첫 액션 사이 한 스텝 공백을 메움)
- CAGE_BODIES는 그대로 — 커플링이라 중지 보상만으로 약지/새끼가 따라 감쌈 (별도 유인 불필요)

### 검증 대기 (학습 중이라 Isaac 실행 금지)
- [ ] smoke test: 환경 생성 + 1 rollout (obs 57 확인)
- [ ] grip_capacity `--gui`: 같이 오므릴 때 이웃 손가락 충돌 여부, 받침 효과 확인

## 2026-07-14 (밤 3) — 보상 구조 논의 + r_T 계획 (구현 전 정리)

### 현재 런 (사용자 진행 중)
- 구성: 새 자세(joint3/4=-1.61/-1.62) + 편 손가락 + 커플링(약지/새끼←중지) + 가상점(0.1, 0.5, 0.9)
- 큐브 질량 0.10 → 0.20kg, episode 6 → 8s (240 step)
- ⚠ **가중치 주의: 활성 태스크(Indy-Wuji-Cube-Grasp)는 CubeGraspRewardsCfg를 씀**
  (`cube_grasp_env_cfg.py:90`). 최근 hold 12 / lift 100 편집은 ChopsticksGraspRewardsCfg에
  들어갔는데 이 클래스는 **어느 태스크에도 연결 안 됨** (functional_grasp 스켈레톤 주석뿐).
  따라서 이번 런의 실효 가중치는 hold 25 / lift 50 / reach 8 / palm 4 / floor 1.0

### 논의로 확정된 해석들
- **Episode_Reward 비중은 회계 착시가 절반**: 비중 = 가중치 × 달성률 × 지속시간.
  hold는 절대형이라 매 스텝 적립 → lift를 못 하는 동안엔 가중치와 무관하게 hold가 커 보임
- **cube_lift 0.004의 의미**: 최하 꼭짓점 기준이라 "모서리 기울이기"로는 0. 0.004는 큐브가
  진짜로 전부 공중에 뜬 순간이 존재했다는 뜻 = PPO가 증폭할 씨앗은 있음
- **일자 들기 실패("자세 바꿔가며 들다 놓침")의 원인 2개**:
  1. 관절공간 절대 액션엔 "위로"라는 방향이 없음 — 단일 관절 회전은 호를 그리며 파지축도 돌림
     (논문이 IK+상대변위를 쓴 이유)
  2. "들고 유지" 보상 층(r_T)이 비어 있음 — 순간 clearance와 안정 유지가 보상에서 구분 안 됨
- **질량**: 가벼울수록 같은 접촉 충격에 잘 튕김(가속도 ∝ 1/m). 100g은 밀어버림을 악화 → 0.2로 복귀
- **가중치 사다리**: reach < hold < lift < r_T 서열 유지. hold를 reach(8) 아래로 내리면 역전
  ("다가가기 > 감싸기") — hold 5안은 기각, 15~20 권장

### 논문 원문 확인 (arXiv:2307.16752, 밤에 재확인)
- **가상점 등분은 논문 명시**: "three equidistant points" × (엄지→중지끝, 엄지→중지마디) = 6점, 내부점
- **중앙점의 관용은 의도된 설계**: "ensures a positive response when an object is positioned
  between the thumb and other fingers imperfectly" — 불완전한 중간 상태에 gradient를 주려는 것
- **연금이 문제 안 되는 마개 = r_T**: 5000 (서열 r_T≫orient 500≫hold 25≫reach 1) + **일회성 +
  성공 시 에피소드 즉시 종료** → "앉아서 hold 벌기" 상한 < "성공 한 방", farming 원천 봉쇄
- constraint-based 버전(Eq.20): 검지끝 오차 <1cm AND 손 회전 <0.15rad AND 물체 z > 테이블+15cm.
  lift 조건 = 암묵적 파지 안정성 제약 (fake success 방지). 보조로 lifting reward(Eq.22)
- **노선 갱신**: 가상점 수술(0.1/0.9로 중앙점 제거)보다 논문 노선(관용 유지 + r_T 층 신설)이
  1순위. 수술은 r_T 넣고도 연금이 문제면 쓰는 카드

### 다음 계획 2가지 방안 (커플링 런 결과 보고 결정)
- **방안 ① r_T 기본형 (1순위, 물리가 판정)**:
  box_ground_clearance > 8cm AND cage gate > 임계 → K스텝(~0.5s) 유지 → 보너스(500~1000) +
  에피소드 종료. 재료(clearance, gate) 전부 기존 코드에 있음. 던지기는 gate/유지 조건에서 걸림
- **방안 ② 면 접촉 조건 (부분 목표 파지)**:
  큐브 yaw가 관측되므로 "엄지는 한 면 + 나머지는 반대 면" 판정 가능 (손끝→면 거리 + 법선 내적,
  접촉센서 불필요). 해 다양성 축소로 학습이 쉬워지나, 자세 지정 대리조건에 당해온 이력 있음.
  **순서: metric으로 먼저** (성공 에피소드가 어느 면을 잡았는지 TensorBoard 기록) → 영상 검증 →
  모서리 끼우기 등 나쁜 해가 실제로 보이면 그때 r_T 조건으로 승격

## 2026-07-14 (밤 4) — 손가락 목표 클램프 (음수 금지)

- 실패 모드 (커플링 런 GUI 관찰, 사용자): 음수 목표 → 검지/중지 과신전으로 확 벌어짐 →
  벌어진 채 바닥에 박힘 → kp 2로는 바닥 접촉을 못 이겨 못 접힘 → 그대로 에피소드 종료
- 조치: `MimicJointActionCfg.target_clamp` 추가. finger[1-3] 목표를 [0, +∞)로 클램프
  - 물리 관절 한계는 그대로 → 접촉에 밀려 젖혀지는 수동 순응성 유지 (한계 자체 축소는 기각)
  - process_actions에서 클램프하므로 커플링 follower(약지/새끼)도 클램프된 목표를 받음
  - 시작 자세(0)가 최대 폄이라 음수 목표는 기능 없음 (오므림 sweep 전부 양수로 검증됨)
- 알려진 부작용: 액션 [-1,0] 데드존 ("완전 폄"에 일괄 매핑). 문제 되면 리매핑(offset/scale 0.8)
- 가중치 이동(hold 12/lift 100 → CubeGraspRewardsCfg)은 사용자 지시로 보류 — 클램프만 적용

### 밤 4 후속 — 클램프 철회 (사용자 결정)
- 데드존 부작용(액션 [-1,0]이 전부 "완전 폄") 우려로 target_clamp를 커밋 직후 되돌림
- 코드는 커플링 커밋(2f1111e) 상태로 복원. 음수 과신전 실패 모드는 미해결로 남음
- 남은 카드: ① 리매핑 (offset 0.8/scale 0.8 -> 목표 [0,1.6], 데드존 없음) ② 그냥 두고
  학습이 스스로 피하는지 관찰 (hand_floor 페널티가 벌어진 채 바닥 박기를 이미 감점함)

## 2026-07-14 (밤 5) — play 진단 출력 추가 + 현재 정책 판정

- `scripts/rsl_rl/play.py`에 최신 run 자동 선택 옵션을 추가함.
  - `--latest_run`
  - `--load_run latest` 또는 `--load_run last`
- `play.py`에 GUI와 같이 보는 진단 옵션을 추가함.
  - `--print_diagnostics`: joint별 torque/velocity/reward/cube metric 출력
  - `--print_contact`: thumb/index/middle/palm contact force 출력
- 별도 상세 스크립트 `scripts/debug/policy_joint_diagnostics.py`도 추가함.
- `python -m py_compile`로 `play.py`와 `policy_joint_diagnostics.py` 문법 확인함.
- 성능 주의: `--print_diagnostics --print_contact --print_action_interval 1`은 터미널 출력/metric/contact 계산 때문에 GUI가 크게 느려짐.
  - 평소에는 `--print_action_interval 10~20` 권장.
  - contact는 필요한 순간에만 켤 것.
- 사용자 play 로그 판정:
  - 안정 구간(step 약 580~710)에서 `|raw| ~= 2.5`, `|applied| ~= 0.83`, `clip ~= 66.7%`.
  - arm 관절은 토크 부족이 아님. `joint1` torque가 약 `3~4%`, err가 약 `0.14rad` 수준.
  - finger 관절은 다수 `tq%=100`으로 effort limit에 붙음.
  - reward raw는 `finger_cage_hold ~= 0.46~0.48`, `cube_lift ~= 0`, `clearance ~= 0`.
  - metric은 `cage_inside_frac ~= 0.58`, `cage_span ~= 0.11m`, `thumb_middle_opposition ~= 0.52`, `thumb_index_opposition ~= 0~0.08`.
- 결론:
  - 현재 정책은 "대충 cage/hold는 만족하지만 실제 하중 지지/lift는 못 하는" local optimum.
  - arm 힘 부족이 아니라 finger action 포화, 불완전한 파지, lift/r_T 계층 부재가 핵심.
  - 다음 수정 후보는 finger action range/scale, finger joint2 처리, negative target 처리, contact/lift/r_T 성공 조건임.
  - reset 직후 출력은 판단에서 제외할 것. reset 직후에는 raw action이 크게 튀고 hold/cage가 0으로 돌아감.

## 2026-07-15 (아침) — 밤샘 런이 캠핑을 탈출해 들었다 + r_T 구현

### 밤샘 런 결과 (사용자 관찰 + 스크린샷)
- TensorBoard: lift reward 지수꼴 증가, Metrics lift > 0.01m 통과 → 밤새 계속 돌림 (사용자 결정)
- 아침 GUI: **큐브를 들었음.** 단, 자세는 손바닥을 하늘로 뒤집어 밑에서 받친 scoop —
  중력이 큐브를 손 안으로 눌러줘서 손가락 힘(0.6 N·m)이 거의 필요 없는 가장 싼 파지법.
  팔은 팔꿈치가 바닥 근처까지 접힘 (manipulability 낮음)
- 평가: 해킹 아님 (최하 꼭짓점 clearance로 진짜 듦). "드는 자세면 뭐든 진짜 파지" 정의상 합격.
  어제 play가 보여준 캠핑은 그 시점 체크포인트의 모습이었고, 학습이 그 국소최적을 탈출한 것
- 남는 문제: ① 접힌 팔로는 이동/배치가 어려움 ② scoop은 pinch가 아니라 functional grasp로
  재사용 불가. 둘 다 보상이 "스타일"을 요구한 적 없어서 생긴 정의의 문제

### r_T 구현 완료 (다음 런부터 적용)
- `ObjectLiftedHeld` 종료 판정 (rewards.py): clearance > 8cm AND gate > 0.3을 15스텝(0.5s)
  연속 유지 → 성공 종료. 캠핑(clearance 0)·들썩(순간)·쳐올리기(유지 불가) 전부 걸러짐
- `CubeGraspTerminationsCfg.success` + `CubeGraspRewardsCfg.lift_success`
  (is_terminated_term, weight 15000 = 로그 스케일 한 방 +500 ≫ hold 캠핑 총액 상한 120)
- 즉시 종료가 마개: "성공 후 hold 연금"과 "hold만 캠핑" 둘 다 기대수익에서 짐
- 사용자 가중치(이번 런): hold 15 / lift 100 / reach 8 / palm 4 (CubeGraspRewardsCfg에 반영됨)

### 열린 결정 (사용자)
1. 이 런 계속 vs r_T 얹어 재학습/resume — scoop이 이미 나왔으니 r_T는 "성공을 빨리 끝내고
   다음 에피소드"로 처리량을 올리고 캠핑 회귀를 막는 용도
2. scoop 자세 수용 여부 — 지금 r_T 정의로는 scoop도 성공. 스타일(위에서 pinch)을 원하면
   성공 조건이나 방향 항에 추가 조건이 필요 (박스 yaw 정렬 카드가 후보)

## 2026-07-15 (오전) — 테이블 도입 (자세 가설 검증 실험)

### 가설과 설계
- 진단: 팔꿈치 바닥 웅크림은 시작 자세 탓이 아니라 **바닥 큐브라는 task 기하** 탓
  (어떤 시작 자세든 손이 바닥까지 내려가야 하고, 든 뒤 팔을 세울 유인이 없음)
- 검증 실험: 큐브를 테이블(BASE_Z=0.40) 위로 → 하강량 60cm → ~17cm. 자세가 개선되면 가설 채택
- manipulability 실측 0.35~0.5 (raw): r_MP(j_max=0.02)는 scoop에서 **완전 데드존** (페널티 0).
  scoop은 특이점 자세가 아님 → r_MP 강화는 이 문제의 손잡이가 아님으로 판정.
  코드 주석의 옛 캘리브레이션("최대 0.113")과 모순 → j_max 만지려면 재측정 선행
- IK 논의: elbow 브랜치는 관절 부호 필터/시작 브랜치 연속성으로 고르는 게 맞지만,
  **우리 scoop은 같은 브랜치 안의 자세**(joint2 부호 동일)라 브랜치 제어로는 안 걸러짐.
  IK 전환의 근거는 자세 교정이 아니라 들기/운반 역학 (별도 결정으로 분리)

### 구현 (cube_grasp_env_cfg.py)
- Support kinematic cuboid 0.5×0.5×0.40 @ (0.62, -0.18), 큐브는 상판 위 (z=0.43)
- surface_z 배선: cube_lift / hand_floor / success(r_T) 전부 BASE_Z 기준으로 __post_init__ 오버라이드.
  metrics는 managers.py:244가 cube_lift.params에서 자동으로 읽음
- 스모크 테스트 통과: action 18 / obs 57 유지, Episode_Termination/success + lift_success 등록 확인

### 이번 런에 들어가는 변경 요약 (fresh 학습)
- 테이블 (신규) + r_T 성공 종료/보너스 (신규) + 커플링 + 가상점(0.1,0.5,0.9)
- 가중치: reach 8 / hold 15 / lift 100 / lift_success 15000 / palm 4 / floor 1.0, 질량 0.20, episode 8s
- 관찰 포인트: ① Episode_Termination/success 비율 ② 팔꿈치가 서는가 (스크린샷)
  ③ palm-up scoop이 유지되는가 ④ 8cm 직전 서성임(성공 회피) 지문

### 오전 2 후속 — A/B 통제: r_T 임시 제외 + 팔 링크 페널티 추가
- r_T(success 종료 + lift_success) 주석처리 (사용자 지적: 테이블 효과를 분리하려면 전 런과
  보상 구성이 같아야 함). 재활성 시 env_cfg_common 두 항 + cube_grasp_env_cfg의
  success surface_z 오버라이드를 세트로 살릴 것
- arm_floor 페널티 추가 (weight 2.0): link[2-6]이 바닥 0.12m 아래로 내려간 깊이 비례 감점.
  링크 선정은 시작 자세 실측 기준 (link1=0.08은 항상 낮아 제외; link2~6 = 0.30~0.71).
  기준면은 테이블이 아니라 바닥 — 팔꿈치는 테이블 옆에 정상적으로 존재 가능
- 절대 페널티(≤0)라 자세가 좋으면 0 → 테이블이 자세를 고치면 이 항은 침묵 (A/B 오염 최소)
- 스모크 테스트 통과: 종료항 1개(time_out), arm_floor 등록, action 18 / obs 57
- 이번 런 = 전 런(scoop) 대비 차이: 테이블 + arm_floor 둘뿐

### 낙하 종료 추가 + surface_z 배선 복구 (2026-07-15 오후)
- cube_dropped 종료 추가: 큐브 중심 < 상판 - 5cm → 즉시 리셋 (테이블 도입으로 생긴
  "떨어뜨림 후 회수불가 에피소드 낭비 + 쫓아 내려가기" 차단). Metrics/cube_lift 음수가
  이 실패 모드의 지문이었음 (보상은 clamp라 음수 불가, metric은 스폰 대비 변위)
- ⚠ 사고 복구: __post_init__의 surface_z 배선(cube_lift/hand_floor)이 편집 중 삭제돼 있었음.
  이대로 학습하면 상판 큐브 clearance가 스폰부터 +BASE_Z → lift 만점에서 시작하는 버그.
  복구 + "지우면 안 됨" 경고 주석. cube_dropped의 minimum_height도 BASE_Z 파생으로 배선
- 사용자 변경 반영된 현재 구성: BASE_Z 0.25, 테이블 0.5×1.5 (y 확장 — 낙하 감소),
  큐브 (0.62, -0.20), arm_floor 주석 (테이블 단독 A/B)
- 스모크 통과: 종료항 time_out + cube_dropped, action 18 / obs 57

## 2026-07-15 (오후 2) — 운반 task 구현 (다음 런용, 테이블 A/B 런과 무관)

### 결정
- 발전 방향 3안 중 **B(랜덤 goal 운반) 채택**: 기계(r_T·gate·종료) 검증 + 파지 강건성 시험.
  C(파지점 개선)의 알맹이는 grip_capacity 스크립트 검증으로 압축, A(젓가락)는 그 다음.
  IK 전환 결정은 젓가락 진입 시점으로
- **orientation 제외**: 대칭 큐브엔 목표 방향 정의 불가(자의적 지표 재도입), scoop으로
  재배향은 별개 기술, 런당 새 축은 하나. 필요 시 TriFinger식 8-keypoint로 업그레이드

### 구현 (obs 57 / action 18 불변)
- `UniformCubeGoalCommand`: 에피소드당 goal을 env-로컬 박스에서 샘플 (리셋만, 1e9s),
  GUI 초록 구 마커, Metrics/cube_goal/error_pos
- **관측 버그 수정**: 기존 cube_to_goal은 월드 고정점 + env_origins 미보정 → 다중 env에서
  env마다 다른 상수(잡음 채널). 커맨드 기반 로컬 프레임으로 교체 (dim 3 유지)
- `ObjectToGoalProgressReward` (500, 논문 orient 자리): 잡은 채(gate, 양수만) goal 접근
  차분 지불. 기준선은 리셋 후 첫 호출 seeding (reward reset 375 < command resample 381)
- `ObjectAtGoalHeld` r_T: goal ±5cm + gate>0.3 유지 0.5s → transport_success +500 + 종료.
  goal이 공중(상판+10~30cm)이라 들었음 자동 함의
- 사다리 완성형: reach(8) < hold(15) < lift(100) < 운반(500) < r_T(15000=+500·종료)

### 사고와 수정
- commands.py의 cfg 전방 참조 주석이 import 시 NameError → mdp 패키지 전체 import 실패
  → play.py 즉사. 문자열 주석으로 수정. py_compile은 못 잡는 종류 — 학습 중이라 스모크를
  건너뛴 구멍. **학습 종료 후 1 env 스모크 필수** (운반 env 런타임 생성은 아직 미검증)
- play 호환 원칙 정리: 차원 일치=하드 제약 / 의미 일치=소프트 제약(옛 체크포인트는
  cube_to_goal 3칸을 무시하도록 학습돼 goal을 안 쫓는 게 정상) / 보상·종료·장면=자유

### 커리큘럼 1단계 — 고정 goal로 축퇴 (사용자 결정)
- 랜덤 운반 전에 "잡아서 들고 +20cm에서 멈춰 유지"부터: goal 범위를 스폰 위 한 점으로
  (lo=hi). 기계는 동일 — 운반 차분층·r_T·마커 전부 그대로, 범위만 축퇴
- 2단계(랜덤 박스) 확장값은 post_init에 주석으로 보존. 논문 curriculum(close-start) 정신
- lift dense는 8cm 포화라 8→20cm 구간의 gradient는 운반 차분층이 담당 (역할 분담 자연스러움)

### 운반 env 스모크 테스트 통과 (2026-07-15)
- 1 env 1 iter: cube_goal 커맨드, cube_transport(500) + transport_success(15000) 보상,
  cube_dropped + success 종료 전부 등록. action 18 / obs 57 불변. 4096 학습 준비 완료

## 로드맵 확정 (2026-07-15, 사용자)
1. 고정 위치 운반 (진행 중)
2. 큐브 → 직육면체 교체 (젓가락 프록시 기하로 접근)
   - 선행 작업: object_half_extent 6군데 하드코딩 중앙화 (BASE_Z 패턴)
3. 랜덤 위치 운반 (구현 완료 — post_init 주석 3줄 복원이 전부)
4. 랜덤 크기 직육면체
   - startup 스케일 랜덤화 + SDF의 env별 half_extent 텐서화 + 관측에 크기 추가(obs 단절)
5. 랜덤 위치+회전+크기 통합
   - 회전 관측 필요 → ④에서 obs 스키마 개정 시 크기+방향을 한 번에 넣어 단절 1회로 묶기
6. 젓가락 (functional grasp) — IK 전환 결정 지점, 목표 파지 g 정의

## 2026-07-15 (저녁) — 1단계 실행 방침 논의 정리

### 파인튜닝 결정 (fresh 대신 테이블 런 체크포인트에서 resume)
- 호환성: obs 57/action 18 불변 → 로드 가능. cube_to_goal 3칸의 의미 변화는 1단계(고정
  goal)에선 무해 — goal이 상수라 그 채널 없이도 과제가 풀림 (채널은 3단계부터 필수)
- 전이 이득: 테이블 정책이 이미 접근+파지+8cm 들기를 함 = 운반의 앞 절반
- 우려 검토 "불안정한 lift가 transport 하강 감점에 억제되지 않나": 수치로 기각 —
  실패 사이클(5cm 들었다 놓침)도 lift(+8) ≫ transport 세금(−0.9)이라 순익 +7.
  하강 감점은 오히려 "놓치면 손해"를 처음 가르치는 안정화 신호
- 감시 지표: 파인튜닝 시작 후 Episode_Reward_Raw/cube_lift가 하락 추세면 억제 실재
  → transport 500→200으로 낮춰 재시작 (헤지 카드)
- 부수 효과: "만세 배회"(goal 지나쳐 1m 치켜들기)는 transport의 멀어짐 감점으로 자동 교정

### 성공 반경/동심원 보상 논의
- goal_radius 5cm = 씨앗 확률용 시작값 (큐브 6cm 기준 "한 개 이내"). 정밀도는 커리큘럼으로:
  1단계 성공 후 3cm로 조이기
- "가까울수록 500~5000" 동심원 제안 평가: 매 스텝 지급(절대형)이면 반경 바깥 껍질
  서성임 연금 재발 → 기각. 정당한 형태 = 포텐셜 볼록화: 차분층의 거리 함수를 선형 d 대신
  φ(d)=1/(0.05+d)로 (DexPoint reach 꼴, telescoping 보존이라 farming 불가)
- 적용 트리거: play에서 "goal 근처 지나침/맴돌기" 관찰 시. 신호 없이 선제 적용 안 함

### cube_transport 동작 확인 (사용자 이해 합치)
- 각 env 안에서 큐브-goal 3D 직선거리를 "좁힌 양"에 지급 (차분), 전진만 gate(잡음) 곱,
  후퇴는 전액 감점. env_origins 보정으로 4096 env 전부 올바름

### 파인튜닝 초기 진단 (run 2026-07-15_15-32-40, 재개 iter 2050 → 2536 시점)
- 재개 직후 일시 퇴행: 낙하율 0.9% → 44% 스파이크 → 33%로 회복 중. error_pos 0.26 → 0.87
- 퇴행 원인 분해:
  1. 관측 분포 이동 — 옛 cube_to_goal(상수−큐브위치)을 정책이 위치 신호로 쓰고 있었고,
     채널 의미 교체로 입력이 흔들려 손놀림이 거칠어짐 ("무시하도록 학습됐다"는 예측은 절반만 맞음)
  2. 낙하에 요금 발생 — transport 하강 감점(쳐내기 한 번 ≈ −20) + cube_dropped 종료(잔여
     수입 몰수). 바닥 시절 공짜였던 거친 취급이 청구되기 시작
  3. value function이 새 보상 스케일에 재적응 중
- 좋은 신호: cube_lift raw 0.08 → 0.11 상승 (transport의 lift 억제 우려는 데이터로 기각),
  낙하율 하락, episode length 회복
- error_pos가 높게 고정된 이유: 종료 시점 거리라 쳐내진/낙하 에피소드가 평균 지배
- **판정 기준 (iter ~5000)**: 낙하 <15% + transport 음수→0 접근 + error_pos <0.4 진입이면
  계속. error_pos가 여전히 0.8대면 관측 이동 데미지 판정 → fresh 전환

### 보상 부호 분리 리팩토링 (2026-07-15 저녁, 사용자 제안)
- 원칙: "페널티면 페널티, 리워드면 리워드" — 항마다 단일 부호
- cube_transport → **best-so-far 차분 (+ 전용)**: 에피소드 최소거리 신기록을 깬 양만
  gate 곱해 지급. 총액 = 시작거리 − 최소거리로 고정 (farming 불가 유지). 후퇴/왕복/재접근 0원
- **drop_penalty 신설 (− 전용)**: cube_dropped 종료 시 정액 −100 (weight −3000).
  기존 ± 혼합형의 "굴러간 거리 비례 벌금"(물리 우연 과세)을 정액으로 교체.
  근거: 회당 ~−20 청구서로도 낙하율 회복이 작동했음 → 5배, r_T(+500)의 1/5
- 적용: 현 파인튜닝 체크포인트(2536+)에서 새 보상으로 재개 권장 (관측 적응 보존)
- 스모크 테스트는 사용자가 직접 실행

## 2026-07-15 (밤) — 다음 판 계획 확정 (사수님 방향 + 논의)

### 사수님 지시
- 젓가락만큼 얇지 않되 직육면체 비율을 유지한 랜덤 직육면체 — 4096 env가 각자 다른 상자
  (살짝 뚱뚱이 ~ 길쭉이). obs에 상자 크기·위치 포함

### 확정 계획 (로드맵 ②+④ 합본)
1. 지금: 고정 위치 운반 런 완주가 관문 (success 발생 + GUI 확인)
2. 다음: 랜덤 직육면체 판 — 커스텀 startup 이벤트(치수 샘플: 단면 3~6cm × 비율 1.5~3,
   scale 적용, half_extent (N,3) 버퍼, env별 스폰 z) + replicate_physics=False
   + half_extent 소비처 6곳 텐서화 + **obs 개정 한 번에: 크기(3) + orientation quat(4) → 64**
   + fresh 학습
3. 나중: 회전 랜덤화 확대(±180°), goal orientation(command), 8-keypoint 표현은 젓가락 카드

### 결정 근거 메모
- orientation을 지금 obs에 넣는 이유: 직육면체는 좁은 면을 가로질러 잡아야 해서 방향이
  기능임 + 접촉 중 회전을 정책이 봐야 함. 단절(fresh)이 어차피 발생하는 시점에 합류
- 초기 yaw ±30° 소량 랜덤화(채널 활성화)는 구현 시점에 사수님 컨펌 후 결정
  ("관측과 변주는 세트" 원칙 — cube_to_goal 깨우기 비용 실측 교훈)
- obs 코드는 현재 런 종료까지 동결 (바꾸면 진행 중 체크포인트 play 불가)

## 2026-07-16 — 오버슈트 진단 + transport-φ 설계 킵 (미구현)

### 진단 (run 17-49-26, 11.4k iter)
- success 0의 원인 = 과녁 통과: clearance 최종 0.33/최대 0.70m (goal 0.20), error_pos 0.17→0.52
- mean_reward 262의 84%가 lift — 높이 무관 동일 지급이라 20cm에서 멈출 이유 없음
- best-so-far 적립이 에피소드당 3.5mm어치: goal 10cm 안에 진입 자체를 못 함 (수직 오버슈트 + 횡 드리프트)
- 구조 원인: lift 8cm 포화 후 균일 연금 + best-so-far 전환으로 "멀어지면 감점" 브레이크 소멸

### 킵한 설계 (transport in-place 수정, 사용자 안 + 조정)
- φ(d) = 0.05/(0.05+d) 역수형 포텐셜, reward = (φ(현재) − φ(베스트))⁺ × gate
- 총액 = φ(최소거리)−φ(시작거리) 고정 (farming 불가, + 전용 유지)
- weight ~4000: 여정 총액 ~103 (r_T의 1/5), 마지막 5cm에 65% 집중 ("가까울수록 크게")
- "±10cm 창"은 기각: 현 정책이 10cm 진입을 못 해 무신호가 됨 — 전 구간 역수가 상위호환
- 실패 신호 정의: φ 적립↑인데 success 0 지속 = "통과만 하고 정착 안 함" → 근접 연금 2차 카드
- (제안했던 별도 goal_proximity 연금 항은 미채택 — 미사용 함수도 제거)

### 결정: 환경 수정(랜덤 직육면체)부터 진행, transport-φ는 그 판에서 함께 적용 검토

## 2026-07-16 (오후) — Indy-Wuji-Box-Transport 태스크 구현 (사수님 방향)

### 방침: 큐브 태스크 동결, 별도 태스크로 복제 (사용자 지시)
- 신규 task: `Indy-Wuji-Box-Transport`, experiment `indy_wuji_box_transport` (로그 폴더 분리)
- 큐브 태스크(Indy-Wuji-Cube-Grasp)는 obs 57 그대로 — 기존 체크포인트 play 호환 유지
- 신규 파일: grasp/box_mdp_cfg.py (MDP cfg 사본+확장), grasp/box_transport_env_cfg.py (장면),
  grasp/indy_wuji_box/{env_cfg, __init__(등록), learning/rsl_rl_cfg}
- 공유 mdp에는 신규 함수 추가 + SDF 헬퍼 3곳의 하위호환 일반화만
  (버퍼 없으면 기존 상수 경로 — 큐브 태스크 동작 불변)

### 랜덤 직육면체 (4096 env 각자 하나)
- `randomize_box_dims` (prestartup): 단면 w~U(3,6cm) 정사각 × 길이 w×U(1.5,3) —
  env별 USD xformOp:scale 적용 + env.box_half_extents (N,3) 버퍼 저장. 런 내내 고정
- `set_box_default_height` (startup): env별 스폰 z = 상판 + 자기 반높이
- scene.replicate_physics = False (env별 지오메트리 파싱). 질량은 0.20 고정 (밀도 변수 통제)
- 보상/판정/metrics의 half_extent는 버퍼 우선 (상수 인자는 fallback)

### 관측 64 = 57 + box_size(3) + box_quat(4)
- 크기: 버퍼×2. 방향: quat (직육면체는 좁은 면을 가로질러 잡아야 해서 방향이 기능).
  초기 회전은 고정 — 채널은 접촉 중 회전에만 신호 ("반쯤 활성", 초기 yaw 랜덤화는 사수님 컨펌 후)
- 보상 구성은 큐브 태스크 사본: reach 8/hold 15/lift 100/transport 500(선형 best-so-far,
  φ 전환은 킵)/r_T 15000/drop 0(2단계 커리큘럼)/goal 고정점 상판+20cm

### 검증
- 스모크(4 env): obs 64, action 18, prestartup/startup 이벤트 실행, 전 항 등록 ✓
- end-to-end 프로브(scripts/debug/box_dims_probe.py): env별 상이 치수(폭 4.1~5.4 × 길이
  6.5~15.2cm), 버퍼=USD scale 일치, 정착 잔차 ±0.0mm ✓

### lift 단계 구성으로 잠금 (사용자 결정)
- Box-Transport에서 transport 층 임시 제외: cube_transport / transport_success / success 종료
  3개 세트 주석처리 — "랜덤 상자를 잡고 드는 것"부터 검증
- 관측(64)과 goal 커맨드는 유지 → transport 재활성 시 체크포인트 이어쓰기 가능 (obs 불변)
- 활성 구성: reach 8 / hold 15 / lift 100 / drop 0 / palm 4 / manip 1 / floor 1,
  종료 = time_out + cube_dropped
- 스모크 통과 (obs 64, 종료항 2개)

### Box lift-only 첫 런 시작 (2026-07-16, 코드 스냅샷 e5d6a68)
- 4096 env fresh. 관전 포인트: ① hold→lift 이륙 (단일 큐브 대비 2~3배 느려도 정상)
  ② 크기별 파지 편차 (얇은 3cm 단면만 실패하면 pinch 물리 한계 신호) ③ drop_penalty 곡선
- 다음 관문: lift 안정화 → transport 3종 세트 재활성 + resume (obs 64 유지라 이어짐)

### 설계 결정: 박스 yaw 정렬 항 없음 (의도, 2026-07-16)
- yaw 정렬 shaping을 넣지 않고 물리가 가르치게 둠 (사용자 결정): 물리는 창 2.8~5.8cm vs
  긴 축 6.5~18cm라 폭을 가로질러 잡는 것만 성립 — 기하가 정답을 강제함
- 정책에는 box_quat 관측(정보)만 제공. 검증: play에서 폭-가로 파지 확인
- 진짜 시험은 초기 yaw 랜덤화 단계 — 그때 실패하면 shaping 검토 (원칙상 젓가락 g로 넘김)

### 큐브 태스크 보상 v2 적용 (2026-07-16, 오버슈트 처방 — 사용자 결정)
- **cube_lift 100 → 50**: 오버슈트 수입의 84%였던 "8cm 포화 후 높이 무관 균일 연금" 축소.
  0~8cm 사다리 역할은 유지
- **cube_transport → 역수 포텐셜 + 창 (킵해둔 φ 설계 + 사용자 창 채택)**:
  φ(d) = 0.05/(0.05 + min(d, 0.10)), reward = (φ신기록 갱신분) × gate, weight 4000
  - 창(±10cm) 밖 지급 0: 창까지는 lift 사다리가 데려옴 (goal이 스폰 위 +17cm라 8cm 들면
    d≈0.09 = 창 안 — 이 배치에서는 창의 무신호 우려가 해소됨)
  - 마지막 5cm에 총액 대부분 집중, 완주 총액 ≈ +89 (r_T의 ~1/5)
  - telescoping 유지 (총액 = φ(최소거리) − φ(창)), + 전용, farming 불가
- box_mdp_cfg의 주석 블록도 새 파라미터로 동기화 (재활성 시 그대로 사용)
- ⚠ 스모크 미실시 (box 학습 중) — 큐브 태스크를 다음에 돌리기/play 하기 전에 1 env 스모크
  필수 (시그니처 변경: distance_max 제거, potential_eps/window 추가)

### transport 창 제거 → 전 구간 역수 φ (사용자 직접 수정, 2026-07-16)
- 근거 (사용자 추론): 스폰 랜덤 ±6/8cm로 "높이는 맞는데 횡으로 창 밖"(d≈13cm)인 에피소드가
  transport 무신호 — 창은 그 구멍을 만들고, 역수 φ는 원래 "멀리서 완만 + 가까이 가파름"의
  2층 구조를 연속으로 내장하므로 창 없는 쪽이 상위호환
- 변경: rewards.py에서 window 파라미터/clamp 제거 (φ = 0.05/(0.05+d) 전 구간),
  env_cfg_common cube_transport에서 "window" 삭제, box_mdp_cfg 주석 블록 동기화,
  arm_floor 사체 블록 정리(사용자). weight 4000 유지 — 완주 총액 ≈ +103 (r_T의 1/5)
- 큐브 보상 v2.1 확정 상태: reach 8 / hold 15 / lift 50(0~8cm 사다리) /
  transport 4000(전구간 역수 φ, best-so-far, gate) / r_T 15000(+500, goal ±5cm 유지 0.5s,
  즉시 종료) / drop 0(2단계 커리큘럼 대기) / palm 4 / manip 1 / floor 1
- 릴레이 구조: lift(0~8cm) → φ(8cm부터 창 없이 연속, 근거리 집중) → r_T(도착·종료)
- ⚠ ObjectToGoalProgressReward 시그니처 변천: distance_max(v1) → potential_eps+window(v2)
  → potential_eps만(v2.1). 옛 파라미터를 cfg에 남기면 env 생성 크래시
- ⚠ 큐브 태스크 스모크 미실시 (box 학습 중) — 큐브를 돌리기/play 전 1 env 스모크 필수

### 정리 결정 (2026-07-16 오후)
- ObjectLiftedHeld 삭제 (사용자 직접): rewards.py 클래스 + env_cfg_common 주석 블록 +
  cube_grasp_env_cfg 참고 주석 3곳 세트로 제거해야 완전
- 킵 카드 추가 — "차분형 +only 통일": 다음 fresh에서 PalmFacingProgressReward를
  best-so-far(+전용)로 전환 검토 (사수님 단일부호 원칙). 차분형이라 어느 쪽이든 farming은
  없고, 겨눔 유지는 reach의 facing gate가 담당하므로 음수 제거 실익은 일관성.
  ⚠ 예외: reach(ObjectCageProgressReward)의 음수는 유지 — drop_penalty 0인 현재
  "쳐내기"에 대한 유일한 즉시 과금이라 일을 하고 있음

### palm_normal_b 문서화 보강 + palm_facing best-so-far 결정 (2026-07-16)
- (0.19, 0.28, 0.94)의 정체를 rewards.py palm_facing_object docstring에 권위 문서화:
  손바닥 법선이 아니라 "파지 개구부 축"(오므린 손끝 수렴 방향, palm_link 로컬).
  도출 실측 3개(법선 ≈ +x, 손가락 +z, 수렴점 (0.065, 0.006, 0.097))와 65° 어긋남 명시.
  측정 스크립트 유실 사실 + 재검증 수단(--show_palm_vectors, FK 재도출) 기록.
  cfg 3곳 주석은 포인터로 교체
- palm_facing을 best-so-far(+전용)로 바꾸기로 결정 (사용자 — (curr−best)⁺ 형태 재도출).
  단순 (curr−prev)⁺ 클램프는 진동 farming(하락 무과금+상승 재적립)이라 기각.
  적용 시점: 다음 재시작 (돌고 있는 런 무관)

### box transport 3종 재활성 (2026-07-16, 관문 통과)
- 관문 판정: lift 이륙 + "잘 잡긴 하는데 잡고 고정이 안 됨" (사용자 play 관찰) —
  잡은 뒤 인센티브 공백의 지문. r_T의 hold_steps(0.5s 유지)가 "고정"을 직접 보상하므로
  별도 orientation 보정 항 없이 3종 해제로 처방 (orientation 항은 2차 카드:
  해제 후에도 success 0이면 물리 진단 먼저 → cube_speed/마찰/orientation 순)
- cube_transport(φ 4000, 전 구간 역수) + transport_success(15000) + success 종료 해제.
  palm_facing best-so-far 전환도 이번 재시작에 함께 반영됨
- 실행 순서: 런 정지 → 1 env 스모크 → 현 체크포인트에서 resume (obs 64 불변).
  drop_penalty는 success 자리 잡은 뒤 −3000 (커리큘럼 순서 유지)

### "B 설계" 카드 확정 — lift 은퇴 + 근접 연금 통합 (2026-07-16, 미적용)
- 사용자 설계: lift 제거 후 "hold gate × error 역수" 연금 하나로 — 공중 유지(lift 역할)와
  error 축소(transport 역할)를 한 항이 동시 지불. r = gate × 0.05/(0.05+d), w~75
- 장점: 일시불이 없어 "스침 후 캠핑" 원천 불가 (머물러야 벌고, 공 안 체류는 success가 회수).
  "transport 유지 조건" 요구도 자동 충족 (연금 = 유지가 지급 조건)
- 안전 산수 (γ=0.99): goal 중심 ~2.5/스텝(공 안은 0.5s 종료), 공 바깥 PV ~110 ≪ r_T 500,
  탁자 캠핑 ~0.3 ≪ 공중 유지 1.5+
- ⚠ 적용 시 cube_lift는 삭제 금지, weight 0 은퇴 — managers.py:244가 surface_z를
  cube_lift.params에서 읽음 (metrics 어긋남 방지)
- 함수(object_goal_proximity)는 rewards.py에 미배선 부품으로 준비됨 (⚠ 헤더 표기).
  적용 시점: 사용자 신호 대기. 현행 box resume은 A(lift 50 + transport 일시불 + r_T)로 관찰
  — cube(v2.1)와 box가 각각 일시불/연금 설계의 자연 A/B가 될 수 있음

### 재학습 원칙 확정 (2026-07-16, 사수님 지시): 보상/태스크가 바뀌면 fresh, resume 금지
- "아예 다른 학습인데 왜 이어서 하냐" — lift 성공본에 transport를 추가했으면 그건 다른
  task이므로 처음부터 학습. resume 결과는 "새 설계의 산물"이 아니라 "옛 정책+적응"이라
  설계 검증으로 무효 (귀속 불가) + 옛 습관 오염 (실측: 오버슈트/만세/재개 퇴행 44%)
- 즉시 적용: box transport 판 fresh / 큐브 v2.1 fresh (기존 resume 권고 철회)
- drop_penalty 2단계는 resume 방식 폐기 → 단일 런 내 curriculum manager
  (modify_reward_weight, N스텝 후 0 → −3000)로 재설계 예정
- resume이 허용되는 경우: 같은 설정의 순수 연장(크래시 복구, iter 추가)뿐

## 2026-07-17 — transport success 보상/종료 연결 및 유지 관찰 정리

- `ObjectAtGoalHeld`의 `hold_steps=15`는 hold reward 누적 시간이 아니라 성공 판정에 필요한
  연속 유지 시간임. 30Hz에서 15 step은 0.5초임.
- 성공 조건은 goal 거리 `< 0.05m`, cage gate `> 0.3`을 15 step 연속 만족하는 것임.
- 성공 step에서 `transport_success(is_terminated_term("success"))`가 raw 1을 읽고,
  현재 weight 30000과 dt 1/30이 적용되어 PPO reward `+1000`을 한 번 지급함.
- 같은 step에 success termination으로 episode를 끝냄. goal 이탈 후 재진입으로 terminal
  reward를 반복 적립하는 것과 성공 뒤 hold/lift 연금을 계속 받는 것을 막는 구조임.
- play에서 0.5초 이후 유지 안정성을 보려면
  `env.terminations.success.params.hold_steps=1000000`을 사용함. 정책과 reward 구조는 그대로고,
  성공 보상/종료만 8초 timeout 뒤로 미뤄짐. `cube_dropped` 종료도 유지됨.
- `env.terminations.success=null`만 쓰면 `transport_success`의 `term_keys="success"` 참조가
  사라져 reward manager 초기화가 실패하므로 사용하지 않음.
- 실행 명령과 주의점은 root `CLI.md`의 `성공 후 유지 상태 확인 Play`에 정리함.

## 2026-07-17 — 큐브 v2.1 수렴 판정 + lift-off(A′) 실험 계획

- 큐브 v2.1 (run 2026-07-16_16-05-23) 수렴: success 89.4% @iter 7,440, 낙하 3.0%,
  error_pos 0.046, 오버슈트 없음 → φ 설계 검증 + 로드맵 ①관문(고정 위치 운반) 통과.
- play 관찰: 파지 진동 + 장기 유지 약함 (성공=0.5초 종료 이후는 미학습 OOD 구간 + gate 0.3
  관대 합격). 유지력 카드: hold_steps 15→30~60 + gate_threshold 0.3→0.4 (다음 큐브 fresh).
  depth_max 증량은 수박씨 배출 이력으로 배제.
- 박스: success 43.5% @iter 8,229 상승 중, 낙하 15.9%로 자연 하락 → drop 커리큘럼 긴급도↓
  (다음 fresh 카드로 유지).
- lift-off 실험(A′) 시작: cube_lift weight 0 (term은 유지 — surface_z metric 배선),
  transport 일시불 유지, fresh. 예상 문제 3개와 대응은 AGENTS.md 로드맵 1-a 절 참조.
  판정 트리: 재도전 무보상·중간 처짐 발생 → B안(gate×φ 연금) fresh / 초반 시드 저하만 →
  lift 훈련바퀴 커리큘럼(modify_reward_weight 50→0) / 무증상 → lift 영구 은퇴 확정.

## 2026-07-17 — run 파라미터 검증·대장 + 추후 계획 확정

- 진행 중 비교 2건, env.yaml 스냅샷으로 파라미터 검증 (사용자 구술과 전부 일치):
  - 큐브: `2026-07-16_16-05-23` (0.2kg, lift 50 — 89.4% 수렴 킵) vs `2026-07-17_23-06-15`
    (0.1kg, lift 50 — 질량만 다른 비교 fresh, 진행 중)
  - 박스: `2026-07-16_16-33-21` (0.1kg, lift 50 — 43.5%+ 확인) vs `2026-07-17_23-15-16`
    (0.1kg, lift 0 — lift-off A′, 진행 중)
  - 공통: transport 4000, r_T 30000(+1000), reach 8/hold 15/drop 0, hold_steps 15.
    두 비교 모두 단일 변수 차이 — run별 상세 대장은 ACTIVITY_2026-07-17.md.
- 추후 계획 순서 확정 (사용자): 비교 승자 고정 → ① goal 위치 랜덤화(command ranges)
  → ② goal orientation obs 포함 학습 → ③ orientation까지 success 판정 포함.
  AGENTS.md 로드맵 '일반화 순서'에 반영함. 각 단계 fresh run.

## 2026-07-17 — 일반화 순서 보완 (사용자 결정)

- 채택: ① 랜덤화 축 확장에 물체 spawn 범위 + **박스 크기 범위**(현행 단면 3~6cm·비율
  1.5~3에서 확장, 목표치는 크기-버킷 실측 후) 추가 ② orientation은 obs+shaping+success를
  한 fresh에 결합 (obs만 단독 학습 금지 — 죽은 채널) ③ 유지력 조이기(hold_steps 15→30)를
  승자 고정 직후 fresh에 편입 (gate_threshold는 별도 차수).
- 보류: 단계별 통과 게이트 수치화 — 버킷별 분해 지표만 유지, 숫자는 단계 진입 시 결정.
- 미확정 제안으로 킵: drop 커리큘럼의 상비 장치 재분류, 젓가락 직전 얇은 물체 브리지.
- AGENTS.md '일반화 순서'에 전체 반영.

## 2026-07-18 — 비교 판정 + B안 배선 구현

- 판정: 큐브 질량 0.1kg 승 (98.2% @6,866 vs 0.2kg 88.0%, 이륙 4배 빠름 — 커리큘럼 초반값로 킵).
  박스 A′(lift 0) 기각 — 들기·운반은 되나(max clearance 0.20, transport 수입 동급) 정착 실패
  (φ 현금화 후 내려놓고 hold 파밍, success 0%). 예상 문제 1+2 혼합 → B안 처방.
  파지 자세 가설은 max 지표로 기각. 지표 함정(final만 보면 오판, per-step 평균 구성 편향)은
  AGENTS.md 1-a 판정 절에 기록.
- 구현: object_goal_proximity를 box_mdp_cfg에 goal_proximity 항으로 배선 (기본 weight 0,
  파일 기본 = A안 승자 구성). 4 env 스모크 통과. hydra 오버라이드 정수 타입 함정 발견
  (0.0 소수점 필수 — CLAUDE.md 추가).
- 다음 라운드 2슬롯: A = B안 fresh (lift 0 / transport 0 / proximity 75 오버라이드),
  B = lift 50 + hold_steps 30 (유지력 조이기). 런치 명령은 ACTIVITY_2026-07-18.md.

## 2026-07-18 — B안 보류, orientation v1 구현

- 사용자 결정: B안 통합 보류 (큐브 배선 철회, 박스 부품 잔류). orientation 성공 + 크기 확대 우선.
- 논문 확인: pre-grasp의 r_orient는 명목 자세 차분형 / TriFinger는 quat error 대신 8-keypoint
  거리 (pos+quat 분리 보상은 ori 학습 느림 — 실측). shaping 필요해지면 keypoint로.
- orientation v1 구현 완료: success에 대칭 최소각 15° 조건 (ObjectAtGoalHeld.ori_limit,
  box 기본 탑재), box_ori_error 지표 4종, obs 불변, 수학 단독 검증 + 4env 스모크 통과.
  다음 박스 fresh = v1 (오버라이드 불필요). v2=opposition 조건, v3=target grasp g 단계 계획은
  AGENTS.md 일반화 순서 참조.

## 2026-07-18 (저녁) — 크기 확장 구현, 2슬롯 라운드 개시

- 슬롯 A 런치됨 (사용자): orientation v1 (ori 15°, 구 크기 3~6cm — 크기 확장 코드 전 스냅샷).
- 크기 확장 구현: randomize_box_dims에 length_range (길이 독립 샘플 U(1.5×폭, 20cm)),
  단면 하한 1.5cm. 젓가락 프록시(1.5×1.5×20) 포함, 샘플 검증 + 스모크 통과.
- 슬롯 B 런치 명령·주의(파일 기본 = ori+새크기 결합)는 ACTIVITY_2026-07-18.md 참조.
- 단면 2.8cm 미만 = 손끝 간격 미실측 구간 경고 유지 — 버킷 분해로 판독.

## 2026-07-18 (오후) — 2슬롯 라운드 런치 (검증됨)

- 슬롯 A `2026-07-18_15-37-42`: ori 15° + 기존 크기 / 슬롯 B `2026-07-18_15-47-34`:
  크기 확장(1.5~6cm × 길이≤20cm) + ori 없음. env.yaml 스냅샷으로 파라미터 확인.
- 파일 기본값 = 슬롯 A 구성 (크기 확장은 CLI 인자). 스폰 자세는 두 슬롯 모두 고정(월드 정렬).

## 2026-07-18 (밤) — 진행 순서 확정

- 고정 ori+고정 pos (슬롯 A, 도는 중) → 랜덤 position (command ranges만, 싼 단계) →
  랜덤 ori (v2 배관 공사, fresh). 순차 투입 원칙. 슬롯 B는 얇은 상자 파지 확인용 —
  크기-버킷 분해로 판독 예정. 상세는 AGENTS.md 로드맵.

## 2026-07-18 (밤 2) — ori v1 기각, keypoint transport(v1.1) 전환

- ori v1 (판정만): iter 4,230 success 0, 운반 자세 72~93° — 씨앗 없음 + 역방향 강화 확인,
  중지. "cage 파지 무유도로는 자세 안 잡힘" 실측 확보.
- v1.1 구현: transport를 keypoint φ로 교체 (위치+자세 통합, TriFinger 측정 + best-so-far
  일시불 지급). 수학 검증 + 스모크 통과. 재런치는 오버라이드 불필요 (파일 기본값).
- 슬롯 B (크기 확장, 20-56-53)는 계속 진행 중.

## 2026-07-19 — WRAP 라운드: A 청신호, B 파지 실패 원인 수정

- 슬롯 A (WRAP+ori): box_ori_error 첫 하락 (0.94→0.73) — 감싸쥐기가 매달림을 잡기 시작.
- 슬롯 B (스틱): 파지 실패 원인 실측 = hand_floor 2cm 존이 스틱 파지 높이를 벌함 →
  clearance 0.01로 완화 (box_mdp_cfg). goal 마커는 구슬 → 반투명 직육면체 (자세 시각화,
  commands.py). 상세·재런치 명령은 ACTIVITY_2026-07-19.md.

## 2026-07-19 (밤) — 스틱 보류 결정, 슬롯 A ori 완성 패키지 구현

- 스틱: 누르기 파밍 + 말단 토크 포화 실측 → 보류 (복귀 카드: 받침 커리큘럼 + 맞물림 팩터
  + 말단 effort). 슬롯 A(WRAP 42°) 완성 우선으로 노선 확정.
- 패키지 구현: lift_height 0.20 (호버 제거) + ori 커리큘럼 45→30→15° (r_T 씨앗, 신규
  curriculums.py) + 최근접 대칭 상대회전 obs (+3 → 67, 크게 되돌리기 방지). 컴파일 통과,
  ⚠ 스모크 미실시 — 런치 전 필수. 상세는 ACTIVITY_2026-07-19.md.

## 2026-07-20 — one-stick A1 functional grasp 분리 태스크 구현

- `Indy-Wuji-Chopsticks-Grasp`를 Cube-Grasp/Box-Transport와 독립된 scene, MDP, PPO, log 구조로 구성함.
- 고정 `2 x 18 x 2cm` 막대 프록시 한 개에 constraint-based target을 추가함:
  object-relative index grip-region point와 stick-relative palm orientation.
- action 18D, observation 63D. 새 관측은 palm-frame index target error 3D와 relative orientation
  axis-angle error 3D이며 기존 checkpoint와 호환하지 않는 fresh 실험임.
- reward는 signed progress `index_grip`, `hand_stick_orientation`과 기존 cage/hold/clearance lift를
  결합함. tool-ready는 index `<2cm`, relative orientation `<15deg`, cage `>0.3`, clearance `>5cm`를
  15 step 유지하면 terminal reward와 종료를 발생시킴.
- `scripts/debug/chopstick_functional_probe.py`에서 모든 observation source diff 0을 확인함.
  첫 zero step relative orientation error는 3.42deg였음.
- GPU 1-env/1-iteration PPO smoke 통과. run:
  `logs/rsl_rl/indy_wuji_chopstick_acquire/2026-07-20_12-16-28`.
- 다음 검증은 128 env 짧은 fresh run에서 index error 감소, cage/lift 이륙, tool-ready 발생을 분리해 봄.

## 2026-07-20 — 젓가락 RL 전체 단계와 A1 즉시 순서 확정

- 최종 정책 구조를 `ACQUIRE(Skill A) -> TOOL_READY -> USE(Skill B)`로 확정함. Skill B는
  low-level finger opening/stability와 high-level arm/object transport로 분리함.
- 단계 명칭을 혼동 없이 고정함: A0=Box world pose, A1=one-stick canonical functional grasp+lift,
  A1.5=one-stick world ready pose, A2=one-stick position/orientation 강건성, A3=near-state pair 형성,
  A4=table-top two-stick acquisition.
- 현재 A1 코드 사실을 재확인함: fixed `2 x 18 x 2cm` stick, action 18D, observation 63D,
  index target은 tail 쪽 45% local +z 표면의 exact point, orientation target은 reset pre-grasp에서 캡처한
  `q_O_H`임. exact point를 region이라고 부르지 않으며 캡처 target을 canonical grasp라고 간주하지 않음.
- policy action에는 arm+thumb/index/middle만 보이지만 `MimicJointActionCfg`가 middle target을
  ring/little에도 복제함. 약지·새끼 passive 고정으로 해석하지 않음.
- 엄지·검지만 막대에 붙고 바닥을 문지르는 hold local optimum을 막기 위한 첫 단일 변경으로
  hold/lift/tool-ready gate를 `min(thumb-index cage, thumb-middle cage)`로 바꿈. reach는 기존 12-point
  mean cage 유지. 가중치, observation/action, hand-floor는 유지했으며 학습 결과는 아직 미확인임.
- 논문식 index grip region과 thumb grip region은 유효한 후속 후보지만 현재 변경에 섞지 않기로 함.
  balanced cage fresh 결과를 먼저 본 뒤 별도 fresh 실험으로 적용 여부를 판단함.
- A1 즉시 순서: balanced cage 학습 판정 -> 수동/scripted canonical grasp feasibility -> 실제 안정
  파지의 `q_O_H` 측정 -> orientation metric 확인 -> stage-latched best-so-far 양수 orientation reward를
  작은 weight로 fresh 검증. 이 절차를 통과하기 전 world ready pose, 두 번째 stick, Skill B로 넘어가지 않음.

## 2026-07-20 — Semantic anchor/contact/stability 설계 후보 정리

- 손가락 joint pose 전체를 외우게 하지 않고 semantic anchor, 허용 자유도, contact topology, 실제 slip
  stability로 functional grasp를 정의하는 원칙을 유지함.
- 현재 exact index point의 첫 확장 후보는 길이 정규화 surface region임. 예시 axial interval은
  `[-0.60, -0.30]`이며 region 내부 error는 0, 외부는 최근접 경계까지 거리로 계산함. 기존 3D error
  observation 형태는 보존 가능하지만 의미 변경이므로 fresh run임.
- 다음 후보는 thumb opposing surface region임. 현재 index `+z`의 단순 반대 `-z`는 테이블에 막혀 있으므로
  canonical grasp에서 실제 접근 가능한 surface axis/sign을 측정해 정함.
- index와 thumb가 같은 axial 구간에 있으면 pinch는 생기지만 긴 축 토크 지지 폭은 작음. 이후
  middle/palm support를 길이축으로 떨어진 region에 추가하고 필요 시 contact axial span을 사용함.
- hand-stick orientation은 장기적으로 directed long-axis와 optional roll로 분리함. tip-tail 반전을 허용하는
  `abs(dot)`은 사용하지 않으며 square proxy만 roll을 강하게 볼 수 있음.
- slip/stability는 `T_H_S=inv(T_W_H)*T_W_S`의 step 간 translation/rotation drift로 metric부터 측정함.
  단순 world `v_stick-v_palm`은 palm 회전 성분 때문에 오판할 수 있음.
- balanced cage는 virtual SDF geometry이고 grouped physical contact와 다름. 둘은 별도 term/metric으로 유지함.
- 적용 순서: index region -> thumb region -> separated support -> axis/roll -> relative drift metric ->
  stability reward/success. hold 감액과 임의 가중치 제안은 raw 분포 확인 전 적용하지 않음.

## 2026-07-20 — A1 index/thumb surface-region 적용

- balanced cage fresh에서도 lift가 나오지 않아 exact index point를 tail-side surface interval로 완화하고
  thumb functional condition을 추가함.
- index: local `+z`, axial `[-0.60, -0.30]`; thumb: local `+x`, axial `[-0.55, -0.25]`.
  surface normal tolerance는 5mm이며 patch 내부 error는 0임.
- obs 63D -> 66D, reward `thumb_grip=20` 추가, tool-ready에 thumb error `<2cm` 추가. action 18D,
  hold 35, lift 100, orientation 0은 유지함.
- `chopstick_functional_probe.py` 1-env source diff 0, PPO 1-iteration smoke 통과. smoke run:
  `indy_wuji_chopsticks_grasp/2026-07-20_17-05-10`.
- observation 의미가 바뀌었으므로 이전 63D checkpoint resume 금지, fresh 학습만 허용함.

## 2026-07-21 — chopstick lift 보상구조를 Box-Transport에 이식 + 스틱 학습

- 젓가락 lift 성공 구조(검지·엄지 명시 balanced tripod gate = `min(엄지-검지, 엄지-중지)`)를
  `box_mdp_cfg.py`에 이식. 이전 코드는 `box_mdp_cfg backup.py`로 백업.
- 파지 gate 전면 교체: 12-point 평균 → `fg_mdp.balanced_tripod_cage_gate`. 신규 클래스
  `BalancedObjectToGoalProgressReward` / `BalancedObjectOrientationProgressReward` /
  `BalancedObjectAtGoalHeld` / `balanced_object_goal_proximity`.
- 가중치·조건: cube_transport 4000->8000, cube_lift 50->150(lift_height 0.08),
  box_orientation activation_distance 0.10->0.015(latch 10cm->1.5cm), index/thumb_grip 40 신규,
  hand_floor weight 1.0->0.0 / clearance 0.02->0.01, success ori_limit 15° 유지.
- 관측 +6(index/thumb grip error) -> obs 67->73. 박스 기본값 2×2×18cm 스틱로 변경.
- 학습 `2026-07-21_18-30-43`(--resume, obs 73, 스틱). step~966: success 0, max clearance 25cm(잘 듦)
  이나 final 9cm로 하강 + error_pos 12.8cm + transport/ori 보상 ≈0 = 정착 계곡 재현.
- 상세는 `ACTIVITY_2026-07-21.md` 참고.

## 2026-07-22 — Box-Transport 운반+orientation을 Chopsticks로 이식

- chopstick lift 성공 기반 위에 box-transport의 "world goal pose 운반 + 자세 매칭" 이식
  (box에 안 얹고 chopsticks에서 진행). 파일: `chopstick_mdp_cfg.py`, `chopsticks_grasp_env_cfg.py`.
- 이식: `_goal_pose_from_command` + `BalancedObjectToGoalProgressReward`(transport) +
  `BalancedObjectAtGoalHeld`(성공) + 커맨드 `cube_goal`(yaw 45°, +20cm) + obs `stick_to_goal`(3)·
  `stick_ori_to_target`(3) → obs 66→72(fresh 필수).
- 핵심 신규 `BalancedObjectOrientationCoupledReward`: sticky latch 대신 **매 스텝 pos<3cm & gate>0.3
  일 때만** orientation 지급 → goal 벗어나 자세만 맞추는 드리프트 0원. best는 valid 스텝만 갱신(파밍 방지).
- 성공 종료 교체: `tool_ready`(functional grasp) → `success`(world goal pose 도달). grip은 dense로만 유지.
- `cube_lift` weight 4000→150, lift_height 0.08→0.20 (4000 lift 연금이 transport 압도해 호버 방지).
- 리워드 시작값: transport 4000 / orientation 4000 / success 30000(+1000) / lift 150.
- py_compile 통과, 참조 함수 존재 확인. ⚠ 스모크 미실시(학습 중) — 런치 전 4-env smoke 필수.
- 상세는 `ACTIVITY_2026-07-22.md` 참고.

## 2026-07-22 (2) — chopstick orientation (c안) + goal_proximity 이식

- `BalancedObjectOrientationCoupledReward` (c안): `best_error`를 goal 안·밖 **모두** 갱신(지급은 안에서만).
  → 밖에서 회전 개선을 best에 넣었다 재진입 때 인출하는 **왕복 파밍(진동) 차단**. 개선은 goal 안에 머문 채 해야 지급.
- `goal_proximity`(gate×φ 위치 연금) chopstick에 이식, **weight 0**(필요 시 켬, 시작값 150 전후).
- 지표 교훈: ori **평균/max가 목표 아님** → success rate + **15°미만 비율**을 볼 것(평균 낮추기 ≠ 임계 통과).
  min/max 큰 스프레드 = 대칭 아님(메트릭이 최근접 처리), 장축이 goal 방향에서 어긋남 = 파지 복권 + 도달성.
- py_compile 통과, obs 72 유지. ⚠ 스모크 미실시. 상세 `ACTIVITY_2026-07-22.md`, 맥락 `AGENTS.md` 2026-07-22 블록.

## 2026-07-22 (3) — chopstick tip/tail 4-대칭 orientation + [계획] 사각뿔대 프록시

- 젓가락 tip(+y)≠tail(-y) → orientation 대칭을 box의 8개(끝-뒤집기 포함)에서 **길이축 4개**로 좁힘.
  chopstick 전용 `stick_tip_ori_error`/`stick_ori_error_nearest_sym`(+`_STICK_TIP_SYMS` 4개) 신설,
  reward·success·obs가 이걸 씀. box 공유 함수(square_prism 8-대칭)는 그대로. 기존 8-대칭 호출은 주석 보존.
  ⚠ box_ori_error metric은 아직 8-대칭 → chopstick 값이 reward보다 관대. 판정은 success로.
- [계획] 젓가락 프록시를 균일 box → **사각뿔대(frustum)** 로 바꿔 사람이 play에서 tip/tail 구분 가능하게.
  단 box SDF가 박스 전용이라 cage/grip SDF 개편 필요(비자명), 나중 작업. 상세 `ACTIVITY_2026-07-22.md`,
  맥락 `AGENTS.md` 2026-07-22 블록.
- py_compile 통과, obs 72 유지.

## 2026-07-22 (4) — box_ori_error metric도 chopstick 4-대칭으로 맞춤
- `square_prism_ori_error`에 optional `syms` 추가(기본 8, box 무영향) + `_SQUARE_PRISM_Y_SYMS_TIP`(4개).
- managers.py가 orientation 항 이름(box_orientation/stick_orientation)으로 태스크 감지해 box_ori_error를
  8/4-대칭으로 계산. orientation_stage_active도 stick_orientation 읽게 고쳐 chopstick에서 작동.
- box 무영향(기본 8-대칭). 세 파일 py_compile 통과. 상세 ACTIVITY_2026-07-22.md.

## 2026-07-22 (5) — 말단 지수 커널(chopstick) + 페널티 3종(box·chopstick)

- 지수 커널(weight 0): `FineProximityReward`(gate×exp(-d/σ_d), σ_d=2cm) + `FineOrientationReward`
  (gate×exp(-θ/σ_θ), σ_θ=10°, pos<3cm 게이팅 + (c)). 선형 보상의 약한 말단 수렴을 e=0에서 가파른
  Laplacian 커널로 보강. pos·ori 분리(곱커널 X). coarse(φ+lift)는 유지 — 모양의 사다리.
  σ<success 임계로 두는 게 핵심(임계 안쪽 refine), 너무 뾰족하면 sparse-like라 넉넉히 시작.
- 페널티 3종(box·chopstick 양쪽): end_effector_speed -0.001(palm 선속도), action_rate -0.005→-0.001,
  action_second_rate -0.0001(비활성/주석, 함수 TODO broken). ⚠ box도 fresh 필요.
- py_compile 통과. 상세 ACTIVITY_2026-07-22.md.

## 2026-07-22 (6) — action_second_rate 활성화(CustomActionManager 배선) + 15-11-01 비교 기록

- `action_second_rate_l2`는 `prevprev_action` 필요 → 표준 ActionManager엔 없어 broken이었음.
  `rl_task_custom_env.py`의 load_managers에서 표준 ActionManager를 **CustomActionManager로 교체**(배선)해
  prevprev_action 채워지게 함. box·chopstick 양쪽 `action_second_rate` RewTerm 활성화(weight -0.0001).
- 15-11-01 run(8-sym·-45°·페널티/커널 없음, orientation stage_active=0으로 미학습) ↔ 현재 코드(4-sym·(c)·
  fine kernel·페널티 3종·action_manager 배선) 비교를 `ACTIVITY_2026-07-22.md`에 저장. 바꾼 파일 5개 명시.
- py_compile 통과(env·box·chopstick). 15-11-01 결과 대기 중.

## 2026-07-22 (7) — Box-Transport에 SimToolReal keypoint 구현 (chopstick과 A/B)

- SimToolReal Eq.2: d=대칭최소 max 꼭짓점 거리, r=gate×max(d_best−d,0)(best-so-far), 성공 d<eps.
  max라 pos·ori 구조적 결합(한 축 드리프트 불가). box에 구현.
- rewards.py `square_prism_keypoint_goal_distance`에 reduce="max" 추가(기본 mean 무영향).
- box_mdp_cfg: `KeypointMaxGoalProgressReward`+`KeypointMaxAtGoalHeld` 신규. keypoint_goal w8000 활성,
  cube_transport 8000→0·box_orientation 4000→0(복구용 유지), success→KeypointMaxAtGoalHeld(eps 0.05).
- A/B: box=keypoint(pos·ori 융합), chopstick=쿼터니안 4-sym+exp. grasp/penalty/lift 공통, 목표항만 다름.
- ⚠ standard PPO라 탐색 병목 가능(SimToolReal은 SAPG). box 다음 run fresh. py_compile 통과. 상세 ACTIVITY.

## 2026-07-22 (8) — Box keypoint 최종: full-pose 8-corner + roll 4-sym (2점 철회)

- 2점(tail/tip)은 roll 무시라 정사각 스틱(roll 90° 대칭)엔 under-constrained(45° roll 성공 처리) → 철회.
  → 8-corner max 거리 + **square_prism_y_tip(roll 4-대칭, flip 제외)** 로 확정 = 위치+자세(roll 포함) full 매칭.
- rewards.py square_prism_keypoint_goal_distance에 "square_prism_y_tip" 옵션 추가. box success도 8-corner keypoint.
- goal_proximity 150→0(위치-only 연금 파밍 방지). 2점 함수는 원형 스틱용 보존.
- A/B: box=8-corner full-pose keypoint ∥ chopstick=쿼터니안 4-sym+exp. py_compile 통과. 상세 ACTIVITY.

## 2026-07-23 — chopstick middle grip region 추가(사용자) + 07-23 A/B 런 중간 판독

- **사용자가 중지(middle)에도 semantic grip region을 추가함.** 엄지(+x)의 대향면인 local **−x**를
  목표로 둬서 엄지-검지-중지 tripod 전부가 명시적 surface region을 가짐.
  `target_grasp.py`에 `middle_grip_error_b`/`middle_grip_error` 신설, `chopstick_mdp_cfg.py`에
  `MIDDLE_CFG`·`MIDDLE_GRIP_REGION`·관측 `middle_grip_error`(3D)·리워드 `middle_grip`(w40) 추가.
  → **policy obs 72 → 75** (fresh 필수, 체크포인트 actor 입력 75D 실측).
- 기록 누락분 소급 확정(현재 코드 = 라이브 런): `stick_transport` 4000→**6000**(eps 0.05→0.08),
  `cube_lift` 150→**300**(lift_h 0.20→**0.15**), `fine_position` 0→**3000**(σ 0.02→**0.05**),
  `fine_orientation` 0→**1500**(σ 10°→**15°**). ⚠ 커널 σ가 success 임계와 같아짐(원 설계는 σ<임계).
- 런 ① box `2026-07-23_10-07-35`(~2350it): success 0, `keypoint_goal` raw≈0, 보상 대부분 cube_lift,
  clearance 0.235/max 0.451, error_pos **0.430** → 잘 들지만 goal 아닌 쪽으로 감(keypoint 미발화).
- 런 ② chopstick `2026-07-23_12-13-23`(~1150it): success 0, **orientation_stage_active final 0.976**
  (=거의 모든 env가 goal 5cm+gate에 한 번은 진입 → 15-11-01의 위치 병목은 뚫림), ori 평균 21.9°
  (min 1.3°/max 62°), error_pos 평균 0.205(진입 후 이탈), 말단 항(fine_*) raw≈0.
- **판정 공백**: ori<15° 비율 지표가 없어 "전부 22°"인지 "절반 5°/절반 40°"인지 구분 불가.
  다음 1순위는 `env/managers.py`에 15°미만 비율 + (pos<5cm & gate & ori<15°) 동시 충족 비율 추가.
- 상세는 `ACTIVITY_2026-07-23.md`.

## 2026-07-23 (2) — 잔차 액션 term 추가(주석 배선) + 두 런 심층 판독

- **잔차(residual) 액션 term 추가**: `CustomResidualJointPositionAction` / `CustomResidualJointActionCfg`.
  `target = 현재 관절각 + action*scale` (뉴로메카 `JointResidualAction`과 같은 구조).
  **배선은 하지 않고 `env_cfg.py`에 주석 블록으로만** 넣음(도는 학습과 혼동 방지).
  예제와 다른 3곳: ① joint_pos_target 폭 26→action_dim(예제는 부분집합에서 RuntimeError)
  ② env_ids unsqueeze 제거(slice에서 AttributeError) ③ clamp_to_limits 옵션(예제엔 없음).
- 잔차 트레이드오프 기록: **공짜 홀드 상실**(action=0이면 예압 0 → 파지 유지에 지속 액션 필요),
  고정 goal에서는 오히려 불리(절대형은 상수해 가능), action_rate가 가속도 벌점으로 의미 변화.
- **두 런 판독 (chopstick 4300it / box 5750it, 둘 다 success 0)** — 같은 실패 모드:
  - 보상의 **84~95%를 `cube_lift`가 차지**(chopstick 668/798, box 1138/1202), 목표 도달 항은 **0.5~0.8%**.
    lift가 `clearance ≥ lift_height`에서 포화해 **높이만 유지하면 매 스텝 나오는 연금**이 됨.
  - **학습이 진행될수록 목표 지표가 단조 악화**: chopstick error_pos 0.194→0.323,
    ori 23.9°→38.9°, stage_active 0.97→0.79. box error_pos 0.322→0.705, displacement 0.355→0.755.
    보상은 오르는데 목표는 멀어짐 = 보상 구조가 목표와 어긋남.
  - **gate가 활성화 임계에 걸침**: 평균 gate 0.30 vs `activation_gate_threshold` 0.3
    → 목표 관련 항이 절반쯤 잘려 나감.
  - A/B(keypoint vs 쿼터니안)는 **아직 유효하지 않음** — 둘 다 lift 연금에 지배당함.
    단 keypoint raw는 5750it 내내 0.0001 고정(순진행 0).
- 다음 조치 후보: ① `cube_lift` 연금 구조 해체(가중치 인하 또는 일회성화) ② gate 임계 정합
  ③ exp 커널은 현재 무의미(d≈0.32에서 exp(-d/0.05)≈0.002) ④ 그 뒤에야 잔차/도달성 판독 가능.
- 상세는 `ACTIVITY_2026-07-23.md`.

## 2026-07-23 (3) — 액션 잔차 전환 + 커플링 해제 + 가중치 전면 재조정 (사용자 적용)

- **chopstick 액션**: 절대형 → **잔차(residual)** (`CustomResidualJointPositionAction`,
  `target = 현재 관절각 + action*scale`, scale 팔 0.3 / 손가락 0.3) + **약지·새끼 커플링 해제**
  (`finger[1-5]`, action 18→26). **policy obs 75 → 91** — joint_pos(18→26)와
  `action_history`(=prev_action, 18→26)가 **둘 다** 커져서 +16. (83으로 잘못 세기 쉬움)
  `WUJI_LEGACY_ACTION=1`로 옛 구성(action 18/obs 75) 복원 → 07-23 이전 체크포인트 play용.
- **가중치 근거 2가지 (구조적 발견)**:
  ① **gate형 weight 1 ≈ progress형 weight 300.** gate형은 240스텝 지급(`2.4W`),
     progress형은 best-so-far라 에피소드당 1회분 예산(`Δ×0.3×W/30`). transport 6000(48점)이
     lift 150(342점)보다 작았던 이유. **weight 숫자를 항 간 직접 비교 금지.**
  ② **lift는 W가 연금 총액, W/h가 바닥에서 떼는 힘.** (150,0.20)의 기울기 7.5/m는
     "못 들고 테이블에서 비빔"의 원인. 07-20 A1의 (100,0.08)=12.5/m가 잘 되던 값.
- **chopstick**: cube_lift 300/0.15 → **100/0.10**, stick_transport 6000 → **30000**,
  stick_orientation 4000 → **30000**, goal_proximity 0 → **75**(transport가 best-so-far라
  머무는 힘이 0인 것 보완), transport_success 30000 → **60000(+2000)**(dense 5배 상향으로
  success가 상대 강등되는 것 복원). fine_* 는 의도적으로 유지(transport 30000이면 말단에서
  φ가 지수커널보다 이미 가파름).
- **box**: keypoint의 `d`가 정규화 안 된 **미터**라 초기 Δd 0.162 → 예산 = 0.00162W.
  8000이면 **13점**뿐(vs lift 500 = 1200점, 92배). keypoint_goal 8000 → **150000**,
  cube_lift 500/0.10 → **150/0.20**, transport_success → **60000**, `middle_grip` 40 신규
  (obs 73 → **76**). goal_proximity는 0 유지(위치-only 파밍이 keypoint 융합을 깎음).
- **대향(opposition) gate는 구현했다가 철회** — grip 자세는 Skill A 필수가 아니라는 사용자 판단.
  "얹어 나르기"(opposition −0.28, cage_span 9.7cm)는 미해결로 남음. 잔차 전환으로 손가락
  dead zone이 사라져 자연 개선될지가 관전 포인트.
- ⚠ 두 태스크가 액션·목표항·lift 모두 달라 **단일변수 A/B 아님**. 패키지 비교로만 읽을 것.
- ⚠ 스모크 미실시. fresh 런치 전 4-env로 action 26 / obs 91 확인 필수.
- 상세는 `ACTIVITY_2026-07-23.md`.

## 2026-07-24 — 밤샘 run 판독 + box도 잔차 전환

- **밤샘 fresh 두 런 (chopstick 잔차 obs91 / box 절대형 obs76)** 판독 (chopstick 4450it, box 3650it):
  - **chopstick 정상 학습, success 0 → 0.0074 (첫 성공).** lift는 됨(max clearance 0.20)이나 유지
    실패(final 0.047). 파지 개선 뚜렷: thumb_index_opposition −0.41 → **+0.33**(같은 쪽→마주봄),
    cage_inside 0.19, action_track_err 0.43, box_ori_error 1.44→0.86. **잔차 전환이 먹힘.**
  - **box 파지 실패.** clearance max 0.24 / final −0.004(거의 못 듦). action_track_err **1.51**,
    opposition **−0.20**(같은 쪽), cage_inside 0.08, keypoint raw 0.0002(gate 뒤라 0원).
    원인 = 절대형 액션의 dead zone(손가락 목표 34%가 limit 밖). **가중치가 아니라 액션 문제.**
- **결정적 관찰**: 파지 성패를 가른 건 keypoint vs 쿼터니안이 아니라 **액션 구조**. 동일 스틱·
  동일 사다리에서 잔차 쓴 chopstick만 파지 성공. 목표항 A/B는 둘 다 파지된 뒤에야 의미 있음.
- → **box도 잔차 + 커플링 해제로 전환** (`grasp/indy_wuji_box/env_cfg.py`, chopstick과 동일).
  scale 팔0.3/손0.3, **action 18→26, obs 76→92**. WUJI_LEGACY_ACTION=1 스위치.
  box 보상은 그대로(keypoint 150000, lift 150/0.20, success 60000). fresh 필수, 스모크 미실시.
- box 런 23-52-08은 판독 후 종료·폴더 삭제(잔차 재런치 준비).
- 상세 `ACTIVITY_2026-07-23.md`.

## 2026-07-24 — Phase 계획 정리 (사용자 설계)

- **Phase 1 (현재)**: 두 젓가락 획득 → functional grasp 형성. 내부 3단계 STATE A(획득+staging)
  → B(standard grip 형성) → C(open-close). 단계 전환은 latch(0.5~1s 연속 만족, 이후 이전 reward
  끄고 bonus 1회). 구현은 **방식 2(단계별 별도 환경/curriculum) 먼저 → 나중에 FSM 결합**.
  topology 가설: 엄지·검지·중지→Stick1, 약지→Stick2, 새끼 보조(스타일 A/B/C 비교).
  BO는 손가락별 axial center만 sweep(surface_axis/sign 고정).
- **Phase 2 (나중)**: 젓가락으로 물체 집어 **큰 목표 공간(바구니)에 놓기**. ⚠ 정밀 pick&place 아님 —
  자세 무관, 목표항은 위치 φ만. 병목은 tip 파지 + open-close 유지(keypoint/자세매칭 불필요).
- 진행 순서: 1-stick 고정 pose 수렴 → **랜덤 pose 도달성 검증(2-stick 전제)** → STATE A → B → C.
- 참고 논문: SimToolReal(keypoint), Dexterous Pre-grasp, Learning to Use Chopsticks(2205.14313).
- 상세는 `ACTIVITY_2026-07-23.md` "Phase 계획", 연구방향 맥락은 `study.md`.

## 2026-07-24 (2) — euler 축 분석 / 랜덤 pose / box keypoint 진행 / 분업

- **goal euler 축 확인(계산)**: 스틱 장축=y. **pitch(y)만 장축 둘레=90° 4-대칭**, roll(x)·yaw(z)는
  tip 방향을 바꿔 **대칭 없음**(젓가락 0°↔180° 정반대). 이전 "yaw 4-대칭" 기록은 오류 정정.
  범위: pitch만 [0,90]이 전체, roll·yaw는 비대칭이라 넓게(양쪽 원하면 마이너스).
- **chopstick 랜덤 pose(fresh)**: roll=(0,180°), yaw=(0,180°), pitch 고정. roll 의도적 개방
  = 스틱 세운 자세까지 포함(수평 전제 tripod엔 부담). roll·yaw 동시라 원인분리 제한.
- **box keypoint 잔차(09-43-43, it2282)**: success it1500 0.002→it2282 **0.057**(지수 초입,
  잔차가 box에서도 먹힘). ⚠ 그런데 keypoint_goal raw 내내 **0.0002로 죽어있음** — 목표 도달하는데
  도달 보상 0 = keypoint가 학습 견인 못 하고 얹혀만 있을 가능성(A/B 핵심질문에 부정 초기신호).
  3000~4000it까지 보고 판정. keypoint distance 로깅 추가 검토(무영향).
- **분업**: chopstick=랜덤 pose 진행 / box=변수 적으니 **중지 해결 테스트베드**(keypoint 잘 되면).
  중지 해법 후보: middle_grip을 per-step 페널티 -w·d(논문 r_contact, 유력) / cage 실제접촉 요구 / 면 스왑.
  현상: 중지 표면 10.6cm 밖, cage 선분만 통과(실파지는 엄지-검지 2점).
- 상세 `ACTIVITY_2026-07-23.md`.

## 2026-07-24 (3) — 재설계·재샘플·success분리·box준비·obs지적

- **grip region 재설계(chopstick, fresh)**: 엄지 목표(+x)와 실제 파지(−x)가 반대였음 규명(보상 약해
  자세상 편한 −x로 감, box도 −x라 일치). **tip/tail 뒤집기**(axial 음수→양수, 파지=+y 끝, 로봇에 8cm
  가까움. 좌표축 무변경 라벨만, orientation은 y축 180° 대칭이 흡수) + **palm-down 배치**(엄지−x/중지+x/검지+z).
  복구용 09-42-28 버전 주석 보존. ⚠ 검지 +z는 palm-up이면 구조적 도달 불가 → 근본은 palm-down 강제(4번).
- **goal euler 축(계산)**: pitch(장축y)만 90° 4-대칭, roll·yaw는 대칭 없음(tip≠tail). 이전 "yaw 4-대칭" 오류 정정.
- **에피소드 내 goal 재샘플 + success 분리(chopstick, 사수님)**: resampling (1e9)→**(5,10)**.
  `_goal_changed` 신설, best-so-far 4개가 goal 변화 감지해 기준선 재장전. **success termination 제거 →
  `GoalReachedBonus`(비종료, goal당 1회)**. timeout 8초 유지(빼면 Manager-Based는 무한 에피소드).
  ⚠ Episode_Termination/success 사라짐→transport_success raw로 봄. 스모크 미실시.
- **box 재샘플 준비**: `_goal_changed` 복제 + keypoint goal 캐시 배선(고정 goal이면 무영향) +
  `KeypointGoalReachedBonus` 정의. 랜덤 런 시 나머지 배선.
- **box keypoint(09-43-43) 잘 됨**: success it3713 **0.59** 지수 상승 중, 예산 keypoint 74% 지배
  ("raw 죽었다" 오판 정정: raw×weight=37점 최대). 4500~5500 포화 예상, 미정지.
- **obs oracle 문제(사수님)**: error 계열(grip_error 9, ori_to_target 3, hand_stick_ori 3)이 시뮬 전용 =
  sim-to-real 불가. 지금은 태스크 성립 확인이라 유지, 실물 전이 시 raw로 교체 or asymmetric critic. **별도 단계.**
- 상세 `ACTIVITY_2026-07-23.md`.

## 2026-07-24 (4) — box success 분리 배선 + chopstick 랜덤런 판독 + obs 분석

- **box도 success 분리 배선**: transport_success→KeypointGoalReachedBonus(비종료, goal당 1회),
  success termination 제거, keypoint goal 캐시 배선. ranges/resampling은 랜덤런 시(고정 09-43-43 기준선
  유지 위해 지금은 안 엶). 도는 run 무영향.
- **chopstick 새 런(15-35-46) 판독**: 파지는 palm-down 재설계로 **더 빠름**(엄지·검지 iter1050에 붙음,
  surface 0.008/0.009 vs 고정런 0.12/0.16). 근데 **box_ori_error 91°**(고정 2°) — 병목은 파지 아니라
  **랜덤 pose 자세**. roll 0~180° 과할 수 있음 → iter2500 추세 보고 축소 판단. lift 못 하는 건 자세 탓.
- **obs 분석(사수님 지적 답)**: 모든 obs = joint_pos(엔코더)·body pose(FK)·**스틱 pose(비전)** 조합.
  스틱 **자세** 의존 obs(grip_error 9·ori_to_target 3·hand_stick_ori 3)가 sim-to-real 병목(얇은 대칭 막대
  6D pose 추정 난). 논문이 error를 actor에 안 넣는 이유: 실물 재현불가+raw로 계산가능+정답힌트 과의존.
  단 goal-relative(stick_to_goal)는 논문도 씀. 자세의존 error는 빼거나 asymmetric critic으로.
  죽은 obs: stick_size(상수), hand_stick_ori(reward 0) — 실물 obs 정리 단계에서 일괄.
- **약지·새끼**: 커플링 해제 후 목표 없어 뜸 → 1스틱은 접어두기(A), 2스틱서 약지 Stick2 목표(미구현).
- 상세 `ACTIVITY_2026-07-24.md`.

## 2026-07-24 (5) — 재샘플 7D key 버그수정 + palm-down 철회

- **goal 재샘플 감지 버그**: `_goal_changed`가 position만 비교했는데 goal은 position 고정+ori만 랜덤 →
  ori 재샘플 못 잡음. GoalReachedBonus/KeypointGoalReachedBonus가 _awarded/_count 리셋 못 해 "goal당
  1회 보너스"가 반쪽 작동. **수정: `_goal_key`(7D pose=pos3+quat4)로 7곳 전부 통일.** 지금 도는 런은
  버그 있는 채 돎(보너스가 에피소드당 1회로 짜게, lift 무관), 수정본은 다음 fresh부터.
  stale 주석 정정(obs 72→91, yaw 4-대칭→pitch 4-대칭).
- **palm-down 철회**: play 진단 결과 재설계 파지가 **중지 벌린 채 2점 pinch** — 원하던 tripod 안 나옴
  (weight 40 약해 정책이 목표 무시 + 검지 +z는 palm-up 도달불가). **grip region 09-42-28(검증 파지)로
  복원**, tip/tail 뒤집기·엄지중지 스왑 철회. palm-down 값은 복구용 주석. 방침: 파지 고정+**랜덤 pose만 도전.**
- 다음 fresh: 09-42-28 파지 + 랜덤 pose(roll/yaw) + 재샘플(7D key) + success분리 + 잔차(obs 91). 스모크 필수.
- 상세 `ACTIVITY_2026-07-24.md`.

## 2026-07-25 — A/B 결론(keypoint) + Phase 1 주먹(wrap) 파지 구조 적용

- **A/B(같은 랜덤 pose)**: box(keypoint) ori min 15°/mean 53° 하강 vs chopstick(쿼터니안) min 34°/mean 87° 벽.
  → 랜덤 pose에선 keypoint 우세(pos·ori 융합, 신호 단절 없음). **자세 목표항 keypoint 채택.**
- **grip 교정**: play서 창발 파지가 엄지−x/중지+x라 region을 거기 맞춤. 새 런(08-47-53) iter 200에
  파지 형성(옛 런 4000 대비 ~3800it 빠름). lift는 마지막 단계라 아직(정상).
- **파지 진단**: grip 보상(progress ~1.3점) ≪ cube_lift(gate 240~800점). 자세는 lift×gate 결합으로만
  형성 → 느슨-lift 위험. **드롭 유발 시에만** gate^k 개입.
- **Phase 1 재정의**: 주먹(5-finger wrap) 파지 + transport + palm-up. wrap이 목표 자체(tool은 Phase 2).
- **코드 적용(주먹 파지)**: 엄지 −x, 나머지 4손가락 +x(반대면). `balanced_penta_cage_gate` +
  `object_lift_in_balanced_penta_cage`(gate^k 노브) 신설. chopstick regions INDEX +z→+x, RING/PINKY
  추가(axial stagger), ring_grip/pinky_grip term, hold·lift를 penta 게이트로. **백업
  chopstick_mdp_cfg.py.tripod_backup_2026-07-25**. py_compile OK, **스모크는 학습 종료 후.**
- **성공-래치 커리큘럼 수식**: `L←max(L,s1)`, `R=(1−L)Rp1+L·Rp2`. 연속 함수=fresh-run 준수.
- 상세 `ACTIVITY_2026-07-25.md`, 설계 `study.md`(2026-07-25 절).

## 2026-07-25 (정정·전환) — keypoint 재평가 + wrap을 box로 이동

- **정정**: 앞서 "keypoint 정체/불필요" 과결론 철회. 09-43-43(box, keypoint, **고정 pose**)은
  err_pos 4.3cm·ori min 3°·success 상승으로 **잘 됐음**. 00-54-01 정체는 **랜덤 pose(1.57~3.14)** 탓,
  방법 아님. chopstick "34° 벽"도 grip 오설정 아티팩트(엄지·중지 x부호)였음.
- **판단**: 랜덤 자세매칭은 Phase1(palm-up 고정)·Phase2(위치만) 어디도 요구 안 함 → 실제 과제는
  keypoint 강한 고정-pose 영역. **keypoint(box) 주력 + 파지 주먹(wrap)**, chopstick은 tripod로 park.
- **코드**: chopstick_mdp_cfg.py → tripod_backup 복원(wrap 철회). box_mdp_cfg.py에 penta wrap 적용
  (ring/pinky cage·region·grip term, _INDEX +z→+x, hold·lift penta, gate_exponent=1.0).
  백업 box_mdp_cfg.py.keypoint_backup_2026-07-25. py_compile OK, **box 스모크 후 fresh**.
- 미결: box를 랜덤 pose 유지할지 실제 요구(고정 palm-up)로 바꿀지.
- 상세 ACTIVITY_2026-07-25.md, 설계 study.md(2026-07-25 절).

- **box goal 고정 확정**: 주먹+keypoint(위치·자세 통합), cube_goal roll/pitch 0·yaw 0.785398 고정,
  resample 1e9. 09-43-43 검증 방식. 랜덤 철회. box_mdp_cfg.py.keypoint_backup에 직전 랜덤값 보존.

## 2026-07-26 — penta(5손가락) 부트스트랩 실패 → quad(4손가락) 축소

- box 21-44-48(penta wrap): penta 게이트가 iter 300 후 하락(0.002→0.0005), lift raw ≈0. 새끼(little)가
  안 붙어(0.088→0.095 멀어짐) min을 0으로 끔. 구조적 실패(닭-달걀 + 새끼 대향 곤란 + 바닥물체 물리).
  기다려도 안 풀리는 것으로 판단.
- 대응(사용자): penta → **quad(엄지+검지+중지+약지)**. 새끼는 게이트에서 빼고 pinky_grip으로만 유도.
- 코드: rewards.py에 _min_cage_gate 헬퍼 + balanced_quad_cage_gate + object_lift_in_balanced_quad_cage
  추가. box finger_cage_hold·cube_lift를 quad로(pinky_cage_cfg 제거). penta 보존. fresh 00-04-17 quad 시작.
- 판별: iter 1000~2000에 finger_cage_hold(quad 게이트) 상승 반전 여부.
- 상세 ACTIVITY_2026-07-26.md.

## 2026-07-27 — `hand_grasp` collision/valley probe

- `scripts/debug/hand_grasp_collision_probe.py` 추가.
- policy 없이 1 env에서 stick-pair 닫힘 각도와 Stick2 valley 위치를 sweep함.
- contact sensor는 실행 중 임시 env cfg에만 추가하는 probe 측정 전용이며 active 학습 cfg는 무변경.
- 실행마다 `logs/debug/hand_grasp_collision_probe/<timestamp>/`에 `probe.log`, config/summary JSON,
  pair/valley CSV를 자동 저장함.
- 사용자가 시뮬레이션을 실행하고 완료를 알리면 최신 결과 파일을 판독함.
- 시뮬레이션은 실행하지 않았고 `py_compile`과 `git diff --check`만 통과함.
## 2026-07-27 — hand_grasp STATE B residual RL

- `2026-07-27_23-15-18` ring joint2 sweep: ring 0/512, other5 440/512, axial miss 0,
  cross-section miss 16.6 mm. stick length/joint2-only 해법을 배제함.
- `hand_grasp` reset을 canonical candidate 6으로 전환함.
- active topology는 thumb/index/middle tip→Stick1, palm/thumb-mid→Stick2,
  ring tip-or-link4→Stick2임.
- anchor 5개를 개별 reward(weight 1씩)로 기록하고 ring/full-stability/success reward와
  1초 success termination을 추가함. anchor 총합은 기존 5×평균과 동일함.
- TensorBoard `Metrics/hand_grasp{,_final,_min,_max}/*`에 개별 접촉력, ring OR,
  contact count/full/quiet gate, stick 속도와 success stable step을 추가함.
- obs/action은 101D/20D 그대로. py_compile 및 정적 diff check 통과, 시뮬 실행은 하지 않음.
- 최초 4096-env reset의 `(4096,) env_ids × (20,) joint_ids` history-buffer indexing 오류를
  `FiniteArticulation`의 env outer-index 확장으로 수정함.

## 2026-07-28 — hand_grasp 6-contact 결합형 reward

- `23-43-11` run은 ring 접촉 약 1.09 N을 얻는 대신 middle→Stick1을 잃은 5/6 해로 수렴했고
  success/quiet/stable-step은 5166 iter까지 0이었음.
- ring support weight 30을 다른 anchor 5개가 모두 0.02 N 이상일 때만 지급하도록 결합함.
- middle surface proximity weight 5, 6-contact hard-gated stability,
  각속도 3 rad/s 초과분 L2 penalty -0.1을 추가함. obs/action 101D/20D 유지, fresh run 필요.

## 2026-07-28 — pose_005 STATE B 유지 학습

- 수동 완성 파지 `2026-07-28_12-39-52/pose_005.json`의 joint actual/PD target과 두 stick
  palm-local pose를 active reset으로 적용함.
- 20D action을 `pose_005 PD target + action×0.1 rad` reference residual로 바꿈. zero action이
  수동 preload를 유지하고 정책은 주변 미세 보정만 수행함.
- acquisition용 middle proximity/ring-coupled reward를 reference joint·두 stick pose,
  6-contact minimum, full-contact quiet stability 중심의 유지 reward로 교체함.
- success에 두 stick reference pose 2 cm/20 deg gate를 추가하고 TensorBoard에 pose error와
  reference-pose-valid metric을 추가함. obs 101D/action 20D 유지, fresh run 필요.
- Isaac은 실행하지 않았고 `py_compile` 정적 검증을 통과함.

## 2026-07-28 — hand_grasp 엄지 distal contact 교정

- GUI에서 Stick1을 실제로 누르는 엄지 부위가 tip이 아니라 tip 바로 아래 마디임을 확인함.
- URDF 체인 기준 해당 body인 `finger1_link4`로 active sensor를 옮기고 semantic 이름을
  `thumb_distal_stick1`로 변경함.
- Stick2의 `finger1_link2` thumb-middle anchor와 과거 CEM 결과 파일은 변경하지 않음.
- 후속 전-link PhysX pair probe에서 실제 Stick1 pair는 `finger1_link3`로 확인되어 sensor path를
  link4에서 link3로 정정함. palm↔Stick2도 240/240, 평균 2.242 N이므로 필수 anchor로 유지함.
- ring의 tip-or-link4 완화는 제거하고 실제 240/240 접촉한 `finger4_tip_link↔Stick2`만 필수로
  확정함. active contact sensor와 semantic group은 정확히 6개임.

## 2026-07-28 — hand_grasp OPEN/CLOSE mode-conditioned reward

- 50:50 per-episode OPEN/CLOSE one-hot command를 추가해 policy observation을 101D에서 103D로 변경함.
- `pose_005`의 local `+y` distal endpoint를 tip으로 정의하고 surface gap target을
  OPEN 0.020 m, CLOSE 0.003 m, mode gap 성공 허용오차를 0.003 m로 설정함.
- Stick1 full pose reward를 in-hand pivot tracking으로 교체하고 Stick2 full pose anchor는 유지함.
- mode별 tip-gap reward와 mode-gap에 결합된 6-contact quiet stability를 적용해 CLOSE에서
  OPEN 정지 자세의 reward farming을 차단함.
- success/metric도 mode gap, Stick1 pivot, Stick2 anchor를 사용하도록 변경함.
- `16-31-57` resolve smoke와 `16-32-21` 1-env/1-iteration smoke가 통과함. fresh run 필요.

## 2026-07-29 — hand_grasp 연속 OPEN/CLOSE와 gap geometry 교정

- old-gap 2 cm run `21-54-22`은 iter 1744 부근이 가장 좋았고 후반에는 quiet/stability가 퇴화함.
- OPEN 3 cm fresh `00-04-22`은 6접촉을 유지하면서 CLOSE와 저속 정착을 포기해 실패했으므로
  OPEN target을 20 mm로 복원함.
- `_tip_surface_gap()`을 endpoint 3D norm에서 7 mm를 빼는 식에서, 공통 장축을 제거한 횡단거리와
  현재 square cross-section 회전의 projected support radius 두 개를 빼는 식으로 변경함.
- `pose_005` reference gap은 old `16.322 mm`에서 corrected `13.310 mm`로 바뀜.
- episode를 30초로 늘리고 command를 3초마다 OPEN/CLOSE 반전시킴. success는 mode마다 한 번
  reward로 주되 종료하지 않고, stick drop 또는 timeout에서만 reset함.
- corrected-gap fresh `11-14-30` 최신 iter 1214는 OPEN/CLOSE raw `0.301/0.287`,
  final gap error `3.28 mm`, full contact `98.0%`, drop `0`, success raw `1.18e-4`임.
- Stick1 pivot과 Stick2 pose 및 여섯 contact는 유지되며, 현재 병목은 각속도 `4.15 rad/s`,
  quiet-valid `9.5%`로 나타나는 mode 전환 뒤 감속/정착임. play에서 OPEN은 나오지만 느림.
- 설정을 iter 1500~2000까지 유지하고, 전환 지연이 계속되면 다음 fresh에서 dwell 4~6초
  curriculum과 mode별 transition-time metric을 검토함.
- 다음 물체 파지 전에 hand/root만 움직이는 scripted motion으로 empty-stick 파지의 이동/회전
  강건성을 먼저 검증함. 상세는 root `ACTIVITY_2026-07-29.md`와 `AGENTS.md`에 기록함.

## 2026-07-29 — hand_grasp Stick2-axis gap

- Stick2를 고정 reference rail, Stick1을 OPEN/CLOSE 이동 stick으로 보는 task 의미에 맞춰
  `_tip_surface_gap()`의 축 투영을 평균 공통축에서 현재 Stick2 local `+y` 장축으로 변경함.
- 사각 단면 projected support radius와 나머지 학습 설정은 유지하고, Stick1 장축 계산 및
  sign-align/average/normalize 연산만 제거함.
- 공통축 run `11-14-30`은 baseline으로 보존하며 새 gap 의미에서는 checkpoint resume 없이
  fresh run을 시작함.

## 2026-07-29 — chopsticks_grasp 1cm: maintained gate 결합 배선
- 1cm 파지 실패 진단: 2cm 성공은 loose-cage(손끝 8cm), 손끝배치 보상이 약한 sum이라 검지 안 감(8.8cm).
- 조치: finger_cage_hold·cube_lift만 maintained_tripod gate(cage×손끝 proximity)로. 백업 pre_maintained_1cm_backup_2026-07-29.
- 판독(12-45-55): 작동 — thumb 0.8/index 2.8cm로 들어옴. 근데 중지 5.1cm가 min 병목→게이트 하드0.
- 조치2: proximity_far 0.05→0.08 (소프트게이트). fresh 대기. 상세 ACTIVITY_2026-07-29.md.

## 2026-07-29 — hand_grasp current-state residual A/B

- OPEN/CLOSE episode/dwell을 `10 s/5 s`로 설정해 mode 전환을 episode당 한 번 학습함.
- `ReferenceResidualJointActionCfg` 대신 기존 `CustomResidualJointActionCfg`를 활성화함:
  `target=current_joint_position+action×0.1`.
- reset PD target도 `pose_005` actual joint position과 같게 두어 수동 preload를 제거함.
- `PREGRASP_JOINT_TARGETS`와 이전 reference action/reset 배선은 재현용 주석으로 보존함.
- obs/action은 `103D/20D` 유지. 의미가 다른 action이므로 기존 checkpoint resume 금지.

## 2026-07-29 (저녁) — chopstick obs 경량화 (sim-to-real oracle 제거)
- grip_error(9) 제거(oracle). stick_pos·stick_in_fingertips → palm-local(신설 object_position_relative_to_bodies_local,
  box/cube 안 건드림, 회전불변+egocentric). B(hand_stick_orientation obs+event+reward) 제거(리셋 우연자세 목표=arbitrary).
  obs 91→79. 보상은 안 끔. 백업 pre_grip_error_removal/pre_B_removal_2026-07-29.
- 미결 #7 stick_ori_to_target: 못 얻는 게 아니라 미리 계산한 error라, raw(스틱 자세+goal 자세)로 교체 예정. 상세 ACTIVITY_2026-07-29.md.

## 2026-07-29 — hand_grasp 각속도 페널티와 palm-relative 속도

- no-angular-penalty run `17-25-14`가 penalty run `16-24-00`보다 낫지 않아 기존
  `-0.1 * clip(max_angular_speed - 3, 0, 10)^2` 항을 다시 활성화함.
- 파지 안정성의 의미를 손 안 slip으로 맞추기 위해 독립 penalty, mode stability, success 속도 gate,
  TensorBoard speed/quiet-valid를 world 속도에서 palm-relative 속도로 변경함.
- `ω_rel=ω_stick-ω_palm`,
  `v_rel=v_stick-v_palm-ω_palm×(p_stick-p_palm)`.
- palm 고정인 현재 OPEN/CLOSE에서는 기존 수치와 거의 같고, floating hand에서는 hand와 함께
  움직이는 stick은 감점하지 않되 손을 못 따라오거나 내부에서 미끄러지는 stick은 감점함.
- 실행 중 run에는 반영되지 않고 fresh run부터 적용됨. obs/action은 `103D/20D` 그대로임.

## 2026-07-29 — hand_grasp distal tip lateral gate

- radial tip gap은 맞지만 Stick1 tip이 Stick2 옆으로 빗겨나는 play 실패를 막기 위해
  `pose_005`의 Stick2-local x-z separation 방향을 기준으로 lateral error를 계산함.
- OPEN/CLOSE tracking과 mode stability에 `exp(-lateral_error/0.005)`를 곱하고,
  success에는 `lateral_error <= 0.005 m`를 hard gate로 추가함.
- `Metrics/hand_grasp{,_final,_min,_max}/tip_lateral_error`를 추가함.
- diagnostic의 `tip_surface_gap`도 예전 3D norm 식에서 active Stick2-axis square-section helper로
  교체해 reward/success/metric 정의를 일치시킴.
- obs/action은 `103D/20D` 불변이나 reward/success 의미가 달라 다음 학습은 fresh run임.

## 2026-07-29 — hand_grasp_object scene scaffold

- 현재 `hand_grasp` package와 `hand_grasp_*.py` debug tools를
  `backups/hand_grasp_pre_object_env_2026-07-29/`에 복구용으로 백업함.
- 기존 OPEN/CLOSE task를 유지하면서 물체 파지용 `hand_grasp_object` task를 별도로 등록함.
- hand와 두 stick의 검증된 상대 자세를 통째로 world x축 약 `-60.16°` 회전해 평균 stick
  long axis를 수평으로 만듦. fixed hand root와 기존 `pose_005` reset은 유지함.
- 두 distal endpoint의 reference midpoint에 `10 mm`, `2 g` dynamic cube를 reset함.
  바로 아래에는 `6 x 6 x 500 mm` kinematic post를 두고 top face를 cube bottom에 맞춤.
- object/support는 scene와 reset에만 추가함. object observation, reward, termination,
  contact sensor, root action은 아직 없음. 기존 MDP는 action `20D`, obs `103D` 그대로임.
- `hand_grasp_object --num_envs 1 --max_iterations 1`에서 scene/event resolve와 24 step 통과함.

## 2026-07-30 — hand_grasp 6-contact gate + collapse termination

- `21-05-32` run은 1500 iter의 contact `5.97/6`, full-contact `96.9%`에서
  3100 iter 이후 약 `3/6`, success `0`으로 policy가 붕괴함. gap/pivot/pose/총 reward도
  같이 감소해 3-contact reward farming으로 판정하지 않음.
- 변경 전 `mdp.py`, `hand_grasp_env_cfg.py`, `rsl_rl_cfg.py`는
  `backups/hand_grasp_pre_contact_gate_2026-07-30/`에 백업함.
- contact reward를 force 0.02 N 포화 기준의 `mean` weight 10 + `min` weight 40으로 구성함.
  OPEN/CLOSE gap+lateral reward에는 여섯 group의 smooth minimum contact gate를 곱함.
  reference/pivot/Stick2 pose는 contact 복구용 ungated 신호로 유지함.
- 6/6을 0.02 N 이상 5 step 획득한 뒤, 0.01 N 이상 접촉이 4개 이하로 10 step
  지속될 때만 `functional_contact_lost` termination을 발생시킴. 5/6은 복구 가능함.
- success·entropy·action·obs는 변경하지 않음. 1-env smoke
  `2026-07-30_09-32-09`에서 reward 11개/termination 5개 resolve 및 24 step 통과함.

## 2026-07-30 — hand_grasp gap/lateral reward 분리

- 기존 OPEN/CLOSE 항의 `contact*exp(-gap/5mm-lateral/5mm)` 결합은 어느 한 축이 나쁘면
  다른 축의 shaping까지 죽이고 로그 원인도 분리하지 못해 폐기함.
- backup: `backups/hand_grasp_pre_lateral_split_2026-07-30/`.
- active gap reward는 Stick2 장축을 제거한 transverse surface gap만 평가:
  `contact_gate*exp(-gap_error/5mm)`.
- lateral은 contact/mode 독립 bounded penalty
  `-5*(1-exp(-lateral_error/5mm))`로 분리함. mode stability의 곱 gate와 success의
  lateral `<=5mm` 조건은 유지함.
- smoke `2026-07-30_10-28-30`: reward 12개, termination 5개, 24 step 통과.

## 2026-07-30 — hand_grasp Stick2 anchor 강화

- `10-31-54` 초반(~330it)은 ring force `0.737 N`인데도 Stick2 error가
  `9.72 mm/33.3°`, palm-relative angular speed `6.45 rad/s`라 약지 힘보다
  Stick2 이동·회전이 실제 병목이었음.
- backup: `backups/hand_grasp_pre_stick2_anchor_2026-07-30/`.
- positive Stick2 pose reward 15를 제거하고 bounded position/orientation penalty를 각각
  weight `-10`, scale `5 mm/10°`로 분리함.
- gap reward와 mode stability에 `exp(-e_pos/5mm-e_ori/10°)` anchor gate를 추가함.
- success Stick2 limit도 `20 mm/20°`에서 `5 mm/10°`로 강화함.
  Stick2는 계속 dynamic이고 fixed로 바꾸지 않음.
- mode dwell/episode `5 s/10 s`, obs/action `103D/20D`, entropy `0.001`,
  6-contact latch는 유지함.
- smoke `2026-07-30_11-25-34`: reward 13개, termination 5개, 24 step 통과.

## 2026-07-30 — `21-05-32` exact baseline active 복원

- contact 6/6이 2450it 부근까지 유지된 `2026-07-29_21-05-32`를 그대로 fresh 재학습하고
  좋은 시점에서 수동 중단하기로 함.
- 직전 강화판은 `backups/hand_grasp_stick2_anchor_2026-07-30/`에 보존함.
- 저장 YAML 기준 episode/dwell은 `10 s/2 s`; 과거 5초 주석이 아니라 이 값을 복원함.
- reward 10개와 termination 4개를 원 run과 동일하게 복원:
  Stick2 positive pose 15, gap+lateral 결합형, contact min 20/scale 0.10,
  후속 contact/anchor/lateral 분리 항과 collapse termination은 비활성.
- CLOSE `3 mm`, gap tolerance `±3 mm`도 exact baseline 유지를 위해 그대로 둠.
- smoke `2026-07-30_12-13-39` 통과. 생성 YAML은 env count/iteration/log path 외
  원 `21-05-32`와 동일함.
- 단일 후속 변경: CLOSE target `0 mm`, OPEN/CLOSE 공통 success tolerance `±0.5 mm`.
  reward sigma `5 mm`와 나머지 baseline 구조는 유지함.

## 2026-07-30 — `12-21-21` canonical backup + axial reward extension

- best OPEN/CLOSE run `logs/rsl_rl/hand_grasp/2026-07-30_12-21-21`의 모든
  checkpoint/event/export/params/git diff를
  `../backups/runs/hand_grasp/2026-07-30_12-21-21_success/`에 백업함.
  원본과 run 파일 25개의 checksum이 일치하며 `BACKUP_INFO.md`를 함께 둠.
- 변경 전 핵심 소스는 `backups/hand_grasp_pre_axial_2026-07-30/`에 보존함.
- 새 실험은 별도 penalty 없이 기존 OPEN/CLOSE/stability 결합 커널에 Stick2-local
  y축 오차를 추가함:
  `exp(-e_gap/5mm-e_lateral/5mm-e_axial/5mm)`,
  `e_axial=|dot(tip1-tip2,y2)-(-4.8016mm)|`.
- metric `tip_axial_offset`, `tip_axial_error`를 추가함. reward term은 10개 그대로이며
  obs/action/success/PPO는 변경하지 않음. 비교는 fresh run으로 함.
- 8-state FSM은 acquire/orient, in-hand placement, functional grasp, approach,
  object pinch, transport, release, terminal로 정리함. 전이는 dwell/hysteresis,
  policy 입력은 phase one-hot+timer, 학습은 phase-conditioned reset을 사용하는 방향이며
  구현은 아직 하지 않음.

## 2026-07-30 — `hand_setting` single-phase implementation

- Scaffold를 열린 손 + 정렬된 dynamic Stick1/2에서 `pose_005` functional grasp까지
  한 정책이 수행하는 실제 MDP로 전환함. 별도 subphase/FSM/command는 사용하지 않음.
- reset은 thumb joint2 `-0.1659`, 나머지 0이며 stick world pose는
  `x=0.055/0.035, y=0, z=0.5195`, identity/zero velocity임.
  state=PD target으로 시작하므로 preload는 정책이 직접 만들어야 함.
- action은 기존 current-joint residual 20D를 재사용하고 OPEN/CLOSE one-hot을 제거해 obs는 101D임.
- reward ladder는 weak q/final stick pose → four central-shaft semantic proximity →
  saturated 6-contact mean → 6-contact min → pose/region-gated completion/stability임.
  mean은 같은 총 weight 5를 여섯 per-contact term(weight 5/6)으로 분해해 1/6부터
  신호와 개별 TensorBoard tag를 주고, hard min은 6/6이 아니면 0임.
- success는 exact six pair force `>=0.02 N`, central-shaft four-region,
  Stick1/2 final pose, palm-relative speed 조건을 30 step 유지하면 pulse 30000과 즉시 종료함.
  episode는 15초, drop threshold는 world z 0.40 m임.
- 최종 smoke `logs/rsl_rl/hand_setting/2026-07-30_17-05-31`에서 command 0,
  obs/action 101D/20D, reward/termination 18/4와 24-step rollout을 확인함.
  기존 hand_grasp는 변경하지 않았고 103D checkpoint resume는 금지함.
- task registry import를 막던 `chopsticks_grasp_env_cfg.py`의 누락
  `STICK_POS=(0.62,-0.20)`을 기존 배치 주석과 동일하게 복구함.

## 2026-07-30 — root backup directory 정리

- root에 모인 loose backup을 `../backups/code/box_transport/`와
  `../backups/code/chopsticks_grasp/`로 task별 분류함.
- 성공 run은 `../backups/runs/hand_grasp/2026-07-30_12-21-21_success/`로 이동함.
- `../backups/README.md`에 분류·naming 규칙을 기록했으며 원본 파일은 삭제하지 않음.

## 2026-07-30 — hand_grasp play keyboard mode CLI

- `scripts/rsl_rl/play.py --keyboard_hand_mode`에서 실행 중 `1=OPEN`, `2=CLOSE`를 선택함.
- Isaac/Carb viewport listener→thread-safe queue→simulation loop 순으로 전달해 CUDA command tensor와
  observation은 simulation thread에서만 갱신함.
- drop/reset 뒤에도 마지막 선택을 복원하고 mode/gap diagnostics를 자동 출력함.
- GUI viewport 입력 전용이며 `--headless`와 함께 쓰면 parser error를 냄.
- 고정 `--hand_mode open|close`와 자동 `--alternate_hand_mode`도 보존하며 세 방식은 상호 배타적임.
- `py_compile`, `git diff --check`, conda 환경의 `play.py --help`를 통과함.

## 2026-07-30 — hand_setting TensorBoard metrics

- `isaac_neuromeka/env/managers.py`의 `CustomRewardManager`에 hand_setting 전용
  configure/compute 경로를 추가함.
- TensorBoard에 `Metrics/hand_setting`, `_final`, `_min`, `_max` 네 family를 기록함.
  각 family는 six-pair force/contact, central-shaft region, Stick1/2 pose error,
  palm-relative speed, `setting_valid`, `success_stable_steps` 등 32개 scalar를 가짐.
- reward/termination cfg의 active sensor group과 threshold를 그대로 읽어 metric과 성공 판정을
  일치시킴.
- `logs/rsl_rl/hand_setting/2026-07-30_17-26-48` smoke가 24 step을 통과했으며
  TensorBoard event 파일에서 총 128개 `Metrics/hand_setting*` scalar tag를 확인함.
- 이미 실행 중인 학습에는 변경된 manager code가 반영되지 않으므로 프로세스 재시작이 필요함.
  obs/action/reward/PPO는 불변이라 최신 hand_setting checkpoint resume는 호환됨.

## 2026-07-30 — hand_setting contact-only shaping A/B

- run `logs/rsl_rl/hand_setting/2026-07-30_17-31-40`은 333 iter에서 mean reward
  `144.7`로 포화했지만 functional contact 약 `1/6`, full-contact/pose-valid/
  setting-valid/success `0`이었음. index/middle/ring contact force도 0이었음.
- easy joint/object reference와 broad shaft region shaping의 local optimum으로 판정함.
- `hand_setting_env_cfg.py`에서 기존 term을 삭제하지 않고 joint reference,
  Stick1/2 reference, four region proximity, setting completion/stability weight를
  `0`으로 변경함.
- six independent contact terms(weight `5/6`), hard contact-min `20`,
  action-rate와 original strict success validator는 유지함. 새 reward/subphase는 없음.
- objective가 변경되어 `17-31-40` resume는 금지하고 다음 학습은 fresh run으로 시작함.

## 2026-07-30 — hand_setting missing-tip approach shaping

- contact-only run `logs/rsl_rl/hand_setting/2026-07-30_18-14-08`은 744 iter,
  약 89분에도 episode max contact `3/6`, hard min/full-contact/success `0`이었음.
- 유지 contact는 thumb-distal, palm, thumb-mid였고 index/middle/ring 최대 force는
  `0.0005/0.0009/0 N`에 불과했음.
- 기존 `_region_term`에 weight 인자를 노출하고 thumb는 `0`,
  index/middle/ring만 `0.5`로 설정함. 새 reward term은 추가하지 않음.
- joint/Stick reference와 setting completion/stability는 계속 `0`; contact terms,
  hard min, strict success, action-rate는 유지함. 다음 run은 fresh로 시작함.

## 2026-07-30 — hand_setting Stick2-first state gate

- GUI replay에서 thumb-side pivot이 Stick2보다 먼저 닫혀 valley 입구를 막는 순서
  failure를 확인함.
- `mdp.py`에 `stick2_seated_gate`,
  `seated_gated_contact_group_strength`,
  `seated_gated_body_box_shaft_region_proximity`를 추가함.
- gate는 Stick2 palm pose `<=15 mm/20°`와 palm/Stick2,
  thumb-mid/Stick2 contact `>=0.02 N` 두 개를 모두 요구함.
- gate 전에는 Stick2 reference pose `12`와 두 anchor contact만 보상함.
  gate 후에만 나머지 four contact, missing index/middle/ring region `0.5`,
  functional-contact min `20`이 켜지며 gate 이탈 시 즉시 0으로 돌아감.
- 새 reward term/FSM/command/observation은 없고 reward 18개,
  obs/action `101D/20D`, strict success/termination을 유지함.
- `CustomRewardManager`에 `stick2_seated`를 추가해 hand-setting metric은
  33개 × 4 family = 132개가 됨.
- smoke `logs/rsl_rl/hand_setting/2026-07-30_21-44-34`가 1 env/1 iter,
  24 policy step을 통과했으며 reset에서 gate와 downstream reward가 0임을 확인함.
- objective 변경으로 기존 checkpoint resume는 금지하고 fresh run으로 비교함.

## 2026-07-30 — hand_setting force-gate correction

- run `logs/rsl_rl/hand_setting/2026-07-30_21-50-01`은 389 iter에도
  `stick2_seated=0`, Stick2 pose-valid `0`, final contact `1/6`이었음.
  final Stick2 error는 약 `45 mm/74°`이고 지속 접촉은 palm–Stick2였음.
- palm/thumb-mid `0.02 N`을 downstream reward의 선행 조건으로 둔 순환 gate를
  active cfg에서 제거함.
- `mdp.py`에 handle-side centerline point와 directed long axis를 비교하는
  `stick2_valley_geometry`, `stick2_in_valley_gate`,
  `valley_gated_*` terms, `StickValleyCenterlineTracking`을 추가함.
- valley point는 Stick2 local `y=-60 mm`; `5 mm/10°` corridor에 들어오면
  force 없이 gate가 열림. gate 전 ring region/contact와 centerline tracking,
  gate 후 나머지 다섯 contact/index-middle region/contact-min이 활성임.
- contact `0.02 N`은 strict success와 diagnostic seated 판정에만 사용함.
- manager metric에 valley point/axis error와 in-valley를 추가해
  36 × 4 = 144 scalar tag를 event 파일에서 확인함.
- smoke `logs/rsl_rl/hand_setting/2026-07-30_22-45-33`은 24 step을 통과했고
  reset에서 gated reward 0, ungated ring region nonzero를 확인함.
- reward 변경이므로 이전 checkpoint resume 없이 fresh run으로 시작함.

## 2026-07-31 — hand_setting valley force validation / thumb probe

- geometry-only run `22-59-53`은 4784 iter에도 in-valley 0이었고 best
  point/axis error는 약 `26.1 mm/45.6°`였음.
- active valley approach는 point/axis score hard-min, loose
  `20 mm/30°` 안의 palm/thumb-middle force-min, strict
  `10 mm/15° + 0.02 N×2` gate로 구성함.
- manager에 geometry/support gate와 thumb joint2
  position/target/action metric을 추가함.
- `scripts/debug/hand_setting_thumb_action_probe.py`로 다른 19 action을
  0으로 두고 joint2만 sweep함. `scale=0.1`에서 실제 joint가
  `-0.1659→+0.458 rad`까지 움직였으나 Stick2 valley error는 약
  `46.4 mm/68.7°`로 불변. 다음은 thumb joint1/joint2 staged sequence를
  검증함.

## 2026-07-31 — hand_setting valley-anchor-only reward

- active weight는 `stick2_valley_approach=12`와
  `valley_anchor_support=12`뿐임.
- Stick1/index/middle/ring/final six-contact/success/action-rate는 weight
  `0`으로 park해 Stick2가 palm에 스치자마자 Stick1 pivot으로 넘어가는
  reward 간섭을 제거함.
- reset은 loose support corridor 밖이므로 geometry 접근 항은 유지하고,
  anchor support는 `20 mm/30°` 안에서만 양측 force-min을 지급함.
- smoke `logs/rsl_rl/hand_setting/2026-07-31_10-05-54`가 24 step 통과함.

## 2026-07-31 — hand_setting finite-shaft valley distance

- exact local `-60 mm` station matching을 제거하고 saved-pose valley
  target과 current 180 mm finite shaft segment 사이 최단거리로 교체함.
- 장축 방향 sliding은 free이며, closest station은 `[-0.09, +0.09] m`로
  clamp해 실제 stick 구간이 valley를 덮어야 함.
- reward/gate/manager metric/probe가 같은 계산을 사용함. TensorBoard
  metric은 `stick2_valley_shaft_distance`.
- final smoke `logs/rsl_rl/hand_setting/2026-07-31_10-32-23`: 24 step 통과,
  action/obs `20/101`, nonzero reward는 valley 두 항뿐임.

## 2026-07-31 — hand_setting spawn feasibility reset

- 두 stick을 `hand_setting`에서만 world `+x`/palm `+z`로 `20 mm`
  이동함. reset world position은 Stick1/2
  `(0.075,0,0.5195)` / `(0.055,0,0.5195) m`; pair separation과
  orientation은 유지함.
- open hand는 `finger1_joint2=-0.1659 rad`, 나머지 관절 `0`이며
  state=target이라 preload가 없음.
- 1-env 5초 zero-action probe: 초기 6 contact force 전부 `0 N`;
  최종 Stick1/2 palm position 약
  `(4.69,0.05,83.97)` / `(11.80,-0.02,55.80) mm`;
  최저 world z `0.5046/0.5118 m`; Stick2 palm force `0.095 N`,
  thumb-mid force `0 N`. 두 stick 모두 drop threshold 위에 남음.
- shared `hand_grasp`와 valley reward 정의는 변경하지 않음. reset 변경이므로
  다음 hand_setting 학습은 fresh run으로 시작함.

## 2026-07-31 — hand_setting Stick2 reference-only A/B

- active reward를 `pose_005` Stick2 palm-relative full-pose tracking
  하나로 단순화함: weight `12`, position/orientation sigma
  `0.10 m/90°`.
- anchor support와 모든 joint/Stick1/contact/region/final reward는
  cfg에서 주석 처리해 Reward Manager에서 제외함. 복구용 정의와
  finite-shaft `-60 mm` 코드는 주석/함수로 보존함.
- `stick2_in_valley`는 reference pose `<=15 mm/20°`와
  palm/thumb-link2 force `>=0.02 N×2`; loose support-ready는
  `30 mm/30°`임.
- TensorBoard valley metric은 `stick2_valley_pose_valid`,
  `stick2_in_valley`를 유지함. 비활성 reward cfg에 종속되던 region score와
  loose support-ready tag는 제거하고 force/pose/shaft-valid metric은 유지함.
- smoke `logs/rsl_rl/hand_setting/2026-07-31_11-59-59`: 1 env,
  1 iter/24 step, obs/action `101/20`, active reward 1개 통과.
- objective 변경이므로 이전 checkpoint resume 없이 fresh run으로 시작함.

## 2026-07-31 — hand_setting paired-reference + thumb-pivot A/B

- active reward를 `two_stick_reference_min=12`와
  `reference_thumb_pivot_min=8` 두 개로 교체함.
- 첫 항은 `pose_005` 기준 Stick1/2 palm-relative full-pose score 중
  낮은 값을 반환해 한 stick만 맞추는 파밍을 막음.
- 둘째 항은 pair score와 `finger1_link3`의 Stick1 local
  `y=-60 mm` pivot-station 접근 score 중 낮은 값을 반환함.
  scale은 pose position/orientation `0.10 m/90°`, thumb approach
  `20 mm`임.
- drop penalty는 추가하지 않고 기존 Stick1/2 drop 즉시 종료만 유지함.
  모든 active reward가 양수라 낙하는 남은 누적 보상을 상실하게 됨.
- 나머지 reward는 계속 미등록 주석 상태이며, 변경 후 `py_compile`,
  active-term 정적 검색, `git diff --check`만 수행함. 다음 학습은 fresh run.
- task reset 코드는 inherited scene spawn에 offset을 더하는 대신 검증된
  Stick1/2 absolute world position `(0.075,0,0.5195)` /
  `(0.055,0,0.5195)`를 직접 사용하도록 단순화함. 실제 배치는 동일함.
- `Metrics/hand_setting{,_final,_min,_max}`에
  `thumb_pivot_distance`(m), `thumb_pivot_score`(0~1)를 추가함.
  active reward와 같은 거리 helper를 공유함.
- `pair_score>=0.50 && thumb_pivot_score>=0.35`인 memoryless Stage-1
  gate를 추가함. gate 뒤 active reward는 joint reference max `8`
  (sigma `0.80`, six-contact progress에 따라 `2`까지 감쇠),
  functional contact mean `5`, min `20`임.
- `hand_grasp`와 맞춰 region reward는 켜지 않았고, 열린 손의 sparse contact
  문제만 mean과 broad q-reference로 연결함. strict success reward
  `30000`도 복구함. metric에는 `stage1_pair_score`, `stage1_ready` 추가.

## 2026-07-31 — hand_setting Stage-1 joint guide 8→2 감쇠

- `15-03-31`은 Stage-1 gate가 평균 `0.92`까지 열렸지만 q-reference raw가
  약 `0.25`, functional contact가 평균 `2.75/6`, min은 계속 `0`에
  머물러 열린 index/middle/ring을 final pose로 보내는 guide가 부족했음.
- Stage-1 q-reference의 config weight를 `2→8`로 올림. 다만
  `m=min_i(clamp(F_i/0.02,0,1))`을 사용해 실제 effective weight를
  `8*(1-0.75m)`으로 계산하므로 여섯 접촉 전에는 `8`, 모두 threshold에
  도달하면 기존 hand_grasp weak prior와 같은 `2`가 됨.
- 이 감쇠는 history/FSM 없이 현재 force만 사용하며, 접촉을 잃으면
  q-reference guide가 자동으로 복구됨. `py_compile`과
  `git diff --check`만 수행했고 다음 비교는 fresh run으로 시작함.

> 08-01~08-03 상세는 `ACTIVITY_2026-08-0{1,2,3}.md` 참조. 아래는 08-04·08-05 요지.

## 2026-08-04 — chopsticks 빈손-뒤집기 해킹 수정 (lift_gate 복원)

- overnight run `23-52-23`이 파지 없이 팜만 뒤집는 해킹(`palm_up_cos` 0.98인데
  `cage_inside` 0, `palm_dist` 0.31m). 원인: 08-03에 `palm_up`을 cage에서 떼며
  `progress×goal_gate`만 남겼는데, 목표가 스틱 테이블 높이와 가까워 테이블에서도
  `goal_gate`가 정확히 0이 아니라 ~0.033 잔여 → 큰 weight가 증폭함.
- 수정: `palm_up`·`stick_in_palm`에 `lift_gate` 복원(cage는 계속 뺌). `lift_gate`는
  clearance 기반이라 테이블에서 정확히 0 → hack 차단 확실. fresh run. 상세 `ACTIVITY_2026-08-04.md`.

## 2026-08-04 — hand_setting per-joint kp/kd 튜닝 + 획득 재시도 (진전 없음)

- wuji.py fingers를 옛 평값(stiffness 2.0/1.0, damping 0.05, effort 0.6)에서 per-joint
  튜닝으로 교체(엄지 joint2 kp 2.0→2.7 UP, joint3 0.75, 비엄지 joint2 0.7, damping joint4
  0.0, effort URDF값). `hand_grasp`는 새 kp/kd로 재학습해도 잘 됨(`17-30-01`).
- 그러나 `hand_setting`(편 손→기능적 파지 획득)은 새 kp/kd로 **진전 없음** — 스틱을 REF로
  못 올리고 freeze. 이 미제가 08-05 진단으로 이어짐.

## 2026-08-05 — hand_setting 획득 진단 대장정 → σ 조정으로 첫 성공

- 유일한 차이로 보였던 kp/kd를 의심하며 진단. **진단 도구 신설**: `finger_reach` task
  (손끝 랜덤 목표 추종 — 제어 격리, 정상 확인), `hand_setting_grasp_reach_probe.py`(정책
  잔차 액션으로 PREGRASP 구동 — 자세 achievable 확정), `hand_setting_0731` task(07-31 env를
  저장 diff에서 그대로 복원 + 현재 kp/kd, 격리용).
- **진짜 원인 = 보상 σ**(kp/kd 아님): 기존 σ(pos 0.10/ori 1.57=90°)가 너무 헐거워 스폰
  스틱 자세가 이미 pair_score 높음 → 정책은 thumb_pivot만 맞추면 됐고, thumb_pivot이
  유클리드 거리만 봐 엄지가 opposition(위) 대신 스틱 밑으로 파고듦("비비기").
- **해결**: pair σ를 `pos 0.01/ori 0.25(~14°)`로 조이고 `thumb_sigma 0.02→0.06`, `success`
  reward weight 30000 one-shot 추가 → **획득 첫 성공.** kp/kd는 튜닝값 그대로 유지.
- 상세 `ACTIVITY_2026-08-05.md`. 도구 함정은 root `CLAUDE.md`/`AGENTS.md`.

## 2026-08-06 — hand_object 학습환경 신설 + tip gap σ 2단 분리

- **hand_object 신설**: `hand_move` 정책을 fine-tuning해 두 젓가락으로 1cm 큐브를 실제
  접촉력으로 집고, 받치던 기둥이 내려가도 안 떨어뜨리게 하는 태스크. obs 103D / action 20D를
  `hand_move`와 동일하게 유지(체크포인트가 shape 변경 없이 로드). 큐브·접촉력은 actor가 못 보고
  보상·종료·메트릭에만 쓴다 — "기존 proprioceptive 상태만으로 안 보이는 물체를 잡고 있을 수
  있나"를 묻는 실험이라 의도적.
- 스틱마다 ContactSensor를 두고 각각 **Cube만 필터**, closing axis 투영으로 스틱별 압착력을
  구해 **`min(score1, score2)`** 로 bilateral 보상. 한쪽만 세게 눌러도 만점이 나오면 안 되므로
  mean/sum이 아니라 min. 힘 부호는 추정하지 않고 isaacsim 테스트에서 유도(`force_matrix_w` =
  필터가 센서에 가하는 힘).
- **yaw/큐브/기둥 위치 3값은 측정 전이라 `None`으로 비워두고 fail-fast.** 임의값으로 채우면
  아무도 검증 안 한 기하로 학습이 돌아가므로. 측정용 calibration 키(`A/D` yaw, `P` 출력,
  `V` 기둥 하강)를 기존 키보드 인프라에 붙였다. 학습 2개가 도는 중이라 Isaac Sim 실행 금지 →
  **런타임 검증·calibration 모두 미완**, 정적 검증만 통과.
- **tip gap σ 2단 분리**: `open/close_tip_gap.sigma`(캡처)는 5mm로 두고
  `mode_grasp_stability.gap_sigma`만 5mm→1mm(수렴)로 조임 — 항을 추가하지 않고 coarse/fine 분리.
  마지막 1mm 구간 보상 차이가 6.2배가 된다. 다만 리셋 구간 gap 압력의 55~71%가 사라지므로
  캡처가 죽는지 확인 필요(신설 메트릭으로 보임).
- **OPEN/CLOSE tip gap error 메트릭 분리** — 기존 `tip_gap_error`는 두 모드가 섞여 어느 쪽이
  안 되는지 알 수 없었다. one-hot 마스킹 누적 후 그 모드가 켜져 있던 시간으로 나눠 조건부 평균.
- **참고**: `success`(예산의 85%)가 `tip_gap_error_limit=0.0005`(±0.5mm)와
  `angular_speed_limit=3.0`(실측 93% 초과)에 막혀 계속 0. σ를 조여도 여기 못 닿으면 그대로 0.
  **미조치**.
- **런타임 검증 통과**(학습 중단 후): `scripts/debug/hand_object_probe.py` 신설, 검사 20개 전부
  통과. hand_move 체크포인트가 actor `103→128→128→20` 으로 로드돼 **fine-tuning 호환 확정**,
  힘 부호 규약도 **실측 확인**(양방향 압착 642회 전부 양수, 음수 0).
- **보정 완료**: 손 스폰과 큐브가 둘 다 고정이므로 보정 대상은 "손이 날아갈 목표 root pose"
  하나뿐이고 큐브는 그 자세의 tip 중점으로 따라온다. GUI 에서 키보드로 손을 몰아 자세를
  잡고 `P` 로 읽었다. 목표 pose (0.0780, −0.0470, 0.3700) / euler (+5.35°, +19.89°, −59.19°),
  큐브 (0.1468, 0.0865, 0.4114).
- **설계 정정**: 처음엔 yaw 하나만 보정하고 root 위치는 스폰 고정으로 만들었는데, 그러면 tip 이
  팜 주위로 호만 그려 12cm 떨어진 큐브에 도달하지 못한다. 사용자 지적으로 **이동+회전**으로
  바꿈. 액션에 `position_command_name` 을 신설했고 미설정이 기본이라 `hand_move` 는 무영향.
- **이동을 2단계로**: 한 직선으로 가면 대각선 접근 중에 계속 회전해 tip 이 큐브 자리를 옆으로
  쓸고 지나간다. 스폰 높이에서 방향·x,y 를 맞춘 뒤(1단계) 순수 수직 하강(2단계). 경유점은
  목표의 z 만 스폰 z 로 바꾼 값이라 계산으로 나오고, 덕분에 2단계가 수직인 게 구조적으로 보장.
- **커리큘럼**: 큐브 보상 3개는 cfg 기본 ON(100/50/200). 1단계는 CLI 로 0 을 줘서 끈다 —
  weight 0 이면 리워드 매니저가 항을 아예 건너뛰고, 메트릭은 센서에서 직접 계산해 그때도 찍힌다.
- **예산 점검에서 걸린 것**: `bilateral_cube_force` 450 점은 젓가락 파지 유지에 게이트가 없다.
  "젓가락 흘리고 큐브만 짓누르기" 손익이 450 vs 490 이라 여유가 8% 뿐. 일단 100 으로 시작하고
  `functional_contact_count` 가 5.5 아래로 떨어지면 50 으로 낮추기로 함.
- 상세 `ACTIVITY_2026-08-06.md`, 태스크 큐레이션 `hand_object_log.md`. 도구 함정은 root `CLAUDE.md`.

## 2026-08-07 — hand_object 최초 파지 성공 / hand_move 3단 커리큘럼 / lateral 보상 신설

- **`hand_object` 가 처음으로 큐브를 잡고 버텼다.** `holding` 0.000 → **0.4375**,
  에피소드의 **48.6%** 가 9초를 완주. 지금까지 큐브 보상 3종이 전부 정확히 0 이었던
  원인은 보상도 기하도 아니라 **상속받은 종료 조건** 이었다.
- `HandObjectTerminationsCfg` 가 `hand_move` 의 `stick2_dropped`(`minimum_height=0.40`)
  를 그대로 물려받았다. `hand_move` 는 손이 스폰 높이에 머무르니 맞는 값이지만
  `hand_object` 는 큐브를 잡으러 **일부러 z=0.365 까지 내려간다.** 하강이 끝나는
  **정확히 2.00초** 에 스틱 루트가 0.40 을 지나 전 에피소드가 오판 종료됐고, CLOSE 가
  2.5초 시작이라 **압착을 한 번도 시도해보지 못했다.** 0.20 으로 오버라이드해서 해결
  (`hand_move` 는 0.40 유지). 앞서 내가 "큐브가 두 팁 사이에 없다, 기하 보정을 다시
  하라" 고 한 진단은 **오진이었고 철회**한다.
- **조건 트리거(파지하면 기둥 제거)가 설계대로 동작함이 처음 확인됐다.**
  `retract_by_grasp` 0.91 — 90.9% 가 파지 조건으로 발동하고 데드라인은 9% 뿐.
  발동 시각 3.50초로 데드라인(5.5초)보다 2초 빠르다. CLOSE 시작 후 1초 만에 조건 충족.
- **`hand_move` 를 3단 커리큘럼으로 재구성**(사용자): 파지자세(5초, 회전 0, 개폐 없음)
  → open/close(15초, 회전 0) → 랜덤 위치. 1단계 완료, `functional_contact_count` 5.98/6.
- 2단계에서 **정책이 파지를 포기하면서까지 끝까지 벌리는** 증상. 보상 실적으로 설명됨 —
  파지 계열이 이미 79% 를 벌고 있어 **비율 문제가 아니다.** 정책이 보는 건 잔여 여지고,
  OPEN 완성 여지 +99.5 vs 파지 전량 −12.0 이라 버리는 게 +87.5 이득이다. 가중치로
  뒤집으려면 w20→w166 이 필요해 비현실적. `OPEN_TIP_GAP` 0.020→0.017 로 대응(사용자).
  `functional_contact_min` 이 12점밖에 못 버는 건 **min 구조** 탓 — 여섯 접촉력 평균은
  전부 포화를 넘는데(0.47~2.12 N) 벌리는 순간 `index_tip_stick1`/`ring_tip_stick2` 가
  떨어져 그 스텝이 0 이 된다(`full_contact` 0.62). 선행 지표는 `full_contact`.
- **`stick_axis_lateral` 보상 항 신설.** 3D 에서 두 직선이 만나려면 위치와 방향이 둘 다
  맞아야 하는데, 기존 `tip_lateral_error` 는 **팁 점의 위치**만 재고 스틱 **축의 방향**은
  아무도 안 봤다(`_tip_geometry_from_palm_poses` 가 stick1 샤프트 축을 계산조차 안 함).
  실측 skew 가 min 2.50° 로 한 번도 0 근처에 안 가는 **상시 편향**이었고 OPEN/CLOSE 가
  같아 모드 게이트 없이 상시로 걸었다. 기존 지수에 얹지 않고 별도 덧셈 항 — 한 지수에
  넣으면 skew 가 나쁠 때 gap gradient 까지 깎인다. 면내각(개폐 자유도)에 gradient 가
  0 임을 수치로 검증. 효과: skew min 2.50° → 0.05°.
- **진단 메트릭 11종 추가**(세 태스크 공용): lateral/axial 의 CLOSE 게이트 버전,
  축 각도의 면내/면외 분해, 리셋 시점 latch 값, 단면 roll. roll 은 두 스틱의 **상대**
  방위는 정렬(4.0°)돼 있으나 **둘 다 면-정면에서 20.6° 돌아간 채 전 구간을 헤맨다** —
  균등분포 평균에 근접해 사실상 무제약. 부작용으로 `support` 가 28.8% 커져
  `close_tip_gap_error` 의 일부가 계측 착시다.
- **보류(근거만 기록)**: `force_saturation` 0.05→0.10~0.15(실측 max 0.2161 N 로 상시
  초과, 신호가 0/1 로 거칠어짐), `reference_distance_sigma` 0.01→0.03(`cube_hold` 600점
  경로가 stability 인자에서 막힘, `cube_distance_to_stick2_tip` min 13.1mm 로 σ 밖).
- 상세 `ACTIVITY_2026-08-07.md`, 태스크 큐레이션 `hand_object_log.md`. 도구 함정은 root `CLAUDE.md`.
- **(같은 날 후속) `cube_hold` 가 0 이던 진짜 원인은 각속도 항이었다.** 게이트 없는
  `cube_distance`/`*_speed_rel` 은 접근·회전 구간을 포함해 오염돼 있었다. `holding` 으로
  게이트한 `hold_distance`/`hold_linear_speed`/`hold_angular_speed` 를 신설해 재보니
  거리는 93.3mm → **8.07mm**(11배 오염, 기하 예측 6~7mm 에 근접해 **팁에 제대로 물려 있음**),
  선속도는 절반으로 줄었는데, **각속도만 10.25 → 10.43 으로 그대로**였다. stability 지수
  12.04 중 **10.43 이 각속도** — 전체의 87% 다.
- 그 10.43 rad/s 는 회전이 아니라 **솔버 노이즈**로 보인다. 큐브가 5mm/3g 이라
  `I = 1.25e-8 kg·m²` 이고, 0.07N 이 2.5mm 편심이면 물리 스텝 한 번에 117 rad/s 가 나온다.
  게다가 지표가 `norm(ω차)` 이라 **알짜 회전 0 인 진동도 평균 10 으로 찍힌다.**
  `hold_distance` 편차가 0.5mm 뿐인 것도 "단단히 물려 안 움직인다"를 가리킨다.
  → `cube_hold` 와 `cube_relative_stability` **둘 다**에서 각속도 항 제거(사용자 판단).
  **곱셈 항에 정책이 못 줄이는 양을 넣으면 σ 오설정보다 나쁘다** — 다른 게 아무리 좋아져도
  목표가 0 에 붙는다. weight·거리 기준점·σ 는 전부 그대로 두었고, 예상 효과는
  `cube_hold` 0.084 → 139 점, `cube_relative_stability` 0.021 → 39 점.
- **내 진단 3건이 틀려서 정정했다**: ① "13.1mm 라 샤프트 안쪽에서 물고 있다"(실제 8.07mm,
  팁에 제대로 물림) ② "각속도 9.98 은 접근 구간 오염일 것"(게이트해도 10.43) ③ "선속도
  하나에 이미 막힌다"(게이트하면 0.80, 안 막힘). 특히 ③ 은 **내가 스스로 오염됐다고
  지적한 값으로 결론을 낸 것**이었고 사용자가 지적해 바로잡았다.
- 거리 기준점 0 → 6mm 는 보류. `exp(-d/σ)` 가 d=0 에서 최대라 "이상적 거리 0"을 요구하는데
  스틱 반두께 3.5 + 큐브 반폭 2.5 = **6.0mm 가 물리적 하한**이라 완벽한 파지도 천장이 0.55 다.
  다만 효과가 일률적으로 1.822배뿐이라 각속도를 안 고치면 무의미했다. 기준값은 실측 8.07 이
  아니라 **기하 이상값 6.0** 을 써야 한다 — 8.07 로 맞추면 현재 자세를 정답으로 고정해
  gradient 가 0 이 되고, 6.0 이면 남는 23% 여지가 roll 정렬 압력으로 작용한다.

## 2026-08-08 — 외란이 파지를 강화한다는 것을 실증

- **결론부터: 사수님 제안(외란 학습)이 맞았고 내 반대가 틀렸다.** 보상 설정이 완전히
  동일하고 외란만 다른 세 런에서 `full_contact` 0.335 → 0.996 / 0.948,
  `min_functional_force` 0.107 → 0.264 → 0.449 N. **외란 세기에 대한 용량-반응**이
  나와서 인과로 볼 근거가 생겼다. `stick1_pivot_error` 도 10.35 → 6.86 → 3.83 mm 로 단조.
- iteration 교란도 배제된다. 외란 없이 it 4983 까지 돌린 런이 `full_contact` 0.583 이
  천장인데, 외란 쪽은 더 적은 iteration 에서 0.95 를 넘는다.
- **메커니즘**: `functional_contact_min` 이 min 게이트라 접촉 하나만 끊겨도 20 weight 가
  통째로 0 이다. 외란이 그 사건을 자주 만드니 **여유를 크게 두는 게 이득**이 된다.
  내가 "min 절벽이 복원을 막는다" 고 걱정한 구조가, 외란과 결합하면 강한 **예방 압력**
  으로 작동한다. `force_scale` 이 0.10 이라 보상은 0.10N 넘으면 더 쥐라고 안 하는데,
  실측 0.449N 은 **보상이 아니라 위험 회피가 만든 여유**다.
- 08-06 "성공-커리큘럼" 체크포인트(it 4983)와 현재(it 6803) 비교: `mean_reward` 1.86배,
  `full_contact` 0.583→0.936, `stick1_pivot_error` 10.15→4.38mm, `open_tip_gap` 7배.
  다만 **Stick2 는 나빠졌다**(위치 2.20→3.96mm, 자세 3.98→6.92deg) — Stick1 을 잘 붙드는
  대신 레일을 양보한 거래로 보이고, Stick2 가 압착축 기준이라 hand_object 에서 봐야 한다.
- **`hand_object` 는 붕괴해서 중단.** 구 보상 체크포인트를 새 보상에 넣은 것이 원인으로
  보인다. it 2231 에 `cube_hold` 3.43(각속도 제거의 성과, 이전 0.084)까지 갔다가 무너져
  it 4113~4116 이 평평해졌다. 죽은 건 **손끝 셋**(index/middle/ring)이고 손바닥·엄지는
  오히려 세졌다 — 엄지 joint2 가 다른 손가락의 토크 2.34배·강성 3.9배라 "Stick2 를
  손바닥에 대고 엄지로 누르기" 가 제일 싼 자세다. 보상이 안 나올 때 물러설 곳이 거기다.
- **내가 이 세션에서 틀린 것 넷** (전부 정정):
  ① `Episode_Reward` 를 합계로 읽어 예산을 15배 부풀림 (실제는 `sum / episode_length_s`).
     이 때문에 "파지가 4% 만 번다"(실제 60%), "파지 버리는 게 +87.5 이득"(실제 −5.7 손해)
     두 결론이 뒤집혔다.
  ② `full_contact` 저점 하나(it 454)를 보고 "단조 발산" 이라 오진. 실제로는 0.03~0.98 로
     진동하며, 능력도 복원력도 있고 **유지를 못 하는 것**이 문제였다.
  ③ hand_object 와 hand_move 가 같은 체크포인트 계보라고 잘못 알았다. 공통점은 보상 설정.
  ④ "이미 깨진 상태를 충분히 겪으니 외란 불필요" — 실증으로 반박됨.
- **미적용**: 외란 적용점을 팁으로(`positions=(0,0.09,0)`, 무게중심이라 지금은 효과가
  약하고 "테이블 충돌" 시나리오와도 안 맞음), 외란 힘 1.0~3.0 상향,
  `force_scale` 0.10→0.8(0.45 이하는 실측 0.449N 때문에 전부 no-op, 그리고 OPEN 과 충돌 +
  시뮬 말단 토크가 실기의 1.3~3.4배라 전이 위험).
- 상세 `ACTIVITY_2026-08-08.md`. 08-07 일지에 §7(lateral 독립항·clamp), §8(Episode_Reward
  정정) 추가. 도구 함정은 root `CLAUDE.md`.

## 2026-08-14 — hand_real/hand_final MuJoCo deploy 1단계

- root `mujoco_deploy/`에 fixed-base Wuji-only policy plumbing 환경을 추가함. 현재 source의
  101D actor slice, post-wrapper action clip, `q+0.1a`, joint clamp/Joint4 target floor,
  30 Hz/120 Hz timing을 설치 package 구현까지 추적해 고정함.
- 기존 `right.xml` position actuator를 원본 수정 없이 사용하며 모든 qpos/dof/actuator를
  20개 physical joint name으로 검증함. frozen Stick reference이므로 task behavior가 아니라
  observation/ONNX/action/position-target interface만 검증함.
- `wuji_mujoco`에서 model/mapping/20관절 command/ONNX/OPEN·CLOSE closed loop 및 viewer PASS.
  실행·한계·source 근거는 `mujoco_deploy/README.md`와 `ISAAC_POLICY_CONTRACT.md` 참고.

## 2026-08-17 — hand_real disturbance A/B + MuJoCo/ArUco 인계

- active `hand_real`은 105D quaternion-history observation, Joint1/2 `0.10`, Joint3 `0.20`,
  Joint4 `0.15` current-joint residual action이다. official URDF effort limit과 task-local
  20-joint Kp/Kd를 사용하고 Stick1 5 mm axial offset은 `0.0`으로 원복돼 있다.
- 약한 외란 source `2026-08-16_21-14-45/model_3800.pt`에서 optimizer까지 이어받은
  true-resume 두 개를 비교 중이다.
  - `0.3~1.2 N`: `2026-08-17_02-50-15`; recovery 0까지 붕괴했다가 latest 9009에서
    contact/final `5.75/5.63`, recovery `91.4%`로 복구. OPEN/CLOSE `11.95/9.45 mm`라
    기하는 아직 source 수준이 아니다.
  - `0.1~0.6 N`: `2026-08-17_11-04-35`; latest 6010에서 contact/final `5.90/5.28`,
    recovery `99.0%`, CLOSE/lateral `2.69/3.07 mm`. OPEN `15.50 mm`라 CLOSE 편향이다.
- MuJoCo는 official fixed-base Wuji right hand, strict name mapping, common 105D adapter,
  Isaac-equivalent action decode, 30/120 Hz scheduler, dynamic sticks와 D435/ArUco provider까지
  구성했다. Real backend는 SDK/encoder/controller contract 확인 전 안전 차단 상태다.
- ArUco Stick1은 `+Y=tail->tip`, `+Z=ID0 outward`로 정의하고 ID1 offset을
  `inv(T_M0_M1)@T_M0_S`로 자동 계산한다. 두 marker pose agreement median은
  `4.84 mm/1.89 deg`이나 position p90 `21.78 mm` long-tail이 남는다.
- Claude가 이어받을 최신 수치·파일·CLI·우선순위는 root
  `CLAUDE_HANDOFF_2026-08-17.md`, 상세 실험 기록은 root `ACTIVITY_2026-08-17.md` 참고.

## 2026-08-18 ~ 08-19 — 실물 한계 채택, Finger Reach, 실제 Wuji 배포

- **실물 SDK로 읽은 관절 한계가 벤더 description보다 20/20 전부 넓다.** 정규화는
  `2(q−center)/(upper−lower)` 아핀이라 포함관계로 넘어갈 수 없고, 좁은 표를 쓰면 실물이
  한계 근처일 때 `|q_norm| > 1`이 나온다(최악 `finger3_joint2` 전범위의 8%). USD 오버라이드와
  `mujoco_deploy/policy_contract.py`를 실물값으로 통일하고 MuJoCo 관절 ROM도 확장했다.
  2026-08-17에 낮췄던 pregrasp `finger3/5_joint3`(1.5512/1.5490)는 1.6272로 복원했다.
- **`hand_real` sim-to-sim은 계약은 맞고 물리가 갈린다.** 리셋 관측은 Isaac과 비트 단위로
  일치하는데 정책을 돌리면 젓가락을 놓친다. dt 수렴 검사와 접촉/마찰 실험을 거쳐 원인은
  **두 시뮬레이터가 서로 다른 손 모델을 쓴다**로 좁혀졌다 — 공식 description에 충돌 메시가
  없어 MuJoCo가 visual convex hull을 쓰고, 손바닥 hull이 오목부를 메워 리셋에서
  `palm↔stick2 −3.94 mm` 관통이 생긴다. 중력을 꺼도 스틱이 1.5 m 밖으로 사출된다.
- **Isaac 액추에이터에 armature가 없다**(`armature: None`). 벤더 MJCF는 0.0002/0.0005이고
  이는 joint4 유효관성의 98.8%다. PhysX 암묵적 적분기가 그 결과 고주파를 눌러 "잘 도는 것처럼"
  보였다. MuJoCo 기본 게인은 벤더 MJCF 동정값으로 정했고 Isaac 쪽은 재학습이 걸려 건드리지 않았다.
- **Finger Reach를 중지 4관절 전용으로 정비**했다(15D 관측 / 4D 액션). 액션이 20D였고,
  관측이 world-frame 오차였고, 스틱 두 개가 같은 위치에 파킹돼 상호관통했고, 커맨드가
  에피소드 중간에 리샘플됐다. 타깃 박스는 기존 범위가 33.8%만 도달 가능해 실물 한계 기준
  도달집합을 계산해 92.8% 박스로 교체했다.
- **실제 Wuji Hand에 정책을 올렸다.** `wujihandpy` 기반 `real_wuji.py` /
  `real_wuji_scheduler.py` / `run_finger_reach_real.py` 신설. 90 Hz 명령 + 30 Hz 정책,
  teleport 없는 시작 자세 이동, 선형 복귀, 단계별 진단 모드. 공유 계약
  (`finger_reach.py`, `policy_contract.py`, `onnx_policy.py`)은 두 백엔드가 그대로 쓴다.
  MuJoCo 경로는 매 변경마다 CSV 최대차 `0.000e+00`으로 회귀 없음을 확인했다.
- **관절 매핑 확정**: `--test-middle-joint 1~4` 네 개 다 MATCHES, 크로스토크 0.008 rad 이하.
  타이밍은 8201틱 중 늦은 틱 1개, 40초 주행 누적 드리프트 +0.0001 s.
- **첫 sim-to-real 비교**: 관절은 4~8도 벌어지는데(J1 7.73°, J2 0.56°) 손끝은 실물 4.64 mm vs
  MuJoCo 8.68 mm로 **실물이 더 정확하다.** 중지가 4 DOF로 3D 목표를 쫓아 여유자유도가 1개이고
  두 플랜트가 다른 널 스페이스 가지에 앉은 것이다. 타깃당 5초→10초로 늘려도 차이가 그대로라
  LowPass 지연이 아니라 정상상태 차이다. 단 손끝 값은 URDF FK 출력이지 측정이 아니다.
- 상세는 root `ACTIVITY_2026-08-18.md`, `ACTIVITY_2026-08-19.md`. CLI는
  `mujoco_deploy/CLI.md`. 실물 SDK 함정은 root `CLAUDE.md`의 "실물 Wuji Hand (wujihandpy)" 절.

### 2026-08-19 (오후) — 시작 자세 선형 이동, 명령 여백 0.95

- **시작 자세 이동도 선형으로**. 복귀만 선형이고 나가는 이동은 고정 타깃+필터(지수형)라
  손가락이 굽은 채(1.68 rad) 시작하면 초기 속도가 21 rad/s였다. `glide_to_pose`로 바꿔
  0.56 rad/s. 실측 등속(0.19 rad/0.33s) 확인. `--start-seconds` 기본 3.0초.
- **궤적 끝점 clamp**. 기계적 스톱에 눌린 관절은 소프트 한계를 넘겨서 읽힌다
  (1.6804522 vs 상한 1.680047, 0.023°). 보간을 측정값에서 시작해 첫 목표가 범위 밖이 되어
  주행이 죽고 20초 데이터가 날아갔다. 끝점 clamp + 복귀 try/except로 로그 보존.
- **run 5 (20초/4타깃) 판독**: 타이밍·추종은 문제 없음 — 지각 0/2438, 정책 스텝 내
  목표 계단 따라잡기 30~32% vs LowPass 2Hz 이론 34.2%로 일치. **문제는 `finger3_joint3`가
  상한에 36.5% 눌러앉아 있고 그 동안 정책이 a3 +0.45로 계속 밀고 있다는 것.** 정착 오차
  4.81 mm인데 J3가 눌린 목표(6.45/5.73)가 안 눌린 목표(3.89/3.15)보다 나쁘다.
- **명령 여백 0.95 도입**. 19만점 격자 IK로 REACH 박스 전체가 1.00/0.95/0.90 모두 100%
  5mm 이내임을 먼저 확인 — **여백이 도달 범위를 깎지 않는다**(여유자유도 1개가 흡수).
  MuJoCo 실측: J3 압착 80.4%→0.0%, 지문 오차 8.80→8.17 mm로 **오히려 개선**.
  정의는 중심±0.95×반범위(Isaac `soft_joint_pos_limit_factor` 규약).
  `COMMAND_TARGET_LIMITS`는 미변경 — 여백은 별도 테이블로 얹어 sim-real 불일치를
  드러나게 뒀다. 실물 기본 0.95 / MuJoCo 기본 1.0, `--parallel-mujoco`가 자동 전달.
  MuJoCo 기본 경로는 변경 전과 CSV 완전 동일.
- 주의: **`hand_real`에 0.95를 그대로 옮기면 pregrasp가 범위 밖**이다(idx 10/18이
  1.6272 vs 상한 1.6244/1.6197). finger_reach에서 공짜였던 것이 파지에선 아니다.
- 상세는 root `ACTIVITY_2026-08-19.md` 7~10절.

### 2026-08-21 ~ 08-24 — 105D 실물 파지 배포, 체크포인트 교체, 전류 상한

- **`mujoco_deploy` → `Deploy` 재편** (41파일). 층 기준은 "공유되는가"가 아니라
  **"바꾸면 학습된 체크포인트가 무효가 되는가"**. 임포트 방향
  `run → backends → policy/vision → common` 을 테스트로 강제.
- **`obs[40:55]` tip_link 프레임 버그**. URDF 두 벌의 `*_tip_fixed` origin z 가 달라
  엄지 3.00 mm / 검지 0.70 mm 상수 바이어스. **물리 부품은 같은 자리**이고 틀린 건
  관측 배선뿐. q 에서 FK 로 푸는 방식으로 교체.
- **105D 실물 파지 첫 완주** (600/600 스텝, 20초). 액션·qt 로깅 추가.
- **명령 여백 0.95 를 계약 테이블에 반영** (사용자 지시). 8/19 의
  `중심 ± f×반범위` 와 **다른 식이다** — `COMMAND_LIMIT_RATIO = 0.95` 를
  `한계 × ratio` 로 곱한다. `COMMAND_TARGET_LIMITS` 를 실제로 좁혔고,
  `OBSERVATION_NORMALIZATION_LIMITS` 는 공장표 그대로 둔다(정규화는 학습과 같아야 함).
  `finger_reach` 는 공장표에서 따로 유도 — 안 그러면 0.95² = 0.90 이 된다.
- **체크포인트 교체 안전장치**: `Deploy/tools/verify_policy_contract.py` 가 학습 런의
  `params/env.yaml` 을 배포 상수와 대조한다. shape 검사로는 못 잡는 pregrasp(78 mrad),
  스칼라 액션 스케일, joint4 override, **학습이 본 모드**를 실제로 잡았다.
- **NEUTRAL `[0,0]` 은 세 번째 학습 상태다.** `hand_real2` 의 5초 커리큘럼은 이것만
  본다 — OPEN/CLOSE 만 보내면 그 체크포인트는 아예 못 돌린다.
- **전류 상한 실험** (1.5 / 1.3 / 0.8 A). 1.3 A 부하 주행은 +0.98 °C/s 로 평형 없이
  76.3 °C 까지 올랐다. **상한을 낮춰도 포화 duty cycle 은 안 준다** (틱당 5.4 → 6.26 관절).
  열 여유는 `I²` 에서 나오는 것이지 포화가 줄어서가 아니다.
- **주행을 죽인 버그 둘**: glide 보간의 float32 1 ULP 초과(0.95 여백 이후 일상적),
  중첩 클로저의 `current_sum +=` 재바인딩(ENABLE 직후 step 0 사망). 둘 다 회귀 테스트.
- **로그에서 체크포인트 역추적**: 실물 CSV 에는 정책 경로가 없다. 기록된 obs 를 후보
  ONNX 에 재생(L1)해 `max|diff| = 0.000e+00` 로 식별. 녹화 재생 쪽은
  `joint_record_*.meta.json` 에 `checkpoint` 가 있다.
- 상세는 root `활동기록/ACTIVITY_2026-08-21.md` ~ `ACTIVITY_2026-08-24.md`.
  CLI 는 `Deploy/CLI.md`, 에이전트 함정은 `Deploy/CLAUDE.md`.
