# Agent Notes

- Main shared baseline: `/home/lsc/IsaacLab` with `env_isaaclab`, Isaac Sim 5.1, IsaacLab 2.3.x.
- Isaac Sim itself comes from the `isaacsim` pip metapackage (v5.1.0.0) installed inside `env_isaaclab`, not from `/home/lsc/isaacsim_pkg` (that's a separate standalone zip extraction, unused by this workflow). IsaacLab is an editable install pointing at `/home/lsc/IsaacLab/source/isaaclab` (source checkout, not the pip binary release the README's install guide assumes — pre-existing setup, not something to change casually).
- `isaac_neuromeka` (this package, published as `nrmk_isaaclab_public`) must be `pip install -e .`'d into `env_isaaclab` from `nrmk_isaaclab_wuji/` before anything imports it. That install pulled in `pandas>=2,<3`, which downgraded an existing `pandas 3.0.3` to `2.3.3` in the env — worth knowing if something else in `env_isaaclab` expected pandas 3.
- `import isaac_neuromeka` (or anything under `isaaclab`) directly from a plain `python -c` fails with `ModuleNotFoundError: No module named 'pxr'` — this is expected, not a bug. `pxr`/`omni` bindings only become importable after `isaaclab.app.AppLauncher` boots Kit, which is why `scripts/rsl_rl/train.py` always launches the app before importing task modules. Never try to smoke-test task registration without going through `AppLauncher` first.
- New working extension: `/home/lsc/wuji_indy_lab_51/nrmk_isaaclab_wuji`.
- Do not modify shared installs or older user assets directly; copy into the new working extension first.
- Neuromeka public is the IsaacLab/Indy7 extension base. Wuji hand assets came from existing Retargeting/GeoRT-style folders.
- Current target direction: Indy7 + Wuji hand for future chopstick manipulation RL.
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
- `Indy-Wuji-Reach` task end-effector body is `palm_link`, not `tcp`. `tcp` is a leftover non-rigid frame under `link6` from the bare indy7 arm (no `RigidBodyAPI` in any of the three usd variants) — the URDF's `tcp -> palm_link` fixed joint got merged away during USD generation. Don't reintroduce `"tcp"` as a body reference in reach/task configs; use `"palm_link"`.
- `scripts/assets/apply_wuji_hand_collision_meshes.py` is the reproducible post-process for the fidelity USD. It de-instances the hand `collisions` Xforms, inserts one collision mesh per Wuji hand link from `isaac_neuromeka/assets/model/urdf/wuji_right/meshes/*_collision.STL`, applies `PhysicsCollisionAPI` + `MeshCollisionAPI(convexHull)`, and blocks nested importer leftovers. Validation target: 26 direct Wuji collision meshes, 0 active nested children, 0 CollisionAPI under hand visuals.

## Latest Handoff Summary

- 현재 task는 `Indy-Wuji-Reach`임.
- 현재 active USD는 `indy7_wuji_right_simplified.usd`임.
- 현재 tracking body는 `palm_link`임.
- 현재 action shape는 6임.
- 현재 policy observation shape는 55임.
- 현재 observation은 arm 6축 joint position/velocity history만 봄.
- hand joints는 articulation에 남아 있음.
- hand joints는 policy action/observation에서 제외됨.
- `joint_vel` reward penalty는 arm 6축만 봄.
- `sim.render_interval = decimation` 적용됨.
- `--num_envs 1 --max_iterations 1` smoke test 통과함.
- `--num_envs 32 --max_iterations 5` test 통과함.
- GUI 실행 확인됨.
- 사용자가 `--num_envs 128 --max_iterations 20` 실행함.
- 다음 권장 run은 `--num_envs 512 --max_iterations 100`임.
- root handoff docs는 `/home/lsc/wuji_indy_lab_51/AGENTS.md`, `/home/lsc/wuji_indy_lab_51/WORKLOG.md`, `/home/lsc/wuji_indy_lab_51/study.md`임.
- 2026-07-08 활동 일지는 `/home/lsc/wuji_indy_lab_51/ACTIVITY_2026-07-08.md`임.
