# Real-ready Wuji Hand 1 deployment stack

This is the first backend of the deployment stack intended for the physical
Wuji Hand 1. It is not a separate MuJoCo-only policy wrapper. The canonical
joint contract, normalization, observation history, action decoder, stick-pose
provider boundary, fingertip FK, and policy runner do not import MuJoCo.

The model is the official right-hand Hand 1 description pinned at commit
`06e5f14cdd1d5fad0a666ca463a668bf609f9534` (`v2026.8.14`). See
[`assets/wuji_description/UPSTREAM.md`](assets/wuji_description/UPSTREAM.md).
The palm is fixed and there is no Indy7.

## Architecture

```text
policy_contract  perception  backend_protocol
       \             |             /
        action + observation + PolicyRunner
                         |
             interface-only dependency
                /                    \
 MujocoWujiHand + Scheduler      RealWujiBackend
          (enabled)          (disabled pending validation)
```

`joint_mapping.py`, `mujoco_wuji.py`, and `mujoco_scheduler.py` contain all
MuJoCo storage IDs, model/data access, physics stepping, target-hold timing, and
viewer synchronization. `PolicyRunner` knows only the backend protocol and has
no physics loop or concrete-backend branch. Its observation adapter is required
explicitly, so the composition root—not the runner—selects the perception
provider.

The current observation is 105D with StickPose7D (`palm xyz + quaternion
wxyz`) for each stick. Quaternion selection uses the same fourfold local-+Y
square-stick symmetry canonicalization as the active Isaac `hand_real` task.
The directed shaft axis remains local `+Y = tail -> tip`.

The scene contains two free 7 x 180 x 7 mm, **10 g each** sticks. Reset center
and directed +Y shaft match Isaac; among the four square-symmetric local-Y
rolls, the variant whose primary-marker +Z points upward (Palm/Base +X) is selected.
Canonicalized policy observation remains equal to the Isaac reset reference.
The hand requests the Isaac pregrasp q; indices 10 and 18 are
explicitly clamped because Isaac's 1.6272 rad values exceed the pinned official
Hand 1 nominal uppers. Four candidate ArUco visuals use DICT_4X4_50 with a
19 mm black boundary. ID0/ID1 belong to Stick1 and ID2/ID3 to Stick2; each
pair is axially staggered with 48-degree roll separation and is flush with the
stick surface. All paper stays at local `Y<=+31.4 mm`, leaving the
`+32..+90 mm` tip/contact region clear. A Palm-Z `-90..0 deg` rendered sweep
found no full-range one-camera layout: Stick1 has zero flush candidates at
`-15 deg` and `0 deg`. The calibrated D435 RGB camera is
1280x720 @ 15 Hz; policy
ticks remain 30 Hz and reuse the latest valid pose between fresh frames.
The external viewer also shows a collision-free D435 body proxy using Intel's
official nominal 90x25x25 mm envelope. Its RGB optical frame is the calibrated
frame; the enclosure is visual-only because the exact optical-to-enclosure CAD
offset is not part of the measured extrinsic.

The viewer scene also contains a first-pass physical testbed. The collision
floor/base plate is `0.630 m (Y) x 0.625 m (Z)`; the 2020 hand post, 4040 camera post,
camera bracket, and D435 enclosure are approximate visual geometry. The exact
calibrated `T_BASE_CAMERA` always places the optical frame--support geometry is
fitted around that pose and never used to recompute it. Palm height is the
explicit temporary constant `HAND_PALM_HEIGHT_TEMP_M = 0.15`, measured from
the plate top (`Base X=0`). The 4040 starts on the lower floor at `X=-0.012 m`.
Base, Palm, and
camera optical frames are shown as RGB axes (red +X, green +Y, blue +Z).

## Commands

Use the existing environment from the project root:

```bash
MUJOCO_PY=/home/lsc/anaconda3/envs/wuji_mujoco/bin/python

$MUJOCO_PY -m Deploy.run.run_policy --inspect-contract
$MUJOCO_PY -m Deploy.run.run_policy --inspect-model
$MUJOCO_PY -m Deploy.run.run_policy --inspect-joints
$MUJOCO_PY -m Deploy.run.run_policy --view-scene
$MUJOCO_PY -m Deploy.run.run_policy --view-camera
$MUJOCO_PY -m Deploy.run.run_policy --validate-fk
MUJOCO_GL=egl $MUJOCO_PY -m Deploy.run.run_policy --inspect-camera
MUJOCO_GL=egl $MUJOCO_PY -m Deploy.run.run_policy --validate-aruco
$MUJOCO_PY -m Deploy.run.run_policy --smoke-backend --policy-steps 30
$MUJOCO_PY -m Deploy.run.run_policy --test-joints
$MUJOCO_PY -m unittest discover -s Deploy/tests -v
$MUJOCO_PY -m Deploy.tools.build_physical_testbed_scene
```

`--onnx-only` and `--run-policy` remain as optional legacy CLI entry points,
but compatibility with an old ONNX contract is not a design requirement.
`--stick-provider synthetic|ground-truth|aruco` selects the provider for policy
modes at the CLI composition root. Synthetic poses validate plumbing only.
The four current marker transforms are explicitly a simulation layout
candidate, not physical calibration. ID1/ID3 must be replaced by transforms
measured after real attachment. `--validate-aruco` checks a deterministic
both-visible diagnostic pose without altering the policy reset contract.

Camera2 is currently a **reset-tail auxiliary candidate**, not a full-range
tracker.  It is level (`0 deg` downward), mounted on the Base/Palm `+Y` side,
and looks horizontally toward `-Y`.  The candidate optical center is Base
`[X,Y,Z]=[0.125,0.200,0.060] m`.  Its 12.5 cm height is the rounded midpoint
of reset ID0/ID2 marker heights (12.10/12.94 cm).  Camera2-only rendered reset
detection of both tail markers passed at this pose; a fixed horizontal camera
does **not** cover the entire Palm-Z `-90..0 deg` sweep.  The previously tested
57.5/60-degree downward arrangements did cover that sweep, but are retained
as comparison results only because a tilted physical mount is undesirable.

To repeat a level-camera mount sweep:

```bash
MUJOCO_GL=egl $MUJOCO_PY -m Deploy.tools.sweep_camera2_mount \
  --down-angle 0 --heights 0.120 0.125 0.130 --side-y 0.15 0.20 0.25
```

Use `--view-scene`, not `python -m mujoco.viewer --mjcf ...`, to inspect the
reset. The raw MJCF contains compile-time placeholders; `--view-scene` creates
the backend first so the Isaac pregrasp and camera-facing symmetry-equivalent
stick reset poses are applied before the GUI opens.
`--view-camera` opens MuJoCo's native viewer locked to the calibrated D435
optical camera, so it works even when OpenCV was installed without HighGUI.
The ArUco provider separately consumes the exact offscreen 1280x720 render.

The generated physical-testbed include is derived from named values in
`scene_contract.py`. The measured plate spans Base `Z=-0.160..+0.465 m`; the
2020 center is at `Y=0, Z=0`, exactly 160 mm from the -Z edge. The 4040 touches
the outside of the +Z edge, so its center is `Z=+0.485 m`, and is offset to
`Y=-0.025 m` (290 mm to the -Y edge, 340 mm to the +Y edge). Profile heights
and bracket shape remain explicitly `TEMPORARY`/`APPROXIMATE`. These fixture
values never modify the calibrated camera or canonical policy frames.

## Contract boundaries

- `OFFICIAL_NOMINAL_PHYSICAL_LIMITS`: pinned description articulation range;
  not connected-hand factory calibration.
- `OBSERVATION_NORMALIZATION_LIMITS`: fixed training/deployment affine range;
  never replaced from runtime hardware values.
- `COMMAND_TARGET_LIMITS`: position-target range with Joint4 indices
  `[3, 7, 11, 15, 19]` floored at zero.
- Action: clip raw output to `[-1, 1]`, then
  `q_target = actual_q_rad + ACTION_SCALE_RAD * action` elementwise, then
  command clamp. `ACTION_SCALE_RAD` is per joint and mirrors Isaac's
  `HAND_REAL_ACTION_SCALE`: Joint3 `0.2`, Joint4 `0.15`, Joint1/Joint2 `0.1`.
  A uniform `0.1` would move Joint3 and Joint4 only 50% and 67% of the trained
  distance. `hand_move`'s uniform `0.1` is a different task and is not deployed
  through this contract.
- Timing: MuJoCo 120 Hz, policy 30 Hz, exactly four held physics steps. A real
  hardware I/O rate is a separate pending value.

The official MJCF armature, joint damping, geometry, mass and inertia are
always preserved, and `ctrlrange` is always narrowed in memory to the canonical
command contract.

The position-servo gains follow `--controller-gains`:

- `deploy` (default): the Isaac-tuned Kp/Kd the policy was actually trained
  against (`DEPLOY_STIFFNESS_NM_PER_RAD` / `DEPLOY_DAMPING_NMS_PER_RAD`, mirroring
  `hand_real_env_cfg.py`'s `HAND_REAL_STIFFNESS` / `HAND_REAL_DAMPING`).
  Sim-to-sim isolates one variable, the physics engine, so MuJoCo must run the
  same commanded controller as training.
- `official`: the pinned vendor MJCF identification (Kp 0.18~0.69, Kv
  0.008~0.031, roughly 4.7 Hz / zeta 0.67 per joint). It is 1.5~7.2x softer than
  the deploy gains and is retained as a deliberate plant-mismatch robustness
  case, not as a baseline.

Effort limits are the official per-joint URDF values in both modes; the MJCF
`forcerange` and Isaac's `HAND_REAL_EFFORT_LIMITS` are identical to them.

Neither mode establishes equivalence to the real firmware controller. Whether
Hand 1 exposes settable Kp/Kd is still `PENDING_REAL_VALIDATION`; if it does
not, the deploy gains describe a hand that does not exist and the sim-to-real
gap lands exactly here.

See [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) for results and unresolved
hardware fields. `ISAAC_POLICY_CONTRACT.md` is retained as a historical audit,
not as the new model/limit/controller source of truth.
