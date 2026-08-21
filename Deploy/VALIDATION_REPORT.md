# Real-ready MuJoCo validation report

Date: 2026-08-16  
Environment: `/home/lsc/anaconda3/envs/wuji_mujoco`  
Source: `wuji-technology/wuji-description` commit
`06e5f14cdd1d5fad0a666ca463a668bf609f9534`, Hand 1 RIGHT

## Architecture migration

Before: an old local `right.xml`, duplicated link4-to-tip offsets, synthetic
sticks inside observation code, and runtime replacement of MuJoCo gains with
Isaac values. It was useful as a policy smoke test but was not a defensible
real-deployment boundary.

After: pinned official URDF/MJCF assets, name-based canonical metadata, three
independent limit tables, explicit tip sites, URDF-parsed standalone FK,
backend and perception protocols, shared observation/action/policy plumbing,
and a deliberately disabled Real backend.

Boundary cleanup: MuJoCo joint names were removed from canonical joint
metadata and moved to the MuJoCo storage mapper. The 120 Hz/four-substep values
and hold loop moved from the canonical contract/CLI rollout into
`mujoco_scheduler.py`. The CLI now composes `PolicyRunner` and
`MujocoScheduler`; it no longer reimplements inference, residual decoding, or
observation history.

Preserved: 20-joint name mapping, action clipping, actual-radian residual,
target clamping, two-sample oldest-to-newest history, duplicated reset sample,
last applied clipped action, OPEN/CLOSE one-hot, finite/shape checks, explicit
mapping failures, and existing CLI entry points.

Removed/replaced: the old MJCF as runtime default, hand-written tip offsets,
and ambiguous single `JOINT_LIMITS`. Isaac Kp/Kd injection was removed in the
2026-08-16 migration and deliberately reinstated on 2026-08-18 as the explicit,
selectable `controller_gains="deploy"` default (see "Model and controller
status"); the 2026-08-16 removal was correct for provenance but wrong for
sim-to-sim, which must hold the commanded controller fixed.

## Model and controller status

- Fixed `right_palm_link`; 20 hinges and 20 position actuators; no Indy7.
- Official mass, inertia, collision/visual meshes, axes, ranges, armature and
  joint damping are retained in every mode.
- The backend sets actuator `ctrlrange` to `COMMAND_TARGET_LIMITS`, and by
  default (2026-08-18) replaces `kp`/`kv` with the Isaac-tuned deploy gains the
  policy was trained against. `controller_gains="official"` restores the vendor
  `kp`/`kv` identification for plant-mismatch sweeps. `forcerange` is the
  official URDF effort in both modes.
- Official default joint armature is `0.0002 kg m^2`, with `0.0005` overrides
  on each Joint1 and thumb Joint2; joint damping is `0.0`. Force ranges and
  identified actuator gains are joint-specific in the official MJCF.
- Simulation `forcerange` is not interpreted as the real hand's current or
  effort unit. Firmware Kp/Kd/equation/saturation remain UNVERIFIED.

## Canonical contract

Policy order is finger-major `finger1_joint1..joint4`, then fingers 2..5.
MuJoCo names are the corresponding `right_fingerN_jointM`. Mapping uses names;
the current numeric addresses happen to be 0..19 but are never assumed.

All three `(20,2)` limit tables are separate, read-only allocations. Physical
and observation values use the official nominal URDF ranges. Command values
match those ranges except Joint4 indices `[3,7,11,15,19]`, whose lower target
is `0 rad`; their physical/observation lower ranges remain negative.

Normalization is `2*(q-center)/(upper-lower)` in canonical actual radians and
is not clipped. Residual action decoding never receives normalized q.

Palm is the official root `right_palm_link`; its local origin and +X/+Y/+Z axes
define the canonical Palm frame. Mount/world transforms are outside the policy
contract. Five massless sites `finger1_tip..finger5_tip` coincide with official
URDF tip-link origins. Site-vs-URDF-FK maximum observed error was
`6.71e-08 m`.

Stick state is pluggable StickPose7D: Palm-relative geometric-center xyz plus
normalized quaternion wxyz. The common quaternion canonicalizer matches
Isaac's four local-+Y square-section symmetries. Synthetic, MuJoCo ground-truth,
and rendered ArUco providers share the same provider boundary.

The deployment scene has two free 7 x 180 x 7 mm sticks at 0.010 kg each.
Reset center and directed +Y shaft match Isaac, while each raw roll is selected
from the four square-symmetric local-Y candidates to put primary ID0/ID2 on
the upward-facing surface. The simulation-only marker layout then places
ID0/ID1 on Stick1 and ID2/ID3 on Stick2 at staggered shaft stations, with each
pair separated by 48 degrees. Marker paper ends by local Y=+31.4 mm, leaving
the distal +32..+90 mm contact region empty, and every marker is flush with its
stick surface. A rendered Palm-Z sweep from -90 to 0 degrees found no complete
one-camera solution: Stick1 had zero candidates at -15 and 0 degrees even when
ID0/ID1, shaft position, and roll were swept. These transforms are not physical
calibration. Fourfold
canonicalization recovers the exact Isaac quaternion
reference. The hand requests the Isaac pregrasp reset, but
policy indices 10 and 18 request 1.6272 rad while pinned official nominal upper
limits are 1.5512 and 1.5490 rad; MuJoCo clamps only those two and reports the
mask rather than changing official physical limits. The D435 calibration snapshot was read from serial
`814412070582` at 1280x720 RGB 15 Hz (firmware `5.17.0.10`); its K and zero
reported distortion coefficients are stored under `assets/`. MuJoCo renders
DICT_4X4_50 markers ID0..ID3 and the provider accepts either marker per stick,
then runs ArucoDetector,
subpixel refinement, IPPE-square candidate selection, marker-to-stick and
camera-to-Palm transforms, raw-pose gates, EMA/SLERP, and HOLD/STALE/LOST.
STALE blocks common PolicyRunner inference and invokes backend safe-stop.
The external scene includes a collision-free D435 enclosure proxy with the
official 90x25x25 mm nominal envelope, rigidly anchored to the calibrated RGB
optical frame. The exact optical-frame-to-enclosure mechanical offset remains
UNVERIFIED and does not affect the rendering camera transform.

The first physical-testbed scene adds a collision-enabled 0.630 m Y by 0.625 m
Z base plate plus visual-only 2020/4040 posts and camera bracket. The plate is
`Z=-0.160..+0.465 m`; the 2020 center is `Y=0,Z=0`, and the outside 4040 center
is `Y=-0.025,Z=+0.485 m` (290/340 mm to the -Y/+Y plate edges). Palm height is
explicitly TEMPORARY at 0.15 m above the plate top (`Base X=0`). The plate top
is 12 mm above the lower floor, and the 4040 starts at that lower floor
(`Base X=-0.012 m`). Profile lengths, bracket dimensions, and the
optical-to-camera-body transform remain approximate. Base/Palm/camera optical XYZ
debug axes and startup transform output make the calibrated and temporary
parts visually and numerically distinguishable.

The physical ID1/ID3 transforms remain pending dual-marker calibration. The
current four transforms are simulation layout candidates and must not be
copied into real deployment configuration.

## Camera2 reset-tail auxiliary mount candidate (2026-08-17)

The user selected a mechanically simple fixed **0-degree downward angle**:
Camera2 is level on the Base/Palm `+Y` side and its optical axis points toward
`-Base Y`.  Its scope is deliberately limited to observing the two reset-tail
markers ID0 and ID2; it is not required to replace Camera1 over the full hand
rotation workspace.

- reset ID0 center in Base: `[0.121009, 0.061799, 0.072649] m`
- reset ID2 center in Base: `[0.129380, 0.049104, 0.047509] m`
- selected optical-center height: Base `X=0.125 m` (rounded midpoint)
- candidate Base pose: `[X,Y,Z]=[0.125,0.200,0.060] m`
- fixed view direction: `-Base Y`; downward angle `0 deg`
- Camera2-only ID0+ID2 detection at reset/Palm-Z `0 deg`: PASS

A full Palm-Z sweep exposed the intentional limitation.  At 5-degree samples
over `-90..0 deg`, the level camera at `X=0.125,Y=0.200 m` saw both tail
markers in `7/19` poses.  Therefore this configuration is VERIFIED only for
the reset-tail auxiliary role, not as a full-range two-stick tracker.

Comparison only: a 57.5-degree downward mount at Base height 0.38--0.40 m and
a 60-degree mount at 0.38--0.43 m produced 19/19 at 5-degree sampling; the
60-degree, 0.40 m candidate also produced 91/91 at 1-degree sampling.  Those
tilted candidates are not selected because mounting them is impractical.

## Executed results

| Check | Result |
|---|---|
| common + MuJoCo + rendered-vision tests (21) | PASS |
| physical testbed dimensions, collision policy, and debug frames | PASS |
| common imports with `mujoco` blocked in isolated process | PASS |
| complete FakeBackend policy tick without MuJoCo | PASS |
| FakeBackend/MuJoCo backend substitution semantics | PASS |
| backend source has no duplicated normalization/residual/action clip | PASS |
| fixed official topology / no Indy7 | PASS |
| unique name-based qpos/dof/actuator mapping (20) | PASS |
| official dynamics preserved except command ctrlrange | PASS |
| three limit tables and Joint4 separation | PASS |
| normalization endpoints/center/no clipping | PASS |
| actual-q residual/action clip/command clamp | PASS |
| explicit tip sites vs standalone URDF FK (5 fingers, 3 poses) | PASS |
| reset/advance history and contiguous finite float32 105D | PASS |
| exact Isaac reset stick pose and 10 g body mass | PASS |
| EGL offscreen 1280x720 calibrated D435 rendering | PASS |
| ID0/ID2 ArUco/IPPE GT comparison in both-visible view: position <5 mm | PASS |
| quaternion geodesic and directed +Y shaft-axis error <5 deg | PASS |
| both-stick pose availability over 20 normal-view perturbations | 20/20 (100%) |
| 15 Hz fresh frame / 30 Hz latest-pose reuse | PASS |
| stale perception command safety gate | PASS |
| one target held for exactly four physics steps | PASS |
| backend smoke (30 policy / 120 physics steps) | PASS |
| single-joint +0.03 rad command, isolation, direction, restore (20/20) | PASS |
| static compile and diff whitespace check | PASS |

## PENDING_REAL_VALIDATION

Connected firmware version, handedness, SDK generation, real joint identifier
order, factory limits, effort/current limits and units, encoder positive
direction, encoder zero offsets, real Palm mounting relation, command-rate
requirement, watchdog, stale-data behavior, firmware position-control equation,
user-settable Kp/Kd, saturation order, temperature/current warnings, and safe
stop behavior. `RealWujiBackend` refuses construction until these are supplied;
it imports no SDK and sends no hardware command.
