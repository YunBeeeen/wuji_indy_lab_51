# Isaac `hand_real` / `hand_final` policy contract

> Historical audit amended for the active 105D `hand_real` observation. It is
> not the source of truth for the Real-ready model, limits, or controller. Use
> `policy_contract.py`, the pinned official description, and
> `VALIDATION_REPORT.md`. Existing ONNX compatibility is not required.

Status: source-audited on 2026-08-14. Current source code and the inspected ONNX
graph take precedence over older worklog notes. This document describes the
actor interface only; it does not claim task-valid behavior with the frozen
stick fixture used by the first MuJoCo smoke test.

## A. Inheritance and ownership

```text
HandRealEnvCfg
└── HandMoveEnvCfg
    ├── HandMoveActionsCfg       (20D hand action + 0D scripted root term)
    └── HandMoveEnvCfg.__post_init__ (timing)

HandFinalEnvCfg
└── HandObjectEnvCfg
    └── HandMoveEnvCfg
```

`HandFinalEnvCfg` explicitly replaces its actor observations with
`HandRealObservationsCfg` and then calls `apply_hand_real_contract()`. The cube,
cube contact, support, and scripted floating-root machinery inherited from
`hand_object` remain environment/reward machinery and are absent from actor
observations. `hand_real` and `hand_final` therefore have the same 105D actor
input and 20D policy action. The MuJoCo deployment intentionally omits the
scripted 0D root action and fixes the palm to world.

Evidence:

- `nrmk_isaaclab_wuji/isaac_neuromeka/tasks/manipulation/hand_grasp/hand_final_env_cfg.py`
  - `HandFinalEnvCfg`, lines 25-37: inherits `HandObjectEnvCfg`, installs
    `HandRealObservationsCfg`, applies the real contract.
  - module lines 1-7: cube/contact are privileged and not actor inputs.
- `.../hand_object_env_cfg.py`
  - `HandObjectEnvCfg`, lines 505-548: inherits `HandMoveEnvCfg` and changes
    object task details/episode duration, not the hand actor action class.
- `.../hand_real_env_cfg.py`
  - `HandRealEnvCfg`, lines 202-213: inherits `HandMoveEnvCfg`.
  - `apply_hand_real_contract`, lines 178-199: reward orientation semantics and
    task-local finger Kp/Kd replacement.

## B. Exact 105D observation

Observation terms are class attributes in the order below. Isaac Lab's
`ObservationManager.compute_group()` iterates configured terms in order and
concatenates the resulting tensors in that order.

| slice | dim | term | frame/content | history semantics | source |
|---|---:|---|---|---|---|
| `[0:20]` | 20 | `joint_pos_history` first slot | canonical joint order, soft-limit normalized | previous policy-step sample | `hand_real_env_cfg.py:127-132` |
| `[20:40]` | 20 | `joint_pos_history` second slot | same | current sample | same |
| `[40:55]` | 15 | `fingertip_pos` | five tip-link origins, palm-frame xyz | current only | `hand_real_env_cfg.py:133-136`, `mdp.py:4437-4452` |
| `[55:62]` | 7 | `stick1_pose_history` first slot | palm xyz + quaternion wxyz | previous | `hand_real_env_cfg.py` |
| `[62:69]` | 7 | `stick1_pose_history` second slot | same | current | same |
| `[69:76]` | 7 | `stick2_pose_history` first slot | palm xyz + quaternion wxyz | previous | `hand_real_env_cfg.py` |
| `[76:83]` | 7 | `stick2_pose_history` second slot | same | current | same |
| `[83:103]` | 20 | `last_action` | post-wrapper-clipped action stored by ActionManager | action which produced current state | `hand_real_mdp.py` |
| `[103:105]` | 2 | `open_close_mode` | OPEN `[1,0]`, CLOSE `[0,1]` | current command | `mdp.py` |

Total: **105 float values**. Stick quaternions are normalized wxyz and selected
against the reset reference over the four local-+Y square-section symmetries;
the final sign is canonicalized to `w>=0`. The active policy group has corruption disabled and
concatenation enabled (`hand_real_env_cfg.py:161-165`). The saved RSL-RL config
also has observation normalization disabled, so no additional empirical actor
normalizer transforms this vector.

### Joint normalization

Installed Isaac Lab source:

- `/home/lsc/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py`
  - `joint_pos_limit_normalized`, lines 222-236: reads
    `asset.data.soft_joint_pos_limits` for the resolved joint IDs.
- `/home/lsc/IsaacLab/source/isaaclab/isaaclab/utils/math.py`
  - `scale_transform`, lines 27-44.

Exact equation:

```text
center = (soft_lower + soft_upper) / 2
q_normalized = 2 * (q - center) / (soft_upper - soft_lower)
```

The current saved `hand_final` environment has
`soft_joint_pos_limit_factor: 1.0`, so its hand soft limits equal the spawned
articulation limits. Terms keep the dtype of the source tensor (float32 in the
training environment and ONNX interface).

The measured `pose_005` values for `finger3_joint3` and `finger5_joint3`
previously exceeded the common URDF upper limit by `0.002673` and `0.000988`
rad. On 2026-08-14 the active Isaac reset/reference constants and this deploy
contract were aligned to the exact `1.6272 rad` upper limit. Inactive historical
copies were left untouched.

### History ordering and reset

- `/home/lsc/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py`
  - `compute_group`, lines 344-435: appends only when
    `update_history=True`, flattens the history tensor, then concatenates terms.
- `/home/lsc/IsaacLab/source/isaaclab/isaaclab/utils/buffers/circular_buffer.py`
  - `buffer`, lines 79-90: oldest entry first, newest entry last.
  - `append`, lines 112-141: the first append fills every history slot with the
    same sample.
- `/home/lsc/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py`
  - `reset`, lines 370-386: resets managers, forwards physics, then computes
    observations with `update_history=True`.
- `/home/lsc/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
  - `step`, lines 232-238: computes observations once after four physics steps,
    with `update_history=True`.

Therefore every 2-sample history is `[previous, current]`, sampled at policy
steps rather than physics substeps. Immediately after reset both slots equal the
reset sample.

### Fingertips and StickPose7D

Canonical tip order is `finger1_tip_link` through `finger5_tip_link`, declared
with `preserve_order=True` in `hand_grasp_env_cfg.py:58-68`.
`fingertip_positions_in_palm()` subtracts the palm world position and applies
the inverse palm quaternion (`mdp.py:4437-4452`).

The active `hand_real_mdp.py` returns the full Palm-relative rigid-body-root
pose as `xyz+wxyz`. It evaluates the four square-section rotations about local
`+Y`, selects the quaternion nearest the reset reference, normalizes it, and
uses `w>=0` for deterministic sign. Local `+Y` remains tail-to-tip.

## C. Exact action contract

Canonical policy order (`hand_grasp_env_cfg.py:47-56`, resolved with
`preserve_order=True`):

```text
00 finger1_joint1   01 finger1_joint2   02 finger1_joint3   03 finger1_joint4
04 finger2_joint1   05 finger2_joint2   06 finger2_joint3   07 finger2_joint4
08 finger3_joint1   09 finger3_joint2   10 finger3_joint3   11 finger3_joint4
12 finger4_joint1   13 finger4_joint2   14 finger4_joint3   15 finger4_joint4
16 finger5_joint1   17 finger5_joint2   18 finger5_joint3   19 finger5_joint4
```

`SceneEntityCfg` resolves names through `find_joints(..., preserve_order=True)`
(`/home/lsc/IsaacLab/source/isaaclab/isaaclab/managers/scene_entity_cfg.py:148-178`).
The installed string resolver documents and implements query-key order when
that flag is true (`isaaclab/utils/string.py:179-250`).

Action stages:

```text
onnx_action                   = actor graph output
action_manager_action         = clip(onnx_action, -1.0, +1.0)
processed_increment           = scale * action_manager_action + 0
unclamped_target              = q_current + processed_increment
q_target                      = clamp(unclamped_target,
                                      effective_lower,
                                      articulation_soft_upper)
effective_lower[joint4s]      = max(articulation_soft_lower, 0 rad)

scale[*_joint1] = 0.10   scale[*_joint3] = 0.20
scale[*_joint2] = 0.10   scale[*_joint4] = 0.15
```

The scale is per joint, not a single scalar. `HAND_REAL_ACTION_SCALE` in
`hand_real_env_cfg.py` builds the 20-entry dictionary and
`apply_hand_real_contract()` installs it on the inherited `hand_action` term, so
`hand_real`, `hand_final`, and `hand_play` all use it. Joint3 and Joint4 were
retuned against larger PD steps; deploying with a uniform `0.1` would move them
only 50% and 67% of the trained distance. The shared `hand_move` term keeps its
uniform `0.1` and is not deployed through this contract. `ACTION_SCALE_RAD` in
`policy_contract.py` is the deploy-side mirror of this table.

Evidence:

- saved `hand_final/2026-08-13_14-15-09/params/agent.yaml`, line 13:
  `clip_actions: 1.0`.
- `/home/lsc/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`,
  lines 151-156: wrapper clip occurs before `env.step`.
- `hand_move_env_cfg.py`, `HandMoveActionsCfg`, lines 274-302: base scale 0.1,
  limit clamp, five Joint4 zero target floors. The scripted root term has
  `action_dim=0`, so total policy action remains 20.
- `hand_real_env_cfg.py`, `HAND_REAL_ACTION_SCALE` and
  `apply_hand_real_contract()`: per-joint scale override for the deployed tasks.
  Every joint is assigned explicitly because Isaac Lab's `JointAction` defaults
  unlisted dictionary entries to `1.0`.
- saved `hand_real/2026-08-16_19-42-24/params/env.yaml`, `actions.hand_action.scale`:
  the per-joint dictionary as actually trained.
- `isaac_neuromeka/mdp/actions/action_cfgs.py`,
  `CustomResidualJointActionCfg`, lines 74-100.
- `isaac_neuromeka/mdp/actions/joint_actions.py`,
  `CustomResidualJointPositionAction`, lines 197-260: cache soft limits, apply
  floors, compute current-q residual, clamp, call
  `set_joint_position_target()`.
- installed `ActionManager.process_action`,
  `/home/lsc/IsaacLab/source/isaaclab/isaaclab/managers/action_manager.py:372-393`:
  moves old `action` to `prev_action`, stores the just-received action, then
  processes terms.

Important consequence: ONNX output and `last_action` are not always identical.
If ONNX exceeds `[-1,1]`, `last_action` contains the clipped vector accepted by
ActionManager. It is not a q target and not `prev_action`.

## D. Timing

```text
physics dt              1/120 s
physics rate            120 Hz
decimation              4
policy dt               1/30 s
policy rate             30 Hz
history sampling        once per policy step (33.333 ms), after 4 physics steps
target hold              same q target for all 4 physics steps
render interval         4 physics steps
```

Source: `HandMoveEnvCfg.__post_init__`, `hand_move_env_cfg.py:790-799`, and
installed `ManagerBasedRLEnv.step`,
`/home/lsc/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:151-238`.

## E. Task-local Isaac Kp/Kd/effort limits

The values are defined by `HAND_REAL_STIFFNESS`, `HAND_REAL_DAMPING`, and
`HAND_REAL_EFFORT_LIMITS` in `hand_real_env_cfg.py` and installed into the
inherited finger actuator by `apply_hand_real_contract()`.

| joint | Kp | Kd | effort [N.m] |
|---|---:|---:|---:|
| finger1_joint1 | 1.7 | 0.04 | 0.4452 |
| finger1_joint2 | 2.7 | 0.05 | 0.4259 |
| finger1_joint3 | 0.75 | 0.0015 | 0.1888 |
| finger1_joint4 | 1.0 | 0.0015 | 0.1468 |
| finger2_joint1 | 2.4 | 0.055 | 0.6188 |
| finger2_joint2 | 0.7 | 0.02 | 0.1822 |
| finger2_joint3 | 0.75 | 0.0015 | 0.2251 |
| finger2_joint4 | 1.6 | 0.0005 | 0.2170 |
| finger3_joint1 | 2.4 | 0.055 | 0.6494 |
| finger3_joint2 | 0.7 | 0.02 | 0.1827 |
| finger3_joint3 | 0.75 | 0.0015 | 0.2078 |
| finger3_joint4 | 1.3 | 0.0001 | 0.2018 |
| finger4_joint1 | 2.4 | 0.055 | 0.6389 |
| finger4_joint2 | 0.7 | 0.02 | 0.1832 |
| finger4_joint3 | 0.75 | 0.0015 | 0.2249 |
| finger4_joint4 | 1.3 | 0.0001 | 0.2044 |
| finger5_joint1 | 2.4 | 0.055 | 0.6441 |
| finger5_joint2 | 0.7 | 0.02 | 0.1798 |
| finger5_joint3 | 0.75 | 0.0015 | 0.2384 |
| finger5_joint4 | 1.25 | 0.0001 | 0.1866 |

The effort limits are the official per-joint right-hand URDF values (N.m). The
Kp/Kd values are **not** from the vendor: they were tuned by the user directly
in Isaac Sim with joint step inputs, and they are 1.5~7.2x stiffer than the
pinned official MJCF identification.

`policy_contract.py` mirrors all three tables as `DEPLOY_STIFFNESS_NM_PER_RAD`,
`DEPLOY_DAMPING_NMS_PER_RAD` and `DEPLOY_EFFORT_LIMITS_NM`, and
`MujocoWujiHand(controller_gains="deploy")` — the default — installs them into
the compiled MuJoCo position servos by policy index. The MJCF declares zero
generic joint damping already, so nothing is added on top of the configured Kd.
This matches the nominal PD equation and caps, but it does **not** make PhysX,
MuJoCo, and the real firmware dynamics identical.

## F. Legacy ONNX provenance (not compatible with active 105D contract)

Inspected graph:

```text
path: nrmk_isaaclab_wuji/logs/rsl_rl/hand_final/
      2026-08-13_14-15-09/exported/policy.onnx
input:  obs, tensor(float), [1, 101]
output: actions, tensor(float), [1, 20]
dynamic batch: no
custom metadata: none
```

This is the superseded 101D graph. It must not be loaded by the active 105D
deployment contract; a new policy/export is required. The historical
hand_real exported graph had the same old fixed shapes.
The current `play.py` exports after loading its selected checkpoint and writes
`exported/policy.onnx` beside that checkpoint (`scripts/rsl_rl/play.py:1075-1094`).
The installed RSL-RL exporter uses the model's named dummy inputs and ONNX opset
18 (`rsl_rl/runners/on_policy_runner.py:181-203`). Running play again overwrites
the same export file. Because the graph has no checkpoint metadata, its exact
checkpoint cannot be established from ONNX bytes alone; provenance must be
recorded externally at export time. The current export timestamp is
2026-08-14 09:47:57 local and the run contains checkpoints through
`model_400.pt`.

## G. MuJoCo model decision

Two MJCF files were found beside the URDF:

- `wuji_right.xml`: fixed hand but its active 20 actuators are `<motor>` torque
  controls; the position block is commented out. It is unsuitable for direct
  q-target application.
- `right.xml`: fixed palm, exactly 20 hinge joints, and exactly 20
  joint-transmitted `<position>` actuators. This is the selected baseline.

The selected file has no free joint, Indy, stick, cube, camera, table, or plate.
It fuses each fixed `finger*_tip_link` into `finger*_link4`; the adapter uses the
five exact fixed-joint translations from `wuji_right.urdf` to reconstruct the
tip-link origins. The source MJCF timestep is 0.002 s, so the backend changes
the loaded model's in-memory timestep to 1/120 s and leaves the source file
unchanged. Its generic actuator ctrl range is also replaced in memory with each
audited joint range so valid q targets above 1.57 rad are not silently clipped.

## H. Frozen stick fixture limitation

The observation adapter uses the original pose_005 palm-frame pregrasp
references with `STICK1_REFERENCE_AXIAL_SHIFT_M = 0.0`:

```text
Stick1 xyz+axis = [ 0.025074348,  0.024245115, 0.096961208,
                    0.806911411, -0.514751995, 0.289697012 ]
Stick2 xyz+axis = [ 0.035598688,  0.016084217, 0.073366970,
                    0.837511325, -0.428649814, 0.338871831 ]
```

Both history slots stay equal. These values are a **synthetic/frozen deploy
fixture**, not MuJoCo ground truth. The closed loop validates observation/action
plumbing only and cannot validate chopstick or cube task behavior.
