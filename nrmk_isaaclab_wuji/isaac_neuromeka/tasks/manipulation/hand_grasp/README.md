# Wuji hand chopstick tasks

This package isolates chopstick manipulation from the Indy arm so grasp
geometry, contact topology, and finger control can be tested independently.
It uses Isaac Lab's manager-based environment API.

## Task boundaries

| Gym task | Initial condition | Learned behavior | Observation / action |
|---|---|---|---|
| `hand_grasp` | Functional two-stick grasp | Alternate OPEN and CLOSE commands while retaining the grasp | 103D / 20D |
| `hand_setting` | Open hand and two aligned dynamic sticks | Form and hold the functional grasp in one continuous transition | 101D / 20D |
| `hand_move` | Functional grasp on a floating, root-controlled hand | Preserve OPEN/CLOSE grasp while the hand pose changes | 103D / 20D |
| `hand_real` | Same simulated task boundary as `hand_move` | Train with deployment-compatible quaternion pose history | 105D / 20D |
| `hand_object` | Functional grasp plus a supported 1 cm cube | Approach, pinch, and retain the cube | 103D / 20D |
| `hand_object_105_distill` | Successful `hand_object` rollout driven by a frozen teacher | Distill 103D velocity policy into the 105D deployable contract | teacher 103D + student 105D / 20D |
| `hand_play` | `hand_object` policy on a table with two plates | Play only; training this ID is refused | 103D / 20D |
| `hand_final_play` | `hand_final` policy on the same table with two plates | Play only; 105D sim-to-real checkpoint inspection | 105D / 20D |
| `finger_reach` | One fingertip and a random target point | Diagnostic mini-reach for PD/action authority | Diagnostic only |
| `hand_grasp_object` | Functional grasp plus target object | Object-interaction scene scaffold | Check the saved run config before loading |

`hand_setting` has no hidden phase or command input. Its 101D observation is:

```text
joint position 20
+ joint velocity 20
+ fingertip position in palm 15
+ two stick poses in palm 14
+ two stick velocities relative to palm 12
+ previous action 20
= 101
```

The shared 20D action is a current-joint residual position command. The action
term converts the policy output into a bounded joint target; it does not make
either stick kinematic or fixed.

## `hand_real` observation contract

`hand_real` inherits `hand_move` physics, reset, action, command, reward,
termination, disturbance, and floating-root controller. Only the actor input is
replaced. It deliberately contains no simulator joint velocity or rigid-body
velocity:

```text
joint position history       40  [q_(t-1), q_t], normalized by joint limits
current fingertip positions  15  five palm-frame xyz values
Stick1 pose history           14  [palm xyz+wxyz]_(t-1), [...]_t
Stick2 pose history           14  [palm xyz+wxyz]_(t-1), [...]_t
last executed action          20  action that produced the current state
OPEN/CLOSE command             2  one-hot
total                        105
```

History is oldest-to-newest. On reset, the first sample is duplicated so the
policy sees zero inferred motion without receiving a fake zero pose. Each Stick
quaternion is palm-frame `wxyz`, normalized, folded across the four physically
equivalent quarter turns about the square stick's local `+y`, selected nearest
the corresponding pose_005 reference, and finally sign-canonicalized with
`w >= 0`. A real vision bridge must reproduce the same marker-to-stick frame,
four-way symmetry fold, and sign convention. The real inference path must
retain the last applied 20D action, compute fingertips from encoder FK, use the
same joint order and limit normalization, and run at the training policy cadence
(currently 30 Hz).

The inherited Stick2 reference reward and OPEN/CLOSE success validator use the
same directed-axis angle in `hand_real`, so they preserve position and shaft
direction without requiring roll. Contact, tip-gap, lateral and axial geometry
remain the original physical training signals.

The task-local residual action scale is `0.1 rad` for Joint1/2, `0.2 rad` for
Joint3, and `0.15 rad` for Joint4 on all five fingers. The 20D order is
unchanged; per-joint `effort_limit_sim` remains the final torque cap.

Simulator-only contact and velocity values may still be used by training
rewards and terminations because they are not actor inputs. A deployment bridge
does not need to reproduce reward signals.

`hand_real` is intentionally incompatible with 103D `hand_move` and 101D
directed-axis-history `hand_real` checkpoints. Joint3/4 also changed from the
old uniform `0.1` action scale to `0.2/0.15`; train it fresh:

```bash
python scripts/rsl_rl/train.py \
  --task hand_real \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000
```

## 103D teacher to 105D student distillation

`hand_object_105_distill` is isolated from the active `hand_object` and
`hand_real` tasks. The frozen
`hand_object/2026-08-08_20-39-52(성공)/model_300.pt` actor receives its exact
old 103D input and drives the rollout. A separate 105D actor sees the current
`hand_real` input and imitates the same physical joint residual. Uniform
teacher actions are converted to the current action units: Joint1/2 `x1`,
Joint3 `x0.5`, Joint4 `x2/3`.

The bridge also freezes the successful run's pose_005 reset, 5.5 s support
retract, five-step retract debounce, and `0.3~0.9 N` disturbance distribution.
Start a fresh supervised run with:

```bash
python scripts/rsl_rl/train.py \
  --task hand_object_105_distill \
  --headless \
  --num_envs 4096 \
  --max_iterations 1000 \
  --init_checkpoint \
  'logs/rsl_rl/hand_object/2026-08-08_20-39-52(성공)/model_300.pt'
```

The distillation checkpoint contains a PPO-compatible `actor_state_dict` for
the student. Continue with fresh critic/optimizer state under current
`hand_real` dynamics:

```bash
python scripts/rsl_rl/train.py \
  --task hand_real \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000 \
  --init_checkpoint \
  'logs/rsl_rl/hand_object_105_distill/<run>/model_<best>.pt' \
  --load_actor_only
```

Distillation removes simulator velocity from the deployed actor input, but it
is not itself a hardware-safety guarantee. PPO fine-tuning, vision-noise
validation, command/joint clamps, current limits, and temperature shutdowns
remain required.

## Reset semantics

The `hand_setting` reset restores two parallel, dynamic sticks and an open
hand. The thumb's second joint starts at `-0.1659 rad`, the measured open
configuration used during the manual pose search. The simulated joint state
and PD target receive the same value, so this is an initial posture rather
than the old target/state mismatch that produced contact preload.

Treat the value as a phase-boundary assumption. To test robustness, compare it
against a zero reset or a controlled reset distribution without changing the
reward at the same time.

## Functional contact topology

The validated setting uses six semantic contact groups:

1. thumb distal link (`finger1_link3`) to Stick1
2. index fingertip to Stick1
3. middle fingertip to Stick1
4. palm to Stick2
5. thumb middle link to Stick2
6. ring fingertip to Stick2

The four narrow finger contacts must also lie on the central 160 mm of the
corresponding shaft. This prevents end-cap reward farming without prescribing
a surface normal or one exact point. Palm and thumb-middle contacts are broad
valley anchors and are checked through their pair forces plus the final Stick2
palm-relative pose.

## `hand_setting` paired-reference + thumb-pivot stage

The active `hand_setting` reset directly assigns task-local absolute world
positions `(0.075, 0, 0.5195)` and `(0.055, 0, 0.5195) m` to
Stick1/Stick2. These are the positions originally validated by shifting the
inherited scene spawn `20 mm`, but the current code stores the final values
without an offset calculation. Their orientation and `20 mm` center separation
remain unchanged. The open hand starts with `finger1_joint2=-0.1659 rad`, all
other joints zero, and simulated state equal to PD target; no grasp preload is
injected. This reset is task-local and does not modify `hand_grasp`.

At reset the six configured semantic pair forces are all zero. In a 5 s
zero-action probe the sticks settled on the upward-facing hand without crossing
the `0.40 m` drop threshold; Stick2 developed palm support while the monitored
thumb-middle contact remained zero.

The task jointly tracks the two stable `hand_grasp/pose_005` palm-relative
stick poses, then brings the thumb distal link toward Stick1's local
`y=-60 mm` pivot station.

**The stage gate and the tracking reward deliberately use different sigmas.**
The gate stays loose so acquisition can unlock at all; the reward is tight so
the reset posture does not already score highly and leave the term without a
usable gradient. Do not collapse these into one number.

```text
gate kernel (SETTING_STAGE1_GATE_PARAMS)
  position sigma     0.10 m
  orientation sigma  1.5708 rad (90 deg)
  thumb sigma        0.020 m

  stick1_score = exp(-stick1_position_error / 0.10
                     -stick1_orientation_error / 90 deg)
  stick2_score = exp(-stick2_position_error / 0.10
                     -stick2_orientation_error / 90 deg)
  pair_score   = min(stick1_score, stick2_score)
  thumb_score  = exp(-thumb_to_stick1_pivot_station_distance / 0.020)

  stage1_ready = pair_score >= 0.65 and thumb_score >= 0.35
  stage2_ready = stage1_ready and max_i |q_i - q_ref_i| <= 0.0873 rad (5 deg)

reward kernel (two_stick_reference_min / reference_thumb_pivot_min)
  position sigma     0.010 m
  orientation sigma  0.25 rad
  thumb sigma        0.060 m
```

`stage1_ready` is memoryless and re-evaluated every step. `stage2_ready` is the
strict all-twenty maximum-error test; the contact terms below fade in on a
separate RMSE ramp rather than waiting for it.

Active reward terms:

```text
two_stick_reference_min           12    min(stick1, stick2) on the tight kernel
reference_thumb_pivot_min          8    min(pair_score, thumb_score)
stage1_joint_reference             8    all-20 prior, joint_sigma 0.80
stage1_missing_joint_reference    12    16 non-thumb joints, mean/min, sigma 0.80
stage1_missing_joint_best_so_far 3000   per-joint best-so-far progress
stage1_semantic_surface_approach   2    live restoring proximity
stage1_index_between               4    index tip between the two shafts
stage2_contact_mean                5    six semantic pair forces, force scale 0.10 N
stage2_contact_min                20    weakest of the six, force scale 0.10 N
```

Stage-2 handover changes the effective weights without any latch or stored
phase:

```text
stage1_joint_reference          8 -> 2   weight_scale = 1 - 0.75 * stage2_ready
stage1_missing_joint_reference 12 -> 0   ratio 0.0, returns if the gate fails
stage2 contact gate = stage1_ready * clip((0.80 - q_RMSE)/(0.80 - 0.0873), 0, 1)
```

`stage1_missing_joint_best_so_far` telescopes, so its `3000` is not comparable
with the per-step weights above. It pays at most one normalized score unit per
episode, which at 30 Hz is roughly `100` reward points. The little finger's
four joints carry `0.25` progress credit instead of `1.0` so it does not close
across the ring finger's route to Stick2; internal normalization keeps the
term's total budget unchanged. The little finger remains at full weight in the
all-20 prior and in the strict Stage-2 gate.

`stage1_semantic_surface_approach` and `stage1_index_between` both use a
one-way Stage-1 unlock, so a transient gate drop does not delete their signal:

```text
semantic approach:
  score_i = clip(1 - surface_distance_i / 0.08, 0, 1)
            index tip -> Stick1, middle tip -> Stick1, ring tip -> Stick2
  reward  = 2 * stage1_unlocked * (0.5 * mean(score_i) + 0.5 * min(score_i))

index between:
  between coordinate  Stick1 centerline 0, Stick2 centerline 1
  slab score          higher between the two centerlines
  stick1 shaft score  higher near the assigned Stick1 surface/shaft
  between_score       = min(slab score, stick1 shaft score)
  pair_maintenance    = clip(pair_score / 0.65, 0, 1)
  reward  = 4 * stage1_unlocked * pair_maintenance * between_score
```

The live `pair_maintenance` factor is what stops the policy from wrecking both
stick poses just to park the index finger between them.

Parked terms are kept as commented blocks for one-line restoration rather than
registered at `weight=0.0`, because a zero weight drops the term from the
Reward Manager entirely:

```text
two_stick_reference_fine_min   6   sigma 0.002 m / 0.05 rad
index_wrong_stick2_contact    -2   index link/tip touching Stick2 outside between
success reward / termination  30000
```

The fine-min kernel was run as an A/B on `2026-08-11_13-18-48` and
`2026-08-11_14-50-02` against the `2026-08-11_12-20-46` coarse baseline. The
thumb became visibly more hesitant and Stage-1 entry itself got worse, so it is
off. Success is parked deliberately: ending the episode on success cuts off
collection of the remaining positive shaping and can make success less
attractive than holding.

The index-between term is not part of the Stage-2 hard gate. It is a
reward-only single-variable experiment; `SETTING_STAGE2_PARAMS` keeps its
index arguments commented so readiness is unchanged.

No explicit drop penalty is active. Either stick crossing world `z=0.40 m`
terminates the episode, which removes the remaining opportunity to collect
future non-negative rewards. The gated joint prior and contact mean bridge the
open-hand acquisition that `hand_grasp` did not need, while contact min matches
its six-contact completion pressure.

`HandSettingSceneCfg` defines five index-to-Stick2 wrong-contact sensors, but
`HandSettingEnvCfg.scene` is deliberately the plain `HandGraspSceneCfg` during
this A/B. Editing those sensors has no effect until the scene assignment and
the parked penalty term are both restored.

TensorBoard exposes the same geometry the reward uses, in all four
hand-setting metric families:

```text
stage1_pair_score           # weaker Stick1/Stick2 score on the gate kernel
stage1_ready                # live hard-gate fraction
stage1_unlocked             # one-way latch fraction
thumb_pivot_distance        # meters; lower is better
thumb_pivot_score           # exp(-distance / 0.020)
all_joint_reference_rmse    # drives the stage-2 contact ramp
all_joint_reference_max_error
all_joint_within_5deg
stage2_ready                # strict maximum-error gate
stage2_contact_progress     # the ramp value actually multiplying contact
missing_joint_best_score
semantic_approach_mean_score / _min_score / _score
index_between_coordinate
index_between_slab_score
index_between_stick1_shaft_score
index_between_score
```

The previous Stick2-valley finite-shaft/`-60 mm` proxy remains in source for
reproduction but is not used by the active reward, gate, or metrics.

## Historical valley experiments

The first reward-ladder run (`2026-07-30_17-31-40`) learned the easy
joint/stick references and broad regions while functional contact stayed near
`1/6`.  The contact-only follow-up (`2026-07-30_18-14-08`) instead saturated
the three reset-adjacent contacts and never exceeded `3/6`.  A visual replay
then exposed the ordering error: the thumb-side pivot closed before Stick2
entered the thumb-index valley.

The geometry-only follow-up `2026-07-30_22-59-53` also failed to enter the
valley. Its best episode extrema were about `26.1 mm` point error and
`45.6 deg` shaft-axis error; the geometry gate remained zero through iteration
4784. Palm and thumb-middle forces sometimes existed outside the valley, so
body-pair contact alone could not distinguish a seated anchor from an outside
push.

The previous finite-shaft design separated approach shaping from final
validation:

- the dense centerline reward (`weight=12`) uses the Stick2 local `y=-60 mm`
  point from `pose_005` only to recover a fixed palm-frame valley target;
- the current Stick2 score is the shortest distance from that target to its
  finite 180 mm centerline, so no particular current shaft station must equal
  `-60 mm` and axial sliding remains free during insertion;
- its score is the minimum of the point and axis exponentials, so the easier
  component cannot compensate for abandoning the harder one;
- inside a loose `20 mm / 30 deg` corridor, `valley_anchor_support`
  (`weight=12`) rewards the weaker of palm--Stick2 and
  thumb-middle--Stick2 forces;
- the strict `stick2_in_valley` gate requires `10 mm / 15 deg` geometry and
  both anchor forces at least `0.02 N`;
- in the active valley-only A/B, every Stick1, ring, six-contact, success, and
  action-rate reward has weight zero. Those terms remain defined only to keep
  their TensorBoard diagnostics.

In that historical A/B, the loose corridor prevented an outside hand contact
from earning an annuity,
while still exposing force shaping before exact seating. The only nonzero
weights are `stick2_valley_approach=12` and `valley_anchor_support=12`.
`stick2_seated` means strict in-valley plus ring support, but is diagnostic
only in this A/B. The experiment intentionally does not learn the Stick1 pivot
or final six-contact grasp yet.

The Stage-1/Stage-2 structure documented above supersedes that historical
valley-only A/B. Two details of the intermediate version are recorded here
because older runs were produced under them:

- `stage1_ready` used `pair_score >= 0.50`; the current threshold is `0.65`,
  raised to reject a premature-contact solution seen around `0.57` while still
  admitting the observed acquisition state near `0.66`;
- the joint-reference fade was driven by the weakest semantic contact,
  `m = min_i(clamp(F_i / 0.02, 0, 1))`, giving effective weight
  `8 * (1 - 0.75 * m)`. The current fade is driven by the strict Stage-2 joint
  gate instead, so it is `8` until all twenty joints are within `5 deg` of
  `pose_005` and `2` after. Both versions are latch-free: losing the condition
  restores the stronger guide automatically.

The strict held-success reward/termination remains the final validator.
It is intentionally stricter than `stick2_seated` and still requires the full
functional grasp for 30 consecutive policy steps.

Success requires all six contact forces to be at least `0.02 N`, all four
narrow contacts to be in their shaft regions, Stick1/Stick2 pose errors to be
inside their configured tolerances, and both sticks to remain below the
palm-relative speed limits.

## TensorBoard diagnostics

`CustomRewardManager` writes the same active success definitions under:

```text
Metrics/hand_setting/
Metrics/hand_setting_final/
Metrics/hand_setting_min/
Metrics/hand_setting_max/
```

The base family is the episode time average, `final` is the terminal sample,
and `min`/`max` are episode extrema. Read the stage gates before the reward
values: a term that looks dead is usually a gate that never opened.

```text
stage1_ready / stage1_unlocked   # acquisition; check this first
stage1_pair_score
all_joint_reference_rmse         # drives stage2_contact_progress
stage2_contact_progress          # how much contact shaping is actually on
stage2_ready                     # strict all-20 within 5 deg
functional_contact_count         # 0..6
shaft_region_count               # 0..4
stick1_position_error
stick2_position_error
stick2_orientation_error
thumb_joint2_position
thumb_joint2_target
thumb_joint2_action
max_linear_speed
max_angular_speed
setting_valid                    # complete per-step success gate
success_stable_steps             # reaches 30 on success
```

The `stick2_valley_pose_valid`, `stick2_in_valley` and `stick2_seated` tags
belong to the historical valley A/B below. They are still written, but no
active reward or gate consumes them.

Individual force and region-score tags identify which semantic contact is
blocking completion. Metric-only code changes do not alter checkpoint shapes,
but a running Python process must be restarted to load them.

## Commands

From the package repository:

```bash
python scripts/rsl_rl/train.py \
  --task hand_setting \
  --headless \
  --num_envs 4096 \
  --max_iterations 50000
```

To test action authority without training or changing the reward, drive only
`finger1_joint2` while all other 19 actions remain zero:

```bash
python scripts/debug/hand_setting_thumb_action_probe.py \
  --task hand_setting \
  --headless
```

The probe sweeps raw magnitudes `0.1, 0.25, 0.5, 0.75, 1.0`, repeatedly
closes/reopens the joint, and saves target/actual joint motion, valley errors,
six contact forces, and stick speeds under
`logs/debug/hand_setting_thumb_action_probe/`.

For the established OPEN/CLOSE task, viewport keyboard control is available:

```bash
python scripts/rsl_rl/play.py \
  --task hand_grasp \
  --load_run <RUN_NAME> \
  --keyboard_hand_mode
```

Focus the Isaac Sim viewport and press `1` for OPEN or `2` for CLOSE.

For `hand_play`, root rotation uses `A/S`, `D/Z`, `X/C`; translation uses the
arrow/Page keys or `I/K`, `J/L`, `U/O`. An external per-link contact-force plot
can be opened without blocking the simulation loop:

```bash
python scripts/rsl_rl/play.py \
  --task hand_play \
  --load_run <HAND_OBJECT_RUN> \
  --manual_root \
  --plot_hand_contact_forces
```

For a 105D `hand_final` checkpoint, use the dedicated compatible sibling:

```bash
# Latest run and latest checkpoint under logs/rsl_rl/hand_final:
python scripts/rsl_rl/play.py \
  --task hand_final_play \
  --num_envs 1 \
  --manual_root

# Latest checkpoint in a selected run (a timestamp prefix is sufficient):
python scripts/rsl_rl/play.py \
  --task hand_final_play \
  --num_envs 1 \
  --manual_root \
  --load_run 2026-08-13_14-15-09

# A particular checkpoint in that run:
python scripts/rsl_rl/play.py \
  --task hand_final_play \
  --num_envs 1 \
  --manual_root \
  --load_run 2026-08-13_14-15-09 \
  --checkpoint model_300.pt
```

An absolute/relative checkpoint path is still accepted. A bare checkpoint
filename is interpreted inside the run selected by `--load_run` (or inside the
latest matching run when `--load_run` is omitted).

The curves are PhysX net contact-force magnitudes for each palm/finger rigid
link, not actuator torque or root-controller wrench.

## Checkpoint compatibility

Always verify the saved `params/env.yaml` before loading:

* `hand_grasp` command-conditioned checkpoints expect 103D observations.
* `hand_setting` checkpoints expect 101D observations and no command.
* `hand_move` and `hand_object` checkpoints expect 103D observations.
* `hand_object_105_distill` checkpoints contain a 105D student actor plus the
  frozen 103D teacher; use the student `actor_state_dict` for PPO fine-tuning.
* Current `hand_real` checkpoints expect 105D observations and must be trained
  fresh rather than initialized from a 101D or 103D actor.
* Changes to Wuji collision overlays alter contact dynamics even when tensor
  dimensions remain equal; compare those configurations with fresh runs.
