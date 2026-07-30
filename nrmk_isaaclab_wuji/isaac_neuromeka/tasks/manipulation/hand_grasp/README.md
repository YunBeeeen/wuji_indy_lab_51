# Wuji hand chopstick tasks

This package isolates chopstick manipulation from the Indy arm so grasp
geometry, contact topology, and finger control can be tested independently.
It uses Isaac Lab's manager-based environment API.

## Task boundaries

| Gym task | Initial condition | Learned behavior | Observation / action |
|---|---|---|---|
| `hand_grasp` | Functional two-stick grasp | Alternate OPEN and CLOSE commands while retaining the grasp | 103D / 20D |
| `hand_setting` | Open hand and two aligned dynamic sticks | Form and hold the functional grasp in one continuous transition | 101D / 20D |
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

## `hand_setting` reward ladder

The reward intentionally provides signals from coarse approach to completion:

1. weak joint and stick-pose references keep exploration near the verified
   solution without requiring one exact joint configuration;
2. four central-shaft proximity terms guide the semantic finger links before
   contact;
3. six independent saturated contact terms provide partial progress and
   per-contact TensorBoard diagnostics;
4. a hard minimum over all six contacts supplies completion pressure;
5. pose/region-gated completion and low-slip stability reward a physically
   valid setting;
6. the held success term terminates after 30 consecutive valid policy steps.

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
and `min`/`max` are episode extrema. Start diagnosis with:

```text
functional_contact_count        # 0..6
shaft_region_count              # 0..4
stick1_position_error
stick2_position_error
max_linear_speed
max_angular_speed
setting_valid                   # complete per-step success gate
success_stable_steps            # reaches 30 on success
```

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

For the established OPEN/CLOSE task, viewport keyboard control is available:

```bash
python scripts/rsl_rl/play.py \
  --task hand_grasp \
  --load_run <RUN_NAME> \
  --keyboard_hand_mode
```

Focus the Isaac Sim viewport and press `1` for OPEN or `2` for CLOSE.

## Checkpoint compatibility

Always verify the saved `params/env.yaml` before loading:

* `hand_grasp` command-conditioned checkpoints expect 103D observations.
* `hand_setting` checkpoints expect 101D observations and no command.
* Changes to Wuji collision overlays alter contact dynamics even when tensor
  dimensions remain equal; compare those configurations with fresh runs.
