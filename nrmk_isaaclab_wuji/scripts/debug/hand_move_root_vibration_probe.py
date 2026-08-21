"""hand_move palm/root vibration diagnosis (2026-08-06).

Answers one question: **is the observed palm jitter a numerical instability of
the root PD controller, or a real physical oscillation?**

Why a dedicated probe
---------------------
The suspected failure mode flips the applied torque sign on *every physics
step* (120 Hz, Nyquist 60 Hz).  ``env.step()`` advances ``decimation`` physics
steps and only exposes the final state, i.e. a 30 Hz sample - which aliases a
120 Hz oscillation into noise.  TensorBoard metrics and ``play.py`` have the
same problem, and the GUI renders at 30 fps too.  So this probe reads the
physics-rate trace buffer that ``HandRootHoldAction`` fills inside
``apply_actions``.

Verdict logic
-------------
For the palm angular velocity and the applied torque, at physics rate:

* **lag-1 autocorrelation r1 close to -1** and a sign-flip rate near 1.0
  -> alternating every step -> numerical instability of the explicit PD.
  The theory: an explicit damping term is unstable once
  ``(I_assumed / I_true) * orientation_kd * dt > 2``.
* **r1 clearly positive**, sign-flip rate well below 0.5
  -> a smooth low-frequency oscillation -> physical resonance or a real
  disturbance the controller is reacting to, not a numerical artefact.

``finger_joint_vel_norm`` is recorded alongside so "the fingers are shaking the
palm" can be separated from "the controller is shaking the palm".

Usage
-----
Run only while no training is in progress (project rule).

The post-convergence window (t >= 4 s) is only reachable with a **trained
policy**: it is what holds the grasp together.  With a constant finger action
the sticks fall out around t = 1 s and the episode ends before the rotation
even starts, so ``--load_run`` is the normal way to use this script.

    # what the palm actually does after the rotation converges
    python scripts/debug/hand_move_root_vibration_probe.py --headless \
        --load_run 2026-08-06_00-12-28 --tag goal_kd40

    # same policy, one gain changed - single-variable comparison
    python scripts/debug/hand_move_root_vibration_probe.py --headless \
        --load_run 2026-08-06_00-12-28 --orientation_kd 10 --tag goal_kd10

Drop ``--headless`` to watch it in the GUI while it records; the trace is
filled inside ``apply_actions`` so it is unaffected by who steps the env.

Each run writes ``result.json`` under
``logs/debug/hand_move_root_vibration/<timestamp>/`` so runs can be compared
afterwards.  The env is rebuilt from the *current* source, not from the
checkpoint's saved config, so check that the gains and the goal range still
match the run being replayed.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="hand_move")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--policy_steps",
    type=int,
    default=0,
    help="policy steps to record (0 = the whole episode, analysed per phase)",
)
parser.add_argument("--effective_inertia", type=float, default=None)
parser.add_argument("--orientation_kp", type=float, default=None)
parser.add_argument("--orientation_kd", type=float, default=None)
parser.add_argument(
    "--finger_grip",
    type=float,
    default=0.0,
    help="constant residual action applied to every finger joint. 0 means no"
    " grip force at all (target = current joint angle), which is why the"
    " sticks fall out within a second. A positive value holds the target"
    " ahead of the measured angle, giving a steady preload of about"
    " kp * scale * value per joint - the documented stand-in for what a"
    " trained policy produces.",
)
parser.add_argument(
    "--pin_sticks",
    action="store_true",
    help="disable gravity on both sticks so they cannot fall out of a passive"
    " grasp. This is what makes the post-convergence window reachable at all:"
    " with a zero finger action the sticks drop after about a second and the"
    " episode ends long before t = 4 s. The sticks keep their inertia and their"
    " contacts, they just have no weight, so the palm still sees them - only"
    " the static gravity load is removed. Same technique as"
    " scripts/debug/grip_capacity.py; no per-step teleporting.",
)
parser.add_argument(
    "--load_run",
    type=str,
    default=None,
    help="run folder under logs/rsl_rl/hand_move to load the policy from."
    " A trained policy is what keeps the grasp alive long enough to reach the"
    " post-convergence window at all - with a constant finger action the"
    " sticks fall out around t = 1 s.",
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="explicit path to a .pt checkpoint"
)
parser.add_argument(
    "--load_checkpoint", type=str, default="model_.*.pt", help="checkpoint name pattern"
)
parser.add_argument(
    "--tag", type=str, default="", help="suffix appended to the output folder"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402


def load_policy(env, task: str):
    """Load a trained rsl_rl policy, mirroring ``scripts/rsl_rl/play.py``.

    Returns ``(wrapped_env, policy, resume_path)``.  The returned env is the
    rsl_rl wrapper: step through it, but read scene/manager state through
    ``wrapped_env.unwrapped``.
    """
    import importlib.metadata as metadata

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    # Required: without it rsl-rl-lib 5.x rejects the deprecated ``policy=``
    # field that this repository's runner cfgs still use (see agent.md).
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, metadata.version("rsl-rl-lib")
    )

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(
            log_root, args_cli.load_run, args_cli.load_checkpoint
        )

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
    return wrapped, policy, resume_path


def lag1_autocorrelation(signal: torch.Tensor) -> float:
    """Pearson correlation between a signal and itself shifted by one sample."""
    if signal.numel() < 3:
        return float("nan")
    x = signal - signal.mean()
    denominator = float((x * x).sum())
    if denominator < 1.0e-20:
        return float("nan")
    return float((x[:-1] * x[1:]).sum() / denominator)


def sign_flip_rate(signal: torch.Tensor, dead_zone: float) -> float:
    """Fraction of consecutive samples whose sign differs.

    Samples inside ``dead_zone`` are ignored so numerical noise around zero
    does not masquerade as an oscillation.
    """
    active = signal[signal.abs() > dead_zone]
    if active.numel() < 3:
        return float("nan")
    return float((torch.sign(active[:-1]) != torch.sign(active[1:])).float().mean())


def dominant_frequency(signal: torch.Tensor, dt: float, dead_zone: float) -> float:
    """Rough frequency estimate from zero crossings, in Hz."""
    active = signal[signal.abs() > dead_zone]
    if active.numel() < 3:
        return float("nan")
    crossings = int((torch.sign(active[:-1]) != torch.sign(active[1:])).sum())
    duration = active.numel() * dt
    return crossings / (2.0 * duration) if duration > 0 else float("nan")


def moving_average(values: torch.Tensor, window: int) -> torch.Tensor:
    """Centred box filter along dim 0, edges held. Used to strip the jitter."""
    if window < 2:
        return values
    pad = window // 2
    padded = torch.cat(
        [values[:1].expand(pad, -1), values, values[-1:].expand(pad, -1)], dim=0
    )
    kernel = torch.ones(1, 1, window, device=values.device) / window
    smoothed = torch.nn.functional.conv1d(
        padded.T.unsqueeze(1), kernel
    ).squeeze(1).T
    return smoothed[: values.shape[0]]


def angular_speed_penalty(
    stick1_ang_vel: torch.Tensor,
    stick2_ang_vel: torch.Tensor,
    palm_ang_vel: torch.Tensor,
    limit: float = 3.0,
    max_excess: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce ``object_pair_angular_speed_excess_l2`` exactly.

    ``speed = max_i |w_stick_i - w_palm|``, ``excess = clamp(speed - limit)``,
    raw reward term = ``excess**2``.  Returns ``(speed, raw)``.
    """
    speed = torch.maximum(
        (stick1_ang_vel - palm_ang_vel).norm(dim=-1),
        (stick2_ang_vel - palm_ang_vel).norm(dim=-1),
    )
    excess = (speed - limit).clamp(min=0.0, max=max_excess)
    return speed, excess.square()


def analyse(values: torch.Tensor, label: str, dt: float) -> dict:
    """Per-axis and magnitude statistics for a (n, 3) signal."""
    magnitude = values.norm(dim=-1)
    # Analyse the axis that actually carries the oscillation.
    axis = int(values.abs().mean(dim=0).argmax())
    signal = values[:, axis]
    dead_zone = 0.02 * float(signal.abs().max()) if float(signal.abs().max()) > 0 else 0.0
    result = {
        "label": label,
        "dominant_axis": "xyz"[axis],
        "rms": float(magnitude.pow(2).mean().sqrt()),
        "peak": float(magnitude.max()),
        "mean": float(magnitude.mean()),
        "lag1_autocorrelation": lag1_autocorrelation(signal),
        "sign_flip_rate": sign_flip_rate(signal, dead_zone),
        "dominant_frequency_hz": dominant_frequency(signal, dt, dead_zone),
    }
    return result


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    # ``--task hand_grasp`` is supported on purpose: it is the control
    # experiment for "did the floating root break the grasp, or does a zero
    # finger action simply never hold the sticks?".  hand_grasp has no root
    # action term, so tracing is skipped and only the drop time is reported.
    root_cfg = getattr(env_cfg.actions, "root_action", None)
    if root_cfg is not None:
        root_cfg.trace_enabled = True
        root_cfg.trace_env_id = 0
        if args_cli.effective_inertia is not None:
            root_cfg.effective_inertia = args_cli.effective_inertia
        if args_cli.orientation_kp is not None:
            root_cfg.orientation_kp = args_cli.orientation_kp
        if args_cli.orientation_kd is not None:
            root_cfg.orientation_kd = args_cli.orientation_kd

    orientation_cfg = getattr(env_cfg.commands, "root_orientation", None)
    schedule = orientation_cfg.schedule if orientation_cfg is not None else None

    raw_env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    use_policy = bool(args_cli.load_run or args_cli.checkpoint)
    policy = None
    resume_path = None
    if use_policy:
        stepper, policy, resume_path = load_policy(raw_env, args_cli.task)
        # Same as play.py: the wrapper has already reset the env.
        obs = stepper.get_observations()
    else:
        stepper = raw_env
        raw_env.reset()
    env = raw_env

    root_term = None
    if root_cfg is not None:
        root_term = env.action_manager.get_term("root_action")
        root_term.reset_trace()

    if args_cli.pin_sticks:
        indices = torch.arange(env.num_envs)
        disabled = torch.ones(env.num_envs, dtype=torch.uint8)
        for stick_name in ("stick1", "stick2"):
            env.scene[stick_name].root_physx_view.set_disable_gravities(
                disabled, indices
            )

    policy_steps = args_cli.policy_steps
    if policy_steps <= 0:
        # Whole episode.  The interesting operating point is *after* the
        # rotation: from t = 4 s the palm has to hold a new orientation while
        # the gravity-loaded sticks keep pulling on it, so the steady-state
        # disturbance torque is far larger than during the initial hold, where
        # the target equals the current pose and the controller is idle.
        episode_length = schedule.episode_length_s if schedule else env.cfg.episode_length_s
        policy_steps = int(round(episode_length / env.step_dt))

    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim), device=env.device
    )
    # The finger residual occupies the first slice of the action vector; the
    # root term (if present) reports action_dim 0, so this covers exactly the
    # 20 finger joints.
    finger_dim = env.action_manager.get_term("hand_action").action_dim
    action[:, :finger_dim] = args_cli.finger_grip

    print("\n=== hand_move root vibration probe ===")
    print(f"  task              = {args_cli.task}")
    if root_cfg is not None:
        print(f"  effective_inertia = {root_cfg.effective_inertia:.3e}")
        print(f"  orientation_kp/kd = {root_cfg.orientation_kp} / {root_cfg.orientation_kd}")
    else:
        print("  root controller   = (none - fixed-base control task)")
    print(f"  finger_grip       = {args_cli.finger_grip}"
          f"{'  <-- ZERO: no grip force' if args_cli.finger_grip == 0.0 else ''}")
    if use_policy:
        print(f"  policy            = {resume_path}")
    else:
        print("  policy            = (none - constant finger action)")
    print(f"  pin_sticks        = {args_cli.pin_sticks}"
          f"{'  (stick gravity off, so they cannot drop out)' if args_cli.pin_sticks else ''}")
    if not use_policy and args_cli.finger_grip == 0.0 and not args_cli.pin_sticks:
        print("  WARNING: no grip force and unpinned sticks -> expect the"
              " episode to end near t = 1 s, before the rotation.")
    print(f"  physics_dt        = {env.physics_dt:.6f} s ({1/env.physics_dt:.0f} Hz)")
    print(f"  step_dt           = {env.step_dt:.6f} s ({1/env.step_dt:.0f} Hz)")
    print(f"  recording         = {policy_steps} policy steps"
          f" ({policy_steps * env.cfg.decimation} physics samples)")
    if schedule is not None:
        print(f"  goal range        = x{schedule.range_x} y{schedule.range_y} z{schedule.range_z}")

    # Verify the gravity override actually took effect instead of assuming it.
    if args_cli.pin_sticks:
        view = env.scene["stick1"].root_physx_view
        getter = getattr(view, "get_disable_gravities", None)
        if getter is not None:
            print(f"  stick1 disable_gravities readback = {getter().tolist()}")
        else:
            print("  (no get_disable_gravities on this view; effect verified via"
                  " the stick height trace below)")

    saw_reset = False
    steps_done = 0
    fired_terms: list[str] = []
    stick_z_min = float("inf")
    stick_z_trace: list[float] = []
    for _ in range(policy_steps):
        previous = int(env.episode_length_buf.max().item())
        if use_policy:
            with torch.inference_mode():
                obs, _, _, _ = stepper.step(policy(obs))
        else:
            stepper.step(action)
        steps_done += 1
        z = min(
            float(env.scene["stick1"].data.root_pos_w[:, 2].min()),
            float(env.scene["stick2"].data.root_pos_w[:, 2].min()),
        )
        stick_z_min = min(stick_z_min, z)
        stick_z_trace.append(z)
        if int(env.episode_length_buf.max().item()) < previous:
            # Report *which* termination fired instead of guessing.  The flags
            # computed just before the reset are still in the manager.
            for name in env.termination_manager.active_terms:
                try:
                    if bool(env.termination_manager.get_term(name).any()):
                        fired_terms.append(name)
                except (KeyError, AttributeError):
                    pass
            saw_reset = True
            break

    if root_term is None:
        print("\n  (no root action term: this run only measures how long the"
              " grasp survives)")
        env.close()
        return

    trace = root_term.trace.clone().cpu()
    dt = env.physics_dt
    print(f"\n  recorded {trace.shape[0]} physics samples"
          f" ({trace.shape[0] * dt:.2f} s)")
    survived_s = None
    if saw_reset:
        survived_s = steps_done * env.step_dt
        print(f"  *** EPISODE ENDED AT t = {survived_s:.2f} s"
              f" (stick drop or time-out) ***")
        if schedule is not None and survived_s < schedule.initial_hold_time_s:
            print("      The grasp failed before the rotation even started, so"
                  " nothing here says anything about the rotation controller.")
    if trace.shape[0] < 8:
        print("  [FAIL] trace is empty - is trace_enabled wired through?")
        env.close()
        return

    columns = {name: i for i, name in enumerate(root_term.TRACE_COLUMNS)}
    ang_vel = trace[:, columns["ang_vel_x"] : columns["ang_vel_x"] + 3]
    torque = trace[:, columns["torque_x"] : columns["torque_x"] + 3]
    orientation_error = trace[
        :, columns["orientation_error_x"] : columns["orientation_error_x"] + 3
    ]
    lin_vel = trace[:, columns["lin_vel_x"] : columns["lin_vel_x"] + 3]
    finger_vel = trace[:, columns["finger_joint_vel_norm"]]

    # -- split into phases --------------------------------------------------
    # With a zero finger action the OPEN/CLOSE command moves nothing, so the
    # whole post-settling stretch is one long "hold at the new orientation".
    phase_bounds = [
        ("initial_hold  (q_cmd = q_start)", 0.0, schedule.initial_hold_time_s),
        ("slerp         (rotating)", schedule.initial_hold_time_s, schedule.slerp_end_time_s),
        ("hold_at_goal  (q_cmd = q_goal)", schedule.slerp_end_time_s, schedule.episode_length_s),
    ]
    sample_time = torch.arange(trace.shape[0], dtype=torch.float) * dt

    def verdict_for(torque_report: dict, ang_report: dict) -> tuple[str, str]:
        r1 = torque_report["lag1_autocorrelation"]
        flip = torque_report["sign_flip_rate"]
        if ang_report["rms"] < 1.0e-3 and torque_report["rms"] < 1.0e-3:
            return "QUIET", "the palm is essentially still in this window."
        if r1 == r1 and r1 < -0.5 and flip > 0.5:
            return "NUMERICAL", (
                f"torque alternates sign nearly every physics step (r1={r1:.3f},"
                f" flip {flip*100:.0f}%) - explicit-damping instability."
                " Reduce effective_inertia * orientation_kd."
            )
        if r1 == r1 and r1 > 0.3:
            return "PHYSICAL", (
                f"smooth low-frequency oscillation (r1={r1:.3f},"
                f" ~{torque_report['dominant_frequency_hz']:.1f} Hz) - a real"
                " disturbance or resonance, not a numerical artefact."
            )
        return "AMBIGUOUS", f"r1={r1:.3f}, flip {flip*100:.0f}% - between the two signatures."

    phases = []
    for label, start, end in phase_bounds:
        mask = (sample_time >= start) & (sample_time < end)
        count = int(mask.sum())
        print(f"\n  ── {label}   t = {start:.1f}~{end:.1f} s   ({count} samples)")
        if count < 8:
            print("     (too few samples - episode ended before this phase)")
            phases.append({"phase": label, "samples": count, "verdict": "NO_DATA"})
            continue
        reports = [
            analyse(ang_vel[mask], "palm angular velocity [rad/s]", dt),
            analyse(torque[mask], "applied torque [N m]", dt),
            analyse(orientation_error[mask], "orientation error [rad]", dt),
            analyse(lin_vel[mask], "palm linear velocity [m/s]", dt),
        ]
        print("     signal                        axis      rms       peak    lag1_r   flip%    f[Hz]")
        for r in reports:
            print(
                f"     {r['label']:28s}  {r['dominant_axis']:>3s} {r['rms']:9.4f} {r['peak']:10.4f}"
                f" {r['lag1_autocorrelation']:8.3f} {r['sign_flip_rate']*100:7.1f}"
                f" {r['dominant_frequency_hz']:8.1f}"
            )
        window_finger = finger_vel[mask]
        print(f"     finger joint speed: rms {float(window_finger.pow(2).mean().sqrt()):.4f},"
              f" peak {float(window_finger.max()):.4f} rad/s")
        verdict, explanation = verdict_for(reports[1], reports[0])
        print(f"     VERDICT: {verdict} - {explanation}")
        phases.append(
            {
                "phase": label,
                "t_start": start,
                "t_end": end,
                "samples": count,
                "signals": reports,
                "finger_joint_vel_rms": float(window_finger.pow(2).mean().sqrt()),
                "finger_joint_vel_peak": float(window_finger.max()),
                "verdict": verdict,
                "explanation": explanation,
            }
        )

    # -- how much of angular_speed_excess is palm jitter? -------------------
    # The reward term is evaluated once per *policy* step, on the state left by
    # the last physics substep, so subsample accordingly before judging it.
    penalty_report = None
    if "stick1_ang_vel_x" in columns:
        s1 = trace[:, columns["stick1_ang_vel_x"] : columns["stick1_ang_vel_x"] + 3]
        s2 = trace[:, columns["stick2_ang_vel_x"] : columns["stick2_ang_vel_x"] + 3]
        # Strip everything above ~5 Hz from the palm: the scripted rotation is
        # under 1 rad/s, so what is removed is jitter, not commanded motion.
        window = max(2, int(round(0.2 / dt)))
        palm_smooth = moving_average(ang_vel, window)

        decim = env.cfg.decimation
        idx = torch.arange(decim - 1, trace.shape[0], decim)
        speed_a, raw_a = angular_speed_penalty(s1[idx], s2[idx], ang_vel[idx])
        speed_b, raw_b = angular_speed_penalty(s1[idx], s2[idx], palm_smooth[idx])

        print("\n  ── angular_speed_excess penalty contamination"
              f"   (limit 3.0 rad/s, weight -0.1, {idx.numel()} policy steps)")
        print(f"     {'':26s} {'as measured':>13s} {'palm de-jittered':>18s}")
        print(f"     {'relative speed rms':26s} {float(speed_a.pow(2).mean().sqrt()):13.4f}"
              f" {float(speed_b.pow(2).mean().sqrt()):18.4f}  rad/s")
        print(f"     {'relative speed peak':26s} {float(speed_a.max()):13.4f}"
              f" {float(speed_b.max()):18.4f}  rad/s")
        print(f"     {'steps over the 3.0 limit':26s} {float((raw_a>0).float().mean())*100:12.1f}%"
              f" {float((raw_b>0).float().mean())*100:17.1f}%")
        print(f"     {'episode raw penalty sum':26s} {float(raw_a.sum()):13.4f}"
              f" {float(raw_b.sum()):18.4f}")
        share = (
            1.0 - float(raw_b.sum()) / float(raw_a.sum()) if float(raw_a.sum()) > 0 else 0.0
        )
        print(f"\n     -> {share*100:.1f}% of the penalty disappears when the palm"
              " jitter is filtered out.")
        if float(raw_a.sum()) < 1.0e-6:
            print("        (the penalty is not firing at all in this run)")
        penalty_report = {
            "relative_speed_rms": float(speed_a.pow(2).mean().sqrt()),
            "relative_speed_rms_dejittered": float(speed_b.pow(2).mean().sqrt()),
            "over_limit_fraction": float((raw_a > 0).float().mean()),
            "over_limit_fraction_dejittered": float((raw_b > 0).float().mean()),
            "raw_penalty_sum": float(raw_a.sum()),
            "raw_penalty_sum_dejittered": float(raw_b.sum()),
            "jitter_share": share,
        }

    reports = [
        analyse(ang_vel, "palm angular velocity [rad/s]", dt),
        analyse(torque, "applied torque [N m]", dt),
        analyse(orientation_error, "orientation error [rad]", dt),
        analyse(lin_vel, "palm linear velocity [m/s]", dt),
    ]
    verdict, explanation = verdict_for(reports[1], reports[0])
    print(f"\n  WHOLE EPISODE VERDICT: {verdict}")
    print(f"  {explanation}")
    print("  (judge the controller on 'hold_at_goal': that is where the sticks"
          " load the palm in the new orientation.)")

    # -- persist ------------------------------------------------------------
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.tag:
        stamp = f"{stamp}_{args_cli.tag}"
    out_dir = os.path.join("logs", "debug", "hand_move_root_vibration", stamp)
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "effective_inertia": root_cfg.effective_inertia,
        "orientation_kp": root_cfg.orientation_kp,
        "orientation_kd": root_cfg.orientation_kd,
        "position_kp": root_cfg.position_kp,
        "position_kd": root_cfg.position_kd,
        "physics_dt": dt,
        "samples": int(trace.shape[0]),
        "ended_early": saw_reset,
        "survived_s": survived_s,
        "finger_grip": args_cli.finger_grip,
        "pin_sticks": bool(args_cli.pin_sticks),
        "policy_checkpoint": resume_path,
        "angular_speed_penalty": penalty_report,
        "verdict": verdict,
        "explanation": explanation,
        "signals": reports,
        "phases": phases,
        "finger_joint_vel_rms": float(finger_vel.pow(2).mean().sqrt()),
        "finger_joint_vel_peak": float(finger_vel.max()),
        "goal_range": {
            "x": list(schedule.range_x),
            "y": list(schedule.range_y),
            "z": list(schedule.range_z),
        },
    }
    with open(os.path.join(out_dir, "result.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    torch.save(trace, os.path.join(out_dir, "trace.pt"))
    print(f"\n  saved: {out_dir}/result.json  (+ trace.pt, columns = {list(columns)})")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
