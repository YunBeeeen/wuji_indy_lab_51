"""Run a 105D grasp policy in MuJoCo through the real hand's bring-up sequence.

This is the MuJoCo twin of ``run_finger_reach_real.py``: same phase names, same
order, same meanings, so a sim-to-sim comparison is reading two logs of one
procedure rather than reconciling two different programs.

    [RESET]    park the hand and put the sticks at their Isaac reset poses
    [GLIDE]    walk the target to the start pose, sticks PINNED
    [SETTLE]   hold there, sticks PINNED
    [RELEASE]  stop pinning -- from here the grasp is the policy's alone
    [SEED]     build the first observation from the settled state
    [RUN]      policy at 30 Hz
    [REPORT]   what actually happened to the sticks

Pinning is the simulator's stand-in for the person who holds the chopsticks
while the hand closes.  Releasing at a named instant is the point of the whole
script: everything before it is staging, and only what happens after it is a
grasp.  ``run_policy.py --run-policy`` starts already released from a teleported
pose, which is why it could never show this.

This is a MuJoCo program.  It never imports hardware modules and it is not a
task-performance evaluation unless a real perception source is supplied --
with ``--stick-provider synthetic`` the policy is shown a constant stick pose.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from ..backends.mujoco_scheduler import (
    MUJOCO_INTEGRATOR,
    MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP,
    SUPPORTED_INTEGRATORS,
    MujocoScheduler,
)
from ..backends.mujoco_wuji import DEFAULT_MODEL_PATH, MujocoWujiHand
from ..policy.observation_adapter import PolicyObservationAdapter
from ..common.policy_contract import (
    ACTION_DIM,
    DEFAULT_RESET_JOINT_POSITIONS,
    OBSERVATION_DIM,
    OBSERVATION_SLICES,
    POLICY_DT,
    POLICY_JOINT_NAMES,
    soft_command_limits,
)
from ..policy.policy_runner import PolicyRunner
from ..common.isaac_reset import (
    ISAAC_PREGRASP_JOINT_POSITIONS_RAD,
    MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ,
)


class ZeroPolicy:
    """Plumbing check: a zero action decodes to ``target = q_current``.

    Useful to prove the phases run, and USELESS as a grasp test -- holding
    ``q_current`` lets the hand chase its own drift, which structurally unwinds
    a grasp (measured 2026-08-20: 47.7 mm slip with a fixed target versus
    387926 mm with zero actions).
    """

    def infer(self, observation):
        return np.zeros(ACTION_DIM, dtype=np.float32)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a 105D policy in MuJoCo with the hardware bring-up sequence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", type=Path, default=None,
                        help="105D ONNX actor. Omit to run the zero-action plumbing check.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--mode", choices=("open", "close"), default="close",
                        help="OPEN/CLOSE one-hot handed to the policy.")
    parser.add_argument("--switch-at", type=float, default=None, metavar="SECONDS",
                        help="Flip to the other mode this far into the policy run.")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="Policy run length, after the bring-up phases.")

    phase = parser.add_argument_group("bring-up")
    phase.add_argument("--start-pose", choices=("pregrasp", "reset"), default="pregrasp",
                       help="Pose to glide TO. pregrasp is where the grasp policy starts.")
    phase.add_argument("--park-pose", choices=("reset", "pregrasp"), default="reset",
                       help="Pose to start FROM, so the glide has something to travel.")
    phase.add_argument("--glide-seconds", type=float, default=3.0)
    phase.add_argument("--settle-seconds", type=float, default=2.0)
    phase.add_argument("--release", choices=("after-settle", "never", "immediately"),
                       default="after-settle",
                       help="When to stop pinning the sticks. 'never' measures joint "
                            "tracking with the grasp taken out of the picture; "
                            "'immediately' is the old teleport-and-go behaviour.")
    phase.add_argument("--limit-margin", type=float, default=1.0, metavar="FRACTION",
                       help="Command only this fraction of each joint's range during "
                            "the bring-up moves. 1.0 is the trained action space.")

    sim = parser.add_argument_group("simulation")
    sim.add_argument("--controller-gains", choices=("vendor", "isaac_tuned"),
                     default="isaac_tuned",
                     help="isaac_tuned reproduces the Kp the policy trained against.")
    sim.add_argument("--physics-substeps", type=int,
                     default=MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP)
    sim.add_argument("--integrator", choices=SUPPORTED_INTEGRATORS, default=MUJOCO_INTEGRATOR)
    sim.add_argument("--stick-provider", choices=("synthetic", "ground-truth", "aruco"),
                     default="ground-truth",
                     help="What the policy is told about the sticks. ground-truth is "
                          "the honest sim-to-sim choice; synthetic feeds a constant.")
    sim.add_argument("--joint-limit-tolerance", type=float, default=0.02, metavar="RAD",
                     help="Abort if a joint is pushed this far past its hardware "
                          "limit. The servo overshoots slightly when the policy "
                          "commands into a stop, so this is not zero.")
    sim.add_argument("--viewer", action="store_true")
    sim.add_argument("--realtime", action="store_true")

    parser.add_argument("--print-interval", type=int, default=15,
                        help="Policy steps between progress lines.")
    parser.add_argument("--out", type=Path, default=None, help="Write per-step CSV here.")
    return parser


def named_pose(name: str) -> np.ndarray:
    if name == "pregrasp":
        return np.asarray(ISAAC_PREGRASP_JOINT_POSITIONS_RAD, dtype=np.float32)
    return np.asarray(DEFAULT_RESET_JOINT_POSITIONS, dtype=np.float32)


def stick_offsets_mm(hand: MujocoWujiHand, reference: np.ndarray) -> np.ndarray:
    """Return per-stick palm-frame displacement from ``reference`` in mm."""

    now = hand.get_stick_poses_in_palm().reshape(2, 7)
    return np.linalg.norm(now[:, :3] - reference[:, :3], axis=1) * 1.0e3


def main() -> int:
    args = build_argument_parser().parse_args()

    # ---- everything checkable before the model is built ----
    if args.policy is not None:
        from ..policy.onnx_policy import OnnxPolicy

        policy = OnnxPolicy(args.policy, OBSERVATION_DIM, ACTION_DIM)
        print(f"[POLICY]   {policy.path}")
        print(f"           {policy.input.shape} -> {policy.output.shape}")
    else:
        policy = ZeroPolicy()
        print("[POLICY]   zero-action plumbing check -- NOT a grasp test")

    hand = MujocoWujiHand(
        args.model,
        physical_limit_tolerance_rad=args.joint_limit_tolerance,
        controller_gains=args.controller_gains,
        physics_substeps=args.physics_substeps,
        integrator=args.integrator,
    )
    if not hand.has_sticks:
        raise RuntimeError(
            f"{args.model} has no sticks; this script stages a two-stick grasp."
        )
    print(f"[MODEL]    {hand.model_summary()}")
    print(f"[CONTRACT] obs {OBSERVATION_DIM}D, action {ACTION_DIM}D, "
          f"{len(POLICY_JOINT_NAMES)} joints")

    park = named_pose(args.park_pose)
    start = named_pose(args.start_pose)
    reference_poses = np.asarray(
        MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ, dtype=np.float64
    )

    viewer = None
    rows: list[list] = []
    failure: BaseException | None = None
    try:
        if args.viewer:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(hand.model, hand.data)
        scheduler = MujocoScheduler(hand, viewer=viewer, realtime=args.realtime)
        print(f"[RATES]    policy {1.0 / POLICY_DT:.1f} Hz, "
              f"{scheduler.substeps} substeps x {scheduler.physics_dt * 1e3:.4f} ms, "
              f"gains={args.controller_gains}, integrator={args.integrator}")

        # ---------------- RESET ----------------
        hand.reset(park)
        hand.set_stick_poses_in_palm(reference_poses)
        print(f"\n[RESET]    parked at {args.park_pose}, sticks at their Isaac reset poses")
        if hand.last_reset_clamped.any():
            clamped = [POLICY_JOINT_NAMES[i] for i in np.flatnonzero(hand.last_reset_clamped)]
            print(f"           park pose clamped to hardware limits at {clamped}")

        # Pin from the very first phase unless the caller asked otherwise: the
        # sticks are resting in a hand that is about to close, and nobody is
        # holding them yet.
        pin = None if args.release == "immediately" else reference_poses

        # ---------------- GLIDE ----------------
        print(f"\n[GLIDE]    {args.park_pose} -> {args.start_pose} over "
              f"{args.glide_seconds:.1f}s, sticks {'PINNED' if pin is not None else 'free'}")
        travel = float(np.abs(start - park).max())
        print(f"           largest joint displacement {travel:.4f} rad "
              f"({travel / max(args.glide_seconds, 1e-9):.3f} rad/s commanded)")
        elapsed, actual, error = scheduler.glide_to_pose(
            start, seconds=args.glide_seconds, limit_fraction=args.limit_margin,
            pin_sticks=pin,
        )
        print(f"[GLIDE]    done in {elapsed:.2f}s, max |q - target| = "
              f"{np.abs(error).max():.5f} rad")

        # ---------------- SETTLE ----------------
        print(f"\n[SETTLE]   {args.settle_seconds:.1f}s at the start pose, "
              f"sticks {'PINNED' if pin is not None else 'free'}")
        settled, drift = scheduler.settle(args.settle_seconds, pin_sticks=pin)
        print(f"[SETTLE]   joint drift during the hold {np.abs(drift).max():.5f} rad")

        # ---------------- RELEASE ----------------
        if args.release == "after-settle":
            pin = None
            print("\n[RELEASE]  sticks let go -- everything after this is the policy's grasp")
        elif args.release == "never":
            print("\n[RELEASE]  never: sticks stay pinned, so this measures joint "
                  "tracking only and any stick number below is meaningless")
        else:
            print("\n[RELEASE]  already free since reset")
        released_reference = hand.get_stick_poses_in_palm().reshape(2, 7).copy()
        print(f"           stick baseline (palm mm) "
              f"{np.round(released_reference[:, :3] * 1e3, 2).tolist()}")

        # ---------------- SEED ----------------
        provider, camera = _make_provider(args.stick_provider, hand)
        adapter = PolicyObservationAdapter(mode=args.mode, stick_provider=provider)
        runner = PolicyRunner(hand, policy, adapter)
        observation = runner.reset()
        print(f"\n[SEED]     stick provider {type(provider).__name__}, mode {args.mode}")
        print(f"           obs {observation.shape[0]}D, "
              f"fingertips {np.round(observation[OBSERVATION_SLICES['fingertips'].slice][:3], 4)} ...")

        # ---------------- RUN ----------------
        steps = max(1, int(round(args.seconds / POLICY_DT)))
        switch_step = (
            None if args.switch_at is None else max(0, int(round(args.switch_at / POLICY_DT)))
        )
        print(f"\n[RUN]      {steps} policy steps ({args.seconds:.1f}s)"
              + (f", mode -> {_other(args.mode)} at step {switch_step}"
                 if switch_step is not None else ""))
        began = time.monotonic()
        for step in range(steps):
            if switch_step is not None and step == switch_step:
                runner.set_mode(_other(args.mode))
                print(f"  [MODE]  -> {_other(args.mode)} at t={step * POLICY_DT:.2f}s")
            decoded, observation = scheduler.run_policy_tick(runner, pin_sticks=pin)
            offsets = stick_offsets_mm(hand, released_reference)
            rows.append(
                [step, step * POLICY_DT, runner.observations.mode,
                 *offsets, int(decoded.target_was_clamped.sum())]
                + list(hand.read_joint_positions())
                + list(decoded.position_target)
                + list(decoded.action_manager_action)
            )
            if step % args.print_interval == 0 or step == steps - 1:
                print(f"  t={step * POLICY_DT:6.2f}s mode={runner.observations.mode:5s} "
                      f"stick1={offsets[0]:8.2f}mm stick2={offsets[1]:8.2f}mm "
                      f"clamp={int(decoded.target_was_clamped.sum()):2d} "
                      f"a=[{decoded.action_manager_action.min():+.3f},"
                      f"{decoded.action_manager_action.max():+.3f}]")

        # ---------------- REPORT ----------------
        offsets = stick_offsets_mm(hand, released_reference)
        print(f"\n[REPORT]   wall clock {time.monotonic() - began:.1f}s")
        print(f"           stick1 moved {offsets[0]:.2f} mm from the release pose")
        print(f"           stick2 moved {offsets[1]:.2f} mm from the release pose")
        if args.release == "never":
            print("           (sticks were pinned throughout -- these are pinning error, "
                  "not grasp quality)")
        held = offsets.max() < 20.0
        print(f"           verdict: {'HELD' if held else 'DROPPED'} "
              f"(threshold 20 mm from release)")
    except (RuntimeError, ValueError) as exc:
        # Losing the rows would cost the run.  Report, write, then exit nonzero.
        failure = exc
        print(f"\n[ABORT]    {type(exc).__name__}: {exc}")
        if rows:
            print(f"           {len(rows)} policy steps completed before this "
                  f"({rows[-1][1]:.2f}s in)")
    finally:
        if viewer is not None:
            from .run_policy import _close_viewer_and_wait

            _close_viewer_and_wait(viewer)

    if args.out is not None and rows:
        _write_csv(args.out, rows)
        print(f"[CSV]      {args.out}  ({len(rows)} rows)")
    return 1 if failure is not None else 0


def _other(mode: str) -> str:
    return "open" if mode == "close" else "close"


def _make_provider(name: str, hand: MujocoWujiHand):
    from .run_policy import create_stick_provider

    return create_stick_provider(hand, name)


def _write_csv(path: Path, rows) -> None:
    header = (
        ["step", "t_s", "mode", "stick1_mm", "stick2_mm", "targets_clamped"]
        + [f"q_{n}" for n in POLICY_JOINT_NAMES]
        + [f"qt_{n}" for n in POLICY_JOINT_NAMES]
        + [f"a_{n}" for n in POLICY_JOINT_NAMES]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
