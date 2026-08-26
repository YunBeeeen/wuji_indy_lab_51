# [run/MuJoCo] 중지 reach 진단을 MuJoCo에서 실행. 실물 판과 같은 CSV 컬럼.
"""Run the middle-finger reach diagnostic in MuJoCo, in the real hand's shape.

The loop mirrors the vendor SDK example: read a 20-joint finger-major state,
build one command frame per policy tick, and re-send it while the plant runs.
On hardware the publisher repeats that frame at 100 Hz under a LowPass
controller; here the same held target is integrated by MuJoCo instead. The
policy tick stays 30 Hz in both.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from ..policy.finger_reach import (
    FingerReachObservationAdapter,
    MIDDLE_JOINT_NAMES,
    MIDDLE_POLICY_INDICES,
    MiddleFingerReachRunner,
    REACH_ACTION_DIM,
    REACH_OBSERVATION_DIM,
    REACH_RANGE_M,
)
from ..backends.mujoco_scheduler import MujocoScheduler
from ..backends.mujoco_wuji import make_finger_reach_backend
from ..common.policy_contract import POLICY_DT, POLICY_JOINT_NAMES


CSV_COLUMNS = (
    ["time", "target_index", "target_palm_x", "target_palm_y", "target_palm_z"]
    + [f"q_prev_{k}" for k in range(1, 5)]
    + [f"q_curr_{k}" for k in range(1, 5)]
    + [f"action_{k}" for k in range(1, 5)]
    + [f"q_target_{k}" for k in range(1, 5)]
    + ["tip_palm_x", "tip_palm_y", "tip_palm_z"]
    + ["error_palm_x", "error_palm_y", "error_palm_z", "error_norm"]
)


class ZeroPolicy:
    """Plumbing smoke stand-in; the reach policy is still training."""

    def infer(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (REACH_OBSERVATION_DIM,):
            raise ValueError(
                f"Expected {REACH_OBSERVATION_DIM}D observation."
            )
        return np.zeros(REACH_ACTION_DIM, dtype=np.float32)


def load_scenario(path: Path) -> dict:
    scenario = json.loads(Path(path).read_text())

    for key in ("targets_palm_m", "target_duration_s"):
        if key not in scenario:
            raise ValueError(f"Scenario is missing {key!r}.")

    controlled = scenario.get("controlled_joints")

    if controlled is not None and list(controlled) != list(MIDDLE_JOINT_NAMES):
        raise ValueError(
            f"Scenario controls {controlled}, "
            f"this backend controls {list(MIDDLE_JOINT_NAMES)}."
        )

    return scenario


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="15->4 ONNX actor. Omitted: run a zero-action plumbing smoke.",
    )

    parser.add_argument(
        "--scenario",
        type=Path,
        default=None,
        help="Deterministic target sequence shared with Isaac and the real hand.",
    )

    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Single palm-frame target in metres.",
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=4.0,
        help="Dwell for a single --target run.",
    )

    parser.add_argument(
        "--target-duration",
        type=float,
        default=None,
        help=(
            "Override the scenario's per-target dwell. "
            "On hardware prefer several seconds: the target steps "
            "discontinuously between waypoints and a short dwell makes "
            "the policy command a large residual immediately."
        ),
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--print-interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--physics-substeps",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--controller-gains",
        choices=("vendor", "isaac_tuned"),
        default="vendor",
        help="vendor: the pinned MJCF's own identified gains (the plant contract). "
             "isaac_tuned: the hand-tuned values from hand_real_env_cfg.py, kept "
             "only to reproduce policies trained before 2026-08-18.",
    )

    # ------------------------------------------------------------
    # Viewer options
    # ------------------------------------------------------------
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open MuJoCo passive viewer during the rollout.",
    )

    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Run approximately in wall-clock real time for visualization.",
    )

    parser.add_argument(
        "--limit-margin",
        type=float,
        default=1.0,
        metavar="FRACTION",
        help=(
            "Command only this fraction of each joint's range, as "
            "centre +- FRACTION*half_range.  Defaults to 1.0, the action space "
            "the policy was trained in, so existing MuJoCo logs stay "
            "reproducible.  The hardware CLI defaults to 0.95 instead and "
            "passes its own value here through --parallel-mujoco, because a "
            "paired comparison must run one action space on both plants."
        ),
    )

    return parser


def _launch_viewer(backend):
    """Open a passive MuJoCo viewer focused on the hand."""

    import mujoco.viewer

    # -------------------------------
    # Brighten the default headlight
    # -------------------------------
    backend.model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    backend.model.vis.headlight.diffuse[:] = [0.85, 0.85, 0.85]
    backend.model.vis.headlight.specular[:] = [0.25, 0.25, 0.25]

    # Optional: lighten background haze a bit
    backend.model.vis.rgba.haze[:] = [0.15, 0.15, 0.15, 1.0]

    viewer = mujoco.viewer.launch_passive(
        backend.model,
        backend.data,
    )

    # Show all geometry groups.
    viewer.opt.geomgroup[:] = 1

    # Automatically frame the reach scene.
    viewer.cam.lookat[:] = backend.model.stat.center
    viewer.cam.distance = 1.5 * backend.model.stat.extent
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0

    viewer.sync()

    return viewer


def _close_viewer_and_wait(viewer, timeout_s: float = 5.0) -> None:
    """Close passive viewer safely before MuJoCo model/data destruction."""

    if viewer is None:
        return

    viewer.close()

    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        try:
            # MuJoCo passive viewer's internal simulation object disappears
            # after the viewer thread has actually shut down.
            if viewer.m is None:
                return
        except Exception:
            return

        time.sleep(0.01)

    print(
        "[WARNING] MuJoCo viewer did not confirm shutdown within "
        f"{timeout_s:.1f}s."
    )


def main() -> int:
    args = build_argument_parser().parse_args()

    # ============================================================
    # Target / scenario
    # ============================================================

    if args.scenario is not None:
        scenario = load_scenario(args.scenario)

        targets = [
            np.asarray(t, dtype=np.float32)
            for t in scenario["targets_palm_m"]
        ]

        duration = float(
            args.target_duration or scenario["target_duration_s"]
        )

        reset_q = scenario.get("initial_all_joint_q")

    else:
        centre = [
            sum(REACH_RANGE_M[axis]) / 2.0
            for axis in ("x", "y", "z")
        ]

        targets = [
            np.asarray(
                args.target if args.target else centre,
                dtype=np.float32,
            )
        ]

        duration = float(
            args.target_duration or args.seconds
        )

        reset_q = None

    # ============================================================
    # MuJoCo backend
    # ============================================================

    backend_kwargs = {
        "controller_gains": args.controller_gains,
    }

    if args.physics_substeps is not None:
        backend_kwargs["physics_substeps"] = args.physics_substeps

    backend = make_finger_reach_backend(**backend_kwargs)

    # ============================================================
    # Policy
    # ============================================================

    if args.policy is not None:
        from ..policy.onnx_policy import OnnxPolicy

        policy = OnnxPolicy(
            args.policy,
            observation_dim=REACH_OBSERVATION_DIM,
            action_dim=REACH_ACTION_DIM,
        )

        print(
            f"[POLICY]   {policy.path.name}  "
            f"{policy.input.shape} -> {policy.output.shape}"
        )

    else:
        policy = ZeroPolicy()

        print(
            "[SMOKE] no --policy given; commanding zero action."
        )

    # ============================================================
    # Runner
    # ============================================================

    runner = MiddleFingerReachRunner(
        backend,
        policy,
        FingerReachObservationAdapter(),
        limit_fraction=args.limit_margin,
    )
    if args.limit_margin < 1.0:
        print(
            f"[LIMITS] command margin {args.limit_margin:.3f} of range; "
            "this is NOT the trained action space and will not reproduce a "
            "default MuJoCo log"
        )

    observation = runner.reset(
        np.asarray(reset_q, dtype=np.float32)
        if reset_q
        else None
    )

    # ============================================================
    # Viewer
    # ============================================================

    viewer = None

    if args.viewer:
        viewer = _launch_viewer(backend)

        print("[VIEWER]   MuJoCo passive viewer opened")

        if args.realtime:
            print("[VIEWER]   realtime visualization enabled")
        else:
            print(
                "[VIEWER]   realtime disabled; simulation may finish quickly"
            )

    # MujocoScheduler already supports viewer synchronization and
    # realtime sleeping.
    scheduler = MujocoScheduler(
        backend,
        viewer=viewer,
        realtime=args.realtime,
    )

    # ============================================================
    # Contract report
    # ============================================================

    print(
        f"[CONTRACT] obs {observation.shape[0]}D, "
        f"action {REACH_ACTION_DIM}D, "
        f"joints {list(MIDDLE_JOINT_NAMES)} "
        f"= canonical {MIDDLE_POLICY_INDICES.tolist()}"
    )

    print(
        f"[TIMING]   policy {1 / POLICY_DT:.1f} Hz, "
        f"physics {1 / backend.model.opt.timestep:.1f} Hz, "
        f"substeps {backend.physics_substeps}, "
        f"integrator {backend.integrator}, "
        f"gains {backend.controller_gains}"
    )

    print(
        f"[SCENE]    {backend.model_path.name}, "
        f"sticks={backend.has_sticks}, "
        f"nq={backend.model.nq}"
    )

    print(
        "[HOLD]     16 non-policy joints pinned to the reset pose"
    )

    # ============================================================
    # Policy rollout
    # ============================================================

    rows = []

    steps_per_target = max(
        1,
        int(round(duration / POLICY_DT)),
    )

    step = 0

    try:
        for target_index, target in enumerate(targets):

            # ----------------------------------------------------
            # New target
            # ----------------------------------------------------

            runner.set_target(target)

            print(
                f"\n[TARGET {target_index}] "
                f"palm xyz = "
                f"[{target[0]:+.5f}, "
                f"{target[1]:+.5f}, "
                f"{target[2]:+.5f}] m"
            )

            # ----------------------------------------------------
            # Run target for requested duration
            # ----------------------------------------------------

            for _ in range(steps_per_target):

                # Observation that will enter the network.
                obs_before = runner.observations.build()

                # NOTE:
                # obs[0:4] is normalized q_previous.
                # RADIANS, not the normalized observation slice: the
                # cross-backend comparison is about joint trajectories, and raw
                # radians are what Isaac and the real encoders report.
                q_prev = runner.observations.q_previous

                # Policy inference + residual q target command.
                decoded = runner.command()

                # Hold this target for exactly one 30-Hz policy tick.
                scheduler.hold_policy_target()

                # Read new q and advance observation history.
                runner.observe_after_hold()

                q_now = backend.read_joint_positions()[
                    MIDDLE_POLICY_INDICES
                ]

                # Logging only; fingertip position is NOT a policy input.
                tip = runner.middle_fingertip_in_palm()

                error = target - tip

                step += 1

                rows.append(
                    [
                        step * POLICY_DT,
                        target_index,
                        *target,
                    ]
                    + list(q_prev)
                    + list(q_now)
                    + list(decoded.clipped_action)
                    + list(decoded.position_target)
                    + list(tip)
                    + list(error)
                    + [float(np.linalg.norm(error))]
                )

                if (
                    step % args.print_interval == 0
                    or step == 1
                ):
                    print(
                        f"step={step:04d} "
                        f"t={step * POLICY_DT:6.3f}s "
                        f"tgt#{target_index} "
                        f"q=[{q_now.min():+.3f},"
                        f"{q_now.max():+.3f}] "
                        f"a=[{decoded.clipped_action.min():+.3f},"
                        f"{decoded.clipped_action.max():+.3f}] "
                        f"err="
                        f"{np.linalg.norm(error) * 1000:7.2f}mm"
                    )

    finally:
        # Always close the viewer, including Ctrl+C / exceptions.
        if viewer is not None:
            _close_viewer_and_wait(viewer)

    # ============================================================
    # Final command
    # ============================================================

    print(
        "\n[FINAL COMMAND FRAME] "
        "SDK finger-major (5, 4) rad"
    )

    for finger, row in enumerate(
        runner.joint_command_grid(),
        start=1,
    ):
        marker = "  <- policy" if finger == 3 else ""

        print(
            f"  finger{finger}: ["
            + ", ".join(f"{v:+.4f}" for v in row)
            + f"]{marker}"
        )

    # ============================================================
    # CSV
    # ============================================================

    if args.csv is not None:

        args.csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.csv.open(
            "w",
            newline="",
        ) as handle:

            writer = csv.writer(handle)

            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)

        print(
            f"\nwrote {len(rows)} rows -> {args.csv}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
