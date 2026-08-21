"""Command-line entry point for Wuji MuJoCo deployment plumbing checks."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..common.fingertip_fk import OFFICIAL_URDF, POLICY_TIP_FRAME_URDF, WujiHand1FingertipFK
from ..backends.joint_mapping import format_joint_mapping
from ..backends.mujoco_scheduler import (
    MUJOCO_INTEGRATOR,
    MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP,
    SUPPORTED_INTEGRATORS,
    MujocoScheduler,
)
from ..backends.mujoco_wuji import DEFAULT_MODEL_PATH, MujocoWujiHand
from ..policy.observation_adapter import PolicyObservationAdapter
from ..policy.onnx_policy import OnnxPolicy
from ..policy.policy_runner import PolicyRunner
from ..backends.real_wuji_backend import pending_validation_report
from ..common.policy_contract import (
    ACTION_DIM,
    COMMAND_TARGET_LIMITS,
    DEFAULT_RESET_JOINT_POSITIONS,
    OFFICIAL_NOMINAL_PHYSICAL_LIMITS,
    OBSERVATION_NORMALIZATION_LIMITS,
    OBSERVATION_DIM,
    OBSERVATION_SLICES,
    POLICY_DT,
    POLICY_JOINT_NAMES,
    contract_summary,
)


#: A copy under models/, not a path into the training logs.  Reaching into
#: ``logs/rsl_rl/<run>/exported/`` meant the default died whenever a run was
#: tidied up -- which it had: the previous value named a run that no longer
#: exists, so ``--run-policy`` failed with FileNotFoundError before touching
#: anything.  See models/README.md.
DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "hand_final_2026-08-21_01-14-36.onnx"
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the deployment and contract-validation CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Fixed-base official Wuji Hand 1 MuJoCo backend and canonical "
            "deployment-contract validator. Synthetic stick observations are "
            "not a task-performance evaluation."
        )
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--inspect-model", action="store_true")
    modes.add_argument("--inspect-joints", action="store_true")
    modes.add_argument("--inspect-contract", action="store_true")
    modes.add_argument("--test-joints", action="store_true")
    modes.add_argument("--onnx-only", action="store_true")
    modes.add_argument("--run-policy", action="store_true")
    modes.add_argument("--validate-fk", action="store_true")
    modes.add_argument("--validate-aruco", action="store_true")
    modes.add_argument("--inspect-camera", action="store_true")
    modes.add_argument("--view-camera", action="store_true")
    modes.add_argument("--view-scene", action="store_true")
    modes.add_argument("--smoke-backend", action="store_true")
    modes.add_argument(
        "--hold-pose",
        action="store_true",
        help=(
            "Command ONE fixed joint target (the Isaac pregrasp) for "
            "--policy-steps and report how far the sticks move.  This is the "
            "grasp test; --smoke-backend is not.  A zero residual action means "
            "target = q_current every step, so the hand re-aims at wherever it "
            "has already slipped to and the grasp unwinds on its own."
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--mode", choices=("open", "close"), default="open")
    parser.add_argument("--policy-steps", type=int, default=300)
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument("--small-delta", type=float, default=0.03)
    parser.add_argument("--joint-test-physics-steps", type=int, default=180)
    parser.add_argument(
        "--controller-gains",
        choices=("vendor", "isaac_tuned"),
        default="vendor",
        help=(
            "vendor (default): the pinned MJCF's own identified gains, i.e. the "
            "plant contract.  isaac_tuned: the hand-tuned values from "
            "hand_real_env_cfg.py, kept only to reproduce policies trained "
            "before 2026-08-18.  The old names 'deploy'/'official' named these "
            "the other way round and are no longer accepted here."
        ),
    )
    parser.add_argument(
        "--physics-substeps",
        type=int,
        default=MUJOCO_PHYSICS_SUBSTEPS_PER_POLICY_STEP,
        help=(
            "MuJoCo integration steps per 30 Hz policy step. Numerical accuracy "
            "only; the hold duration stays 1/30 s regardless."
        ),
    )
    parser.add_argument(
        "--integrator",
        choices=SUPPORTED_INTEGRATORS,
        default=MUJOCO_INTEGRATOR,
        help="MuJoCo integrator. The vendor MJCF declares rk4; implicitfast is MuJoCo's recommended default.",
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--debug-observation", action="store_true")
    parser.add_argument(
        "--stick-provider",
        choices=("synthetic", "ground-truth", "aruco"),
        default="synthetic",
        help="Palm-relative StickPose7D source used by ONNX/closed-loop modes.",
    )
    parser.add_argument(
        "--joint-limit-tolerance",
        type=float,
        default=0.02,
        help=(
            "Maximum numerical actual-q overshoot beyond a MuJoCo joint limit "
            "before aborting. Position targets are always clamped exactly."
        ),
    )
    return parser


def inspect_model(hand: MujocoWujiHand) -> None:
    """Print every model joint and actuator before the canonical map."""

    import mujoco

    print("[MUJOCO MODEL]")
    print(hand.model_summary())
    print("\n[joints]")
    for joint_id in range(hand.model.njnt):
        name = mujoco.mj_id2name(hand.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        print(
            f"  id={joint_id:02d} name={name:<20} "
            f"type={int(hand.model.jnt_type[joint_id])} "
            f"qpos={int(hand.model.jnt_qposadr[joint_id]):02d} "
            f"dof={int(hand.model.jnt_dofadr[joint_id]):02d} "
            f"range={tuple(float(v) for v in hand.model.jnt_range[joint_id])}"
        )
    print("\n[actuators]")
    for actuator_id in range(hand.model.nu):
        name = mujoco.mj_id2name(
            hand.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
        )
        joint_id = int(hand.model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(
            hand.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        )
        print(
            f"  id={actuator_id:02d} name={name:<20} joint={joint_name:<20} "
            f"ctrlrange={tuple(float(v) for v in hand.model.actuator_ctrlrange[actuator_id])} "
            f"kp={float(hand.model.actuator_gainprm[actuator_id, 0]):.6g} "
            f"kd={-float(hand.model.actuator_biasprm[actuator_id, 2]):.6g} "
            f"effort={float(hand.model.actuator_forcerange[actuator_id, 1]):.6g}"
        )
    print()
    print(format_joint_mapping(hand.model, hand.mapping))


def test_single_joint_commands(
    hand: MujocoWujiHand,
    delta: float,
    physics_steps: int,
) -> bool:
    """Command each mapped actuator alone, then restore it to the baseline."""

    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("--small-delta must be finite and positive.")
    if physics_steps < 1:
        raise ValueError("--joint-test-physics-steps must be positive.")
    # Exercise away from hard stops: the official thumb Joint1 lower limit is
    # +0.0475 rad, where a 0.03 rad perturbation can be dominated by the model's
    # static/contact response even though its mapping is correct.
    baseline = (
        COMMAND_TARGET_LIMITS[:, 0]
        + np.float32(0.25)
        * (COMMAND_TARGET_LIMITS[:, 1] - COMMAND_TARGET_LIMITS[:, 0])
    ).astype(np.float32)
    all_passed = True
    print("[SINGLE-JOINT POSITION COMMAND TEST]")
    for policy_index, joint_name in enumerate(POLICY_JOINT_NAMES):
        hand.reset(baseline)
        # Establish the gravity-loaded equilibrium for the unchanged baseline
        # command.  The existing MJCF gains are intentionally not claimed to
        # match PhysX, so comparing against the raw reset q would mislabel a
        # steady gravity offset as a mapping failure.
        hand.apply_position_targets(baseline)
        hand.step(physics_steps)
        equilibrium = hand.read_joint_positions_policy_order()
        baseline_ctrl = hand.control_snapshot()
        requested = float(baseline[policy_index] + delta)
        applied = min(requested, float(COMMAND_TARGET_LIMITS[policy_index, 1]))
        target = baseline.copy()
        target[policy_index] = applied
        hand.apply_position_targets(target)
        hand.step(physics_steps)
        q_after = hand.read_joint_positions_policy_order()
        moved = float(q_after[policy_index] - equilibrium[policy_index])
        target_passed = moved > min(delta * 0.05, 1.0e-3)
        actuator_id = int(hand.mapping.policy_to_mujoco_actuator[policy_index])
        changed_ctrl = np.flatnonzero(
            np.abs(hand.control_snapshot() - baseline_ctrl) > 1.0e-12
        )
        ctrl_isolated = np.array_equal(changed_ctrl, np.asarray([policy_index]))

        hand.apply_position_targets(baseline)
        hand.step(physics_steps)
        restored_error = abs(
            float(
                hand.read_joint_positions_policy_order()[policy_index]
                - equilibrium[policy_index]
            )
        )
        restored = restored_error < max(delta * 0.25, 5.0e-3)
        passed = target_passed and ctrl_isolated and restored
        all_passed &= passed
        print(
            f"  [{'PASS' if passed else 'FAIL'}] Policy[{policy_index:02d}] "
            f"{joint_name:<20} actuator={actuator_id:02d} "
            f"requested={requested:+.5f} applied={applied:+.5f} "
            f"moved={moved:+.5f} restored_err={restored_error:.5f}"
        )
    print(f"single-joint command test: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def run_onnx_only(
    hand: MujocoWujiHand,
    policy: OnnxPolicy,
    mode: str,
    debug_observation: bool,
    stick_provider,
) -> bool:
    """Validate fixed-shape deterministic inference without applying action."""

    adapter = PolicyObservationAdapter(mode=mode, stick_provider=stick_provider)
    runner = PolicyRunner(hand, policy, adapter)
    observation = runner.reset()
    print(f"[STICK PROVIDER] {type(stick_provider).__name__}")
    if debug_observation:
        for name, value in adapter.debug_slices().items():
            term = OBSERVATION_SLICES[name]
            print(f"  obs[{term.start:03d}:{term.stop:03d}] {name}: {value}")
    first = policy.infer(observation)
    second = policy.infer(observation)
    deterministic = np.array_equal(first, second)
    print("[ONNX]")
    print(policy.describe())
    print(
        f"  observation: shape={observation.shape}, dtype={observation.dtype}, "
        f"finite={bool(np.isfinite(observation).all())}"
    )
    print(
        f"  action: shape={first.shape}, dtype={first.dtype}, "
        f"finite={bool(np.isfinite(first).all())}, min={first.min():+.6f}, "
        f"max={first.max():+.6f}, deterministic={deterministic}"
    )
    return deterministic


def run_closed_loop(
    hand: MujocoWujiHand,
    policy: OnnxPolicy,
    mode: str,
    policy_steps: int,
    print_interval: int,
    viewer: bool,
    realtime: bool,
    debug_observation: bool,
    stick_provider,
) -> None:
    """Run the 30 Hz policy / 120 Hz fixed-base MuJoCo loop."""

    if policy_steps < 1 or print_interval < 1:
        raise ValueError("Policy steps and print interval must be positive.")
    adapter = PolicyObservationAdapter(mode=mode, stick_provider=stick_provider)
    runner = PolicyRunner(hand, policy, adapter)
    observation = runner.reset()
    if np.any(hand.last_reset_clamped):
        names = [
            POLICY_JOINT_NAMES[index]
            for index in np.flatnonzero(hand.last_reset_clamped)
        ]
        print(f"[WARNING] Reset q clamped to MuJoCo limits for: {names}")
    print(f"[STICK PROVIDER] {type(stick_provider).__name__}")
    print(
        f"[CLOSED LOOP] policy={1.0 / POLICY_DT:.1f} Hz, "
        f"physics={1.0 / hand.model.opt.timestep:.1f} Hz, "
        f"substeps={hand.physics_substeps}, integrator={hand.integrator}, "
        f"gains={hand.controller_gains}"
    )
    if debug_observation:
        for name, value in adapter.debug_slices().items():
            print(f"  {name}: {value}")

    active_viewer = _launch_viewer(hand) if viewer else None
    scheduler = MujocoScheduler(hand, viewer=active_viewer, realtime=realtime)
    try:
        for policy_step in range(policy_steps):
            decoded, observation = scheduler.run_policy_tick(runner)
            onnx_action = decoded.onnx_action
            q_next = hand.read_joint_positions_policy_order()
            lower = OFFICIAL_NOMINAL_PHYSICAL_LIMITS[:, 0]
            upper = OFFICIAL_NOMINAL_PHYSICAL_LIMITS[:, 1]
            violation = np.maximum(0.0, np.maximum(lower - q_next, q_next - upper))
            max_violation = float(np.max(violation))
            if policy_step % print_interval == 0 or policy_step == policy_steps - 1:
                print(
                    f"step={policy_step:04d} t={(policy_step + 1) * POLICY_DT:7.3f}s "
                    f"onnx=[{onnx_action.min():+.3f},{onnx_action.max():+.3f}] "
                    f"action=[{decoded.action_manager_action.min():+.3f},"
                    f"{decoded.action_manager_action.max():+.3f}] "
                    f"q=[{q_next.min():+.3f},{q_next.max():+.3f}] "
                    f"target=[{decoded.position_target.min():+.3f},"
                    f"{decoded.position_target.max():+.3f}] "
                    f"action_clip={int(decoded.action_was_clipped.sum())} "
                    f"target_clamp={int(decoded.target_was_clamped.sum())}"
                    f" q_limit_overshoot={max_violation:.5f}"
                )
    finally:
        if active_viewer is not None:
            _close_viewer_and_wait(active_viewer)
    print("closed-loop timing/interface smoke test: PASS")


def _launch_viewer(hand: MujocoWujiHand):
    """Import GUI support only when explicitly requested."""

    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(hand.model, hand.data)
    # MuJoCo's default mjvOption enables only geom groups 0..2.  Fixture and
    # camera proxies intentionally live in group 3 and debug axes in group 4,
    # so explicitly enable them in the external scene viewer.  These display
    # settings do not affect the calibrated optical-camera renderer.
    viewer.opt.geomgroup[:] = 1
    # Frame the complete testbed on startup instead of the much smaller hand.
    # Users can still orbit/zoom normally after the viewer opens.
    viewer.cam.lookat[:] = hand.model.stat.center
    viewer.cam.distance = 1.35 * hand.model.stat.extent
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0
    return viewer


def validate_fingertip_fk(hand: MujocoWujiHand) -> bool:
    """Compare explicit MuJoCo sites with standalone official-URDF FK.

    This is a MODEL-CONSISTENCY check between the vendor description and the
    MuJoCo model built from it -- deliberately the official URDF, not the
    trained tip frames.  It says nothing about whether those tips are the ones
    the policy learned; ``policy_tip_frame_delta`` below reports that.
    """

    fk = WujiHand1FingertipFK(OFFICIAL_URDF)
    lower = OFFICIAL_NOMINAL_PHYSICAL_LIMITS[:, 0]
    upper = OFFICIAL_NOMINAL_PHYSICAL_LIMITS[:, 1]
    samples = (
        DEFAULT_RESET_JOINT_POSITIONS,
        lower + np.float32(0.25) * (upper - lower),
        lower + np.float32(0.75) * (upper - lower),
    )
    passed = True
    print("[MUJOCO SITE / OFFICIAL URDF FK]")
    for sample_index, q in enumerate(samples):
        hand.reset(q)
        mujoco_tips = hand.get_fingertip_positions_in_palm()
        urdf_tips = fk.fingertip_positions_in_palm(q)
        error = np.abs(mujoco_tips - urdf_tips).reshape(5, 3)
        maximum = float(error.max())
        sample_passed = maximum <= 2.0e-6
        passed &= sample_passed
        print(
            f"  [{'PASS' if sample_passed else 'FAIL'}] sample={sample_index} "
            f"max_abs_error={maximum:.9g} m"
        )
    print(f"fingertip FK/site validation: {'PASS' if passed else 'FAIL'}")
    print()
    print_policy_tip_frame_delta(samples)
    return passed


def print_policy_tip_frame_delta(samples) -> None:
    """Report how far MuJoCo's own tips sit from the frames the policy saw.

    Informational, never a gate: the two URDFs are simply different models and
    the observation is built from POLICY_TIP_FRAME_URDF regardless.  Printing it
    keeps the 3 mm thumb offset visible instead of buried in a diff.
    """

    official = WujiHand1FingertipFK(OFFICIAL_URDF)
    trained = WujiHand1FingertipFK(POLICY_TIP_FRAME_URDF)
    print(f"[OFFICIAL vs POLICY TIP FRAMES]  ({POLICY_TIP_FRAME_URDF.label})")
    stacked = np.stack(
        [
            np.linalg.norm(
                (
                    official.fingertip_positions_in_palm(q)
                    - trained.fingertip_positions_in_palm(q)
                ).reshape(5, 3),
                axis=1,
            )
            for q in samples
        ]
    )
    for finger in range(5):
        column = stacked[:, finger] * 1.0e3
        print(
            f"  finger{finger + 1}: {column.min():7.4f} .. {column.max():7.4f} mm"
        )
    print("  observations use the policy tip frames on every backend.")


def smoke_backend(
    hand: MujocoWujiHand,
    policy_steps: int,
    viewer=None,
    realtime: bool = False,
) -> bool:
    """Exercise common residual decoding and exact target-hold cadence without ONNX.

    With ``--viewer`` this is also the only mode that runs physics on the
    grasp scene while you watch it.  ``--view-scene`` deliberately does not
    step, so the sticks there sit wherever the reset put them; if you want to
    know whether the hand actually HOLDS them, it has to be this one.

    The zero action means the hand keeps requesting its reset pose, so any
    stick motion reported below is the scene settling or failing on its own,
    not something a policy did.
    """

    if policy_steps < 1:
        raise ValueError("--policy-steps must be positive.")
    class ZeroPolicy:
        def infer(self, observation):
            return np.zeros(ACTION_DIM, dtype=np.float32)

    runner = PolicyRunner(hand, ZeroPolicy(), PolicyObservationAdapter())
    runner.reset()
    scheduler = MujocoScheduler(hand, viewer=viewer, realtime=realtime)
    start_count = hand.physics_step_count

    try:
        sticks_at_start = hand.get_stick_poses_in_palm().reshape(-1, 7).copy()
    except Exception:
        sticks_at_start = None

    if sticks_at_start is not None:
        print(
            f"[STICK DRIFT] palm-frame displacement from the reset pose, "
            f"zero action ({policy_steps} steps = {policy_steps / 30.0:.1f} s)"
        )
        print(f"  {'t[s]':>6}" + "".join(f"{f'Stick{i + 1}':>12}" for i in range(len(sticks_at_start))))

    report_every = max(1, policy_steps // 10)

    for step in range(policy_steps):
        scheduler.run_policy_tick(runner)

        if sticks_at_start is None or (step + 1) % report_every:
            continue

        now = hand.get_stick_poses_in_palm().reshape(-1, 7)
        moved = np.linalg.norm(now[:, :3] - sticks_at_start[:, :3], axis=1) * 1000.0
        print(f"  {(step + 1) / 30.0:6.2f}" + "".join(f"{v:9.1f}mm" for v in moved))

    actual_steps = hand.physics_step_count - start_count
    expected_steps = policy_steps * hand.physics_substeps
    passed = actual_steps == expected_steps and hand.health().ok
    print(
        f"backend smoke: {'PASS' if passed else 'FAIL'} "
        f"policy_steps={policy_steps} physics_steps={actual_steps}/{expected_steps}"
    )
    if sticks_at_start is not None:
        print(
            "  NOTE: this is NOT a grasp test. The zero action decodes to "
            "target = q_current, so the hand re-aims at its own drift and the "
            "grasp unwinds by construction. Use --hold-pose for that."
        )
    return passed


def hold_pose(
    hand: MujocoWujiHand,
    policy_steps: int,
    viewer=None,
    realtime: bool = False,
) -> bool:
    """Hold one fixed joint target and report whether the sticks stay put.

    The target never changes, so this asks the only question that matters for
    a grasp: with the hand actively holding the pose it was recorded in, does
    the object stay in it?

    Contrast with ``--smoke-backend``, whose zero action decodes to
    ``target = q_current`` -- the hand re-aims at its own drift, so the grasp
    decays no matter how good the contact model is.  Measured over 10 s:
    fixed target left Stick1 at 47.7 mm, zero action at 387926 mm.
    """

    if policy_steps < 1:
        raise ValueError("--policy-steps must be positive.")

    import mujoco

    from ..common.isaac_reset import ISAAC_PREGRASP_JOINT_POSITIONS_RAD

    target = np.clip(
        ISAAC_PREGRASP_JOINT_POSITIONS_RAD,
        COMMAND_TARGET_LIMITS[:, 0],
        COMMAND_TARGET_LIMITS[:, 1],
    ).astype(np.float32)

    hand.reset()
    scheduler = MujocoScheduler(hand, viewer=viewer, realtime=realtime)
    at_start = hand.get_stick_poses_in_palm().reshape(-1, 7).copy()
    joints_at_start = hand.read_joint_positions().copy()

    print("[HOLD POSE]")
    print(f"  target: Isaac pregrasp, held for {policy_steps} steps "
          f"= {policy_steps / 30.0:.1f} s")
    print("  the target never changes, so any stick motion is the grasp failing")
    print(f"  {'t[s]':>6}" + "".join(f"{f'Stick{i + 1}':>12}" for i in range(len(at_start)))
          + f"{'관절 이탈':>12}")

    report_every = max(1, policy_steps // 10)

    for step in range(policy_steps):
        hand.write_joint_position_targets(target)
        scheduler.hold_policy_target()

        if (step + 1) % report_every:
            continue

        now = hand.get_stick_poses_in_palm().reshape(-1, 7)
        moved = np.linalg.norm(now[:, :3] - at_start[:, :3], axis=1) * 1000.0
        drift = np.abs(hand.read_joint_positions() - joints_at_start).max() * 1000.0
        print(f"  {(step + 1) / 30.0:6.2f}" + "".join(f"{v:9.1f}mm" for v in moved)
              + f"{drift:9.1f}mrad")

    print("\n  [CONTACTS AT END]")
    data = hand.data
    body_name = lambda geom: (
        mujoco.mj_id2name(hand.model, mujoco.mjtObj.mjOBJ_BODY, hand.model.geom_bodyid[geom])
        or "?"
    ).replace("right_", "")
    stick_contacts = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = body_name(contact.geom1), body_name(contact.geom2)
        if "stick" not in first + second:
            continue
        stick_contacts += 1
        print(f"    {contact.dist * 1000:8.3f} mm   {first} <-> {second}")
    if not stick_contacts:
        print("    스틱과 닿아 있는 것이 하나도 없다 - 이미 놓쳤다")

    final = hand.get_stick_poses_in_palm().reshape(-1, 7)
    worst = float(np.linalg.norm(final[:, :3] - at_start[:, :3], axis=1).max() * 1000.0)
    held = worst < 15.0
    print(f"\n  hold: {'PASS' if held else 'FAIL'}  "
          f"최대 이동 {worst:.1f} mm (기준 15 mm)")
    print("  기준 15 mm 는 임의로 고른 값이다. Isaac 에서 같은 측정을 하기 전까지 "
          "'몇 mm 면 정상인지'는 모른다.")
    return held


def inspect_camera(hand: MujocoWujiHand) -> None:
    import mujoco

    from ..vision.aruco_perception import CameraCalibration, DEFAULT_CALIBRATION_PATH

    camera_id = mujoco.mj_name2id(
        hand.model, mujoco.mjtObj.mjOBJ_CAMERA, "d435_rgb"
    )
    calibration = CameraCalibration.load()
    print("[D435 RGB CAMERA]")
    print(f"snapshot: {DEFAULT_CALIBRATION_PATH}")
    print(f"resolution: {calibration.width}x{calibration.height} @ 15 Hz")
    print(f"K:\n{calibration.matrix}")
    print(f"distortion: {calibration.distortion}")
    print(f"MuJoCo focalpixel: {hand.model.cam_intrinsic[camera_id, :2]}")
    print("marker dictionary: DICT_4X4_50; primary IDs: Stick1=0, Stick2=2")
    _print_scene_frames()


def _print_scene_frames() -> None:
    from ..vision.sim_aruco import (T_BASE_CAMERA, T_BASE_PALM)
    from ..common.stick_pose import rotation_matrix_to_quaternion_wxyz

    palm_quaternion = rotation_matrix_to_quaternion_wxyz(T_BASE_PALM[:3, :3])
    camera_quaternion = rotation_matrix_to_quaternion_wxyz(T_BASE_CAMERA[:3, :3])
    print("[SCENE FRAMES: Base +X=up, +Y=right, +Z=forward]")
    print(f"  T_BASE_PALM position: {T_BASE_PALM[:3, 3]}")
    print(f"  T_BASE_PALM quaternion wxyz: {palm_quaternion}")
    print(f"  T_BASE_PALM rotation:\n{T_BASE_PALM[:3, :3]}")
    print(f"  Palm axes in Base (columns +X,+Y,+Z):\n{T_BASE_PALM[:3, :3]}")
    print(f"  T_BASE_CAMERA position: {T_BASE_CAMERA[:3, 3]}")
    print(f"  T_BASE_CAMERA quaternion wxyz: {camera_quaternion}")
    print(f"  T_BASE_CAMERA rotation:\n{T_BASE_CAMERA[:3, :3]}")
    print(f"  Camera optical forward (+Z) in Base: {T_BASE_CAMERA[:3, 2]}")


def view_reset_scene(hand: MujocoWujiHand) -> None:
    """Show the backend-initialized Isaac reset scene without policy inference."""

    viewer = _launch_viewer(hand)
    clamped_names = tuple(
        POLICY_JOINT_NAMES[index] for index in np.flatnonzero(hand.last_reset_clamped)
    )
    print("[RESET SCENE VIEWER]")
    print("  hand: Isaac functional-pregrasp request")
    print("  sticks: Isaac centers/shaft with ID0/ID2 on the upward-facing surface")
    print("  visible fixture: base plate, 2020 hand post, 4040 camera post, D435 proxy")
    print("  frame axes: red=+X, green=+Y, blue=+Z")
    print(f"  nominal-limit clamp: {clamped_names or 'none'}")
    print("  simulation is paused; close the viewer window to exit")
    _print_scene_frames()
    try:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    finally:
        _close_viewer_and_wait(viewer)


def view_calibrated_camera(hand: MujocoWujiHand) -> None:
    """Display the calibrated D435 viewpoint in MuJoCo's native GUI.

    The deployment environment intentionally may use a headless OpenCV build;
    visualization therefore must not depend on ``cv2.imshow``.  ArUco still
    consumes the exact 1280x720 offscreen pixels from ``MujocoCameraSource``.
    """

    import mujoco

    camera_id = mujoco.mj_name2id(
        hand.model, mujoco.mjtObj.mjOBJ_CAMERA, "d435_rgb"
    )
    if camera_id < 0:
        raise RuntimeError("MuJoCo camera 'd435_rgb' is missing.")
    viewer = _launch_viewer(hand)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id
    print("[D435 RGB VIEW] calibrated optical-camera viewpoint")
    print("  close the MuJoCo viewer window to exit")
    print("  ArUco input remains the exact offscreen 1280x720 render")
    try:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    finally:
        _close_viewer_and_wait(viewer)


def validate_aruco(hand: MujocoWujiHand) -> bool:
    from ..vision.aruco_perception import ArucoStickPoseProvider
    from ..backends.mujoco_perception import (
        MujocoCameraSource,
        MujocoGroundTruthStickPoseProvider,
    )
    from ..vision.sim_aruco import frontal_camera_validation_stick_poses
    from ..common.stick_pose import (
        quaternion_geodesic_error_deg,
        quaternion_to_rotation_matrix_wxyz,
    )

    hand.set_stick_poses_in_palm(frontal_camera_validation_stick_poses())
    expected = MujocoGroundTruthStickPoseProvider(hand).sample()
    camera = MujocoCameraSource(hand)
    provider = ArucoStickPoseProvider(camera)
    passed = True
    try:
        actual = provider.sample()
        print("[RENDERED ARUCO / MUJOCO GROUND TRUTH]")
        for index, (ground_truth, estimate) in enumerate(zip(
            (expected.stick1, expected.stick2), (actual.stick1, actual.stick2)
        ), start=1):
            position_mm = float(np.linalg.norm(ground_truth[:3] - estimate[:3]) * 1000.0)
            rotation_deg = quaternion_geodesic_error_deg(
                ground_truth[3:], estimate[3:]
            )
            gt_axis = quaternion_to_rotation_matrix_wxyz(ground_truth[3:])[:, 1]
            estimated_axis = quaternion_to_rotation_matrix_wxyz(estimate[3:])[:, 1]
            axis_deg = float(np.degrees(np.arccos(np.clip(gt_axis @ estimated_axis, -1, 1))))
            item_passed = position_mm < 5.0 and rotation_deg < 5.0 and axis_deg < 5.0
            passed &= item_passed
            print(
                f"  [{'PASS' if item_passed else 'FAIL'}] Stick{index}: "
                f"position={position_mm:.3f} mm rotation={rotation_deg:.3f} deg "
                f"shaft_axis={axis_deg:.3f} deg"
            )
        detections = 0
        sample_count = 20
        for sample_index in range(sample_count):
            phase = 2.0 * np.pi * sample_index / sample_count
            hand.set_stick_poses_in_palm(frontal_camera_validation_stick_poses(
                camera_x_positions=(
                    -0.12 + 0.004 * np.sin(phase),
                    0.10 + 0.004 * np.cos(phase),
                ),
                camera_y=0.092 + 0.003 * np.cos(phase),
            ))
            provider.reset()
            try:
                provider.sample()
                detections += 1
            except RuntimeError:
                pass
        availability = detections / sample_count
        availability_passed = availability > 0.95
        passed &= availability_passed
        print(
            f"  [{'PASS' if availability_passed else 'FAIL'}] both-stick pose "
            f"availability={detections}/{sample_count} ({availability:.1%})"
        )
    finally:
        camera.close()
    print(f"rendered ArUco validation: {'PASS' if passed else 'FAIL'}")
    return passed


def create_stick_provider(hand: MujocoWujiHand, name: str):
    """Composition-root selection; common runner remains backend-neutral."""

    if name == "synthetic":
        from ..common.perception import SyntheticStickPoseProvider

        return SyntheticStickPoseProvider(), None
    if name == "ground-truth":
        from ..backends.mujoco_perception import MujocoGroundTruthStickPoseProvider

        return MujocoGroundTruthStickPoseProvider(hand), None
    if name == "aruco":
        from ..vision.aruco_perception import ArucoStickPoseProvider
        from ..backends.mujoco_perception import MujocoCameraSource

        camera = MujocoCameraSource(hand)
        return ArucoStickPoseProvider(camera), camera
    raise ValueError(f"Unknown stick provider {name!r}.")


def _close_viewer_and_wait(active_viewer, timeout_s: float = 5.0) -> None:
    """Wait for the passive viewer thread before model/data destruction.

    MuJoCo's public ``Handle.__exit__`` requests asynchronous shutdown but does
    not join the Linux viewer thread.  Returning immediately can let Python
    destroy ``MjModel``/``MjData`` while that thread is still tearing down GLFW,
    which was observed as a post-PASS segmentation fault.  ``Handle.m`` remains
    non-None until the viewer-owned simulate object has actually been destroyed.
    """

    active_viewer.close()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if active_viewer.m is None:
                return
        except Exception:
            # The C++ simulate object has already been destroyed.
            return
        time.sleep(0.01)
    print(
        "[WARNING] MuJoCo viewer did not confirm shutdown within "
        f"{timeout_s:.1f}s."
    )


def main() -> int:
    args = build_argument_parser().parse_args()
    if not any(
        (
            args.inspect_model,
            args.inspect_joints,
            args.inspect_contract,
            args.test_joints,
            args.onnx_only,
            args.run_policy,
            args.validate_fk,
            args.validate_aruco,
            args.inspect_camera,
            args.view_camera,
            args.view_scene,
            args.smoke_backend,
            args.hold_pose,
        )
    ):
        args.inspect_model = True

    if args.inspect_contract:
        print(contract_summary())
        print("\n[PHYSICAL NOMINAL LIMITS]\n", OFFICIAL_NOMINAL_PHYSICAL_LIMITS)
        print("\n[OBSERVATION NORMALIZATION LIMITS]\n", OBSERVATION_NORMALIZATION_LIMITS)
        print("\n[COMMAND TARGET LIMITS]\n", COMMAND_TARGET_LIMITS)
        print()
        print(pending_validation_report())
        return 0

    hand = MujocoWujiHand(
        args.model,
        physical_limit_tolerance_rad=args.joint_limit_tolerance,
        controller_gains=args.controller_gains,
        physics_substeps=args.physics_substeps,
        integrator=args.integrator,
    )
    if args.inspect_model:
        inspect_model(hand)
        return 0
    if args.inspect_joints:
        print(format_joint_mapping(hand.model, hand.mapping))
        return 0
    if args.test_joints:
        return 0 if test_single_joint_commands(
            hand, args.small_delta, args.joint_test_physics_steps
        ) else 1
    if args.validate_fk:
        return 0 if validate_fingertip_fk(hand) else 1
    if args.validate_aruco:
        return 0 if validate_aruco(hand) else 1
    if args.inspect_camera:
        inspect_camera(hand)
        return 0
    if args.view_camera:
        view_calibrated_camera(hand)
        return 0
    if args.view_scene:
        view_reset_scene(hand)
        return 0
    if args.hold_pose:
        viewer = _launch_viewer(hand) if args.viewer else None
        try:
            ok = hold_pose(hand, args.policy_steps, viewer, args.realtime)
        finally:
            if viewer is not None:
                _close_viewer_and_wait(viewer)
        return 0 if ok else 1
    if args.smoke_backend:
        viewer = _launch_viewer(hand) if args.viewer else None
        try:
            ok = smoke_backend(hand, args.policy_steps, viewer, args.realtime)
        finally:
            if viewer is not None:
                _close_viewer_and_wait(viewer)
        return 0 if ok else 1

    policy = OnnxPolicy(args.policy, OBSERVATION_DIM, ACTION_DIM)
    stick_provider, camera = create_stick_provider(hand, args.stick_provider)
    try:
        if args.onnx_only:
            return 0 if run_onnx_only(
                hand, policy, args.mode, args.debug_observation, stick_provider
            ) else 1
        run_closed_loop(
            hand,
            policy,
            args.mode,
            args.policy_steps,
            args.print_interval,
            args.viewer,
            args.realtime,
            args.debug_observation,
            stick_provider,
        )
    finally:
        if camera is not None:
            camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
