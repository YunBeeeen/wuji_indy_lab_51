"""hand_object construction / sensor / force / retract probe (2026-08-06).

Runs the scripted ``hand_object`` episode with a zero finger action and checks
everything that has never been executed before.  This is the smoke test that
has to pass before any calibration or training, because the whole task rests on
a contact-force sign convention that was *derived* from an isaacsim unit test
rather than measured.

Checks performed:

    1.  dimensions: action 20D, observation 103D (identical to hand_move, so a
        hand_move checkpoint loads)
    2.  the two cube-filtered contact sensors exist and their ``force_matrix_w``
        has the expected ``(N, B, M, 3)`` shape
    3.  reset geometry: nothing is in contact at spawn, the cube sits on the
        support, and the support column clears the hand and both sticks
    4.  scripted OPEN/CLOSE switch happens once, at the configured time
    5.  the support retracts smoothly (no teleport) and only within its window,
        and stays down afterwards
    6.  per-environment independence of the support state
    7.  contact-force sign: with ``--squeeze`` the probe drives the fingers
        closed and reports whether both inward forces come out positive, which
        is what validates STICK1/STICK2_CUBE_FORCE_SIGN
    8.  no NaN anywhere in the force pipeline, including when the tips coincide
    9.  drop termination fires only after the support is fully retracted

Run only while no training is in progress (project rule).

    python scripts/debug/hand_object_probe.py --headless
    python scripts/debug/hand_object_probe.py --headless --squeeze
    python scripts/debug/hand_object_probe.py --squeeze          # GUI
    python scripts/debug/hand_object_probe.py --headless --num_envs 4
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="hand_object")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--squeeze",
    action="store_true",
    help=(
        "drive the fingers towards CLOSE with a constant positive residual so a"
        " real contact force appears; without it the forces stay at zero and the"
        " sign convention cannot be checked."
    ),
)
parser.add_argument(
    "--squeeze_action",
    type=float,
    default=0.6,
    help="constant action value used for --squeeze, in the policy's [-1, 1] range",
)
parser.add_argument(
    "--yaw_deg",
    type=float,
    default=0.0,
    help=(
        "relative yaw about the hand's own z for this run. The real goal pose is"
        " unmeasured (the HAND_OBJECT_TARGET_* constants are None); this only"
        " fills one in so the scripted trajectory can run - not a calibration."
    ),
)
parser.add_argument(
    "--uncalibrated",
    action="store_true",
    help=(
        "leave the goal-pose constants unset, as a calibration session does."
        " Exercises the fallback where the scripted trajectory holds the spawn"
        " pose instead of raising - the path that blocked the first calibration"
        " attempt."
    ),
)
parser.add_argument(
    "--goal_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("DX", "DY", "DZ"),
    help="goal root position as an offset from the spawn position, in metres.",
)
parser.add_argument(
    "--cube_at_tips",
    action="store_true",
    help=(
        "PROBE-ONLY geometry: put the cube at the measured reset tip midpoint on a"
        " short floating stub, so a contact actually forms at yaw 0 and the force"
        " sign can be checked. The shipped config instead places the cube where the"
        " tips end up after the calibrated yaw, on a column from the ground."
    ),
)
parser.add_argument(
    "--load_run",
    type=str,
    default=None,
    help=(
        "hand_object experiment 안의 런 이름. 정책 없이는 젓가락이 닫히지 않아"
        " 접촉력이 0이고 부호 검증이 불가능하다. hand_move 체크포인트는 --checkpoint 로."
    ),
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="체크포인트 .pt 경로 직접 지정"
)
parser.add_argument(
    "--load_checkpoint", type=str, default="model_.*.pt", help="체크포인트 이름 패턴"
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=15,
    help="policy steps between debug text lines (0 disables)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402
from isaac_neuromeka.tasks.manipulation.hand_grasp import (  # noqa: E402
    hand_object_env_cfg as ho_cfg,
    hand_object_mdp as ho_mdp,
)

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((PASS if ok else FAIL, name, detail))
    return ok


def fmt(values, digits: int = 4) -> str:
    return "(" + ", ".join(f"{float(v):+.{digits}f}" for v in values) + ")"


def load_policy(env, task: str):
    """Wrap the env for rsl_rl and load a checkpoint. Returns (wrapped, policy, path).

    A trained policy is what makes the force check meaningful: a constant action
    does not close the tips, so the contact force stays at zero and the sign
    convention cannot be observed at all.
    """
    import importlib.metadata as metadata

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root = os.path.abspath(
            os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        )
        resume_path = get_checkpoint_path(
            log_root, args_cli.load_run, args_cli.load_checkpoint
        )

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(
        wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    # strict=True inside: a shape mismatch here is exactly the hand_move
    # checkpoint compatibility check, so let it raise rather than catching it.
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
    return wrapped, policy, resume_path


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    # The calibration constants are deliberately unset; this probe is exactly
    # the thing that has to work before they can be measured.
    env_cfg.require_calibration = False
    schedule = ho_mdp.HAND_OBJECT_SCHEDULE
    # Fill in the unmeasured yaw so the scripted trajectory can run at all.
    # Both command terms hold *copies* of the module schedule (configclass
    # deep-copies mutable defaults), so setting the module object alone is not
    # enough - the copies in this env_cfg have to be set too.
    yaw = math.radians(args_cli.yaw_deg)
    euler = (0.0, 0.0, yaw)
    # HAND_ROOT_POS is the spawn position; the goal is it plus the offset.
    from isaac_neuromeka.tasks.manipulation.hand_grasp.hand_grasp_env_cfg import (
        HAND_ROOT_POS,
    )

    goal_pos = tuple(
        float(a) + float(b) for a, b in zip(HAND_ROOT_POS, args_cli.goal_offset)
    )
    override_goal = args_cli.uncalibrated or args_cli.yaw_deg != 0.0 or any(
        v != 0.0 for v in args_cli.goal_offset
    )
    if args_cli.uncalibrated:
        print("[probe] uncalibrated: goal pose left as None (spawn pose held)")
    elif not override_goal:
        print(
            f"[probe] using the calibrated goal pose from hand_object_mdp.py: "
            f"pos {schedule.target_root_pos_e}, euler {schedule.target_euler_rad}"
        )
    if override_goal:
        for cfg in (
            schedule,
            env_cfg.commands.open_close.schedule,
            env_cfg.commands.root_orientation.schedule,
            env_cfg.commands.support.schedule,
        ):
            cfg.target_euler_rad = None if args_cli.uncalibrated else euler
            cfg.target_root_pos_e = None if args_cli.uncalibrated else goal_pos
    schedule.validate()
    print(f"[probe] goal root pose: pos {goal_pos}, euler_deg (0, 0, {args_cli.yaw_deg})")

    if args_cli.cube_at_tips:
        # Measured reset tip midpoint (see hand_object_env_cfg's docstring).
        # The stub floats: a kinematic body needs no column to the ground, and a
        # full-height column this close to the palm would intersect the fingers.
        cube_pos = (0.1134, 0.0223, 0.6043)
        stub_height = 0.005
        top_z = cube_pos[2] - 0.5 * ho_cfg.OBJECT_SIZE[2] - ho_cfg.SUPPORT_TOP_CLEARANCE
        env_cfg.scene.object.init_state.pos = cube_pos
        env_cfg.scene.object_support.init_state.pos = (
            cube_pos[0],
            cube_pos[1],
            top_z - 0.5 * stub_height,
        )
        env_cfg.scene.object_support.spawn.size = (
            ho_cfg.SUPPORT_CROSS_SECTION,
            ho_cfg.SUPPORT_CROSS_SECTION,
            stub_height,
        )
        print(
            f"[probe] cube_at_tips: cube {cube_pos}, "
            f"{stub_height * 1000:.0f}mm floating stub beneath"
        )

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    obs_dict, _ = env.reset()

    device = env.device
    dt = env.step_dt
    num_envs = env.num_envs

    # -- 1. dimensions --------------------------------------------------
    action_dim = env.action_manager.total_action_dim
    policy_obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
    obs_dim = int(policy_obs.shape[1])
    check("action 20D (hand_move 와 동일)", action_dim == 20, f"{action_dim}")
    check("observation 103D (hand_move 와 동일)", obs_dim == 103, f"{obs_dim}")

    # -- 2. sensors -----------------------------------------------------
    for sensor_name in (ho_cfg.STICK1_CUBE_SENSOR, ho_cfg.STICK2_CUBE_SENSOR):
        present = sensor_name in env.scene.sensors
        if not check(f"센서 '{sensor_name}' 존재", present):
            continue
        matrix = env.scene.sensors[sensor_name].data.force_matrix_w
        ok = matrix is not None and matrix.ndim == 4 and matrix.shape[0] == num_envs
        check(
            f"'{sensor_name}'.force_matrix_w shape (N,B,M,3)",
            ok,
            "None" if matrix is None else str(tuple(matrix.shape)),
        )

    # -- 3. reset geometry ----------------------------------------------
    cube = env.scene[ho_cfg.OBJECT.name]
    support = env.scene[ho_cfg.OBJECT_SUPPORT.name]
    robot = env.scene["robot"]
    origin = env.scene.env_origins

    tip1, tip2 = ho_mdp.stick_tip_positions_w(
        env, ho_cfg.STICK_1, ho_cfg.STICK_2, ho_cfg.STICK_TIP_OFFSET_O
    )
    cube_pos = cube.data.root_pos_w
    support_pos = support.data.root_pos_w

    # play 에서 P 키가 찍는 것과 같은 블록. GUI 없이도 포맷과 계산이 도는지 확인한다.
    from isaac_neuromeka.tasks.manipulation.hand_grasp import hand_move_mdp as hm_mdp

    print()
    print(
        hm_mdp.geometry_report(
            env,
            stick1_cfg=ho_cfg.STICK_1,
            stick2_cfg=ho_cfg.STICK_2,
            tip_offset_o=ho_cfg.STICK_TIP_OFFSET_O,
        )
    )

    print("\n--- reset 기하 (env 0, env-local) " + "-" * 34)
    print(f"  Stick1 tip        {fmt(tip1[0] - origin[0])}")
    print(f"  Stick2 tip        {fmt(tip2[0] - origin[0])}")
    print(f"  tip 중점          {fmt(0.5 * (tip1[0] + tip2[0]) - origin[0])}")
    print(f"  tip 간격          {float((tip1[0] - tip2[0]).norm()) * 1000:.2f} mm")
    print(f"  cube 중심         {fmt(cube_pos[0] - origin[0])}")
    print(f"  support 중심      {fmt(support_pos[0] - origin[0])}")

    # Cube must be resting on the support, not floating and not sunk into it.
    cube_bottom = cube_pos[:, 2] - 0.5 * ho_cfg.OBJECT_SIZE[2]
    support_top = support_pos[:, 2] + 0.5 * env_cfg.scene.object_support.spawn.size[2]
    gap = cube_bottom - support_top
    check(
        "cube 가 support 위에 접촉 없이 얹힘 (0 <= gap <= 2mm)",
        bool(((gap >= -1.0e-6) & (gap <= 0.002)).all()),
        f"gap = {float(gap[0]) * 1000:.3f} mm",
    )

    # Nothing may be touching at spawn: the sticks, the fingers and the column.
    hand_xy = robot.data.body_pos_w[:, :, :2] - origin[:, None, :2]
    support_xy = support_pos[:, :2] - origin[:, :2]
    hand_clearance = (hand_xy - support_xy[:, None, :]).norm(dim=-1).min(dim=1).values
    half_diag = 0.5 * ho_cfg.SUPPORT_CROSS_SECTION * (2 ** 0.5)
    clearance_mm = float((hand_clearance - half_diag).min()) * 1000
    if args_cli.cube_at_tips:
        # --cube_at_tips 는 큐브를 tip 중점에 놓으므로 기둥이 손가락 바로 옆이다.
        # 이 배치는 접촉력 검증용이지 배송되는 기하가 아니라서 판정 대상이 아님.
        results.append(("SKIP", "손 링크가 support 기둥에서 30mm 이상",
                        f"--cube_at_tips 전용 기하 (실측 {clearance_mm:.1f} mm)"))
    else:
        check("손 링크 원점이 support 기둥에서 30mm 이상",
              clearance_mm > 30.0, f"최소 {clearance_mm:.1f} mm")

    net1 = env.scene.sensors[ho_cfg.STICK1_CUBE_SENSOR].data.force_matrix_w
    net2 = env.scene.sensors[ho_cfg.STICK2_CUBE_SENSOR].data.force_matrix_w
    if net1 is not None and net2 is not None:
        contact_at_reset = float(net1.abs().max() + net2.abs().max())
        check(
            "reset 에서 스틱-큐브 접촉 없음",
            contact_at_reset < 1.0e-6,
            f"max|F| = {contact_at_reset:.2e} N",
        )

    # -- rollout --------------------------------------------------------
    #
    # Everything below is keyed on each environment's OWN episode clock
    # (``episode_length_buf``), never on the loop counter.  Environments reset
    # independently the moment the cube drops, so a check written against the
    # global step index reads a mixture of two episodes and reports nonsense -
    # which is exactly what the first version of this probe did.
    support_term = env.command_manager.get_term("support")
    orientation_term = env.command_manager.get_term("root_orientation")
    close_start = schedule.open_close_start_time_s
    retract_deadline = schedule.support_retract_deadline_time_s
    retract_end = schedule.support_retract_end_time_s
    distance = schedule.support_retract_distance_m
    per_step = distance * dt / schedule.support_retract_duration_s

    use_policy = bool(args_cli.load_run or args_cli.checkpoint)
    wrapped = policy = None
    if use_policy:
        wrapped, policy, resume_path = load_policy(env, args_cli.task)
        print(f"[probe] policy: {resume_path}")
        obs, _ = wrapped.get_observations(), None
    constant_action = torch.zeros(num_envs, action_dim, device=device)
    if args_cli.squeeze:
        constant_action[:] = args_cli.squeeze_action

    # rollout long enough to cover several episodes even with early drops
    total_steps = int(round(3.0 * schedule.episode_length_s / dt))

    mode_violation = 0
    retract_violation = 0
    retract_rise_violation = 0
    triggers: list[tuple[float, bool]] = []   # (시각, 파지조건으로 발동했는가)
    pose_error_after_move = []
    stage1_z_drift = []      # 정렬 구간에서 z 가 스폰 높이를 벗어난 양
    stage2_xy_drift = []     # 하강 구간에서 x,y 가 목표를 벗어난 양
    max_offset_jump = 0.0
    nan_seen = False
    drop_before_retract = 0
    episodes_done = 0
    contacts: list[tuple[float, float, float]] = []   # (f1, f2, elapsed)
    hold_forces: list[float] = []                      # min(f1,f2) while retracted
    previous_offset = support_term.support_offset_m.clone()
    previous_elapsed = env.episode_length_buf.clone()
    drop_times: list[float] = []
    post_retract_heights: list[float] = []

    print("\n--- rollout " + "-" * 56)
    for step in range(total_steps):
        # 종료항은 step "안에서" 평가되고 그 직후 _reset_idx 가 support 를 되돌리므로,
        # step 이 끝난 뒤 retracted_gate 를 읽으면 항상 0 이다(리셋된 값).
        # 평가가 일어난 시각을 스텝 진입 전 시계로 따로 잡아둔다.
        eval_time = (env.episode_length_buf.float() + 1.0) * dt
        if use_policy:
            with torch.inference_mode():
                action = policy(obs)
            obs, _, dones, _ = wrapped.step(action)
        else:
            _, _, terminated, truncated, _ = env.step(constant_action)
            dones = terminated | truncated

        elapsed = env.episode_length_buf.float() * dt
        just_reset = env.episode_length_buf < previous_elapsed
        previous_elapsed = env.episode_length_buf.clone()
        episodes_done += int(dones.sum())

        # -- 4. OPEN/CLOSE against each env's own clock
        mode = env.command_manager.get_command("open_close")
        should_close = elapsed >= close_start
        mode_violation += int(((mode[:, 1] > 0.5) != should_close).sum())

        # -- 5. support retract invariants.
        #    The offset is no longer a function of elapsed time: the column goes
        #    as soon as the grasp is real and at the deadline at the latest.  So
        #    rather than one expected value there are three properties:
        #      a) it never moves before CLOSE begins (nothing to grip yet), and
        #      b) it is fully down once the deadline path would have finished
        #         (the fallback must be unconditional), and
        #      c) it never rises again inside an episode (the trigger latches).
        offset = support_term.support_offset_m
        too_early = (offset > 1.0e-9) & (elapsed < close_start)
        too_late = (offset < distance - 1.0e-6) & (elapsed >= retract_end)
        retract_violation += int((too_early | too_late).sum())

        started = (offset > 1.0e-9) & (previous_offset <= 1.0e-9) & (~just_reset)
        by_grasp = support_term.metrics["retract_by_grasp"]
        for i in torch.nonzero(started).flatten().tolist():
            triggers.append((float(elapsed[i]), float(by_grasp[i]) > 0.5))

        delta = offset - previous_offset
        if bool((~just_reset).any()):
            retract_rise_violation += int((delta[~just_reset] < -1.0e-9).sum())
        jump = delta.abs()
        # A reset legitimately snaps the column back up; only smooth motion
        # within a continuing episode is being checked here.
        max_offset_jump = max(max_offset_jump, float(jump[~just_reset].max()) if (~just_reset).any() else 0.0)
        previous_offset = offset.clone()

        # -- 5b. the root must actually ARRIVE at the goal, not just turn.
        #    Sampled during the settle window, after the interpolation has
        #    finished and before CLOSE begins.
        settled = (elapsed >= schedule.slerp_end_time_s) & (elapsed < close_start)
        if bool(settled.any()):
            goal = orientation_term.position_goal_w
            actual = env.scene["robot"].data.root_link_pos_w
            pose_error_after_move.append(
                float((actual - goal).norm(dim=-1)[settled].max())
            )

        # -- 5d. two-stage approach: align at spawn height, then descend.
        #    Checked on the *command* rather than the measured pose so a PD lag
        #    does not masquerade as a trajectory bug.
        target = orientation_term.target_pos_w
        if target is not None:
            fraction = schedule.approach_fraction
            move_a = (elapsed - schedule.initial_hold_time_s) / max(
                schedule.rotation_interpolation_time_s, 1e-9
            )
            start_z = orientation_term.start_pos_w[:, 2]
            goal_xy = orientation_term.position_goal_w[:, :2]
            aligning = (move_a > 0.0) & (move_a < fraction)
            descending = (move_a > fraction) & (move_a <= 1.0)
            if bool(aligning.any()):
                stage1_z_drift.append(
                    float((target[:, 2] - start_z)[aligning].abs().max())
                )
            if bool(descending.any()):
                stage2_xy_drift.append(
                    float((target[:, :2] - goal_xy)[descending].abs().max())
                )

        # -- 5e. an unheld cube must actually fall clear of the column.
        #    The column sits under the cube, so too short a retract just lowers
        #    the pedestal and the cube lands back on it - which silently removes
        #    the whole point of the hold phase.
        cube_z = cube.data.root_pos_w[:, 2] - origin[:, 2]
        fully_out = support_term.retracted_gate > 0.5
        if bool(fully_out.any()):
            post_retract_heights.append(float(cube_z[fully_out].min()))

        # -- 7/8. forces
        inward1, inward2, axis, raw1, raw2 = ho_mdp.cube_inward_forces(
            env,
            ho_cfg.STICK_1,
            ho_cfg.STICK_2,
            ho_cfg.STICK_TIP_OFFSET_O,
            ho_cfg.STICK1_CUBE_SENSOR,
            ho_cfg.STICK2_CUBE_SENSOR,
        )
        if torch.isnan(torch.stack([inward1, inward2])).any() or torch.isnan(axis).any():
            nan_seen = True
        touching = (raw1.norm(dim=-1) > 1.0e-6) | (raw2.norm(dim=-1) > 1.0e-6)
        for i in torch.nonzero(touching).flatten().tolist():
            contacts.append((float(inward1[i]), float(inward2[i]), float(elapsed[i])))
        retracted = support_term.retracted_gate > 0.5
        for i in torch.nonzero(retracted & touching).flatten().tolist():
            hold_forces.append(float(min(inward1[i], inward2[i])))

        # -- 9. drop termination must not fire before the column is clear.
        #    Checked by evaluation *time*, not by reading retracted_gate after
        #    the step: the reset that follows a termination has already put the
        #    column back up by then, so that read is always 0 and the check
        #    would fail on every single drop.
        if "cube_dropped" in env.termination_manager.active_terms:
            dropped = env.termination_manager.get_term("cube_dropped")
            for i in torch.nonzero(dropped).flatten().tolist():
                t = float(eval_time[i])
                drop_times.append(t)
                if t < retract_end - 1.0e-6:
                    drop_before_retract += 1

        if args_cli.print_interval and step % args_cli.print_interval == 0:
            mode_text = "OPEN " if float(mode[0, 0]) > 0.5 else "CLOSE"
            print(
                f"  step={step:3d} t0={float(elapsed[0]):5.2f}s {mode_text}"
                f" support={float(offset[0]) * 1000:5.1f}mm"
                f" retracted={float(support_term.retracted_gate[0]):.0f}"
                f" f1={float(inward1[0]):+.5f} f2={float(inward2[0]):+.5f}"
                f" done={int(dones.sum())}"
            )

    check("OPEN/CLOSE 가 각 env 의 에피소드 시계와 일치", mode_violation == 0,
          f"불일치 {mode_violation} env-step")
    check("support 는 CLOSE 이전에 안 내려가고, 데드라인까지는 반드시 내려감",
          retract_violation == 0, f"위반 {retract_violation} env-step")
    check("support 가 도중에 다시 올라오지 않음 (latch)",
          retract_rise_violation == 0, f"상승 {retract_rise_violation} env-step")
    if triggers:
        grasp_fired = [t for t, g in triggers if g]
        clock_fired = [t for t, g in triggers if not g]
        print(
            f"  [트리거] 총 {len(triggers)}회 — 파지조건 {len(grasp_fired)}회"
            + (f" (평균 {sum(grasp_fired) / len(grasp_fired):.2f}s)" if grasp_fired else "")
            + f", 데드라인 {len(clock_fired)}회"
        )
        check("모든 트리거가 CLOSE 시작 이후",
              min(t for t, _ in triggers) >= close_start - 1.0e-6,
              f"최초 {min(t for t, _ in triggers):.2f}s, CLOSE {close_start:.2f}s")
    check("순간이동 없음 (리셋 제외, 스텝당 <= 예상 1.5배)",
          max_offset_jump <= per_step * 1.5 + 1.0e-9,
          f"최대 {max_offset_jump * 1000:.3f} mm/step, 예상 {per_step * 1000:.3f}")
    if stage1_z_drift:
        check("1단계(정렬)에서 높이가 스폰 z 유지", max(stage1_z_drift) < 1.0e-4,
              f"최대 {max(stage1_z_drift) * 1000:.3f} mm 이탈")
    if stage2_xy_drift:
        check("2단계(하강)에서 x,y 가 목표 고정 (순수 z 하강)",
              max(stage2_xy_drift) < 1.0e-4,
              f"최대 {max(stage2_xy_drift) * 1000:.3f} mm 이탈")

    if args_cli.uncalibrated:
        worst = max(pose_error_after_move) if pose_error_after_move else 0.0
        check("미보정 시 스폰 자세를 유지 (raise 하지 않음)", worst < 0.010,
              f"스폰 대비 최대 {worst * 1000:.2f} mm")
    elif pose_error_after_move:
        worst = max(pose_error_after_move)
        check("이동 완료 후 root 가 목표 위치에 도달 (오차 < 10mm)", worst < 0.010,
              f"최대 {worst * 1000:.2f} mm")
    else:
        results.append(("SKIP", "root 목표 위치 도달", "settle 구간 표본 없음"))
    if post_retract_heights:
        lowest = min(post_retract_heights)
        spawn_z = float(cube.data.default_root_state[0, 2])
        fell = spawn_z - lowest
        # With no trained cube policy the cube is not held, so it must fall at
        # least past the drop threshold; anything less means the column is
        # still holding it up.
        check("기둥 하강 후 안 잡힌 큐브가 실제로 낙하",
              fell > schedule.cube_drop_height_m,
              f"최저 {lowest:.4f} m (스폰 {spawn_z:.4f}), 낙하 {fell * 1000:.1f} mm "
              f"> 임계 {schedule.cube_drop_height_m * 1000:.0f} mm")
    check("force 파이프라인에 NaN 없음", not nan_seen)
    check("support 완전 하강 전 drop 종료 없음", drop_before_retract == 0,
          f"drop {len(drop_times)}회 중 {drop_before_retract}회가 이름"
          + (f", 최소 t={min(drop_times):.2f}s (retract 완료 {retract_end:.2f}s)"
             if drop_times else ""))
    print(f"\n  에피소드 종료 {episodes_done}회, 접촉 {len(contacts)} env-step")

    # -- 5c. manual override must leave the keyboard's position target alone.
    #    The scripted position trajectory writes the action term's target buffer
    #    every physics step, and that is the same buffer the translation keys
    #    write; if the command term does not stand down under manual override,
    #    I/K/J/L/U/O silently do nothing while the rotation keys still work.
    #
    #    Done straight after an explicit reset: an episode ending mid-test would
    #    re-capture the target from the live root pose and look like the bug.
    env.reset()
    orientation_term.enable_manual_override(True)
    env.command_manager.get_term("open_close").enable_manual_override(True)
    root_action = env.action_manager.get_term("root_action")
    before = root_action.target_root_pos_w.clone()
    nudge = torch.zeros(num_envs, 3, device=device)
    nudge[:, 0] = 0.01
    root_action.add_target_position_delta(nudge)
    _, _, terminated, truncated, _ = env.step(constant_action)
    if bool((terminated | truncated).any()):
        results.append(("SKIP", "manual override 시 키보드 이동 목표 유지",
                        "테스트 스텝에서 에피소드가 끝나 측정 불가"))
    else:
        moved = float((root_action.target_root_pos_w - before)[:, 0].min())
        check("manual override 시 키보드 이동 목표가 유지됨 (I/K 동작)",
              moved > 0.009, f"+10mm 명령 후 {moved * 1000:+.2f} mm 남음")
    orientation_term.enable_manual_override(False)
    env.command_manager.get_term("open_close").enable_manual_override(False)

    # -- 6. per-env independence ----------------------------------------
    if num_envs > 1:
        support_term._current_offset[:] = 0.0
        support_term._current_offset[0] = 0.5 * distance
        support_term._write_support_pose()
        z = support.data.root_pos_w[:, 2] - origin[:, 2]
        check("support 상태가 env 별로 독립",
              bool((z[1:] - z[1]).abs().max() < 1.0e-6)
              and float(z[0]) < float(z[1]) - 1.0e-4,
              f"env0 {float(z[0]):.4f} vs env1 {float(z[1]):.4f}")

    # -- 7. force sign --------------------------------------------------
    strong = [c for c in contacts if max(abs(c[0]), abs(c[1])) > 1.0e-3]
    if not strong:
        results.append(("SKIP", "force 부호 검증",
                        f"유의미한 접촉 없음 (접촉 {len(contacts)}, 최대 "
                        f"{max((max(abs(a), abs(b)) for a, b, _ in contacts), default=0.0):.2e} N)"
                        " — 학습된 정책(--load_run/--checkpoint)이 필요"))
    else:
        both_pos = sum(1 for a, b, _ in strong if a > 0 and b > 0)
        both_neg = sum(1 for a, b, _ in strong if a < 0 and b < 0)
        mixed = len(strong) - both_pos - both_neg
        best = max(strong, key=lambda c: min(c[0], c[1]))
        print(f"  유의 접촉 {len(strong)}: 둘다양수 {both_pos}, 둘다음수 {both_neg}, 혼합 {mixed}")
        print(f"  min(f1,f2) 최대: f1={best[0]:+.5f} f2={best[1]:+.5f} N (t={best[2]:.2f}s)")
        check("압착 시 inward force 두 개 모두 양수 (부호 규약 검증)",
              both_pos > both_neg, f"양수 {both_pos} vs 음수 {both_neg}"
              + ("  -> STICK1/2_CUBE_FORCE_SIGN 을 뒤집을 것" if both_neg > both_pos else ""))

    if hold_forces:
        hold = torch.tensor(hold_forces)
        print(f"\n  [force_saturation 참고] 기둥 하강 후 min(f1,f2):"
              f" 표본 {len(hold_forces)},"
              f" 중앙값 {float(hold.median()):.5f} N,"
              f" 90퍼센타일 {float(hold.quantile(0.9)):.5f} N,"
              f" 최대 {float(hold.max()):.5f} N")

    env.close()

    print("\n" + "=" * 72)
    for status, name, detail in results:
        print(f"  {status:4s} {name}" + (f"   [{detail}]" if detail else ""))
    failed = [r for r in results if r[0] == FAIL]
    print("=" * 72)
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")


if __name__ == "__main__":
    main()
    simulation_app.close()
