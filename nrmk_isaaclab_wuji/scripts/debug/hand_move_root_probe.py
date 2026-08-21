"""hand_move floating-root / orientation-schedule probe (2026-08-05).

Drives ``hand_move`` with a zero finger action and checks the scripted root
rotation machinery.  No training, no reward changes.

Scenario presets (``--scenario``):

    zero        range_x/y/z all (0, 0)          -> q_goal must equal q_start
    pitch       relative pitch fixed at +N deg
    roll        relative roll  fixed at +N deg
    yaw         relative yaw   fixed at +N deg
    full        the configured default range (random)

Checks performed:

    1.  dimensions: action 20D, observation 103D, action_history 20D
    2.  reset state matches the hand_grasp functional grasp
        (joint positions/targets, palm-relative stick poses, zero root velocity)
    3.  q_start captured after the reset events (equals the live root pose)
    4.  q_goal == q_start (x) q_delta, and the geodesic angle matches the
        commanded Euler magnitude for a single-axis scenario
    5.  SLERP timing: q_cmd == q_start before the hold ends, moving during the
        interpolation window, == q_goal from the SLERP end onward and for the
        rest of the episode
    6.  the root PD controller keeps running after the SLERP ends: the wrench
        buffer is refreshed every physics step and the root tracks q_goal
    7.  OPEN/CLOSE schedule: CLOSE until the start time, then alternating
        segments of the configured length
    8.  the root does not translate while it rotates
    9.  palm-relative stick pose and angular speed stay bounded

Run only while no training is in progress (project rule: no Isaac Sim probe
during train.py).

    python scripts/debug/hand_move_root_probe.py --headless --scenario zero
    python scripts/debug/hand_move_root_probe.py --headless --scenario pitch
    python scripts/debug/hand_move_root_probe.py --headless --scenario full
    python scripts/debug/hand_move_root_probe.py --scenario yaw --no_slerp
    python scripts/debug/hand_move_root_probe.py --scenario pitch   # GUI markers
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="hand_move")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--scenario",
    type=str,
    default="zero",
    choices=("zero", "pitch", "roll", "yaw", "full"),
)
parser.add_argument(
    "--angle_deg",
    type=float,
    default=10.0,
    help="fixed relative angle for the single-axis scenarios",
)
parser.add_argument("--no_slerp", action="store_true", help="set use_slerp=False")
parser.add_argument("--no_smoothstep", action="store_true", help="set use_smoothstep=False")
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

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    quat_apply_inverse,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402
from isaac_neuromeka.tasks.manipulation.hand_grasp.hand_grasp_env_cfg import (  # noqa: E402
    HAND_JOINT_NAMES,
    HAND_ROOT_POS,
    HAND_ROOT_ROT,
    PREGRASP_JOINT_POSITIONS,
    PREGRASP_STICK1_POSITION_P,
    PREGRASP_STICK1_QUATERNION_P,
    PREGRASP_STICK2_POSITION_P,
    PREGRASP_STICK2_QUATERNION_P,
)

PASS, FAIL = "PASS", "FAIL"


def fmt(values, digits: int = 5) -> list[float]:
    return [round(float(v), digits) for v in values]


def geodesic_deg(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
    delta = quat_mul(quat_a, quat_conjugate(quat_b))
    return torch.rad2deg(2.0 * torch.acos(delta[:, 0].abs().clamp(max=1.0)))


def apply_scenario(schedule) -> None:
    """Rewrite the sampling ranges in place for the selected scenario."""
    angle = math.radians(args_cli.angle_deg)
    zero = (0.0, 0.0)
    fixed = (angle, angle)
    if args_cli.scenario == "zero":
        schedule.range_x, schedule.range_y, schedule.range_z = zero, zero, zero
    elif args_cli.scenario == "roll":
        schedule.range_x, schedule.range_y, schedule.range_z = fixed, zero, zero
    elif args_cli.scenario == "pitch":
        schedule.range_x, schedule.range_y, schedule.range_z = zero, fixed, zero
    elif args_cli.scenario == "yaw":
        schedule.range_x, schedule.range_y, schedule.range_z = zero, zero, fixed
    # "full" leaves the configured default range untouched.
    if args_cli.no_slerp:
        schedule.use_slerp = False
    if args_cli.no_smoothstep:
        schedule.use_smoothstep = False
    schedule.validate()


class Probe:
    def __init__(self, env, orientation_term, open_close_term, schedule):
        self.env = env
        self.device = env.device
        self.robot = env.scene["robot"]
        self.stick1 = env.scene["stick1"]
        self.stick2 = env.scene["stick2"]
        self.palm_id = self.robot.find_bodies(["palm_link"])[0][0]
        self.hand_joint_ids, _ = self.robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
        self.root_term = env.action_manager.get_term("root_action")
        self.orientation = orientation_term
        self.open_close = open_close_term
        self.schedule = schedule
        self.zero_action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim), device=self.device
        )
        self.results: list[tuple[str, str, str]] = []
        self.samples: list[dict] = []

    # -- readouts -------------------------------------------------------
    def stick_pose_in_palm(self, stick):
        palm_pos = self.robot.data.body_pos_w[:, self.palm_id]
        palm_quat = self.robot.data.body_quat_w[:, self.palm_id]
        pos_p = quat_apply_inverse(palm_quat, stick.data.root_pos_w - palm_pos)
        quat_p = quat_mul(quat_conjugate(palm_quat), stick.data.root_quat_w)
        return pos_p.clone(), quat_p.clone()

    def stick_ang_speed_in_palm(self, stick) -> torch.Tensor:
        palm_ang_vel = self.robot.data.body_ang_vel_w[:, self.palm_id]
        return (stick.data.root_ang_vel_w - palm_ang_vel).norm(dim=-1)

    def snapshot(self) -> dict:
        elapsed = float(self.env.episode_length_buf.max().item()) * self.env.step_dt
        root_quat = self.robot.data.root_link_quat_w
        stick1_p, _ = self.stick_pose_in_palm(self.stick1)
        stick2_p, _ = self.stick_pose_in_palm(self.stick2)
        return {
            "t": elapsed,
            "phase": int(self.orientation.phase[0].item()),
            "alpha": float(self.orientation.alpha[0, 0].item()),
            "q_cmd": self.orientation.target_quat_w[0].clone(),
            "q_goal": self.orientation.goal_quat_w[0].clone(),
            "q_start": self.orientation.start_quat_w[0].clone(),
            "root_pos": self.robot.data.root_link_pos_w.clone(),
            "root_quat": root_quat.clone(),
            "err_to_cmd_deg": float(
                geodesic_deg(self.orientation.target_quat_w, root_quat)[0].item()
            ),
            "err_to_goal_deg": float(
                geodesic_deg(self.orientation.goal_quat_w, root_quat)[0].item()
            ),
            "root_ang_speed": float(
                self.robot.data.root_link_ang_vel_w.norm(dim=-1)[0].item()
            ),
            "mode_open": bool(self.open_close.command[0, 0] > 0.5),
            "segment": int(self.open_close.segment[0].item()),
            "stick1_p": stick1_p.clone(),
            "stick2_p": stick2_p.clone(),
            "stick_ang_speed": float(
                torch.maximum(
                    self.stick_ang_speed_in_palm(self.stick1),
                    self.stick_ang_speed_in_palm(self.stick2),
                ).max().item()
            ),
            "wrench": self._root_wrench_norm(),
        }

    def _root_wrench_norm(self) -> tuple[float, float]:
        """Magnitude of the wrench currently in the persistent PhysX buffer."""
        composer = getattr(self.robot, "permanent_wrench_composer", None)
        if composer is None:
            return (float("nan"), float("nan"))
        # ``composed_force`` is a warp array; the torch view is the *_as_torch
        # property.  Shape is (num_envs, num_bodies, 3) and the hand root is
        # body 0 (palm_link).
        force = composer.composed_force_as_torch[:, 0]
        torque = composer.composed_torque_as_torch[:, 0]
        return (float(force.norm(dim=-1).max().item()), float(torque.norm(dim=-1).max().item()))

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.results.append((name, PASS if ok else FAIL, detail))
        print(f"  [{PASS if ok else FAIL}] {name}: {detail}")

    def run_episode(self) -> None:
        """Step one full episode, recording a snapshot every policy step."""
        max_steps = int(round(self.schedule.episode_length_s / self.env.step_dt))
        self.samples.append(self.snapshot())
        for step in range(max_steps - 1):
            self.env.step(self.zero_action)
            sample = self.snapshot()
            self.samples.append(sample)
            if args_cli.print_interval and step % args_cli.print_interval == 0:
                print(
                    f"    t={sample['t']:5.2f}s phase={sample['phase']}"
                    f" alpha={sample['alpha']:.3f}"
                    f" err_cmd={sample['err_to_cmd_deg']:6.2f}deg"
                    f" err_goal={sample['err_to_goal_deg']:6.2f}deg"
                    f" w={sample['root_ang_speed']:5.3f}rad/s"
                    f" mode={'OPEN ' if sample['mode_open'] else 'CLOSE'}"
                    f" seg={sample['segment']}"
                    f" |F|={sample['wrench'][0]:6.3f} |T|={sample['wrench'][1]:6.3f}"
                )

    def at_time(self, target_time: float) -> dict:
        """Snapshot closest to (and not before) the requested elapsed time."""
        for sample in self.samples:
            if sample["t"] >= target_time - 1.0e-9:
                return sample
        return self.samples[-1]


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    # Both command terms hold their own (deep-copied) schedule; patch both.
    schedule = env_cfg.commands.root_orientation.schedule
    apply_scenario(schedule)
    apply_scenario(env_cfg.commands.open_close.schedule)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    orientation_term = env.command_manager.get_term("root_orientation")
    open_close_term = env.command_manager.get_term("open_close")
    probe = Probe(env, orientation_term, open_close_term, schedule)
    origin = env.scene.env_origins
    device = env.device

    print("\n=== hand_move probe ===")
    print(f"  scenario            = {args_cli.scenario} ({args_cli.angle_deg} deg)")
    print(f"  use_slerp           = {schedule.use_slerp}, use_smoothstep = {schedule.use_smoothstep}")
    print(f"  ranges              = x{schedule.range_x} y{schedule.range_y} z{schedule.range_z}")
    print(f"  step_dt / physics_dt= {env.step_dt:.5f} / {env.physics_dt:.5f}")
    print(f"  episode_length_s    = {env.cfg.episode_length_s}")

    # ------------------------------------------------------------ 1
    print("\n[1] dimensions")
    action_dim = env.action_manager.total_action_dim
    obs_dim = int(env.observation_manager.group_obs_dim["policy"][0])
    history_dim = int(env.action_manager.prev_action.shape[1])
    print(f"      action term dims = {env.action_manager.action_term_dim}")
    probe.check("total action dimension is 20", action_dim == 20, f"{action_dim}")
    probe.check("policy observation dimension is 103", obs_dim == 103, f"{obs_dim}")
    probe.check("action history dimension is 20", history_dim == 20, f"{history_dim}")

    # ------------------------------------------------------------ 2, 3
    print("\n[2/3] reset state vs hand_grasp functional grasp")
    robot = probe.robot
    joint_pos = robot.data.joint_pos[:, probe.hand_joint_ids]
    joint_target = robot.data.joint_pos_target[:, probe.hand_joint_ids]
    reference_joints = torch.tensor(PREGRASP_JOINT_POSITIONS, device=device).unsqueeze(0)
    joint_pos_error = (joint_pos - reference_joints).abs().max().item()
    joint_target_error = (joint_target - reference_joints).abs().max().item()

    stick1_p, stick1_q = probe.stick_pose_in_palm(probe.stick1)
    stick2_p, stick2_q = probe.stick_pose_in_palm(probe.stick2)
    ref1_p = torch.tensor(PREGRASP_STICK1_POSITION_P, device=device).unsqueeze(0)
    ref2_p = torch.tensor(PREGRASP_STICK2_POSITION_P, device=device).unsqueeze(0)
    ref1_q = torch.tensor(PREGRASP_STICK1_QUATERNION_P, device=device).unsqueeze(0)
    ref2_q = torch.tensor(PREGRASP_STICK2_QUATERNION_P, device=device).unsqueeze(0)
    stick1_pos_err = (stick1_p - ref1_p).norm(dim=-1).max().item()
    stick2_pos_err = (stick2_p - ref2_p).norm(dim=-1).max().item()
    stick1_ori_err = geodesic_deg(stick1_q, ref1_q.expand_as(stick1_q)).max().item()
    stick2_ori_err = geodesic_deg(stick2_q, ref2_q.expand_as(stick2_q)).max().item()

    root_pos_local = robot.data.root_link_pos_w - origin
    nominal_pos = torch.tensor(HAND_ROOT_POS, device=device).unsqueeze(0)
    nominal_quat = torch.tensor(HAND_ROOT_ROT, device=device).unsqueeze(0)
    root_lin = robot.data.root_link_lin_vel_w.norm(dim=-1).max().item()
    root_ang = robot.data.root_link_ang_vel_w.norm(dim=-1).max().item()

    probe.check(
        "hand joint positions == PREGRASP_JOINT_POSITIONS",
        joint_pos_error < 1.0e-4,
        f"max |dq| = {joint_pos_error:.2e} rad",
    )
    probe.check(
        "hand joint targets == PREGRASP_JOINT_POSITIONS",
        joint_target_error < 1.0e-4,
        f"max |dq_target| = {joint_target_error:.2e} rad",
    )
    probe.check(
        "Stick1/Stick2 palm-relative pose == pose_005 reference",
        max(stick1_pos_err, stick2_pos_err) < 1.0e-3
        and max(stick1_ori_err, stick2_ori_err) < 1.0,
        f"pos {stick1_pos_err*1000:.3f}/{stick2_pos_err*1000:.3f} mm,"
        f" ori {stick1_ori_err:.3f}/{stick2_ori_err:.3f} deg",
    )
    probe.check(
        "root starts at HAND_ROOT_POS/ROT with zero velocity",
        (root_pos_local - nominal_pos).norm(dim=-1).max().item() < 1.0e-4
        and geodesic_deg(robot.data.root_link_quat_w, nominal_quat.expand(args_cli.num_envs, 4)).max().item() < 0.05
        and max(root_lin, root_ang) < 1.0e-4,
        f"|v|={root_lin:.2e} m/s, |w|={root_ang:.2e} rad/s",
    )

    q_start = orientation_term.start_quat_w
    probe.check(
        "q_start captured after the reset events (not stale)",
        geodesic_deg(q_start, robot.data.root_link_quat_w).max().item() < 1.0e-3,
        f"|q_start - q_root| = {geodesic_deg(q_start, robot.data.root_link_quat_w).max().item():.2e} deg",
    )

    # ------------------------------------------------------------ 4
    print("\n[4] relative goal sampling")
    delta = orientation_term.delta_euler
    q_goal = orientation_term.goal_quat_w
    expected_goal = quat_mul(
        q_start, quat_from_euler_xyz(delta[:, 0], delta[:, 1], delta[:, 2])
    )
    goal_composition_err = geodesic_deg(q_goal, expected_goal).max().item()
    goal_angle_deg = geodesic_deg(q_goal, q_start).max().item()
    print(f"      sampled delta rpy = {fmt(torch.rad2deg(delta[0]), 4)} deg")
    print(f"      goal geodesic     = {goal_angle_deg:.4f} deg")
    probe.check(
        "q_goal == q_start (x) q_delta",
        goal_composition_err < 1.0e-3,
        f"composition error {goal_composition_err:.2e} deg",
    )
    if args_cli.scenario == "zero":
        probe.check(
            "zero range -> q_goal == q_start",
            goal_angle_deg < 1.0e-3,
            f"geodesic angle {goal_angle_deg:.2e} deg",
        )
    elif args_cli.scenario in ("roll", "pitch", "yaw"):
        probe.check(
            f"single-axis {args_cli.scenario} magnitude matches the command",
            abs(goal_angle_deg - args_cli.angle_deg) < 1.0e-2,
            f"{goal_angle_deg:.4f} deg vs commanded {args_cli.angle_deg:.4f} deg",
        )

    # ------------------------------------------------------------ run
    print(f"\n[run] stepping one full {schedule.episode_length_s} s episode with zero finger action")
    probe.run_episode()

    # ------------------------------------------------------------ 5
    print("\n[5] SLERP timing")
    hold_end = schedule.initial_hold_time_s
    slerp_end = schedule.slerp_end_time_s
    mid = hold_end + 0.5 * schedule.rotation_interpolation_time_s
    before = probe.at_time(hold_end - env.step_dt)
    at_mid = probe.at_time(mid)
    after = probe.at_time(slerp_end)
    at_end = probe.samples[-1]

    cmd_vs_start_before = float(geodesic_deg(before["q_cmd"].unsqueeze(0), before["q_start"].unsqueeze(0))[0])
    cmd_vs_goal_after = float(geodesic_deg(after["q_cmd"].unsqueeze(0), after["q_goal"].unsqueeze(0))[0])
    cmd_vs_goal_end = float(geodesic_deg(at_end["q_cmd"].unsqueeze(0), at_end["q_goal"].unsqueeze(0))[0])
    cmd_vs_start_mid = float(geodesic_deg(at_mid["q_cmd"].unsqueeze(0), at_mid["q_start"].unsqueeze(0))[0])

    probe.check(
        "q_cmd == q_start before the hold ends",
        cmd_vs_start_before < 1.0e-3,
        f"t={before['t']:.2f}s, |q_cmd - q_start| = {cmd_vs_start_before:.2e} deg",
    )
    probe.check(
        "q_cmd == q_goal from the SLERP end onwards",
        cmd_vs_goal_after < 1.0e-3,
        f"t={after['t']:.2f}s, |q_cmd - q_goal| = {cmd_vs_goal_after:.2e} deg",
    )
    probe.check(
        "q_cmd stays pinned at q_goal until the episode ends",
        cmd_vs_goal_end < 1.0e-3,
        f"t={at_end['t']:.2f}s, |q_cmd - q_goal| = {cmd_vs_goal_end:.2e} deg",
    )
    if args_cli.scenario != "zero" and schedule.use_slerp:
        probe.check(
            "q_cmd actually interpolates mid-window",
            cmd_vs_start_mid > 1.0e-3,
            f"t={at_mid['t']:.2f}s, moved {cmd_vs_start_mid:.4f} deg from q_start",
        )
    if args_cli.no_slerp and args_cli.scenario != "zero":
        just_after_hold = probe.at_time(hold_end)
        step_jump = float(
            geodesic_deg(
                just_after_hold["q_cmd"].unsqueeze(0), just_after_hold["q_goal"].unsqueeze(0)
            )[0]
        )
        probe.check(
            "use_slerp=False steps q_cmd straight to q_goal at the hold end",
            step_jump < 1.0e-3,
            f"t={just_after_hold['t']:.2f}s, |q_cmd - q_goal| = {step_jump:.2e} deg",
        )

    # ------------------------------------------------------------ 6
    print("\n[6] controller lifecycle after the SLERP ends")
    post_slerp = [s for s in probe.samples if s["t"] >= slerp_end]
    wrench_written = all(
        not math.isnan(s["wrench"][0]) and not math.isnan(s["wrench"][1]) for s in post_slerp
    )
    settled = probe.at_time(schedule.open_close_start_time_s)
    final_error = at_end["err_to_goal_deg"]
    max_post_error = max(s["err_to_goal_deg"] for s in post_slerp)
    probe.check(
        "wrench buffer is still being written after the SLERP ends",
        wrench_written,
        f"{len(post_slerp)} post-SLERP samples, all readable",
    )
    probe.check(
        "root converges to q_goal and stays there",
        final_error < 5.0 and max_post_error < 45.0,
        f"error at open/close start {settled['err_to_goal_deg']:.3f} deg,"
        f" at episode end {final_error:.3f} deg, worst after SLERP {max_post_error:.3f} deg",
    )
    print(
        f"      angular speed at slerp end / open-close start ="
        f" {after['root_ang_speed']:.4f} / {settled['root_ang_speed']:.4f} rad/s"
    )

    # ------------------------------------------------------------ 7
    print("\n[7] OPEN/CLOSE schedule")
    start_time = schedule.open_close_start_time_s
    segment_time = schedule.open_close_segment_time_s
    close_before = all(not s["mode_open"] for s in probe.samples if s["t"] < start_time - 1e-9)
    probe.check(
        "CLOSE for the whole pre-open/close window",
        close_before,
        f"t < {start_time}s",
    )
    expected_first_open = bool(open_close_term.first_mode_open[0].item())
    segment_modes = []
    for index in range(schedule.num_open_close_segments):
        probe_time = start_time + (index + 0.5) * segment_time
        sample = probe.at_time(probe_time)
        segment_modes.append((index, sample["segment"], sample["mode_open"]))
    alternating = all(
        observed_segment == index
        and observed_open == (expected_first_open if index % 2 == 0 else not expected_first_open)
        for index, observed_segment, observed_open in segment_modes
    )
    print(
        "      segments = "
        + ", ".join(
            f"{i}:{'OPEN' if o else 'CLOSE'}(seg={s})" for i, s, o in segment_modes
        )
    )
    probe.check(
        "segments alternate from the sampled first mode",
        alternating,
        f"first_mode_open={expected_first_open}",
    )

    # ------------------------------------------------------------ 8, 9
    print("\n[8/9] root translation and stick stability")
    translation = max(
        float((s["root_pos"] - probe.samples[0]["root_pos"]).norm(dim=-1).max().item())
        for s in probe.samples
    )
    stick1_shift = max(
        float((s["stick1_p"] - probe.samples[0]["stick1_p"]).norm(dim=-1).max().item())
        for s in probe.samples
    )
    stick2_shift = max(
        float((s["stick2_p"] - probe.samples[0]["stick2_p"]).norm(dim=-1).max().item())
        for s in probe.samples
    )
    peak_stick_ang = max(s["stick_ang_speed"] for s in probe.samples)
    probe.check(
        "root does not translate while it rotates",
        translation < 0.01,
        f"peak translation {translation*1000:.3f} mm",
    )
    probe.check(
        "palm-relative stick pose stays bounded",
        max(stick1_shift, stick2_shift) < 0.02,
        f"stick1 {stick1_shift*1000:.2f} mm, stick2 {stick2_shift*1000:.2f} mm"
        " (zero finger action, so some drift is expected)",
    )
    probe.check(
        "palm-relative stick angular speed stays under the penalty threshold",
        peak_stick_ang < 3.0,
        f"peak |w_rel| {peak_stick_ang:.4f} rad/s (reward threshold 3.0)",
    )

    # ------------------------------------------------------------ summary
    print("\n=== summary ===")
    failures = [name for name, status, _ in probe.results if status == FAIL]
    for name, status, detail in probe.results:
        print(f"  {status:4s}  {name}  |  {detail}")
    print(f"\n  {len(probe.results) - len(failures)}/{len(probe.results)} checks passed")
    if failures:
        print("  failed: " + ", ".join(failures))

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
