"""Report the actual PhysX contact pairs in a saved ``hand_grasp`` pose."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_POSE = (
    PROJECT_ROOT
    / "logs/debug/hand_grasp_keyboard/2026-07-28_12-39-52/pose_005.json"
)

parser = argparse.ArgumentParser(
    description="Replay one saved hand_grasp pose and enumerate its contact pairs."
)
parser.add_argument("--task", type=str, default="hand_grasp")
parser.add_argument("--pose-file", type=Path, default=DEFAULT_POSE)
parser.add_argument("--settle-steps", type=int, default=60)
parser.add_argument("--sample-steps", type=int, default=240)
parser.add_argument("--force-threshold", type=float, default=1.0e-4)
parser.add_argument(
    "--output-root",
    type=Path,
    default=PROJECT_ROOT / "logs/debug/hand_grasp_contact_pairs",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.pose_file = args_cli.pose_file.expanduser().resolve()
if not args_cli.pose_file.is_file():
    parser.error(f"pose file not found: {args_cli.pose_file}")
if args_cli.settle_steps < 0 or args_cli.sample_steps <= 0:
    parser.error("settle-steps must be non-negative and sample-steps must be positive")
if args_cli.force_threshold < 0.0:
    parser.error("force-threshold must be non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

import isaac_neuromeka.tasks  # noqa: F401, E402
from isaac_neuromeka.tasks.manipulation.hand_grasp.hand_grasp_env_cfg import (  # noqa: E402
    HAND_JOINT_NAMES,
)


HAND_BODY_NAMES = ["palm_link"] + [
    body_name
    for finger in range(1, 6)
    for body_name in (
        f"finger{finger}_link1",
        f"finger{finger}_link2",
        f"finger{finger}_link3",
        f"finger{finger}_link4",
        f"finger{finger}_tip_link",
    )
]
OBJECT_NAMES = ("Stick1", "Stick2")
SEMANTIC_CONTACT_PAIRS = {
    "thumb_distal_stick1": ("finger1_link3", "stick1"),
    "index_tip_stick1": ("finger2_tip_link", "stick1"),
    "middle_tip_stick1": ("finger3_tip_link", "stick1"),
    "ring_tip_stick2": ("finger4_tip_link", "stick2"),
    "palm_stick2": ("palm_link", "stick2"),
    "thumb_mid_stick2": ("finger1_link2", "stick2"),
}


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _step(scene, sim, steps: int, render: bool) -> None:
    dt = sim.get_physics_dt()
    for _ in range(steps):
        scene.write_data_to_sim()
        sim.step(render=render)
        scene.update(dt)


def _update_record(records: dict, pair: tuple[str, str], force: float) -> None:
    record = records.setdefault(
        pair,
        {
            "body_pair": list(pair),
            "active_steps": 0,
            "force_sum_N": 0.0,
            "max_force_N": 0.0,
        },
    )
    record["active_steps"] += 1
    record["force_sum_N"] += force
    record["max_force_N"] = max(record["max_force_N"], force)


def main() -> None:
    pose = json.loads(args_cli.pose_file.read_text(encoding="utf-8"))
    if pose.get("joint_names") != HAND_JOINT_NAMES:
        raise ValueError("Saved pose joint order does not match HAND_JOINT_NAMES.")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.episode_length_s = 1.0e9
    reset_params = env_cfg.events.reset_pregrasp.params
    reset_params["joint_positions"] = tuple(pose["joint_actual_positions_rad"])
    reset_params["joint_position_targets"] = tuple(pose["joint_target_positions_rad"])
    reset_params["stick1_position_p"] = tuple(pose["stick1_pose_palm"]["position"])
    reset_params["stick1_quaternion_p"] = tuple(
        pose["stick1_pose_palm"]["quaternion_wxyz"]
    )
    reset_params["stick2_position_p"] = tuple(pose["stick2_pose_palm"]["position"])
    reset_params["stick2_quaternion_p"] = tuple(
        pose["stick2_pose_palm"]["quaternion_wxyz"]
    )

    filter_paths = [
        f"{{ENV_REGEX_NS}}/Robot/{body_name}" for body_name in HAND_BODY_NAMES
    ] + [f"{{ENV_REGEX_NS}}/{name}" for name in OBJECT_NAMES]
    filter_labels = HAND_BODY_NAMES + list(OBJECT_NAMES)
    sensor_names = {}
    for body_name in HAND_BODY_NAMES:
        sensor_name = f"pair_probe_{body_name}"
        sensor_names[body_name] = sensor_name
        setattr(
            env_cfg.scene,
            sensor_name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
                filter_prim_paths_expr=filter_paths,
                update_period=0.0,
                history_length=1,
                track_pose=False,
                debug_vis=False,
            ),
        )
    if hasattr(env_cfg.scene, "lazy_sensor_update"):
        env_cfg.scene.lazy_sensor_update = False

    output_dir = (
        args_cli.output_root.expanduser().resolve()
        / datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()
        scene = env.scene
        sim = env.sim
        robot = scene["robot"]
        joint_ids, resolved_names = robot.find_joints(
            HAND_JOINT_NAMES,
            preserve_order=True,
        )
        if resolved_names != HAND_JOINT_NAMES:
            raise RuntimeError(f"Unexpected hand joint order: {resolved_names}")
        target = torch.tensor(
            pose["joint_target_positions_rad"],
            device=env.device,
            dtype=robot.data.joint_pos.dtype,
        ).unsqueeze(0)
        robot.set_joint_position_target(target, joint_ids=joint_ids)

        render = not args_cli.headless
        _step(scene, sim, args_cli.settle_steps, render)

        semantic_link_positions = {}
        for label, (body_name, object_name) in SEMANTIC_CONTACT_PAIRS.items():
            body_ids, resolved_body_names = robot.find_bodies(
                [body_name],
                preserve_order=True,
            )
            if resolved_body_names != [body_name]:
                raise RuntimeError(
                    f"Unexpected body resolution for {body_name}: "
                    f"{resolved_body_names}"
                )
            obj = scene[object_name]
            body_pos_w = robot.data.body_pos_w[:, body_ids[0]]
            body_pos_o = quat_apply_inverse(
                obj.data.root_quat_w,
                body_pos_w - obj.data.root_pos_w,
            )
            semantic_link_positions[label] = {
                "body_name": body_name,
                "object_name": object_name,
                "body_origin_position_object_frame_m": body_pos_o[0],
            }

        object_records: dict[tuple[str, str], dict] = {}
        hand_records: dict[tuple[str, str], dict] = {}
        for _ in range(args_cli.sample_steps):
            robot.set_joint_position_target(target, joint_ids=joint_ids)
            _step(scene, sim, 1, render)
            seen_hand_pairs: set[tuple[str, str]] = set()
            for first, sensor_name in sensor_names.items():
                matrix = scene.sensors[sensor_name].data.force_matrix_w
                if matrix is None:
                    continue
                force_by_filter = torch.linalg.vector_norm(
                    matrix[0].reshape(-1, len(filter_labels), 3),
                    dim=-1,
                ).sum(dim=0)
                active_indices = torch.nonzero(
                    force_by_filter > args_cli.force_threshold,
                    as_tuple=False,
                ).flatten()
                for filter_index in active_indices.tolist():
                    second = filter_labels[filter_index]
                    if first == second:
                        continue
                    force = float(force_by_filter[filter_index].item())
                    if second in OBJECT_NAMES:
                        _update_record(object_records, (first, second), force)
                    else:
                        pair = tuple(sorted((first, second)))
                        if pair in seen_hand_pairs:
                            continue
                        seen_hand_pairs.add(pair)
                        _update_record(hand_records, pair, force)

        def finish(records: dict[tuple[str, str], dict]) -> list[dict]:
            rows = []
            for record in records.values():
                active_steps = record["active_steps"]
                record["active_fraction"] = active_steps / args_cli.sample_steps
                record["mean_active_force_N"] = (
                    record.pop("force_sum_N") / active_steps
                )
                rows.append(record)
            return sorted(
                rows,
                key=lambda row: (-row["active_fraction"], -row["max_force_N"]),
            )

        object_rows = finish(object_records)
        hand_rows = finish(hand_records)
        result = {
            "pose_file": args_cli.pose_file,
            "settle_steps": args_cli.settle_steps,
            "sample_steps": args_cli.sample_steps,
            "force_threshold_N": args_cli.force_threshold,
            "semantic_link_reference_positions": semantic_link_positions,
            "hand_object_pairs": object_rows,
            "hand_hand_pairs": hand_rows,
        }
        output_path = output_dir / "contact_pairs.json"
        _write_json(output_path, result)

        print(
            f"[hand_grasp contact-pair probe] pose={args_cli.pose_file.name} "
            f"threshold={args_cli.force_threshold:g}N",
            flush=True,
        )
        print("[hand <-> stick]", flush=True)
        for row in object_rows:
            print(
                f"  {row['body_pair'][0]} <-> {row['body_pair'][1]} "
                f"active={row['active_fraction']:.3f} "
                f"mean={row['mean_active_force_N']:.4f}N "
                f"max={row['max_force_N']:.4f}N",
                flush=True,
            )
        print("[hand <-> hand]", flush=True)
        for row in hand_rows:
            print(
                f"  {row['body_pair'][0]} <-> {row['body_pair'][1]} "
                f"active={row['active_fraction']:.3f} "
                f"mean={row['mean_active_force_N']:.4f}N "
                f"max={row['max_force_N']:.4f}N",
                flush=True,
            )
        print("[semantic link origins in stick frame]", flush=True)
        for label, record in semantic_link_positions.items():
            position_mm = [
                1000.0 * value
                for value in record["body_origin_position_object_frame_m"]
            ]
            print(
                f"  {label}: "
                f"[{position_mm[0]:+.2f}, {position_mm[1]:+.2f}, "
                f"{position_mm[2]:+.2f}] mm",
                flush=True,
            )
        print(f"Saved {output_path}", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
