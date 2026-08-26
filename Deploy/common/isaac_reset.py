# [common] Isaac 리셋 상수 — pregrasp 20관절 각도와 palm 기준 젓가락 두 개의 시작 포즈. 실제 값.
"""Isaac's own reset numbers.  Real values, shared by every backend.

Where the twenty finger joints start (the pregrasp pose) and where the two
chopsticks sit relative to the palm at that moment.  These come out of the
Isaac environment the policy was trained in, so MuJoCo and the physical hand
both have to start from them or they are not reproducing the same episode.

Nothing here is a camera, a marker, or a guess.  The simulated ArUco fixture
that used to share this file lives in ``vision/sim_aruco.py``, and the physical
camera rig lives in ``vision/deploy_rig.py``.
"""

from __future__ import annotations

import numpy as np

#: Stick geometry, from the Isaac scene.
STICK_SIZE_M = np.asarray([0.007, 0.180, 0.007], dtype=np.float64)
STICK_MASS_KG = 0.010

# The pose every episode RESETS to, and therefore the pose the real hand GLIDES
# to before the policy takes over.  Isaac applies it through the
# ``reset_pregrasp`` event; ``run_hand_policy_real`` walks to it during GLIDE
# and ``mujoco_wuji.reset()`` teleports to it.  One pose, three consumers.
#
# It belongs to a CHECKPOINT, not to the hand: each training run records its own
# in ``params/env.yaml``.  Deploying a policy against the wrong one starts it
# from somewhere it never trained, which does not raise -- it just grasps badly.
# ``read_pregrasp_from_env_yaml`` below reads a run's own value, and
# ``tools/verify_policy_contract`` reports the difference.
#
# 2026-08-23: switched to hand_real2/2026-08-23_16-47-00 (model_750), whose
# reset differs from the previous pose by up to 78.0 mrad (finger2_joint2).
# The previous value is kept below rather than deleted: the hand_real lineage
# (2026-08-18_23-57-25, 2026-08-19_02-20-20) still trains from it, so switching
# back is a matter of moving the comment, not of re-deriving numbers.
#
# CAVEAT on index 18 (finger5_joint3), unchanged from the old pose: 1.6272 rad
# == 93.2317 deg was a placeholder upper limit, i.e. that joint was SATURATED
# when the pose was recorded.  It also sits 35.8 mrad above the deployed
# COMMAND_LIMIT_RATIO bound, so the real runner clamps it and says so.
# Index 10 is no longer at the placeholder -- the new pose has 1.5926 there.

# -- hand_real lineage (pose_005), 2026-08-18 .. 2026-08-19 runs.  ACTIVE.
ISAAC_PREGRASP_JOINT_POSITIONS_RAD = np.asarray(
    [0.5345742259, 0.8214717428, 0.0257641812, 0.0236253070,
     0.7266738102, 0.1332869837, 1.1353251203, 1.3972301575,
     0.4016543424, 0.0103005540, 1.5925557413, 1.1031022734,
     0.8597054933, 0.0217672060, 1.3284198815, 0.3220070672,
     0.7154092789, 0.0788998753, 1.6272000000, 0.2546040118,
], dtype=np.float32,
)
# -- hand_real , 2026-08-24_09-29-55
#ISAAC_PREGRASP_JOINT_POSITIONS_RAD = np.asarray(
#    [0.5377866626, 0.8436813951, 0.0377136655, -0.0000001810,
#     0.7017297745, 0.0553143807, 1.1822255850, 1.4215219021,
#     0.4649881423, -0.0292181600, 1.6272000000, 1.1032750607,
#     0.9151425958, -0.0129909236, 1.3248542547, 0.3182539344,
#     0.7154092789, 0.0788998753, 1.6272000000, 0.2546040118], dtype=np.float32,
#)
# -- hand_real2/2026-08-23_16-47-00 (model_750).  Kept for switching back;
#    do not delete.
#ISAAC_PREGRASP_JOINT_POSITIONS_RAD = np.asarray(
#    [0.5345742106, 0.8214717507, 0.0257641803, 0.0236253068,
#     0.7266737819, 0.1332869828, 1.1353250742, 1.3972301483,
#     0.4016543329, 0.0103005543, 1.5925557613, 1.1031023264,
#     0.8597055078, 0.0217672065, 1.3284199238, 0.3220070601,
#     0.7154092789, 0.0788998753, 1.6272000074, 0.2546040118], dtype=np.float32,
#)
ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ = np.asarray(
    [[0.0250743479, 0.0242451150, 0.0969612077,
      0.4618085623, -0.0092124203, -0.1713383496, -0.8702247143],
     [0.0355986878, 0.0160842165, 0.0733669698,
      0.2051235586, -0.6018196344, -0.4935579300, -0.5934122205]], dtype=np.float64,
)
STICK_REFERENCE_QUATERNIONS_PALM_WXYZ = ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ[:, 3:].copy()


def _primary_marker_up_symmetric_reset_poses() -> np.ndarray:
    """Choose the square-symmetric reset roll with ID0/ID2 on top.

    Primary marker centers remain at stick-local Y=-90 mm, Z=+3.5 mm.  The
    square section admits four policy-equivalent shaft rolls; select the one
    whose local +Z marker normal points most toward Palm/Base +X (up).
    """

    from .stick_pose import (
        quaternion_multiply_wxyz,
        quaternion_to_rotation_matrix_wxyz,
    )

    poses = ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ.copy()
    up_p = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    for stick_index, source_pose in enumerate(ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ):
        best_score = -np.inf
        best_quaternion = source_pose[3:].copy()
        for quarter_turn in range(4):
            half_angle = quarter_turn * np.pi / 4.0
            roll_y = np.asarray(
                [np.cos(half_angle), 0.0, np.sin(half_angle), 0.0],
                dtype=np.float64,
            )
            candidate = quaternion_multiply_wxyz(source_pose[3:], roll_y)
            rotation_p_s = quaternion_to_rotation_matrix_wxyz(candidate)
            marker_normal_p = rotation_p_s[:, 2]
            score = float(marker_normal_p @ up_p)
            if score > best_score:
                best_score = score
                best_quaternion = candidate
        poses[stick_index, 3:] = best_quaternion
    return poses


MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ = (
    _primary_marker_up_symmetric_reset_poses()
)


def _primary_marker_up_symmetric_reset_poses() -> np.ndarray:
    """Choose the square-symmetric reset roll with ID0/ID2 on top.

    Primary marker centers remain at stick-local Y=-90 mm, Z=+3.5 mm.  The
    square section admits four policy-equivalent shaft rolls; select the one
    whose local +Z marker normal points most toward Palm/Base +X (up).
    """

    from .stick_pose import (
        quaternion_multiply_wxyz,
        quaternion_to_rotation_matrix_wxyz,
    )

    poses = ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ.copy()
    up_p = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    for stick_index, source_pose in enumerate(ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ):
        best_score = -np.inf
        best_quaternion = source_pose[3:].copy()
        for quarter_turn in range(4):
            half_angle = quarter_turn * np.pi / 4.0
            roll_y = np.asarray(
                [np.cos(half_angle), 0.0, np.sin(half_angle), 0.0],
                dtype=np.float64,
            )
            candidate = quaternion_multiply_wxyz(source_pose[3:], roll_y)
            rotation_p_s = quaternion_to_rotation_matrix_wxyz(candidate)
            marker_normal_p = rotation_p_s[:, 2]
            score = float(marker_normal_p @ up_p)
            if score > best_score:
                best_score = score
                best_quaternion = candidate
        poses[stick_index, 3:] = best_quaternion
    return poses


MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ = (
    _primary_marker_up_symmetric_reset_poses()
)



def read_pregrasp_from_env_yaml(target) -> np.ndarray:
    """Read the pregrasp a training run actually reset to, from its env.yaml.

    ``ISAAC_PREGRASP_JOINT_POSITIONS_RAD`` above is ONE run's pose, baked in.
    It is also the anchor for ``ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ`` -- those
    stick poses were recorded with the hand in that exact pose -- so editing the
    constant to chase a new checkpoint silently invalidates MuJoCo's initial
    grasp geometry.  On hardware the sticks come from the cameras and no such
    coupling exists, so the real runner can take the pose per run instead.
    That is what this function is for; the constant stays put.

    Parsed by regex rather than a YAML loader: env.yaml carries
    ``!!python/tuple`` and ``!!python/object`` tags, so a safe loader refuses
    the file and an unsafe one would run constructors out of it.

    Accepts a run directory, its ``params`` directory, ``env.yaml`` itself, or a
    path inside the run (e.g. ``exported/policy.onnx``).
    """

    import re
    from pathlib import Path

    target = Path(target).expanduser().resolve()
    if target.is_file() and target.suffix in (".yaml", ".yml"):
        env_yaml = target
    else:
        for candidate in (target / "params" / "env.yaml", target / "env.yaml",
                          target.parent / "params" / "env.yaml",
                          target.parent.parent / "params" / "env.yaml"):
            if candidate.is_file():
                env_yaml = candidate
                break
        else:
            raise FileNotFoundError(f"No params/env.yaml found for {target}.")

    text = env_yaml.read_text()
    start = re.search(r"^(\s*)reset_pregrasp:\s*$", text, re.M)
    if start is None:
        raise ValueError(f"{env_yaml} has no reset_pregrasp event.")
    key = re.search(r"^\s*joint_positions:.*$", text[start.end():], re.M)
    if key is None:
        raise ValueError(f"{env_yaml} reset_pregrasp has no joint_positions.")
    tail = text[start.end() + key.end():]
    values = []
    for line in tail.splitlines():
        # `$` in MULTILINE stops before the newline, so `tail` opens with one --
        # splitlines() then yields '' first.  Breaking on it reads zero joints
        # and reports the file as malformed when it is fine.
        if not line.strip():
            continue
        item = re.match(r"^\s*-\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s*$", line)
        if item is None:
            break  # end of the sequence -- do not wander into the next key
        values.append(float(item.group(1)))
    expected = len(ISAAC_PREGRASP_JOINT_POSITIONS_RAD)
    if len(values) != expected:
        raise ValueError(
            f"{env_yaml}: read {len(values)} pregrasp joints, expected {expected}.")
    return np.asarray(values, dtype=np.float32)


for _array in (STICK_SIZE_M, ISAAC_PREGRASP_JOINT_POSITIONS_RAD,
               ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ,
               STICK_REFERENCE_QUATERNIONS_PALM_WXYZ,
               MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ):
    _array.setflags(write=False)
