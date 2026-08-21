from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from Deploy.common.isaac_reset import (ISAAC_PREGRASP_JOINT_POSITIONS_RAD, ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ, MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ, STICK_REFERENCE_QUATERNIONS_PALM_WXYZ, STICK_MASS_KG)
from Deploy.vision.sim_aruco import (D435_NOMINAL_BODY_SIZE_M, BASE_PLATE_DEPTH_Z_M, BASE_PLATE_NEGATIVE_Z_EDGE_M, BASE_PLATE_THICKNESS_M, BASE_PLATE_WIDTH_Y_M, CAMERA_SUPPORT_CENTER_Y_M, CAMERA_SUPPORT_CENTER_Z_M, CAMERA_SUPPORT_FLOOR_X_M, CAMERA_SUPPORT_HEIGHT_APPROX_M, HAND_PROFILE_CENTER_Z_M, R_WORLD_BASE, T_BASE_CAMERA, T_MARKER_STICK_BY_ID)


class VisionSceneTests(unittest.TestCase):
    def test_scene_stick_mass_camera_and_reset_pose(self) -> None:
        import mujoco

        from Deploy.backends.mujoco_wuji import MujocoWujiHand

        hand = MujocoWujiHand()
        for name in ("stick1", "stick2"):
            body_id = mujoco.mj_name2id(hand.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.assertAlmostEqual(float(hand.model.body_mass[body_id]), STICK_MASS_KG, places=9)
        for marker_number in range(4):
            marker_body_id = mujoco.mj_name2id(
                hand.model, mujoco.mjtObj.mjOBJ_BODY,
                f"aruco{marker_number}_candidate_mount",
            )
            expected_stick_marker = np.linalg.inv(
                T_MARKER_STICK_BY_ID[marker_number]
            )
            np.testing.assert_allclose(
                hand.model.body_pos[marker_body_id],
                expected_stick_marker[:3, 3],
                atol=1e-8,
            )
        np.testing.assert_allclose(
            hand.get_stick_poses_in_palm(),
            MUJOCO_VISIBLE_STICK_RESET_POSES_PALM_XYZ_WXYZ,
            atol=8e-7,
            rtol=0.0,
        )
        from Deploy.common.stick_pose import (
            canonicalize_square_stick_quaternion,
            quaternion_geodesic_error_deg,
        )
        for index, pose in enumerate(hand.get_stick_poses_in_palm()):
            from Deploy.common.stick_pose import quaternion_to_rotation_matrix_wxyz
            marker_normal_p = quaternion_to_rotation_matrix_wxyz(pose[3:])[:, 2]
            self.assertGreater(float(marker_normal_p[0]), 0.0)
            canonical = canonicalize_square_stick_quaternion(
                pose[3:], STICK_REFERENCE_QUATERNIONS_PALM_WXYZ[index]
            )
            self.assertLess(
                quaternion_geodesic_error_deg(
                    canonical, ISAAC_STICK_RESET_POSES_PALM_XYZ_WXYZ[index, 3:]
                ),
                1.0e-4,
            )
        expected_q = np.clip(
            ISAAC_PREGRASP_JOINT_POSITIONS_RAD,
            hand.model.jnt_range[:20, 0],
            hand.model.jnt_range[:20, 1],
        )
        np.testing.assert_allclose(hand.read_joint_positions(), expected_q, atol=1e-7)
        # Until 2026-08-17 indices 10 and 18 clamped here, because Isaac's reset
        # pose was measured against an older URDF whose joint3 upper was a
        # uniform 93.2317 deg placeholder.  Isaac's limits and that reset pose
        # are now both aligned to the pinned official description, so a
        # clamp on any joint means the two sides have drifted apart again.
        np.testing.assert_array_equal(np.flatnonzero(hand.last_reset_clamped), [])
        camera_id = mujoco.mj_name2id(
            hand.model, mujoco.mjtObj.mjOBJ_CAMERA, "d435_rgb"
        )
        np.testing.assert_array_equal(hand.model.cam_resolution[camera_id], [1280, 720])
        np.testing.assert_allclose(
            hand.model.cam_intrinsic[camera_id, :2],
            [926.550964355469, 926.377807617188],
            atol=1e-4,
        )
        np.testing.assert_allclose(
            hand.data.cam_xpos[camera_id],
            R_WORLD_BASE @ T_BASE_CAMERA[:3, 3],
            atol=1e-7,
        )
        cv_to_mujoco_camera = np.diag([1.0, -1.0, -1.0])
        np.testing.assert_allclose(
            hand.data.cam_xmat[camera_id].reshape(3, 3),
            R_WORLD_BASE @ T_BASE_CAMERA[:3, :3] @ cv_to_mujoco_camera,
            atol=1e-6,
        )
        camera_body_id = mujoco.mj_name2id(
            hand.model, mujoco.mjtObj.mjOBJ_BODY, "d435_rgb_visual"
        )
        camera_geom_id = mujoco.mj_name2id(
            hand.model, mujoco.mjtObj.mjOBJ_GEOM, "d435_body_visual"
        )
        self.assertGreaterEqual(camera_body_id, 0)
        np.testing.assert_allclose(
            2.0 * hand.model.geom_size[camera_geom_id],
            D435_NOMINAL_BODY_SIZE_M,
            atol=1e-9,
        )
        floor_id = mujoco.mj_name2id(
            hand.model, mujoco.mjtObj.mjOBJ_GEOM, "base_plate_collision"
        )
        np.testing.assert_allclose(
            2.0 * hand.model.geom_size[floor_id],
            [BASE_PLATE_DEPTH_Z_M, BASE_PLATE_WIDTH_Y_M,
             BASE_PLATE_THICKNESS_M],
            atol=1e-9,
        )
        self.assertNotEqual(int(hand.model.geom_contype[floor_id]), 0)
        floor_center_base = R_WORLD_BASE.T @ hand.model.geom_pos[floor_id]
        self.assertAlmostEqual(
            float(floor_center_base[2] - BASE_PLATE_DEPTH_Z_M / 2.0),
            BASE_PLATE_NEGATIVE_Z_EDGE_M,
        )
        hand_post_id = mujoco.mj_name2id(
            hand.model, mujoco.mjtObj.mjOBJ_GEOM, "hand_2020_profile_visual"
        )
        camera_post_id = mujoco.mj_name2id(
            hand.model, mujoco.mjtObj.mjOBJ_GEOM, "camera_4040_profile_visual"
        )
        hand_post_base = R_WORLD_BASE.T @ hand.model.geom_pos[hand_post_id]
        camera_post_base = R_WORLD_BASE.T @ hand.model.geom_pos[camera_post_id]
        self.assertAlmostEqual(float(hand_post_base[2]), HAND_PROFILE_CENTER_Z_M)
        self.assertAlmostEqual(float(camera_post_base[1]), CAMERA_SUPPORT_CENTER_Y_M)
        self.assertAlmostEqual(float(camera_post_base[2]), CAMERA_SUPPORT_CENTER_Z_M)
        self.assertAlmostEqual(
            float(camera_post_base[0] - CAMERA_SUPPORT_HEIGHT_APPROX_M / 2.0),
            CAMERA_SUPPORT_FLOOR_X_M,
        )
        for name in (
            "hand_2020_profile_visual",
            "camera_4040_profile_visual",
            "camera_bracket_visual",
            "base_axis_x",
            "palm_axis_x",
            "camera_optical_axis_x",
        ):
            self.assertGreaterEqual(
                mujoco.mj_name2id(hand.model, mujoco.mjtObj.mjOBJ_GEOM, name), 0
            )

    def test_offscreen_aruco_ground_truth_accuracy_and_15hz_supply(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        script = r'''
import numpy as np
from Deploy.vision.aruco_perception import ArucoStickPoseProvider
from Deploy.backends.mujoco_perception import MujocoCameraSource, MujocoGroundTruthStickPoseProvider
from Deploy.backends.mujoco_wuji import MujocoWujiHand
from Deploy.vision.sim_aruco import frontal_camera_validation_stick_poses
from Deploy.common.stick_pose import quaternion_geodesic_error_deg, quaternion_to_rotation_matrix_wxyz

hand = MujocoWujiHand()
hand.set_stick_poses_in_palm(frontal_camera_validation_stick_poses())
ground_truth = MujocoGroundTruthStickPoseProvider(hand).sample()
camera = MujocoCameraSource(hand)
provider = ArucoStickPoseProvider(camera)
estimate = provider.sample()
assert estimate.fresh
for expected, actual in zip(
    (ground_truth.stick1, ground_truth.stick2),
    (estimate.stick1, estimate.stick2),
):
    position_error_mm = np.linalg.norm(expected[:3] - actual[:3]) * 1000.0
    rotation_error_deg = quaternion_geodesic_error_deg(expected[3:], actual[3:])
    expected_axis = quaternion_to_rotation_matrix_wxyz(expected[3:])[:, 1]
    actual_axis = quaternion_to_rotation_matrix_wxyz(actual[3:])[:, 1]
    axis_error_deg = np.degrees(np.arccos(np.clip(expected_axis @ actual_axis, -1.0, 1.0)))
    assert position_error_mm < 5.0, position_error_mm
    assert rotation_error_deg < 5.0, rotation_error_deg
    assert axis_error_deg < 5.0, axis_error_deg

# 30 Hz policy vs 15 Hz fresh RGB: one fresh frame every two policy holds,
# while the intermediate call reuses the latest valid pose.  Advance by
# ``physics_substeps`` rather than a hard-coded 4 so this stays a statement
# about policy steps, not about whichever integration timestep is configured.
policy_step = hand.physics_substeps
hand.step(policy_step)
held = provider.sample()
assert not held.fresh, (hand.model.opt.timestep, policy_step)
hand.step(policy_step)
fresh = provider.sample()
assert fresh.fresh, (hand.model.opt.timestep, policy_step)
camera.close()
'''
        environment = os.environ.copy()
        environment["MUJOCO_GL"] = "egl"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
