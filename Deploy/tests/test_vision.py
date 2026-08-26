# [test] 렌더링 ArUco 픽스처로 마커 검출·포즈 복원 검증.
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import types
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


class StaleLadderTests(unittest.TestCase):
    """When both cameras lose a stick, how long before the hand is frozen.

    The ladder is the only thing standing between "one dropped frame" and
    "freeze the grasp mid-hold", so it is exercised directly rather than only
    through a live camera.  ``_advance`` is driven with synthetic selections and
    a synthetic clock; no RealSense, no ArUco, no tracker frames.
    """

    def setUp(self) -> None:
        from Deploy.vision.provider import DualCameraStickPoseProvider

        self.provider = DualCameraStickPoseProvider(
            25.0, acknowledge_candidate_geometry=True
        )

    @staticmethod
    def _seen(position=(0.02, 0.02, 0.09)):
        return {"pose": {"position": np.asarray(position, dtype=np.float64),
                         "quaternion": np.asarray([1.0, 0.0, 0.0, 0.0])},
                "source": "MAIN", "reason": "MAIN_SEES_AT_LEAST_ONE_MARKER"}

    @staticmethod
    def _blind():
        return {"pose": None, "source": "NONE",
                "reason": "MAIN_MISSED_BOTH_SIDE_INVALID"}

    def test_first_sighting_is_reinit_then_valid(self) -> None:
        from Deploy.common.perception import PoseState

        _, first = self.provider._advance(0, self._seen(), 1000.0)
        _, second = self.provider._advance(0, self._seen(), 1033.0)
        self.assertIs(first, PoseState.REINIT)
        self.assertIs(second, PoseState.VALID)

    def test_ladder_is_time_based_not_frame_based(self) -> None:
        # The camera went 15 -> 30 Hz mid-project.  A frame-count ladder would
        # have silently halved every threshold; these must not move.
        from Deploy.common.perception import PoseState

        self.provider._advance(0, self._seen(), 0.0)
        for elapsed, expected in ((33.0, PoseState.HOLD),     # 1 frame @30Hz
                                  (99.0, PoseState.HOLD),
                                  (101.0, PoseState.STALE),
                                  (249.0, PoseState.STALE),
                                  (251.0, PoseState.LOST)):
            pose, state = self.provider._advance(0, self._blind(), elapsed)
            self.assertIs(state, expected, f"at {elapsed} ms blind")
            # The last good pose is held throughout -- the policy keeps a value
            # to act on; freezing the COMMAND is safe_stop's job, not this one.
            self.assertIsNotNone(pose)

    def test_recovery_clears_the_blind_clock(self) -> None:
        from Deploy.common.perception import PoseState

        self.provider._advance(0, self._seen(), 0.0)
        self.provider._advance(0, self._blind(), 200.0)          # STALE
        _, state = self.provider._advance(0, self._seen(), 210.0)
        self.assertIs(state, PoseState.VALID)
        _, state = self.provider._advance(0, self._blind(), 260.0)
        self.assertIs(state, PoseState.HOLD)                     # 50 ms, not 260

    def test_never_seen_is_lost_not_hold(self) -> None:
        from Deploy.common.perception import PoseState

        pose, state = self.provider._advance(1, self._blind(), 500.0)
        self.assertIsNone(pose)
        self.assertIs(state, PoseState.LOST)

    def test_hold_threshold_must_sit_below_stale(self) -> None:
        from Deploy.vision.provider import DualCameraStickPoseProvider

        with self.assertRaises(ValueError):
            DualCameraStickPoseProvider(25.0, hold_after_ms=300.0,
                                        stale_after_ms=250.0,
                                        acknowledge_candidate_geometry=True)

    def test_policy_runner_stops_on_stale_but_rides_out_hold(self) -> None:
        # The contract this ladder exists to feed: HOLD keeps running, STALE
        # freezes.  Verified against PolicyRunner itself, not restated here.
        from Deploy.common.perception import PoseState
        from Deploy.policy.policy_runner import PolicyRunner
        import inspect

        source = inspect.getsource(PolicyRunner.command)
        self.assertIn("PoseState.STALE", source)
        self.assertIn("PoseState.LOST", source)
        self.assertNotIn("PoseState.HOLD", source)


class FirstFrameWaitTests(unittest.TestCase):
    """Opening a camera is not the same as having a frame from it.

    ``pipeline.start()`` returns before any frame arrives, and the tracker
    threads then need a frame plus a detection pass.  Sampling in that gap gives
    ``latest() is None``, which the MAIN/SIDE arbitration reads as "saw
    nothing" -- identical to markers being out of view.  That cost a hardware
    run: the failure said "are the cameras seeing the markers?" when the true
    answer was "no frame has arrived yet".
    """

    class FakeWorker:
        def __init__(self, name, ready_after_polls):
            self.name = name
            self._left = ready_after_polls

        def latest(self):
            if self._left > 0:
                self._left -= 1
                return None
            return object()

    def _provider(self):
        from Deploy.vision.provider import DualCameraStickPoseProvider

        provider = DualCameraStickPoseProvider(
            25.000097, acknowledge_candidate_geometry=True
        )
        provider.tracker = types.SimpleNamespace(
            MAIN_CAMERA_SERIAL="814412070582", SIDE_CAMERA_SERIAL="342222074358"
        )
        return provider

    def test_waits_until_both_cameras_deliver(self) -> None:
        provider = self._provider()
        provider._workers = [self.FakeWorker("MAIN", 3), self.FakeWorker("SIDE", 8)]
        warmup = provider.wait_for_first_frames(timeout_s=5.0)
        self.assertEqual(set(warmup), {"MAIN", "SIDE"})
        # Recorded as gauges so a slow-starting camera is visible in the report.
        self.assertIn("side_warmup_ms", provider.timing.labels)

    def test_a_silent_camera_is_named_and_blamed_on_the_camera(self) -> None:
        provider = self._provider()
        provider._workers = [self.FakeWorker("MAIN", 0),
                             self.FakeWorker("SIDE", 10**9)]
        with self.assertRaises(RuntimeError) as caught:
            provider.wait_for_first_frames(timeout_s=0.2)
        message = str(caught.exception)
        self.assertIn("SIDE", message)
        self.assertNotIn("MAIN,", message)
        # Must not send the operator to move the chopsticks.
        self.assertIn("카메라 문제", message)
        self.assertIn("342222074358", message)

    def test_failure_message_separates_the_three_causes(self) -> None:
        provider = self._provider()
        provider._modules = (types.SimpleNamespace(TARGET_IDS=(0, 1)),
                             types.SimpleNamespace(TARGET_IDS=(2, 3)))
        from Deploy.vision.provider import StickSourceReport

        report = StickSourceReport(stick1_reason="MAIN_MISSED_BOTH_SIDE_INVALID",
                                   stick2_reason="MAIN_MISSED_BOTH_SIDE_INVALID")
        no_frame = provider._first_pose_failure(None, None, report)
        self.assertIn("프레임 없음", no_frame)

        seen_nothing = types.SimpleNamespace(detected={})
        seen_marker = types.SimpleNamespace(detected={0: None})
        message = provider._first_pose_failure(seen_nothing, seen_marker, report)
        self.assertIn("마커가 안 보입니다", message)   # MAIN: frame, no markers
        self.assertIn("[0]", message)                  # SIDE: marker 0 seen
        self.assertNotIn("프레임 없음", message)

    def test_reset_waits_for_a_fresh_frame_before_returning(self) -> None:
        """The exact sequence a hardware run failed on.

        start() -> reset() -> sample().  reset() clears each worker's latest
        frame together with the tracker history, and the caller samples on the
        next line, so reset must re-wait or that sample sees nothing.
        """

        class Resettable:
            def __init__(self, name, warm_polls):
                self.name = name
                self.warm_polls = warm_polls
                self._left = warm_polls

            def reset(self):
                self._left = self.warm_polls      # frames thrown away

            def latest(self):
                if self._left > 0:
                    self._left -= 1
                    return None
                return object()

        provider = self._provider()
        provider._workers = [Resettable("MAIN", 2), Resettable("SIDE", 4)]
        provider.wait_for_first_frames(timeout_s=5.0)
        self.assertIsNotNone(provider._workers[0].latest())

        provider.reset()
        # After reset the frames must be back, not None.
        self.assertIsNotNone(provider._workers[0].latest())
        self.assertIsNotNone(provider._workers[1].latest())

    def test_reset_without_open_cameras_does_not_wait(self) -> None:
        # reset() is also reachable before start(); it must not block there.
        provider = self._provider()
        provider.reset()


class CrossCameraSyncTests(unittest.TestCase):
    """One observation built from two cameras needs them close in time.

    The arbitration picks a camera per stick, so Stick1 can come from MAIN and
    Stick2 from SIDE -- measured 386 of 450 steps on 2026-08-21.  Their newest
    frames are up to a frame apart (p95 49 ms), and at the sticks' own
    21-24 mm/s that is about 1 mm of error in the stick1-to-stick2 geometry the
    grasp depends on.  The task is chopstick manipulation, so "the sticks are
    not moving" is not an available assumption.

    The tracker already finds the closest-in-time pair (7-8.5 ms) and prints it;
    it just never feeds it to the pose path, because a display has no use for
    it.  Re-selecting from that pair costs a little frame age and buys back most
    of the skew -- but only when the sources actually differ.
    """

    def setUp(self) -> None:
        from Deploy.vision.provider import DualCameraStickPoseProvider

        self.P = DualCameraStickPoseProvider

    def test_same_camera_is_not_treated_as_cross_camera(self) -> None:
        # Both on MAIN: a nearest-pair frame would just be older for nothing.
        self.assertFalse(self.P._cross_camera([{"source": "MAIN"}, {"source": "MAIN"}]))
        self.assertFalse(self.P._cross_camera([{"source": "MAIN"}, {"source": "NONE"}]))

    def test_split_sources_are_detected(self) -> None:
        self.assertTrue(self.P._cross_camera([{"source": "MAIN"}, {"source": "SIDE"}]))
        self.assertTrue(self.P._cross_camera([{"source": "SIDE"}, {"source": "MAIN"}]))

    def test_resync_can_be_switched_off(self) -> None:
        provider = self.P(25.000097, acknowledge_candidate_geometry=True,
                          prefer_synchronised_pair=False)
        self.assertFalse(provider.prefer_synchronised_pair)

    def test_nearest_pair_comes_from_worker_history(self) -> None:
        # It must read history_snapshot(), not latest(): latest() is the very
        # thing whose skew is being avoided.
        import inspect

        source = inspect.getsource(self.P._nearest_pair)
        self.assertIn("history_snapshot()", source)
        self.assertNotIn("latest()", source)
        self.assertIn("find_nearest_timestamp_pair", source)
