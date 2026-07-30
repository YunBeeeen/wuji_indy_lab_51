"""Wuji-hand-only functional pre-grasp learning with two chopstick proxies."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp
from isaac_neuromeka.assets import WUJI_RIGHT_CFG
from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg
from isaac_neuromeka.mdp.actions import CustomResidualJointActionCfg
# Previous preload-injecting A/B configuration is intentionally preserved
# below as comments.  Re-enable this import only when reproducing that baseline.
# from isaac_neuromeka.mdp.actions import ReferenceResidualJointActionCfg
from isaac_neuromeka.utils.etc import EmptyCfg

from . import mdp as hand_grasp_mdp


HAND_ROOT_POS = (0.0, 0.0, 0.50)
# The palm_link mesh's local +x surface normal is aligned with world +z.
HAND_ROOT_ROT = (0.0, 0.7071067811865476, 0.0, 0.7071067811865476)

STICK_SIZE = (0.007, 0.18, 0.007)
# Local +y is the long axis. Identity keeps it along world +y, perpendicular
# to the fingers (the hand's local +z, mapped to world +x).
STICK_ROT = (1.0, 0.0, 0.0, 0.0)
STICK_Z = 0.5195

# The default collision/depenetration settings are too coarse for a 7 mm,
# 10 g body: a shallow overlap can be resolved as a visible launch.
SMALL_CONTACT_OFFSET = 0.001
SMALL_REST_OFFSET = 0.0
SMALL_MAX_DEPENETRATION_VELOCITY = 0.2

HAND_JOINT_NAMES = [
    f"finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]
HAND_JOINTS = SceneEntityCfg(
    "robot",
    joint_names=HAND_JOINT_NAMES,
    preserve_order=True,
)
PALM = SceneEntityCfg("robot", body_names=["palm_link"])
FINGERTIPS = SceneEntityCfg(
    "robot",
    body_names=[
        "finger1_tip_link",
        "finger2_tip_link",
        "finger3_tip_link",
        "finger4_tip_link",
        "finger5_tip_link",
    ],
    preserve_order=True,
)
STICK_1 = SceneEntityCfg("stick1")
STICK_2 = SceneEntityCfg("stick2")

# Manually validated functional grasp saved by hand_grasp_keyboard.py:
# logs/debug/hand_grasp_keyboard/2026-07-28_12-39-52/pose_005.json.
# The measured state and commanded target intentionally differ.  The commanded
# targets are preserved below so the previous preload-injecting baseline remains
# reproducible, but the active residual experiment resets target=measured state
# and requires the policy to create any necessary contact preload.
PREGRASP_JOINT_POSITIONS = (
    0.5377866626,
    0.8436813951,
    0.0377136655,
    -0.0000001810,
    0.7017297745,
    0.0553143807,
    1.1822255850,
    1.4215219021,
    0.4649881423,
    -0.0292181600,
    1.6298730373,
    1.1032750607,
    0.9151425958,
    -0.0129909236,
    1.3248542547,
    0.3182539344,
    0.7154092789,
    0.0788998753,
    1.6281884909,
    0.2546040118,
)
PREGRASP_JOINT_TARGETS = (
    0.5835183859,
    0.9338999987,
    0.0,
    0.0,
    0.6981317997,
    0.0237611812,
    1.2566378117,
    1.6271998882,
    0.4886921048,
    0.0,
    1.6271998882,
    0.8592541218,
    0.9203640223,
    -0.0349065810,
    1.3654001951,
    0.6587172747,
    0.7155851722,
    0.1047197506,
    1.6271998882,
    -0.1006772667,
)
PREGRASP_STICK1_POSITION_P = (
    0.0250743479,
    0.0242451150,
    0.0969612077,
)
PREGRASP_STICK1_QUATERNION_P = (
    0.4618085623,
    -0.0092124203,
    -0.1713383496,
    -0.8702247143,
)
PREGRASP_STICK2_POSITION_P = (
    0.0355986878,
    0.0160842165,
    0.0733669698,
)
PREGRASP_STICK2_QUATERNION_P = (
    0.2051235586,
    -0.6018196344,
    -0.4935579300,
    -0.5934122205,
)

# Both saved sticks use local +y as the distal chopstick tip.  The reference
# Under the transverse square-section gap metric, the validated grasp starts
# near 13.3 mm.  Start continuous recovery learning with the already proven
# 20 mm OPEN target before expanding the motion range.
STICK_TIP_OFFSET_O = (0.0, 0.5 * STICK_SIZE[1], 0.0)
STICK1_PIVOT_OFFSET_O = (0.0, -0.06, 0.0)
OPEN_TIP_GAP = 0.020
CLOSE_TIP_GAP = 0.0
# pose_005 distal-tip separation, projected into the Stick2-local x-z plane.
# This is a validated grasp-relative direction, not an inferred palm/opening
# normal.  It lets rewards reject tips that reach the right radial gap by
# sliding sideways around Stick2.
TIP_SEPARATION_DIRECTION_STICK2 = (
    0.8876932852,
    0.0,
    -0.4604352630,
)
TIP_LATERAL_SIGMA = 0.005
TIP_LATERAL_ERROR_LIMIT = 0.005
# pose_005 distal-tip offset projected onto Stick2 local +y.  The validated
# grasp is intentionally not forced to zero axial offset.
TIP_AXIAL_OFFSET_STICK2 = -0.0048015763
TIP_AXIAL_SIGMA = 0.005

CONTACT_SENSOR_NAMES = {
    "thumb_distal_stick1": "thumb_distal_stick1",
    "index_tip_stick1": "index_tip_stick1",
    "middle_tip_stick1": "middle_tip_stick1",
    "ring_tip_stick2": "ring_tip_stick2",
    "palm_stick2": "palm_stick2",
    "thumb_mid_stick2": "thumb_mid_stick2",
}
ANCHOR_CONTACT_GROUPS = (
    (CONTACT_SENSOR_NAMES["thumb_distal_stick1"],),
    (CONTACT_SENSOR_NAMES["index_tip_stick1"],),
    (CONTACT_SENSOR_NAMES["middle_tip_stick1"],),
    (CONTACT_SENSOR_NAMES["palm_stick2"],),
    (CONTACT_SENSOR_NAMES["thumb_mid_stick2"],),
)
RING_SUPPORT_GROUPS = (
    (CONTACT_SENSOR_NAMES["ring_tip_stick2"],),
)
FUNCTIONAL_CONTACT_GROUPS = ANCHOR_CONTACT_GROUPS + RING_SUPPORT_GROUPS

MODE_HELD_PARAMS = {
    "command_name": "open_close",
    "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
    "palm_cfg": PALM,
    "stick1_cfg": STICK_1,
    "stick2_cfg": STICK_2,
    "stick1_pivot_offset_o": STICK1_PIVOT_OFFSET_O,
    "stick1_tip_offset_o": STICK_TIP_OFFSET_O,
    "stick2_tip_offset_o": STICK_TIP_OFFSET_O,
    "stick_thickness": STICK_SIZE[0],
    "open_target_gap": OPEN_TIP_GAP,
    "close_target_gap": CLOSE_TIP_GAP,
    "stick1_reference_position_p": PREGRASP_STICK1_POSITION_P,
    "stick1_reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
    "stick2_reference_position_p": PREGRASP_STICK2_POSITION_P,
    "stick2_reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
    "reference_separation_direction_stick2": TIP_SEPARATION_DIRECTION_STICK2,
    "contact_threshold": 0.02,
    "pivot_error_limit": 0.015,
    "tip_gap_error_limit": 0.0005,
    "lateral_error_limit": TIP_LATERAL_ERROR_LIMIT,
    "stick2_position_error_limit": 0.02,
    "stick2_orientation_error_limit": 0.3490658504,
    "linear_speed_limit": 0.15,
    "angular_speed_limit": 3.0,
}


def _stick_cfg(name: str, pos_x: float, color: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(pos_x, 0.0, STICK_Z),
            rot=STICK_ROT,
        ),
        spawn=sim_utils.CuboidCfg(
            size=STICK_SIZE,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=SMALL_MAX_DEPENETRATION_VELOCITY,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=SMALL_CONTACT_OFFSET,
                rest_offset=SMALL_REST_OFFSET,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                metallic=0.0,
                roughness=0.5,
            ),
        ),
    )


@configclass
class HandGraspSceneCfg(InteractiveSceneCfg):
    """A fixed-base Wuji hand and two independent 7 mm square chopstick proxies."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    robot = WUJI_RIGHT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=WUJI_RIGHT_CFG.spawn.replace(
            rigid_props=WUJI_RIGHT_CFG.spawn.rigid_props.replace(
                max_depenetration_velocity=SMALL_MAX_DEPENETRATION_VELOCITY,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=SMALL_CONTACT_OFFSET,
                rest_offset=SMALL_REST_OFFSET,
            ),
            articulation_props=WUJI_RIGHT_CFG.spawn.articulation_props.replace(
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=WUJI_RIGHT_CFG.init_state.replace(
            pos=HAND_ROOT_POS,
            rot=HAND_ROOT_ROT,
        ),
    )
    stick1 = _stick_cfg("Stick1", 0.055, (0.95, 0.62, 0.12))
    stick2 = _stick_cfg("Stick2", 0.035, (0.12, 0.72, 0.32))

    thumb_distal_stick1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger1_link3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    index_tip_stick1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger2_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    middle_tip_stick1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger3_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick1"],
        update_period=0.0,
    )
    ring_tip_stick2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger4_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    palm_stick2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/palm_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )
    thumb_mid_stick2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/finger1_link2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Stick2"],
        update_period=0.0,
    )


@configclass
class HandGraspActionsCfg:
    # Active experiment: do not inject the manually saved PD preload.
    # target = current_joint_position + action * 0.1.  The policy must therefore
    # keep a non-zero residual where contact force is needed while preserving
    # the six semantic contacts and tracking the OPEN/CLOSE command.
    hand_action = CustomResidualJointActionCfg(
        asset_name="robot",
        joint_names=HAND_JOINT_NAMES,
        preserve_order=True,
        scale=0.1,
        clamp_to_limits=True,
    )

    # Previous preload-injecting baseline, retained for exact reproduction.
    # It is disabled because target=PREGRASP_JOINT_TARGETS at reset supplies the
    # manually discovered contact force instead of testing whether RL learns it.
    # hand_action = ReferenceResidualJointActionCfg(
    #     asset_name="robot",
    #     joint_names=HAND_JOINT_NAMES,
    #     preserve_order=True,
    #     scale=0.1,
    #     clamp_to_limits=True,
    #     reference_positions=PREGRASP_JOINT_TARGETS,
    # )


@configclass
class HandGraspCommandsCfg:
    # Exact 21-05-32 baseline: alternate every 2 s inside a 10 s episode.
    open_close = hand_grasp_mdp.OpenCloseModeCommandCfg(
        resampling_time_range=(0.5, 0.5),
        debug_vis=True,
        open_probability=0.5,
    )


@configclass
class HandGraspObservationsCfg:
    """103D policy state with an explicit OPEN/CLOSE command.

    Dimension breakdown:
        20 normalized joint positions
      + 20 joint velocities
      + 15 palm-frame fingertip positions (5 fingertips x xyz)
      + 14 palm-frame stick poses (2 sticks x [xyz + quaternion])
      + 12 palm-frame relative stick velocities (2 sticks x [linear + angular])
      + 20 previous actions
      +  2 OPEN/CLOSE one-hot mode
      = 103.
    """
    # joint vel x -> joint position history

    @configclass
    class PolicyCfg(ObsGroup):
        # 103 = 20 + 20 + (5 x 3) + (2 x 7) + (2 x 6) + 20 + 2.
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": HAND_JOINTS},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": HAND_JOINTS},
            scale=0.2,
        )
        fingertip_pos = ObsTerm(
            func=hand_grasp_mdp.fingertip_positions_in_palm,
            params={"palm_cfg": PALM, "fingertip_cfg": FINGERTIPS},
        )
        stick1_pose = ObsTerm(
            func=hand_grasp_mdp.object_pose_in_palm,
            params={"palm_cfg": PALM, "object_cfg": STICK_1},
        )
        stick2_pose = ObsTerm(
            func=hand_grasp_mdp.object_pose_in_palm,
            params={"palm_cfg": PALM, "object_cfg": STICK_2},
        )
        stick1_velocity = ObsTerm(
            func=hand_grasp_mdp.object_velocity_in_palm,
            params={"palm_cfg": PALM, "object_cfg": STICK_1},
            scale=0.2,
        )
        stick2_velocity = ObsTerm(
            func=hand_grasp_mdp.object_velocity_in_palm,
            params={"palm_cfg": PALM, "object_cfg": STICK_2},
            scale=0.2,
        )
        action_history = ObsTerm(func=mdp.action_history)
        open_close_mode = ObsTerm(
            func=hand_grasp_mdp.open_close_mode,
            params={"command_name": "open_close"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class HandGraspEventCfg:
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_pregrasp = EventTerm(
        func=hand_grasp_mdp.reset_to_functional_pregrasp,
        mode="reset",
        params={
            "hand_cfg": HAND_JOINTS,
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "joint_positions": PREGRASP_JOINT_POSITIONS,
            # Previous preload baseline (disabled): passing the manually saved
            # PD target here injected contact force before the policy acted.
            # "joint_position_targets": PREGRASP_JOINT_TARGETS,
            "stick1_position_p": PREGRASP_STICK1_POSITION_P,
            "stick1_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
            "stick2_position_p": PREGRASP_STICK2_POSITION_P,
            "stick2_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
        },
    )


@configclass
class HandGraspRewardsCfg:
    """Keep the functional grasp while tracking the requested tip-gap mode."""

    # This is deliberately a weak prior.  Contact mechanics may settle at joint
    # angles slightly different from the saved measured state.
    joint_reference = RewTerm(
        func=hand_grasp_mdp.JointReferenceTracking,
        weight=2.0,
        params={
            "asset_cfg": HAND_JOINTS,
            "reference_joint_positions": PREGRASP_JOINT_POSITIONS,
            "sigma": 0.20,
        },
    )
    # Freeze only the in-hand support point.  Full Stick1 pose tracking would
    # prohibit the rotation needed to open and close the distal tips.
    stick1_pivot = RewTerm(
        func=hand_grasp_mdp.ObjectPointReferenceTracking,
        weight=10.0,
        params={
            "palm_cfg": PALM,
            "object_cfg": STICK_1,
            "point_o": STICK1_PIVOT_OFFSET_O,
            "reference_position_p": PREGRASP_STICK1_POSITION_P,
            "reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
            "sigma": 0.01,
        },
    )
    # The lower stick remains the reference rail for open-close control.
    stick2_reference_pose = RewTerm(
        func=hand_grasp_mdp.ObjectReferencePoseTracking,
        weight=15.0,
        params={
            "palm_cfg": PALM,
            "object_cfg": STICK_2,
            "reference_position_p": PREGRASP_STICK2_POSITION_P,
            "reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
            "position_sigma": 0.01,
            "orientation_sigma": 0.1745329252,
        },
    )
    # Only the active mode contributes.  Keeping separate terms makes OPEN and
    # CLOSE learning visible independently in TensorBoard.
    open_tip_gap = RewTerm(
        func=hand_grasp_mdp.mode_tip_gap_tracking,
        weight=20.0,
        params={
            "command_name": "open_close",
            "mode_index": 0,
            "target_gap": OPEN_TIP_GAP,
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "stick1_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick2_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick_thickness": STICK_SIZE[0],
            "reference_separation_direction_stick2": TIP_SEPARATION_DIRECTION_STICK2,
            "reference_axial_offset_stick2": TIP_AXIAL_OFFSET_STICK2,
            "sigma": 0.005,
            "lateral_sigma": TIP_LATERAL_SIGMA,
            "axial_sigma": TIP_AXIAL_SIGMA,
        },
    )
    close_tip_gap = RewTerm(
        func=hand_grasp_mdp.mode_tip_gap_tracking,
        weight=20.0,
        params={
            "command_name": "open_close",
            "mode_index": 1,
            "target_gap": CLOSE_TIP_GAP,
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "stick1_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick2_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick_thickness": STICK_SIZE[0],
            "reference_separation_direction_stick2": TIP_SEPARATION_DIRECTION_STICK2,
            "reference_axial_offset_stick2": TIP_AXIAL_OFFSET_STICK2,
            "sigma": 0.005,
            "lateral_sigma": TIP_LATERAL_SIGMA,
            "axial_sigma": TIP_AXIAL_SIGMA,
        },
    )
    # Exact 21-05-32 contact shaping.
    functional_contact_min = RewTerm(
        func=hand_grasp_mdp.contact_group_strength,
        weight=20.0,
        params={
            "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "force_scale": 0.10,
            "reduction": "min",
        },
    )
    mode_grasp_stability = RewTerm(
        func=hand_grasp_mdp.mode_grasp_stability,
        weight=50.0,
        params={
            "command_name": "open_close",
            "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "stick1_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick2_tip_offset_o": STICK_TIP_OFFSET_O,
            "stick_thickness": STICK_SIZE[0],
            "open_target_gap": OPEN_TIP_GAP,
            "close_target_gap": CLOSE_TIP_GAP,
            "reference_separation_direction_stick2": TIP_SEPARATION_DIRECTION_STICK2,
            "reference_axial_offset_stick2": TIP_AXIAL_OFFSET_STICK2,
            "gap_sigma": 0.005,
            "lateral_sigma": TIP_LATERAL_SIGMA,
            "axial_sigma": TIP_AXIAL_SIGMA,
            "force_scale": 0.10,
            "contact_threshold": 0.02,
            "linear_speed_scale": 0.10,
            "angular_speed_scale": 2.0,
        },
    )
    angular_speed_excess = RewTerm(
        func=hand_grasp_mdp.object_pair_angular_speed_excess_l2,
        weight=-0.1,
        params={
            "palm_cfg": PALM,
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "angular_speed_limit": 3.0,
            "max_excess": 10.0,
        },
    )
    # One pulse per mode interval.  Success no longer ends the episode; the
    # command rearms this bonus when it alternates OPEN/CLOSE.
    success = RewTerm(
        func=hand_grasp_mdp.OpenCloseModeHeld,
        weight=30000.0,
        params={
            **MODE_HELD_PARAMS,
            "hold_steps": 30,
            "one_shot_per_mode": True,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)


@configclass
class HandGraspTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    stick1_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.40, "asset_cfg": STICK_1},
    )
    stick2_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.40, "asset_cfg": STICK_2},
    )
    success = DoneTerm(
        func=hand_grasp_mdp.OpenCloseModeHeld,
        params={
            **MODE_HELD_PARAMS,
            # Keep the term for canonical metrics, but make it unreachable so
            # a successful mode does not reset the continuous sequence.
            "hold_steps": 1_000_000,
        },
    )


@configclass
class HandGraspEnvCfg(NrmkRLEnvCfg):
    """Mode-conditioned OPEN/CLOSE learning around the validated hand grasp."""

    scene: HandGraspSceneCfg = HandGraspSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: HandGraspObservationsCfg = HandGraspObservationsCfg()
    actions: HandGraspActionsCfg = HandGraspActionsCfg()
    commands: HandGraspCommandsCfg = HandGraspCommandsCfg()
    rewards: HandGraspRewardsCfg = HandGraspRewardsCfg()
    terminations: HandGraspTerminationsCfg = HandGraspTerminationsCfg()
    events: HandGraspEventCfg = HandGraspEventCfg()
    curriculum = EmptyCfg()
    costs = EmptyCfg()

    actor_obs_list: list = ["policy"]
    critic_obs_list: list | None = None
    teacher_obs_list: list | None = None

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        # Exact 21-05-32 baseline: five 2 s mode intervals per episode.
        self.episode_length_s = 10.0
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        self.viewer.eye = (0.55, 0.55, 0.78)
        self.viewer.lookat = (0.045, 0.0, 0.51)
