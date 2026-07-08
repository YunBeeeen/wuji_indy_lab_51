from __future__ import annotations

import os
import pdb  # noqa:F401

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

##
# Scene definition
##
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg

# Test camera
from isaaclab.sensors import (  # noqa: F401
    ContactSensor,
    ContactSensorCfg,
    FrameTransformer,
    FrameTransformerCfg,
    RayCasterCfg,
    TiledCamera,
    TiledCameraCfg,
    save_images_to_file,
)
from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.terrains import TerrainImporterCfg  # noqa: F401
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise  # noqa: F401
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: F401

import isaac_neuromeka.mdp as mdp

# from isaac_neuromeka.assets import MOBY_CFG
from isaac_neuromeka.assets import MOBY200_CFG
from isaac_neuromeka.assets.articulation import FiniteArticulationCfg

# Import common environment configuration
from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg  # TODO: move one level up
from isaac_neuromeka.mdp.actions import (  # noqa: F401
    ClampedJointPositionAction,
    CustomJointPositionAction,
    JointVelocityAction,
)
from isaac_neuromeka.mdp.actions.action_cfgs import TerrainFloatingBaseVelocityActionCfg
from isaac_neuromeka.mdp.actions.base_actions import TerrainFloatingBaseVelocityAction
from isaac_neuromeka.terrain.mesh_terrain_cfg import (  # noqa: F401
    MeshTerrainImporterCfg,
)
from isaac_neuromeka.utils.etc import EmptyCfg


def lidar_radial_pattern_with_range(cfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a horizontal radial LiDAR pattern over ``cfg.angle_range`` in degrees."""
    angle_range_rad = [np.deg2rad(cfg.angle_range[0]), np.deg2rad(cfg.angle_range[1])]
    azimuth = torch.linspace(angle_range_rad[0], angle_range_rad[1], cfg.num_rays, device=device)

    ray_directions = torch.stack([torch.cos(azimuth), torch.sin(azimuth), torch.zeros_like(azimuth)], dim=-1)
    ray_directions = torch.nn.functional.normalize(ray_directions, p=2.0, dim=-1)
    ray_starts = torch.zeros_like(ray_directions)

    return ray_starts, ray_directions


@configclass
class PlanarLidarPatternCfg(PatternBaseCfg):
    """2D LiDAR pattern in the sensor XY plane."""

    func = lidar_radial_pattern_with_range

    num_rays: int = 271
    angle_range: tuple[float, float] = (-135.0, 135.0)


@configclass
class MobySceneCfg(InteractiveSceneCfg):

    ##############################
    # Terrain can be set as (1) plane ground (2) scanned scene
    #
    # Scanned scene should be prepared in USD file format.
    # To convert mesh files to USD files, follow below steps:
    # (1) Prepare GLB/OBJ/STL files of the scene (e.g., opensource HM3D dataset, Manual 3D scanning of the scene, etc.)
    # (2) Convert mesh files to USD format using the converter in "IsaacLab". The command is as follows:
    #     ```
    #     python scripts/tools/convert_mesh.py /PATH/TO/MESH/FILE(.glb, .obs, .stl) /PATH/TO/USD/FILE(.usd) --collision-approximation meshSimplification
    #     ```
    # (3) Set the USD file path in the "usd_path" field of the TerrainImporterCfg.

    # # Choice 1: Plane ground
    # terrain = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    # )

    # Choice 2: Scanned scene
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        # usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/../../assets/scene/hm3d_1/usd/hm3d_1.usd",   # Example 1: HM3D dataset
        # usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/../../assets/scene/hm3d_2/usd/hm3d_2.usd",   # Example 2: HM3D dataset
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/../../assets/scene/nrmk_2nd_floor/usd/nrmk_2nd_floor.usd",  # Example 3: Manually scanned scene
        env_spacing=5.0,
    )
    ##############################

    # robots
    robot: FiniteArticulationCfg = MOBY200_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # target object
    obstacle: RigidObjectCfg = None

    # contact sensor # TODO: add contact sensor (also enable in moby.py in isaac_neuromeka/assets)
    contact_sensors = None  # To be added in the future for base collision detection

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    camera_front = TiledCameraCfg(
        prim_path="/World/envs/env_.*/robot/base_link/camera_front",  # TODO: adjust the pose
        offset=TiledCameraCfg.OffsetCfg(pos=(0.5, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=640,
        height=480,
    )

    lidar = RayCasterCfg(
        # prim_path="{ENV_REGEX_NS}/robot/front_nanoscan3_sensor_link",
        prim_path="{ENV_REGEX_NS}/robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
        update_period=0.0,
        mesh_prim_paths=["/World/ground"],
        ray_alignment="base",
        pattern_cfg=PlanarLidarPatternCfg(num_rays=1024, angle_range=(-180.0, 180.0)),
        max_distance=10.0,
        debug_vis=True,
    )


##
# Environment configuration
##


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RobotStatesCfg(ObsGroup):
        p = ObsTerm(func=mdp.position_in_world, params={"body_name": "base_link"})
        orientation = ObsTerm(func=mdp.orientation_in_world, params={"body_name": "base_link"})
        base_vel = ObsTerm(
            func=mdp.finite_body_vel_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        image = ObsTerm(
            func=mdp.image_unnormalized, params={"sensor_cfg": SceneEntityCfg("camera_front"), "data_type": "rgb"}
        )
        depth_image = ObsTerm(
            func=mdp.image_unnormalized, params={"sensor_cfg": SceneEntityCfg("camera_front"), "data_type": "depth"}
        )
        pointcloud = ObsTerm(func=mdp.lidar_pointcloud, params={"sensor_cfg": SceneEntityCfg("lidar")})

        # TODO: add arm states

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy = RobotStatesCfg()


@configclass
class ActionsCfg:
    vel_action: ActionTerm = TerrainFloatingBaseVelocityActionCfg(
        class_type=TerrainFloatingBaseVelocityAction,
        asset_name="robot",
        velocity_scale=1.0,
        yaw_rate_scale=1.0,
        offset_z_pos=0.02,
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # reset_robot_pose = EventTerm(
    #     func=mdp.reset_pose_mesh_terrain,
    #     mode="reset",
    #     params={
    #         "pose_range":  {"yaw": (0.0, 3.14)},
    #         "velocity_range":  {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
    #     },
    # )

    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.7, 1.2),
            "operation": "abs",
            "distribution": "uniform",
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class MobyDeployEnvCfg(NrmkRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: MobySceneCfg = MobySceneCfg(num_envs=1, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    commands = EmptyCfg()
    actions = ActionsCfg()
    rewards = EmptyCfg()
    curriculum = EmptyCfg()  # Not used for now
    costs = EmptyCfg()  # Not used for now
    terminations = TerminationsCfg()  # Not used for now
    events = EventCfg()

    actor_obs_list = ["policy"]

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # task settings
        self.sim.dt = 1.0 / 200.0
        self.decimation = 20  # 20 * 1/200 = 0.1s # Control at 10 Hz
        self.episode_length_s = 100.0

        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
