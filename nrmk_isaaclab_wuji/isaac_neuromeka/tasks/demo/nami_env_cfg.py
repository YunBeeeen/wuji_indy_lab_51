from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg  # noqa: F401
from isaaclab.terrains import TerrainImporterCfg  # noqa: F401
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: F401

import isaac_neuromeka.mdp as mdp
from isaac_neuromeka.assets import NAMI_CFG
from isaac_neuromeka.assets.articulation import FiniteArticulationCfg

# Import common environment configuration
from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg
from isaac_neuromeka.mdp.actions.action_cfgs import TerrainFloatingBaseVelocityActionCfg
from isaac_neuromeka.mdp.actions.base_actions import TerrainFloatingBaseVelocityAction
from isaac_neuromeka.utils.etc import EmptyCfg


@configclass
class NamiSceneCfg(InteractiveSceneCfg):

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
    robot: FiniteArticulationCfg = NAMI_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # cameras
    camera_front = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/head_yaw/camera_front",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.1080, 0.0, 0.2332), rot=(0.9848078, 0.0, 0.1736482, 0.0), convention="world"
        ),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[604.8516, 0, 321.95575, 0, 604.3739, 238.7731, 0, 0, 1],  # Realsense D435 intrinsics
            width=640,
            height=480,
            clipping_range=(0.1, 10.0),
        ),
        width=640,
        height=480,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class ObservationsCfg:  #
    """Observation specifications for the MDP."""

    @configclass
    class RobotStatesCfg(ObsGroup):
        q = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        qdot = ObsTerm(func=mdp.finite_joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        p = ObsTerm(func=mdp.position_in_world, params={"body_name": "base_link"})
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
        offset_z_pos=0.05,
    )


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class NamiNavDeployEnvCfg(NrmkRLEnvCfg):
    """Configuration for the environment."""

    # Scene settings
    scene: NamiSceneCfg = NamiSceneCfg(num_envs=1, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    commands = EmptyCfg()
    actions = ActionsCfg()
    rewards = EmptyCfg()
    curriculum = EmptyCfg()
    costs = EmptyCfg()
    terminations = TerminationsCfg()
    events = EventCfg()

    actor_obs_list = ["policy"]

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # task settings
        self.sim.dt = 1.0 / 200.0
        self.decimation = 20  # 20 * 1/200 = 0.1s # Control at 10 Hz
        self.episode_length_s = 1000.0

        # viewer settings
        self.viewer.origin_type = "asset_root"  # set to "world" for global view
        self.viewer.asset_name = "robot"

        # back view
        self.viewer.eye = (-1.3, 0.0, 1.7)
        self.viewer.lookat = (1.3, 0.0, 1.0)

        # # front view
        # self.viewer.eye = (2.5, 0.0, 2)
        # self.viewer.lookat = (-0.8, 0.0, 0.5)
