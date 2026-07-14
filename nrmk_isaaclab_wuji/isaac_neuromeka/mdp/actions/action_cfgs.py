from dataclasses import MISSING  # noqa: F401

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg, JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaac_neuromeka.mdp.actions.base_actions import (
    FloatingBaseVelocityAction,
    TerrainFloatingBaseVelocityAction,
)
from isaac_neuromeka.mdp.actions.joint_actions import (
    ClampedJointPositionAction,
    JointResidualAction,
    MimicJointPositionAction,
)


@configclass
class ResidualJointActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = JointResidualAction

    offset: float | dict[str, float] = 0.0

    cmd_name: str = "ee_pose"

    use_default_offset = True

    # for experiment
    repulsive_force_coeff: float = 0.1


@configclass
class ClampedJointActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = ClampedJointPositionAction

    offset: float | dict[str, float] = 0.0

    clamp_range: tuple[float, float] = (-1.0, 1.0)

    cmd_name: str = "ee_pose"

    use_default_offset = True


@configclass
class MimicJointActionCfg(JointPositionActionCfg):
    """CustomJointPositionAction + follower 커플링.

    mimic: {follower 관절 이름: source(액션) 관절 이름}. follower는 joint_names에 넣지 말 것
    (액션/관측 차원에 안 들어가고 목표만 복사받음). MimicJointPositionAction 참고.
    """

    class_type: type[ActionTerm] = MimicJointPositionAction

    mimic: dict[str, str] = {}


@configclass
class FloatingBaseVelocityActionCfg(ActionTermCfg):
    """
    Use Vx, Wz to control the robot like a floating base.
    """

    class_type: type[ActionTerm] = FloatingBaseVelocityAction
    asset_name = "robot"

    debug_vis = True

    # 속도 스케일: [-1, 1] → [-velocity_scale, +velocity_scale]
    velocity_scale: float = 0.5  # m/s 단위
    yaw_rate_scale: float = 0.5  # rad/s 단위

    max_linear_accel: float = 2.0  # m/s²
    max_angular_accel: float = 5.0  # rad/s²

    max_front_velocity: float = 1.6  # m/s
    max_yaw_rate: float = 5.0  # rad/s

    front_idx: int = 0  # x axis facing front

    fixed_z_pos: float = 0.0

    # For PD controller to fix roll, pitch angles
    angle_kp: float = 10000.0
    angle_kd: float = 500.0

    # For PD controller to fix z position (hovering)
    z_kp: float = 1000.0
    z_kd: float = 200.0

    xy_kp: float = 1000.0
    xy_kd: float = 200.0

    yaw_kp: float = 1000.0
    yaw_kd: float = 200.0

    # debug_vis

    target_vel_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target_velocity",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            )
        },
    )


@configclass
class TerrainFloatingBaseVelocityActionCfg(FloatingBaseVelocityActionCfg):
    """
    Use Vx, Wz to control the robot like a floating base over mesh terrain.
    """

    class_type: type[ActionTerm] = TerrainFloatingBaseVelocityAction

    offset_z_pos: float = 0.0  # constant offset from terrain height to base height


# @configclass
# class IKResidualActionCfg(ResidualJointActionCfg):
#     ik_method: str = "dls"
#     ik_body_name: str = "tcp"
