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
    CustomResidualJointPositionAction,
    JointResidualAction,
    MimicJointPositionAction,
    ReferenceResidualJointPositionAction,
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

# action scale 조정, residual joint 방식으로 바꾸라고 하심, (뉴로메카 예제 똑같이) , clamp 수정 /

@configclass
class MimicJointActionCfg(JointPositionActionCfg):
    """CustomJointPositionAction + follower 커플링.

    mimic: {follower 관절 이름: source(액션) 관절 이름}. follower는 joint_names에 넣지 말 것
    (액션/관측 차원에 안 들어가고 목표만 복사받음). MimicJointPositionAction 참고.
    """

    class_type: type[ActionTerm] = MimicJointPositionAction

    mimic: dict[str, str] = {}


@configclass
class CustomResidualJointActionCfg(JointActionCfg):
    """잔차형(residual) 위치 액션 — 뉴로메카 예제 `ResidualJointActionCfg`와 같은 구조 (2026-07-23).

    예제처럼 `JointActionCfg`를 상속하고 `offset` 기본 0.0을 쓴다(= default_joint_pos를 더하지
    않음). 커플링(mimic)은 없다 — 약지/새끼까지 정책이 직접 제어하는 구성을 위한 term.

    joint_names에 `finger[1-5]`를 넣으면 action dim 18 → 26, 관측 `joint_pos`도 26이 되어
    policy obs 75 → **91**이 된다(fresh 필수). joint_pos(18→26)뿐 아니라
    `action_history`(= action_manager.prev_action)도 액션 차원을 따라가므로 +8이 두 번 붙는다.

    ⚠ scale 단위가 절대형과 다름: "기본자세로부터의 변위"가 아니라 **스텝당 증분**.
      최대속도 ≈ (kp/kd)×scale [rad/s], 최대 유지토크 ≈ kp×scale.
      자세한 건 `CustomResidualJointPositionAction` docstring 참고.
    """

    class_type: type[ActionTerm] = CustomResidualJointPositionAction

    offset: float | dict[str, float] = 0.0

    # 예제엔 없는 추가 옵션. 관절 스토퍼를 계속 미는 토크를 없앰. 예제 그대로 가려면 False.
    clamp_to_limits: bool = True


@configclass
class ReferenceResidualJointActionCfg(CustomResidualJointActionCfg):
    """Position residual around a fixed joint-command reference.

    ``target = reference_positions + action * scale + offset``.  This preserves
    a manually measured contact preload while the policy learns small
    stabilization corrections around it.
    """

    class_type: type[ActionTerm] = ReferenceResidualJointPositionAction

    reference_positions: tuple[float, ...] = MISSING


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
