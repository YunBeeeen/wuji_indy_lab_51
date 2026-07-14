from __future__ import annotations

import isaac_neuromeka.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaac_neuromeka.assets import INDY7_WUJI_RIGHT_CFG
from isaac_neuromeka.mdp.actions import CustomJointPositionAction
from isaac_neuromeka.tasks.manipulation.grasp.cube_grasp_env_cfg import (  # noqa: F401
    CubeGraspEnvCfg,
)

##
# Environment configuration
##

@configclass
class Indy7WujiCubeGraspEnvCfg(CubeGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = INDY7_WUJI_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.render_interval = self.decimation
        # 약지/새끼는 액션에 없는데(dim 축소) 액추에이터는 stiffness=20으로 붙어 있어서, 편 자세(0.0)에
        # 뻣뻣하게 고정된 채 손에서 제일 앞으로 튀어나와 있었음 (tip x: 약지 .841 > 새끼 .838 > 중지 .837
        # > 검지 .825). 접근하면 이 둘이 큐브를 먼저 쳐서 12cm 밀어내고 반작용으로 팔이 튕겨나감.
        # 3지 파지처럼 안쪽으로 접어둠. asset의 finger[1-5] 키는 반드시 지울 것 (정규식 중복 매칭).
        # ★ 2026-07-13_21-57-31 run 값 (params/env.yaml 에서 복원)
        joint_pos = self.scene.robot.init_state.joint_pos
        joint_pos.pop("finger[1-5]_joint[1-4]", None)
        joint_pos.update(
            {
                "joint0": 0.0,
                "joint1": -0.45,
                "joint2": -1.85,
                # 2026-07-14 손목 탐색(40만 샘플) 최상위: 파지중심이 손바닥보다 5.7cm 아래로
                # 내려와 위에서 덮치는 방향이 됨. manip 0.0645 -> 0.0803 (특이점에서 더 멂).
                # 이전값 joint3=0.0, joint4=1.2 는 손바닥이 하늘을 봐서 큐브에 손을 대려면
                # 손목을 크게 돌려야 했고 그 과정에서 큐브를 계속 쳐냈음.
                "joint3": -1.61,
                "joint4": -1.62,
                "joint5": 2.356194490192345,
                # 전 손가락 폄. 접어두면(1.2) 접힌 손가락끼리 자가 충돌로 씨름하다 튕기며
                # 파지 지점을 쓸었음 (실측: 접힘 유지 오차 1.27rad, 토크 포화 82%).
                # 편 자세는 목표=기본=0이라 정적이고, GUI 관찰로도 편 쪽이 큐브를 안 침.
                "finger[1-5]_joint[1-4]": 0.0,
            }
        )

        controlled_joint_names = ["joint[0-5]", "finger[1-3]_joint[1-4]"]

        def controlled_joint_cfg():
            return SceneEntityCfg("robot", joint_names=controlled_joint_names)

        self.observations.policy.joint_pos.params = {"asset_cfg": controlled_joint_cfg()}

        self.actions.arm_action = mdp.JointPositionActionCfg(
            class_type=CustomJointPositionAction,
            asset_name="robot",
            joint_names=controlled_joint_names,
            # 2026-07-13_21-57-31 run 값: 스칼라 1.0
            # 절대 위치 명령임 (target = default + scale * action). scale이 곧 "도달 반경".
            # clip_actions=1.0 과 짝 -> 관절 목표가 default ± 1.0 rad 로 묶임.
            scale=1.0,
            use_default_offset=True,
        )
