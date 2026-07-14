from __future__ import annotations

import isaac_neuromeka.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaac_neuromeka.assets import INDY7_WUJI_RIGHT_CFG
from isaac_neuromeka.mdp.actions import CustomJointPositionAction
from isaac_neuromeka.tasks.manipulation.functional_grasp.chopsticks_grasp_env_cfg import (  # noqa: F401
    ChopsticksGraspEnvCfg,
)

##
# Environment configuration
##

@configclass
class Indy7WujiChopsticksEnvCfg(ChopsticksGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = INDY7_WUJI_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.render_interval = self.decimation
        # 약지/새끼는 액션에 없는데(dim 축소) 액추에이터는 stiffness=20으로 붙어 있어서, 편 자세(0.0)에
        # 뻣뻣하게 고정된 채 손에서 제일 앞으로 튀어나와 있었음 (tip x: 약지 .841 > 새끼 .838 > 중지 .837
        # > 검지 .825). 접근하면 이 둘이 큐브를 먼저 쳐서 12cm 밀어내고 반작용으로 팔이 튕겨나감.
        # 3지 파지처럼 안쪽으로 접어둠 (사람도 엄지-검지-중지로 집을 땐 약지/새끼를 접음).
        # asset의 finger[1-5] 키는 반드시 지울 것 — 안 그러면 정규식이 중복 매칭돼서 죽음.
        joint_pos = self.scene.robot.init_state.joint_pos
        joint_pos.pop("finger[1-5]_joint[1-4]", None)
        joint_pos.update(
            {
                "joint1": -0.45,
                "joint2": -1.85,
                "joint4": 1.20,
                "finger[1-3]_joint[1-4]": 0.0,  # 제어하는 3지는 그대로 편 채로 시작
                # 약지/새끼는 액션에 없는데(dim 축소) 액추에이터는 stiffness=20으로 붙어 있어서,
                # 편 자세(0.0)면 뻣뻣하게 고정된 채 손에서 제일 앞으로 튀어나옴 -> 접근할 때
                # 이 둘이 큐브를 먼저 쳐서 12cm 밀어내고 반작용으로 팔이 튕겨나갔음.
                # 3지 파지처럼 안쪽으로 접어둠 (사람도 엄지-검지-중지로 집을 땐 약지/새끼를 접음).
                "finger[4-5]_joint1": 1.20,  # 한계 +1.636
                "finger[4-5]_joint2": 0.0,  # 벌림은 중립
                "finger[4-5]_joint3": 1.20,  # 한계 +1.627
                "finger[4-5]_joint4": 1.20,  # 한계 +1.627
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
            # 절대 위치 명령임 (target = default + scale * action). 즉 scale이 곧 "도달 반경".
            # 0.2 + clip 없음이었는데, 그러면 |a|=1로는 관절이 11°밖에 안 움직여서 큐브(55cm 아래)에
            # 절대 못 닿음. 정책이 살려고 |a|를 5, 10까지 밀어내다가 발산했음 (실측 |Δa| 최대 9.66
            # = 관절 목표가 한 step에 110° 점프 -> 팔이 전속력으로 왕복 -> 큐브를 67cm 날림).
            # 1.0 + clip_actions=1.0 이면 목표가 default ± 1.0 rad로 묶임. 4096x8 샘플링으로
            # 그 안에 손끝 3개가 큐브 표면 2mm까지 닿는 자세가 있음을 확인함 (0.5면 3.6cm, 도달 불가).
            scale=1.0,
            use_default_offset=True,
        )
