"""Indy7 + Wuji 오버라이드 (Box-Transport). grasp/indy_wuji/env_cfg.py의 사본 (2026-07-16).

시작 자세/커플링/게인은 큐브 태스크에서 검증된 값 그대로. 설계 근거 주석은 원본 참고.
"""

from __future__ import annotations

import isaac_neuromeka.mdp as mdp  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaac_neuromeka.assets import INDY7_WUJI_RIGHT_CFG
from isaac_neuromeka.mdp.actions import MimicJointActionCfg
from isaac_neuromeka.tasks.manipulation.grasp.box_transport_env_cfg import (  # noqa: F401
    BoxTransportEnvCfg,
)


@configclass
class Indy7WujiBoxTransportEnvCfg(BoxTransportEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = INDY7_WUJI_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.render_interval = self.decimation
        # asset의 finger[1-5] 키는 반드시 지울 것 (정규식 중복 매칭).
        joint_pos = self.scene.robot.init_state.joint_pos
        joint_pos.pop("finger[1-5]_joint[1-4]", None)
        joint_pos.update(
            {
                "joint0": 0.0,
                "joint1": -0.45,
                "joint2": -1.85,
                # 2026-07-14 손목 탐색 최상위 (위에서 덮치는 방향)
                "joint3": -1.61,
                "joint4": -1.62,
                "joint5": 2.356194490192345,
                # 전 손가락 폄 (접힘은 자가 충돌로 폐기)
                "finger[1-5]_joint[1-4]": 0.0,
            }
        )

        controlled_joint_names = ["joint[0-5]", "finger[1-3]_joint[1-4]"]

        def controlled_joint_cfg():
            return SceneEntityCfg("robot", joint_names=controlled_joint_names)

        self.observations.policy.joint_pos.params = {"asset_cfg": controlled_joint_cfg()}

        self.actions.arm_action = MimicJointActionCfg(
            asset_name="robot",
            joint_names=controlled_joint_names,
            # 절대 위치 명령 (target = default + scale * action), clip_actions=1.0과 짝
            scale=1.0,
            use_default_offset=True,
            # 약지/새끼 커플링 (Schunk SIH식): 액션 18D/관측 차원 불변, 받침 손가락 추가
            mimic={
                "finger4_joint1": "finger3_joint1",
                "finger4_joint2": "finger3_joint2",
                "finger4_joint3": "finger3_joint3",
                "finger4_joint4": "finger3_joint4",
                "finger5_joint1": "finger3_joint1",
                "finger5_joint2": "finger3_joint2",
                "finger5_joint3": "finger3_joint3",
                "finger5_joint4": "finger3_joint4",
            },
        )

        # 커플링으로 finger4-5도 매 스텝 목표를 받으므로 전 손가락 제조사 게인으로 통일
        # (asset 기본의 finger4-5 kp 20은 chopsticks 접힘 유지용 — 여기서만 override)
        fingers = self.scene.robot.actuators["fingers"]
        fingers.stiffness = {
            "finger[1-5]_joint[1-2]": 2.0,
            "finger[1-5]_joint[3-4]": 1.0,
        }
        fingers.damping = 0.05
