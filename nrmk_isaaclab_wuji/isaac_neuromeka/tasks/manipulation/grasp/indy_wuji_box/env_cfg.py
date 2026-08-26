"""Indy7 + Wuji 오버라이드 (Box-Transport). grasp/indy_wuji/env_cfg.py의 사본 (2026-07-16).

시작 자세/커플링/게인은 큐브 태스크에서 검증된 값 그대로. 설계 근거 주석은 원본 참고.
"""

from __future__ import annotations

import os

import isaac_neuromeka.mdp as mdp  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaac_neuromeka.assets import INDY7_WUJI_RIGHT_CFG
from isaac_neuromeka.mdp.actions import CustomResidualJointActionCfg, MimicJointActionCfg  # noqa: F401
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

        # ── 액션 구조 스위치 (2026-07-24, chopstick과 동일) ──────────────────────────
        # 기본 = 잔차(residual) + 커플링 해제: action 26, raw-pose policy obs 69.
        #   근거: 07-23_23-52-08 런(절대형 유지)이 파지 실패 — action_track_err 1.5,
        #   thumb_index_opposition −0.20(같은 쪽), cage_inside_frac 0.08, keypoint raw 0.0002.
        #   같은 스틱에 잔차를 쓴 chopstick은 opposition +0.33/track_err 0.43으로 파지 성공.
        #   → box의 병목은 keypoint 가중치가 아니라 절대형 액션의 dead zone이었음.
        # `WUJI_LEGACY_ACTION=1` 이면 절대형 + mimic action 18, raw-pose obs 53.
        #   관측 계약도 바뀌었으므로 예전 76D 체크포인트와는 더 이상 호환되지 않는다.
        legacy_action = os.environ.get("WUJI_LEGACY_ACTION", "") == "1"
        if legacy_action:
            print("[INFO] WUJI_LEGACY_ACTION=1 → 절대형 + mimic 액션 (action 18 / raw-pose obs 53)")

        # 약지·새끼 mimic 커플링 해제 — 20개 손가락 관절을 정책이 직접 제어. fresh 필수.
        # obs 53 → 69: joint_pos(18→26) + action_history(=prev_action, 18→26) = +16.
        controlled_joint_names = (
            ["joint[0-5]", "finger[1-3]_joint[1-4]"]
            if legacy_action
            else ["joint[0-5]", "finger[1-5]_joint[1-4]"]
        )

        def controlled_joint_cfg():
            return SceneEntityCfg("robot", joint_names=controlled_joint_names)

        self.observations.policy.joint_pos.params = {"asset_cfg": controlled_joint_cfg()}

        if legacy_action:
            # 2026-07-24 이전 구성: 절대형 + 약지·새끼 mimic 커플링.
            self.actions.arm_action = MimicJointActionCfg(
                asset_name="robot",
                joint_names=controlled_joint_names,
                # 절대 위치 명령 (target = default + scale * action), clip_actions=1.0과 짝
                scale=1.0,
                use_default_offset=True,
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
        else:
            # 잔차 + 커플링 해제 (chopstick과 동일). target = 현재 관절각 + action*scale.
            # scale 단위가 스텝당 증분: 팔 0.3(1.5rad/s), 손가락 0.3(유지토크 kp×0.3).
            self.actions.arm_action = CustomResidualJointActionCfg(
                asset_name="robot",
                joint_names=controlled_joint_names,
                scale={
                    "joint[0-5]": 0.3,
                    "finger[1-5]_joint[1-4]": 0.3,
                },
                clamp_to_limits=True,
            )

        # 커플링으로 finger4-5도 매 스텝 목표를 받으므로 전 손가락 제조사 게인으로 통일
        # (asset 기본의 finger4-5 kp 20은 chopsticks 접힘 유지용 — 여기서만 override)
        fingers = self.scene.robot.actuators["fingers"]
        fingers.stiffness = {
            "finger[1-5]_joint[1-2]": 2.0,
            "finger[1-5]_joint[3-4]": 1.0,
        }
        fingers.damping = 0.05
