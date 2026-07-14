from __future__ import annotations

import isaac_neuromeka.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaac_neuromeka.assets import INDY7_WUJI_RIGHT_CFG
from isaac_neuromeka.mdp.actions import MimicJointActionCfg
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
        # asset의 finger[1-5] 키는 반드시 지울 것 (정규식 중복 매칭).
        # 약지/새끼 이력: 폄(초기) -> 접음(큐브를 쳐냄 방지) -> 다시 폄(접힌 채 자가 충돌)
        # -> 2026-07-14 중지 커플링 (아래 mimic). 이제 액션을 따라 같이 오므려짐.
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

        self.actions.arm_action = MimicJointActionCfg(
            asset_name="robot",
            joint_names=controlled_joint_names,
            # 2026-07-13_21-57-31 run 값: 스칼라 1.0
            # 절대 위치 명령임 (target = default + scale * action). scale이 곧 "도달 반경".
            # clip_actions=1.0 과 짝 -> 관절 목표가 default ± 1.0 rad 로 묶임.
            scale=1.0,
            use_default_offset=True,
            # 2026-07-14 약지/새끼 커플링: 중지의 목표를 그대로 따라감 (논문 Schunk SIH의
            # 관절 커플링 방식). 액션 18D / 관측 57D는 그대로, 감싸는 받침 손가락 2개 추가.
            # 근거: palm-up 실측에서 0.30kg을 버틴 주체가 새끼/손바닥 — 받침 능력은 검증됨.
            # 이 손의 집게는 물리는 창이 좁아(오므림 30~50%) 받침이 있으면 성공 조건이 느슨해짐.
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
            # 2026-07-14 손가락 목표는 음수 금지 (물리 한계는 그대로 -> 수동 순응성 유지).
            # 시작 자세(0)가 이미 최대 폄이라 음수 목표는 기능이 없고, 실제로는 과신전으로
            # 벌어진 채 바닥에 박혀 kp 2로 못 접히고 에피소드가 끝나는 실패 모드만 만들었음.
            # 파지에 필요한 동작은 전부 양수 방향 (grip_capacity 오므림 sweep으로 검증).
            # 부작용: 액션 [-1,0]이 전부 "완전 폄"에 매핑되는 데드존. 폄 근처에서 미적대면
            # 리매핑(offset 0.8 / scale 0.8)으로 교체할 것.
            target_clamp={"finger[1-3]_joint[1-4]": (0.0, None)},
        )

        # 커플링으로 finger4-5도 매 스텝 목표를 받으므로, 접힘을 gain으로 붙들던 예외(kp 20,
        # kd 0.5)를 걷어내고 전 손가락을 제조사 값으로 통일. asset 기본값은 건드리지 않음 —
        # chopsticks/functional_grasp는 접힌 finger4-5를 gain으로 유지해야 해서 kp 20이 필요함.
        fingers = self.scene.robot.actuators["fingers"]
        fingers.stiffness = {
            "finger[1-5]_joint[1-2]": 2.0,
            "finger[1-5]_joint[3-4]": 1.0,
        }
        fingers.damping = 0.05
