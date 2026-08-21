"""Indy7 + Wuji override for one-stick functional grasp."""

from __future__ import annotations

import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaac_neuromeka.assets import INDY7_WUJI_RIGHT_CFG
from isaac_neuromeka.mdp.actions import CustomResidualJointActionCfg, MimicJointActionCfg  # noqa: F401
from isaac_neuromeka.tasks.manipulation.functional_grasp.chopsticks_grasp_env_cfg import (
    ChopsticksGraspEnvCfg,
)


@configclass
class Indy7WujiChopsticksGraspEnvCfg(ChopsticksGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = INDY7_WUJI_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.render_interval = self.decimation

        joint_pos = self.scene.robot.init_state.joint_pos
        joint_pos.pop("finger[1-5]_joint[1-4]", None)
        joint_pos.update(
            {
                "joint0": 0.0,
                "joint1": -0.45,
                "joint2": -1.85,
                "joint3": -1.61,
                "joint4": -1.62,
                "joint5": 2.356194490192345,
                "finger[1-5]_joint[1-4]": 0.0,
            }
        )

        # ── 액션 구조 스위치 (2026-07-23) ────────────────────────────────────────────
        # 기본 = 잔차(residual) + 커플링 해제: action 26, policy obs 91.
        #   (joint_pos 18→26 + action_history 18→26 = +16. action_history가 prev_action이라
        #    액션 차원을 따라감 — obs 83으로 잘못 세기 쉬움)
        # `WUJI_LEGACY_ACTION=1` 이면 2026-07-23 이전 구성(절대형 + 약지·새끼 mimic):
        #   action 18, policy obs 75. **옛 체크포인트를 play할 때만** 쓸 것.
        #   예) WUJI_LEGACY_ACTION=1 python scripts/rsl_rl/play.py \
        #         --task Indy-Wuji-Chopsticks-Grasp --num_envs 1 --load_run 2026-07-23_12-13-23
        legacy_action = os.environ.get("WUJI_LEGACY_ACTION", "") == "1"
        if legacy_action:
            print("[INFO] WUJI_LEGACY_ACTION=1 → 절대형 + mimic 액션 (action 18 / obs 75)")

        # 약지·새끼 mimic 커플링 해제 — 20개 손가락 관절을 정책이 직접 제어. fresh 필수.
        controlled_joint_names = (
            ["joint[0-5]", "finger[1-3]_joint[1-4]"]
            if legacy_action
            else ["joint[0-5]", "finger[1-5]_joint[1-4]"]
        )

        def controlled_joint_cfg():
            return SceneEntityCfg("robot", joint_names=controlled_joint_names)

        self.observations.policy.joint_pos.params = {"asset_cfg": controlled_joint_cfg()}

        # ── 잔차(residual) 액션 — 활성 (2026-07-23) ──────────────────────────────────
        # target = 현재 관절각 + action*scale. 뉴로메카 예제(JointResidualAction)와 같은 구조.
        # 전환 근거: 절대형은 target = default + action인데 손가락 default가 0(굽힘 하한)이라
        #   액션 범위의 약 34%가 관절 limit 밖 = 도달 불가였음 (2026-07-23 진단).
        #
        # scale 단위가 절대형과 다름(기본자세로부터의 변위 → 스텝당 증분).
        #   최대속도 ≈ (kp/kd)×scale [rad/s] : 팔 5×0.3=1.5 (속도한계 2.775~3.3)
        #   최대 유지토크 ≈ kp×scale         : 손가락 joint1·2는 2×0.3=0.6 (effort_limit과 동일)
        #                                      joint3·4는 1×0.3=0.3 (URDF 스펙과 동일)
        #
        # ⚠ fresh 필수 (액션 의미가 바뀜 — 절대형 체크포인트 resume 금지)
        # ⚠ 잔차는 기본자세로 당기는 스프링이 없음(a=0 → "제자리 유지", 예압 0).
        #   파지 유지에 지속적인 양수 액션이 필요하고 유지 토크 ≈ kp × action × scale.
        #   스모크에서 눈으로 볼 것: ① 리셋 직후 손이 안 튀는가 ② 스틱을 잡은 뒤 놓지 않는가.
        # ⚠ action_rate 페널티 의미가 "속도 벌점"에서 "가속도 벌점"으로 바뀜 — 가중치 재검토 대상.
        if legacy_action:
            # 2026-07-23 이전 모든 런의 구성: 절대형 + 약지·새끼 mimic 커플링.
            # target = default_joint_pos + action*1.0
            self.actions.arm_action = MimicJointActionCfg(
                asset_name="robot",
                joint_names=controlled_joint_names,
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
            self.actions.arm_action = CustomResidualJointActionCfg(
                asset_name="robot",
                joint_names=controlled_joint_names,
                scale={
                    "joint[0-5]": 0.3,
                    # 2026-07-30: 손가락 scale 0.1→**0.3** 복구. 0.1은 1cm 얇은 스틱용으로
                    #   hand_grasp의 0.1 rad 정밀도에 맞춘 값이었으나, 됐던 2cm 파지(run 09-42-28)는
                    #   0.3이었음. 잔차 유지토크 ≈ kp×action×scale라 0.1이면 joint1·2 최대토크가
                    #   effort_limit(0.6)의 1/3(0.2)로 떨어져 2cm를 못 들었음(obs 검증 런 22-51-28 회귀).
                    #   (1cm 얇은 스틱 작업 재개 시 다시 0.1로.)
                    "finger[1-5]_joint[1-4]": 0.3,
                },
                clamp_to_limits=True,  # 예제와 100% 동일하게 가려면 False
            )

        # ⚠ mimic follower(finger[4-5])는 액션 관절과 겹치면 안 됨. legacy 분기가
        #   controlled_joint_names를 finger[1-3]으로 함께 되돌리는 이유 —
        #   finger[1-5]인 채 MimicJointActionCfg를 쓰면 __init__이 ValueError로 죽는다.
        #
        # 참고: chopstick_mdp_cfg.py의 hold_folded_fingers(finger[4-5] 목표를 기본자세로 고정하는
        #   리셋 이벤트)는 이제 정책이 직접 제어하므로 무의미해짐. 잔차 액션의 reset이 26관절 전부
        #   "현재 자세"로 목표를 채우고, 이벤트는 그보다 먼저 실행되므로 덮어써짐 — 무해해서 남겨 둠.
        # ──────────────────────────────────────────────────────────────────────────────

        # 2026-08-04: 손가락 stiffness/damping 오버라이드 제거(사용자). 이제 wuji.py의 WUJI_RIGHT_CFG가
        #   per-joint 튜닝된 kp/kd(wuji_actuator_parameters.py)를 담으므로 그 값을 그대로 사용한다.
        #   옛 평값(stiffness 2.0/1.0·damping 0.05)이 여기서 per-joint 튜닝을 덮어써 무효화하던 것을 제거 —
        #   joint4 kd=0.0 등 튜닝이 이제 chopsticks에도 적용됨.
        #   ⚠ grasp/indy_wuji·grasp/indy_wuji_box에는 아직 같은 오버라이드가 남아 있음(그 태스크는 미변경).
