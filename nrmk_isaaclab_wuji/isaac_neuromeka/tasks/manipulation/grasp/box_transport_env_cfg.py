"""Indy-Wuji-Box-Transport 장면/환경 설정 (2026-07-16).

cube_grasp_env_cfg.py의 사본 + 랜덤 직육면체 확장 (큐브 태스크 동결 방침으로 별도 파일).
차이: env별 상자 치수 랜덤화(prestartup) → replicate_physics=False, obs 64.
asset 이름은 mdp 파라미터 호환을 위해 "cube"를 유지함 (실체는 직육면체).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import (
    ReachSceneCfg,
)

# 테이블/기준면. 큐브 태스크(2026-07-15 실험값)와 동일 — 팔꿈치 웅크림 방지 검증된 기하.
BASE_Z = 0.25                     # 테이블 상판 높이
BOX_BASE_SIZE = 0.06              # spawn 큐보이드 기준 치수 (스케일 랜덤화의 분모)
CUBE_POS = (0.62, -0.20)          # 상자 스폰 x/y

import isaac_neuromeka.mdp as mdp  # noqa: F401, E402
from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg  # noqa: E402
from isaac_neuromeka.tasks.manipulation.grasp.box_mdp_cfg import (  # noqa: E402
    BoxTransportActionsCfg,
    BoxTransportCommandsCfg,
    BoxTransportEventCfg,
    BoxTransportObservationsCfg,
    BoxTransportRewardsCfg,
    BoxTransportTerminationsCfg,
)
from isaac_neuromeka.utils.etc import EmptyCfg  # noqa: E402


@configclass
class BoxTransportSceneCfg(ReachSceneCfg):
    """Scene config: 테이블 + env별 크기가 달라질 기준 직육면체."""

    # 테이블 (kinematic). 가장자리 = 낙하 감지기 (쳐냄 -> 낙하 종료 -> 빠른 리셋)
    support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(CUBE_POS[0], CUBE_POS[1], BASE_Z / 2),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 1.0, BASE_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.45, 0.4),
                metallic=0.0,
                roughness=0.8,
            ),
        ),
    )

    # 기준 상자 (BOX_BASE_SIZE 정육면체로 스폰 -> prestartup에서 env별 스케일 적용됨).
    # 이름 "cube" 유지: mdp 파라미터들(SceneEntityCfg("cube"))과 호환.
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            # z는 placeholder — set_box_default_height(startup)가 env별 반높이로 보정
            pos=(CUBE_POS[0], CUBE_POS[1], BASE_Z + BOX_BASE_SIZE / 2),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BOX_BASE_SIZE, BOX_BASE_SIZE, BOX_BASE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            # 질량은 크기와 무관하게 고정 (밀도가 env마다 다름 — 변수 통제 의도)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.20),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.55, 0.1),
                metallic=0.0,
                roughness=0.5,
            ),
        ),
    )


@configclass
class BoxTransportEnvCfg(NrmkRLEnvCfg):
    """Randomized-box grasp & fixed-goal transport environment."""

    scene: BoxTransportSceneCfg = BoxTransportSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: BoxTransportObservationsCfg = BoxTransportObservationsCfg()
    actions: BoxTransportActionsCfg = BoxTransportActionsCfg()
    commands: BoxTransportCommandsCfg = BoxTransportCommandsCfg()
    rewards: BoxTransportRewardsCfg | EmptyCfg = BoxTransportRewardsCfg()
    terminations: BoxTransportTerminationsCfg = BoxTransportTerminationsCfg()
    events: BoxTransportEventCfg | EmptyCfg = BoxTransportEventCfg()
    curriculum = EmptyCfg()
    costs = EmptyCfg()

    actor_obs_list: list = ["policy"]
    critic_obs_list: list | None = None
    teacher_obs_list: list | None = None

    def __post_init__(self):
        """Post initialization."""
        # 접촉 태스크 제어 주기/버퍼 (큐브 태스크와 동일 근거 — cube_grasp_env_cfg.py 주석 참고)
        self.sim.dt = 1.0 / 60.0
        self.decimation = 2  # 30 Hz
        self.episode_length_s = 8.0
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        # ★ env별 지오메트리: 물리 복제를 꺼야 env마다 다른 스케일이 파싱됨
        # (startup이 느려지는 비용. step 속도는 거의 무관)
        self.scene.replicate_physics = False

        # ★ 받침면 기준 배선 (지우면 안 됨 — 없으면 lift가 스폰부터 만점인 버그.
        #   2026-07-15 큐브 태스크에서 실제 유실 사고). metrics는 cube_lift 것을 자동으로 읽음.
        self.rewards.cube_lift.params["surface_z"] = BASE_Z
        self.rewards.hand_floor.params["surface_z"] = BASE_Z
        # env별 스폰 높이 보정 이벤트의 기준면
        self.events.set_box_height.params["surface_z"] = BASE_Z
        # 낙하 종료: 상자 중심이 상판 5cm 아래 = 테이블 밖 (최소 상자 반높이 1.5cm라 정상 파지 중 불가)
        self.terminations.cube_dropped.params["minimum_height"] = BASE_Z - 0.05
        # 운반 goal (1단계: 스폰 위 +20cm 고정점. 2단계 랜덤 박스 확장값은 큐브 태스크 주석 참고)
        self.commands.cube_goal.ranges.pos_x = (CUBE_POS[0], CUBE_POS[0])
        self.commands.cube_goal.ranges.pos_y = (CUBE_POS[1], CUBE_POS[1])
        self.commands.cube_goal.ranges.pos_z = (BASE_Z + 0.20, BASE_Z + 0.20)

        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
