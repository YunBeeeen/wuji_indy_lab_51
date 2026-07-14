from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaac_neuromeka.tasks.manipulation.reach.reach_env_cfg import (
    ReachSceneCfg,
)


import isaac_neuromeka.mdp as mdp  # noqa: F401
from isaac_neuromeka.env.rl_task_env_cfg import NrmkRLEnvCfg

# Import common environment configuration
from isaac_neuromeka.tasks.manipulation.common.env_cfg_common import (  # noqa: F401
    CubeGraspActionsCfg,
    CubeGraspCommandsCfg,
    CubeGraspEventCfg,
    CubeGraspObservationsCfg,
    CubeGraspRewardsCfg,
    CubeGraspTerminationsCfg,
)
from isaac_neuromeka.utils.etc import EmptyCfg



@configclass
class ChopsticksGraspSceneCfg(ReachSceneCfg):
    """직육면체(= 젓가락의 단순화 모델)를 잡는 씬.

    정육면체는 대칭이라 "목표 파지 회전"을 정의할 수 없음 -> 논문의 r_hr / r_orient를 못 씀.
    직육면체는 긴 축이 있어서 목표 파지 자세가 해석적으로 정의됨 (짧은 축을 가로질러 잡기).
    나중에 실제 젓가락 메시로 교체할 것. entity 이름(cube)은 reward/metric 공유 때문에 아직 유지함.
    """

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            # x=0.45 (베이스로부터 0.485m)에서 옮김.
            # 손바닥을 아래로 향한 자세들을 샘플링해보면, "최고" manipulability는 거리에 따라 거의
            # 안 변하는데 (0.116 vs 최적점 0.125) "좋은 자세의 밀도"가 크게 다름:
            # manip>0.08인 비율이 0.485m에선 11.9%, 여기선 22.6%. 파지가 불가능했던 게 아니라
            # "가능하지만 찾기 어려웠던" 것. 논문도 물체를 manipulation workspace 안에 스폰함.
            # 단, 더 멀리는 금물: Indy7 도달거리가 약 0.8m이고 완전히 뻗은 팔은 그 자체가 특이점임.
            # (x=0.82면 큐브가 0.84m -> 팔 밖)
            pos=(0.62, -0.18, 0.015),  # 3cm 면으로 누워 있으므로 중심이 z=0.015
            # 긴 축(로컬 z)을 월드 x로 눕힘 (y축 기준 90도). 바닥에 누운 막대기.
            # 이 자세에선 바로 못 잡으므로 먼저 세우거나 굴려야 함 -> 그게 pre-grasp manipulation.
            rot=(0.70711, 0.0, 0.70711, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.16),  # 3cm x 3cm x 16cm 막대. 젓가락 프록시
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.30),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.4, 1.0),
                metallic=0.0,
                roughness=0.5,
            ),
        ),
    )


@configclass
class ChopsticksGraspEnvCfg(NrmkRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ChopsticksGraspSceneCfg = ChopsticksGraspSceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: CubeGraspObservationsCfg = CubeGraspObservationsCfg()
    actions: CubeGraspActionsCfg = CubeGraspActionsCfg()
    commands: CubeGraspCommandsCfg = CubeGraspCommandsCfg()
    # MDP settings
    rewards: CubeGraspRewardsCfg | EmptyCfg = CubeGraspRewardsCfg()
    terminations: CubeGraspTerminationsCfg = CubeGraspTerminationsCfg()
    events: CubeGraspEventCfg | EmptyCfg = CubeGraspEventCfg()
    curriculum = EmptyCfg()  # Not used for now
    # CMDP settings
    costs = EmptyCfg()  # Not used for now

    #
    actor_obs_list: list = ["policy"]  # ["proprioception", "point_cloud", "privileged"]
    critic_obs_list: list | None = None  # None: same as actor_obs_list
    teacher_obs_list: list | None = None  # None: same as actor_obs_list

    def __post_init__(self):
        """Post initialization."""
        # task settings
        # 24였음 -> 0.4초/step = 2.5 Hz. 정책이 내는 건 "관절 목표 위치"이고 그 목표를 0.4초 내내
        # 고정한 채 PD가 밀어붙임. 즉 바닥에 닿아도 0.4초가 지나야 알아챔 -> 모든 접촉이 슬램이 됨.
        # (실측: 손이 바닥에 처박고 87cm까지 튕겨 오르길 반복, 큐브가 67cm 날아감)
        # reach는 접촉이 없어서 2.5 Hz로 충분했지만 파지는 접촉이 전부임.
        # 30 Hz는 IsaacLab 공식 접촉 task들의 하한임 (Lift-Cube 50, ShadowHand 40~60, Allegro 30).
        # 15 Hz(dec 4)면 판단 사이에 손이 3.3cm(큐브 반 개) 움직여서 아슬아슬함.
        self.sim.dt = 1.0 / 60.0
        self.decimation = 2  # 33ms/step -> 30 Hz
        self.episode_length_s = 8.0  # -> 240 step/episode (기존 20)
        # 2026-07-10 run이 2**18에서 약 263k patch로 overflow남. 지금은 finger_cage_hold가 손가락을
        # 오므리게 해서 접촉이 더 늘어남. overflow는 크래시가 아니라 "접촉을 조용히 버림" -> 손이
        # 큐브를 통과하고 cage reward가 안 오름 -> "reward 설계가 잘못됨"과 구별이 불가능해짐.
        self.sim.physx.gpu_max_rigid_patch_count = 2**20

        # reward는 아직 CubeGraspRewardsCfg를 공유함 (half_extent가 정육면체 기준으로 하드코딩됨).
        # 여기서 직육면체 치수로 덮어씀. reward를 논문 방식으로 재설계할 때 제대로 분리할 것.
        # metric(managers.py)은 scene cfg에서 크기를 읽으므로 자동으로 맞춰짐.
        half = tuple(x / 2.0 for x in self.scene.cube.spawn.size)
        for term in ("finger_cage_reach", "finger_cage_hold", "cube_lift"):
            self.rewards.__dict__[term].params["object_half_extent"] = half

        # viewer settings
        self.viewer.eye = (2.5, 2.5, 2.5)
