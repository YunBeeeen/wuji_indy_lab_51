"""목표 파지 g = (hp, hr, hj) 정의.

논문(Dexterous Pre-grasp Manipulation)은 g를 외부 oracle/데이터셋에서 받는다.
우리는 직육면체이므로 **치수에서 해석적으로 계산**한다. 젓가락으로 넘어가면 constraint-based
표현(검지끝 3D 위치 + EE 회전)으로 바꾸면 되고, 이 모듈의 인터페이스는 그대로 쓴다.

핵심: **g는 "물체 로컬 좌표계"에서 정의한다.**
    물체가 굴러가면 g도 같이 월드에서 움직인다.
    -> "지금 물체 자세로는 g에 도달할 수 없다" 상황이 자연스럽게 생기고,
       정책은 먼저 물체를 굴려서(r_orient) g를 도달 가능하게 만들어야 한다.
       그게 논문 제목의 pre-grasp manipulation이다.
    로봇 기준으로 정의하면 이 성질이 사라진다. 반드시 물체 기준으로 둘 것.
"""

from __future__ import annotations

import torch

# -----------------------------------------------------------------------------
# 흐름
#
#   물체 pose (월드)          손 상태 (월드)
#        │                         │
#        ▼                         │
#   g_local (물체 로컬, 상수)       │
#        │                         │
#        ▼                         │
#   g_world = 물체pose ∘ g_local   │      <- 매 step 계산
#        │                         │
#        └──────────┬──────────────┘
#                   ▼
#        Δhp = |hp_world - hp_hand|      (위치 오차)
#        Δhr = angle(hr_world, hr_hand)  (회전 오차)
#        Δhj = mean|hj_target - hj_hand| (관절 오차)
#                   │
#                   ├─> r_hp, r_hr, r_hj  (전부 차분형)  -> r_grasp
#                   ├─> r_T  (세 오차가 전부 임계 이하면 1, 아니면 0)
#                   └─> observation (정책이 목표를 알아야 하므로)
# -----------------------------------------------------------------------------


def box_target_grasp_local(
    half_extent: tuple[float, float, float],
    approach_clearance: float = 0.0,
) -> dict[str, torch.Tensor]:
    """직육면체의 목표 파지를 물체 로컬 좌표로 계산.

    직육면체(3 x 3 x 16 cm)의 파지 논리:
      - 긴 축(로컬 z)을 **가로질러** 잡는다. 짧은 축(3cm)이 손가락 사이 간극에 들어감.
      - 손은 물체의 옆면에서 접근. 파지 개구부(엄지-손가락 사이)가 물체 중심을 향해야 함.
      - 젓가락으로 바뀌면 "긴 축의 특정 지점을 검지로" 로 바뀜 -> 여기만 고치면 됨.

    Returns:
        hp: (3,)  손 위치 목표 (물체 로컬)
        hr: (4,)  손 회전 목표 (물체 로컬, quaternion wxyz)
        hj: (N,)  손가락 관절 목표 (물체와 무관하므로 로컬/월드 구분 없음)

    TODO:
      - hp: 물체 옆면에서 `half_extent[0] + palm_offset` 만큼 떨어진 점.
            palm_offset은 palm_link 원점에서 파지 중심까지의 거리 (실측 약 0.13 m,
            palm 로컬 (0.19, 0.28, 0.94) 방향으로).
      - hr: 파지 개구부 축이 -hp 방향(= 물체 중심 쪽)을 향하고,
            손가락이 닫히는 평면이 물체의 긴 축과 수직이 되는 quaternion.
            -> palm 로컬의 개구부 축 (0.19, 0.28, 0.94)를 목표 방향에 정렬시키는 회전.
      - hj: 짧은 축(3cm)을 감쌌을 때의 손가락 관절각.
            cage_span이 물체 폭 근처가 되는 값. 샘플링이나 IK로 한 번 구해서 상수로 박아둘 것.
    """
    raise NotImplementedError


def target_grasp_world(
    object_pos_w: torch.Tensor,   # (N, 3)
    object_quat_w: torch.Tensor,  # (N, 4)
    hp_local: torch.Tensor,       # (3,)
    hr_local: torch.Tensor,       # (4,)
) -> tuple[torch.Tensor, torch.Tensor]:
    """물체 로컬의 g를 월드로 변환. 매 step 호출.

    hp_w = object_pos + R(object_quat) @ hp_local
    hr_w = object_quat * hr_local        (quaternion 곱)

    **이게 pre-grasp manipulation의 핵심.** 물체가 굴러가면 g_world가 따라 움직이므로,
    정책은 "물체를 어떻게 놓아야 g에 도달할 수 있는가"를 스스로 풀어야 한다.
    """
    raise NotImplementedError


def nominal_object_rotation() -> torch.Tensor:
    """물체의 "정상(nominal)" 자세. r_orient가 이쪽으로 물체를 돌리도록 보상.

    논문: "the object z-axis points upwards, and the object x-axis (the direction of the tool tip)
           points away from the hand."

    직육면체(막대)의 경우:
      - 지금 초기 자세는 **누워 있음** (긴 축이 월드 x) -> 그대로는 잡기 어려움
      - nominal은 **세워진 자세** (긴 축이 월드 z) 또는 "긴 축이 손에서 멀어지는 방향"
      - 이 값이 곧 "정책이 물체를 어떤 자세로 만들어야 하는가"를 정의함

    **여기가 pre-grasp manipulation의 난이도를 정하는 손잡이임.**
    초기 자세와 nominal이 같으면 굴릴 필요가 없어서 과제가 무의미해지고,
    너무 멀면 학습이 안 됨. 커리큘럼으로 조절할 것.
    """
    raise NotImplementedError
