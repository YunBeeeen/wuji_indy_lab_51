"""functional_grasp 전용 MDP 항들 (논문 방식).

기존 것은 재사용:
    isaac_neuromeka.mdp.rewards 의
        ObjectCageProgressReward   -> r_reach
        object_in_finger_cage      -> r_hold
        arm_manipulability_penalty -> r_MP
        object_lift_in_cage        -> r_lift
        hand_floor_penalty, action_rate_l2
        _box_signed_distance, cage_points, box_ground_clearance

여기서 새로 만드는 것:
    target_grasp.py  — 목표 파지 g = (hp, hr, hj) 정의 및 월드 변환
    rewards.py       — r_hp, r_hr, r_hj (r_grasp), r_orient, r_T
"""

# from .target_grasp import *  # noqa
# from .rewards import *  # noqa
