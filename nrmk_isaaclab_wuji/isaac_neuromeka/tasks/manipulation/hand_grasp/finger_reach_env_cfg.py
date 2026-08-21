"""Middle-finger reach diagnostic environment (Sim-to-Sim / Sim-to-Real).

The purpose is NOT manipulation learning.  It is the simplest possible probe of
"given the same policy, the same initial joint state and the same target
command, do Isaac / MuJoCo / the real Wuji Hand move the same way?"

Only the middle finger is controlled (4 joints).  The other sixteen joints are
held at their reset pose.  The policy sees a palm-frame *target position*
rather than a fingertip *error*, so no forward kinematics is needed to build
the observation on any backend -- FK differences therefore cannot leak into the
policy input and can be measured separately from the logged trajectories.

To probe a different finger, change ``REACH_FINGER_INDEX``.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp
from isaac_neuromeka.mdp.actions.action_cfgs import CustomResidualJointActionCfg
from . import mdp as hand_grasp_mdp
from .hand_grasp_env_cfg import HandGraspEnvCfg, HandGraspSceneCfg
from .hand_real_mdp import last_applied_action


# --------------------------------------------------------------------------- #
# Canonical contract.  Every backend maps by joint NAME; raw array order is
# never assumed to agree between Isaac, MuJoCo and the real hand.
# --------------------------------------------------------------------------- #
REACH_FINGER_INDEX = 3
REACH_TIP_LINK = f"finger{REACH_FINGER_INDEX}_tip_link"
PALM_BODY_NAME = "palm_link"

# Explicit list, never a regex: ``preserve_order=True`` makes SceneEntityCfg
# resolve in *this* order, which becomes the policy action order.
MIDDLE_JOINT_NAMES = [
    f"finger{REACH_FINGER_INDEX}_joint1",
    f"finger{REACH_FINGER_INDEX}_joint2",
    f"finger{REACH_FINGER_INDEX}_joint3",
    f"finger{REACH_FINGER_INDEX}_joint4",
]
# The sixteen joints the policy does NOT control.  They are held, not free.
OTHER_JOINT_NAMES = [
    f"finger{finger}_joint{joint}"
    for finger in (1, 2, 3, 4, 5)
    for joint in (1, 2, 3, 4)
    if finger != REACH_FINGER_INDEX
]

MIDDLE_JOINTS = SceneEntityCfg(
    "robot",
    joint_names=MIDDLE_JOINT_NAMES,
    preserve_order=True,
)
OTHER_JOINTS = SceneEntityCfg(
    "robot",
    joint_names=OTHER_JOINT_NAMES,
    preserve_order=True,
)
REACH_TIP_CFG = SceneEntityCfg("robot", body_names=[REACH_TIP_LINK])

# Palm-local sampling box for the target.  These are palm-frame metres:
# ``FingerTipReachCommand`` samples ``p_local`` inside this box and only then
# rotates it by the palm quaternion (``commands.py``).
#
# Measured 2026-08-18 by sweeping the four middle-finger joints over the REAL
# hand's factory limits (SDK v1.7.0) with the Joint4 command floor at 0 rad,
# then asking how far a uniformly sampled box point is from the reachable set:
#
#   box [cm]                          within 5 mm   median   p90
#   x 0.0~5.0  y -3.0~3.0  z 3.0~10.0   28.3 %      9.74 mm  22.62 mm  <- previous
#   x 2.5~5.0  y -0.5~2.0  z 8.5~11.5   92.8 %      2.95 mm   4.66 mm  <- selected
#   x 2.0~6.0  y -1.0~2.5  z 6.0~12.0   82.7 %      3.18 mm   6.45 mm
#   x 0.5~6.0  y -1.5~3.0  z 6.5~14.0   58.3 %      4.21 mm  12.34 mm
#
# The previous box was mostly unreachable: with reward exp(-d/0.02) a median
# target capped the achievable reward at 0.61, so the policy was being graded
# on geometry rather than on control.  The selected box brackets the pregrasp
# fingertip (palm [3.63, 0.69, 9.80] cm).  Widen it only after the diagnostic
# trajectory comparison is working.
REACH_RANGE_X = (0.025, 0.050)
REACH_RANGE_Y = (0.010, 0.010)
REACH_RANGE_Z = (0.085, 0.115)

# Park the sticks far away AND apart from each other.  Parking two dynamic
# rigid bodies at an identical pose makes them interpenetrate on the first
# step, which injects contact energy into a scene that should be quiet.
STICK1_PARK_POS = (2.0, 2.0, 0.5)
STICK2_PARK_POS = (2.0, 2.4, 0.5)


def finger_target_position_in_palm(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Return the reach target as palm-frame ``[x, y, z]`` -- the deploy value.

    ``FingerTipReachCommand`` samples ``p_local`` inside the palm-frame box and
    keeps it as ``target_palm``; its public ``command`` property is the
    env-frame world offset instead (``commands.py``: ``target_e = target_w -
    env_origins``).  Reading ``target_palm`` therefore hands the policy the
    literal number a MuJoCo or real-hand operator types in -- no camera, no
    forward kinematics, no frame round-trip.
    """

    return env.command_manager.get_term(command_name).target_palm


@configclass
class FingerReachCommandsCfg:
    finger_target = mdp.FingerTipReachCommandCfg(
        asset_name="robot",
        body_name=REACH_TIP_LINK,
        palm_body_name=PALM_BODY_NAME,
        range_x=REACH_RANGE_X,
        range_y=REACH_RANGE_Y,
        range_z=REACH_RANGE_Z,
        # Longer than the episode: resample on reset only, so one episode is
        # one target and the trajectory is directly comparable across backends.
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
    )


@configclass
class FingerReachActionsCfg:
    """4D middle-finger residual action.

    Deliberately NOT ``HandSettingActionsCfg``: that term drives all twenty
    joints, which defeats a single-finger diagnostic.  Decoding, per
    ``CustomResidualJointPositionAction.process_actions``:

        processed = raw_action * scale
        target    = q_current[middle] + processed
        target    = clamp(target, soft_joint_pos_limits, lower_overrides)
        set_joint_position_target(target, joint_ids=middle_only)

    ``raw_action`` is already wrapper-clipped to [-1, 1] by RSL-RL.
    """

    hand_action = CustomResidualJointActionCfg(
        asset_name="robot",
        joint_names=MIDDLE_JOINT_NAMES,
        preserve_order=True,
        scale=0.1,
        clamp_to_limits=True,
        # No distal floor: the command range is the connected hand's own factory
        # range, so Isaac, MuJoCo and the real hand share one action space.
        #
        # The floor was inherited from the twenty-joint grasp tasks, which have
        # since dropped theirs too -- a joint clamped at exactly 0.000 requests a
        # negative residual every step and delivers 0.000 N.m.  Measured against
        # model_500 it was inert anyway: joint4 never went below +0.186 rad, and
        # MuJoCo trajectories with and without the floor were identical to
        # 0.000 mrad over 120 policy steps.
        joint_position_lower_overrides=None,
    )


@configclass
class FingerReachObservationsCfg:
    """15D deployable policy input.

    obs[0:4]   previous normalized middle joint position
    obs[4:8]   current  normalized middle joint position
    obs[8:11]  target position in the palm frame [x, y, z]
    obs[11:15] last raw middle-finger policy action
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_history = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": MIDDLE_JOINTS},
            history_length=2,
            flatten_history_dim=True,
        )
        target_position_palm = ObsTerm(
            func=finger_target_position_in_palm,
            params={"command_name": "finger_target"},
        )
        # ``action_manager.action`` -- the action that produced this state.
        # ``mdp.action_history`` would be ``prev_action``, one step older.
        last_action = ObsTerm(func=last_applied_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FingerReachRewardsCfg:
    # Fingertip FK is used for the reward and the logs, never for the
    # observation, so a kinematics mismatch stays measurable and separable.
    finger_reach = RewTerm(
        func=hand_grasp_mdp.body_reach_command_tracking,
        weight=10.0,
        params={
            "body_cfg": REACH_TIP_CFG,
            "command_name": "finger_target",
            "sigma": 0.02,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)


@configclass
class FingerReachTerminationsCfg:
    # Time-out only: we want the whole approach -> converge -> hold trajectory.
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class FingerReachEventCfg:
    reset = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # An action term writes position targets for its own joints only, so the
    # other sixteen would drift toward target 0 after a reset.  Pin them to the
    # reset pose ONCE per episode; do not re-issue q_target = q_current every
    # step, which would let drift or an external force redefine the hold.
    hold_other_joints = EventTerm(
        func=mdp.hold_joints_at_default,
        mode="reset",
        params={"asset_cfg": OTHER_JOINTS},
    )


@configclass
class FingerReachEnvCfg(HandGraspEnvCfg):
    """Single-finger reach diagnostic: 15D observation, 4D action."""

    scene: HandGraspSceneCfg = HandGraspSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: FingerReachObservationsCfg = FingerReachObservationsCfg()
    actions: FingerReachActionsCfg = FingerReachActionsCfg()
    commands: FingerReachCommandsCfg = FingerReachCommandsCfg()
    rewards: FingerReachRewardsCfg = FingerReachRewardsCfg()
    terminations: FingerReachTerminationsCfg = FingerReachTerminationsCfg()
    events: FingerReachEventCfg = FingerReachEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 4.0

        self.scene.stick1 = self.scene.stick1.replace(
            init_state=self.scene.stick1.init_state.replace(pos=STICK1_PARK_POS)
        )
        self.scene.stick2 = self.scene.stick2.replace(
            init_state=self.scene.stick2.init_state.replace(pos=STICK2_PARK_POS)
        )
