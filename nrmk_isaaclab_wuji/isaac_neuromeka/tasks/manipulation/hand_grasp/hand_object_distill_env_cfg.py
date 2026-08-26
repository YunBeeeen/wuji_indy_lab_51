"""Isolated 103D-teacher to 105D-student distillation environment.

This task is a compatibility bridge, not a replacement for either parent:

* scene, cube objective and scripted motion come from ``hand_object``;
* the ``teacher`` observation reproduces the successful 2026-08-08 103D
  velocity contract;
* the ``student`` observation is exactly the deployable 105D ``hand_real``
  quaternion-history contract;
* rollout uses the successful run's reset/schedule/disturbance distribution;
* the action term uses today's deployable per-joint residual scales.

The teacher-driven distillation algorithm converts the teacher's old uniform
0.1-rad action into those current action units before stepping this task.  No
existing task configuration is mutated.
"""

from __future__ import annotations

import copy

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
)
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp

from . import hand_object_distill_mdp
from . import hand_object_mdp
from . import hand_real_mdp
from . import mdp as hand_grasp_mdp
from .hand_grasp_env_cfg import (
    FINGERTIPS,
    HAND_JOINT_NAMES,
    HAND_JOINTS,
    PALM,
    PREGRASP_STICK1_QUATERNION_P,
    PREGRASP_STICK2_QUATERNION_P,
    STICK_1,
    STICK_2,
)
from .hand_object_env_cfg import HandObjectEnvCfg
from .hand_real_env_cfg import HAND_REAL_ACTION_SCALE


STUDENT_ACTION_SCALE_RAD = tuple(
    HAND_REAL_ACTION_SCALE[joint_name] for joint_name in HAND_JOINT_NAMES
)
if STUDENT_ACTION_SCALE_RAD != hand_object_distill_mdp.STUDENT_ACTION_SCALE_RAD:
    raise ValueError(
        "distillation student action scale has drifted from hand_real; update the "
        "shared distillation contract before training"
    )

# Exact non-policy environment values in the successful teacher run.  They are
# frozen here because the active hand_object curriculum has since changed.
TEACHER_SUPPORT_RETRACT_DEADLINE_S = 5.5
TEACHER_SUPPORT_RETRACT_DEBOUNCE_STEPS = 5
TEACHER_DISTURBANCE_TIME_RANGE_S = (5.5, 12.0)
TEACHER_DISTURBANCE_DURATION_S = 0.10
TEACHER_DISTURBANCE_FORCE_RANGE_N = (0.3, 0.9)
TEACHER_DISTURBANCE_PROBABILITY = 1.0


@configclass
class HandObjectDistillationObservationsCfg:
    """Two simultaneous actor inputs: deployable 105D student and frozen 103D teacher."""

    @configclass
    class StudentCfg(ObsGroup):
        # Keep this term order byte-for-byte compatible with HandRealObservationsCfg.
        joint_pos_history = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": HAND_JOINTS},
            history_length=2,
            flatten_history_dim=True,
        )
        fingertip_pos = ObsTerm(
            func=hand_grasp_mdp.fingertip_positions_in_palm,
            params={"palm_cfg": PALM, "fingertip_cfg": FINGERTIPS},
        )
        stick1_pose_history = ObsTerm(
            func=hand_real_mdp.canonical_object_pose_in_palm,
            params={
                "palm_cfg": PALM,
                "object_cfg": STICK_1,
                "reference_quaternion_p": PREGRASP_STICK1_QUATERNION_P,
            },
            history_length=2,
            flatten_history_dim=True,
        )
        stick2_pose_history = ObsTerm(
            func=hand_real_mdp.canonical_object_pose_in_palm,
            params={
                "palm_cfg": PALM,
                "object_cfg": STICK_2,
                "reference_quaternion_p": PREGRASP_STICK2_QUATERNION_P,
            },
            history_length=2,
            flatten_history_dim=True,
        )
        last_action = ObsTerm(func=hand_real_mdp.last_applied_action)
        open_close_mode = ObsTerm(
            func=hand_grasp_mdp.open_close_mode,
            params={"command_name": "open_close"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        # Exact saved 103D order: q20, dq20, tips15, stick poses14,
        # stick velocities12, previous action20, mode2.
        joint_pos = ObsTerm(
            func=hand_object_distill_mdp.teacher_joint_pos_limit_normalized,
            params={"asset_cfg": HAND_JOINTS},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": HAND_JOINTS},
            scale=0.2,
        )
        fingertip_pos = ObsTerm(
            func=hand_grasp_mdp.fingertip_positions_in_palm,
            params={"palm_cfg": PALM, "fingertip_cfg": FINGERTIPS},
        )
        stick1_pose = ObsTerm(
            func=hand_grasp_mdp.object_pose_in_palm,
            params={"palm_cfg": PALM, "object_cfg": STICK_1},
        )
        stick2_pose = ObsTerm(
            func=hand_grasp_mdp.object_pose_in_palm,
            params={"palm_cfg": PALM, "object_cfg": STICK_2},
        )
        stick1_velocity = ObsTerm(
            func=hand_grasp_mdp.object_velocity_in_palm,
            params={"palm_cfg": PALM, "object_cfg": STICK_1},
            scale=0.2,
        )
        stick2_velocity = ObsTerm(
            func=hand_grasp_mdp.object_velocity_in_palm,
            params={"palm_cfg": PALM, "object_cfg": STICK_2},
            scale=0.2,
        )
        action_history = ObsTerm(
            func=hand_object_distill_mdp.teacher_previous_action,
            params={"student_action_scale_rad": STUDENT_ACTION_SCALE_RAD},
        )
        open_close_mode = ObsTerm(
            func=hand_grasp_mdp.open_close_mode,
            params={"command_name": "open_close"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    student: StudentCfg = StudentCfg()
    teacher: TeacherCfg = TeacherCfg()


def _successful_teacher_schedule() -> hand_object_mdp.HandObjectScheduleCfg:
    """Return an independent copy of the model_300 rollout schedule."""
    schedule = copy.deepcopy(hand_object_mdp.HAND_OBJECT_SCHEDULE)
    schedule.support_retract_deadline_time_s = TEACHER_SUPPORT_RETRACT_DEADLINE_S
    schedule.validate()
    return schedule


@configclass
class HandObjectDistillationEnvCfg(HandObjectEnvCfg):
    """Teacher-preserving rollout with a deployment-compatible student input."""

    observations: HandObjectDistillationObservationsCfg = (
        HandObjectDistillationObservationsCfg()
    )

    # This is a training bridge only.  A checkpoint produced here is played as
    # hand_real/hand_final after PPO fine-tuning, never under this task ID.
    checkpoint_source_experiment: str | None = None

    def __post_init__(self):
        super().__post_init__()

        # Student action units must already match hand_real.  The runner maps
        # the old teacher output so the actual physical residual remains 0.1*a.
        hand_action = self.actions.hand_action.replace(
            scale=dict(HAND_REAL_ACTION_SCALE),
            joint_position_lower_overrides=None,
        )
        self.actions = self.actions.replace(hand_action=hand_action)

        # Restore the exact successful teacher reset instead of today's shared
        # 4 mm-clearance PREGRASP_JOINT_POSITIONS.
        teacher_pregrasp = hand_object_distill_mdp.TEACHER_PREGRASP_JOINT_POSITIONS
        self.events.reset_pregrasp.params["joint_positions"] = teacher_pregrasp
        self.rewards.joint_reference.params["reference_joint_positions"] = (
            teacher_pregrasp
        )

        # All three command terms and cube-drop termination own independent
        # configclass copies; update every copy so no hidden 7.5 s boundary is
        # left in the 5.5 s teacher rollout.
        teacher_schedule = _successful_teacher_schedule()
        self.commands.open_close.schedule = copy.deepcopy(teacher_schedule)
        self.commands.root_orientation.schedule = copy.deepcopy(teacher_schedule)
        self.commands.support.schedule = copy.deepcopy(teacher_schedule)
        self.commands.support.retract_trigger_debounce_steps = (
            TEACHER_SUPPORT_RETRACT_DEBOUNCE_STEPS
        )
        self.terminations.cube_dropped.params["schedule"] = copy.deepcopy(
            teacher_schedule
        )
        self.episode_length_s = teacher_schedule.episode_length_s

        disturbance = self.events.stick_disturbance
        disturbance.params["time_range_s"] = TEACHER_DISTURBANCE_TIME_RANGE_S
        disturbance.params["duration_s"] = TEACHER_DISTURBANCE_DURATION_S
        disturbance.params["force_range_n"] = TEACHER_DISTURBANCE_FORCE_RANGE_N
        disturbance.params["probability"] = TEACHER_DISTURBANCE_PROBABILITY
