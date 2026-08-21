"""Final 105D sim-to-real policy with the hand_object cube objective.

``hand_final`` preserves the deployable ``hand_real`` actor contract while
borrowing the calibrated cube scene, scripted root translation, support
retraction, cube rewards and cube-drop termination from ``hand_object``.  The
cube and its forces remain privileged reward/metric inputs and are not added to
the actor observation.
"""

from isaaclab.utils import configclass

from . import hand_object_mdp
from .hand_object_env_cfg import HandObjectCommandsCfg, HandObjectEnvCfg
from .hand_real_env_cfg import (
    HAND_REAL_POLICY_OBS_DIM,
    HandRealObservationsCfg,
    apply_hand_real_contract,
)


# Keep this task's support top slightly narrower than the 5 mm cube while
# preserving the calibrated support height and centre used by hand_object.
HAND_FINAL_SUPPORT_CROSS_SECTION = 0.006

# Push the scripted goal 7 mm further along the hand's own +z, i.e. the
# direction the fingers extend (world +x at reset; (0.936, 0.339, 0.093) once
# the calibrated goal rotation is applied, so it is mostly world +x there).
#
# Body frame on purpose.  ``tip_target_offset_w`` next to it is world-frame and
# stays at hand_object's 2026-08-08 value; "reach further the way the fingers
# point" is not a world-axis statement, so it needs its own knob.  The cube and
# the support do not move - only the root pose the trajectory flies to.
HAND_FINAL_GOAL_ROOT_OFFSET_R = (0.0, 0.003, 0.007)


@configclass
class HandFinalCommandsCfg(HandObjectCommandsCfg):
    """``hand_object``'s commands with the goal root pushed along hand +z.

    ``root_orientation`` is re-declared rather than patched in
    :meth:`HandFinalEnvCfg.__post_init__`.  ``configclass`` runs a user
    ``__post_init__`` *before* it deepcopies members
    (``configclass.py:93``), so mutating an inherited command term there would
    reach the object ``HandObjectEnvCfg`` still holds as its own field default
    and silently move hand_object's goal too.  Declaring the field here gives
    hand_final its own instance.
    """

    root_orientation = hand_object_mdp.HandObjectRootOrientationCommandCfg(
        goal_root_offset_r=HAND_FINAL_GOAL_ROOT_OFFSET_R,
    )


@configclass
class HandFinalEnvCfg(HandObjectEnvCfg):
    """Cube pick-and-hold fine-tuning for a 105D ``hand_real`` checkpoint."""

    observations: HandRealObservationsCfg = HandRealObservationsCfg()
    commands: HandFinalCommandsCfg = HandFinalCommandsCfg()

    # play.py may use this namespace as its checkpoint fallback before the
    # first hand_final run exists.  Training uses --init_checkpoint explicitly.
    checkpoint_source_experiment: str | None = "hand_real"

    def __post_init__(self):
        super().__post_init__()
        apply_hand_real_contract(self)

        # Task-local geometry A/B: hand_object keeps its original 6 x 6 mm
        # column, whereas hand_final supports the 5 mm cube on a 4 x 4 mm top.
        # Only x/y change; z retains the calibrated top height below the cube.
        support_size = self.scene.object_support.spawn.size
        self.scene.object_support.spawn.size = (
            HAND_FINAL_SUPPORT_CROSS_SECTION,
            HAND_FINAL_SUPPORT_CROSS_SECTION,
            support_size[2],
        )

        # Keep the interface contract explicit near the combined task.  The
        # dimension is derived from HandRealObservationsCfg; this constant is a
        # documentation/checkpoint compatibility assertion for maintainers.
        if HAND_REAL_POLICY_OBS_DIM != 105:
            raise ValueError(
                "hand_final requires the 105D hand_real observation contract"
            )
