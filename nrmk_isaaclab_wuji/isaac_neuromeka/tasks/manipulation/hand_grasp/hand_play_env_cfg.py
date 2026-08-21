"""Keyboard-driven play scene: a table, two plates, one cube (2026-08-07).

Inheritance::

    HandGraspEnvCfg -> HandMoveEnvCfg -> HandObjectEnvCfg -> HandPlayEnvCfg

``hand_play`` exists to fly a trained ``hand_object`` policy around a scene that
looks like the real task instead of the training rig.  The retracting support
column is gone; in its place there is a 40 cm table with two flat plates on it
and the 5 mm cube sitting in the left one.

**Manual control only.**  There is no scripted trajectory here and training this
task is refused - see ``require_manual_root`` below for why.

    python scripts/rsl_rl/play.py --task hand_play --num_envs 1 --manual_root \
        --load_experiment hand_object --load_run <run> env.episode_length_s=300.0

Policy interface is untouched
-----------------------------
103D observation, 20D finger action, the same six functional-contact sensors and
the same two cube-filtered stick sensors.  A ``hand_object`` checkpoint loads
with no shape change - the scene changed, the policy's view of it did not.

Height budget, and the one real conflict
----------------------------------------
The training geometry has the palm dropping to ``z = 0.365`` to reach a cube on
a 6 mm column.  A table cannot allow that: its top is at 0.404, so the
calibrated goal pose is *below the tabletop*.  Nothing here tries to reconcile
the two.  The hand spawns at ``z = 0.50``, comfortably above the table, and the
operator flies it down from there; the scripted trajectory is simply not used.

The table height is chosen so the cube ends up exactly where the policy was
trained to expect it::

    table top    0.4039
    + plate      0.0050   ->  plate top 0.4089
    + cube half  0.0025   ->  cube centre 0.4114   = HAND_OBJECT_CUBE_POS_E[2]

so the left plate reproduces the training cube position to the millimetre.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from . import hand_object_mdp
from .hand_object_env_cfg import (
    CUBE_HALF_HEIGHT,
    OBJECT_FRICTION,
    OBJECT_MASS,
    OBJECT_RESTITUTION,
    OBJECT_SIZE,
    SMALL_CONTACT_OFFSET,
    SMALL_MAX_DEPENETRATION_VELOCITY,
    SMALL_REST_OFFSET,
    HandObjectCommandsCfg,
    HandObjectEnvCfg,
    HandObjectEventCfg,
    HandObjectRewardsCfg,
    HandObjectSceneCfg,
    HandObjectTerminationsCfg,
)
from .hand_real_env_cfg import (
    HAND_REAL_POLICY_OBS_DIM,
    HandRealObservationsCfg,
    apply_hand_real_contract,
)

# ---------------------------------------------------------------------------
# Furniture geometry.  All env-local, like every other position in this package.
# ---------------------------------------------------------------------------

# Where the two plate centres sit.  The left one is placed on the calibrated
# cube position so a loaded policy meets the object exactly where it learnt to,
# and the right one mirrors it through the hand's y = 0 plane.
PLATE_X = 0.1468
PLATE_Y = 0.0865

PLATE_RADIUS = 0.040          # 8 cm across
PLATE_THICKNESS = 0.005       # low enough not to foul the tips at a 17 mm gap
PLATE_COLOR = (0.55, 0.55, 0.58)

# Derived so the cube lands on the calibrated height; see the module docstring.
TABLE_TOP_Z = 0.4114 - CUBE_HALF_HEIGHT - PLATE_THICKNESS
TABLE_TOP_THICKNESS = 0.030
TABLE_TOP_SIZE = (0.60, 0.50, TABLE_TOP_THICKNESS)
# Pushed forward of the hand: the hand is at x = 0 and the plates at x = 0.147,
# so the top spans x -0.10 .. 0.50 and leaves the operator room behind the hand.
TABLE_CENTER_XY = (0.20, 0.0)

TABLE_LEG_SIZE = 0.040
TABLE_LEG_INSET = 0.060       # from the top's edge, so the legs read as legs
TABLE_COLOR = (0.16, 0.10, 0.05)

_TABLE_TOP_CENTER_Z = TABLE_TOP_Z - 0.5 * TABLE_TOP_THICKNESS
_LEG_HEIGHT = TABLE_TOP_Z - TABLE_TOP_THICKNESS      # ground to the underside
_PLATE_CENTER_Z = TABLE_TOP_Z + 0.5 * PLATE_THICKNESS
CUBE_POS_E = (PLATE_X, PLATE_Y, TABLE_TOP_Z + PLATE_THICKNESS + CUBE_HALF_HEIGHT)


def _static_box(
    name: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    """A collidable, immovable box.

    ``AssetBaseCfg`` with no ``rigid_props`` gives a plain static collider - no
    rigid body, no articulation, nothing for PhysX to integrate.  That is what
    furniture should be.  The support column had to be a kinematic
    ``RigidObjectCfg`` only because the retract wrote poses into it every step;
    nothing moves the table.
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=OBJECT_FRICTION,
                dynamic_friction=OBJECT_FRICTION,
                restitution=OBJECT_RESTITUTION,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, metallic=0.0, roughness=0.8
            ),
        ),
    )


def _plate(name: str, y: float) -> AssetBaseCfg:
    """One flat grey disc, static like the table."""
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(PLATE_X, y, _PLATE_CENTER_Z)),
        spawn=sim_utils.CylinderCfg(
            radius=PLATE_RADIUS,
            height=PLATE_THICKNESS,
            axis="Z",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=OBJECT_FRICTION,
                dynamic_friction=OBJECT_FRICTION,
                restitution=OBJECT_RESTITUTION,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=PLATE_COLOR, metallic=0.0, roughness=0.6
            ),
        ),
    )


def _leg(index: int, sign_x: float, sign_y: float) -> AssetBaseCfg:
    """One leg, sitting under a corner of the top and standing on the ground."""
    half_x = 0.5 * TABLE_TOP_SIZE[0] - TABLE_LEG_INSET
    half_y = 0.5 * TABLE_TOP_SIZE[1] - TABLE_LEG_INSET
    return _static_box(
        f"TableLeg{index}",
        (TABLE_LEG_SIZE, TABLE_LEG_SIZE, _LEG_HEIGHT),
        (
            TABLE_CENTER_XY[0] + sign_x * half_x,
            TABLE_CENTER_XY[1] + sign_y * half_y,
            0.5 * _LEG_HEIGHT,
        ),
        TABLE_COLOR,
    )


@configclass
class HandPlaySceneCfg(HandObjectSceneCfg):
    """``hand_object``'s scene with the column swapped for table + plates.

    ``object_support = None`` removes the inherited prim outright -
    ``InteractiveScene._add_entities_from_cfg`` skips ``None`` entries
    (``interactive_scene.py:732``).  The command term, the two support-gated
    rewards and the drop termination that referenced it are removed alongside,
    below; leaving any of them would fail at construction on ``scene[...]``.
    """

    object_support = None

    table_top = _static_box(
        "TableTop",
        TABLE_TOP_SIZE,
        (TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], _TABLE_TOP_CENTER_Z),
        TABLE_COLOR,
    )
    table_leg_1 = _leg(1, +1.0, +1.0)
    table_leg_2 = _leg(2, +1.0, -1.0)
    table_leg_3 = _leg(3, -1.0, +1.0)
    table_leg_4 = _leg(4, -1.0, -1.0)

    plate_left = _plate("PlateLeft", +PLATE_Y)
    plate_right = _plate("PlateRight", -PLATE_Y)

    # The same cube, re-seated on the left plate.
    #
    # Declared in full rather than derived from the parent: ``configclass``
    # turns these into dataclass fields, so ``HandObjectSceneCfg.object`` is an
    # ``AttributeError`` and ``.replace()`` on it never gets the chance to run.
    # Every property below is copied verbatim from ``HandObjectSceneCfg`` -
    # ``activate_contact_sensors`` in particular, without which the two
    # cube-filtered stick sensors report nothing at all.
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CUBE_POS_E,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_SIZE,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=SMALL_MAX_DEPENETRATION_VELOCITY,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=SMALL_CONTACT_OFFSET,
                rest_offset=SMALL_REST_OFFSET,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=OBJECT_FRICTION,
                dynamic_friction=OBJECT_FRICTION,
                restitution=OBJECT_RESTITUTION,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.42, 0.95),
                metallic=0.0,
                roughness=0.5,
            ),
        ),
    )


@configclass
class HandPlayCommandsCfg(HandObjectCommandsCfg):
    """Only the two commands the manual controller needs.

    ``open_close`` (latched by 1/2) and ``root_orientation`` (the keyboard writes
    into its buffers) stay.  ``support`` goes: its ``__init__`` resolves
    ``scene["object_support"]``, which no longer exists.
    """

    support = None


@configclass
class HandPlayRewardsCfg(HandObjectRewardsCfg):
    """Grasp rewards only; the two support-gated cube rewards are dropped.

    Nothing is optimised here - this is a play scene - but every term is still
    evaluated and logged each step, so a term holding a stale
    ``support_command_name`` would raise on the first step rather than quietly
    read zero.
    """

    cube_relative_stability = None
    cube_hold = None


@configclass
class HandPlayTerminationsCfg(HandObjectTerminationsCfg):
    """No cube-drop failure: with no column to retract, it can never fire.

    ``CubeDropped`` requires the support to be fully retracted before it will
    look at the geometry, and it reads the support command term to find that
    out.  Both are gone.  A cube knocked off the plate simply stays on the
    floor until the operator presses ``C``.
    """

    cube_dropped = None


@configclass
class HandPlayEventCfg(HandObjectEventCfg):
    """Manual play keeps deterministic physics by disabling stick disturbances."""

    stick_disturbance = None


@configclass
class HandPlayEnvCfg(HandObjectEnvCfg):
    """Manual-only play environment.  Not trainable, deliberately."""

    scene: HandPlaySceneCfg = HandPlaySceneCfg(num_envs=1, env_spacing=2.0)
    commands: HandPlayCommandsCfg = HandPlayCommandsCfg()
    rewards: HandPlayRewardsCfg = HandPlayRewardsCfg()
    terminations: HandPlayTerminationsCfg = HandPlayTerminationsCfg()
    events: HandPlayEventCfg = HandPlayEventCfg()

    # Refuse to run without keyboard control.
    #
    # The scripted trajectory flies the palm to z = 0.365, which is 39 mm below
    # this table's top surface - it would drive the hand straight through the
    # furniture.  Rather than invent an unvalidated goal pose, the whole
    # scripted path is disabled and the operator drives.  ``play.py`` and
    # ``train.py`` both check this flag and stop with an explanation.
    require_manual_root: bool = True

    # ``--load_run`` alone then resolves against the hand_object folder, so the
    # operator does not have to remember ``--load_experiment`` every session.
    checkpoint_source_experiment: str = "hand_object"

    # hand_object was trained from the reset grasp under OPEN for the first
    # 2.5 s.  Starting manual play in CLOSE instead asks the policy to close in
    # free space before it has approached the cube and can immediately disturb
    # an otherwise valid six-contact reset grasp.
    manual_root_initial_mode_index: int = 0  # OPEN

    def __post_init__(self):
        super().__post_init__()
        # Long enough that a session is not interrupted by a reset; the operator
        # resets with ``R`` when they want one.
        self.episode_length_s = 300.0


@configclass
class HandFinalPlayEnvCfg(HandPlayEnvCfg):
    """Play a 105D ``hand_final`` policy in the table-and-plates scene.

    ``hand_play`` remains the checkpoint-compatible 103D view for legacy
    ``hand_object`` policies.  This sibling changes only the deployable policy
    observation and task-local real-hand action/actuator contract; furniture, cube,
    keyboard control and the play-only termination behavior are inherited
    unchanged.
    """

    observations: HandRealObservationsCfg = HandRealObservationsCfg()
    checkpoint_source_experiment: str = "hand_final"

    def __post_init__(self):
        super().__post_init__()
        apply_hand_real_contract(self)
        if HAND_REAL_POLICY_OBS_DIM != 105:
            raise ValueError(
                "hand_final_play requires the 105D hand_real observation contract"
            )
