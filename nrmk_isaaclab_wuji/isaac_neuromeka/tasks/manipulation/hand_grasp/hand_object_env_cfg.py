"""Cube pick-and-hold fine-tuned from ``hand_move`` (2026-08-06).

Inheritance::

    HandGraspEnvCfg  ->  HandMoveEnvCfg  ->  HandObjectEnvCfg

The task: starting from the validated functional chopsticks grasp, open the
tips, rotate the floating hand by one calibrated angle so the cube ends up
between them, close on the cube, then have the support column slide away and
keep holding.

One task ID, two ways to run it
-------------------------------
``hand_object`` is registered once and serves training, manual play and
calibration - the same arrangement ``hand_move`` already uses.  The three
scripted command terms (``open_close``, ``root_orientation``, ``support``) each
carry a manual override, and ``play.py --manual_root`` flips all three at once::

    training     : python scripts/rsl_rl/train.py --task hand_object --headless
    play/calib   : python scripts/rsl_rl/play.py  --task hand_object \
                       --num_envs 1 --manual_root env.episode_length_s=300.0

In manual mode the root pose comes from the keyboard instead of the schedule, the
mode is latched by ``1``/``2`` instead of switching at 2.5 s, and the support
only drops when ``V`` is pressed.  That last one matters: with the timed
retract still running, a play session would lose its support 5.5 s in, wherever
the operator happened to have flown the hand.

Policy interface is frozen
--------------------------
103D observation, 20D finger action - identical to ``hand_move``, so its
checkpoint fine-tunes here with no shape change.  The cube, the contact forces
and the phase are visible to rewards, terminations and metrics only.  The actor
never sees them, which is deliberate: the point is to test whether the existing
proprioceptive state is enough to hold an object it cannot observe.

Why the hand and the sticks are not touched
-------------------------------------------
Their spawn, the ``pose_005`` reset and the whole reward set come from
``hand_move`` unchanged, and this file adds no override for any of them.  When
the geometry does not line up, the cube, the support and the yaw are what move
- never the validated grasp.

Calibration must happen before training
---------------------------------------
The spawn pose and the cube are both fixed, so what has to be found is the
**root pose the hand flies to**: where it ends up after the move/rotate window,
with the distal tips straddling the cube.  The constants are ``None`` until
someone measures it; constructing the training config with them unset raises,
because a plausible-looking guess would produce a run whose geometry nobody
ever checked.  The procedure:

1. Start the play/calibration command above and load a ``hand_move`` checkpoint
   (the fingers have to be doing something sensible for the forces to mean
   anything).
2. Fly the hand: ``I/K/J/L/U/O`` translate, ``Q/E`` roll, ``W/S`` pitch,
   ``A/D`` yaw - all in the hand's own frame, the same frame the scripted
   trajectory uses.  ``1`` = OPEN, ``2`` = CLOSE.
3. Press ``P`` repeatedly and drive ``cube - tip midpoint`` towards zero with
   the cube between the two tips.  **Do not move the cube**; it is the fixed
   target.
4. Paste the two lines ``P`` prints into ``hand_object_mdp.py``::

       HAND_OBJECT_TARGET_ROOT_POS_E = (...)
       HAND_OBJECT_TARGET_EULER_RAD  = (...)

   ``HAND_OBJECT_CUBE_POS_E`` / ``..._SUPPORT_POS_E`` are the cube and support
   positions from the same block (they never moved).
5. For the force: CLOSE on the cube, press ``V`` to drop the support, and watch
   ``min(inward1, inward2)``.  Compare attempts that hold against ones that
   drop, and set ``HAND_OBJECT_FORCE_SATURATION_N`` near the bottom of the
   holding range.  Also confirm both inward forces read positive during the
   squeeze - that checks the contact-force sign convention rather than assuming
   it.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaac_neuromeka.mdp as mdp

from . import hand_object_mdp
from . import mdp as hand_grasp_mdp
from .hand_grasp_env_cfg import (
    FUNCTIONAL_CONTACT_GROUPS,
    SMALL_CONTACT_OFFSET,
    SMALL_MAX_DEPENETRATION_VELOCITY,
    SMALL_REST_OFFSET,
    STICK_1,
    STICK_2,
    STICK_TIP_OFFSET_O,
)
from .hand_move_env_cfg import (
    HandMoveActionsCfg,
    HandMoveCommandsCfg,
    HandMoveEnvCfg,
    HandMoveEventCfg,
    HandMoveRewardsCfg,
    HandMoveSceneCfg,
    HandMoveTerminationsCfg,
)
from .hand_move_root_actions import HandRootHoldActionCfg
from .hand_object_mdp import HAND_OBJECT_SCHEDULE

# ---------------------------------------------------------------------------
# Cube and support geometry.  Every position here is **env-local**: the value
# Isaac Lab offsets by each environment's origin when it replicates the scene.
# Storing a world coordinate would put all but env 0 in the wrong place.
# ---------------------------------------------------------------------------
OBJECT_SIZE = (0.01, 0.01, 0.01)
OBJECT_MASS = 0.003
OBJECT_FRICTION = 1.0
OBJECT_RESTITUTION = 0.0

# Gap left between the support's top face and the cube's bottom face, so the
# two start in contact-free proximity rather than interpenetrating.
SUPPORT_TOP_CLEARANCE = 0.0005
# A deliberately thin column: it has to sit under a cube held between two
# chopstick tips 23 mm apart without becoming what they grip.
SUPPORT_CROSS_SECTION = 0.006

CUBE_HALF_HEIGHT = 0.5 * OBJECT_SIZE[2]

# Play-only fallback placement, used while the calibration constants are unset
# so the scene can still be built and flown around.  PROVISIONAL: the reset tip
# midpoint (0.113, 0.022, 0.604) advanced 12 cm along the stick pointing axis,
# which puts the cube in front of the open tips at zero yaw.  This is *not* the
# training placement - that one comes from the calibrated yaw.
PROVISIONAL_CUBE_POS_E = (0.148, 0.083, 0.401)

OBJECT = SceneEntityCfg("object")
OBJECT_SUPPORT = SceneEntityCfg("object_support")

STICK1_CUBE_SENSOR = "stick1_cube_contact"
STICK2_CUBE_SENSOR = "stick2_cube_contact"

# Reproduce the OPEN target used by the successful
# hand_object/2026-08-08_20-39-52 run.  Keep this task-local so restoring
# hand_object does not change hand_move's active 20 mm target.
HAND_OBJECT_OPEN_TARGET_GAP = 0.017


def _cube_position_e() -> tuple[float, float, float]:
    """Calibrated cube centre if measured, else the provisional play position."""
    if hand_object_mdp.HAND_OBJECT_CUBE_POS_E is not None:
        return tuple(hand_object_mdp.HAND_OBJECT_CUBE_POS_E)
    return PROVISIONAL_CUBE_POS_E


def _support_position_e() -> tuple[float, float, float]:
    """Support centre, derived from the cube unless measured separately.

    The derived column runs from the ground plane up to exactly
    :data:`SUPPORT_TOP_CLEARANCE` below the cube's bottom face, which is what
    makes "cube resting on support" true without an initial penetration.
    """
    if hand_object_mdp.HAND_OBJECT_SUPPORT_POS_E is not None:
        return tuple(hand_object_mdp.HAND_OBJECT_SUPPORT_POS_E)
    cube = _cube_position_e()
    top_z = cube[2] - CUBE_HALF_HEIGHT - SUPPORT_TOP_CLEARANCE
    return (cube[0], cube[1], 0.5 * top_z)


def _support_size() -> tuple[float, float, float]:
    """Column cross-section and height, so its top face lands under the cube."""
    support = _support_position_e()
    return (SUPPORT_CROSS_SECTION, SUPPORT_CROSS_SECTION, 2.0 * support[2])


@configclass
class HandObjectSceneCfg(HandMoveSceneCfg):
    """The ``hand_move`` scene plus a supported cube and two cube-only sensors.

    ``robot``, ``stick1``, ``stick2`` and the six functional-grasp contact
    sensors are inherited untouched.
    """

    object_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObjectSupport",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_support_position_e(),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=_support_size(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # Kinematic: the retract moves it by writing poses, and nothing
                # in the scene is allowed to push it around.
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=SMALL_CONTACT_OFFSET,
                rest_offset=SMALL_REST_OFFSET,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=OBJECT_FRICTION,
                dynamic_friction=OBJECT_FRICTION,
                restitution=OBJECT_RESTITUTION,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.35, 0.35, 0.38),
                metallic=0.0,
                roughness=0.8,
            ),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_cube_position_e(),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=OBJECT_SIZE,
            # Required for the two cube-filtered sensors below to report at all.
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

    # One sensor per stick, each filtered to the cube alone.
    #
    # A single sensor on the cube would return one summed vector, and the whole
    # bilateral-force idea needs the two contributions kept apart: "Stick1 is
    # mashing the cube into the support" and "both sticks are squeezing" look
    # identical once the forces are added together.
    #
    # Filtering to the cube also excludes everything else touching a stick -
    # the fingers, the other stick, the support - so what is measured is only
    # the grip on the object.  It does not restrict contact to the *tip*: the
    # whole stick body reports.  That is acceptable while the cube spawns
    # between the tips, and if shaft-pressing shows up in the traces a
    # contact-point gate is the follow-up, not a second rigid body.
    stick1_cube_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Stick1",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
    )
    stick2_cube_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Stick2",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
    )


@configclass
class HandObjectActionsCfg(HandMoveActionsCfg):
    """``hand_move``'s 20D finger action, with the root also translating.

    ``hand_action`` is inherited untouched, so the action vector stays 20D and a
    ``hand_move`` checkpoint still fits.  The only change is that the root
    controller now takes its *position* target from the same command term that
    supplies its attitude, instead of holding the position captured at reset.

    Holding the position is right for ``hand_move``, where the question is
    whether the grasp survives a rotation.  Here the hand has to reach a cube
    at a fixed place, and turning on the spot leaves the tips swinging on an arc
    about the palm that never arrives.
    """

    root_action = HandRootHoldActionCfg(
        asset_name="robot",
        rotation_action_dim=0,
        orientation_command_name="root_orientation",
        # Same term, both targets.  ``HandObjectRootOrientationCommand`` owns the
        # position trajectory as well as the attitude one, so the name is now a
        # misnomer - it is kept because ``hand_move`` binds this term by name and
        # so does ``HandMoveManualRootController``, which both tasks share.
        position_command_name="root_orientation",
    )


@configclass
class HandObjectCommandsCfg(HandMoveCommandsCfg):
    """``hand_move``'s two scripted commands, retimed, plus the support column.

    The term *names* ``open_close`` and ``root_orientation`` are kept, because
    the observation term, every reward and the success termination bind to them
    by name.  Only the schedules behind them change.
    """

    open_close = hand_object_mdp.HandObjectOpenCloseCommandCfg()
    root_orientation = hand_object_mdp.HandObjectRootOrientationCommandCfg()
    # The support term recomputes the same inward forces the force reward uses,
    # so it needs the same entities.  Passed explicitly rather than defaulted
    # inside the MDP module: the stick names, the tip offset and the sensor
    # names are defined here, and a second copy of them would be free to drift.
    support = hand_object_mdp.HandObjectSupportCommandCfg(
        stick1_cfg=STICK_1,
        stick2_cfg=STICK_2,
        tip_offset_o=STICK_TIP_OFFSET_O,
        stick1_sensor_name=STICK1_CUBE_SENSOR,
        stick2_sensor_name=STICK2_CUBE_SENSOR,
        force_saturation=hand_object_mdp.HAND_OBJECT_FORCE_SATURATION_N,
    )


@configclass
class HandObjectRewardsCfg(HandMoveRewardsCfg):
    """Every ``hand_move`` reward, unchanged, plus four cube terms.

    Nothing inherited is re-weighted or gated differently.  In particular
    ``close_tip_gap`` keeps ``target_gap = 0`` and does **not** fade out as
    contact force appears: "close the tips" and "squeeze the cube" are separate
    instructions here, and coupling them at this stage would hide which of the
    two the policy failed at.

    All four are on by default.  To run the curriculum's first stage instead -
    inherited hand_grasp rewards only, while the six contacts and OPEN/CLOSE
    settle at the new goal pose - zero them from the command line::

        train.py --task hand_object --headless \
            --init_checkpoint <hand_move .pt> \
            env.rewards.bilateral_cube_contact.weight=0.0 \
            env.rewards.bilateral_cube_force.weight=0.0 \
            env.rewards.cube_relative_stability.weight=0.0 \
            env.rewards.cube_hold.weight=0.0

    A zero weight is not merely a small reward: the manager skips the term
    outright (``reward_manager.py:146``, and the same check in this project's
    ``managers.py:1947``), so it costs nothing and behaves exactly like
    commenting the term out.  The cube metrics keep reporting either way - they
    are computed from the sensors, not from these rewards - so stage 1 still
    shows what the contact forces are doing.  Hydra needs the decimal point; an
    int raises.

    Weights are PROVISIONAL.  Budget, remembering that the
    manager scales by ``dt`` so a term held at 1.0 for ``T`` seconds is worth
    ``weight x T``:

        bilateral_cube_contact   15 x 4.5 s (CLOSE window)     =   67.5
        bilateral_cube_force     50 x 4.5 s (CLOSE window)     =  225
        cube_relative_stability  50 x 3.0 s (after retract)    =  150
        cube_hold               200 x 3.0 s (after retract)    =  600

    The acquisition term is deliberately not multiplied by the weakest-six-
    contact gate; it must remain available while the policy discovers contact.
    The other cube objectives retain that strict gate.  For scale, the inherited
    ``mode_grasp_stability`` is worth about 350 over the same episode.  The hold
    reward is intentionally the largest new term, but it remains unreachable by
    sacrificing the functional grasp.
    """

    bilateral_cube_contact = RewTerm(
        func=hand_object_mdp.bilateral_cube_contact_acquisition,
        weight=15.0,
        params={
            "command_name": "open_close",
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "tip_offset_o": STICK_TIP_OFFSET_O,
            "stick1_sensor_name": STICK1_CUBE_SENSOR,
            "stick2_sensor_name": STICK2_CUBE_SENSOR,
            "force_saturation": (
                hand_object_mdp.HAND_OBJECT_CONTACT_ACQUISITION_SATURATION_N
            ),
        },
    )

    bilateral_cube_force = RewTerm(
        func=hand_object_mdp.bilateral_cube_force,
        weight=50.0,
        params={
            "command_name": "open_close",
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "cube_cfg": OBJECT,
            "tip_offset_o": STICK_TIP_OFFSET_O,
            "stick1_sensor_name": STICK1_CUBE_SENSOR,
            "stick2_sensor_name": STICK2_CUBE_SENSOR,
            "functional_sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "force_saturation": hand_object_mdp.HAND_OBJECT_FORCE_SATURATION_N,
            "functional_force_scale": (
                hand_object_mdp.HAND_OBJECT_FUNCTIONAL_FORCE_SCALE_N
            ),
        },
    )
    cube_relative_stability = RewTerm(
        func=hand_object_mdp.cube_relative_stability,
        weight=50.0,
        params={
            "support_command_name": "support",
            "cube_cfg": OBJECT,
            "stick2_cfg": STICK_2,
            "tip_offset_o": STICK_TIP_OFFSET_O,
            "functional_sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "functional_force_scale": (
                hand_object_mdp.HAND_OBJECT_FUNCTIONAL_FORCE_SCALE_N
            ),
        },
    )
    cube_hold = RewTerm(
        func=hand_object_mdp.cube_hold,
        weight=200.0,
        params={
            "command_name": "open_close",
            "support_command_name": "support",
            "stick1_cfg": STICK_1,
            "stick2_cfg": STICK_2,
            "cube_cfg": OBJECT,
            "tip_offset_o": STICK_TIP_OFFSET_O,
            "stick1_sensor_name": STICK1_CUBE_SENSOR,
            "stick2_sensor_name": STICK2_CUBE_SENSOR,
            "functional_sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
            "force_saturation": hand_object_mdp.HAND_OBJECT_FORCE_SATURATION_N,
            "functional_force_scale": (
                hand_object_mdp.HAND_OBJECT_FUNCTIONAL_FORCE_SCALE_N
            ),
        },
    )


@configclass
class HandObjectTerminationsCfg(HandMoveTerminationsCfg):
    """``hand_move``'s terminations plus a debounced cube-drop failure.

    ``time_out`` stays False, so this is a genuine failure and bootstrapping is
    cut - which is the entire penalty for dropping.  No negative drop reward is
    added: a large one is the usual way to teach a policy never to attempt the
    grasp at all.
    """

    # Disabled to reproduce hand_object/2026-08-08_20-39-52: transient loss of
    # a functional contact must leave time for policy recovery.  Explicit None
    # is required because deleting this override would inherit the active
    # HandMoveTerminationsCfg term.
    functional_contact_lost = None

    # Previous strict termination, retained for an easy rollback:
    # functional_contact_lost = DoneTerm(
    #     func=hand_grasp_mdp.FunctionalContactLoss,
    #     params={
    #         "sensor_groups": FUNCTIONAL_CONTACT_GROUPS,
    #         "acquire_threshold": 0.10,
    #         "release_threshold": 0.05,
    #         "acquire_hold_steps": 5,
    #         "minimum_retained_contacts": 6,
    #         "loss_hold_steps": 15,
    #     },
    # )

    # 2026-08-07: 상속받은 0.40 m 로는 hand_object 가 아예 학습되지 않는다.
    #
    # ``hand_move`` 는 손이 스폰 높이에 머무르므로 "스틱 루트가 0.40 아래 = 손에서
    # 빠짐"이 성립하지만, 이 태스크는 큐브를 잡으러 **일부러 z = 0.365 까지 내려간다**.
    # 하강이 끝나는 2.0 s 에 스틱 루트가 0.40 을 지나면서 전 에피소드가 오판 종료됐다
    # (run 12-45-12: Episode_Termination/stick2_dropped = 1.0000,
    #  mean_episode_length 59.96 steps = 정확히 2.00 s).  CLOSE 는 2.5 s 시작이라
    # 압착을 한 번도 시도하지 못했고, 그래서 큐브 보상 3종과 holding 이 전부 0 이었다.
    # "큐브에 안 닿는다"로 보였던 증상의 실제 원인이 이것이다.
    #
    # 0.20 m 는 하강 목표(0.365)·큐브(0.411)·기둥 상단(0.409) 어느 것보다 한참 아래고,
    # 손에서 빠진 스틱은 테이블/바닥으로 떨어지므로 진짜 낙하는 그대로 잡힌다.
    stick1_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.20, "asset_cfg": STICK_1},
    )
    stick2_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.20, "asset_cfg": STICK_2},
    )
    cube_dropped = DoneTerm(
        func=hand_object_mdp.CubeDropped,
        params={
            "command_name": "open_close",
            "support_command_name": "support",
            "cube_cfg": OBJECT,
            "stick2_cfg": STICK_2,
            "tip_offset_o": STICK_TIP_OFFSET_O,
            "schedule": HAND_OBJECT_SCHEDULE,
        },
    )


@configclass
class HandObjectEventCfg(HandMoveEventCfg):
    """Identical to ``hand_move``; no cube-specific reset term is needed.

    The inherited ``reset_all`` is ``mdp.reset_scene_to_default``, which
    restores *every* rigid object - cube and support included - to its
    ``init_state`` with zero velocity, and ``reset_pregrasp`` then re-establishes
    the functional grasp.  The support's retract bookkeeping follows in
    ``HandObjectSupportCommand._resample_command``.

    ``hand_grasp_mdp.reset_object_between_stick_tips`` is deliberately unused:
    dropping the cube straight into the tip gap would skip the approach this
    task exists to learn.
    """


@configclass
class HandObjectEnvCfg(HandMoveEnvCfg):
    """Cube pick-and-hold, fine-tuned from a ``hand_move`` checkpoint."""

    scene: HandObjectSceneCfg = HandObjectSceneCfg(num_envs=4096, env_spacing=1.0)
    actions: HandObjectActionsCfg = HandObjectActionsCfg()
    commands: HandObjectCommandsCfg = HandObjectCommandsCfg()
    rewards: HandObjectRewardsCfg = HandObjectRewardsCfg()
    terminations: HandObjectTerminationsCfg = HandObjectTerminationsCfg()
    events: HandObjectEventCfg = HandObjectEventCfg()

    # Refuse to build the scripted task on unmeasured geometry.  Set to False
    # for a play/calibration session, where the operator supplies the pose:
    #     play.py ... env.require_calibration=false
    #
    # The check itself lives in ``HandObjectRootOrientationCommand.__init__``,
    # not here.  ``hydra_task_config`` instantiates this config - running
    # ``__post_init__`` - *before* it applies any command-line override
    # (``isaaclab_tasks/utils/hydra.py:80`` vs ``:90``), so raising here would
    # kill the process before ``env.require_calibration=false`` could be read
    # and the calibration session could never start.  Command terms are built
    # during ``gym.make``, which does see the overridden config.
    require_calibration: bool = True

    # Where ``--load_run`` should look when this task has no runs of its own.
    #
    # hand_object logs under its own experiment name so its runs never mix with
    # hand_move's, but that also means ``logs/rsl_rl/hand_object/`` does not
    # exist until the first training run - and until then the only checkpoints
    # worth playing are hand_move's.  play.py falls back to this name in that
    # case, so ``--load_run <date>`` keeps working with no extra flags.
    checkpoint_source_experiment: str | None = "hand_move"

    def __post_init__(self):
        super().__post_init__()
        HAND_OBJECT_SCHEDULE.validate()
        # hand_move currently owns a 20 mm OPEN target.  Override it locally
        # with hand_object's successful 2026-08-08 value (17 mm), while keeping
        # this task's 0.9 N disturbance ceiling.
        open_target_gap = HAND_OBJECT_OPEN_TARGET_GAP
        self.rewards.open_tip_gap.params["target_gap"] = open_target_gap
        self.rewards.mode_grasp_stability.params["open_target_gap"] = open_target_gap
        self.rewards.success.params["open_target_gap"] = open_target_gap
        self.terminations.success.params["open_target_gap"] = open_target_gap
        # Play-only subclasses deliberately remove this event.  Their config is
        # already installed when this parent post-init runs, so only tune the
        # training disturbance when the term is actually present.
        if self.events.stick_disturbance is not None:
            self.events.stick_disturbance.params["force_range_n"] = (0.3, 0.9)
        # The parent takes its episode length from HAND_MOVE_SCHEDULE (15 s);
        # this task's script is 7 s.
        self.episode_length_s = HAND_OBJECT_SCHEDULE.episode_length_s

        # Frame the cube, the tips and the support top together.
        cube = _cube_position_e()
        self.viewer.eye = (cube[0] + 0.45, cube[1] - 0.30, cube[2] + 0.22)
        self.viewer.lookat = (cube[0], cube[1], cube[2])


# Backwards-compatible aliases: the previous class names still resolve, so any
# existing import or registration keeps working.  Same objects, not copies.
HandObjectGraspSceneCfg = HandObjectSceneCfg
HandObjectGraspEventCfg = HandObjectEventCfg
HandObjectGraspEnvCfg = HandObjectEnvCfg
