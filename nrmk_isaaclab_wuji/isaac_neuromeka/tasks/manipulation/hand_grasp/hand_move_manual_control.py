"""Keyboard control of the ``hand_move`` floating root, for play only.

The one invariant this module exists to preserve:

    **The keyboard never writes a root pose.**  It only moves the *targets* of
    the existing root PD controller.  The hand is still driven by
    ``HandRootHoldAction`` through PhysX external wrenches, every physics step,
    exactly as during training.  Nothing here writes a *hand* root pose or
    velocity.  (:meth:`~HandMoveManualRootController.reset_object` does write a
    pose, but to the free cube of ``hand_object`` - a body with no controller
    and no such invariant.  See its docstring.)

The second invariant:

    **The trained finger policy keeps running.**  This module produces no
    finger action at all.  Keys ``1``/``2`` only choose the OPEN/CLOSE command
    that goes into the policy's 103D observation; the policy decides what the
    fingers do, while the operator moves the wrist.

Why the scripted trajectory has to be switched off rather than overwritten
-------------------------------------------------------------------------
``HandRootHoldAction`` re-reads ``HandMoveRootOrientationCommand.target_quat_w``
on *every physics step*, and the command term recomputes that buffer from the
episode clock at the end of *every* env step.  So calling
``set_target_orientation`` from a play loop would be undone within one step.
Both command terms therefore get an explicit manual-override flag that disables
their scripted update; the operator then owns the buffers the controller reads.

This module is imported by ``scripts/rsl_rl/play.py`` only when
``--manual_root`` is passed, and it imports ``carb``/``omni`` lazily inside
:meth:`HandMoveManualRootController.attach` so that headless training never
touches them.
"""

from __future__ import annotations

import math
import queue

import torch

# ---------------------------------------------------------------------------
# Key map.  Edit here; nothing below hard-codes a key name.
#
# Avoid Q/W/E/R because the viewport also uses them for its selection and
# transform tools.  Arrow keys are more likely to be intercepted, so
# translation is bound to BOTH the arrows and an IJKL/UO block - whichever
# survives on a given setup will work.
# ---------------------------------------------------------------------------
TRANSLATION_KEYS: dict[str, tuple[int, float]] = {
    # key name        (world axis, sign)
    "UP": (0, +1.0),
    "DOWN": (0, -1.0),
    "LEFT": (1, +1.0),
    "RIGHT": (1, -1.0),
    "PAGE_UP": (2, +1.0),
    "PAGE_DOWN": (2, -1.0),
    # Arrow-free duplicates
    "I": (0, +1.0),
    "K": (0, -1.0),
    "J": (1, +1.0),
    "L": (1, -1.0),
    "U": (2, +1.0),
    "O": (2, -1.0),
}

ROTATION_KEYS: dict[str, tuple[int, float]] = {
    # key name   (palm-local axis, sign);  x = roll, y = pitch, z = yaw
    "A": (0, +1.0),
    "S": (0, -1.0),
    "D": (1, +1.0),
    "Z": (1, -1.0),
    "X": (2, +1.0),
    "C": (2, -1.0),
}

MODE_KEYS: dict[str, int] = {"KEY_1": 0, "KEY_2": 1}  # 0 = OPEN, 1 = CLOSE
SYNC_KEY = "H"          # snap the target to the actual current root pose
RESTORE_KEY = "G"       # restore the target to this episode's reset pose
ENV_RESET_KEY = "R"     # full env reset (hand, sticks and object all respawn)
OBJECT_RESET_KEY = "B"  # respawn only the object, leaving the hand where it is
SUPPORT_KEY = "V"       # hand_object: start lowering the cube's support column
CALIBRATION_KEY = "P"   # print the geometry/calibration readout for this task
RECORD_KEY = "M"        # start/stop recording measured hand joint angles

# The scene entity the ``C`` key respawns.  Absent in plain ``hand_move``, in
# which case the key is simply reported as unavailable.
OBJECT_ASSET_NAME = "object"
# The command term the ``V`` key drives.  Only ``hand_object`` has one.
SUPPORT_COMMAND_NAME = "support"
SUPPORT_ASSET_NAME = "object_support"


def normalize_carb_name(value) -> str:
    """Return a bare key/event name from whatever carb handed the callback.

    ``isaaclab.devices.keyboard`` reads ``event.input.name`` directly, which
    assumes carb delivers ``KeyboardInput``/``KeyboardEventType`` enums.  This
    Kit build hands the callback plain **strings** for at least some events, so
    ``.name`` raises ``AttributeError`` - and because the exception happens
    inside a carb subscription it is swallowed and re-printed on every single
    keystroke instead of failing once.

    Accept both shapes, and strip any ``KeyboardInput.``/``KeyboardEventType.``
    prefix that ``str()`` on an enum would leave behind.
    """
    name = getattr(value, "name", None)
    if name is None:
        name = str(value)
    name = name.rsplit(".", 1)[-1].upper()
    # A bare digit means this build reports "1"/"2" where the enum is KEY_1/KEY_2.
    if len(name) == 1 and name.isdigit():
        name = f"KEY_{name}"
    return name


class HandMoveManualRootController:
    """Drive the ``hand_move`` root PD targets from the keyboard."""

    def __init__(
        self,
        env,
        translation_speed: float = 0.1,          # m/s
        rotation_speed: float = math.radians(30.0),  # rad/s
        max_translation_from_start: float = 0.30,  # m
        initial_mode_index: int = 1,               # CLOSE
        print_interval_s: float = 1.0,
    ):
        self._env = env
        self.translation_speed = translation_speed
        self.rotation_speed = rotation_speed
        self.max_translation_from_start = max_translation_from_start
        self.print_interval_s = print_interval_s

        self._root_action = env.action_manager.get_term("root_action")
        self._orientation_term = env.command_manager.get_term("root_orientation")
        self._open_close_term = env.command_manager.get_term("open_close")
        # Only ``hand_object`` has a support column; plain ``hand_move`` does
        # not, and the V/P keys report themselves as unavailable there.
        self._support_term = None
        if SUPPORT_COMMAND_NAME in env.command_manager.active_terms:
            self._support_term = env.command_manager.get_term(SUPPORT_COMMAND_NAME)
        # Set by whoever constructs the controller; a zero-argument callable
        # returning the text that the ``P`` key prints.  Left None for plain
        # ``hand_move``, where there is nothing task-specific to report.
        self.calibration_reporter = None
        # Set by whoever constructs the controller; a zero-argument callable
        # that flips joint recording on/off for the ``M`` key.  Left None when
        # the session was started without a recorder, in which case ``M``
        # reports itself as unavailable rather than doing nothing silently.
        self.record_toggle = None

        self._device = env.device
        self._pressed: set[str] = set()
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self._input_interface = None
        self._keyboard = None
        self._subscription = None
        self._elapsed_since_print = 0.0
        self._mode_index = initial_mode_index
        self._mode_dirty = False
        # Set by the ``R`` key, consumed by the play loop, which is the only
        # place that may legally call ``env.reset()``.
        self._env_reset_requested = False
        self._callback_error_reported = False
        self._unbound_reported: set[str] = set()

    # -- lifecycle ------------------------------------------------------
    def attach(self) -> None:
        """Take over the scripted trajectory and subscribe to the keyboard."""
        import carb.input
        import omni.appwindow

        self._carb_input = carb.input

        self._orientation_term.enable_manual_override(True)
        self._open_close_term.enable_manual_override(True, self._mode_index)
        if self._support_term is not None:
            # Without this the column would drop on the schedule's clock, a few
            # seconds in, wherever the operator happened to have flown the hand.
            self._support_term.enable_manual_override(True)
        self.sync_to_current_pose()

        def _on_key(event) -> bool:
            # Everything here is inside a carb subscription: an exception is not
            # propagated, it is printed once per keystroke forever.  So compare
            # by normalised name (see normalize_carb_name) and never let a
            # surprise escape.
            try:
                name = normalize_carb_name(event.input)
                kind = normalize_carb_name(event.type)
                if kind == "KEY_PRESS":
                    self._events.put((name, True))
                elif kind == "KEY_RELEASE":
                    self._events.put((name, False))
                # KEY_REPEAT is deliberately ignored: the target is integrated
                # from step_dt, so the OS repeat rate must not affect motion.
            except Exception as exc:  # noqa: BLE001 - see comment above
                if not self._callback_error_reported:
                    self._callback_error_reported = True
                    print(
                        f"[WARN] manual root keyboard callback failed once "
                        f"({type(exc).__name__}: {exc}); further occurrences "
                        f"are suppressed.",
                        flush=True,
                    )
            return True

        self._input_interface = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input_interface.subscribe_to_keyboard_events(
            self._keyboard, _on_key
        )
        print(self.help_text(), flush=True)

    def detach(self) -> None:
        """Unsubscribe and drop all input state."""
        if self._subscription is not None and self._input_interface is not None:
            self._input_interface.unsubscribe_to_keyboard_events(
                self._keyboard, self._subscription
            )
        self._subscription = None
        self._keyboard = None
        self._input_interface = None
        self._pressed.clear()
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                break

    def help_text(self) -> str:
        object_line = (
            f"         {OBJECT_RESET_KEY}                respawn the object only (hand stays put)\n"
            if self.has_object
            else ""
        )
        if self._support_term is not None:
            object_line += (
                f"         {SUPPORT_KEY}                lower the cube's support column\n"
            )
        if self.calibration_reporter is not None:
            object_line += (
                f"         {CALIBRATION_KEY}                print the geometry / calibration block\n"
            )
        if self.record_toggle is not None:
            object_line += (
                f"         {RECORD_KEY}                start / stop recording joint angles to CSV\n"
            )
        return (
            "[INFO] hand_move manual root control\n"
            "         1 / 2            OPEN / CLOSE command (policy still drives the fingers)\n"
            "         Up/Down or I/K   world +X / -X\n"
            "         Left/Right or J/L world +Y / -Y\n"
            "         PgUp/PgDn or U/O  world +Z / -Z\n"
            "         A / S            palm-local roll  + / -\n"
            "         D / Z            palm-local pitch + / -\n"
            "         X / C            palm-local yaw   + / -\n"
            f"         {SYNC_KEY}                snap target to the current actual pose\n"
            f"         {RESTORE_KEY}                restore target to this episode's reset pose\n"
            f"         {ENV_RESET_KEY}                full env reset (hand, sticks and object)\n"
            + object_line
            + f"       speeds: {self.translation_speed:.3f} m/s, "
            f"{math.degrees(self.rotation_speed):.1f} deg/s; "
            f"target bounded to {self.max_translation_from_start * 100:.0f} cm from reset"
        )

    @property
    def has_object(self) -> bool:
        """True when the scene carries a resettable object (``hand_object``)."""
        return OBJECT_ASSET_NAME in self._env.scene.rigid_objects

    # -- target manipulation --------------------------------------------
    def sync_to_current_pose(self) -> None:
        """Make the targets equal the actual root pose (key ``H``)."""
        robot = self._env.scene["robot"]
        self._root_action.set_target_position(robot.data.root_link_pos_w.clone())
        self._orientation_term.set_manual_target_quat(
            robot.data.root_link_quat_w.clone()
        )

    def restore_reset_pose(self) -> None:
        """Return the targets to the pose captured at the last reset (``R``)."""
        self._root_action.set_target_position(self._root_action.start_root_pos_w.clone())
        self._orientation_term.set_manual_target_quat(
            self._root_action.start_root_quat_w.clone()
        )

    def reset_object(self) -> bool:
        """Respawn only the object at its ``init_state`` (key ``B``).

        This *is* a ``write_root_pose_to_sim``, but on the **cube**, not on the
        hand.  The invariant this module protects is that the hand root is only
        ever moved by the PD controller's wrench; a free rigid body that the
        operator wants to put back on its stand has no such controller and no
        such invariant.  The pose written is exactly what
        ``mdp.reset_scene_to_default`` would write, so ``B`` and a full reset
        leave the cube in the same place - ``B`` just does not disturb the hand.
        """
        obj = self._env.scene.rigid_objects.get(OBJECT_ASSET_NAME)
        if obj is None:
            return False
        state = obj.data.default_root_state.clone()
        state[:, :3] += self._env.scene.env_origins
        obj.write_root_pose_to_sim(state[:, :7])
        obj.write_root_velocity_to_sim(state[:, 7:])
        return True

    def print_calibration(self) -> bool:
        """Print the task's calibration block (key ``P``).

        Wrapped so a diagnostic that hits a missing entity cannot take down a
        live GUI session - the operator would lose the pose they were setting up.
        """
        if self.calibration_reporter is None:
            print(
                f"[WARN] this task has no calibration report; "
                f"'{CALIBRATION_KEY}' does nothing here.",
                flush=True,
            )
            return False
        try:
            print(self.calibration_reporter(), flush=True)
        except Exception as exc:  # noqa: BLE001 - see docstring
            print(f"[WARN] calibration report failed: {type(exc).__name__}: {exc}", flush=True)
            return False
        return True

    def toggle_recording(self) -> bool:
        """Start or stop the joint recorder (key ``M``).

        Wrapped like :meth:`print_calibration`: a recorder that fails to open
        its file must not take down a live GUI session, because the operator
        would lose the pose they were setting up.
        """
        if self.record_toggle is None:
            print(
                f"[WARN] this session has no joint recorder; "
                f"'{RECORD_KEY}' does nothing here.",
                flush=True,
            )
            return False
        try:
            self.record_toggle()
        except Exception as exc:  # noqa: BLE001 - see docstring
            print(f"[WARN] joint recording failed: {type(exc).__name__}: {exc}", flush=True)
            return False
        return True

    def consume_env_reset_request(self) -> bool:
        """Return (and clear) whether ``R`` asked the play loop for a reset."""
        requested = self._env_reset_requested
        self._env_reset_requested = False
        return requested

    def on_env_reset(self) -> None:
        """Re-seed after an episode reset and drop stale key state.

        ``HandRootHoldAction.reset`` and the command term's ``_resample_command``
        have already re-captured the fresh functional-grasp pose by the time
        this is called, so the targets are correct; what has to be cleared is
        the operator's held keys, otherwise a key held across a reset would
        keep integrating from the old intent.
        """
        self._pressed.clear()
        self.sync_to_current_pose()

    # -- per-step update -------------------------------------------------
    def update(self, dt: float) -> bool:
        """Drain key events and integrate the targets. Returns True if the
        OPEN/CLOSE command changed (the caller should refresh observations)."""
        mode_changed = self._drain_events()

        translation = [0.0, 0.0, 0.0]
        for key in self._pressed:
            binding = TRANSLATION_KEYS.get(key)
            if binding is not None:
                axis, sign = binding
                translation[axis] += sign
        rotation = [0.0, 0.0, 0.0]
        for key in self._pressed:
            binding = ROTATION_KEYS.get(key)
            if binding is not None:
                axis, sign = binding
                rotation[axis] += sign

        # Opposite keys held together cancel, which falls out of the summation.
        if any(translation):
            delta = torch.tensor(translation, device=self._device) * (
                self.translation_speed * dt
            )
            self._root_action.add_target_position_delta(
                delta.unsqueeze(0).expand(self._env.num_envs, 3),
                max_distance_from_start=self.max_translation_from_start,
            )
        if any(rotation):
            delta = torch.tensor(rotation, device=self._device) * (
                self.rotation_speed * dt
            )
            self._orientation_term.apply_manual_local_rotation(
                delta.unsqueeze(0).expand(self._env.num_envs, 3)
            )

        self._elapsed_since_print += dt
        if self.print_interval_s > 0.0 and self._elapsed_since_print >= self.print_interval_s:
            self._elapsed_since_print = 0.0
            print(self.status_line(), flush=True)
        return mode_changed

    def _drain_events(self) -> bool:
        mode_changed = False
        while True:
            try:
                name, is_press = self._events.get_nowait()
            except queue.Empty:
                break
            if is_press:
                if name in MODE_KEYS:
                    new_mode = MODE_KEYS[name]
                    if new_mode != self._mode_index:
                        self._mode_index = new_mode
                        self._open_close_term.set_manual_mode(new_mode)
                        mode_changed = True
                        print(
                            f"[INFO] manual OPEN/CLOSE -> "
                            f"{'OPEN' if new_mode == 0 else 'CLOSE'}",
                            flush=True,
                        )
                elif name == SYNC_KEY:
                    self.sync_to_current_pose()
                    print("[INFO] manual target synced to the current root pose.", flush=True)
                elif name == RESTORE_KEY:
                    self.restore_reset_pose()
                    print("[INFO] manual target restored to the reset pose.", flush=True)
                elif name == ENV_RESET_KEY:
                    # Deferred: resetting mid-update would invalidate the very
                    # buffers this method is about to integrate.
                    self._env_reset_requested = True
                    print("[INFO] full env reset requested.", flush=True)
                elif name == OBJECT_RESET_KEY:
                    if self.reset_object():
                        print("[INFO] object respawned at its initial pose.", flush=True)
                    else:
                        print(
                            f"[WARN] no '{OBJECT_ASSET_NAME}' in this scene; "
                            f"'{OBJECT_RESET_KEY}' does nothing here.",
                            flush=True,
                        )
                elif name == SUPPORT_KEY:
                    if self._support_term is not None:
                        self._support_term.trigger_manual_retract(True)
                        print(
                            "[INFO] support column retracting; the grasp is now "
                            "on its own.",
                            flush=True,
                        )
                    else:
                        print(
                            f"[WARN] no '{SUPPORT_COMMAND_NAME}' command in this "
                            f"scene; '{SUPPORT_KEY}' does nothing here.",
                            flush=True,
                        )
                elif name == CALIBRATION_KEY:
                    self.print_calibration()
                elif name == RECORD_KEY:
                    self.toggle_recording()
                else:
                    self._pressed.add(name)
                    self._report_unbound_key(name)
            else:
                self._pressed.discard(name)
        return mode_changed

    def _report_unbound_key(self, name: str) -> None:
        """Announce the first few unrecognised key names, once each.

        This build's carb reports key names as strings rather than enums, so if
        the naming ever differs from what the key maps above assume (``W`` vs
        ``KEY_W``, say) the symptom would be silent: keys simply do nothing.
        Printing the raw name the first time it is seen turns that into an
        obvious message.  Capped so ordinary typing cannot flood the console.
        """
        if name in TRANSLATION_KEYS or name in ROTATION_KEYS:
            return
        if len(self._unbound_reported) >= 12 or name in self._unbound_reported:
            return
        self._unbound_reported.add(name)
        print(f"[INFO] manual root: key '{name}' is not bound.", flush=True)

    # -- diagnostics -----------------------------------------------------
    def status_line(self) -> str:
        robot = self._env.scene["robot"]
        origin = self._env.scene.env_origins[0]
        current_pos = robot.data.root_link_pos_w[0] - origin
        current_quat = robot.data.root_link_quat_w[0]
        target_pos = self._root_action.target_root_pos_w[0] - origin
        target_quat = self._orientation_term.target_quat_w[0]

        position_error = float((target_pos - current_pos).norm())
        delta = quat_mul_single(target_quat, quat_conjugate_single(current_quat))
        orientation_error = math.degrees(2.0 * math.acos(min(1.0, abs(float(delta[0])))))

        held = sorted(
            k for k in self._pressed if k in TRANSLATION_KEYS or k in ROTATION_KEYS
        )
        return (
            f"[manual] pos {fmt3(current_pos)} -> {fmt3(target_pos)}"
            f" (err {position_error * 100:5.2f} cm) |"
            f" quat {fmt4(current_quat)} -> {fmt4(target_quat)}"
            f" (err {orientation_error:5.2f} deg) |"
            f" mode {'OPEN ' if self._mode_index == 0 else 'CLOSE'} |"
            f" keys {','.join(held) if held else '-'}"
        )


def fmt3(vector) -> str:
    return "(" + ", ".join(f"{float(v):+.3f}" for v in vector) + ")"


def fmt4(vector) -> str:
    return "(" + ", ".join(f"{float(v):+.3f}" for v in vector) + ")"


def quat_conjugate_single(quat: torch.Tensor) -> torch.Tensor:
    return torch.stack([quat[0], -quat[1], -quat[2], -quat[3]])


def quat_mul_single(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
