"""Construct the hardware backend, with an explicit record of what is verified.

``RealWujiHand`` in ``real_wuji.py`` is the SDK wrapper.  This module is the
gate in front of it: it names, item by item, what hardware runs have actually
established and what has not been measured yet, and it checks that the object
really satisfies ``WujiBackend`` instead of assuming it.

This module exists separately because "the SDK works" and "this is safe to hand
a 20-joint policy" are different claims.  ``finger_reach`` established the first
on four joints.  The second adds every motor at once, the pregrasp pose, and a
grasp whose preload must survive a fault -- so the list below is kept honest
rather than deleted.
"""

from __future__ import annotations

from ..common.backend_protocol import WujiBackend


#: Established by hardware runs, not by reading the vendor documentation.
VERIFIED_ON_HARDWARE = (
    "SDK generation: wujihandpy 1.7.0 (NOT the older wuji_sdk)",
    "firmware 1.2.1, product SN read back",
    "handedness: contract is the pinned wuji-description RIGHT body; "
    "read_handedness() is a tactile field and reports 0 on this right hand",
    "factory joint limits read from the hand and pinned as REAL_HAND_FACTORY_LIMITS",
    "encoder sign/zero per joint, via the single-joint bring-up move",
    "command rate: 90 Hz publish with 30 Hz policy, measured round trips",
    "closed-loop residual policy driving four joints (finger_reach)",
)

#: Not measured.  Each of these is a thing the code must therefore not assume.
PENDING_REAL_VALIDATION = (
    "firmware watchdog: whether the last target is held when publishing stops. "
    "safe_stop() therefore freezes the target and relies on the caller to keep "
    "publishing it, rather than trusting the firmware to hold.",
    "controller semantics: reading q inside realtime_controller with "
    "enable_upstream=True is not a vendor-demonstrated pattern.",
    "twenty-joint thermal behaviour at the pregrasp pose. Measured 2026-08-19: "
    "holding pregrasp WITHOUT chopsticks stalls the fingers against each other, "
    "saturating current at 1.5 A and reaching 88.4 C on finger1_joint2 in 96 s. "
    "Any twenty-joint bring-up must reach pregrasp only with the sticks in place.",
)


def make_real_backend(**kwargs) -> WujiBackend:
    """Return a ``WujiBackend``-conforming handle on the physical hand.

    ``kwargs`` go straight to ``RealWujiHand``.  Importing is deferred so this
    module stays importable in environments without ``wujihandpy`` -- the
    MuJoCo environment does not have it, and the contract tests run there.
    """

    from .real_wuji import RealWujiHand

    backend = RealWujiHand(**kwargs)
    if not isinstance(backend, WujiBackend):
        missing = [
            name
            for name in ("joint_identifiers", "read_joint_positions",
                         "write_joint_position_targets", "health", "safe_stop")
            if not callable(getattr(backend, name, None))
        ]
        raise RuntimeError(f"RealWujiHand does not satisfy WujiBackend; missing {missing}.")
    return backend


def pending_validation_report() -> str:
    """Render both lists for a run header, so neither is silently forgotten."""

    lines = ["[VERIFIED]"]
    lines += [f"  + {item}" for item in VERIFIED_ON_HARDWARE]
    lines.append("[NOT MEASURED]")
    lines += [f"  ! {item}" for item in PENDING_REAL_VALIDATION]
    return "\n".join(lines)
