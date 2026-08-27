# [backend/실물] 하드웨어 백엔드 생성 게이트 — 실측으로 검증된 항목과 미측정 항목을 명시적으로 나열.
"""실물 SDK 래퍼를 공통 백엔드로 생성하는 안전 게이트.
실행 전 실측 검증 항목과 미확인 항목 구분."""

from __future__ import annotations

from ..common.backend_protocol import WujiBackend


#: 제조사 문서가 아닌 실제 하드웨어 실행으로 확인한 항목.
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

#: 아직 실측하지 않아 코드에서 가정하면 안 되는 항목.
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
    """실물 손을 ``WujiBackend``로 생성 후 필수 함수 검사."""

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
    """검증 완료·미완료 항목을 실행 헤더 문자열로 정리."""

    lines = ["[VERIFIED]"]
    lines += [f"  + {item}" for item in VERIFIED_ON_HARDWARE]
    lines.append("[NOT MEASURED]")
    lines += [f"  ! {item}" for item in PENDING_REAL_VALIDATION]
    return "\n".join(lines)
