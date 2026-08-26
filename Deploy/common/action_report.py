# [common] 정책 액션/관절 목표를 사람이 읽는 줄로. 실물·MuJoCo 러너 공용 포맷터.
"""Render a decoded policy action so "is it commanding anything?" is answerable.

``|target - q|`` alone cannot answer that question: the residual is
``ACTION_SCALE_RAD * action`` with a per-joint scale (0.1/0.1/0.2/0.15 rad), so
the same displacement means a very different network output depending on which
joint moved.  These helpers print the action itself.

Lives in ``common/`` because both the real-hand runner and the MuJoCo runner
need the identical view -- comparing sim against hardware is the whole point,
and two hand-written format strings drift.
"""

from __future__ import annotations

import numpy as np

from .policy_contract import ACTION_DIM, ACTION_SCALE_RAD, POLICY_JOINT_NAMES

#: Joints per finger, in canonical policy order.
_PER_FINGER = 4
_FINGERS = ACTION_DIM // _PER_FINGER


def finger_row(values, fmt: str) -> str:
    """Lay 20 policy-ordered numbers out as five groups of four, one per finger.

    A flat 20-vector is unreadable at 30 Hz; grouped by finger you can see at a
    glance which finger the policy is driving.  Column *j* of every group is the
    same joint number.
    """

    values = np.asarray(values, dtype=np.float64).reshape(-1)[:ACTION_DIM]
    return " ".join(
        "[" + " ".join(format(float(v), fmt)
                       for v in values[f * _PER_FINGER:(f + 1) * _PER_FINGER]) + "]"
        for f in range(_FINGERS)
    )


def action_summary(decoded) -> str:
    """One line: peak action with its joint name, RMS, and how many hit the clip.

    ``clip`` counts outputs that sat on the +-1 ActionManager clip -- a
    rail-to-rail policy -- which is a different thing from the ``clamp`` count,
    which counts targets pushed back by the command limits.
    """

    a = np.asarray(decoded.onnx_action, dtype=np.float64)
    peak = int(np.argmax(np.abs(a)))
    return (f"a|max|={abs(float(a[peak])):.3f}({POLICY_JOINT_NAMES[peak]}) "
            f"rms={float(np.sqrt(np.mean(np.square(a)))):.3f} "
            f"clip={int(np.asarray(decoded.action_was_clipped).sum()):2d}")


def action_detail_lines(decoded, q_current, indent: str = "           ") -> list[str]:
    """The action, the joint target it produced, and the q it was added to.

    Three lines rather than two: ``qt`` is only interpretable next to ``q``,
    because the contract is residual -- ``qt = clamp(q + scale * a)``.
    """

    q = np.asarray(q_current, dtype=np.float64).reshape(-1)[:ACTION_DIM]
    return [
        f"{indent}a  " + finger_row(decoded.onnx_action, "+.2f"),
        f"{indent}qt " + finger_row(decoded.position_target, "+.3f"),
        f"{indent}q  " + finger_row(q, "+.3f"),
    ]


def print_action_detail(decoded, q_current, indent: str = "           ") -> None:
    for line in action_detail_lines(decoded, q_current, indent):
        print(line)


def action_verdict(mean_peak: float, max_peak: float, steps: int) -> list[str]:
    """End-of-run lines saying whether the policy commanded anything at all.

    Reported from an accumulator rather than re-derived from the CSV so it
    survives a run that aborted -- an aborted run is exactly the one where you
    need to know whether the policy was alive up to that point.
    """

    lines = [f"[ACTION]     스텝당 |a|max 평균 {mean_peak:.3f}, "
             f"전체 최대 {max_peak:.3f}  ({steps} 스텝)"]
    if max_peak < 1e-3:
        lines.append("             액션이 0 입니다 -- 정책이 아무것도 명령하지 "
                     "않았습니다. ONNX 출력부터 볼 것.")
    else:
        lo, hi = float(ACTION_SCALE_RAD.min()), float(ACTION_SCALE_RAD.max())
        lines.append(f"             관절 스케일 {lo:.2f}~{hi:.2f} rad 이므로 "
                     f"|a|max {mean_peak:.3f} 는 틱당 "
                     f"{mean_peak * lo * 1000.0:.0f}~{mean_peak * hi * 1000.0:.0f} mrad "
                     f"의 잔차 명령입니다.")
    return lines
