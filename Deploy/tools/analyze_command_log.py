# [tool] 90Hz 명령 로그를 읽어 정책 스텝 내부(틱 간 거동)를 들여다본다.
"""Read a ``*_90hz.csv`` command-rate log and look inside the policy step.

The policy samples q every 33.3 ms, so the 30 Hz log shows only where the joint
was at each decision point.  Whatever happens in between -- overshoot, ringing,
a joint that sticks and then breaks free -- is invisible there.  The command-rate
log samples three times as often and this reads the difference out.

Written after two open questions the 30 Hz logs could not settle: whether the
sim/real joint gap is servo gain or friction, and why finger3_joint3 once took
ten times longer to move when its path crossed zero.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ..policy.finger_reach import MIDDLE_JOINT_NAMES


def load(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows.")
    return {key: np.asarray([float(r[key]) for r in rows]) for key in rows[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="A *_90hz.csv written by --log-commands.")
    parser.add_argument("--divider", type=int, default=3,
                        help="Command ticks per policy step (90/30 = 3).")
    args = parser.parse_args()

    log = load(args.log)
    Q = np.stack([log[f"q_curr_{k}"] for k in range(1, 5)], 1)
    T = np.stack([log[f"q_target_{k}"] for k in range(1, 5)], 1)
    t = log["time"]
    dt = np.diff(t)
    print(f"{args.log}  {len(t)} ticks, {t[-1]-t[0]:.2f} s")
    print(f"  tick period: mean {dt.mean()*1000:.3f} ms, p95 {np.percentile(dt,95)*1000:.3f}, "
          f"max {dt.max()*1000:.3f}")

    print(f"\n  servo tracking |q_target - q| [mrad]")
    print(f"    {'joint':<24}{'mean':>9}{'p95':>9}{'max':>9}")
    for index, name in enumerate(MIDDLE_JOINT_NAMES):
        e = np.abs(T[:, index] - Q[:, index]) * 1000
        print(f"    {name:<24}{e.mean():>9.2f}{np.percentile(e,95):>9.2f}{e.max():>9.2f}")

    # A new target lands on every divider-th tick.  Comparing the joint's motion
    # in the tick right after against the rest of the step shows whether it
    # responds immediately or only once the filter has wound up.
    print(f"\n  response within the policy step [mrad moved per tick]")
    print(f"    {'joint':<24}{'tick 0':>9}{'tick 1':>9}{'tick 2':>9}")
    step = np.diff(Q, axis=0) * 1000
    phase = np.arange(len(step)) % args.divider
    for index, name in enumerate(MIDDLE_JOINT_NAMES):
        cells = "".join(
            f"{np.abs(step[phase == p, index]).mean():>9.3f}" for p in range(args.divider)
        )
        print(f"    {name:<24}{cells}")

    # Overshoot: the joint passed its target and came back.
    print(f"\n  sign changes of (q_target - q)  = crossings of the target")
    for index, name in enumerate(MIDDLE_JOINT_NAMES):
        e = T[:, index] - Q[:, index]
        crossings = int(np.sum(np.diff(np.sign(e[np.abs(e) > 1e-5])) != 0))
        print(f"    {name:<24}{crossings:>6}  "
              f"({'overshoot/ringing present' if crossings > 4 else 'monotonic approach'})")

    # Motion near zero, where the one anomalous slow move happened.
    print(f"\n  speed vs |q|, to test the near-zero stiction seen on joint3")
    speed = np.abs(np.diff(Q, axis=0)) / np.maximum(dt[:, None], 1e-9)
    for index, name in enumerate(MIDDLE_JOINT_NAMES):
        near = np.abs(Q[:-1, index]) < 0.05
        far = ~near
        if near.sum() < 10 or far.sum() < 10:
            print(f"    {name:<24}not enough samples on both sides of |q|=0.05")
            continue
        moving = speed[:, index] > 1e-3
        near_speed = speed[near & moving, index].mean() if (near & moving).any() else float("nan")
        far_speed = speed[far & moving, index].mean() if (far & moving).any() else float("nan")
        print(f"    {name:<24}|q|<0.05: {near_speed:6.3f} rad/s   "
              f"|q|>0.05: {far_speed:6.3f} rad/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
