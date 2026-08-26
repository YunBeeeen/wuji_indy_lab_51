# [tool] 서로 다른 백엔드의 reach 로그를 컬럼 단위로 비교.
"""Compare finger-reach logs from different backends, column by column.

The three backends never run together: the scenario file fixes the target
sequence so Isaac, MuJoCo and the real hand can each replay it whenever, and
the comparison happens afterwards on the CSVs.  This tool is that comparison.

Rows are matched on POLICY STEP ORDINAL, not on the ``time`` column.  Both logs
write exactly one row per policy step, but they timestamp different instants
within it -- MuJoCo stamps ``(step+1)/30`` at the end of its physics hold, the
real hand stamps measured ``monotonic()`` elapsed at the start of the tick -- so
the two time columns never coincide and matching on them found nothing at all.
A run that stopped early still lines up over the steps both completed.

Reported in the diagnostic priority order: joint trajectory first, because a
difference there explains everything downstream, and fingertip position last,
because it can differ even when the joints agree -- which is the signature of a
kinematics mismatch rather than a control one.
"""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..policy.finger_reach import MIDDLE_JOINT_NAMES, REACH_FINGER_INDEX
from ..common.fingertip_fk import DEFAULT_URDF_PATH


def _rpy(rpy):
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _tip_chain():
    """Palm -> controlled fingertip joint chain from the pinned official URDF."""

    root = ET.parse(DEFAULT_URDF_PATH).getroot()
    joints = {}
    for element in root.findall("joint"):
        origin, axis = element.find("origin"), element.find("axis")
        transform = np.eye(4)
        if origin is not None:
            transform[:3, :3] = _rpy([float(v) for v in origin.attrib.get("rpy", "0 0 0").split()])
            transform[:3, 3] = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
        strip = lambda name: name[6:] if name.startswith("right_") else name
        joints[strip(element.find("child").attrib["link"])] = dict(
            name=strip(element.attrib["name"]),
            parent=strip(element.find("parent").attrib["link"]),
            transform=transform,
            axis=np.asarray([float(v) for v in axis.attrib.get("xyz", "0 0 0").split()])
            if axis is not None else np.zeros(3),
            joint_type=element.attrib["type"],
        )
    chain, link = [], f"finger{REACH_FINGER_INDEX}_tip_link"
    while link in joints:
        chain.append(joints[link])
        link = joints[link]["parent"]
    return chain[::-1]


def fingertip_in_palm(chain, q_middle) -> np.ndarray:
    """Palm-frame fingertip position. MODEL OUTPUT, not a measurement.

    The real hand reports joint angles and nothing Cartesian, so its fingertip
    can only be inferred through this kinematic chain.  That inference assumes
    the hardware's link lengths and encoder zeros match the URDF, neither of
    which has been verified against the physical hand.
    """

    angles = dict(zip(MIDDLE_JOINT_NAMES, q_middle))
    pose = np.eye(4)
    for entry in chain:
        local = entry["transform"].copy()
        if entry["joint_type"] == "revolute":
            axis = entry["axis"] / np.linalg.norm(entry["axis"])
            theta = angles[entry["name"]]
            skew = np.asarray([[0, -axis[2], axis[1]],
                               [axis[2], 0, -axis[0]],
                               [-axis[1], axis[0], 0]])
            local[:3, :3] = local[:3, :3] @ (
                np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * skew @ skew
            )
        pose = pose @ local
    return pose[:3, 3]

GROUPS = (
    ("q", "joint trajectory q(t)", [f"q_curr_{k}" for k in range(1, 5)], "rad", 1000.0, "mrad"),
    ("action", "raw policy action", [f"action_{k}" for k in range(1, 5)], "-", 1000.0, "1e-3"),
    ("q_target", "decoded joint target", [f"q_target_{k}" for k in range(1, 5)], "rad", 1000.0, "mrad"),
    ("tip", "fingertip position", ["tip_palm_x", "tip_palm_y", "tip_palm_z"], "m", 1000.0, "mm"),
    ("error", "target error norm", ["error_norm"], "m", 1000.0, "mm"),
)


def load(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows.")
    return {key: np.asarray([float(r[key]) for r in rows]) for key in rows[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Baseline log, usually the Isaac one.")
    parser.add_argument("others", type=Path, nargs="+", help="Logs to compare against it.")
    parser.add_argument("--tolerance-mrad", type=float, default=10.0,
                        help="Joint agreement threshold used for the verdict line.")
    args = parser.parse_args()

    reference = load(args.reference)
    print(f"reference : {args.reference}  ({len(reference['time'])} rows)")

    for path in args.others:
        other = load(path)
        shared_steps = min(len(reference["time"]), len(other["time"]))
        if shared_steps == 0:
            print(f"\n{path}: no rows to compare.")
            continue
        ri = oi = np.arange(shared_steps)

        print(f"\ncompare   : {path}  ({len(other['time'])} rows, "
              f"{shared_steps} shared policy steps)")
        mismatched_targets = int(
            np.count_nonzero(
                np.abs(reference["target_palm_x"][ri] - other["target_palm_x"][oi]) > 1e-6
            )
        )
        if mismatched_targets:
            print(f"  WARNING: {mismatched_targets} samples were commanded different targets; "
                  "these logs are not comparable.")

        print(f"  {'quantity':<22}{'max diff':>12}{'mean diff':>12}{'rms':>12}   unit")
        verdict = None
        for key, label, columns, _unit, scale, out_unit in GROUPS:
            # Only compare columns BOTH logs carry.  The real hand has no
            # Cartesian fingertip measurement, so its log has no tip_palm_* or
            # error_norm; checking only `other` broke whenever the real log was
            # passed as the reference.
            shared = [c for c in columns if c in reference and c in other]
            if not shared:
                continue
            deltas = np.concatenate(
                [np.abs(reference[c][ri] - other[c][oi]) for c in shared]
            )
            rms = float(np.sqrt(np.mean(deltas ** 2)))
            print(f"  {label:<22}{deltas.max()*scale:>12.4f}{deltas.mean()*scale:>12.4f}"
                  f"{rms*scale:>12.4f}   {out_unit}")
            if key == "q":
                verdict = deltas.max() * 1000.0

        # --- per-joint, so a single large joint cannot hide behind an average
        tip_errors: dict[str, float] = {}
        joint_columns = [f"q_curr_{k}" for k in range(1, 5)]
        if all(c in reference and c in other for c in joint_columns):
            QA = np.stack([reference[c][ri] for c in joint_columns], 1)
            QB = np.stack([other[c][oi] for c in joint_columns], 1)

            # Split transition from settled: the two are different questions.
            # A gap right after a target change means one plant is slower; a gap
            # that survives into the settled window means they stop in different
            # places, which no amount of extra time will fix.
            targets = reference.get("target_index")
            settled = np.ones(len(ri), dtype=bool)
            if targets is not None:
                boundaries = list(np.flatnonzero(np.diff(targets[ri])) + 1) + [0]
                for boundary in boundaries:
                    settled[boundary:boundary + 60] = False

            print(f"\n  per joint [mrad / deg]        {'all':>16}{'settled':>18}")
            for index, name in enumerate(MIDDLE_JOINT_NAMES):
                delta = np.abs(QA[:, index] - QB[:, index])
                text_all = f"{delta.mean()*1000:7.1f} / {np.degrees(delta.mean()):5.2f}"
                text_set = (f"{delta[settled].mean()*1000:7.1f} / "
                            f"{np.degrees(delta[settled].mean()):5.2f}")
                print(f"    {name:<24}{text_all:>16}{text_set:>18}")
            if targets is not None:
                moving = ~settled
                print(f"    {'(transition window)':<24}"
                      f"{np.abs(QA-QB)[moving].mean()*1000:9.1f} mrad over "
                      f"{int(moving.sum())} steps")

            # --- fingertip vs the commanded target, which is what the task is
            axes = ["target_palm_x", "target_palm_y", "target_palm_z"]
            if all(a in reference and a in other for a in axes):
                chain = _tip_chain()
                tip_errors = {}
                print(f"\n  fingertip vs target [mm]   (FK from q -- MODEL, not measured)")
                for label, log, Q, idx in (
                    (args.reference.name, reference, QA, ri),
                    (path.name, other, QB, oi),
                ):
                    goal = np.stack([log[a][idx] for a in axes], 1)
                    tip = np.stack([fingertip_in_palm(chain, q) for q in Q])
                    error = goal - tip
                    norm = np.linalg.norm(error, axis=1)
                    per_axis = "  ".join(
                        f"{a}{error[settled][:, i].mean()*1000:+6.2f}"
                        for i, a in enumerate("xyz")
                    )
                    tip_errors[label] = float(norm[settled].mean())
                    print(f"    {label:<34}settled {norm[settled].mean()*1000:6.2f}   {per_axis}")
                    if "error_norm" in log:
                        print(f"    {'  (log column, for cross-check)':<34}"
                              f"        {log['error_norm'][idx][settled].mean()*1000:6.2f}")

        if verdict is not None:
            if verdict <= args.tolerance_mrad:
                print(f"\n  -> joints agree within {args.tolerance_mrad:.1f} mrad.")
            elif len(tip_errors) == 2 and abs(np.diff(list(tip_errors.values()))[0]) < 0.005:
                # Four joints chasing a 3D point leave one redundant degree of
                # freedom, so two plants can settle on different postures that
                # put the fingertip in nearly the same place.  Joint divergence
                # is then a fact about posture, not about task performance, and
                # sending someone to audit the observation pipeline over it
                # wastes their time.
                print(f"\n  -> joints differ ({verdict:.1f} mrad) but the fingertip lands within "
                      f"{abs(np.diff(list(tip_errors.values()))[0])*1000:.1f} mm of the same "
                      "distance from target.\n     The finger has 4 DoF for a 3D goal, so the "
                      "redundant DoF absorbs it: different posture, same task result.\n"
                      "     Look at the per-axis fingertip bias above before suspecting the "
                      "contract.")
            else:
                print(f"\n  -> joints DIVERGE ({verdict:.2f} mrad > {args.tolerance_mrad:.1f}) "
                      "and the fingertip errors differ too.\n     Check observation, "
                      "normalization, joint mapping, action decoding, timing, actuator model, "
                      "physics -- in that order.")

        if "error_norm" in reference and "error_norm" in other:
            print(f"  {'final target error':<22}reference "
                  f"{reference['error_norm'][ri][-1]*1000:7.2f} mm"
                  f"   vs {other['error_norm'][oi][-1]*1000:7.2f} mm")
        elif not tip_errors:
            print("  (no fingertip comparison: a log is missing the target columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
