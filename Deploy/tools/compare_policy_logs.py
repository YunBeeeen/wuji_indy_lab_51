# [tool] 105D 정책 로그 비교. obs 재생으로 배선을 먼저 검증하고, 그 다음에 블록별 차이.
"""Compare 105D grasp-policy runs -- and say which comparisons mean anything.

Comparing a real trajectory against a simulated one step by step is the obvious
thing to do and it is nearly useless here.  The loop is closed through contact:
the policy's own output changes the next observation, so any difference at all
-- 1 mm of stick placement, one frame of camera latency -- compounds within a
few ticks.  Two MuJoCo runs from slightly different seeds diverge the same way.
"Real and sim look nothing alike after 3 seconds" is the expected result and
tells you nothing about whether the deployment is wired correctly.

What IS decisive is that the policy is a deterministic function.  So this tool
works in three levels, and they answer different questions:

  L1  REPLAY   Feed each logged observation back through the ONNX and compare
               with the action that was logged next to it.  Same file, same
               network -- this MUST match to float precision.  A mismatch means
               the deployment's own decode is not what it recorded, which is a
               bug in one run and needs no second run to find.

  L2  CROSS    Feed run A's observations through run B's ONNX.  Both runs are
               supposed to carry the same policy, so this must also match.  A
               mismatch means the two runs are not the same network -- the
               single most likely deployment mistake and completely invisible
               in a trajectory plot.

  L3  BLOCKS   Only now compare the observations themselves, block by block
               (q, fingertips, stick1/2, last_action, mode).  These SHOULD
               differ; the useful output is WHICH block is out of family and by
               how much, because that localises the sim-to-real gap.  Ranges
               are compared, not per-step values, precisely because per-step
               alignment is meaningless in a closed loop.

L1 and L2 are pass/fail.  L3 is a measurement, never a verdict.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ..common.policy_contract import (
    ACTION_CLIP,
    ACTION_DIM,
    OBSERVATION_DIM,
    OBSERVATION_SLICES,
    POLICY_JOINT_NAMES,
    observation_csv_columns,
)

#: Float32 round-trip through CSV text costs about this much.  Anything above
#: it is a real difference, not a formatting artefact.
REPLAY_TOLERANCE = 1e-5


def _read(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path}: no rows.")
    columns = {}
    for key in rows[0]:
        try:
            columns[key] = np.asarray([float(row[key]) for row in rows])
        except (TypeError, ValueError):
            columns[key] = np.asarray([row[key] for row in rows], dtype=object)
    return columns


def _matrix(columns: dict[str, np.ndarray], names) -> np.ndarray:
    missing = [n for n in names if n not in columns]
    if missing:
        raise SystemExit(
            f"Log is missing {len(missing)} column(s), first {missing[0]!r}.  "
            "Logs written before observation logging was added cannot be "
            "replayed -- rerun with --out.")
    return np.column_stack([columns[n] for n in names]).astype(np.float32)


def observations_of(columns) -> np.ndarray:
    return _matrix(columns, observation_csv_columns())


def actions_of(columns) -> np.ndarray:
    return _matrix(columns, [f"a_{n}" for n in POLICY_JOINT_NAMES])


def targets_of(columns) -> np.ndarray:
    return _matrix(columns, [f"qt_{n}" for n in POLICY_JOINT_NAMES])


def joints_of(columns) -> np.ndarray:
    return _matrix(columns, [f"q_{n}" for n in POLICY_JOINT_NAMES])


def replay(onnx_path: Path, observations: np.ndarray) -> np.ndarray:
    """Run the ONNX actor over recorded observations, clipped as the wrapper does."""

    from ..policy.onnx_policy import OnnxPolicy

    policy = OnnxPolicy(onnx_path, OBSERVATION_DIM, ACTION_DIM)
    out = np.empty((observations.shape[0], ACTION_DIM), dtype=np.float32)
    for i, observation in enumerate(observations):
        out[i] = policy.infer(observation)
    return np.clip(out, -ACTION_CLIP, ACTION_CLIP)


def _report_match(label: str, logged: np.ndarray, replayed: np.ndarray) -> bool:
    n = min(len(logged), len(replayed))
    diff = np.abs(logged[:n] - replayed[:n])
    worst = int(np.unravel_index(np.argmax(diff), diff.shape)[1]) if diff.size else 0
    ok = diff.max() <= REPLAY_TOLERANCE if diff.size else True
    print(f"  {label:28s} max|diff| {diff.max():.3e} "
          f"({POLICY_JOINT_NAMES[worst]})  {'PASS' if ok else 'FAIL'}")
    if not ok:
        rows = np.flatnonzero(diff.max(axis=1) > REPLAY_TOLERANCE)
        print(f"     {rows.size}/{n} 스텝 불일치, 첫 스텝 {int(rows[0])}")
    return ok


def compare_blocks(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> None:
    print(f"\n[L3 BLOCKS]  관측 블록별 범위 -- 차이가 나는 게 정상입니다.")
    print(f"  {'블록':22s} {name_a:>26s} {name_b:>26s}   {'중앙값차':>10s}")
    for name, term in OBSERVATION_SLICES.items():
        ba, bb = a[:, term.slice], b[:, term.slice]
        print(f"  {name:22s} "
              f"[{ba.min():+8.3f},{ba.max():+8.3f}] μ{ba.mean():+7.3f} "
              f"[{bb.min():+8.3f},{bb.max():+8.3f}] μ{bb.mean():+7.3f} "
              f"   {abs(float(np.median(ba)) - float(np.median(bb))):9.4f}")
    print("  두 런의 스텝은 정렬되지 않습니다 (폐루프라 궤적은 반드시 갈라집니다).")
    print("  볼 것은 어느 블록이 통째로 벗어났는가입니다 -- 예: stick 블록만 범위가")
    print("  다르면 비전/배치 문제, joint 블록이 다르면 pregrasp 나 정규화 문제입니다.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_a", type=Path, help="첫 번째 런 CSV (--out 으로 쓴 것).")
    parser.add_argument("log_b", type=Path, nargs="?",
                        help="두 번째 런 CSV. 없으면 L1 만 수행합니다.")
    parser.add_argument("--onnx-a", type=Path, default=None,
                        help="log_a 를 만든 policy.onnx. L1/L2 에 필요합니다.")
    parser.add_argument("--onnx-b", type=Path, default=None,
                        help="log_b 를 만든 policy.onnx. 생략하면 --onnx-a 를 씁니다.")
    args = parser.parse_args()

    a = _read(args.log_a)
    obs_a, act_a = observations_of(a), actions_of(a)
    print(f"[LOG A] {args.log_a}  {len(obs_a)} 스텝")
    b = obs_b = act_b = None
    if args.log_b is not None:
        b = _read(args.log_b)
        obs_b, act_b = observations_of(b), actions_of(b)
        print(f"[LOG B] {args.log_b}  {len(obs_b)} 스텝")

    ok = True
    if args.onnx_a is not None:
        print(f"\n[L1 REPLAY]  기록된 obs -> ONNX -> 기록된 action 과 대조 "
              f"(허용 {REPLAY_TOLERANCE:.0e})")
        print("             여기서 FAIL 이면 그 런 하나가 이미 틀린 것입니다.")
        ok &= _report_match("A: obs_A -> onnx_A", act_a, replay(args.onnx_a, obs_a))
        onnx_b = args.onnx_b or args.onnx_a
        if obs_b is not None:
            ok &= _report_match("B: obs_B -> onnx_B", act_b, replay(onnx_b, obs_b))
            if args.onnx_b is not None:
                print(f"\n[L2 CROSS]   A 의 obs 를 B 의 ONNX 에 넣습니다.")
                print("             같은 정책이어야 하므로 PASS 여야 합니다. "
                      "FAIL = 서로 다른 체크포인트입니다.")
                ok &= _report_match("A: obs_A -> onnx_B", act_a,
                                    replay(args.onnx_b, obs_a))
    else:
        print("\n[L1 REPLAY]  건너뜀 -- --onnx-a 를 주면 배선을 먼저 검증합니다. "
              "그게 유일한 pass/fail 비교입니다.")

    if obs_b is not None:
        compare_blocks(obs_a, obs_b, args.log_a.stem[:26], args.log_b.stem[:26])
        print(f"\n[궤적]       참고용 -- 판정 근거로 쓰지 마십시오.")
        for label, ma, mb in (("action", act_a, act_b),
                              ("qt", targets_of(a), targets_of(b)),
                              ("q", joints_of(a), joints_of(b))):
            print(f"  {label:8s} |{label}| 평균  A {np.abs(ma).mean():.4f}   "
                  f"B {np.abs(mb).mean():.4f}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
