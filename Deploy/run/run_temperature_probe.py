"""Read-only timing probe for wujihandpy temperature APIs.

This tool never enables a motor and never writes a target.  It measures the
installed SDK rather than assuming that a 20-joint SDO duration divides evenly
across individual joints.  In particular it separates:

* synchronous response latency;
* async response completion latency;
* unchecked request submission cost; and
* cached ``get`` cost.

The unchecked API has no completion/freshness token.  Equal consecutive motor
temperatures therefore cannot prove when its cache was refreshed; async
completion is the trustworthy measurement of response latency.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import numpy as np


def _summary(label: str, samples_ms: list[float]) -> None:
    values = np.asarray(samples_ms, dtype=np.float64)
    print(
        f"{label:28s} n={values.size:3d}  mean={values.mean():7.3f} ms  "
        f"p95={np.percentile(values, 95):7.3f} ms  max={values.max():7.3f} ms"
    )


def _sync_samples(call, repeats: int) -> tuple[list[float], object]:
    elapsed = []
    value = None
    for _ in range(repeats):
        began = time.perf_counter()
        value = call()
        elapsed.append((time.perf_counter() - began) * 1000.0)
    return elapsed, value


async def _async_samples(joint, repeats: int) -> tuple[list[float], float]:
    elapsed = []
    value = float("nan")
    for _ in range(repeats):
        began = time.perf_counter()
        value = float(await joint.read_joint_temperature_async())
        elapsed.append((time.perf_counter() - began) * 1000.0)
    return elapsed, value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Wuji temperature reads without enabling any motor."
    )
    parser.add_argument("--finger", type=int, choices=range(1, 6), default=3,
                        help="1-based finger number")
    parser.add_argument("--joint", type=int, choices=range(1, 5), default=1,
                        help="1-based joint number")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--unchecked-wait-ms", type=float, default=50.0,
                        help="Wait after each unchecked request before reading cache")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if args.unchecked_wait_ms < 0.0:
        raise ValueError("--unchecked-wait-ms must be non-negative")

    import wujihandpy

    hand = wujihandpy.Hand()
    finger = hand.finger(args.finger - 1)
    joint = finger.joint(args.joint - 1)
    print(f"SDK {getattr(wujihandpy, '__version__', 'unknown')}  "
          f"finger{args.finger}_joint{args.joint}")
    print("READ ONLY: no motor enable and no target write\n")

    full_ms, full_value = _sync_samples(hand.read_joint_temperature, args.repeats)
    finger_ms, finger_value = _sync_samples(finger.read_joint_temperature, args.repeats)
    joint_ms, joint_value = _sync_samples(joint.read_joint_temperature, args.repeats)
    _summary("20-joint synchronous", full_ms)
    _summary("4-joint synchronous", finger_ms)
    _summary("1-joint synchronous", joint_ms)
    print(f"latest sync values: all max={float(np.max(full_value)):.1f}C, "
          f"finger max={float(np.max(finger_value)):.1f}C, joint={float(joint_value):.1f}C")

    async_ms, async_value = asyncio.run(_async_samples(joint, args.repeats))
    _summary("1-joint async completion", async_ms)
    print(f"latest async value: {async_value:.1f}C")

    submit_ms = []
    get_ms = []
    cached = []
    wait_s = args.unchecked_wait_ms / 1000.0
    for _ in range(args.repeats):
        began = time.perf_counter()
        joint.read_joint_temperature_unchecked()
        submit_ms.append((time.perf_counter() - began) * 1000.0)
        if wait_s:
            time.sleep(wait_s)
        began = time.perf_counter()
        cached.append(float(joint.get_joint_temperature()))
        get_ms.append((time.perf_counter() - began) * 1000.0)
    _summary("1-joint unchecked submit", submit_ms)
    _summary("1-joint cached get", get_ms)
    print(f"cached values after {args.unchecked_wait_ms:g} ms: "
          f"min={min(cached):.1f}C max={max(cached):.1f}C")
    if statistics.pstdev(cached) == 0.0:
        print("NOTE: unchanged cache values cannot prove response freshness; "
              "use async completion latency for scheduling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
