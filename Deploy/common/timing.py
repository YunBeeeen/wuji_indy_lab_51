# [common] 정책 틱 안의 단계별 지연 계측. 어디서 예산을 먹는지 사후에 알아내기 위한 것.
"""Per-stage latency accounting inside one policy tick.

A policy step has a hard budget -- 1/30 s -- and when a run misses it, "the loop
was late" is not a finding.  This records where the time actually went, so the
answer is a stage name rather than a guess.

Two kinds of number, deliberately separated:

* **stages** are durations: how long a step of work took.  Measured with
  ``perf_counter``, which is monotonic and unaffected by clock adjustments.
* **gauges** are instantaneous readings that are not durations at all -- the age
  of the camera frame being used, the arrival skew between MAIN and SIDE.  These
  answer "how stale was the input", which no duration can.

Mixing the two is how a timing report starts lying: a 5 ms ``sample()`` that
consumed a 90 ms-old frame is fast and wrong at once, and only the gauge says so.

Cost: ``perf_counter`` is tens of nanoseconds, so instrumenting every stage of a
30 Hz loop is free at the resolution being measured.  It is always on -- timing
you have to switch on is timing you do not have when the bad run happens.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np


class StageTimer:
    """Accumulate named stage durations and gauge readings.

    Not thread-safe by design: each timer belongs to one loop.  The camera
    threads report through gauges taken from their own frame timestamps, so no
    cross-thread accumulation happens here.
    """

    def __init__(self, budget_ms: float | None = None, name: str = ""):
        self.name = name
        self.budget_ms = float(budget_ms) if budget_ms else None
        self._stages: dict[str, list[float]] = {}
        self._gauges: dict[str, list[float]] = {}
        self._order: list[str] = []
        self.last: dict[str, float] = {}

    @contextmanager
    def stage(self, label: str):
        began = time.perf_counter()
        try:
            yield
        finally:
            self.record(label, (time.perf_counter() - began) * 1000.0)

    def record(self, label: str, milliseconds: float) -> None:
        if label not in self._stages:
            self._stages[label] = []
            self._order.append(label)
        self._stages[label].append(float(milliseconds))
        self.last[label] = float(milliseconds)

    def gauge(self, label: str, value: float) -> None:
        """Record a non-duration reading (frame age, skew, ...)."""

        if value is None or not np.isfinite(value):
            return
        self._gauges.setdefault(label, []).append(float(value))
        self.last[label] = float(value)

    def reset(self) -> None:
        self._stages.clear()
        self._gauges.clear()
        self._order.clear()
        self.last.clear()

    # -- reading out ------------------------------------------------------
    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self._order) + tuple(self._gauges)

    def stats(self, label: str) -> dict[str, float]:
        values = np.asarray(self._stages.get(label) or self._gauges.get(label) or [])
        if values.size == 0:
            return {}
        return {"n": float(values.size), "mean": float(values.mean()),
                "p95": float(np.percentile(values, 95)), "max": float(values.max())}

    #: Wrapper stages that measure a whole tick rather than a step of work.
    #: Ranking them as bottlenecks would always name the wrapper.
    AGGREGATE_LABELS = ("total",)

    def slowest(self) -> tuple[str, float] | None:
        """The stage with the worst p95 -- the bottleneck, in one call."""

        ranked = [(l, self.stats(l).get("p95", 0.0)) for l in self._order
                  if l not in self.AGGREGATE_LABELS]
        return max(ranked, key=lambda x: x[1]) if ranked else None

    def unaccounted_ms(self) -> float | None:
        """Mean total minus the mean of its parts.

        The number that says the instrumentation is incomplete.  A tick whose
        parts sum to 1 ms inside a 12 ms total is not a fast tick -- it is 11 ms
        happening somewhere nobody is looking, which is exactly the case this
        whole module exists to prevent.
        """

        if "total" not in self._stages:
            return None
        if not any(k not in self.AGGREGATE_LABELS for k in self._stages):
            # A timer holding only the wrapper has no parts to account for, so
            # "unaccounted == total" would be noise rather than a finding.
            return None
        total = float(np.mean(self._stages["total"]))
        parts = sum(float(np.mean(v)) for k, v in self._stages.items()
                    if k not in self.AGGREGATE_LABELS)
        return total - parts

    def over_budget(self) -> int:
        """Ticks whose total exceeded the budget, if a total stage is recorded."""

        if self.budget_ms is None or "total" not in self._stages:
            return 0
        return int(np.sum(np.asarray(self._stages["total"]) > self.budget_ms))

    def report(self) -> str:
        if not self._stages and not self._gauges:
            return f"[TIMING{' ' + self.name if self.name else ''}] 기록 없음"

        lines = [f"[TIMING{' ' + self.name if self.name else ''}]"
                 + (f"  예산 {self.budget_ms:.1f} ms/틱" if self.budget_ms else "")]
        lines.append(f"  {'단계':<26}{'n':>6}{'평균':>10}{'p95':>10}{'최대':>10}"
                     + ("      예산%" if self.budget_ms else ""))
        for label in self._order:
            s = self.stats(label)
            if not s:
                continue
            share = (f"   {100.0 * s['p95'] / self.budget_ms:8.1f}%"
                     if self.budget_ms else "")
            lines.append(f"  {label:<26}{int(s['n']):6d}{s['mean']:9.2f}ms"
                         f"{s['p95']:9.2f}ms{s['max']:9.2f}ms{share}")
        for label in self._gauges:
            s = self.stats(label)
            lines.append(f"  {label:<26}{int(s['n']):6d}{s['mean']:9.2f}  "
                         f"{s['p95']:9.2f}  {s['max']:9.2f}   (게이지)")

        unaccounted = self.unaccounted_ms()
        if unaccounted is not None:
            share = (f" ({100.0 * unaccounted / self.budget_ms:.1f}% 예산)"
                     if self.budget_ms else "")
            lines.append(f"  {'(미계상)':<26}{'':>6}{unaccounted:9.2f}ms{share}"
                         + ("   <- 계측 안 된 구간" if unaccounted > 1.0 else ""))

        worst = self.slowest()
        if worst and worst[1] > 0:
            lines.append(f"  → 최대 병목: {worst[0]} (p95 {worst[1]:.2f} ms)")
        if self.budget_ms and "total" in self._stages:
            n = len(self._stages["total"])
            late = self.over_budget()
            lines.append(f"  → 예산 초과 {late}/{n} 틱"
                         + (f"  ({100.0 * late / n:.1f}%)" if n else ""))
        return "\n".join(lines)

    def csv_columns(self) -> tuple[str, ...]:
        return tuple(f"ms_{l}" for l in self._order) + tuple(f"g_{l}" for l in self._gauges)

    def csv_row(self) -> tuple[float, ...]:
        return tuple(self.last.get(l, float("nan")) for l in self._order) + \
               tuple(self.last.get(l, float("nan")) for l in self._gauges)
