# [tool] 학습 런의 params/env.yaml 을 배포 계약과 대조. 체크포인트를 갈아끼우기 전에 돌릴 것.
"""Check that a checkpoint was TRAINED on the contract this package deploys.

Swapping ``--policy`` already works: the ONNX is loaded by path and
``OnnxPolicy`` rejects a file whose input/output shapes are not 105/20.  But
shape is almost nothing.  The network carries weights and dimensions; it does
NOT carry the pregrasp pose it started from, the per-joint action scale that
turns its output into radians, the joint ORDER those twenty numbers are in, the
normalization range its inputs were divided by, or the OPEN/CLOSE schedule it
saw.  Every one of those lives in ``common/policy_contract.py`` and
``common/isaac_reset.py`` as a hard-coded constant, and every one of them can
differ between two runs that are both "105D in, 20D out".

A mismatch there does not raise.  It produces a hand that moves smoothly and
wrongly -- the single most expensive failure mode this stack has, because it
looks like a tuning problem.

Isaac writes ``params/env.yaml`` beside every run.  That file states what the
policy was actually trained with, so this tool reads it and diffs it against
the deployed constants.  Run it whenever a checkpoint changes.

Deliberately NOT automatic on every deploy run: env.yaml lives next to the
training log, which may be pruned or moved, and a missing file must not stop a
run that is otherwise fine.  This is a check you run, not a gate you trip over.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from ..common.isaac_reset import ISAAC_PREGRASP_JOINT_POSITIONS_RAD
from ..common.policy_contract import (
    ACTION_DIM,
    ACTION_SCALE_RAD,
    COMMAND_LIMIT_RATIO,
    OBSERVATION_DIM,
    OBSERVATION_NORMALIZATION_LIMITS,
    POLICY_JOINT_NAMES,
)

#: Isaac's own schedule fields, and the deploy CLI defaults that mirror them.
SCHEDULE_DEFAULTS = {
    "open_close_start_time_s": ("--open-lead-seconds", 5.0),
    "open_close_segment_time_s": ("--segment-seconds", 2.0),
}


def find_env_yaml(target: Path) -> Path:
    """Accept a run dir, a params dir, an ONNX path, or the yaml itself."""

    target = target.expanduser().resolve()
    if target.is_file() and target.suffix in (".yaml", ".yml"):
        return target
    for candidate in (
        target / "params" / "env.yaml",
        target / "env.yaml",
        target.parent / "params" / "env.yaml",          # .../exported/policy.onnx
        target.parent.parent / "params" / "env.yaml",
    ):
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"No params/env.yaml found for {target}.  Pass the training run "
        "directory, or the yaml itself.")


def _block(text: str, header: str) -> str:
    """Return the indented block under ``header`` at column 0 or 2.

    Parsed by text rather than by ``yaml.load``: env.yaml embeds
    ``!!python/tuple`` and ``!!python/object`` tags, so a safe loader refuses it
    and an unsafe loader would execute constructors from a file this tool is
    supposed to be inspecting rather than trusting.
    """

    match = re.search(rf"^(\s*){re.escape(header)}:\s*$", text, re.M)
    if match is None:
        return ""
    indent = len(match.group(1))
    lines = []
    for line in text[match.end():].splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        depth = len(line) - len(line.lstrip())
        # A YAML sequence is written at the SAME indent as its key, so a
        # dedent-to-equal only ends the block when the line is not a list item.
        # Getting this wrong returns an empty block, which then reads as "the
        # trained config has zero joints" rather than as a parse failure.
        if depth < indent or (depth == indent and not stripped.startswith("-")):
            break
        lines.append(line)
    return "\n".join(lines)


def _scalar(text: str, key: str):
    match = re.search(rf"^\s*{re.escape(key)}:\s*(\S+)\s*$", text, re.M)
    return None if match is None else match.group(1)


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        self.failures += not ok
        print(f"  [{'OK  ' if ok else 'DIFF'}] {label}" + (f"  {detail}" if detail else ""))
        return ok

    def skip(self, label: str, why: str) -> None:
        print(f"  [ -- ] {label}  {why}")


def verify(env_yaml: Path) -> int:
    text = env_yaml.read_text()
    report = Report()
    print(f"[SOURCE]   {env_yaml}")

    action = _block(text, "hand_action")
    print("\n[ACTION]   액션 계약")
    names = re.findall(r"^\s*-\s*(finger\d_joint\d)\s*$", _block(action, "joint_names"), re.M)
    report.check("관절 순서 (preserve_order)", tuple(names) == POLICY_JOINT_NAMES,
                 "" if tuple(names) == POLICY_JOINT_NAMES else
                 f"학습 {len(names)}개, 배포 {ACTION_DIM}개")
    report.check("preserve_order: true", _scalar(action, "preserve_order") == "true",
                 f"학습값 {_scalar(action, 'preserve_order')}")
    report.check("clamp_to_limits: true", _scalar(action, "clamp_to_limits") == "true",
                 f"학습값 {_scalar(action, 'clamp_to_limits')}")
    # A scalar `null` means no override.  A nested BLOCK means the old cosmetic
    # Joint4=0 floor is present, which freezes five distal joints the deployed
    # contract lets go negative -- _scalar returns None for both, so the two
    # cases have to be told apart or the report reads as a parse failure.
    overrides = _block(action, "joint_position_lower_overrides")
    override_scalar = _scalar(action, "joint_position_lower_overrides")
    if overrides.strip():
        detail = ("학습에 하한 override 가 있음: "
                  + ", ".join(sorted(set(re.findall(r"(finger\d_joint\d)", overrides)))))
        report.check("joint4 하한 override 없음", False, detail)
    else:
        report.check("joint4 하한 override 없음", override_scalar == "null",
                     f"학습값 {override_scalar}")

    trained_scale = {
        m.group(1): float(m.group(2))
        for m in re.finditer(r"^\s*(finger\d_joint\d):\s*([\d.eE+-]+)\s*$",
                             _block(action, "scale"), re.M)
    }
    if len(trained_scale) == ACTION_DIM:
        deployed = {n: float(s) for n, s in zip(POLICY_JOINT_NAMES, ACTION_SCALE_RAD)}
        # float32 round-trip: 0.1 stores as 0.10000000149011612.  Comparing at
        # 1e-9 flags all twenty as mismatched, which is noise, not a finding.
        bad = {k: (v, deployed[k]) for k, v in trained_scale.items()
               if abs(v - deployed[k]) > 1e-6}
        report.check("관절별 액션 스케일", not bad,
                     "" if not bad else
                     "; ".join(f"{k} 학습 {a} != 배포 {b}" for k, (a, b) in bad.items()))
    elif not trained_scale:
        # A SCALAR scale means one number for all twenty joints -- what
        # hand_move used before HAND_REAL_ACTION_SCALE.  Deploying such a
        # checkpoint through this contract would move Joint3 twice and Joint4
        # 1.5x as far as it was trained to.  Silent, and it looks like the
        # policy being aggressive.
        scalar = _scalar(action, "scale")
        uniform = scalar is not None and scalar not in ("null", "None")
        distinct = "/".join(f"{v:g}" for v in dict.fromkeys(
            float(x) for x in ACTION_SCALE_RAD))
        report.check(
            "관절별 액션 스케일", not uniform,
            f"학습은 전 관절 단일 {scalar} -- 배포는 관절별 {distinct}"
            if uniform else f"scale 을 읽지 못함 (값 {scalar})")
    else:
        report.skip("관절별 액션 스케일", f"scale 항목 {len(trained_scale)}개만 읽힘")

    print("\n[OBS]      관측 계약")
    policy_obs = _block(_block(text, "observations"), "policy")
    terms = re.findall(r"^    ([a-z_]+):\s*$", policy_obs, re.M)
    expected = ["joint_pos_history", "fingertip_pos", "last_action", "open_close_mode"]
    missing = [t for t in expected if t not in terms]
    report.check("관측 term 구성", not missing,
                 f"항목 {terms}" if missing else f"{len(terms)}개 term")
    histories = [int(m) for m in re.findall(r"^\s*history_length:\s*(\d+)\s*$", policy_obs, re.M)]
    report.check("history_length 2가 3개 (q, stick1, stick2)",
                 histories.count(2) == 3, f"실제 {histories}")

    print("\n[POSE]     리셋/기동 자세")
    reset_block = _block(text, "reset_pregrasp")
    positions = _block(reset_block, "joint_positions")
    if not positions:
        # ``joint_positions: !!python/tuple`` -- the tag sits on the key line,
        # so the bare-key regex misses it.  Take everything after the key.
        match = re.search(r"^(\s*)joint_positions:.*$", reset_block, re.M)
        positions = reset_block[match.end():] if match else ""
    pregrasp = [float(v) for v in re.findall(
        r"^\s*-\s*(-?[\d.]+(?:[eE]-?\d+)?)\s*$", positions, re.M)][:ACTION_DIM]
    if len(pregrasp) == ACTION_DIM:
        delta = np.abs(np.asarray(pregrasp, dtype=np.float32)
                       - ISAAC_PREGRASP_JOINT_POSITIONS_RAD)
        worst = int(np.argmax(delta))
        report.check("pregrasp 자세", float(delta.max()) < 1e-5,
                     f"최대 차 {1000.0 * delta.max():.2f} mrad ({POLICY_JOINT_NAMES[worst]})")
    else:
        report.skip("pregrasp 자세", f"{len(pregrasp)}개만 읽힘 (20 필요)")

    print("\n[SCHEDULE] OPEN/CLOSE 스케줄 -- CLI 기본값과 대조")
    # The mode the policy actually saw, which is NOT decided by the schedule
    # fields alone.  `neutral_before_open_close` emits [0, 0] until
    # open_close_start_time_s, so a task whose episode ENDS at that boundary
    # trains on NEUTRAL and never sees OPEN or CLOSE.  Nothing about the
    # observation shape reveals this, and driving such a checkpoint with the
    # OPEN/CLOSE schedule feeds it an input outside its training set --
    # hand_real2's 5 s grasp curriculum is exactly that case (2026-08-23).
    episode = _scalar(text, "episode_length_s")
    start = _scalar(text, "open_close_start_time_s")
    neutral = (_scalar(_block(_block(text, "commands"), "open_close"),
                       "neutral_before_open_close") == "true")
    if episode is not None and start is not None:
        never_alternates = float(episode) <= float(start)
        if never_alternates and neutral:
            report.check("학습이 본 모드", False,
                         f"episode_length_s {episode} <= open_close_start {start} "
                         f"이고 neutral_before_open_close=true -- 이 정책은 "
                         f"NEUTRAL [0,0] 만 봤습니다. --mode neutral 로 배포할 것 "
                         f"(--mode schedule/open/close 는 학습 밖 입력).")
        elif never_alternates:
            report.check("학습이 본 모드", False,
                         f"episode_length_s {episode} <= open_close_start {start} "
                         f"-- 교대 구간에 도달하지 못했습니다. --mode open 고정으로 "
                         f"배포할 것.")
        else:
            first = "NEUTRAL" if neutral else "OPEN"
            report.check("학습이 본 모드", True,
                         f"{first} {start}s 리드 -> 교대 (episode {episode}s)")
            if neutral:
                print(f"         └ 리드 구간이 NEUTRAL 입니다. 배포 스케줄의 "
                      f"리드는 OPEN 이라 첫 {start}s 가 다릅니다.")
    for key, (flag, default) in SCHEDULE_DEFAULTS.items():
        raw = _scalar(text, key)
        if raw is None:
            report.skip(f"{key}", "env.yaml 에 없음")
            continue
        report.check(f"{key} -> {flag}", abs(float(raw) - default) < 1e-9,
                     f"학습 {raw} vs 배포 기본 {default}  ({flag} 로 맞출 것)")

    print("\n[LIMITS]   한계")
    print(f"  [ -- ] 정규화 범위  env.yaml 에 없음 -- USD "
          f"(wuji_right_filtered.usda) 가 출처. 상한 예: finger5_joint3 "
          f"{OBSERVATION_NORMALIZATION_LIMITS[18, 1]:+.6f}")
    print(f"  [note] 명령 한계는 배포에서 x{COMMAND_LIMIT_RATIO} 로 좁혀져 있습니다 "
          f"-- 학습(1.0)과 의도적으로 다릅니다.")

    print(f"\n[결과]     {'전부 일치' if not report.failures else f'{report.failures}건 불일치'}"
          f"  (obs {OBSERVATION_DIM}D / action {ACTION_DIM}D)")
    if report.failures:
        print("           불일치 항목은 ONNX 로드로는 절대 안 잡힙니다 -- "
              "손이 부드럽게 틀리게 움직입니다.")
    return 1 if report.failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diff a training run's env.yaml against the deployed contract.")
    parser.add_argument("run", type=Path,
                        help="학습 런 디렉터리, params/env.yaml, 또는 policy.onnx 경로.")
    args = parser.parse_args()
    return verify(find_env_yaml(args.run))


if __name__ == "__main__":
    raise SystemExit(main())
