"""Auto Tune JSON and candidate-summary CSV persistence."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..gain_io import save_json
from .config import ResolvedAutoTuneConfig
from .result import AutoTuneOutcome


CANDIDATE_CSV_FIELDS = (
    "rank",
    "candidate_id",
    "stage",
    "generation_reason",
    "identified_natural_frequency",
    "identified_damping_ratio",
    "identified_effective_inertia",
    "identification_normalized_rmse",
    "status",
    "kp",
    "kd",
    "direction",
    "settling_time",
    "percentage_overshoot",
    "steady_state_error",
    "rms_computed_effort",
    "rms_applied_effort",
    "peak_computed_effort",
    "peak_applied_effort",
    "maximum_velocity",
    "saturation_count",
    "total_score",
    "failure_reason",
)

def build_autotune_document(
    *,
    metadata: dict[str, Any],
    version_info: dict[str, Any],
    config: ResolvedAutoTuneConfig,
    outcome: AutoTuneOutcome,
    original_gains: dict[str, float],
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a complete, portable Auto Tune result without touching assets."""

    selected = outcome.selected
    return {
        "schema": "isaaclab_pd_autotune_v1",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "asset_file": metadata.get("asset_file"),
        "asset_cfg_name": metadata.get("asset_cfg_name"),
        "joint_name": config.joint_name,
        "actuator_group": config.actuator_group,
        "isaac_sim_version": version_info.get("isaac_sim", "unknown"),
        "isaac_lab_version": version_info.get("isaac_lab", "unknown"),
        "physics_dt": config.physics_dt,
        "resolved_configuration": config.to_dict(),
        "value_sources": {key: value.value for key, value in config.value_sources.items()},
        "search_strategy": "response_identification_guided_v3",
        "search_budget": config.search_budget,
        "preset": config.preset.value,
        "torque_policy": config.torque_policy.value,
        "custom_weights": dict(config.custom_weights),
        "original_gains": dict(original_gains),
        "selected_gains": (
            {
                "stiffness": selected.kp,
                "damping": selected.kd,
                "effort_limit": config.effort_limit,
            }
            if selected is not None
            else None
        ),
        "outcome": outcome.to_dict(),
    }


def build_autotune_document_from_payload(
    *,
    metadata: dict[str, Any],
    version_info: dict[str, Any],
    resolved_configuration: dict[str, Any],
    outcome: dict[str, Any],
    original_gains: dict[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the same result schema from child-process event payloads."""

    selected_id = outcome.get("selected_candidate_id")
    selected = next(
        (
            result
            for result in outcome.get("candidates", [])
            if result.get("candidate", {}).get("candidate_id") == selected_id
        ),
        None,
    )
    selected_gains = None
    if selected is not None:
        selected_gains = {
            "stiffness": selected["candidate"]["kp"],
            "damping": selected["candidate"]["kd"],
            "effort_limit": resolved_configuration.get("effort_limit"),
        }
    return {
        "schema": "isaaclab_pd_autotune_v1",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "asset_file": metadata.get("asset_file"),
        "asset_cfg_name": metadata.get("asset_cfg_name"),
        "joint_name": resolved_configuration.get("joint_name"),
        "actuator_group": resolved_configuration.get("actuator_group"),
        "isaac_sim_version": version_info.get("isaac_sim", "unknown"),
        "isaac_lab_version": version_info.get("isaac_lab", "unknown"),
        "physics_dt": resolved_configuration.get("physics_dt", metadata.get("physics_dt")),
        "resolved_configuration": dict(resolved_configuration),
        "value_sources": dict(resolved_configuration.get("value_sources", {})),
        "search_strategy": "response_identification_guided_v3",
        "search_budget": resolved_configuration.get("search_budget"),
        "preset": resolved_configuration.get("preset"),
        "torque_policy": resolved_configuration.get("torque_policy"),
        "custom_weights": dict(resolved_configuration.get("custom_weights", {})),
        "original_gains": dict(original_gains),
        "selected_gains": selected_gains,
        "outcome": dict(outcome),
    }


def save_autotune_json(path: Path | str, document: dict[str, Any]) -> Path:
    return save_json(path, document)


def save_candidate_summary_csv(path: Path | str, outcome: AutoTuneOutcome) -> Path:
    """Write one row per candidate, ordered by selection score/status."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        outcome.candidates,
        key=lambda item: (
            0 if item is outcome.selected else 1,
            item.total_score if item.total_score is not None else float("inf"),
            item.candidate.candidate_id,
        ),
    )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_CSV_FIELDS)
        writer.writeheader()
        for rank, result in enumerate(ordered, start=1):
            failure = result.hard_failure_reasons or result.performance_violations
            estimate = result.candidate.model_estimate or {}
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": result.candidate.candidate_id,
                    "stage": result.candidate.stage,
                    "generation_reason": result.candidate.generation_reason,
                    "identified_natural_frequency": estimate.get("natural_frequency"),
                    "identified_damping_ratio": estimate.get("damping_ratio"),
                    "identified_effective_inertia": estimate.get("effective_inertia"),
                    "identification_normalized_rmse": estimate.get("normalized_rmse"),
                    "status": result.status.value,
                    "kp": result.kp,
                    "kd": result.kd,
                    "direction": result.direction,
                    "settling_time": result.settling_time,
                    "percentage_overshoot": result.percentage_overshoot,
                    "steady_state_error": result.steady_state_error,
                    "rms_computed_effort": result.rms_computed_effort,
                    "rms_applied_effort": result.rms_applied_effort,
                    "peak_computed_effort": result.peak_computed_effort,
                    "peak_applied_effort": result.peak_applied_effort,
                    "maximum_velocity": result.maximum_velocity,
                    "saturation_count": result.saturation_count,
                    "total_score": result.total_score,
                    "failure_reason": " | ".join(failure),
                }
            )
    return destination


def save_candidate_summary_payload_csv(path: Path | str, outcome: dict[str, Any]) -> Path:
    """Write candidate rows received from the simulation child."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected_id = outcome.get("selected_candidate_id")
    candidates = sorted(
        outcome.get("candidates", []),
        key=lambda item: (
            0 if item["candidate"]["candidate_id"] == selected_id else 1,
            item.get("total_score") if item.get("total_score") is not None else float("inf"),
            item["candidate"]["candidate_id"],
        ),
    )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_CSV_FIELDS)
        writer.writeheader()
        for rank, result in enumerate(candidates, start=1):
            candidate = result["candidate"]
            estimate = candidate.get("model_estimate") or {}
            failure = result.get("hard_failure_reasons") or result.get("performance_violations") or []
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": candidate["candidate_id"],
                    "stage": candidate["stage"],
                    "generation_reason": candidate.get("generation_reason"),
                    "identified_natural_frequency": estimate.get("natural_frequency"),
                    "identified_damping_ratio": estimate.get("damping_ratio"),
                    "identified_effective_inertia": estimate.get("effective_inertia"),
                    "identification_normalized_rmse": estimate.get("normalized_rmse"),
                    "status": result["status"],
                    "kp": candidate["kp"],
                    "kd": candidate["kd"],
                    "direction": result["direction"],
                    "settling_time": result.get("settling_time"),
                    "percentage_overshoot": result.get("percentage_overshoot"),
                    "steady_state_error": result.get("steady_state_error"),
                    "rms_computed_effort": result.get("rms_computed_effort"),
                    "rms_applied_effort": result.get("rms_applied_effort"),
                    "peak_computed_effort": result.get("peak_computed_effort"),
                    "peak_applied_effort": result.get("peak_applied_effort"),
                    "maximum_velocity": result.get("maximum_velocity"),
                    "saturation_count": result.get("saturation_count"),
                    "total_score": result.get("total_score"),
                    "failure_reason": " | ".join(failure),
                }
            )
    return destination
