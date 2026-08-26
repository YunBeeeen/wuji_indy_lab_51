"""Gain, session, and JSON configuration persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path | str) -> dict[str, Any]:
    """Load and validate a top-level JSON object."""

    file_path = Path(path).expanduser().resolve()
    value = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {file_path}")
    return value


def save_json(path: Path | str, value: dict[str, Any]) -> Path:
    """Write a human-readable JSON object atomically enough for tuning output."""

    file_path = Path(path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = file_path.with_suffix(file_path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(file_path)
    return file_path


def build_gain_document(
    asset_file: str,
    asset_cfg_name: str,
    physics_dt: float,
    gains: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Create the portable gain-file schema without modifying the asset source."""

    return {
        "asset_file": str(Path(asset_file).expanduser().resolve()),
        "asset_cfg_name": asset_cfg_name,
        "physics_dt": float(physics_dt),
        "joints": gains,
    }
