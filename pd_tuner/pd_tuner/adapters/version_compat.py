"""Runtime capability checks and environment version reporting."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any


REQUIRED_ARTICULATION_METHODS = (
    "write_joint_stiffness_to_sim",
    "write_joint_damping_to_sim",
    "write_joint_effort_limit_to_sim",
    "set_joint_position_target",
    "write_data_to_sim",
)


def validate_articulation_api(articulation: Any) -> None:
    """Fail clearly when the installed Isaac Lab lacks required public APIs."""

    missing = [name for name in REQUIRED_ARTICULATION_METHODS if not callable(getattr(articulation, name, None))]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "This Isaac Lab version lacks runtime articulation APIs required by the tuner: " + joined
        )


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def collect_version_info() -> dict[str, str]:
    """Collect portable environment information after Isaac Sim starts."""

    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda or "unavailable")
        cuda_device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    except Exception:
        torch_version = "unavailable"
        cuda_version = "unavailable"
        cuda_device = "unavailable"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ubuntu": platform.freedesktop_os_release().get("PRETTY_NAME", "unknown"),
        "isaac_lab": _package_version("isaaclab"),
        "isaac_sim": _package_version("isaacsim"),
        "pyside6": _package_version("PySide6"),
        "pyqtgraph": _package_version("pyqtgraph"),
        "torch": torch_version,
        "cuda": cuda_version,
        "cuda_device": cuda_device,
    }
