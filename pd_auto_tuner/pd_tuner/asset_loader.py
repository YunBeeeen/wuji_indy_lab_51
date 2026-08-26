"""Dynamic discovery and import of user-supplied Isaac Lab asset configs."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class AssetConfigSummary:
    """A valid articulation configuration found in a Python module."""

    name: str
    class_name: str
    actuator_group_names: tuple[str, ...]


class AssetImportError(RuntimeError):
    """Raised with user-facing context when an asset module cannot be imported."""

    def __init__(self, asset_file: Path, cause: BaseException):
        missing = cause.name if isinstance(cause, ModuleNotFoundError) else None
        details = f"Failed to import robot asset: {asset_file}\n{type(cause).__name__}: {cause}"
        if missing:
            details += (
                f"\n\nMissing module: {missing}\n"
                "Select the correct Project root or install the package in the "
                "Isaac Lab Python environment."
            )
        super().__init__(details)
        self.asset_file = asset_file
        self.cause = cause
        self.missing_module = missing


def default_asset_directories(search_root: Path) -> list[Path]:
    """Find conventional asset directories without requiring one layout."""

    root = search_root.expanduser().resolve()
    candidates: list[Path] = []
    for name in ("assets", "robots", "robot_assets"):
        path = root / name
        if path.is_dir():
            candidates.append(path)
    source = root / "source"
    if source.is_dir():
        candidates.extend(path for path in source.glob("*/assets") if path.is_dir())
    # Also support a common installed-project layout without naming a project.
    candidates.extend(
        path
        for path in root.glob("*/assets")
        if path.is_dir() and path not in candidates
    )
    return sorted(set(candidates))


def discover_asset_files(asset_directory: Path, recursive: bool = True) -> list[Path]:
    """Return importable-looking Python files under an arbitrary directory."""

    directory = asset_directory.expanduser().resolve()
    if not directory.is_dir():
        return []
    iterator: Iterable[Path] = directory.rglob("*.py") if recursive else directory.glob("*.py")
    return sorted(
        path
        for path in iterator
        if path.name != "__init__.py" and not path.name.startswith("_")
    )


def _add_import_paths(asset_file: Path, project_root: Path | None) -> None:
    paths = [asset_file.parent]
    if project_root is not None:
        paths.insert(0, project_root)
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _module_name_from_root(asset_file: Path, project_root: Path | None) -> str | None:
    if project_root is None:
        return None
    try:
        relative = asset_file.relative_to(project_root).with_suffix("")
    except ValueError:
        return None
    if all(part.isidentifier() for part in relative.parts):
        return ".".join(relative.parts)
    return None


def load_asset_module(asset_file: Path | str, project_root: Path | str | None = None) -> ModuleType:
    """Dynamically load an asset file, preserving package imports when possible."""

    file_path = Path(asset_file).expanduser().resolve()
    root_path = Path(project_root).expanduser().resolve() if project_root else None
    if not file_path.is_file():
        raise FileNotFoundError(f"Asset Python file not found: {file_path}")
    _add_import_paths(file_path, root_path)
    importlib.invalidate_caches()
    module_name = _module_name_from_root(file_path, root_path)
    try:
        if module_name:
            existing = sys.modules.get(module_name)
            if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == file_path:
                return importlib.reload(existing)
            return importlib.import_module(module_name)

        digest = hashlib.sha1(str(file_path).encode("utf-8")).hexdigest()[:12]
        unique_name = f"pd_tuner_user_asset_{digest}"
        spec = importlib.util.spec_from_file_location(unique_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create an import spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = module
        spec.loader.exec_module(module)
        return module
    except BaseException as exc:
        raise AssetImportError(file_path, exc) from exc


def articulation_configs(module: ModuleType) -> dict[str, Any]:
    """Return every concrete ``ArticulationCfg`` object exposed by a module."""

    try:
        from isaaclab.assets import ArticulationCfg
    except ImportError as exc:
        raise RuntimeError(
            "Isaac Lab is not importable in this Python environment; cannot validate ArticulationCfg objects."
        ) from exc
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_") and isinstance(value, ArticulationCfg)
    }


def inspect_asset_file(
    asset_file: Path | str,
    project_root: Path | str | None = None,
) -> list[AssetConfigSummary]:
    """Import a file and summarize its valid articulation configs."""

    module = load_asset_module(asset_file, project_root)
    configs = articulation_configs(module)
    return [
        AssetConfigSummary(
            name=name,
            class_name=type(cfg).__name__,
            actuator_group_names=tuple(cfg.actuators.keys()),
        )
        for name, cfg in sorted(configs.items())
    ]


def resolve_asset_cfg(
    asset_file: Path | str,
    cfg_name: str,
    project_root: Path | str | None = None,
) -> Any:
    """Load and return one named, validated articulation configuration."""

    module = load_asset_module(asset_file, project_root)
    configs = articulation_configs(module)
    if cfg_name not in configs:
        available = ", ".join(sorted(configs)) or "<none>"
        raise KeyError(f"ArticulationCfg {cfg_name!r} not found. Available: {available}")
    return configs[cfg_name]
