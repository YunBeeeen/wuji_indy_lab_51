"""Short-lived Isaac-enabled child used to validate asset configuration objects."""

from __future__ import annotations

from dataclasses import asdict
import time
import traceback
from typing import Any


def run_asset_inspector(asset_file: str, project_root: str | None, result_queue: Any) -> None:
    """Start Isaac headless, dynamically import one module, and return valid configs."""

    simulation_app = None
    try:
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher({"headless": True}).app
        from .asset_loader import inspect_asset_file

        summaries = inspect_asset_file(asset_file, project_root)
        result_queue.put(
            {
                "ok": True,
                "asset_file": asset_file,
                "summaries": [asdict(summary) for summary in summaries],
                "wall_time_sent": time.time(),
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "asset_file": asset_file,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "wall_time_sent": time.time(),
            }
        )
    finally:
        # Queue.put() uses a background feeder thread. Flush it before
        # SimulationApp tears down the process-wide Kit framework, otherwise
        # the GUI can remain stuck on "Inspecting…" even though import passed.
        try:
            result_queue.close()
            result_queue.join_thread()
        except (AttributeError, OSError):
            pass
        if simulation_app is not None:
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
