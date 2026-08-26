"""Module entry point for ``python -m pd_tuner``."""

from __future__ import annotations

import multiprocessing as mp

from .launcher import main


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
