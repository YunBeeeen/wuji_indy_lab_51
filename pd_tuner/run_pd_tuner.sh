#!/usr/bin/env bash
set -euo pipefail

TUNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${TUNER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${ISAACLAB_PYTHON:-}" ]]; then
    PYTHON_CMD=("${ISAACLAB_PYTHON}")
elif python -c 'import isaaclab, isaacsim' >/dev/null 2>&1; then
    PYTHON_CMD=(python)
elif [[ -n "${ISAACLAB_ROOT:-}" && -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    PYTHON_CMD=("${ISAACLAB_ROOT}/isaaclab.sh" -p)
elif [[ -n "${ISAACSIM_PATH:-}" && -x "${ISAACSIM_PATH}/python.sh" ]]; then
    PYTHON_CMD=("${ISAACSIM_PATH}/python.sh")
else
    cat >&2 <<'EOF'
Could not find a Python executable that imports both Isaac Lab and Isaac Sim.

Checked:
  1. ISAACLAB_PYTHON
  2. the active `python`
  3. $ISAACLAB_ROOT/isaaclab.sh -p
  4. $ISAACSIM_PATH/python.sh

Examples:
  ISAACLAB_ROOT=/path/to/IsaacLab ./run_pd_tuner.sh
  ISAACLAB_PYTHON=/path/to/python.sh ./run_pd_tuner.sh
EOF
    exit 2
fi

if ! "${PYTHON_CMD[@]}" -c 'import isaaclab, isaacsim' >/dev/null 2>&1; then
    printf 'Selected Python command failed Isaac imports: %q ' "${PYTHON_CMD[@]}" >&2
    printf '\nSet ISAACLAB_ROOT, ISAACSIM_PATH, or ISAACLAB_PYTHON correctly.\n' >&2
    exit 2
fi

exec "${PYTHON_CMD[@]}" -m pd_tuner "$@"
