#!/usr/bin/env bash
set -euo pipefail

TUNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ISAACLAB_PYTHON:-}" ]]; then
    PYTHON_CMD=("${ISAACLAB_PYTHON}")
elif python -c 'import isaaclab, isaacsim' >/dev/null 2>&1; then
    PYTHON_CMD=(python)
elif [[ -n "${ISAACLAB_ROOT:-}" && -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
    PYTHON_CMD=("${ISAACLAB_ROOT}/isaaclab.sh" -p)
elif [[ -n "${ISAACSIM_PATH:-}" && -x "${ISAACSIM_PATH}/python.sh" ]]; then
    PYTHON_CMD=("${ISAACSIM_PATH}/python.sh")
else
    echo "No compatible Isaac Lab Python found. Set ISAACLAB_PYTHON, ISAACLAB_ROOT, or ISAACSIM_PATH." >&2
    exit 2
fi

printf 'GUI packages will be installed only into: %q ' "${PYTHON_CMD[@]}"
printf '\nRequirements: %s\n' "${TUNER_DIR}/requirements-gui.txt"
read -r -p "Continue? [y/N] " answer
if [[ ! "${answer}" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi
"${PYTHON_CMD[@]}" -m pip install -r "${TUNER_DIR}/requirements-gui.txt"
