#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${ROBOLAB_IMAGE:-$SCRIPT_DIR/robolab-isaaclab-2.2.0.sif}"
REPO="${ROBOLAB_REPO:-$SCRIPT_DIR}"
VENV="${UV_PROJECT_ENVIRONMENT:-$REPO/.venv}"

if [[ ! -f "$IMAGE" ]]; then
    echo "Apptainer image not found: $IMAGE" >&2
    echo "Set ROBOLAB_IMAGE=/path/to/robolab-isaaclab-2.2.0.sif or place it beside this script." >&2
    exit 1
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export UV_PROJECT_ENVIRONMENT="$VENV"

# Use bash -c (not -lc) to skip host .bashrc / conda init.
# Explicitly activate the venv so uv never falls back to host Python.
apptainer exec --nv \
    --bind "$REPO:/workspace/robolab" \
    --bind /gscratch:/gscratch \
    --pwd /workspace/robolab \
    "$IMAGE" \
    bash -c '
        export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
        export PATH="$VIRTUAL_ENV/bin:/usr/local/bin:$HOME/.local/bin:$PATH"
        unset CONDA_DEFAULT_ENV CONDA_PREFIX
        "$@"
    ' bash "$@"
