#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${ROBOLAB_REGISTRY:-robolab}"
IMAGE_TAG="${1:-$(git rev-parse --short HEAD)}"

docker run --rm -it \
    --gpus all \
    --network=host \
    --entrypoint /bin/bash \
    -w /workspace/robolab \
    "${IMAGE_NAME}:${IMAGE_TAG}"
