#!/usr/bin/env bash
# Build the robolab Docker image (~2 min, ~42 GB).
set -euo pipefail

IMAGE_NAME="${ROBOLAB_REGISTRY:-robolab}"
PUSH=false
OPENPI_COMMIT=""

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --push) PUSH=true ;;
        --openpi-commit=*) OPENPI_COMMIT="${arg#*=}" ;;
        *) IMAGE_TAG="$arg" ;;
    esac
done

IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

ROBOLAB_DOCKER_DIR="$(dirname "$(realpath -s "$0")")"
ROBOLAB_DIR="$(realpath "${ROBOLAB_DOCKER_DIR}/../")"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG}"

BUILD_ARGS=()
if [ -n "$OPENPI_COMMIT" ]; then
    BUILD_ARGS+=(--build-arg "OPENPI_COMMIT=${OPENPI_COMMIT}")
fi

docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" \
             --network=host \
             "${BUILD_ARGS[@]}" \
             -f "${ROBOLAB_DOCKER_DIR}/Dockerfile" \
             "${ROBOLAB_DIR}"

echo "Built ${IMAGE_NAME}:${IMAGE_TAG}"

if [ "$PUSH" = true ]; then
    echo "Pushing ${IMAGE_NAME}:${IMAGE_TAG}"
    docker push "${IMAGE_NAME}:${IMAGE_TAG}"
    echo "Pushed ${IMAGE_NAME}:${IMAGE_TAG}"
fi
