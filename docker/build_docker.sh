#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the robolab Docker image (~2 min, ~42 GB).
set -euo pipefail

IMAGE_NAME="${ROBOLAB_REGISTRY:-robolab}"
PUSH=false
OPENPI_COMMIT=""
# IsaacLab/IsaacSim stack selector -> base image tag (see docker/Dockerfile).
# 2.2.0 = IsaacSim 5.0 (default); newer stacks remain selectable.
ISAACLAB_TAG="2.2.0"
TAG_SUFFIX=""
RUNTIME_USER="0:0"

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --push) PUSH=true ;;
        --openpi-commit=*) OPENPI_COMMIT="${arg#*=}" ;;
        --isaac50) ISAACLAB_TAG="2.2.0"; TAG_SUFFIX=""; RUNTIME_USER="0:0" ;;
        --isaac51) ISAACLAB_TAG="2.3.0"; TAG_SUFFIX="-isaac51"; RUNTIME_USER="0:0" ;;
        --isaac60) ISAACLAB_TAG="3.0.0-beta2"; TAG_SUFFIX="-isaac60"; RUNTIME_USER="1000:1000" ;;
        --isaaclab-tag=*) ISAACLAB_TAG="${arg#*=}"; TAG_SUFFIX="-isaaclab-${arg#*=}" ;;
        *) IMAGE_TAG="$arg" ;;
    esac
done

# Default the image tag to the git HEAD, suffixed by the stack so simulator
# images from the same commit don't collide in the registry.
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)${TAG_SUFFIX}}"

ROBOLAB_DOCKER_DIR="$(dirname "$(realpath -s "$0")")"
ROBOLAB_DIR="$(realpath "${ROBOLAB_DOCKER_DIR}/../")"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG}"

echo "Using IsaacLab base image tag: ${ISAACLAB_TAG}"

BUILD_ARGS=(
    --build-arg "ISAACLAB_TAG=${ISAACLAB_TAG}"
    --build-arg "ROBOLAB_RUNTIME_USER=${RUNTIME_USER}"
)
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
