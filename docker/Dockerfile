# IsaacLab/IsaacSim stack is selected by the base image tag:
#   2.2.0        -> Isaac Lab 2.2.0 / Isaac Sim 5.0 (default)
#   2.3.0        -> Isaac Lab 2.3.x / Isaac Sim 5.1
#   3.0.0-beta2  -> Isaac Lab 3.0 beta 2 / Isaac Sim 6.0
# The project's isaac extras are NOT installed here: the
# base image already provides the matching IsaacLab/IsaacSim, so `pip install
# -e .` only pulls the non-isaac deps.
ARG ISAACLAB_TAG=2.2.0
FROM nvcr.io/nvidia/isaac-lab:${ISAACLAB_TAG}
ARG ROBOLAB_RUNTIME_USER=0:0

# Temporarily elevate only for system packages; Isaac Lab 3 defaults to UID 1000.
USER 0:0
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    htop \
    iputils-ping \
    git-lfs \
    nvtop \
    tmux \
    vim \
    zip \
    && rm -rf /var/lib/apt/lists/*
USER ${ROBOLAB_RUNTIME_USER}

# openpi-client (pinned to openpi server commit, no local deps)
ARG OPENPI_COMMIT=aa6420561529593114160d05e5ad155792b272f3
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-cache-dir --no-deps \
    "openpi-client @ git+https://github.com/xuningy/openpi@${OPENPI_COMMIT}#subdirectory=packages/openpi-client"

# Assets — each subdirectory is a separate layer for better caching.
# Placed before code COPY so they stay cached across code changes.
# Ordered smallest to largest so cheaper layers come first.
COPY --chown=1000:1000 assets/robots/      /workspace/robolab/assets/robots/
COPY --chown=1000:1000 assets/materials/   /workspace/robolab/assets/materials/
COPY --chown=1000:1000 assets/fixtures/    /workspace/robolab/assets/fixtures/
COPY --chown=1000:1000 assets/backgrounds/ /workspace/robolab/assets/backgrounds/
COPY --chown=1000:1000 assets/scenes/      /workspace/robolab/assets/scenes/
COPY --chown=1000:1000 assets/objects/     /workspace/robolab/assets/objects/

# Accept NVIDIA/Isaac Sim EULAs
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y

# Copy project code (assets/ excluded via .dockerignore)
COPY --chown=1000:1000 . /workspace/robolab
WORKDIR /workspace/robolab

# Install Python dependencies
# GIT_LFS_SKIP_SMUDGE=1 avoids pulling large LFS objects during build
RUN GIT_LFS_SKIP_SMUDGE=1 /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-cache-dir -e .
