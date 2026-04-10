FROM nvcr.io/nvidia/isaac-lab:2.2.0

# System packages
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

# openpi-client (pinned to openpi server commit, no local deps)
ARG OPENPI_COMMIT=aa6420561529593114160d05e5ad155792b272f3
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-cache-dir --no-deps \
    "openpi-client @ git+https://github.com/xuningy/openpi@${OPENPI_COMMIT}#subdirectory=packages/openpi-client"

# Assets — each subdirectory is a separate layer for better caching.
# Placed before code COPY so they stay cached across code changes.
# Ordered smallest to largest so cheaper layers come first.
COPY assets/robots/      /workspace/robolab/assets/robots/
COPY assets/materials/   /workspace/robolab/assets/materials/
COPY assets/fixtures/    /workspace/robolab/assets/fixtures/
COPY assets/backgrounds/ /workspace/robolab/assets/backgrounds/
COPY assets/scenes/      /workspace/robolab/assets/scenes/
COPY assets/objects/     /workspace/robolab/assets/objects/

# Accept NVIDIA/Isaac Sim EULAs
ENV OMNI_KIT_ACCEPT_EULA=YES
ENV ACCEPT_EULA=Y

# Copy project code (assets/ excluded via .dockerignore)
COPY . /workspace/robolab
WORKDIR /workspace/robolab

# Install Python dependencies
# GIT_LFS_SKIP_SMUDGE=1 avoids pulling large LFS objects during build
RUN GIT_LFS_SKIP_SMUDGE=1 /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-cache-dir -e .
