# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Evaluation orchestration: episode runner plus the inference ABC, with the
factory / concrete clients re-exported from the top-level
:mod:`robolab_policy_client` package for convenience.
"""

from .base_client import InferenceClient
from .episode import run_episode
from .summarize import summarize_run

# Re-export the factory + registry from the top-level package. Safe: the
# package uses guarded imports so missing backend deps don't break this.
from robolab_policy_client import POLICY_REGISTRY, create_client

__all__ = [
    "InferenceClient",
    "POLICY_REGISTRY",
    "create_client",
    "run_episode",
    "summarize_run",
]
