# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Concrete policy inference clients plus the factory / registry.

Each module in this package is a backend-specific client that inherits from
:class:`robolab.eval.base_client.InferenceClient`. :mod:`.runtime` owns the
``POLICY_REGISTRY`` and ``create_client`` factory.

Backend-specific runtime deps (openpi_client, zmq, json_numpy, requests,
websockets) are imported lazily here so that missing deps only disable the
affected client, not the whole package.
"""

from .runtime import POLICY_REGISTRY, create_client

__all__ = ["POLICY_REGISTRY", "create_client"]

try:
    from .pi0_family import Pi0DroidJointposClient
    __all__.append("Pi0DroidJointposClient")
except ImportError:
    pass

try:
    from .gr00t import GR00TDroidJointposClient
    __all__.append("GR00TDroidJointposClient")
except ImportError:
    pass

try:
    from .dreamzero import DreamZeroClient
    __all__.append("DreamZeroClient")
except ImportError:
    pass
