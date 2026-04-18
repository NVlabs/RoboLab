# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Inference-client runtime: POLICY_REGISTRY + create_client factory.

Parallel to ``robolab/core/environments/runtime.py`` for environments: given a
registered backend name plus kwargs, return a constructed
:class:`InferenceClient`. Callers should not branch on backend name — this
module owns the registry lookup and kwarg-filtering.
"""

import inspect
from typing import Any

from robolab.eval.base_client import InferenceClient

# Populate the registry with whatever backends can be imported. A missing
# backend dep (e.g. openpi_client, zmq, json_numpy) excludes that backend
# without breaking the import of the runtime itself. Pi0 handles several
# trained variants via its own ``policy_variant`` kwarg (forwarded by
# :func:`create_client` when the class accepts it), so the same class appears
# under multiple keys.
POLICY_REGISTRY: dict[str, type[InferenceClient]] = {}

try:
    from .pi0_family import Pi0DroidJointposClient
    for _name in ("pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"):
        POLICY_REGISTRY[_name] = Pi0DroidJointposClient
except ImportError:
    pass

try:
    from .gr00t import GR00TDroidJointposClient
    POLICY_REGISTRY["gr00t"] = GR00TDroidJointposClient
except ImportError:
    pass

try:
    from .dreamzero import DreamZeroClient
    POLICY_REGISTRY["dreamzero"] = DreamZeroClient
except ImportError:
    pass


def create_client(name: str, **kwargs: Any) -> InferenceClient:
    """Construct the inference client for a given backend name.

    The client class is looked up in :data:`POLICY_REGISTRY`. Kwargs are
    filtered by the class's ``__init__`` signature so each backend only
    receives the subset of arguments it actually declares — e.g. Pi0 accepts
    ``remote_uri`` and ``policy_variant`` while GR00T silently drops both.

    The backend name itself is passed as ``policy_variant`` when the client
    accepts that parameter (used by Pi0 to pick its per-variant default
    horizon). Explicit ``policy_variant`` in kwargs wins over this default.

    Args:
        name: Backend identifier (e.g. ``"pi0"``, ``"pi05"``, ``"gr00t"``).
            Must be a key in :data:`POLICY_REGISTRY`.
        **kwargs: Client constructor arguments. Values of ``None`` are
            dropped so the client's own defaults apply.

    Returns:
        A constructed :class:`InferenceClient` subclass instance.

    Raises:
        ValueError: If ``name`` is not a registered backend.
    """
    name = name.lower()
    try:
        cls = POLICY_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unsupported policy '{name}'. Known: {sorted(POLICY_REGISTRY)}"
        ) from None

    candidate = {"policy_variant": name, **kwargs}
    accepted = set(inspect.signature(cls.__init__).parameters)
    filtered = {k: v for k, v in candidate.items() if k in accepted and v is not None}
    return cls(**filtered)
