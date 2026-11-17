# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode success from an Isaac Lab termination manager."""

from collections.abc import Sequence

import torch


def success_terms_fired(termination_manager, env_ids: Sequence[int]) -> torch.Tensor:
    """Whether a termination term named ``success*`` fired, for each env in ``env_ids``.

    Isaac Lab's ``termination_manager.terminated`` is the OR of every non-timeout term, so it is
    also set when a failure term (object fell off the table, robot fell) ends the episode. Success
    is read from the ``success*`` terms by name instead, so any other non-timeout term ends the
    episode as a failure. Tasks with one ``success_<object>`` term per acceptable object
    (GrabABagelTask, GrabAFruitTask) succeed when any of those terms fired.

    Returns a bool tensor of shape ``(len(env_ids),)``.
    """
    fired = torch.zeros_like(termination_manager.terminated[env_ids], dtype=torch.bool)
    for name in termination_manager.active_terms:
        if name.startswith("success"):
            fired |= termination_manager.get_term(name)[env_ids].bool()
    return fired
