# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run orchestration through RoboLab's native evaluator."""
from policies.ai_native.common.episode import attach_episode_metadata


def run_orchestrated_episode(env, env_cfg, episode, client, **options):
    from robolab.eval.episode import run_episode

    if options.get("enable_gt_state"):
        raise ValueError("Orchestration accepts RGB and proprioception only")
    client.bind_environment(env, env_cfg)
    results, messages, timing = run_episode(env, env_cfg, episode, client, **options)
    # Shared cleanup logs reset exceptions. Surface asynchronous errors and
    # failures to acknowledge the terminal action before publishing results.
    client.check_worker()
    attach_episode_metadata(results, client.episode_metadata)
    return results, messages, timing
