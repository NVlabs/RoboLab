# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create the working directory for the robot-tool-only Codex session."""
from pathlib import Path


def prepare_workspace(audit, skill_root):
    agent = Path(audit).resolve() / 'agent'
    agent.mkdir()
    (agent / 'AGENTS.md').write_text(
        'Use only the supplied robot service and bounded episode-memory tools.\n'
        'Camera observations and robot context are supplied in the conversation.\n'
        'Paths in robot-tool arguments identify host-managed artifacts.\n')
    return agent
