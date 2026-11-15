# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab lifecycle adapter for Cosmos Action + Astra orchestration."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from robolab.eval.base_client import InferenceClient

from .bridge import EpisodeBridge


@dataclass(frozen=True)
class OrchestratedConfig:
    policy_uri: str
    codex: str = "codex"
    model: str = "gpt-6-astra-medium"
    cosmos_api_key_env: str | None = None
    student_action_horizon: int = 15
    require_full_student_chunk: bool = False
    max_decisions: int = 3000
    timeout: float = 120
    seed: int = 2040
    robolab_revision: str = "unknown"

    def __post_init__(self):
        if self.model not in tuple(f"gpt-6-astra-{effort}" for effort in ("low", "medium", "high", "xhigh", "max")):
            raise ValueError("Use an exact supported Astra model ID, e.g. gpt-6-astra-medium")
        if not self.policy_uri.startswith(("ws://", "wss://")):
            raise ValueError("policy_uri must be a websocket URL")
        if not 1 <= self.student_action_horizon <= 32:
            raise ValueError("student_action_horizon must be in 1..32")
        if self.require_full_student_chunk and self.student_action_horizon != 32:
            raise ValueError("full student chunks require horizon 32")
        if self.max_decisions < 0 or self.timeout <= 0 or self.seed < 0:
            raise ValueError("Invalid decision budget, timeout, or seed")

    @property
    def reasoning_effort(self):
        return self.model.rsplit("-", 1)[1]

    @property
    def routed_model(self):
        return self.model.rsplit("-", 1)[0]

    @property
    def provider_name(self):
        return "openai"


class OrchestratedClient(InferenceClient):
    """One environment and episode, bound by the shared episode wrapper.

    The simulator thread steps approved actions. A separate persistent Astra
    worker blocks until those actions have actually executed before observing
    and reviewing the next Cosmos proposal.
    """

    def __init__(self, config: OrchestratedConfig, output: str | Path):
        super().__init__()
        self.config = config
        self.output = Path(output).resolve()
        self.runtime = None

    def bind_environment(self, env, env_cfg):
        if self.runtime is not None:
            raise RuntimeError("OrchestratedClient supports one episode per instance")
        token = None
        if self.config.cosmos_api_key_env is not None:
            token = os.environ.get(self.config.cosmos_api_key_env)
            if not token:
                raise ValueError(f"Missing Cosmos credential environment variable {self.config.cosmos_api_key_env}")
        self.output = self.output / ("attempt-" + uuid.uuid4().hex[:12])
        self.output.mkdir(parents=True, exist_ok=False)
        # Like Agentic, inherit CODEX_HOME and let Codex handle authentication.
        # Episode output remains separate; never copy credentials into artifacts.
        self.runtime = EpisodeBridge(
            env, env_cfg, self.output, self.config.codex, self.config.policy_uri,
            token=token, robolab_revision=self.config.robolab_revision,
            max_decisions=self.config.max_decisions, timeout=self.config.timeout,
            student_action_horizon=self.config.student_action_horizon,
            require_full_student_chunk=self.config.require_full_student_chunk,
            seed_base=self.config.seed,
            model=self.config.routed_model,
            effort=self.config.reasoning_effort, provider=self.config.provider_name,
        )

    def prepare_episode(self, episode_idx, instruction):
        if self.runtime is None:
            raise RuntimeError("Bind the environment before preparing an episode")
        # Retrieve tracked values explicitly. Every selected field is shown to
        # both teacher and student; no unselected task context enters prompts.
        if isinstance(instruction, str):
            text = instruction
        else:
            fields = list(instruction.items())
            text = fields[0][1] if len(fields) == 1 else "\n\n".join(
                f"{name}: {value}" for name, value in fields)
        if not text.strip():
            raise ValueError("An instruction is required")
        self.runtime.instruction = text
        self.runtime.begin_episode(episode_idx)

    def infer(self, obs, instruction, *, env_id=0):
        if env_id != 0:
            raise ValueError("Orchestration supports environment 0 only")
        if self.runtime is None or self.runtime.instruction is None:
            raise RuntimeError("prepare_episode must precede inference")
        return self.runtime.infer(obs)

    # Orchestration owns the complete inference loop, like Agentic. The base
    # query/response hooks do not apply to its blocking robot-tool conversation.
    def _extract_observation(self, raw_obs, *, env_id=0):
        raise RuntimeError("Orchestrated observations require episode state; call infer")

    def _pack_request(self, extracted_obs, instruction):
        raise RuntimeError("Orchestrated requests require episode state; call infer")

    def _query_server(self, request):
        raise RuntimeError("Orchestrated queries require episode state; call infer")

    def _unpack_response(self, response):
        raise RuntimeError("Orchestrated responses require episode state; call infer")

    def reset(self, *, env_id=None):
        if env_id not in (None, 0):
            raise ValueError("Orchestration supports environment 0 only")
        if self.runtime is not None:
            self.runtime.reset()
        super().reset(env_id=env_id)

    def check_worker(self):
        if self.runtime is not None:
            self.runtime.check_worker(timeout=self.config.timeout)

    def episode_metadata(self, *, env_id=0):
        path = self.output / "episode" / "controller" / "result.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        return dict(backend="orchestrated", teacher_model=self.config.routed_model,
                    reasoning_effort=self.config.reasoning_effort, native_tools="off",
                    selected_model_id=self.config.model, routed_model=self.config.routed_model,
                    model_provider=self.config.provider_name,
                    student_action_horizon=self.config.student_action_horizon,
                    max_decisions=self.config.max_decisions,
                    orchestration=result)

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
