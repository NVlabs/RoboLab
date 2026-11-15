# Astranomy: AI-Native Robot Policies

This directory contains the RoboLab policy implementations for [Astranomy](https://research.nvidia.com/labs/srl/projects/astranomy/), an empirical study of GPT-6 Astra as a native robot policy. Astranomy studies direct action generation, robot control through a persistent agent conversation, and model supervision of a robot foundation policy.

## Policies

| Policy | Description | Documentation |
|---|---|---|
| Direct-API (Direct Policy) | Astra predicts robot actions directly through API calls. | [Direct-API README](direct_api/README.md) |
| Agentic | Astra controls the robot through Codex, with conversation history and optional tools. | [Agentic README](agentic/README.md) |
| Orchestrated | Astra reviews actions proposed by a robot foundation policy and decides when to intervene. | [Orchestrated README](orchestrated/README.md) |

Each policy's README covers its setup, supported features, configuration, and usage.
