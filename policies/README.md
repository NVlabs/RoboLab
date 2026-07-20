# Inference Clients and Policy Server Setup

RoboLab uses a **server-client architecture**: your model runs as a standalone server process, and RoboLab connects to it through a lightweight inference client during evaluation.

Each `policies/<policy>/` folder is one backend and contains:

- `client.py` — the concrete `InferenceClient` subclass that speaks WebSocket / ZMQ / HTTP to a remote policy server.
- `run.py` — the runner script. Defines policy-specific argparse flags, builds a `make_client(args)` closure, and calls `run_evaluation(args, policy="<name>", client_factory=make_client)`.
- `__init__.py` — re-exports the client class.
- `README.md` — server install, server launch, and run instructions for that backend.

## The `InferenceClient` contract

All concrete clients inherit from the `InferenceClient` ABC in `robolab/eval/base_client.py`:

```python
from robolab.eval import InferenceClient

class InferenceClient(ABC):
    # Hooks subclasses must implement:
    def _extract_observation(self, raw_obs, *, env_id=0) -> dict: ...
    def _pack_request(self, extracted_obs, instruction) -> Any: ...
    def _query_server(self, request) -> Any: ...
    def _unpack_response(self, response) -> np.ndarray: ...
    # Provided by the base: infer(), reset(), close(), chunking state.
```

Each `run.py` imports its client class directly and constructs it inline — there is no central registry or factory:

```python
from policies.pi0_family.client import Pi0DroidJointposClient

client = Pi0DroidJointposClient(remote_host="localhost", remote_port=8000, policy_variant="pi05")
```

For writing your own inference client, see [Evaluating a New Policy](../docs/policy.md).

## Common CLI Options

For the full CLI reference, see [Running Environments](../docs/environment_run.md#run-cli-reference).
Use `policies/<policy>/run.py`

```bash
# Run on all benchmark tasks headlessly
uv run python policies/<policy>/run.py --headless

# Run on a specific task
uv run python policies/<policy>/run.py --task BananaInBowlTask

# Run on a tag of tasks
uv run python policies/<policy>/run.py --tag pick_place

# Run multiple runs per task (total episodes = num_runs * num_envs)
uv run python policies/<policy>/run.py --headless --num-runs 5 --num-envs 2

# Resume a previous run
uv run python policies/<policy>/run.py --headless --output-folder-name my_previous_run

# Disable subtask checking (on by default)
uv run python policies/<policy>/run.py --headless --disable-subtask
```
