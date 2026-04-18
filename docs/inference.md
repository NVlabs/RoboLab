# Inference Clients and Policy Server Setup

RoboLab uses a **server-client architecture**: your model runs as a standalone server process, and RoboLab connects to it through a lightweight inference client during evaluation.

## Built-in Inference Clients

| Policy | Client Class | Protocol | Default Port | Dependencies |
|--------|-------------|----------|-------------|--------------|
| Pi0 / Pi0-fast / Pi05 | `Pi0DroidJointposClient` | WebSocket (OpenPI) | 8000 | `openpi-client` |
| GR00T | `GR00TDroidJointposClient` | ZMQ | 5555 | `zmq`, `msgpack` |

Concrete clients live in the top-level `robolab_policy_client/` package (sibling to `robolab/`, installed together). They all inherit from the `InferenceClient` ABC in `robolab/eval/base_client.py`:

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

The `create_client(name, **kwargs)` factory looks a backend up in
`POLICY_REGISTRY` and constructs it with signature-filtered kwargs:

```python
from robolab.eval import create_client

client = create_client("pi0", remote_host="localhost", remote_port=8000)

# Or import a client class directly from the top-level package:
from robolab_policy_client import Pi0DroidJointposClient
```

For writing your own inference client, see [Evaluating a New Policy](policy.md).

---

## OpenPI (Pi0 / Pi0-fast / Pi05)

OpenPI uses a WebSocket-based policy server. The server runs separately (in its own environment) and RoboLab connects via the `openpi-client` package.

### Install the server

1. Clone [`git@github.com:xuningy/openpi.git`](https://github.com/xuningy/openpi) and follow install instructions there. **Do not** install OpenPI in the same virtual environment as RoboLab — it runs separately.

2. Install the OpenPI **client** in the RoboLab environment:
   ```bash
   cd robolab
   uv pip install -e ../openpi/packages/openpi-client
   ```

### Start the policy server

Open a separate terminal and launch the server. We set `XLA_PYTHON_CLIENT_MEM_FRACTION` to 50% to avoid JAX consuming all GPU memory.

**Pi05:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
```

**Pi0-fast:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi0_fast_droid_jointpos
```

**Pi0:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/pi0_droid_jointpos
```

**PaliGemma Binning:**
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=paligemma_binning_droid_jointpos \
    --policy.dir=gs://openpi-assets-simeval/paligemma_binning_droid_jointpos
```

### Run evaluation

```bash
cd robolab
uv run python examples/policy/run_eval.py --policy pi05 --headless
```

The default connection is `localhost:8000`. To change:
```bash
uv run python examples/policy/run_eval.py --policy pi05 --remote-host <HOST> --remote-port <PORT>
```

---

## GR00T N1.6

RoboLab ships a built-in GR00T inference client (`robolab_policy_client/gr00t.py`) that communicates via ZMQ.

### Install the server

1. Make sure your `CUDA_HOME` and `PATH` is adequately set in your `.bashrc`. Otherwise, set it explicitly:
    ```bash
    export CUDA_HOME=/usr/local/cuda-12.4
    export PATH=/usr/local/cuda-12.4/bin:$PATH
    ```

2. Clone and install:
    ```bash
    git clone --recurse-submodules https://github.com/nadunRanawaka1/Isaac-GR00T-n16-droid.git
    cd Isaac-GR00T-n16-droid
    git checkout fa1fd91f4798e333b7cd1e9d5a32fe55f105a16b
    uv sync --python 3.10
    uv pip install -e .
    ```

3. Download the model checkpoint [oss-droid-v0.zip](https://nvidia-my.sharepoint.com/personal/nranawakaara_nvidia_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fnranawakaara%5Fnvidia%5Fcom%2FDocuments%2Fgr00t%5Fcheckpoints%2Foss%2Ddroid%2Dv0%2Ezip&parent=%2Fpersonal%2Fnranawakaara%5Fnvidia%5Fcom%2FDocuments%2Fgr00t%5Fcheckpoints) and unzip.

### Start the policy server

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path /path/to/oss-droid-v0/checkpoint-25000 \
    --embodiment-tag OXE_DROID_JOINT_POSITION_RELATIVE \
    --use-sim-policy-wrapper \
    --host 0.0.0.0 --port 5555
```

### Run evaluation

```bash
cd robolab
uv run python examples/policy/run_eval.py --policy gr00t --remote-host 0.0.0.0 --remote-port 5555 --headless
```

---

## Common CLI Options

For the full CLI reference, see [Running Environments](environment_run.md#run_evalpy-cli-reference).

```bash
# Run on all benchmark tasks headlessly
uv run python examples/policy/run_eval.py --policy <policy> --headless

# Run on a specific task
uv run python examples/policy/run_eval.py --policy <policy> --task BananaInBowlTask

# Run on a tag of tasks
uv run python examples/policy/run_eval.py --policy <policy> --tag pick_place

# Run multiple runs per task (total episodes = num_runs * num_envs)
uv run python examples/policy/run_eval.py --policy <policy> --headless --num-runs 5 --num_envs 2

# Resume a previous run
uv run python examples/policy/run_eval.py --policy <policy> --headless --output-folder-name my_previous_run

# Enable subtask checking
uv run python examples/policy/run_eval.py --policy <policy> --headless --enable-subtask
```
