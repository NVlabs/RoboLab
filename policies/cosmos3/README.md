# Cosmos 3

[**Cosmos 3**](https://huggingface.co/collections/nvidia/cosmos3) is a suite of omnimodal world models designed to jointly process and generate language, images, video, audio, and action sequences. This directory provides the RoboLab client for [**Cosmos3-Nano-Policy-DROID**](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID), a World-Action Model (WAM) obtained by post-training Cosmos 3 on the [DROID](https://droid-dataset.github.io/) dataset.

[`client.py`](./client.py) provides the `Cosmos3Client` class that connects to a policy server hosting Cosmos3-Nano-Policy-DROID over the OpenPI WebSocket protocol. The client uses RoboLab's shared MsgPack/NumPy WebSocket transport, including optional Bearer authentication, and does not depend on `openpi-client`.

Below is a quickstart for bringing up the policy server and running an evaluation from a RoboLab client.

## Server

First, clone [`cosmos-framework`](https://github.com/NVIDIA/cosmos-framework):

```Shell
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
```

Build the Docker image:

```Shell
docker build \
  -t cosmos-framework:latest \
  .
```

Set your Hugging Face token and launch the container, which installs the dependencies:

```Shell
# Set your Hugging Face token (https://huggingface.co/settings/tokens):
export HF_TOKEN=<your_hf_token>

docker run \
  -it \
  -e HF_HOME=/workspace/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  --net host \
  --rm \
  --runtime nvidia \
  -v .:/workspace \
  -v /workspace/.venv \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  cosmos-framework:latest \
  bash -c '\
    uv sync \
      --all-extras \
      --group=cu130-train \
      --group=policy-server && \
    exec bash; \
  '
```

Inside the container, start the policy server:

```Shell
python -m cosmos_framework.scripts.action_policy_server_robolab \
  --port 8000
```

## Client

Clone [`RoboLab`](https://github.com/NVlabs/RoboLab):

```Shell
git clone https://github.com/NVlabs/RoboLab.git
cd RoboLab
```

Build the Docker image:

```Shell
./docker/build_docker.sh latest
```

Launch the container:

```Shell
./docker/run_docker.sh latest
```

Run a task against the policy server. This opens a viewer window for real-time visualization of the simulation:

```Shell
python policies/cosmos3/run.py \
  --task BananaInBowlTask
```

The host and port form above targets the Cosmos reference server at
`ws://localhost:8000`. To use a routed endpoint such as vLLM-Omni's standard
OpenPI endpoint, pass the complete URI:

```Shell
python policies/cosmos3/run.py \
  --remote-uri ws://localhost:8000/v1/realtime/robot/openpi \
  --task BananaInBowlTask
```

If the server was started with API-key authentication, pass the key as a
Bearer token with `--remote-token`, or set `COSMOS3_API_TOKEN`. The command-line
value takes precedence over the environment variable:

```Shell
export COSMOS3_API_TOKEN=<your_api_key>
python policies/cosmos3/run.py \
  --remote-uri wss://policy.example/v1/realtime/robot/openpi \
  --task BananaInBowlTask
```

RoboLab sends a distinct `session_id` for every episode and parallel
environment. The VoLo Cosmos3 client sends that policy-session identifier as
well as VoLo's separate `__episode_id`; proxies should preserve both fields.

To evaluate across multiple sub-environments in parallel in headless mode:

```Shell
python policies/cosmos3/run.py \
  --task BananaInBowlTask \
  --num-envs 10 \
  --headless
```
