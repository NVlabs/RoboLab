# Cosmos 3

[**Cosmos 3**](https://huggingface.co/collections/nvidia/cosmos3) is a suite of omnimodal world models designed to jointly process and generate language, images, video, audio, and action sequences. This directory provides the RoboLab client for [**Cosmos3-Nano-Policy-DROID**](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID), a World-Action Model (WAM) obtained by post-training Cosmos 3 on the [DROID](https://droid-dataset.github.io/) dataset.

[`client.py`](./client.py) provides the `Cosmos3Client` class that connects to a policy server hosting Cosmos3-Nano-Policy-DROID over the OpenPI WebSocket protocol. `Cosmos3Client` requires the [`openpi-client`](https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client) package.

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

To evaluate across multiple sub-environments in parallel in headless mode:

```Shell
python policies/cosmos3/run.py \
  --task BananaInBowlTask \
  --num-envs 10 \
  --headless
```


## Galbot contact setting

Galbot evaluations use `--embodiment galbot`. To select the 120 Hz gripper
contact setting tested with recorded `SauceBottlesCrateTask` actions, add:

```text
--galbot-init-pose dataset_mean --galbot-gain-profile reduced --galbot-sim-hz 120 --galbot-decimation 8 --galbot-position-iterations 255
```

The position iteration option raises both the robot solver request and the
scene limit. With eight physics steps per action, the control rate stays at
15 Hz. The option defaults to 128 iterations; these flags select the tested
setting explicitly. The recorded-action tests reduced sampled overlap but did
not eliminate it or establish closed-loop policy success.
