# Evaluating a New Policy

This guide walks through how to evaluate your own policy against the RoboLab benchmark. You do **not** need to fork or modify RoboLab — everything can live in your own separate repository that imports `robolab` as a dependency.

RoboLab uses a **server-client architecture**: your model runs as a standalone server (any framework, any GPU), and a lightweight inference client inside the simulator sends observations and receives actions.

**Prerequisites:** You need registered environments before running evaluation. For DROID with joint-position actions, RoboLab ships a built-in registration you can use directly. If you need custom observations, a different robot, or different simulation parameters, first follow the [Environment Registration](environment_registration.md) guide.

## Your Repository Structure

```
my_policy_eval/
  my_policy/
    __init__.py
    inference_client.py        # Your inference client (Step 1)
  run_eval.py                  # Your evaluation script (Step 2)
  requirements.txt             # includes robolab as a dependency
```

## Step 1: Implement an Inference Client

Subclass `robolab.inference.InferenceClient`:

```python
# my_policy/inference_client.py

import numpy as np
from robolab.inference.base_client import InferenceClient


class MyPolicyClient(InferenceClient):
    def __init__(self, remote_host: str = "localhost", remote_port: int = 8000) -> None:
        # Connect to your model server
        ...

    def infer(self, obs: dict, instruction: str) -> dict:
        # For the default DROID registration, obs contains:
        #   obs["image_obs"]["external_cam"]    - (N, H, W, 3) torch tensor, uint8
        #   obs["image_obs"]["wrist_cam"]       - (N, H, W, 3) torch tensor, uint8
        #   obs["proprio_obs"]["arm_joint_pos"] - (N, 7) torch tensor, float32
        #   obs["proprio_obs"]["gripper_pos"]   - (N, 1) torch tensor, float32

        # Extract observations for this env (N = num_envs; index by env_id)
        image = obs["image_obs"]["external_cam"][0].cpu().numpy()
        joint_pos = obs["proprio_obs"]["arm_joint_pos"][0].cpu().numpy()

        # Call your model server and get back an action
        action = self._query_server(image, joint_pos, instruction)

        # Return dict with "action" (np.ndarray) and "viz" (np.ndarray for display)
        return {
            "action": action,  # shape (8,): 7 joint positions + 1 gripper {0, 1}
            "viz": image,      # any RGB image for the live visualization window
        }

    def reset(self):
        # Called between episodes. Clear any internal state (action buffers, etc.)
        ...
```

**Key contract:**
- `infer()` receives the raw observation dict from the simulator and a language instruction string.
- It must return a dict with `"action"` (a numpy array, typically 8-dim: 7 joints + 1 gripper) and `"viz"` (an image array for the live display window).
- `reset()` is called at the end of each episode.

See the [existing clients](#existing-clients-as-reference) for complete working examples.

## Step 2: Write Your Evaluation Script and Run It

For the full evaluation script template, CLI reference, and run instructions, see [Running Environments](environment_run.md#writing-an-evaluation-script).

In short:

1. **Install robolab** as a dependency:
   ```bash
   cd /path/to/robolab && uv pip install -e .
   ```

2. **Install your package** so its modules are importable:
   ```bash
   cd /path/to/my_policy_eval && uv pip install -e .
   ```

3. **Start your model server** (in a separate terminal):
   ```bash
   python -m my_model.serve --checkpoint /path/to/model --port 8000
   ```

4. **Run evaluation**:
   ```bash
   # Run on all benchmark tasks
   python run_eval.py --headless

   # Run on a specific task
   python run_eval.py --task BananaInBowlTask

   # Run on a tag of tasks
   python run_eval.py --tag pick_place

   # Run multiple runs with parallel envs (total episodes = num_runs * num_envs)
   python run_eval.py --headless --num-runs 5 --num_envs 2

   # Custom server address
   python run_eval.py --remote-host 10.0.0.1 --remote-port 5555
   ```

5. **View results**: Results are saved to `output/<timestamp>_my_policy/`. See [Analysis and Results Parsing](analysis.md) for summarization tools.

## Existing Clients as Reference

| Client | Protocol | File |
|--------|----------|------|
| Pi0 / Pi0-fast / Pi05 | WebSocket (OpenPI) | `robolab/inference/pi0_family.py` |
| GR00T | ZMQ | `robolab/inference/gr00t.py` |

See [Inference Clients](inference.md) for server setup instructions.
