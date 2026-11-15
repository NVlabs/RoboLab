# Orchestrated: Cosmos Action + Astra

Cosmos Action proposes a 32-step DROID joint-position trajectory. A persistent
GPT-6 Astra conversation reviews the first 15 steps by default, using three camera
views, measured robot state, and a robot-only forward-kinematics preview. Astra chooses:

- `student`: execute an unchanged prefix of the proposal (1–15 steps by default).
- `edit`: adjust the proposal's EEF trajectory or gripper for 1–5 steps.
- `eef`: issue an absolute EEF correction for 1–5 steps.

The gate permits intervention on observed execution failure or a mismatch between
the proposed intent and the current subgoal. EEF commands use bounded IK. The
simulator waits while Astra reasons; execution replies contain fresh observations
after the approved actions finish. No speculative simulation or task/object ground
truth is provided to the policy. Only native success or timeout ends the episode.
At the decision cap, the host holds measured joint/gripper state until native
termination. Reaching that cap does **not** mean task success.

## Setup

Follow the repository [installation instructions](../../../README.md) on a Linux
GPU machine with IsaacSim, IsaacLab, and the RoboLab assets. Python dependencies
are included in the shared RoboLab installation. Install the tested Codex CLI
version:

```bash
npm install -g @openai/codex@0.163.0
codex --version  # codex-cli 0.163.0
```

Pass `--codex-bin /path/to/codex` when keeping this version alongside another
installation. The model-facing prompts preserve the reference policy, including
Simplified Chinese decision explanations.

The backend uses DROID with native Codex tools disabled. Astra gets three robot
tools and five bounded episode-memory tools.

### Configure Astra

Astra runs through Codex CLI's `openai` provider. Use a Codex login with access
to GPT-6 Astra. For API-key authentication, set `OPENAI_API_KEY` and configure
a dedicated Codex home:

```bash
export ASTRA_CODEX_HOME="/absolute/path/outside/checkout/astra-codex-home"
mkdir -p "$ASTRA_CODEX_HOME"
printf '%s' "$OPENAI_API_KEY" | CODEX_HOME="$ASTRA_CODEX_HOME" codex login --with-api-key
```

To use an existing Codex login, set `ASTRA_CODEX_HOME` to that Codex home instead.
The evaluation command below passes it to Codex as `CODEX_HOME`.

### Start the Cosmos policy server

Deploy [Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID)
as an OpenPI-compatible WebSocket policy server. Follow the
[Cosmos server setup guide](../../cosmos3/README.md#server), then pass the running
server's URL to `--policy-uri`. The reference server uses `ws://localhost:8000`;
for a remote deployment, supply its complete `ws://` or `wss://` URL, including
any route path required by that server. Keep the server running during evaluation.

For endpoints using `Authorization: Api-Key ...`, set a credential environment
variable and pass its name with `--cosmos-api-key-env YOUR_KEY_VARIABLE`.
Omit this option for unauthenticated servers.

## Run an evaluation

Run from the RoboLab repository root, using the installed Python environment:

```bash
CODEX_HOME="$ASTRA_CODEX_HOME" python policies/ai_native/orchestrated/run.py \
  --task BananaThenRubiksCubeTask --headless \
  --num-envs 1 --num-runs 1 --seed 2040 \
  --policy-uri ws://localhost:8000 \
  --model gpt-6-astra-medium --allow-model-requests \
  --instruction-type default --video-mode sensor \
  --output-folder-name /absolute/path/outside/checkout
```

## Configuration

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `gpt-6-astra-medium` | Astra model plus reasoning effort (`low`, `medium`, `high`, `xhigh`, `max`) |
| `--codex-bin` | `codex` | Codex executable |
| `--decision-timeout` | `120` | Wait limit between completed agent tool calls, in seconds |
| `--allow-model-requests` | Off | Include this flag to enable live model requests |
| `--policy-uri` | Required | URL of the running Cosmos WebSocket policy server |
| `--cosmos-api-key-env` | Unset | Credential environment variable for an authenticated Cosmos endpoint |
| `--student-action-horizon` | `15` | Leading actions retained from each 32-step Cosmos proposal |
| `--require-full-student-chunk` | Off | Require all 32 student actions; use with `--student-action-horizon 32` |
| `--max-decisions` | `3000` | Robot-decision budget; `0` disables this cap |

Orchestrated uses Codex's `openai` provider. `--model gpt-6-astra-medium`
selects model `gpt-6-astra` with `medium` reasoning effort. Codex uses the
credentials and configuration in `CODEX_HOME`.

Use `--require-full-student-chunk --student-action-horizon 32` to require
acceptance of the entire proposal in `student` mode. Otherwise, Astra selects a
prefix up to `--student-action-horizon` steps.
Edits are limited to the available proposal length, up to five steps. Absolute
EEF corrections can use up to five steps regardless of the student horizon.

`--max-decisions` counts robot decisions. Once the budget is exhausted, the host
holds the measured robot state until native success or timeout. Lowering this
budget does not shorten the task's native time limit. `--decision-timeout` limits
the wait between completed agent tool calls, rather than total evaluation
duration. Reasoning events alone do not reset this deadline.

Use `--instruction-type` to select task instruction fields. The selected fields
are supplied to both models and recorded in the evaluation results.

### Relationship to Astranomy

This backend preserves the reference Orchestrated policy's prompts and action
interface: medium reasoning, a 15-step Cosmos review horizon, 1–5-step
interventions, three camera views, bounded episode-memory tools, and native tools
disabled. DROID runs at 15 Hz, with simulation paused during model inference.

The [Astranomy study-design prose](https://research.nvidia.com/labs/srl/projects/astranomy/#study-design)
and the released backend differ in some runtime settings. The defaults here
preserve the reference implementation rather than silently changing its inputs:

| Setting | Study-design prose | This backend |
|---|---|---|
| Astra camera images | 384 × 216 pixels per view | Aspect-preserving JPEG thumbnails, at most 640 pixels per side |
| Seeds | Five seeds, 0–4 | Default seed 2040; set `--seed` to change it |
| Failure handling | No walltime or retry limit | Default 120-second agent inactivity timeout; bounded tool-input rejection handling; up to three Cosmos request attempts |
| Completion tokens | 8,192 per model call | No explicit completion-token cap set by this backend; the configured model runtime governs it |

Matching these settings does not guarantee identical model outputs or reproduce
the published success rate.

## Results and lifecycle

The shared RoboLab evaluator owns resets, stepping, native termination, video,
HDF5 trajectories, and result summaries. `episode.py` binds the environment,
checks worker completion after native cleanup, and attaches policy metadata.
`client.py` supplies the evaluator's inference interface, selected instructions,
and credentials. `agent/run.py` runs the Astra conversation; `control/client.py`
handles its robot service calls; `bridge.py` supplies approved actions to the
simulator and acknowledges execution. Package marker files (`__init__.py`) are
omitted from the tree below.

```text
orchestrated/
├── README.md                        # Setup, configuration, and evaluation outputs
├── LICENSE-MIT                      # License for the included code and prompts
├── run.py                           # Parse CLI, start IsaacLab, invoke shared evaluator
├── client.py                        # Adapt configuration and instructions to infer/reset
├── episode.py                       # Bind environment, check completion, attach metadata
├── bridge.py                        # Queue approved actions and return execution feedback
├── cosmos3_rollout.py               # Cosmos tool setup, prompts, tracking, and recovery holds
├── settings.py                      # Default Astra model and reasoning effort
├── io.py                            # Atomic JSON writes and input-validation helpers
├── cosmos/
│   ├── student.py                   # Request Cosmos actions, validate and retain a prefix
│   ├── observation.py               # Pack camera/proprio inputs with RoboLab's Cosmos codec
│   └── transport.py                 # MsgPack WebSocket connection, authentication, timeouts
├── agent/
│   ├── run.py                       # Maintain Codex conversation and dispatch robot/memory tools
│   ├── transport.py                 # Launch Codex app-server; exchange and log JSON-RPC
│   ├── episode_memory.py            # Store and recall episode observations, history, and notes
│   ├── schema.py                    # Define the structured action-response tool schema
│   ├── workspace.py                 # Create the Codex working directory and tool instructions
│   ├── SKILL.md                     # Tell Astra how to use services and choose actions
│   ├── gate_prompt.md               # Explain when intervention is justified
│   └── context/
│       ├── teacher_context.md       # Observation interpretation and decision guidance
│       ├── eef_control.md           # Coordinate frames, action limits, and IK behavior
│       └── cosmos_geometry.md       # Gripper calibration and camera extrinsics
└── control/
    ├── client.py                    # Handle start/infer/execute calls and save execution records
    ├── gate_assessment.py           # Validate task-progress and intervention evidence
    ├── validation.py                # Reject stale requests and invalid action targets or lengths
    ├── action_edit_kinematics.py    # Compute robot FK and offsets for edited trajectories
    ├── eef_controller.py            # Convert EEF targets to bounded joint targets using IK
    └── decision_log.py              # Emit public structured decision events
```

Robot control services use an in-process bridge to the native evaluator.

Each task's `orchestration/attempt-*/` contains prompt hashes and an `episode/`
directory with endpoint identity, public decision records, proposed/executed
actions, history, timing, and result counters. Prior attempts are preserved on
retries. Endpoint identity is declared rather than independently verified when
the server reports empty metadata.

The shared evaluator saves native video, HDF5 trajectories, and episode summaries
under the evaluation output directory. Within each orchestration episode,
`controller/history.json` records executed decisions, `controller/result.json`
records completion and counters, and `sim/codex_decisions/` records submitted
decisions. `codex_workspace/` contains `rpc_in.jsonl` and `rpc_out.jsonl`, which record the Codex protocol
messages and may include observations, encoded images, and model reasoning events.

## License attribution

Parts of this backend are adapted from
[GPT-as-Policy](https://github.com/anonymous-report-421/GPT-as-Policy) by
Yu-Mool Shu and Lipxin Zheng, released under the MIT license and included here
as [LICENSE-MIT](LICENSE-MIT). The agent prompts (`agent/SKILL.md`,
`agent/gate_prompt.md`, `agent/context/teacher_context.md`,
`agent/context/eef_control.md`), the agent loop and transport (`agent/run.py`,
`agent/transport.py`, `agent/schema.py`), the robot-control layer (`control/`),
and `io.py` are adapted from its `hybrid_rollout/robolab/` implementation; each
adapted source file names its upstream file in its header. The Cosmos
integration (`cosmos/`, `cosmos3_rollout.py`, `bridge.py`), the episode-memory
tools, the camera and gripper calibration context, and the RoboLab adapters are
NVIDIA code under the repository's Apache-2.0 license.
