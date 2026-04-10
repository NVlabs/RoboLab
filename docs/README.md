# RoboLab Documentation

## How RoboLab Works

RoboLab dynamically combines **tasks** with user-specified **robot**, **observations**, **actions**, and **simulation parameters** at environment registration time. The core concepts are:

- **[Objects](objects.md)** — USD object assets with physics properties for manipulation
- **[Scenes](scene.md)** — USD-based environments containing objects, fixtures, and spatial layout
- **[Tasks](task.md)** — Language instructions, termination criteria, and scene bindings
- **[Subtask Checking](subtask.md)** — Granular progress tracking within tasks
- **[Conditionals](task_conditionals.md)** — Predicate logic for defining success/failure conditions
- **[Event Tracking](event_tracking.md)** — Monitoring task-relevant events during execution
- **[Task Libraries](task_libraries.md)** — Managing task collections, generating metadata, and viewing statistics
- **[Robots](robots.md)** — Robot articulation configs, actuators, and action spaces
- **[Cameras](camera.md)** — Scene cameras and robot-attached cameras
- **[Lighting](lighting.md)** — Scene lighting (sphere, directional, and custom lights)
- **[Backgrounds](background.md)** — HDR/EXR dome light backgrounds
- **[Environment Registration](environment_registration.md)** — How tasks are combined with robot/observation/action configs into runnable Gymnasium environments
- **[Environment Generation](environment_generation.md)** — Contact sensor creation, subtask trackers, and runtime environment internals
- **[Inference Clients](inference.md)** — Built-in policy clients and server setup instructions (OpenPI, GR00T)
- **[Running Environments](environment_run.md)** — Creating environments, evaluation scripts, CLI reference, and robustness testing
- **[Data Storage and Output](data.md)** — Output directory structure, HDF5 layout, and episode result fields
- **[Analysis and Results Parsing](analysis.md)** — Scripts for summarizing, comparing, and auditing experiment results

## Development Workflow

If you're building a completely new benchmark and workflow, follow the steps below in order.
Otherwise, pick whichever applies to your use case.

### Creating new assets, tasks, and benchmarks

1. **[Creating New Objects](objects.md)** — Author USD object assets with rigid body, collision, and friction properties. Includes pipeline for catalog generation, screenshots, and physics tuning.
2. **[Creating New Scenes](scene.md)** — Compose objects into USD scenes using IsaacSim. Includes settling, metadata generation, and screenshot utilities.
3. **[Creating New Tasks](task.md)** — Define task dataclasses with language instructions, termination criteria, and scene bindings. Tasks can live in your own repository.
4. **[Managing Task Libraries](task_libraries.md)** — Organize tasks into collections, generate metadata (JSON, CSV, README), and compute statistics.

### Configuring robots, cameras, lighting, and backgrounds

- **[Robots](robots.md)** — Define or customize robot articulation, actuators, and action spaces. Use built-in configs (DROID, Franka) or bring your own from IsaacLab.
- **[Cameras](camera.md)** — Set up scene cameras and robot-attached cameras (e.g., wrist cameras).
- **[Lighting](lighting.md)** — Configure scene lighting for evaluation or robustness testing.
- **[Backgrounds](background.md)** — Set HDR/EXR dome light backgrounds for realistic scene rendering.

### Evaluating a new policy against the benchmark

1. **[Setting Up Environment Registration](environment_registration.md)** — Register tasks with your robot/observation/action/simulation settings. For DROID with joint-position actions, the built-in registration can be used directly.
2. **[Evaluating a New Policy](policy.md)** — Implement an inference client for your model and run multi-task evaluation. Everything can live in your own separate repository.

### AI Workflows

- **[Scene Generation](scene.md#ai-workflows-scene-generation)** — Generate USD scenes from natural language using the `/robolab-scenegen` Claude Code skill. See [`skills/robolab-scenegen/`](../skills/robolab-scenegen/).
- **[Task Generation](task.md#ai-workflows-task-generation)** — Generate task files from natural language using the `/robolab-taskgen` Claude Code skill. See [`skills/robolab-taskgen/`](../skills/robolab-taskgen/).

### Running and debugging

- **[Running Environments](environment_run.md)** — Creating environments, evaluation scripts, CLI reference, and robustness testing
- **[Data Storage and Output](data.md)** — Output directory structure, HDF5 layout, and episode result fields
- **[Analysis and Results Parsing](analysis.md)** — Scripts for summarizing, comparing, and auditing experiment results
- **[Debugging](debug.md)** — Verbose/debug flags, world state inspection, and diagnostic scripts
