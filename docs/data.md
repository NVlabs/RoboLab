# Data Storage and Output

By default, proprio data will be recorded in the hdf5 file. To turn on image data recording, `robolab.constants.RECORD_IMAGE_DATA` needs to be set to True.

## Single-Env vs Multi-Env Recording

RoboLab supports running multiple parallel environments per task. The recording system handles both cases:

**Single-env** (`--num_envs 1`, default):
- One episode per run, written to `run_{run_idx}.hdf5` as `demo_0`
- Videos: `{instruction}_{run_idx}.mp4`
- Subtask logs: `log_{run_idx}_env0.json`

**Multi-env** (`--num_envs N`):
- N parallel episodes per run, all written to `run_{run_idx}.hdf5` as `demo_0` through `demo_{N-1}`
- Each `demo_i` corresponds to `env_id=i` with independent trajectory data
- Videos: `{instruction}_{run_idx}_env{env_id}.mp4` (one per env)
- Subtask logs: `log_{run_idx}_env{env_id}.json` (one per env)
- Each env has its own subtask state machine, event tracker, and termination state
- Envs terminate independently — when an env succeeds, it freezes while others continue
- HDF5 recording uses auto-flush (every 500 steps) with concurrent per-env streaming — each env's buffer is flushed to its own demo without interfering with other envs
- Total episodes = `num_runs * num_envs`

> **Tip:** Prefer increasing `--num_envs` for more episodes. Only increase `--num-runs` (default 1) if you run out of GPU memory with the desired `num_envs`. For example, to run 24 episodes when only 12 fit in VRAM: `--num_envs 12 --num-runs 2`.

The `episode_results.jsonl` tracks each env independently with a unique `episode` ID (`run_idx * num_envs + env_id`) and an `env_id` field. Each line is a self-contained JSON object (JSONL format) for safe append-only writes.

## Output Directory Structure

```
output/
└── <output_folder>/
    ├── episode_results.jsonl
    ├── <ENV_NAME>/
        ├── run_0.hdf5                       # Run 0 data (demo_0..demo_{N-1} for N envs)
        ├── run_1.hdf5                       # Run 1 data (if num_runs > 1)
        ├── log_0_env0.json                  # Run 0, env 0 subtask log
        ├── log_0_env1.json                  # Run 0, env 1 subtask log
        ├── {instruction}_0_env0.mp4         # Run 0, env 0 observation video
        ├── {instruction}_0_env0_viewport.mp4
        ├── {instruction}_0_env1.mp4         # Run 0, env 1 observation video
        ├── {instruction}_0_env1_viewport.mp4
        └── env_cfg.json
```
For `ENV_NAME`, refer to [environment registration](environment_registration.md).

## HDF5 Data Structure

Each `run_{i}.hdf5` file contains the following hierarchical structure (`h5glance run_0.hdf5`)
```
└data (2 attributes)
  ├demo_0 (2 attributes)
  │ ├actions      [float32: 470 × 8]
  │ ├initial_state
  │ │ ├articulation
  │ │ │ └robot
  │ │ │   ├joint_position [float32: 1 × 13]
  │ │ │   ├joint_velocity [float32: 1 × 13]
  │ │ │   ├root_pose      [float32: 1 × 7]
  │ │ │   └root_velocity  [float32: 1 × 6]
  │ │ └rigid_object
  │ │   ├banana
  │ │   │ ├root_pose      [float32: 1 × 7]
  │ │   │ └root_velocity  [float32: 1 × 6]
  │ │   ├bowl
  │ │   │ ├root_pose      [float32: 1 × 7]
  │ │   │ └root_velocity  [float32: 1 × 6]
  │ │   └rubiks_cube
  │ │     ├root_pose      [float32: 1 × 7]
  │ │     └root_velocity  [float32: 1 × 6]
  │ ├obs
  │ │ ├arm_joint_pos      [float32: 470]
  │ │ ├external_cam       [uint8: 470 × 720 × 1280 × 3]
  │ │ ├gripper_pos        [float32: 470]
  │ │ └wrist_cam  [uint8: 470 × 720 × 1280 × 3]
  │ ├states
  │ │ ├articulation
  │ │ │ └robot
  │ │ │   ├joint_position [float32: 470 × 13]
  │ │ │   ├joint_velocity [float32: 470 × 13]
  │ │ │   ├root_pose      [float32: 470 × 7]
  │ │ │   └root_velocity  [float32: 470 × 6]
  │ │ └rigid_object
  │ │   ├banana
  │ │   │ ├root_pose      [float32: 470 × 7]
  │ │   │ └root_velocity  [float32: 470 × 6]
  │ │   ├bowl
  │ │   │ ├root_pose      [float32: 470 × 7]
  │ │   │ └root_velocity  [float32: 470 × 6]
  │ │   └rubiks_cube
  │ │     ├root_pose      [float32: 470 × 7]
  │ │     └root_velocity  [float32: 470 × 6]
  │ ├subtask
  │ │ ├completed  [uint8: 470]
  │ │ ├score      [float32: 470]
  │ │ └status     [uint16: 470]
  │ └bbox
  │   ├bbox_mm
  │   │ ├banana       [int16: 470 × 8 × 3]   # OBB corners in mm (relative to env origin)
  │   │ ├bowl         [int16: 470 × 8 × 3]
  │   │ └rubiks_cube  [int16: 470 × 8 × 3]
  │   └centroid
  │     ├banana       [float16: 470 × 3]      # centroid in meters (relative to env origin)
  │     ├bowl         [float16: 470 × 3]
  │     └rubiks_cube  [float16: 470 × 3]
  ├demo_1
    ....
```

## Data Structure Details

### Episodes
- **`demo_i`**: Episode data for `env_id=i`. In single-env mode, only `demo_0` exists. In multi-env mode, `demo_0` through `demo_{N-1}` exist in each `run_{run_idx}.hdf5`.

### Data Components
- **`actions`**: Robot control commands (8-dimensional for joint positions)
- **`initial_state`**: Starting configuration of robot and objects
- **`obs`**: Observations including joint positions, camera images, and gripper state
- **`states`**: Full state trajectory of robot and objects over time
- **`subtask`**: Task progress tracking metrics
- **`bbox`**: Per-step oriented bounding box data for all rigid objects
  - **`bbox_mm/{name}`**: OBB corners as `int16` in millimeters, shape `(T, 8, 3)`. Corner order: `[0-3]` bottom face (z-low), `[4-7]` top face (z-high). Positions are relative to env origin. Convert to meters: `corners_m = data[:] / 1000.0`
  - **`centroid/{name}`**: Object centroid as `float16` in meters, shape `(T, 3)`. Relative to env origin.

### Object Tracking
The system tracks three main objects:
- **`banana`**: Position and velocity in 3D space
- **`bowl`**: Container object state
- **`rubiks_cube`**: Manipulation target object

## File Descriptions

- **`run_{i}.hdf5`**: Per-run data file containing episode data for all envs in that run. Each `demo_j` within the file corresponds to `env_id=j`. Contains:
  - Robot observations (joint states, end-effector poses, etc.)
  - Actions taken by the policy
  - Task-specific metrics and success/failure indicators
  - Episode timestamps and metadata

- **`env_cfg.json`**: Configuration file containing the environment setup, including:
  - Task definition and parameters
  - Robot configuration
  - Observation and action space specifications
  - Scene and lighting settings

## Episode Results

#### `episode_results.jsonl`
The main results file, located at the top level of the output folder. Uses JSONL format (one JSON object per line) for safe append-only writes. Each line corresponds to one episode and contains task metadata, success/failure status, subtask scoring, trajectory metrics, and error events.

> **Note:** Legacy folders may contain `episode_results.json` (JSON array format). All analysis tools read both formats transparently.

```jsonl
{"env_name": "RubiksCubeAndBananaTask", "task_name": "RubiksCubeAndBananaTask", "run_name": "RubiksCubeAndBananaTask_1", "episode": 1, "policy": "pi05", "instruction": "Put the cube and the banana in the bowl", "instruction_type": "default", "attributes": ["conjunction", "simple"], "success": true, "score": 1.0, "reason": "Completed subtask 'pick_and_place' 1/1", "episode_step": 232, "duration": 15.467, "dt": 0.06666666666666667, "metrics": {"ee_sparc": -3.971, "joint_sparc_mean": -6.190, "ee_isj": 213.389, "joint_isj": 5408.055, "ee_path_length": 1.126, "joint_rmse_mean": 0.022, "ee_speed_max": 0.195, "ee_speed_mean": 0.068}, "events": {"TARGET_OBJECT_DROPPED": 4}}
```

**Fields:**
- `env_name`: Environment name (also the subfolder name on disk)
- `task_name`: Base task class name (multiple env_names can share the same task_name)
- `run_name`: Unique run identifier (`<ENV_NAME>_<RUN_IDX>`)
- `run`: Run index (which batch of parallel envs)
- `episode`: Global episode number (`run_idx * num_envs + env_id`)
- `env_id`: Which parallel environment within the run (0 to num_envs-1)
- `policy`: Policy backend used
- `instruction`: Language instruction for the task
- `instruction_type`: Instruction variant used (e.g., `default`, `vague`, `specific`)
- `attributes`: Task attribute tags (e.g., `simple`, `color`, `spatial`, `conjunction`)
- `success`: Whether the episode succeeded
- `score`: Subtask completion score (0.0 to 1.0), present when subtask tracking is enabled
- `reason`: Completion or failure reason, present when subtask tracking is enabled
- `episode_step`: Number of simulation steps in the episode
- `duration`: Episode duration in seconds
- `dt`: Simulation timestep (seconds per step)
- `metrics`: Trajectory quality metrics, computed automatically at the end of each episode from HDF5 data
  - `ee_sparc`: End-effector SPARC smoothness (more negative = smoother)
  - `joint_sparc_mean`: Mean SPARC across arm joints
  - `ee_isj`: End-effector Integrated Squared Jerk
  - `joint_isj`: Joint space Integrated Squared Jerk
  - `ee_path_length`: End-effector path length in meters
  - `joint_rmse_mean`: Mean joint tracking error (action vs actual)
  - `ee_speed_max`: Maximum end-effector speed (m/s)
  - `ee_speed_mean`: Mean end-effector speed (m/s)
- `timing`: Wall-clock timing breakdown per episode (always recorded)
  - `policy_inference_s`: Total time spent in policy server queries
  - `policy_inference_avg_ms`: Average policy query time per step
  - `env_step_s`: Total time spent in `env.step()` (physics + observations + termination)
  - `env_step_avg_ms`: Average env step time per step
  - `video_write_s`: Total time spent encoding video frames
  - `video_write_avg_ms`: Average video write time per step
  - `wall_total_s`: Sum of all timed sections
  - `it_per_sec`: Steps per second (`steps / wall_total_s`)
- `events`: Error event counts, extracted from episode log files at the end of each episode
  - `WRONG_OBJECT_GRABBED`: Robot grabbed an unintended object
  - `GRIPPER_HIT_TABLE`: Gripper made contact with table
  - `GRIPPER_HIT_OBJECT`: Gripper collided with an object
  - `GRIPPER_FULLY_CLOSED`: Gripper closed fully (empty grasp)
  - `TARGET_OBJECT_DROPPED`: Target object was grabbed but dropped mid-transport
  - `MULTIPLE_OBJECTS_GRABBED`: Multiple objects grabbed simultaneously
  - `OBJECT_BUMPED`: Non-target object was nudged
  - `OBJECT_MOVED`: Non-target object was significantly displaced
  - `OBJECT_OUT_OF_SCENE`: Object fell off table or moved outside workspace
  - `wrong_objects_grabbed`: List of wrong object names that were grabbed

## See Also

- [Analysis and Results Parsing](analysis.md) — Scripts for summarizing, comparing, and auditing experiment results
