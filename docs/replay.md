# Replaying Recorded Episodes

RoboLab records every episode to HDF5 ([Data Storage and Output](data.md)). `examples/run_recorded.py` replays a recorded episode: it rebuilds the environment, restores the recorded initial scene state, and steps the recorded actions open-loop while the usual termination checking, subtask tracking, and video/HDF5 recording run as normal. Use it to verify recordings, sanity-check task conditionals against a known trajectory, or debug environment changes against a reference episode. The replay helpers (config overlay, state restore, validation, provenance) live in `robolab/core/replay/` for use in your own drivers.

## Quick Start

Replay the bundled demonstration:

```bash
python examples/run_recorded.py --headless
```

Replay your own recording:

```bash
python examples/run_recorded.py --task <TaskName> --recorded-data-folder <folder> --headless
```

The script expects this layout:

```
<recorded-data-folder>/
└── <TaskName>/
    ├── data.hdf5        # the recording (override the filename with --file)
    └── env_cfg.json     # env config saved by the recording run (used by default, see below)
```

Evaluation runs already produce this layout (they write `run_{i}.hdf5` instead of `data.hdf5`), so an eval output replays directly with `--file`, and `--episode` selects the demo within the file:

```bash
# Replay env 2's episode from run_0.hdf5 of an eval output
python examples/run_recorded.py --task <TaskName> --recorded-data-folder output/<run_folder> --file run_0.hdf5 --episode 2 --headless
```

By default `demo_0` is replayed; in a multi-env recording, `demo_i` is env `i`'s episode. Results are written to `output/playback_<folder>_<task>/`, including the replay's own exported HDF5 (`run_<episode>.hdf5`), videos, and subtask logs.

## What Playback Restores

A faithful replay needs three things from the recording, and all three are restored by default:

| Restored | Source | Notes |
|---|---|---|
| Initial scene state | `data.hdf5` → `demo_0/initial_state` | Robot joints and object poses are reset via `env.reset_to()`, exactly as recorded. Without this, a fresh `reset()` re-settles objects from USD poses and the replay diverges mid-episode. |
| Actions | `data.hdf5` → `demo_0/actions` | Stepped open-loop; no policy is involved. |
| Environment config | `env_cfg.json` sidecar | Object init poses, physics/solver params, termination params, seed, instruction — the exact values the episode was recorded with (see next section). |

Runtime choices always come from the CLI, never the recording: `--num_envs`, `--device`, `--headless`, rendering settings, and recorder configuration.

## Replaying with the Recorded Env Config (`--env-config`)

By default (`--env-config recorded`), playback overlays the `env_cfg.json` saved next to the recording onto a freshly built config, so replay is unaffected by later changes to the repo's task and scene definitions. Fields that no longer exist in the current config schema (or changed type) are skipped and listed in a warning.

Pass `--env-config current` to rebuild the config from the current repo instead — useful to ask "how does this recorded trajectory behave under my *new* task definition?". A yellow warning is printed whenever the recorded config is not fully in effect (sidecar missing, `current` requested, or fields skipped), since the env config then differs from recording time and behavior may diverge.

The overlay restores config **values**, not code or assets. It cannot protect against:

- changed predicate/conditional *implementations* (the config stores only the callable's import path),
- changed USD file *contents* on disk (the config stores only asset paths),
- a different simulator stack (see below).

## Faithful Reproduction Checklist

Contact-rich physics amplifies tiny numerical differences, so reproducing a recorded outcome requires matching the recording context:

1. **Same simulator stack.** Isaac Sim 5.x and 6.0 ship different PhysX builds; recorded outcomes are not invariant across them. RoboLab translates recorded WXYZ scene states for Isaac Lab 3's internal XYZW convention, but that does not remove physics drift. Recordings carry `isaaclab_version` / `isaacsim_version` / `recorded_at` HDF5 attrs and playback prints a notice on mismatch.
2. **Single env, recorded and replayed.** All parallel envs share one batched physics scene, so a trajectory recorded in a multi-env batch evolves slightly differently when replayed alone (and vice versa). Record with `--num_envs 1` and replay with `--num_envs 1` (the default); playback prints a notice when replaying with more.
3. **Recorded env config in effect.** Keep the `env_cfg.json` sidecar next to the recording and leave `--env-config recorded` (the default).
4. **Bundle the replay's own export.** Replay→replay is deterministic. To create a demonstration file that reproduces reliably, replay a recorded success once and keep the HDF5 that the *replay* exports (`output/playback_.../run_0.hdf5`, small — no image observations) together with the `env_cfg.json` from that same playback folder. The bundled `examples/recorded_data/RubiksCubeAndBananaTask/` demo was produced this way.

Do not add settling steps before replay: settling is part of the recorded action stream, and the initial-state restore already puts the scene in the exact recorded pre-settle state.

## Validating Replay Fidelity (`--validate-states`)

The recording also stores the full scene state at every step (`demo_0/states`). `--validate-states` compares the simulated state against it each step and reports drift — turning "the replay diverged" from a guess into a measurement:

```bash
python examples/run_recorded.py --headless --validate-states
```

A faithful replay reports:

```
STATE VALIDATION: replay tracked the recording over 648 steps; max drift 0.0000 on None (tolerance 0.01).
```

A diverging replay reports when and where it left the recording:

```
STATE VALIDATION: drift first exceeded tolerance 0.01 at step 431 (rigid_object/banana/root_pose: 0.0413).
STATE VALIDATION: replay diverged from the recording. Max drift 42.2192 at step 406 on
rigid_object/banana/root_velocity; first exceeded tolerance 0.01 at step 0. Fields over tolerance (8/12): ...
```

Comparison uses env 0 (matching the single-env recipe). Note that quaternion sign flips (`q` vs `-q` encode the same rotation) are not normalized, so a pose drift of ~2.0 on an otherwise tracking replay usually indicates a sign flip rather than a real divergence.

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--task`, `-t` | `RubiksCubeAndBananaTask` | Task name; must be a folder inside the recorded data folder. |
| `--recorded-data-folder`, `--dir` | `examples/recorded_data` | Folder containing `<TaskName>/<file>`. |
| `--file` | `data.hdf5` | HDF5 filename inside `<recorded-data-folder>/<TaskName>/` (e.g. `run_0.hdf5`). |
| `--episode` | `0` | Demo index to replay (`demo_<episode>`; in multi-env recordings `demo_i` is env `i`'s episode). |
| `--env-config {recorded,current}` | `recorded` | Replay with the recorded `env_cfg.json` (faithful) or the current repo's config. |
| `--validate-states` | off | Compare sim state against the recorded per-step states and report drift. |
| `--num_envs` | `1` | Parallel envs; keep at 1 for faithful reproduction. |
| `--disable-subtask` | on | Disable subtask progress checking during replay. |
| `--headless` | off | Run without a viewer (standard AppLauncher flag). |
