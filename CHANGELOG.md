# Changelog

## [0.3.3] - 2026-11-15

### Added

- Opt-in Isaac Lab 3.0 beta 2 patch 1 / Isaac Sim 6.0.1 support with
  Python 3.12, a resolved dependency lock, setup documentation, and task
  preflight commands. Isaac Sim 5.0 remains the default.
  ([#47](https://github.com/NVlabs/RoboLab/pull/47))
- New lighting presets (`lighting_selection=`): nine light-and-sky setups,
  from the default sphere light to window, ceiling, dark, and spotlight scenes
  (`docs/lighting.md`).
- Orchestrated AI-native policy: a Cosmos Action model proposes joint
  trajectories, and an agent reviews each chunk with camera views and robot
  state before executing, editing, or replacing it with end-effector
  corrections.
- GPU details stored in recordings and compared during replay.
- Initial-pose randomization (`--init-pose-noise`, with `--init-pose-objects`
  and `--init-pose-workspace`): each episode starts objects in a different,
  collision-checked layout that is reproducible per seed. Tasks can keep
  objects together with `object_groups` or in place with `fixed_objects`
  (`docs/init_pose_randomization.md`).

### Changed

- The `robolab/tasks/randomize_initial_pose/` task variants are removed; use
  `--init-pose-noise` instead.

### Fixed

- Removed a bowl and banana that were stuck inside the table in
  WhiteMugInCenterOfTableTask's scene. Scores on that task may change.
- DROID/Robotiq rendering and plate/fork USD hierarchies on Isaac Sim 6.
- Quaternion conversion at simulator boundaries, including explicit WXYZ
  configuration defaults and consistent orientations in multi-camera resets.
- Evaluation display-mode detection across Isaac Lab 2 and 3.
- `env_spacing` passed to `create_env` / `parse_env_cfg` now changes the
  spacing between parallel environments.
  ([#41](https://github.com/NVlabs/RoboLab/pull/41))
- The VoLo runner defaults `--open-loop-horizon` to 8, matching the 8-action
  chunks the VoLo proxy serves; the old default crashed on the first tool
  chunk. ([#37](https://github.com/NVlabs/RoboLab/pull/37))

## [0.3.2] - 2026-10-18

### Added

- Astranomy Direct-API and Codex Agent-EEF policy backends, with configurable
  model access, action representations, history selection, and inference logs.
- Optional episode context sent once before inference, allowing compatible
  policy clients to use video demonstrations for in-context learning (ICL).
- Object-counting task predicates, multiple tracked instruction fields, and
  support for multi-articulation end-effector recording and gripper labels.

### Changed

- Cosmos3 supports full WebSocket URIs, optional Bearer authentication, and
  separate policy sessions for parallel environments. Cosmos3 and DreamZero
  share a MsgPack/NumPy WebSocket transport.
- Batched policy replans are divided into bounded per-request environment
  groups.

### Fixed

- Ground meshes no longer occlude cleaning-scene backgrounds.
- Recorded subtask conditionals resolve correctly during replay.

## [0.3.1] - 2026-08-11

### Added

- Kinova Gen3 (Robotiq 2F-85) fixed-base support: robot config, joint-position
  environment registrations, and an example runner
  (`examples/run_kinova_jointpos.py`).
  ([#23](https://github.com/NVlabs/RoboLab/pull/23))
- Floor-standing robot support: a robot config can declare the
  `root_z_above_ground` label (the reach of its lowest colliders below the
  root), and the environment factory places the root on each task scene's
  authored ground plane plus that offset (`docs/robots.md`).
- Per-scene ground-height lock (`tests/test_scene_ground.py`): the task table
  is a dynamic rigid body resting on the ground plane, so the authored ground
  height sets the tabletop height and is now locked per scene.

### Changed

- Every robot config must now declare the `ee_recorder_bodies` label mapping
  HDF5 channel names to the articulation bodies recorded for end-effector
  pose (`{}` opts out). **Breaking for custom robot configs**: environment
  generation fails with a `ValueError` naming the config until the label is
  added — see `docs/robots.md#end-effector-pose-recording`.
- Remaining task-scene ground planes aligned to the canonical -0.697 height;
  legacy scenes keep their original -0.65 ground to preserve replay
  compatibility with existing recordings.

## [0.3.0] - 2026-08-07

### Added

- Explicit coordinate-frame contract: end-effector observations are now
  published in the robot-root frame, with a compatibility shim for older
  recordings (`robolab/core/logging/frame_compat.py`) and a reference doc
  (`docs/frames.md`).
- Ground-truth state export during stepping: `--enable-gt-state` makes every
  runner emit a raw per-env world-state snapshot each step under
  `obs["gt_state"]` (`robolab.eval.GroundTruthStateExporter`; schema in
  `docs/environment_run.md`).
- VoLo policy backend (`policies/volo/`): a proxy wrapper over existing
  clients that adds depth observations, camera calibration, and ground-truth
  state metadata to the inference payload.
- Robot-owned scene fixtures with per-robot fixture selection
  (`robolab/core/environments/scene_fixture.py`): a robot config declares the
  table fixture it is mounted on (or none, for robots with their own base),
  and the environment factory swaps the task scene's default fixture
  accordingly.
- Multi-gripper support in the task conditional system: robots can declare
  named gripper groups (e.g. `"gripper": ["left", "right"]` on a bimanual
  robot), and success/failure predicates can target any member of a group or
  require several grippers simultaneously (see `docs/task_conditionals.md`
  and `docs/event_tracking.md`).
- Pull request template and documentation of accepted PR types in
  `CONTRIBUTING.md`.
- News section in the README and an ecosystem page (`docs/ecosystem.md`).

### Fixed

- Explicitly-named tasks now resolve correctly from scoped content-pack
  folders.

## [0.2.1] - 2026-07-20

### Added

- Faithful replay of recorded episodes: new `robolab/core/replay/` module with
  recorded env-config overlay, initial-state restore, and per-step state
  validation; driver script `examples/run_recorded.py`; user guide
  `docs/replay.md`. ([#19](https://github.com/NVlabs/RoboLab/issues/19))
- Batched inference support in the eval client interface.
- The pytest install-verification suite now ships in `tests/` (documented
  fresh-install check: `uv run pytest tests/`).

### Changed

- Subtask progress tracking is enabled by default during evaluation.
- Updated the GR00T N1.7 client.
  ([#15](https://github.com/NVlabs/RoboLab/pull/15))
- Overrode isaacsim's stale `websockets==12.0` pin so `uv lock` resolves a
  modern websockets for the policy clients.

### Fixed

- `WorldState` extras handling on IsaacLab 2.3 (`XformPrimView`).

## [0.2.0] - 2026-07-07

### Added

- IsaacSim 5.1 / IsaacLab 2.3 support alongside 5.0 / 2.2, selected via the
  mutually-exclusive `isaac50` / `isaac51` extras.
- Relative differential-IK environment registrations
  (`examples/run_rel_ik_demo.py`).
- Realtime/pathtracing RTX renderer selection via `--rendering-type`.
- Poly Haven attribution files for the curated backgrounds.

### Changed

- Bundled backgrounds trimmed to a curated indoor set. More backgrounds can be obtained from Poly Haven.
- End-effector pose position is recorded in the env-local frame.
- Docs: asset contribution guidelines, conditionals
  restructure; generalized VRAM sizing guide.

### Fixed

- HDF5 episode success is derived from `terminated` rather than a named
  termination term.

## [0.1.1] - 2026-05-31

### Added

- Cosmos3 policy client; refreshed DreamZero, GR00T, and Pi0-family clients.
- Results dashboard (`robolab-dashboard` CLI).
- `/robolab-scenegen` and `/robolab-taskgen` Claude Code skills.
- Convex-hull placement predicates, adaptive sampling, and DROID IK control.
- `CONTRIBUTING.md` and a `uv`-based install flow.

### Changed

- Per-policy reorganization: one runner per backend under
  `policies/<policy>/run.py`.
- Docs revamp: per-policy READMEs, SPDX headers, `THIRD_PARTY_NOTICES.md`.
- Episode videos encoded as streaming H.264 (libx264) so they play in
  browsers.

### Fixed

- Placement order enforced in then-sequenced tasks.
- Frozen-env data leaks in the episode recorder.

## [0.1.0] - 2026-04-10

Initial public release: 100+ benchmark tasks,
DROID and Franka embodiments, server-client policy evaluation (Pi0 family,
GR00T, DreamZero), and HDF5 episode recording with analysis tools.

[0.3.3]: https://github.com/NVlabs/RoboLab/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/NVlabs/RoboLab/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/NVlabs/RoboLab/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/NVlabs/RoboLab/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/NVlabs/RoboLab/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NVlabs/RoboLab/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/NVlabs/RoboLab/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/NVlabs/RoboLab/releases/tag/v0.1.0
