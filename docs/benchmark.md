# Benchmark

The RoboLab Benchmark evaluates generalist robot manipulation policies across three **competency axes** spanning three **difficulty levels**. It consists of **120 tasks** defined in [`robolab/tasks/benchmark/`](../robolab/tasks/benchmark/).

## At a Glance

| Stat | Value |
|:--|:--|
| Tasks | 120 |
| Average subtasks per task | 2.02 |
| Average objects per task | 9.0 |
| Average difficulty score | 2.90 |
| Evaluation time | ~40 GPU hours / 120 tasks  (assuming ~200ms inference step)|
| Average speed | 1.4 it/s (assuming ~200ms inference step) |

Difficulty distribution: **simple** 64 (53.3%) · **moderate** 39 (32.5%) · **complex** 17 (14.2%)

## Competency Axes

The benchmark evaluates policies along three competency axes:

- **Visual** — recognition of visual traits: color, semantics, and size
- **Relational** — understanding of inter-object temporal, numerical, and spatial relationships
- **Procedural** — action-oriented reasoning including affordances, reorientation, and stacking

### Task Attributes

Attributes are organized into three higher-level categories (mapping defined in [`robolab/constants.py`](../robolab/constants.py) → `BENCHMARK_TASK_CATEGORIES`):

| Category | Attributes | Tasks |
|:--|:--|:--|
| **Visual** | color (26), semantics (60), size (6) | 84 |
| **Relational** | conjunction (8), counting (7), spatial (29) | 42 |
| **Procedural** | affordance (12), reorientation (6), sorting (12), stacking (6) | 34 |

Tasks may carry multiple attributes across axes. An additional `vague` tag (7 tasks) indicates ambiguous language instructions. 3 tasks are untagged.

## Comparison to DROID Dataset

The benchmark emphasizes multi-step and two-step tasks compared to the DROID training distribution. In addition, only **68.7%** of benchmark objects appear in the DROID training vocabulary (word-level overlap: 68 of 99 benchmark object words appear in DROID's 2,760-word vocabulary).

## Difficulty Scoring

A task's difficulty score combines manipulation volume with skill complexity:

```
difficulty_score = num_subtasks + max(skill_weight)
```

where `num_subtasks` is the number of subtask actions and `max(skill_weight)` is the highest weight among the task's non-difficulty attributes. The skill weights are:

| Weight | Attributes |
|:--|:--|
| 0 | color, semantics, size, conjunction, vague |
| +1 | spatial |
| +2 | counting, sorting, stacking, affordance |
| +3 | reorientation |

The score is mapped to a label using fixed thresholds (defined in `DIFFICULTY_THRESHOLDS`):

| Label | Score Range |
|:--|:--|
| **simple** | score ≤ 2 |
| **moderate** | score 3–4 |
| **complex** | score ≥ 5 |

## Key Files

| File | Description |
|:--|:--|
| [`robolab/constants.py`](../robolab/constants.py) | `SKILL_WEIGHTS`, `DIFFICULTY_THRESHOLDS`, `BENCHMARK_TASK_CATEGORIES` |
| [`robolab/core/task/subtask_utils.py`](../robolab/core/task/subtask_utils.py) | `compute_difficulty_score(num_subtasks, attributes)` |
| [`robolab/tasks/_utils/load_task_info.py`](../robolab/tasks/_utils/load_task_info.py) | Metadata pipeline — populates `num_subtasks`, `difficulty_score`, `difficulty_label` |
| [`robolab/tasks/_utils/generate_task_metadata.py`](../robolab/tasks/_utils/generate_task_metadata.py) | Generates `task_metadata.json`, `task_report.txt`, `task_table.csv` |
| [`robolab/tasks/_utils/compute_task_statistics.py`](../robolab/tasks/_utils/compute_task_statistics.py) | Prints summary statistics to stdout |
| [`robolab/tasks/_metadata/`](../robolab/tasks/_metadata/) | Pre-generated metadata: `task_metadata.json`, `task_report.txt`, `task_table.csv` |

## Generating Statistics

```bash
# Print difficulty distribution
python robolab/tasks/_utils/compute_task_statistics.py --difficulty

# Include per-task breakdown
python robolab/tasks/_utils/compute_task_statistics.py --difficulty -v

# Regenerate metadata files (task_metadata.json, task_report.txt, task_table.csv)
python robolab/tasks/_utils/generate_task_metadata.py
```
