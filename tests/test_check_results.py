# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""analysis/check_results.py matches episodes to the HDF5 files the runner writes."""

import json
import re

import h5py

from analysis.check_results import check_folder, diagnose_hdf5, print_results


def write_hdf5(path, demos):
    with h5py.File(path, "w") as f:
        for demo in demos:
            f.create_group(f"data/{demo}").attrs["num_samples"] = 1


def make_results_folder(root):
    """Two runs x two envs (run_*.hdf5), one legacy data.hdf5 task, one task with no HDF5."""
    rows = [{"env_name": "MultiEnvTask", "run": run, "env_id": env_id,
             "episode": run * 2 + env_id, "success": True}
            for run in range(2) for env_id in range(2)]
    rows += [{"env_name": "LegacyTask", "episode": i, "success": True} for i in range(3)]
    rows += [{"env_name": "NoDataTask", "run": 0, "env_id": i, "episode": i, "success": False} for i in range(2)]
    (root / "episode_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    for task in ("MultiEnvTask", "LegacyTask", "NoDataTask"):
        (root / task).mkdir()
    # Each run file holds demo_{env_id}; env 1 of run 1 was never written.
    write_hdf5(root / "MultiEnvTask" / "run_0.hdf5", ["demo_0", "demo_1"])
    write_hdf5(root / "MultiEnvTask" / "run_1.hdf5", ["demo_0"])
    # Legacy layout holds demo_{episode}; episode 1 was never written.
    write_hdf5(root / "LegacyTask" / "data.hdf5", ["demo_0", "demo_2"])


def test_check_folder_maps_episodes_to_run_files(tmp_path):
    make_results_folder(tmp_path)

    results = check_folder(str(tmp_path))

    assert (results["total"], results["valid"]) == (9, 5)
    assert (results["missing_demo"], results["missing_hdf5"]) == (2, 2)
    assert results["missing_hdf5_envs"] == ["NoDataTask"]
    assert sorted((c["env_name"], c["episode"]) for c in results["corrupt_episodes"]) == [
        ("LegacyTask", 1), ("MultiEnvTask", 3), ("NoDataTask", 0), ("NoDataTask", 1),
    ]


def test_diagnose_and_report_list_only_missing_episodes(tmp_path, capsys):
    make_results_folder(tmp_path)

    diagnose_hdf5(str(tmp_path))
    print_results(check_folder(str(tmp_path)), str(tmp_path))
    out = re.sub(r"\033\[[0-9;]*m", "", capsys.readouterr().out)

    multi_env = out.split("Environment: MultiEnvTask")[1].split("Environment:")[0]
    assert "Missing in HDF5: [3]" in multi_env
    assert "Extra in HDF5" not in multi_env
    legacy = out.split("Environment: LegacyTask")[1].split("Environment:")[0]
    assert "Missing in HDF5: [1]" in legacy
    assert "Tasks with no HDF5 file:\n  - NoDataTask" in out
