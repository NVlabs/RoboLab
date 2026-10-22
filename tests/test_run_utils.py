# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run offsets and stopping rules are independent of simulator startup."""

import argparse
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from robolab.eval.run_utils import episode_ids_for_run, is_evaluation_complete, is_run_complete
from robolab.eval.runner import add_common_eval_args, run_evaluation


def results(ids, *, task='Task', success=False):
    return [{'env_name': task, 'episode': i, 'success': success} for i in ids]


def test_offset_maps_to_episode_ids():
    assert episode_ids_for_run(0, 2) == [0, 1]
    assert episode_ids_for_run(3, 2) == [6, 7]


def test_complete_run_requires_all_ids_for_the_same_task():
    saved = results([6]) + results([7], task='OtherTask')
    assert not is_run_complete(saved, 'Task', 3, 2)
    assert is_run_complete(saved + results([7]), 'Task', 3, 2)


def test_fixed_budget_checks_only_requested_range():
    options = dict(num_envs=2, num_runs=2, run_index_offset=3)
    assert not is_evaluation_complete(results([0, 1, 6, 7, 8, 10]), 'Task', **options)
    assert is_evaluation_complete(results([6, 7, 8, 9]), 'Task', **options)


def test_duplicate_rows_do_not_fill_missing_fixed_budget_episodes():
    assert not is_run_complete(results([0, 0]), 'Task', 0, 2)


def test_adaptive_resume_does_not_stop_after_default_first_run():
    assert not is_evaluation_complete(results([0]), 'Task', num_envs=1, num_runs=1, adaptive_max=100)


def test_adaptive_stops_at_maximum_and_ignores_other_tasks():
    options = dict(num_envs=1, num_runs=1, adaptive_max=3)
    saved = results([0, 1]) + results(range(10), task='OtherTask')
    assert not is_evaluation_complete(saved, 'Task', **options)
    assert is_evaluation_complete(saved + results([2]), 'Task', **options)


def test_adaptive_can_stop_at_precision_before_maximum():
    assert is_evaluation_complete(results(range(10), success=True), 'Task',
        num_envs=1, num_runs=1, adaptive_max=100, ci_pp_width=0.5)
    assert not is_evaluation_complete(results(range(10), success=True), 'Task',
        num_envs=1, num_runs=1, adaptive_max=100, ci_pp_width=0.01)


def test_adaptive_rejects_nonzero_offset():
    with pytest.raises(ValueError, match='not supported with adaptive'):
        is_evaluation_complete([], 'Task', num_envs=1, num_runs=1, run_index_offset=1, adaptive_max=100)


@pytest.fixture
def runner_harness(monkeypatch, tmp_path):
    import robolab.constants as constants

    state = SimpleNamespace(saved=[], executed=[], headless=[])
    state.env = SimpleNamespace(close=Mock(), reset_eval_state=Mock(), sim=SimpleNamespace(has_gui=Mock(return_value=False)))
    state.client = SimpleNamespace(close=Mock())
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.01), decimation=10, instruction='test')
    state.create_env = Mock(return_value=(state.env, cfg))

    def execute(**kwargs):
        state.executed.append(kwargs['episode'])
        state.headless.append(kwargs['headless'])
        assert len(state.executed) < 10, 'Runner failed to stop at its budget'
        return [], [], {}

    def summarize(**kwargs):
        # Return a new list as the real summarizer may do; the runner must use it
        # for subsequent completion checks.
        ids = episode_ids_for_run(kwargs['run_idx'], kwargs['num_envs'])
        previous = [row for row in kwargs['episode_results'] if row['episode'] not in ids]
        state.saved = previous + results(ids)
        return state.saved

    modules = {
        'robolab.core.environments.factory': {'get_envs': lambda **kwargs: ['Task']},
        'robolab.core.environments.runtime': {'create_env': state.create_env},
        'robolab.core.logging.results': {
            'init_experiment': lambda folder: ('unused.jsonl', state.saved),
            'summarize_experiment_results': lambda *args, **kwargs: None,
        },
        'robolab.core.utils.print_utils': {'print_experiment_summary': lambda **kwargs: None},
        'robolab.eval.episode': {'run_episode': execute},
        'robolab.eval.summarize': {'summarize_run': summarize},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(constants, 'PACKAGE_DIR', str(tmp_path))
    monkeypatch.setattr(constants, 'set_output_dir', lambda folder: None)
    monkeypatch.setattr(constants, 'ENABLE_SUBTASK_PROGRESS_CHECKING', False)

    def run(*cli, legacy_headless=True):
        parser = argparse.ArgumentParser()
        add_common_eval_args(parser)
        args = parser.parse_args(['--task', 'Task', '--output-folder-name', 'test', *cli])
        args.device = 'cpu'
        if legacy_headless:
            args.headless = True
        original_args = vars(args).copy()
        def make_client(received_args):
            assert received_args is args
            return state.client
        run_evaluation(args, policy='test', client_factory=make_client)
        assert vars(args) == original_args

    state.run = run
    return state


@pytest.mark.parametrize('has_gui', [False, True])
@pytest.mark.parametrize('gui_is_property', [False, True])
def test_runner_uses_actual_gui_state_without_legacy_headless_argument(runner_harness, has_gui, gui_is_property):
    h = runner_harness
    h.env.sim.has_gui = has_gui if gui_is_property else Mock(return_value=has_gui)
    h.run(legacy_headless=False)
    assert h.headless == [not has_gui]


def test_runner_resumes_adaptive_after_saved_first_batch(runner_harness):
    h = runner_harness
    h.saved = results([0])
    h.run('--num-episodes-adaptive', '3')
    assert h.executed == [1, 2]
    assert [row['episode'] for row in h.saved] == [0, 1, 2]
    h.env.close.assert_called_once()
    h.client.close.assert_called_once()


def test_runner_resumes_fixed_range_with_offset(runner_harness):
    h = runner_harness
    h.saved = results([4, 5])
    h.run('--num-envs', '2', '--num-runs', '2', '--run-index-offset', '2')
    assert h.executed == [3]
    assert [row['episode'] for row in h.saved] == [4, 5, 6, 7]


def test_runner_recovers_partial_batch_within_requested_range(runner_harness):
    h = runner_harness
    h.saved = results([4, 6, 7])
    h.run('--num-envs', '2', '--num-runs', '2', '--run-index-offset', '2')
    assert h.executed == [2]
    assert sorted(row['episode'] for row in h.saved) == [4, 5, 6, 7]


@pytest.mark.parametrize('adaptive', [False, True])
def test_completed_task_skips_simulator_creation(runner_harness, adaptive):
    h = runner_harness
    h.saved = results([0, 1, 2])
    h.run(*(['--num-episodes-adaptive', '3'] if adaptive else ['--num-runs', '3']))
    h.create_env.assert_not_called()
    assert h.executed == []


def test_adaptive_precision_can_skip_simulator_creation(runner_harness):
    h = runner_harness
    h.saved = results(range(10), success=True)
    h.run('--num-episodes-adaptive', '100', '--ci-pp-width', '0.5')
    h.create_env.assert_not_called()
