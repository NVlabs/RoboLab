# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""Pytest configuration for RoboLab install-verification tests.

Boots Isaac Sim once at conftest load so test files can freely import
isaaclab/robolab modules at their top level. Auto-accepts the Omniverse
EULA so a fresh install works headless without any prompts.
"""

import os
import sys

# Accept the Omniverse EULA non-interactively. Must be set before any
# isaaclab import. setdefault → user can still override.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")

import cv2  # noqa: E402, F401  must be imported before isaaclab

from isaaclab.app import AppLauncher  # noqa: E402

# Launch Isaac Sim once for the whole pytest session. Carb's logger is told to
# only emit warnings or worse so install-verification output isn't drowned in
# Isaac Sim's [Info] startup/shutdown chatter.
_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
    fast_shutdown=True,
    carb_settings={
        "/log/level": "warn",
        "/log/outputStreamLevel": "warn",
        "/log/fileLogLevel": "warn",
    },
)
simulation_app = _launcher.app


def pytest_addoption(parser):
    parser.addoption(
        "--task",
        default="BananaInBowlTask",
        help="Task class name for test_run_empty (default: BananaInBowlTask). "
             "The test runs the first registered env matching this task.",
    )
    parser.addoption(
        "--env-name",
        default=None,
        help="Full registered env name for test_run_empty (e.g. BananaInBowlTaskHomeOffice). "
             "If set, overrides --task.",
    )


import pytest


@pytest.fixture
def task_arg(request):
    return request.config.getoption("--task")


@pytest.fixture
def env_name_arg(request):
    return request.config.getoption("--env-name")


@pytest.fixture
def app_launcher_class():
    """The bootstrap class, before Kit replaces the isaaclab.app namespace."""
    return AppLauncher


def pytest_sessionfinish(session, exitstatus):
    session.config._robolab_exit_status = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Release Kit after pytest prints its report, preserving pytest's status."""
    try:
        import omni.kit.app

        sys.stdout.flush()
        sys.stderr.flush()
        omni.kit.app.get_app().post_quit(getattr(config, "_robolab_exit_status", 0))
        simulation_app.close()
    except Exception:
        pass
