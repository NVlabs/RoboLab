# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Pretty-print helpers for RoboLab experiment logging."""

CYAN = "\033[96m"
RESET = "\033[0m"


def print_experiment_summary(
    task_envs: list[str],
    filter_str: str,
    num_envs: int,
    num_episodes: int,
    policy: str,
    instruction_type: str,
    output_dir: str,
):
    """Print a formatted experiment summary block in cyan."""
    print(f"{CYAN}{'═' * 62}")
    print(f"  RoboLab Experiment")
    print(f"{'─' * 62}")
    print(f"  Filter         : {filter_str}")
    print(f"  Environments   : {', '.join(task_envs)}")
    print(f"  Num Tasks      : {len(task_envs)}")
    print(f"  Num Envs       : {num_envs}")
    print(f"  Episodes       : {num_episodes} per task")
    print(f"  Policy         : {policy}")
    print(f"  Instr. Type    : {instruction_type}")
    print(f"  Output         : {output_dir}")
    print(f"{'═' * 62}{RESET}")


def print_env_info(
    env_name: str,
    instruction: str,
    instruction_type: str,
    seed: int,
    policy: str,
    scene_name: str,
    attributes: list[str] | None = None,
):
    """Print a formatted per-environment info block in cyan."""
    attrs = ', '.join(attributes) if attributes else 'none'
    print(f"{CYAN}┌{'─' * 62}┐")
    print(f"│  Environment : {env_name:<45} │")
    print(f"├{'─' * 62}┤")
    print(f"│  Instruction : {str(instruction):<45} │")
    print(f"│  Instr. Type : {instruction_type:<45} │")
    print(f"│  Seed        : {seed:<45} │")
    print(f"│  Policy      : {policy:<45} │")
    print(f"│  Scene       : {scene_name:<45} │")
    print(f"│  Attributes  : {attrs:<45} │")
    print(f"└{'─' * 62}┘{RESET}")
