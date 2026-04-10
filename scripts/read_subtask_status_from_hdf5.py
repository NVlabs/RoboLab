# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os

from robolab.core.logging.results import load_timestep_from_config, print_all_episodes, print_episode_subtask_status


def main():
    parser = argparse.ArgumentParser(
        description='Read and display subtask status information from HDF5 files.'
    )
    parser.add_argument(
        'file',
        type=str,
        help='Path to the HDF5 file'
    )
    parser.add_argument(
        '-e', '--episode',
        type=int,
        default=None,
        help='Episode number to display (e.g., 0 for demo_0). If not provided, shows all episodes.'
    )

    args = parser.parse_args()

    # Load timestep from config file
    config_path = os.path.join(os.path.dirname(args.file), "env_cfg.json")
    step_dt = load_timestep_from_config(config_path)

    print(f"File: {args.file}")
    if step_dt is not None:
        print(f"Loaded timestep: {step_dt:.4f}s ({1/step_dt:.2f} Hz)\n")
    else:
        print("No timestep information available. Only step indices will be shown.\n")

    if args.episode is not None:
        print_episode_subtask_status(args.episode, args.file, step_dt, indent="")
    else:
        print_all_episodes(args.file, step_dt)

if __name__ == '__main__':
    main()
