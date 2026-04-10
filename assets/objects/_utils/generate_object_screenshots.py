# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import os

from robolab.constants import OBJECT_DIR, SCENE_DIR


def _find_usd_files_excluding_materials(dataset_path: str):
    usd_files = []
    for root, dirs, files in os.walk(dataset_path):
        # prune materials and '_' dirs
        dirs[:] = [d for d in dirs if d.lower() != 'materials' and not d.startswith('_')]
        for f in files:
            if f.endswith((".usd", ".usda", ".usdc", ".usdz")):
                usd_files.append(os.path.join(root, f))
    return usd_files


def main():
    """This script takes a screenshot of a USD file or a folder of USD files. Adds default lighting to a scene."""

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})

    from robolab.core.utils.usd_utils import get_aabb # noqa
    from robolab.core.utils.render_utils import render_stage_frame # noqa

    parser = argparse.ArgumentParser(description="Render object dataset screenshots")
    parser.add_argument('--datasets', nargs='+', default=['hope'], help="Datasets to render under assets/objects (e.g., vomp hope ycb)")
    args, _ = parser.parse_known_args()

    object_datasets = args.datasets

    # get_aabb()


    for object_dataset in object_datasets:
        object_dir = os.path.join(OBJECT_DIR, object_dataset)
        output_folder = os.path.join(OBJECT_DIR, "_images", object_dataset)
        os.makedirs(output_folder, exist_ok=True)

        resolution = (240, 240)
        skip_frames = 100
        add_lighting = True
        ground_position = -0.05
        camera_position = (0.2, 0.03, 0.1)
        camera_target = (0.0, 0.0, 0.05)

        if os.path.isdir(object_dir):
            usds = _find_usd_files_excluding_materials(object_dir)
        elif object_dir.endswith((".usd", ".usda", ".usdc", ".usdz")):
            usds = [object_dir]
        else:
            raise ValueError(f"Invalid object path: {object_dir}")

        for usd_path in usds:
            output_path = render_stage_frame(
                app,
                usd_path=usd_path,
                output_dir=output_folder,
                resolution=resolution,
                skip_frames=skip_frames,
                add_lighting=add_lighting,
                add_ground=True,
                camera_position=camera_position,
                camera_target=camera_target,
                ground_position=ground_position,
            )
            print(f"Rendered {usd_path} to {output_path}")


    app.close()

if __name__ == "__main__":
    main()
