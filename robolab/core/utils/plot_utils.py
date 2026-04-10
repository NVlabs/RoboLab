# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import matplotlib.pyplot as plt
import numpy as np


def plot_objects(obj_poses: list[dict] | dict, episode_id = 0, title="", image_path=None):

    fig = plt.figure()
    colors = ['y', 'r', 'b', 'g', 'm', 'c', 'k']

    if isinstance(obj_poses, dict):
        obj_poses = [obj_poses]
        episode_ids = [episode_id]
    elif isinstance(obj_poses, list):
        episode_ids = [i for i in range(len(obj_poses))]

    # Create subplot once, outside the loop
    ax = fig.add_subplot(111, projection='3d')

    for i, obj_pose_dict in enumerate(obj_poses):
        for idx, (object, pose) in enumerate(obj_pose_dict.items()):
            position = pose[:3]
            object_label = f"{object}_{episode_ids[i]}"
            ax.scatter(position[0], position[1], position[2], color=colors[idx], s=30, marker='o', label=object_label)
            ax.text(position[0], position[1], position[2], object_label, color=colors[idx], fontsize=9)

    # Configure axes once after all data is plotted
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    # Set top-down view (elevation=90 looks straight down, azimuth=0)
    ax.view_init(elev=45, azim=120)
    # Set axis limits from -1 to 1
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-0.5, 0.5)

    plt.tight_layout()

    # Save to file
    plt.savefig(image_path)
    plt.close(fig)
