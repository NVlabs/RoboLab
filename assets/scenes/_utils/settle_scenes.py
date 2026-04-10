# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Scene settling utility for physics stabilization.

This script opens USD scene files, runs physics simulation to allow objects to settle
into stable positions, and saves the stabilized scenes. This is useful for ensuring
that scenes start in physically-stable configurations rather than with objects in
mid-air or unstable positions.

Usage:
    # Settle a single scene (creates *.usda)
    python settle_scenes.py --scene banana_bowl.usda

    # Settle multiple scenes
    python settle_scenes.py --scene banana_bowl.usda apple_plate.usda

    # Settle all scenes in a directory
    python settle_scenes.py --scene /path/to/scenes/

    # Replace original scenes instead of creating new files
    python settle_scenes.py --scene banana_bowl.usda apple_plate.usda --replace

    # Settle and generate screenshots
    python settle_scenes.py --scene banana_bowl.usda --screenshot

    # Settle and save screenshots to a custom directory
    python settle_scenes.py --scene banana_bowl.usda --screenshot --screenshot-dir /path/to/output
"""

import os

from isaacsim import SimulationApp

from robolab.constants import SCENE_DIR
from robolab.core.scenes import utils as scene_utils


def clean_physics_for_export(stage):
    """Clean physics artifacts from a settled stage before export.

    Removes:
    - physics:velocity and physics:angularVelocity on all prims (residual
      velocities cause objects to drift when reloaded)
    - PhysicsScene prims (conflict with Isaac Lab's own physics scene when
      the USD is loaded as a reference; objects end up with mismatched solver
      settings)
    """
    from pxr import Usd

    prims_to_remove = []
    for prim in stage.Traverse():
        # Clear velocity attributes
        for attr_name in ("physics:velocity", "physics:angularVelocity"):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsAuthored():
                attr.Clear()

        # Mark PhysicsScene prims for removal (can't remove while traversing)
        if prim.GetTypeName() == "PhysicsScene":
            prims_to_remove.append(prim.GetPath())

    for prim_path in prims_to_remove:
        stage.RemovePrim(prim_path)


def open_and_save_scene(scene_path: str, output_path: str, simulation_app: SimulationApp):
    """Open a USD scene, run physics simulation to settle objects, and save the result.

    Args:
        scene_path: Path to the input USD scene file
        output_path: Path where the stabilized scene should be saved
        simulation_app: Active SimulationApp instance for running the simulation
    """
    import omni.timeline
    import omni.usd
    omni.usd.get_context().open_stage(scene_path)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(300):  # ~5s at 60Hz (adjust as needed)
        simulation_app.update()
    timeline.pause()
    stage = omni.usd.get_context().get_stage()
    clean_physics_for_export(stage)
    stage.GetRootLayer().Export(output_path)

def open_and_save_multiple_scenes(scene_paths: list[str], replace: bool = False,
                                   screenshot: bool = False, screenshot_dir: str | None = None):
    """Process multiple USD scenes to stabilize physics.

    Args:
        scene_paths: List of paths to USD scene files to process
        replace: If True, replaces original files with stabilized versions.
                If False, creates new files in the SCENE_DIR.
        screenshot: If True, generates a screenshot of each scene after settling.
        screenshot_dir: Directory to save screenshots. Defaults to SCENE_DIR/_images.
    """
    simulation_app = SimulationApp({"headless": True})

    if screenshot:
        from robolab.core.utils.render_utils import render_stage_frame
        if screenshot_dir is None:
            screenshot_dir = os.path.join(SCENE_DIR, "_images")

    for usd_path in scene_paths:
        base = os.path.basename(usd_path)
        usd_dir = os.path.dirname(usd_path)
        name, ext = os.path.splitext(base)
        print(f"Settling scene: {usd_path}")

        if replace:
            # Save to temporary file first
            temp_output_path = os.path.join(usd_dir, f"{name}_temp{ext}")
            open_and_save_scene(usd_path, temp_output_path, simulation_app)
            # Delete the old file and rename the new file
            if os.path.exists(usd_path):
                os.remove(usd_path)
            os.rename(temp_output_path, usd_path)
            print(f"Deleted temp: {temp_output_path} and replaced scene: {usd_path}")
            settled_path = usd_path
        else:
            output_path = os.path.join(SCENE_DIR, f"{name}{ext}")
            open_and_save_scene(usd_path, output_path, simulation_app)
            print(f"Saved stabilized scene: {output_path}")
            settled_path = output_path

        # Generate screenshot of the settled scene
        if screenshot:
            screenshot_path = render_stage_frame(
                simulation_app,
                usd_path=settled_path,
                output_dir=screenshot_dir,
                resolution=(640, 480),
                skip_frames=100,
                camera_position=(-0.3, 0.3, 0.7),
                camera_target=(0.5, 0.0, 0.0),
                add_lighting=True,
            )
            print(f"Screenshot saved: {screenshot_path}")

    simulation_app.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Settle a scene")
    parser.add_argument("--scene", type=str, nargs="+", default=[SCENE_DIR], help="Path(s) to scene(s) or directories to settle")
    parser.add_argument("--replace", action="store_true", help="Replace the scene if it already exists")
    parser.add_argument("--screenshot", action="store_true", help="Generate a screenshot after settling each scene")
    parser.add_argument("--screenshot-dir", type=str, default=None, help="Directory to save screenshots (default: SCENE_DIR/_images)")
    args = parser.parse_args()
    scene_paths = []
    for scene in args.scene:
        if os.path.isdir(scene):
            from robolab.core.scenes.utils import get_scenes_from_folder
            scene_paths.extend(get_scenes_from_folder(scene))
        else:
            scene_path = scene_utils.find_scene_file(scene, SCENE_DIR)
            if scene_path:
                scene_paths.append(scene_path)
            else:
                raise ValueError(f"Scene not found: {scene}")
    open_and_save_multiple_scenes(scene_paths, args.replace,
                                  screenshot=args.screenshot,
                                  screenshot_dir=args.screenshot_dir)
