#!/usr/bin/env python3
"""
Scatter Objects Script

This script takes all objects from handal, hope, hot3d, ycb, and vomp datasets,
places them on a ground plane scene, and renders a single image using IsaacSim.

Objects are positioned using their bounding box information to ensure they sit
properly on the ground plane.

Usage:
    python scatter_objects.py [--datasets handal hope hot3d ycb vomp] [--output-dir OUTPUT_DIR] [--grid-spacing 0.15]
"""

import os
import json
import math
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Scatter objects from datasets onto a ground plane and render")
    parser.add_argument('--datasets', nargs='+', default=['handal', 'hope', 'hot3d', 'ycb', 'fruits_veggies'],
                        help="Datasets to include (default: handal hope hot3d ycb vomp)")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Output directory for rendered image (default: assets/objects/_utils/output)")
    parser.add_argument('--grid-spacing', type=float, default=0.10,
                        help="Spacing between objects in grid layout (meters)")
    parser.add_argument('--resolution', type=int, nargs=2, default=[1280, 720],
                        help="Output image resolution (width height)")
    parser.add_argument('--skip-frames', type=int, default=30,
                        help="Frames to skip before capturing (for scene stabilization)")
    parser.add_argument('--randomize', action='store_true',
                        help="Randomize object orientations and drop from staggered heights")
    parser.add_argument('--drop-height-min', type=float, default=0.1,
                        help="Minimum drop height in meters (default: 0.1)")
    parser.add_argument('--drop-height-max', type=float, default=0.5,
                        help="Maximum drop height in meters (default: 1.0)")
    args, _ = parser.parse_known_args()

    # Initialize IsaacSim - must be done before importing other omni/pxr modules
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})

    # Now import the rest
    from pxr import UsdGeom, Gf
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.api import World
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from omni.isaac.core.utils.viewports import set_camera_view
    import omni.isaac.core.utils.prims as prim_utils
    import omni.usd
    from PIL import Image

    # Setup paths
    script_dir = Path(__file__).parent
    objects_dir = script_dir.parent

    catalog_path = objects_dir / "object_catalog.json"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the object catalog
    print(f"Loading object catalog from: {catalog_path}")
    with open(catalog_path, 'r') as f:
        catalog = json.load(f)

    # Filter objects by dataset
    target_datasets = set(args.datasets)
    all_objects = [obj for obj in catalog if obj.get('dataset') in target_datasets]

    # Separate vomp shelf/container objects from main objects
    objects_to_add = []
    side_objects = []  # Shelves and containers go off to the side
    for obj in all_objects:
        obj_class = obj.get('class', '').lower()
        if obj.get('dataset') == 'vomp' and obj_class in ('container', 'shelf'):
            side_objects.append(obj)
        else:
            objects_to_add.append(obj)

    print(f"\nFiltering objects from datasets: {target_datasets}")
    print(f"Found {len(objects_to_add)} objects to add to scene")
    if side_objects:
        print(f"Found {len(side_objects)} shelf/container objects (will be placed off to the side)")

    # Count per dataset
    dataset_counts = {}
    for obj in objects_to_add:
        ds = obj.get('dataset', 'unknown')
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    for ds, count in sorted(dataset_counts.items()):
        print(f"  - {ds}: {count} objects")

    if not objects_to_add:
        print("No objects found! Exiting.")
        app.close()
        return

    # Create a new stage
    print("\nCreating new USD stage...")
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    # Set up world and default prim
    world_prim = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world_prim)

    # Calculate grid layout
    num_objects = len(objects_to_add)
    grid_size = math.ceil(math.sqrt(num_objects))
    spacing = args.grid_spacing

    # Use spacing directly (allows overlapping objects)
    effective_spacing = spacing

    print(f"\nGrid layout: {grid_size}x{grid_size}, spacing: {effective_spacing:.3f}m")

    # Add objects to the scene
    print("\nAdding objects to scene...")
    placed_objects = []

    # Pre-generate unique drop heights if randomizing
    if args.randomize:
        # Generate evenly spaced drop heights so each object drops from a different height
        drop_heights = np.linspace(args.drop_height_min, args.drop_height_max, len(objects_to_add))
        np.random.shuffle(drop_heights)  # Shuffle so it's not predictable by position
        print(f"  Randomize enabled: dropping from heights {args.drop_height_min:.2f}m to {args.drop_height_max:.2f}m")

    total_objects = len(objects_to_add)
    for idx, obj in enumerate(objects_to_add):
        name = obj.get('name', f'object_{idx}')
        usd_path = obj.get('usd_path')
        dims = obj.get('dims', [0.1, 0.1, 0.1])

        # Progress output every 25 objects
        if idx % 25 == 0:
            print(f"  Loading objects: {idx}/{total_objects} ({100*idx//total_objects}%)")

        if not usd_path or not os.path.exists(usd_path):
            print(f"  [SKIP] {name}: USD file not found at {usd_path}")
            continue

        # Calculate grid position (centered around origin)
        row = idx // grid_size
        col = idx % grid_size

        # Center the grid
        offset_x = -(grid_size - 1) * effective_spacing / 2
        offset_y = -(grid_size - 1) * effective_spacing / 2

        x = offset_x + col * effective_spacing
        y = offset_y + row * effective_spacing

        # Add random position jitter if randomizing
        if args.randomize:
            # Random offset up to 50% of grid spacing in each direction
            jitter_range = effective_spacing * 0.5
            x += np.random.uniform(-jitter_range, jitter_range)
            y += np.random.uniform(-jitter_range, jitter_range)

        # Calculate Z position
        if args.randomize:
            # Drop from randomized height
            z = drop_heights[idx]
        else:
            # Place on ground (z=0), lifted by half height so object sits on ground
            z = dims[2] / 2.0

        # Generate random orientation if randomizing
        if args.randomize:
            # Random quaternion (uniform distribution on SO(3))
            u1, u2, u3 = np.random.random(3)
            qw = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
            qx = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
            qy = np.sqrt(u1) * np.sin(2 * np.pi * u3)
            qz = np.sqrt(u1) * np.cos(2 * np.pi * u3)
            orientation = (qw, qx, qy, qz)
        else:
            orientation = None

        # Create unique prim path
        unique_name = name
        counter = 1
        while stage.GetPrimAtPath(f"/World/{unique_name}").IsValid():
            unique_name = f"{name}_{counter}"
            counter += 1

        prim_path = f"/World/{unique_name}"

        try:
            # Create Xform for the object
            xform = UsdGeom.Xform.Define(stage, prim_path)
            prim = xform.GetPrim()

            # Add payload (reference to the actual USD file)
            prim.GetPayloads().AddPayload(usd_path)

            # Set transform
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()

            # Add translate operation
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(x, y, z))

            # Add rotation if randomizing
            if orientation is not None:
                orient_op = xformable.AddOrientOp()
                orient_op.Set(Gf.Quatd(orientation[0], orientation[1], orientation[2], orientation[3]))

            placed_objects.append({
                'name': unique_name,
                'position': (x, y, z),
                'orientation': orientation,
                'dims': dims,
                'dataset': obj.get('dataset', 'unknown')
            })

        except Exception as e:
            print(f"  [ERROR] Failed to add {name}: {e}")
            continue

    print(f"\nSuccessfully placed {len(placed_objects)} objects")

    # Place shelf/bin objects off to the side
    if side_objects:
        print(f"\nPlacing {len(side_objects)} shelf/container objects off to the side...")
        side_offset_y = (grid_size * effective_spacing / 2) + 2.0  # 2m beyond the main grid

        for idx, obj in enumerate(side_objects):
            name = obj.get('name', f'side_object_{idx}')
            usd_path = obj.get('usd_path')
            dims = obj.get('dims', [0.5, 0.5, 0.5])

            if not usd_path or not os.path.exists(usd_path):
                print(f"  [SKIP] {name}: USD file not found at {usd_path}")
                continue

            # Place in a row along Y, offset from main grid
            x = idx * 1.5  # 1.5m spacing between side objects
            y = side_offset_y
            z = dims[2] / 2.0  # Sit on ground

            # Create unique prim path
            unique_name = name
            counter = 1
            while stage.GetPrimAtPath(f"/World/{unique_name}").IsValid():
                unique_name = f"{name}_{counter}"
                counter += 1

            prim_path = f"/World/{unique_name}"

            try:
                xform = UsdGeom.Xform.Define(stage, prim_path)
                prim = xform.GetPrim()
                prim.GetPayloads().AddPayload(usd_path)

                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()

                translate_op = xformable.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(x, y, z))

                placed_objects.append({
                    'name': unique_name,
                    'position': (x, y, z),
                    'orientation': None,
                    'dims': dims,
                    'dataset': obj.get('dataset', 'unknown'),
                    'side_object': True
                })

            except Exception as e:
                print(f"  [ERROR] Failed to add {name}: {e}")
                continue

        print(f"  Placed {len([o for o in placed_objects if o.get('side_object')])} shelf/container objects")

    # Add ground plane
    print("\nAdding ground plane...")
    GroundPlane(prim_path="/World/GroundPlane", z_position=0.0, size=50)

    # Add lighting
    print("Adding lighting...")
    prim_utils.create_prim(
        "/World/DistantLight",
        "DistantLight",
        attributes={
            "inputs:color": (1.0, 1.0, 1.0),
            "inputs:enableColorTemperature": True,
            "inputs:colorTemperature": 6500.0,
            "inputs:intensity": 2.0,
            "inputs:exposure": 10.0,
            "inputs:angle": 30.0,
        }
    )

    prim_utils.create_prim(
        "/World/DomeLight",
        "DomeLight",
        attributes={
            "inputs:intensity": 2.0,
            "inputs:color": (1.0, 1.0, 1.0),
            "inputs:enableColorTemperature": True,
            "inputs:colorTemperature": 6150,
            "inputs:exposure": 8.0,
            "inputs:texture:format": "latlong",
        }
    )

    # Set white background
    import omni.kit.commands
    from pxr import UsdRender
    render_settings = stage.GetPrimAtPath("/Render/RenderProduct/Vars")
    if not render_settings.IsValid():
        # Set the dome light to act as white background by removing texture
        dome_prim = stage.GetPrimAtPath("/World/DomeLight")
        if dome_prim.IsValid():
            # Use solid white color as background
            dome_prim.GetAttribute("inputs:texture:file").Clear()

    # Set the viewport background color to white via carb settings
    import carb.settings
    settings = carb.settings.get_settings()
    settings.set("/rtx/post/backgroundZeroAlpha/backgroundDefaultColor", (1.0, 1.0, 1.0, 1.0))

    # Initialize world
    print("\nInitializing world...")
    world = World(physics_dt=0.0167, rendering_dt=1/60)
    world.reset()

    # Setup camera - position to see all objects from the front
    grid_extent = (grid_size - 1) * effective_spacing / 2
    camera_distance = max(grid_extent * 3, 2.0)  # At least 2m away
    camera_height = max(grid_extent * 1.5, 3.0)  # At least 1m high

    camera_position = (camera_distance, 0.0, camera_height)  # Front view (looking along -X)
    camera_target = (0.0, 0.0, 0.0)  # Look at center, slightly above ground

    print(f"Camera position: {camera_position}")
    print(f"Camera target: {camera_target}")

    resolution = tuple(args.resolution)
    camera = Camera(prim_path="/OmniverseKit_Persp", resolution=resolution, frequency=20)

    set_camera_view(
        eye=np.array(camera_position),
        target=np.array(camera_target),
        camera_prim_path="/OmniverseKit_Persp"
    )
    camera.initialize()

    # Set camera intrinsics for good framing
    camera_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
    camera_prim.GetAttribute("focalLength").Set(24.0)
    camera_prim.GetAttribute("horizontalAperture").Set(20.955)

    # Render the scene
    print(f"\nRendering scene (waiting {args.skip_frames} frames for stabilization)...")

    output_path = output_dir / "scattered_objects.png"

    i = 0
    while app.is_running():
        world.step(render=True)
        if i % 10 == 0:
            print(f"  Rendering frame {i}/{args.skip_frames}...")
        if i == args.skip_frames:
            # Capture the frame
            rgba_image = camera.get_rgba()
            rgb_image = rgba_image[:, :, :3]

            # Save image
            pil_image = Image.fromarray(rgb_image.astype(np.uint8))
            pil_image.save(output_path)
            print(f"\n✓ Image saved to: {output_path}")

            world.stop()
            break
        i += 1

    # Save placement info
    info_path = output_dir / "scattered_objects_info.json"
    placement_info = {
        'datasets': list(target_datasets),
        'num_objects': len(placed_objects),
        'grid_size': grid_size,
        'spacing': effective_spacing,
        'randomize': args.randomize,
        'drop_height_range': [args.drop_height_min, args.drop_height_max] if args.randomize else None,
        'camera_position': camera_position,
        'camera_target': camera_target,
        'resolution': resolution,
        'objects': placed_objects
    }
    with open(info_path, 'w') as f:
        json.dump(placement_info, f, indent=2)
    print(f"✓ Placement info saved to: {info_path}")

    # Cleanup
    omni.usd.get_context().close_stage()
    app.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
