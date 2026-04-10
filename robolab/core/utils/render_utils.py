# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os

import numpy as np
from PIL import Image


def render_stage_frame(app,
                        usd_path,
                        output_dir=None,
                        skip_frames=100,
                        resolution=(640, 480),
                        add_lighting=True,
                        add_ground=False,
                        camera_position=None,
                        camera_target=None,
                        focal_length=None,
                        horizontal_aperture=None,
                        ground_position=-0.2
                ):
    """Render a frame from a USD stage file.

    Args:
        app: Isaac Sim SimulationApp instance
        usd_path: Path to the USD file to render
        output_dir: Directory to save the rendered image (optional)
        skip_frames: Number of frames to skip before rendering for scene stabilization
        resolution: Tuple of (width, height) for the rendered image
        add_lighting: Whether to add default scene lighting
        add_ground: Whether to add a ground plane
        camera_position: Camera position as (x, y, z). If None, uses (2.5, 0, 1.5)
        camera_target: Camera target as (x, y, z). If None, uses (0.5, 0.0, 0.0)
        focal_length: Camera focal length in cm. If None, uses 24.0 cm
        horizontal_aperture: Camera horizontal aperture in cm. If None, uses 20.955 cm
        ground_position: Z-position of the ground plane

    Returns:
        str: Path to the saved image file

    Note:
        When camera parameters are None, default values are used to ensure
        consistent framing across different USD files, regardless of their
        embedded camera settings. This prevents the "far away" rendering issue
        caused by inconsistent camera intrinsics in USD files.
    """
    import omni.isaac.core.utils.prims as prim_utils
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects.ground_plane import GroundPlane
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.sensors.camera import Camera
    from omni.isaac.core.utils.viewports import set_camera_view
    from pxr import Gf, UsdGeom

    _ = open_stage(str(usd_path))
    stage = omni.usd.get_context().get_stage()
    world = World(physics_dt=0.0167, rendering_dt=1/60)
    world.reset()

    # prim = stage.GetDefaultPrim()
    # xform = UsdGeom.Xformable(prim)

    # # Remove existing transform ops for a clean set
    # xform.ClearXformOpOrder()

    # # Add a new orient op for 90° about X
    # orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    # orient_op.Set(Gf.Quatf(0.7071068, Gf.Vec3f(1, 0, 0)))

    camera = Camera(prim_path="/OmniverseKit_Persp", resolution=resolution, frequency=20)

    # Set default camera position if not provided to ensure consistent framing
    if camera_position is None or camera_target is None:
        # Default to a reasonable view position
        camera_position = (2.5, 0, 1.5)
        camera_target = (0.5, 0.0, 0.0)

    set_camera_view(
        eye=np.array(camera_position),
        target=np.array(camera_target),
        camera_prim_path="/OmniverseKit_Persp"
    )
    camera.initialize()

    # Set consistent camera intrinsics to ensure consistent framing regardless of USD file settings
    if focal_length is None:
        focal_length = 24.0  # Default from Isaac Sim PinholeCameraCfg
    if horizontal_aperture is None:
        horizontal_aperture = 20.955  # Default from Isaac Sim PinholeCameraCfg

    camera_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
    camera_prim.GetAttribute("focalLength").Set(focal_length)
    camera_prim.GetAttribute("horizontalAperture").Set(horizontal_aperture)

    if add_lighting:
        light = prim_utils.create_prim(
            "/World/distant_light",
            "DistantLight",
            attributes={
                "inputs:color": (1.0, 1.0, 1.0),
                "inputs:enableColorTemperature": True,
                "inputs:colorTemperature": 7250.0,
                "inputs:intensity": 1.0,
                "inputs:exposure": 10.0,
                "inputs:angle": 30,
                }
            )

        light = prim_utils.create_prim(
            "/World/dome_light",
            "DomeLight",
            attributes={
                "inputs:intensity": 1.0,
                "inputs:color": (1.0, 1.0, 1.0),
                "inputs:enableColorTemperature": True,
                "inputs:colorTemperature": 6150,
                "inputs:exposure": 9.0,
                "inputs:texture:format": "latlong",
                }
            )

    if add_ground:
        GroundPlane(prim_path="/World/GroundPlane", z_position=ground_position)

    # Strip the extension from the USD path and keep only the filename
    usd_filename = os.path.splitext(os.path.basename(usd_path))[0]

    # Render frames until we reach the desired frame
    i = 0
    while app.is_running():
        world.step(render=True)
        if i == skip_frames:
            # Get the RGBA image from camera
            rgba_image = camera.get_rgba()
            # Convert to RGB (remove alpha channel)
            rgb_image = rgba_image[:, :, :3]

            # Save the image to PNG file
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{usd_filename}.png")

            # Convert numpy array to PIL Image and save
            pil_image = Image.fromarray(rgb_image.astype(np.uint8))
            pil_image.save(output_path)
            print(f"Image saved to: {output_path}")
            world.stop()
            break
        i += 1

    omni.usd.get_context().close_stage()
    return output_path
