# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
from PIL import Image

from robolab.core.utils.isaaclab_compat import ISAACLAB_USES_XYZW


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
    import omni.usd
    from pxr import Gf, UsdGeom

    if ISAACLAB_USES_XYZW:
        import isaaclab.sim as prim_utils
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaacsim.core.experimental.utils import app as app_utils
        from isaacsim.core.rendering_manager import ViewportManager

        app_utils.enable_extension("isaacsim.sensors.experimental.rtx")
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

        open_stage = prim_utils.open_stage
        world = None

        def set_camera_view(*, eye, target, camera_prim_path):
            ViewportManager.set_camera_view(
                camera_prim_path, eye=list(eye), target=list(target)
            )

    else:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects.ground_plane import GroundPlane
        from isaacsim.core.utils import prims as prim_utils
        from isaacsim.core.utils.stage import open_stage
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera

    _ = open_stage(str(usd_path))
    if ISAACLAB_USES_XYZW:
        world = SimulationContext(SimulationCfg(dt=0.0167, render_interval=1))
        stage = world.stage
    else:
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

    if ISAACLAB_USES_XYZW:
        rtx_camera = RtxCamera("/OmniverseKit_Persp", tick_rate=20.0)
        camera = CameraSensor(
            rtx_camera,
            resolution=(resolution[1], resolution[0]),
            annotators=["rgb"],
        )
    else:
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
    if not ISAACLAB_USES_XYZW:
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
        if ISAACLAB_USES_XYZW:
            ground_cfg = prim_utils.GroundPlaneCfg()
            ground_cfg.func(
                "/World/GroundPlane", ground_cfg, translation=(0.0, 0.0, ground_position)
            )
        else:
            GroundPlane(prim_path="/World/GroundPlane", z_position=ground_position)

    # Strip the extension from the USD path and keep only the filename
    usd_filename = os.path.splitext(os.path.basename(usd_path))[0]

    # Render frames until we reach the desired frame
    i = 0
    while app.is_running():
        world.step(render=True)
        if i >= skip_frames:
            if ISAACLAB_USES_XYZW:
                rgb_data, _ = camera.get_data("rgb")
                if rgb_data is None:
                    i += 1
                    continue
                rgb_image = rgb_data.numpy()
            else:
                # Convert the legacy RGBA image to RGB.
                rgb_image = camera.get_rgba()[:, :, :3]

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

    if ISAACLAB_USES_XYZW:
        SimulationContext.clear_instance()
    else:
        omni.usd.get_context().close_stage()
    return output_path
