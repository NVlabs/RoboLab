# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os

import robolab.constants
from robolab.constants import ASSET_DIR, DEFAULT_TASK_SUBFOLDERS, TASK_DIR

CAMERAS_USDA_PATH = os.path.join(ASSET_DIR, "cameras.usda")


def load_cameras_from_usda(usda_path, width=1280, height=720):
    """Load camera configs from a .usda file for use with TiledCameraCfg.

    Creates individual CameraCfg entries to spawn camera prims, plus a single
    TiledCameraCfg that captures all cameras efficiently in one batched render.

    TiledCamera requires uniform resolution, so width/height are set once for
    all cameras.

    Args:
        usda_path: Path to the .usda file containing camera definitions.
        width: Render width for all cameras.
        height: Render height for all cameras.

    Returns:
        Tuple of (camera_cfgs_list, num_cameras) where camera_cfgs_list includes
        both individual spawner configclasses and a TiledCameraCfg configclass.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg, TiledCameraCfg
    from isaaclab.utils import configclass
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usda_path)
    if stage is None:
        raise FileNotFoundError(f"Could not open USD stage: {usda_path}")

    spawn_cfgs = []
    camera_names = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Camera):
            continue

        cam = UsdGeom.Camera(prim)
        name = prim.GetName()
        camera_names.append(name)

        focal_length = cam.GetFocalLengthAttr().Get()
        focus_distance = cam.GetFocusDistanceAttr().Get()
        h_aperture = cam.GetHorizontalApertureAttr().Get()
        v_aperture = cam.GetVerticalApertureAttr().Get()

        pos = (0.0, 0.0, 0.0)
        rot = (1.0, 0.0, 0.0, 0.0)
        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t = op.Get()
                pos = (float(t[0]), float(t[1]), float(t[2]))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                q = op.Get()
                rot = (float(q.GetReal()), *[float(v) for v in q.GetImaginary()])

        @configclass
        class _CamSpawnCfg:
            pass

        _CamSpawnCfg.__name__ = f"{name}SpawnCfg"
        _CamSpawnCfg.__qualname__ = f"{name}SpawnCfg"

        setattr(_CamSpawnCfg, name, CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=focus_distance,
                horizontal_aperture=h_aperture,
                vertical_aperture=v_aperture,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=pos, rot=rot, convention="opengl"
            ),
        ))

        spawn_cfgs.append(_CamSpawnCfg)
        print(f"  Loaded camera: {name} (focal_length={focal_length}, pos={pos})")

    num_cameras = len(camera_names)
    if num_cameras == 0:
        raise ValueError(f"No cameras found in {usda_path}")

    # TiledCameraCfg matches ALL spawned camera prims via regex
    names_regex = "|".join(camera_names)
    tiled_prim_path = f"{{ENV_REGEX_NS}}/({names_regex})"

    @configclass
    class _TiledCameraCfg:
        tiled_camera = TiledCameraCfg(
            prim_path=tiled_prim_path,
            spawn=None,
            width=width,
            height=height,
            data_types=["rgb", "depth"],
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="ros"
            ),
        )

    all_cfgs = spawn_cfgs + [_TiledCameraCfg]

    print(f"[RoboLab] Loaded {num_cameras} cameras from {usda_path}")
    print(f"[RoboLab] TiledCamera prim_path: {tiled_prim_path}")
    print(f"[RoboLab] TiledCamera resolution: {width}x{height}")
    return all_cfgs, num_cameras


"""
Scene registration:

For the same task, we can register multiple variants. For example, the following script will register something like this:

Task Name                | Environment                       | Config Class                            | Reg | Tags
---------------------------------------------------------------------------------------------------------------------------------------------------
BagelOnPlateTableTask    | BagelOnPlateTableTaskHomeOffice   | BagelOnPlateTableTaskHomeOfficeEnvCfg   | ✓   | all, pick_place
BagelOnPlateTableTask    | BagelOnPlateTableTaskBilliardHall | BagelOnPlateTableTaskBilliardHallEnvCfg | ✓   | all, pick_place
BananaInBowlTableTask    | BananaInBowlTableTaskHomeOffice   | BananaInBowlTableTaskHomeOfficeEnvCfg   | ✓   | all, pick_place
BananaInBowlTableTask    | BananaInBowlTableTaskBilliardHall | BananaInBowlTableTaskBilliardHallEnvCfg | ✓   | all, pick_place

The columns are:
- Task Name: The base task class name (groups variants together)
- Environment: The registered environment name (also the Gymnasium ID)
- Config Class: The generated configuration class name
- Reg: Registration status (✓ = registered with Gymnasium)
- Tags: Tag names this environment belongs to

"""
def auto_register_droid_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, lighting_intensity=None, task=None):
    """Automatically discover and register tasks.

    Uses TiledCameraCfg for efficient batched camera rendering.
    Cameras are loaded from cameras.usda.

    Access the tiled camera output in your policy via:
        tiled_camera = env.scene["tiled_camera"]
        rgb = tiled_camera.data.output["rgb"]           # (num_envs * num_cameras, H, W, C)
        rgb_per_env = rgb.view(num_envs, num_cameras, H, W, C)

    Args:
        task_dirs: Subdirectories to search for tasks.
        lighting_intensity: Optional lighting intensity override.
        task: If provided, only register the specified task(s) instead of discovering
              all tasks. Accepts a single task name/filename/path (str) or a list of them.
              Significantly faster when running a subset of tasks.
    """
    from robolab.core.environments.factory import auto_discover_and_create_cfgs, create_env_cfg
    from robolab.core.observations.observation_utils import generate_obs_cfg

    # from robolab.registrations.droid_jointpos.observations import ImageObsCfg
    from robolab.robots.droid import (
        DroidCfg,
        DroidJointPositionActionCfg,
        ProprioceptionObservationCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg

    # from robolab.core.observations.observation_utils import generate_image_obs_from_cameras
    # from robolab.variations.camera import OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    camera_cfgs, num_cameras = load_cameras_from_usda(CAMERAS_USDA_PATH)

    # ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])

    # Image observations bypass the obs manager — access tiled_camera directly in policy.
    ObservationCfg = generate_obs_cfg({
        # "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
        # "viewport_cam": ViewportCameraCfg(),
    })

    shared_kwargs = dict(
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidJointPositionActionCfg(),
        robot_cfg=DroidCfg,
        camera_cfg=camera_cfgs,
        # camera_cfg=[OverShoulderLeftCameraCfg, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )

    if task is not None:
        tasks = task if isinstance(task, list) else [task]
        print(f"\033[96m[RoboLab] Registering {len(tasks)} task(s): {tasks}\033[0m")
        for t in tasks:
            create_env_cfg(
                t,
                task_dir=TASK_DIR,
                env_prefix="",
                env_postfix="",
                **shared_kwargs,
            )
    else:
        print(f"\033[96m[RoboLab] Registering all tasks in {task_dirs}\033[0m")
        for subdir in task_dirs:
            auto_discover_and_create_cfgs(
                task_dir=TASK_DIR,
                task_subdirs=[subdir],
                pattern="*.py",
                env_prefix="",
                env_postfix="",
                **shared_kwargs,
            )

    if robolab.constants.VERBOSE:
        from robolab.core.environments.factory import print_env_table
        print_env_table()
