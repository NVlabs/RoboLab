# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render randomized initial poses, one image per env, and a GIF per scene.

Runs no policy: it resets each env with the initial-pose randomization event
(the same one ``--init-pose-noise`` adds to the eval runners), steps physics
briefly, and renders. With ``--view top-down`` (default) the camera looks
straight down and each object's footprint is drawn on top: green = moved group,
red = fixed, yellow = workspace. With ``--view viewport`` it renders the eval
runners' viewport camera with no footprints. The first frame of each GIF is the
default layout; each frame's caption gives the largest shift and turn allowed
(reduced on crowded episodes). GIFs are encoded with ffmpeg when it is installed
(much better colors), otherwise with Pillow.

Tasks that share a scene USD are rendered once (the first task found).

Usage:
    python examples/check_init_pose.py --headless --task-dirs benchmark --init-pose-noise 0.1
    python examples/check_init_pose.py --headless --task BananaInBowlTask --init-pose-noise 0.2 90 --num-envs 8
    python examples/check_init_pose.py --headless --task BananaInBowlTask --init-pose-noise 0.1 --view viewport
    # Split the scenes across 3 processes:
    python examples/check_init_pose.py --headless --init-pose-noise 0.1 --shard 0 3

Output (default output/init_pose_check/<label>/, or <label>_viewport/ with --view viewport):
    <task>/env<i>_run<r>.png, <task>/default.png, <task>.gif
    <task>/init_pose_groups.json   groups and settings
    <task>/episode_motions.json    per episode: each group's sampled center (m) and yaw (rad)
    summary_shard<k>.json          per-task timing, status and errors
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import cv2  # noqa: F401  Must be imported before isaaclab.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", nargs="+", default=None, help="Tasks to check (default: every task in --task-dirs).")
parser.add_argument("--task-dirs", "--task_dirs", nargs="+", default=["benchmark"],
                    help="Task folders under robolab/tasks/ (default: benchmark).")
parser.add_argument("--num-envs", "--num_envs", type=int, default=16, help="Parallel envs, one layout each.")
parser.add_argument("--num-runs", "--num_runs", type=int, default=1, help="Resets per env (run indices 0..N-1).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--init-pose-noise", type=float, nargs="+", required=True, metavar="X",
                    help="X [YAW_DEG], as for the eval runners.")
parser.add_argument("--init-pose-objects", nargs="+", default=None, metavar="NAME")
parser.add_argument("--init-pose-workspace", type=float, nargs=4, default=None,
                    metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"))
parser.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("INDEX", "COUNT"),
                    help="Process only scenes INDEX, INDEX+COUNT, ... of the sorted unique scenes.")
parser.add_argument("--output-dir", default=None, help="Default: output/init_pose_check/<settings label>.")
parser.add_argument("--gif-frame-ms", type=int, default=300)
parser.add_argument("--gif-width", type=int, default=600, help="GIF width in pixels; the PNGs keep full size.")
parser.add_argument("--view", choices=["top-down", "viewport"], default="top-down",
                    help="top-down: camera straight down with footprints drawn on top; viewport: the eval "
                         "runners' viewport camera, no footprints.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PIL import Image  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.constants import PACKAGE_DIR, set_output_dir  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.events.init_pose_layout import default_yaw_range_deg, transform_group  # noqa: E402
from robolab.core.events.reset_pose import INIT_POSE_EVENT_NAME  # noqa: E402
from robolab.eval.runner import init_pose_event_from_args, init_pose_label  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.variations.camera import EgocentricMirroredCameraCfg  # noqa: E402

robolab.constants.RECORD_IMAGE_DATA = False

# Top-down camera over the table center, looking straight down; image up = +x (away
# from the robot), image right = -y. Env-local meters.
CAMERA_NAME = "init_pose_top_down_camera"
VIEWPORT_CAMERA_NAME = "egocentric_mirrored_camera"  # attribute of EgocentricMirroredCameraCfg
CAMERA_XY = (0.55, 0.0)
CAMERA_HEIGHT_M = 1.5
IMAGE_WIDTH, IMAGE_HEIGHT = 800, 560
FOCAL_LENGTH, HORIZONTAL_APERTURE = 24.0, 17.6
PIXELS_PER_UNIT_DEPTH = FOCAL_LENGTH / HORIZONTAL_APERTURE * IMAGE_WIDTH  # pixels per meter at 1 m depth
# Physics steps after each reset (1/120 s each): lets the renderer pick up the teleported poses
# and shows whether any moved object gets pushed (e.g. by an undetected overlap).
SETTLE_STEPS = 10
DRIFT_WARNING_M = 0.01


@configclass
class TopDownCameraCfg:
    init_pose_top_down_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/init_pose_top_down_camera",
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=FOCAL_LENGTH,
            focus_distance=CAMERA_HEIGHT_M,
            horizontal_aperture=HORIZONTAL_APERTURE,
            vertical_aperture=HORIZONTAL_APERTURE * IMAGE_HEIGHT / IMAGE_WIDTH,
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(*CAMERA_XY, CAMERA_HEIGHT_M), rot=(0.7071068, 0.0, 0.0, -0.7071068),
                                        convention="opengl"),
    )


def to_pixels(points_xy: np.ndarray, table_top_z: float) -> np.ndarray:
    """Project env-local points on the table top to image pixels."""
    scale = PIXELS_PER_UNIT_DEPTH / (CAMERA_HEIGHT_M - table_top_z)
    u = IMAGE_WIDTH / 2 - (points_xy[..., 1] - CAMERA_XY[1]) * scale
    v = IMAGE_HEIGHT / 2 - (points_xy[..., 0] - CAMERA_XY[0]) * scale
    return np.stack([u, v], axis=-1).round().astype(np.int32)


def draw_overlay(image: np.ndarray, term, motions) -> np.ndarray:
    """In the top-down view, draw the workspace and every footprint (green moved, red fixed); else return as is."""
    if args.view != "top-down":
        return image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    draw_footprints(image, term, motions)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_footprints(image: np.ndarray, term, motions) -> None:
    """Draw the workspace (yellow) and every footprint (green moved, red fixed) on a BGR top-down image."""
    ws = term.workspace
    corners = np.array([[ws.x_min, ws.y_min], [ws.x_min, ws.y_max], [ws.x_max, ws.y_max], [ws.x_max, ws.y_min]])
    cv2.polylines(image, [to_pixels(corners, term.table_top_z)], True, (0, 220, 255), 2)
    by_name = {fp.name: fp for fp in term.footprints}
    for group, motion in zip(term.groups, motions):
        color = (60, 200, 60) if group.moves else (60, 60, 230)
        for polygon in transform_group(group, [by_name[name] for name in group.members], motion):
            cv2.polylines(image, [to_pixels(polygon, term.table_top_z)], True, color, 2)


def gif_frame(image: np.ndarray, caption: str) -> np.ndarray:
    """Resize an RGB frame to --gif-width and draw the caption in a bar along the top."""
    height = round(image.shape[0] * args.gif_width / image.shape[1])
    frame = cv2.resize(image, (args.gif_width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (0, 0), (args.gif_width, 22), (0, 0, 0), -1)
    cv2.putText(frame, caption, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def write_gif(frames: list[np.ndarray], path: str) -> None:
    """Write RGB frames as a looping GIF, with a palette chosen per frame by ffmpeg if available."""
    if shutil.which("ffmpeg") is None:
        Image.fromarray(frames[0]).save(path, save_all=True, append_images=[Image.fromarray(f) for f in frames[1:]],
                                        duration=args.gif_frame_ms, loop=0)
        return
    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(tmp, f"{i:04d}.png"))
        palette = "split[a][b];[a]palettegen=stats_mode=single[p];[b][p]paletteuse=new=1:dither=sierra2_4a"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(1000 / args.gif_frame_ms),
                        "-i", os.path.join(tmp, "%04d.png"), "-vf", palette, path], check=True)


def settle_and_measure_drift(env, term) -> dict[str, list[float]]:
    """Step physics briefly so the renderer sees the new poses; return each moved object's xy drift (m) per env."""
    for _ in range(SETTLE_STEPS):
        env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt * SETTLE_STEPS)
    origins = env.scene.env_origins.cpu().numpy()
    drift = {}
    for name, written in term.last_written_positions.items():
        actual = env.scene.rigid_objects[name].data.root_pos_w.cpu().numpy() - origins
        drift[name] = np.linalg.norm(actual[:, :2] - written[:, :2], axis=-1).round(4).tolist()
    return drift


def capture(env) -> np.ndarray:
    """Render and return the camera's rgb images, (num_envs, H, W, 3) uint8."""
    for _ in range(4):
        env.sim.render()
    camera = env.scene.sensors[CAMERA_NAME if args.view == "top-down" else VIEWPORT_CAMERA_NAME]
    camera.update(dt=0.0, force_recompute=True)
    return camera.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8)


def unique_scene_envs(env_names: list[str]) -> list[tuple[str, str, list[str]]]:
    """(scene usd, first env, all envs) per scene, sorted by scene path."""
    by_scene: dict[str, list[str]] = {}
    for name in env_names:
        usd_path = parse_env_cfg(name, num_envs=1).scene.scene.spawn.usd_path
        by_scene.setdefault(usd_path, []).append(name)
    return [(usd, names[0], names) for usd, names in sorted(by_scene.items())]


def check_scene(task_env: str, task_dir: str) -> dict:
    """Render the default layout and every env x run layout of one task; return timing."""
    set_output_dir(task_dir)
    start = time.perf_counter()
    env, _ = create_env(task_env, device=args.device, seed=args.seed, num_envs=args.num_envs,
                        events=init_pose_event_from_args(args))
    created_s = time.perf_counter() - start
    try:
        term_cfg = env.event_manager.get_term_cfg(INIT_POSE_EVENT_NAME)
        term = term_cfg.func
        noise_values = dict(noise_m=term_cfg.params["noise_m"], yaw_range_deg=term_cfg.params["yaw_range_deg"])
        noise_m = noise_values["noise_m"]
        yaw_deg = noise_values["yaw_range_deg"]
        yaw_deg = default_yaw_range_deg(noise_m) if yaw_deg is None else yaw_deg
        term.reset_seconds.clear()  # drop any reset done while creating the env

        frames = []
        drift_by_object: dict[str, float] = {}
        drift_by_episode: dict[int, float] = {}
        episode_motions = {}
        reduced_noise = {}
        for run in range(args.num_runs):
            env.run_index = run
            env.reset()
            for name, per_env in settle_and_measure_drift(env, term).items():
                drift_by_object[name] = max(drift_by_object.get(name, 0.0), max(per_env))
                for env_id, value in enumerate(per_env):
                    episode = run * env.num_envs + env_id
                    drift_by_episode[episode] = max(drift_by_episode.get(episode, 0.0), value)
            images = capture(env)
            for env_id in range(env.num_envs):
                episode = run * env.num_envs + env_id
                scale = term.last_noise_scales[env_id]
                caption = f"{task_env} | shift <= {scale * noise_m:.2f} m, turn <= {scale * yaw_deg:.0f} deg"
                if scale < 1.0:
                    reduced_noise[episode] = round(scale, 3)
                    caption += " (crowded)"
                frame = draw_overlay(images[env_id], term, term.last_motions[env_id])
                Image.fromarray(frame).save(os.path.join(task_dir, f"env{env_id}_run{run}.png"))
                frames.append(gif_frame(frame, caption))
                episode_motions[episode] = [
                    None if m is None else {"center_xy": [round(float(v), 6) for v in m.center_xy],
                                            "yaw_rad": round(m.yaw_rad, 6)}
                    for m in term.last_motions[env_id]]
        with open(os.path.join(task_dir, "episode_motions.json"), "w") as f:
            json.dump(episode_motions, f, indent=1)
        reset_seconds = list(term.reset_seconds)

        # Default layout: zero noise keeps every group where it is.
        term_cfg.params.update(noise_m=0.0, yaw_range_deg=0.0)
        env.reset()
        settle_and_measure_drift(env, term)
        default_frame = draw_overlay(capture(env)[0], term, term.last_motions[0])
        Image.fromarray(default_frame).save(os.path.join(task_dir, "default.png"))
        frames.insert(0, gif_frame(default_frame, f"{task_env} | default layout"))
        gif_path = task_dir + ".gif"
        write_gif(frames, gif_path)
        return {
            "status": "ok",
            "env_create_seconds": round(created_s, 3),
            "setup_seconds": round(term.setup_seconds, 4),
            "reset_seconds_per_reset": [round(s, 4) for s in reset_seconds],
            "max_drift_m_by_object": drift_by_object,
            "episodes_with_drift": {e: d for e, d in drift_by_episode.items() if d > DRIFT_WARNING_M},
            "drift_warning": max(drift_by_object.values(), default=0.0) > DRIFT_WARNING_M,
            "episodes_with_reduced_noise": reduced_noise,
            "groups": term.report(**noise_values)["groups"],
            "gif": gif_path,
        }
    finally:
        env.close()


def main() -> int:
    viewport_camera = TopDownCameraCfg if args.view == "top-down" else EgocentricMirroredCameraCfg
    auto_register_droid_envs(task_dirs=args.task_dirs, task=args.task, cameras=[], viewport_camera=viewport_camera)
    env_names = get_envs(task=args.task) if args.task else get_envs()
    scenes = unique_scene_envs(env_names)
    index, count = args.shard
    scenes = scenes[index::count]
    label = init_pose_label(args) + ("_viewport" if args.view == "viewport" else "")
    output_dir = args.output_dir or os.path.join(PACKAGE_DIR, "output", "init_pose_check", label)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[check_init_pose] {len(scenes)} scenes in shard {index}/{count} -> {output_dir}", flush=True)

    summary = {"settings": {"label": init_pose_label(args), "view": args.view, "num_envs": args.num_envs,
                            "num_runs": args.num_runs, "seed": args.seed}, "tasks": {}}
    summary_path = os.path.join(output_dir, f"summary_shard{index}.json")
    for scene_usd, task_env, sharing in scenes:
        task_dir = os.path.join(output_dir, task_env)
        os.makedirs(task_dir, exist_ok=True)
        print(f"[check_init_pose] {task_env} ({os.path.basename(scene_usd)})", flush=True)
        try:
            result = check_scene(task_env, task_dir)
        except Exception as exc:  # report and continue with the next scene
            traceback.print_exc()
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        result.update(scene=os.path.basename(scene_usd), tasks_sharing_scene=sharing)
        summary["tasks"][task_env] = result
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[check_init_pose] {task_env}: {result['status']}", flush=True)

    failed = [name for name, r in summary["tasks"].items() if r["status"] != "ok"]
    print(f"[check_init_pose] done: {len(scenes) - len(failed)} ok, {len(failed)} failed {failed} -> {summary_path}",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        # Omniverse teardown can hang after all results are written; exit directly.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
