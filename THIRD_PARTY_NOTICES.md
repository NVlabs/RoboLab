# Third-Party Notices

RoboLab is distributed under the [Apache License 2.0](./LICENSE).

It depends on the following third-party open-source packages, each governed by its own license. The list reflects packages declared in `pyproject.toml`; transitive dependencies fetched by `uv sync` carry their own upstream licenses, which take precedence over anything stated here.

## Direct dependencies

| Package | Version | License | License URL |
|---|---|---|---|
| isaaclab | 2.2.0 / 2.3.2.post1 | BSD-3-Clause | https://github.com/isaac-sim/IsaacLab/blob/main/LICENSE |
| isaacsim | 5.0.0.0 / 5.1.0.0 | Apache-2.0 | https://github.com/isaac-sim/IsaacSim/blob/main/LICENSE.md |
| torch | 2.7.0+cu128 | BSD-3-Clause | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| numpy | 1.26.0 | BSD-3-Clause | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| gymnasium | 1.2.0 | MIT | https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE |
| openai | 2.28.0 | Apache-2.0 | https://github.com/openai/openai-python/blob/main/LICENSE |
| requests | 2.32.3 | Apache-2.0 | https://github.com/psf/requests/blob/main/LICENSE |
| json-numpy | 2.1.1 | MIT | https://github.com/Crimson-Crow/json-numpy/blob/main/LICENSE.txt |
| pyzmq | 27.1.0 | BSD-3-Clause | https://github.com/zeromq/pyzmq/blob/main/LICENSE.md |
| msgpack | 1.1.2 | Apache-2.0 | https://github.com/msgpack/msgpack-python/blob/main/COPYING |
| msgpack-numpy | 0.4.8 | BSD-3-Clause | https://github.com/lebedov/msgpack-numpy/blob/master/LICENSE.md |
| websockets | 12.0 / 17.1 | BSD-3-Clause | https://github.com/python-websockets/websockets/blob/main/LICENSE |
| opencv-python | 4.11.0.86 | MIT | https://github.com/opencv/opencv-python/blob/4.x/LICENSE.txt |
| pillow | 11.2.1 | HPND (MIT-CMU) | https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| imageio | 2.37.0 | BSD-2-Clause | https://github.com/imageio/imageio/blob/master/LICENSE |
| scipy | 1.15.3 | BSD-3-Clause | https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| matplotlib | 3.10.0 | PSF-based (Matplotlib License) | https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE |
| h5py | 3.16.0 | BSD-3-Clause | https://github.com/h5py/h5py/blob/master/LICENSE |
| PyYAML | 6.0.2 | MIT | https://github.com/yaml/pyyaml/blob/main/LICENSE |
| pandas | 3.0.2 | BSD-3-Clause | https://github.com/pandas-dev/pandas/blob/main/LICENSE |
| tqdm | 4.67.3 | MIT AND MPL-2.0 | https://github.com/tqdm/tqdm/blob/master/LICENCE |
| tyro | 0.9.17 | MIT | https://github.com/brentyi/tyro/blob/main/LICENSE |
| python-dotenv | 1.2.2 | BSD-3-Clause | https://github.com/theskumar/python-dotenv/blob/main/LICENSE |
| setuptools | 80.10.2 | MIT | https://github.com/pypa/setuptools/blob/main/LICENSE |
| fastapi | 0.115.7 | MIT | https://github.com/fastapi/fastapi/blob/master/LICENSE |
| uvicorn | 0.29.0 | BSD-3-Clause | https://github.com/encode/uvicorn/blob/master/LICENSE.md |
| jinja2 | 3.1.6 | BSD-3-Clause | https://github.com/pallets/jinja/blob/main/LICENSE.txt |

## Optional dependencies

Installed only when the corresponding extra is requested (`uv sync --extra analysis` or `--extra all`).

| Package | Version | License | License URL |
|---|---|---|---|
| sbi | 0.26.1 | Apache-2.0 | https://github.com/sbi-dev/sbi/blob/main/LICENSE.txt |

## System dependencies

The following are not Python packages and must be installed via the host OS package manager (see [README](./README.md#installation)):

| Component | Source | Notes |
|---|---|---|
| ffmpeg | distro package (`apt install ffmpeg`, etc.) | Used for video recording. License terms are governed by the distribution's ffmpeg build. |

## Bundled assets

RoboLab bundles 3D objects, backgrounds, materials, fixtures, scenes, and robots under `assets/`. Where a folder ships its own `LICENSE` file, that local license governs and takes precedence over the entry below.

### Object datasets (`assets/objects/`)

| Folder | License | Source / Provenance |
|---|---|---|
| `handal` | CC BY-NC-SA 4.0 | Derivative of the [HANDAL](https://github.com/NVlabs/HANDAL) dataset, © 2023 NVIDIA Corporation. See [`assets/objects/handal/LICENSE`](./assets/objects/handal/LICENSE). |
| `hope` | CC BY-NC-SA 4.0 | Derivative of the [HOPE](https://github.com/swtyree/hope-dataset) dataset, © 2021 NVIDIA Corporation. See [`assets/objects/hope/LICENSE`](./assets/objects/hope/LICENSE). |
| `hot3d` | CC BY-SA 4.0, with HOT3D Dataset License Agreement non-sale restriction | Derivative of the [HOT3D](https://www.projectaria.com/datasets/hot3d/license/) Model Data, © Meta Platforms Technologies, LLC. See [`assets/objects/hot3d/LICENSE`](./assets/objects/hot3d/LICENSE). |
| `ycb` | MIT | Derivative of the [YCB-Video](https://github.com/yuxng/YCB_Video_toolbox) dataset, © 2017 Yu Xiang. See [`assets/objects/ycb/LICENSE`](./assets/objects/ycb/LICENSE). |
| `basic` | CC BY-NC-SA 4.0 | © 2026 NVIDIA Corporation. |
| `fruits_veggies` | CC BY-NC-SA 4.0 | © 2026 NVIDIA Corporation. |
| `objaverse` | CC BY-NC-SA 4.0 | © 2026 NVIDIA Corporation. |
| `vomp` | CC BY 4.0 | © 2026 NVIDIA Corporation. |

### Backgrounds, materials, fixtures, scenes, robots

| Folder | License | Source / Provenance |
|---|---|---|
| `assets/backgrounds` | CC0 1.0 | HDRI backgrounds sourced from [Poly Haven](https://polyhaven.com), dedicated to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). |
| `assets/fixtures` | CC BY-NC-SA 4.0 | © 2026 NVIDIA Corporation. |
| `assets/materials` | CC BY-NC-SA 4.0 | © 2026 NVIDIA Corporation. |
| `assets/scenes` | CC BY-NC-SA 4.0 | © 2026 NVIDIA Corporation. |
| `assets/robots` (except locally licensed subfolders) | CC BY-NC-SA 4.0 | © 2026 NVIDIA Corporation. |
| `assets/robots/kinova_gen3_robotiq_2f85` | BSD-3-Clause | Derived from [Kinova `ros2_kortex`](https://github.com/Kinovarobotics/ros2_kortex) and [PickNik `ros2_robotiq_gripper`](https://github.com/PickNikRobotics/ros2_robotiq_gripper). See the two license files in the asset directory. |

## Bundled source code

| Path | License | Source / Provenance |
|---|---|---|
| `policies/ai_native/orchestrated/` (files marked `SPDX-License-Identifier: MIT` and the agent prompts listed in its README) | MIT | Adapted from [GPT-as-Policy](https://github.com/anonymous-report-421/GPT-as-Policy), © 2026 Yu-Mool Shu and Lipxin Zheng. See [`policies/ai_native/orchestrated/LICENSE-MIT`](./policies/ai_native/orchestrated/LICENSE-MIT). |
