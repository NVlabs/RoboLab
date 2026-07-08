# Contributions

## Acknowledgements

RoboLab is a research project developed and maintained by the [Seattle Robotics Lab (SRL)](https://research.nvidia.com/labs/srl/) at NVIDIA Research, with contributions from across NVIDIA.

NVIDIA contributors: Xuning Yang, Rishit Dagli, Jonathan Tremblay, Yu-Wei Chao, Alperen Degirmenci, Liang Hao, Fabio Ramos, Alex Zook, Siyi Chen, Renato Gasoto.

We thank the following additional contributors for their help in building RoboLab: Arhan Jain, Karl Pertsch.


## How to contribute
If you encounter issues or have suggestions, please open an [Issue](https://github.com/NVlabs/RoboLab/issues) in this repository.

### PRs
1. Fork the repository and create your branch from main.
2. Make your changes and ensure tests pass.
3. Describe in detail what your feature does, or if it fixes issues.
4. Rebase to the latest main.
5. Sign off all your commits (see DCO section below).
6. Submit a pull request.

### Assets: Objects, Scenes, Tasks, Robots, etc
We welcome new asset contributions via PRs. For each asset type, you must make sure that they follow these instructions:

- **Objects** — see [`docs/objects.md`](docs/objects.md)
  - Self-contained `.usd`/`.usda` files with physics, placed under `assets/objects/<your_dataset_name>/`.
  - Appropriate license is placed in each folder, and added to `THIRD_PARTY_NOTICES.md`.
- **Scenes** — see [`docs/scene.md`](docs/scene.md)
  - USD layouts of objects + fixtures (no robots/lighting/backgrounds).
  - All object references must be relative (no global paths) and self-contained within your PR.
  - Scenes must be settled and accompanied by appropriate updates to scene metadata, `_images`, etc.
- **Tasks** — see [`docs/task_libraries.md`](docs/task_libraries.md)
  - Self-contained `.py`, placed under `robolab/tasks/<your_benchmark_name>/`.
- **Robots** — see [`docs/robots.md`](docs/robots.md)
  - A `@configclass` with a `robot` field (`ArticulationCfg`), and any and all appropriate action/observation configs.
- **Variations** (camera, lighting, background) — see [`docs/camera.md`](docs/camera.md), [`docs/lighting.md`](docs/lighting.md), [`docs/background.md`](docs/background.md)

> [!NOTE]
> Each object or task contribution must live in its own folder, and the folder name must reference your work (e.g. the dataset or task name). Objects must carry their own inherited license.

In your PR, please add any appropriate citations, contact information and affiliation for the assets. We will cite your work.

## Developer Certificate of Origin (DCO)

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

  You can also configure commit signing as default for this repo with:
  ```
  git config commit.gpgsign true
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```
