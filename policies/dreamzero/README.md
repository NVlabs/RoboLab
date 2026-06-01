# DreamZero

## DreamZero Server

Follow instructions on running the DreamZero server [here](https://github.com/dreamzero0/dreamzero#running-the-inference-server). The checkpoint to use for DROID is [DreamZero-DROID](https://huggingface.co/GEAR-Dreams/DreamZero-DROID).

## Run evaluation

On the RoboLab side, run the following:

```bash
uv run python policies/dreamzero/run.py --task BananaInBowlTask --headless --remote-host xx.xx.xx.xxx --remote-port 5000
```

## Backend-specific flags

| Flag | Default | Description |
|------|---------|-------------|
| `--remote-host` | `localhost` | Policy server host. |
| `--remote-port` | `5000` | Policy server port. |
| `--open-loop-horizon` | client default | Actions executed per predicted chunk before re-querying. |
| `--dz-binarize-gripper` | off | Re-enable gripper binarization at 0.5 threshold (ablation). |
| `--dz-resize {area,linear,pad}` | `area` | Image resize method. `area`/`linear` change aspect ratio if source differs from the 180x320 target; `pad` letterboxes. |
| `--cam2-source {black,right,head,duplicate}` | `black` | Second exterior camera: `black` (training dropout), `right` (over-shoulder), `head` (front overhead), `duplicate` (copy of left). `right`/`head` attach the extra cameras at registration time. |

See the [policies README](../README.md) for shared client architecture and common CLI options.
