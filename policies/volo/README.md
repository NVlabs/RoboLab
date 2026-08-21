# VoLo

This directory provides RoboLab clients for [VoLo](https://chicychen.github.io/VoLo/), a physical orchestrator for open-vocabulary long-horizon manipulation. See the VoLo project page for installation and setup instructions.

The additional tasks from the VoLo project are available at the [RoboVoLo](https://github.com/NVlabs/RoboVoLo) task library.

After starting the policy server and VoLo proxy, run an evaluation with:

```bash
python policies/volo/run.py \
  --policy pi05 \
  --task BananaInBowlTask \
  --remote-host localhost \
  --remote-port 8001
```

The runner supports `cosmos3` and the Pi0-family policies. See the [policies README](../README.md) for shared evaluation options.

`--remote-uri` is available for every backend and overrides `--remote-host`
and `--remote-port`. For an authenticated Cosmos3 endpoint, use
`--remote-token` or the `COSMOS3_API_TOKEN` environment variable; this token
option does not apply to the Pi0-family client.

## How the proxy backend works

The VoLo clients wrap existing backends (cosmos3, pi0 family — selected via `--policy`) for use behind a VoLo inference proxy. They compose `OrchestratorMetadataMixin` ([`metadata.py`](./metadata.py)) over the backend client, producing a request that is a strict superset of the backend's wire format: depth images, camera calibration, front RGB, an `__episode_id`, and (with `--enable-gt-state`) simulator ground-truth state with client-side lift tracking and grasp detection derived from the raw per-step snapshot. Cosmos3 requests also retain the backend's per-environment `session_id`; it identifies policy state, while `__episode_id` identifies the VoLo orchestration episode. Proxies should forward both without treating them as aliases.

The runner registers environments via [`registration.py`](./registration.py), which turns on depth rendering plus `<camera>_depth` and `<camera>_pos/_quat/_K` observation terms, and adds the `robovolo` content-pack folder to task discovery; standard backends never pay that cost.
