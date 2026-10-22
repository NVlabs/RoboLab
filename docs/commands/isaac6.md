# Isaac Sim 6 validation commands

Use a separate Python 3.12 environment for the Isaac Sim 6 stack:

```bash
uv sync --python 3.12 --extra isaac60 --extra test
uv run --no-sync pytest tests/
uv run --no-sync python examples/run_kinova_jointpos.py --headless --num-steps 180
uv run --no-sync python examples/run_abs_ik_demo.py --headless
uv run --no-sync python scripts/preflight_tasks.py --headless --task RubiksCubeAndBananaTask
uv run --no-sync python scripts/preflight_tasks.py --headless --report output/isaac60_preflight.jsonl
./docker/build_docker.sh --isaac60
```

The task preflight report is resumable. These commands describe required runtime
validation; importing the changes does not establish a new GPU test result.
