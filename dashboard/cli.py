# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""CLI entry point: `robolab-dashboard --output-dir <path> [--port 8080]`."""

import argparse
from pathlib import Path

import uvicorn

from dashboard.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboLab eval results dashboard")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional. Directory containing run subfolders. If omitted, "
                             "the dashboard launches with whatever sources were previously "
                             "added via the UI (persisted to "
                             "~/.config/robolab-dashboard/sources.json); add more from the "
                             "sidebar at runtime.")
    parser.add_argument("--scenes-metadata-dir", type=Path, default=None,
                        help="Directory containing scene_metadata.json (default: auto-detect "
                             "from robolab.constants.PACKAGE_DIR/assets/scenes/_metadata "
                             "or a sibling robolab/ checkout)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind host (default: 0.0.0.0 — accessible on LAN)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code change (dev only)")
    args = parser.parse_args()

    output_dir: Path | None = args.output_dir.resolve() if args.output_dir else None
    if output_dir is not None and not output_dir.exists():
        # Don't hard-error — just warn and let the user add via the sidebar.
        print(f"warning: --output-dir does not exist ({output_dir}); starting empty")
        output_dir = None

    app = create_app(output_dir, scenes_dir=args.scenes_metadata_dir)
    print(f"Serving {output_dir or '(no initial output dir)'} on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
