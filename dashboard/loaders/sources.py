# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Persistent registry of output directories the dashboard reads from.

Stored as a flat JSON list at ``$XDG_CONFIG_HOME/robolab-dashboard/sources.json``
(defaults to ``~/.config/robolab-dashboard/sources.json``). The dashboard CLI
seeds this with ``--output-dir`` on first launch; users can then add/remove
sources from the UI without restarting.
"""

import json
import os
from pathlib import Path


def _config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "robolab-dashboard" / "sources.json"


class SourceRegistry:
    def __init__(self, path: Path | None = None):
        self.path = path or _config_path()

    def load(self) -> list[Path]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return []
        out: list[Path] = []
        seen: set[Path] = set()
        for item in raw if isinstance(raw, list) else []:
            try:
                p = Path(item).expanduser().resolve()
            except (TypeError, ValueError):
                continue
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def save(self, dirs: list[Path]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [str(p) for p in dirs]
        self.path.write_text(json.dumps(payload, indent=2))

    def add(self, p: Path) -> list[Path]:
        p = p.expanduser().resolve()
        if not p.exists() or not p.is_dir():
            raise ValueError(f"not a directory: {p}")
        dirs = self.load()
        if p not in dirs:
            dirs.append(p)
            self.save(dirs)
        return dirs

    def remove(self, p: Path) -> list[Path]:
        p = p.expanduser().resolve()
        dirs = [d for d in self.load() if d != p]
        self.save(dirs)
        return dirs

    def seed_if_empty(self, initial: Path) -> list[Path]:
        dirs = self.load()
        if dirs:
            return dirs
        initial = initial.expanduser().resolve()
        if initial.is_dir():
            self.save([initial])
            return [initial]
        return []
