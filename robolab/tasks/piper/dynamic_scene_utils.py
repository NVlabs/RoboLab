# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for building runtime Piper pick-place scenes."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from robolab.constants import OBJECT_CATALOG_PATH, PACKAGE_DIR, SCENE_DIR, get_timestamp, resolve_catalog_path
from robolab.core.scenes.utils import find_scene_file


DEFAULT_BASE_SCENE = "piper_pick_place_base.usda"
DEFAULT_OBJECT_POS = (0.328, 0.0, 0.20)
DEFAULT_OBJECT_ROT = (1.0, 0.0, 0.0, 0.0)
DEFAULT_OBJECT_SCALE = (1.0, 1.0, 1.0)
GENERATED_SCENE_ENV = "ROBOLAB_PIPER_DYNAMIC_SCENE"
OBJECT_NAME_ENV = "ROBOLAB_PIPER_DYNAMIC_OBJECT_NAME"
OBJECT_NAMES_ENV = "ROBOLAB_PIPER_DYNAMIC_OBJECT_NAMES"
INSTRUCTION_ENV = "ROBOLAB_PIPER_DYNAMIC_INSTRUCTION"

_PAYLOAD_RE = re.compile(r"@([^@]+)@")
_VALID_PRIM_CHARS_RE = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class DynamicObjectSpec:
    name: str
    usd_path: str
    pos: tuple[float, float, float] = DEFAULT_OBJECT_POS
    rot: tuple[float, float, float, float] = DEFAULT_OBJECT_ROT
    scale: tuple[float, float, float] = DEFAULT_OBJECT_SCALE


def sanitize_prim_name(name: str) -> str:
    """Return a USD-safe prim name."""
    cleaned = _VALID_PRIM_CHARS_RE.sub("_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("Object prim name is empty after sanitization.")
    if cleaned[0].isdigit():
        cleaned = f"object_{cleaned}"
    return cleaned


def resolve_object_usd(dynamic_object: str | None = None, dynamic_object_usd: str | None = None) -> tuple[str, str]:
    """Resolve an object catalog name or USD path to ``(object_name, usd_path)``."""
    if dynamic_object_usd:
        usd_path = _resolve_repo_path(dynamic_object_usd)
        object_name = dynamic_object or Path(usd_path).stem
        return sanitize_prim_name(object_name), usd_path

    if not dynamic_object:
        raise ValueError("Provide --dynamic-object or --dynamic-object-usd.")

    possible_path = _maybe_existing_path(dynamic_object)
    if possible_path:
        return sanitize_prim_name(Path(possible_path).stem), possible_path

    with open(OBJECT_CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    for entry in catalog:
        if entry.get("name") == dynamic_object:
            return sanitize_prim_name(dynamic_object), resolve_catalog_path(entry["usd_path"])

    known = sorted({entry.get("name", "") for entry in catalog if entry.get("name")})
    preview = ", ".join(known[:40])
    raise ValueError(f"Object '{dynamic_object}' not found in catalog. First known names: {preview}")


def sample_dynamic_objects(
    *,
    target_object_name: str | None,
    target_object_usd_path: str | None,
    count: int,
    categories: Iterable[str] | None = None,
    datasets: Iterable[str] | None = None,
    object_pool: Iterable[str] | None = None,
    sample_with_replacement: bool = False,
    seed: int | None = None,
    center: Iterable[float] = DEFAULT_OBJECT_POS,
    area: Iterable[float] = (0.22, 0.20),
    z: float | None = None,
    object_rot: Iterable[float] = DEFAULT_OBJECT_ROT,
    scale: Iterable[float] = DEFAULT_OBJECT_SCALE,
) -> tuple[DynamicObjectSpec, list[DynamicObjectSpec]]:
    """Build target + randomized distractor specs for a dynamic pick-place scene.

    By default, distractors are sampled without replacement and therefore need
    distinct catalog entries. ``sample_with_replacement`` permits repeated
    catalog entries; each repeated instance receives a unique USD prim name.
    """
    if count < 1:
        raise ValueError("--dynamic-object-count must be >= 1.")

    rng = random.Random(seed)
    center_tuple = tuple(float(v) for v in center)
    if len(center_tuple) != 3:
        raise ValueError("center must have 3 values.")
    area_tuple = tuple(float(v) for v in area)
    if len(area_tuple) != 2:
        raise ValueError("--dynamic-object-area must have 2 values.")
    scale_tuple = tuple(float(v) for v in scale)
    if len(scale_tuple) != 3:
        raise ValueError("--dynamic-object-scale must have 3 values.")
    object_rot_tuple = tuple(float(v) for v in object_rot)
    if len(object_rot_tuple) != 4:
        raise ValueError("--dynamic-object-rot must have 4 values.")

    catalog = _load_object_catalog()

    candidates = _filter_catalog(catalog, categories=categories, datasets=datasets, object_pool=object_pool)
    if not candidates and count > 1:
        raise ValueError("No catalog objects matched the requested dynamic object filters.")

    if target_object_usd_path:
        if not target_object_name:
            target_object_name = Path(target_object_usd_path).stem
        target_name, target_usd = resolve_object_usd(target_object_name, target_object_usd_path)
    elif target_object_name:
        target_name, target_usd = resolve_object_usd(target_object_name, None)
    else:
        if not candidates:
            raise ValueError("Provide --dynamic-object, --dynamic-object-usd, or random object filters.")
        target_entry = rng.choice(candidates)
        target_name = sanitize_prim_name(target_entry["name"])
        target_usd = resolve_catalog_path(target_entry["usd_path"])

    selected_entries = []
    distractor_count = count - 1
    if distractor_count > 0:
        if sample_with_replacement:
            # Include the target's catalog entry as a possible distractor. This
            # is intentional: repeated instances are assigned distinct prim
            # names below (e.g. ``lime01``, ``lime01_01``).
            selected_entries = [rng.choice(candidates) for _ in range(distractor_count)]
        else:
            target_catalog_name = target_object_name or target_name
            distractor_candidates = [
                entry for entry in candidates
                if entry.get("name") != target_catalog_name and entry.get("usd_path") != _catalog_relpath(target_usd)
            ]
            if len(distractor_candidates) < distractor_count:
                raise ValueError(
                    f"Requested {distractor_count} distractor object(s), but only "
                    f"{len(distractor_candidates)} matched the filters. "
                    "Use --dynamic-object-sample-with-replacement to allow repeated objects."
                )
            selected_entries = rng.sample(distractor_candidates, distractor_count)

    positions = _sample_positions(rng, count=count, center=center_tuple, area=area_tuple, z=z)
    rotations = [object_rot_tuple, *[_random_yaw_quat(rng) for _ in range(max(0, count - 1))]]

    target_spec = DynamicObjectSpec(
        name=sanitize_prim_name(target_name),
        usd_path=target_usd,
        pos=positions[0],
        rot=rotations[0],
        scale=scale_tuple,
    )

    used_names = {target_spec.name}
    distractor_specs = []
    for idx, entry in enumerate(selected_entries, start=1):
        base_name = sanitize_prim_name(entry["name"])
        prim_name = _unique_name(base_name, used_names)
        used_names.add(prim_name)
        distractor_specs.append(
            DynamicObjectSpec(
                name=prim_name,
                usd_path=resolve_catalog_path(entry["usd_path"]),
                pos=positions[idx],
                rot=rotations[idx],
                scale=scale_tuple,
            )
        )

    return target_spec, distractor_specs


def build_instruction(object_name: str) -> str:
    return f"Pick up the {object_name.replace('_', ' ')} and place it in the box"


def generate_dynamic_pick_place_scene(
    *,
    object_name: str,
    object_usd_path: str,
    base_scene: str = DEFAULT_BASE_SCENE,
    output_dir: str | None = None,
    object_pos: Iterable[float] = DEFAULT_OBJECT_POS,
    object_rot: Iterable[float] = DEFAULT_OBJECT_ROT,
    object_scale: Iterable[float] = DEFAULT_OBJECT_SCALE,
) -> str:
    """Generate a scene by inserting one dynamic object into a base Piper scene."""
    target_spec = DynamicObjectSpec(
        name=object_name,
        usd_path=object_usd_path,
        pos=tuple(float(v) for v in object_pos),
        rot=tuple(float(v) for v in object_rot),
        scale=tuple(float(v) for v in object_scale),
    )
    return generate_dynamic_pick_place_scene_from_specs(
        target=target_spec,
        distractors=[],
        base_scene=base_scene,
        output_dir=output_dir,
    )


def generate_dynamic_pick_place_scene_from_specs(
    *,
    target: DynamicObjectSpec,
    distractors: Iterable[DynamicObjectSpec] = (),
    base_scene: str = DEFAULT_BASE_SCENE,
    output_dir: str | None = None,
) -> str:
    """Generate a scene by inserting target and distractor objects into a base Piper scene."""
    base_scene_path = find_scene_file(base_scene, SCENE_DIR)
    if not os.path.exists(base_scene_path):
        raise FileNotFoundError(f"Base scene not found: {base_scene}")

    object_specs = [target, *list(distractors)]
    _validate_object_specs(object_specs)

    if output_dir is None:
        output_dir = os.path.join(PACKAGE_DIR, "output", "generated_scenes", get_timestamp())
    os.makedirs(output_dir, exist_ok=True)

    with open(base_scene_path, "r", encoding="utf-8") as f:
        text = f.read()

    text = _rewrite_relative_payloads(text, os.path.dirname(base_scene_path))
    object_block = "\n".join(_format_object_block_from_spec(spec) for spec in object_specs)

    marker = '    def Xform "GroundPlane"'
    if marker not in text:
        raise ValueError(f"Base scene has no insertion marker: {marker}")
    text = text.replace(marker, object_block + "\n" + marker, 1)

    suffix = target.name if len(object_specs) == 1 else f"{target.name}_{len(object_specs)}objects"
    output_path = os.path.join(output_dir, f"piper_pick_place_{suffix}.usda")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path


def export_dynamic_env(
    scene_path: str,
    object_name: str,
    instruction: str | None = None,
    object_names: Iterable[str] | None = None,
) -> None:
    """Expose the generated scene to the runtime task module through env vars."""
    os.environ[GENERATED_SCENE_ENV] = scene_path
    os.environ[OBJECT_NAME_ENV] = sanitize_prim_name(object_name)
    if object_names is None:
        object_names = [object_name]
    os.environ[OBJECT_NAMES_ENV] = json.dumps([sanitize_prim_name(name) for name in object_names])
    os.environ[INSTRUCTION_ENV] = instruction or build_instruction(object_name)


def settle_scene_in_place(
    scene_path: str,
    simulation_app,
    *,
    object_names: Iterable[str] | None = None,
    sequential_drop: bool = False,
    steps: int = 300,
    steps_per_object: int = 120,
    final_steps: int = 120,
    hold_z_offset: float = 0.35,
) -> None:
    """Run physics settling on a generated scene and save it in place."""
    if sequential_drop:
        _settle_scene_sequential(
            scene_path,
            simulation_app,
            object_names=list(object_names or []),
            steps_per_object=steps_per_object,
            final_steps=final_steps,
            hold_z_offset=hold_z_offset,
        )
        return

    import importlib.util

    settle_path = os.path.join(PACKAGE_DIR, "assets", "scenes", "_utils", "settle_scenes.py")
    spec = importlib.util.spec_from_file_location("robolab_runtime_settle_scenes", settle_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load settle utility: {settle_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    temp_path = scene_path + ".settled.usda"
    module.open_and_save_scene(scene_path, temp_path, simulation_app)
    os.replace(temp_path, scene_path)


def _settle_scene_sequential(
    scene_path: str,
    simulation_app,
    *,
    object_names: list[str],
    steps_per_object: int,
    final_steps: int,
    hold_z_offset: float,
) -> None:
    if not object_names:
        raise ValueError("Sequential drop requires at least one object name.")

    from pxr import Gf, Sdf
    import omni.timeline
    import omni.usd

    clean_physics_for_export = _load_clean_physics_for_export()
    temp_path = scene_path + ".settled.usda"

    omni.usd.get_context().open_stage(scene_path)
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()

    prims = []
    original_positions = {}
    for name in object_names:
        prim = stage.GetPrimAtPath(f"/World/{sanitize_prim_name(name)}")
        if not prim or not prim.IsValid():
            raise ValueError(f"Object prim not found for sequential drop: /World/{name}")
        prims.append((sanitize_prim_name(name), prim))
        pos = _get_translate(prim)
        original_positions[sanitize_prim_name(name)] = pos
        _set_bool_attr(prim, "physics:rigidBodyEnabled", True)
        _clear_velocities(prim)

    highest_z = max(pos[2] for pos in original_positions.values()) + hold_z_offset
    for idx, (name, prim) in enumerate(prims):
        pos = original_positions[name]
        _set_translate(prim, (pos[0], pos[1], highest_z + idx * 0.05))
        _set_bool_attr(prim, "physics:kinematicEnabled", True)

    timeline.play()
    for drop_idx, (name, prim) in enumerate(prims):
        pos = original_positions[name]
        _clear_velocities(prim)
        _set_translate(prim, pos)
        _set_bool_attr(prim, "physics:kinematicEnabled", False)
        print(f"\033[96m[RoboLab] Sequential settle drop {drop_idx + 1}/{len(prims)}: {name}\033[0m")
        _run_updates(simulation_app, steps_per_object)

    _run_updates(simulation_app, final_steps)
    timeline.pause()

    stage = omni.usd.get_context().get_stage()
    clean_physics_for_export(stage)
    stage.GetRootLayer().Export(temp_path)
    os.replace(temp_path, scene_path)


def _load_clean_physics_for_export():
    import importlib.util

    settle_path = os.path.join(PACKAGE_DIR, "assets", "scenes", "_utils", "settle_scenes.py")
    spec = importlib.util.spec_from_file_location("robolab_runtime_settle_scenes", settle_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load settle utility: {settle_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.clean_physics_for_export


def _run_updates(simulation_app, steps: int) -> None:
    for _ in range(steps):
        simulation_app.update()


def _get_translate(prim) -> tuple[float, float, float]:
    attr = prim.GetAttribute("xformOp:translate")
    value = attr.Get() if attr and attr.IsValid() else None
    if value is None:
        return DEFAULT_OBJECT_POS
    return (float(value[0]), float(value[1]), float(value[2]))


def _set_translate(prim, pos: tuple[float, float, float]) -> None:
    from pxr import Gf, Sdf

    attr = prim.GetAttribute("xformOp:translate")
    if not attr or not attr.IsValid():
        attr = prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
    attr.Set(Gf.Vec3d(*pos))


def _set_bool_attr(prim, name: str, value: bool) -> None:
    from pxr import Sdf

    attr = prim.GetAttribute(name)
    if not attr or not attr.IsValid():
        attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool)
    attr.Set(bool(value))


def _clear_velocities(prim) -> None:
    from pxr import Gf, Sdf

    for attr_name in ("physics:velocity", "physics:angularVelocity"):
        attr = prim.GetAttribute(attr_name)
        if not attr or not attr.IsValid():
            attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Vector3f)
        attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))


def _resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    candidate = os.path.join(PACKAGE_DIR, path)
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(path)


def _maybe_existing_path(path: str) -> str | None:
    candidates = [
        path,
        os.path.join(PACKAGE_DIR, path),
        os.path.join(PACKAGE_DIR, "assets", "objects", path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def _load_object_catalog() -> list[dict]:
    with open(OBJECT_CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_catalog(
    catalog: list[dict],
    *,
    categories: Iterable[str] | None,
    datasets: Iterable[str] | None,
    object_pool: Iterable[str] | None,
) -> list[dict]:
    category_set = {c for c in (categories or []) if c}
    dataset_set = {d for d in (datasets or []) if d}
    object_set = {o for o in (object_pool or []) if o}

    out = []
    for entry in catalog:
        if not entry.get("rigid_body") or entry.get("static_body"):
            continue
        if object_set and entry.get("name") not in object_set:
            continue
        if category_set and entry.get("class") not in category_set:
            continue
        if dataset_set and entry.get("dataset") not in dataset_set:
            continue
        out.append(entry)
    return out


def _catalog_relpath(abs_path: str) -> str:
    try:
        return os.path.relpath(abs_path, PACKAGE_DIR)
    except ValueError:
        return abs_path


def _sample_positions(
    rng: random.Random,
    *,
    count: int,
    center: tuple[float, float, float],
    area: tuple[float, float],
    z: float | None,
) -> list[tuple[float, float, float]]:
    if count == 1:
        return [(center[0], center[1], center[2] if z is None else float(z))]

    cols = max(1, int(count**0.5 + 0.999))
    rows = max(1, (count + cols - 1) // cols)
    x_step = area[0] / max(cols, 1)
    y_step = area[1] / max(rows, 1)
    x0 = center[0] - area[0] / 2.0 + x_step / 2.0
    y0 = center[1] - area[1] / 2.0 + y_step / 2.0
    positions = []
    for idx in range(count):
        row, col = divmod(idx, cols)
        jitter_x = rng.uniform(-0.2, 0.2) * x_step
        jitter_y = rng.uniform(-0.2, 0.2) * y_step
        positions.append((
            x0 + col * x_step + jitter_x,
            y0 + row * y_step + jitter_y,
            center[2] if z is None else float(z),
        ))
    rng.shuffle(positions)
    return positions


def _random_yaw_quat(rng: random.Random) -> tuple[float, float, float, float]:
    import math

    yaw = rng.uniform(-math.pi, math.pi)
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _unique_name(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        return base_name
    idx = 1
    while True:
        candidate = f"{base_name}_{idx:02d}"
        if candidate not in used_names:
            return candidate
        idx += 1


def _validate_object_specs(specs: list[DynamicObjectSpec]) -> None:
    reserved = {"scene", "table", "pick_box", "place_box", "GroundPlane"}
    names = set()
    for spec in specs:
        name = sanitize_prim_name(spec.name)
        if name in reserved:
            raise ValueError(f"Object name '{name}' conflicts with an existing scene prim.")
        if name in names:
            raise ValueError(f"Duplicate object prim name: {name}")
        names.add(name)

        usd_path = _resolve_repo_path(spec.usd_path)
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"Object USD not found: {usd_path}")


def _rewrite_relative_payloads(text: str, base_dir: str) -> str:
    def replace(match: re.Match[str]) -> str:
        asset = match.group(1)
        if os.path.isabs(asset) or "://" in asset:
            return match.group(0)
        return f"@{os.path.abspath(os.path.join(base_dir, asset))}@"

    return _PAYLOAD_RE.sub(replace, text)


def _format_usd_tuple(values: tuple[float, ...]) -> str:
    return "(" + ", ".join(f"{v:.10g}" for v in values) + ")"


def _format_object_block(
    *,
    object_name: str,
    object_usd_path: str,
    object_pos: tuple[float, ...],
    object_rot: tuple[float, ...],
    object_scale: tuple[float, ...],
) -> str:
    if len(object_pos) != 3:
        raise ValueError("--dynamic-object-pos must have 3 values.")
    if len(object_rot) != 4:
        raise ValueError("--dynamic-object-rot must have 4 values in qw qx qy qz order.")
    if len(object_scale) != 3:
        raise ValueError("--dynamic-object-scale must have 3 values.")

    return f'''    def "{object_name}" (
        prepend payload = @{object_usd_path}@
    )
    {{
        vector3f physics:angularVelocity
        vector3f physics:velocity
        quatf xformOp:orient = {_format_usd_tuple(object_rot)}
        float3 xformOp:scale = {_format_usd_tuple(object_scale)}
        double3 xformOp:translate = {_format_usd_tuple(object_pos)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}
'''


def _format_object_block_from_spec(spec: DynamicObjectSpec) -> str:
    return _format_object_block(
        object_name=sanitize_prim_name(spec.name),
        object_usd_path=_resolve_repo_path(spec.usd_path),
        object_pos=tuple(float(v) for v in spec.pos),
        object_rot=tuple(float(v) for v in spec.rot),
        object_scale=tuple(float(v) for v in spec.scale),
    )
