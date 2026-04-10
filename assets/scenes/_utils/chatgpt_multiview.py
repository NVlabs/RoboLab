#!/usr/bin/env python3
"""
Render BOP objects with PyVista and (optionally) generate ChatGPT descriptions.
This gives nicer rasterisation than pyrender without needing an OpenGL context.
"""

import os
import sys
import json
import time
import base64
import io
from pathlib import Path

# Optional: load environment variables from .env so OPENAI_API_KEY can be set there
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    # Try usual .env file first; if not found, fallback to .venv (text key store)
    if Path('.env').is_file():
        load_dotenv('.env')
    elif Path('.venv').is_file():
        load_dotenv('.venv')
    else:
        load_dotenv()
except ImportError:
    pass  # dotenv not available – fall back to existing environment

import numpy as np
from PIL import Image
import pyvista as pv  # pip install pyvista
import trimesh        # still used for bounding‐box centring
from openai import OpenAI
import argparse
import zipfile
import tempfile

# Use off-screen rendering globally (headless / SSH).  PyVista API differs across versions.
try:
    pv.global_theme.off_screen = True  # PyVista ≥ 0.43
except AttributeError:
    try:
        pv.rcParams["off_screen"] = True  # Older versions
    except Exception:
        pass  # Fallback – we still pass off_screen=True per-plotter

class PyVistaBOPRenderer:
    def __init__(self, dataset_path: str | Path, output_dir: str = "renders_pyvista", debug: bool=False):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.renders_dir = self.output_dir / "renders"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.renders_dir.mkdir(exist_ok=True)

        # Discover USD files recursively
        self.usd_files = self._discover_usd_files()

        # OpenAI client (optional)
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.vision_model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")
        if self.client:
            print(f"Using ChatGPT vision model: {self.vision_model}")

        self.width, self.height = 640, 480
        print(f"Found {len(self.usd_files)} USD files to render under {self.dataset_path}")

        self.debug = debug

    def _discover_usd_files(self):
        """Recursively find all *.usd and *.usdz files under dataset_path (objects)."""
        # Include both plain USD files and zipped USDZ packages
        usd_files = list(self.dataset_path.rglob("*.usd"))
        usd_files += list(self.dataset_path.rglob("*.usdz"))
        return sorted(usd_files)

    def _prepare_usd_path(self, usd_path: Path):
        """If given a .usdz archive, unzip it to a temp dir and return the root USD path.
        Returns (resolved_path, tmp_dir or None).  Caller should tmp_dir.cleanup() when done.
        """
        if usd_path.suffix.lower() != ".usdz":
            return usd_path, None

        tmp_dir = tempfile.TemporaryDirectory()
        try:
            with zipfile.ZipFile(usd_path, "r") as zf:
                zf.extractall(tmp_dir.name)
        except Exception as e:
            if self.debug:
                print(f"[WARNING] Failed to unzip {usd_path}: {e}")
            tmp_dir.cleanup()
            return usd_path, None

        root_candidates = (
            list(Path(tmp_dir.name).glob("*.usd"))
            + list(Path(tmp_dir.name).glob("*.usda"))
            + list(Path(tmp_dir.name).glob("*.usdc"))
        )
        if not root_candidates:
            # search recursively
            root_candidates = (
                list(Path(tmp_dir.name).rglob("*.usd"))
                + list(Path(tmp_dir.name).rglob("*.usda"))
                + list(Path(tmp_dir.name).rglob("*.usdc"))
            )

        root_file = None
        for cand in root_candidates:
            if cand.stem == usd_path.stem:
                root_file = cand
                break
        if root_file is None and root_candidates:
            root_file = root_candidates[0]

        if root_file is None:
            if self.debug:
                print(f"[WARNING] No USD file found inside {usd_path}")
            tmp_dir.cleanup()
            return usd_path, None

        return root_file, tmp_dir

    # --------------------------------------------------- internal helpers
    def _usd_to_trimesh(self, usd_path: Path):
        """Convert a .usd mesh (or mesh hierarchy) to (trimesh, uvs) tuple.
        uvs is a (N, 2) array of texture coordinates or None. Requires pxr USD.
        """
        try:
            from pxr import Usd, UsdGeom, UsdShade, Sdf
        except ImportError:
            if self.debug:
                print("[WARNING] pxr module not available – cannot read USD meshes.")
            return None

        stage = Usd.Stage.Open(str(usd_path))
        vertices_list = []
        faces_list = []
        uvs_list = []
        vert_offset = 0

        # ------------------------------------------------------------------
        # Attempt to find a diffuse/albedo texture referenced in the material
        # ------------------------------------------------------------------
        texture_from_material: Path | None = None
        try:
            for prim in stage.Traverse():
                if prim.IsA(UsdShade.Shader):
                    shader = UsdShade.Shader(prim)
                    for tex_name in ("diffuse_texture", "albedo_texture", "baseColor_texture"):
                        inp = shader.GetInput(tex_name)
                        if inp and inp.HasAuthoredValue():
                            asset_path = inp.Get().assetPath if isinstance(inp.Get(), Sdf.AssetPath) else None
                            if asset_path:
                                texture_from_material = usd_path.parent / asset_path
                                break
                    if texture_from_material:
                        break
        except Exception:
            pass

        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                verts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
                counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
                indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)

                if len(verts) == 0 or len(indices) == 0:
                    continue

                # Fan-triangulate each polygon
                start = 0
                for c in counts:
                    face_indices = indices[start:start + c]
                    for j in range(1, c - 1):
                        faces_list.append([
                            vert_offset + face_indices[0],
                            vert_offset + face_indices[j],
                            vert_offset + face_indices[j + 1],
                        ])
                    start += c

                vertices_list.append(verts)
                vert_offset += len(verts)

                # UV coordinates – attempt to fetch primvar 'st'
                uv_vals = None
                try:
                    # Dump all primvars for debugging
                    if self.debug:
                        print(f"Mesh prim: {prim.GetPath()}")
                        for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
                            arr = pv.Get() if pv.HasAuthoredValue() else None
                            arr_len = len(arr) if arr is not None else 0
                            print(
                                f"    primvar '{pv.GetBaseName()}': interp={pv.GetInterpolation()}, "
                                f"authored={pv.HasAuthoredValue()}, indexed={pv.IsIndexed()}, len={arr_len}"
                            )

                    # Prefer PrimvarsAPI retrieval
                    uv_attr = UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st')
                    if (not uv_attr) or (not uv_attr.HasValue()):
                        # Fallback: iterate over primvars list
                        for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
                            if pv.GetBaseName() in ('st', 'uv') and pv.HasValue():
                                uv_attr = pv
                                break
                        else:
                            uv_attr = None
                    if uv_attr and uv_attr.HasValue():
                        uv_raw = uv_attr.Get()
                        # Convert list of Gf.Vec2f -> Nx2 float32
                        uv_vals = np.array([[v[0], v[1]] for v in uv_raw], dtype=np.float32)
                        # Handle indexed primvars
                        if uv_attr.IsIndexed():
                            try:
                                idx = np.array(uv_attr.GetIndices(), dtype=np.int64)
                                uv_vals = uv_vals[idx]
                            except Exception:
                                pass
                except Exception:
                    uv_vals = None

                if uv_vals is None or len(uv_vals) == 0:
                    # Fallback: create dummy UVs so array sizes stay aligned
                    uv_vals = np.zeros((len(verts), 2), dtype=np.float32)
                uvs_list.append(uv_vals)

        if not vertices_list or not faces_list:
            if self.debug:
                print(f"[WARNING] No mesh prims found in {usd_path}")
            return None

        vertices = np.vstack(vertices_list)
        faces = np.asarray(faces_list, dtype=np.int64)
        uvs = np.vstack(uvs_list)
        # Determine if we have meaningful UVs (not all zeros)
        has_meaningful_uv = not np.allclose(uvs, 0.0)
        if not has_meaningful_uv:
            uvs = None

        return (trimesh.Trimesh(vertices=vertices, faces=faces, process=False), uvs, texture_from_material)

    def _load_mesh(self, usd_path: Path):
        """Load mesh and UVs from USD."""
        result = self._usd_to_trimesh(usd_path)
        return result  # may be None or (mesh, uvs, tex_path)

    # --------------------------------------------------------- rendering
    def _render_single(self, usd_path: Path) -> np.ndarray | None:
        usd_path_resolved, tmp_dir = self._prepare_usd_path(usd_path)
        loaded = self._load_mesh(usd_path_resolved)
        if loaded is None:
            print(f"Mesh not found or failed to load for {usd_path}")
            if tmp_dir:
                tmp_dir.cleanup()
            return None
        tri_mesh, uvs, material_tex_path = loaded
        model_dir = usd_path_resolved.parent
        # Center mesh and compute bbox from geometry
        bounds = tri_mesh.bounds  # (min, max)
        center = (bounds[0] + bounds[1]) / 2
        tri_mesh.vertices -= center
        bbox_size = bounds[1] - bounds[0]

        # Rotate: make original Z axis become world Y (up).
        v = tri_mesh.vertices
        tri_mesh.vertices = np.column_stack((v[:, 0], v[:, 2], -v[:, 1]))

        # Convert to PyVista mesh
        pv_mesh = pv.wrap(tri_mesh)
        has_uv = False
        if uvs is not None and len(uvs) == pv_mesh.n_points:
            pv_mesh.point_data['Texture Coordinates'] = uvs  # PyVista convention
            try:
                pv_mesh.active_t_coords_name = 'Texture Coordinates'
            except AttributeError:
                pass  # older PyVista versions
            has_uv = True

        if self.debug:
            if has_uv:
                print(
                    f"  -> UVs ok: {len(uvs)} coords, min={uvs.min(axis=0)}, max={uvs.max(axis=0)}"
                )
            else:
                if uvs is None:
                    print("  -> no UV data found; generating planar UVs")
                    # Simple planar projection onto XZ plane
                    verts = pv_mesh.points
                    min_xz = verts[:, [0, 2]].min(axis=0)
                    max_xz = verts[:, [0, 2]].max(axis=0)
                    size_xz = np.where((max_xz - min_xz) == 0.0, 1.0, (max_xz - min_xz))
                    gen_uv = (verts[:, [0, 2]] - min_xz) / size_xz
                    pv_mesh.point_data['Texture Coordinates'] = gen_uv.astype(np.float32)
                    try:
                        pv_mesh.active_t_coords_name = 'Texture Coordinates'
                    except AttributeError:
                        pass
                    has_uv = True
                    print("  -> planar UVs generated")
                else:
                    print(
                        f"  -> UV data length mismatch (uvs={len(uvs)}, points={pv_mesh.n_points})"
                    )

        # ------------------------------------------------ texture support
        texture = None
        selected_tex_path = None
        tex_candidates = [
            usd_path_resolved.with_suffix('.png'),
            usd_path_resolved.with_suffix('.jpg'),
            usd_path_resolved.parent / 'textures' / (usd_path_resolved.stem + '.png'),
            usd_path_resolved.parent / 'textures' / (usd_path_resolved.stem + '.jpg'),
        ]
        if material_tex_path is not None:
            tex_candidates.insert(0, material_tex_path)
        for tex_path in tex_candidates:
            if tex_path.exists():
                try:
                    texture = pv.read_texture(str(tex_path))
                    if self.debug:
                        print(f"  -> texture loaded: {tex_path.relative_to(self.dataset_path)}")
                    selected_tex_path = tex_path
                    break
                except Exception as e:
                    if self.debug:
                        print(f"  -> texture load failed ({e}) for {tex_path}")
        if self.debug:
            if selected_tex_path is None:
                try:
                    rel_path_dbg = usd_path_resolved.relative_to(self.dataset_path)
                except ValueError:
                    rel_path_dbg = usd_path_resolved.name
                print(f"  -> no texture found for {rel_path_dbg}")

        # Setup plotter
        plotter = pv.Plotter(off_screen=True, window_size=(self.width, self.height))
        plotter.set_background("white")
        if texture is not None and has_uv:
            plotter.add_mesh(pv_mesh, texture=texture, smooth_shading=True)
        else:
            plotter.add_mesh(pv_mesh, color="lightgray", smooth_shading=True, specular=0.3)

        # Show axes for orientation if debug
        if self.debug:
            plotter.show_axes()

        # Camera – place along +Z looking at origin
        distance = float(bbox_size.max() * 2.5)
        plotter.camera.position = (0.0, 0.0, distance)
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.up = (0.0, 1.0, 0.0)
        plotter.camera.view_angle = 30

        if self.debug:
            print(f"  camera position: {plotter.camera.position}, focal {plotter.camera.focal_point}, distance {distance:.2f}")
            print(f"  object bbox (mm): {bbox_size}")

        plotter.render()  # ensure camera change is applied
        img = plotter.screenshot(return_img=True)
        plotter.close()
        if tmp_dir:
            tmp_dir.cleanup()
        return img

    # ------------------------------------------------ helper: tight crop
    def _tight_crop(self, img: np.ndarray, margin: int = 5) -> np.ndarray:
        # mask of non-white pixels (any channel < 250)
        mask = ~(img > 250).all(axis=2)
        coords = np.argwhere(mask)
        if coords.size == 0:
            return img  # nothing found, return original
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        y0 = max(0, y0 - margin)
        x0 = max(0, x0 - margin)
        y1 = min(img.shape[0], y1 + margin)
        x1 = min(img.shape[1], x1 + margin)
        return img[y0:y1, x0:x1]

    def _render_four_views(self, usd_path: Path) -> np.ndarray | None:
        usd_path_resolved, tmp_dir = self._prepare_usd_path(usd_path)
        loaded = self._load_mesh(usd_path_resolved)
        if loaded is None:
            print(f"Mesh not found or failed to load for {usd_path}")
            if tmp_dir:
                tmp_dir.cleanup()
            return None
        tri_mesh, uvs, material_tex_path = loaded
        model_dir = usd_path_resolved.parent
        # Center mesh and compute bbox from geometry
        bounds = tri_mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        tri_mesh.vertices -= center
        bbox_size = bounds[1] - bounds[0]

        # Rotate: make original Z axis become world Y (up).
        v = tri_mesh.vertices
        tri_mesh.vertices = np.column_stack((v[:, 0], v[:, 2], -v[:, 1]))

        # Convert to PyVista mesh
        pv_mesh = pv.wrap(tri_mesh)
        has_uv = False
        if uvs is not None and len(uvs) == pv_mesh.n_points:
            pv_mesh.point_data['Texture Coordinates'] = uvs
            try:
                pv_mesh.active_t_coords_name = 'Texture Coordinates'
            except AttributeError:
                pass
            has_uv = True

        if self.debug:
            if has_uv:
                print(
                    f"   UVs ok: {len(uvs)} coords, min={uvs.min(axis=0)}, max={uvs.max(axis=0)}"
                )
            else:
                if uvs is None:
                    print("   no UV data found; generating planar UVs")
                    verts = pv_mesh.points
                    min_xz = verts[:, [0, 2]].min(axis=0)
                    max_xz = verts[:, [0, 2]].max(axis=0)
                    size_xz = np.where((max_xz - min_xz) == 0.0, 1.0, (max_xz - min_xz))
                    gen_uv = (verts[:, [0, 2]] - min_xz) / size_xz
                    pv_mesh.point_data['Texture Coordinates'] = gen_uv.astype(np.float32)
                    try:
                        pv_mesh.active_t_coords_name = 'Texture Coordinates'
                    except AttributeError:
                        pass
                    has_uv = True
                    print("   planar UVs generated")
                else:
                    print(
                        f"   UV data length mismatch (uvs={len(uvs)}, points={pv_mesh.n_points})"
                    )

        # ------------------------------------------------ texture support
        texture = None
        selected_tex_path = None
        tex_candidates = [
            usd_path_resolved.with_suffix('.png'),
            usd_path_resolved.with_suffix('.jpg'),
            usd_path_resolved.parent / 'textures' / (usd_path_resolved.stem + '.png'),
            usd_path_resolved.parent / 'textures' / (usd_path_resolved.stem + '.jpg'),
        ]
        if material_tex_path is not None:
            tex_candidates.insert(0, material_tex_path)
        for tex_path in tex_candidates:
            if tex_path.exists():
                try:
                    texture = pv.read_texture(str(tex_path))
                    if self.debug:
                        print(f"  -> texture loaded: {tex_path.relative_to(self.dataset_path)}")
                    selected_tex_path = tex_path
                    break
                except Exception as e:
                    if self.debug:
                        print(f"  -> texture load failed ({e}) for {tex_path}")
        if self.debug:
            if selected_tex_path is None:
                try:
                    rel_path_dbg = usd_path_resolved.relative_to(self.dataset_path)
                except ValueError:
                    rel_path_dbg = usd_path_resolved.name
                print(f"  -> no texture found for {rel_path_dbg}")

        # Setup plotter
        plotter = pv.Plotter(off_screen=True, window_size=(self.width, self.height))
        plotter.set_background("white")
        if texture is not None and has_uv:
            plotter.add_mesh(pv_mesh, texture=texture, smooth_shading=True)
        else:
            plotter.add_mesh(pv_mesh, color="lightgray", smooth_shading=True, specular=0.3)

        # Show axes for orientation if debug
        if self.debug:
            plotter.show_axes()

        # generate 4 views around Y axis
        distance = float(bbox_size.max() * 2.5)
        views = [(45, 30), (135, 30), (225, -30), (315, -30)]  # distinct az/elev
        crops = []
        for az, el in views:
            az_rad, el_rad = np.radians(az), np.radians(el)
            x = distance * np.cos(el_rad) * np.sin(az_rad)
            y = distance * np.sin(el_rad)
            z = distance * np.cos(el_rad) * np.cos(az_rad)
            plotter.camera.position = (x, y, z)
            plotter.camera.focal_point = (0.0, 0.0, 0.0)
            plotter.camera.up = (0.0, 1.0, 0.0)
            plotter.camera.view_angle = 30
            if self.debug:
                print(f"   view az={az} el={el}: cam=({x:.1f},{y:.1f},{z:.1f})")
            plotter.render()  # ensure camera change is applied
            img = plotter.screenshot(return_img=True)
            crops.append(self._tight_crop(img))

        plotter.close()
        if tmp_dir:
            tmp_dir.cleanup()
        # create 2×2 grid composite
        h_max = max(im.shape[0] for im in crops)
        w_max = max(im.shape[1] for im in crops)
        canvas = np.ones((h_max*2, w_max*2, 3), dtype=np.uint8) * 255
        for idx, crop in enumerate(crops):
            row = idx // 2
            col = idx % 2
            y_offset = row * h_max + (h_max - crop.shape[0])//2
            x_offset = col * w_max + (w_max - crop.shape[1])//2
            canvas[y_offset:y_offset+crop.shape[0], x_offset:x_offset+crop.shape[1]] = crop
        return canvas

    # ------------------------------------------------------ description
    def _image_to_base64(self, img: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _describe(self, img: np.ndarray):
        """Return a dict with keys: name, class, mass_kg, description (or error)."""
        print(self.client)
        if not self.client:
            return {
                "name": None,
                "class": None,
                "mass_kg": None,
                "description": "(description skipped – no OPENAI_API_KEY)",
            }

        prompt_text = (
            "You are provided an image of a single daily-life object. "
            "Respond ONLY in valid JSON, with the following keys: "
            "name (short object name), class (broad category), mass_kg (estimated weight in kilograms, number), description (two sentences describing appearance, colors, any text)."
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{self._image_to_base64(img)}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=200,
            )
            content = resp.choices[0].message.content.strip()
            print(content)
            # Attempt to locate JSON in response
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Attempt to extract JSON-like substring
                import re

                match = re.search(r"\{[\s\S]*\}", content)
                if match:
                    try:
                        data = json.loads(match.group(0))
                    except Exception:
                        data = {"description": content}
                else:
                    data = {"description": content}

            # Ensure required keys
            for k in ("name", "class", "mass_kg", "description"):
                data.setdefault(k, None)
            return data
        except Exception as e:
            return {"name": None, "class": None, "mass_kg": None, "description": f"Error: {e}"}

    # ------------------------------------------------------ main loop
    def run(self, skip_description=False):
        json_path = self.output_dir / "object_descriptions.json"
        # Load existing JSON if present
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    results = json.load(f)
            except Exception as e:
                print(f"[WARNING] Could not parse existing JSON, starting fresh: {e}")
                results = {}
        else:
            results = {}
        for usd_path in self.usd_files:
            print(f"Rendering {usd_path.relative_to(self.dataset_path)} …", flush=True)
            img = self._render_four_views(usd_path)
            if img is None:
                print("  failed 🛑")
                continue
            # save image
            rel_name = usd_path.stem  # might be obj_000001 or similar
            render_path = self.renders_dir / f"{rel_name}.png"
            Image.fromarray(img).save(render_path)
            desc_data = self._describe(img) if not skip_description else {
                "name": None,
                "class": None,
                "mass_kg": None,
                "description": "(skipped)",
            }
            print("  done")
            obj_key = str(usd_path.relative_to(self.dataset_path))
            results[obj_key] = {
                "render_path": str(render_path.relative_to(self.output_dir)),
                **desc_data,
            }
            # incremental save – overwrite only after updating dict (atomic strategy)
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            time.sleep(0.5)
        print("All objects rendered.")
        return results


def main():
    ap = argparse.ArgumentParser(description="Render BOP objects with PyVista")
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output_dir", default="renders_pyvista")
    ap.add_argument("--skip_description", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not Path(args.dataset_path).exists():
        print("Dataset path not found", file=sys.stderr)
        sys.exit(1)

    renderer = PyVistaBOPRenderer(args.dataset_path, args.output_dir, debug=args.debug)
    renderer.run(skip_description=args.skip_description)


if __name__ == "__main__":
    main() 