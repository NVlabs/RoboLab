# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from PIL import Image

from robolab.constants import BACKGROUND_ASSET_DIR
from robolab.core.utils.csv_utils import save_markdown_table


def convert_hdri_to_png(image_path, png_path, width=None, height=None, overwrite=False):
    """Convert HDR/EXR to PNG, optionally resizing. Skip or overwrite existing PNG."""
    png_path = Path(png_path)
    if png_path.exists() and not overwrite:
        # Skip conversion
        return False

    try:


        image = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return False

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Percentile scaling for better contrast
        max_val = np.percentile(image, 99)
        ldr = np.clip(image / max_val, 0, 1)

        # Gamma correction for display
        gamma = 1.0 / 2.2
        ldr = np.power(ldr, gamma)

        # Convert to 0-255
        ldr = np.clip(ldr * 255, 0, 255).astype(np.uint8)

        img = Image.fromarray(ldr)
        if width or height:
            img.thumbnail((width if width else img.width, height if height else img.height))
        img.save(png_path)
        return True

    except Exception as e:
        print(f"Error converting {image_path} to PNG: {e}")
        return False

def generate_background_data(folder_path, image_width=80, overwrite_existing=False):
    """Generate structured data for all background images."""
    folder = Path(folder_path)

    # Recursively find all HDR and EXR files
    image_files = sorted(folder.rglob("*.hdr")) + sorted(folder.rglob("*.exr"))
    if not image_files:
        print(f"No .hdr or .exr files found in {folder}")
        return None

    # Group image files by their parent folder
    folder_dict = defaultdict(list)
    for image_file in image_files:
        folder_dict[image_file.parent].append(image_file)

    # Collect all data
    background_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_backgrounds": len(image_files),
            "total_folders": len(folder_dict),
            "image_width": image_width
        },
        "folders": {}
    }

    for parent_folder in sorted(folder_dict.keys()):
        rel_parent = parent_folder.relative_to(folder)
        folder_key = str(rel_parent)

        background_data["folders"][folder_key] = {
            "folder_path": folder_key,
            "backgrounds": []
        }

        for image_file in sorted(folder_dict[parent_folder]):
            stem = image_file.stem
            txt_file = image_file.parent / f"{stem}.txt"
            png_file = image_file.parent / f"{stem}.png"

            # Convert HDR or EXR to PNG if it doesn't exist
            success = convert_hdri_to_png(
                image_file,
                png_file,
                width=image_width,
                overwrite=overwrite_existing
            )
            if success:
                print(f"Converted {image_file} -> {png_file}")

            # Use relative paths
            rel_image = image_file.relative_to(folder)
            rel_png = png_file.relative_to(folder)

            # Get description
            description = ""
            if txt_file.exists():
                description = txt_file.read_text(encoding="utf-8").strip()

            # Create background entry
            background_entry = {
                "path": str(rel_image),
                "image": str(rel_png) if png_file.exists() else None,
                "description": description
            }

            background_data["folders"][folder_key]["backgrounds"].append(background_entry)

    return background_data


def save_background_json(background_data, folder_path, output_json="backgrounds.json"):
    """Save background data as JSON file."""
    if background_data is None:
        return None

    folder = Path(folder_path)
    json_path = folder / output_json

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(background_data, f, indent=2, ensure_ascii=False)

    print(f"JSON data saved: {json_path}")
    return json_path


def _wrap_description(description, max_line_length=60):
    """Wrap description text intelligently at commas for better readability."""
    if len(description) <= max_line_length:
        return description

    # Split by commas and rebuild with line breaks
    parts = [part.strip() for part in description.split(',')]
    if len(parts) <= 1:
        return description

    lines = []
    current_line = parts[0]

    for part in parts[1:]:
        # Check if adding this part would exceed the line length
        test_line = current_line + "," + part
        if len(test_line) <= max_line_length:
            current_line = test_line
        else:
            # Start a new line
            lines.append(current_line)
            current_line = part

    # Add the last line
    if current_line:
        lines.append(current_line)

    # Join with HTML line breaks for markdown table compatibility
    return "<br>".join(lines)


def generate_markdown_from_data(background_data, folder_path, output_md="README.md", wrap_descriptions=True):
    """Generate markdown table from background data using csv_utils."""
    if background_data is None:
        return

    folder = Path(folder_path)
    output_path = folder / output_md
    image_width = background_data["metadata"]["image_width"]

    # Convert nested data to flat list of dictionaries for csv_utils
    table_data = []
    multiple_folders = len(background_data["folders"]) > 1

    for folder_key, folder_info in background_data["folders"].items():
        for bg in folder_info["backgrounds"]:
            # Escape pipe characters in description to avoid breaking table
            description = bg["description"].replace("|", "\\|").replace("\n", " ") if bg["description"] else "_No description available._"

            # Add intelligent wrapping for comma-separated tags
            if wrap_descriptions and description != "_No description available._":
                description = _wrap_description(description, 40)

            # Get image cell
            if bg["image"]:
                preview = f'<a href="{bg["image"]}"><img src="{bg["image"]}" width="{image_width}"/></a>'
            else:
                preview = "_No image available._"

            row_data = {
                "Background": f"`{bg['path']}`",
                "Description": description,
                "Preview": preview
            }

            # Add folder column if there are multiple folders
            if multiple_folders:
                row_data["Folder"] = f"`{folder_key}`"

            table_data.append(row_data)

    # Set column order
    if multiple_folders:
        headers = ["Folder", "Background", "Description", "Preview"]
    else:
        headers = ["Background", "Description", "Preview"]

    # Generate title and description
    timestamp = datetime.fromisoformat(background_data["metadata"]["generated_at"]).strftime("%Y-%m-%d %H:%M:%S")
    title = "Background Assets"
    description = f"For `outdoor` backgrounds, download them [here](https://drive.google.com/file/d/1ldbQ3n4rCYwtEeDeqf-Rtujq9n94VFiL/view?usp=sharing)(~2GB). This table was generated automatically from {background_data['metadata']['total_backgrounds']} backgrounds. Last updated: {timestamp}"

    # Use csv_utils save_markdown_table or fallback
    if save_markdown_table is not None:
        save_markdown_table(
            csv_input=table_data,
            output_path=str(output_path),
            title=title,
            description=description,
            headers=headers,
            align="left",
            path_type="relative"
        )
    else:
        # Fallback to manual generation if csv_utils not available
        _fallback_generate_markdown(table_data, output_path, title, description, headers)


def _fallback_generate_markdown(table_data, output_path, title, description, headers):
    """Fallback markdown generation if csv_utils is not available."""
    md_lines = []

    if title:
        md_lines.append(f"# {title}\n")
    if description:
        md_lines.append(f"{description}\n")

    # Add table header
    md_lines.append("| " + " | ".join(headers) + " |")
    md_lines.append("| " + " | ".join(":--" for _ in headers) + " |")

    # Add data rows
    for row in table_data:
        row_values = [str(row.get(header, "")) for header in headers]
        md_lines.append("| " + " | ".join(row_values) + " |")

    markdown_content = "\n".join(md_lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"Markdown generated: {output_path}")


def generate_background_markdown(folder_path, output_md="README.md", output_json="backgrounds.json", image_width=80, overwrite_existing=False, wrap_descriptions=True):
    """Main function that generates both JSON and Markdown files."""
    # Generate structured data
    background_data = generate_background_data(folder_path, image_width, overwrite_existing)

    # Save JSON file
    json_path = save_background_json(background_data, folder_path, output_json)

    # Generate Markdown from data
    generate_markdown_from_data(background_data, folder_path, output_md, wrap_descriptions)

    return background_data, json_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HDR/EXR images to PNG and generate JSON data and Markdown table recursively."
    )
    parser.add_argument("--folder_path", default=BACKGROUND_ASSET_DIR, help="Path to folder containing .hdr, .exr, .png, and .txt files")
    parser.add_argument("--output-md", default="README.md", help="Name of output Markdown file")
    parser.add_argument("--output-json", default="backgrounds.json", help="Name of output JSON file")
    parser.add_argument("--image-width", type=int, default=600, help="Width of images in Markdown")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files")
    parser.add_argument("--no-wrap", action="store_true", help="Disable intelligent text wrapping for descriptions")
    args = parser.parse_args()

    data, json_path = generate_background_markdown(
        args.folder_path,
        output_md=args.output_md,
        output_json=args.output_json,
        image_width=args.image_width,
        overwrite_existing=args.overwrite,
        wrap_descriptions=not args.no_wrap
    )
