# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import csv
import os
from datetime import datetime
from typing import Any, Optional, cast

from robolab.core.utils.file_utils import convert_file_path, get_relative_path


def csv_to_markdown_table(
    csv_input: str | list[list[str]] | list[dict[str, Any]],
    headers: Optional[list[str]] = None,
    align: str = "left"
) -> str:
    """
    Convert CSV data to a markdown table format.

    Args:
        csv_input: Can be one of:
            - Path to a CSV file (str)
            - list of lists representing CSV rows
            - list of dictionaries (keys become headers)
        headers: Optional list of column headers. If None and csv_input is a file,
                headers are taken from the first row.
        align: Table alignment ('left', 'center', 'right'). Default is 'left'.

    Returns:
        str: Markdown formatted table

    Examples:
        # From file path
        markdown = csv_to_markdown_table("data.csv")

        # From list of lists
        data = [["Name", "Age"], ["Alice", "25"], ["Bob", "30"]]
        markdown = csv_to_markdown_table(data)

        # From list of dictionaries
        data = [{"Name": "Alice", "Age": "25"}, {"Name": "Bob", "Age": "30"}]
        markdown = csv_to_markdown_table(data)
    """
    # Handle different input types
    if isinstance(csv_input, str):
        # Input is a file path
        if not os.path.exists(csv_input):
            raise FileNotFoundError(f"CSV file not found: {csv_input}")

        with open(csv_input, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return ""

        if headers is None:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            data_rows = rows

    elif isinstance(csv_input, list) and csv_input and isinstance(csv_input[0], dict):
        # Input is list of dictionaries
        dict_data = csv_input  # type: ignore
        if headers is None:
            headers = list(dict_data[0].keys())  # type: ignore
        data_rows = [[str(row.get(header, "")) for header in headers] for row in dict_data]  # type: ignore

    elif isinstance(csv_input, list):
        # Input is list of lists
        if not csv_input:
            return ""

        list_data = csv_input  # type: ignore
        if headers is None:
            headers = list_data[0]  # type: ignore
            data_rows = list_data[1:]  # type: ignore
        else:
            data_rows = list_data  # type: ignore
    else:
        raise ValueError("csv_input must be a file path, list of lists, or list of dictionaries")

    if not headers:
        return ""

    # Create alignment string
    align_chars = {
        "left": ":--",
        "center": ":-:",
        "right": "--:"
    }
    alignment = align_chars.get(align, ":--")

    # Build markdown table
    markdown_lines = []

    # Header row
    header_row = "| " + " | ".join(str(header) for header in headers) + " |"
    markdown_lines.append(header_row)

    # Separator row
    separator_row = "| " + " | ".join(alignment for _ in headers) + " |"
    markdown_lines.append(separator_row)

    # Data rows
    for row in data_rows:
        # Ensure row has same length as headers
        padded_row = [str(cell) if i < len(row) else "" for i, cell in enumerate(row)] + [""] * (len(headers) - len(row))
        padded_row = padded_row[:len(headers)]  # Trim if too long
        data_row = "| " + " | ".join(padded_row) + " |"
        markdown_lines.append(data_row)

    return "\n".join(markdown_lines)


def get_markdown_image_text(
    filename_to_img: str,
    relative_dir: str,
    image_dir: str,
    image_ext: str='.png',
    size: Optional[tuple[int, int]] = None,
    subtitle: Optional[str] = None
) -> str:
    """Helper function to find filename_to_img in image_dir and return a markdown image text, relative to relative_dir.
    Assumes that the image filename is the same as the filename_to_img, but with the extension replaced with image_ext.

    Args:
        filename_to_img: The filename to find in image_dir
        relative_dir: The directory to use as the base for the relative path
        image_dir: The directory to search for the image
        image_ext: The extension of the image file
        size: Optional tuple of (width, height) in pixels for custom image sizing
        subtitle: Optional subtitle text to display below the image (forces HTML img tag)

    Returns:
        str: Markdown image text or HTML img tag if size or subtitle is specified
    """
    # Get extension of current filename
    ext = os.path.splitext(filename_to_img)[1]

    image_filename = filename_to_img.replace(ext, image_ext)
    image_path = os.path.join(image_dir, image_filename)

    if not os.path.exists(image_path):
        return "No image"

    # Calculate relative path from output directory to image
    relative_image_path = get_relative_path(image_path, relative_dir)
    # Normalize path separators for markdown (use forward slashes)
    relative_image_path = relative_image_path.replace(os.sep, '/')

    # Use HTML if subtitle or size is provided, otherwise use markdown
    use_html = subtitle is not None or size is not None

    if use_html:
        # Build HTML img tag
        if size is not None:
            width, height = size
            image_ref = f'<img src="{relative_image_path}" alt="{image_filename}" width="{width}" height="{height}">'
        else:
            image_ref = f'<img src="{relative_image_path}" alt="{image_filename}">'

        # Add subtitle if provided
        if subtitle is not None:
            return f"{image_ref}<br>{subtitle}"
        return image_ref
    else:
        # Use markdown syntax
        return f"![{image_filename}]({relative_image_path})"

def add_images_to_csv(
    csv_file_path: str,
    image_dir: str,
    column_name_to_img: str = "scene",
    image_column_name: str = "Preview",
    relative_dir: str = None,
    size: Optional[tuple[int, int]] = None,
    replace_column: bool = False
) -> list[list[str]]:
    """
    Load a CSV file and append image references for corresponding PNG files.

    Args:
        csv_file_path: Path to the CSV file to load
        image_dir: Directory where PNG images are stored
        column_name_to_img: Name of the column containing USD filenames
        image_column_name: Name to give the new image column (ignored if replace_column=True)
        relative_dir: Directory to use as base for relative paths
        size: Optional tuple of (width, height) in pixels for custom image sizing
        replace_column: If True, replace the content of column_name_to_img with the image
                       and original text as subtitle. If False, add a new column.

    Returns:
        list of rows with image column added or replaced (including header row)
    """

    if relative_dir is None:
        relative_dir = os.path.dirname(os.path.abspath(csv_file_path))

    # Load CSV file
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

    with open(csv_file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []

    headers = rows[0]
    data_rows = rows[1:]

    try:
        scene_col_idx = headers.index(column_name_to_img)
    except ValueError:
        # Scene column not found, return original data
        return rows

    if replace_column:
        # Replace the existing column with image + subtitle
        new_headers = headers
        new_data_rows = []

        for row in data_rows:
            # Pad row to match header length
            padded_row = [str(cell) if i < len(row) else "" for i, cell in enumerate(row)] + [""] * (len(headers) - len(row))
            padded_row = padded_row[:len(headers)]

            scene_filename = padded_row[scene_col_idx]
            # Use get_markdown_image_text with subtitle parameter
            image_ref = get_markdown_image_text(
                scene_filename,
                relative_dir,
                image_dir,
                image_ext='.png',
                size=size,
                subtitle=scene_filename
            )

            # Replace column content with image and subtitle
            new_row = padded_row.copy()
            new_row[scene_col_idx] = image_ref
            new_data_rows.append(new_row)
    else:
        # Add image column to headers
        new_headers = headers + [image_column_name]
        new_data_rows = []

        for row in data_rows:
            # Pad row to match header length
            padded_row = [str(cell) if i < len(row) else "" for i, cell in enumerate(row)] + [""] * (len(headers) - len(row))
            padded_row = padded_row[:len(headers)]

            scene_filename = padded_row[scene_col_idx]
            image_ref = get_markdown_image_text(scene_filename, relative_dir, image_dir, image_ext='.png', size=size)
            new_row = padded_row + [image_ref]
            new_data_rows.append(new_row)

    return [new_headers] + new_data_rows

def save_markdown_table(
    csv_input: str | list[list[str]] | list[dict[str, Any]],
    output_path: str,
    title: str = "",
    description: str = "",
    headers: Optional[list[str]] = None,
    align: str = "left",
    path_type: str = "absolute" # One of "absolute", "relative", or "filename_only"
) -> None:
    """
    Convert CSV data to markdown table and save to file.

    Args:
        csv_input: CSV data (same as csv_to_markdown_table)
        output_path: Path where to save the markdown file
        title: Optional title for the markdown document
        description: Optional description for the markdown document
        headers: Optional list of column headers
        align: Table alignment ('left', 'center', 'right')
        path_type: How to handle file paths - "absolute" (default), "relative", or "filename_only"
    """

    def check_cell_is_html(cell: str) -> bool:
        return '<img' in cell or '<br' in cell or '<div' in cell or '<p' in cell or '<span' in cell or '<a' in cell or '<b' in cell or '<i' in cell or '<u' in cell or '<s' in cell or '<sup' in cell or '<sub' in cell or '<em' in cell or '<strong' in cell or '<code' in cell or '<pre' in cell or '<blockquote' in cell or '<hr' in cell or '<br' in cell or '<div' in cell or '<p' in cell or '<span' in cell or '<a' in cell or '<b' in cell or '<i' in cell or '<u' in cell or '<s' in cell or '<sup' in cell or '<sub' in cell or '<em' in cell or '<strong' in cell or '<code' in cell or '<pre' in cell or '<blockquote' in cell or '<hr' in cell

    # Convert file paths based on path_type if not "absolute"
    if path_type != "absolute":
        # Validate path_type parameter
        if path_type not in ["relative", "filename_only"]:
            raise ValueError(f"Invalid path_type '{path_type}'. Must be 'absolute', 'relative', or 'filename_only'")

        output_dir = os.path.dirname(os.path.abspath(output_path))

        # Process CSV data to convert file paths
        if isinstance(csv_input, str):
            # Load CSV file and convert paths
            if not os.path.exists(csv_input):
                raise FileNotFoundError(f"CSV file not found: {csv_input}")

            with open(csv_input, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            if rows:
                # Convert paths in all data rows (skip header)
                converted_rows = [rows[0]]  # Keep header as-is
                for row in rows[1:]:
                    converted_row = []
                    for cell in row:
                        # Skip HTML content (contains <img or <br tags)
                        if check_cell_is_html(cell):
                            converted_row.append(cell)
                        else:
                            converted_row.append(convert_file_path(cell, path_type, output_dir))
                    converted_rows.append(converted_row)
                csv_input = cast(list[list[str]], converted_rows)

        elif isinstance(csv_input, list) and csv_input and isinstance(csv_input[0], dict):
            # Handle list of dictionaries
            converted_data = []
            for row_dict in csv_input:
                if isinstance(row_dict, dict):
                    converted_dict = {}
                    for key, value in row_dict.items():
                        # Skip HTML content (contains <img or <br tags)
                        cell_str = str(value)
                        if check_cell_is_html(cell_str):
                            converted_dict[key] = cell_str
                        else:
                            converted_dict[key] = convert_file_path(cell_str, path_type, output_dir)
                    converted_data.append(converted_dict)
            csv_input = cast(list[dict[str, Any]], converted_data)

        elif isinstance(csv_input, list):
            # Handle list of lists
            if csv_input and not isinstance(csv_input[0], dict):
                # Convert paths in all data rows (skip header if it exists)
                list_data = csv_input  # Type narrowing
                converted_rows = [list_data[0]]  # Keep first row as-is (assumed header)
                for row in list_data[1:]:
                    if isinstance(row, list):
                        converted_row = []
                        for cell in row:
                            # Skip HTML content (contains <img or <br tags)
                            cell_str = str(cell)
                            if check_cell_is_html(cell_str):
                                converted_row.append(cell_str)
                            else:
                                converted_row.append(convert_file_path(cell_str, path_type, output_dir))
                        converted_rows.append(converted_row)
                csv_input = cast(list[list[str]], converted_rows)

    # Convert CSV data to markdown
    markdown_content = csv_to_markdown_table(csv_input, headers, align)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    default_language = "This table was generated automatically from CSV data. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_content = f"{markdown_content}\n\n{default_language}"

    if description:
        markdown_content = f"{description}\n\n{markdown_content}"
    if title:
        markdown_content = f"# {title}\n\n{markdown_content}"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"Markdown table saved to {output_path}")


def csv_string_to_markdown_table(csv_string: str, align: str = "left") -> str:
    """
    Convert CSV string content to markdown table.

    Args:
        csv_string: CSV content as string
        align: Table alignment ('left', 'center', 'right')

    Returns:
        str: Markdown formatted table
    """
    # Parse CSV string
    lines = csv_string.strip().split('\n')
    reader = csv.reader(lines)
    rows = list(reader)

    return csv_to_markdown_table(rows, align=align)
