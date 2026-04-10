#!/usr/bin/env python3
"""Convert RoboLab output to LeRobot v3.0 format.

Usage:
    python -m robolab.core.export.convert_to_lerobot --input output/my_experiment --output output/my_experiment_lerobot

    # Or from robolab directory:
    python scripts/convert_to_lerobot.py --input output/video_test --output output/video_test_lerobot
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert RoboLab output to LeRobot v3.0 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert a single experiment
    python scripts/convert_to_lerobot.py --input output/video_test

    # Specify custom output directory
    python scripts/convert_to_lerobot.py --input output/video_test --output /path/to/lerobot_dataset

    # Specify robot type and FPS
    python scripts/convert_to_lerobot.py --input output/video_test --robot-type droid --fps 30
        """,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to RoboLab output directory (contains task folders with data.hdf5)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path for LeRobot output directory (default: <input>/lerobot)"
    )
    parser.add_argument(
        "--robot-type",
        default="franka",
        help="Robot type for dataset metadata (default: franka)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Frames per second (default: 15.0)"
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face repository ID (optional)"
    )

    args = parser.parse_args()

    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        sys.exit(1)

    # Import here to avoid slow imports when just showing help
    from robolab.core.export.lerobot_exporter import export_to_lerobot

    try:
        output_path = export_to_lerobot(
            robolab_output_dir=str(input_path),
            lerobot_output_dir=args.output,
            robot_type=args.robot_type,
            fps=args.fps,
        )
        print(f"\nSuccess! LeRobot dataset created at: {output_path}")
        print(f"\nTo visualize, run the lerobot-dataset-visualizer and open:")
        print(f"  file://{output_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
