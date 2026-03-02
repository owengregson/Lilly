#!/usr/bin/env python3
"""Prepare V3 training segments from preprocessed Parquet data."""
import argparse
import sys
from pathlib import Path

from lilly.cli.ui import BannerAnimator, ProgressUI, print_banner, t
from lilly.core.config import PROCESSED_DIR, V3_SEGMENT_DIR
from lilly.data.segment_v3 import process_chunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V3 training segments")
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=V3_SEGMENT_DIR)
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    print_banner()

    animator = BannerAnimator()
    ui = ProgressUI(3, animator)
    animator.start()

    # Step 1: Scan Parquet files
    label = "Scanning Parquet files"
    ui.begin(label)
    parquet_files = sorted(args.input_dir.glob("*.parquet"))
    if args.max_files > 0:
        parquet_files = parquet_files[:args.max_files]
    if not parquet_files:
        ui.fail(label, f"No Parquet files found in {args.input_dir}")
        animator.stop()
        sys.exit(1)
    ui.done(label, f"{len(parquet_files)} files")

    # Step 2: Extract V3 segments
    label = "Extracting V3 segments"
    ui.begin(label)
    total_segments = 0
    for i, pf in enumerate(parquet_files, 1):
        n = process_chunk(pf, args.output_dir)
        total_segments += n
        ui.update(f"{i}/{len(parquet_files)} chunks | {total_segments:,} segments")
    ui.done(label, f"{total_segments:,} segments")

    # Step 3: Summary
    label = "Summary"
    ui.begin(label)
    ui.done(label, f"{total_segments:,} segments in {args.output_dir}")

    animator.stop()
    ui.finish()

    print()
    print(f"  {t.GREEN}Segmentation complete.{t.RESET}")
    print(f"  Output: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
