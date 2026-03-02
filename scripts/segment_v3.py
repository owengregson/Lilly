#!/usr/bin/env python3
"""Prepare V3 training segments from preprocessed Parquet data."""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from lilly.cli.ui import BannerAnimator, ProgressUI, print_banner, t
from lilly.core.config import PROCESSED_DIR, V3_SEGMENT_DIR
from lilly.core.hardware import detect_workers
from lilly.data.segment_v3 import process_chunk, validate_segment_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V3 training segments")
    parser.add_argument("--input-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=V3_SEGMENT_DIR)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of workers (0 = auto-detect)")
    parser.add_argument("--reprocess", action="store_true",
                        help="Reprocess all files (ignore existing output)")
    args = parser.parse_args()

    if args.workers <= 0:
        args.workers = detect_workers()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_banner()

    animator = BannerAnimator()
    ui = ProgressUI(3, animator)
    animator.start()

    # Step 1: Scan Parquet files and validate existing output
    label = "Scanning Parquet files"
    ui.begin(label)
    parquet_files = sorted(args.input_dir.glob("*.parquet"))
    if args.max_files > 0:
        parquet_files = parquet_files[:args.max_files]
    if not parquet_files:
        ui.fail(label, f"No Parquet files found in {args.input_dir}")
        animator.stop()
        sys.exit(1)

    remaining = []
    skipped = 0
    corrupt = 0
    for pf in parquet_files:
        expected = args.output_dir / f"segments_{pf.stem}.npz"
        if expected.exists() and not args.reprocess:
            if validate_segment_npz(expected):
                skipped += 1
            else:
                # Corrupt / incomplete — remove and reprocess
                expected.unlink()
                remaining.append(pf)
                corrupt += 1
        else:
            remaining.append(pf)

    detail = f"{len(parquet_files)} files"
    if skipped:
        detail += f" ({skipped} done"
        if corrupt:
            detail += f", {corrupt} corrupt/replaced"
        detail += f", {len(remaining)} remaining)"
    elif corrupt:
        detail += f" ({corrupt} corrupt/replaced)"
    ui.done(label, detail)

    # Step 2: Extract V3 segments (parallel)
    label = "Extracting V3 segments"
    if not remaining:
        ui.skip(label, "all files already processed")
    else:
        ui.begin(label)
        total_segments = 0
        files_done = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_chunk, pf, args.output_dir): pf
                for pf in remaining
            }
            for future in as_completed(futures):
                try:
                    n = future.result()
                    total_segments += n
                except Exception:
                    pass
                files_done += 1
                ui.update(
                    f"{files_done}/{len(remaining)} chunks | "
                    f"{total_segments:,} segments"
                )

        ui.done(label, f"{total_segments:,} new segments")

    # Step 3: Summary
    label = "Summary"
    ui.begin(label)
    all_npz = list(args.output_dir.glob("segments_*.npz"))
    ui.done(label, f"{len(all_npz)} segment files in {args.output_dir}")

    animator.stop()
    ui.finish()

    print()
    print(f"  {t.GREEN}Segmentation complete.{t.RESET}")
    print(f"  Output: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
