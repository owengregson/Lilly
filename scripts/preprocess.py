#!/usr/bin/env python3
"""Preprocess raw keystroke data into Parquet chunks."""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import pandas as pd

from lilly.cli.ui import BannerAnimator, ProgressUI, print_banner, t
from lilly.core.config import PROCESSED_DIR, RAW_DIR
from lilly.core.hardware import detect_workers
from lilly.data.preprocess import (
    find_keystroke_files,
    process_file,
    validate_parquet,
    write_chunk_parquet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess 136M keystrokes dataset")
    parser.add_argument("--data-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of workers (0 = auto-detect)")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--reprocess", action="store_true",
                        help="Reprocess all chunks (ignore existing output)")
    args = parser.parse_args()

    if args.workers <= 0:
        args.workers = detect_workers()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_banner()

    animator = BannerAnimator()
    ui = ProgressUI(3, animator)
    animator.start()

    # Step 1: Scan for keystroke files
    label = "Scanning for keystroke files"
    ui.begin(label)
    files = find_keystroke_files(args.data_dir)
    if not files:
        ui.fail(label, f"No keystroke files found in {args.data_dir}")
        animator.stop()
        sys.exit(1)
    if args.max_files > 0:
        files = files[:args.max_files]
    ui.done(label, f"{len(files)} files, {args.workers} workers")

    # Step 2: Process chunks (with resume + validation)
    label = "Processing chunks"
    ui.begin(label)

    total_chunks = (len(files) + args.chunk_size - 1) // args.chunk_size
    chunks_written = 0
    chunks_skipped = 0
    chunks_replaced = 0
    files_done = 0

    for chunk_idx, chunk_start in enumerate(range(0, len(files), args.chunk_size)):
        out_path = args.output_dir / f"keystrokes_chunk_{chunk_idx:04d}.parquet"
        chunk_file_count = min(args.chunk_size, len(files) - chunk_start)

        # Resume: skip chunks whose output already exists and is valid
        if out_path.exists() and not args.reprocess:
            if validate_parquet(out_path):
                files_done += chunk_file_count
                chunks_skipped += 1
                ui.update(f"chunk {chunk_idx} verified, skipping")
                continue
            else:
                # Corrupt / incomplete — remove and reprocess
                out_path.unlink()
                chunks_replaced += 1

        chunk_files = files[chunk_start:chunk_start + args.chunk_size]
        chunk_dfs: List[pd.DataFrame] = []

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_file, f): f for f in chunk_files}
            for future in as_completed(futures):
                try:
                    df = future.result()
                    if not df.empty:
                        chunk_dfs.append(df)
                except Exception:
                    pass
                files_done += 1
                ui.update(f"chunk {chunk_idx}/{total_chunks} | file {files_done}/{len(files)}")

        if chunk_dfs:
            combined = pd.concat(chunk_dfs, ignore_index=True)
            write_chunk_parquet(combined, out_path)

        chunks_written += 1

    detail = f"{chunks_written} chunks written"
    if chunks_skipped:
        detail += f", {chunks_skipped} skipped (verified)"
    if chunks_replaced:
        detail += f", {chunks_replaced} replaced (were corrupt)"
    ui.done(label, detail)

    # Step 3: Summary
    label = "Summary"
    ui.begin(label)
    all_parquet = list(args.output_dir.glob("keystrokes_chunk_*.parquet"))
    ui.done(label, f"{len(all_parquet)} total chunks in {args.output_dir}")

    animator.stop()
    ui.finish()

    print()
    print(f"  {t.GREEN}Preprocessing complete.{t.RESET}")
    print(f"  Output: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
