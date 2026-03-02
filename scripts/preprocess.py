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
from lilly.data.preprocess import (
    _auto_detect_workers,
    find_keystroke_files,
    process_file,
    write_chunk_parquet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess 136M keystrokes dataset")
    parser.add_argument("--data-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of workers (0 = auto-detect: 75%% of CPU cores)")
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()

    if args.workers <= 0:
        args.workers = _auto_detect_workers()

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

    # Step 2: Process chunks
    label = "Processing chunks"
    ui.begin(label)

    total_sessions = 0
    total_keystrokes = 0
    chunk_idx = 0
    files_done = 0

    for chunk_start in range(0, len(files), args.chunk_size):
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
                ui.update(f"chunk {chunk_idx} | file {files_done}/{len(files)}")

        if chunk_dfs:
            combined = pd.concat(chunk_dfs, ignore_index=True)
            out_path = args.output_dir / f"keystrokes_chunk_{chunk_idx:04d}.parquet"
            write_chunk_parquet(combined, out_path)

            n_sessions = combined["session_id"].nunique()
            n_keystrokes = len(combined)
            total_sessions += n_sessions
            total_keystrokes += n_keystrokes

        chunk_idx += 1

    ui.done(label, f"{chunk_idx} chunks written")

    # Step 3: Summary
    label = "Summary"
    ui.begin(label)
    ui.done(
        label,
        f"{total_sessions:,} sessions, {total_keystrokes:,} keystrokes",
    )

    animator.stop()
    ui.finish()

    print()
    print(f"  {t.GREEN}Preprocessing complete.{t.RESET}")
    print(f"  Output: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
