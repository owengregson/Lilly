#!/usr/bin/env python3
"""Download the Aalto 136M Keystrokes dataset."""
import argparse
import sys

from lilly.cli.ui import BannerAnimator, ProgressUI, print_banner, t
from lilly.core.config import DATASET_URL, DATASET_ZIP, RAW_DIR
from lilly.data.download import download, expected_keystroke_count, extract, verify


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Aalto 136M Keystrokes dataset")
    parser.add_argument("--url", type=str, default=DATASET_URL)
    parser.add_argument("--dest", type=str, default=str(DATASET_ZIP))
    parser.add_argument("--data-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--redownload", action="store_true",
                        help="Re-extract even if files already exist")
    args = parser.parse_args()

    from pathlib import Path

    dest = Path(args.dest)
    data_dir = Path(args.data_dir)

    print_banner()

    animator = BannerAnimator()
    ui = ProgressUI(4, animator)
    animator.start()

    # Step 1: Check dataset status
    label = "Checking dataset status"
    ui.begin(label)
    existing_count = verify(data_dir) if data_dir.exists() else 0
    if dest.exists():
        size_gb = dest.stat().st_size / 1e9
        ui.done(label, f"zip exists ({size_gb:.2f} GB), {existing_count} files extracted")
    else:
        ui.done(label, "not yet downloaded")

    # Step 2: Download
    label = "Downloading dataset"
    ui.begin(label)

    def dl_progress(downloaded: int, total: int) -> None:
        if total > 0:
            pct = downloaded * 100 // total
            mb = downloaded / 1e6
            ui.update(f"{mb:.0f} MB ({pct}%)")

    result = download(url=args.url, dest=dest, progress_callback=dl_progress)
    if result["status"] == "skipped":
        size_gb = result["size_bytes"] / 1e9
        ui.done(label, f"already downloaded ({size_gb:.2f} GB)")
    else:
        size_gb = result["size_bytes"] / 1e9
        ui.done(label, f"{size_gb:.2f} GB")

    # Step 3: Extract (skip if all expected files are already present)
    label = "Extracting archive"
    expected = expected_keystroke_count(dest)
    extraction_complete = (
        existing_count > 0
        and expected > 0
        and existing_count >= expected
        and not args.redownload
    )

    if extraction_complete:
        ui.skip(label, f"all {existing_count} files already extracted")
    else:
        if existing_count > 0 and expected > 0:
            detail = f"incomplete ({existing_count}/{expected} files)"
        else:
            detail = ""
        ui.begin(label)
        if detail:
            ui.update(detail)

        def ext_progress(extracted: int, total: int) -> None:
            ui.update(f"{extracted}/{total} files")

        extract(zip_path=dest, dest_dir=data_dir, progress_callback=ext_progress)
        ui.done(label, str(data_dir))

    # Step 4: Verify
    label = "Verifying files"
    ui.begin(label)
    count = verify(data_dir)
    if count == 0:
        ui.fail(label, "No keystroke files found after extraction!")
        animator.stop()
        sys.exit(1)
    if expected > 0 and count < expected:
        ui.warn(label, f"only {count}/{expected} keystroke files (some may be missing)")
    else:
        ui.done(label, f"{count} keystroke files")

    animator.stop()
    ui.finish()

    print()
    print(f"  {t.GREEN}Done.{t.RESET} {count} participant files ready for preprocessing.")
    print()


if __name__ == "__main__":
    main()
