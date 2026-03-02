"""Download and extract the Aalto 136M Keystrokes dataset.

Pure library module — no prints, no CLI. Scripts own the UI layer.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable, Optional

import requests

from lilly.core.config import DATASET_URL, DATASET_ZIP, RAW_DIR


def download(
    url: str = DATASET_URL,
    dest: Path = DATASET_ZIP,
    chunk_size: int = 8192,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Download a file, optionally reporting progress.

    Args:
        url: URL to download.
        dest: Local file path to save to.
        chunk_size: Bytes per read chunk.
        progress_callback: Called with (bytes_downloaded, total_bytes).

    Returns:
        {"status": "downloaded"|"skipped", "size_bytes": int}
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        local_size = dest.stat().st_size
        head = requests.head(url, timeout=30)
        remote_size = int(head.headers.get("Content-Length", 0))
        if local_size == remote_size and remote_size > 0:
            return {"status": "skipped", "size_bytes": local_size}

    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))

    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback:
                progress_callback(downloaded, total)

    return {"status": "downloaded", "size_bytes": dest.stat().st_size}


def extract(
    zip_path: Path = DATASET_ZIP,
    dest_dir: Path = RAW_DIR,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Extract a zip file, optionally reporting progress.

    Args:
        zip_path: Path to the zip file.
        dest_dir: Directory to extract into.
        progress_callback: Called with (files_extracted, total_files).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, dest_dir)
            if progress_callback:
                progress_callback(i, total)


def verify(data_dir: Path = RAW_DIR) -> int:
    """Count extracted keystroke files.

    Returns:
        Number of *_keystrokes.txt files found.
    """
    return len(list(data_dir.rglob("*_keystrokes.txt")))
