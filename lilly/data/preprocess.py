"""Parse raw keystroke files into structured typing sessions.

Reads the Aalto 136M Keystrokes dataset (tab-separated text files) and
produces Parquet files with aligned keystroke sequences ready for feature
extraction.

Uses C-backed pandas/numpy/pyarrow operations for ~10-20x speedup over
the previous pure-Python implementation.

Pure library module — no prints, no CLI. Scripts own the UI layer.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lilly.core.config import (
    ACTION_BACKSPACE,
    ACTION_CORRECT,
    ACTION_ERROR,
    BACKSPACE_KEY,
    MAX_IKI_MS,
    MAX_KEYSTROKES_PER_SESSION,
    MIN_KEYSTROKES_PER_SESSION,
    MODIFIER_KEYS,
)
from lilly.core.encoding import wpm_to_bucket

# Column names for the raw TSV files (Aalto 136M Keystrokes format)
_TSV_COLUMNS = [
    "uid", "session_id", "sentence", "section",
    "keystroke_id", "press_time", "release_time", "letter",
]
_TSV_DTYPES = {
    "uid": str,
    "session_id": str,
    "sentence": str,
    "section": str,
    "keystroke_id": "int64",
    "press_time": "float64",
    "release_time": "float64",
    "letter": str,
}

# Modifier keys as a numpy-friendly set for np.isin filtering
_MODIFIER_ARRAY = np.array(sorted(MODIFIER_KEYS), dtype=object)

# Explicit pyarrow schema for output Parquet files
PARQUET_SCHEMA = pa.schema([
    ("session_id", pa.string()),
    ("uid", pa.string()),
    ("target_sentence", pa.string()),
    ("keystroke_idx", pa.int32()),
    ("typed_key", pa.string()),
    ("target_char", pa.string()),
    ("action", pa.int8()),
    ("press_time", pa.float64()),
    ("release_time", pa.float64()),
    ("iki", pa.float32()),
    ("hold_time", pa.float32()),
    ("target_pos", pa.int16()),
    ("buffer_len", pa.int16()),
    ("wpm", pa.float32()),
    ("wpm_bucket", pa.int8()),
    ("session_total_chars", pa.int32()),
    ("session_total_errors", pa.int32()),
    ("session_total_backspaces", pa.int32()),
])


def parse_keystroke_file(filepath: Path) -> pd.DataFrame:
    """Parse a single participant's keystroke file using the pandas C engine.

    Returns a DataFrame with columns matching _TSV_COLUMNS, sorted by
    (uid, session_id, sentence, press_time).
    """
    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            header=0,
            names=_TSV_COLUMNS,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7],
            dtype=_TSV_DTYPES,
            engine="c",
            on_bad_lines="skip",
            na_filter=False,
            encoding="utf-8",
            encoding_errors="replace",
        )
    except Exception:
        return pd.DataFrame(columns=_TSV_COLUMNS)

    if df.empty:
        return df

    # Drop the unused 'section' column
    df.drop(columns=["section"], inplace=True)

    # Sort by session grouping + chronological order (C-backed sort)
    df.sort_values(
        ["uid", "session_id", "sentence", "press_time"],
        inplace=True,
        kind="mergesort",
    )
    df.reset_index(drop=True, inplace=True)

    return df


def replay_session_arrays(
    target: str,
    letters: np.ndarray,
    press_times: np.ndarray,
    release_times: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    """Replay a keystroke sequence against a target sentence using numpy arrays.

    Tracks a buffer length counter and classifies each keystroke as correct,
    error, or backspace. Skips modifier keys.

    Returns a dict of parallel arrays, or None if too few keystrokes survive.
    """
    if not target or len(letters) == 0:
        return None

    n = len(letters)

    # --- Filter modifier keys using np.isin (C-backed) ---
    keep_mask = ~np.isin(letters, _MODIFIER_ARRAY)
    if not np.any(keep_mask):
        return None

    letters = letters[keep_mask]
    press_times = press_times[keep_mask]
    release_times = release_times[keep_mask]
    n = len(letters)

    if n < MIN_KEYSTROKES_PER_SESSION:
        return None

    # --- Pre-allocate output arrays ---
    action = np.empty(n, dtype=np.int8)
    target_pos = np.empty(n, dtype=np.int16)
    buffer_len = np.empty(n, dtype=np.int16)
    typed_key = np.empty(n, dtype=object)
    target_char = np.empty(n, dtype=object)

    # --- Sequential buffer state machine (tight scalar loop) ---
    buf_len = 0
    target_len = len(target)

    for i in range(n):
        letter = letters[i]

        if letter == BACKSPACE_KEY:
            target_pos[i] = max(0, buf_len - 1)
            if buf_len > 0:
                buf_len -= 1
            action[i] = ACTION_BACKSPACE
            typed_key[i] = "BKSP"
            target_char[i] = ""
        else:
            pos = buf_len
            target_pos[i] = pos
            if pos < target_len:
                expected = target[pos]
                is_correct = letter == expected
            else:
                expected = ""
                is_correct = False

            action[i] = ACTION_CORRECT if is_correct else ACTION_ERROR
            typed_key[i] = letter
            target_char[i] = expected
            buf_len += 1

        buffer_len[i] = buf_len

    # --- Vectorize IKI computation (np.diff + np.clip — C-backed) ---
    iki = np.empty(n, dtype=np.float32)
    iki[0] = 0.0
    if n > 1:
        iki[1:] = np.diff(press_times).astype(np.float32)
    np.clip(iki, 0.0, MAX_IKI_MS, out=iki)

    # --- Vectorize hold time (C-backed) ---
    hold_time = (release_times - press_times).astype(np.float32)
    np.clip(hold_time, 0.0, 2000.0, out=hold_time)

    return {
        "typed_key": typed_key,
        "target_char": target_char,
        "action": action,
        "press_time": press_times,
        "release_time": release_times,
        "iki": iki,
        "hold_time": hold_time,
        "target_pos": target_pos,
        "buffer_len": buffer_len,
    }


def compute_wpm_vectorized(actions: np.ndarray, press_times: np.ndarray) -> float:
    """Compute words-per-minute using vectorized numpy operations."""
    chars_typed = int(np.sum(actions != ACTION_BACKSPACE))
    if chars_typed < 2:
        return 0.0

    duration_min = (press_times[-1] - press_times[0]) / 60000.0
    if duration_min <= 0:
        return 0.0

    return (chars_typed / 5.0) / duration_min


def process_file(filepath: Path) -> pd.DataFrame:
    """Process a single participant file into a DataFrame of typed keystrokes.

    Uses groupby instead of manual dict-of-lists, and builds DataFrames
    directly from dict-of-arrays.
    """
    raw_df = parse_keystroke_file(filepath)
    if raw_df.empty:
        return pd.DataFrame()

    session_dfs: List[pd.DataFrame] = []

    for (uid, sid, sentence), group_df in raw_df.groupby(
        ["uid", "session_id", "sentence"], sort=False
    ):
        if len(group_df) > MAX_KEYSTROKES_PER_SESSION:
            continue

        letters = group_df["letter"].values
        press_times = group_df["press_time"].values
        release_times = group_df["release_time"].values

        result = replay_session_arrays(
            sentence, letters, press_times, release_times
        )
        if result is None:
            continue

        # Compute WPM (vectorized)
        wpm = compute_wpm_vectorized(result["action"], result["press_time"])
        if wpm <= 0 or wpm > 300:
            continue

        n = len(result["action"])
        total_errors = int(np.sum(result["action"] == ACTION_ERROR))
        total_backspaces = int(np.sum(result["action"] == ACTION_BACKSPACE))

        # Session hash for a stable session ID
        session_hash = hashlib.md5(
            f"{uid}:{sid}:{sentence}".encode()
        ).hexdigest()[:12]

        # Build DataFrame directly from dict-of-arrays (no list-of-dicts)
        session_df = pd.DataFrame({
            "session_id": session_hash,
            "uid": uid,
            "target_sentence": sentence,
            "keystroke_idx": np.arange(n, dtype=np.int32),
            "typed_key": result["typed_key"],
            "target_char": result["target_char"],
            "action": result["action"],
            "press_time": result["press_time"],
            "release_time": result["release_time"],
            "iki": result["iki"],
            "hold_time": result["hold_time"],
            "target_pos": result["target_pos"],
            "buffer_len": result["buffer_len"],
            "wpm": np.float32(wpm),
            "wpm_bucket": np.int8(wpm_to_bucket(wpm)),
            "session_total_chars": np.int32(n),
            "session_total_errors": np.int32(total_errors),
            "session_total_backspaces": np.int32(total_backspaces),
        })
        session_dfs.append(session_df)

    if not session_dfs:
        return pd.DataFrame()

    return pd.concat(session_dfs, ignore_index=True)


def write_chunk_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """Write a DataFrame to Parquet with explicit schema and dictionary encoding."""
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    pq.write_table(
        table,
        out_path,
        compression="snappy",
        write_statistics=False,
        use_dictionary=["session_id", "uid", "target_sentence"],
    )


def find_keystroke_files(data_dir: Path) -> List[Path]:
    """Find all *_keystrokes.txt files recursively."""
    files: List[Path] = []
    for p in data_dir.rglob("*_keystrokes.txt"):
        files.append(p)

    if not files:
        for subdir in data_dir.iterdir():
            if subdir.is_dir():
                for p in subdir.rglob("*_keystrokes.txt"):
                    files.append(p)
                if files:
                    break

    files.sort()
    return files


def _auto_detect_workers() -> int:
    """Auto-detect worker count: 75% of CPU cores, clamped to [1, 128]."""
    cpu_count = os.cpu_count() or 4
    return max(1, min(int(cpu_count * 0.75), 128))


