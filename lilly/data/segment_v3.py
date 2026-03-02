"""V3 segmentation pipeline: Parquet -> .npz segment chunks.

Reads preprocessed Parquet chunks, groups by session, computes style vectors,
splits at pause boundaries, and saves V3 segment dicts as .npz files.

Pure library module — no prints, no CLI. Scripts own the UI layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from lilly.core.config import (
    MAX_SEGMENT_KEYSTROKES,
    MAX_TARGET_CHARS,
    MIN_SEGMENT_KEYSTROKES,
    PAUSE_THRESHOLD_MS,
)
from lilly.core.encoding import char_to_id
from lilly.data.style import compute_style_vector


def extract_v3_segments(
    session_df: pd.DataFrame,
    style_vector: np.ndarray,
    target_sentence: str,
) -> List[dict]:
    """Extract V3 segments from a single session.

    Args:
        session_df: DataFrame with keystroke rows for one session.
        style_vector: (16,) float32 style vector for this session.
        target_sentence: The target sentence text.

    Returns:
        List of segment dicts ready for .npz storage.
    """
    keystrokes = session_df.to_dict("records")
    if not keystrokes:
        return []

    # Split at pause boundaries
    groups: List[List[dict]] = []
    current: List[dict] = [keystrokes[0]]

    for ks in keystrokes[1:]:
        if ks["iki"] >= PAUSE_THRESHOLD_MS:
            groups.append(current)
            current = [ks]
        else:
            current.append(ks)
    if current:
        groups.append(current)

    segments = []
    prev_context_chars = [0] * 4
    prev_context_actions = [0] * 4
    prev_context_delays = [0.0] * 4

    for group in groups:
        # Split long groups
        sub_groups = _split_long_group(group)
        for sub_group in sub_groups:
            if len(sub_group) < MIN_SEGMENT_KEYSTROKES:
                # Still update context
                if sub_group:
                    tail = sub_group[-4:]
                    prev_context_chars = [char_to_id(ks["typed_key"]) for ks in tail]
                    prev_context_actions = [int(ks["action"]) for ks in tail]
                    prev_context_delays = [float(ks["iki"]) for ks in tail]
                continue

            seg = _build_segment_dict(
                sub_group, style_vector, target_sentence,
                prev_context_chars, prev_context_actions, prev_context_delays,
            )
            if seg is not None:
                segments.append(seg)

            # Update context from tail of this group
            tail = sub_group[-4:]
            prev_context_chars = [char_to_id(ks["typed_key"]) for ks in tail]
            prev_context_actions = [int(ks["action"]) for ks in tail]
            prev_context_delays = [float(ks["iki"]) for ks in tail]

    return segments


def _build_segment_dict(
    keystrokes: List[dict],
    style_vector: np.ndarray,
    target_sentence: str,
    prev_chars: list,
    prev_actions: list,
    prev_delays: list,
) -> dict | None:
    """Build a single V3 segment dict from a group of keystrokes."""
    # Compute target text for this segment
    target_text = _compute_target_text(keystrokes, target_sentence)
    if not target_text or len(target_text) > MAX_TARGET_CHARS:
        return None

    n = len(keystrokes)
    # Encoder chars: target text char IDs
    enc_chars = np.zeros(MAX_TARGET_CHARS, dtype=np.int32)
    for i, ch in enumerate(target_text[:MAX_TARGET_CHARS]):
        enc_chars[i] = char_to_id(ch)
    enc_len = min(len(target_text), MAX_TARGET_CHARS)

    # Decoder data: keystroke char IDs, delays, and actions
    dec_chars = np.zeros(MAX_SEGMENT_KEYSTROKES, dtype=np.int32)
    dec_delays = np.zeros(MAX_SEGMENT_KEYSTROKES, dtype=np.float32)
    dec_actions = np.zeros(MAX_SEGMENT_KEYSTROKES, dtype=np.int32)

    for i, ks in enumerate(keystrokes[:MAX_SEGMENT_KEYSTROKES]):
        dec_chars[i] = char_to_id(ks["typed_key"])
        dec_delays[i] = float(ks["iki"])
        dec_actions[i] = int(ks["action"])

    dec_len = min(n, MAX_SEGMENT_KEYSTROKES)

    # Context arrays (padded to 4)
    ctx_chars = np.zeros(4, dtype=np.int32)
    ctx_actions = np.zeros(4, dtype=np.int32)
    ctx_delays = np.zeros(4, dtype=np.float32)
    for i, v in enumerate(prev_chars[-4:]):
        ctx_chars[4 - len(prev_chars[-4:]) + i] = v
    for i, v in enumerate(prev_actions[-4:]):
        ctx_actions[4 - len(prev_actions[-4:]) + i] = v
    for i, v in enumerate(prev_delays[-4:]):
        ctx_delays[4 - len(prev_delays[-4:]) + i] = v

    return {
        "encoder_chars": enc_chars,
        "encoder_lengths": np.int32(enc_len),
        "decoder_chars": dec_chars,
        "decoder_delays": dec_delays,
        "decoder_actions": dec_actions,
        "decoder_lengths": np.int32(dec_len),
        "style_vector": style_vector.copy(),
        "prev_context_chars": ctx_chars,
        "prev_context_actions": ctx_actions,
        "prev_context_delays": ctx_delays,
    }


def _compute_target_text(keystrokes: List[dict], sentence: str) -> str:
    """Determine what target text a group of keystrokes covers."""
    min_pos = None
    max_pos = None
    for ks in keystrokes:
        pos = ks.get("target_pos", 0)
        if ks["action"] == 0:  # CORRECT
            if min_pos is None or pos < min_pos:
                min_pos = pos
            if max_pos is None or pos > max_pos:
                max_pos = pos
    if min_pos is None:
        return ""
    end = min(max_pos + 1, len(sentence))
    return sentence[min_pos:end]


def _split_long_group(group: List[dict]) -> List[List[dict]]:
    """Split a keystroke group that exceeds MAX_SEGMENT_KEYSTROKES."""
    if len(group) <= MAX_SEGMENT_KEYSTROKES:
        return [group]

    result: List[List[dict]] = []
    start = 0
    while start < len(group):
        end = min(start + MAX_SEGMENT_KEYSTROKES, len(group))
        if end < len(group):
            best = end
            for i in range(end - 1, max(start + MIN_SEGMENT_KEYSTROKES, end - 15) - 1, -1):
                if group[i]["typed_key"] == " ":
                    best = i + 1
                    break
            end = best
        result.append(group[start:end])
        start = end
    return result


def process_chunk(parquet_path: Path, output_dir: Path) -> int:
    """Process a single Parquet chunk into V3 segments.

    Returns the number of segments extracted.
    """
    df = pd.read_parquet(parquet_path)

    if "session_id" not in df.columns:
        return 0

    all_segments = []

    for session_id, session_df in df.groupby("session_id"):
        session_df = session_df.sort_values("keystroke_idx").reset_index(drop=True)
        style_vec = compute_style_vector(session_df)

        target = (
            session_df["target_sentence"].iloc[0]
            if "target_sentence" in session_df.columns
            else ""
        )
        segments = extract_v3_segments(session_df, style_vec, target)
        all_segments.extend(segments)

    if not all_segments:
        return 0

    # Stack into arrays
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_name = parquet_path.stem
    out_path = output_dir / f"segments_{chunk_name}.npz"

    arrays = {}
    for key in all_segments[0]:
        arrays[key] = np.stack([s[key] for s in all_segments])

    np.savez(out_path, **arrays)
    return len(all_segments)


