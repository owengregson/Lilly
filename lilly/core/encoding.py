"""Unified character encoding and WPM bucket mapping.

Provides consistent character-to-ID and ID-to-character conversion
used across both V1 and V2 model versions.
"""

from __future__ import annotations

from lilly.core.config import (
    ASCII_OFFSET,
    BACKSPACE_TOKEN,
    END_TOKEN,
    PAD_TOKEN,
    START_TOKEN,
    WPM_BUCKET_EDGES,
)


def char_to_id(ch: str) -> int:
    """Encode a character as an integer ID.

    0=PAD, 1-95=printable ASCII (space..tilde), 96=BACKSPACE.
    Unknown/non-printable characters map to PAD.
    """
    if not ch or ch == "":
        return PAD_TOKEN
    if ch == "BKSP":
        return BACKSPACE_TOKEN
    if len(ch) > 1:
        return PAD_TOKEN
    code = ord(ch)
    if 32 <= code <= 126:
        return code - ASCII_OFFSET  # 1..95
    return PAD_TOKEN


def id_to_char(cid: int) -> str:
    """Decode an integer ID back to a character string.

    Handles V2 special tokens (END, START) as well as V1 tokens.
    """
    if cid == PAD_TOKEN:
        return ""
    if cid == BACKSPACE_TOKEN:
        return "BKSP"
    if cid == END_TOKEN:
        return "<END>"
    if cid == START_TOKEN:
        return "<START>"
    if 1 <= cid <= 95:
        return chr(cid + ASCII_OFFSET)
    return ""


def wpm_to_bucket(wpm: float) -> int:
    """Map WPM to a bucket index (0 to NUM_WPM_BUCKETS-1)."""
    for i in range(len(WPM_BUCKET_EDGES) - 1):
        if wpm < WPM_BUCKET_EDGES[i + 1]:
            return i
    return len(WPM_BUCKET_EDGES) - 2
