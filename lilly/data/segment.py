"""Inference-time text segmentation utility.

Splits text into word-boundary segments for autoregressive generation.
"""

from __future__ import annotations

import random
from typing import List


def split_text_into_inference_segments(
    text: str,
    min_words: int = 2,
    max_words: int = 4,
    seed: int | None = None,
) -> List[str]:
    """Split text into segments for inference at word boundaries."""
    if not text:
        return []

    rng = random.Random(seed)
    space_positions = [i for i, ch in enumerate(text) if ch == " "]

    if not space_positions:
        return [text]

    segments: List[str] = []
    start = 0

    while start < len(text):
        n_words = rng.randint(min_words, max_words)
        spaces_after = [p for p in space_positions if p >= start]

        if len(spaces_after) < n_words:
            segments.append(text[start:])
            break

        split_at = spaces_after[n_words - 1]
        segments.append(text[start:split_at])
        start = split_at

    return segments
