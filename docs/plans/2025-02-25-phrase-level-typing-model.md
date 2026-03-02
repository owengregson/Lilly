# Phrase-Level Typing Model (V2) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the V1 LSTM next-keystroke model with a Transformer encoder-decoder that generates complete phrase-level keystroke sequences (with errors, corrections, and timing) in one shot.

**Architecture:** Small Transformer seq2seq. Encoder processes target text + context. Decoder autoregressively generates `(char_id, delay_ms)` pairs for the full keystroke sequence including errors and corrections. Trained on Aalto 136M keystrokes reprocessed into pause-delimited segments.

**Tech Stack:** Python 3.11+, TensorFlow 2.15+, NumPy <2.0, Pandas, PyArrow, TensorFlow.js (export)

**Design Doc:** `docs/plans/2025-02-25-phrase-level-typing-model-design.md`

---

## Task 1: V2 Configuration

**Files:**
- Create: `ml/v2/__init__.py`
- Create: `ml/v2/config.py`

**Step 1: Create the v2 package**

```bash
mkdir -p ml/v2
```

**Step 2: Write empty init**

Create `ml/v2/__init__.py` as an empty file.

**Step 3: Write V2 config**

Create `ml/v2/config.py` with all constants for the V2 pipeline. This mirrors V1's `config.py` but with V2-specific values.

```python
"""Configuration for the V2 phrase-level typing model."""

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent  # ml/
V2_ROOT = Path(__file__).parent              # ml/v2/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "v2_processed"
SEGMENT_DIR = DATA_DIR / "v2_segments"
MODEL_DIR = PROJECT_ROOT / "models" / "v2"
EXPORT_DIR = PROJECT_ROOT / "export" / "v2"

# ---------------------------------------------------------------------------
# Character encoding (same as V1 + START and END tokens)
# ---------------------------------------------------------------------------
PAD_TOKEN = 0
ASCII_OFFSET = 31  # char_id = ord(c) - 31 for printable chars (gives 1..95)
BACKSPACE_TOKEN = 96
END_TOKEN = 97
START_TOKEN = 98
NUM_CHAR_CLASSES = 99  # 0=PAD, 1-95=ASCII, 96=BKSP, 97=END, 98=START

# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
PAUSE_THRESHOLD_MS = 300.0  # IKI gap that defines segment boundary
MIN_SEGMENT_KEYSTROKES = 3
MAX_SEGMENT_KEYSTROKES = 64
MAX_TARGET_CHARS = 32  # max characters in target text per segment

# ---------------------------------------------------------------------------
# Preprocessing (reused from V1)
# ---------------------------------------------------------------------------
MIN_KEYSTROKES_PER_SESSION = 20
MAX_KEYSTROKES_PER_SESSION = 5000
MODIFIER_KEYS = frozenset({"SHIFT", "CAPS_LOCK", "CTRL", "ALT", "TAB"})
BACKSPACE_KEY = "BKSP"

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
MAX_IKI_MS = 5000.0
MIN_IKI_MS = 10.0
MAX_HOLD_MS = 2000.0
MIN_HOLD_MS = 5.0

# ---------------------------------------------------------------------------
# Action labels (kept for preprocessing compatibility)
# ---------------------------------------------------------------------------
ACTION_CORRECT = 0
ACTION_ERROR = 1
ACTION_BACKSPACE = 2

# ---------------------------------------------------------------------------
# WPM persona buckets (same as V1)
# ---------------------------------------------------------------------------
WPM_BUCKET_EDGES = [0, 30, 45, 60, 75, 90, 105, 120, 140, 170, 999]
NUM_WPM_BUCKETS = len(WPM_BUCKET_EDGES) - 1  # 10

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    num_char_classes: int = NUM_CHAR_CLASSES
    num_wpm_buckets: int = NUM_WPM_BUCKETS
    max_encoder_len: int = MAX_TARGET_CHARS
    max_decoder_len: int = MAX_SEGMENT_KEYSTROKES
    char_embed_dim: int = 32
    wpm_embed_dim: int = 16
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    # Context: previous segment's last N keystrokes
    context_tail_len: int = 4

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    lr_warmup_steps: int = 1000
    lr_decay_patience: int = 3
    lr_decay_factor: float = 0.5
    early_stop_patience: int = 7
    val_split: float = 0.1
    test_split: float = 0.05
    timing_loss_weight: float = 1.0
    char_loss_weight: float = 1.0
    shuffle_buffer: int = 100_000
    seed: int = 42
    max_samples: int = 0  # 0 = all
```

**Step 4: Verify the config imports**

Run: `cd /Users/owengregson/Documents/Lilly && python -c "from v2.config import ModelConfig, TrainConfig; print('OK')"`

Expected: `OK`

**Step 5: Commit**

```bash
git add ml/v2/__init__.py ml/v2/config.py
git commit -m "feat(v2): add V2 configuration for phrase-level typing model"
```

---

## Task 2: Segmentation Logic

**Files:**
- Create: `ml/v2/segment.py`
- Create: `ml/v2/test_segment.py`

This module takes a list of aligned keystrokes (from V1's `replay_session`) and splits them into pause-delimited segments. Each segment contains the target text substring and the full keystroke sequence (including errors and corrections).

**Step 1: Write the failing tests**

Create `ml/v2/test_segment.py`:

```python
"""Tests for segment extraction logic."""

import sys
from pathlib import Path

# Allow imports from ml/ root
sys.path.insert(0, str(Path(__file__).parent.parent))

from v2.segment import (
    Segment,
    extract_segments,
    compute_target_text_for_segment,
    char_to_id,
    id_to_char,
    split_text_into_inference_segments,
)
from v2.config import (
    BACKSPACE_TOKEN,
    END_TOKEN,
    START_TOKEN,
    PAD_TOKEN,
    PAUSE_THRESHOLD_MS,
)


# ---------------------------------------------------------------------------
# Helpers: make fake aligned keystrokes (dict-based, matches preprocess output)
# ---------------------------------------------------------------------------

def make_keystroke(typed_key, target_char, action, iki, target_pos, buffer_len):
    """Create a keystroke dict matching the preprocessed parquet schema."""
    return {
        "typed_key": typed_key,
        "target_char": target_char,
        "action": action,
        "iki": iki,
        "hold_time": iki * 0.3,
        "target_pos": target_pos,
        "buffer_len": buffer_len,
    }


# ---------------------------------------------------------------------------
# Character encoding
# ---------------------------------------------------------------------------

def test_char_to_id_printable():
    assert char_to_id("a") == ord("a") - 31
    assert char_to_id(" ") == ord(" ") - 31  # space = 1
    assert char_to_id("~") == ord("~") - 31  # tilde = 95

def test_char_to_id_special():
    assert char_to_id("BKSP") == BACKSPACE_TOKEN  # 96
    assert char_to_id("") == PAD_TOKEN  # 0

def test_id_to_char_roundtrip():
    for ch in "abcXYZ 123!@#":
        assert id_to_char(char_to_id(ch)) == ch
    assert id_to_char(BACKSPACE_TOKEN) == "BKSP"
    assert id_to_char(END_TOKEN) == "<END>"
    assert id_to_char(START_TOKEN) == "<START>"
    assert id_to_char(PAD_TOKEN) == ""


# ---------------------------------------------------------------------------
# Target text extraction from keystroke segment
# ---------------------------------------------------------------------------

def test_compute_target_clean():
    """Clean typing: target = joined typed keys."""
    keystrokes = [
        make_keystroke("h", "h", 0, 80, 0, 1),
        make_keystroke("i", "i", 0, 90, 1, 2),
    ]
    assert compute_target_text_for_segment(keystrokes, "hi there") == "hi"

def test_compute_target_with_error_correction():
    """Error + backspace + retype: target still = 'hi'."""
    keystrokes = [
        make_keystroke("h", "h", 0, 80, 0, 1),
        make_keystroke("o", "i", 1, 70, 1, 2),   # error
        make_keystroke("BKSP", "", 2, 180, 1, 1), # backspace
        make_keystroke("i", "i", 0, 90, 1, 2),    # retype
    ]
    assert compute_target_text_for_segment(keystrokes, "hi there") == "hi"

def test_compute_target_with_space():
    """Target spans a word boundary including space."""
    keystrokes = [
        make_keystroke("h", "h", 0, 80, 0, 1),
        make_keystroke("i", "i", 0, 90, 1, 2),
        make_keystroke(" ", " ", 0, 120, 2, 3),
    ]
    assert compute_target_text_for_segment(keystrokes, "hi there") == "hi "


# ---------------------------------------------------------------------------
# Segment extraction from session
# ---------------------------------------------------------------------------

def test_extract_segments_single_burst():
    """All keystrokes < pause threshold = one segment."""
    keystrokes = [
        make_keystroke("h", "h", 0, 80, 0, 1),
        make_keystroke("i", "i", 0, 90, 1, 2),
    ]
    segments = extract_segments(keystrokes, "hi", wpm_bucket=5, session_len=2)
    assert len(segments) == 1
    assert segments[0].target_text == "hi"
    assert len(segments[0].keystroke_char_ids) == 2
    assert len(segments[0].keystroke_delays) == 2

def test_extract_segments_pause_split():
    """A 400ms gap splits into two segments."""
    keystrokes = [
        make_keystroke("h", "h", 0, 80, 0, 1),
        make_keystroke("i", "i", 0, 90, 1, 2),
        make_keystroke(" ", " ", 0, 400, 2, 3),  # pause > 300ms
        make_keystroke("y", "y", 0, 85, 3, 4),
        make_keystroke("o", "o", 0, 95, 4, 5),
    ]
    segments = extract_segments(keystrokes, "hi yo", wpm_bucket=5, session_len=5)
    assert len(segments) == 2
    assert segments[0].target_text == "hi"
    assert segments[1].target_text == " yo"

def test_extract_segments_includes_errors():
    """Error keystrokes are included in the segment's keystroke sequence."""
    keystrokes = [
        make_keystroke("h", "h", 0, 80, 0, 1),
        make_keystroke("o", "i", 1, 70, 1, 2),   # error: typed 'o' instead of 'i'
        make_keystroke("BKSP", "", 2, 180, 1, 1), # backspace
        make_keystroke("i", "i", 0, 90, 1, 2),    # retype correct
    ]
    segments = extract_segments(keystrokes, "hi", wpm_bucket=5, session_len=4)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.target_text == "hi"
    # 4 keystrokes: h, o(error), BKSP, i
    assert len(seg.keystroke_char_ids) == 4
    assert seg.keystroke_char_ids[1] == char_to_id("o")  # the error char
    assert seg.keystroke_char_ids[2] == BACKSPACE_TOKEN

def test_extract_segments_context_propagation():
    """Second segment gets tail context from first segment."""
    keystrokes = [
        make_keystroke("a", "a", 0, 80, 0, 1),
        make_keystroke("b", "b", 0, 90, 1, 2),
        make_keystroke("c", "c", 0, 85, 2, 3),
        make_keystroke("d", "d", 0, 400, 3, 4),  # pause
        make_keystroke("e", "e", 0, 80, 4, 5),
    ]
    segments = extract_segments(keystrokes, "abcde", wpm_bucket=5, session_len=5)
    assert len(segments) == 2
    # Second segment should have context from first
    assert len(segments[1].prev_context) > 0

def test_short_segments_filtered():
    """Segments with < MIN_SEGMENT_KEYSTROKES are discarded."""
    keystrokes = [
        make_keystroke("a", "a", 0, 80, 0, 1),
        make_keystroke("b", "b", 0, 400, 1, 2),  # pause → segment 1 = just "a" (1 ks)
        make_keystroke("c", "c", 0, 80, 2, 3),
        make_keystroke("d", "d", 0, 90, 3, 4),
        make_keystroke("e", "e", 0, 85, 4, 5),
    ]
    segments = extract_segments(keystrokes, "abcde", wpm_bucket=5, session_len=5)
    # First segment (just "a") should be filtered out (< 3 keystrokes)
    assert all(len(s.keystroke_char_ids) >= 3 for s in segments)


# ---------------------------------------------------------------------------
# Inference-time text splitting
# ---------------------------------------------------------------------------

def test_split_text_simple():
    """Split short text into a single segment."""
    segments = split_text_into_inference_segments("hi", seed=42)
    assert len(segments) == 1
    assert segments[0] == "hi"

def test_split_text_multiple_words():
    """Longer text gets split into multi-word segments."""
    text = "the quick brown fox jumps over the lazy dog"
    segments = split_text_into_inference_segments(text, seed=42)
    assert len(segments) >= 2
    # Rejoined should equal original
    assert "".join(segments) == text

def test_split_text_preserves_all_chars():
    """No characters lost during splitting."""
    text = "Hello, world! This is a test."
    segments = split_text_into_inference_segments(text, seed=42)
    assert "".join(segments) == text
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/owengregson/Documents/Lilly && python -m pytest v2/test_segment.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'v2.segment'`

**Step 3: Write the segment module**

Create `ml/v2/segment.py`:

```python
"""Segmentation logic for the V2 phrase-level typing model.

Splits keystroke sessions into pause-delimited segments, where each segment
contains the target text and the full keystroke sequence (errors, corrections,
timing) as the human actually typed it.

Also provides inference-time text splitting (heuristic word-boundary segments).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

from v2.config import (
    ASCII_OFFSET,
    BACKSPACE_TOKEN,
    END_TOKEN,
    MAX_SEGMENT_KEYSTROKES,
    MIN_SEGMENT_KEYSTROKES,
    PAD_TOKEN,
    PAUSE_THRESHOLD_MS,
    START_TOKEN,
)


# ---------------------------------------------------------------------------
# Character encoding
# ---------------------------------------------------------------------------

def char_to_id(ch: str) -> int:
    """Encode a character as an integer ID.

    0=PAD, 1-95=printable ASCII (space..tilde), 96=BACKSPACE, 97=END, 98=START.
    """
    if not ch or ch == "":
        return PAD_TOKEN
    if ch == "BKSP":
        return BACKSPACE_TOKEN
    if len(ch) > 1:
        return PAD_TOKEN
    code = ord(ch)
    if 32 <= code <= 126:
        return code - ASCII_OFFSET
    return PAD_TOKEN


def id_to_char(cid: int) -> str:
    """Decode an integer ID back to a character string."""
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


# ---------------------------------------------------------------------------
# Segment data structure
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """A single typing segment extracted from a session."""
    target_text: str
    keystroke_char_ids: List[int]  # char IDs of what was actually typed
    keystroke_delays: List[float]  # IKI in ms for each keystroke
    wpm_bucket: int
    segment_index: int  # position in session
    session_len: int  # total keystrokes in session (fatigue proxy)
    prev_context: List[int] = field(default_factory=list)  # tail char_ids from previous segment
    sentence_pos: float = 0.0  # normalized position in sentence


# ---------------------------------------------------------------------------
# Target text extraction
# ---------------------------------------------------------------------------

def compute_target_text_for_segment(
    keystrokes: List[dict], full_sentence: str
) -> str:
    """Determine what target text a segment of keystrokes covers.

    Replays the segment's keystrokes to find the net characters produced,
    then returns the corresponding substring of the full sentence.
    """
    buffer: List[str] = []
    min_pos = None
    max_pos = None

    for ks in keystrokes:
        if ks["typed_key"] == "BKSP":
            if buffer:
                buffer.pop()
        else:
            pos = ks["target_pos"]
            if ks["action"] == 0:  # CORRECT
                if min_pos is None or pos < min_pos:
                    min_pos = pos
                if max_pos is None or pos > max_pos:
                    max_pos = pos
            buffer.append(ks["typed_key"])

    if min_pos is None:
        return ""

    # The target text is the substring from min to max+1 of the sentence
    end = min(max_pos + 1, len(full_sentence))
    return full_sentence[min_pos:end]


# ---------------------------------------------------------------------------
# Segment extraction from session keystrokes
# ---------------------------------------------------------------------------

def extract_segments(
    keystrokes: List[dict],
    target_sentence: str,
    wpm_bucket: int,
    session_len: int,
    pause_threshold: float = PAUSE_THRESHOLD_MS,
) -> List[Segment]:
    """Split a session's keystrokes into pause-delimited segments.

    A new segment starts whenever the IKI exceeds pause_threshold.
    Segments shorter than MIN_SEGMENT_KEYSTROKES are discarded.
    Segments longer than MAX_SEGMENT_KEYSTROKES are split at the nearest
    word boundary.
    """
    if not keystrokes:
        return []

    # Split into raw groups by pause threshold
    groups: List[List[dict]] = []
    current_group: List[dict] = [keystrokes[0]]

    for ks in keystrokes[1:]:
        if ks["iki"] >= pause_threshold:
            groups.append(current_group)
            current_group = [ks]
        else:
            current_group.append(ks)

    if current_group:
        groups.append(current_group)

    # Convert groups to Segments
    segments: List[Segment] = []
    prev_tail: List[int] = []

    for seg_idx, group in enumerate(groups):
        # Enforce max length: split at word boundaries if needed
        sub_groups = _split_long_group(group)

        for sub_group in sub_groups:
            if len(sub_group) < MIN_SEGMENT_KEYSTROKES:
                # Still update context for continuity
                if sub_group:
                    tail_ids = [char_to_id(ks["typed_key"]) for ks in sub_group]
                    prev_tail = tail_ids[-4:]
                continue

            target_text = compute_target_text_for_segment(sub_group, target_sentence)

            char_ids = [char_to_id(ks["typed_key"]) for ks in sub_group]
            delays = [ks["iki"] for ks in sub_group]

            # Compute sentence position from first keystroke's target_pos
            first_pos = sub_group[0]["target_pos"]
            sentence_pos = first_pos / max(len(target_sentence), 1)

            segment = Segment(
                target_text=target_text,
                keystroke_char_ids=char_ids,
                keystroke_delays=delays,
                wpm_bucket=wpm_bucket,
                segment_index=len(segments),
                session_len=session_len,
                prev_context=list(prev_tail),
                sentence_pos=sentence_pos,
            )
            segments.append(segment)

            # Update context tail
            prev_tail = char_ids[-4:]

    return segments


def _split_long_group(group: List[dict]) -> List[List[dict]]:
    """Split a keystroke group that exceeds MAX_SEGMENT_KEYSTROKES.

    Tries to split at space characters (word boundaries) for clean segments.
    """
    if len(group) <= MAX_SEGMENT_KEYSTROKES:
        return [group]

    result: List[List[dict]] = []
    start = 0

    while start < len(group):
        end = min(start + MAX_SEGMENT_KEYSTROKES, len(group))

        if end < len(group):
            # Try to find a word boundary (space) to split at
            best_split = end
            for i in range(end - 1, max(start + MIN_SEGMENT_KEYSTROKES, end - 15) - 1, -1):
                if group[i]["typed_key"] == " ":
                    best_split = i + 1  # split after the space
                    break
            end = best_split

        result.append(group[start:end])
        start = end

    return result


# ---------------------------------------------------------------------------
# Inference-time text splitting
# ---------------------------------------------------------------------------

def split_text_into_inference_segments(
    text: str,
    min_words: int = 2,
    max_words: int = 4,
    seed: int | None = None,
) -> List[str]:
    """Split text into segments for inference.

    Splits at word boundaries with randomized segment lengths
    (min_words to max_words per segment) to mimic natural burst patterns.
    """
    if not text:
        return []

    rng = random.Random(seed)

    # Find word boundary positions (indices of spaces)
    space_positions = [i for i, ch in enumerate(text) if ch == " "]

    if not space_positions:
        return [text]

    segments: List[str] = []
    start = 0

    while start < len(text):
        # Pick segment length in words
        n_words = rng.randint(min_words, max_words)

        # Find the nth space after start
        spaces_after = [p for p in space_positions if p >= start]

        if len(spaces_after) < n_words:
            # Remaining text is one segment
            segments.append(text[start:])
            break

        # Split after the nth word (at the space)
        split_at = spaces_after[n_words - 1]
        segments.append(text[start:split_at])
        start = split_at  # Next segment starts at the space

    return segments
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/owengregson/Documents/Lilly && python -m pytest v2/test_segment.py -v`

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add ml/v2/segment.py ml/v2/test_segment.py
git commit -m "feat(v2): add pause-based segmentation logic with tests"
```

---

## Task 3: Data Preprocessing (Aalto → Segments)

**Files:**
- Create: `ml/v2/preprocess.py`

This script reuses V1's `parse_keystroke_file` and `replay_session` functions (imported from the parent `preprocess` module), then applies V2 segmentation and saves segment-level `.npz` files.

**Step 1: Write the preprocessor**

Create `ml/v2/preprocess.py`:

```python
#!/usr/bin/env python3
"""Reprocess Aalto 136M keystrokes into pause-delimited segments for V2.

Reuses V1's parsing and replay logic, then applies V2 segmentation to produce
segment-level training examples saved as compressed NumPy archives.

Each .npz file contains arrays for a batch of segments:
    encoder_chars:    (N, max_target_len) int32   - target text char IDs
    encoder_lengths:  (N,) int32                  - actual target text length
    decoder_chars:    (N, max_ks_len) int32        - keystroke char IDs
    decoder_delays:   (N, max_ks_len) float32      - keystroke delays (ms)
    decoder_lengths:  (N,) int32                   - actual keystroke seq length
    wpm_buckets:      (N,) int32                   - WPM persona bucket
    prev_contexts:    (N, context_tail_len) int32   - previous segment tail
    sentence_pos:     (N,) float32                  - position in sentence
    session_fracs:    (N,) float32                  - segment_index / total_segments (fatigue)

Usage:
    python -m v2.preprocess                     # process all files
    python -m v2.preprocess --max-files 50      # first 50 files
    python -m v2.preprocess --workers 8         # parallel
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

# V1 imports for parsing
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess import (
    parse_keystroke_file,
    replay_session,
    compute_wpm,
    wpm_to_bucket,
    find_keystroke_files,
)
from config import MAX_KEYSTROKES_PER_SESSION

# V2 imports
from v2.config import (
    MAX_SEGMENT_KEYSTROKES,
    MAX_TARGET_CHARS,
    ModelConfig,
    RAW_DIR,
    SEGMENT_DIR,
)
from v2.segment import Segment, char_to_id, extract_segments


def process_file_to_segments(filepath: Path) -> List[Segment]:
    """Parse one participant file and extract all segments."""
    sessions_raw = parse_keystroke_file(filepath)
    all_segments: List[Segment] = []

    for (uid, sid, sentence), keystrokes in sessions_raw.items():
        if len(keystrokes) > MAX_KEYSTROKES_PER_SESSION:
            continue

        aligned = replay_session(sentence, keystrokes)
        if aligned is None:
            continue

        wpm = compute_wpm(aligned)
        if wpm <= 0 or wpm > 300:
            continue

        bucket = wpm_to_bucket(wpm)

        # Convert AlignedKeystroke objects to dicts for segment extraction
        ks_dicts = [
            {
                "typed_key": ks.typed_key,
                "target_char": ks.target_char,
                "action": ks.action,
                "iki": ks.iki,
                "hold_time": ks.hold_time,
                "target_pos": ks.target_pos,
                "buffer_len": ks.buffer_len,
            }
            for ks in aligned
        ]

        segments = extract_segments(
            ks_dicts, sentence, wpm_bucket=bucket, session_len=len(aligned)
        )
        all_segments.extend(segments)

    return all_segments


def segments_to_arrays(
    segments: List[Segment],
    cfg: ModelConfig,
) -> dict:
    """Pack a list of Segments into padded NumPy arrays."""
    n = len(segments)
    max_enc = cfg.max_encoder_len
    max_dec = cfg.max_decoder_len
    ctx_len = cfg.context_tail_len

    encoder_chars = np.zeros((n, max_enc), dtype=np.int32)
    encoder_lengths = np.zeros(n, dtype=np.int32)
    decoder_chars = np.zeros((n, max_dec), dtype=np.int32)
    decoder_delays = np.zeros((n, max_dec), dtype=np.float32)
    decoder_lengths = np.zeros(n, dtype=np.int32)
    wpm_buckets = np.zeros(n, dtype=np.int32)
    prev_contexts = np.zeros((n, ctx_len), dtype=np.int32)
    sentence_pos = np.zeros(n, dtype=np.float32)
    session_fracs = np.zeros(n, dtype=np.float32)

    for i, seg in enumerate(segments):
        # Encoder: target text characters
        target_ids = [char_to_id(ch) for ch in seg.target_text[:max_enc]]
        enc_len = len(target_ids)
        encoder_chars[i, :enc_len] = target_ids
        encoder_lengths[i] = enc_len

        # Decoder: keystroke sequence
        dec_len = min(len(seg.keystroke_char_ids), max_dec)
        decoder_chars[i, :dec_len] = seg.keystroke_char_ids[:dec_len]
        decoder_delays[i, :dec_len] = seg.keystroke_delays[:dec_len]
        decoder_lengths[i] = dec_len

        # Metadata
        wpm_buckets[i] = seg.wpm_bucket

        # Previous context (pad from left if shorter than ctx_len)
        ctx = seg.prev_context[-ctx_len:]
        if ctx:
            prev_contexts[i, ctx_len - len(ctx):] = ctx

        sentence_pos[i] = seg.sentence_pos
        session_fracs[i] = seg.segment_index / max(seg.session_len, 1)

    return {
        "encoder_chars": encoder_chars,
        "encoder_lengths": encoder_lengths,
        "decoder_chars": decoder_chars,
        "decoder_delays": decoder_delays,
        "decoder_lengths": decoder_lengths,
        "wpm_buckets": wpm_buckets,
        "prev_contexts": prev_contexts,
        "sentence_pos": sentence_pos,
        "session_fracs": session_fracs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Aalto data into V2 segments")
    parser.add_argument("--data-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=SEGMENT_DIR)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = find_keystroke_files(args.data_dir)
    if not files:
        print(f"No keystroke files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.max_files > 0:
        files = files[:args.max_files]

    print(f"Found {len(files)} keystroke files")
    print(f"Output: {args.output_dir}")
    print()

    cfg = ModelConfig()
    total_segments = 0
    chunk_idx = 0

    for chunk_start in range(0, len(files), args.chunk_size):
        chunk_files = files[chunk_start:chunk_start + args.chunk_size]
        chunk_segments: List[Segment] = []

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_file_to_segments, f): f for f in chunk_files}
            pbar = tqdm(as_completed(futures), total=len(futures), desc=f"Chunk {chunk_idx}")
            for future in pbar:
                try:
                    segs = future.result()
                    chunk_segments.extend(segs)
                    pbar.set_postfix(segments=len(chunk_segments))
                except Exception as e:
                    print(f"\nError: {e}", file=sys.stderr)

        if chunk_segments:
            arrays = segments_to_arrays(chunk_segments, cfg)
            out_path = args.output_dir / f"segments_chunk_{chunk_idx:04d}.npz"
            np.savez_compressed(out_path, **arrays)
            n = len(chunk_segments)
            total_segments += n
            print(f"  Chunk {chunk_idx}: {n:,} segments → {out_path.name}")

        chunk_idx += 1

    print()
    print("=" * 60)
    print(f"V2 preprocessing complete.")
    print(f"  Total segments: {total_segments:,}")
    print(f"  Output chunks:  {chunk_idx}")
    print(f"  Output dir:     {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test (dry run with --max-files 1)**

Run: `cd /Users/owengregson/Documents/Lilly && python -m v2.preprocess --max-files 1 --workers 1`

Expected: Processes 1 file, prints segment count, creates `data/v2_segments/segments_chunk_0000.npz`.

**Step 3: Verify output format**

Run: `cd /Users/owengregson/Documents/Lilly && python -c "
import numpy as np
d = np.load('data/v2_segments/segments_chunk_0000.npz')
for k in sorted(d.files): print(f'{k}: {d[k].shape} {d[k].dtype}')
"`

Expected: All array shapes match the spec (encoder_chars has 2 dims, decoder_chars has 2 dims, etc.).

**Step 4: Commit**

```bash
git add ml/v2/preprocess.py
git commit -m "feat(v2): add segment-level preprocessing pipeline"
```

---

## Task 4: Dataset Pipeline

**Files:**
- Create: `ml/v2/dataset.py`

Loads `.npz` segment files, constructs (encoder_inputs, decoder_inputs, decoder_labels) tuples, splits train/val/test, and builds `tf.data.Dataset` objects with padding and batching.

**Step 1: Write dataset module**

Create `ml/v2/dataset.py`:

```python
"""TF data pipeline for V2 segment-level training examples.

Loads .npz segment files, prepares encoder inputs, decoder inputs
(teacher-forced, shifted right with START token), and decoder labels
(shifted left with END token).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from v2.config import (
    END_TOKEN,
    START_TOKEN,
    PAD_TOKEN,
    SEGMENT_DIR,
    ModelConfig,
    TrainConfig,
)


def load_segment_files(data_dir: Path, max_files: int = 0) -> dict:
    """Load and concatenate all segment .npz files."""
    files = sorted(data_dir.glob("segments_chunk_*.npz"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No segment files in {data_dir}")

    arrays = {}
    for f in files:
        data = np.load(f)
        for key in data.files:
            arrays.setdefault(key, []).append(data[key])

    return {k: np.concatenate(v) for k, v in arrays.items()}


def prepare_decoder_io(
    decoder_chars: np.ndarray,
    decoder_delays: np.ndarray,
    decoder_lengths: np.ndarray,
    max_len: int,
) -> tuple:
    """Prepare teacher-forced decoder inputs and labels.

    Decoder input:  [START, c1, c2, ..., cn]  (shifted right)
    Decoder label:  [c1, c2, ..., cn, END]    (shifted left)

    Delay input:    [0, d1, d2, ..., dn]      (shifted right)
    Delay label:    [d1, d2, ..., dn, 0]      (shifted left)
    """
    n = len(decoder_chars)
    # +1 for START/END tokens
    input_len = max_len + 1
    label_len = max_len + 1

    dec_input_chars = np.full((n, input_len), PAD_TOKEN, dtype=np.int32)
    dec_input_delays = np.zeros((n, input_len), dtype=np.float32)
    dec_label_chars = np.full((n, label_len), PAD_TOKEN, dtype=np.int32)
    dec_label_delays = np.zeros((n, label_len), dtype=np.float32)
    dec_label_mask = np.zeros((n, label_len), dtype=np.float32)

    for i in range(n):
        length = int(decoder_lengths[i])

        # Input: START + keystrokes
        dec_input_chars[i, 0] = START_TOKEN
        dec_input_chars[i, 1:length + 1] = decoder_chars[i, :length]
        dec_input_delays[i, 0] = 0.0
        dec_input_delays[i, 1:length + 1] = decoder_delays[i, :length]

        # Label: keystrokes + END
        dec_label_chars[i, :length] = decoder_chars[i, :length]
        dec_label_chars[i, length] = END_TOKEN
        dec_label_delays[i, :length] = decoder_delays[i, :length]
        dec_label_mask[i, :length + 1] = 1.0  # mask includes the END token

    return dec_input_chars, dec_input_delays, dec_label_chars, dec_label_delays, dec_label_mask


def build_datasets(
    data_dir: Path,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    max_files: int = 0,
) -> tuple:
    """Build train, validation, test tf.data.Datasets.

    Returns (train_ds, val_ds, test_ds, n_total).
    """
    print("Loading segment files...")
    arrays = load_segment_files(data_dir, max_files)

    n_total = len(arrays["encoder_chars"])
    print(f"  Total segments: {n_total:,}")

    if train_cfg.max_samples > 0:
        n_total = min(n_total, train_cfg.max_samples)
        arrays = {k: v[:n_total] for k, v in arrays.items()}
        print(f"  Capped to: {n_total:,}")

    # Shuffle
    rng = np.random.default_rng(train_cfg.seed)
    perm = rng.permutation(n_total)
    arrays = {k: v[perm] for k, v in arrays.items()}

    # Prepare decoder I/O
    dec_in_chars, dec_in_delays, dec_lbl_chars, dec_lbl_delays, dec_lbl_mask = (
        prepare_decoder_io(
            arrays["decoder_chars"],
            arrays["decoder_delays"],
            arrays["decoder_lengths"],
            model_cfg.max_decoder_len,
        )
    )

    # Normalize delays to log-space for training
    import math
    dec_in_delays_log = np.log(np.clip(dec_in_delays, 1.0, 5000.0))
    dec_in_delays_log[dec_in_delays == 0] = 0.0  # keep padding as 0
    dec_lbl_delays_log = np.log(np.clip(dec_lbl_delays, 1.0, 5000.0))
    dec_lbl_delays_log[dec_lbl_delays == 0] = 0.0

    # Split
    n_test = int(n_total * train_cfg.test_split)
    n_val = int(n_total * train_cfg.val_split)
    n_train = n_total - n_val - n_test
    print(f"  Train: {n_train:,}  Val: {n_val:,}  Test: {n_test:,}")

    def _make_ds(start: int, end: int, shuffle: bool) -> tf.data.Dataset:
        s = slice(start, end)
        inputs = {
            "encoder_chars": arrays["encoder_chars"][s],
            "encoder_lengths": arrays["encoder_lengths"][s],
            "decoder_input_chars": dec_in_chars[s],
            "decoder_input_delays": dec_in_delays_log[s],
            "wpm_bucket": arrays["wpm_buckets"][s],
            "prev_context": arrays["prev_contexts"][s],
            "sentence_pos": arrays["sentence_pos"][s],
            "session_frac": arrays["session_fracs"][s],
        }
        labels = {
            "char_labels": dec_lbl_chars[s],
            "delay_labels": dec_lbl_delays_log[s],
            "label_mask": dec_lbl_mask[s],
        }

        ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
        if shuffle:
            ds = ds.shuffle(
                buffer_size=min(train_cfg.shuffle_buffer, end - start),
                seed=train_cfg.seed,
            )
        ds = ds.batch(train_cfg.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _make_ds(0, n_train, shuffle=True)
    val_ds = _make_ds(n_train, n_train + n_val, shuffle=False)
    test_ds = _make_ds(n_train + n_val, n_total, shuffle=False)

    return train_ds, val_ds, test_ds, n_total
```

**Step 2: Smoke test dataset loading**

Run: `cd /Users/owengregson/Documents/Lilly && python -c "
from v2.dataset import build_datasets
from v2.config import ModelConfig, TrainConfig, SEGMENT_DIR
train_ds, val_ds, test_ds, n = build_datasets(SEGMENT_DIR, ModelConfig(), TrainConfig(), max_files=1)
for inputs, labels in train_ds.take(1):
    for k, v in inputs.items(): print(f'  input {k}: {v.shape}')
    for k, v in labels.items(): print(f'  label {k}: {v.shape}')
"`

Expected: Prints shapes for all input/label tensors. Encoder chars should be (batch, 32), decoder input chars should be (batch, 65).

**Step 3: Commit**

```bash
git add ml/v2/dataset.py
git commit -m "feat(v2): add segment-level dataset pipeline"
```

---

## Task 5: Transformer Model Architecture

**Files:**
- Create: `ml/v2/model.py`

The core model: a small Transformer encoder-decoder with two output projections (char_id softmax, delay LogNormal params) per decoder step.

**Step 1: Write the model**

Create `ml/v2/model.py`. This is the most complex file. Key decisions:
- Use Keras's `MultiHeadAttention` layer (built into TF 2.15+)
- Sinusoidal positional encoding
- Encoder: target chars + WPM + context → encoder hidden states
- Decoder: causal self-attention + cross-attention → char_id + delay per step
- Custom LogNormal NLL loss (same as V1, proven to work)

```python
"""V2 Transformer encoder-decoder model for phrase-level typing generation.

Encoder: target text characters + WPM + context → hidden states
Decoder: autoregressive keystroke generation → (char_id, delay) per step

Two output projections per decoder step:
    char_logits:  (batch, dec_len, num_char_classes) → softmax
    delay_params: (batch, dec_len, 2)                → [mu, log_sigma] for LogNormal
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from tensorflow import keras

from v2.config import ModelConfig


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def sinusoidal_positional_encoding(max_len: int, d_model: int) -> tf.Tensor:
    """Compute sinusoidal positional encoding matrix."""
    positions = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
    dims = np.arange(d_model)[np.newaxis, :]       # (1, d_model)

    angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])

    return tf.constant(angles, dtype=tf.float32)  # (max_len, d_model)


# ---------------------------------------------------------------------------
# Transformer encoder layer
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = keras.layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(dim_ff, activation="relu"),
            keras.layers.Dense(d_model),
        ])
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)

    def call(self, x, padding_mask=None, training=False):
        attn = self.mha(x, x, attention_mask=padding_mask, training=training)
        x = self.norm1(x + self.drop1(attn, training=training))
        ffn = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn, training=training))
        return x


# ---------------------------------------------------------------------------
# Transformer decoder layer
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.self_attn = keras.layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead
        )
        self.cross_attn = keras.layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(dim_ff, activation="relu"),
            keras.layers.Dense(d_model),
        ])
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        self.norm3 = keras.layers.LayerNormalization()
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)
        self.drop3 = keras.layers.Dropout(dropout)

    def call(self, x, encoder_output, causal_mask=None,
             encoder_padding_mask=None, training=False):
        # Causal self-attention
        self_attn = self.self_attn(
            x, x, attention_mask=causal_mask, training=training
        )
        x = self.norm1(x + self.drop1(self_attn, training=training))

        # Cross-attention to encoder
        cross_attn = self.cross_attn(
            x, encoder_output, attention_mask=encoder_padding_mask,
            training=training,
        )
        x = self.norm2(x + self.drop2(cross_attn, training=training))

        # Feed-forward
        ffn = self.ffn(x)
        x = self.norm3(x + self.drop3(ffn, training=training))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TypingTransformer(keras.Model):
    def __init__(self, cfg: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        # Embeddings
        self.char_embed = keras.layers.Embedding(
            cfg.num_char_classes, cfg.char_embed_dim, name="char_embedding"
        )
        self.wpm_embed = keras.layers.Embedding(
            cfg.num_wpm_buckets, cfg.wpm_embed_dim, name="wpm_embedding"
        )

        # Encoder projection (char_embed + context → d_model)
        self.encoder_proj = keras.layers.Dense(cfg.d_model, name="encoder_proj")

        # Decoder projection (char_embed + delay → d_model)
        self.delay_proj = keras.layers.Dense(cfg.char_embed_dim, name="delay_proj")
        self.decoder_proj = keras.layers.Dense(cfg.d_model, name="decoder_proj")

        # Positional encodings (pre-computed)
        self.enc_pos = sinusoidal_positional_encoding(
            cfg.max_encoder_len + cfg.context_tail_len + 4, cfg.d_model
        )
        self.dec_pos = sinusoidal_positional_encoding(
            cfg.max_decoder_len + 2, cfg.d_model
        )

        # Encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(
                cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout,
                name=f"encoder_{i}",
            )
            for i in range(cfg.num_encoder_layers)
        ]

        # Decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(
                cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout,
                name=f"decoder_{i}",
            )
            for i in range(cfg.num_decoder_layers)
        ]

        # Output heads
        self.char_head = keras.layers.Dense(
            cfg.num_char_classes, name="char_output"
        )
        self.delay_head = keras.layers.Dense(2, name="delay_output")  # mu, log_sigma

        # WPM conditioning: project to d_model for addition
        self.wpm_proj = keras.layers.Dense(cfg.d_model, name="wpm_proj")

    def _encode(self, encoder_chars, encoder_lengths, wpm_bucket,
                prev_context, training=False):
        """Encode target text + context into hidden states."""
        # Embed target characters
        char_emb = self.char_embed(encoder_chars)  # (B, enc_len, embed)

        # Project to d_model
        enc = self.encoder_proj(char_emb)  # (B, enc_len, d_model)

        # Add positional encoding
        seq_len = tf.shape(enc)[1]
        enc = enc + self.enc_pos[:seq_len]

        # Add WPM conditioning (broadcast across sequence)
        wpm_emb = self.wpm_embed(wpm_bucket)  # (B, 1, wpm_dim)
        wpm_emb = tf.squeeze(wpm_emb, axis=1)  # (B, wpm_dim)
        wpm_cond = self.wpm_proj(wpm_emb)  # (B, d_model)
        enc = enc + wpm_cond[:, tf.newaxis, :]

        # Create padding mask from encoder_lengths
        max_len = tf.shape(encoder_chars)[1]
        positions = tf.range(max_len)[tf.newaxis, :]  # (1, max_len)
        padding_mask = positions < encoder_lengths[:, tf.newaxis]  # (B, max_len)
        # MultiHeadAttention expects (B, T, S) attention mask
        # For self-attention: (B, enc_len, enc_len)
        attn_mask = padding_mask[:, tf.newaxis, :]  # (B, 1, enc_len)

        # Run encoder layers
        for layer in self.encoder_layers:
            enc = layer(enc, padding_mask=attn_mask, training=training)

        return enc, padding_mask

    def _decode(self, decoder_input_chars, decoder_input_delays,
                encoder_output, encoder_padding_mask, training=False):
        """Decode keystroke sequence with teacher forcing."""
        # Embed decoder characters
        char_emb = self.char_embed(decoder_input_chars)  # (B, dec_len, embed)

        # Embed delays
        delay_expanded = decoder_input_delays[:, :, tf.newaxis]  # (B, dec_len, 1)
        delay_emb = self.delay_proj(delay_expanded)  # (B, dec_len, embed)

        # Combine char + delay embeddings
        combined = tf.concat([char_emb, delay_emb], axis=-1)  # (B, dec_len, 2*embed)
        dec = self.decoder_proj(combined)  # (B, dec_len, d_model)

        # Add positional encoding
        dec_len = tf.shape(dec)[1]
        dec = dec + self.dec_pos[:dec_len]

        # Causal mask for self-attention
        causal_mask = tf.linalg.band_part(
            tf.ones((dec_len, dec_len)), -1, 0
        )  # lower triangular (dec_len, dec_len)
        causal_mask = causal_mask[tf.newaxis, :, :]  # (1, dec_len, dec_len)

        # Cross-attention mask from encoder padding
        cross_mask = encoder_padding_mask[:, tf.newaxis, :]  # (B, 1, enc_len)

        # Run decoder layers
        for layer in self.decoder_layers:
            dec = layer(
                dec, encoder_output,
                causal_mask=causal_mask,
                encoder_padding_mask=cross_mask,
                training=training,
            )

        # Output projections
        char_logits = self.char_head(dec)  # (B, dec_len, num_classes)
        delay_params = self.delay_head(dec)  # (B, dec_len, 2)

        return char_logits, delay_params

    def call(self, inputs, training=False):
        """Forward pass for training (teacher forcing)."""
        encoder_output, encoder_padding_mask = self._encode(
            inputs["encoder_chars"],
            inputs["encoder_lengths"],
            inputs["wpm_bucket"][:, tf.newaxis],  # ensure (B, 1)
            inputs["prev_context"],
            training=training,
        )

        char_logits, delay_params = self._decode(
            inputs["decoder_input_chars"],
            inputs["decoder_input_delays"],
            encoder_output,
            encoder_padding_mask,
            training=training,
        )

        return {"char_logits": char_logits, "delay_params": delay_params}


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class LogNormalNLL(keras.losses.Loss):
    """Negative log-likelihood of a LogNormal distribution (per-token)."""

    def __init__(self, **kwargs):
        super().__init__(name="log_normal_nll", **kwargs)

    def call(self, y_true, y_pred):
        # y_true: (batch, seq) log-IKI
        # y_pred: (batch, seq, 2) [mu, log_sigma]
        mu = y_pred[:, :, 0]
        log_sigma = y_pred[:, :, 1]
        sigma = tf.exp(tf.clip_by_value(log_sigma, -5.0, 5.0))

        nll = (
            0.5 * tf.square((y_true - mu) / (sigma + 1e-6))
            + log_sigma
            + 0.5 * math.log(2.0 * math.pi)
        )
        return nll  # (batch, seq) — masked externally


def compute_loss(model_output, labels, char_weight=1.0, timing_weight=1.0):
    """Compute combined masked loss for char prediction and timing.

    Both losses are masked by label_mask to ignore padding positions.
    """
    char_logits = model_output["char_logits"]
    delay_params = model_output["delay_params"]
    mask = labels["label_mask"]  # (batch, seq)

    # Character loss: sparse cross-entropy
    char_loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    char_loss = char_loss_fn(labels["char_labels"], char_logits)  # (batch, seq)
    char_loss = tf.reduce_sum(char_loss * mask) / tf.reduce_sum(mask)

    # Timing loss: LogNormal NLL
    timing_nll = LogNormalNLL()(labels["delay_labels"], delay_params)  # (batch, seq)
    timing_loss = tf.reduce_sum(timing_nll * mask) / tf.reduce_sum(mask)

    total = char_weight * char_loss + timing_weight * timing_loss

    return total, char_loss, timing_loss


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig) -> TypingTransformer:
    """Instantiate the model and build it with dummy input."""
    model = TypingTransformer(cfg)

    # Build by calling with dummy data
    dummy = {
        "encoder_chars": tf.zeros((1, cfg.max_encoder_len), dtype=tf.int32),
        "encoder_lengths": tf.constant([[5]], dtype=tf.int32),
        "decoder_input_chars": tf.zeros((1, cfg.max_decoder_len + 1), dtype=tf.int32),
        "decoder_input_delays": tf.zeros((1, cfg.max_decoder_len + 1), dtype=tf.float32),
        "wpm_bucket": tf.zeros((1,), dtype=tf.int32),
        "prev_context": tf.zeros((1, cfg.context_tail_len), dtype=tf.int32),
        "sentence_pos": tf.zeros((1,), dtype=tf.float32),
        "session_frac": tf.zeros((1,), dtype=tf.float32),
    }
    model(dummy, training=False)

    return model
```

**Step 2: Verify model builds and has reasonable param count**

Run: `cd /Users/owengregson/Documents/Lilly && python -c "
from v2.model import build_model
from v2.config import ModelConfig
model = build_model(ModelConfig())
model.summary()
print(f'Total params: {model.count_params():,}')
"`

Expected: Model summary prints. Total params should be roughly 250K-500K.

**Step 3: Verify forward pass produces correct output shapes**

Run: `cd /Users/owengregson/Documents/Lilly && python -c "
import tensorflow as tf
from v2.model import build_model
from v2.config import ModelConfig
cfg = ModelConfig()
model = build_model(cfg)
dummy = {
    'encoder_chars': tf.zeros((4, cfg.max_encoder_len), dtype=tf.int32),
    'encoder_lengths': tf.constant([[5],[8],[3],[10]], dtype=tf.int32),
    'decoder_input_chars': tf.zeros((4, cfg.max_decoder_len + 1), dtype=tf.int32),
    'decoder_input_delays': tf.zeros((4, cfg.max_decoder_len + 1), dtype=tf.float32),
    'wpm_bucket': tf.zeros((4,), dtype=tf.int32),
    'prev_context': tf.zeros((4, cfg.context_tail_len), dtype=tf.int32),
    'sentence_pos': tf.zeros((4,), dtype=tf.float32),
    'session_frac': tf.zeros((4,), dtype=tf.float32),
}
out = model(dummy)
print(f'char_logits: {out[\"char_logits\"].shape}')  # (4, 65, 99)
print(f'delay_params: {out[\"delay_params\"].shape}')  # (4, 65, 2)
"`

Expected: `char_logits: (4, 65, 99)` and `delay_params: (4, 65, 2)`.

**Step 4: Commit**

```bash
git add ml/v2/model.py
git commit -m "feat(v2): add Transformer encoder-decoder model architecture"
```

---

## Task 6: Training Loop

**Files:**
- Create: `ml/v2/train.py`

Custom training loop (not `model.fit`) because we need masked loss computation. Uses `tf.GradientTape`.

**Step 1: Write training script**

Create `ml/v2/train.py`:

```python
#!/usr/bin/env python3
"""Train the V2 phrase-level typing model.

Usage:
    python -m v2.train                           # defaults
    python -m v2.train --epochs 50               # more epochs
    python -m v2.train --max-files 5             # limit data
    python -m v2.train --max-samples 100000      # cap samples
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from v2.config import MODEL_DIR, SEGMENT_DIR, ModelConfig, TrainConfig
from v2.dataset import build_datasets
from v2.model import TypingTransformer, build_model, compute_loss


def train_step(model, optimizer, inputs, labels, train_cfg):
    """Single training step with gradient tape."""
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        total_loss, char_loss, timing_loss = compute_loss(
            outputs, labels,
            char_weight=train_cfg.char_loss_weight,
            timing_weight=train_cfg.timing_loss_weight,
        )

    grads = tape.gradient(total_loss, model.trainable_variables)
    # Gradient clipping
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Compute char accuracy (masked)
    mask = labels["label_mask"]
    char_logits = outputs["char_logits"]
    preds = tf.argmax(char_logits, axis=-1, output_type=tf.int32)
    correct = tf.cast(tf.equal(preds, labels["char_labels"]), tf.float32)
    accuracy = tf.reduce_sum(correct * mask) / tf.reduce_sum(mask)

    return total_loss, char_loss, timing_loss, accuracy


def val_step(model, inputs, labels, train_cfg):
    """Single validation step (no gradients)."""
    outputs = model(inputs, training=False)
    total_loss, char_loss, timing_loss = compute_loss(
        outputs, labels,
        char_weight=train_cfg.char_loss_weight,
        timing_weight=train_cfg.timing_loss_weight,
    )

    mask = labels["label_mask"]
    preds = tf.argmax(outputs["char_logits"], axis=-1, output_type=tf.int32)
    correct = tf.cast(tf.equal(preds, labels["char_labels"]), tf.float32)
    accuracy = tf.reduce_sum(correct * mask) / tf.reduce_sum(mask)

    return total_loss, char_loss, timing_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V2 typing model")
    parser.add_argument("--data-dir", type=Path, default=SEGMENT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.max_samples is not None:
        train_cfg.max_samples = args.max_samples

    run_name = args.run_name or datetime.now().strftime("v2_run_%Y%m%d_%H%M%S")
    run_dir = args.model_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V2 Phrase-Level Typing Model - Training")
    print("=" * 60)
    print(f"  Run:         {run_name}")
    print(f"  Epochs:      {train_cfg.epochs}")
    print(f"  Batch size:  {train_cfg.batch_size}")
    print(f"  LR:          {train_cfg.learning_rate}")
    print()

    # Data
    train_ds, val_ds, test_ds, n_total = build_datasets(
        args.data_dir, model_cfg, train_cfg, max_files=args.max_files
    )

    # Model
    model = build_model(model_cfg)
    model.summary()

    # Optimizer with cosine decay
    total_train_steps = n_total // train_cfg.batch_size * train_cfg.epochs
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=train_cfg.learning_rate,
        decay_steps=total_train_steps,
        alpha=1e-6,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=train_cfg.weight_decay,
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    csv_path = run_dir / "training_log.csv"

    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,train_char_loss,train_timing_loss,train_acc,"
                "val_loss,val_char_loss,val_timing_loss,val_acc\n")

    for epoch in range(train_cfg.epochs):
        # Train
        train_losses, train_char_losses, train_timing_losses, train_accs = [], [], [], []
        for inputs, labels in train_ds:
            tl, cl, tl2, acc = train_step(model, optimizer, inputs, labels, train_cfg)
            train_losses.append(tl.numpy())
            train_char_losses.append(cl.numpy())
            train_timing_losses.append(tl2.numpy())
            train_accs.append(acc.numpy())

        # Validate
        val_losses, val_char_losses, val_timing_losses, val_accs = [], [], [], []
        for inputs, labels in val_ds:
            vl, vcl, vtl, vacc = val_step(model, inputs, labels, train_cfg)
            val_losses.append(vl.numpy())
            val_char_losses.append(vcl.numpy())
            val_timing_losses.append(vtl.numpy())
            val_accs.append(vacc.numpy())

        # Metrics
        t_loss = np.mean(train_losses)
        t_char = np.mean(train_char_losses)
        t_time = np.mean(train_timing_losses)
        t_acc = np.mean(train_accs)
        v_loss = np.mean(val_losses)
        v_char = np.mean(val_char_losses)
        v_time = np.mean(val_timing_losses)
        v_acc = np.mean(val_accs)

        print(f"Epoch {epoch+1}/{train_cfg.epochs}: "
              f"loss={t_loss:.4f} char={t_char:.4f} timing={t_time:.4f} acc={t_acc:.4f} | "
              f"val_loss={v_loss:.4f} val_char={v_char:.4f} val_timing={v_time:.4f} val_acc={v_acc:.4f}")

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{t_loss},{t_char},{t_time},{t_acc},"
                    f"{v_loss},{v_char},{v_time},{v_acc}\n")

        # Checkpoint
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            model.save(str(run_dir / "best_model.keras"))
            print(f"  -> Saved best model (val_loss={v_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Final evaluation
    print("\nEvaluating on test set...")
    model = keras.models.load_model(
        str(run_dir / "best_model.keras"), compile=False
    )
    test_losses, test_accs = [], []
    for inputs, labels in test_ds:
        tl, _, _, acc = val_step(model, inputs, labels, train_cfg)
        test_losses.append(tl.numpy())
        test_accs.append(acc.numpy())

    test_loss = np.mean(test_losses)
    test_acc = np.mean(test_accs)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    # Save metadata
    meta = {
        "model_config": model_cfg.__dict__,
        "train_config": train_cfg.__dict__,
        "total_samples": n_total,
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "run_name": run_name,
    }
    with open(run_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Model saved to: {run_dir}")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test with tiny data**

Run: `cd /Users/owengregson/Documents/Lilly && python -m v2.train --max-files 1 --max-samples 500 --epochs 2 --batch-size 32 --run-name smoke_test`

Expected: Trains for 2 epochs, prints loss/accuracy per epoch, saves `models/v2/smoke_test/best_model.keras`.

**Step 3: Commit**

```bash
git add ml/v2/train.py
git commit -m "feat(v2): add custom training loop with masked loss"
```

---

## Task 7: Inference / Sequence Generation

**Files:**
- Create: `ml/v2/generate.py`

Autoregressive generation: for each segment, encode the target text, then decode keystroke-by-keystroke, feeding each sampled output back as input to the next step.

**Step 1: Write generator**

Create `ml/v2/generate.py`:

```python
#!/usr/bin/env python3
"""Generate realistic typing sequences using the V2 model.

Autoregressive inference: encode target text, then decode one keystroke
at a time, feeding each sampled (char_id, delay) back as input.

Usage:
    python -m v2.generate models/v2/run_XXXX/best_model.keras "Hello, world!"
    python -m v2.generate models/v2/run_XXXX/best_model.keras --wpm 80
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from v2.config import (
    END_TOKEN,
    MAX_IKI_MS,
    MIN_IKI_MS,
    START_TOKEN,
    PAD_TOKEN,
    ModelConfig,
)
from v2.segment import char_to_id, id_to_char, split_text_into_inference_segments


class GeneratedKeystroke(NamedTuple):
    key: str
    delay_ms: float
    cumulative_ms: float


def wpm_to_bucket(wpm: float) -> int:
    from v2.config import WPM_BUCKET_EDGES
    for i in range(len(WPM_BUCKET_EDGES) - 1):
        if wpm < WPM_BUCKET_EDGES[i + 1]:
            return i
    return len(WPM_BUCKET_EDGES) - 2


def sample_lognormal(mu: float, log_sigma: float) -> float:
    sigma = math.exp(np.clip(log_sigma, -5.0, 5.0))
    log_val = np.random.normal(mu, sigma)
    return float(np.clip(np.exp(log_val), MIN_IKI_MS, MAX_IKI_MS))


def weighted_sample(logits: np.ndarray, temperature: float = 1.0) -> int:
    if temperature != 1.0:
        logits = logits / temperature
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


def generate_segment(
    model: keras.Model,
    target_text: str,
    wpm_bucket: int,
    prev_context: List[int],
    cfg: ModelConfig,
    temperature: float = 0.8,
    max_steps: int = 0,
) -> List[GeneratedKeystroke]:
    """Generate keystroke sequence for one segment via autoregressive decoding."""
    if max_steps <= 0:
        max_steps = len(target_text) * 3 + 5

    # Encode target text
    target_ids = [char_to_id(ch) for ch in target_text[:cfg.max_encoder_len]]
    enc_len = len(target_ids)

    encoder_chars = np.zeros((1, cfg.max_encoder_len), dtype=np.int32)
    encoder_chars[0, :enc_len] = target_ids

    # Prepare static inputs
    static_inputs = {
        "encoder_chars": tf.constant(encoder_chars),
        "encoder_lengths": tf.constant([[enc_len]], dtype=tf.int32),
        "wpm_bucket": tf.constant([wpm_bucket], dtype=tf.int32),
        "prev_context": tf.constant(
            [_pad_context(prev_context, cfg.context_tail_len)], dtype=tf.int32
        ),
        "sentence_pos": tf.constant([0.0], dtype=tf.float32),
        "session_frac": tf.constant([0.0], dtype=tf.float32),
    }

    # Encode once
    encoder_output, encoder_padding_mask = model._encode(
        static_inputs["encoder_chars"],
        static_inputs["encoder_lengths"],
        static_inputs["wpm_bucket"][:, tf.newaxis],
        static_inputs["prev_context"],
        training=False,
    )

    # Autoregressive decoding
    dec_max_len = cfg.max_decoder_len + 1
    dec_chars = np.zeros((1, dec_max_len), dtype=np.int32)
    dec_delays = np.zeros((1, dec_max_len), dtype=np.float32)
    dec_chars[0, 0] = START_TOKEN

    keystrokes: List[GeneratedKeystroke] = []
    cumulative_ms = 0.0

    for step in range(1, min(max_steps + 1, dec_max_len)):
        # Run decoder up to current position
        char_logits, delay_params = model._decode(
            tf.constant(dec_chars[:, :step]),
            tf.constant(dec_delays[:, :step]),
            encoder_output,
            encoder_padding_mask,
            training=False,
        )

        # Sample from last position
        last_logits = char_logits[0, -1].numpy()  # (num_classes,)
        last_delay = delay_params[0, -1].numpy()  # (2,)

        char_id = weighted_sample(last_logits, temperature)

        if char_id == END_TOKEN:
            break
        if char_id == PAD_TOKEN:
            break

        delay_ms = sample_lognormal(last_delay[0], last_delay[1])
        cumulative_ms += delay_ms

        key = id_to_char(char_id)
        keystrokes.append(GeneratedKeystroke(key, delay_ms, cumulative_ms))

        # Feed back
        if step < dec_max_len:
            dec_chars[0, step] = char_id
            dec_delays[0, step] = math.log(max(delay_ms, 1.0))

    return keystrokes


def _pad_context(ctx: List[int], length: int) -> List[int]:
    """Left-pad context to target length."""
    if len(ctx) >= length:
        return ctx[-length:]
    return [PAD_TOKEN] * (length - len(ctx)) + ctx


def generate_full_text(
    model: keras.Model,
    text: str,
    wpm: float = 100.0,
    temperature: float = 0.8,
    cfg: ModelConfig | None = None,
    seed: int | None = None,
) -> List[GeneratedKeystroke]:
    """Generate keystroke sequence for full text, split into segments."""
    if cfg is None:
        cfg = ModelConfig()
    if seed is not None:
        np.random.seed(seed)

    bucket = wpm_to_bucket(wpm)
    segments = split_text_into_inference_segments(text, seed=seed)

    all_keystrokes: List[GeneratedKeystroke] = []
    prev_context: List[int] = []
    cumulative_ms = 0.0

    for seg_text in segments:
        seg_keystrokes = generate_segment(
            model, seg_text, bucket, prev_context, cfg, temperature
        )

        # Adjust cumulative times
        for ks in seg_keystrokes:
            cumulative_ms += ks.delay_ms
            all_keystrokes.append(GeneratedKeystroke(ks.key, ks.delay_ms, cumulative_ms))

        # Update context
        if seg_keystrokes:
            prev_context = [
                char_to_id(ks.key) for ks in seg_keystrokes[-4:]
            ]

    return all_keystrokes


def print_sequence(keystrokes: List[GeneratedKeystroke], text: str) -> None:
    """Pretty-print generated keystroke sequence."""
    print(f"\nTarget: \"{text}\"")
    print(f"Keystrokes: {len(keystrokes)}")

    if keystrokes:
        total_ms = keystrokes[-1].cumulative_ms
        print(f"Time: {total_ms:.0f}ms ({total_ms/1000:.1f}s)")

    # Reconstruct
    buffer = []
    errors = 0
    backspaces = 0
    for ks in keystrokes:
        if ks.key == "BKSP":
            if buffer:
                buffer.pop()
            backspaces += 1
        else:
            buffer.append(ks.key)

    result = "".join(buffer)
    correct = len(keystrokes) - errors - backspaces
    print(f"Errors: {errors}  Backspaces: {backspaces}")
    print(f"Result: \"{result}\"")
    print(f"Match: {result == text}")

    print(f"\n{'#':>4}  {'Key':>6}  {'Delay':>8}  {'Cumul':>10}")
    print("-" * 40)
    for i, ks in enumerate(keystrokes):
        key_display = repr(ks.key) if ks.key != "BKSP" else "BKSP"
        print(f"{i+1:4d}  {key_display:>6s}  {ks.delay_ms:7.1f}ms  {ks.cumulative_ms:9.1f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate typing sequence (V2)")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("text", nargs="?", default=None)
    parser.add_argument("--wpm", type=float, default=100.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    text = args.text or "The quick brown fox jumps over the lazy dog."

    print("Loading model...")
    model = keras.models.load_model(str(args.model_path), compile=False)

    keystrokes = generate_full_text(model, text, wpm=args.wpm, temperature=args.temperature)
    print_sequence(keystrokes, text)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add ml/v2/generate.py
git commit -m "feat(v2): add autoregressive sequence generation"
```

---

## Task 8: Live Preview

**Files:**
- Create: `ml/v2/live_preview.py`

Real-time terminal playback using the V2 generator. Reuses the renderer pattern from V1's `live_preview.py`.

**Step 1: Write live preview**

Create `ml/v2/live_preview.py`:

```python
#!/usr/bin/env python3
"""Live preview of V2 phrase-level typing model.

Types text in real-time in the terminal with model-predicted timing,
errors, and corrections — all generated by the model as coherent sequences.

Usage:
    python -m v2.live_preview                              # interactive
    python -m v2.live_preview "Hello, world!"              # specific text
    python -m v2.live_preview --wpm 80 --speed 0.5 "text"  # slow playback
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")
from tensorflow import keras  # noqa: E402

from v2.config import ModelConfig  # noqa: E402
from v2.generate import generate_full_text, GeneratedKeystroke  # noqa: E402

RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"


def play_keystrokes(
    keystrokes: list[GeneratedKeystroke],
    target: str,
    speed: float = 1.0,
) -> None:
    """Play back keystrokes in real-time to the terminal."""
    buffer: list[str] = []
    errors = 0
    backspaces = 0
    correct = 0

    sys.stdout.write(CURSOR_HIDE)
    sys.stdout.write(f"\n{DIM}Target:{RESET} {target}\n\n")
    sys.stdout.write(f"{DIM}Typing:{RESET} ")
    sys.stdout.flush()

    start = time.time()

    for ks in keystrokes:
        time.sleep(ks.delay_ms / 1000.0 / speed)

        if ks.key == "BKSP":
            backspaces += 1
            if buffer:
                buffer.pop()
                sys.stdout.write("\b \b")
        elif ks.key in ("<END>", "<START>", ""):
            continue
        else:
            buffer.append(ks.key)
            # Color: red if it's going to be backspaced (heuristic: check if next is BKSP)
            # Simpler: just print normally, errors show as the model decided
            correct += 1
            sys.stdout.write(ks.key)

        sys.stdout.flush()

    elapsed = time.time() - start
    result = "".join(buffer)
    total_ks = correct + backspaces
    eff_wpm = (len(result) / 5.0) / (elapsed / 60.0) if elapsed > 0 else 0

    sys.stdout.write(CURSOR_SHOW)
    sys.stdout.write(f"\n\n{DIM}{'─' * 50}{RESET}\n")
    sys.stdout.write(f"  {BOLD}Session Complete{RESET}\n")
    sys.stdout.write(f"  Time:       {elapsed:.1f}s\n")
    sys.stdout.write(f"  Keystrokes: {total_ks}")
    sys.stdout.write(f"  ({GREEN}{correct} chars{RESET} {YELLOW}{backspaces} backspaces{RESET})\n")
    sys.stdout.write(f"  Effective:  {eff_wpm:.0f} WPM\n")
    match = result == target
    tag = f"{GREEN}YES{RESET}" if match else f"{RED}NO{RESET}"
    sys.stdout.write(f"  Match:      {tag}\n")
    if not match:
        sys.stdout.write(f"  Got:        \"{result}\"\n")
    sys.stdout.write(f"{DIM}{'─' * 50}{RESET}\n\n")
    sys.stdout.flush()


def find_latest_model() -> Path | None:
    """Find the most recent V2 best_model.keras."""
    from v2.config import MODEL_DIR
    runs = sorted(MODEL_DIR.glob("v2_run_*/best_model.keras"))
    return runs[-1] if runs else None


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 live typing preview")
    parser.add_argument("text", nargs="?", default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--wpm", type=float, default=100.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model_path = args.model or find_latest_model()
    if model_path is None or not model_path.exists():
        print("No trained V2 model found. Run v2.train first.")
        sys.exit(1)

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    print(f"{DIM}Loading model from {model_path.name}...{RESET}", end="", flush=True)
    model = keras.models.load_model(str(model_path), compile=False)
    print(f" {GREEN}done{RESET}")

    try:
        while True:
            if args.text:
                text = args.text
            else:
                print(f"\n{BOLD}Enter text to type{RESET} {DIM}(Ctrl+C to quit):{RESET}")
                try:
                    text = input("> ").strip()
                except EOFError:
                    break
                if not text:
                    continue

            text = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
            if not text:
                print(f"{YELLOW}No valid characters.{RESET}")
                if args.text:
                    break
                continue

            keystrokes = generate_full_text(
                model, text, wpm=args.wpm, temperature=args.temperature,
                seed=args.seed,
            )
            play_keystrokes(keystrokes, text, speed=args.speed)

            if args.text:
                break

    except KeyboardInterrupt:
        sys.stdout.write(CURSOR_SHOW)
        print(f"\n\n{DIM}Interrupted.{RESET}\n")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add ml/v2/live_preview.py
git commit -m "feat(v2): add real-time live preview for V2 model"
```

---

## Task 9: Evaluation

**Files:**
- Create: `ml/v2/evaluate.py`

V2 evaluation focuses on:
1. Character prediction accuracy (per-token, masked)
2. Timing MAE (log-space and ms)
3. Reconstruction accuracy (does generated text match target?)
4. Error rate distribution (compare model vs dataset)

**Step 1: Write evaluation script**

Create `ml/v2/evaluate.py`. Follow the same pattern as V1's `evaluate.py` — load best model, run on test set, save metrics JSON and plots.

Key metrics to compute:
- **Token accuracy**: % of decoder tokens correctly predicted (teacher-forced)
- **Timing MAE**: mean absolute error of predicted log-IKI vs actual
- **Reconstruction rate**: % of segments where autoregressive generation produces correct text
- **Error rate comparison**: model's error rate vs dataset's actual error rate

The exact code follows V1 patterns but uses V2 dataset and model. The script loads the model, iterates test batches for teacher-forced metrics, then generates a sample of segments autoregressively for reconstruction rate.

**Step 2: Commit**

```bash
git add ml/v2/evaluate.py
git commit -m "feat(v2): add evaluation script with reconstruction metrics"
```

---

## Task 10: Export to TF.js

**Files:**
- Create: `ml/v2/export.py`

Same pipeline as V1: Keras → SavedModel → TF.js graph model with uint8 quantization. The V2 model uses custom layers (`TypingTransformer`), so the export uses `model.export()` for the SavedModel step.

**Step 1: Write export script**

Follow V1's `export.py` exactly — it already handles custom models. Adjust paths to V2 directories.

**Step 2: Commit**

```bash
git add ml/v2/export.py
git commit -m "feat(v2): add TF.js export pipeline"
```

---

## Task 11: Full Pipeline Integration Test

**Step 1: Run full pipeline end-to-end with small data**

```bash
cd /Users/owengregson/Documents/Lilly

# Preprocess (small)
python -m v2.preprocess --max-files 5 --workers 2

# Train (small)
python -m v2.train --max-files 1 --max-samples 1000 --epochs 3 --batch-size 32 --run-name integration_test

# Generate
python -m v2.generate models/v2/integration_test/best_model.keras "Hello world"

# Live preview
python -m v2.live_preview --model models/v2/integration_test/best_model.keras "Hello world"
```

Expected: All commands complete without errors. Generated keystrokes print. Live preview plays back.

**Step 2: Run full preprocessing on all data**

```bash
python -m v2.preprocess --workers 8
```

Expected: Processes all 136M keystrokes into segment files. Reports total segment count (expect 5-10M).

**Step 3: Train full model**

```bash
python -m v2.train --epochs 30
```

Expected: Trains to convergence. Best model saved. Char accuracy should be >85% on validation set.

**Step 4: Evaluate and generate**

```bash
# Evaluate
python -m v2.evaluate models/v2/<run_name>/best_model.keras

# Live demo
python -m v2.live_preview
```

**Step 5: Commit final state**

```bash
git add -A ml/v2/
git commit -m "feat(v2): complete phrase-level typing model pipeline"
```

---

## Execution Order Summary

| Task | Description | Depends On |
|------|-------------|------------|
| 1 | V2 config | — |
| 2 | Segmentation logic + tests | 1 |
| 3 | Data preprocessing | 1, 2 |
| 4 | Dataset pipeline | 1, 3 |
| 5 | Model architecture | 1 |
| 6 | Training loop | 4, 5 |
| 7 | Inference / generation | 5 |
| 8 | Live preview | 7 |
| 9 | Evaluation | 4, 5, 7 |
| 10 | TF.js export | 5 |
| 11 | Integration test | all |

Tasks 1-2 are sequential. Tasks 3-5 can be done in parallel after 2. Tasks 6-10 depend on 5 being done. Task 11 is the final integration sweep.
