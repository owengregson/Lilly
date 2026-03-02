"""Tests for lilly.data.segment — pause-based keystroke segmentation.

Adapted from v2/test_segment.py with imports from the restructured package.
"""

import unittest

from lilly.core.config import BACKSPACE_TOKEN, END_TOKEN, PAD_TOKEN, START_TOKEN
from lilly.core.encoding import char_to_id, id_to_char
from lilly.data.segment import (
    Segment,
    compute_target_text_for_segment,
    extract_segments,
    split_text_into_inference_segments,
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

class TestCharEncoding(unittest.TestCase):
    """Character encoding sanity checks (via lilly.core.encoding)."""

    def test_char_to_id_printable(self):
        self.assertEqual(char_to_id("a"), ord("a") - 31)
        self.assertEqual(char_to_id(" "), ord(" ") - 31)  # space = 1
        self.assertEqual(char_to_id("~"), ord("~") - 31)  # tilde = 95

    def test_char_to_id_special(self):
        self.assertEqual(char_to_id("BKSP"), BACKSPACE_TOKEN)  # 96
        self.assertEqual(char_to_id(""), PAD_TOKEN)  # 0

    def test_id_to_char_roundtrip(self):
        for ch in "abcXYZ 123!@#":
            self.assertEqual(id_to_char(char_to_id(ch)), ch)
        self.assertEqual(id_to_char(BACKSPACE_TOKEN), "BKSP")
        self.assertEqual(id_to_char(END_TOKEN), "<END>")
        self.assertEqual(id_to_char(START_TOKEN), "<START>")
        self.assertEqual(id_to_char(PAD_TOKEN), "")


# ---------------------------------------------------------------------------
# Target text extraction from keystroke segment
# ---------------------------------------------------------------------------

class TestComputeTargetText(unittest.TestCase):
    """Test compute_target_text_for_segment."""

    def test_compute_target_clean(self):
        """Clean typing: target = joined typed keys."""
        keystrokes = [
            make_keystroke("h", "h", 0, 80, 0, 1),
            make_keystroke("i", "i", 0, 90, 1, 2),
        ]
        self.assertEqual(compute_target_text_for_segment(keystrokes, "hi there"), "hi")

    def test_compute_target_with_error_correction(self):
        """Error + backspace + retype: target still = 'hi'."""
        keystrokes = [
            make_keystroke("h", "h", 0, 80, 0, 1),
            make_keystroke("o", "i", 1, 70, 1, 2),   # error
            make_keystroke("BKSP", "", 2, 180, 1, 1),  # backspace
            make_keystroke("i", "i", 0, 90, 1, 2),    # retype
        ]
        self.assertEqual(compute_target_text_for_segment(keystrokes, "hi there"), "hi")

    def test_compute_target_with_space(self):
        """Target spans a word boundary including space."""
        keystrokes = [
            make_keystroke("h", "h", 0, 80, 0, 1),
            make_keystroke("i", "i", 0, 90, 1, 2),
            make_keystroke(" ", " ", 0, 120, 2, 3),
        ]
        self.assertEqual(compute_target_text_for_segment(keystrokes, "hi there"), "hi ")


# ---------------------------------------------------------------------------
# Segment extraction from session
# ---------------------------------------------------------------------------

class TestExtractSegments(unittest.TestCase):
    """Test extract_segments splitting and filtering."""

    def test_extract_segments_single_burst(self):
        """All keystrokes < pause threshold = one segment."""
        keystrokes = [
            make_keystroke("h", "h", 0, 80, 0, 1),
            make_keystroke("e", "e", 0, 85, 1, 2),
            make_keystroke("y", "y", 0, 90, 2, 3),
        ]
        segments = extract_segments(keystrokes, "hey", wpm_bucket=5, session_len=3)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].target_text, "hey")
        self.assertEqual(len(segments[0].keystroke_char_ids), 3)
        self.assertEqual(len(segments[0].keystroke_delays), 3)

    def test_extract_segments_pause_split(self):
        """A 400ms gap splits into two segments."""
        keystrokes = [
            make_keystroke("h", "h", 0, 80, 0, 1),
            make_keystroke("e", "e", 0, 85, 1, 2),
            make_keystroke("y", "y", 0, 90, 2, 3),
            make_keystroke(" ", " ", 0, 400, 3, 4),  # pause > 300ms
            make_keystroke("y", "y", 0, 85, 4, 5),
            make_keystroke("o", "o", 0, 95, 5, 6),
            make_keystroke("u", "u", 0, 80, 6, 7),
        ]
        segments = extract_segments(keystrokes, "hey you", wpm_bucket=5, session_len=7)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].target_text, "hey")
        self.assertEqual(segments[1].target_text, " you")

    def test_extract_segments_includes_errors(self):
        """Error keystrokes are included in the segment's keystroke sequence."""
        keystrokes = [
            make_keystroke("h", "h", 0, 80, 0, 1),
            make_keystroke("o", "i", 1, 70, 1, 2),   # error: typed 'o' instead of 'i'
            make_keystroke("BKSP", "", 2, 180, 1, 1),  # backspace
            make_keystroke("i", "i", 0, 90, 1, 2),    # retype correct
        ]
        segments = extract_segments(keystrokes, "hi", wpm_bucket=5, session_len=4)
        self.assertEqual(len(segments), 1)
        seg = segments[0]
        self.assertEqual(seg.target_text, "hi")
        # 4 keystrokes: h, o(error), BKSP, i
        self.assertEqual(len(seg.keystroke_char_ids), 4)
        self.assertEqual(seg.keystroke_char_ids[1], char_to_id("o"))  # the error char
        self.assertEqual(seg.keystroke_char_ids[2], BACKSPACE_TOKEN)

    def test_extract_segments_context_propagation(self):
        """Second segment gets tail context from first segment."""
        keystrokes = [
            make_keystroke("a", "a", 0, 80, 0, 1),
            make_keystroke("b", "b", 0, 90, 1, 2),
            make_keystroke("c", "c", 0, 85, 2, 3),
            make_keystroke("d", "d", 0, 400, 3, 4),  # pause
            make_keystroke("e", "e", 0, 80, 4, 5),
            make_keystroke("f", "f", 0, 85, 5, 6),
            make_keystroke("g", "g", 0, 90, 6, 7),
        ]
        segments = extract_segments(keystrokes, "abcdefg", wpm_bucket=5, session_len=7)
        self.assertEqual(len(segments), 2)
        # Second segment should have context from first
        self.assertGreater(len(segments[1].prev_context), 0)

    def test_short_segments_filtered(self):
        """Segments with < MIN_SEGMENT_KEYSTROKES are discarded."""
        keystrokes = [
            make_keystroke("a", "a", 0, 80, 0, 1),
            make_keystroke("b", "b", 0, 400, 1, 2),  # pause -> segment 1 = just "a" (1 ks)
            make_keystroke("c", "c", 0, 80, 2, 3),
            make_keystroke("d", "d", 0, 90, 3, 4),
            make_keystroke("e", "e", 0, 85, 4, 5),
        ]
        segments = extract_segments(keystrokes, "abcde", wpm_bucket=5, session_len=5)
        # First segment (just "a") should be filtered out (< 3 keystrokes)
        for s in segments:
            self.assertGreaterEqual(len(s.keystroke_char_ids), 3)


# ---------------------------------------------------------------------------
# Inference-time text splitting
# ---------------------------------------------------------------------------

class TestSplitTextInference(unittest.TestCase):
    """Test split_text_into_inference_segments."""

    def test_split_text_simple(self):
        """Split short text into a single segment."""
        segments = split_text_into_inference_segments("hi", seed=42)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], "hi")

    def test_split_text_multiple_words(self):
        """Longer text gets split into multi-word segments."""
        text = "the quick brown fox jumps over the lazy dog"
        segments = split_text_into_inference_segments(text, seed=42)
        self.assertGreaterEqual(len(segments), 2)
        # Rejoined should equal original
        self.assertEqual("".join(segments), text)

    def test_split_text_preserves_all_chars(self):
        """No characters lost during splitting."""
        text = "Hello, world! This is a test."
        segments = split_text_into_inference_segments(text, seed=42)
        self.assertEqual("".join(segments), text)


if __name__ == "__main__":
    unittest.main()
