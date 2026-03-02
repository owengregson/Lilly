"""Tests for lilly.data.segment — inference-time text segmentation."""

import unittest

from lilly.data.segment import split_text_into_inference_segments


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
