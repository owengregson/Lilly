"""Tests for lilly.core.encoding — character encoding and WPM bucket mapping."""

import unittest

from lilly.core.encoding import char_to_id, id_to_char, wpm_to_bucket


class TestCharToId(unittest.TestCase):
    """Test char_to_id conversion."""

    def test_char_to_id_printable(self):
        """Printable ASCII maps to ord(c) - 31."""
        self.assertEqual(char_to_id(" "), 1)   # space (0x20) -> 1
        self.assertEqual(char_to_id("a"), 66)  # ord('a')=97, 97-31=66
        self.assertEqual(char_to_id("~"), 95)  # ord('~')=126, 126-31=95

    def test_char_to_id_backspace(self):
        """BKSP string maps to backspace token 96."""
        self.assertEqual(char_to_id("BKSP"), 96)

    def test_char_to_id_pad(self):
        """Empty string and other edge cases map to PAD (0)."""
        self.assertEqual(char_to_id(""), 0)
        self.assertEqual(char_to_id(""), 0)  # truly empty
        # Multi-character unknown strings (not BKSP) map to PAD
        self.assertEqual(char_to_id("XY"), 0)
        # Non-printable single character maps to PAD
        self.assertEqual(char_to_id("\x01"), 0)


class TestIdToChar(unittest.TestCase):
    """Test id_to_char conversion."""

    def test_id_to_char_roundtrip(self):
        """For all printable ASCII, char_to_id(id_to_char(id)) == id."""
        for code in range(32, 127):  # space through tilde
            ch = chr(code)
            cid = char_to_id(ch)
            self.assertEqual(id_to_char(cid), ch)
            self.assertEqual(char_to_id(id_to_char(cid)), cid)

    def test_id_to_char_special(self):
        """Special tokens decode correctly."""
        self.assertEqual(id_to_char(0), "")         # PAD
        self.assertEqual(id_to_char(96), "BKSP")    # BACKSPACE
        self.assertEqual(id_to_char(97), "<END>")    # END
        self.assertEqual(id_to_char(98), "<START>")  # START


class TestWpmToBucket(unittest.TestCase):
    """Test WPM to bucket mapping.

    Bucket edges: [0, 30, 45, 60, 75, 90, 105, 120, 140, 170, 999]
    """

    def test_wpm_to_bucket(self):
        self.assertEqual(wpm_to_bucket(25), 0)   # 0 <= 25 < 30
        self.assertEqual(wpm_to_bucket(50), 2)   # 45 <= 50 < 60
        self.assertEqual(wpm_to_bucket(100), 5)  # 90 <= 100 < 105
        self.assertEqual(wpm_to_bucket(200), 9)  # 170 <= 200 < 999


if __name__ == "__main__":
    unittest.main()
