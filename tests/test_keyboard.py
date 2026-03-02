"""Tests for lilly.core.keyboard — QWERTY layout utilities."""

import unittest

from lilly.core.keyboard import (
    SHIFT_MAP,
    get_neighbors,
    key_distance,
    same_finger,
    same_hand,
)


class TestKeyDistance(unittest.TestCase):
    """Test key_distance calculations."""

    def test_key_distance_same_key(self):
        """Distance from a key to itself is zero."""
        self.assertEqual(key_distance("a", "a"), 0.0)

    def test_key_distance_adjacent(self):
        """Adjacent keys have a small positive distance."""
        dist = key_distance("a", "s")
        self.assertGreater(dist, 0)
        self.assertLess(dist, 2.0)

    def test_key_distance_unknown(self):
        """Unknown keys return distance 0.0."""
        self.assertEqual(key_distance("a", "\u00a7"), 0.0)


class TestGetNeighbors(unittest.TestCase):
    """Test neighbor lookup."""

    def test_get_neighbors(self):
        """'f' should have neighbors including 'd', 'g', 'r', 'v'."""
        neighbors = get_neighbors("f")
        for expected in ["d", "g", "r", "v"]:
            self.assertIn(expected, neighbors)


class TestFingerAndHand(unittest.TestCase):
    """Test same_finger and same_hand checks."""

    def test_same_finger(self):
        """'e' and 'd' are both typed with finger index 2."""
        self.assertTrue(same_finger("e", "d"))

    def test_same_hand(self):
        """'a' and 's' are both left hand; 'a' and 'j' are different hands."""
        self.assertTrue(same_hand("a", "s"))
        self.assertFalse(same_hand("a", "j"))


class TestShiftMap(unittest.TestCase):
    """Test shifted character mapping."""

    def test_shift_map(self):
        """Shifted characters map to their base keys."""
        self.assertEqual(SHIFT_MAP["A"], "a")
        self.assertEqual(SHIFT_MAP["!"], "1")
        self.assertEqual(SHIFT_MAP["@"], "2")


if __name__ == "__main__":
    unittest.main()
