"""Tests for GPU auto-tuning — lilly.core.gpu and config changes."""

import unittest

from lilly.core.config import V3TrainConfig


class TestV3TrainConfigPrefetch(unittest.TestCase):
    """V3TrainConfig should have a prefetch_buffer field."""

    def test_default_prefetch_buffer_is_zero(self):
        cfg = V3TrainConfig()
        self.assertEqual(cfg.prefetch_buffer, 0)

    def test_prefetch_buffer_custom_value(self):
        cfg = V3TrainConfig(prefetch_buffer=8)
        self.assertEqual(cfg.prefetch_buffer, 8)


if __name__ == "__main__":
    unittest.main()
