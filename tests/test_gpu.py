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


from lilly.core.gpu import GPUProfile, GPU_PROFILES, CPU_PROFILE


class TestGPUProfile(unittest.TestCase):
    """GPUProfile dataclass and registry."""

    def test_cpu_profile_exists(self):
        self.assertEqual(CPU_PROFILE.name, "CPU")
        self.assertEqual(CPU_PROFILE.vram_gb, 0)
        self.assertEqual(CPU_PROFILE.batch_size, 32)

    def test_known_profiles_registry(self):
        self.assertIn("A10", GPU_PROFILES)
        self.assertIn("T4", GPU_PROFILES)
        self.assertIn("H100", GPU_PROFILES)

    def test_a10_profile_values(self):
        p = GPU_PROFILES["A10"]
        self.assertEqual(p.batch_size, 256)
        self.assertEqual(p.shuffle_buffer, 100_000)
        self.assertEqual(p.prefetch_buffer, 8)

    def test_h100_profile_values(self):
        p = GPU_PROFILES["H100"]
        self.assertEqual(p.batch_size, 512)
        self.assertEqual(p.shuffle_buffer, 200_000)


from lilly.core.gpu import _profile_from_vram


class TestVRAMFallback(unittest.TestCase):
    """VRAM-based fallback for unknown GPUs."""

    def test_low_vram_gets_cpu_profile(self):
        p = _profile_from_vram("Unknown GPU", 4.0)
        self.assertEqual(p.batch_size, 32)  # CPU tier

    def test_medium_vram_gets_t4_profile(self):
        p = _profile_from_vram("Unknown GPU", 12.0)
        self.assertEqual(p.batch_size, 128)  # T4 tier

    def test_large_vram_gets_a10_profile(self):
        p = _profile_from_vram("Unknown GPU", 24.0)
        self.assertEqual(p.batch_size, 256)  # A10 tier

    def test_very_large_vram_gets_a100_profile(self):
        p = _profile_from_vram("Unknown GPU", 48.0)
        self.assertEqual(p.batch_size, 512)  # A100 tier

    def test_preserves_gpu_name(self):
        p = _profile_from_vram("RTX 4090", 24.0)
        self.assertEqual(p.name, "RTX 4090")


if __name__ == "__main__":
    unittest.main()
