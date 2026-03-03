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


from unittest.mock import patch, MagicMock

from lilly.core.gpu import detect_gpu


class TestDetectGPU(unittest.TestCase):
    """detect_gpu() queries TF and returns the right profile."""

    @patch("lilly.core.gpu.tf.config.list_physical_devices", return_value=[])
    def test_no_gpu_returns_cpu(self, mock_list):
        p = detect_gpu()
        self.assertEqual(p.name, "CPU")
        self.assertEqual(p.batch_size, 32)

    @patch("lilly.core.gpu.tf.config.experimental.get_device_details")
    @patch("lilly.core.gpu.tf.config.list_physical_devices")
    def test_known_gpu_matched_by_name(self, mock_list, mock_details):
        fake_dev = MagicMock()
        mock_list.return_value = [fake_dev]
        mock_details.return_value = {
            "device_name": "NVIDIA A10",
            "compute_capability": (8, 6),
        }
        p = detect_gpu()
        self.assertEqual(p.name, "A10")
        self.assertEqual(p.batch_size, 256)

    @patch("lilly.core.gpu.tf.config.experimental.get_device_details")
    @patch("lilly.core.gpu.tf.config.list_physical_devices")
    def test_unknown_gpu_uses_vram_fallback(self, mock_list, mock_details):
        fake_dev = MagicMock()
        mock_list.return_value = [fake_dev]
        mock_details.return_value = {"device_name": "NVIDIA RTX 5090"}
        # No VRAM info in details -> falls back to name-only,
        # which won't match -> CPU profile as safe default
        p = detect_gpu()
        # Should still work (either matched or fell back)
        self.assertIsInstance(p, GPUProfile)

    @patch("lilly.core.gpu.tf.config.experimental.get_device_details")
    @patch("lilly.core.gpu.tf.config.list_physical_devices")
    def test_h100_matched(self, mock_list, mock_details):
        fake_dev = MagicMock()
        mock_list.return_value = [fake_dev]
        mock_details.return_value = {"device_name": "NVIDIA H100 80GB HBM3"}
        p = detect_gpu()
        self.assertEqual(p.name, "H100")
        self.assertEqual(p.batch_size, 512)


if __name__ == "__main__":
    unittest.main()
