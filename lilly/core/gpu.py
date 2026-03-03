"""GPU detection and training auto-tuning.

Detects available GPU hardware via TensorFlow and returns a GPUProfile
with recommended training hyperparameters. CLI flags always override.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUProfile:
    """Recommended training settings for a specific GPU."""

    name: str
    vram_gb: float
    batch_size: int
    shuffle_buffer: int
    prefetch_buffer: int  # 0 = tf.data.AUTOTUNE


# ── Known profiles ────────────────────────────────────────────────────────
CPU_PROFILE = GPUProfile(
    name="CPU", vram_gb=0, batch_size=32,
    shuffle_buffer=10_000, prefetch_buffer=2,
)

GPU_PROFILES: dict[str, GPUProfile] = {
    "T4": GPUProfile(
        name="T4", vram_gb=16, batch_size=128,
        shuffle_buffer=50_000, prefetch_buffer=4,
    ),
    "A10": GPUProfile(
        name="A10", vram_gb=24, batch_size=256,
        shuffle_buffer=100_000, prefetch_buffer=8,
    ),
    "L4": GPUProfile(
        name="L4", vram_gb=24, batch_size=256,
        shuffle_buffer=100_000, prefetch_buffer=8,
    ),
    "A100-40": GPUProfile(
        name="A100-40", vram_gb=40, batch_size=512,
        shuffle_buffer=200_000, prefetch_buffer=16,
    ),
    "A100-80": GPUProfile(
        name="A100-80", vram_gb=80, batch_size=512,
        shuffle_buffer=200_000, prefetch_buffer=16,
    ),
    "H100": GPUProfile(
        name="H100", vram_gb=80, batch_size=512,
        shuffle_buffer=200_000, prefetch_buffer=16,
    ),
}
