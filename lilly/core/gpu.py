"""GPU detection and training auto-tuning.

Detects available GPU hardware via TensorFlow and returns a GPUProfile
with recommended training hyperparameters. CLI flags always override.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

try:
    import tensorflow as tf
except ImportError:
    # Lightweight stub for environments without TF.
    # Provides attribute structure for test-time @patch() traversal.
    # detect_gpu() will return CPU_PROFILE when using this stub.
    tf = SimpleNamespace(  # type: ignore[assignment]
        config=SimpleNamespace(
            list_physical_devices=lambda *a: [],
            experimental=SimpleNamespace(
                get_device_details=lambda *a: {},
                get_memory_info=lambda *a: {},
            ),
        ),
    )

from lilly.core.config import V3TrainConfig


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


def _profile_from_vram(name: str, vram_gb: float) -> GPUProfile:
    """Create a profile for an unknown GPU based on its VRAM."""
    if vram_gb < 8:
        base = CPU_PROFILE
    elif vram_gb < 20:
        base = GPU_PROFILES["T4"]
    elif vram_gb < 35:
        base = GPU_PROFILES["A10"]
    else:
        base = GPU_PROFILES["A100-80"]
    return GPUProfile(
        name=name, vram_gb=vram_gb, batch_size=base.batch_size,
        shuffle_buffer=base.shuffle_buffer, prefetch_buffer=base.prefetch_buffer,
    )


def _match_gpu_name(device_name: str) -> str | None:
    """Match a TF device name string against known GPU profile keys."""
    upper = device_name.upper()
    # Check each profile key (ordered specific -> general)
    for key in ("A100-80", "A100-40", "H100", "A10", "L4", "T4"):
        if key == "A100-80" and "A100" in upper and "80" in upper:
            return key
        if key == "A100-40" and "A100" in upper and "80" not in upper:
            return key
        if key in ("H100", "A10", "L4", "T4") and key in upper:
            return key
    return None


def detect_gpu() -> GPUProfile:
    """Detect the available GPU and return its profile.

    Falls back to CPU_PROFILE if no GPU is found.
    Falls back to VRAM heuristic if GPU is not in the registry.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return CPU_PROFILE

    device = gpus[0]
    try:
        details = tf.config.experimental.get_device_details(device)
    except Exception:
        return CPU_PROFILE

    device_name = details.get("device_name", "")
    matched_key = _match_gpu_name(device_name)
    if matched_key is not None:
        return GPU_PROFILES[matched_key]

    # Try VRAM-based fallback
    try:
        mem_info = tf.config.experimental.get_memory_info(device.name)
        vram_gb = mem_info.get("total", 0) / (1024 ** 3)
        if vram_gb > 0:
            return _profile_from_vram(device_name, vram_gb)
    except Exception:
        pass

    # Last resort: unknown GPU, use conservative T4-tier defaults
    return _profile_from_vram(device_name, 16.0)


def auto_tune_config(
    train_cfg: V3TrainConfig,
    gpu_profile: GPUProfile,
    overrides: dict | None = None,
) -> V3TrainConfig:
    """Return a new V3TrainConfig tuned for the detected GPU.

    Fields from gpu_profile are applied first, then any CLI overrides
    take priority.
    """
    overrides = overrides or {}
    gpu_values = {
        "batch_size": gpu_profile.batch_size,
        "shuffle_buffer": gpu_profile.shuffle_buffer,
        "prefetch_buffer": gpu_profile.prefetch_buffer,
    }
    # CLI overrides win
    gpu_values.update(overrides)
    return replace(train_cfg, **gpu_values)
