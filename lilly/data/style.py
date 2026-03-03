"""Style vector computation for V3 typing persona conditioning.

Computes a 16-dimensional style vector from session-level typing statistics.
See the design doc for feature descriptions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lilly.core.config import (
    ACTION_BACKSPACE,
    ACTION_ERROR,
    PAUSE_THRESHOLD_MS,
    STYLE_DIM,
)


def compute_style_vector(df: pd.DataFrame) -> np.ndarray:
    """Compute 16-dim style vector from a session DataFrame.

    Args:
        df: DataFrame with columns: action, iki, hold_time, typed_key,
            target_char, target_pos, wpm

    Returns:
        (16,) float32 array
    """
    vec = np.zeros(STYLE_DIM, dtype=np.float32)

    if len(df) < 3:
        return vec

    ikis = df["iki"].values.astype(np.float64)
    actions = df["action"].values
    if "hold_time" in df.columns:
        hold_times = df["hold_time"].values.astype(np.float64)
    else:
        hold_times = np.zeros(len(df))
    typed_keys = df["typed_key"].values

    # Clamp IKIs to valid range
    ikis_clipped = np.clip(ikis, 10.0, 5000.0)
    log_ikis = np.log(ikis_clipped)

    # dim 0: mean_iki_log
    vec[0] = float(np.mean(log_ikis))

    # dim 1: std_iki_log (rhythm regularity)
    vec[1] = float(np.std(log_ikis)) if len(log_ikis) > 1 else 0.0

    # dim 2: median_iki_log (robust central tendency)
    vec[2] = float(np.median(log_ikis))

    # dim 3: iki_skewness (clamped to [-10, 10] to prevent extreme values)
    if len(log_ikis) > 2 and np.std(log_ikis) > 1e-6:
        mean_val = np.mean(log_ikis)
        std_val = np.std(log_ikis)
        skew = float(np.mean(((log_ikis - mean_val) / std_val) ** 3))
        vec[3] = float(np.clip(skew, -10.0, 10.0))
    else:
        vec[3] = 0.0

    # dim 4-5: mean_burst_length, std_burst_length
    burst_lengths = _compute_burst_lengths(ikis, PAUSE_THRESHOLD_MS)
    if len(burst_lengths) > 0:
        vec[4] = float(np.mean(burst_lengths))
        vec[5] = float(np.std(burst_lengths)) if len(burst_lengths) > 1 else 0.0

    # dim 6: error_rate
    n_total = len(actions)
    n_errors = np.sum(actions == ACTION_ERROR)
    vec[6] = float(n_errors / max(n_total, 1))

    # dim 7: correction_rate (fraction of errors corrected within 3 keystrokes)
    if n_errors > 0:
        corrections = 0
        for i in range(len(actions)):
            if actions[i] == ACTION_ERROR:
                # Check if backspace follows within 3 keystrokes
                for j in range(i + 1, min(i + 4, len(actions))):
                    if actions[j] == ACTION_BACKSPACE:
                        corrections += 1
                        break
        vec[7] = float(corrections / n_errors)

    # dim 8: mean_correction_latency_log
    correction_latencies = _compute_correction_latencies(actions, ikis)
    if len(correction_latencies) > 0:
        vec[8] = float(np.mean(np.log(np.clip(correction_latencies, 10.0, 5000.0))))

    # dim 9: bigram_speed_variance
    vec[9] = _compute_bigram_speed_variance(typed_keys, ikis)

    # dim 10: pause_frequency
    n_pauses = np.sum(ikis > PAUSE_THRESHOLD_MS)
    vec[10] = float(n_pauses / max(n_total, 1))

    # dim 11: mean_hold_time_log (filter out missing/zero hold times before clipping)
    valid_hold_mask = hold_times > 0
    if np.any(valid_hold_mask):
        hold_valid = np.clip(hold_times[valid_hold_mask], 5.0, 2000.0)
        vec[11] = float(np.mean(np.log(hold_valid)))

    # dim 12: iki_autocorrelation (lag-1)
    if len(log_ikis) > 2:
        corr = np.corrcoef(log_ikis[:-1], log_ikis[1:])
        vec[12] = float(corr[0, 1]) if np.isfinite(corr[0, 1]) else 0.0

    # dim 13: word_boundary_slowdown (space-IKI / within-word-IKI ratio)
    space_mask = np.array([k == " " for k in typed_keys])
    if np.any(space_mask) and np.any(~space_mask):
        space_ikis = ikis_clipped[space_mask]
        nonspace_ikis = ikis_clipped[~space_mask]
        mean_nonspace = np.mean(nonspace_ikis)
        if mean_nonspace > 0:
            vec[13] = float(np.mean(space_ikis) / mean_nonspace)

    # dim 14: error_burst_rate (fraction of errors that are clustered)
    if n_errors > 1:
        error_indices = np.where(actions == ACTION_ERROR)[0]
        clustered = sum(1 for i in range(1, len(error_indices))
                       if error_indices[i] - error_indices[i-1] <= 3)
        vec[14] = float(clustered / len(error_indices))

    # dim 15: session_speedup_trend (linear slope of rolling IKI)
    if len(log_ikis) > 10:
        window = min(20, len(log_ikis) // 2)
        rolling = np.convolve(log_ikis, np.ones(window)/window, mode='valid')
        if len(rolling) > 1:
            x = np.arange(len(rolling), dtype=np.float64)
            slope = np.polyfit(x, rolling, 1)[0]
            vec[15] = float(slope)

    return vec


def _compute_burst_lengths(ikis: np.ndarray, threshold: float) -> list[int]:
    """Compute burst lengths (keystrokes between pauses)."""
    bursts = []
    current = 0
    for iki in ikis:
        if iki >= threshold:
            if current > 0:
                bursts.append(current)
            current = 1
        else:
            current += 1
    if current > 0:
        bursts.append(current)
    return bursts


def _compute_correction_latencies(actions: np.ndarray, ikis: np.ndarray) -> list[float]:
    """Compute error-to-backspace latencies."""
    latencies = []
    for i in range(len(actions)):
        if actions[i] == ACTION_ERROR:
            cumulative = 0.0
            for j in range(i + 1, min(i + 4, len(actions))):
                cumulative += ikis[j]
                if actions[j] == ACTION_BACKSPACE:
                    latencies.append(cumulative)
                    break
    return latencies


def _compute_bigram_speed_variance(typed_keys: np.ndarray, ikis: np.ndarray) -> float:
    """Compute variance of per-bigram mean IKIs."""
    if len(typed_keys) < 2:
        return 0.0

    bigram_ikis: dict[str, list[float]] = {}
    for i in range(1, len(typed_keys)):
        bigram = f"{typed_keys[i-1]}{typed_keys[i]}"
        bigram_ikis.setdefault(bigram, []).append(float(ikis[i]))

    # Only use bigrams with enough samples
    means = [np.mean(v) for v in bigram_ikis.values() if len(v) >= 2]
    if len(means) < 2:
        return 0.0
    return float(np.var(means))


@dataclass
class StyleNormalizer:
    """Z-score normalizer for style vectors."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, vectors: np.ndarray) -> StyleNormalizer:
        """Fit normalizer from an array of style vectors."""
        mean = vectors.mean(axis=0).astype(np.float32)
        std = vectors.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0  # avoid division by zero
        return cls(mean=mean, std=std)

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Apply z-score normalization."""
        return ((vectors - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, normed: np.ndarray) -> np.ndarray:
        """Reverse z-score normalization."""
        return (normed * self.std + self.mean).astype(np.float32)

    def save(self, path: Path) -> None:
        """Save normalizer to JSON."""
        data = {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> StyleNormalizer:
        """Load normalizer from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            mean=np.array(data["mean"], dtype=np.float32),
            std=np.array(data["std"], dtype=np.float32),
        )
