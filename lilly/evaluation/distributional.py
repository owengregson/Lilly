"""Tier 2 distributional metrics for V3 model evaluation.

Compares statistical distributions of generated vs real typing sequences
using Wasserstein distance, KS tests, and autocorrelation metrics.
"""

from __future__ import annotations

import numpy as np

from lilly.core.config import PAUSE_THRESHOLD_MS, V3ModelConfig
from lilly.inference.generator import generate_v3_full


def compute_tier2_metrics(
    model,
    dataset,
    cfg: V3ModelConfig,
    n_samples: int = 1000,
    style_vector: np.ndarray | None = None,
) -> dict:
    """Compute Tier 2 distributional metrics.

    Generates sequences autoregressively and compares distributions
    against real sequences from the dataset.

    Requires scipy for statistical tests.
    """
    from scipy.stats import ks_2samp, wasserstein_distance

    if style_vector is None:
        style_vector = np.zeros(cfg.style_dim, dtype=np.float32)

    # Collect real sequences from dataset
    real_ikis = []
    real_actions = []
    count = 0
    for batch_inputs, batch_labels in dataset:
        delays = batch_labels["delay_labels"].numpy()
        masks = batch_labels["label_mask"].numpy()
        actions = batch_labels["action_labels"].numpy()
        for b in range(len(masks)):
            for t in range(masks.shape[1]):
                if masks[b, t] > 0:
                    real_ikis.append(float(delays[b, t]))
                    real_actions.append(int(actions[b, t]))
            count += 1
            if count >= n_samples:
                break
        if count >= n_samples:
            break

    real_ikis = np.array(real_ikis)

    # Generate sequences
    gen_ikis = []
    gen_actions = []
    test_texts = [
        "the quick brown fox",
        "hello world",
        "typing test sentence",
        "a simple phrase",
        "the lazy dog jumps",
    ]

    for i in range(min(n_samples, 200)):
        text = test_texts[i % len(test_texts)]
        keystrokes = generate_v3_full(
            model, text, style_vector, cfg,
            temperatures={"action": 0.8, "timing": 0.8, "char": 0.8},
        )
        for ks in keystrokes:
            gen_ikis.append(float(np.log(max(ks.delay_ms, 1.0))))
            gen_actions.append(ks.action)

    gen_ikis = np.array(gen_ikis) if gen_ikis else np.zeros(1)

    metrics = {}

    # IKI distribution comparison
    if len(real_ikis) > 0 and len(gen_ikis) > 0:
        metrics["iki_wasserstein"] = float(wasserstein_distance(real_ikis, gen_ikis))
        ks_stat, ks_pval = ks_2samp(real_ikis, gen_ikis)
        metrics["iki_ks_statistic"] = float(ks_stat)
        metrics["iki_ks_pvalue"] = float(ks_pval)

    # Burst length comparison
    real_bursts = _extract_bursts(np.exp(real_ikis))
    gen_bursts = _extract_bursts(np.exp(gen_ikis))
    if len(real_bursts) > 0 and len(gen_bursts) > 0:
        metrics["burst_length_wasserstein"] = float(
            wasserstein_distance(real_bursts, gen_bursts)
        )

    # Pause duration comparison
    real_pauses = real_ikis[np.exp(real_ikis) > PAUSE_THRESHOLD_MS]
    gen_pauses = gen_ikis[np.exp(gen_ikis) > PAUSE_THRESHOLD_MS]
    if len(real_pauses) > 0 and len(gen_pauses) > 0:
        metrics["pause_duration_wasserstein"] = float(
            wasserstein_distance(real_pauses, gen_pauses)
        )

    # Correction latency comparison
    real_corrections = _extract_correction_latencies_from_actions(
        real_actions, np.exp(real_ikis)
    )
    gen_corrections = _extract_correction_latencies_from_actions(
        gen_actions, np.exp(gen_ikis)
    )
    if len(real_corrections) > 0 and len(gen_corrections) > 0:
        metrics["correction_latency_wasserstein"] = float(
            wasserstein_distance(real_corrections, gen_corrections)
        )

    # IKI autocorrelation comparison
    real_autocorr = _compute_autocorrelation(real_ikis)
    gen_autocorr = _compute_autocorrelation(gen_ikis)
    metrics["iki_autocorrelation_mae"] = float(abs(real_autocorr - gen_autocorr))

    return metrics


def _extract_bursts(ikis_ms: np.ndarray) -> list[int]:
    """Split IKIs at pauses, return burst lengths."""
    bursts = []
    current = 0
    for iki in ikis_ms:
        if iki >= PAUSE_THRESHOLD_MS:
            if current > 0:
                bursts.append(current)
            current = 1
        else:
            current += 1
    if current > 0:
        bursts.append(current)
    return bursts


def _extract_correction_latencies_from_actions(
    actions: list[int], ikis_ms: np.ndarray
) -> list[float]:
    """Find error->backspace delays."""
    latencies = []
    n = min(len(actions), len(ikis_ms))
    for i in range(n):
        if actions[i] == 1:  # error
            cumulative = 0.0
            for j in range(i + 1, min(i + 4, n)):
                cumulative += ikis_ms[j]
                if actions[j] == 2:  # backspace
                    latencies.append(cumulative)
                    break
    return latencies


def _compute_autocorrelation(ikis: np.ndarray) -> float:
    """Compute lag-1 autocorrelation of IKI series."""
    if len(ikis) < 3:
        return 0.0
    corr = np.corrcoef(ikis[:-1], ikis[1:])
    val = corr[0, 1]
    return float(val) if np.isfinite(val) else 0.0
