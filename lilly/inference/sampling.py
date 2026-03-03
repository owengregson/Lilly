"""Shared sampling utilities for model inference."""

from __future__ import annotations

import math

import numpy as np

from lilly.core.config import MAX_IKI_MS, MIN_IKI_MS


def sample_lognormal(mu: float, log_sigma: float) -> float:
    """Sample from LogNormal given mu and log_sigma in log-space."""
    sigma = math.exp(np.clip(log_sigma, -5.0, 5.0))
    log_val = np.random.normal(mu, sigma)
    return float(np.clip(np.exp(log_val), MIN_IKI_MS, MAX_IKI_MS))


def weighted_sample(probs: np.ndarray, temperature: float = 1.0) -> int:
    """Sample from a probability distribution with temperature."""
    probs = np.asarray(probs, dtype=np.float64)
    if temperature != 1.0:
        logits = np.log(probs + 1e-10) / temperature
        logits = logits - np.max(logits)  # numerical stability
        probs = np.exp(logits) / np.sum(np.exp(logits))
    else:
        # Renormalize to prevent float32 precision drift from softmax
        probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def weighted_sample_logits(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Sample from logits (unnormalized) with temperature."""
    if temperature != 1.0:
        logits = logits / temperature
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


def sample_mdn(
    pi: np.ndarray, mu: np.ndarray, log_sigma: np.ndarray,
    temperature: float = 1.0,
) -> tuple[float, int]:
    """Sample from an MDN mixture distribution.

    Args:
        pi: (K,) mixture weights
        mu: (K,) component means (log-space)
        log_sigma: (K,) component log-std
        temperature: scaling for mixture weights and sigma

    Returns:
        (delay_ms, component_index)
    """
    # Apply temperature to mixture weights
    pi = np.asarray(pi, dtype=np.float64)
    if temperature != 1.0:
        log_pi = np.log(pi + 1e-10) / temperature
        pi = np.exp(log_pi - np.max(log_pi))
    pi = pi / np.sum(pi)  # always renormalize

    k = int(np.random.choice(len(pi), p=pi))
    sigma = math.exp(np.clip(log_sigma[k], -5.0, 5.0))
    if temperature != 1.0:
        sigma *= temperature
    z = np.random.normal(mu[k], sigma)
    delay_ms = float(np.clip(np.exp(z), MIN_IKI_MS, MAX_IKI_MS))
    return delay_ms, k
