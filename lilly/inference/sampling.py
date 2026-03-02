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
    if temperature != 1.0:
        logits = np.log(probs + 1e-10) / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))
    return int(np.random.choice(len(probs), p=probs))


def weighted_sample_logits(logits: np.ndarray, temperature: float = 1.0) -> int:
    """Sample from logits (unnormalized) with temperature."""
    if temperature != 1.0:
        logits = logits / temperature
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))
