"""V3 evaluation visualization suite.

Creates matplotlib plots for IKI distributions, burst patterns,
action confusion matrices, MDN components, and style interpolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def plot_iki_distributions(
    real_ikis: np.ndarray,
    gen_ikis: np.ndarray,
    save_path: Path,
    real_actions: np.ndarray | None = None,
    gen_actions: np.ndarray | None = None,
) -> None:
    """Plot per-action IKI histograms comparing real vs generated."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3 if real_actions is not None else 1,
                              figsize=(15, 5))
    if real_actions is None:
        axes = [axes]

    action_names = ["correct", "error", "backspace"]

    if real_actions is not None:
        for idx, name in enumerate(action_names):
            ax = axes[idx]
            real_mask = real_actions == idx
            gen_mask = gen_actions == idx
            if np.any(real_mask):
                ax.hist(real_ikis[real_mask], bins=50, alpha=0.5, label="Real", density=True)
            if np.any(gen_mask):
                ax.hist(gen_ikis[gen_mask], bins=50, alpha=0.5, label="Generated", density=True)
            ax.set_title(f"IKI Distribution ({name})")
            ax.set_xlabel("log(IKI)")
            ax.set_ylabel("Density")
            ax.legend()
    else:
        axes[0].hist(real_ikis, bins=50, alpha=0.5, label="Real", density=True)
        axes[0].hist(gen_ikis, bins=50, alpha=0.5, label="Generated", density=True)
        axes[0].set_title("IKI Distribution")
        axes[0].set_xlabel("log(IKI)")
        axes[0].legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_burst_patterns(
    real_ikis: np.ndarray,
    gen_ikis: np.ndarray,
    save_path: Path,
    n_points: int = 200,
) -> None:
    """Plot time series of IKI values (burst patterns)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    n = min(n_points, len(real_ikis))
    ax1.plot(np.exp(real_ikis[:n]), linewidth=0.5)
    ax1.set_title("Real Typing (IKI time series)")
    ax1.set_ylabel("IKI (ms)")
    ax1.set_ylim(0, 1000)

    n = min(n_points, len(gen_ikis))
    ax2.plot(np.exp(gen_ikis[:n]), linewidth=0.5, color="orange")
    ax2.set_title("Generated Typing (IKI time series)")
    ax2.set_ylabel("IKI (ms)")
    ax2.set_xlabel("Keystroke index")
    ax2.set_ylim(0, 1000)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_action_confusion(
    pred_actions: np.ndarray,
    true_actions: np.ndarray,
    save_path: Path,
) -> None:
    """Plot action prediction confusion matrix."""
    import matplotlib.pyplot as plt

    labels = ["correct", "error", "backspace"]
    n_classes = 3
    matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

    for t, p in zip(true_actions, pred_actions):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            matrix[t, p] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Action Confusion Matrix")

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_mdn_components(
    model,
    save_path: Path,
) -> None:
    """Visualize learned MDN components for each action head."""
    import matplotlib.pyplot as plt
    import tensorflow as tf

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    action_names = ["correct", "error", "backspace"]

    # Create dummy input to get MDN outputs
    from lilly.core.config import V3ModelConfig
    cfg = V3ModelConfig()
    dec_len = cfg.max_decoder_len + 1
    dummy = {
        "encoder_chars": tf.zeros((1, cfg.max_encoder_len), dtype=tf.int32),
        "encoder_lengths": tf.constant([[5]], dtype=tf.int32),
        "decoder_input_chars": tf.zeros((1, dec_len), dtype=tf.int32),
        "decoder_input_delays": tf.zeros((1, dec_len), dtype=tf.float32),
        "decoder_input_actions": tf.zeros((1, dec_len), dtype=tf.int32),
        "style_vector": tf.zeros((1, cfg.style_dim)),
        "prev_context_chars": tf.zeros((1, cfg.context_tail_len), dtype=tf.int32),
        "prev_context_actions": tf.zeros((1, cfg.context_tail_len), dtype=tf.int32),
        "prev_context_delays": tf.zeros((1, cfg.context_tail_len), dtype=tf.float32),
    }
    outputs = model(dummy, training=False)

    x = np.linspace(1.0, 8.0, 200)  # log-space IKI range

    for idx, name in enumerate(action_names):
        key = f"timing_{name}"
        pi = outputs[key][0][0, 0].numpy()
        mu = outputs[key][1][0, 0].numpy()
        log_sigma = outputs[key][2][0, 0].numpy()

        ax = axes[idx]
        mixture = np.zeros_like(x)
        for k in range(len(pi)):
            sigma = np.exp(log_sigma[k])
            component = pi[k] * np.exp(-0.5 * ((x - mu[k]) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            ax.plot(x, component, alpha=0.3, label=f"k={k}")
            mixture += component
        ax.plot(x, mixture, "k-", linewidth=2, label="mixture")
        ax.set_title(f"MDN Components ({name})")
        ax.set_xlabel("log(IKI)")
        ax.set_ylabel("Density")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_style_interpolation(
    model,
    cfg,
    save_path: Path,
    n_points: int = 5,
) -> None:
    """Generate at interpolated styles and plot IKI distributions."""
    import matplotlib.pyplot as plt
    from lilly.inference.generator import generate_v3_full

    fig, ax = plt.subplots(figsize=(10, 6))
    text = "the quick brown fox"

    for i in range(n_points):
        style = np.zeros(cfg.style_dim, dtype=np.float32)
        # Vary mean_iki_log from fast to slow
        style[0] = 3.0 + i * 0.5  # log-IKI from ~20ms to ~150ms
        keystrokes = generate_v3_full(
            model, text, style, cfg,
            temperatures={"action": 0.5, "timing": 0.5, "char": 0.5},
        )
        ikis = [np.log(max(ks.delay_ms, 1.0)) for ks in keystrokes]
        if ikis:
            ax.hist(ikis, bins=30, alpha=0.3, label=f"style[0]={style[0]:.1f}")

    ax.set_title("Style Interpolation (varying mean_iki_log)")
    ax.set_xlabel("log(IKI)")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
