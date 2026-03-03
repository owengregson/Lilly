"""Tier 3 realism metrics for V3 model evaluation.

Includes discriminator-based realism scoring and style consistency checks.
"""

from __future__ import annotations

import numpy as np
from tensorflow import keras

from lilly.core.config import V3ModelConfig
from lilly.inference.generator import generate_v3_full


def train_discriminator(
    real_sequences: list[np.ndarray],
    generated_sequences: list[np.ndarray],
    max_len: int = 200,
) -> float:
    """Train a small LSTM classifier to distinguish real vs generated sequences.

    Args:
        real_sequences: List of (N, 3) arrays [delay_log, action, is_space]
        generated_sequences: List of (N, 3) arrays
        max_len: Maximum sequence length for padding

    Returns:
        Test accuracy (0.5 = indistinguishable = perfect model)
    """
    # Pad and create labels
    all_seqs = []
    all_labels = []

    for seq in real_sequences:
        padded = np.zeros((max_len, 3), dtype=np.float32)
        length = min(len(seq), max_len)
        padded[:length] = seq[:length]
        all_seqs.append(padded)
        all_labels.append(1)  # real

    for seq in generated_sequences:
        padded = np.zeros((max_len, 3), dtype=np.float32)
        length = min(len(seq), max_len)
        padded[:length] = seq[:length]
        all_seqs.append(padded)
        all_labels.append(0)  # generated

    X = np.array(all_seqs)
    y = np.array(all_labels, dtype=np.float32)

    # Shuffle with fixed seed for reproducibility
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    # Split 80/20
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build small LSTM classifier
    model = keras.Sequential([
        keras.layers.LSTM(64, input_shape=(max_len, 3)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0,
              validation_split=0.1)

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return float(accuracy)


def compute_realism_score(
    model,
    dataset,
    cfg: V3ModelConfig,
    n_samples: int = 500,
    style_vector: np.ndarray | None = None,
) -> dict:
    """Compute realism score using discriminator.

    Returns:
        dict with 'discriminator_accuracy' and 'realism_score'
    """
    if style_vector is None:
        style_vector = np.zeros(cfg.style_dim, dtype=np.float32)

    # Collect real sequences
    real_sequences = []
    count = 0
    for batch_inputs, batch_labels in dataset:
        delays = batch_labels["delay_labels"].numpy()
        actions = batch_labels["action_labels"].numpy()
        masks = batch_labels["label_mask"].numpy()
        for b in range(len(masks)):
            seq = []
            for t in range(masks.shape[1]):
                if masks[b, t] > 0:
                    seq.append([delays[b, t], float(actions[b, t]), 0.0])
            if len(seq) > 5:
                real_sequences.append(np.array(seq, dtype=np.float32))
                count += 1
            if count >= n_samples:
                break
        if count >= n_samples:
            break

    # Generate sequences
    test_texts = [
        "the quick brown fox", "hello world",
        "typing test sentence", "a simple phrase",
        "the lazy dog jumps", "machine learning model",
    ]
    gen_sequences = []
    for i in range(min(n_samples, len(real_sequences))):
        text = test_texts[i % len(test_texts)]
        keystrokes = generate_v3_full(
            model, text, style_vector, cfg,
            temperatures={"action": 0.8, "timing": 0.8, "char": 0.8},
        )
        seq = []
        for ks in keystrokes:
            seq.append([
                float(np.log(max(ks.delay_ms, 1.0))),
                float(ks.action),
                1.0 if ks.key == " " else 0.0,
            ])
        if len(seq) > 5:
            gen_sequences.append(np.array(seq, dtype=np.float32))

    if len(real_sequences) < 10 or len(gen_sequences) < 10:
        return {"discriminator_accuracy": 0.5, "realism_score": 0.5}

    disc_acc = train_discriminator(real_sequences, gen_sequences)

    return {
        "discriminator_accuracy": disc_acc,
        "realism_score": 1.0 - disc_acc,
    }


def check_style_consistency(
    model,
    cfg: V3ModelConfig,
    wpm_targets: list[float] | None = None,
    error_targets: list[float] | None = None,
) -> dict:
    """Check if generated output matches target WPM and error rate settings.

    This is a placeholder that builds style vectors from target settings
    and measures if the generated output has matching statistics.
    """
    if wpm_targets is None:
        wpm_targets = [40.0, 80.0, 120.0]
    if error_targets is None:
        error_targets = [0.02, 0.05, 0.10]

    results = {}
    text = "the quick brown fox jumps over the lazy dog"

    for wpm_target in wpm_targets:
        # Build a style vector biased toward this WPM
        style = np.zeros(cfg.style_dim, dtype=np.float32)
        # mean_iki_log: approximate log(60000 / (5 * wpm))
        target_iki_ms = 60000.0 / (5.0 * max(wpm_target, 1.0))
        style[0] = float(np.log(max(target_iki_ms, 10.0)))

        keystrokes = generate_v3_full(
            model, text, style, cfg,
            temperatures={"action": 0.5, "timing": 0.5, "char": 0.5},
        )

        if keystrokes:
            total_time_s = keystrokes[-1].cumulative_ms / 1000.0
            n_chars = len(text)
            actual_wpm = (n_chars / 5.0) / max(total_time_s / 60.0, 0.001)
            results[f"wpm_{wpm_target:.0f}_actual"] = actual_wpm

    return results
