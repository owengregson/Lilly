"""Unified model evaluation for V1 (LSTM) and V2 (Transformer) models.

Provides metrics computation and visualization for both model versions:
- V1: timing MAE, per-action accuracy breakdown, confusion matrix plots
- V2: teacher-forced metrics, autoregressive reconstruction accuracy
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from lilly.core.config import (
    ACTION_BACKSPACE,
    ACTION_CORRECT,
    ACTION_ERROR,
    V1TrainConfig,
    V2ModelConfig,
    V2TrainConfig,
)
from lilly.core.encoding import id_to_char
from lilly.inference.generator import generate_v2_segment
from lilly.models.transformer import compute_loss


# ---------------------------------------------------------------------------
# V1 Evaluation
# ---------------------------------------------------------------------------


def evaluate_v1(
    model: keras.Model,
    test_ds: tf.data.Dataset,
) -> dict:
    """Run V1 model evaluation and collect detailed metrics.

    Runs ``model.evaluate`` on the test dataset, then collects predictions
    for 50 batches to compute timing MAE (log-space and ms) and per-action
    accuracy breakdown.

    Parameters
    ----------
    model : keras.Model
        Compiled V1 LSTM model.
    test_ds : tf.data.Dataset
        Test dataset yielding ``(inputs, labels, sample_weights)`` tuples.

    Returns
    -------
    dict
        Evaluation metrics dict. Includes scalar metrics plus private arrays
        ``_pred_timing``, ``_true_iki_log``, ``_pred_action``, and
        ``_true_action`` for downstream plotting.
    """
    results = model.evaluate(test_ds, verbose=1, return_dict=True)

    # Collect predictions for analysis
    all_pred_timing = []
    all_pred_action = []
    all_true_iki = []
    all_true_action = []

    for inputs, labels, _ in test_ds.take(50):  # sample 50 batches
        preds = model.predict(inputs, verbose=0)

        all_pred_timing.append(preds["timing"])
        all_pred_action.append(np.argmax(preds["action"], axis=-1))
        all_true_iki.append(labels["timing"].numpy())
        all_true_action.append(labels["action"].numpy())

    pred_timing = np.concatenate(all_pred_timing)
    pred_action = np.concatenate(all_pred_action)
    true_iki_log = np.concatenate(all_true_iki)
    true_action = np.concatenate(all_true_action)

    # Timing accuracy: MAE on log-IKI
    pred_mu = pred_timing[:, 0]
    timing_mae = np.mean(np.abs(pred_mu - true_iki_log))
    timing_mae_ms = np.mean(np.abs(np.exp(pred_mu) - np.exp(true_iki_log)))

    # Action accuracy breakdown
    action_names = ["correct", "error", "backspace"]
    action_metrics: dict = {}
    for i, name in enumerate(action_names):
        mask = true_action == i
        if mask.sum() > 0:
            acc = np.mean(pred_action[mask] == i)
            action_metrics[f"action_{name}_acc"] = float(acc)
            action_metrics[f"action_{name}_count"] = int(mask.sum())

    return {
        **results,
        "timing_mae_log": float(timing_mae),
        "timing_mae_ms": float(timing_mae_ms),
        **action_metrics,
        "_pred_timing": pred_timing,
        "_true_iki_log": true_iki_log,
        "_pred_action": pred_action,
        "_true_action": true_action,
    }


# ---------------------------------------------------------------------------
# V1 Plotting
# ---------------------------------------------------------------------------


def plot_timing_distribution(
    pred_timing: np.ndarray,
    true_iki_log: np.ndarray,
    save_path: Path,
) -> None:
    """Plot predicted vs actual IKI distributions.

    Creates a 3-subplot figure:
    1. Log-IKI histogram comparison (predicted mu vs actual)
    2. Actual vs predicted scatter plot (up to 5000 samples)
    3. Predicted sigma (uncertainty) distribution

    Parameters
    ----------
    pred_timing : np.ndarray
        Shape ``(N, 2)`` with columns ``[mu, log_sigma]``.
    true_iki_log : np.ndarray
        Shape ``(N,)`` of actual log-IKI values.
    save_path : Path
        Where to save the figure (PNG).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Log-IKI histogram comparison
    ax = axes[0]
    pred_mu = pred_timing[:, 0]
    ax.hist(true_iki_log, bins=100, alpha=0.5, label="Actual log(IKI)", density=True)
    ax.hist(pred_mu, bins=100, alpha=0.5, label="Predicted mu", density=True)
    ax.set_xlabel("log(IKI ms)")
    ax.set_ylabel("Density")
    ax.set_title("IKI Distribution (log-space)")
    ax.legend()

    # 2. Actual vs predicted IKI scatter
    ax = axes[1]
    n_plot = min(5000, len(pred_mu))
    idx = np.random.choice(len(pred_mu), n_plot, replace=False)
    ax.scatter(true_iki_log[idx], pred_mu[idx], alpha=0.1, s=1)
    ax.plot(
        [true_iki_log.min(), true_iki_log.max()],
        [true_iki_log.min(), true_iki_log.max()],
        "r--",
        linewidth=1,
    )
    ax.set_xlabel("Actual log(IKI)")
    ax.set_ylabel("Predicted mu")
    ax.set_title("Timing Prediction Accuracy")

    # 3. Predicted sigma distribution
    ax = axes[2]
    pred_log_sigma = pred_timing[:, 1]
    pred_sigma = np.exp(np.clip(pred_log_sigma, -5, 5))
    ax.hist(pred_sigma, bins=100, alpha=0.7, color="green")
    ax.set_xlabel("Predicted sigma")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Uncertainty")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Timing plot saved: {save_path}")


def plot_action_confusion(
    pred_action: np.ndarray,
    true_action: np.ndarray,
    save_path: Path,
) -> None:
    """Plot action prediction confusion matrix.

    Uses sklearn to build a row-normalized confusion matrix with both
    percentages and raw counts displayed in each cell.

    Parameters
    ----------
    pred_action : np.ndarray
        Shape ``(N,)`` of predicted action indices.
    true_action : np.ndarray
        Shape ``(N,)`` of true action indices.
    save_path : Path
        Where to save the figure (PNG).
    """
    from sklearn.metrics import confusion_matrix

    labels = [ACTION_CORRECT, ACTION_ERROR, ACTION_BACKSPACE]
    names = ["Correct", "Error", "Backspace"]
    cm = confusion_matrix(true_action, pred_action, labels=labels)

    # Normalize by row
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Action Prediction Confusion Matrix")

    for i in range(len(names)):
        for j in range(len(names)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


# ---------------------------------------------------------------------------
# V2 Evaluation
# ---------------------------------------------------------------------------


def teacher_forced_metrics(
    model: keras.Model,
    dataset: tf.data.Dataset,
    train_cfg: V2TrainConfig,
) -> dict:
    """Compute teacher-forced token accuracy and timing MAE on a V2 dataset.

    Iterates over the dataset, running the model in inference mode with
    teacher forcing to compute character loss, timing loss, token accuracy,
    and timing MAE in log-space.

    Parameters
    ----------
    model : keras.Model
        Trained V2 ``TypingTransformer`` model.
    dataset : tf.data.Dataset
        V2 evaluation dataset yielding ``(inputs, labels)`` tuples.
    train_cfg : V2TrainConfig
        Training config with ``char_loss_weight`` and ``timing_loss_weight``.

    Returns
    -------
    dict
        Keys: ``total_loss``, ``char_loss``, ``timing_loss``,
        ``token_accuracy``, ``timing_mae_log``.
    """
    all_losses = []
    all_char_losses = []
    all_timing_losses = []
    all_accs = []
    timing_maes = []

    for inputs, labels in dataset:
        outputs = model(inputs, training=False)
        total, char_loss, timing_loss = compute_loss(
            outputs,
            labels,
            char_weight=train_cfg.char_loss_weight,
            timing_weight=train_cfg.timing_loss_weight,
        )

        mask = labels["label_mask"]
        preds = tf.argmax(outputs["char_logits"], axis=-1, output_type=tf.int32)
        correct = tf.cast(tf.equal(preds, labels["char_labels"]), tf.float32)
        acc = tf.reduce_sum(correct * mask) / tf.reduce_sum(mask)

        # Timing MAE in log-space
        delay_mu = outputs["delay_params"][:, :, 0]  # predicted mu
        delay_true = labels["delay_labels"]
        mae = tf.reduce_sum(tf.abs(delay_mu - delay_true) * mask) / tf.reduce_sum(
            mask
        )

        all_losses.append(total.numpy())
        all_char_losses.append(char_loss.numpy())
        all_timing_losses.append(timing_loss.numpy())
        all_accs.append(acc.numpy())
        timing_maes.append(mae.numpy())

    return {
        "total_loss": float(np.mean(all_losses)),
        "char_loss": float(np.mean(all_char_losses)),
        "timing_loss": float(np.mean(all_timing_losses)),
        "token_accuracy": float(np.mean(all_accs)),
        "timing_mae_log": float(np.mean(timing_maes)),
    }


def reconstruction_metrics(
    model: keras.Model,
    dataset: tf.data.Dataset,
    cfg: V2ModelConfig,
    n_samples: int = 100,
) -> dict:
    """Evaluate autoregressive reconstruction accuracy for V2.

    Generates keystroke sequences for a sample of segments and replays
    them to check whether the net output text matches the target.

    Parameters
    ----------
    model : keras.Model
        Trained V2 ``TypingTransformer`` model.
    dataset : tf.data.Dataset
        V2 evaluation dataset yielding ``(inputs, labels)`` tuples.
    cfg : V2ModelConfig
        Model configuration.
    n_samples : int
        Number of segments to evaluate (default 100).

    Returns
    -------
    dict
        Keys: ``reconstruction_accuracy``, ``n_evaluated``, ``n_matches``,
        ``mean_error_rate``, ``mean_generated_length``.
    """
    matches = 0
    total = 0
    error_rates = []
    generated_lengths = []

    for inputs, labels in dataset:
        batch_size = inputs["encoder_chars"].shape[0]
        for i in range(min(batch_size, n_samples - total)):
            if total >= n_samples:
                break

            # Extract target text from encoder chars
            enc_chars = inputs["encoder_chars"][i].numpy()
            enc_len = int(inputs["encoder_lengths"][i].numpy())
            target_text = "".join(id_to_char(int(c)) for c in enc_chars[:enc_len])

            if not target_text:
                continue

            wpm_bucket = int(inputs["wpm_bucket"][i].numpy())
            prev_ctx = inputs["prev_context"][i].numpy().tolist()

            # Generate autoregressively
            keystrokes = generate_v2_segment(
                model, target_text, wpm_bucket, prev_ctx, cfg, temperature=0.8
            )

            # Replay to get output text
            buffer: list[str] = []
            n_backspaces = 0
            for ks in keystrokes:
                if ks.key == "BKSP":
                    if buffer:
                        buffer.pop()
                    n_backspaces += 1
                elif ks.key not in ("<END>", "<START>", ""):
                    buffer.append(ks.key)

            result = "".join(buffer)
            is_match = result == target_text
            matches += int(is_match)
            total += 1

            # Error rate for this segment
            if keystrokes:
                error_rate = n_backspaces / len(keystrokes)
                error_rates.append(error_rate)
            generated_lengths.append(len(keystrokes))

        if total >= n_samples:
            break

    return {
        "reconstruction_accuracy": matches / max(total, 1),
        "n_evaluated": total,
        "n_matches": matches,
        "mean_error_rate": float(np.mean(error_rates)) if error_rates else 0.0,
        "mean_generated_length": (
            float(np.mean(generated_lengths)) if generated_lengths else 0.0
        ),
    }
