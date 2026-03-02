"""Tier 1 point metrics for V3 model evaluation.

Computes action accuracy/F1, timing MAE/NLL, and error character accuracy
on a held-out dataset.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import tensorflow as tf

from lilly.training.losses import V3LossConfig, compute_v3_loss


def compute_tier1_metrics(model, dataset, loss_cfg: V3LossConfig | None = None) -> dict:
    """Compute Tier 1 point metrics on a dataset.

    Args:
        model: TypingTransformerV3 model
        dataset: tf.data.Dataset yielding (inputs, labels) tuples
        loss_cfg: V3LossConfig (uses defaults if None)

    Returns:
        dict with metric names -> float values
    """
    if loss_cfg is None:
        loss_cfg = V3LossConfig()

    # Accumulators
    all_true_actions = []
    all_pred_actions = []
    all_timing_nll = []
    all_timing_mae = []
    error_correct = 0
    error_total = 0
    error_top3_correct = 0
    total_masked = 0

    for batch_inputs, batch_labels in dataset:
        outputs = model(batch_inputs, training=False)
        mask = batch_labels["label_mask"].numpy()

        # Action predictions
        action_probs = outputs["action_probs"].numpy()
        pred_actions = np.argmax(action_probs, axis=-1)
        true_actions = batch_labels["action_labels"].numpy()

        for b in range(len(mask)):
            for t in range(mask.shape[1]):
                if mask[b, t] > 0:
                    all_true_actions.append(true_actions[b, t])
                    all_pred_actions.append(pred_actions[b, t])
                    total_masked += 1

        # Timing NLL (using combined loss)
        _, components = compute_v3_loss(outputs, batch_labels, loss_cfg)
        all_timing_nll.append(float(components["timing"]))

        # Timing MAE (exp-space, using correct-MDN mean)
        pi, mu, _ = outputs["timing_correct"]
        best_k = tf.argmax(pi, axis=-1).numpy()
        for b in range(len(mask)):
            for t in range(mask.shape[1]):
                if mask[b, t] > 0 and true_actions[b, t] == 0:
                    pred_log = mu.numpy()[b, t, best_k[b, t]]
                    true_log = batch_labels["delay_labels"].numpy()[b, t]
                    pred_ms = np.exp(pred_log)
                    true_ms = np.exp(true_log)
                    all_timing_mae.append(abs(pred_ms - true_ms))

        # Error char accuracy (on error samples only)
        error_logits = outputs["error_char_logits"].numpy()
        error_labels = batch_labels["error_char_labels"].numpy()
        for b in range(len(mask)):
            for t in range(mask.shape[1]):
                if mask[b, t] > 0 and true_actions[b, t] == 1:
                    error_total += 1
                    pred_char = np.argmax(error_logits[b, t])
                    if pred_char == error_labels[b, t]:
                        error_correct += 1
                    top3 = np.argsort(error_logits[b, t])[-3:]
                    if error_labels[b, t] in top3:
                        error_top3_correct += 1

    all_true = np.array(all_true_actions)
    all_pred = np.array(all_pred_actions)

    # Overall accuracy
    action_accuracy = float(np.mean(all_true == all_pred)) if len(all_true) > 0 else 0.0

    # Per-class F1
    f1_scores = {}
    for cls, name in [(0, "correct"), (1, "error"), (2, "backspace")]:
        tp = np.sum((all_pred == cls) & (all_true == cls))
        fp = np.sum((all_pred == cls) & (all_true != cls))
        fn = np.sum((all_pred != cls) & (all_true == cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        f1_scores[f"action_f1_{name}"] = float(f1)

    return {
        "action_accuracy": action_accuracy,
        **f1_scores,
        "timing_nll": float(np.mean(all_timing_nll)) if all_timing_nll else 0.0,
        "timing_mae_ms": float(np.mean(all_timing_mae)) if all_timing_mae else 0.0,
        "error_char_accuracy": float(error_correct / max(error_total, 1)),
        "error_char_top3": float(error_top3_correct / max(error_total, 1)),
        "total_samples": total_masked,
    }
