"""V3 loss functions: Focal Loss, MDN Mixture NLL, combined loss."""

from __future__ import annotations

import math
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras


@dataclass
class V3LossConfig:
    action_weight: float = 3.0
    timing_weight: float = 1.0
    error_char_weight: float = 1.0
    position_weight: float = 0.1
    focal_gamma: float = 2.0
    focal_alpha: tuple = (0.25, 0.5, 0.5)


class FocalLoss:
    """Focal loss for imbalanced action classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Returns per-sample loss (B, T) for external masking.
    """

    def __init__(self, alpha: list[float], gamma: float = 2.0):
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = gamma

    def __call__(self, y_true, y_pred):
        """
        y_true: (B, T) int32 action labels
        y_pred: (B, T, num_actions) float32 probabilities
        Returns: (B, T) per-position focal loss
        """
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true_onehot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

        p_t = tf.reduce_sum(y_pred * y_true_onehot, axis=-1)  # (B, T)
        alpha_t = tf.reduce_sum(self.alpha * y_true_onehot, axis=-1)  # (B, T)

        focal_weight = alpha_t * tf.pow(1.0 - p_t, self.gamma)
        ce = -tf.math.log(p_t)

        return focal_weight * ce  # (B, T)


def mdn_mixture_nll(y_true, pi, mu, log_sigma):
    """Negative log-likelihood of a LogNormal mixture density.

    Args:
        y_true: (B, T) target values in log-space
        pi: (B, T, K) mixture weights (sum to 1)
        mu: (B, T, K) component means
        log_sigma: (B, T, K) component log-std

    Returns:
        (B, T) per-position NLL
    """
    sigma = tf.exp(log_sigma)  # already clipped in MDNHead
    y = y_true[:, :, tf.newaxis]  # (B, T, 1)

    # Log-probability of each component (Gaussian in log-space = LogNormal)
    log_probs = (
        -0.5 * tf.square((y - mu) / (sigma + 1e-6))
        - log_sigma
        - 0.5 * math.log(2.0 * math.pi)
    )  # (B, T, K)

    # Log-sum-exp with mixture weights
    log_pi = tf.math.log(pi + 1e-10)
    log_mixture = tf.reduce_logsumexp(log_pi + log_probs, axis=-1)  # (B, T)

    return -log_mixture  # (B, T)


def compute_v3_loss(outputs, labels, cfg: V3LossConfig):
    """Compute the combined V3 loss.

    Args:
        outputs: dict with keys action_probs, timing_correct/error/backspace,
                 error_char_logits, position_pred
        labels: dict with keys action_labels, delay_labels, error_char_labels,
                position_labels, label_mask

    Returns:
        (total_loss, component_dict)
    """
    mask = labels["label_mask"]  # (B, T)
    mask_sum = tf.maximum(tf.reduce_sum(mask), 1.0)

    # --- Action loss (focal) ---
    focal = FocalLoss(alpha=list(cfg.focal_alpha), gamma=cfg.focal_gamma)
    action_loss_raw = focal(labels["action_labels"], outputs["action_probs"])  # (B, T)
    action_loss = tf.reduce_sum(action_loss_raw * mask) / mask_sum

    # --- Timing loss (per-action masked MDN NLL) ---
    action_labels = labels["action_labels"]
    delay_labels = labels["delay_labels"]

    timing_loss = tf.constant(0.0)
    for action_idx, key in enumerate(["timing_correct", "timing_error", "timing_backspace"]):
        pi, mu, log_sigma = outputs[key]
        action_mask = tf.cast(tf.equal(action_labels, action_idx), tf.float32) * mask
        action_mask_sum = tf.maximum(tf.reduce_sum(action_mask), 1.0)
        nll = mdn_mixture_nll(delay_labels, pi, mu, log_sigma)
        timing_loss = timing_loss + tf.reduce_sum(nll * action_mask) / action_mask_sum

    # --- Error char loss (masked to action=ERROR) ---
    error_mask = tf.cast(tf.equal(action_labels, 1), tf.float32) * mask
    error_mask_sum = tf.maximum(tf.reduce_sum(error_mask), 1.0)
    error_ce = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )(labels["error_char_labels"], outputs["error_char_logits"])
    error_char_loss = tf.reduce_sum(error_ce * error_mask) / error_mask_sum

    # --- Position loss (auxiliary) ---
    position_pred = tf.squeeze(outputs["position_pred"], axis=-1)  # (B, T)
    position_mse = tf.square(position_pred - labels["position_labels"])
    position_loss = tf.reduce_sum(position_mse * mask) / mask_sum

    # --- Combine ---
    total = (
        cfg.action_weight * action_loss
        + cfg.timing_weight * timing_loss
        + cfg.error_char_weight * error_char_loss
        + cfg.position_weight * position_loss
    )

    components = {
        "action": action_loss,
        "timing": timing_loss,
        "error_char": error_char_loss,
        "position": position_loss,
    }

    return total, components
