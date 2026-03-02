"""Custom loss functions shared across model versions."""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow import keras


class LogNormalNLL(keras.losses.Loss):
    """Negative log-likelihood of a LogNormal distribution (V1: per-sample).

    Model predicts (mu, log_sigma) for each sample.
    Target is log(IKI_ms).
    """

    def __init__(self, **kwargs):
        super().__init__(name="log_normal_nll", **kwargs)

    def call(self, y_true, y_pred):
        mu = y_pred[:, 0]
        log_sigma = y_pred[:, 1]
        sigma = tf.exp(tf.clip_by_value(log_sigma, -5.0, 5.0))

        nll = (
            0.5 * tf.square((y_true - mu) / (sigma + 1e-6))
            + log_sigma
            + 0.5 * math.log(2.0 * math.pi)
        )
        return tf.reduce_mean(nll)


class LogNormalNLLSeq(keras.losses.Loss):
    """Negative log-likelihood of a LogNormal distribution (V2: per-token).

    Returns per-position NLL for external masking.
    y_true: (batch, seq) log-IKI
    y_pred: (batch, seq, 2) [mu, log_sigma]
    """

    def __init__(self, **kwargs):
        super().__init__(name="log_normal_nll_seq", **kwargs)

    def call(self, y_true, y_pred):
        mu = y_pred[:, :, 0]
        log_sigma = y_pred[:, :, 1]
        sigma = tf.exp(tf.clip_by_value(log_sigma, -5.0, 5.0))

        nll = (
            0.5 * tf.square((y_true - mu) / (sigma + 1e-6))
            + log_sigma
            + 0.5 * math.log(2.0 * math.pi)
        )
        return nll  # (batch, seq) -- masked externally


class MaskedSparseCE(keras.losses.Loss):
    """Sparse categorical cross-entropy masked to only error samples.

    The error_char prediction is only meaningful when the action is ERROR (1).
    The mask is applied externally via sample_weight during training.
    """

    def __init__(self, **kwargs):
        super().__init__(name="masked_error_ce", **kwargs)

    def call(self, y_true, y_pred):
        return keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False
        )
