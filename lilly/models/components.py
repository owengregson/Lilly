"""V3 model components: FiLM, MDN, ActionGate, ErrorCharHead."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from lilly.core.keyboard import key_distance


class FiLMModulation(keras.layers.Layer):
    """Feature-wise Linear Modulation for style conditioning.

    Applies learned scale (gamma) and shift (beta) to input features
    based on a style vector: output = gamma * x + beta.
    """

    def __init__(self, d_model: int, style_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self._d_model = d_model
        self._style_dim = style_dim
        self.gamma_proj = keras.layers.Dense(d_model, name="film_gamma")
        self.beta_proj = keras.layers.Dense(d_model, name="film_beta")

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self._d_model, "style_dim": self._style_dim})
        return config

    def call(self, x, style_vector):
        gamma = 1.0 + self.gamma_proj(style_vector)  # (B, d_model)
        beta = self.beta_proj(style_vector)            # (B, d_model)
        return gamma[:, tf.newaxis, :] * x + beta[:, tf.newaxis, :]


class MDNHead(keras.layers.Layer):
    """Mixture Density Network head for LogNormal timing prediction.

    Outputs (pi, mu, log_sigma) for n_components LogNormal components.
    """

    def __init__(self, n_components: int = 8, hidden_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self._n_components = n_components
        self._hidden_dim = hidden_dim
        self.hidden = keras.layers.Dense(hidden_dim, activation="relu", name="mdn_hidden")
        self.params = keras.layers.Dense(n_components * 3, name="mdn_params")

    def get_config(self):
        config = super().get_config()
        config.update({"n_components": self._n_components, "hidden_dim": self._hidden_dim})
        return config

    def call(self, h):
        x = self.hidden(h)
        raw = self.params(x)  # (B, T, n_components * 3)
        shape = tf.shape(raw)
        raw = tf.reshape(raw, (shape[0], shape[1], self._n_components, 3))

        pi_logits = raw[:, :, :, 0]
        mu = raw[:, :, :, 1]
        log_sigma = tf.clip_by_value(raw[:, :, :, 2], -5.0, 5.0)

        pi = tf.nn.softmax(pi_logits, axis=-1)
        return pi, mu, log_sigma


class ActionGate(keras.layers.Layer):
    """Action prediction gate: correct / error / backspace."""

    def __init__(self, hidden_dim: int = 64, num_actions: int = 3, **kwargs):
        super().__init__(**kwargs)
        self._hidden_dim = hidden_dim
        self._num_actions = num_actions
        self.hidden = keras.layers.Dense(hidden_dim, activation="relu", name="gate_hidden")
        self.output_proj = keras.layers.Dense(num_actions, activation="softmax", name="gate_out")

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self._hidden_dim, "num_actions": self._num_actions})
        return config

    def call(self, h):
        return self.output_proj(self.hidden(h))


def _build_distance_matrix() -> np.ndarray:
    """Precompute (97, 97) QWERTY distance matrix for char IDs 0..96."""
    matrix = np.zeros((97, 97), dtype=np.float32)
    for i in range(1, 96):  # printable chars (IDs 1..95)
        ch_i = chr(i + 31)
        for j in range(1, 96):
            ch_j = chr(j + 31)
            matrix[i, j] = key_distance(ch_i, ch_j)
    return matrix


class ErrorCharHead(keras.layers.Layer):
    """Error character prediction with QWERTY distance bias.

    Predicts which wrong key was typed, biased toward keys physically
    near the target key on a QWERTY keyboard.
    """

    def __init__(self, num_chars: int = 97, learnable_alpha: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._num_chars = num_chars
        self._learnable_alpha = learnable_alpha
        self.logit_proj = keras.layers.Dense(num_chars, name="error_logits")
        dist_matrix = _build_distance_matrix()
        self.distance_matrix = tf.constant(dist_matrix, dtype=tf.float32)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="qwerty_alpha", shape=(), initializer="ones",
            trainable=self._learnable_alpha,
        )
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"num_chars": self._num_chars,
                        "learnable_alpha": self._learnable_alpha})
        return config

    def call(self, h, target_char_ids):
        logits = self.logit_proj(h)  # (B, T, 97)
        target_ids_clipped = tf.clip_by_value(target_char_ids, 0, 96)
        distances = tf.gather(self.distance_matrix, target_ids_clipped)  # (B, T, 97)
        biased_logits = logits - self.alpha * distances
        return biased_logits
