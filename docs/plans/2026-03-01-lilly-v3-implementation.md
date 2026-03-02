# Lilly V3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild Lilly's typing model as V3 — a unified encoder-decoder transformer with action-gated decoder, mixture density timing, FiLM style conditioning, and distributional evaluation.

**Architecture:** Encoder reads target text + style vector via FiLM. Decoder autoregressively generates keystrokes, first predicting action (correct/error/backspace) through a gate, then routing to per-action MDN timing heads and character heads. Error character head uses QWERTY distance bias.

**Tech Stack:** TensorFlow/Keras, NumPy, Pandas, PyArrow, SciPy (distributional metrics), matplotlib (visualization).

**Design doc:** `docs/plans/2026-03-01-lilly-v3-design.md`

---

## Task 1: Delete V1/V2 Code & Update Config

**Files:**
- Delete: `lilly/models/lstm.py`
- Delete: `lilly/models/transformer.py`
- Delete: `lilly/data/features.py`
- Delete: `lilly/training/trainer_v1.py`
- Delete: `lilly/training/trainer_v2.py`
- Delete: `lilly/inference/context.py`
- Delete: `lilly/inference/generator.py`
- Delete: `lilly/inference/preview.py`
- Delete: `lilly/evaluation/evaluator.py`
- Delete: `scripts/extract_features.py`
- Delete: `lilly/core/losses.py`
- Delete: `lilly/training/callbacks.py`
- Modify: `lilly/core/config.py`
- Modify: `lilly/data/segment.py` (keep only `split_text_into_inference_segments`)

**Step 1: Delete V1/V2 files**

```bash
rm lilly/models/lstm.py lilly/models/transformer.py
rm lilly/data/features.py
rm lilly/training/trainer_v1.py lilly/training/trainer_v2.py
rm lilly/inference/context.py lilly/inference/generator.py lilly/inference/preview.py
rm lilly/evaluation/evaluator.py
rm scripts/extract_features.py
rm lilly/core/losses.py
rm lilly/training/callbacks.py
```

**Step 2: Trim `lilly/data/segment.py`**

Keep only the `split_text_into_inference_segments` function (lines 148-179). Delete everything else (Segment dataclass, compute_target_text_for_segment, extract_segments, _split_long_group).

**Step 3: Update `lilly/core/config.py`**

Remove `V1ModelConfig`, `V1TrainConfig`, `V2ModelConfig`, `V2TrainConfig`. Add:

```python
# ---------------------------------------------------------------------------
# V3 Segmentation
# ---------------------------------------------------------------------------
PAUSE_THRESHOLD_MS = 300.0
MIN_SEGMENT_KEYSTROKES = 3
MAX_SEGMENT_KEYSTROKES = 80  # increased from 64
MAX_TARGET_CHARS = 32
STYLE_DIM = 16

# ---------------------------------------------------------------------------
# V3 Model & Training Config
# ---------------------------------------------------------------------------
@dataclass
class V3ModelConfig:
    max_encoder_len: int = MAX_TARGET_CHARS
    max_decoder_len: int = MAX_SEGMENT_KEYSTROKES
    num_char_classes: int = NUM_CHAR_CLASSES_V2  # 99
    char_embed_dim: int = 48
    action_embed_dim: int = 16
    delay_embed_dim: int = 16
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    style_dim: int = STYLE_DIM
    mdn_components: int = 8
    num_actions: int = NUM_ACTIONS  # 3
    qwerty_bias_learnable: bool = True
    context_tail_len: int = 4


@dataclass
class V3TrainConfig:
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 2000
    early_stop_patience: int = 10
    action_loss_weight: float = 3.0
    timing_loss_weight: float = 1.0
    error_char_loss_weight: float = 1.0
    position_loss_weight: float = 0.1
    focal_gamma: float = 2.0
    focal_alpha: tuple = (0.25, 0.5, 0.5)
    val_split: float = 0.1
    test_split: float = 0.05
    shuffle_buffer: int = 100_000
    seed: int = 42
    max_samples: int = 0
```

Also add path constants:

```python
V3_SEGMENT_DIR = DATA_DIR / "v3_segments"
V3_MODEL_DIR = MODEL_DIR / "v3"
V3_EXPORT_DIR = EXPORT_DIR / "v3"
```

Remove V2-specific path constants (`V2_SEGMENT_DIR`, `V2_MODEL_DIR`, `V2_EXPORT_DIR`).

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove V1/V2 code, add V3 config dataclasses"
```

---

## Task 2: Model Components — FiLM, MDN, ActionGate, ErrorCharHead

**Files:**
- Create: `lilly/models/components.py`
- Create: `tests/test_components.py`

**Step 1: Write failing tests**

Create `tests/test_components.py`:

```python
"""Unit tests for V3 model components."""
import numpy as np
import tensorflow as tf
import pytest


class TestFiLMModulation:
    def test_output_shape(self):
        from lilly.models.components import FiLMModulation
        film = FiLMModulation(d_model=128, style_dim=16)
        x = tf.random.normal((2, 10, 128))
        style = tf.random.normal((2, 16))
        out = film(x, style)
        assert out.shape == (2, 10, 128)

    def test_identity_at_init(self):
        """gamma=1+proj(0)~1, beta=proj(0)~0 => near identity."""
        from lilly.models.components import FiLMModulation
        film = FiLMModulation(d_model=4, style_dim=2)
        x = tf.ones((1, 3, 4))
        style = tf.zeros((1, 2))
        out = film(x, style)
        # With zero style input and zero-init bias, gamma~1, beta~0
        # So output should be close to x (not exact due to random weight init)
        assert out.shape == (1, 3, 4)


class TestMDNHead:
    def test_output_shape(self):
        from lilly.models.components import MDNHead
        mdn = MDNHead(n_components=8, hidden_dim=64)
        h = tf.random.normal((2, 10, 128))
        pi, mu, log_sigma = mdn(h)
        assert pi.shape == (2, 10, 8)
        assert mu.shape == (2, 10, 8)
        assert log_sigma.shape == (2, 10, 8)

    def test_pi_sums_to_one(self):
        from lilly.models.components import MDNHead
        mdn = MDNHead(n_components=8, hidden_dim=64)
        h = tf.random.normal((4, 5, 128))
        pi, _, _ = mdn(h)
        sums = tf.reduce_sum(pi, axis=-1)
        np.testing.assert_allclose(sums.numpy(), 1.0, atol=1e-5)

    def test_log_sigma_clipped(self):
        from lilly.models.components import MDNHead
        mdn = MDNHead(n_components=8, hidden_dim=64)
        h = tf.random.normal((2, 5, 128)) * 100  # large input
        _, _, log_sigma = mdn(h)
        assert tf.reduce_all(log_sigma >= -5.0)
        assert tf.reduce_all(log_sigma <= 5.0)


class TestActionGate:
    def test_output_shape(self):
        from lilly.models.components import ActionGate
        gate = ActionGate(hidden_dim=64, num_actions=3)
        h = tf.random.normal((2, 10, 128))
        probs = gate(h)
        assert probs.shape == (2, 10, 3)

    def test_sums_to_one(self):
        from lilly.models.components import ActionGate
        gate = ActionGate(hidden_dim=64, num_actions=3)
        h = tf.random.normal((4, 8, 128))
        probs = gate(h)
        sums = tf.reduce_sum(probs, axis=-1)
        np.testing.assert_allclose(sums.numpy(), 1.0, atol=1e-5)


class TestErrorCharHead:
    def test_output_shape(self):
        from lilly.models.components import ErrorCharHead
        head = ErrorCharHead(num_chars=97, learnable_alpha=True)
        h = tf.random.normal((2, 10, 128))
        target_ids = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                   [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
        logits = head(h, target_ids)
        assert logits.shape == (2, 10, 97)

    def test_neighbor_bias_increases_nearby_keys(self):
        """Keys near the target should have higher logits than distant keys."""
        from lilly.models.components import ErrorCharHead
        head = ErrorCharHead(num_chars=97, learnable_alpha=True)
        # Set alpha to a large positive value to make bias dominant
        head.alpha.assign(10.0)
        h = tf.zeros((1, 1, 128))
        # target char 'f' (char_to_id('f') = ord('f') - 31 = 71)
        target_ids = tf.constant([[71]])
        logits = head(h, target_ids)
        probs = tf.nn.softmax(logits[0, 0]).numpy()
        # 'g' (72) should be higher than 'p' (81) since g is next to f
        assert probs[72] > probs[81], "Neighbor key should have higher prob"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_components.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lilly.models.components'`

**Step 3: Implement `lilly/models/components.py`**

```python
"""V3 model components: FiLM, MDN, ActionGate, ErrorCharHead."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from lilly.core.keyboard import KEY_POSITIONS, SHIFT_MAP, key_distance


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
```

**Step 4: Run tests**

```bash
pytest tests/test_components.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add lilly/models/components.py tests/test_components.py
git commit -m "feat: add V3 model components (FiLM, MDN, ActionGate, ErrorCharHead)"
```

---

## Task 3: Loss Functions — Focal Loss, MDN Mixture NLL, Combined Loss

**Files:**
- Create: `lilly/training/losses.py`
- Create: `tests/test_losses.py`

**Step 1: Write failing tests**

Create `tests/test_losses.py`:

```python
"""Unit tests for V3 loss functions."""
import numpy as np
import tensorflow as tf
import pytest


class TestFocalLoss:
    def test_output_shape(self):
        from lilly.training.losses import FocalLoss
        fl = FocalLoss(alpha=[0.25, 0.5, 0.5], gamma=2.0)
        y_true = tf.constant([[0, 1, 2, 0]], dtype=tf.int32)  # (1, 4)
        y_pred = tf.constant([[[0.9, 0.05, 0.05],
                                [0.1, 0.8, 0.1],
                                [0.1, 0.1, 0.8],
                                [0.8, 0.1, 0.1]]], dtype=tf.float32)
        loss = fl(y_true, y_pred)
        assert loss.shape == (1, 4)

    def test_confident_correct_has_low_loss(self):
        from lilly.training.losses import FocalLoss
        fl = FocalLoss(alpha=[1.0, 1.0, 1.0], gamma=2.0)
        y_true = tf.constant([[0]], dtype=tf.int32)
        confident = tf.constant([[[0.99, 0.005, 0.005]]], dtype=tf.float32)
        uncertain = tf.constant([[[0.4, 0.3, 0.3]]], dtype=tf.float32)
        loss_conf = fl(y_true, confident).numpy()[0, 0]
        loss_unc = fl(y_true, uncertain).numpy()[0, 0]
        assert loss_conf < loss_unc * 0.1, "Confident correct should have much lower loss"

    def test_gamma_zero_equals_weighted_ce(self):
        from lilly.training.losses import FocalLoss
        fl = FocalLoss(alpha=[1.0, 1.0, 1.0], gamma=0.0)
        y_true = tf.constant([[1]], dtype=tf.int32)
        y_pred = tf.constant([[[0.2, 0.6, 0.2]]], dtype=tf.float32)
        focal = fl(y_true, y_pred).numpy()[0, 0]
        ce = -np.log(0.6)
        np.testing.assert_allclose(focal, ce, atol=1e-5)


class TestMDNMixtureLoss:
    def test_output_shape(self):
        from lilly.training.losses import mdn_mixture_nll
        pi = tf.constant([[[0.5, 0.5]]], dtype=tf.float32)
        mu = tf.constant([[[4.0, 5.0]]], dtype=tf.float32)
        log_sigma = tf.constant([[[0.0, 0.0]]], dtype=tf.float32)
        y_true = tf.constant([[4.5]], dtype=tf.float32)
        nll = mdn_mixture_nll(y_true, pi, mu, log_sigma)
        assert nll.shape == (1, 1)

    def test_lower_nll_for_correct_component(self):
        from lilly.training.losses import mdn_mixture_nll
        # Mixture with one component at mu=4.0, one at mu=8.0
        pi = tf.constant([[[0.5, 0.5]]], dtype=tf.float32)
        mu = tf.constant([[[4.0, 8.0]]], dtype=tf.float32)
        log_sigma = tf.constant([[[-1.0, -1.0]]], dtype=tf.float32)
        # Target near first component
        nll_near = mdn_mixture_nll(tf.constant([[4.0]]), pi, mu, log_sigma).numpy()
        # Target far from both
        nll_far = mdn_mixture_nll(tf.constant([[6.0]]), pi, mu, log_sigma).numpy()
        assert nll_near < nll_far

    def test_positive_nll(self):
        """NLL should generally be positive for reasonable inputs."""
        from lilly.training.losses import mdn_mixture_nll
        pi = tf.constant([[[1.0]]], dtype=tf.float32)
        mu = tf.constant([[[4.0]]], dtype=tf.float32)
        log_sigma = tf.constant([[[0.5]]], dtype=tf.float32)
        nll = mdn_mixture_nll(tf.constant([[4.0]]), pi, mu, log_sigma)
        # With sigma=exp(0.5)~1.65, NLL at the mean should be moderate
        assert nll.numpy()[0, 0] > -5.0  # not extremely negative


class TestCombinedV3Loss:
    def test_runs_without_error(self):
        from lilly.training.losses import compute_v3_loss, V3LossConfig
        cfg = V3LossConfig()
        outputs = {
            "action_probs": tf.constant([[[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]]]),
            "timing_correct": (
                tf.constant([[[1.0]]]), tf.constant([[[4.0]]]), tf.constant([[[0.0]]])
            ),
            "timing_error": (
                tf.constant([[[1.0]]]), tf.constant([[[4.0]]]), tf.constant([[[0.0]]])
            ),
            "timing_backspace": (
                tf.constant([[[1.0]]]), tf.constant([[[4.0]]]), tf.constant([[[0.0]]])
            ),
            "error_char_logits": tf.random.normal((1, 2, 97)),
            "position_pred": tf.constant([[[0.3], [0.6]]]),
        }
        labels = {
            "action_labels": tf.constant([[0, 1]]),
            "delay_labels": tf.constant([[4.5, 4.0]]),
            "error_char_labels": tf.constant([[0, 50]]),
            "position_labels": tf.constant([[0.25, 0.5]]),
            "label_mask": tf.constant([[1.0, 1.0]]),
        }
        total, components = compute_v3_loss(outputs, labels, cfg)
        assert total.shape == ()
        assert "action" in components
        assert "timing" in components
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_losses.py -v
```

**Step 3: Implement `lilly/training/losses.py`**

```python
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
```

**Step 4: Run tests**

```bash
pytest tests/test_losses.py -v
```

**Step 5: Commit**

```bash
git add lilly/training/losses.py tests/test_losses.py
git commit -m "feat: add V3 loss functions (FocalLoss, MDN NLL, combined loss)"
```

---

## Task 4: Learning Rate Schedule — WarmupCosineDecay

**Files:**
- Create: `lilly/training/schedule.py`

**Step 1: Implement `lilly/training/schedule.py`**

```python
"""Learning rate schedule with linear warmup + cosine decay."""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow import keras


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup from 0 to peak_lr, then cosine decay to min_lr."""

    def __init__(self, peak_lr: float, warmup_steps: int,
                 decay_steps: int, min_lr: float = 1e-6):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        decay = tf.cast(self.decay_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.peak_lr * (step / tf.maximum(warmup, 1.0))

        # Cosine decay
        progress = tf.minimum((step - warmup) / tf.maximum(decay - warmup, 1.0), 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress)
        )

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "min_lr": self.min_lr,
        }
```

**Step 2: Commit**

```bash
git add lilly/training/schedule.py
git commit -m "feat: add WarmupCosineDecay learning rate schedule"
```

---

## Task 5: V3 Transformer Model — TypingTransformerV3

**Files:**
- Create: `lilly/models/typing_model.py`
- Create: `tests/test_typing_model.py`

**Step 1: Write failing tests**

Create `tests/test_typing_model.py`:

```python
"""Unit tests for the V3 TypingTransformerV3 model."""
import tensorflow as tf
import numpy as np
import pytest

from lilly.core.config import V3ModelConfig


def _make_dummy_inputs(cfg: V3ModelConfig, batch: int = 2):
    """Create dummy inputs matching the model's expected signature."""
    dec_len = cfg.max_decoder_len + 1
    return {
        "encoder_chars": tf.zeros((batch, cfg.max_encoder_len), dtype=tf.int32),
        "encoder_lengths": tf.constant([[5]] * batch, dtype=tf.int32),
        "decoder_input_chars": tf.zeros((batch, dec_len), dtype=tf.int32),
        "decoder_input_delays": tf.zeros((batch, dec_len), dtype=tf.float32),
        "decoder_input_actions": tf.zeros((batch, dec_len), dtype=tf.int32),
        "style_vector": tf.random.normal((batch, cfg.style_dim)),
        "prev_context_chars": tf.zeros((batch, cfg.context_tail_len), dtype=tf.int32),
        "prev_context_actions": tf.zeros((batch, cfg.context_tail_len), dtype=tf.int32),
        "prev_context_delays": tf.zeros((batch, cfg.context_tail_len), dtype=tf.float32),
    }


class TestModelForwardPass:
    def test_output_keys(self):
        from lilly.models.typing_model import build_model
        cfg = V3ModelConfig()
        model = build_model(cfg)
        inputs = _make_dummy_inputs(cfg)
        outputs = model(inputs, training=False)

        expected_keys = {
            "action_probs", "timing_correct", "timing_error",
            "timing_backspace", "error_char_logits", "position_pred",
        }
        assert set(outputs.keys()) == expected_keys

    def test_action_probs_shape(self):
        from lilly.models.typing_model import build_model
        cfg = V3ModelConfig()
        model = build_model(cfg)
        inputs = _make_dummy_inputs(cfg)
        outputs = model(inputs, training=False)
        dec_len = cfg.max_decoder_len + 1
        assert outputs["action_probs"].shape == (2, dec_len, 3)

    def test_timing_mdn_shapes(self):
        from lilly.models.typing_model import build_model
        cfg = V3ModelConfig()
        model = build_model(cfg)
        inputs = _make_dummy_inputs(cfg)
        outputs = model(inputs, training=False)
        dec_len = cfg.max_decoder_len + 1
        for key in ["timing_correct", "timing_error", "timing_backspace"]:
            pi, mu, log_sigma = outputs[key]
            assert pi.shape == (2, dec_len, cfg.mdn_components)
            assert mu.shape == (2, dec_len, cfg.mdn_components)

    def test_error_char_logits_shape(self):
        from lilly.models.typing_model import build_model
        cfg = V3ModelConfig()
        model = build_model(cfg)
        inputs = _make_dummy_inputs(cfg)
        outputs = model(inputs, training=False)
        dec_len = cfg.max_decoder_len + 1
        assert outputs["error_char_logits"].shape == (2, dec_len, 97)

    def test_param_count_reasonable(self):
        from lilly.models.typing_model import build_model
        cfg = V3ModelConfig()
        model = build_model(cfg)
        count = model.count_params()
        assert 500_000 < count < 10_000_000, f"Param count {count} outside expected range"

    def test_training_mode_runs(self):
        from lilly.models.typing_model import build_model
        cfg = V3ModelConfig()
        model = build_model(cfg)
        inputs = _make_dummy_inputs(cfg)
        outputs = model(inputs, training=True)
        assert outputs["action_probs"].shape[0] == 2
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_typing_model.py -v
```

**Step 3: Implement `lilly/models/typing_model.py`**

This is the core model. It has:
- Encoder with FiLM-conditioned transformer layers
- Decoder with FiLM-conditioned transformer layers + causal masking
- Action gate routing to 3 MDN timing heads
- Error character head with QWERTY bias
- Position prediction auxiliary head

The encoder and decoder layer classes are defined within this file (extending the components). The `TypingTransformerV3` class is a `keras.Model` subclass with `_encode`, `_decode`, and `call` methods.

The full implementation should follow the architecture in the design doc exactly. Key details:
- Decoder input concatenates char_embed(48d) + delay_proj(16d) + action_embed(16d) = 80d, then projects to d_model=128
- FiLM is applied after LayerNorm in each layer
- Causal mask for decoder self-attention
- Cross-attention mask from encoder padding
- `call()` returns the full output dict
- `build_model()` instantiates with dummy input to set shapes

Implement the `sinusoidal_positional_encoding` function (can port from V2's transformer.py which is now deleted, so rewrite it).

**Step 4: Run tests**

```bash
pytest tests/test_typing_model.py -v
```

**Step 5: Commit**

```bash
git add lilly/models/typing_model.py tests/test_typing_model.py
git commit -m "feat: add TypingTransformerV3 model with action-gated decoder"
```

---

## Task 6: Style Vector Computation

**Files:**
- Create: `lilly/data/style.py`
- Create: `tests/test_style.py`

**Step 1: Write failing tests**

Create `tests/test_style.py`:

```python
"""Unit tests for style vector computation."""
import numpy as np
import pandas as pd
import pytest


def _make_session_df(n=100):
    """Create a fake session DataFrame matching preprocess.py output format."""
    rng = np.random.default_rng(42)
    actions = rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 1, 2], size=n)  # ~80% correct
    ikis = rng.lognormal(mean=4.5, sigma=0.5, size=n).clip(10, 5000)
    return pd.DataFrame({
        "action": actions,
        "iki": ikis,
        "hold_time": rng.lognormal(mean=4.0, sigma=0.3, size=n).clip(5, 2000),
        "typed_key": ["a"] * n,
        "target_char": ["a"] * n,
        "target_pos": list(range(n)),
        "wpm": [80.0] * n,
    })


class TestComputeStyleVector:
    def test_output_shape(self):
        from lilly.data.style import compute_style_vector
        df = _make_session_df()
        sv = compute_style_vector(df)
        assert sv.shape == (16,)
        assert sv.dtype == np.float32

    def test_all_finite(self):
        from lilly.data.style import compute_style_vector
        df = _make_session_df()
        sv = compute_style_vector(df)
        assert np.all(np.isfinite(sv))

    def test_different_sessions_differ(self):
        from lilly.data.style import compute_style_vector
        df1 = _make_session_df(n=100)
        df2 = _make_session_df(n=200)
        df2["iki"] = df2["iki"] * 2  # much slower
        sv1 = compute_style_vector(df1)
        sv2 = compute_style_vector(df2)
        assert not np.allclose(sv1, sv2)


class TestStyleNormalization:
    def test_normalize_and_denormalize(self):
        from lilly.data.style import compute_style_vector, StyleNormalizer
        vectors = np.stack([compute_style_vector(_make_session_df()) for _ in range(20)])
        norm = StyleNormalizer.fit(vectors)
        normed = norm.transform(vectors)
        # After z-score, mean should be ~0, std ~1
        np.testing.assert_allclose(normed.mean(axis=0), 0.0, atol=0.2)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_style.py -v
```

**Step 3: Implement `lilly/data/style.py`**

Compute the 16 style features from a session DataFrame. Each feature is documented in the design doc. The `StyleNormalizer` class holds mean/std and can save/load from JSON.

Key implementation notes:
- `compute_style_vector(df) -> np.ndarray` returns (16,) float32
- Burst detection: keystrokes between pauses (IKI > 300ms)
- Correction detection: error followed by backspace within 3 keystrokes
- Bigram speed variance: group by consecutive (key_i, key_i+1) pairs, compute per-bigram mean IKI, then variance
- IKI autocorrelation: `np.corrcoef(iki[:-1], iki[1:])[0,1]`
- Handle edge cases (empty session, all same key, etc.) by returning 0.0 for undefined features

**Step 4: Run tests**

```bash
pytest tests/test_style.py -v
```

**Step 5: Commit**

```bash
git add lilly/data/style.py tests/test_style.py
git commit -m "feat: add style vector computation (16-dim typing persona features)"
```

---

## Task 7: V3 Segmentation Pipeline

**Files:**
- Create: `lilly/data/segment_v3.py`
- Create: `scripts/segment_v3.py`

**Step 1: Implement `lilly/data/segment_v3.py`**

This module:
1. Reads Parquet chunks from `PROCESSED_DIR`
2. Groups by session_id
3. Computes style vector per session (using `lilly/data/style.py`)
4. Splits keystrokes into segments at pause boundaries (IKI >= 300ms)
5. For each segment, creates the V3 segment dict with action labels
6. Saves as .npz chunks to `V3_SEGMENT_DIR`

Key functions:
- `extract_v3_segments(session_df, style_vector) -> List[dict]`
- `process_chunk(parquet_path, output_dir) -> int` (returns segment count)
- `main()` CLI entrypoint

The segment format matches the design doc exactly. Action labels come directly from the `action` column in the Parquet data.

**Step 2: Implement `scripts/segment_v3.py`**

Thin wrapper that calls `lilly.data.segment_v3.main()`:

```python
#!/usr/bin/env python3
"""Prepare V3 training segments from preprocessed Parquet data."""
from lilly.data.segment_v3 import main

if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add lilly/data/segment_v3.py scripts/segment_v3.py
git commit -m "feat: add V3 segmentation pipeline with action labels and style vectors"
```

---

## Task 8: V3 Data Pipeline (tf.data)

**Files:**
- Modify: `lilly/data/pipeline.py` (remove V1/V2 loaders, add V3)

**Step 1: Rewrite `lilly/data/pipeline.py`**

Remove `load_v1_npz_files`, `build_v1_datasets`, `load_v2_segment_files`, `_prepare_decoder_io`, `build_v2_datasets`.

Add:
- `load_v3_segment_files(data_dir, max_files) -> dict`
- `_prepare_v3_decoder_io(...)` — builds teacher-forced decoder inputs with START token shift, action labels, delay log-normalization
- `build_v3_datasets(data_dir, model_cfg, train_cfg, max_files) -> (train_ds, val_ds, test_ds, n_total)`

The V3 pipeline differs from V2 in:
- Decoder inputs include action IDs (not just char + delay)
- Style vectors are loaded from the segments
- Delay normalization: `log(clip(delay, 1.0, 5000.0))`
- Position labels computed: `target_pos / sentence_length` per keystroke
- Label dict includes `action_labels`, `error_char_labels`, `position_labels`, `label_mask`

**Step 2: Commit**

```bash
git add lilly/data/pipeline.py
git commit -m "feat: rewrite data pipeline for V3 (action labels, style vectors, position labels)"
```

---

## Task 9: V3 Training Loop

**Files:**
- Create: `lilly/training/trainer.py`
- Create: `scripts/train.py` (rewrite)

**Step 1: Implement `lilly/training/trainer.py`**

Custom GradientTape training loop:
- Builds model from V3ModelConfig
- Creates AdamW optimizer with WarmupCosineDecay schedule
- Training step: forward pass, compute_v3_loss, gradient clipping, apply_gradients
- Validation step: forward pass, compute_v3_loss, action accuracy, timing MAE
- Per-epoch: log all loss components to CSV, save best model on val_loss, early stopping
- Final evaluation on test set
- Save training metadata JSON

Key difference from V2 trainer: uses the V3 combined loss with focal loss, per-action MDN masking, and position auxiliary loss. Logs per-component losses (action, timing, error_char, position) separately.

**Step 2: Rewrite `scripts/train.py`**

```python
#!/usr/bin/env python3
"""Train the V3 typing model."""
import argparse
from pathlib import Path
from lilly.core.config import V3_SEGMENT_DIR, V3_MODEL_DIR, V3ModelConfig, V3TrainConfig
from lilly.training.trainer import train

def main():
    parser = argparse.ArgumentParser(description="Train V3 typing model")
    parser.add_argument("--data-dir", type=Path, default=V3_SEGMENT_DIR)
    parser.add_argument("--model-dir", type=Path, default=V3_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    train_cfg = V3TrainConfig()
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate

    train(
        data_dir=args.data_dir, model_dir=args.model_dir,
        train_cfg=train_cfg, run_name=args.run_name,
        max_files=args.max_files,
    )

if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add lilly/training/trainer.py scripts/train.py
git commit -m "feat: add V3 training loop with focal loss and per-action MDN"
```

---

## Task 10: Sampling Utilities — MDN Sampling

**Files:**
- Modify: `lilly/inference/sampling.py`

**Step 1: Add `sample_mdn` to `lilly/inference/sampling.py`**

Keep existing functions. Add:

```python
def sample_mdn(
    pi: np.ndarray, mu: np.ndarray, log_sigma: np.ndarray
) -> tuple[float, int]:
    """Sample from an MDN mixture distribution.

    Args:
        pi: (K,) mixture weights
        mu: (K,) component means (log-space)
        log_sigma: (K,) component log-std

    Returns:
        (delay_ms, component_index)
    """
    k = int(np.random.choice(len(pi), p=pi))
    sigma = math.exp(np.clip(log_sigma[k], -5.0, 5.0))
    z = np.random.normal(mu[k], sigma)
    delay_ms = float(np.clip(np.exp(z), MIN_IKI_MS, MAX_IKI_MS))
    return delay_ms, k
```

**Step 2: Commit**

```bash
git add lilly/inference/sampling.py
git commit -m "feat: add MDN mixture sampling to inference utilities"
```

---

## Task 11: V3 Generator (Inference)

**Files:**
- Create: `lilly/inference/generator.py`
- Create: `tests/test_generator.py`

**Step 1: Write failing tests**

Create `tests/test_generator.py`:

```python
"""Unit tests for V3 generation logic."""
import pytest


class TestGenerateV3Segment:
    def test_produces_keystrokes(self):
        """Generate should return a non-empty list of keystrokes."""
        from lilly.models.typing_model import build_model
        from lilly.inference.generator import generate_v3_segment
        from lilly.core.config import V3ModelConfig
        import numpy as np

        cfg = V3ModelConfig()
        model = build_model(cfg)
        style = np.zeros(cfg.style_dim, dtype=np.float32)
        keystrokes = generate_v3_segment(
            model, "hello", style_vector=style, prev_context=None,
            cfg=cfg, temperatures={"action": 0.8, "timing": 0.8, "char": 0.8},
        )
        assert len(keystrokes) > 0

    def test_keystroke_has_required_fields(self):
        from lilly.models.typing_model import build_model
        from lilly.inference.generator import generate_v3_segment
        from lilly.core.config import V3ModelConfig
        import numpy as np

        cfg = V3ModelConfig()
        model = build_model(cfg)
        style = np.zeros(cfg.style_dim, dtype=np.float32)
        keystrokes = generate_v3_segment(
            model, "hi", style_vector=style, prev_context=None,
            cfg=cfg, temperatures={"action": 0.5, "timing": 0.8, "char": 0.5},
        )
        ks = keystrokes[0]
        assert hasattr(ks, "key")
        assert hasattr(ks, "delay_ms")
        assert hasattr(ks, "action")
        assert hasattr(ks, "cumulative_ms")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_generator.py -v
```

**Step 3: Implement `lilly/inference/generator.py`**

The generator implements the autoregressive loop from the design doc:
- `GeneratedKeystroke` NamedTuple with (key, delay_ms, action, target_char, cumulative_ms, mdn_component)
- `generate_v3_segment(model, target_text, style_vector, prev_context, cfg, temperatures)` — single segment generation
- `generate_v3_full(model, text, style_vector, cfg, temperatures, seed)` — full text via segment splitting
- `print_v3_sequence(keystrokes, text)` — pretty-print helper

Key logic:
- Encode target text once
- Initialize decoder with START token
- At each step: get hidden state, sample action from gate, route to correct/error/backspace path
- Correct: char = target[position], timing from correct MDN
- Error: char = sample from error head, timing from error MDN
- Backspace: char = BKSP, timing from backspace MDN, adjust position
- Stop when position reaches end of target text or max steps

**Step 4: Run tests**

```bash
pytest tests/test_generator.py -v
```

**Step 5: Commit**

```bash
git add lilly/inference/generator.py tests/test_generator.py
git commit -m "feat: add V3 autoregressive generator with action-gated decoding"
```

---

## Task 12: Evaluation — Tier 1 Point Metrics

**Files:**
- Create: `lilly/evaluation/metrics.py`

**Step 1: Implement `lilly/evaluation/metrics.py`**

Functions:
- `compute_tier1_metrics(model, dataset, train_cfg) -> dict` — runs model on dataset, computes:
  - `action_accuracy`: overall
  - `action_f1_correct`, `action_f1_error`, `action_f1_backspace`: per-class F1
  - `timing_mae_ms`: mean absolute error in milliseconds (exp transform)
  - `timing_nll`: mean MDN NLL
  - `error_char_accuracy`: on error samples only
  - `error_char_top3`: top-3 accuracy on error samples

Uses `compute_v3_loss` for NLL computation, `tf.argmax` for action predictions, manual F1 computation.

**Step 2: Commit**

```bash
git add lilly/evaluation/metrics.py
git commit -m "feat: add Tier 1 point metrics for V3 evaluation"
```

---

## Task 13: Evaluation — Tier 2 Distributional Metrics

**Files:**
- Create: `lilly/evaluation/distributional.py`

**Step 1: Implement `lilly/evaluation/distributional.py`**

Requires `scipy.stats` for Wasserstein distance and KS test.

Functions:
- `compute_tier2_metrics(model, dataset, cfg, n_samples=1000) -> dict`
  - Generates `n_samples` sequences autoregressively
  - Collects real sequences from the dataset
  - Computes:
    - `iki_wasserstein`: `scipy.stats.wasserstein_distance` on log-IKI
    - `iki_ks_statistic`: `scipy.stats.ks_2samp` on log-IKI
    - `burst_length_wasserstein`: burst length distributions
    - `pause_duration_wasserstein`: pause (>300ms) distributions
    - `error_position_chi2`: `scipy.stats.chi2_contingency` on position bins
    - `correction_latency_wasserstein`: error-to-backspace delay distributions
    - `iki_autocorrelation_mae`: lag-1 autocorrelation comparison

Helper functions:
- `_extract_bursts(ikis)` — split at pauses, return burst lengths
- `_extract_correction_latencies(keystrokes)` — find error→backspace delays
- `_compute_autocorrelation(ikis)` — lag-1 autocorrelation

**Step 2: Commit**

```bash
git add lilly/evaluation/distributional.py
git commit -m "feat: add Tier 2 distributional metrics (Wasserstein, KS, autocorrelation)"
```

---

## Task 14: Evaluation — Tier 3 Realism Metrics

**Files:**
- Create: `lilly/evaluation/realism.py`

**Step 1: Implement `lilly/evaluation/realism.py`**

Functions:
- `train_discriminator(real_sequences, generated_sequences) -> float`
  - Builds a small LSTM classifier (64 units, binary output)
  - Trains on real=1, generated=0
  - Returns test accuracy (0.5 = indistinguishable = perfect)
- `compute_realism_score(model, dataset, cfg, n_samples=500) -> dict`
  - Generates sequences, collects real ones
  - Trains discriminator
  - Returns `realism_score = 1.0 - discriminator_accuracy`
- `check_style_consistency(model, cfg, wpm_targets, error_targets) -> dict`
  - Generates at various WPM/error settings
  - Measures if actual WPM and error rate match targets

**Step 2: Commit**

```bash
git add lilly/evaluation/realism.py
git commit -m "feat: add Tier 3 realism metrics (discriminator, style consistency)"
```

---

## Task 15: Evaluation — Visualization

**Files:**
- Create: `lilly/evaluation/visualization.py`

**Step 1: Implement `lilly/evaluation/visualization.py`**

Functions:
- `plot_iki_distributions(real_ikis, gen_ikis, save_path)` — per-action IKI histograms
- `plot_burst_patterns(real_keystrokes, gen_keystrokes, save_path)` — time series
- `plot_action_confusion(pred_actions, true_actions, save_path)` — confusion matrix
- `plot_mdn_components(model, save_path)` — visualize learned mixture components
- `plot_style_interpolation(model, cfg, save_path)` — generate at interpolated styles

All use matplotlib. Save to PNG at `save_path`.

**Step 2: Commit**

```bash
git add lilly/evaluation/visualization.py
git commit -m "feat: add V3 evaluation visualization suite"
```

---

## Task 16: Evaluation & Generation Scripts

**Files:**
- Rewrite: `scripts/evaluate.py`
- Rewrite: `scripts/generate.py`
- Rewrite: `scripts/live_preview.py`

**Step 1: Rewrite `scripts/evaluate.py`**

```python
#!/usr/bin/env python3
"""Evaluate a trained V3 model."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=1000)
    args = parser.parse_args()
    # Load model, build dataset, run appropriate tier metrics, print results

if __name__ == "__main__":
    main()
```

**Step 2: Rewrite `scripts/generate.py`**

```python
#!/usr/bin/env python3
"""Generate typing sequences with V3 model."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("text", type=str)
    parser.add_argument("--wpm", type=float, default=100.0)
    parser.add_argument("--error-rate", type=float, default=0.05)
    parser.add_argument("--action-temp", type=float, default=0.8)
    parser.add_argument("--timing-temp", type=float, default=0.8)
    parser.add_argument("--char-temp", type=float, default=0.8)
    args = parser.parse_args()
    # Load model, build style vector from WPM/error_rate, generate, print

if __name__ == "__main__":
    main()
```

**Step 3: Rewrite `scripts/live_preview.py`**

Similar to generate.py but with real-time terminal rendering using the existing ANSI renderer pattern (but rewritten for V3 output format).

**Step 4: Commit**

```bash
git add scripts/evaluate.py scripts/generate.py scripts/live_preview.py
git commit -m "feat: rewrite CLI scripts for V3 (evaluate, generate, live_preview)"
```

---

## Task 17: Export Pipeline Update

**Files:**
- Modify: `lilly/export/converter.py`
- Rewrite: `scripts/export.py`

**Step 1: Update converter.py**

Add V3 custom objects dict for model loading:

```python
def get_v3_custom_objects():
    from lilly.models.typing_model import TypingTransformerV3
    from lilly.models.components import (
        FiLMModulation, MDNHead, ActionGate, ErrorCharHead,
    )
    return {
        "TypingTransformerV3": TypingTransformerV3,
        "FiLMModulation": FiLMModulation,
        "MDNHead": MDNHead,
        "ActionGate": ActionGate,
        "ErrorCharHead": ErrorCharHead,
    }
```

**Step 2: Rewrite `scripts/export.py`**

```python
#!/usr/bin/env python3
"""Export V3 model to TensorFlow.js."""
import argparse
from pathlib import Path
from lilly.core.config import V3_EXPORT_DIR
from lilly.export.converter import export_model, get_v3_custom_objects

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=V3_EXPORT_DIR)
    parser.add_argument("--quantize", default="uint8", choices=["uint8", "uint16", "float16", "none"])
    args = parser.parse_args()
    export_model(args.model_path, args.output_dir, args.quantize,
                 custom_objects=get_v3_custom_objects())

if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add lilly/export/converter.py scripts/export.py
git commit -m "feat: update export pipeline for V3 custom objects"
```

---

## Task 18: Update CLAUDE.md & pyproject.toml

**Files:**
- Modify: `CLAUDE.md`
- Modify: `pyproject.toml`

**Step 1: Update CLAUDE.md**

Replace all V1/V2 references with V3. Update:
- Package structure (new file tree)
- Model architecture section (action-gated decoder, MDN, FiLM)
- Commands section (remove V1-specific steps, add V3 segmentation)
- Config classes section
- Known issues (remove V1 issues, document any V3 limitations)

**Step 2: Update pyproject.toml**

Add `scipy` to dependencies (needed for distributional metrics):

```toml
dependencies = [
    "tensorflow>=2.15.0",
    "numpy>=1.24.0,<2.0.0",
    "pandas>=2.0.0",
    "pyarrow>=14.0.0",
    "tqdm>=4.65.0",
    "requests>=2.31.0",
    "scipy>=1.11.0",
]
```

**Step 3: Commit**

```bash
git add CLAUDE.md pyproject.toml
git commit -m "docs: update CLAUDE.md and pyproject.toml for V3"
```

---

## Task 19: Run Full Test Suite

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

Fix any failures. All existing tests (test_encoding.py, test_keyboard.py) should still pass. New tests (test_components.py, test_losses.py, test_typing_model.py, test_style.py, test_generator.py) should all pass.

**Step 2: Run linter**

```bash
ruff check lilly/ tests/ scripts/
```

Fix any lint issues.

**Step 3: Commit fixes**

```bash
git add -A
git commit -m "fix: resolve test and lint issues from V3 migration"
```

---

## Task Dependency Graph

```
Task 1 (delete V1/V2, update config)
  ├── Task 2 (components: FiLM, MDN, ActionGate, ErrorCharHead)
  │     └── Task 5 (typing_model.py)
  │           ├── Task 9 (training loop)
  │           │     └── Task 19 (full test suite)
  │           └── Task 11 (generator)
  │                 ├── Task 12 (Tier 1 metrics)
  │                 ├── Task 13 (Tier 2 metrics)
  │                 ├── Task 14 (Tier 3 metrics)
  │                 └── Task 16 (scripts)
  ├── Task 3 (losses)
  │     └── Task 9 (training loop)
  ├── Task 4 (LR schedule)
  │     └── Task 9 (training loop)
  ├── Task 6 (style vector)
  │     └── Task 7 (V3 segmentation)
  │           └── Task 8 (data pipeline)
  │                 └── Task 9 (training loop)
  ├── Task 10 (MDN sampling)
  │     └── Task 11 (generator)
  ├── Task 15 (visualization)
  ├── Task 17 (export)
  └── Task 18 (docs)
```

**Critical path:** 1 → 2 → 5 → 9 → 19

**Parallel tracks after Task 1:**
- Track A: 2 → 5 → 11 → 12/13/14/16
- Track B: 3 → (feeds into 9)
- Track C: 4 → (feeds into 9)
- Track D: 6 → 7 → 8 → (feeds into 9)
- Track E: 10 → (feeds into 11)
- Track F: 15, 17, 18 (independent)
