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
