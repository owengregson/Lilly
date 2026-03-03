"""V3 training loop with custom GradientTape.

Implements the full training pipeline: model construction, optimizer setup,
training/validation steps, checkpointing, early stopping, and logging.

Pure library module — no prints. Scripts own the UI layer.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from lilly.core.config import V3ModelConfig, V3TrainConfig
from lilly.data.pipeline import build_v3_datasets
from lilly.models.typing_model import build_model
from lilly.training.losses import V3LossConfig, compute_v3_loss
from lilly.training.schedule import WarmupCosineDecay


def _compute_action_accuracy(outputs, labels):
    """Compute masked action accuracy."""
    mask = labels["label_mask"]
    pred_actions = tf.argmax(outputs["action_probs"], axis=-1)
    true_actions = tf.cast(labels["action_labels"], tf.int64)
    correct = tf.cast(tf.equal(pred_actions, true_actions), tf.float32)
    return tf.reduce_sum(correct * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def _compute_timing_mae(outputs, labels):
    """Compute masked timing MAE in log-space."""
    mask = labels["label_mask"]
    # Use the correct-action MDN mean as prediction
    pi, mu, _ = outputs["timing_correct"]
    # Select the most probable component's mu via one-hot gather
    best_k = tf.argmax(pi, axis=-1)  # (B, T)
    best_k_onehot = tf.one_hot(best_k, depth=tf.shape(mu)[-1])  # (B, T, K)
    pred_delay_log = tf.reduce_sum(mu * best_k_onehot, axis=-1)  # (B, T)
    true_delay_log = labels["delay_labels"]
    mae_log = tf.abs(pred_delay_log - true_delay_log) * mask
    return tf.reduce_sum(mae_log) / tf.maximum(tf.reduce_sum(mask), 1.0)


@tf.function
def train_step(model, optimizer, inputs, labels, loss_cfg):
    """Single training step with gradient computation."""
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        total_loss, components = compute_v3_loss(outputs, labels, loss_cfg)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    action_acc = _compute_action_accuracy(outputs, labels)
    timing_mae = _compute_timing_mae(outputs, labels)

    return total_loss, components, action_acc, timing_mae


@tf.function
def val_step(model, inputs, labels, loss_cfg):
    """Single validation step (no gradients)."""
    outputs = model(inputs, training=False)
    total_loss, components = compute_v3_loss(outputs, labels, loss_cfg)
    action_acc = _compute_action_accuracy(outputs, labels)
    timing_mae = _compute_timing_mae(outputs, labels)
    return total_loss, components, action_acc, timing_mae


def train(
    data_dir: Path,
    model_dir: Path,
    model_cfg: V3ModelConfig | None = None,
    train_cfg: V3TrainConfig | None = None,
    run_name: str | None = None,
    max_files: int = 0,
    progress_callback: Optional[Callable] = None,
    batch_callback: Optional[Callable] = None,
    datasets: tuple | None = None,
    model: tf.keras.Model | None = None,
) -> tuple[Path, dict]:
    """Train the V3 model.

    Args:
        data_dir: Directory containing V3 segment .npz files.
        model_dir: Parent directory for run output.
        model_cfg: Model configuration.
        train_cfg: Training configuration.
        run_name: Name for this training run.
        max_files: Max segment files to load (0 = all).
        progress_callback: Called with (epoch, train_loss, val_loss, acc, lr, time_s)
            at the end of each epoch.
        batch_callback: Called during training with
            (epoch, batch, steps_per_epoch, total_epochs, batch_loss).
            Throttled internally to ~4 updates/sec.
        datasets: Pre-built (train_ds, val_ds, test_ds, n_total) tuple to avoid
            reloading data. If None, datasets are built from data_dir.
        model: Pre-built model. If None, a new model is built from model_cfg.

    Returns:
        (run_dir, metadata_dict)
    """
    if model_cfg is None:
        model_cfg = V3ModelConfig()
    if train_cfg is None:
        train_cfg = V3TrainConfig()

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"run_{timestamp}"
    run_dir = model_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets (or reuse pre-built ones)
    if datasets is not None:
        train_ds, val_ds, test_ds, n_total = datasets
    else:
        train_ds, val_ds, test_ds, n_total = build_v3_datasets(
            data_dir, model_cfg, train_cfg, max_files=max_files,
        )

    # Build model (or reuse pre-built one)
    if model is None:
        model = build_model(model_cfg)

    # Compute total steps for LR schedule
    steps_per_epoch = max(1, n_total // train_cfg.batch_size)
    total_steps = steps_per_epoch * train_cfg.epochs

    # Build optimizer
    lr_schedule = WarmupCosineDecay(
        peak_lr=train_cfg.learning_rate,
        warmup_steps=train_cfg.warmup_steps,
        decay_steps=total_steps,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=train_cfg.weight_decay,
        beta_2=0.98,
        epsilon=1e-9,
    )

    loss_cfg = V3LossConfig(
        action_weight=train_cfg.action_loss_weight,
        timing_weight=train_cfg.timing_loss_weight,
        error_char_weight=train_cfg.error_char_loss_weight,
        position_weight=train_cfg.position_loss_weight,
        focal_gamma=train_cfg.focal_gamma,
        focal_alpha=train_cfg.focal_alpha,
    )

    # CSV logger
    csv_path = run_dir / "training_log.csv"
    best_val_loss = float("inf")
    patience_counter = 0
    epoch = 0

    csv_file = open(csv_path, "w", newline="")
    try:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "epoch", "train_loss", "train_action", "train_timing",
            "train_error_char", "train_position", "train_acc",
            "train_timing_mae",
            "val_loss", "val_action", "val_timing",
            "val_error_char", "val_position", "val_acc", "val_timing_mae",
            "lr", "time_s",
        ])

        for epoch in range(1, train_cfg.epochs + 1):
            epoch_start = time.time()

            # --- Training ---
            train_losses = {"total": [], "action": [], "timing": [],
                           "error_char": [], "position": []}
            train_accs = []
            train_maes = []

            _last_batch_cb = epoch_start
            for batch_idx, (batch_inputs, batch_labels) in enumerate(train_ds, 1):
                total, comps, acc, mae = train_step(
                    model, optimizer, batch_inputs, batch_labels, loss_cfg,
                )
                train_losses["total"].append(float(total))
                train_losses["action"].append(float(comps["action"]))
                train_losses["timing"].append(float(comps["timing"]))
                train_losses["error_char"].append(float(comps["error_char"]))
                train_losses["position"].append(float(comps["position"]))
                train_accs.append(float(acc))
                train_maes.append(float(mae))

                if batch_callback:
                    now = time.time()
                    if now - _last_batch_cb >= 0.25:
                        _last_batch_cb = now
                        batch_callback(
                            epoch, batch_idx, steps_per_epoch,
                            train_cfg.epochs, float(total),
                        )

            # --- Validation ---
            val_losses = {"total": [], "action": [], "timing": [],
                         "error_char": [], "position": []}
            val_accs = []
            val_maes = []

            for batch_inputs, batch_labels in val_ds:
                total, comps, acc, mae = val_step(
                    model, batch_inputs, batch_labels, loss_cfg,
                )
                val_losses["total"].append(float(total))
                val_losses["action"].append(float(comps["action"]))
                val_losses["timing"].append(float(comps["timing"]))
                val_losses["error_char"].append(float(comps["error_char"]))
                val_losses["position"].append(float(comps["position"]))
                val_accs.append(float(acc))
                val_maes.append(float(mae))

            # --- Epoch stats ---
            epoch_time = time.time() - epoch_start
            train_loss = np.mean(train_losses["total"])
            val_loss = np.mean(val_losses["total"])
            current_lr = float(lr_schedule(optimizer.iterations))

            if progress_callback:
                progress_callback(
                    epoch, float(train_loss), float(val_loss),
                    float(np.mean(train_accs)), current_lr, epoch_time,
                )

            # Log to CSV
            csv_writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{np.mean(train_losses['action']):.6f}",
                f"{np.mean(train_losses['timing']):.6f}",
                f"{np.mean(train_losses['error_char']):.6f}",
                f"{np.mean(train_losses['position']):.6f}",
                f"{np.mean(train_accs):.6f}",
                f"{np.mean(train_maes):.6f}",
                f"{val_loss:.6f}",
                f"{np.mean(val_losses['action']):.6f}",
                f"{np.mean(val_losses['timing']):.6f}",
                f"{np.mean(val_losses['error_char']):.6f}",
                f"{np.mean(val_losses['position']):.6f}",
                f"{np.mean(val_accs):.6f}",
                f"{np.mean(val_maes):.6f}",
                f"{current_lr:.8f}",
                f"{epoch_time:.2f}",
            ])
            csv_file.flush()

            # --- Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_path = run_dir / "best_model.keras"
                model.save(str(best_path))
            else:
                patience_counter += 1
                if patience_counter >= train_cfg.early_stop_patience:
                    break
    finally:
        csv_file.close()

    # Save final model
    final_path = run_dir / "final_model.keras"
    model.save(str(final_path))

    # Save metadata
    metadata = {
        "run_name": run_name,
        "model_config": {k: v for k, v in model_cfg.__dict__.items()},
        "train_config": {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in train_cfg.__dict__.items()},
        "total_segments": n_total,
        "best_val_loss": float(best_val_loss),
        "epochs_trained": epoch,
        "total_params": model.count_params(),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return run_dir, metadata
