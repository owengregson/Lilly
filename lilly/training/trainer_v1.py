"""V1 LSTM model trainer using Keras .fit() API."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from lilly.core.config import MODEL_DIR, TFRECORD_DIR, V1ModelConfig, V1TrainConfig
from lilly.data.pipeline import build_v1_datasets
from lilly.models.lstm import build_model, compile_model
from lilly.training.callbacks import make_callbacks


def compute_error_sample_weights(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Add sample weights for the error_char head.

    The error_char loss should only apply when the true action is ERROR (1).
    """
    def add_weights(inputs, labels):
        action_labels = labels["action"]
        error_mask = tf.cast(tf.equal(action_labels, 1), tf.float32)

        sample_weights = {
            "timing": tf.ones_like(labels["timing"]),
            "action": tf.ones_like(tf.cast(labels["action"], tf.float32)),
            "error_char": error_mask,
        }
        return inputs, labels, sample_weights

    return dataset.map(add_weights, num_parallel_calls=tf.data.AUTOTUNE)


def train(
    data_dir: Path = TFRECORD_DIR,
    model_dir: Path = MODEL_DIR,
    model_cfg: V1ModelConfig | None = None,
    train_cfg: V1TrainConfig | None = None,
    run_name: str | None = None,
    max_files: int = 0,
    resume_from: str | None = None,
) -> Path:
    """Run V1 model training. Returns path to the run directory."""
    model_cfg = model_cfg or V1ModelConfig()
    train_cfg = train_cfg or V1TrainConfig()
    run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Lilly V1 Typing Model - Training")
    print("=" * 60)
    print(f"  Run:          {run_name}")
    print(f"  Data dir:     {data_dir}")
    print(f"  Epochs:       {train_cfg.epochs}")
    print(f"  Batch size:   {train_cfg.batch_size}")
    print(f"  LR:           {train_cfg.learning_rate}")
    print()

    # Build datasets
    train_ds, val_ds, test_ds, n_total = build_v1_datasets(
        data_dir, train_cfg, max_files=max_files
    )

    # Add error masking sample weights
    train_ds = compute_error_sample_weights(train_ds)
    val_ds = compute_error_sample_weights(val_ds)
    test_ds = compute_error_sample_weights(test_ds)

    # Build model
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        model = keras.models.load_model(resume_from, compile=False)
    else:
        model = build_model(model_cfg)

    compile_model(
        model,
        learning_rate=train_cfg.learning_rate,
        timing_weight=train_cfg.timing_loss_weight,
        action_weight=train_cfg.action_loss_weight,
        error_char_weight=train_cfg.error_char_loss_weight,
    )
    model.summary()
    print()

    # Callbacks
    callbacks = make_callbacks(model_dir, train_cfg, run_name)

    # Train
    print("Starting training...")
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=train_cfg.epochs, callbacks=callbacks, verbose=1,
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_ds, verbose=1, return_dict=True)
    print("\nTest results:")
    for key, val in test_results.items():
        print(f"  {key}: {val:.4f}")

    # Save final model
    run_dir = model_dir / run_name
    final_path = run_dir / "final_model.keras"
    model.save(str(final_path))
    print(f"\nFinal model saved: {final_path}")

    # Save metadata
    meta = {
        "model_config": model_cfg.__dict__,
        "train_config": train_cfg.__dict__,
        "total_samples": n_total,
        "test_results": {k: float(v) for k, v in test_results.items()},
        "total_params": int(model.count_params()),
        "run_name": run_name,
    }
    meta_path = run_dir / "training_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Training metadata saved: {meta_path}")
    print("\nDone.")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V1 typing prediction model")
    parser.add_argument("--data-dir", type=Path, default=TFRECORD_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_cfg = V1TrainConfig()
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate
    if args.max_samples is not None:
        train_cfg.max_samples = args.max_samples

    train(
        data_dir=args.data_dir, model_dir=args.model_dir,
        train_cfg=train_cfg, run_name=args.run_name,
        max_files=args.max_files, resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
