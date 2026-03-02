"""V2 Transformer model trainer using custom GradientTape loop."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from lilly.core.config import V2_MODEL_DIR, V2_SEGMENT_DIR, V2ModelConfig, V2TrainConfig
from lilly.data.pipeline import build_v2_datasets
from lilly.models.transformer import (
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    TypingTransformer,
    build_model,
    compute_loss,
)


def train_step(model, optimizer, inputs, labels, train_cfg):
    """Single training step with gradient tape."""
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        total_loss, char_loss, timing_loss = compute_loss(
            outputs, labels,
            char_weight=train_cfg.char_loss_weight,
            timing_weight=train_cfg.timing_loss_weight,
        )

    grads = tape.gradient(total_loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    mask = labels["label_mask"]
    char_logits = outputs["char_logits"]
    preds = tf.argmax(char_logits, axis=-1, output_type=tf.int32)
    correct = tf.cast(tf.equal(preds, labels["char_labels"]), tf.float32)
    accuracy = tf.reduce_sum(correct * mask) / tf.reduce_sum(mask)

    return total_loss, char_loss, timing_loss, accuracy


def val_step(model, inputs, labels, train_cfg):
    """Single validation step (no gradients)."""
    outputs = model(inputs, training=False)
    total_loss, char_loss, timing_loss = compute_loss(
        outputs, labels,
        char_weight=train_cfg.char_loss_weight,
        timing_weight=train_cfg.timing_loss_weight,
    )

    mask = labels["label_mask"]
    preds = tf.argmax(outputs["char_logits"], axis=-1, output_type=tf.int32)
    correct = tf.cast(tf.equal(preds, labels["char_labels"]), tf.float32)
    accuracy = tf.reduce_sum(correct * mask) / tf.reduce_sum(mask)

    return total_loss, char_loss, timing_loss, accuracy


def train(
    data_dir: Path = V2_SEGMENT_DIR,
    model_dir: Path = V2_MODEL_DIR,
    model_cfg: V2ModelConfig | None = None,
    train_cfg: V2TrainConfig | None = None,
    run_name: str | None = None,
    max_files: int = 0,
) -> Path:
    """Run V2 model training. Returns path to the run directory."""
    model_cfg = model_cfg or V2ModelConfig()
    train_cfg = train_cfg or V2TrainConfig()
    run_name = run_name or datetime.now().strftime("v2_run_%Y%m%d_%H%M%S")
    run_dir = model_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Lilly V2 Phrase-Level Typing Model - Training")
    print("=" * 60)
    print(f"  Run:         {run_name}")
    print(f"  Epochs:      {train_cfg.epochs}")
    print(f"  Batch size:  {train_cfg.batch_size}")
    print(f"  LR:          {train_cfg.learning_rate}")
    print()

    # Data
    train_ds, val_ds, test_ds, n_total = build_v2_datasets(
        data_dir, model_cfg, train_cfg, max_files=max_files
    )

    # Model
    model = build_model(model_cfg)
    model.summary()

    # Optimizer with cosine decay
    total_train_steps = max(1, n_total // train_cfg.batch_size * train_cfg.epochs)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=train_cfg.learning_rate,
        decay_steps=total_train_steps, alpha=1e-6,
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule, weight_decay=train_cfg.weight_decay,
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    csv_path = run_dir / "training_log.csv"

    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,train_char_loss,train_timing_loss,train_acc,"
                "val_loss,val_char_loss,val_timing_loss,val_acc\n")

    for epoch in range(train_cfg.epochs):
        train_losses, train_char_losses, train_timing_losses, train_accs = [], [], [], []
        for inputs, labels in train_ds:
            tl, cl, tl2, acc = train_step(model, optimizer, inputs, labels, train_cfg)
            train_losses.append(tl.numpy())
            train_char_losses.append(cl.numpy())
            train_timing_losses.append(tl2.numpy())
            train_accs.append(acc.numpy())

        val_losses, val_char_losses, val_timing_losses, val_accs = [], [], [], []
        for inputs, labels in val_ds:
            vl, vcl, vtl, vacc = val_step(model, inputs, labels, train_cfg)
            val_losses.append(vl.numpy())
            val_char_losses.append(vcl.numpy())
            val_timing_losses.append(vtl.numpy())
            val_accs.append(vacc.numpy())

        t_loss = np.mean(train_losses)
        t_char = np.mean(train_char_losses)
        t_time = np.mean(train_timing_losses)
        t_acc = np.mean(train_accs)
        v_loss = np.mean(val_losses)
        v_char = np.mean(val_char_losses)
        v_time = np.mean(val_timing_losses)
        v_acc = np.mean(val_accs)

        print(f"Epoch {epoch+1}/{train_cfg.epochs}: "
              f"loss={t_loss:.4f} char={t_char:.4f} timing={t_time:.4f} acc={t_acc:.4f} | "
              f"val_loss={v_loss:.4f} val_char={v_char:.4f} val_timing={v_time:.4f} val_acc={v_acc:.4f}")

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{t_loss},{t_char},{t_time},{t_acc},"
                    f"{v_loss},{v_char},{v_time},{v_acc}\n")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            model.save(str(run_dir / "best_model.keras"))
            print(f"  -> Saved best model (val_loss={v_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Final evaluation
    print("\nEvaluating on test set...")
    custom_objects = {
        "TypingTransformer": TypingTransformer,
        "TransformerEncoderLayer": TransformerEncoderLayer,
        "TransformerDecoderLayer": TransformerDecoderLayer,
    }
    model = keras.models.load_model(
        str(run_dir / "best_model.keras"), compile=False,
        custom_objects=custom_objects,
    )
    test_losses, test_accs = [], []
    for inputs, labels in test_ds:
        tl, _, _, acc = val_step(model, inputs, labels, train_cfg)
        test_losses.append(tl.numpy())
        test_accs.append(acc.numpy())

    test_loss = np.mean(test_losses)
    test_acc = np.mean(test_accs)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    # Save metadata
    meta = {
        "model_config": model_cfg.__dict__,
        "train_config": train_cfg.__dict__,
        "total_samples": n_total,
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "run_name": run_name,
    }
    with open(run_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Model saved to: {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V2 typing model")
    parser.add_argument("--data-dir", type=Path, default=V2_SEGMENT_DIR)
    parser.add_argument("--model-dir", type=Path, default=V2_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    train_cfg = V2TrainConfig()
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
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
