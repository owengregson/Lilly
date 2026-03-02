"""Shared training callbacks for Keras model training."""

from __future__ import annotations

from pathlib import Path

from tensorflow import keras

from lilly.core.config import V1TrainConfig


def make_callbacks(
    model_dir: Path,
    train_cfg: V1TrainConfig,
    run_name: str,
) -> list:
    """Create training callbacks for V1 Keras .fit() training."""
    callbacks = []

    ckpt_path = model_dir / run_name / "best_model.keras"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path), monitor="val_loss",
            save_best_only=True, verbose=1,
        )
    )

    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=train_cfg.early_stop_patience,
            restore_best_weights=True, verbose=1,
        )
    )

    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=train_cfg.lr_decay_factor,
            patience=train_cfg.lr_decay_patience, min_lr=1e-6, verbose=1,
        )
    )

    tb_dir = model_dir / run_name / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        keras.callbacks.TensorBoard(
            log_dir=str(tb_dir), histogram_freq=1, write_graph=False,
        )
    )

    csv_path = model_dir / run_name / "training_log.csv"
    callbacks.append(keras.callbacks.CSVLogger(str(csv_path)))

    return callbacks
