#!/usr/bin/env python3
"""Train the V3 typing model."""
import argparse
from pathlib import Path

from lilly.cli.ui import BannerAnimator, ProgressUI, print_banner, t
from lilly.core.config import V3_MODEL_DIR, V3_SEGMENT_DIR, V3ModelConfig, V3TrainConfig
from lilly.data.pipeline import build_v3_datasets
from lilly.models.typing_model import build_model
from lilly.training.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V3 typing model")
    parser.add_argument("--data-dir", type=Path, default=V3_SEGMENT_DIR)
    parser.add_argument("--model-dir", type=Path, default=V3_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    model_cfg = V3ModelConfig()
    train_cfg = V3TrainConfig()
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_cfg.learning_rate = args.learning_rate

    print_banner()

    animator = BannerAnimator()
    ui = ProgressUI(4, animator)
    animator.start()

    # Step 1: Load datasets
    label = "Loading datasets"
    ui.begin(label)
    train_ds, val_ds, test_ds, n_total = build_v3_datasets(
        args.data_dir, model_cfg, train_cfg, max_files=args.max_files,
    )
    ui.done(label, f"{n_total:,} samples")

    # Step 2: Build model
    label = "Building model"
    ui.begin(label)
    model = build_model(model_cfg)
    params = model.count_params()
    ui.done(label, f"{params:,} parameters")

    # Step 3: Training
    label = "Training"
    ui.begin(label)

    def on_epoch(epoch, train_loss, val_loss, acc, lr, time_s):
        ui.update(
            f"epoch {epoch}/{train_cfg.epochs} | "
            f"loss={train_loss:.4f}/{val_loss:.4f} | "
            f"acc={acc:.3f} | {time_s:.1f}s"
        )

    run_dir, metadata = train(
        data_dir=args.data_dir, model_dir=args.model_dir,
        model_cfg=model_cfg, train_cfg=train_cfg,
        run_name=args.run_name, max_files=args.max_files,
        progress_callback=on_epoch,
    )
    epochs_trained = metadata.get("epochs_trained", "?")
    best_loss = metadata.get("best_val_loss", 0)
    ui.done(label, f"{epochs_trained} epochs, best_val_loss={best_loss:.4f}")

    # Step 4: Saving
    label = "Saving model"
    ui.begin(label)
    ui.done(label, str(run_dir))

    animator.stop()
    ui.finish()

    print()
    print(f"  {t.GREEN}Training complete.{t.RESET}")
    print(f"  Run directory: {run_dir}")
    print()


if __name__ == "__main__":
    main()
