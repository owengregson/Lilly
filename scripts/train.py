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
