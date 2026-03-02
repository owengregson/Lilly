#!/usr/bin/env python3
"""Train a model (V1 or V2).

Usage:
    python scripts/train.py              # V1 default
    python scripts/train.py --version v2 # V2 transformer
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Train Lilly typing model")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1",
                       help="Model version (default: v1)")
    args, remaining = parser.parse_known_args()

    # Re-inject remaining args so the trainer sees them
    sys.argv = [sys.argv[0]] + remaining

    if args.version == "v2":
        from lilly.training.trainer_v2 import main as train_main
    else:
        from lilly.training.trainer_v1 import main as train_main

    train_main()


if __name__ == "__main__":
    main()
