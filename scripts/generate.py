#!/usr/bin/env python3
"""Generate a typing sequence.

Usage:
    python scripts/generate.py models/run_XXX/final_model.keras "Hello!"
    python scripts/generate.py --version v2 models/v2/run_XXX/best_model.keras "Hello!"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser(description="Generate typing sequence")
    parser.add_argument("model_path", type=Path, help="Path to .keras model")
    parser.add_argument("text", nargs="?", default=None, help="Text to type")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--input-file", type=Path, help="Read text from file")
    parser.add_argument("--wpm", type=float, default=100.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    if args.input_file:
        text = args.input_file.read_text().strip()
    elif args.text:
        text = args.text
    else:
        text = "The quick brown fox jumps over the lazy dog."

    if args.version == "v2":
        from lilly.inference.preview import load_v2_model
        from lilly.inference.generator import generate_v2_full, print_v2_sequence

        model = load_v2_model(args.model_path)
        keystrokes = generate_v2_full(model, text, wpm=args.wpm, temperature=args.temperature)
        print_v2_sequence(keystrokes, text)
    else:
        model = keras.models.load_model(str(args.model_path), compile=False)
        from lilly.inference.generator import generate_v1, print_v1_sequence

        keystrokes = generate_v1(model, text, wpm=args.wpm, temperature=args.temperature)
        print_v1_sequence(keystrokes, text)


if __name__ == "__main__":
    main()
