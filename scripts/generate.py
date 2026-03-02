#!/usr/bin/env python3
"""Generate typing sequences with V3 model.

Usage:
    python scripts/generate.py models/v3/run_XXX/best_model.keras "Hello, world!"
    python scripts/generate.py models/v3/run_XXX/best_model.keras "Test" --wpm 60
    python scripts/generate.py models/v3/run_XXX/best_model.keras "Test" --action-temp 0.5
"""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser(description="Generate V3 typing sequence")
    parser.add_argument("model_path", type=Path, help="Path to .keras model")
    parser.add_argument("text", nargs="?", default=None, help="Text to type")
    parser.add_argument("--input-file", type=Path, help="Read text from file")
    parser.add_argument("--wpm", type=float, default=100.0,
                        help="Target WPM (used to set style vector)")
    parser.add_argument("--error-rate", type=float, default=0.05,
                        help="Target error rate (used to set style vector)")
    parser.add_argument("--action-temp", type=float, default=0.8)
    parser.add_argument("--timing-temp", type=float, default=0.8)
    parser.add_argument("--char-temp", type=float, default=0.8)
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

    # Filter to printable ASCII
    text = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
    if not text:
        print("No valid characters in input.")
        return

    from lilly.core.config import V3ModelConfig
    from lilly.export.converter import get_v3_custom_objects
    from lilly.inference.generator import generate_v3_full, print_v3_sequence

    cfg = V3ModelConfig()

    print(f"Loading model: {args.model_path}")
    model = keras.models.load_model(
        str(args.model_path), compile=False,
        custom_objects=get_v3_custom_objects(),
    )

    # Build style vector from WPM and error rate
    style = np.zeros(cfg.style_dim, dtype=np.float32)
    target_iki_ms = 60000.0 / (5.0 * max(args.wpm, 1.0))
    style[0] = float(np.log(max(target_iki_ms, 10.0)))  # mean_iki_log
    style[1] = 0.5  # std_iki_log (moderate variance)
    style[4] = args.error_rate  # error_rate

    temperatures = {
        "action": args.action_temp,
        "timing": args.timing_temp,
        "char": args.char_temp,
    }

    print(f"Target: {text!r}")
    print(f"Style: WPM={args.wpm}, error_rate={args.error_rate}")
    print(f"Temperatures: {temperatures}")
    print()

    keystrokes = generate_v3_full(
        model, text, style, cfg,
        temperatures=temperatures,
        seed=args.seed,
    )
    print_v3_sequence(keystrokes, text)

    # Summary stats
    if keystrokes:
        total_ms = keystrokes[-1].cumulative_ms
        n_correct = sum(1 for ks in keystrokes if ks.action == 0)
        n_errors = sum(1 for ks in keystrokes if ks.action == 1)
        n_backspace = sum(1 for ks in keystrokes if ks.action == 2)
        actual_wpm = (len(text) / 5.0) / max(total_ms / 60000.0, 0.001)
        print(f"\nSummary:")
        print(f"  Total time: {total_ms:.0f}ms")
        print(f"  Keystrokes: {len(keystrokes)} (correct={n_correct}, error={n_errors}, backspace={n_backspace})")
        print(f"  Actual WPM: {actual_wpm:.1f}")


if __name__ == "__main__":
    main()
