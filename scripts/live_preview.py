#!/usr/bin/env python3
"""Live typing preview in the terminal for V3 model.

Usage:
    python scripts/live_preview.py "Hello, world!"
    python scripts/live_preview.py --model models/v3/run_XXX/best_model.keras "Test"
    python scripts/live_preview.py --wpm 80 --speed 0.5 "Some text"
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

tf.get_logger().setLevel("ERROR")

# ANSI escape codes
BOLD = "\033[1m"
DIM = "\033[2m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"
CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"


def play_keystrokes(keystrokes, text: str, speed: float = 1.0) -> None:
    """Render keystrokes in real-time in the terminal."""
    sys.stdout.write(CURSOR_HIDE)
    sys.stdout.write(f"\n{DIM}Target: {text!r}{RESET}\n\n")
    sys.stdout.flush()

    buffer = []
    for ks in keystrokes:
        delay_s = (ks.delay_ms / 1000.0) * speed
        time.sleep(max(delay_s, 0.001))

        if ks.action == 0:  # correct
            sys.stdout.write(ks.key)
            buffer.append(ks.key)
        elif ks.action == 1:  # error
            sys.stdout.write(f"{RED}{ks.key}{RESET}")
            buffer.append(ks.key)
        elif ks.action == 2:  # backspace
            if buffer:
                buffer.pop()
                sys.stdout.write("\b \b")

        sys.stdout.flush()

    # Final summary
    total_ms = keystrokes[-1].cumulative_ms if keystrokes else 0
    n_errors = sum(1 for ks in keystrokes if ks.action == 1)
    n_backspace = sum(1 for ks in keystrokes if ks.action == 2)
    actual_wpm = (len(text) / 5.0) / max(total_ms / 60000.0, 0.001) if total_ms > 0 else 0

    sys.stdout.write(f"\n\n{DIM}---{RESET}\n")
    sys.stdout.write(
        f"{DIM}{total_ms:.0f}ms | {actual_wpm:.0f} WPM | "
        f"{n_errors} errors | {n_backspace} backspaces{RESET}\n"
    )
    sys.stdout.write(CURSOR_SHOW)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Live V3 typing preview")
    parser.add_argument("text", nargs="?", default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--wpm", type=float, default=100.0)
    parser.add_argument("--error-rate", type=float, default=0.05)
    parser.add_argument("--action-temp", type=float, default=0.8)
    parser.add_argument("--timing-temp", type=float, default=0.8)
    parser.add_argument("--char-temp", type=float, default=0.8)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (0.5 = half speed)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    from lilly.core.config import V3_MODEL_DIR, V3ModelConfig
    from lilly.export.converter import get_v3_custom_objects
    from lilly.inference.generator import generate_v3_full

    cfg = V3ModelConfig()

    # Find model
    if args.model is not None:
        model_path = args.model
    else:
        candidates = sorted(V3_MODEL_DIR.glob("run_*/best_model.keras"))
        if not candidates:
            print("No trained V3 model found. Specify --model path.")
            sys.exit(1)
        model_path = candidates[-1]

    print(f"{DIM}Loading model: {model_path}{RESET}")
    model = keras.models.load_model(
        str(model_path), compile=False,
        custom_objects=get_v3_custom_objects(),
    )

    # Build style vector
    style = np.zeros(cfg.style_dim, dtype=np.float32)
    target_iki_ms = 60000.0 / (5.0 * max(args.wpm, 1.0))
    style[0] = float(np.log(max(target_iki_ms, 10.0)))
    style[1] = 0.5
    style[4] = args.error_rate

    temperatures = {
        "action": args.action_temp,
        "timing": args.timing_temp,
        "char": args.char_temp,
    }

    try:
        while True:
            if args.text:
                text = args.text
            else:
                print(f"\n{BOLD}Enter text to type{RESET} {DIM}(Ctrl+C to quit):{RESET}")
                try:
                    text = input("> ").strip()
                except EOFError:
                    break
                if not text:
                    continue

            text = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
            if not text:
                print(f"{YELLOW}No valid characters.{RESET}")
                if args.text:
                    break
                continue

            keystrokes = generate_v3_full(
                model, text, style, cfg,
                temperatures=temperatures,
                seed=args.seed,
            )
            play_keystrokes(keystrokes, text, speed=args.speed)

            if args.text:
                break
    except KeyboardInterrupt:
        sys.stdout.write(CURSOR_SHOW)
        print(f"\n\n{DIM}Interrupted.{RESET}\n")


if __name__ == "__main__":
    main()
