#!/usr/bin/env python3
"""Live typing preview in the terminal.

Usage:
    python scripts/live_preview.py "Hello, world!"
    python scripts/live_preview.py --version v2 "Hello!"
    python scripts/live_preview.py --wpm 80 --speed 0.5 "Some text"
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")


def main():
    parser = argparse.ArgumentParser(description="Live typing preview")
    parser.add_argument("text", nargs="?", default=None)
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--wpm", type=float, default=100.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    if args.version == "v2":
        from lilly.inference.preview import (
            load_v2_model, find_latest_v2_model,
            play_v2_keystrokes, CURSOR_SHOW, DIM, BOLD, YELLOW, RESET,
        )
        from lilly.inference.generator import generate_v2_full

        model_path = args.model or find_latest_v2_model()
        if model_path is None or not model_path.exists():
            print("No trained V2 model found.")
            sys.exit(1)

        model = load_v2_model(model_path)

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

                keystrokes = generate_v2_full(model, text, wpm=args.wpm,
                                             temperature=args.temperature, seed=args.seed)
                play_v2_keystrokes(keystrokes, text, speed=args.speed)

                if args.text:
                    break
        except KeyboardInterrupt:
            sys.stdout.write(CURSOR_SHOW)
            print(f"\n\n{DIM}Interrupted.{RESET}\n")
    else:
        from lilly.inference.preview import load_v1_model, live_generate_v1, CURSOR_SHOW, DIM, BOLD, YELLOW, RESET

        if args.model is None:
            from lilly.core.config import MODEL_DIR
            candidates = sorted(MODEL_DIR.glob("run_*/best_model.keras"))
            if not candidates:
                print("No trained V1 model found.")
                sys.exit(1)
            model_path = candidates[-1]
        else:
            model_path = args.model

        model = load_v1_model(model_path)

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

                live_generate_v1(model, text, wpm=args.wpm,
                                temperature=args.temperature, speed=args.speed)

                if args.text:
                    break
        except KeyboardInterrupt:
            sys.stdout.write(CURSOR_SHOW)
            print(f"\n\n{DIM}Interrupted.{RESET}\n")


if __name__ == "__main__":
    main()
