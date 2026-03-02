"""Real-time terminal typing preview for V1 and V2 models."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

tf.get_logger().setLevel("ERROR")
from tensorflow import keras  # noqa: E402

from lilly.core.config import (  # noqa: E402
    ACTION_BACKSPACE,
    ACTION_CORRECT,
    ACTION_ERROR,
    MODEL_DIR,
    SEQ_LEN,
    V2ModelConfig,
    V2_MODEL_DIR,
)
from lilly.core.encoding import id_to_char, wpm_to_bucket  # noqa: E402
from lilly.core.losses import LogNormalNLL  # noqa: E402
from lilly.inference.context import ContextWindow  # noqa: E402
from lilly.inference.generator import (  # noqa: E402
    GeneratedKeystrokeV2,
    generate_v2_full,
)
from lilly.inference.sampling import sample_lognormal, weighted_sample  # noqa: E402

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"
CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"


# ---------------------------------------------------------------------------
# V1 Live Preview
# ---------------------------------------------------------------------------

class LiveRenderer:
    """Renders V1 typing session to the terminal in real-time."""

    def __init__(self, target_text: str):
        self.target = target_text
        self.buffer: List[str] = []
        self.errors = 0
        self.backspaces = 0
        self.correct = 0
        self.start_time = 0.0

    def start(self) -> None:
        self.start_time = time.time()
        sys.stdout.write(CURSOR_HIDE)
        sys.stdout.write(f"\n{DIM}Target:{RESET} {self.target}\n\n")
        sys.stdout.write(f"{DIM}Typing:{RESET} ")
        sys.stdout.flush()

    def type_correct(self, ch: str) -> None:
        self.correct += 1
        self.buffer.append(ch)
        sys.stdout.write(ch)
        sys.stdout.flush()

    def type_error(self, ch: str) -> None:
        self.errors += 1
        self.buffer.append(ch)
        sys.stdout.write(f"{RED}{ch}{RESET}")
        sys.stdout.flush()

    def type_backspace(self) -> None:
        self.backspaces += 1
        if self.buffer:
            self.buffer.pop()
            sys.stdout.write("\b \b")
            sys.stdout.flush()

    def finish(self) -> None:
        elapsed = time.time() - self.start_time
        result = "".join(self.buffer)
        total_ks = self.correct + self.errors + self.backspaces
        effective_wpm = (self.correct / 5.0) / (elapsed / 60.0) if elapsed > 0 else 0

        sys.stdout.write(CURSOR_SHOW)
        sys.stdout.write(f"\n\n{DIM}{'─' * 50}{RESET}\n")
        sys.stdout.write(f"  {BOLD}Session Complete{RESET}\n")
        sys.stdout.write(f"  Time:       {elapsed:.1f}s\n")
        sys.stdout.write(f"  Keystrokes: {total_ks}")
        sys.stdout.write(
            f"  ({GREEN}{self.correct} correct{RESET}"
            f" {RED}{self.errors} errors{RESET}"
            f" {YELLOW}{self.backspaces} backspaces{RESET})\n"
        )
        sys.stdout.write(f"  Error rate: {self.errors / max(total_ks, 1) * 100:.1f}%\n")
        sys.stdout.write(f"  Effective:  {effective_wpm:.0f} WPM\n")
        sys.stdout.write(f"  Output:     \"{result}\"\n")
        sys.stdout.write(f"{DIM}{'─' * 50}{RESET}\n\n")
        sys.stdout.flush()


def live_generate_v1(
    model: keras.Model,
    text: str,
    wpm: float = 100.0,
    temperature: float = 0.8,
    speed: float = 1.0,
) -> None:
    """Generate and display a V1 typing session in real-time."""
    max_total = len(text) * 3
    bucket = wpm_to_bucket(wpm)
    ctx = ContextWindow(SEQ_LEN, bucket)
    renderer = LiveRenderer(text)

    buffer: List[str] = []
    total_ks = 0

    renderer.start()

    while len(buffer) < len(text) and total_ks < max_total:
        target_pos = len(buffer)
        target_char = text[target_pos] if target_pos < len(text) else ""

        inputs = ctx.to_model_input()
        predictions = model.predict(inputs, verbose=0)

        timing_params = predictions["timing"][0]
        action_probs = predictions["action"][0]
        error_probs = predictions["error_char"][0]

        delay_ms = sample_lognormal(timing_params[0], timing_params[1])
        action_idx = weighted_sample(action_probs, temperature)

        if action_idx == ACTION_CORRECT:
            time.sleep(delay_ms / 1000.0 / speed)
            buffer.append(target_char)
            renderer.type_correct(target_char)
            ctx.push(
                target_char, target_char, ACTION_CORRECT,
                delay_ms, delay_ms * 0.3, target_pos, len(text), text,
            )

        elif action_idx == ACTION_ERROR:
            error_char_id = weighted_sample(error_probs, temperature)
            error_char = id_to_char(error_char_id)
            if not error_char or error_char == "BKSP":
                error_char = "x"

            time.sleep(delay_ms / 1000.0 / speed)
            buffer.append(error_char)
            renderer.type_error(error_char)
            ctx.push(
                error_char, target_char, ACTION_ERROR,
                delay_ms, delay_ms * 0.3, target_pos, len(text), text,
            )

        elif action_idx == ACTION_BACKSPACE:
            if buffer:
                time.sleep(delay_ms / 1000.0 / speed)
                buffer.pop()
                renderer.type_backspace()
                ctx.push(
                    "BKSP", "", ACTION_BACKSPACE,
                    delay_ms, delay_ms * 0.3,
                    max(0, len(buffer)), len(text), text,
                )
            else:
                time.sleep(delay_ms / 1000.0 / speed)
                buffer.append(target_char)
                renderer.type_correct(target_char)
                ctx.push(
                    target_char, target_char, ACTION_CORRECT,
                    delay_ms, delay_ms * 0.3, target_pos, len(text), text,
                )

        total_ks += 1

    renderer.finish()


# ---------------------------------------------------------------------------
# V2 Live Preview
# ---------------------------------------------------------------------------

def play_v2_keystrokes(
    keystrokes: list[GeneratedKeystrokeV2],
    target: str,
    speed: float = 1.0,
) -> None:
    """Play back V2 keystrokes in real-time to the terminal."""
    buffer: list[str] = []
    backspaces = 0
    correct = 0

    sys.stdout.write(CURSOR_HIDE)
    sys.stdout.write(f"\n{DIM}Target:{RESET} {target}\n\n")
    sys.stdout.write(f"{DIM}Typing:{RESET} ")
    sys.stdout.flush()

    start = time.time()

    for ks in keystrokes:
        time.sleep(ks.delay_ms / 1000.0 / speed)

        if ks.key == "BKSP":
            backspaces += 1
            if buffer:
                buffer.pop()
                sys.stdout.write("\b \b")
        elif ks.key in ("<END>", "<START>", ""):
            continue
        else:
            buffer.append(ks.key)
            correct += 1
            sys.stdout.write(ks.key)

        sys.stdout.flush()

    elapsed = time.time() - start
    result = "".join(buffer)
    total_ks = correct + backspaces
    eff_wpm = (len(result) / 5.0) / (elapsed / 60.0) if elapsed > 0 else 0

    sys.stdout.write(CURSOR_SHOW)
    sys.stdout.write(f"\n\n{DIM}{'─' * 50}{RESET}\n")
    sys.stdout.write(f"  {BOLD}Session Complete{RESET}\n")
    sys.stdout.write(f"  Time:       {elapsed:.1f}s\n")
    sys.stdout.write(f"  Keystrokes: {total_ks}")
    sys.stdout.write(f"  ({GREEN}{correct} chars{RESET} {YELLOW}{backspaces} backspaces{RESET})\n")
    sys.stdout.write(f"  Effective:  {eff_wpm:.0f} WPM\n")
    match = result == target
    tag = f"{GREEN}YES{RESET}" if match else f"{RED}NO{RESET}"
    sys.stdout.write(f"  Match:      {tag}\n")
    if not match:
        sys.stdout.write(f"  Got:        \"{result}\"\n")
    sys.stdout.write(f"{DIM}{'─' * 50}{RESET}\n\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

DEFAULT_V1_MODEL = MODEL_DIR / "run_20260224_173441" / "best_model.keras"


def load_v1_model(path: Path) -> keras.Model:
    """Load a V1 Keras model with custom objects."""
    print(f"{DIM}Loading model from {path.name}...{RESET}", end="", flush=True)
    model = keras.models.load_model(
        str(path), compile=False,
        custom_objects={"LogNormalNLL": LogNormalNLL},
    )
    print(f" {GREEN}done{RESET}")
    return model


def load_v2_model(path: Path) -> keras.Model:
    """Load a V2 Keras model with custom objects."""
    from lilly.models.transformer import (
        TransformerDecoderLayer,
        TransformerEncoderLayer,
        TypingTransformer,
    )
    print(f"{DIM}Loading model from {path.name}...{RESET}", end="", flush=True)
    model = keras.models.load_model(
        str(path), compile=False,
        custom_objects={
            "TypingTransformer": TypingTransformer,
            "TransformerEncoderLayer": TransformerEncoderLayer,
            "TransformerDecoderLayer": TransformerDecoderLayer,
        },
    )
    print(f" {GREEN}done{RESET}")
    return model


def find_latest_v2_model() -> Path | None:
    """Find the most recent V2 best_model.keras."""
    runs = sorted(V2_MODEL_DIR.glob("v2_run_*/best_model.keras"))
    return runs[-1] if runs else None
