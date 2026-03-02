"""Keystroke sequence generation for V1 and V2 models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from lilly.core.config import (
    ACTION_BACKSPACE,
    ACTION_CORRECT,
    ACTION_ERROR,
    END_TOKEN,
    MAX_IKI_MS,
    PAD_TOKEN,
    SEQ_LEN,
    START_TOKEN,
    V2ModelConfig,
)
from lilly.core.encoding import char_to_id, id_to_char, wpm_to_bucket
from lilly.inference.context import ContextWindow
from lilly.inference.sampling import sample_lognormal, weighted_sample, weighted_sample_logits


class GeneratedKeystroke(NamedTuple):
    key: str
    delay_ms: float
    action: str
    target_char: str
    cumulative_ms: float


class GeneratedKeystrokeV2(NamedTuple):
    key: str
    delay_ms: float
    cumulative_ms: float


# ---------------------------------------------------------------------------
# V1 generation
# ---------------------------------------------------------------------------

def generate_v1(
    model: keras.Model,
    text: str,
    wpm: float = 100.0,
    temperature: float = 0.8,
    max_total_keystrokes: int = 0,
) -> List[GeneratedKeystroke]:
    """Generate a V1 keystroke sequence for the given text.

    The model predicts timing, action, and error character at each step.
    Errors trigger immediate backspace+retype correction.
    """
    if max_total_keystrokes <= 0:
        max_total_keystrokes = len(text) * 3

    bucket = wpm_to_bucket(wpm)
    ctx = ContextWindow(SEQ_LEN, bucket)
    keystrokes: List[GeneratedKeystroke] = []

    buffer: List[str] = []
    cumulative_ms = 0.0
    total_ks = 0

    while len(buffer) < len(text) and total_ks < max_total_keystrokes:
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
            buffer.append(target_char)
            cumulative_ms += delay_ms

            keystrokes.append(GeneratedKeystroke(
                key=target_char, delay_ms=delay_ms, action="correct",
                target_char=target_char, cumulative_ms=cumulative_ms,
            ))
            ctx.push(
                target_char, target_char, ACTION_CORRECT,
                delay_ms, delay_ms * 0.3, target_pos, len(text), text,
            )

        elif action_idx == ACTION_ERROR:
            error_char_id = weighted_sample(error_probs, temperature)
            error_char = id_to_char(error_char_id)
            if not error_char or error_char == "BKSP":
                error_char = "x"

            buffer.append(error_char)
            cumulative_ms += delay_ms

            keystrokes.append(GeneratedKeystroke(
                key=error_char, delay_ms=delay_ms, action="error",
                target_char=target_char, cumulative_ms=cumulative_ms,
            ))
            ctx.push(
                error_char, target_char, ACTION_ERROR,
                delay_ms, delay_ms * 0.3, target_pos, len(text), text,
            )

            # Immediate correction: backspace + retype
            bksp_delay = delay_ms * 0.6
            cumulative_ms += bksp_delay
            buffer.pop()

            keystrokes.append(GeneratedKeystroke(
                key="BKSP", delay_ms=bksp_delay, action="backspace",
                target_char="", cumulative_ms=cumulative_ms,
            ))
            ctx.push(
                "BKSP", "", ACTION_BACKSPACE,
                bksp_delay, bksp_delay * 0.3, target_pos, len(text), text,
            )

            retype_delay = delay_ms * 0.85
            cumulative_ms += retype_delay
            buffer.append(target_char)

            keystrokes.append(GeneratedKeystroke(
                key=target_char, delay_ms=retype_delay, action="correct",
                target_char=target_char, cumulative_ms=cumulative_ms,
            ))
            ctx.push(
                target_char, target_char, ACTION_CORRECT,
                retype_delay, retype_delay * 0.3, target_pos, len(text), text,
            )
            total_ks += 2

        elif action_idx == ACTION_BACKSPACE:
            if buffer:
                cumulative_ms += delay_ms
                buffer.pop()

                keystrokes.append(GeneratedKeystroke(
                    key="BKSP", delay_ms=delay_ms, action="backspace",
                    target_char="", cumulative_ms=cumulative_ms,
                ))
                ctx.push(
                    "BKSP", "", ACTION_BACKSPACE,
                    delay_ms, delay_ms * 0.3,
                    max(0, len(buffer)), len(text), text,
                )
            else:
                buffer.append(target_char)
                cumulative_ms += delay_ms

                keystrokes.append(GeneratedKeystroke(
                    key=target_char, delay_ms=delay_ms, action="correct",
                    target_char=target_char, cumulative_ms=cumulative_ms,
                ))
                ctx.push(
                    target_char, target_char, ACTION_CORRECT,
                    delay_ms, delay_ms * 0.3, target_pos, len(text), text,
                )

        total_ks += 1

    return keystrokes


# ---------------------------------------------------------------------------
# V2 generation
# ---------------------------------------------------------------------------

def _pad_context(ctx: List[int], length: int) -> List[int]:
    """Left-pad context to target length."""
    if len(ctx) >= length:
        return ctx[-length:]
    return [PAD_TOKEN] * (length - len(ctx)) + ctx


def generate_v2_segment(
    model: keras.Model,
    target_text: str,
    wpm_bucket: int,
    prev_context: List[int],
    cfg: V2ModelConfig,
    temperature: float = 0.8,
    max_steps: int = 0,
) -> List[GeneratedKeystrokeV2]:
    """Generate keystroke sequence for one V2 segment via autoregressive decoding."""
    if max_steps <= 0:
        max_steps = len(target_text) * 3 + 5

    target_ids = [char_to_id(ch) for ch in target_text[:cfg.max_encoder_len]]
    enc_len = len(target_ids)

    encoder_chars = np.zeros((1, cfg.max_encoder_len), dtype=np.int32)
    encoder_chars[0, :enc_len] = target_ids

    static_inputs = {
        "encoder_chars": tf.constant(encoder_chars),
        "encoder_lengths": tf.constant([[enc_len]], dtype=tf.int32),
        "wpm_bucket": tf.constant([wpm_bucket], dtype=tf.int32),
        "prev_context": tf.constant(
            [_pad_context(prev_context, cfg.context_tail_len)], dtype=tf.int32
        ),
        "sentence_pos": tf.constant([0.0], dtype=tf.float32),
        "session_frac": tf.constant([0.0], dtype=tf.float32),
    }

    encoder_output, encoder_padding_mask = model._encode(
        static_inputs["encoder_chars"],
        static_inputs["encoder_lengths"],
        static_inputs["wpm_bucket"][:, tf.newaxis],
        static_inputs["prev_context"],
        training=False,
    )

    dec_max_len = cfg.max_decoder_len + 1
    dec_chars = np.zeros((1, dec_max_len), dtype=np.int32)
    dec_delays = np.zeros((1, dec_max_len), dtype=np.float32)
    dec_chars[0, 0] = START_TOKEN

    keystrokes: List[GeneratedKeystrokeV2] = []
    cumulative_ms = 0.0

    for step in range(1, min(max_steps + 1, dec_max_len)):
        char_logits, delay_params = model._decode(
            tf.constant(dec_chars[:, :step]),
            tf.constant(dec_delays[:, :step]),
            encoder_output, encoder_padding_mask,
            training=False,
        )

        last_logits = char_logits[0, -1].numpy()
        last_delay = delay_params[0, -1].numpy()

        char_id = weighted_sample_logits(last_logits, temperature)

        if char_id == END_TOKEN or char_id == PAD_TOKEN:
            break

        delay_ms = sample_lognormal(last_delay[0], last_delay[1])
        cumulative_ms += delay_ms

        key = id_to_char(char_id)
        keystrokes.append(GeneratedKeystrokeV2(key, delay_ms, cumulative_ms))

        if step < dec_max_len:
            dec_chars[0, step] = char_id
            dec_delays[0, step] = math.log(max(delay_ms, 1.0))

    return keystrokes


def generate_v2_full(
    model: keras.Model,
    text: str,
    wpm: float = 100.0,
    temperature: float = 0.8,
    cfg: V2ModelConfig | None = None,
    seed: int | None = None,
) -> List[GeneratedKeystrokeV2]:
    """Generate V2 keystroke sequence for full text, split into segments."""
    from lilly.data.segment import split_text_into_inference_segments

    if cfg is None:
        cfg = V2ModelConfig()
    if seed is not None:
        np.random.seed(seed)

    bucket = wpm_to_bucket(wpm)
    segments = split_text_into_inference_segments(text, seed=seed)

    all_keystrokes: List[GeneratedKeystrokeV2] = []
    prev_context: List[int] = []
    cumulative_ms = 0.0

    for seg_text in segments:
        seg_keystrokes = generate_v2_segment(
            model, seg_text, bucket, prev_context, cfg, temperature
        )

        for ks in seg_keystrokes:
            cumulative_ms += ks.delay_ms
            all_keystrokes.append(GeneratedKeystrokeV2(ks.key, ks.delay_ms, cumulative_ms))

        if seg_keystrokes:
            prev_context = [char_to_id(ks.key) for ks in seg_keystrokes[-4:]]

    return all_keystrokes


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_v1_sequence(keystrokes: List[GeneratedKeystroke], text: str) -> None:
    """Pretty-print a V1 generated keystroke sequence."""
    print(f"\nTarget text: \"{text}\"")
    print(f"Total keystrokes: {len(keystrokes)}")
    total_ms = keystrokes[-1].cumulative_ms if keystrokes else 0
    print(f"Total time: {total_ms:.0f}ms ({total_ms/1000:.1f}s)")

    n_errors = sum(1 for ks in keystrokes if ks.action == "error")
    n_backspaces = sum(1 for ks in keystrokes if ks.action == "backspace")
    n_correct = sum(1 for ks in keystrokes if ks.action == "correct")
    effective_wpm = (n_correct / 5.0) / (total_ms / 60000.0) if total_ms > 0 else 0

    print(f"Effective WPM: {effective_wpm:.0f}")
    print(f"Errors: {n_errors}  Backspaces: {n_backspaces}  Correct: {n_correct}")
    print(f"Error rate: {n_errors / max(len(keystrokes), 1) * 100:.1f}%")
    print()

    print(f"{'#':>4}  {'Key':>6}  {'Delay':>8}  {'Action':>10}  {'Target':>6}  {'Cumul':>10}")
    print("-" * 56)

    for i, ks in enumerate(keystrokes):
        key_display = repr(ks.key) if ks.key not in ("BKSP",) else "BKSP"
        target_display = repr(ks.target_char) if ks.target_char else "-"
        print(
            f"{i+1:4d}  {key_display:>6s}  {ks.delay_ms:7.1f}ms  "
            f"{ks.action:>10s}  {target_display:>6s}  {ks.cumulative_ms:9.1f}ms"
        )

    buffer = []
    for ks in keystrokes:
        if ks.key == "BKSP":
            if buffer:
                buffer.pop()
        else:
            buffer.append(ks.key)

    result = "".join(buffer)
    print(f"\nReconstructed: \"{result}\"")
    print(f"Matches target: {result == text}")


def print_v2_sequence(keystrokes: List[GeneratedKeystrokeV2], text: str) -> None:
    """Pretty-print a V2 generated keystroke sequence."""
    print(f"\nTarget: \"{text}\"")
    print(f"Keystrokes: {len(keystrokes)}")

    if keystrokes:
        total_ms = keystrokes[-1].cumulative_ms
        print(f"Time: {total_ms:.0f}ms ({total_ms/1000:.1f}s)")

    buffer = []
    backspaces = 0
    for ks in keystrokes:
        if ks.key == "BKSP":
            if buffer:
                buffer.pop()
            backspaces += 1
        else:
            buffer.append(ks.key)

    result = "".join(buffer)
    correct = len(keystrokes) - backspaces
    print(f"Chars: {correct}  Backspaces: {backspaces}")
    print(f"Result: \"{result}\"")
    print(f"Match: {result == text}")

    print(f"\n{'#':>4}  {'Key':>6}  {'Delay':>8}  {'Cumul':>10}")
    print("-" * 40)
    for i, ks in enumerate(keystrokes):
        key_display = repr(ks.key) if ks.key != "BKSP" else "BKSP"
        print(f"{i+1:4d}  {key_display:>6s}  {ks.delay_ms:7.1f}ms  {ks.cumulative_ms:9.1f}ms")
