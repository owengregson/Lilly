"""V3 autoregressive keystroke generator.

Implements the action-gated generation loop: at each step, predict action,
route to correct/error/backspace path, sample timing from per-action MDN.
"""

from __future__ import annotations

from typing import List, NamedTuple

import numpy as np
import tensorflow as tf

from lilly.core.config import (
    BACKSPACE_TOKEN,
    START_TOKEN,
    V3ModelConfig,
)
from lilly.core.encoding import char_to_id, id_to_char
from lilly.data.segment import split_text_into_inference_segments
from lilly.inference.sampling import sample_mdn, weighted_sample, weighted_sample_logits


class GeneratedKeystroke(NamedTuple):
    """A single generated keystroke event."""
    key: str
    delay_ms: float
    action: int  # 0=correct, 1=error, 2=backspace
    target_char: str
    cumulative_ms: float
    mdn_component: int


def generate_v3_segment(
    model,
    target_text: str,
    style_vector: np.ndarray,
    prev_context: dict | None,
    cfg: V3ModelConfig,
    temperatures: dict[str, float] | None = None,
    max_steps: int | None = None,
    seed: int | None = None,
) -> List[GeneratedKeystroke]:
    """Generate keystrokes for a single text segment.

    Args:
        model: TypingTransformerV3 model
        target_text: Text to type
        style_vector: (style_dim,) float32 style vector
        prev_context: dict with prev_chars, prev_actions, prev_delays (or None)
        cfg: V3ModelConfig
        temperatures: dict with keys 'action', 'timing', 'char'
        max_steps: Maximum decoder steps (default: 3x target length)
        seed: Random seed

    Returns:
        List of GeneratedKeystroke
    """
    if seed is not None:
        np.random.seed(seed)

    temps = {"action": 0.8, "timing": 0.8, "char": 0.8}
    if temperatures:
        temps.update(temperatures)

    if max_steps is None:
        max_steps = max(len(target_text) * 3, 10)

    # Encode target text
    enc_chars = np.zeros((1, cfg.max_encoder_len), dtype=np.int32)
    for i, ch in enumerate(target_text[:cfg.max_encoder_len]):
        enc_chars[0, i] = char_to_id(ch)
    enc_len = np.array([[min(len(target_text), cfg.max_encoder_len)]], dtype=np.int32)

    # Style vector
    style = style_vector.reshape(1, cfg.style_dim).astype(np.float32)

    # Context
    ctx_chars = np.zeros((1, cfg.context_tail_len), dtype=np.int32)
    ctx_actions = np.zeros((1, cfg.context_tail_len), dtype=np.int32)
    ctx_delays = np.zeros((1, cfg.context_tail_len), dtype=np.float32)
    if prev_context is not None:
        tail_len = cfg.context_tail_len
        for key, arr in [("chars", ctx_chars), ("actions", ctx_actions),
                         ("delays", ctx_delays)]:
            vals = prev_context.get(key, [])[-tail_len:]
            offset = tail_len - len(vals)
            for i, v in enumerate(vals):
                arr[0, offset + i] = v

    # Initialize decoder sequence with START token
    dec_len = cfg.max_decoder_len + 1
    dec_chars = np.zeros((1, dec_len), dtype=np.int32)
    dec_delays = np.zeros((1, dec_len), dtype=np.float32)
    dec_actions = np.zeros((1, dec_len), dtype=np.int32)
    dec_chars[0, 0] = START_TOKEN

    keystrokes: List[GeneratedKeystroke] = []
    position = 0  # Current position in target text
    cumulative_ms = 0.0

    for step in range(min(max_steps, dec_len - 1)):
        # Build inputs
        inputs = {
            "encoder_chars": tf.constant(enc_chars),
            "encoder_lengths": tf.constant(enc_len),
            "decoder_input_chars": tf.constant(dec_chars),
            "decoder_input_delays": tf.constant(dec_delays),
            "decoder_input_actions": tf.constant(dec_actions),
            "style_vector": tf.constant(style),
            "prev_context_chars": tf.constant(ctx_chars),
            "prev_context_actions": tf.constant(ctx_actions),
            "prev_context_delays": tf.constant(ctx_delays),
        }

        # Forward pass
        outputs = model(inputs, training=False)

        # Get predictions at current step
        action_probs = outputs["action_probs"][0, step].numpy()

        # Sample action with temperature
        action = weighted_sample(action_probs, temps["action"])

        # Route to action-specific heads
        if action == 0:  # CORRECT
            if position >= len(target_text):
                break  # Done typing
            char = target_text[position]
            char_id = char_to_id(char)
            pi, mu, log_sigma = (
                outputs["timing_correct"][0][0, step].numpy(),
                outputs["timing_correct"][1][0, step].numpy(),
                outputs["timing_correct"][2][0, step].numpy(),
            )
            delay_ms, mdn_k = sample_mdn(pi, mu, log_sigma, temps["timing"])
            position += 1

        elif action == 1:  # ERROR
            # Sample error character from QWERTY-biased logits
            error_logits = outputs["error_char_logits"][0, step].numpy()
            char_id = weighted_sample_logits(error_logits, temps["char"])
            char = id_to_char(char_id)
            pi, mu, log_sigma = (
                outputs["timing_error"][0][0, step].numpy(),
                outputs["timing_error"][1][0, step].numpy(),
                outputs["timing_error"][2][0, step].numpy(),
            )
            delay_ms, mdn_k = sample_mdn(pi, mu, log_sigma, temps["timing"])
            # Position doesn't advance on error

        else:  # BACKSPACE
            char = "BKSP"
            char_id = BACKSPACE_TOKEN
            pi, mu, log_sigma = (
                outputs["timing_backspace"][0][0, step].numpy(),
                outputs["timing_backspace"][1][0, step].numpy(),
                outputs["timing_backspace"][2][0, step].numpy(),
            )
            delay_ms, mdn_k = sample_mdn(pi, mu, log_sigma, temps["timing"])
            # Adjust position backward if possible
            if position > 0:
                position -= 1

        cumulative_ms += delay_ms
        target_ch = target_text[min(position, len(target_text) - 1)] if target_text else ""

        keystrokes.append(GeneratedKeystroke(
            key=char,
            delay_ms=delay_ms,
            action=action,
            target_char=target_ch,
            cumulative_ms=cumulative_ms,
            mdn_component=mdn_k,
        ))

        # Update decoder sequence for next step
        next_pos = step + 1
        if next_pos < dec_len:
            dec_chars[0, next_pos] = char_id
            dec_delays[0, next_pos] = float(np.log(max(delay_ms, 1.0)))
            dec_actions[0, next_pos] = action

        # Check if we're done
        if position >= len(target_text) and action == 0:
            break

    return keystrokes


def generate_v3_full(
    model,
    text: str,
    style_vector: np.ndarray,
    cfg: V3ModelConfig,
    temperatures: dict[str, float] | None = None,
    seed: int | None = None,
) -> List[GeneratedKeystroke]:
    """Generate keystrokes for full text by splitting into segments.

    Segments text at word boundaries and propagates context between segments.
    """
    segments = split_text_into_inference_segments(text, seed=seed)

    all_keystrokes: List[GeneratedKeystroke] = []
    prev_context = None
    cumulative_offset = 0.0

    for seg_text in segments:
        ks_list = generate_v3_segment(
            model, seg_text, style_vector, prev_context,
            cfg, temperatures, seed=None,
        )

        # Adjust cumulative times
        for ks in ks_list:
            adjusted = GeneratedKeystroke(
                key=ks.key,
                delay_ms=ks.delay_ms,
                action=ks.action,
                target_char=ks.target_char,
                cumulative_ms=ks.cumulative_ms + cumulative_offset,
                mdn_component=ks.mdn_component,
            )
            all_keystrokes.append(adjusted)

        if ks_list:
            cumulative_offset = all_keystrokes[-1].cumulative_ms
            # Build context for next segment
            tail = ks_list[-cfg.context_tail_len:]
            prev_context = {
                "chars": [char_to_id(ks.key) for ks in tail],
                "actions": [ks.action for ks in tail],
                "delays": [float(np.log(max(ks.delay_ms, 1.0))) for ks in tail],
            }

    return all_keystrokes


def print_v3_sequence(keystrokes: List[GeneratedKeystroke], text: str = "") -> None:
    """Pretty-print a generated keystroke sequence."""
    action_names = {0: "CORRECT", 1: "ERROR", 2: "BKSP"}
    if text:
        print(f"Target: {text!r}")
    print(f"{'Step':>4}  {'Action':>8}  {'Key':>6}  {'Delay':>8}  {'Cumul':>9}  {'MDN':>3}")
    print("-" * 52)
    for i, ks in enumerate(keystrokes):
        key_disp = repr(ks.key) if ks.key != "BKSP" else "BKSP"
        print(
            f"{i:4d}  {action_names.get(ks.action, '?'):>8}  "
            f"{key_disp:>6}  {ks.delay_ms:7.1f}ms  "
            f"{ks.cumulative_ms:8.1f}ms  {ks.mdn_component:3d}"
        )
