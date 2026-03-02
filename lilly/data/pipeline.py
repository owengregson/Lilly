"""TensorFlow data pipeline builder for V3 model training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from lilly.core.config import (
    END_TOKEN,
    PAD_TOKEN,
    START_TOKEN,
    V3ModelConfig,
    V3TrainConfig,
)


# ---------------------------------------------------------------------------
# V3 Pipeline
# ---------------------------------------------------------------------------

def load_v3_segment_files(data_dir: Path, max_files: int = 0) -> dict:
    """Load and concatenate all V3 segment .npz files."""
    files = sorted(data_dir.glob("segments_*.npz"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No segment files in {data_dir}")

    arrays = {}
    for f in files:
        data = np.load(f)
        for key in data.files:
            arrays.setdefault(key, []).append(data[key])

    return {k: np.concatenate(v) for k, v in arrays.items()}


def _prepare_v3_decoder_io(
    decoder_chars: np.ndarray,
    decoder_delays: np.ndarray,
    decoder_actions: np.ndarray,
    decoder_lengths: np.ndarray,
    max_len: int,
) -> dict:
    """Prepare teacher-forced decoder inputs and labels for V3.

    Decoder input:  [START, c1, c2, ..., cn]  (shifted right)
    Decoder label:  [c1, c2, ..., cn, END]    (shifted left)

    Includes action IDs in decoder inputs for teacher forcing.
    """
    n = len(decoder_chars)
    input_len = max_len + 1  # +1 for START token

    # Decoder inputs (teacher-forced, shifted right)
    dec_input_chars = np.full((n, input_len), PAD_TOKEN, dtype=np.int32)
    dec_input_delays = np.zeros((n, input_len), dtype=np.float32)
    dec_input_actions = np.zeros((n, input_len), dtype=np.int32)

    # Labels (shifted left)
    dec_label_chars = np.full((n, input_len), PAD_TOKEN, dtype=np.int32)
    dec_label_delays = np.zeros((n, input_len), dtype=np.float32)
    dec_label_actions = np.zeros((n, input_len), dtype=np.int32)
    dec_label_mask = np.zeros((n, input_len), dtype=np.float32)

    # Error char labels (the char that was typed for ERROR actions)
    dec_error_char_labels = np.zeros((n, input_len), dtype=np.int32)

    # Position labels (normalized position in target sentence)
    dec_position_labels = np.zeros((n, input_len), dtype=np.float32)

    for i in range(n):
        length = int(decoder_lengths[i])

        # Input: START token + actual keystrokes
        dec_input_chars[i, 0] = START_TOKEN
        dec_input_chars[i, 1:length + 1] = decoder_chars[i, :length]
        dec_input_delays[i, 0] = 0.0
        dec_input_delays[i, 1:length + 1] = decoder_delays[i, :length]
        dec_input_actions[i, 0] = 0  # START gets "correct" action
        dec_input_actions[i, 1:length + 1] = decoder_actions[i, :length]

        # Labels: actual keystrokes + END token
        dec_label_chars[i, :length] = decoder_chars[i, :length]
        dec_label_chars[i, length] = END_TOKEN
        dec_label_delays[i, :length] = decoder_delays[i, :length]
        dec_label_actions[i, :length] = decoder_actions[i, :length]
        dec_label_mask[i, :length + 1] = 1.0

        # Error char labels = the actual typed char for error positions
        for j in range(length):
            if decoder_actions[i, j] == 1:  # ERROR
                dec_error_char_labels[i, j] = decoder_chars[i, j]

        # Position labels: simple linear ramp for now
        # (real position tracking would need target_pos from preprocessing)
        if length > 0:
            positions = np.linspace(0.0, 1.0, length)
            dec_position_labels[i, :length] = positions

    return {
        "dec_input_chars": dec_input_chars,
        "dec_input_delays": dec_input_delays,
        "dec_input_actions": dec_input_actions,
        "dec_label_actions": dec_label_actions,
        "dec_label_delays": dec_label_delays,
        "dec_label_mask": dec_label_mask,
        "dec_error_char_labels": dec_error_char_labels,
        "dec_position_labels": dec_position_labels,
    }


def build_v3_datasets(
    data_dir: Path,
    model_cfg: V3ModelConfig,
    train_cfg: V3TrainConfig,
    max_files: int = 0,
) -> tuple:
    """Build V3 train, validation, test tf.data.Datasets.

    Returns (train_ds, val_ds, test_ds, n_total).
    """
    arrays = load_v3_segment_files(data_dir, max_files)

    n_total = len(arrays["encoder_chars"])

    if train_cfg.max_samples > 0:
        n_total = min(n_total, train_cfg.max_samples)
        arrays = {k: v[:n_total] for k, v in arrays.items()}

    # Shuffle
    rng = np.random.default_rng(train_cfg.seed)
    perm = rng.permutation(n_total)
    arrays = {k: v[perm] for k, v in arrays.items()}

    # Prepare decoder I/O
    decoder_io = _prepare_v3_decoder_io(
        arrays["decoder_chars"],
        arrays["decoder_delays"],
        arrays["decoder_actions"],
        arrays["decoder_lengths"],
        model_cfg.max_decoder_len,
    )

    # Log-normalize delays
    dec_in_delays_log = np.log(np.clip(decoder_io["dec_input_delays"], 1.0, 5000.0))
    dec_in_delays_log[decoder_io["dec_input_delays"] == 0] = 0.0
    dec_lbl_delays_log = np.log(np.clip(decoder_io["dec_label_delays"], 1.0, 5000.0))
    dec_lbl_delays_log[decoder_io["dec_label_delays"] == 0] = 0.0

    # Split
    n_test = int(n_total * train_cfg.test_split)
    n_val = int(n_total * train_cfg.val_split)
    n_train = n_total - n_val - n_test

    def _make_ds(start, end, shuffle):
        s = slice(start, end)
        inputs = {
            "encoder_chars": arrays["encoder_chars"][s],
            "encoder_lengths": arrays["encoder_lengths"][s],
            "decoder_input_chars": decoder_io["dec_input_chars"][s],
            "decoder_input_delays": dec_in_delays_log[s],
            "decoder_input_actions": decoder_io["dec_input_actions"][s],
            "style_vector": arrays["style_vector"][s],
            "prev_context_chars": arrays["prev_context_chars"][s],
            "prev_context_actions": arrays["prev_context_actions"][s],
            "prev_context_delays": arrays["prev_context_delays"][s],
        }
        labels = {
            "action_labels": decoder_io["dec_label_actions"][s],
            "delay_labels": dec_lbl_delays_log[s],
            "error_char_labels": decoder_io["dec_error_char_labels"][s],
            "position_labels": decoder_io["dec_position_labels"][s],
            "label_mask": decoder_io["dec_label_mask"][s],
        }

        ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
        if shuffle:
            ds = ds.shuffle(
                buffer_size=min(train_cfg.shuffle_buffer, end - start),
                seed=train_cfg.seed,
            )
        ds = ds.batch(train_cfg.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    return (
        _make_ds(0, n_train, shuffle=True),
        _make_ds(n_train, n_train + n_val, shuffle=False),
        _make_ds(n_train + n_val, n_total, shuffle=False),
        n_total,
    )
