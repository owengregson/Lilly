"""Sliding context window for V1 autoregressive generation."""

from __future__ import annotations

import math
from collections import deque
from typing import List

import numpy as np

from lilly.core.config import (
    ACTION_CORRECT,
    ACTION_ERROR,
    MAX_IKI_MS,
    MIN_IKI_MS,
    NUM_DENSE_FEATURES,
    PAD_TOKEN,
    SEQ_LEN,
)
from lilly.core.encoding import char_to_id
from lilly.core.keyboard import key_distance


class ContextWindow:
    """Maintains the sliding window of recent keystrokes for V1 model input."""

    def __init__(self, seq_len: int = SEQ_LEN, wpm_bucket: int = 5):
        self.seq_len = seq_len
        self.wpm_bucket = wpm_bucket

        self.typed_ids = deque([PAD_TOKEN] * seq_len, maxlen=seq_len)
        self.target_ids = deque([PAD_TOKEN] * seq_len, maxlen=seq_len)
        self.action_ids = deque([ACTION_CORRECT] * seq_len, maxlen=seq_len)
        self.dense = deque(
            [np.zeros(NUM_DENSE_FEATURES, dtype=np.float32)] * seq_len,
            maxlen=seq_len,
        )

        self.prev_key = ""
        self.prev_iki = 0.0
        self.chars_since_error = 0
        self.consecutive_correct = 0
        self.iki_history: List[float] = []
        self.error_history: List[int] = []
        self.total_chars = 0

    def push(
        self,
        typed_key: str,
        target_char: str,
        action: int,
        iki_ms: float,
        hold_ms: float,
        target_pos: int,
        sentence_len: int,
        sentence: str,
    ) -> None:
        """Add a keystroke to the context window."""
        self.typed_ids.append(char_to_id(typed_key))
        self.target_ids.append(char_to_id(target_char) if target_char else PAD_TOKEN)
        self.action_ids.append(action)

        feat = np.zeros(NUM_DENSE_FEATURES, dtype=np.float32)

        # 0: iki_log
        iki_clipped = max(MIN_IKI_MS, min(iki_ms, MAX_IKI_MS))
        feat[0] = math.log(iki_clipped) / math.log(MAX_IKI_MS)

        # 1: hold_time_log
        hold_clipped = max(5.0, min(hold_ms, 2000.0))
        feat[1] = math.log(hold_clipped) / math.log(2000.0)

        # 2: key_distance
        if self.prev_key and typed_key != "BKSP" and self.prev_key != "BKSP":
            feat[2] = min(key_distance(self.prev_key, typed_key) / 10.0, 1.0)

        # 3: pos_in_sentence
        feat[3] = target_pos / max(sentence_len, 1)

        # 4-7: word position features
        if target_pos < len(sentence):
            ws = target_pos
            while ws > 0 and sentence[ws - 1] not in " \t\n":
                ws -= 1
            we = target_pos
            while we < len(sentence) and sentence[we] not in " \t\n":
                we += 1
            wl = max(we - ws, 1)
            feat[4] = (target_pos - ws) / wl
            feat[5] = 1.0 if target_pos == ws else 0.0
            feat[6] = 1.0 if target_pos == we - 1 else 0.0
            feat[7] = 1.0 if sentence[target_pos] in ".!?;:" else 0.0

        # 8-9: error tracking
        if action == ACTION_ERROR:
            self.chars_since_error = 0
            self.consecutive_correct = 0
        elif action == ACTION_CORRECT:
            self.chars_since_error += 1
            self.consecutive_correct += 1
        else:
            self.consecutive_correct = 0

        feat[8] = min(self.chars_since_error, 50) / 50.0
        feat[9] = min(self.consecutive_correct, 50) / 50.0

        # 10-12: rolling stats
        self.iki_history.append(iki_clipped)
        self.error_history.append(1 if action == ACTION_ERROR else 0)
        if len(self.iki_history) > 10:
            self.iki_history.pop(0)
        if len(self.error_history) > 20:
            self.error_history.pop(0)

        feat[10] = np.mean(self.iki_history) / MAX_IKI_MS
        feat[11] = (np.std(self.iki_history) / MAX_IKI_MS) if len(self.iki_history) > 1 else 0.0
        feat[12] = np.mean(self.error_history)

        # 13: elapsed fraction
        self.total_chars += 1
        feat[13] = min(self.total_chars / max(sentence_len, 1), 1.0)

        self.dense.append(feat)
        self.prev_key = typed_key
        self.prev_iki = iki_ms

    def to_model_input(self) -> dict:
        """Convert current window to model input tensors."""
        return {
            "typed_chars": np.array([list(self.typed_ids)], dtype=np.int32),
            "target_chars": np.array([list(self.target_ids)], dtype=np.int32),
            "actions": np.array([list(self.action_ids)], dtype=np.int32),
            "dense_features": np.array([list(self.dense)], dtype=np.float32),
            "wpm_bucket": np.array([[self.wpm_bucket]], dtype=np.int32),
        }
