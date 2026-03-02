"""V3 Typing Transformer model with action-gated decoder.

Encoder reads target text + style vector via FiLM conditioning.
Decoder autoregressively generates keystrokes with action gate routing
to per-action MDN timing heads and character heads.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from tensorflow import keras

from lilly.core.config import V3ModelConfig
from lilly.models.components import (
    ActionGate,
    ErrorCharHead,
    FiLMModulation,
    MDNHead,
)


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """Generate sinusoidal positional encoding matrix.

    Returns (max_len, d_model) float32 array.
    """
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(max_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


class EncoderLayer(keras.layers.Layer):
    """Single encoder layer with self-attention, FFN, and FiLM conditioning."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, style_dim: int, **kwargs):
        super().__init__(**kwargs)
        self._config = {
            "d_model": d_model, "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout, "style_dim": style_dim,
        }
        self.self_attn = keras.layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead,
            dropout=dropout, name="self_attn",
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(dim_feedforward, activation="relu"),
            keras.layers.Dense(d_model),
        ], name="ffn")
        self.norm1 = keras.layers.LayerNormalization(name="norm1")
        self.norm2 = keras.layers.LayerNormalization(name="norm2")
        self.film1 = FiLMModulation(d_model, style_dim, name="film1")
        self.film2 = FiLMModulation(d_model, style_dim, name="film2")
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    def call(self, x, style_vector, attention_mask=None, training=False):
        # Self-attention with residual
        normed = self.norm1(x)
        normed = self.film1(normed, style_vector)
        attn_out = self.self_attn(
            normed, normed, attention_mask=attention_mask, training=training,
        )
        x = x + self.dropout1(attn_out, training=training)

        # FFN with residual
        normed = self.norm2(x)
        normed = self.film2(normed, style_vector)
        ffn_out = self.ffn(normed)
        x = x + self.dropout2(ffn_out, training=training)

        return x


class DecoderLayer(keras.layers.Layer):
    """Single decoder layer with causal self-attention, cross-attention, FFN, and FiLM."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, style_dim: int, **kwargs):
        super().__init__(**kwargs)
        self._config = {
            "d_model": d_model, "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout, "style_dim": style_dim,
        }
        self.self_attn = keras.layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead,
            dropout=dropout, name="causal_self_attn",
        )
        self.cross_attn = keras.layers.MultiHeadAttention(
            num_heads=nhead, key_dim=d_model // nhead,
            dropout=dropout, name="cross_attn",
        )
        self.ffn = keras.Sequential([
            keras.layers.Dense(dim_feedforward, activation="relu"),
            keras.layers.Dense(d_model),
        ], name="ffn")
        self.norm1 = keras.layers.LayerNormalization(name="norm1")
        self.norm2 = keras.layers.LayerNormalization(name="norm2")
        self.norm3 = keras.layers.LayerNormalization(name="norm3")
        self.film1 = FiLMModulation(d_model, style_dim, name="film1")
        self.film2 = FiLMModulation(d_model, style_dim, name="film2")
        self.film3 = FiLMModulation(d_model, style_dim, name="film3")
        self.dropout1 = keras.layers.Dropout(dropout)
        self.dropout2 = keras.layers.Dropout(dropout)
        self.dropout3 = keras.layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update(self._config)
        return config

    def call(self, x, encoder_output, style_vector,
             causal_mask=None, cross_attn_mask=None, training=False):
        # Causal self-attention
        normed = self.norm1(x)
        normed = self.film1(normed, style_vector)
        attn_out = self.self_attn(
            normed, normed, attention_mask=causal_mask, training=training,
        )
        x = x + self.dropout1(attn_out, training=training)

        # Cross-attention to encoder
        normed = self.norm2(x)
        normed = self.film2(normed, style_vector)
        cross_out = self.cross_attn(
            normed, encoder_output, attention_mask=cross_attn_mask,
            training=training,
        )
        x = x + self.dropout2(cross_out, training=training)

        # FFN
        normed = self.norm3(x)
        normed = self.film3(normed, style_vector)
        ffn_out = self.ffn(normed)
        x = x + self.dropout3(ffn_out, training=training)

        return x


class TypingTransformerV3(keras.Model):
    """V3 Typing Transformer with action-gated decoder."""

    def __init__(self, cfg: V3ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

        # --- Embeddings ---
        self.encoder_char_embed = keras.layers.Embedding(
            cfg.num_char_classes, cfg.char_embed_dim, name="enc_char_embed",
        )
        self.encoder_proj = keras.layers.Dense(cfg.d_model, name="enc_proj")

        self.decoder_char_embed = keras.layers.Embedding(
            cfg.num_char_classes, cfg.char_embed_dim, name="dec_char_embed",
        )
        self.decoder_action_embed = keras.layers.Embedding(
            cfg.num_actions + 1, cfg.action_embed_dim, name="dec_action_embed",
        )  # +1 for padding action
        self.decoder_delay_proj = keras.layers.Dense(
            cfg.delay_embed_dim, name="dec_delay_proj",
        )
        self.decoder_input_proj = keras.layers.Dense(
            cfg.d_model, name="dec_input_proj",
        )

        # --- Context embeddings ---
        self.context_char_embed = keras.layers.Embedding(
            cfg.num_char_classes, cfg.char_embed_dim, name="ctx_char_embed",
        )
        self.context_action_embed = keras.layers.Embedding(
            cfg.num_actions + 1, cfg.action_embed_dim, name="ctx_action_embed",
        )
        self.context_delay_proj = keras.layers.Dense(
            cfg.delay_embed_dim, name="ctx_delay_proj",
        )
        self.context_proj = keras.layers.Dense(
            cfg.d_model, name="ctx_proj",
        )

        # --- Positional encodings (precomputed constants) ---
        max_pos = max(cfg.max_encoder_len, cfg.max_decoder_len + 1 + cfg.context_tail_len) + 1
        pe = sinusoidal_positional_encoding(max_pos, cfg.d_model)
        self.pe = tf.constant(pe, dtype=tf.float32)

        # --- Encoder layers ---
        self.encoder_layers = [
            EncoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward,
                        cfg.dropout, cfg.style_dim, name=f"enc_layer_{i}")
            for i in range(cfg.num_encoder_layers)
        ]
        self.encoder_norm = keras.layers.LayerNormalization(name="enc_final_norm")

        # --- Decoder layers ---
        self.decoder_layers = [
            DecoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward,
                        cfg.dropout, cfg.style_dim, name=f"dec_layer_{i}")
            for i in range(cfg.num_decoder_layers)
        ]
        self.decoder_norm = keras.layers.LayerNormalization(name="dec_final_norm")

        # --- Output heads ---
        self.action_gate = ActionGate(
            hidden_dim=64, num_actions=cfg.num_actions, name="action_gate",
        )
        self.timing_correct = MDNHead(
            n_components=cfg.mdn_components, hidden_dim=64, name="timing_correct",
        )
        self.timing_error = MDNHead(
            n_components=cfg.mdn_components, hidden_dim=64, name="timing_error",
        )
        self.timing_backspace = MDNHead(
            n_components=cfg.mdn_components, hidden_dim=64, name="timing_backspace",
        )
        self.error_char_head = ErrorCharHead(
            num_chars=97,
            learnable_alpha=cfg.qwerty_bias_learnable,
            name="error_char_head",
        )
        self.position_head = keras.layers.Dense(1, name="position_pred")

        # --- Dropout ---
        self.embed_dropout = keras.layers.Dropout(cfg.dropout)

    def _make_causal_mask(self, seq_len):
        """Create causal attention mask for decoder self-attention."""
        # Keras MHA expects: True = attend, False = mask
        mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=tf.bool), -1, 0,
        )
        return mask

    def _encode(self, encoder_chars, encoder_lengths, style_vector, training=False):
        """Encode target text with FiLM-conditioned transformer layers."""
        # Embed + project
        x = self.encoder_char_embed(encoder_chars)  # (B, S, char_embed_dim)
        x = self.encoder_proj(x)  # (B, S, d_model)

        # Add positional encoding
        seq_len = tf.shape(encoder_chars)[1]
        x = x + self.pe[:seq_len]
        x = self.embed_dropout(x, training=training)

        # Padding mask: (B, S) -> bool, True = valid
        indices = tf.range(seq_len)[tf.newaxis, :]  # (1, S)
        enc_mask = indices < encoder_lengths  # (B, S)
        # For MHA: (B, 1, 1, S) bool mask
        attn_mask = enc_mask[:, tf.newaxis, tf.newaxis, :]

        for layer in self.encoder_layers:
            x = layer(x, style_vector, attention_mask=attn_mask, training=training)

        x = self.encoder_norm(x)
        return x, enc_mask

    def _decode(self, decoder_input_chars, decoder_input_delays,
                decoder_input_actions, prev_context_chars,
                prev_context_actions, prev_context_delays,
                encoder_output, enc_mask, style_vector, training=False):
        """Decode with causal self-attention and cross-attention to encoder."""
        # --- Build context embeddings ---
        ctx_char = self.context_char_embed(prev_context_chars)
        ctx_action = self.context_action_embed(prev_context_actions)
        ctx_delay = self.context_delay_proj(prev_context_delays[:, :, tf.newaxis])
        ctx = tf.concat([ctx_char, ctx_action, ctx_delay], axis=-1)
        ctx = self.context_proj(ctx)  # (B, ctx_len, d_model)

        # --- Build decoder input embeddings ---
        dec_char = self.decoder_char_embed(decoder_input_chars)
        dec_action = self.decoder_action_embed(decoder_input_actions)
        dec_delay = self.decoder_delay_proj(decoder_input_delays[:, :, tf.newaxis])
        dec = tf.concat([dec_char, dec_action, dec_delay], axis=-1)
        dec = self.decoder_input_proj(dec)  # (B, T, d_model)

        # --- Concatenate context + decoder ---
        x = tf.concat([ctx, dec], axis=1)  # (B, ctx_len + T, d_model)

        # Add positional encoding
        total_len = tf.shape(x)[1]
        x = x + self.pe[:total_len]
        x = self.embed_dropout(x, training=training)

        # Causal mask for full (context + decoder) sequence
        causal_mask = self._make_causal_mask(total_len)

        # Cross-attention mask: (B, 1, 1, S) from encoder padding
        cross_mask = enc_mask[:, tf.newaxis, tf.newaxis, :]

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, style_vector,
                     causal_mask=causal_mask, cross_attn_mask=cross_mask,
                     training=training)

        x = self.decoder_norm(x)

        # Strip context prefix, keep only decoder positions
        ctx_len = tf.shape(prev_context_chars)[1]
        decoder_hidden = x[:, ctx_len:, :]  # (B, T, d_model)

        return decoder_hidden

    def call(self, inputs, training=False):
        """Forward pass returning all output heads.

        Args:
            inputs: dict with keys matching _make_dummy_inputs
            training: bool

        Returns:
            dict with keys: action_probs, timing_correct, timing_error,
                 timing_backspace, error_char_logits, position_pred
        """
        # Encode
        encoder_output, enc_mask = self._encode(
            inputs["encoder_chars"], inputs["encoder_lengths"],
            inputs["style_vector"], training=training,
        )

        # Decode
        decoder_hidden = self._decode(
            inputs["decoder_input_chars"],
            inputs["decoder_input_delays"],
            inputs["decoder_input_actions"],
            inputs["prev_context_chars"],
            inputs["prev_context_actions"],
            inputs["prev_context_delays"],
            encoder_output, enc_mask,
            inputs["style_vector"],
            training=training,
        )

        # Output heads
        action_probs = self.action_gate(decoder_hidden)
        timing_correct = self.timing_correct(decoder_hidden)
        timing_error = self.timing_error(decoder_hidden)
        timing_backspace = self.timing_backspace(decoder_hidden)

        # Error char head needs target char IDs matching decoder length.
        # Pad or tile encoder_chars to match decoder timesteps.
        dec_len = tf.shape(decoder_hidden)[1]
        enc_len = tf.shape(inputs["encoder_chars"])[1]
        # Pad encoder chars with zeros if decoder is longer than encoder
        padded_enc_chars = tf.pad(
            inputs["encoder_chars"],
            [[0, 0], [0, tf.maximum(dec_len - enc_len, 0)]],
        )
        target_ids_for_error = padded_enc_chars[:, :dec_len]
        error_char_logits = self.error_char_head(
            decoder_hidden, target_ids_for_error
        )

        position_pred = self.position_head(decoder_hidden)

        return {
            "action_probs": action_probs,
            "timing_correct": timing_correct,
            "timing_error": timing_error,
            "timing_backspace": timing_backspace,
            "error_char_logits": error_char_logits,
            "position_pred": position_pred,
        }


def build_model(cfg: V3ModelConfig | None = None) -> TypingTransformerV3:
    """Instantiate and build the V3 model with a dummy forward pass."""
    if cfg is None:
        cfg = V3ModelConfig()

    model = TypingTransformerV3(cfg, name="typing_transformer_v3")

    # Dummy forward pass to build all layers
    dec_len = cfg.max_decoder_len + 1
    dummy_inputs = {
        "encoder_chars": tf.zeros((1, cfg.max_encoder_len), dtype=tf.int32),
        "encoder_lengths": tf.constant([[1]], dtype=tf.int32),
        "decoder_input_chars": tf.zeros((1, dec_len), dtype=tf.int32),
        "decoder_input_delays": tf.zeros((1, dec_len), dtype=tf.float32),
        "decoder_input_actions": tf.zeros((1, dec_len), dtype=tf.int32),
        "style_vector": tf.zeros((1, cfg.style_dim), dtype=tf.float32),
        "prev_context_chars": tf.zeros((1, cfg.context_tail_len), dtype=tf.int32),
        "prev_context_actions": tf.zeros((1, cfg.context_tail_len), dtype=tf.int32),
        "prev_context_delays": tf.zeros((1, cfg.context_tail_len), dtype=tf.float32),
    }
    model(dummy_inputs, training=False)

    return model
