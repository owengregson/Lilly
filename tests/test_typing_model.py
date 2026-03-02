"""Unit tests for the V3 TypingTransformerV3 model."""
import tensorflow as tf

from lilly.core.config import V3ModelConfig


def _make_dummy_inputs(cfg: V3ModelConfig, batch: int = 2):
    """Create dummy inputs matching the model's expected signature."""
    dec_len = cfg.max_decoder_len + 1
    return {
        "encoder_chars": tf.zeros((batch, cfg.max_encoder_len), dtype=tf.int32),
        "encoder_lengths": tf.constant([[5]] * batch, dtype=tf.int32),
        "decoder_input_chars": tf.zeros((batch, dec_len), dtype=tf.int32),
        "decoder_input_delays": tf.zeros((batch, dec_len), dtype=tf.float32),
        "decoder_input_actions": tf.zeros((batch, dec_len), dtype=tf.int32),
        "style_vector": tf.random.normal((batch, cfg.style_dim)),
        "prev_context_chars": tf.zeros((batch, cfg.context_tail_len), dtype=tf.int32),
        "prev_context_actions": tf.zeros((batch, cfg.context_tail_len), dtype=tf.int32),
        "prev_context_delays": tf.zeros((batch, cfg.context_tail_len), dtype=tf.float32),
    }


class TestModelForwardPass:
    def test_output_keys(self):
        from lilly.models.typing_model import build_model
        cfg = V3ModelConfig()
        model = build_model(cfg)
        inputs = _make_dummy_inputs(cfg)
        outputs = model(inputs, training=False)

        expected_keys = {
            "action_probs", "timing_correct", "timing_error",
            "timing_backspace", "error_char_logits", "position_pred",
        }
        assert set(outputs.keys()) == expected_keys

    def test_action_probs_shape(self):
        from lilly.models.typing_model import build_model
        cfg = V3ModelConfig()
        model = build_model(cfg)
        inputs = _make_dummy_inputs(cfg)
        outputs = model(inputs, training=False)
        dec_len = cfg.max_decoder_len + 1
        assert outputs["action_probs"].shape == (2, dec_len, 3)

    def test_timing_mdn_shapes(self):
        from lilly.models.typing_model import build_model
        cfg = V3ModelConfig()
        model = build_model(cfg)
        inputs = _make_dummy_inputs(cfg)
        outputs = model(inputs, training=False)
        dec_len = cfg.max_decoder_len + 1
        for key in ["timing_correct", "timing_error", "timing_backspace"]:
            pi, mu, log_sigma = outputs[key]
            assert pi.shape == (2, dec_len, cfg.mdn_components)
            assert mu.shape == (2, dec_len, cfg.mdn_components)

    def test_error_char_logits_shape(self):
        from lilly.models.typing_model import build_model
        cfg = V3ModelConfig()
        model = build_model(cfg)
        inputs = _make_dummy_inputs(cfg)
        outputs = model(inputs, training=False)
        dec_len = cfg.max_decoder_len + 1
        assert outputs["error_char_logits"].shape == (2, dec_len, 97)

    def test_param_count_reasonable(self):
        from lilly.models.typing_model import build_model
        cfg = V3ModelConfig()
        model = build_model(cfg)
        count = model.count_params()
        assert 500_000 < count < 10_000_000, f"Param count {count} outside expected range"

    def test_training_mode_runs(self):
        from lilly.models.typing_model import build_model
        cfg = V3ModelConfig()
        model = build_model(cfg)
        inputs = _make_dummy_inputs(cfg)
        outputs = model(inputs, training=True)
        assert outputs["action_probs"].shape[0] == 2
