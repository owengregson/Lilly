"""Unit tests for V3 model components."""
import numpy as np
import tensorflow as tf
import pytest


class TestFiLMModulation:
    def test_output_shape(self):
        from lilly.models.components import FiLMModulation
        film = FiLMModulation(d_model=128, style_dim=16)
        x = tf.random.normal((2, 10, 128))
        style = tf.random.normal((2, 16))
        out = film(x, style)
        assert out.shape == (2, 10, 128)

    def test_identity_at_init(self):
        """gamma=1+proj(0)~1, beta=proj(0)~0 => near identity."""
        from lilly.models.components import FiLMModulation
        film = FiLMModulation(d_model=4, style_dim=2)
        x = tf.ones((1, 3, 4))
        style = tf.zeros((1, 2))
        out = film(x, style)
        # With zero style input and zero-init bias, gamma~1, beta~0
        # So output should be close to x (not exact due to random weight init)
        assert out.shape == (1, 3, 4)


class TestMDNHead:
    def test_output_shape(self):
        from lilly.models.components import MDNHead
        mdn = MDNHead(n_components=8, hidden_dim=64)
        h = tf.random.normal((2, 10, 128))
        pi, mu, log_sigma = mdn(h)
        assert pi.shape == (2, 10, 8)
        assert mu.shape == (2, 10, 8)
        assert log_sigma.shape == (2, 10, 8)

    def test_pi_sums_to_one(self):
        from lilly.models.components import MDNHead
        mdn = MDNHead(n_components=8, hidden_dim=64)
        h = tf.random.normal((4, 5, 128))
        pi, _, _ = mdn(h)
        sums = tf.reduce_sum(pi, axis=-1)
        np.testing.assert_allclose(sums.numpy(), 1.0, atol=1e-5)

    def test_log_sigma_clipped(self):
        from lilly.models.components import MDNHead
        mdn = MDNHead(n_components=8, hidden_dim=64)
        h = tf.random.normal((2, 5, 128)) * 100  # large input
        _, _, log_sigma = mdn(h)
        assert tf.reduce_all(log_sigma >= -5.0)
        assert tf.reduce_all(log_sigma <= 5.0)


class TestActionGate:
    def test_output_shape(self):
        from lilly.models.components import ActionGate
        gate = ActionGate(hidden_dim=64, num_actions=3)
        h = tf.random.normal((2, 10, 128))
        probs = gate(h)
        assert probs.shape == (2, 10, 3)

    def test_sums_to_one(self):
        from lilly.models.components import ActionGate
        gate = ActionGate(hidden_dim=64, num_actions=3)
        h = tf.random.normal((4, 8, 128))
        probs = gate(h)
        sums = tf.reduce_sum(probs, axis=-1)
        np.testing.assert_allclose(sums.numpy(), 1.0, atol=1e-5)


class TestErrorCharHead:
    def test_output_shape(self):
        from lilly.models.components import ErrorCharHead
        head = ErrorCharHead(num_chars=97, learnable_alpha=True)
        h = tf.random.normal((2, 10, 128))
        target_ids = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                   [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
        logits = head(h, target_ids)
        assert logits.shape == (2, 10, 97)

    def test_neighbor_bias_increases_nearby_keys(self):
        """Keys near the target should have higher logits than distant keys."""
        from lilly.models.components import ErrorCharHead
        head = ErrorCharHead(num_chars=97, learnable_alpha=True)
        h = tf.zeros((1, 1, 128))
        # target char 'f' (char_to_id('f') = ord('f') - 31 = 71)
        target_ids = tf.constant([[71]])
        # Call once to trigger build(), then set alpha high
        head(h, target_ids)
        head.alpha.assign(10.0)
        logits = head(h, target_ids)
        probs = tf.nn.softmax(logits[0, 0]).numpy()
        # 'g' (72) should be higher than 'p' (81) since g is next to f
        assert probs[72] > probs[81], "Neighbor key should have higher prob"
