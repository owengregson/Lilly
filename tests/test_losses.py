"""Unit tests for V3 loss functions."""
import numpy as np
import tensorflow as tf


class TestFocalLoss:
    def test_output_shape(self):
        from lilly.training.losses import FocalLoss
        fl = FocalLoss(alpha=[0.25, 0.5, 0.5], gamma=2.0)
        y_true = tf.constant([[0, 1, 2, 0]], dtype=tf.int32)  # (1, 4)
        y_pred = tf.constant([[[0.9, 0.05, 0.05],
                                [0.1, 0.8, 0.1],
                                [0.1, 0.1, 0.8],
                                [0.8, 0.1, 0.1]]], dtype=tf.float32)
        loss = fl(y_true, y_pred)
        assert loss.shape == (1, 4)

    def test_confident_correct_has_low_loss(self):
        from lilly.training.losses import FocalLoss
        fl = FocalLoss(alpha=[1.0, 1.0, 1.0], gamma=2.0)
        y_true = tf.constant([[0]], dtype=tf.int32)
        confident = tf.constant([[[0.99, 0.005, 0.005]]], dtype=tf.float32)
        uncertain = tf.constant([[[0.4, 0.3, 0.3]]], dtype=tf.float32)
        loss_conf = fl(y_true, confident).numpy()[0, 0]
        loss_unc = fl(y_true, uncertain).numpy()[0, 0]
        assert loss_conf < loss_unc * 0.1, "Confident correct should have much lower loss"

    def test_gamma_zero_equals_weighted_ce(self):
        from lilly.training.losses import FocalLoss
        fl = FocalLoss(alpha=[1.0, 1.0, 1.0], gamma=0.0)
        y_true = tf.constant([[1]], dtype=tf.int32)
        y_pred = tf.constant([[[0.2, 0.6, 0.2]]], dtype=tf.float32)
        focal = fl(y_true, y_pred).numpy()[0, 0]
        ce = -np.log(0.6)
        np.testing.assert_allclose(focal, ce, atol=1e-5)


class TestMDNMixtureLoss:
    def test_output_shape(self):
        from lilly.training.losses import mdn_mixture_nll
        pi = tf.constant([[[0.5, 0.5]]], dtype=tf.float32)
        mu = tf.constant([[[4.0, 5.0]]], dtype=tf.float32)
        log_sigma = tf.constant([[[0.0, 0.0]]], dtype=tf.float32)
        y_true = tf.constant([[4.5]], dtype=tf.float32)
        nll = mdn_mixture_nll(y_true, pi, mu, log_sigma)
        assert nll.shape == (1, 1)

    def test_lower_nll_for_correct_component(self):
        from lilly.training.losses import mdn_mixture_nll
        # Mixture with one component at mu=4.0, one at mu=8.0
        pi = tf.constant([[[0.5, 0.5]]], dtype=tf.float32)
        mu = tf.constant([[[4.0, 8.0]]], dtype=tf.float32)
        log_sigma = tf.constant([[[-1.0, -1.0]]], dtype=tf.float32)
        # Target near first component
        nll_near = mdn_mixture_nll(tf.constant([[4.0]]), pi, mu, log_sigma).numpy()
        # Target far from both
        nll_far = mdn_mixture_nll(tf.constant([[6.0]]), pi, mu, log_sigma).numpy()
        assert nll_near < nll_far

    def test_positive_nll(self):
        """NLL should generally be positive for reasonable inputs."""
        from lilly.training.losses import mdn_mixture_nll
        pi = tf.constant([[[1.0]]], dtype=tf.float32)
        mu = tf.constant([[[4.0]]], dtype=tf.float32)
        log_sigma = tf.constant([[[0.5]]], dtype=tf.float32)
        nll = mdn_mixture_nll(tf.constant([[4.0]]), pi, mu, log_sigma)
        # With sigma=exp(0.5)~1.65, NLL at the mean should be moderate
        assert nll.numpy()[0, 0] > -5.0  # not extremely negative


class TestCombinedV3Loss:
    def test_runs_without_error(self):
        from lilly.training.losses import V3LossConfig, compute_v3_loss
        cfg = V3LossConfig()
        outputs = {
            "action_probs": tf.constant([[[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]]]),
            "timing_correct": (
                tf.constant([[[1.0]]]), tf.constant([[[4.0]]]), tf.constant([[[0.0]]])
            ),
            "timing_error": (
                tf.constant([[[1.0]]]), tf.constant([[[4.0]]]), tf.constant([[[0.0]]])
            ),
            "timing_backspace": (
                tf.constant([[[1.0]]]), tf.constant([[[4.0]]]), tf.constant([[[0.0]]])
            ),
            "error_char_logits": tf.random.normal((1, 2, 97)),
            "position_pred": tf.constant([[[0.3], [0.6]]]),
        }
        labels = {
            "action_labels": tf.constant([[0, 1]]),
            "delay_labels": tf.constant([[4.5, 4.0]]),
            "error_char_labels": tf.constant([[0, 50]]),
            "position_labels": tf.constant([[0.25, 0.5]]),
            "label_mask": tf.constant([[1.0, 1.0]]),
        }
        total, components = compute_v3_loss(outputs, labels, cfg)
        assert total.shape == ()
        assert "action" in components
        assert "timing" in components
