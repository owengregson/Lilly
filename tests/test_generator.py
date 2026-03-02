"""Unit tests for V3 generation logic."""
import pytest


class TestGenerateV3Segment:
    def test_produces_keystrokes(self):
        """Generate should return a non-empty list of keystrokes."""
        from lilly.models.typing_model import build_model
        from lilly.inference.generator import generate_v3_segment
        from lilly.core.config import V3ModelConfig
        import numpy as np

        cfg = V3ModelConfig()
        model = build_model(cfg)
        style = np.zeros(cfg.style_dim, dtype=np.float32)
        keystrokes = generate_v3_segment(
            model, "hello", style_vector=style, prev_context=None,
            cfg=cfg, temperatures={"action": 0.8, "timing": 0.8, "char": 0.8},
        )
        assert len(keystrokes) > 0

    def test_keystroke_has_required_fields(self):
        from lilly.models.typing_model import build_model
        from lilly.inference.generator import generate_v3_segment
        from lilly.core.config import V3ModelConfig
        import numpy as np

        cfg = V3ModelConfig()
        model = build_model(cfg)
        style = np.zeros(cfg.style_dim, dtype=np.float32)
        keystrokes = generate_v3_segment(
            model, "hi", style_vector=style, prev_context=None,
            cfg=cfg, temperatures={"action": 0.5, "timing": 0.8, "char": 0.5},
        )
        ks = keystrokes[0]
        assert hasattr(ks, "key")
        assert hasattr(ks, "delay_ms")
        assert hasattr(ks, "action")
        assert hasattr(ks, "cumulative_ms")
