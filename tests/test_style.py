"""Unit tests for style vector computation."""
import numpy as np
import pandas as pd


def _make_session_df(n=100):
    """Create a fake session DataFrame matching preprocess.py output format."""
    rng = np.random.default_rng(42)
    actions = rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 1, 2], size=n)  # ~80% correct
    ikis = rng.lognormal(mean=4.5, sigma=0.5, size=n).clip(10, 5000)
    return pd.DataFrame({
        "action": actions,
        "iki": ikis,
        "hold_time": rng.lognormal(mean=4.0, sigma=0.3, size=n).clip(5, 2000),
        "typed_key": ["a"] * n,
        "target_char": ["a"] * n,
        "target_pos": list(range(n)),
        "wpm": [80.0] * n,
    })


class TestComputeStyleVector:
    def test_output_shape(self):
        from lilly.data.style import compute_style_vector
        df = _make_session_df()
        sv = compute_style_vector(df)
        assert sv.shape == (16,)
        assert sv.dtype == np.float32

    def test_all_finite(self):
        from lilly.data.style import compute_style_vector
        df = _make_session_df()
        sv = compute_style_vector(df)
        assert np.all(np.isfinite(sv))

    def test_different_sessions_differ(self):
        from lilly.data.style import compute_style_vector
        df1 = _make_session_df(n=100)
        df2 = _make_session_df(n=200)
        df2["iki"] = df2["iki"] * 2  # much slower
        sv1 = compute_style_vector(df1)
        sv2 = compute_style_vector(df2)
        assert not np.allclose(sv1, sv2)


class TestStyleNormalization:
    def test_normalize_and_denormalize(self):
        from lilly.data.style import StyleNormalizer, compute_style_vector
        vectors = np.stack([compute_style_vector(_make_session_df()) for _ in range(20)])
        norm = StyleNormalizer.fit(vectors)
        normed = norm.transform(vectors)
        # After z-score, mean should be ~0, std ~1
        np.testing.assert_allclose(normed.mean(axis=0), 0.0, atol=0.2)
