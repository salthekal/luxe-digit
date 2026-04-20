"""Tests for the ML feature extractor.

These tests do not require ROOT, so they run on any machine with
torch and numpy installed. Marked tests that do need ROOT (for the
upstream pipeline integration) should be tagged with the
``requires_root`` marker.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from luxedigit.ml_extractor import (
    MLExtractor,
    MLExtractorConfig,
    ProfileNormalizer,
    StripProfileMLP,
    synthesize_training_set,
)


# ─────────────────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────────────────


def test_synthesize_training_set_shapes():
    profiles, labels = synthesize_training_set(n_samples=100, n_strips=64, seed=0)
    assert profiles.shape == (100, 64)
    assert labels.shape == (100, 4)
    assert profiles.dtype == np.float32
    assert labels.dtype == np.float32


def test_synthesize_reproducible():
    p1, l1 = synthesize_training_set(n_samples=50, seed=42)
    p2, l2 = synthesize_training_set(n_samples=50, seed=42)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(l1, l2)


def test_synthesize_label_ranges():
    _, labels = synthesize_training_set(n_samples=1000, n_strips=64, seed=1)
    amp, mean, sigma, bck = labels.T
    assert (amp > 0).all()
    assert (sigma > 0).all()
    assert (mean > 0).all() and (mean < 64).all()
    assert (bck >= 0).all()


# ─────────────────────────────────────────────────────────────────────────
# Normalizer
# ─────────────────────────────────────────────────────────────────────────


def test_normalizer_roundtrip():
    x = torch.randn(100, 64) * 500 + 1000
    y = torch.randn(100, 4) * 10 + 50
    norm = ProfileNormalizer()
    norm.fit(x, y)
    y_back = norm.denorm_y(norm.norm_y(y))
    torch.testing.assert_close(y_back, y, rtol=1e-5, atol=1e-5)


def test_normalizer_raises_before_fit():
    norm = ProfileNormalizer()
    with pytest.raises(RuntimeError, match="not fitted"):
        norm.norm_x(torch.zeros(1, 64))


# ─────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────


def test_model_output_shape():
    cfg = MLExtractorConfig(n_strips=64)
    model = StripProfileMLP(cfg)
    x = torch.randn(8, 64)
    y = model(x)
    assert y.shape == (8, 4)


def test_model_respects_n_strips():
    cfg = MLExtractorConfig(n_strips=128, n_params=5)
    model = StripProfileMLP(cfg)
    x = torch.randn(4, 128)
    assert model(x).shape == (4, 5)


# ─────────────────────────────────────────────────────────────────────────
# End-to-end (tiny model, few epochs — this should still converge on synth)
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_extractor():
    cfg = MLExtractorConfig(
        n_strips=64,
        hidden=(64, 32),
        dropout=0.05,
        epochs=5,
        batch_size=64,
        early_stopping_patience=10,  # don't early-stop in a 5-epoch test
        seed=0,
    )
    return MLExtractor(cfg)


def test_fit_reduces_loss(tiny_extractor):
    profiles, labels = synthesize_training_set(n_samples=500, n_strips=64, seed=0)
    history = tiny_extractor.fit(profiles, labels, verbose=False)
    assert len(history["train_loss"]) >= 1
    # First vs last epoch — must have learned *something*
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_predict_shapes(tiny_extractor):
    profiles, labels = synthesize_training_set(n_samples=200, n_strips=64, seed=0)
    tiny_extractor.fit(profiles, labels, verbose=False)

    test_x, _ = synthesize_training_set(n_samples=20, n_strips=64, seed=99)
    result = tiny_extractor.predict(test_x, mc_samples=5)

    assert result["params"].shape == (20, 4)
    assert result["std"].shape == (20, 4)
    assert result["labels"] == ("amp", "mean", "sigma", "bck")
    assert (result["std"] >= 0).all()


def test_predict_raises_before_fit():
    ext = MLExtractor(MLExtractorConfig(epochs=1))
    with pytest.raises(RuntimeError, match="not fitted"):
        ext.predict(np.zeros((1, 64), dtype=np.float32))


def test_save_load_roundtrip(tiny_extractor, tmp_path):
    profiles, labels = synthesize_training_set(n_samples=200, n_strips=64, seed=0)
    tiny_extractor.fit(profiles, labels, verbose=False)

    path = tmp_path / "model.pt"
    tiny_extractor.save(path)
    assert path.exists()

    loaded = MLExtractor.load(path)
    test_x, _ = synthesize_training_set(n_samples=10, n_strips=64, seed=5)

    # Deterministic check requires eval mode; MC-dropout is inherently stochastic,
    # so we compare the deterministic forward pass.
    loaded.model.eval()
    tiny_extractor.model.eval()

    x_n = loaded.norm.norm_x(torch.as_tensor(test_x, dtype=torch.float32)).to(loaded.cfg.device)
    with torch.no_grad():
        y1 = loaded.model(x_n)
        y2 = tiny_extractor.model(x_n)
    torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────────


def test_fit_rejects_wrong_profile_shape(tiny_extractor):
    with pytest.raises(ValueError, match="2D"):
        tiny_extractor.fit(np.zeros(64, dtype=np.float32), np.zeros((1, 4), dtype=np.float32))


def test_fit_rejects_wrong_label_shape(tiny_extractor):
    with pytest.raises(ValueError, match="labels"):
        tiny_extractor.fit(
            np.zeros((10, 64), dtype=np.float32), np.zeros((10, 3), dtype=np.float32)
        )


def test_fit_rejects_mismatched_n():
    cfg = MLExtractorConfig(n_strips=64, epochs=1)
    ext = MLExtractor(cfg)
    with pytest.raises(ValueError, match="same N"):
        ext.fit(np.zeros((10, 64), dtype=np.float32), np.zeros((5, 4), dtype=np.float32))


def test_fit_rejects_wrong_strip_count():
    cfg = MLExtractorConfig(n_strips=64, epochs=1)
    ext = MLExtractor(cfg)
    with pytest.raises(ValueError, match="strips"):
        ext.fit(np.zeros((10, 32), dtype=np.float32), np.zeros((10, 4), dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────
# Markers for future ROOT-dependent tests
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.requires_root
def test_roundtrip_against_upstream_fit():
    """Compare ML predictions against upstream Gaussian fit on real MC data.

    Skipped unless ROOT is installed (see pyproject pytest markers).
    """
    pytest.skip("Requires ROOT + real MC input files; implement when available.")
