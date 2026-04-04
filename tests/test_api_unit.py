"""
Unit tests for api.py helper functions.
These run without loading the ML models — fast CI checks.
"""
import numpy as np
import pytest
import sys, os
import types
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Stub out heavy dependencies so tests run without tensorflow/keras installed
def _stub_keras():
    keras = types.ModuleType("keras")
    keras.layers = types.ModuleType("keras.layers")
    # Dense.from_config patch in api.py needs __func__ on from_config
    dense_mock = mock.MagicMock()
    real_method = lambda cls, c: c  # dummy classmethod
    dense_mock.from_config = classmethod(real_method)
    keras.layers.Dense = dense_mock
    keras.models = types.ModuleType("keras.models")
    keras.models.load_model = mock.MagicMock(return_value=mock.MagicMock())
    sys.modules.setdefault("keras", keras)
    sys.modules.setdefault("keras.layers", keras.layers)
    sys.modules.setdefault("keras.models", keras.models)

    tf = types.ModuleType("tensorflow")
    sys.modules.setdefault("tensorflow", tf)

    mne = types.ModuleType("mne")
    mne.io = types.ModuleType("mne.io")
    sys.modules.setdefault("mne", mne)
    sys.modules.setdefault("mne.io", mne.io)

def _stub_anthropic():
    anthropic_mod = types.ModuleType("anthropic")
    anthropic_mod.Anthropic = mock.MagicMock()
    sys.modules.setdefault("anthropic", anthropic_mod)

_stub_keras()
_stub_anthropic()

with mock.patch("joblib.load", return_value=mock.MagicMock()):
    import api as api_mod


# ---------------------------------------------------------------------------
# _bandpass_filter
# ---------------------------------------------------------------------------
def test_bandpass_filter_shape():
    sig = np.random.randn(3000).astype(np.float32)
    out = api_mod._bandpass_filter(sig, 100)
    assert out.shape == sig.shape


def test_bandpass_filter_reduces_dc():
    # DC offset should be attenuated by high-pass at 0.5 Hz
    sig = np.ones(3000, dtype=np.float32) * 100.0
    out = api_mod._bandpass_filter(sig, 100)
    assert np.abs(out).mean() < 1.0


# ---------------------------------------------------------------------------
# _validate_signal
# ---------------------------------------------------------------------------
def test_validate_signal_ok():
    sig = np.random.randn(3000).astype(np.float32)
    api_mod._validate_signal(sig)  # should not raise


def test_validate_signal_flat():
    from fastapi import HTTPException
    sig = np.ones(3000, dtype=np.float32)
    with pytest.raises(HTTPException) as exc:
        api_mod._validate_signal(sig)
    assert exc.value.status_code == 422
    assert "flat" in exc.value.detail.lower()


def test_validate_signal_nan():
    from fastapi import HTTPException
    sig = np.random.randn(3000).astype(np.float32)
    sig[100] = float("nan")
    with pytest.raises(HTTPException) as exc:
        api_mod._validate_signal(sig)
    assert exc.value.status_code == 422


def test_validate_signal_too_long():
    from fastapi import HTTPException
    sig = np.random.randn(100 * 3600 * 25).astype(np.float32)  # 25 hours
    with pytest.raises(HTTPException) as exc:
        api_mod._validate_signal(sig)
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# _extract_spectral_features
# ---------------------------------------------------------------------------
def test_spectral_features_shape():
    sig = np.random.randn(3000).astype(np.float32)
    feats = api_mod._extract_spectral_features(sig, fs=100)
    assert feats.shape == (15,)


def test_spectral_features_finite():
    sig = np.random.randn(6000).astype(np.float32)
    feats = api_mod._extract_spectral_features(sig, fs=100)
    assert np.isfinite(feats).all()


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------
def test_compute_metrics_score_range():
    stages = [0] * 20 + [2] * 40 + [3] * 20 + [4] * 20  # realistic night
    confidence = [0.9] * len(stages)
    m = api_mod._compute_metrics(stages, confidence)
    assert 0 <= m.sleep_quality_score <= 100
    assert m.recording_minutes > 0
    assert m.sleep_efficiency >= 0


def test_compute_metrics_all_wake():
    stages = [0] * 100
    confidence = [0.8] * 100
    m = api_mod._compute_metrics(stages, confidence)
    assert m.tst_min == 0.0
    assert m.sleep_efficiency == 0.0
    assert m.pct_n3 == 0.0
    assert m.pct_rem == 0.0


def test_compute_metrics_perfect_sleep():
    # High efficiency, low WASO, some N3 + REM
    stages = [0] * 2 + [2] * 30 + [3] * 12 + [4] * 14 + [2] * 5
    confidence = [0.95] * len(stages)
    m = api_mod._compute_metrics(stages, confidence)
    assert m.sleep_quality_score > 50


# ---------------------------------------------------------------------------
# _rule_based_explain
# ---------------------------------------------------------------------------
def test_rule_based_explain_returns_three_tips():
    stages = [0] * 20 + [2] * 40 + [3] * 10 + [4] * 10
    confidence = [0.85] * len(stages)
    m = api_mod._compute_metrics(stages, confidence)
    explanation, tips = api_mod._rule_based_explain(m, "Fair")
    assert isinstance(explanation, str) and len(explanation) > 10
    assert len(tips) == 3


def test_rule_based_explain_poor_sleep():
    # All wake — very poor sleep
    stages = [0] * 100
    confidence = [0.7] * 100
    m = api_mod._compute_metrics(stages, confidence)
    explanation, tips = api_mod._rule_based_explain(m, "Poor")
    assert len(tips) == 3
    assert all(isinstance(t, str) for t in tips)
