"""
Integration tests for the async EDF job flow and supporting endpoints.

Tests use the real FastAPI app (via TestClient) but:
- CNN+LSTM model is mocked (no TensorFlow needed)
- MNE's read_raw_edf is mocked to return synthetic EEG (no real EDF needed)
- App lifespan is bypassed; state is set directly before the TestClient starts
"""
import io
import time
import types
import unittest.mock as mock
from contextlib import asynccontextmanager
import sys
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stub heavy deps before any app imports
# ---------------------------------------------------------------------------
def _stub_module(name):
    m = types.ModuleType(name)
    sys.modules.setdefault(name, m)
    return m

# keras stubs
_keras        = _stub_module("keras")
_keras_layers = _stub_module("keras.layers")
_keras_models = _stub_module("keras.models")
_keras.layers = _keras_layers
_keras.models = _keras_models

_dense_mock = mock.MagicMock()
_dense_mock.from_config = classmethod(lambda cls, c: c)
_keras_layers.Dense = _dense_mock
_keras_models.load_model = mock.MagicMock()   # won't be called — lifespan bypassed

_stub_module("tensorflow")

# anthropic stub
_anthropic = _stub_module("anthropic")
_anthropic.Anthropic = mock.MagicMock()

# mne stub — read_raw_edf returns a synthetic 10-minute EEG signal
_mne    = _stub_module("mne")
_mne_io = _stub_module("mne.io")
_mne.io = _mne_io

class _FakeRaw:
    ch_names = ["EEG Fpz-Cz"]
    info     = {"sfreq": 100.0}
    def __init__(self, signal): self._sig = signal.astype(np.float32)
    def pick_channels(self, chs): return self
    def pick_types(self, **kw):   return self
    def copy(self):               return self
    def get_data(self):           return self._sig[np.newaxis, :]

def _fake_read_raw_edf(path, preload=True, verbose=False):
    n = 100 * 60 * 10   # 10 minutes at 100 Hz
    t = np.linspace(0, 600, n, dtype=np.float32)
    sig = (
        1.5 * np.sin(2 * np.pi * 1.0 * t)
        + 0.8 * np.sin(2 * np.pi * 6.0 * t)
        + 0.1 * np.random.randn(n).astype(np.float32)
    )
    return _FakeRaw(sig)

# Patch on whatever mne.io module is currently registered — handles the case
# where the unit-test stub was registered first via setdefault.
sys.modules["mne.io"].read_raw_edf = _fake_read_raw_edf
sys.modules["mne"].io = sys.modules["mne.io"]

# ---------------------------------------------------------------------------
# Mock CNN+LSTM model: always predicts N2 (stage 2) with 60 % confidence
# ---------------------------------------------------------------------------
class _MockModel:
    def predict(self, X, verbose=0):
        n_seqs = X.shape[0]
        probs = np.full((n_seqs, 5), 0.1, dtype=np.float32)
        probs[:, 2] = 0.6
        return probs

_mock_cnn = _MockModel()
_mock_reg = mock.MagicMock(predict=lambda X: np.array([42.0]))

# ---------------------------------------------------------------------------
# Import app modules (stubs are already in sys.modules)
# ---------------------------------------------------------------------------
import joblib  # noqa: E402

with mock.patch.object(joblib, "load", return_value=_mock_reg):
    from app import state          # noqa: E402
    from app.main import app       # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: bypass lifespan, inject mocks directly into state
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    state.model            = _mock_cnn
    state.reg_model        = _mock_reg
    state.anthropic_client = None

    # Replace the lifespan with a no-op so TestClient doesn't try to load models
    @asynccontextmanager
    async def _noop_lifespan(application):
        yield

    app.router.lifespan_context = _noop_lifespan

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fake_edf_upload():
    """Minimal bytes uploaded as an .edf file (MNE stub ignores content)."""
    return ("night.edf", io.BytesIO(b"\x00" * 512), "application/octet-stream")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["cnn_lstm_loaded"] is True
    assert body["regression_loaded"] is True


# ---------------------------------------------------------------------------
# /predict  (JSON EEG array)
# ---------------------------------------------------------------------------
def test_predict_json_returns_epochs_and_metrics(client):
    signal = (np.random.randn(5 * 3000) * 10).tolist()
    r = client.post("/predict", json={"eeg_signal": signal, "smooth": False})
    assert r.status_code == 200
    body = r.json()
    assert "epochs" in body and "metrics" in body
    assert len(body["epochs"]) == 5
    assert 0 <= body["metrics"]["sleep_quality_score"] <= 100


def test_predict_json_rejects_flat_signal(client):
    r = client.post("/predict", json={"eeg_signal": [0.0] * 15000})
    assert r.status_code == 422
    assert "flat" in r.json()["detail"].lower()


def test_predict_json_rejects_too_short(client):
    r = client.post("/predict", json={"eeg_signal": np.random.randn(100).tolist()})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /predict/edf  →  /jobs/{job_id}  — the core async lifecycle
# ---------------------------------------------------------------------------
def test_edf_job_full_lifecycle(client):
    """Submit EDF → receive job_id immediately → poll until done → check shape."""
    r = client.post(
        "/predict/edf",
        files={"file": _fake_edf_upload()},
        params={"channel": "EEG Fpz-Cz", "smooth": "true"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "processing"
    job_id = body["job_id"]

    # Poll until done (max 15 seconds — background thread does real signal processing)
    deadline = time.time() + 15
    result = None
    while time.time() < deadline:
        poll = client.get(f"/jobs/{job_id}")
        assert poll.status_code == 200
        jbody = poll.json()
        if jbody["status"] == "done":
            result = jbody["result"]
            break
        if jbody["status"] == "failed":
            pytest.fail(f"Job failed unexpectedly: {jbody.get('error')}")
        time.sleep(0.25)

    assert result is not None, "Job did not complete within 15 s"
    assert "epochs" in result and "metrics" in result
    assert len(result["epochs"]) > 0
    assert all(e["stage_name"] in ("Wake", "N1", "N2", "N3", "REM") for e in result["epochs"])
    assert 0 <= result["metrics"]["sleep_quality_score"] <= 100


def test_edf_job_submit_returns_immediately(client):
    """The /predict/edf endpoint must return before inference finishes."""
    import time
    t0 = time.time()
    r = client.post(
        "/predict/edf",
        files={"file": _fake_edf_upload()},
    )
    elapsed = time.time() - t0
    assert r.status_code == 200
    assert elapsed < 5.0, f"Expected immediate response, took {elapsed:.1f}s"


def test_edf_job_rejects_non_edf(client):
    r = client.post(
        "/predict/edf",
        files={"file": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")},
    )
    assert r.status_code == 422
    assert "edf" in r.json()["detail"].lower()


def test_jobs_404_for_unknown_id(client):
    r = client.get("/jobs/does-not-exist-xyz")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /predict/quick-score
# ---------------------------------------------------------------------------
def test_quick_score_returns_score_in_range(client):
    r = client.post(
        "/predict/quick-score",
        files={"file": _fake_edf_upload()},
    )
    assert r.status_code == 200
    body = r.json()
    assert "predicted_score" in body
    assert 0 <= body["predicted_score"] <= 100


# ---------------------------------------------------------------------------
# /explain  (rule-based fallback — no Anthropic key set in tests)
# ---------------------------------------------------------------------------
def test_explain_returns_explanation_and_three_tips(client):
    metrics = {
        "recording_minutes":  480.0,
        "tst_min":            380.0,
        "sleep_latency_min":  12.0,
        "sleep_efficiency":   0.79,
        "waso_min":           25.0,
        "pct_n3":             0.18,
        "pct_rem":            0.21,
        "fragmentation_index": 0.08,
        "confidence_mean":    0.88,
        "sleep_quality_score": 61.5,
    }
    r = client.post("/explain", json={"metrics": metrics, "language": "english"})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["explanation"], str) and len(body["explanation"]) > 10
    assert len(body["tips"]) == 3
    assert body["ai_available"] is False   # no ANTHROPIC_API_KEY in test env