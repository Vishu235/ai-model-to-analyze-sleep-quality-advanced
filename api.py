import os
import tempfile
import uuid
import threading
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

import joblib
import mne
import numpy as np
import keras
import keras.layers
import tensorflow as tf
import anthropic
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis as sp_kurtosis

# Model was saved with an older Keras 3 that included 'quantization_config' in Dense.
# Keras 3.6+ removed this field. Patch from_config to drop it gracefully.
_orig_dense_from_config = keras.layers.Dense.from_config.__func__

@classmethod
def _patched_dense_from_config(cls, config):
    config.pop("quantization_config", None)
    return _orig_dense_from_config(cls, config)

keras.layers.Dense.from_config = _patched_dense_from_config
from pydantic import BaseModel
from scipy.ndimage import median_filter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "inference", "sleep_best_ckpt.keras"),
)
TARGET_FS = 100
EPOCH_SECONDS = 30
SAMPLES_PER_EPOCH = TARGET_FS * EPOCH_SECONDS  # 3000
SEQ_LEN = 5
SMOOTH_KERNEL = 5

STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

# Bandpass filter config (same as training pipeline)
LOWCUT = 0.5
HIGHCUT = 30.0
FILTER_ORDER = 4

# Sleep quality thresholds
SE_MIN, SE_GOOD = 0.70, 0.90
WASO_BAD_MIN = 90.0
LATENCY_REF_MIN = 45.0
N3_TARGET, REM_TARGET = 0.20, 0.22
FRAG_BAD_REF = 0.12
W_SE, W_WASO, W_LATENCY, W_REST, W_FRAG = 0.30, 0.20, 0.15, 0.20, 0.15

# ---------------------------------------------------------------------------
# Anthropic client (optional — only if ANTHROPIC_API_KEY is set)
# ---------------------------------------------------------------------------
_ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_anthropic_client: Optional[anthropic.Anthropic] = None


# ---------------------------------------------------------------------------
# API Key Auth
# ---------------------------------------------------------------------------
_API_KEY = os.environ.get("API_KEY", "")   # set in Railway env vars
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str = Security(_api_key_header)):
    if not _API_KEY:
        return  # auth disabled when API_KEY env var is not set
    if key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# Regression model paths
REG_MODEL_PATH = os.environ.get(
    "REG_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "regression_model.pkl"),
)

# ---------------------------------------------------------------------------
# Global model handles
# ---------------------------------------------------------------------------
_model     = None   # CNN+LSTM classification model
_reg_model = None   # Regression model (quick score)

# ---------------------------------------------------------------------------
# In-memory job store for async EDF processing
# ---------------------------------------------------------------------------
_jobs: Dict[str, Dict[str, Any]] = {}   # job_id -> {status, result, error}
_jobs_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _reg_model, _anthropic_client
    _model = keras.models.load_model(MODEL_PATH, compile=False, safe_mode=False)
    if os.path.exists(REG_MODEL_PATH):
        _reg_model = joblib.load(REG_MODEL_PATH)
    if _ANTHROPIC_API_KEY:
        _anthropic_client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)
    yield


app = FastAPI(title="Sleep Staging API", lifespan=lifespan)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    eeg_signal: List[float]   # raw EEG samples at 100 Hz
    smooth: bool = True


class EpochResult(BaseModel):
    epoch: int
    stage_id: int
    stage_name: str
    confidence: float


class Metrics(BaseModel):
    recording_minutes: float
    tst_min: float
    sleep_latency_min: float
    sleep_efficiency: float
    waso_min: float
    pct_n3: float
    pct_rem: float
    fragmentation_index: float
    confidence_mean: float
    sleep_quality_score: float


class PredictResponse(BaseModel):
    epochs: List[EpochResult]
    metrics: Metrics


# Limits
MAX_SAMPLES = TARGET_FS * 3600 * 24   # 24 hours max
MAX_EDF_MB  = 500                      # 500 MB file size limit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_signal(signal: np.ndarray) -> None:
    """Raise HTTPException if the signal is unusable."""
    if len(signal) > MAX_SAMPLES:
        raise HTTPException(
            status_code=422,
            detail=f"Signal too long: {len(signal) / TARGET_FS / 3600:.1f}h. Maximum is 10 hours.",
        )
    if not np.isfinite(signal).all():
        raise HTTPException(
            status_code=422,
            detail="Signal contains NaN or Inf values. Check your data for corrupt samples.",
        )
    if signal.std() < 1e-6:
        raise HTTPException(
            status_code=422,
            detail="Signal is flat (std ≈ 0). The electrode may be disconnected or the channel is empty.",
        )


def _bandpass_filter(signal: np.ndarray, fs: int) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(FILTER_ORDER, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def _extract_spectral_features(signal: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Extract 15 spectral + statistical features used by the regression model."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs * 4))
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
    bp = {}
    for b_name, (lo, hi) in bands.items():
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        bp[b_name] = float(_trapz(psd[idx], freqs[idx]))
    total = sum(bp.values()) + 1e-10
    feats = (
        [bp["delta"], bp["theta"], bp["alpha"], bp["beta"]]
        + [bp[b] / total for b in ["delta", "theta", "alpha", "beta"]]
        + [bp["delta"] / (bp["beta"] + 1e-10), bp["theta"] / (bp["alpha"] + 1e-10)]
        + [float(signal.mean()), float(signal.std()),
           float(skew(signal)), float(sp_kurtosis(signal)),
           float(np.sqrt(np.mean(signal ** 2)))]
    )
    return np.array(feats, dtype=np.float32)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _compute_metrics(stages: list, confidence: list) -> Metrics:
    n = len(stages)
    recording_minutes = (n * EPOCH_SECONDS) / 60.0

    sleep_epochs = [s for s in stages if s != 0]
    tst_min = len(sleep_epochs) * EPOCH_SECONDS / 60.0
    sleep_efficiency = tst_min / recording_minutes if recording_minutes > 0 else 0.0

    first_sleep = next((i for i, s in enumerate(stages) if s != 0), None)
    sleep_latency_min = (
        first_sleep * EPOCH_SECONDS / 60.0 if first_sleep is not None else recording_minutes
    )

    waso_epochs = 0
    if first_sleep is not None:
        for s in stages[first_sleep:]:
            if s == 0:
                waso_epochs += 1
    waso_min = waso_epochs * EPOCH_SECONDS / 60.0

    pct_n3 = stages.count(3) / len(sleep_epochs) if sleep_epochs else 0.0
    pct_rem = stages.count(4) / len(sleep_epochs) if sleep_epochs else 0.0

    transitions = sum(1 for i in range(1, n) if stages[i] != stages[i - 1])
    fragmentation_index = transitions / n if n > 0 else 0.0
    confidence_mean = float(np.mean(confidence)) if confidence else 0.0

    se_score = _clip01((sleep_efficiency - SE_MIN) / (SE_GOOD - SE_MIN)) * 100
    waso_score = _clip01((WASO_BAD_MIN - waso_min) / WASO_BAD_MIN) * 100
    lat_score = _clip01((LATENCY_REF_MIN - sleep_latency_min) / LATENCY_REF_MIN) * 100
    n3_score = _clip01(pct_n3 / N3_TARGET) * 100
    rem_score = _clip01(pct_rem / REM_TARGET) * 100
    rest_score = 0.5 * (n3_score + rem_score)
    frag_score = _clip01((FRAG_BAD_REF - fragmentation_index) / FRAG_BAD_REF) * 100

    quality = (
        W_SE * se_score
        + W_WASO * waso_score
        + W_LATENCY * lat_score
        + W_REST * rest_score
        + W_FRAG * frag_score
    )

    return Metrics(
        recording_minutes=round(recording_minutes, 2),
        tst_min=round(tst_min, 2),
        sleep_latency_min=round(sleep_latency_min, 2),
        sleep_efficiency=round(sleep_efficiency, 4),
        waso_min=round(waso_min, 2),
        pct_n3=round(pct_n3, 4),
        pct_rem=round(pct_rem, 4),
        fragmentation_index=round(fragmentation_index, 4),
        confidence_mean=round(confidence_mean, 4),
        sleep_quality_score=round(quality, 2),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "cnn_lstm_loaded": _model is not None,
        "regression_loaded": _reg_model is not None,
        "ai_explain_available": _anthropic_client is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _=Depends(require_api_key)):
    signal = np.array(req.eeg_signal, dtype=np.float32)

    _validate_signal(signal)

    n_epochs = len(signal) // SAMPLES_PER_EPOCH
    min_samples = SEQ_LEN * SAMPLES_PER_EPOCH
    if len(signal) < min_samples:
        raise HTTPException(
            status_code=422,
            detail=f"Need at least {min_samples} samples ({SEQ_LEN * EPOCH_SECONDS}s at {TARGET_FS} Hz). Got {len(signal)}.",
        )

    signal = signal[: n_epochs * SAMPLES_PER_EPOCH]

    # Bandpass filter: 0.5–30 Hz (same as training pipeline)
    signal = _bandpass_filter(signal, TARGET_FS)

    epochs = signal.reshape(n_epochs, SAMPLES_PER_EPOCH)

    # Z-score normalize per epoch
    mean = epochs.mean(axis=1, keepdims=True)
    std = epochs.std(axis=1, keepdims=True) + 1e-8
    epochs = (epochs - mean) / std

    # Sequences: (n_seqs, SEQ_LEN, 3000, 1)
    X = np.stack(
        [epochs[i : i + SEQ_LEN] for i in range(n_epochs - SEQ_LEN + 1)], axis=0
    )[..., np.newaxis]

    probs = _model.predict(X, verbose=0)
    pred_seq = probs.argmax(axis=1)
    conf_seq = probs.max(axis=1)

    stages = [0] * n_epochs
    confidence = [0.0] * n_epochs
    for i in range(len(pred_seq)):
        idx = i + SEQ_LEN - 1
        stages[idx] = int(pred_seq[i])
        confidence[idx] = float(conf_seq[i])
    for i in range(SEQ_LEN - 1):
        stages[i] = stages[SEQ_LEN - 1]
        confidence[i] = confidence[SEQ_LEN - 1]

    if req.smooth:
        stages = median_filter(np.array(stages), size=SMOOTH_KERNEL).tolist()

    epoch_results = [
        EpochResult(
            epoch=i,
            stage_id=stages[i],
            stage_name=STAGE_NAMES[stages[i]],
            confidence=round(confidence[i], 4),
        )
        for i in range(n_epochs)
    ]

    return PredictResponse(
        epochs=epoch_results,
        metrics=_compute_metrics(stages, confidence),
    )


# ---------------------------------------------------------------------------
# Explain endpoint — RAG: Claude explains sleep results in plain language
# ---------------------------------------------------------------------------
class ExplainRequest(BaseModel):
    metrics: Metrics
    language: str = "english"   # "english" or "hindi"


class ExplainResponse(BaseModel):
    explanation: str
    tips: List[str]
    ai_available: bool


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest, _=Depends(require_api_key)):
    """Use Claude to explain sleep metrics in plain language with actionable tips."""
    m = req.metrics
    score = m.sleep_quality_score

    if score >= 75:
        label = "Great"
    elif score >= 55:
        label = "Good"
    elif score >= 35:
        label = "Fair"
    else:
        label = "Poor"

    if _anthropic_client is None:
        explanation, tips = _rule_based_explain(m, label)
        return ExplainResponse(explanation=explanation, tips=tips, ai_available=False)

    lang_instruction = (
        "Respond in simple Hindi (Devanagari script)." if req.language == "hindi"
        else "Respond in clear, simple English."
    )

    prompt = f"""You are a sleep health assistant. A patient's overnight EEG sleep study produced these results:

- Sleep Quality Score: {score:.1f}/100 ({label})
- Total Recording: {m.recording_minutes:.0f} minutes
- Total Sleep Time: {m.tst_min:.0f} minutes
- Sleep Efficiency: {m.sleep_efficiency*100:.1f}% (normal: >85%)
- Sleep Latency: {m.sleep_latency_min:.1f} minutes to fall asleep (normal: <20 min)
- WASO (wake after sleep onset): {m.waso_min:.0f} minutes (normal: <30 min)
- Deep Sleep (N3): {m.pct_n3*100:.1f}% of sleep (normal: 15-25%)
- REM Sleep: {m.pct_rem*100:.1f}% of sleep (normal: 20-25%)
- Fragmentation Index: {m.fragmentation_index*100:.1f}% (lower is better)
- Model Confidence: {m.confidence_mean*100:.0f}%

{lang_instruction}

Provide:
1. A 2-3 sentence plain-language explanation of what these results mean.
2. Exactly 3 specific, actionable tips to improve sleep quality.

Format as JSON only — no extra text:
{{"explanation": "...", "tips": ["tip1", "tip2", "tip3"]}}"""

    try:
        response = _anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        import json as _json
        text = response.content[0].text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        parsed = _json.loads(text[start:end])
        return ExplainResponse(
            explanation=parsed["explanation"],
            tips=parsed["tips"],
            ai_available=True,
        )
    except Exception:
        explanation, tips = _rule_based_explain(m, label)
        return ExplainResponse(explanation=explanation, tips=tips, ai_available=False)


def _rule_based_explain(m: Metrics, label: str):
    """Rule-based fallback when ANTHROPIC_API_KEY is not set."""
    parts = []
    if m.sleep_efficiency < 0.75:
        parts.append(f"Your sleep efficiency is low at {m.sleep_efficiency*100:.0f}% (target >85%), meaning much of your time in bed was spent awake.")
    else:
        parts.append(f"Your sleep efficiency is {m.sleep_efficiency*100:.0f}%, which is {'good' if m.sleep_efficiency >= 0.85 else 'acceptable'}.")
    if m.waso_min > 30:
        parts.append(f"You woke up frequently after falling asleep, losing {m.waso_min:.0f} minutes of sleep.")
    if m.pct_n3 < 0.10:
        parts.append("Your deep sleep (N3) is below normal, which may affect physical recovery.")
    if m.pct_rem < 0.15:
        parts.append("Your REM sleep is low, which can impact memory and mood.")
    explanation = " ".join(parts) if parts else f"Your sleep quality is {label.lower()} with a score of {m.sleep_quality_score:.0f}/100."

    tips = []
    if m.sleep_latency_min > 20:
        tips.append("Maintain a consistent sleep schedule — same bedtime and wake time every day.")
    if m.waso_min > 30:
        tips.append("Avoid screens and bright light 1 hour before bed to reduce nighttime waking.")
    if m.pct_n3 < 0.15:
        tips.append("Exercise regularly (not within 3 hours of bedtime) to increase deep sleep.")
    if m.pct_rem < 0.20:
        tips.append("Limit alcohol — even small amounts significantly reduce REM sleep.")
    if m.fragmentation_index > 0.10:
        tips.append("Keep your bedroom cool (18-20°C) and dark to reduce sleep fragmentation.")
    defaults = [
        "Avoid caffeine after 2 PM.",
        "Use your bed only for sleep — not for work or screens.",
        "Try 10 minutes of breathing exercises before bed.",
    ]
    for d in defaults:
        if len(tips) >= 3:
            break
        if d not in tips:
            tips.append(d)
    return explanation, tips[:3]

# ---------------------------------------------------------------------------
# Quick Score endpoint — Regression model (no CNN+LSTM needed)
# ---------------------------------------------------------------------------
class QuickScoreResponse(BaseModel):
    predicted_score: float
    score_range: str = "0-100"
    model: str = "RandomForest (spectral EEG features)"
    note: str = "Fast estimate. Use /predict/edf for full staging + accurate score."


@app.post("/predict/quick-score", response_model=QuickScoreResponse)
async def predict_quick_score(
    file: UploadFile = File(...),
    channel: str = "EEG Fpz-Cz",
    _=Depends(require_api_key),
):
    """Fast sleep quality score using regression model on spectral EEG features.
    Does NOT run CNN+LSTM — returns in seconds instead of minutes."""
    if _reg_model is None:
        raise HTTPException(status_code=503, detail="Regression model not loaded.")
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=422, detail="Only .edf files are supported.")

    content = await file.read()
    if len(content) > MAX_EDF_MB * 1024 * 1024:
        raise HTTPException(status_code=422,
            detail=f"File too large: {len(content)/1024/1024:.1f} MB. Max {MAX_EDF_MB} MB.")

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(content); tmp_path = tmp.name

    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        if channel in raw.ch_names:
            raw.pick_channels([channel])
        else:
            eeg_raw = raw.copy().pick_types(eeg=True)
            if not eeg_raw.ch_names:
                raise HTTPException(status_code=422,
                    detail=f"Channel '{channel}' not found. Available: {raw.ch_names}")
            raw = eeg_raw; raw.pick_channels([raw.ch_names[0]])
        if raw.info["sfreq"] != TARGET_FS:
            raw.resample(TARGET_FS, verbose=False)
        signal = raw.get_data()[0].astype(np.float32)
    finally:
        os.unlink(tmp_path)

    _validate_signal(signal)
    signal = _bandpass_filter(signal, TARGET_FS)
    feats  = _extract_spectral_features(signal, TARGET_FS).reshape(1, -1)
    score  = float(np.clip(_reg_model.predict(feats)[0], 0, 100))
    return QuickScoreResponse(predicted_score=round(score, 1))


# ---------------------------------------------------------------------------
# Async EDF processing — runs CNN+LSTM in background thread, no timeout
# ---------------------------------------------------------------------------
def _run_edf_job(job_id: str, content: bytes, channel: str, smooth: bool):
    """Runs in a background thread. Saves result to _jobs[job_id]."""
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        if channel in raw.ch_names:
            raw.pick_channels([channel])
        else:
            eeg_raw = raw.copy().pick_types(eeg=True)
            if not eeg_raw.ch_names:
                raise ValueError(f"Channel '{channel}' not found. Available: {raw.ch_names}")
            raw = eeg_raw
            raw.pick_channels([raw.ch_names[0]])
        if raw.info["sfreq"] != TARGET_FS:
            raw.resample(TARGET_FS, verbose=False)
        signal = raw.get_data()[0].astype(np.float32)
    finally:
        os.unlink(tmp_path)

    if len(signal) > MAX_SAMPLES:
        raise ValueError(f"Signal too long: {len(signal)/TARGET_FS/3600:.1f}h. Max 24h.")
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or Inf values.")
    if signal.std() < 1e-6:
        raise ValueError("Signal is flat — electrode may be disconnected.")

    n_epochs = len(signal) // SAMPLES_PER_EPOCH
    if n_epochs < SEQ_LEN:
        raise ValueError(f"EDF too short. Need ≥{SEQ_LEN * EPOCH_SECONDS}s. Got {len(signal)/TARGET_FS:.1f}s.")

    signal = signal[: n_epochs * SAMPLES_PER_EPOCH]
    signal = _bandpass_filter(signal, TARGET_FS)
    epochs = signal.reshape(n_epochs, SAMPLES_PER_EPOCH)
    mean = epochs.mean(axis=1, keepdims=True)
    std  = epochs.std(axis=1, keepdims=True) + 1e-8
    epochs = (epochs - mean) / std

    X = np.stack(
        [epochs[i : i + SEQ_LEN] for i in range(n_epochs - SEQ_LEN + 1)], axis=0
    )[..., np.newaxis]

    probs    = _model.predict(X, verbose=0)
    pred_seq = probs.argmax(axis=1)
    conf_seq = probs.max(axis=1)

    stages     = [0]   * n_epochs
    confidence = [0.0] * n_epochs
    for i in range(len(pred_seq)):
        stages[i + SEQ_LEN - 1]     = int(pred_seq[i])
        confidence[i + SEQ_LEN - 1] = float(conf_seq[i])
    for i in range(SEQ_LEN - 1):
        stages[i]     = stages[SEQ_LEN - 1]
        confidence[i] = confidence[SEQ_LEN - 1]

    if smooth:
        stages = median_filter(np.array(stages), size=SMOOTH_KERNEL).tolist()

    result = PredictResponse(
        epochs=[
            EpochResult(epoch=i, stage_id=stages[i],
                        stage_name=STAGE_NAMES[stages[i]],
                        confidence=round(confidence[i], 4))
            for i in range(n_epochs)
        ],
        metrics=_compute_metrics(stages, confidence),
    )
    with _jobs_lock:
        _jobs[job_id] = {"status": "done", "result": result.model_dump()}


def _start_edf_job(job_id: str, content: bytes, channel: str, smooth: bool):
    def _worker():
        try:
            _run_edf_job(job_id, content, channel, smooth)
        except Exception as exc:
            with _jobs_lock:
                _jobs[job_id] = {"status": "failed", "error": str(exc)}
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


class JobResponse(BaseModel):
    job_id: str
    status: str   # "processing"


class JobStatusResponse(BaseModel):
    status: str           # "processing" | "done" | "failed"
    result: Optional[PredictResponse] = None
    error: Optional[str] = None


@app.post("/predict/edf", response_model=JobResponse)
async def predict_edf(
    file: UploadFile = File(...),
    channel: str = "EEG Fpz-Cz",
    smooth: bool = True,
    _=Depends(require_api_key),
):
    """Submit EDF for async processing. Returns job_id — poll GET /jobs/{job_id}."""
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=422, detail="Only .edf files are supported.")

    content = await file.read()
    if len(content) > MAX_EDF_MB * 1024 * 1024:
        raise HTTPException(
            status_code=422,
            detail=f"File too large: {len(content)/1024/1024:.1f} MB. Max {MAX_EDF_MB} MB.",
        )

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "processing"}

    _start_edf_job(job_id, content, channel, smooth)
    return JobResponse(job_id=job_id, status="processing")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, _=Depends(require_api_key)):
    """Poll for EDF processing result."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] == "done":
        return JobStatusResponse(status="done", result=job["result"])
    if job["status"] == "failed":
        return JobStatusResponse(status="failed", error=job["error"])
    return JobStatusResponse(status="processing")
