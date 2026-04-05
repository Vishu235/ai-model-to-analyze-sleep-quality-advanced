# Sleep Staging AI — Complete Project Documentation

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Railway](https://img.shields.io/badge/Deployed-Railway-purple)

**Live URL:** https://sleep-staging-api-production.up.railway.app

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [ML Model — CNN+LSTM](#3-ml-model--cnnlstm)
4. [Dataset & Preprocessing](#4-dataset--preprocessing)
5. [API Design & Endpoints](#5-api-design--endpoints)
6. [Signal Processing Pipeline](#6-signal-processing-pipeline)
7. [Sleep Quality Scoring](#7-sleep-quality-scoring)
8. [Regression Model (Quick Score)](#8-regression-model-quick-score)
9. [AI Explanation — Claude Haiku](#9-ai-explanation--claude-haiku)
10. [Async Job Queue](#10-async-job-queue)
11. [Structured Logging](#11-structured-logging)
12. [Model Evaluation Scripts](#12-model-evaluation-scripts)
13. [Docker & Deployment](#13-docker--deployment)
14. [Problems Faced & Solutions](#14-problems-faced--solutions)
15. [Project Structure](#15-project-structure)
16. [How to Run Locally](#16-how-to-run-locally)
17. [Environment Variables](#17-environment-variables)
18. [API Reference](#18-api-reference)

---

## 1. Project Overview

This project is an AI-powered REST API that takes a raw EEG (brain signal) recording from an overnight sleep study and automatically:

- Classifies every 30-second epoch into a sleep stage (Wake / N1 / N2 / N3 / REM)
- Computes clinical sleep quality metrics (efficiency, latency, WASO, etc.)
- Assigns a sleep quality score (0–100)
- Generates a plain-language explanation with 3 actionable health tips

### Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| ML Model | CNN + LSTM (Keras / TF 2.17) | Local waveform features (CNN) + temporal context across epochs (LSTM) |
| API | FastAPI | Async, fast, auto-generates `/docs` Swagger UI |
| Regression | scikit-learn RandomForest | Instant sleep score from spectral EEG features |
| AI Explanation | Claude Haiku (Anthropic API) | Plain-language explanation with rule-based fallback |
| Container | Docker | Consistent environment across local and cloud |
| Hosting | Railway | Git-push deployment |

---

## 2. System Architecture

The project has three distinct layers:

```
┌──────────────────────────────────────────┐
│           Training Layer                 │  src/  — CNN+LSTM training code
│        (runs once, offline)              │  gen_dataset.py — regression data
└──────────────────┬───────────────────────┘
                   │ produces
                   ▼
          inference/sleep_best_ckpt.keras
          regression_model.pkl

┌──────────────────────────────────────────┐
│             API Layer                    │  app/  — FastAPI server
│       (runs 24/7 on Railway)             │  api.py — entry point
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│          Evaluation Layer                │  evaluate_model.py
│       (runs once, for report)            │  cross_dataset_eval.py
└──────────────────────────────────────────┘
```

### Server Startup Sequence

When `uvicorn api:app` is run:

```
api.py  ──imports──►  app/main.py
                           │
                    1. configure_logging()       ← JSON structured logs
                    2. Keras compatibility patch ← fixes Dense.from_config
                    3. load CNN+LSTM model       ← into state.model
                    4. load RandomForest         ← into state.reg_model
                    5. init Anthropic client     ← into state.anthropic_client
                    6. register routes + middleware
                    7. yield  ← server is live
```

### Global State (`app/state.py`)

All three model handles live as module-level globals, populated once during startup:

```python
model            = None  # CNN+LSTM Keras model
reg_model        = None  # RandomForest regression model
anthropic_client = None  # Anthropic client (None if no API key)
```

All routes import `from app import state` and read these directly. Safe because they are set once at startup and never mutated again.

---

## 3. ML Model — CNN+LSTM

### Why CNN+LSTM?

Sleep staging is a sequential problem. A single 30-second epoch alone is ambiguous — clinicians also look at the epochs before and after to make a decision. The architecture mirrors this:

- **CNN** reads each individual epoch and extracts local waveform features (spindles, K-complexes, slow waves)
- **LSTM** sees a sequence of 5 consecutive epochs and learns temporal transitions (N2 usually follows N1, N3 follows N2, etc.)

```
Input: 5 consecutive EEG epochs  [batch, 5, 3000, 1]
                 ↓
     TimeDistributed CNN
  (each epoch processed independently)
                 ↓
          LSTM layer
  (learns: N2→N3→REM patterns etc.)
                 ↓
       Dense + Softmax (5 classes)
                 ↓
  Output: Stage for the last (5th) epoch
```

### Input Shape

```
(batch_size,  5,    3000,   1)
     ↑        ↑      ↑      ↑
  samples   seq   samples  channel
           len   per epoch
           (5)   (30s × 100Hz)
```

### CNN Feature Extraction

- Small kernels → high-frequency features (sleep spindles ~12–15 Hz, K-complexes)
- Large kernels → slow waves (delta waves dominant in N3)
- Pooling layers reduce dimensionality before LSTM

### LSTM Context

- 5-epoch sequence = 2.5 minutes of temporal context
- Learns the cyclical structure: Wake → N1 → N2 → N3 → REM → N2 → ...

### Post-Processing

After inference, a median filter (size=5) is applied to the predicted stage sequence:

```python
stages = median_filter(stages, size=5)
```

This removes single-epoch noise spikes (e.g., one N1 epoch surrounded by N2 on both sides is almost certainly a misclassification).

### Output

- 5 classes: Wake (0), N1 (1), N2 (2), N3 (3), REM (4)
- Per-epoch confidence score (0.0 – 1.0)
- **Accuracy: 94.59%** on PhysioNet Sleep-EDF test set

### Keras Compatibility Patch

The model was saved with an intermediate Keras 3 version that stored a `quantization_config` key in Dense layer configs. Newer Keras 3.6+ removed it, causing a load error. The patch in `app/main.py` intercepts `Dense.from_config` and silently drops the key:

```python
_orig = keras.layers.Dense.from_config.__func__

@classmethod
def _patched_dense_from_config(cls, config):
    config.pop("quantization_config", None)
    return _orig(cls, config)

keras.layers.Dense.from_config = _patched_dense_from_config
```

This must run before `keras.models.load_model` is called.

---

## 4. Dataset & Preprocessing

**Dataset:** PhysioNet Sleep-EDF Expanded — 153 whole-night polysomnography (PSG) recordings with expert-annotated sleep stage labels.

- **EEG Channel:** Fpz-Cz (single channel used)
- **Sampling Rate:** 100 Hz (recordings at other rates are resampled)
- **Epoch Length:** 30 seconds = 3000 samples
- **Labels:** W, N1, N2, N3/N4 (merged into N3), REM

### Training vs Evaluation Splits

| Split | Subjects | Use |
|-------|---------|-----|
| Sleep-EDF Telemetry (ST*) | Training subjects | Model was trained on this |
| Sleep-EDF Cassette (SC*) | Held-out subjects | Never seen during training — used for generalisation evaluation |

### Preprocessing Pipeline

```
Raw EEG signal (continuous)
         ↓
Bandpass filter: 0.5–30 Hz (Butterworth order 4)
         ↓
Segment into 30-second epochs (3000 samples each)
         ↓
Z-score normalise each epoch independently:
    epoch = (epoch - mean) / (std + 1e-8)
         ↓
Build sliding windows of 5 epochs each
         ↓
CNN+LSTM prediction
```

### Why Bandpass Filter?

| Frequency | Noise source | Action |
|-----------|-------------|--------|
| < 0.5 Hz | DC drift, slow movement artifacts | Remove |
| > 30 Hz | EMG (muscle), power line (50/60 Hz) | Remove |
| 0.5–30 Hz | All clinically relevant brain waves | Keep |

### Why Z-score Normalisation?

Different EEG recordings have different amplitude scales depending on electrode placement and skin resistance. Z-score normalisation makes each epoch amplitude-independent — the model sees relative waveform patterns, not absolute voltages. This is critical for generalisation across different patients and recording setups.

---

## 5. API Design & Endpoints

### Why FastAPI?

- Automatic OpenAPI docs at `/docs` with zero extra code
- Async support for file uploads (non-blocking I/O)
- Pydantic v2 for automatic input validation and serialisation
- Significantly faster than Flask under concurrent load

### Endpoints

```
GET  /                    → Frontend dashboard (static/index.html)
GET  /health              → Model load status check
POST /predict             → JSON EEG array → full sleep staging (sync)
POST /predict/edf         → EDF file upload → job_id (async)
GET  /jobs/{job_id}       → Poll for EDF processing result
POST /predict/quick-score → Fast regression score from EDF (sync, ~1s)
POST /explain             → AI explanation of sleep metrics
```

### Authentication

All endpoints accept an optional `X-API-Key` header.

- If `API_KEY` env var is **not set** → auth is disabled (open API, good for development)
- If `API_KEY` **is set** → any missing or wrong key returns 401

### Input Validation (`app/services/signal.py`)

Every signal passes through three checks before inference:

```python
# 1. Length limit
if len(signal) > TARGET_FS * 3600 * 24:   # > 24 hours → reject

# 2. Corrupt values
if not np.isfinite(signal).all():           # NaN or Inf → reject

# 3. Dead electrode
if signal.std() < 1e-6:                    # flat signal → reject
```

File uploads additionally check:
- File extension must be `.edf`
- File size must be ≤ 500 MB

---

## 6. Signal Processing Pipeline

The full pipeline inside `app/services/inference.py:run_inference()`:

```
Validated EEG signal (float32 array at 100 Hz)
         │
         ▼
bandpass_filter()              [app/services/signal.py]
└─ Butterworth 0.5–30 Hz
         │
         ▼
Segment into 3000-sample epochs
Z-score normalise per epoch
         │
         ▼
Build sliding windows of 5 epochs
Shape: [n_windows, 5, 3000, 1]
         │
         ▼
state.model.predict(X)
→ probs  [n_windows, 5]     (softmax probabilities)
→ stage  = argmax(probs)
→ conf   = max(probs)
         │
         ▼
Fill-forward first 4 epochs (no LSTM context yet)
         │
         ▼
median_filter(size=5)         [smoothing]
         │
         ▼
compute_metrics(stages, conf) [app/services/scoring.py]
         │
         ▼
PredictResponse {epochs: [...], metrics: {...}}
```

**Why sliding windows?** The model classifies epoch `i` using epochs `i-4` through `i`. So the first classifiable epoch is epoch 4 (index 4). The first 4 epochs are fill-forwarded with the same prediction as epoch 4.

---

## 7. Sleep Quality Scoring

`compute_metrics()` in `app/services/scoring.py` calculates clinical sleep metrics from the predicted stage sequence:

| Metric | Calculation |
|--------|------------|
| **Total Sleep Time (TST)** | All non-Wake epochs × 30s |
| **Sleep Efficiency** | TST ÷ Total Recording Time |
| **Sleep Latency** | Time from recording start to first non-Wake epoch |
| **WASO** | Wake After Sleep Onset — Wake epochs occurring after first sleep epoch |
| **% N3** | Deep sleep epochs ÷ total sleep epochs |
| **% REM** | REM epochs ÷ total sleep epochs |
| **Fragmentation Index** | Number of stage transitions ÷ total epochs |

### Quality Score Formula (0–100)

Each sub-metric is normalised against AASM clinical thresholds, then combined:

```
Score = 0.30 × sleep_efficiency_score
      + 0.20 × waso_score
      + 0.15 × latency_score
      + 0.20 × (n3_score + rem_score) / 2
      + 0.15 × fragmentation_score
```

Example thresholds used for normalisation:

| Sub-score | "Good" threshold | "Bad" threshold |
|-----------|-----------------|-----------------|
| Sleep efficiency | ≥ 90% | < 70% |
| WASO | 0 min | ≥ 90 min |
| Sleep latency | 0 min | ≥ 45 min |
| N3 % | ≥ 20% | 0% |
| REM % | ≥ 22% | 0% |

---

## 8. Regression Model (Quick Score)

### Purpose

`/predict/quick-score` endpoint returns a sleep quality score estimate in ~1 second, without running the full CNN+LSTM (which takes minutes for overnight recordings).

### How It Works

```
EDF file upload
      ↓
MNE reads signal → resample to 100 Hz
      ↓
bandpass_filter()
      ↓
extract_spectral_features()  ← 15 features via Welch PSD
      ↓
reg_model.predict(features)  ← RandomForest
      ↓
QuickScoreResponse {predicted_score: 72.3}
```

### 15 Spectral Features Used

```python
# Absolute band powers (Welch PSD)
power_delta, power_theta, power_alpha, power_beta

# Relative band powers (each ÷ total power)
rel_delta, rel_theta, rel_alpha, rel_beta

# Band ratios
delta/beta       # high ratio → deep sleep dominance
theta/alpha      # high ratio → drowsiness / light sleep

# Time-domain statistics
mean, std, skewness, kurtosis, RMS
```

### Training (`gen_dataset.py`)

1. Run CNN+LSTM on 102 PhysioNet recordings
2. Compute quality score per recording via `compute_metrics()`
3. Extract 15 spectral features from raw EEG
4. Save to `regression_dataset.csv` (89 usable samples)
5. Train RandomForest, evaluate by MAE on held-out split
6. Save best model as `regression_model.pkl`

---

## 9. AI Explanation — Claude Haiku

### Endpoint: `POST /explain`

Takes the `Metrics` object from a prediction, returns a plain-language explanation and 3 actionable tips.

### How It Works

```
Metrics object (sleep_efficiency, waso_min, pct_n3, etc.)
         ↓
Assign quality label (Great / Good / Fair / Poor)
         ↓
if ANTHROPIC_API_KEY set:
    Build structured prompt → Claude Haiku → parse JSON response
else:
    rule_based_explain()  ← threshold-based fallback
         ↓
ExplainResponse {explanation: "...", tips: [...], ai_available: bool}
```

### Prompt Design

The prompt provides all clinical metrics in structured form and asks Claude to return strict JSON:

```
You are a sleep health assistant. Results:
- Sleep Quality Score: 41.6/100 (Fair)
- Sleep Efficiency: 31.7% (normal: >85%)
- WASO: 420 min (normal: <30 min)
...

Return JSON only: {"explanation": "...", "tips": ["...", "...", "..."]}
```

Supports `"language": "hindi"` — instructs Claude to respond in Devanagari script.

### Rule-Based Fallback

When `ANTHROPIC_API_KEY` is not set (or the API call fails), `rule_based_explain()` applies hardcoded threshold logic:

- `sleep_efficiency < 75%` → explains low efficiency
- `waso_min > 30` → explains frequent waking
- `pct_n3 < 10%` → explains low deep sleep
- `pct_rem < 15%` → explains low REM
- Always returns exactly 3 tips from a prioritised list

This makes the `/explain` endpoint fully functional with no paid services required.

---

## 10. Async Job Queue

### Why Async?

Large EDF files (8–24 hour recordings) take 5–20 minutes to process through CNN+LSTM. Railway (and most cloud proxies) have a **30-second connection timeout** — a synchronous request would drop with a 502 error.

### Solution: Job Queue Pattern

```
Client                     Server
  │                           │
  │── POST /predict/edf ────► │  validates file, creates job_id
  │◄── {job_id: "abc123"} ─── │  returns immediately (< 1 sec)
  │                           │  [background thread processes EDF]
  │── GET /jobs/abc123 ─────► │
  │◄── {status: "processing"} │  client polls every few seconds
  │── GET /jobs/abc123 ─────► │
  │◄── {status: "done", ...}  │  full result returned
```

### Implementation (`app/services/jobs.py`)

```python
# In-memory store
_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()   # thread-safe access

# Job states: "processing" → "done" | "failed"
# Each job stores: {status, result, error, created_at}
```

### Job TTL & Cleanup

A daemon thread runs every 10 minutes and evicts jobs older than 1 hour:

```python
JOB_TTL_SECONDS = 3600

def _cleanup_loop():
    while True:
        time.sleep(600)
        cutoff = time.monotonic() - JOB_TTL_SECONDS
        with _jobs_lock:
            expired = [id for id, j in _jobs.items() if j["created_at"] < cutoff]
            for id in expired:
                del _jobs[id]

threading.Thread(target=_cleanup_loop, daemon=True, name="job-cleanup").start()
```

This prevents unbounded memory growth on long-running deployments.

---

## 11. Structured Logging

`app/logging_config.py` installs a JSON formatter on the root logger at startup. Every log call produces one JSON object per line:

```json
{"ts": "2026-04-05T10:30:00Z", "level": "INFO", "logger": "app.services.jobs",
 "message": "EDF job complete", "job_id": "abc-123", "epochs": 960, "quality_score": 72.3}
```

Extra fields passed via `extra={...}` are automatically merged into the JSON payload. Stack traces on errors are included as a list of strings under `"exc"`.

The HTTP middleware in `app/main.py` logs every request and response:

```json
{"ts": "...", "level": "INFO", "message": "Request",  "method": "POST", "path": "/predict/edf"}
{"ts": "...", "level": "INFO", "message": "Response", "method": "POST", "path": "/predict/edf", "status": 200}
```

Noisy third-party loggers (`mne`, `uvicorn.access`) are quietened to WARNING level.

---

## 12. Model Evaluation Scripts

### `evaluate_model.py` — Standard Evaluation

Runs the CNN+LSTM on a folder of EDF files and outputs four artefacts:

```bash
python evaluate_model.py \
    --data_dir  /path/to/sleep-cassette/ \
    --model_path inference/sleep_best_ckpt.keras \
    --out_dir   evaluation/
```

**Outputs:**

| File | Contents |
|------|---------|
| `confusion_matrix.png` | Normalised heatmap (Wake/N1/N2/N3/REM) |
| `per_class_metrics.csv` | Precision, Recall, F1, Support per stage |
| `classification_report.txt` | sklearn text report + overall accuracy + Cohen's Kappa |
| `summary.json` | Machine-readable summary of all metrics |

### `cross_dataset_eval.py` — Generalisation Study

Evaluates the model on both the Telemetry split (training distribution) and the Cassette split (held-out subjects) and measures the performance gap between them:

```bash
python cross_dataset_eval.py \
    --telemetry_dir /path/to/sleep-telemetry/ \
    --cassette_dir  /path/to/sleep-cassette/ \
    --model_path    inference/sleep_best_ckpt.keras \
    --out_dir       evaluation/cross_dataset/
```

**Outputs:**

| File | Contents |
|------|---------|
| `telemetry_summary.json` | Metrics on training-distribution subjects |
| `cassette_summary.json` | Metrics on held-out subjects |
| `cross_dataset_report.txt` | Side-by-side comparison table |
| `generalisation_gap.json` | Accuracy/F1/Kappa delta between splits |

The generalisation gap quantifies how much performance degrades when the model encounters subjects from a different recording setup than it was trained on.

---

## 13. Docker & Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Layer 1: dependencies (cached if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 2: app code (changes frequently)
COPY api.py .
COPY app/ app/
COPY static/ static/
COPY inference/sleep_best_ckpt.keras inference/sleep_best_ckpt.keras
COPY regression_model.pkl .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300"]
```

**Why layer order matters:** `requirements.txt` → `pip install` → app code. If only `api.py` changes, Docker reuses the cached pip install layer, making rebuilds fast.

**Why `${PORT:-8000}`:** Railway dynamically assigns a port via `$PORT` env var. `:-8000` is the fallback for local development.

### `railway.json`

```json
{
  "build":  {"builder": "DOCKERFILE"},
  "deploy": {
    "healthcheckPath":    "/health",
    "healthcheckTimeout": 300,
    "restartPolicyType":  "ON_FAILURE"
  }
}
```

The 300-second health check timeout allows time for the CNN+LSTM model to finish loading on startup before Railway declares the deployment unhealthy.

### Deployment Flow

```
git push origin main
      ↓
Railway detects push (GitHub webhook)
      ↓
Builds Docker image from Dockerfile
      ↓
Starts container with $PORT env var
      ↓
Health check: GET /health (waits up to 300s)
      ↓
Traffic routed to new container
```

---

## 14. Problems Faced & Solutions

### Problem 1: Keras 3 Model Load Error
**Error:** `unrecognized keyword argument: quantization_config`

**Cause:** Model was saved with an intermediate Keras 3 version that wrote `quantization_config` into Dense layer configs. Newer Keras removed it.

**Fix:** Monkey-patch `Dense.from_config` to drop the key before reconstruction.

---

### Problem 2: Railway 502 Gateway Timeout
**Error:** HTTP 502 on EDF uploads for large overnight recordings.

**Cause:** Railway's reverse proxy has a ~30-second timeout. CNN+LSTM inference on 8–24 hour recordings takes 5–20 minutes.

**Fix:** Async job queue pattern — `POST /predict/edf` returns a `job_id` immediately; client polls `GET /jobs/{id}`.

---

### Problem 3: Railway Health Check Failure
**Error:** Service kept restarting on deployment.

**Cause:** Hardcoded port `8000` in the Dockerfile CMD; Railway assigns a dynamic `$PORT`.

**Fix:**
```dockerfile
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

---

### Problem 4: EDF Upload 422 Error
**Error:** Valid 22-hour EDF files rejected with "Signal too long".

**Cause:** `MAX_SAMPLES` was set for 10 hours, not 24.

**Fix:**
```python
MAX_SAMPLES = TARGET_FS * 3600 * 24   # 24 hours
```

---

### Problem 5: numpy Version Mismatch in Pickle
**Error:** `numpy.random.MT19937 not known` when loading `regression_model.pkl`.

**Cause:** Regression model pickled locally with numpy 2.x. Docker container uses numpy 1.26.4 (required by tensorflow-cpu==2.17.0). Pickle format is not numpy-version-agnostic.

**Fix:** Retrain the regression model inside the Docker container so the pickle format matches the runtime environment.

---

### Problem 6: `np.trapz` Removed in numpy 2.0
**Error:** `AttributeError: module 'numpy' has no attribute 'trapz'`

**Cause:** `np.trapz` was renamed to `np.trapezoid` in numpy 2.0. Local environment had numpy 2.x, Docker had 1.26.4.

**Fix:**
```python
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
```

---

### Problem 7: `python-multipart` Missing
**Error:** `422 Unprocessable Entity` on all file upload endpoints.

**Cause:** FastAPI requires `python-multipart` for `UploadFile` support but does not declare it as a hard dependency.

**Fix:** Added `python-multipart>=0.0.9` to `requirements.txt`.

---

### Problem 8: Integration Test — Lifespan Bypass
**Error:** Tests failed because `lifespan` tried to call `keras.models.load_model` with no model file present in CI.

**Cause:** `TestClient` triggers the FastAPI lifespan, which tries to load the real Keras model.

**Fix:** Override the lifespan with a no-op before creating the TestClient, and set model handles directly:

```python
state.model = _mock_cnn
state.reg_model = _mock_reg
app.router.lifespan_context = _noop_lifespan
with TestClient(app) as c:
    yield c
```

---

### Problem 9: MNE Stub Collision Across Test Files
**Error:** Tests in `test_integration.py` failed when run together with `test_api_unit.py`.

**Cause:** Both files register an MNE stub via `sys.modules.setdefault(...)`. The first file to run "wins" — the stub from the second file is silently ignored.

**Fix:** Instead of `setdefault`, directly assign to the already-registered module:
```python
sys.modules["mne.io"].read_raw_edf = _fake_read_raw_edf
sys.modules["mne"].io = sys.modules["mne.io"]
```

---

## 15. Project Structure

```
sleep-staging-api/
│
├── api.py                          # Thin entry point — imports app.main.app
│                                   # Backward-compat re-exports for tests
│
├── Dockerfile                      # Container definition
├── requirements.txt                # Python dependencies
├── railway.json                    # Railway deployment config
│
├── app/                            # FastAPI application package
│   ├── main.py                     # Keras patch, lifespan, HTTP middleware
│   ├── config.py                   # All constants and env var reads
│   ├── state.py                    # Global model handles (set at startup)
│   ├── auth.py                     # X-API-Key dependency (no-op if not configured)
│   ├── logging_config.py           # JSON structured logging setup
│   │
│   ├── models/
│   │   └── schemas.py              # Pydantic request/response models
│   │
│   ├── routes/
│   │   ├── predict.py              # /predict, /predict/edf, /predict/quick-score, /jobs/{id}
│   │   └── explain.py              # /explain (Claude Haiku + rule-based fallback)
│   │
│   └── services/
│       ├── signal.py               # validate_signal, bandpass_filter, extract_spectral_features
│       ├── inference.py            # run_inference() — shared CNN+LSTM pipeline
│       ├── scoring.py              # compute_metrics, rule_based_explain
│       └── jobs.py                 # Async job queue, EDF loading, TTL cleanup thread
│
├── inference/
│   └── sleep_best_ckpt.keras       # Trained CNN+LSTM model (94.59% accuracy)
│
├── src/                            # Offline training code (not used by the API)
│   ├── dl_data.py                  # Dataset loading and preprocessing
│   ├── dl_model.py                 # CNN+LSTM architecture definition
│   ├── dl_main.py                  # Training loop
│   └── modal_train.py              # Cloud GPU training (Modal)
│
├── gen_dataset.py                  # Generate regression_dataset.csv from CNN+LSTM
├── regression_dataset.csv          # 89 training samples for regression model
├── regression_model.pkl            # Trained RandomForest model
│
├── evaluate_model.py               # Confusion matrix + per-class metrics
├── cross_dataset_eval.py           # Generalisation gap: Telemetry vs Cassette split
│
├── static/
│   └── index.html                  # Frontend dashboard
│
└── tests/
    ├── test_api_unit.py            # 13 unit tests (signal, scoring, validation)
    └── test_integration.py         # 10 integration tests (all API endpoints)
```

---

## 16. How to Run Locally

### Option A: Python directly (recommended for development)

```bash
# Install dependencies
pip install -r requirements.txt

# Start server with auto-reload
uvicorn api:app --reload
```

Server runs at `http://127.0.0.1:8000`

| URL | What it shows |
|-----|--------------|
| `http://127.0.0.1:8000/docs` | Interactive Swagger UI — test all endpoints |
| `http://127.0.0.1:8000/health` | Model load status |
| `http://127.0.0.1:8000/redoc` | Alternative API docs |

### Option B: Docker

```bash
# Build
docker build -t sleep-staging-api .

# Run
docker run -p 8000:8000 sleep-staging-api
```

### Run Tests

```bash
pip install pytest python-multipart anthropic
pytest tests/ -v
```

Expected: **23 tests passed** (13 unit + 10 integration)

### Run Model Evaluation

```bash
# Standard evaluation on Cassette split
python evaluate_model.py \
    --data_dir /path/to/sleep-cassette/ \
    --out_dir  evaluation/

# Cross-dataset generalisation study
python cross_dataset_eval.py \
    --cassette_dir  /path/to/sleep-cassette/ \
    --telemetry_dir /path/to/sleep-telemetry/ \
    --out_dir       evaluation/cross_dataset/
```

---

## 17. Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `PORT` | No | 8000 | Server port (Railway sets this automatically) |
| `API_KEY` | No | `""` | API key authentication (disabled when empty) |
| `ANTHROPIC_API_KEY` | No | `""` | Claude Haiku for AI explanations (rule-based fallback when empty) |
| `MODEL_PATH` | No | `inference/sleep_best_ckpt.keras` | Path to CNN+LSTM model |
| `REG_MODEL_PATH` | No | `regression_model.pkl` | Path to regression model |
| `DATA_DIR` | No | `data/sleep-cassette` | Used by `gen_dataset.py` only |
| `OUT_CSV` | No | `regression_dataset.csv` | Used by `gen_dataset.py` only |

The API is fully functional with **no env vars set** — it just uses rule-based explanations instead of Claude, and skips `/predict/quick-score` if the regression model file is missing.

---

## 18. API Reference

### GET /health

```json
{
  "status": "ok",
  "cnn_lstm_loaded":      true,
  "regression_loaded":    true,
  "ai_explain_available": false
}
```

---

### POST /predict

Synchronous. JSON body with raw EEG samples. Best for short recordings or testing.

**Request:**
```json
{
  "eeg_signal": [0.12, -0.34, 0.56, ...],
  "smooth": true
}
```

Minimum length: `SEQ_LEN × SAMPLES_PER_EPOCH = 5 × 3000 = 15000 samples` (150 seconds at 100 Hz).

**Response:** See `PredictResponse` below.

---

### POST /predict/edf

Asynchronous. Upload an EDF file and receive a `job_id`. Poll `/jobs/{job_id}` for results.

**Request:** `multipart/form-data`
- `file`: EDF file (max 500 MB)
- `channel`: EEG channel name (default: `EEG Fpz-Cz`)
- `smooth`: Apply median filter (default: `true`)

**Response:**
```json
{"job_id": "abc123-...", "status": "processing"}
```

---

### GET /jobs/{job_id}

**Response (processing):**
```json
{"status": "processing"}
```

**Response (done):**
```json
{
  "status": "done",
  "result": {
    "epochs": [
      {"epoch": 0, "stage_id": 0, "stage_name": "Wake", "confidence": 0.98},
      {"epoch": 1, "stage_id": 1, "stage_name": "N1",   "confidence": 0.74},
      ...
    ],
    "metrics": {
      "recording_minutes":   450.0,
      "tst_min":             380.0,
      "sleep_latency_min":   12.5,
      "sleep_efficiency":    0.844,
      "waso_min":            28.0,
      "pct_n3":              0.18,
      "pct_rem":             0.21,
      "fragmentation_index": 0.08,
      "confidence_mean":     0.91,
      "sleep_quality_score": 67.4
    }
  }
}
```

**Response (failed):**
```json
{"status": "failed", "error": "Signal is flat — electrode may be disconnected."}
```

---

### POST /predict/quick-score

Fast score using the RandomForest regression model. No per-epoch staging.

**Request:** `multipart/form-data` — EDF file + optional `channel`

**Response:**
```json
{
  "predicted_score": 33.4,
  "score_range":     "0-100",
  "model":           "RandomForest (spectral EEG features)",
  "note":            "Fast estimate. Use /predict/edf for full staging + accurate score."
}
```

---

### POST /explain

AI explanation of sleep results.

**Request:**
```json
{
  "metrics": {
    "recording_minutes":   450.0,
    "tst_min":             380.0,
    "sleep_latency_min":   12.5,
    "sleep_efficiency":    0.844,
    "waso_min":            28.0,
    "pct_n3":              0.18,
    "pct_rem":             0.21,
    "fragmentation_index": 0.08,
    "confidence_mean":     0.91,
    "sleep_quality_score": 67.4
  },
  "language": "english"
}
```

`"language"` accepts `"english"` (default) or `"hindi"` (Devanagari script).

**Response:**
```json
{
  "explanation":   "Your sleep efficiency is good at 84%...",
  "tips": [
    "Maintain a consistent sleep schedule.",
    "Avoid screens 1 hour before bed.",
    "Exercise regularly to increase deep sleep."
  ],
  "ai_available": true
}
```

`"ai_available": false` means the rule-based fallback was used.

---

### Sleep Stage Labels

| ID | Name | Clinical Description |
|----|------|---------------------|
| 0  | Wake | Awake |
| 1  | N1   | Light sleep onset — easily disturbed |
| 2  | N2   | Light sleep — sleep spindles and K-complexes |
| 3  | N3   | Deep slow-wave sleep — hardest to wake from |
| 4  | REM  | Rapid Eye Movement — dreaming, memory consolidation |

---

## License

Model trained on the PhysioNet Sleep-EDF Expanded dataset (open access, PhysioNet Restricted Health Data License 1.5.0).
