# Sleep Staging AI — Complete Project Documentation

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Railway](https://img.shields.io/badge/Deployed-Railway-purple)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-black)

**Live URL:** https://sleep-staging-api-production.up.railway.app

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Model Architecture](#2-model-architecture)
3. [Dataset](#3-dataset)
4. [API Design](#4-api-design)
5. [Regression Model](#5-regression-model)
6. [AI Explanation (RAG)](#6-ai-explanation-rag)
7. [Docker & Deployment](#7-docker--deployment)
8. [DevOps — CI/CD Pipeline](#8-devops--cicd-pipeline)
9. [MLOps — Auto Retrain Pipeline](#9-mlops--auto-retrain-pipeline)
10. [Problems Faced & Solutions](#10-problems-faced--solutions)
11. [Project Structure](#11-project-structure)
12. [How to Run Locally](#12-how-to-run-locally)
13. [Environment Variables](#13-environment-variables)
14. [API Reference](#14-api-reference)

---

## 1. Project Overview

Is project ka goal hai ek EEG (brain signal) file upload karo aur AI automatically bataye:
- Raat bhar kaunse sleep stages the (Wake / N1 / N2 / N3 / REM)
- Sleep quality score (0–100)
- Hypnogram chart (visual sleep pattern)
- Plain language explanation with health tips

### Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| ML Model | CNN + LSTM (Keras/TF 2.17) | Sleep staging needs both local features (CNN) and time context (LSTM) |
| API | FastAPI | Fast, async, auto docs at `/docs` |
| Regression | scikit-learn RandomForest | Quick sleep score from spectral features |
| AI Explanation | Claude Haiku (Anthropic API) | Plain language explanation of results |
| Container | Docker | Consistent environment everywhere |
| Hosting | Railway | Simple git-push deploy |
| CI/CD | GitHub Actions | Auto test + build on every push |

---

## 2. Model Architecture

### Why CNN + LSTM?

Sleep staging ek sequential problem hai. Sirf ek epoch (30 seconds) dekhne se stage predict karna mushkil hai — doctor bhi pehle wale aur baad wale epochs dekhkar decide karta hai.

```
Input: 5 consecutive EEG epochs → [epoch1, epoch2, epoch3, epoch4, epoch5]
                                           ↓
                              TimeDistributed CNN
                    (each epoch processed independently)
                                           ↓
                              LSTM layer
                    (learns: N2 usually follows N1, etc.)
                                           ↓
                              Dense + Softmax
                                           ↓
                    Output: Stage for last epoch (Wake/N1/N2/N3/REM)
```

### Input Shape
```
(batch_size, 5, 3000, 1)
     ↑       ↑   ↑    ↑
  samples  seq  samples  channel
          len  per epoch
          (5)  (30s×100Hz)
```

### CNN Feature Extraction
- **Small kernels** — high frequency features (spindles, K-complexes)
- **Large kernels** — slow waves (delta waves in N3)
- **Pooling** — dimensionality reduction

### LSTM Context
- 5-epoch sequence = 2.5 minutes of context
- Learns sleep cycle transitions: Wake→N1→N2→N3→REM→N2→...

### Output
- 5 classes: Wake(0), N1(1), N2(2), N3(3), REM(4)
- Confidence score per epoch
- **Accuracy: 94.59%** on PhysioNet Sleep-EDF test set

### Post-Processing
```python
# Median filter (size=5) smooths out single-epoch noise
stages = median_filter(stages, size=5)
```

---

## 3. Dataset

**PhysioNet Sleep-EDF Expanded** — 153 whole-night PSG recordings

- **EEG Channel:** Fpz-Cz (single channel)
- **Sampling Rate:** 100 Hz
- **Epoch Length:** 30 seconds = 3000 samples
- **Labels:** W, N1, N2, N3/N4 (merged → N3), REM

### Preprocessing Pipeline
```
Raw EEG signal
    ↓
Bandpass filter: 0.5–30 Hz (Butterworth order 4)
    ↓
Z-score normalize per epoch: (x - mean) / std
    ↓
Sliding window sequences: (n_epochs - 5 + 1) sequences
    ↓
CNN+LSTM prediction
```

### Why Bandpass Filter?
- Below 0.5 Hz: DC drift, movement artifacts → remove
- Above 30 Hz: EMG noise, power line artifacts → remove
- 0.5–30 Hz: All clinically relevant sleep waves (delta, theta, alpha, sigma, beta)

### Why Z-score Normalization?
- Different EEG recordings have different amplitudes (electrode placement, skin resistance)
- Z-score makes each epoch amplitude-independent
- Model sees relative patterns, not absolute voltages

---

## 4. API Design

### Why FastAPI?
- Automatic OpenAPI docs at `/docs`
- Async support (important for file uploads)
- Pydantic validation (automatic input checking)
- Much faster than Flask for concurrent requests

### Endpoints

```
GET  /           → Frontend dashboard (index.html)
GET  /health     → Model load status
POST /predict    → JSON EEG array → sleep stages
POST /predict/edf → EDF file upload → job_id (async)
GET  /jobs/{id}  → Poll for EDF processing result
POST /predict/quick-score → Fast regression score
POST /explain    → Claude AI explains results
```

### Why Async EDF Processing?

**Problem:** Large EDF files (8-24 hours) take 5-20 minutes to process with CNN+LSTM.
Railway (and most cloud platforms) have a **30-second proxy timeout** — the connection would drop before processing finishes → 502 error.

**Solution: Job Queue Pattern**
```
Client                     Server
  |                           |
  |-- POST /predict/edf ----→ |  (returns job_id in <1 sec)
  |← {job_id: "abc123"} ----- |
  |                           |  (background thread processes EDF)
  |-- GET /jobs/abc123 -----→ |
  |← {status: "processing"} - |  (poll every 3 seconds)
  |-- GET /jobs/abc123 -----→ |
  |← {status: "processing"} - |
  ...
  |-- GET /jobs/abc123 -----→ |
  |← {status: "done", result} |  (frontend renders results)
```

### API Key Authentication

```python
# Optional — only enforced when API_KEY env var is set
# Header: X-API-Key: your-key
```

Agar `API_KEY` env var set nahi hai → auth disabled (open API).
Production mein Railway pe set karo for security.

### Input Validation
```python
MAX_SAMPLES = 100 * 3600 * 24   # 24 hours max
MAX_EDF_MB  = 500                # 500 MB file size limit

# Checks:
# - Signal too long (> 24h)
# - NaN or Inf values
# - Flat signal (std < 1e-6) → disconnected electrode
# - File too large (> 500 MB)
# - Wrong file format (not .edf)
```

---

## 5. Regression Model

### Purpose
`/predict/quick-score` endpoint — fast sleep quality score **without running CNN+LSTM**.
Returns in ~2 seconds vs 5-20 minutes for full staging.

### Features Used (15 spectral + statistical)
```python
# Frequency band power (absolute)
power_delta, power_theta, power_alpha, power_beta

# Relative band power
rel_delta, rel_theta, rel_alpha, rel_beta

# Band ratios
ratio_delta_beta   # high delta/beta = deep sleep
ratio_theta_alpha  # theta dominance = light sleep / drowsiness

# Time domain statistics
mean, std, skewness, kurtosis, rms
```

### Training Data Generation (`gen_dataset.py`)
```
102 PhysioNet recordings
    ↓
CNN+LSTM → sleep stages per recording
    ↓
quality_score() function → ground truth score
    ↓
Extract 15 spectral features from raw EEG
    ↓
Save to regression_dataset.csv
    ↓
Train RandomForest + GradientBoosting
    ↓
Pick best by MAE on test split
    ↓
Save regression_model.pkl
```

89 usable samples (13 skipped — no N3 annotations).

### Why Trained Inside Docker?
**Problem:** Locally trained model used numpy 2.x format. Docker container has numpy 1.26.4 (required by tensorflow-cpu==2.17.0).

**Solution:** Retrain model inside the Docker container so pickle format matches the runtime environment.

```bash
docker run --rm -v dataset.csv:/app/... sleep-staging-api python3 -c "...train..."
```

---

## 6. AI Explanation (RAG)

### What is RAG here?
RAG = Retrieval Augmented Generation. Yahan "retrieval" matlab hai CNN+LSTM ke results (metrics) ko Claude ke context mein dena.

```
Sleep metrics (from CNN+LSTM)
    ↓
Structured prompt → Claude Haiku
    ↓
Plain English/Hindi explanation + 3 actionable tips
```

### Why Claude Haiku?
- Fastest + cheapest Claude model
- Sufficient for structured JSON output
- ~0.5 seconds response time

### Prompt Design
```
You are a sleep health assistant. Results:
- Sleep Quality Score: 41.6/100 (Fair)
- Sleep Efficiency: 31.7% (normal: >85%)
- WASO: 420 min (normal: <30 min)
...

Respond in English/Hindi.
Return JSON: {"explanation": "...", "tips": ["...", "...", "..."]}
```

### Fallback (No API Key)
Agar `ANTHROPIC_API_KEY` set nahi hai → rule-based explanation:
- Low efficiency → specific message
- High WASO → specific message
- Low N3 → specific message
- Always returns exactly 3 tips

---

## 7. Docker & Deployment

### Dockerfile Design

```dockerfile
FROM python:3.10-slim          # slim = smaller image, fewer vulnerabilities

WORKDIR /app

# Layer 1: dependencies (cached if requirements.txt unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 2: app code (changes frequently, separate layer)
COPY api.py .
COPY static/ static/
COPY inference/sleep_best_ckpt.keras inference/sleep_best_ckpt.keras
COPY regression_model.pkl .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 300"]
```

**Why layer order matters:**
- `requirements.txt` copy → `pip install` → app code
- Agar sirf `api.py` change hoa → pip install layer cached → fast rebuild

**Why `${PORT:-8000}`:**
Railway dynamically assigns a port via `$PORT` env var. `:-8000` is fallback for local.

### Railway Deployment

```
git push origin main
    ↓
Railway detects push (GitHub webhook)
    ↓
Builds Docker image from Dockerfile
    ↓
Runs container with $PORT env var
    ↓
Health check: GET /health
    ↓
Traffic routed to new container
```

### railway.json
```json
{
  "build": {"builder": "DOCKERFILE"},
  "deploy": {
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300,
    "restartPolicyType": "ON_FAILURE"
  }
}
```

---

## 8. DevOps — CI/CD Pipeline

**File:** `.github/workflows/ci.yml`

### Why CI/CD?
Bina CI/CD ke:
- Code push karo → manually test karo → manually deploy karo
- Kisi step mein bhool ho sakti hai
- Broken code production mein ja sakta hai

CI/CD ke saath:
- Code push karo → automatically test → automatically deploy
- Broken code block ho jaata hai before deploy

### Pipeline

```
git push to main
    ↓
Job 1: Unit Tests (ubuntu-latest, Python 3.10)
    - pip install test dependencies
    - pytest tests/test_api_unit.py
    - 13 tests: bandpass filter, validation, spectral features, metrics, explain
    ↓ (only if tests pass)
Job 2: Docker Build Check
    - Build Docker image
    - GHA layer caching (faster builds)
    ↓ (only if build passes)
Railway auto-deploys
```

### Test Coverage (13 tests)

| Test | What it checks |
|------|---------------|
| `test_bandpass_filter_shape` | Output shape same as input |
| `test_bandpass_filter_reduces_dc` | DC offset removed |
| `test_validate_signal_ok` | Valid signal passes |
| `test_validate_signal_flat` | Flat signal → 422 error |
| `test_validate_signal_nan` | NaN values → 422 error |
| `test_validate_signal_too_long` | >24h → 422 error |
| `test_spectral_features_shape` | Returns 15 features |
| `test_spectral_features_finite` | No NaN in features |
| `test_compute_metrics_score_range` | Score in 0-100 |
| `test_compute_metrics_all_wake` | All wake → 0% efficiency |
| `test_compute_metrics_perfect_sleep` | Good sleep → score >50 |
| `test_rule_based_explain_returns_three_tips` | Always 3 tips |
| `test_rule_based_explain_poor_sleep` | Handles edge case |

### Why Mock keras/tensorflow in Tests?
Tests ko model files ki zaroorat nahi honi chahiye — sirf business logic test karo.
CI mein `tensorflow` install karna 10+ minutes aur 2GB disk lega.
Mock se tests 3 seconds mein complete hote hain.

---

## 9. MLOps — Auto Retrain Pipeline

**File:** `.github/workflows/mlops_retrain.yml`

### Trigger Conditions
```yaml
on:
  push:
    paths: ["regression_dataset.csv"]   # Auto: naya data push hone pe
  workflow_dispatch:                     # Manual: GitHub UI se
    inputs:
      dataset_url: ...                   # Optional: external dataset URL
```

### Pipeline Steps
```
New regression_dataset.csv pushed
    ↓
Install: numpy, scikit-learn, pandas, joblib
    ↓
Load dataset → train RF + GB models
    ↓
Compare MAE on test split → pick best
    ↓
Print metrics (MAE, R², sample count)
    ↓
Commit regression_model.pkl back to repo
    ↓
Railway auto-deploys new model [skip ci]
```

### Why [skip ci]?
Model commit ke baad CI trigger na ho (infinite loop avoid karna).

---

## 10. Problems Faced & Solutions

### Problem 1: Keras 3 Model Load Error
**Error:** `unrecognized keyword argument: quantization_config`

**Cause:** Model was saved with an intermediate Keras 3 version that added `quantization_config` to Dense layer config. Newer Keras 3.6+ removed it.

**Fix:**
```python
_orig = keras.layers.Dense.from_config.__func__
@classmethod
def _patched(cls, config):
    config.pop("quantization_config", None)
    return _orig(cls, config)
keras.layers.Dense.from_config = _patched
```

### Problem 2: Railway Health Check Failure
**Error:** Service kept restarting

**Cause:** CMD had hardcoded port 8000, Railway assigns dynamic `$PORT`

**Fix:**
```dockerfile
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

### Problem 3: EDF Upload "Server Error"
**Error:** 422 Unprocessable Entity

**Cause:** PhysioNet files are 22-24 hours. `MAX_SAMPLES` was set for 10 hours.

**Fix:** Increased to 24 hours:
```python
MAX_SAMPLES = TARGET_FS * 3600 * 24
```

### Problem 4: numpy Version Mismatch
**Error:** `numpy.random.MT19937 not known`

**Cause:** Regression model pickled locally with numpy 2.x. Docker container uses numpy 1.26.4 (required by tensorflow 2.17).

**Fix:** Retrain model inside the Docker container:
```bash
docker run --rm -v dataset.csv:/app/... sleep-staging-api python3 -c "...train..."
```

### Problem 5: 502 Gateway Timeout
**Error:** HTTP 502 on EDF upload

**Cause:** Railway proxy has ~30 second timeout. CNN+LSTM inference on large files takes 5-20 minutes.

**Fix:** Async job queue pattern:
- POST `/predict/edf` → returns `job_id` instantly
- Background thread processes EDF
- Client polls `GET /jobs/{job_id}` every 3 seconds

### Problem 6: np.trapz vs np.trapezoid
**Error:** `AttributeError: module 'numpy' has no attribute 'trapz'` (numpy 2.0+)

**Cause:** `np.trapz` renamed to `np.trapezoid` in numpy 2.0. Docker has 1.26.4, local has 2.x.

**Fix:**
```python
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
```

### Problem 7: GitHub Push Rejected (workflow scope)
**Error:** `refusing to allow a Personal Access Token to create or update workflow`

**Cause:** GitHub PAT token didn't have `workflow` scope for pushing `.github/workflows/` files.

**Fix:** Created new PAT with both `repo` + `workflow` scopes.

---

## 11. Project Structure

```
sleep-staging-api/
├── api.py                          # FastAPI application (main file)
├── Dockerfile                      # Container definition
├── requirements.txt                # Python dependencies
├── railway.json                    # Railway deployment config
├── gen_dataset.py                  # Generate regression training data
├── regression_dataset.csv          # 89 training samples
├── regression_model.pkl            # Trained RandomForest model
│
├── inference/
│   └── sleep_best_ckpt.keras       # CNN+LSTM model (94.59% accuracy)
│
├── static/
│   └── index.html                  # Frontend dashboard
│
├── tests/
│   └── test_api_unit.py            # 13 unit tests
│
└── .github/
    └── workflows/
        ├── ci.yml                  # CI: test + docker build
        └── mlops_retrain.yml       # MLOps: auto retrain on new data
```

---

## 12. How to Run Locally

### Option A: Docker (Recommended)
```bash
# Build
docker build -t sleep-staging-api .

# Run
docker run -p 8000:8000 sleep-staging-api

# Open browser
# http://localhost:8000
```

### Option B: Python directly
```bash
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Run Tests
```bash
pip install pytest python-multipart anthropic
pytest tests/test_api_unit.py -v
```

---

## 13. Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `PORT` | No | 8000 | Server port (Railway sets this) |
| `API_KEY` | No | "" | API authentication (disabled if empty) |
| `ANTHROPIC_API_KEY` | No | "" | Claude AI explanations (rule-based fallback if empty) |
| `MODEL_PATH` | No | `inference/sleep_best_ckpt.keras` | CNN+LSTM model path |
| `REG_MODEL_PATH` | No | `regression_model.pkl` | Regression model path |

---

## 14. API Reference

### GET /health
```json
{
  "status": "ok",
  "cnn_lstm_loaded": true,
  "regression_loaded": true,
  "ai_explain_available": false
}
```

### POST /predict/edf
Upload EDF file → returns job_id

**Request:** `multipart/form-data`
- `file`: EDF file
- `channel`: EEG channel name (default: `EEG Fpz-Cz`)
- `smooth`: Apply median filter (default: `true`)

**Response:**
```json
{"job_id": "abc123-...", "status": "processing"}
```

### GET /jobs/{job_id}
Poll for result

**Response (done):**
```json
{
  "status": "done",
  "result": {
    "epochs": [
      {"epoch": 0, "stage_id": 0, "stage_name": "Wake", "confidence": 0.98},
      ...
    ],
    "metrics": {
      "recording_minutes": 450.0,
      "tst_min": 380.0,
      "sleep_latency_min": 12.5,
      "sleep_efficiency": 0.844,
      "waso_min": 28.0,
      "pct_n3": 0.18,
      "pct_rem": 0.21,
      "fragmentation_index": 0.08,
      "confidence_mean": 0.91,
      "sleep_quality_score": 67.4
    }
  }
}
```

### POST /predict/quick-score
Fast score using regression model (no CNN+LSTM)

**Request:** `multipart/form-data` — EDF file

**Response:**
```json
{
  "predicted_score": 33.4,
  "score_range": "0-100",
  "model": "RandomForest (spectral EEG features)",
  "note": "Fast estimate. Use /predict/edf for full staging + accurate score."
}
```

### POST /explain
AI explanation of sleep results

**Request:**
```json
{
  "metrics": { ...metrics object... },
  "language": "english"
}
```

**Response:**
```json
{
  "explanation": "Your sleep efficiency is low at 31%...",
  "tips": [
    "Maintain a consistent sleep schedule.",
    "Avoid screens 1 hour before bed.",
    "Exercise regularly to increase deep sleep."
  ],
  "ai_available": true
}
```

### Sleep Quality Score Formula
```
Score = 0.30 × sleep_efficiency_score
      + 0.20 × waso_score
      + 0.15 × latency_score
      + 0.20 × (n3_score + rem_score) / 2
      + 0.15 × fragmentation_score
```

### Sleep Stage Labels
| ID | Name | Description |
|----|------|-------------|
| 0 | Wake | Awake |
| 1 | N1 | Light sleep onset |
| 2 | N2 | Light sleep (spindles, K-complexes) |
| 3 | N3 | Deep sleep (slow waves) |
| 4 | REM | REM sleep (dreaming) |

---

## License
Model trained on PhysioNet Sleep-EDF dataset (open access).
