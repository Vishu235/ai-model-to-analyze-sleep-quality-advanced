import uuid
import tempfile
import os

import mne
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.auth import require_api_key
from app import state
from app.config import TARGET_FS, MAX_SAMPLES, MAX_EDF_MB, SAMPLES_PER_EPOCH, SEQ_LEN
from app.models.schemas import (
    PredictRequest, PredictResponse,
    QuickScoreResponse,
    JobResponse, JobStatusResponse,
)
from app.services.signal import validate_signal, bandpass_filter, extract_spectral_features
from app.services.inference import run_inference
from app.services import jobs as job_store

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _=Depends(require_api_key)):
    signal = np.array(req.eeg_signal, dtype=np.float32)
    validate_signal(signal)

    min_samples = SEQ_LEN * SAMPLES_PER_EPOCH
    if len(signal) < min_samples:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Need at least {min_samples} samples "
                f"({SEQ_LEN * 30}s at {TARGET_FS} Hz). Got {len(signal)}."
            ),
        )

    return run_inference(signal, smooth=req.smooth)


@router.post("/predict/quick-score", response_model=QuickScoreResponse)
async def predict_quick_score(
    file: UploadFile = File(...),
    channel: str = "EEG Fpz-Cz",
    _=Depends(require_api_key),
):
    """Fast sleep quality score via regression model. Returns in seconds, not minutes."""
    if state.reg_model is None:
        raise HTTPException(status_code=503, detail="Regression model not loaded.")
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=422, detail="Only .edf files are supported.")

    content = await file.read()
    if len(content) > MAX_EDF_MB * 1024 * 1024:
        raise HTTPException(
            status_code=422,
            detail=f"File too large: {len(content)/1024/1024:.1f} MB. Max {MAX_EDF_MB} MB.",
        )

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
                raise HTTPException(
                    status_code=422,
                    detail=f"Channel '{channel}' not found. Available: {raw.ch_names}",
                )
            raw = eeg_raw
            raw.pick_channels([raw.ch_names[0]])
        if raw.info["sfreq"] != TARGET_FS:
            raw.resample(TARGET_FS, verbose=False)
        signal = raw.get_data()[0].astype(np.float32)
    finally:
        os.unlink(tmp_path)

    validate_signal(signal)
    signal = bandpass_filter(signal)
    feats  = extract_spectral_features(signal).reshape(1, -1)
    score  = float(np.clip(state.reg_model.predict(feats)[0], 0, 100))
    return QuickScoreResponse(predicted_score=round(score, 1))


@router.post("/predict/edf", response_model=JobResponse)
async def predict_edf(
    file: UploadFile = File(...),
    channel: str = "EEG Fpz-Cz",
    smooth: bool = True,
    _=Depends(require_api_key),
):
    """Submit EDF for async CNN+LSTM processing. Poll GET /jobs/{job_id} for results."""
    if not file.filename.lower().endswith(".edf"):
        raise HTTPException(status_code=422, detail="Only .edf files are supported.")

    content = await file.read()
    if len(content) > MAX_EDF_MB * 1024 * 1024:
        raise HTTPException(
            status_code=422,
            detail=f"File too large: {len(content)/1024/1024:.1f} MB. Max {MAX_EDF_MB} MB.",
        )

    job_id = str(uuid.uuid4())
    job_store.create_job(job_id)
    job_store.start_edf_job(job_id, content, channel, smooth)
    return JobResponse(job_id=job_id, status="processing")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, _=Depends(require_api_key)):
    """Poll for EDF processing result."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] == "done":
        return JobStatusResponse(status="done", result=job["result"])
    if job["status"] == "failed":
        return JobStatusResponse(status="failed", error=job["error"])
    return JobStatusResponse(status="processing")