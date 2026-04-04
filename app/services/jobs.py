"""Async job queue for long-running EDF inference jobs."""
import threading
import tempfile
import os
from typing import Dict, Any

import mne
import numpy as np

from app.config import TARGET_FS, MAX_SAMPLES, MAX_EDF_MB
from app.services.inference import run_inference

# In-memory job store: job_id -> {status, result, error}
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def create_job(job_id: str) -> None:
    with _jobs_lock:
        _jobs[job_id] = {"status": "processing"}


def get_job(job_id: str):
    with _jobs_lock:
        return _jobs.get(job_id)


def _load_edf_signal(content: bytes, channel: str) -> np.ndarray:
    """Write EDF bytes to a temp file, load with MNE, return float32 signal."""
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
                raise ValueError(
                    f"Channel '{channel}' not found. Available: {raw.ch_names}"
                )
            raw = eeg_raw
            raw.pick_channels([raw.ch_names[0]])
        if raw.info["sfreq"] != TARGET_FS:
            raw.resample(TARGET_FS, verbose=False)
        return raw.get_data()[0].astype(np.float32)
    finally:
        os.unlink(tmp_path)


def _run_edf_job(job_id: str, content: bytes, channel: str, smooth: bool) -> None:
    """Execute inference for one EDF job and write result into _jobs."""
    signal = _load_edf_signal(content, channel)

    if len(signal) > MAX_SAMPLES:
        raise ValueError(f"Signal too long: {len(signal)/TARGET_FS/3600:.1f}h. Max 24h.")
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or Inf values.")
    if signal.std() < 1e-6:
        raise ValueError("Signal is flat — electrode may be disconnected.")

    result = run_inference(signal, smooth=smooth)
    with _jobs_lock:
        _jobs[job_id] = {"status": "done", "result": result.model_dump()}


def start_edf_job(job_id: str, content: bytes, channel: str, smooth: bool) -> None:
    """Spawn a daemon thread to run EDF inference in the background."""
    def _worker():
        try:
            _run_edf_job(job_id, content, channel, smooth)
        except Exception as exc:
            with _jobs_lock:
                _jobs[job_id] = {"status": "failed", "error": str(exc)}

    threading.Thread(target=_worker, daemon=True).start()