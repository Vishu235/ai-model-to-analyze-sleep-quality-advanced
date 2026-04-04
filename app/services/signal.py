import numpy as np
from fastapi import HTTPException
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis as sp_kurtosis

from app.config import (
    TARGET_FS, FILTER_ORDER, LOWCUT, HIGHCUT, MAX_SAMPLES,
)

# numpy 2.0 compatibility
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def validate_signal(signal: np.ndarray) -> None:
    """Raise HTTPException if the signal is unusable."""
    if len(signal) > MAX_SAMPLES:
        raise HTTPException(
            status_code=422,
            detail=f"Signal too long: {len(signal) / TARGET_FS / 3600:.1f}h. Maximum is 24 hours.",
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


def bandpass_filter(signal: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Butterworth bandpass filter 0.5–30 Hz (matches training pipeline)."""
    nyq = 0.5 * fs
    b, a = butter(FILTER_ORDER, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def extract_spectral_features(signal: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Extract 15 spectral + statistical features used by the regression model."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs * 4))
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
    bp = {}
    for b_name, (lo, hi) in bands.items():
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        bp[b_name] = float(_trapz(psd[idx], freqs[idx]))
    total = sum(bp.values()) + 1e-10
    feats = (
        [bp["delta"], bp["theta"], bp["alpha"], bp["beta"]]
        + [bp[b] / total for b in ["delta", "theta", "alpha", "beta"]]
        + [bp["delta"] / (bp["beta"] + 1e-10), bp["theta"] / (bp["alpha"] + 1e-10)]
        + [
            float(signal.mean()), float(signal.std()),
            float(skew(signal)), float(sp_kurtosis(signal)),
            float(np.sqrt(np.mean(signal ** 2))),
        ]
    )
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Private aliases kept for backward compatibility with existing tests
# ---------------------------------------------------------------------------
_validate_signal          = validate_signal
_bandpass_filter          = bandpass_filter
_extract_spectral_features = extract_spectral_features