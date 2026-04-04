"""Shared CNN+LSTM inference logic used by both /predict and the async EDF job."""
import numpy as np
from scipy.ndimage import median_filter

from app import state
from app.config import SAMPLES_PER_EPOCH, SEQ_LEN, SMOOTH_KERNEL, STAGE_NAMES
from app.models.schemas import EpochResult, PredictResponse
from app.services.signal import bandpass_filter
from app.services.scoring import compute_metrics


def run_inference(signal: np.ndarray, smooth: bool = True) -> PredictResponse:
    """
    Run CNN+LSTM sleep staging on a pre-validated EEG signal.

    Expects signal to already be validated (finite, non-flat, correct length).
    Raises ValueError if the signal is too short for even one sequence.
    """
    n_epochs = len(signal) // SAMPLES_PER_EPOCH
    if n_epochs < SEQ_LEN:
        raise ValueError(
            f"Signal too short. Need ≥{SEQ_LEN * 30}s of EEG. "
            f"Got {len(signal) / 100:.1f}s ({n_epochs} epochs)."
        )

    signal = signal[: n_epochs * SAMPLES_PER_EPOCH]
    signal = bandpass_filter(signal)

    epochs = signal.reshape(n_epochs, SAMPLES_PER_EPOCH)
    mean   = epochs.mean(axis=1, keepdims=True)
    std    = epochs.std(axis=1, keepdims=True) + 1e-8
    epochs = (epochs - mean) / std

    X = np.stack(
        [epochs[i : i + SEQ_LEN] for i in range(n_epochs - SEQ_LEN + 1)], axis=0
    )[..., np.newaxis]

    probs    = state.model.predict(X, verbose=0)
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

    epoch_results = [
        EpochResult(
            epoch      = i,
            stage_id   = stages[i],
            stage_name = STAGE_NAMES[stages[i]],
            confidence = round(confidence[i], 4),
        )
        for i in range(n_epochs)
    ]

    return PredictResponse(
        epochs  = epoch_results,
        metrics = compute_metrics(stages, confidence),
    )