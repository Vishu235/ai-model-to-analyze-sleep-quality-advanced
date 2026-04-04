import os
import json
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd
import mne
import tensorflow as tf
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter


# ==========================================================
# CONFIGURATION (EDIT ONLY THIS SECTION)
# ==========================================================


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.environ.get("INFERENCE_MODEL_PATH", os.path.join(THIS_DIR, "sleep_best_ckpt.keras"))
EDF_PATH = os.environ.get("INFERENCE_EDF_PATH", os.path.join(THIS_DIR, "data", "SC4001E0-PSG.edf"))

OUTDIR = os.environ.get("INFERENCE_OUTPUT_DIR", os.path.join(THIS_DIR, "outputs", "run_01"))

CHANNEL_NAME = "EEG Fpz-Cz"
TARGET_FS = 100
EPOCH_SECONDS = 30
LOWCUT = 0.5
HIGHCUT = 30.0
FILTER_ORDER = 4

SEQ_LEN = 5
NORMALIZATION = "per_epoch"  # per_epoch / per_recording / none
SMOOTH = True
SMOOTH_KERNEL = 5

# Realistic scoring references (can be tuned after validation on a cohort).
LATENCY_REF_MIN = 45.0
SE_MIN_REF = 0.70
SE_GOOD_REF = 0.90
WASO_BAD_REF_MIN = 90.0
N3_TARGET = 0.20
REM_TARGET = 0.22
FRAG_BAD_REF = 0.12

# Weights for sleep quality score (sum should be 1.0).
W_SE = 0.30
W_WASO = 0.20
W_LATENCY = 0.15
W_RESTORATIVE = 0.20
W_FRAGMENTATION = 0.15


# ==========================================================
# LABEL MAP
# ==========================================================

STAGE_ID_TO_NAME = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}


# ==========================================================
# UTILITY FUNCTIONS
# ==========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class CompatibleBatchNormalization(tf.keras.layers.BatchNormalization):
    """Handles legacy configs where BatchNormalization axis may be serialized as [n]."""

    @classmethod
    def from_config(cls, config):
        axis = config.get("axis", -1)
        if isinstance(axis, (list, tuple)) and len(axis) == 1:
            config["axis"] = axis[0]
        return super().from_config(config)


def load_trained_model(primary_model_path):
    """Load model with compatibility fallbacks across Keras/TensorFlow serialization differences."""
    candidate_paths = [primary_model_path]

    # Add sibling fallback models for convenience if the chosen path fails.
    default_candidates = [
        os.path.join(THIS_DIR, "sleep_best_ckpt.keras"),
        os.path.join(THIS_DIR, "final_model_clean.h5"),
    ]
    for p in default_candidates:
        if p not in candidate_paths:
            candidate_paths.append(p)

    errors = []
    for model_path in candidate_paths:
        if not os.path.exists(model_path):
            errors.append(f"{model_path}: file not found")
            continue
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            errors.append(f"{model_path}: {e}")

        # Compatibility retry for legacy BatchNormalization axis serialization.
        try:
            return tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={"BatchNormalization": CompatibleBatchNormalization},
            )
        except Exception as e:
            errors.append(f"{model_path} (compatible BN): {e}")

    raise RuntimeError(
        "Unable to load model. Tried:\n- " + "\n- ".join(errors)
    )


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def epoch_signal(signal, fs, epoch_seconds):
    samples_per_epoch = int(epoch_seconds * fs)
    n_epochs = len(signal) // samples_per_epoch
    signal = signal[:n_epochs * samples_per_epoch]
    return signal.reshape(n_epochs, samples_per_epoch)


def normalize_epochs(epochs, mode):
    if mode == "none":
        return epochs

    if mode == "per_epoch":
        mean = epochs.mean(axis=1, keepdims=True)
        std = epochs.std(axis=1, keepdims=True) + 1e-8
        return (epochs - mean) / std

    if mode == "per_recording":
        mean = epochs.mean()
        std = epochs.std() + 1e-8
        return (epochs - mean) / std

    raise ValueError("Invalid normalization mode")


def create_sequences(epochs, seq_len):
    return np.stack(
        [epochs[i:i + seq_len] for i in range(len(epochs) - seq_len + 1)],
        axis=0
    )[..., np.newaxis]


# ==========================================================
# SLEEP METRICS
# ==========================================================

@dataclass
class SleepMetrics:
    recording_minutes: float
    analyzed_minutes: float
    tst_min: float
    sleep_latency_min: float
    sleep_efficiency: float
    waso_min: float
    pct_n3: float
    pct_rem: float
    fragmentation_index: float
    confidence_mean: float
    main_sleep_start_epoch: int
    main_sleep_end_epoch: int
    sleep_quality_score_0_100: float
    model_reliability_score_0_100: float
    sqi_0_100: float
    component_scores: Dict[str, float]
    weighted_contributions: Dict[str, float]


def _clip01(x):
    return max(0.0, min(1.0, float(x)))


def _detect_main_sleep_window(stages):
    """Find the dominant sleep bout and add context margins."""
    n = len(stages)
    sleep_idx = np.where(np.array(stages) != 0)[0]
    if len(sleep_idx) == 0:
        return 0, n - 1

    runs = []
    start = sleep_idx[0]
    prev = sleep_idx[0]
    for idx in sleep_idx[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = idx
        prev = idx
    runs.append((start, prev))

    # Select longest continuous sleep run as anchor for "previous night" sleep.
    anchor_start, anchor_end = max(runs, key=lambda r: r[1] - r[0])
    margin_epochs = int((30 * 60) / EPOCH_SECONDS)  # 30 minutes on each side

    s = max(0, anchor_start - margin_epochs)
    e = min(n - 1, anchor_end + margin_epochs)
    return s, e


def compute_sleep_metrics(stages, confidence):
    n = len(stages)
    recording_minutes = (n * EPOCH_SECONDS) / 60

    w_start, w_end = _detect_main_sleep_window(stages)
    window_stages = stages[w_start:w_end + 1]
    window_conf = confidence[w_start:w_end + 1]

    analyzed_minutes = (len(window_stages) * EPOCH_SECONDS) / 60

    sleep_epochs = [s for s in window_stages if s != 0]
    tst_min = len(sleep_epochs) * EPOCH_SECONDS / 60
    sleep_efficiency = tst_min / analyzed_minutes if analyzed_minutes > 0 else 0

    first_sleep_idx = next((i for i, s in enumerate(window_stages) if s != 0), None)
    sleep_latency_min = (
        first_sleep_idx * EPOCH_SECONDS / 60 if first_sleep_idx is not None else analyzed_minutes
    )

    waso_epochs = 0
    if first_sleep_idx is not None:
        for s in window_stages[first_sleep_idx:]:
            if s == 0:
                waso_epochs += 1
    waso_min = waso_epochs * EPOCH_SECONDS / 60

    def pct(stage):
        return window_stages.count(stage) / len(sleep_epochs) if sleep_epochs else 0

    pct_n3 = pct(3)
    pct_rem = pct(4)

    transitions = sum(1 for i in range(1, len(window_stages)) if window_stages[i] != window_stages[i - 1])
    fragmentation_index = transitions / len(window_stages) if window_stages else 0

    confidence_mean = float(np.mean(window_conf)) if window_conf else 0.0

    # Normalize individual components to [0,100].
    se_score = _clip01((sleep_efficiency - SE_MIN_REF) / (SE_GOOD_REF - SE_MIN_REF)) * 100
    waso_score = _clip01((WASO_BAD_REF_MIN - waso_min) / WASO_BAD_REF_MIN) * 100
    latency_score = _clip01((LATENCY_REF_MIN - sleep_latency_min) / LATENCY_REF_MIN) * 100
    n3_score = _clip01(pct_n3 / N3_TARGET) * 100
    rem_score = _clip01(pct_rem / REM_TARGET) * 100
    restorative_score = 0.5 * (n3_score + rem_score)
    frag_score = _clip01((FRAG_BAD_REF - fragmentation_index) / FRAG_BAD_REF) * 100
    conf_score = _clip01((confidence_mean - 0.50) / 0.50) * 100

    weighted = {
        "sleep_efficiency": W_SE * se_score,
        "waso": W_WASO * waso_score,
        "sleep_latency": W_LATENCY * latency_score,
        "restorative": W_RESTORATIVE * restorative_score,
        "fragmentation": W_FRAGMENTATION * frag_score,
    }
    sleep_quality_score = sum(weighted.values())
    model_reliability_score = conf_score

    # Keep backward-compatible SQI field as a balanced blend.
    sqi = (
        0.90 * sleep_quality_score
        + 0.10 * model_reliability_score
    )
    sqi = max(0.0, min(100.0, sqi))

    component_scores = {
        "se_score": se_score,
        "waso_score": waso_score,
        "latency_score": latency_score,
        "n3_score": n3_score,
        "rem_score": rem_score,
        "restorative_score": restorative_score,
        "fragmentation_score": frag_score,
        "confidence_score": conf_score,
    }

    return SleepMetrics(
        recording_minutes=recording_minutes,
        analyzed_minutes=analyzed_minutes,
        tst_min=tst_min,
        sleep_latency_min=sleep_latency_min,
        sleep_efficiency=sleep_efficiency,
        waso_min=waso_min,
        pct_n3=pct_n3,
        pct_rem=pct_rem,
        fragmentation_index=fragmentation_index,
        confidence_mean=confidence_mean,
        main_sleep_start_epoch=int(w_start),
        main_sleep_end_epoch=int(w_end),
        sleep_quality_score_0_100=sleep_quality_score,
        model_reliability_score_0_100=model_reliability_score,
        sqi_0_100=sqi
        ,
        component_scores=component_scores,
        weighted_contributions=weighted
    )


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main():
    ensure_dir(OUTDIR)

    print("Loading trained model...")
    model = load_trained_model(MODEL_PATH)
    print("Model loaded successfully.\n")

    print("Loading EDF...")
    raw = mne.io.read_raw_edf(EDF_PATH, preload=True, verbose=False)

    if CHANNEL_NAME in raw.ch_names:
        raw.pick_channels([CHANNEL_NAME])
    else:
        raw.pick_types(eeg=True)
        raw.pick_channels([raw.ch_names[0]])

    if raw.info["sfreq"] != TARGET_FS:
        raw.resample(TARGET_FS)

    signal = raw.get_data()[0]

    print("Filtering...")
    signal = bandpass_filter(signal, TARGET_FS, LOWCUT, HIGHCUT, FILTER_ORDER)

    print("Epoching...")
    epochs = epoch_signal(signal, TARGET_FS, EPOCH_SECONDS)

    print("Normalizing...")
    epochs = normalize_epochs(epochs, NORMALIZATION)

    print("Creating sequences...")
    X = create_sequences(epochs, SEQ_LEN)

    print("Running inference...")
    probs = model.predict(X, verbose=0)
    pred_seq = probs.argmax(axis=1)
    conf_seq = probs.max(axis=1)

    n_epochs = len(epochs)
    stages = [-1] * n_epochs
    confidence = [0] * n_epochs

    for i in range(len(pred_seq)):
        idx = i + SEQ_LEN - 1
        stages[idx] = int(pred_seq[i])
        confidence[idx] = float(conf_seq[i])

    for i in range(SEQ_LEN - 1):
        stages[i] = stages[SEQ_LEN - 1]
        confidence[i] = confidence[SEQ_LEN - 1]

    if SMOOTH:
        stages = median_filter(np.array(stages), size=SMOOTH_KERNEL).tolist()

    metrics = compute_sleep_metrics(stages, confidence)

    pd.DataFrame({
        "epoch": np.arange(len(stages)),
        "stage_id": stages,
        "stage_name": [STAGE_ID_TO_NAME[s] for s in stages],
        "confidence": confidence
    }).to_csv(os.path.join(OUTDIR, "hypnogram.csv"), index=False)

    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(asdict(metrics), f, indent=2)

    print("\n===== RESULTS =====")
    print(f"Sleep Quality Score (0-100): {metrics.sleep_quality_score_0_100:.2f}")
    print(f"Model Reliability Score (0-100): {metrics.model_reliability_score_0_100:.2f}")
    print(f"Overall Sleep Score (0-100): {metrics.sqi_0_100:.2f}")
    print(f"Recording (min): {metrics.recording_minutes:.2f}")
    print(f"Analyzed Main Sleep Window (min): {metrics.analyzed_minutes:.2f}")
    print(f"TST (min): {metrics.tst_min:.2f}")
    print(f"Sleep Latency (min): {metrics.sleep_latency_min:.2f}")
    print(f"Sleep Efficiency: {metrics.sleep_efficiency*100:.2f}%")
    print(f"%N3: {metrics.pct_n3*100:.2f}%")
    print(f"%REM: {metrics.pct_rem*100:.2f}%")
    print(f"WASO (min): {metrics.waso_min:.2f}")
    print(f"Fragmentation Index: {metrics.fragmentation_index:.4f}")
    print(f"Mean Prediction Confidence: {metrics.confidence_mean:.4f}")
    print("Score Breakdown (weighted):")
    for k, v in metrics.weighted_contributions.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()