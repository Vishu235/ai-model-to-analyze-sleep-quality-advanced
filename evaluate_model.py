"""
evaluate_model.py — Confusion matrix + per-class metrics for the CNN+LSTM model.

Usage:
    python evaluate_model.py --data_dir /path/to/sleep-cassette \
                              --model_path inference/sleep_best_ckpt.keras \
                              --out_dir evaluation/

Outputs (in --out_dir):
    confusion_matrix.png      — normalised heatmap
    per_class_metrics.csv     — precision, recall, F1, support per stage
    classification_report.txt — sklearn text report
    summary.json              — overall accuracy, macro-F1, Cohen's Kappa

Requires: tensorflow, mne, scikit-learn, matplotlib, seaborn
"""
import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Keras compatibility patch (same as in production)
# ---------------------------------------------------------------------------
import keras
import keras.layers

_orig = keras.layers.Dense.from_config.__func__

@classmethod
def _patched(cls, config):
    config.pop("quantization_config", None)
    return _orig(cls, config)

keras.layers.Dense.from_config = _patched

import mne
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

STAGE_NAMES  = ["Wake", "N1", "N2", "N3", "REM"]
EVENT_ID     = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}
TARGET_FS    = 100
EPOCH_SEC    = 30
SPE          = TARGET_FS * EPOCH_SEC   # samples per epoch
SEQ_LEN      = 5
SMOOTH_SIZE  = 5


def _bandpass(signal, fs=TARGET_FS):
    nyq = 0.5 * fs
    b, a = butter(4, [0.5 / nyq, 30.0 / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def load_recording(psg_path, hyp_path, channel="EEG Fpz-Cz"):
    """Load one PSG + hypnogram pair. Returns (data, labels) arrays or None."""
    try:
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        ch  = channel if channel in raw.ch_names else raw.copy().pick_types(eeg=True).ch_names[0]
        raw.pick_channels([ch])
        if raw.info["sfreq"] != TARGET_FS:
            raw.resample(TARGET_FS, verbose=False)

        ann = mne.read_annotations(hyp_path)
        raw.set_annotations(ann, emit_warning=False)
        events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID,
                                                  chunk_duration=30., verbose=False)
        ep = mne.Epochs(raw, events, event_id=EVENT_ID,
                        tmin=0, tmax=EPOCH_SEC - (1 / TARGET_FS),
                        baseline=None, preload=True, verbose=False)

        data   = ep.get_data()[:, 0, :].astype(np.float32)
        labels = ep.events[:, 2].astype(int)
        return data, labels
    except Exception as exc:
        print(f"  SKIP {os.path.basename(psg_path)}: {exc}", flush=True)
        return None, None


def run_inference(model, data):
    """Run CNN+LSTM on one recording's epoch array. Returns predicted stage list."""
    ne = len(data)
    if ne < SEQ_LEN:
        return None

    m   = data.mean(axis=1, keepdims=True)
    s   = data.std(axis=1, keepdims=True) + 1e-8
    norm = (data - m) / s

    X = np.stack([norm[i:i + SEQ_LEN] for i in range(ne - SEQ_LEN + 1)])[..., np.newaxis]
    probs    = model.predict(X, verbose=0)
    pred_seq = probs.argmax(axis=1)

    preds = np.zeros(ne, int)
    for i in range(len(pred_seq)):
        preds[i + SEQ_LEN - 1] = pred_seq[i]
    for i in range(SEQ_LEN - 1):
        preds[i] = preds[SEQ_LEN - 1]

    return median_filter(preds, size=SMOOTH_SIZE).tolist()


def plot_confusion_matrix(cm, out_path):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not installed — skipping confusion matrix plot.")
        return

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted Stage", fontsize=12)
    ax.set_ylabel("True Stage",      fontsize=12)
    ax.set_title("CNN+LSTM Sleep Staging — Normalised Confusion Matrix", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN+LSTM sleep staging model")
    parser.add_argument("--data_dir",   required=True, help="Path to sleep-cassette/ folder")
    parser.add_argument("--model_path", default="inference/sleep_best_ckpt.keras")
    parser.add_argument("--channel",    default="EEG Fpz-Cz")
    parser.add_argument("--max_files",  type=int, default=None,
                        help="Limit number of recordings (useful for quick test runs)")
    parser.add_argument("--out_dir",    default="evaluation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load model ---
    print(f"Loading model from {args.model_path} …")
    model = keras.models.load_model(args.model_path, compile=False, safe_mode=False)
    print("Model loaded.\n")

    # --- Find PSG / Hypnogram pairs ---
    import glob
    psgs = sorted(glob.glob(os.path.join(args.data_dir, "*E0-PSG.edf")))
    if args.max_files:
        psgs = psgs[: args.max_files]

    pairs = []
    for p in psgs:
        sub = os.path.basename(p)[:6]
        hyp = glob.glob(os.path.join(args.data_dir, f"{sub}*Hypnogram.edf"))
        if hyp:
            pairs.append((p, hyp[0]))

    print(f"Found {len(pairs)} recordings.\n")
    if not pairs:
        print("No PSG/Hypnogram pairs found. Check --data_dir path.")
        sys.exit(1)

    # --- Collect predictions and ground truth ---
    all_true, all_pred = [], []

    for idx, (psg, hyp) in enumerate(pairs):
        print(f"[{idx+1}/{len(pairs)}] {os.path.basename(psg)}", flush=True)
        data, labels = load_recording(psg, hyp, channel=args.channel)
        if data is None:
            continue

        preds = run_inference(model, data)
        if preds is None:
            print("  SKIP: too short")
            continue

        # Align lengths (label count may differ from epoch count by ±1)
        min_len = min(len(labels), len(preds))
        all_true.extend(labels[:min_len].tolist())
        all_pred.extend(preds[:min_len])

    if not all_true:
        print("No predictions collected. Exiting.")
        sys.exit(1)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # --- Metrics ---
    accuracy = float((all_true == all_pred).mean())
    macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
    kappa    = float(cohen_kappa_score(all_true, all_pred))
    cm       = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3, 4])
    report   = classification_report(
        all_true, all_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=STAGE_NAMES,
        zero_division=0,
    )

    print("\n" + "="*60)
    print(f"Overall Accuracy : {accuracy*100:.2f}%")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Cohen's Kappa    : {kappa:.4f}")
    print("\nPer-class Report:")
    print(report)

    # --- Save outputs ---
    # 1. Confusion matrix plot
    plot_confusion_matrix(cm, os.path.join(args.out_dir, "confusion_matrix.png"))

    # 2. Per-class CSV
    import csv
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true, all_pred, labels=[0, 1, 2, 3, 4], zero_division=0
    )
    csv_path = os.path.join(args.out_dir, "per_class_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Stage", "Precision", "Recall", "F1", "Support"])
        for i, name in enumerate(STAGE_NAMES):
            writer.writerow([name,
                              round(float(precision[i]), 4),
                              round(float(recall[i]),    4),
                              round(float(f1[i]),        4),
                              int(support[i])])
    print(f"Saved: {csv_path}")

    # 3. Text report
    txt_path = os.path.join(args.out_dir, "classification_report.txt")
    with open(txt_path, "w") as f:
        f.write(f"Overall Accuracy : {accuracy*100:.2f}%\n")
        f.write(f"Macro F1         : {macro_f1:.4f}\n")
        f.write(f"Cohen's Kappa    : {kappa:.4f}\n\n")
        f.write(report)
    print(f"Saved: {txt_path}")

    # 4. Summary JSON
    summary = {
        "n_epochs":       int(len(all_true)),
        "n_recordings":   len(pairs),
        "accuracy":       round(accuracy, 6),
        "macro_f1":       round(macro_f1, 6),
        "cohen_kappa":    round(kappa, 6),
        "per_class": {
            name: {
                "precision": round(float(precision[i]), 4),
                "recall":    round(float(recall[i]),    4),
                "f1":        round(float(f1[i]),        4),
                "support":   int(support[i]),
            }
            for i, name in enumerate(STAGE_NAMES)
        },
    }
    json_path = os.path.join(args.out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
