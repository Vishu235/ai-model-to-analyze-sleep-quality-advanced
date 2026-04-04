"""
cross_dataset_eval.py — Cross-dataset generalisation study.

Compares model performance across two splits of the PhysioNet Sleep-EDF
Expanded dataset:
  - Telemetry (SC*)  — the subjects the model was trained on
  - Cassette  (ST*)  — held-out subjects never seen during training

This evaluates whether the model generalises beyond its training distribution.

Usage:
    python cross_dataset_eval.py \
        --cassette_dir  /path/to/sleep-cassette/ \
        --telemetry_dir /path/to/sleep-telemetry/ \
        --model_path    inference/sleep_best_ckpt.keras \
        --out_dir       evaluation/cross_dataset/

If you only have one split, pass just that directory and omit the other flag.

Outputs (in --out_dir):
    cassette_summary.json     — per-class metrics on Cassette subjects
    telemetry_summary.json    — per-class metrics on Telemetry subjects
    cross_dataset_report.txt  — side-by-side comparison table
    generalisation_gap.json   — accuracy/F1/Kappa delta between splits
"""
import argparse
import json
import os
import sys
import glob

import numpy as np

# Keras patch (same as production)
import keras, keras.layers  # noqa: E401

_orig = keras.layers.Dense.from_config.__func__
@classmethod
def _p(cls, c):
    c.pop("quantization_config", None)
    return _orig(cls, c)
keras.layers.Dense.from_config = _p

import mne
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt
from sklearn.metrics import (
    classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, precision_recall_fscore_support,
)

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
EVENT_ID    = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}
TARGET_FS = 100
EPOCH_SEC = 30
SPE       = TARGET_FS * EPOCH_SEC
SEQ_LEN   = 5


def _bandpass(signal):
    nyq = 0.5 * TARGET_FS
    b, a = butter(4, [0.5 / nyq, 30.0 / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def load_pairs(data_dir, psg_pattern="*PSG.edf"):
    """Find all (PSG, Hypnogram) pairs in a directory."""
    pairs = []
    for p in sorted(glob.glob(os.path.join(data_dir, psg_pattern))):
        sub = os.path.basename(p)[:6]
        hyp = glob.glob(os.path.join(data_dir, f"{sub}*Hypnogram.edf"))
        if hyp:
            pairs.append((p, hyp[0]))
    return pairs


def load_recording(psg, hyp, channel="EEG Fpz-Cz"):
    try:
        raw = mne.io.read_raw_edf(psg, preload=True, verbose=False)
        ch  = channel if channel in raw.ch_names else raw.copy().pick_types(eeg=True).ch_names[0]
        raw.pick_channels([ch])
        if raw.info["sfreq"] != TARGET_FS:
            raw.resample(TARGET_FS, verbose=False)
        ann = mne.read_annotations(hyp)
        raw.set_annotations(ann, emit_warning=False)
        events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID,
                                                  chunk_duration=30., verbose=False)
        ep = mne.Epochs(raw, events, event_id=EVENT_ID,
                        tmin=0, tmax=EPOCH_SEC - (1 / TARGET_FS),
                        baseline=None, preload=True, verbose=False)
        return ep.get_data()[:, 0, :].astype(np.float32), ep.events[:, 2].astype(int)
    except Exception as exc:
        print(f"  SKIP {os.path.basename(psg)}: {exc}", flush=True)
        return None, None


def predict_recording(model, data):
    ne = len(data)
    if ne < SEQ_LEN:
        return None
    m = data.mean(axis=1, keepdims=True)
    s = data.std(axis=1, keepdims=True) + 1e-8
    norm = (data - m) / s
    X = np.stack([norm[i:i + SEQ_LEN] for i in range(ne - SEQ_LEN + 1)])[..., np.newaxis]
    probs = model.predict(X, verbose=0)
    preds = np.zeros(ne, int)
    ps    = probs.argmax(axis=1)
    for i in range(len(ps)):
        preds[i + SEQ_LEN - 1] = ps[i]
    for i in range(SEQ_LEN - 1):
        preds[i] = preds[SEQ_LEN - 1]
    return median_filter(preds, size=5).tolist()


def evaluate_split(model, pairs, label, channel="EEG Fpz-Cz"):
    """Evaluate model on a list of (psg, hyp) pairs. Returns metrics dict."""
    if not pairs:
        return None

    all_true, all_pred = [], []
    for idx, (psg, hyp) in enumerate(pairs):
        print(f"  [{label}] [{idx+1}/{len(pairs)}] {os.path.basename(psg)}", flush=True)
        data, labels = load_recording(psg, hyp, channel=channel)
        if data is None:
            continue
        preds = predict_recording(model, data)
        if preds is None:
            continue
        n = min(len(labels), len(preds))
        all_true.extend(labels[:n].tolist())
        all_pred.extend(preds[:n])

    if not all_true:
        return None

    t = np.array(all_true)
    p = np.array(all_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        t, p, labels=[0, 1, 2, 3, 4], zero_division=0
    )

    return {
        "split":        label,
        "n_recordings": len(pairs),
        "n_epochs":     int(len(t)),
        "accuracy":     round(float((t == p).mean()), 6),
        "macro_f1":     round(float(f1_score(t, p, average="macro", zero_division=0)), 6),
        "cohen_kappa":  round(float(cohen_kappa_score(t, p)), 6),
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


def write_report(results, out_path):
    lines = ["Cross-Dataset Generalisation Report", "=" * 60, ""]
    for r in results:
        if r is None:
            continue
        lines.append(f"Split         : {r['split']}")
        lines.append(f"Recordings    : {r['n_recordings']}")
        lines.append(f"Epochs        : {r['n_epochs']}")
        lines.append(f"Accuracy      : {r['accuracy']*100:.2f}%")
        lines.append(f"Macro F1      : {r['macro_f1']:.4f}")
        lines.append(f"Cohen's Kappa : {r['cohen_kappa']:.4f}")
        lines.append("")
        lines.append(f"{'Stage':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>10}")
        lines.append("-" * 50)
        for name, m in r["per_class"].items():
            lines.append(
                f"{name:<8} {m['precision']:>10.4f} {m['recall']:>8.4f} "
                f"{m['f1']:>8.4f} {m['support']:>10}"
            )
        lines.append("")
        lines.append("-" * 60)
        lines.append("")

    # Generalisation gap (if two splits available)
    valid = [r for r in results if r is not None]
    if len(valid) == 2:
        a, b = valid[0], valid[1]
        lines.append("Generalisation Gap (Telemetry → Cassette)")
        lines.append(f"  Accuracy drop : {(a['accuracy'] - b['accuracy'])*100:+.2f}%")
        lines.append(f"  Macro F1 drop : {(a['macro_f1'] - b['macro_f1']):+.4f}")
        lines.append(f"  Kappa drop    : {(a['cohen_kappa'] - b['cohen_kappa']):+.4f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--cassette_dir",  default=None,
                        help="Path to sleep-cassette/ folder (SC* recordings)")
    parser.add_argument("--telemetry_dir", default=None,
                        help="Path to sleep-telemetry/ folder (ST* recordings)")
    parser.add_argument("--model_path",    default="inference/sleep_best_ckpt.keras")
    parser.add_argument("--channel",       default="EEG Fpz-Cz")
    parser.add_argument("--max_files",     type=int, default=None)
    parser.add_argument("--out_dir",       default="evaluation/cross_dataset")
    args = parser.parse_args()

    if not args.cassette_dir and not args.telemetry_dir:
        parser.error("Provide at least one of --cassette_dir or --telemetry_dir")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading model from {args.model_path} …")
    model = keras.models.load_model(args.model_path, compile=False, safe_mode=False)
    print("Model loaded.\n")

    results = []

    if args.telemetry_dir:
        pairs = load_pairs(args.telemetry_dir, psg_pattern="*PSG.edf")
        if args.max_files:
            pairs = pairs[:args.max_files]
        print(f"Telemetry split: {len(pairs)} recordings")
        r = evaluate_split(model, pairs, label="Telemetry (train distribution)", channel=args.channel)
        results.append(r)
        if r:
            with open(os.path.join(args.out_dir, "telemetry_summary.json"), "w") as f:
                json.dump(r, f, indent=2)
            print(f"\nTelemetry — Accuracy: {r['accuracy']*100:.2f}%  Kappa: {r['cohen_kappa']:.4f}")

    if args.cassette_dir:
        pairs = load_pairs(args.cassette_dir, psg_pattern="*PSG.edf")
        if args.max_files:
            pairs = pairs[:args.max_files]
        print(f"\nCassette split: {len(pairs)} recordings")
        r = evaluate_split(model, pairs, label="Cassette (held-out subjects)", channel=args.channel)
        results.append(r)
        if r:
            with open(os.path.join(args.out_dir, "cassette_summary.json"), "w") as f:
                json.dump(r, f, indent=2)
            print(f"\nCassette — Accuracy: {r['accuracy']*100:.2f}%  Kappa: {r['cohen_kappa']:.4f}")

    write_report(results, os.path.join(args.out_dir, "cross_dataset_report.txt"))

    # Generalisation gap JSON
    valid = [r for r in results if r is not None]
    if len(valid) == 2:
        gap = {
            "from_split":    valid[0]["split"],
            "to_split":      valid[1]["split"],
            "accuracy_drop": round(valid[0]["accuracy"] - valid[1]["accuracy"], 6),
            "macro_f1_drop": round(valid[0]["macro_f1"] - valid[1]["macro_f1"], 6),
            "kappa_drop":    round(valid[0]["cohen_kappa"] - valid[1]["cohen_kappa"], 6),
        }
        with open(os.path.join(args.out_dir, "generalisation_gap.json"), "w") as f:
            json.dump(gap, f, indent=2)
        print(f"\nGeneralisation gap saved.")

    print("\nCross-dataset evaluation complete.")


if __name__ == "__main__":
    main()