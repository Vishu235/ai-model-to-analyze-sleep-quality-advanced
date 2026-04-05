"""
load_data.py
Loads Sleep-EDF Cassette recordings directly from a flat directory of EDF files.
Works with both a local flat folder and Modal's /data volume.
"""
import glob
import os

import mne
import numpy as np

from config import SLEEP_CASSETTE_PATH, EPOCH_LENGTH_SEC

FIXED_EVENT_ID = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,   # N4 merged into N3 (AASM 2007+)
    "Sleep stage R": 4,
}


def _find_pairs(data_dir, max_subjects=None):
    """
    Find all (PSG, Hypnogram) EDF pairs in data_dir.
    Sleep-EDF Cassette naming convention: SC4NNNX0-PSG.edf + SC4NNNXy-Hypnogram.edf
    where NNN = subject ID, X = night index.
    """
    psg_files = sorted(glob.glob(os.path.join(data_dir, "*PSG.edf")))

    # Sleep-EDF Cassette has 20 subjects (0–19), each with up to 2 nights.
    # The cap of 50 was originally set for the broader Sleep-EDF Expanded
    # dataset which includes Telemetry subjects (IDs up to ~78).
    # Keeping the cap explicit allows quick subset runs (e.g. max_subjects=6).
    if max_subjects is not None:
        # Subject IDs are encoded in the first 6 characters: SC4NNX
        seen = {}
        for p in psg_files:
            sid = os.path.basename(p)[:6]
            seen.setdefault(sid, []).append(p)
        ordered = list(seen.keys())[:max_subjects]
        psg_files = [seen[s][0] for s in ordered]   # one recording per subject

    pairs = []
    for psg in psg_files:
        sub = os.path.basename(psg)[:6]
        hyp_matches = glob.glob(os.path.join(data_dir, f"{sub}*Hypnogram.edf"))
        if hyp_matches:
            pairs.append((psg, hyp_matches[0]))
        else:
            print(f"  WARNING: no Hypnogram found for {sub}, skipping.")

    return pairs


def load_all_subjects_epochs(max_subjects=None):
    """
    Load Sleep-EDF recordings from SLEEP_CASSETTE_PATH.

    Returns
    -------
    final_epochs : mne.Epochs
        Concatenated epochs across all subjects.
    subject_ids : np.ndarray
        One subject ID string per epoch (used by GroupKFold).
    """
    print(f"Data directory : {SLEEP_CASSETTE_PATH}")
    pairs = _find_pairs(SLEEP_CASSETTE_PATH, max_subjects=max_subjects)
    print(f"Found {len(pairs)} PSG/Hypnogram pairs.")

    if not pairs:
        raise RuntimeError(
            f"No PSG/Hypnogram pairs found in {SLEEP_CASSETTE_PATH}. "
            "Check that DATA_DIR or config.SLEEP_CASSETTE_PATH is correct."
        )

    all_epochs_list = []
    all_subject_ids = []

    for idx, (psg_path, hyp_path) in enumerate(pairs):
        subject_id = os.path.basename(psg_path)[:6]
        print(f"[{idx+1}/{len(pairs)}] Processing {subject_id} ...", flush=True)

        try:
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

            # Pick EEG Fpz-Cz; fall back to first available EEG channel
            if "EEG Fpz-Cz" in raw.ch_names:
                raw.pick_channels(["EEG Fpz-Cz"])
            else:
                eeg_raw = raw.copy().pick_types(eeg=True)
                if not eeg_raw.ch_names:
                    print(f"  SKIP {subject_id}: no EEG channel found.")
                    continue
                raw.pick_channels([eeg_raw.ch_names[0]])

            annotations = mne.read_annotations(hyp_path)
            raw.set_annotations(annotations, emit_warning=False)

            try:
                events, _ = mne.events_from_annotations(
                    raw, event_id=FIXED_EVENT_ID,
                    chunk_duration=30.0, verbose=False,
                )
            except ValueError:
                print(f"  SKIP {subject_id}: no matching sleep stage annotations.")
                continue

            epochs = mne.Epochs(
                raw, events,
                event_id=FIXED_EVENT_ID,
                tmin=0,
                tmax=EPOCH_LENGTH_SEC - (1.0 / raw.info["sfreq"]),
                baseline=None,
                preload=True,
                verbose=False,
                on_missing="ignore",
            )

            n = len(epochs)
            if n > 0:
                all_epochs_list.append(epochs)
                all_subject_ids.extend([subject_id] * n)
                print(f"  {n} epochs loaded.")
            else:
                print(f"  SKIP {subject_id}: 0 epochs after windowing.")

        except Exception as exc:
            print(f"  SKIP {subject_id}: {exc}")
            continue

    if not all_epochs_list:
        raise RuntimeError("No epochs loaded. Check data path and file format.")

    final_epochs = mne.concatenate_epochs(all_epochs_list)
    print(f"\nTotal epochs loaded: {len(final_epochs)} across {len(all_epochs_list)} subjects.")
    return final_epochs, np.array(all_subject_ids)