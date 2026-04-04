"""
dl_data.py
"""
import numpy as np
import mne 
from mne.epochs import Epochs


def _zscore_per_epoch(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)

def prepare_cnn_lstm_data(epochs, seq_len=5):
    """
    Args:
        epochs: MNE Epochs object
        seq_len: int
    """
    data = epochs.get_data()          # Shape: (N, Ch, Time)
    labels = epochs.events[:, -1]     # Shape: (N,)

    # Extract single channel (EEG) -> Shape: (N, Time)
    x = data[:, 0, :]                 

    x = _zscore_per_epoch(x)

    X_seq, y_seq = [], []
    
    for i in range(len(x) - seq_len + 1):
        X_seq.append(x[i : i + seq_len])
        y_seq.append(labels[i + seq_len - 1]) 

    X = np.array(X_seq)[..., np.newaxis] 
    y = np.array(y_seq)
    
    return X, y


def prepare_cnn_lstm_data_by_subject(epochs, subject_ids, seq_len=5):
    """
    Builds sequences strictly within each subject to avoid context leakage
    across subject boundaries.

    Args:
        epochs: MNE Epochs object
        subject_ids: np.ndarray with one subject id per epoch
        seq_len: int
    """
    data = epochs.get_data()          # Shape: (N, Ch, Time)
    labels = epochs.events[:, -1]     # Shape: (N,)

    if len(subject_ids) != len(data):
        raise ValueError("subject_ids length must match number of epochs")

    x = data[:, 0, :]                 # Shape: (N, Time)
    x = _zscore_per_epoch(x)

    X_seq, y_seq, groups = [], [], []

    # Keep insertion order of subjects as loaded.
    ordered_subjects = list(dict.fromkeys(subject_ids.tolist()))

    for sid in ordered_subjects:
        idx = np.where(subject_ids == sid)[0]
        if len(idx) < seq_len:
            continue

        xs = x[idx]
        ys = labels[idx]

        for i in range(len(xs) - seq_len + 1):
            X_seq.append(xs[i : i + seq_len])
            y_seq.append(ys[i + seq_len - 1])
            groups.append(sid)

    X = np.array(X_seq)[..., np.newaxis]
    y = np.array(y_seq)
    groups = np.array(groups)

    return X, y, groups