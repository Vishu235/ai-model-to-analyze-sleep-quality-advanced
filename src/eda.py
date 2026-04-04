"""
eda.py

Exploratory Data Analysis (EDA) module for Sleep-EDF project.
Generates plots to understand data distribution and EEG characteristics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from load_data import load_all_subjects_epochs
from preprocess import extract_bandpower_features


# -----------------------------
# Output directory for plots
# -----------------------------
PLOT_DIR = os.path.join("results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def plot_class_distribution(y):
    """
    Plots sleep stage class distribution.
    """
    counter = Counter(y)

    plt.figure(figsize=(8, 5))
    plt.bar(counter.keys(), counter.values())
    plt.xlabel("Sleep Stage Label")
    plt.ylabel("Number of Epochs")
    plt.title("Sleep Stage Class Distribution")
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, "class_distribution.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved class distribution plot: {save_path}")


def plot_raw_eeg_signal(epochs, epoch_idx=0):
    """
    Plots raw EEG signal for one epoch (30 seconds).
    """
    data = epochs.get_data()[epoch_idx]  # shape: (channels, samples)
    signal = data[0]  # first EEG channel

    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.title("Raw EEG Signal (Single Epoch)")
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, "raw_eeg_signal.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved raw EEG signal plot: {save_path}")


def plot_bandpower_features(X):
    """
    Plots bandpower feature distribution.
    """
    feature_names = ["Delta", "Theta", "Alpha", "Beta"]

    plt.figure(figsize=(8, 5))
    plt.boxplot(X, labels=feature_names)
    plt.ylabel("Band Power Value")
    plt.title("EEG Bandpower Feature Distribution")
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, "bandpower_distribution.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved bandpower feature plot: {save_path}")


def run_eda():
    """
    Runs complete EDA pipeline.
    """
    print("Starting EDA pipeline...")

    # Load combined epochs from multiple subjects
    epochs = load_all_subjects_epochs()

    # Extract features
    X, y = extract_bandpower_features(epochs)

    # Generate plots
    plot_class_distribution(y)
    plot_raw_eeg_signal(epochs)
    plot_bandpower_features(X)

    print("EDA completed successfully.")


if __name__ == "__main__":
    run_eda()
