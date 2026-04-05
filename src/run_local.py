"""
run_local.py — Run 5-fold cross-validation training locally.

Usage:
    cd src/
    python run_local.py

Override data path without editing config.py:
    set DATA_DIR=D:\path\to\sleep-cassette && python run_local.py

Speed vs quality trade-offs:
    Full run (paper numbers):  max_subjects=None, epochs=35  (~4-8 hrs on CPU)
    Quick validation run:      max_subjects=6,    epochs=2   (~10 min on CPU)
"""
import os
from dl_main import train_model

config = {
    # None = use all subjects found in DATA_DIR / config.SLEEP_CASSETTE_PATH
    # Set to a small number (e.g. 6) for a quick smoke-test
    "max_subjects": None,

    "seq_len":    5,    # LSTM context window in epochs (must match saved model)
    "epochs":     35,   # training epochs per fold
    "batch_size": 32,
}

if __name__ == "__main__":
    print("=" * 60)
    print("Sleep Staging — 5-Fold CV Training (Local)")
    print("=" * 60)
    print(f"Data directory : {os.environ.get('DATA_DIR', 'using config.py default')}")
    print(f"Config         : {config}")
    print()
    mean_acc = train_model(config)
    print(f"\nFinal mean accuracy across folds: {mean_acc*100:.2f}%")