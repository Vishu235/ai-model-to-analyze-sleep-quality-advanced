import os
import numpy as np
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint 
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, cohen_kappa_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from load_data import load_all_subjects_epochs
from dl_data import prepare_cnn_lstm_data_by_subject
from dl_model import build_cnn_lstm

def train_model(config=None):
    # wandb init 
    run = wandb.init(project="sleep-edf-5class", config=config)
    cfg = wandb.config

    print(f"Loading epochs...")
    epochs_data, subject_ids = load_all_subjects_epochs(max_subjects=cfg.max_subjects)

    print("preparing subject-safe sequences...")
    X, y, groups = prepare_cnn_lstm_data_by_subject(
        epochs_data,
        subject_ids,
        seq_len=cfg.seq_len,
    )

    if len(X) == 0:
        raise RuntimeError("No training sequences produced. Check data loading and seq_len.")

    gkf = GroupKFold(n_splits=5) 
    train_idx, test_idx = next(gkf.split(X, y, groups))
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weight = dict(zip(classes, cw))

    model = build_cnn_lstm(X_tr.shape[1:], len(classes))

    callbacks = [    # Final Shape: (Samples, Seq_Len, Time, 1)

        EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        WandbMetricsLogger(log_freq="epoch"),
        WandbModelCheckpoint("sleep_best_ckpt.keras", save_best_only=True, monitor="val_loss") 

    ]

    print("Training...")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_te, y_te),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        class_weight=class_weight,
        callbacks=callbacks
    )

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    y_pred = model.predict(X_te, verbose=0).argmax(axis=1)

    macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_te, y_pred)

    wandb.log(
        {
            "test_loss": float(loss),
            "test_accuracy": float(acc),
            "test_macro_f1": float(macro_f1),
            "test_kappa": float(kappa),
        }
    )

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Cohen Kappa: {kappa:.4f}")
    
    run.finish()
    return acc