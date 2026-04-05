import gc
import os
import numpy as np
import wandb
import tensorflow as tf
from types import SimpleNamespace
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, cohen_kappa_score,
    precision_recall_fscore_support, classification_report,
)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from load_data import load_all_subjects_epochs
from dl_data import prepare_cnn_lstm_data_by_subject
from dl_model import build_cnn_lstm

STAGE_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
N_FOLDS = 5

DEFAULT_CONFIG = {
    "max_subjects": None,
    "seq_len": 5,
    "epochs": 35,
    "batch_size": 32,
    "eval_batch_size": 256,   # batched predict/evaluate — avoids GPU OOM on large test sets
}


def _safe_wandb_log(payload, run):
    if run is None:
        return
    try:
        wandb.log(payload)
    except Exception as exc:
        print(f"WARNING: wandb.log failed, continuing without remote logging: {exc}")


def train_model(config=None):
    merged_cfg = dict(DEFAULT_CONFIG)
    if config:
        merged_cfg.update(config)
    cfg = SimpleNamespace(**merged_cfg)

    run = None
    try:
        run = wandb.init(project="sleep-edf-5class", config=merged_cfg)
        cfg = SimpleNamespace(**dict(wandb.config))
    except Exception as exc:
        # Transient DNS/network errors should not abort expensive training jobs.
        print(f"WARNING: wandb.init failed, continuing without wandb: {exc}")

    print("Loading epochs...")
    epochs_data, subject_ids = load_all_subjects_epochs(max_subjects=cfg.max_subjects)

    print("Preparing subject-safe sequences...")
    X, y, groups = prepare_cnn_lstm_data_by_subject(
        epochs_data,
        subject_ids,
        seq_len=cfg.seq_len,
    )

    if len(X) == 0:
        raise RuntimeError("No training sequences produced. Check data loading and seq_len.")

    # ------------------------------------------------------------------
    # 5-fold cross-validation with GroupKFold.
    # Groups = subject IDs, so no subject appears in both train and test
    # within any fold — prevents data leakage across subject boundaries.
    # Metrics are collected per fold and summarised as mean ± std at the
    # end, as required for conference paper reporting.
    # ------------------------------------------------------------------
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_metrics = []   # list of dicts, one per fold
    best_acc = -1.0
    best_model_path = None

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        # Clear Keras graph and free GPU memory from the previous fold before
        # building a new model. Without this, residual tensors accumulate and
        # cause OOM on fold 2+ even on a 16 GB T4.
        K.clear_session()
        gc.collect()

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/{N_FOLDS}")
        print(f"  Train sequences: {len(train_idx)}  |  Test sequences: {len(test_idx)}")

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        classes = np.unique(y_tr)
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        class_weight = dict(zip(classes, cw))

        model = build_cnn_lstm(X_tr.shape[1:], len(classes))

        # Use tf.data so training and validation data are streamed in batches
        # and never fully staged on GPU at once — avoids OOM on large folds.
        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
            .shuffle(buffer_size=min(len(X_tr), 10000), seed=42)
            .batch(cfg.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_te, y_te))
            .batch(cfg.eval_batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        ckpt_path = f"sleep_best_ckpt_fold{fold_idx}.keras"
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]
        if run is not None:
            callbacks.extend([
                WandbMetricsLogger(log_freq="epoch"),
                WandbModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss"),
            ])

        print(f"  Training fold {fold_idx}...")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.epochs,
            class_weight=class_weight,
            callbacks=callbacks,
        )

        # --- Evaluate (batched to stay within GPU memory) ---
        loss, acc = model.evaluate(val_ds, verbose=0)
        y_pred = model.predict(val_ds, verbose=0).argmax(axis=1)

        macro_f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
        kappa    = float(cohen_kappa_score(y_te, y_pred))

        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            y_te, y_pred, labels=[0, 1, 2, 3, 4], zero_division=0
        )

        per_class = {
            STAGE_NAMES[i]: {
                "precision": round(float(precision[i]), 4),
                "recall":    round(float(recall[i]),    4),
                "f1":        round(float(f1_per_class[i]), 4),
                "support":   int(support[i]),
            }
            for i in range(len(STAGE_NAMES))
        }

        fold_result = {
            "fold":      fold_idx,
            "accuracy":  float(acc),
            "macro_f1":  macro_f1,
            "kappa":     kappa,
            "loss":      float(loss),
            "per_class": per_class,
        }
        fold_metrics.append(fold_result)

        print(f"  Fold {fold_idx} — Acc: {acc:.4f}  Macro-F1: {macro_f1:.4f}  Kappa: {kappa:.4f}")
        print(classification_report(y_te, y_pred, target_names=STAGE_NAMES, zero_division=0))

        _safe_wandb_log({
            f"fold{fold_idx}/test_accuracy": float(acc),
            f"fold{fold_idx}/test_macro_f1": macro_f1,
            f"fold{fold_idx}/test_kappa":    kappa,
            f"fold{fold_idx}/test_loss":     float(loss),
            **{f"fold{fold_idx}/f1_{STAGE_NAMES[i]}": float(f1_per_class[i])
               for i in range(len(STAGE_NAMES))},
        }, run)

        # Keep track of best fold for final model checkpoint
        if float(acc) > best_acc:
            best_acc = float(acc)
            best_model_path = ckpt_path

        # Explicitly release GPU memory before next fold.
        # del model removes the Python reference; K.clear_session() + gc.collect()
        # triggers TF/Keras to free the underlying GPU allocations.
        del model, train_ds, val_ds, X_tr, X_te, y_tr, y_te
        K.clear_session()
        gc.collect()

    # ------------------------------------------------------------------
    # Aggregate results across all folds — mean ± std for paper reporting
    # ------------------------------------------------------------------
    accs      = [m["accuracy"]  for m in fold_metrics]
    f1s       = [m["macro_f1"]  for m in fold_metrics]
    kappas    = [m["kappa"]     for m in fold_metrics]

    mean_acc   = float(np.mean(accs));    std_acc   = float(np.std(accs))
    mean_f1    = float(np.mean(f1s));     std_f1    = float(np.std(f1s))
    mean_kappa = float(np.mean(kappas));  std_kappa = float(np.std(kappas))

    print(f"\n{'='*60}")
    print(f"5-Fold Cross-Validation Summary")
    print(f"{'='*60}")
    print(f"  Accuracy  : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  Macro F1  : {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Kappa     : {mean_kappa:.4f} ± {std_kappa:.4f}")
    print(f"\nPer-fold breakdown:")
    for m in fold_metrics:
        print(f"  Fold {m['fold']}: Acc={m['accuracy']*100:.2f}%  "
              f"F1={m['macro_f1']:.4f}  Kappa={m['kappa']:.4f}")
    print(f"\nBest model checkpoint: {best_model_path}")

    _safe_wandb_log({
        "cv/mean_accuracy":  mean_acc,
        "cv/std_accuracy":   std_acc,
        "cv/mean_macro_f1":  mean_f1,
        "cv/std_macro_f1":   std_f1,
        "cv/mean_kappa":     mean_kappa,
        "cv/std_kappa":      std_kappa,
    }, run)

    if run is not None:
        run.finish()
    return mean_acc