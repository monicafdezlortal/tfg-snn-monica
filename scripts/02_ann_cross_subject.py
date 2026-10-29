"""
Experimento 02 - ANN (MLP) cross-subject sobre las Extracted Features de SEED (de_LDS).
  - 4 canales (FT7, FT8, T7, T8) x 5 bandas = 20 features, o 62 x 5 = 310
  - Normalizacion z-score por sujeto (sin leakage entre sujetos)
  - sujetos 1-3 -> test, 4-5 -> val, 6-15 -> train

Uso:
  python scripts/02_ann_cross_subject.py --channels 4 --seed 42
Salida:
  results/02_ann_cross_subject/<4ch|62ch>_seed<N>/
    config.json, metrics.json, confusion_matrix.csv, model.pth
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from eeg_emotion import config as cfg
from eeg_emotion.data import (load_seed_features, normalize_per_subject,
                              cross_subject_masks, describe_split)
from eeg_emotion.models import EmotionANN
from eeg_emotion.training import (set_seed, class_weights, n_params, evaluate, fit,
                                  compute_metrics, run_dir, save_run)

EXPERIMENT = "02_ann_cross_subject"

# ============================================================
# HIPERPARAMETROS
# ============================================================
HIDDEN        = 128
EPOCHS        = 50
BATCH_SIZE    = 256
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", type=int, choices=[4, 62], default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help="solo para pruebas rapidas (por defecto %(default)s)")
    p.add_argument("--out", default=None, help="carpeta de salida (por defecto en results/)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device  = cfg.DEVICE
    out_dir = args.out or run_dir(EXPERIMENT, args.channels, args.seed)
    ckpt    = os.path.join(out_dir, "model.pth")
    print(f"Dispositivo: {device}\n")

    X_raw, meta = load_seed_features(args.channels)
    y, subject = meta["y"], meta["subject"]
    X = normalize_per_subject(X_raw, subject)
    input_dim = X.shape[1]
    print(f"Features: {cfg.FEATURE_KEY}  X={X.shape}  ({args.channels} canales x 5 bandas)")

    # --- Split cross-subject held-out ---
    train_mask, val_mask, test_mask = cross_subject_masks(subject)
    y_train = y[train_mask]
    describe_split(subject, (train_mask, val_mask, test_mask))

    def make_ds(mask):
        return TensorDataset(torch.from_numpy(X[mask]), torch.from_numpy(y[mask]))

    train_ds, val_ds, test_ds = make_ds(train_mask), make_ds(val_mask), make_ds(test_mask)
    n_train, n_val = len(train_ds), len(val_ds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    print(f"  train={n_train}  val={n_val}  test={len(test_ds)}")

    # --- Modelo ---
    model   = EmotionANN(input_dim=input_dim, hidden=HIDDEN, num_classes=3,
                         dropout=DROPOUT).to(device)
    optim   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optim, mode='max', factor=0.5, patience=7, min_lr=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_train, device))
    print(f"Parametros entrenables: {n_params(model):,}\n")

    best_val_acc, best_epoch, history = fit(model, train_loader, val_loader, optim, sched,
                                            loss_fn, args.epochs, ckpt, device)

    # --- Evaluacion final en test ---
    test_loss, test_acc, preds, true = evaluate(model, test_loader, device)
    print(f"\nTEST  loss={test_loss:.4f}  acc={test_acc:.4f}")
    print()
    print(classification_report(true, preds, target_names=cfg.CLASS_NAMES, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(true, preds))
    print(f"\nModelo guardado en: {ckpt}")

    config = {
        "experiment": EXPERIMENT, "model": "EmotionANN", "channels": args.channels,
        "seed": args.seed, "feature_key": cfg.FEATURE_KEY, "input_dim": input_dim,
        "normalization": "z-score por sujeto",
        "train_subjects": sorted(np.unique(subject[train_mask]).tolist()),
        "val_subjects": cfg.VAL_SUBJECTS, "test_subjects": cfg.TEST_SUBJECTS,
        "hidden": HIDDEN, "epochs": args.epochs, "batch_size": BATCH_SIZE, "lr": LR,
        "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "n_params": n_params(model),
        "device": device, "torch_threads": torch.get_num_threads(),
    }
    metrics = {"seed": args.seed, **compute_metrics(true, preds), "test_loss": test_loss,
               "best_val_acc": best_val_acc, "best_epoch": best_epoch, "history": history}
    save_run(out_dir, config, metrics)


if __name__ == "__main__":
    main()
