"""
Experimento 03 - SNN (SpikingJelly, neuronas LIF) cross-subject sobre SEED de_LDS.
Split cross-subject held-out: sujetos 1-3 -> test, 4-5 -> val, 6-15 -> train.
La validacion usa sujetos enteros: con random_split las ventanas contiguas
(casi identicas tras el suavizado LDS) caian en train y val a la vez y
val_acc salia ~100%, sin informar de la generalizacion a sujetos nuevos
(ver legacy/train_snn_random_split.py).

Entrada: z-score por sujeto mapeado a [0,1] (clip +-4 sigma) y codificado en
spikes por rate coding (Bernoulli) con T pasos, regenerado en cada epoch.

Uso:
  python scripts/03_snn_cross_subject.py --channels 4 --seed 42
Salida:
  results/03_snn_cross_subject/<4ch|62ch>_seed<N>/
    config.json, metrics.json, confusion_matrix.csv, model.pth
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from eeg_emotion import config as cfg
from eeg_emotion.data import (load_seed_features, zscore_per_subject, to_spike_rate,
                              cross_subject_masks, describe_split,
                              SpikeDataset, spike_collate)
from eeg_emotion.models import EmotionSNN
from eeg_emotion.training import (set_seed, class_weights, n_params, evaluate, fit,
                                  compute_metrics, run_dir, save_run)

EXPERIMENT = "03_snn_cross_subject"

# ============================================================
# HIPERPARAMETROS
# ============================================================
T           = 50       # pasos temporales del rate coding
HIDDEN      = 128
EPOCHS      = 50
BATCH_SIZE  = 256
LR          = 1e-3
WEIGHT_DECAY= 1e-4
DROPOUT     = 0.3


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

    # --- Cargar datos (sujetos en orden 1..15, como el antiguo .npz) ---
    X_raw, meta = load_seed_features(args.channels, order="subject")
    y, subject = meta["y"], meta["subject"]

    # --- Normalizacion por sujeto ---
    # Cada sujeto se normaliza con su propia media y std por feature.
    # Los sujetos de test usan sus propias estadisticas -> sin leakage.
    # Tras z-score se mapea a [0,1] clipando en ±4σ para el rate coding.
    X_subj = to_spike_rate(zscore_per_subject(X_raw, subject))
    input_dim = X_subj.shape[1]
    print(f"Normalizacion por sujeto: rango global [{X_subj.min():.3f}, {X_subj.max():.3f}]")

    # --- Split cross-subject held-out ---
    train_mask, val_mask, test_mask = cross_subject_masks(subject)
    X_train, y_train = X_subj[train_mask], y[train_mask]
    X_val,   y_val   = X_subj[val_mask],   y[val_mask]
    X_test,  y_test  = X_subj[test_mask],  y[test_mask]
    describe_split(subject, (train_mask, val_mask, test_mask))

    # --- Dataset y DataLoaders ---
    train_ds = SpikeDataset(X_train, y_train, T)
    val_ds   = SpikeDataset(X_val,   y_val,   T)
    test_ds  = SpikeDataset(X_test,  y_test,  T)
    n_train, n_val = len(train_ds), len(val_ds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=spike_collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=spike_collate, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=spike_collate, num_workers=0)

    print(f"  train={n_train}  val={n_val}  test={len(test_ds)}\n")

    # --- Class weights (penaliza mas los errores en clases minoritarias en train) ---
    w_tensor = class_weights(y_train, device)
    weights = w_tensor.cpu().numpy()
    print(f"\nClass weights -> negativa={weights[0]:.3f}  "
          f"neutra={weights[1]:.3f}  positiva={weights[2]:.3f}")

    # --- Modelo ---
    model   = EmotionSNN(input_dim=input_dim, hidden=HIDDEN, num_classes=3,
                         dropout=DROPOUT).to(device)
    optim   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Reduce LR x0.5 si val_acc no mejora en 7 epochs; minimo 1e-5
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optim, mode='max', factor=0.5, patience=7, min_lr=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=w_tensor)
    print(f"Parametros entrenables: {n_params(model):,}\n")

    best_val_acc, best_epoch, history = fit(model, train_loader, val_loader, optim, sched,
                                            loss_fn, args.epochs, ckpt, device,
                                            spiking=True)

    # --- Evaluacion final en test ---
    test_loss, test_acc, preds, true = evaluate(model, test_loader, device, spiking=True)
    print(f"\nTEST  loss={test_loss:.4f}  acc={test_acc:.4f}")
    print()
    print(classification_report(true, preds, target_names=cfg.CLASS_NAMES, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(true, preds))
    print(f"\nModelo guardado en: {ckpt}")

    config = {
        "experiment": EXPERIMENT, "model": "EmotionSNN", "channels": args.channels,
        "seed": args.seed, "feature_key": cfg.FEATURE_KEY, "input_dim": input_dim,
        "normalization": "z-score por sujeto -> [0,1] (clip ±4σ)",
        "encoding": "rate coding Bernoulli", "T": T,
        "neuron": "LIF tau=2.0 v_threshold=0.5 surrogate=ATan",
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
