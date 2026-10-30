"""
Experimento 01 - Replica del protocolo de Zheng & Lu (2015) sobre SEED, para
comparar nuestra ANN con los resultados del paper en igualdad de condiciones.

Protocolo del paper (subject-dependent, por sesion):
  - cada sesion de cada sujeto es un "experimento" independiente
  - trials 1-9 de la sesion -> train, trials 10-15 -> test
  - se entrena un modelo nuevo por sesion y se reporta media +- std
  - features DE suavizadas (de_LDS), 4 canales (FT7, FT8, T7, T8) o 62 x 5 bandas

Modelos evaluados:
  - ANN: misma arquitectura que 02_ann_cross_subject.py (EmotionANN), 50 epocas
    fijas: en este protocolo no hay validacion y el test no se usa para elegir epoca
  - SVM lineal: baseline del propio paper. C se elige con validacion cruzada
    agrupada por trials (GroupKFold) DENTRO del train de cada sesion

Referencias del paper para comparar:
  - 4 canales:            82.88 % +- 10.92
  - 62 canales, SVM:      83.99 % +-  9.72
  - 62 canales, DBN:      86.08 % +-  8.34

Uso:
  python scripts/01_zheng_lu_ann_svm.py --channels 4 --seed 42
  python scripts/01_zheng_lu_ann_svm.py --channels 62 --sessions 3
Salida:
  results/01_zheng_lu_ann_svm/<4ch|62ch>[_3ses]_seed<N>/
    config.json, metrics.json, sessions.csv,
    confusion_matrix_EmotionANN.csv, confusion_matrix_LinearSVC.csv
"""

import argparse
import csv
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, GroupKFold

from eeg_emotion import config as cfg
from eeg_emotion.data import load_seed_features, zheng_sessions, zscore_train_test
from eeg_emotion.models import EmotionANN
from eeg_emotion.training import set_seed, n_params, compute_metrics, run_dir, save_run

EXPERIMENT = "01_zheng_lu_ann_svm"

# ============================================================
# HIPERPARAMETROS
# ============================================================
# ANN (los mismos que 02_ann_cross_subject.py)
HIDDEN       = 128
EPOCHS       = 50
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.3

# SVM lineal
SVM_C_GRID   = 2.0 ** np.arange(-10, 11, 2)
SVM_CV_FOLDS = 3
SVM_MAX_ITER = 20000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", type=int, choices=[4, 62], default=4)
    p.add_argument("--sessions", type=int, choices=[2, 3], default=2,
                   help="sesiones por sujeto: 2 como el paper (30 experimentos) o 3 (45)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=EPOCHS,
                   help="solo para pruebas rapidas (por defecto %(default)s)")
    p.add_argument("--out", default=None, help="carpeta de salida (por defecto en results/)")
    return p.parse_args()


# ============================================================
# MODELOS
# ============================================================
def run_ann(X_tr, y_tr, X_te, seed, epochs, device):
    """Entrena una EmotionANN nueva durante `epochs` epocas y predice el test."""
    torch.manual_seed(seed)
    model = EmotionANN(input_dim=X_tr.shape[1], hidden=HIDDEN, num_classes=3,
                       dropout=DROPOUT).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss_fn(model(x), y).backward()
            optim.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_te).to(device)).argmax(1).cpu().numpy()
    return preds, n_params(model)


def run_svm(X_tr, y_tr, trial_tr, X_te):
    """C por validacion cruzada agrupada por trials dentro del train (no se mira el test)."""
    grid = GridSearchCV(LinearSVC(max_iter=SVM_MAX_ITER),
                        {"C": SVM_C_GRID},
                        cv=GroupKFold(n_splits=SVM_CV_FOLDS))
    grid.fit(X_tr, y_tr, groups=trial_tr)
    return grid.predict(X_te), float(grid.best_params_["C"])


def summarize(per_session, y_true_all, y_pred_all):
    """Media +- std entre sesiones (como el paper) y matriz de confusion acumulada."""
    acc = np.array([s["accuracy"] for s in per_session])
    f1  = np.array([s["f1_macro"] for s in per_session])
    total = compute_metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))
    return {
        "accuracy": float(acc.mean()), "accuracy_std_sesiones": float(acc.std()),
        "f1_macro": float(f1.mean()), "f1_macro_std_sesiones": float(f1.std()),
        "confusion_matrix": total["confusion_matrix"], "n_test": total["n_test"],
    }


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    set_seed(args.seed)
    device = cfg.DEVICE
    tag = f"{args.channels}ch" + ("" if args.sessions == 2 else f"_{args.sessions}ses")
    out_dir = args.out or run_dir(EXPERIMENT, tag, args.seed)
    sessions = zheng_sessions(args.sessions)
    print(f"Dispositivo: {device}")
    print(f"{len(sessions)} experimentos ({args.sessions} sesiones x sujeto), "
          f"{args.channels} canales, {cfg.FEATURE_KEY}\n")

    print(f"{'Suj':>4} {'Ses':>4} {'ANN':>8} {'SVM':>8} {'C':>10}")
    print("-" * 39)
    rows = []
    res = {"EmotionANN": ([], [], []), "LinearSVC": ([], [], [])}   # (por sesion, true, pred)
    for subj, ses, fp in sessions:
        X, meta = load_seed_features(args.channels, files=[fp])
        X, y, trial = X.astype(np.float32), meta["y"], meta["trial"]
        tr, te = np.isin(trial, cfg.ZHENG_TRAIN_TRIALS), np.isin(trial, cfg.ZHENG_TEST_TRIALS)
        X_tr, X_te = zscore_train_test(X[tr], X[te])

        pred_ann, n_par = run_ann(X_tr, y[tr], X_te, args.seed, args.epochs, device)
        pred_svm, best_c = run_svm(X_tr, y[tr], trial[tr], X_te)

        row = {"subject": subj, "session": ses, "file": os.path.basename(fp),
               "n_train": int(tr.sum()), "n_test": int(te.sum())}
        for name, pred in (("EmotionANN", pred_ann), ("LinearSVC", pred_svm)):
            m = compute_metrics(y[te], pred)
            per_session, true_all, pred_all = res[name]
            per_session.append({"subject": subj, "session": ses,
                                "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
            true_all.append(y[te])
            pred_all.append(pred)
            row[f"acc_{name}"], row[f"f1_{name}"] = m["accuracy"], m["f1_macro"]
        row["svm_C"] = best_c
        rows.append(row)
        print(f"{subj:4d} {ses:4d} {row['acc_EmotionANN']*100:7.2f}% "
              f"{row['acc_LinearSVC']*100:7.2f}% {best_c:10.4g}")

    modelos = {name: {**summarize(*res[name]), "por_sesion": res[name][0]} for name in res}
    print("-" * 39)
    for name, m in modelos.items():
        print(f"{name:10s}: {m['accuracy']*100:.2f} % +- {m['accuracy_std_sesiones']*100:.2f}"
              f"   (F1 macro {m['f1_macro']*100:.2f} % +- {m['f1_macro_std_sesiones']*100:.2f})")
    print("\nPaper (Zheng & Lu 2015): 4 canales 82.88 % +- 10.92 | "
          "62 canales SVM 83.99 % | DBN 86.08 %")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sessions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    config = {
        "experiment": EXPERIMENT, "models": list(modelos), "channels": args.channels,
        "sessions_per_subject": args.sessions, "n_experiments": len(sessions),
        "seed": args.seed, "feature_key": cfg.FEATURE_KEY,
        "train_trials": cfg.ZHENG_TRAIN_TRIALS, "test_trials": cfg.ZHENG_TEST_TRIALS,
        "normalization": "z-score con estadisticas del train de cada sesion",
        "ann": {"hidden": HIDDEN, "epochs": args.epochs, "batch_size": BATCH_SIZE,
                "lr": LR, "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT,
                "class_weights": False, "drop_last": True, "n_params": n_par},
        "svm": {"C_grid": SVM_C_GRID.tolist(), "cv": f"GroupKFold({SVM_CV_FOLDS}) por trial",
                "max_iter": SVM_MAX_ITER},
        "device": device, "torch_threads": torch.get_num_threads(),
    }
    save_run(out_dir, config, {"seed": args.seed, "modelos": modelos})


if __name__ == "__main__":
    main()
