"""
Replica el protocolo experimental de Zheng & Lu (2015) sobre SEED para poder
comparar nuestra ANN con los resultados del paper en igualdad de condiciones.

Protocolo del paper (subject-dependent, por sesion):
  - cada sesion de cada sujeto es un "experimento" independiente
  - trials 1-9 de la sesion -> train, trials 10-15 -> test
  - se entrena un modelo nuevo por sesion y se reporta media +- std
  - features DE suavizadas (de_LDS), aqui 4 canales (FT7, FT8, T7, T8) x 5 bandas

Modelos evaluados:
  - MLP: misma arquitectura que train_ann.py
  - SVM lineal: baseline del propio paper (tambien usan linear kernel)

Referencias del paper para comparar:
  - 4 canales:            82.88 % +- 10.92
  - 62 canales, SVM:      83.99 % +-  9.72
  - 62 canales, DBN:      86.08 % +-  8.34
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, GroupKFold

# ============================================================
# CONFIG
# ============================================================
DATA_DIR    = ("data/ExtractedFeatures_1s" if os.path.isdir("data/ExtractedFeatures_1s")
               else "ExtractedFeatures_1s")
FEATURE_KEY = "de_LDS"

CHANNEL_INDICES  = [14, 22, 23, 31]    # FT7, FT8, T7, T8
USE_ALL_CHANNELS = False               # True -> 62 canales x 5 bandas = 310

# El paper usa 2 sesiones por sujeto (30 experimentos).
# El SEED publico trae 3; con 2 replicamos el paper exactamente.
SESSIONS_PER_SUBJECT = 2
N_TRAIN_TRIALS       = 9

# Hiperparametros del MLP (los mismos que train_ann.py)
HIDDEN       = 128
EPOCHS       = 50        # fijo: en este protocolo no hay validacion, no se usa el test para elegir epoca
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.3
SEED         = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DATOS
# ============================================================
def list_sessions(data_dir):
    """Devuelve [(sujeto, n_sesion, ruta)] ordenando las sesiones de cada sujeto por fecha."""
    files = [f for f in sorted(glob.glob(os.path.join(data_dir, "*.mat")))
             if "label" not in os.path.basename(f)]
    by_subj = {}
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]      # p.ej. "1_20131027"
        subj, date = name.split("_")[0], name.split("_")[1]
        by_subj.setdefault(int(subj), []).append((date, fp))

    sessions = []
    for subj in sorted(by_subj):
        ordered = sorted(by_subj[subj])[:SESSIONS_PER_SUBJECT]
        for k, (_, fp) in enumerate(ordered, start=1):
            sessions.append((subj, k, fp))
    return sessions


def load_session(fp, labels):
    """X (N, F), y (N,), trial (N,) de una sola sesion."""
    mat = loadmat(fp)
    X, y, trial = [], [], []
    for t in range(15):
        de = mat[f"{FEATURE_KEY}{t + 1}"]                     # (62, N_seg, 5)
        if not USE_ALL_CHANNELS:
            de = de[CHANNEL_INDICES]                          # (4, N_seg, 5)
        de = de.transpose(1, 0, 2)                            # (N_seg, C, 5)
        n = de.shape[0]
        X.append(de.reshape(n, -1))
        y.append(np.full(n, labels[t], dtype=np.int64))
        trial.append(np.full(n, t, dtype=np.int64))
    return np.concatenate(X).astype(np.float32), np.concatenate(y), np.concatenate(trial)


def zscore(X_train, X_test):
    """Normaliza con estadisticas SOLO del train de la sesion (sin leakage)."""
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


# ============================================================
# MODELOS
# ============================================================
class EmotionANN(nn.Module):
    """Misma arquitectura que train_ann.py."""
    def __init__(self, input_dim=20, hidden=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def run_mlp(X_tr, y_tr, X_te, y_te):
    torch.manual_seed(SEED)
    model = EmotionANN(input_dim=X_tr.shape[1], hidden=HIDDEN).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    for _ in range(EPOCHS):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            loss_fn(model(x), y).backward()
            optim.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_te).to(DEVICE)).argmax(1).cpu().numpy()
    return float((preds == y_te).mean())


def run_svm(X_tr, y_tr, trial_tr, X_te, y_te):
    # C se elige con validacion cruzada por trials DENTRO del train (no se mira el test)
    grid = GridSearchCV(LinearSVC(max_iter=20000),
                        {"C": 2.0 ** np.arange(-10, 11, 2)},
                        cv=GroupKFold(n_splits=3))
    grid.fit(X_tr, y_tr, groups=trial_tr)
    return float((grid.predict(X_te) == y_te).mean())


# ============================================================
# MAIN
# ============================================================
def main():
    np.random.seed(SEED)
    labels = (loadmat(os.path.join(DATA_DIR, "label.mat"))["label"].flatten() + 1).astype(np.int64)
    sessions = list_sessions(DATA_DIR)
    print(f"Dispositivo: {DEVICE}")
    print(f"{len(sessions)} experimentos ({SESSIONS_PER_SUBJECT} sesiones x sujeto), "
          f"{'62' if USE_ALL_CHANNELS else '4'} canales, {FEATURE_KEY}\n")

    print(f"{'Suj':>4} {'Ses':>4} {'MLP':>8} {'SVM':>8}")
    print("-" * 28)
    acc_mlp, acc_svm = [], []
    for subj, ses, fp in sessions:
        X, y, trial = load_session(fp, labels)
        tr, te = trial < N_TRAIN_TRIALS, trial >= N_TRAIN_TRIALS
        X_tr, X_te = zscore(X[tr], X[te])

        a_mlp = run_mlp(X_tr, y[tr], X_te, y[te])
        a_svm = run_svm(X_tr, y[tr], trial[tr], X_te, y[te])
        acc_mlp.append(a_mlp)
        acc_svm.append(a_svm)
        print(f"{subj:4d} {ses:4d} {a_mlp*100:7.2f}% {a_svm*100:7.2f}%")

    print("-" * 28)
    print(f"MLP : {np.mean(acc_mlp)*100:.2f} % +- {np.std(acc_mlp)*100:.2f}")
    print(f"SVM : {np.mean(acc_svm)*100:.2f} % +- {np.std(acc_svm)*100:.2f}")
    print("\nPaper (Zheng & Lu 2015): 4 canales 82.88 % +- 10.92 | "
          "62 canales SVM 83.99 % | DBN 86.08 %")


if __name__ == "__main__":
    main()
