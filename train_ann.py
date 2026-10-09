"""
Entrena una ANN (MLP) sobre las Extracted Features de SEED (de_LDS).
  - Mismas features: 4 canales (FT7, FT8, T7, T8) x 5 bandas = 20
    (o los 62 canales = 310 con USE_ALL_CHANNELS=True)
  - Normalizacion z-score por sujeto (sin leakage entre sujetos)
  - sujetos 1-3 -> test, 4-5 -> val, 6-15 -> train
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# CONFIG
# ============================================================
DATA_DIR    = ("data/ExtractedFeatures_1s" if os.path.isdir("data/ExtractedFeatures_1s")
               else "ExtractedFeatures_1s")
CKPT_PATH   = "results/best_ann.pth"
FEATURE_KEY = "de_LDS"

# Canales criticos segun Zheng & Lu 2015 (mismos que build_dataset.py)
CHANNEL_INDICES  = [14, 22, 23, 31]    # FT7, FT8, T7, T8
USE_ALL_CHANNELS = False               # True -> 62 canales x 5 bandas = 310

HIDDEN        = 128
EPOCHS        = 50
BATCH_SIZE    = 256
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3
VAL_SUBJECTS  = [4, 5]
TEST_SUBJECTS = [1, 2, 3]
SEED          = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# CARGA DE DATOS
# ============================================================
def load_seed_features(data_dir):
    """Devuelve X (N, F), y (N,), subject (N,) leyendo los .mat de SEED."""
    trial_labels = loadmat(os.path.join(data_dir, "label.mat"))["label"].flatten()
    labels_mapped = (trial_labels + 1).astype(np.int64)   # -1/0/1 -> 0/1/2

    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    mat_files = [f for f in mat_files if "label" not in os.path.basename(f)]

    all_X, all_y, all_subj = [], [], []
    for fp in mat_files:
        subj_id = int(os.path.basename(fp).split("_")[0])
        mat = loadmat(fp)
        for trial_idx in range(15):
            key = f"{FEATURE_KEY}{trial_idx + 1}"
            if key not in mat:
                print(f"  AVISO: {key} no encontrado en {os.path.basename(fp)}")
                continue
            de = mat[key]                                  # (62, N_seg, 5)
            if not USE_ALL_CHANNELS:
                de = de[CHANNEL_INDICES]                   # (4, N_seg, 5)
            de = de.transpose(1, 0, 2)                     # (N_seg, C, 5)
            n_seg = de.shape[0]
            all_X.append(de.reshape(n_seg, -1))
            all_y.append(np.full(n_seg, labels_mapped[trial_idx], dtype=np.int64))
            all_subj.append(np.full(n_seg, subj_id, dtype=np.int64))

    X = np.concatenate(all_X).astype(np.float64)
    y = np.concatenate(all_y)
    subject = np.concatenate(all_subj)
    return X, y, subject


def normalize_per_subject(X, subject):
    """z-score por sujeto y feature. Cada sujeto usa solo sus estadisticas."""
    X_norm = np.zeros_like(X, dtype=np.float32)
    for s in np.unique(subject):
        mask  = subject == s
        mu    = X[mask].mean(axis=0, keepdims=True)
        sigma = X[mask].std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        X_norm[mask] = ((X[mask] - mu) / sigma).astype(np.float32)
    return X_norm


# ============================================================
# MODELO
# ============================================================
class EmotionANN(nn.Module):
    """
    Equivalente no-spiking de EmotionSNN (train_snn.py):
    Linear -> BN -> ReLU -> Dropout  (x2)  -> Linear (logits)
    Entrada : (B, F) features continuas
    Salida  : (B, 3) logits
    """
    def __init__(self, input_dim=20, hidden=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# HELPERS
# ============================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_true, losses = [], [], []
    loss_fn = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        losses.append(loss_fn(logits, y).item())
        all_preds.append(logits.argmax(1).cpu())
        all_true.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    true  = torch.cat(all_true).numpy()
    acc   = (preds == true).mean()
    return float(np.mean(losses)), float(acc), preds, true


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Dispositivo: {DEVICE}\n")

    X_raw, y, subject = load_seed_features(DATA_DIR)
    X = normalize_per_subject(X_raw, subject)
    input_dim = X.shape[1]
    print(f"Features: {FEATURE_KEY}  X={X.shape}  "
          f"({'62 canales' if USE_ALL_CHANNELS else '4 canales'} x 5 bandas)")

    # --- Split cross-subject held-out ---
    test_mask  = np.isin(subject, TEST_SUBJECTS)
    val_mask   = np.isin(subject, VAL_SUBJECTS)
    train_mask = ~(test_mask | val_mask)
    y_train    = y[train_mask]

    print(f"Train sujetos : {sorted(np.unique(subject[train_mask]).tolist())}")
    print(f"Val   sujetos : {sorted(np.unique(subject[val_mask]).tolist())}")
    print(f"Test  sujetos : {sorted(np.unique(subject[test_mask]).tolist())}")

    def make_ds(mask):
        return TensorDataset(torch.from_numpy(X[mask]), torch.from_numpy(y[mask]))

    train_ds, val_ds, test_ds = make_ds(train_mask), make_ds(val_mask), make_ds(test_mask)
    n_train, n_val = len(train_ds), len(val_ds)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    print(f"  train={n_train}  val={n_val}  test={len(test_ds)}")

    # --- Class weights (igual que train_snn.py) ---
    counts  = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = counts.sum() / (3.0 * counts)
    weights = weights / weights.mean()
    w_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    # --- Modelo ---
    model   = EmotionANN(input_dim=input_dim, hidden=HIDDEN, num_classes=3).to(DEVICE)
    optim   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optim, mode='max', factor=0.5, patience=7, min_lr=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=w_tensor)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros entrenables: {n_params:,}\n")

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    best_val_acc = 0.0
    best_epoch   = 0

    print(f"{'Ep':>4} {'train_loss':>10} {'val_loss':>9} {'val_acc':>8} {'lr':>8} {'best':>5}")
    print("-" * 55)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for x, y_batch in train_loader:
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
            optim.zero_grad()
            loss = loss_fn(model(x), y_batch)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        val_loss, val_acc, _, _ = evaluate(model, val_loader)
        sched.step(val_acc)
        current_lr = optim.param_groups[0]['lr']

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), CKPT_PATH)
            marker = " <--"

        print(f"{epoch:4d} {np.mean(train_losses):10.4f} {val_loss:9.4f} "
              f"{val_acc:8.4f} {current_lr:8.2e}{marker}")

    # --- Evaluacion final en test ---
    print(f"\nMejor val_acc: {best_val_acc:.4f} (epoch {best_epoch})")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

    test_loss, test_acc, preds, true = evaluate(model, test_loader)
    print(f"\nTEST  loss={test_loss:.4f}  acc={test_acc:.4f}")
    print()
    print(classification_report(true, preds,
                                target_names=["negativa", "neutra", "positiva"],
                                digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(true, preds))
    print(f"\nModelo guardado en: {CKPT_PATH}")


if __name__ == "__main__":
    main()
