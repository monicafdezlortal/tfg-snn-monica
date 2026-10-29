"""
Entrena una SNN con SpikingJelly sobre el dataset SEED preprocesado.
Split cross-subject held-out: sujetos 1-3 -> test, sujetos 4-15 -> train/val.
La normalizacion se recalcula solo sobre train para evitar data leakage.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

from spikingjelly.activation_based import neuron, functional, surrogate, layer

# ============================================================
# CONFIG
# ============================================================
NPZ_PATH    = "data/seed_snn_dataset.npz"
CKPT_PATH   = "data/best_snn.pth"

T           = 50       # pasos temporales (debe coincidir con build_dataset.py)
HIDDEN      = 128
EPOCHS      = 50
BATCH_SIZE  = 256
LR          = 1e-3
WEIGHT_DECAY= 1e-4
DROPOUT     = 0.3
VAL_FRAC    = 0.15     # fraccion del train que va a validacion
TEST_SUBJECTS = [1, 2, 3]   # sujetos reservados como test (cross-subject)
SEED        = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DATASET
# ============================================================
class SpikeDataset(Dataset):
    """
    Genera spikes por rate coding en cada __getitem__ (no en disco).
    Esto actua como data augmentation estocastica en cada epoch.
    """
    def __init__(self, X_norm: np.ndarray, y: np.ndarray, T: int):
        self.X = torch.tensor(X_norm, dtype=torch.float32)  # (N, 20)
        self.y = torch.tensor(y,      dtype=torch.long)
        self.T = T

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]                                      # (20,)
        # Bernoulli: spike si random < probabilidad = valor normalizado
        rand   = torch.rand(self.T, x.shape[0])             # (T, 20)
        spikes = (rand < x.unsqueeze(0)).float()             # (T, 20)
        return spikes, self.y[idx]


# ============================================================
# MODELO
# ============================================================
class EmotionSNN(nn.Module):
    """
    SNN de 2 capas ocultas con neuronas LIF.
    Fixes aplicados:
    - BatchNorm antes de cada LIF para que la entrada este en rango util
    - v_threshold=0.5 para que las neuronas disparen (con umbral=1.0 y
      pesos Kaiming la membrana raramente llega al umbral -> neuronas muertas)
    - Sin LIF en la capa de salida: CrossEntropyLoss espera logits reales,
      no spikes binarios
    Entrada : (T, B, 20) spike trains
    Salida  : (B, 3)     logits
    """
    def __init__(self, input_dim=20, hidden=128, num_classes=3):
        super().__init__()

        self.fc1   = layer.Linear(input_dim, hidden)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.lif1  = neuron.LIFNode(tau=2.0, v_threshold=0.5,
                                    surrogate_function=surrogate.ATan())
        self.drop1 = nn.Dropout(DROPOUT)

        self.fc2   = layer.Linear(hidden, hidden // 2)
        self.bn2   = nn.BatchNorm1d(hidden // 2)
        self.lif2  = neuron.LIFNode(tau=2.0, v_threshold=0.5,
                                    surrogate_function=surrogate.ATan())
        self.drop2 = nn.Dropout(DROPOUT)

        self.fc3   = layer.Linear(hidden // 2, num_classes)
        # Sin LIF aqui: la salida son logits directos para CrossEntropyLoss

    def forward(self, x_seq):
        # x_seq : (T, B, 20)
        spike_acc = None
        for t in range(x_seq.shape[0]):
            h = self.drop1(self.lif1(self.bn1(self.fc1(x_seq[t]))))
            h = self.drop2(self.lif2(self.bn2(self.fc2(h))))
            h = self.fc3(h)
            spike_acc = h if spike_acc is None else spike_acc + h
        return spike_acc / x_seq.shape[0]               # (B, num_classes)


# ============================================================
# HELPERS
# ============================================================
def collate_fn(batch):
    """Convierte lista de (T,20) y labels en tensores (T,B,20) y (B,)."""
    spikes, labels = zip(*batch)
    spikes = torch.stack(spikes, dim=1)   # (T, B, 20)
    labels = torch.stack(labels, dim=0)   # (B,)
    return spikes, labels


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
        functional.reset_net(model)
    preds = torch.cat(all_preds).numpy()
    true  = torch.cat(all_true).numpy()
    acc   = (preds == true).mean()
    return float(np.mean(losses)), float(acc), preds, true


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Dispositivo: {DEVICE}\n")

    # --- Cargar datos ---
    data    = np.load(NPZ_PATH)
    # Reconstruir valores DE brutos (los guardados estan normalizados globalmente)
    X_norm_global = data["X_features"].astype(np.float64)
    g_min = data["norm_min"].astype(np.float64)
    g_max = data["norm_max"].astype(np.float64)
    X_raw = X_norm_global * (g_max - g_min) + g_min   # DE originales

    y       = data["y"]        # (N,)
    subject = data["subject"]  # (N,)

    # --- Normalizacion por sujeto ---
    # Cada sujeto se normaliza con su propia media y std por feature.
    # Los sujetos de test usan sus propias estadisticas -> sin leakage.
    # Tras z-score se mapea a [0,1] clipando en ±4σ para el rate coding.
    X_subj = np.zeros_like(X_raw, dtype=np.float32)
    for s in np.unique(subject):
        mask  = subject == s
        x_s   = X_raw[mask]
        mu    = x_s.mean(axis=0, keepdims=True)
        sigma = x_s.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        z     = (x_s - mu) / sigma                    # z-score del sujeto
        X_subj[mask] = ((np.clip(z, -4, 4) + 4) / 8).astype(np.float32)

    print(f"Normalizacion por sujeto: rango global [{X_subj.min():.3f}, {X_subj.max():.3f}]")

    # --- Split cross-subject held-out ---
    test_mask  = np.isin(subject, TEST_SUBJECTS)
    train_mask = ~test_mask

    X_trainval, y_trainval = X_subj[train_mask], y[train_mask]
    X_test,     y_test     = X_subj[test_mask],  y[test_mask]

    train_subjs = sorted(np.unique(subject[train_mask]).tolist())
    test_subjs  = sorted(np.unique(subject[test_mask]).tolist())
    print(f"Train sujetos : {train_subjs}")
    print(f"Test  sujetos : {test_subjs}")
    print(f"Train+val     : {len(y_trainval)} muestras")
    print(f"Test          : {len(y_test)} muestras")

    # --- Dataset y DataLoaders ---
    full_ds = SpikeDataset(X_trainval, y_trainval, T)
    n_val   = int(len(full_ds) * VAL_FRAC)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    test_ds  = SpikeDataset(X_test, y_test, T)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    print(f"  train={n_train}  val={n_val}  test={len(test_ds)}\n")

    # --- Class weights (penaliza mas los errores en clases minoritarias en train) ---
    counts  = np.bincount(y_trainval, minlength=3).astype(np.float32)
    weights = (counts.sum() / (3.0 * counts))          # inverso de frecuencia
    weights = weights / weights.mean()                  # escala relativa a 1.0
    w_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"\nClass weights -> negativa={weights[0]:.3f}  "
          f"neutra={weights[1]:.3f}  positiva={weights[2]:.3f}")

    # --- Modelo ---
    model   = EmotionSNN(input_dim=20, hidden=HIDDEN, num_classes=3).to(DEVICE)
    optim   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Reduce LR x0.5 si val_acc no mejora en 7 epochs; minimo 1e-5
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optim, mode='max', factor=0.5, patience=7, min_lr=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=w_tensor)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametros entrenables: {n_params:,}\n")

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
            logits = model(x)
            loss   = loss_fn(logits, y_batch)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
            functional.reset_net(model)

        val_loss, val_acc, _, _ = evaluate(model, val_loader)
        sched.step(val_acc)   # ReduceLROnPlateau necesita la metrica
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
