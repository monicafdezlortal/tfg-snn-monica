"""
Random search de hiperparametros sobre la SNN SEED.
- 15 combinaciones aleatorias
- Max 20 epochs por combinacion, early stopping patience=5 sobre val_loss
- Resultados en data/hparam_results.csv
- Reutiliza la normalizacion por sujeto y el split de train_snn.py
- La mejor configuracion se elige por val_acc (sujetos de validacion);
  test_acc solo se reporta, no se usa para elegir
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spikingjelly.activation_based import neuron, functional, surrogate, layer

# Reutilizamos constantes y helpers de train_snn
from train_snn import (
    SpikeDataset, collate_fn, evaluate,
    NPZ_PATH, TEST_SUBJECTS, VAL_SUBJECTS, T, SEED, BATCH_SIZE, DEVICE,
)

CSV_PATH   = "data/hparam_results.csv"
N_TRIALS   = 15
MAX_EPOCHS = 20
PATIENCE   = 5     # early stopping sobre val_loss
RNG        = np.random.default_rng(SEED)


# ============================================================
# MODELO FLEXIBLE (1 o 2 capas ocultas)
# ============================================================
class FlexSNN(nn.Module):
    def __init__(self, input_dim, hidden, n_layers, v_threshold, tau, dropout):
        super().__init__()
        self.fcs   = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.lifs  = nn.ModuleList()
        self.drops = nn.ModuleList()

        in_dim = input_dim
        for _ in range(n_layers):
            self.fcs.append(layer.Linear(in_dim, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
            self.lifs.append(neuron.LIFNode(
                tau=float(tau),
                v_threshold=float(v_threshold),
                surrogate_function=surrogate.ATan()
            ))
            self.drops.append(nn.Dropout(float(dropout)))
            in_dim = hidden

        self.out = layer.Linear(in_dim, 3)

    def forward(self, x_seq):
        acc = None
        for t in range(x_seq.shape[0]):
            h = x_seq[t]
            for fc, bn, lif, drop in zip(self.fcs, self.bns, self.lifs, self.drops):
                h = drop(lif(bn(fc(h))))
            h = self.out(h)
            acc = h if acc is None else acc + h
        return acc / x_seq.shape[0]


# ============================================================
# CARGA DE DATOS (igual que train_snn, una sola vez)
# ============================================================
def load_data():
    data          = np.load(NPZ_PATH)
    X_norm_global = data["X_features"].astype(np.float64)
    g_min         = data["norm_min"].astype(np.float64)
    g_max         = data["norm_max"].astype(np.float64)
    X_raw         = X_norm_global * (g_max - g_min) + g_min

    y       = data["y"]
    subject = data["subject"]

    X_subj = np.zeros_like(X_raw, dtype=np.float32)
    for s in np.unique(subject):
        mask  = subject == s
        x_s   = X_raw[mask]
        mu    = x_s.mean(axis=0, keepdims=True)
        sigma = x_s.std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        z     = (x_s - mu) / sigma
        X_subj[mask] = ((np.clip(z, -4, 4) + 4) / 8).astype(np.float32)

    test_mask  = np.isin(subject, TEST_SUBJECTS)
    val_mask   = np.isin(subject, VAL_SUBJECTS)
    train_mask = ~(test_mask | val_mask)

    X_train, y_train = X_subj[train_mask], y[train_mask]
    X_val,   y_val   = X_subj[val_mask],   y[val_mask]
    X_test,  y_test  = X_subj[test_mask],  y[test_mask]

    counts  = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = (counts.sum() / (3.0 * counts))
    weights = weights / weights.mean()
    w_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    return X_train, y_train, X_val, y_val, X_test, y_test, w_tensor


# ============================================================
# ENTRENAMIENTO DE UN TRIAL
# ============================================================
def run_trial(hp, X_train, y_train, X_val, y_val, X_test, y_test, w_tensor):
    torch.manual_seed(SEED)

    train_ds = SpikeDataset(X_train, y_train, T)
    val_ds   = SpikeDataset(X_val,   y_val,   T)
    test_ds  = SpikeDataset(X_test,  y_test,  T)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model   = FlexSNN(
        input_dim=20,
        hidden=hp["hidden"],
        n_layers=hp["n_layers"],
        v_threshold=hp["v_threshold"],
        tau=hp["tau"],
        dropout=hp["dropout"],
    ).to(DEVICE)

    optim   = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=w_tensor)

    best_val_loss  = float("inf")
    best_state     = None
    epochs_no_impr = 0
    epochs_done    = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y_batch in train_loader:
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
            optim.zero_grad()
            loss = loss_fn(model(x), y_batch)
            loss.backward()
            optim.step()
            functional.reset_net(model)

        val_loss, val_acc, _, _ = evaluate(model, val_loader)
        epochs_done = epoch

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_impr = 0
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, val_acc,  _,    _    = evaluate(model, val_loader)
    _, test_acc, preds, true = evaluate(model, test_loader)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        **hp,
        "n_params":   n_params,
        "epochs_run": epochs_done,
        "val_acc":    round(val_acc,  4),
        "test_acc":   round(test_acc, 4),
    }


# ============================================================
# ESPACIO DE BUSQUEDA Y MUESTREO
# ============================================================
def sample_hp():
    return {
        "hidden":      int(RNG.integers(64, 257)),          # 64..256
        "n_layers":    int(RNG.integers(1, 3)),             # 1 o 2
        "v_threshold": round(float(RNG.uniform(0.5, 1.5)), 3),
        "tau":         round(float(RNG.uniform(1.1, 4.0)), 3),  # LIFNode exige tau > 1
        "dropout":     round(float(RNG.uniform(0.1, 0.4)), 3),
        "lr":          float(np.exp(RNG.uniform(np.log(5e-4), np.log(5e-3)))),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("Cargando datos...")
    X_tr, y_tr, X_va, y_va, X_te, y_te, w = load_data()
    print(f"  Train: {len(y_tr)}  Val: {len(y_va)}  Test: {len(y_te)}\n")

    os.makedirs("data", exist_ok=True)
    fieldnames = ["trial", "hidden", "n_layers", "v_threshold", "tau",
                  "dropout", "lr", "n_params", "epochs_run", "val_acc", "test_acc"]

    # --- Reanudacion: cargar trials ya completados del CSV ---
    results       = []
    completed_ids = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                results.append({k: (int(v) if k in ("trial","hidden","n_layers","epochs_run","n_params")
                                    else float(v)) for k, v in row.items()})
                completed_ids.add(int(row["trial"]))
        print(f"Reanudando: {len(completed_ids)} trials ya completados "
              f"({sorted(completed_ids)})\n")

    write_mode = "a" if completed_ids else "w"
    with open(CSV_PATH, write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not completed_ids:
            writer.writeheader()

        for trial in range(1, N_TRIALS + 1):
            hp = sample_hp()   # el RNG es determinista: misma semilla -> mismos hp
            if trial in completed_ids:
                print(f"Trial {trial:2d}/{N_TRIALS}  [ya completado, saltando]")
                continue

            print(f"Trial {trial:2d}/{N_TRIALS}  "
                  f"hidden={hp['hidden']}  layers={hp['n_layers']}  "
                  f"thr={hp['v_threshold']}  tau={hp['tau']}  "
                  f"drop={hp['dropout']}  lr={hp['lr']:.2e}  ...",
                  end="", flush=True)

            t0  = time.time()
            row = run_trial(hp, X_tr, y_tr, X_va, y_va, X_te, y_te, w)
            row["trial"] = trial
            elapsed = time.time() - t0

            print(f"  val={row['val_acc']:.4f}  test={row['test_acc']:.4f}  "
                  f"ep={row['epochs_run']}  {elapsed:.0f}s")

            writer.writerow({k: row[k] for k in fieldnames})
            f.flush()
            results.append(row)

    # --- Top 5 por val_acc (el test no se usa para elegir) ---
    results.sort(key=lambda r: r["val_acc"], reverse=True)
    print(f"\n{'='*70}")
    print(f"TOP 5 por val_acc  (resultados completos en {CSV_PATH})")
    print(f"{'='*70}")
    hdr = f"{'#':>3} {'test':>6} {'val':>6} {'hidden':>7} {'layers':>7} "
    hdr += f"{'thr':>5} {'tau':>5} {'drop':>5} {'lr':>8} {'ep':>4}"
    print(hdr)
    print("-" * 70)
    for i, r in enumerate(results[:5], 1):
        print(f"{i:3d} {r['test_acc']:6.4f} {r['val_acc']:6.4f} "
              f"{r['hidden']:7d} {r['n_layers']:7d} "
              f"{r['v_threshold']:5.2f} {r['tau']:5.2f} "
              f"{r['dropout']:5.2f} {r['lr']:8.2e} {r['epochs_run']:4d}")

    best = results[0]
    print(f"\nMejor configuracion:")
    for k, v in best.items():
        if k != "trial":
            print(f"  {k:12s}: {v}")


if __name__ == "__main__":
    main()
