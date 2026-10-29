"""
Random search de hiperparametros para la ANN (MLP) sobre SEED de_LDS.

Protocolo (sin tocar el test hasta el final):
  - Sujetos 1-3 -> test, reservados. Solo se usan una vez al final.
  - Sujetos 4-15 -> validacion cruzada por sujetos en 4 folds de 3 sujetos.
    Cada configuracion se entrena 4 veces (una por fold) y se puntua con la
    media de val_acc. Con un solo par de sujetos de validacion el ranking
    dependeria mucho de que sujetos tocasen.
  - La mejor configuracion (por val_acc media) se reentrena con los 12
    sujetos durante la media de epochs optimas de los folds, con 3 semillas,
    y se evalua en test.

Resultados:
  results/ann_hparam_results.csv  una fila por configuracion (se puede reanudar)
  results/ann_best.txt            resumen de la mejor configuracion + test
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn

import train_ann as base
from train_ann import load_seed_features, normalize_per_subject, CHANNEL_INDICES

CSV_PATH     = "results/ann_hparam_results.csv"
SUMMARY_PATH = "results/ann_best.txt"
N_TRIALS     = 40
MAX_EPOCHS   = 40
PATIENCE     = 6        # early stopping sobre val_acc
TEST_SUBJECTS = [1, 2, 3]
FOLDS = [[4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
FINAL_SEEDS  = [0, 1, 2]
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

FIELDS = ["trial", "channels", "hidden", "n_layers", "dropout", "lr",
          "weight_decay", "batch_size", "n_params", "best_epochs",
          "val_acc_mean", "val_acc_std", "fold_accs", "seconds"]


# ============================================================
# MODELO FLEXIBLE
# ============================================================
class FlexANN(nn.Module):
    """n_layers bloques Linear -> BN -> ReLU -> Dropout, anchura hidden,
    hidden//2, hidden//4... (igual patron que EmotionANN) y Linear final."""
    def __init__(self, input_dim, hidden, n_layers, dropout):
        super().__init__()
        layers, in_dim = [], input_dim
        for i in range(n_layers):
            out_dim = max(hidden // (2 ** i), 16)
            layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# ENTRENAMIENTO
# ============================================================
def class_weights(y):
    counts = np.bincount(y, minlength=3).astype(np.float32)
    w = counts.sum() / (3.0 * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def accuracy(model, X, y, bs=4096):
    model.eval()
    preds = torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)])
    return (preds == y).float().mean().item()


def train_model(hp, X_tr, y_tr, X_val=None, y_val=None, epochs=None, seed=SEED):
    """Entrena con early stopping si hay val. Devuelve (modelo, best_val_acc, best_epoch)."""
    epochs = epochs or MAX_EPOCHS
    torch.manual_seed(seed)
    model = FlexANN(X_tr.shape[1], hp["hidden"], hp["n_layers"], hp["dropout"]).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_tr.cpu().numpy()))
    bs = hp["batch_size"]

    best_acc, best_epoch, best_state, no_impr = -1.0, 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr), device=DEVICE)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:          # BatchNorm necesita >1 muestra
                continue
            optim.zero_grad()
            loss_fn(model(X_tr[idx]), y_tr[idx]).backward()
            optim.step()

        if X_val is None:
            continue
        acc = accuracy(model, X_val, y_val)
        if acc > best_acc:
            best_acc, best_epoch, no_impr = acc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_impr += 1
            if no_impr >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc, best_epoch


# ============================================================
# ESPACIO DE BUSQUEDA
# ============================================================
def sample_hp(rng):
    return {
        "channels":     str(rng.choice(["4ch", "62ch"])),
        "hidden":       int(rng.choice([32, 64, 128, 256, 512])),
        "n_layers":     int(rng.integers(1, 4)),                           # 1..3
        "dropout":      round(float(rng.uniform(0.0, 0.6)), 3),
        "lr":           float(np.exp(rng.uniform(np.log(1e-4), np.log(3e-3)))),
        "weight_decay": float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-2)))),
        "batch_size":   int(rng.choice([128, 256, 512])),
    }


# Trial 0 = configuracion actual de train_ann.py, como referencia
BASELINE_HP = {"channels": "4ch", "hidden": 128, "n_layers": 2, "dropout": 0.3,
               "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 256}


def parse_row(row):
    hp = {"channels": row["channels"], "hidden": int(row["hidden"]),
          "n_layers": int(row["n_layers"]), "dropout": float(row["dropout"]),
          "lr": float(row["lr"]), "weight_decay": float(row["weight_decay"]),
          "batch_size": int(row["batch_size"])}
    return {**row, **hp, "trial": int(row["trial"]),
            "val_acc_mean": float(row["val_acc_mean"]),
            "best_epochs": [int(e) for e in row["best_epochs"].split("|")]}


# ============================================================
# MAIN
# ============================================================
def main():
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"Dispositivo: {DEVICE}  hilos: {torch.get_num_threads()}")

    # Carga los 62 canales una vez; la variante 4ch son columnas de esta matriz
    base.USE_ALL_CHANNELS = True
    X_raw, y, subject = load_seed_features(base.DATA_DIR)
    X_all = normalize_per_subject(X_raw, subject)                    # (N, 310)
    cols_4ch = [c * 5 + b for c in CHANNEL_INDICES for b in range(5)]
    X_by_ch = {"62ch": torch.from_numpy(X_all).to(DEVICE),
               "4ch":  torch.from_numpy(np.ascontiguousarray(X_all[:, cols_4ch])).to(DEVICE)}
    y_t = torch.from_numpy(y).to(DEVICE)
    print(f"Datos: {X_all.shape[0]} muestras  (4ch=20 / 62ch=310 features)\n")

    fold_masks = [np.isin(subject, f) for f in FOLDS]
    pool_mask  = np.isin(subject, sum(FOLDS, []))
    test_mask  = np.isin(subject, TEST_SUBJECTS)

    # --- Reanudacion ---
    results = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            results = [parse_row(r) for r in csv.DictReader(f)]
        print(f"Reanudando: {len(results)} trials ya completados\n")
    done = {r["trial"] for r in results}

    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(SEED)
    with open(CSV_PATH, "a" if done else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            writer.writeheader()

        for trial in range(N_TRIALS + 1):
            hp = BASELINE_HP if trial == 0 else sample_hp(rng)   # RNG determinista
            if trial in done:
                continue

            t0 = time.time()
            X = X_by_ch[hp["channels"]]
            fold_accs, best_epochs = [], []
            for k, val_mask in enumerate(fold_masks):
                tr_mask = pool_mask & ~val_mask
                model, acc, ep = train_model(hp, X[tr_mask], y_t[tr_mask],
                                             X[val_mask], y_t[val_mask], seed=SEED + k)
                fold_accs.append(acc)
                best_epochs.append(ep)

            n_params = sum(p.numel() for p in model.parameters())
            row = {**hp, "trial": trial, "n_params": n_params,
                   "best_epochs": "|".join(map(str, best_epochs)),
                   "val_acc_mean": round(float(np.mean(fold_accs)), 4),
                   "val_acc_std":  round(float(np.std(fold_accs)), 4),
                   "fold_accs":    "|".join(f"{a:.4f}" for a in fold_accs),
                   "seconds":      round(time.time() - t0)}
            writer.writerow({k: row[k] for k in FIELDS})
            f.flush()
            results.append({**row, "best_epochs": best_epochs})

            print(f"Trial {trial:2d}/{N_TRIALS}  {hp['channels']:>4} h={hp['hidden']:3d} "
                  f"L={hp['n_layers']} drop={hp['dropout']:.2f} lr={hp['lr']:.1e} "
                  f"wd={hp['weight_decay']:.1e} bs={hp['batch_size']:3d}  "
                  f"val={row['val_acc_mean']:.4f}±{row['val_acc_std']:.4f}  "
                  f"ep={row['best_epochs']}  {row['seconds']}s", flush=True)

    # --- Ranking por val_acc media (el test no interviene) ---
    results.sort(key=lambda r: r["val_acc_mean"], reverse=True)
    print(f"\n{'=' * 70}\nTOP 5 por val_acc media (CV 4 folds por sujeto)\n{'=' * 70}")
    for i, r in enumerate(results[:5], 1):
        print(f"{i}. trial {r['trial']:2d}  val={r['val_acc_mean']:.4f}  {r['channels']} "
              f"h={r['hidden']} L={r['n_layers']} drop={r['dropout']} "
              f"lr={r['lr']:.1e} wd={r['weight_decay']:.1e} bs={r['batch_size']}")

    # --- Reentreno final de la mejor y de la baseline, evaluacion en test ---
    baseline = next(r for r in results if r["trial"] == 0)
    lines = []
    for name, r in [("MEJOR", results[0]), ("BASELINE (train_ann.py)", baseline)]:
        hp = {k: r[k] for k in BASELINE_HP}
        n_epochs = max(1, int(round(np.mean(r["best_epochs"]))))
        X = X_by_ch[hp["channels"]]
        test_accs = []
        for s in FINAL_SEEDS:
            model, _, _ = train_model(hp, X[pool_mask], y_t[pool_mask], epochs=n_epochs, seed=s)
            test_accs.append(accuracy(model, X[test_mask], y_t[test_mask]))
        lines += [f"{name}: trial {r['trial']}",
                  f"  hiperparametros : {hp}",
                  f"  val_acc CV      : {r['val_acc_mean']:.4f} ± {float(r['val_acc_std']):.4f}",
                  f"  epochs finales  : {n_epochs} (media de las optimas por fold)",
                  f"  TEST acc        : {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}  "
                  f"(semillas {FINAL_SEEDS}: {', '.join(f'{a:.4f}' for a in test_accs)})",
                  ""]

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Resumen guardado en {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
