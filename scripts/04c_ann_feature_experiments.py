"""
Experimentos para mejorar la ANN cross-subject en SEED, por etapas (greedy):
  1. Features de entrada   : de_LDS, psd_LDS, de+psd, de+asimetrias (dasm, rasm)...
  2. Contexto temporal     : concatenar las k ventanas ANTERIORES del mismo trial
                             (causal: solo pasado, valido en tiempo real)
  3. Regularizacion        : ruido gaussiano en la entrada, mixup, label smoothing
Cada etapa parte de la mejor opcion de la anterior.

Protocolo igual que ann_hparam_search.py:
  - Sujetos 1-3 -> test, solo al final.
  - Sujetos 4-15 -> CV por sujetos en 4 folds de 3; se elige por val_acc media.
  - Final: la mejor combinacion y la referencia (de_LDS 62ch) se reentrenan
    con los 12 sujetos y se evaluan en test con 3 semillas, mas un ensemble
    de 5 semillas de la mejor.

Resultados:
  results/ann_experiments.csv       una fila por experimento (se puede reanudar)
  results/ann_experiments_best.txt  resumen final con test
"""

import os
import csv
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat

from ann_hparam_search import FlexANN, class_weights, FOLDS, TEST_SUBJECTS
from train_ann import DATA_DIR, normalize_per_subject

CSV_PATH     = "results/ann_experiments.csv"
SUMMARY_PATH = "results/ann_experiments_best.txt"
MAX_EPOCHS   = 40
PATIENCE     = 6
FINAL_SEEDS  = [0, 1, 2]
ENSEMBLE_SEEDS = [10, 11, 12, 13, 14]
SEED         = 42
N_THREADS    = 4        # deja CPU libre para ann_hparam_search.py
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Configuracion de red fija para comparar entradas (razonable segun la
# prueba inicial: 62 canales + lr bajo + dropout alto funcionaba mejor)
NET = {"hidden": 256, "n_layers": 2, "dropout": 0.5,
       "lr": 3e-4, "weight_decay": 1e-4, "batch_size": 256}

FEATURE_SETS = {
    "de":            ["de_LDS"],
    "psd":           ["psd_LDS"],
    "de+psd":        ["de_LDS", "psd_LDS"],
    "de+dasm+rasm":  ["de_LDS", "dasm_LDS", "rasm_LDS"],
    "de+asm+dcau":   ["de_LDS", "asm_LDS", "dcau_LDS"],
    "de_movingAve":  ["de_movingAve"],
}
CONTEXTS = [0, 2, 4, 9]          # n. de ventanas anteriores concatenadas
REGS = {
    "none":          {},
    "noise0.2":      {"noise": 0.2},
    "noise0.5":      {"noise": 0.5},
    "mixup0.2":      {"mixup": 0.2},
    "smooth0.1":     {"smooth": 0.1},
    "noise0.2+smooth0.1": {"noise": 0.2, "smooth": 0.1},
}

FIELDS = ["stage", "name", "features", "context", "reg", "input_dim",
          "val_acc_mean", "val_acc_std", "fold_accs", "best_epochs", "seconds"]


# ============================================================
# DATOS
# ============================================================
def load_all_features(keys):
    """Devuelve {key: X (N, C*5) z-score por sujeto}, y, subject, trial_id.
    trial_id identifica cada trial de cada sesion; dentro de un trial las
    ventanas quedan en orden temporal."""
    labels = loadmat(os.path.join(DATA_DIR, "label.mat"))["label"].flatten() + 1
    files = sorted(f for f in glob.glob(os.path.join(DATA_DIR, "*.mat"))
                   if "label" not in os.path.basename(f))
    feats = {k: [] for k in keys}
    ys, subjs, tids = [], [], []
    tid = 0
    for fp in files:
        subj_id = int(os.path.basename(fp).split("_")[0])
        mat = loadmat(fp)
        for t in range(1, 16):
            n = mat[f"de_LDS{t}"].shape[1]
            for k in keys:
                a = mat[f"{k}{t}"].transpose(1, 0, 2)          # (n, C, 5)
                feats[k].append(a.reshape(n, -1))
            ys.append(np.full(n, labels[t - 1], dtype=np.int64))
            subjs.append(np.full(n, subj_id, dtype=np.int64))
            tids.append(np.full(n, tid, dtype=np.int64))
            tid += 1
    y, subject, trial_id = map(np.concatenate, (ys, subjs, tids))
    out = {}
    for k in keys:
        X = np.concatenate(feats[k]).astype(np.float64)
        out[k] = normalize_per_subject(X, subject)
    return out, y, subject, trial_id


def add_context(X, trial_id, k):
    """Concatena [x_{t-k}, ..., x_{t-1}, x_t] dentro de cada trial.
    Al inicio del trial se repite la primera ventana (padding causal)."""
    if k == 0:
        return X
    starts = np.r_[0, np.flatnonzero(np.diff(trial_id)) + 1]
    pos = np.arange(len(X)) - np.repeat(starts, np.diff(np.r_[starts, len(X)]))
    offs = np.arange(k, -1, -1)                                # k..0
    idx = np.arange(len(X))[:, None] - np.minimum(offs[None, :], pos[:, None])
    return X[idx].reshape(len(X), -1)


# ============================================================
# ENTRENAMIENTO
# ============================================================
@torch.no_grad()
def predict_proba(model, X, bs=4096):
    model.eval()
    return torch.cat([F.softmax(model(X[i:i + bs]), 1) for i in range(0, len(X), bs)])


def train_model(X_tr, y_tr, X_val=None, y_val=None, reg=None, epochs=None, seed=SEED):
    reg = reg or {}
    epochs = epochs or MAX_EPOCHS
    torch.manual_seed(seed)
    model = FlexANN(X_tr.shape[1], NET["hidden"], NET["n_layers"], NET["dropout"]).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=NET["lr"], weight_decay=NET["weight_decay"])
    w = class_weights(y_tr.cpu().numpy())
    smooth = reg.get("smooth", 0.0)
    bs = NET["batch_size"]

    best_acc, best_epoch, best_state, no_impr = -1.0, 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr), device=DEVICE)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xb, yb = X_tr[idx], y_tr[idx]
            if reg.get("noise"):
                xb = xb + reg["noise"] * torch.randn_like(xb)
            optim.zero_grad()
            if reg.get("mixup"):
                lam = float(np.random.beta(reg["mixup"], reg["mixup"]))
                j = torch.randperm(len(xb), device=DEVICE)
                out = model(lam * xb + (1 - lam) * xb[j])
                loss = (lam * F.cross_entropy(out, yb, weight=w, label_smoothing=smooth)
                        + (1 - lam) * F.cross_entropy(out, yb[j], weight=w, label_smoothing=smooth))
            else:
                loss = F.cross_entropy(model(xb), yb, weight=w, label_smoothing=smooth)
            loss.backward()
            optim.step()

        if X_val is None:
            continue
        acc = (predict_proba(model, X_val).argmax(1) == y_val).float().mean().item()
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
# MAIN
# ============================================================
def main():
    torch.set_num_threads(N_THREADS)
    np.random.seed(SEED)
    all_keys = sorted({k for ks in FEATURE_SETS.values() for k in ks})
    print(f"Cargando {all_keys} ...", flush=True)
    feats, y, subject, trial_id = load_all_features(all_keys)
    y_t = torch.from_numpy(y).to(DEVICE)
    fold_masks = [np.isin(subject, f) for f in FOLDS]
    pool_mask  = np.isin(subject, sum(FOLDS, []))
    test_mask  = np.isin(subject, TEST_SUBJECTS)
    print(f"{len(y)} muestras\n", flush=True)

    def build_X(fset, ctx):
        X = np.concatenate([feats[k] for k in FEATURE_SETS[fset]], axis=1)
        X = add_context(X, trial_id, ctx)
        return torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(DEVICE)

    # --- Reanudacion ---
    done = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            for r in csv.DictReader(f):
                done[r["name"]] = r
        print(f"Reanudando: {len(done)} experimentos ya hechos\n")

    os.makedirs("results", exist_ok=True)
    f = open(CSV_PATH, "a" if done else "w", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if not done:
        writer.writeheader()

    def run_cv(stage, fset, ctx, reg_name):
        name = f"{fset}|ctx{ctx}|{reg_name}"
        if name in done:
            r = done[name]
            print(f"[{stage}] {name:40s} val={float(r['val_acc_mean']):.4f}  (ya hecho)")
            return float(r["val_acc_mean"]), [int(e) for e in r["best_epochs"].split("|")]
        t0 = time.time()
        X = build_X(fset, ctx)
        accs, eps = [], []
        for k, val_mask in enumerate(fold_masks):
            tr_mask = pool_mask & ~val_mask
            _, acc, ep = train_model(X[tr_mask], y_t[tr_mask], X[val_mask], y_t[val_mask],
                                     reg=REGS[reg_name], seed=SEED + k)
            accs.append(acc)
            eps.append(ep)
        row = {"stage": stage, "name": name, "features": fset, "context": ctx,
               "reg": reg_name, "input_dim": X.shape[1],
               "val_acc_mean": round(float(np.mean(accs)), 4),
               "val_acc_std": round(float(np.std(accs)), 4),
               "fold_accs": "|".join(f"{a:.4f}" for a in accs),
               "best_epochs": "|".join(map(str, eps)),
               "seconds": round(time.time() - t0)}
        writer.writerow(row)
        f.flush()
        done[name] = row
        print(f"[{stage}] {name:40s} val={row['val_acc_mean']:.4f}±{row['val_acc_std']:.4f}  "
              f"folds={row['fold_accs']}  ep={row['best_epochs']}  {row['seconds']}s", flush=True)
        return row["val_acc_mean"], eps

    # --- Etapa 1: features ---
    res = {fs: run_cv("1-features", fs, 0, "none") for fs in FEATURE_SETS}
    best_fs = max(res, key=lambda k: res[k][0])
    print(f"  -> mejor features: {best_fs}\n", flush=True)

    # --- Etapa 2: contexto temporal ---
    res = {c: run_cv("2-contexto", best_fs, c, "none") for c in CONTEXTS}
    best_ctx = max(res, key=lambda k: res[k][0])
    print(f"  -> mejor contexto: {best_ctx}\n", flush=True)

    # --- Etapa 3: regularizacion ---
    res = {r: run_cv("3-reg", best_fs, best_ctx, r) for r in REGS}
    best_reg = max(res, key=lambda k: res[k][0])
    best_val, best_eps = res[best_reg]
    print(f"  -> mejor regularizacion: {best_reg}\n", flush=True)
    f.close()

    # --- Final: test ---
    ref = done["de|ctx0|none"]
    ref_val = float(ref["val_acc_mean"])
    ref_eps = [int(e) for e in str(ref["best_epochs"]).split("|")]
    lines = []
    for label, fs, ctx, rg, val, eps in [
            ("MEJOR", best_fs, best_ctx, best_reg, best_val, best_eps),
            ("REFERENCIA de_LDS 62ch", "de", 0, "none", ref_val, ref_eps)]:
        X = build_X(fs, ctx)
        n_ep = max(1, int(round(np.mean(eps))))
        accs = []
        for s in FINAL_SEEDS:
            m, _, _ = train_model(X[pool_mask], y_t[pool_mask], reg=REGS[rg], epochs=n_ep, seed=s)
            accs.append((predict_proba(m, X[test_mask]).argmax(1) == y_t[test_mask]).float().mean().item())
        lines += [f"{label}: features={fs}  contexto={ctx}  reg={rg}  input_dim={X.shape[1]}",
                  f"  red             : {NET}",
                  f"  val_acc CV      : {val:.4f}",
                  f"  epochs finales  : {n_ep}",
                  f"  TEST acc        : {np.mean(accs):.4f} ± {np.std(accs):.4f}  "
                  f"({', '.join(f'{a:.4f}' for a in accs)})"]
        if label == "MEJOR":
            probs = sum(predict_proba(train_model(X[pool_mask], y_t[pool_mask], reg=REGS[rg],
                                                  epochs=n_ep, seed=s)[0], X[test_mask])
                        for s in ENSEMBLE_SEEDS)
            ens = (probs.argmax(1) == y_t[test_mask]).float().mean().item()
            lines.append(f"  TEST ensemble {len(ENSEMBLE_SEEDS)} semillas: {ens:.4f}")
        lines.append("")

    summary = "\n".join(lines)
    print(summary)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as fo:
        fo.write(summary)
    print(f"Resumen guardado en {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
