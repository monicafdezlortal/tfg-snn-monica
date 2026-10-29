"""
Experimento 04a - Random search de hiperparametros para la ANN (MLP) sobre SEED de_LDS.

Protocolo (sin tocar el test hasta el final):
  - Sujetos 1-3 -> test, reservados. Solo se usan una vez al final.
  - Sujetos 4-15 -> validacion cruzada por sujetos en 4 folds de 3 sujetos.
    Cada configuracion se entrena 4 veces (una por fold) y se puntua con la
    media de val_acc. Con un solo par de sujetos de validacion el ranking
    dependeria mucho de que sujetos tocasen.
  - La mejor configuracion (por val_acc media) se reentrena con los 12
    sujetos durante la media de epochs optimas de los folds, con 3 semillas,
    y se evalua en test.

--channels both (por defecto) busca tambien el numero de canales (4ch/62ch);
--channels 4|62 fuerza todos los trials a ese numero de canales (el resto de
hiperparametros sorteados no cambia).

Uso:
  python scripts/04a_ann_hparam_search.py
Salida (se puede reanudar):
  results/04a_ann_hparam_search/<both|4ch|62ch>_seed<N>/
    ann_hparam_results.csv  una fila por configuracion
    ann_best.txt            resumen de la mejor configuracion + test
    config.json, metrics.json, confusion_matrix.csv (de la mejor, semilla 0)
"""

import argparse
import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn

from eeg_emotion import config as cfg
from eeg_emotion.data import (load_seed_features, normalize_per_subject, channel_columns,
                              cv_fold_masks)
from eeg_emotion.models import FlexANN
from eeg_emotion.training import class_weights, compute_metrics, run_dir, save_run

EXPERIMENT   = "04a_ann_hparam_search"
N_TRIALS     = 40
MAX_EPOCHS   = 40
PATIENCE     = 6        # early stopping sobre val_acc
FINAL_SEEDS  = [0, 1, 2]
DEVICE       = cfg.DEVICE

FIELDS = ["trial", "channels", "hidden", "n_layers", "dropout", "lr",
          "weight_decay", "batch_size", "n_params", "best_epochs",
          "val_acc_mean", "val_acc_std", "fold_accs", "seconds"]


# ============================================================
# ENTRENAMIENTO
# ============================================================
@torch.no_grad()
def predict(model, X, bs=4096):
    model.eval()
    return torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)])


def accuracy(model, X, y, bs=4096):
    return (predict(model, X, bs) == y).float().mean().item()


def train_model(hp, X_tr, y_tr, X_val=None, y_val=None, epochs=None, seed=42,
                max_epochs=MAX_EPOCHS):
    """Entrena con early stopping si hay val. Devuelve (modelo, best_val_acc, best_epoch)."""
    epochs = epochs or max_epochs
    torch.manual_seed(seed)
    model = FlexANN(X_tr.shape[1], hp["hidden"], hp["n_layers"], hp["dropout"]).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y_tr.cpu().numpy(), DEVICE))
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


# Trial 0 = configuracion de 02_ann_cross_subject.py, como referencia
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", choices=["4", "62", "both"], default="both")
    p.add_argument("--seed", type=int, default=42,
                   help="semilla del sorteo de hiperparametros y de los folds")
    p.add_argument("--trials", type=int, default=N_TRIALS,
                   help="n. de configuraciones aleatorias (ademas de la baseline)")
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                   help="solo para pruebas rapidas (por defecto %(default)s)")
    p.add_argument("--out", default=None, help="carpeta de salida (por defecto en results/)")
    return p.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    seed = args.seed
    ch_tag = "both" if args.channels == "both" else f"{args.channels}ch"
    out_dir = args.out or run_dir(EXPERIMENT, ch_tag, seed)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ann_hparam_results.csv")
    summary_path = os.path.join(out_dir, "ann_best.txt")

    def force_channels(hp):
        return hp if ch_tag == "both" else {**hp, "channels": ch_tag}

    torch.set_num_threads(os.cpu_count() or 1)
    print(f"Dispositivo: {DEVICE}  hilos: {torch.get_num_threads()}")

    # Carga los 62 canales una vez; la variante 4ch son columnas de esta matriz
    X_raw, meta = load_seed_features(62)
    y, subject = meta["y"], meta["subject"]
    X_all = normalize_per_subject(X_raw, subject)                    # (N, 310)
    cols_4ch = channel_columns(cfg.CHANNEL_INDICES)
    X_by_ch = {"62ch": torch.from_numpy(X_all).to(DEVICE),
               "4ch":  torch.from_numpy(np.ascontiguousarray(X_all[:, cols_4ch])).to(DEVICE)}
    y_t = torch.from_numpy(y).to(DEVICE)
    print(f"Datos: {X_all.shape[0]} muestras  (4ch=20 / 62ch=310 features)\n")

    fold_masks, pool_mask = cv_fold_masks(subject)
    test_mask = np.isin(subject, cfg.TEST_SUBJECTS)

    # --- Reanudacion ---
    results = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            results = [parse_row(r) for r in csv.DictReader(f)]
        print(f"Reanudando: {len(results)} trials ya completados\n")
    done = {r["trial"] for r in results}

    rng = np.random.default_rng(seed)
    with open(csv_path, "a" if done else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            writer.writeheader()

        for trial in range(args.trials + 1):
            hp = BASELINE_HP if trial == 0 else sample_hp(rng)   # RNG determinista
            hp = force_channels(hp)
            if trial in done:
                continue

            t0 = time.time()
            X = X_by_ch[hp["channels"]]
            fold_accs, best_epochs = [], []
            for k, val_mask in enumerate(fold_masks):
                tr_mask = pool_mask & ~val_mask
                model, acc, ep = train_model(hp, X[tr_mask], y_t[tr_mask],
                                             X[val_mask], y_t[val_mask], seed=seed + k,
                                             max_epochs=args.max_epochs)
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

            print(f"Trial {trial:2d}/{args.trials}  {hp['channels']:>4} h={hp['hidden']:3d} "
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
    lines, final = [], {}
    y_test = y_t[test_mask]
    for name, r in [("MEJOR", results[0]), ("BASELINE (02_ann_cross_subject.py)", baseline)]:
        hp = {k: r[k] for k in BASELINE_HP}
        n_epochs = max(1, int(round(np.mean(r["best_epochs"]))))
        X = X_by_ch[hp["channels"]]
        test_accs, per_seed = [], []
        for s in FINAL_SEEDS:
            model, _, _ = train_model(hp, X[pool_mask], y_t[pool_mask], epochs=n_epochs, seed=s)
            preds = predict(model, X[test_mask])
            test_accs.append((preds == y_test).float().mean().item())
            per_seed.append({"seed": s, **compute_metrics(y_test.cpu().numpy(),
                                                          preds.cpu().numpy())})
        final[name] = {"trial": r["trial"], "hp": hp, "epochs": n_epochs,
                       "val_acc_cv": r["val_acc_mean"],
                       "test_acc_mean": float(np.mean(test_accs)),
                       "test_acc_std": float(np.std(test_accs)), "per_seed": per_seed}
        lines += [f"{name}: trial {r['trial']}",
                  f"  hiperparametros : {hp}",
                  f"  val_acc CV      : {r['val_acc_mean']:.4f} ± {float(r['val_acc_std']):.4f}",
                  f"  epochs finales  : {n_epochs} (media de las optimas por fold)",
                  f"  TEST acc        : {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}  "
                  f"(semillas {FINAL_SEEDS}: {', '.join(f'{a:.4f}' for a in test_accs)})",
                  ""]

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Resumen guardado en {summary_path}")

    config = {"experiment": EXPERIMENT, "channels": ch_tag, "seed": seed,
              "n_trials": args.trials, "max_epochs": args.max_epochs, "patience": PATIENCE,
              "cv_folds": cfg.CV_FOLDS, "test_subjects": cfg.TEST_SUBJECTS,
              "final_seeds": FINAL_SEEDS, "feature_key": cfg.FEATURE_KEY}
    best = final["MEJOR"]
    metrics = {"seed": seed, **best["per_seed"][0], "final": final}
    save_run(out_dir, config, metrics)


if __name__ == "__main__":
    main()
