"""
Experimento 04b - Random search de hiperparametros sobre la SNN SEED.
- 15 combinaciones aleatorias
- Max 20 epochs por combinacion, early stopping patience=5 sobre val_loss
- Mismos datos, normalizacion por sujeto y split que 03_snn_cross_subject.py
- La mejor configuracion se elige por val_acc (sujetos de validacion);
  test_acc solo se reporta, no se usa para elegir

Uso:
  python scripts/04b_snn_hparam_search.py --channels 4
Salida (se puede reanudar):
  results/04b_snn_hparam_search/<4ch|62ch>_seed<N>/
    hparam_results.csv  una fila por configuracion
    config.json, metrics.json, confusion_matrix.csv (de la mejor)
"""

import argparse
import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional

from eeg_emotion import config as cfg
from eeg_emotion.data import (load_seed_features, zscore_per_subject, to_spike_rate,
                              cross_subject_masks, SpikeDataset, spike_collate)
from eeg_emotion.models import FlexSNN
from eeg_emotion.training import (class_weights, evaluate, compute_metrics,
                                  run_dir, save_run)

EXPERIMENT = "04b_snn_hparam_search"
N_TRIALS   = 15
MAX_EPOCHS = 20
PATIENCE   = 5     # early stopping sobre val_loss
T          = 50    # mismos que 03_snn_cross_subject.py
BATCH_SIZE = 256
DEVICE     = cfg.DEVICE

FIELDNAMES = ["trial", "hidden", "n_layers", "v_threshold", "tau",
              "dropout", "lr", "n_params", "epochs_run", "val_acc", "test_acc"]


# ============================================================
# CARGA DE DATOS (igual que 03_snn_cross_subject.py, una sola vez)
# ============================================================
def load_data(channels):
    X_raw, meta = load_seed_features(channels, order="subject")
    y, subject = meta["y"], meta["subject"]
    X_subj = to_spike_rate(zscore_per_subject(X_raw, subject))

    train_mask, val_mask, test_mask = cross_subject_masks(subject)
    X_train, y_train = X_subj[train_mask], y[train_mask]
    X_val,   y_val   = X_subj[val_mask],   y[val_mask]
    X_test,  y_test  = X_subj[test_mask],  y[test_mask]

    w_tensor = class_weights(y_train, DEVICE)
    return X_train, y_train, X_val, y_val, X_test, y_test, w_tensor


# ============================================================
# ENTRENAMIENTO DE UN TRIAL
# ============================================================
def run_trial(hp, X_train, y_train, X_val, y_val, X_test, y_test, w_tensor,
              seed, max_epochs=MAX_EPOCHS):
    torch.manual_seed(seed)

    train_ds = SpikeDataset(X_train, y_train, T)
    val_ds   = SpikeDataset(X_val,   y_val,   T)
    test_ds  = SpikeDataset(X_test,  y_test,  T)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=spike_collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=spike_collate, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=spike_collate, num_workers=0)

    model   = FlexSNN(
        input_dim=X_train.shape[1],
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

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y_batch in train_loader:
            x, y_batch = x.to(DEVICE), y_batch.to(DEVICE)
            optim.zero_grad()
            loss = loss_fn(model(x), y_batch)
            loss.backward()
            optim.step()
            functional.reset_net(model)

        val_loss, val_acc, _, _ = evaluate(model, val_loader, DEVICE, spiking=True)
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

    _, val_acc,  _,    _    = evaluate(model, val_loader, DEVICE, spiking=True)
    _, test_acc, preds, true = evaluate(model, test_loader, DEVICE, spiking=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    row = {
        **hp,
        "n_params":   n_params,
        "epochs_run": epochs_done,
        "val_acc":    round(val_acc,  4),
        "test_acc":   round(test_acc, 4),
    }
    return row, compute_metrics(true, preds)


# ============================================================
# ESPACIO DE BUSQUEDA Y MUESTREO
# ============================================================
def sample_hp(rng):
    return {
        "hidden":      int(rng.integers(64, 257)),          # 64..256
        "n_layers":    int(rng.integers(1, 3)),             # 1 o 2
        "v_threshold": round(float(rng.uniform(0.5, 1.5)), 3),
        "tau":         round(float(rng.uniform(1.1, 4.0)), 3),  # LIFNode exige tau > 1
        "dropout":     round(float(rng.uniform(0.1, 0.4)), 3),
        "lr":          float(np.exp(rng.uniform(np.log(5e-4), np.log(5e-3)))),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", type=int, choices=[4, 62], default=4)
    p.add_argument("--seed", type=int, default=42,
                   help="semilla del sorteo de hiperparametros y del entrenamiento")
    p.add_argument("--trials", type=int, default=N_TRIALS)
    p.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                   help="solo para pruebas rapidas (por defecto %(default)s)")
    p.add_argument("--out", default=None, help="carpeta de salida (por defecto en results/)")
    return p.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = args.out or run_dir(EXPERIMENT, args.channels, args.seed)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "hparam_results.csv")

    print("Cargando datos...")
    X_tr, y_tr, X_va, y_va, X_te, y_te, w = load_data(args.channels)
    print(f"  Train: {len(y_tr)}  Val: {len(y_va)}  Test: {len(y_te)}\n")

    # --- Reanudacion: cargar trials ya completados del CSV ---
    results       = []
    completed_ids = set()
    test_metrics  = {}          # metricas completas de los trials de esta ejecucion
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                results.append({k: (int(v) if k in ("trial","hidden","n_layers","epochs_run","n_params")
                                    else float(v)) for k, v in row.items()})
                completed_ids.add(int(row["trial"]))
        print(f"Reanudando: {len(completed_ids)} trials ya completados "
              f"({sorted(completed_ids)})\n")

    write_mode = "a" if completed_ids else "w"
    with open(csv_path, write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not completed_ids:
            writer.writeheader()

        for trial in range(1, args.trials + 1):
            hp = sample_hp(rng)   # el RNG es determinista: misma semilla -> mismos hp
            if trial in completed_ids:
                print(f"Trial {trial:2d}/{args.trials}  [ya completado, saltando]")
                continue

            print(f"Trial {trial:2d}/{args.trials}  "
                  f"hidden={hp['hidden']}  layers={hp['n_layers']}  "
                  f"thr={hp['v_threshold']}  tau={hp['tau']}  "
                  f"drop={hp['dropout']}  lr={hp['lr']:.2e}  ...",
                  end="", flush=True)

            t0  = time.time()
            row, test_metrics[trial] = run_trial(hp, X_tr, y_tr, X_va, y_va, X_te, y_te, w,
                                                 seed=args.seed, max_epochs=args.max_epochs)
            row["trial"] = trial
            elapsed = time.time() - t0

            print(f"  val={row['val_acc']:.4f}  test={row['test_acc']:.4f}  "
                  f"ep={row['epochs_run']}  {elapsed:.0f}s")

            writer.writerow({k: row[k] for k in FIELDNAMES})
            f.flush()
            results.append(row)

    # --- Top 5 por val_acc (el test no se usa para elegir) ---
    results.sort(key=lambda r: r["val_acc"], reverse=True)
    print(f"\n{'='*70}")
    print(f"TOP 5 por val_acc  (resultados completos en {csv_path})")
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

    config = {"experiment": EXPERIMENT, "channels": args.channels, "seed": args.seed,
              "n_trials": args.trials, "max_epochs": args.max_epochs, "patience": PATIENCE,
              "T": T, "batch_size": BATCH_SIZE, "val_subjects": cfg.VAL_SUBJECTS,
              "test_subjects": cfg.TEST_SUBJECTS, "feature_key": cfg.FEATURE_KEY}
    # Las metricas completas solo existen si el mejor trial se ha entrenado en
    # esta ejecucion; si viene de un CSV reanudado, solo hay accuracy.
    metrics = {"seed": args.seed, "best": best,
               **test_metrics.get(best["trial"], {"accuracy": best["test_acc"]})}
    save_run(out_dir, config, metrics)


if __name__ == "__main__":
    main()
