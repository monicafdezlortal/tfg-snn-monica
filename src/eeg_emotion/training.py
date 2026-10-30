"""
Bucle de entrenamiento, evaluacion, metricas, semillas y guardado de resultados.
"""

import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from spikingjelly.activation_based import functional

from . import config as cfg


# ============================================================
# REPRODUCIBILIDAD
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# ENTRENAMIENTO
# ============================================================
def class_weights(y, device=cfg.DEVICE):
    """Pesos inversos a la frecuencia de cada clase en train, escalados a
    media 1 (penaliza mas los errores en clases minoritarias)."""
    counts = np.bincount(y, minlength=3).astype(np.float32)
    w = counts.sum() / (3.0 * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, loader, device=cfg.DEVICE, spiking=False):
    """Devuelve (loss media, accuracy, preds, true). Si spiking=True se
    resetea el estado de las neuronas LIF tras cada batch."""
    model.eval()
    all_preds, all_true, losses = [], [], []
    loss_fn = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(loss_fn(logits, y).item())
        all_preds.append(logits.argmax(1).cpu())
        all_true.append(y.cpu())
        if spiking:
            functional.reset_net(model)
    preds = torch.cat(all_preds).numpy()
    true  = torch.cat(all_true).numpy()
    acc   = (preds == true).mean()
    return float(np.mean(losses)), float(acc), preds, true


def fit(model, train_loader, val_loader, optim, sched, loss_fn, epochs, ckpt_path,
        device=cfg.DEVICE, spiking=False):
    """
    Entrena `epochs` epochs, guarda en ckpt_path el modelo con mejor val_acc
    y lo recarga al terminar. sched es ReduceLROnPlateau sobre val_acc.
    Devuelve (best_val_acc, best_epoch, historial por epoch).
    """
    os.makedirs(os.path.dirname(str(ckpt_path)), exist_ok=True)
    best_val_acc, best_epoch, history = 0.0, 0, []

    print(f"{'Ep':>4} {'train_loss':>10} {'val_loss':>9} {'val_acc':>8} {'lr':>8} {'best':>5}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y_batch in train_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            optim.zero_grad()
            loss = loss_fn(model(x), y_batch)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
            if spiking:
                functional.reset_net(model)

        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, spiking)
        sched.step(val_acc)   # ReduceLROnPlateau necesita la metrica
        current_lr = optim.param_groups[0]['lr']

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), ckpt_path)
            marker = " <--"

        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)),
                        "val_loss": val_loss, "val_acc": val_acc, "lr": current_lr})
        print(f"{epoch:4d} {np.mean(train_losses):10.4f} {val_loss:9.4f} "
              f"{val_acc:8.4f} {current_lr:8.2e}{marker}")

    print(f"\nMejor val_acc: {best_val_acc:.4f} (epoch {best_epoch})")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return best_val_acc, best_epoch, history


# ============================================================
# METRICAS Y RESULTADOS
# ============================================================
def compute_metrics(y_true, y_pred):
    """accuracy, F1 macro, F1 por clase y matriz de confusion."""
    labels = list(range(len(cfg.CLASS_NAMES)))
    f1_cls = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "f1_per_class": {c: float(f) for c, f in zip(cfg.CLASS_NAMES, f1_cls)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_test": int(len(y_true)),
    }


def run_dir(experiment, channels, seed, root=cfg.RESULTS_DIR):
    """results/<experimento>/<canales>_seed<N>/"""
    ch = channels if isinstance(channels, str) else f"{channels}ch"
    path = os.path.join(str(root), experiment, f"{ch}_seed{seed}")
    os.makedirs(path, exist_ok=True)
    return path


def _write_confusion_matrix(path, cm):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["real\\predicha"] + cfg.CLASS_NAMES)
        for name, row in zip(cfg.CLASS_NAMES, cm):
            w.writerow([name] + row)


def save_run(out_dir, config, metrics):
    """Guarda config.json, metrics.json y confusion_matrix.csv en out_dir.
    Si metrics trae varios modelos (metrics["modelos"][nombre]), escribe una
    confusion_matrix_<nombre>.csv por modelo."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        _write_confusion_matrix(os.path.join(out_dir, "confusion_matrix.csv"), cm)
    for name, m in metrics.get("modelos", {}).items():
        if m.get("confusion_matrix") is not None:
            _write_confusion_matrix(os.path.join(out_dir, f"confusion_matrix_{name}.csv"),
                                    m["confusion_matrix"])
    print(f"Resultados guardados en: {out_dir}")
