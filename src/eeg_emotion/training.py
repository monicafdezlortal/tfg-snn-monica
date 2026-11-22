"""
Bucle de entrenamiento, evaluacion, metricas, semillas y guardado de resultados.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
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
# METRICAS Y RESULTADOS (en eeg_emotion.results, sin torch; se
# reexportan aqui para que los scripts 01-04 no cambien)
# ============================================================
from .results import compute_metrics, run_dir, save_run, _write_confusion_matrix  # noqa: E402,F401
