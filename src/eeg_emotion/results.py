"""
Metricas y guardado de resultados (config.json, metrics.json,
confusion_matrix.csv). No depende de torch, asi que lo usan tanto los
experimentos 01-04 (a traves de eeg_emotion.training) como el 05 (Keras/Akida).
"""

import csv
import json
import os

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import config as cfg


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
