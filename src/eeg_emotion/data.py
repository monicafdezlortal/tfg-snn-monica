"""
Carga de SEED (ExtractedFeatures_1s), seleccion de canales, normalizacion,
splits y codificacion en spikes.
"""

import os
import glob
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from . import config as cfg


# ============================================================
# CARGA
# ============================================================
def session_files(data_dir=cfg.DATA_DIR):
    """Ficheros de sesion ("<sujeto>_<fecha>.mat") en orden alfabetico."""
    files = sorted(glob.glob(os.path.join(str(data_dir), "*.mat")))
    return [f for f in files if "label" not in os.path.basename(f)]


def load_labels(data_dir=cfg.DATA_DIR):
    """Etiquetas de los 15 trials: -1/0/1 -> 0/1/2 (CrossEntropyLoss)."""
    labels = loadmat(os.path.join(str(data_dir), "label.mat"))["label"].flatten()
    return (labels + 1).astype(np.int64)


def channel_indices(channels):
    """Indices de canal para --channels 4|62 (None = los 62)."""
    if channels == 4:
        return cfg.CHANNEL_INDICES
    if channels == 62:
        return None
    raise ValueError(f"channels debe ser 4 o 62, no {channels}")


def channel_columns(indices=cfg.CHANNEL_INDICES):
    """Columnas de la matriz de 62 canales (canal*5 + banda) para esos canales."""
    return [c * cfg.N_BANDS + b for c in indices for b in range(cfg.N_BANDS)]


def zheng_sessions(n_sessions=2, data_dir=cfg.DATA_DIR):
    """[(sujeto, n_sesion, ruta)] con los sujetos en orden numerico y, dentro
    de cada sujeto, sus primeras n_sessions sesiones por fecha. Zheng & Lu
    2015 usan 2 sesiones por sujeto (30 experimentos); el SEED publico trae 3."""
    by_subj = {}
    for fp in session_files(data_dir):
        subj, date = os.path.splitext(os.path.basename(fp))[0].split("_")
        by_subj.setdefault(int(subj), []).append((date, fp))
    return [(subj, k, fp)
            for subj in sorted(by_subj)
            for k, (_, fp) in enumerate(sorted(by_subj[subj])[:n_sessions], start=1)]


def load_seed(feature_keys=(cfg.FEATURE_KEY,), channels=4, data_dir=cfg.DATA_DIR,
              order="file", files=None):
    """
    Lee las features de todas las sesiones de SEED.

    Devuelve (feats, meta):
      feats[key] : (N, C*5) float64, una fila por ventana de 1 s; columnas
                   canal-major (canal*5 + banda)
      meta       : dict de arrays (N,) -> y, subject, session (1-3),
                   trial (1-15) y trial_id (unico por sesion y trial)

    channels=4 selecciona FT7, FT8, T7, T8; channels=62 los deja todos
    (las features por pares de electrodos, dasm/rasm..., solo con 62).
    order="file"    : orden alfabetico de ficheros (10_, 11_, ..., 1_, 2_...)
    order="subject" : sujetos 1..15 en orden numerico (orden del antiguo
                      data/seed_snn_dataset.npz, que usaba la SNN)
    files : lista de ficheros a leer (por defecto todas las sesiones de
            data_dir); p.ej. [ruta] para una sola sesion
    """
    feature_keys = list(feature_keys)
    idx = channel_indices(channels)
    labels = load_labels(data_dir)

    feats = {k: [] for k in feature_keys}
    cols = {k: [] for k in ("y", "subject", "session", "trial", "trial_id")}
    sessions_seen = {}
    tid = 0
    for fp in (session_files(data_dir) if files is None else files):
        name = os.path.basename(fp)
        subj_id = int(name.split("_")[0])                   # "1_20131027.mat" -> 1
        sessions_seen[subj_id] = sessions_seen.get(subj_id, 0) + 1
        mat = loadmat(fp)
        for t in range(1, 16):
            missing = [k for k in feature_keys if f"{k}{t}" not in mat]
            if missing:
                print(f"  AVISO: {missing[0]}{t} no encontrado en {name}")
                continue
            n = mat[f"{feature_keys[0]}{t}"].shape[1]
            for k in feature_keys:
                a = mat[f"{k}{t}"]                          # (C, n, 5)
                if idx is not None:
                    a = a[idx]
                a = a.transpose(1, 0, 2)                    # (n, C, 5)
                feats[k].append(a.reshape(n, -1))
            cols["y"].append(np.full(n, labels[t - 1], dtype=np.int64))
            cols["subject"].append(np.full(n, subj_id, dtype=np.int64))
            cols["session"].append(np.full(n, sessions_seen[subj_id], dtype=np.int64))
            cols["trial"].append(np.full(n, t, dtype=np.int64))
            cols["trial_id"].append(np.full(n, tid, dtype=np.int64))
            tid += 1

    feats = {k: np.concatenate(v).astype(np.float64) for k, v in feats.items()}
    meta = {k: np.concatenate(v) for k, v in cols.items()}

    if order == "subject":
        perm = np.argsort(meta["subject"], kind="stable")
        feats = {k: v[perm] for k, v in feats.items()}
        meta = {k: v[perm] for k, v in meta.items()}
    elif order != "file":
        raise ValueError(f"order debe ser 'file' o 'subject', no {order}")
    return feats, meta


def load_seed_features(channels=4, feature_key=cfg.FEATURE_KEY, data_dir=cfg.DATA_DIR,
                       order="file", files=None):
    """Atajo para una sola feature: devuelve (X (N, C*5) float64, meta)."""
    feats, meta = load_seed((feature_key,), channels, data_dir, order, files)
    return feats[feature_key], meta


# ============================================================
# NORMALIZACION
# ============================================================
def zscore_per_subject(X, subject):
    """z-score por sujeto y feature (float64). Cada sujeto usa solo sus
    estadisticas, asi los sujetos de test no filtran informacion al train."""
    Z = np.zeros_like(X, dtype=np.float64)
    for s in np.unique(subject):
        mask  = subject == s
        mu    = X[mask].mean(axis=0, keepdims=True)
        sigma = X[mask].std(axis=0, keepdims=True)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        Z[mask] = (X[mask] - mu) / sigma
    return Z


def normalize_per_subject(X, subject):
    """z-score por sujeto en float32 (entrada de la ANN)."""
    return zscore_per_subject(X, subject).astype(np.float32)


def zscore_train_test(X_train, X_test):
    """z-score con las estadisticas SOLO del train (protocolo por sesion de
    Zheng & Lu: el test no se usa para normalizar)."""
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


def to_spike_rate(Z, clip=4):
    """Mapea un z-score a [0, 1] clipando en +-clip sigma (probabilidad de
    spike para el rate coding de la SNN)."""
    return ((np.clip(Z, -clip, clip) + clip) / (2 * clip)).astype(np.float32)


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
# SPLITS
# ============================================================
def cross_subject_masks(subject, test_subjects=cfg.TEST_SUBJECTS,
                        val_subjects=cfg.VAL_SUBJECTS):
    """Mascaras (train, val, test) por sujetos enteros."""
    test_mask  = np.isin(subject, test_subjects)
    val_mask   = np.isin(subject, val_subjects)
    train_mask = ~(test_mask | val_mask)
    return train_mask, val_mask, test_mask


def cv_fold_masks(subject, folds=cfg.CV_FOLDS):
    """Mascaras de validacion de cada fold y mascara del conjunto de sujetos
    que se reparten entre los folds."""
    fold_masks = [np.isin(subject, f) for f in folds]
    pool_mask  = np.isin(subject, sum(folds, []))
    return fold_masks, pool_mask


def describe_split(subject, masks, names=("Train", "Val  ", "Test ")):
    for name, mask in zip(names, masks):
        print(f"{name} sujetos : {sorted(np.unique(subject[mask]).tolist())}")


# ============================================================
# CODIFICACION EN SPIKES (SNN)
# ============================================================
class SpikeDataset(Dataset):
    """
    Genera spikes por rate coding en cada __getitem__ (no en disco).
    Esto actua como data augmentation estocastica en cada epoch.
    """
    def __init__(self, X_norm, y, T):
        self.X = torch.tensor(X_norm, dtype=torch.float32)  # (N, F)
        self.y = torch.tensor(y,      dtype=torch.long)
        self.T = T

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]                                      # (F,)
        # Bernoulli: spike si random < probabilidad = valor normalizado
        rand   = torch.rand(self.T, x.shape[0])             # (T, F)
        spikes = (rand < x.unsqueeze(0)).float()             # (T, F)
        return spikes, self.y[idx]


def spike_collate(batch):
    """Convierte lista de (T,F) y labels en tensores (T,B,F) y (B,)."""
    spikes, labels = zip(*batch)
    spikes = torch.stack(spikes, dim=1)   # (T, B, F)
    labels = torch.stack(labels, dim=0)   # (B,)
    return spikes, labels
