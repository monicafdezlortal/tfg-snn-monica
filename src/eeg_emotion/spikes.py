"""
Codificacion en spikes para la SNN (rate coding). Es el unico codigo de carga
de datos que necesita torch; vive aparte para que eeg_emotion.data se pueda
usar sin torch (entorno .venv-akida del experimento 05).
"""

import torch
from torch.utils.data import Dataset


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
