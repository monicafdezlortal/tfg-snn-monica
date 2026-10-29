"""
Modelos: ANN (MLP) y su equivalente spiking (SNN LIF con SpikingJelly).
"""

import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer


# ============================================================
# ANN
# ============================================================
class EmotionANN(nn.Module):
    """
    Equivalente no-spiking de EmotionSNN:
    Linear -> BN -> ReLU -> Dropout  (x2)  -> Linear (logits)
    Entrada : (B, F) features continuas
    Salida  : (B, 3) logits
    """
    def __init__(self, input_dim=20, hidden=128, num_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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
# SNN
# ============================================================
class EmotionSNN(nn.Module):
    """
    SNN de 2 capas ocultas con neuronas LIF.
    - BatchNorm antes de cada LIF para que la entrada este en rango util
    - v_threshold=0.5 para que las neuronas disparen (con umbral=1.0 y
      pesos Kaiming la membrana raramente llega al umbral -> neuronas muertas)
    - Sin LIF en la capa de salida: CrossEntropyLoss espera logits reales,
      no spikes binarios
    Entrada : (T, B, F) spike trains
    Salida  : (B, 3)    logits (media temporal)
    """
    def __init__(self, input_dim=20, hidden=128, num_classes=3, dropout=0.3):
        super().__init__()

        self.fc1   = layer.Linear(input_dim, hidden)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.lif1  = neuron.LIFNode(tau=2.0, v_threshold=0.5,
                                    surrogate_function=surrogate.ATan())
        self.drop1 = nn.Dropout(dropout)

        self.fc2   = layer.Linear(hidden, hidden // 2)
        self.bn2   = nn.BatchNorm1d(hidden // 2)
        self.lif2  = neuron.LIFNode(tau=2.0, v_threshold=0.5,
                                    surrogate_function=surrogate.ATan())
        self.drop2 = nn.Dropout(dropout)

        self.fc3   = layer.Linear(hidden // 2, num_classes)
        # Sin LIF aqui: la salida son logits directos para CrossEntropyLoss

    def forward(self, x_seq):
        # x_seq : (T, B, F)
        spike_acc = None
        for t in range(x_seq.shape[0]):
            h = self.drop1(self.lif1(self.bn1(self.fc1(x_seq[t]))))
            h = self.drop2(self.lif2(self.bn2(self.fc2(h))))
            h = self.fc3(h)
            spike_acc = h if spike_acc is None else spike_acc + h
        return spike_acc / x_seq.shape[0]               # (B, num_classes)


class FlexSNN(nn.Module):
    """n_layers capas ocultas iguales (Linear -> BN -> LIF -> Dropout) con
    umbral y tau configurables, y Linear final de logits."""
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
