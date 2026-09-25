"""
Diagnostica por que la SNN no aprende: mide tasa de spikes por capa,
LR actual y umbral de las neuronas LIF.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import train_snn as tr   # reutiliza Dataset y collate_fn

DEVICE = "cpu"
data    = np.load(tr.NPZ_PATH)
X       = data["X_features"][data["session"] <= 2]
y       = data["y"][data["session"] <= 2]
x_min   = X.min(0, keepdims=True); x_max = X.max(0, keepdims=True)
span    = np.where((x_max - x_min) < 1e-8, 1.0, x_max - x_min)
X       = np.clip((X - x_min) / span, 0, 1).astype(np.float32)

ds     = tr.SpikeDataset(X, y, tr.T)
loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=tr.collate_fn)
x_batch, _ = next(iter(loader))   # (T=50, B=512, 20)

model = tr.EmotionSNN(input_dim=20, hidden=128, num_classes=3).to(DEVICE)

# --- Registro de spikes por capa ---
spike_records = {}
def make_hook(name):
    def hook(mod, inp, out):
        spike_records[name] = out.detach().mean().item()
    return hook

model.lif1.register_forward_hook(make_hook("lif1"))
model.lif2.register_forward_hook(make_hook("lif2"))
model.lif3.register_forward_hook(make_hook("lif3"))

model.eval()
with torch.no_grad():
    _ = model(x_batch)
functional.reset_net(model)

print("=" * 50)
print("DIAGNOSTICO SNN")
print("=" * 50)
print(f"\nLearning rate configurado : {tr.LR}")
print(f"Optimizador               : Adam")
print()

for lif_name in ["lif1", "lif2", "lif3"]:
    lif_mod = getattr(model, lif_name)
    rate    = spike_records.get(lif_name, float("nan"))
    print(f"{lif_name}: umbral={lif_mod.v_threshold}  "
          f"tau={lif_mod.tau}  "
          f"tasa_spikes={rate:.4f}  "
          f"({'MUERTA' if rate < 0.05 else 'OK'})")

print()
print("DIAGNOSIS:")
print("  Si tasa_spikes < 0.05 en alguna capa -> neurona muerta,")
print("  el gradiente no fluye -> loss = ln(3) = 1.0986 para siempre.")
print()
print("  Solucion: bajar umbral (v_threshold) o anadir BatchNorm antes del LIF.")
print("  La capa de salida (lif3) NUNCA debe ser LIF: usar Linear directo como logits.")
