"""
Pipeline de preprocesado SEED -> dataset SNN
Entrada : data/ExtractedFeatures_1s/*.mat  (de_LDS, ya suavizadas)
Salida  : data/seed_snn_dataset.npz
          X_spikes   : (N, T=50, 20)  float32  spike trains rate-coded
          X_features : (N, 20)        float32  DE normalizada [0,1]
          y          : (N,)           int64    0=negativa 1=neutra 2=positiva
"""

import os
import glob
import numpy as np
from scipy.io import loadmat

# ============================================================
# CONFIG
# ============================================================
DATA_DIR    = ("data/ExtractedFeatures_1s" if os.path.isdir("data/ExtractedFeatures_1s")
               else "ExtractedFeatures_1s")
OUTPUT_PATH = "data/seed_snn_dataset.npz"

# Canales criticos segun Zheng & Lu 2015 (indices 0-based en el array de 62)
CHANNEL_NAMES   = ['FT7', 'FT8', 'T7', 'T8']
CHANNEL_INDICES = [14, 22, 23, 31]

# 5 bandas DE (dimension 2 del array de ExtractedFeatures)
BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']

T    = 50   # pasos temporales para rate coding
SEED = 42


# ============================================================
# PASO 1: CARGAR ETIQUETAS
# ============================================================
print("=" * 55)
print("PASO 1 — Etiquetas de los 15 trials")
print("=" * 55)

label_mat    = loadmat(os.path.join(DATA_DIR, "label.mat"))
trial_labels = label_mat['label'].flatten()        # [-1, 0, 1, ...]
print(f"Etiquetas originales  : {trial_labels}")

# CrossEntropyLoss requiere clases 0..C-1
label_map = {-1: 0, 0: 1, 1: 2}
labels_mapped = np.array([label_map[l] for l in trial_labels], dtype=np.int64)
print(f"Etiquetas remapeadas  : {labels_mapped}")
print(f"  (0=negativa, 1=neutra, 2=positiva)")


# ============================================================
# PASO 2: CARGAR FEATURES DE TODOS LOS SUJETOS
# ============================================================
print("\n" + "=" * 55)
print("PASO 2 — Carga de features de_LDS (4 canales, 5 bandas)")
print("=" * 55)

mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.mat")))
mat_files = [f for f in mat_files if "label" not in os.path.basename(f)]
print(f"Archivos .mat encontrados: {len(mat_files)}")

all_X    = []
all_y    = []
all_subj = []   # id de sujeto (1-15)
all_sess = []   # numero de sesion (1, 2 o 3) dentro de cada sujeto

# Agrupa archivos por sujeto para asignar numero de sesion
from collections import defaultdict
subj_files = defaultdict(list)
for fp in mat_files:
    basename = os.path.basename(fp)
    subj_id  = int(basename.split("_")[0])   # "1_20131027.mat" -> 1
    subj_files[subj_id].append(fp)

for subj_id in sorted(subj_files.keys()):
    sessions = sorted(subj_files[subj_id])   # orden cronologico por fecha
    for sess_num, file_path in enumerate(sessions, start=1):
        mat  = loadmat(file_path)
        name = os.path.basename(file_path)
        segs_subj = 0

        for trial_idx in range(15):
            key = f"de_LDS{trial_idx + 1}"
            if key not in mat:
                print(f"  AVISO: {key} no encontrado en {name}")
                continue

            de     = mat[key]                          # (62, N_seg, 5)
            de_4ch = de[CHANNEL_INDICES, :, :]         # (4, N_seg, 5)
            de_4ch = de_4ch.transpose(1, 0, 2)         # (N_seg, 4, 5)
            N_seg  = de_4ch.shape[0]
            de_flat = de_4ch.reshape(N_seg, -1)        # (N_seg, 20)

            label   = labels_mapped[trial_idx]
            y_trial = np.full(N_seg, label, dtype=np.int64)

            all_X.append(de_flat.astype(np.float64))
            all_y.append(y_trial)
            all_subj.append(np.full(N_seg, subj_id,  dtype=np.int64))
            all_sess.append(np.full(N_seg, sess_num, dtype=np.int64))
            segs_subj += N_seg

        print(f"  Sujeto {subj_id:2d} sesion {sess_num}: {name}  ({segs_subj} ventanas)")

X    = np.concatenate(all_X,    axis=0)
y    = np.concatenate(all_y,    axis=0)
subj = np.concatenate(all_subj, axis=0)
sess = np.concatenate(all_sess, axis=0)

print(f"\nX shape total : {X.shape}   (muestras x features)")
print(f"y shape total : {y.shape}")
unique, counts = np.unique(y, return_counts=True)
label_names = {0: 'negativa', 1: 'neutra', 2: 'positiva'}
for cls, cnt in zip(unique, counts):
    print(f"  Clase {cls} ({label_names[cls]}): {cnt} muestras ({cnt/len(y)*100:.1f}%)")


# ============================================================
# PASO 3: NORMALIZACION [0, 1] POR FEATURE
# ============================================================
print("\n" + "=" * 55)
print("PASO 3 — Normalizacion min-max por feature")
print("=" * 55)

# Min-max global (toda la muestra) para cada una de las 20 features
# En entrenamiento real deberas calcular min/max solo sobre el train split
x_min = X.min(axis=0, keepdims=True)   # (1, 20)
x_max = X.max(axis=0, keepdims=True)   # (1, 20)
span  = np.where((x_max - x_min) < 1e-8, 1.0, x_max - x_min)

X_norm = (X - x_min) / span
X_norm = np.clip(X_norm, 0.0, 1.0).astype(np.float32)

print(f"X_norm : min={X_norm.min():.4f}  max={X_norm.max():.4f}  mean={X_norm.mean():.4f}")
print(f"Shape  : {X_norm.shape}")
print()
print("Detalle por feature (canal x banda):")
feat_idx = 0
for ch in CHANNEL_NAMES:
    for band in BAND_NAMES:
        col = X_norm[:, feat_idx]
        print(f"  [{feat_idx:2d}] {ch}-{band:5s}: min={col.min():.3f}  max={col.max():.3f}  mean={col.mean():.3f}")
        feat_idx += 1


# ============================================================
# PASO 4: RATE CODING -> (N, T, 20)
# ============================================================
print("\n" + "=" * 55)
print(f"PASO 4 — Rate coding  (T={T} pasos temporales)")
print("=" * 55)

rng = np.random.default_rng(SEED)
N, F = X_norm.shape

# Para cada muestra y timestep: spike si random < p (p = valor normalizado)
# Generamos en bloques para no saturar RAM
BLOCK = 5000
spikes_list = []
for start in range(0, N, BLOCK):
    end   = min(start + BLOCK, N)
    block = X_norm[start:end]                          # (B, 20)
    rand  = rng.random((end - start, T, F), dtype=np.float32)  # (B, T, 20)
    spk   = (rand < block[:, np.newaxis, :]).astype(np.float32) # (B, T, 20)
    spikes_list.append(spk)

spikes = np.concatenate(spikes_list, axis=0)  # (N, T, 20)

print(f"Spike tensor shape : {spikes.shape}")
print(f"Tasa media de spikes: {spikes.mean():.4f}  (deberia ≈ mean de X_norm = {X_norm.mean():.4f})")
print(f"Memoria aproximada : {spikes.nbytes / 1e6:.1f} MB")


# ============================================================
# PASO 5: GUARDAR
# ============================================================
print("\n" + "=" * 55)
print("PASO 5 — Guardando dataset")
print("=" * 55)

os.makedirs("data", exist_ok=True)
np.savez_compressed(
    OUTPUT_PATH,
    X_spikes=spikes,                         # (N, T=50, 20)  float32
    X_features=X_norm,                       # (N, 20)        float32
    y=y,                                     # (N,)           int64
    subject=subj,                            # (N,)           int64  sujeto 1-15
    session=sess,                            # (N,)           int64  sesion 1/2/3
    channel_indices=np.array(CHANNEL_INDICES),
    norm_min=x_min.astype(np.float32),
    norm_max=x_max.astype(np.float32),
)

size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
print(f"Guardado en : {OUTPUT_PATH}  ({size_mb:.1f} MB)")
print()
print("RESUMEN FINAL:")
print(f"  X_spikes   : {spikes.shape}  <-- listo para SpikingJelly")
print(f"  X_features : {X_norm.shape}")
print(f"  y          : {y.shape}")
print(f"  subject    : {subj.shape}  (sujetos 1-{subj.max()})")
print(f"  session    : {sess.shape}  (sesiones 1-{sess.max()})")
print(f"  T          : {T} pasos temporales")
print(f"  Features   : 4 canales x 5 bandas = 20")
print()
print("Split subject-dependent:")
train_mask = sess <= 2
test_mask  = sess == 3
print(f"  Train (sesiones 1+2) : {train_mask.sum()} muestras")
print(f"  Test  (sesion 3)     : {test_mask.sum()} muestras")
