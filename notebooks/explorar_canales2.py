from scipy.io import loadmat
import numpy as np

# Exploramos un archivo Preprocessed_EEG por si tiene info de canales
mat = loadmat(r"SEED\SEED_EEG\Preprocessed_EEG\1_20131027.mat")

keys = [k for k in mat.keys() if not k.startswith("__")]
print("Claves en Preprocessed_EEG/1_20131027.mat:")
for k in keys:
    v = mat[k]
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")

# Lista estandar SEED 62 canales (ESI NeuroScan 10-20 extended)
# Fuente: dataset oficial SEED y papers que lo usan
SEED_62_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4',           # 0-4
    'F7',  'F5',  'F3',  'F1',  'FZ',             # 5-9
    'F2',  'F4',  'F6',  'F8',                    # 10-13
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ',            # 14-18
    'FC2', 'FC4', 'FC6', 'FT8',                   # 19-22
    'T7',  'CP5', 'CP3', 'CP1', 'CPZ',            # 23-27
    'CP2', 'CP4', 'CP6', 'T8',                    # 28-31
    'TP9', 'P7',  'P5',  'P3',  'P1',             # 32-36
    'PZ',  'P2',  'P4',  'P6',  'P8',  'TP10',   # 37-42
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',  # 43-49
    'CB1', 'O1',  'OZ',  'O2',  'CB2',            # 50-54
    'M1',  'TP7', 'CP5', 'CP6', 'TP8', 'M2',     # 55-60  (algunos overlaps posibles)
    'CB2'                                          # 61
]

print()
print(f"Lista tiene {len(SEED_62_CHANNELS)} entradas")
target = ['FT7', 'FT8', 'T7', 'T8']
for name in target:
    idx = [i for i, n in enumerate(SEED_62_CHANNELS) if n == name]
    print(f"  {name}: indices = {idx}")
