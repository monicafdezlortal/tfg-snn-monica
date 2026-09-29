from scipy.io import loadmat
import numpy as np

# --- 1. Explorar un archivo de features ---
mat = loadmat("data/ExtractedFeatures_1s/1_20131027.mat")
print("=== Claves en 1_20131027.mat ===")
keys = [k for k in mat.keys() if not k.startswith("__")]
print(keys)
print()

for key in keys[:3]:  # primeras 3 claves
    arr = mat[key]
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
print()

# --- 2. Explorar el archivo de etiquetas ---
label_mat = loadmat("data/ExtractedFeatures_1s/label.mat")
print("=== Claves en label.mat ===")
label_keys = [k for k in label_mat.keys() if not k.startswith("__")]
print(label_keys)
print()

for k in label_keys:
    print(f"  {k}: {label_mat[k]}")
