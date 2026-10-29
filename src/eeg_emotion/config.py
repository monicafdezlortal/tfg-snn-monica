"""
Constantes compartidas por todos los experimentos: rutas, canales y splits.
"""

from pathlib import Path

import torch

# ============================================================
# RUTAS
# ============================================================
REPO_ROOT   = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"

# Features de SEED ya extraidas (ventanas de 1 s). Se buscan en data/ y, si
# no estan, en la raiz del repo (mismo criterio que los scripts originales).
DATA_DIR = (REPO_ROOT / "data" / "ExtractedFeatures_1s"
            if (REPO_ROOT / "data" / "ExtractedFeatures_1s").is_dir()
            else REPO_ROOT / "ExtractedFeatures_1s")

FEATURE_KEY = "de_LDS"      # Differential Entropy suavizada con LDS

# ============================================================
# CANALES Y BANDAS
# ============================================================
# Canales criticos segun Zheng & Lu 2015 (indices 0-based en el array de 62)
CHANNEL_NAMES   = ["FT7", "FT8", "T7", "T8"]
CHANNEL_INDICES = [14, 22, 23, 31]
N_CHANNELS      = 62
BAND_NAMES      = ["delta", "theta", "alpha", "beta", "gamma"]
N_BANDS         = len(BAND_NAMES)

CLASS_NAMES = ["negativa", "neutra", "positiva"]   # etiquetas SEED -1/0/1 -> 0/1/2

# ============================================================
# SPLITS
# ============================================================
# Cross-subject: los sujetos de test no se ven nunca durante el entrenamiento
TEST_SUBJECTS = [1, 2, 3]
VAL_SUBJECTS  = [4, 5]
# Validacion cruzada por sujetos (busquedas de hiperparametros) sobre 4-15
CV_FOLDS = [[4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]

# Protocolo de Zheng & Lu 2015 (dentro de cada sesion)
ZHENG_TRAIN_TRIALS = list(range(1, 10))     # trials 1-9
ZHENG_TEST_TRIALS  = list(range(10, 16))    # trials 10-15

# ============================================================
# HARDWARE
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
