"""
train_snn_seed.py
─────────────────────────────────────────────────────────────────────────────
SNN Fully-Connected para clasificación de emociones con el dataset SEED.

Características extraídas: de_LDS  (Differential Entropy + LDS suavizado)
  Forma por trial: (62 canales, N ventanas, 5 bandas) → aplanado a (N, 310)

Etiquetas SEED:   -1 = negativa  |  0 = neutra  |  1 = positiva
  Remapeadas a:    0              |  1            |  2   (para CrossEntropyLoss)

Estructura de archivos esperada
───────────────────────────────
data/ExtractedFeatures_1s/
    label.mat
    1_20131027.mat
    1_20131030.mat
    ...
    15_20130709.mat

Cómo ejecutar
─────────────
    python train_snn_seed.py

Dependencias
────────────
    pip install torch spikingjelly scipy numpy scikit-learn matplotlib seaborn
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")          # sin pantalla; cambia a "TkAgg" si quieres ventana
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import loadmat

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from spikingjelly.activation_based import neuron, surrogate, functional


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN  ← edita solo este bloque
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = r"C:\Users\Usuario\OneDrive\Documentos\UNI\TFG\tfg-snn-monica\data\ExtractedFeatures_1s"
LABEL_FILE = os.path.join(DATA_DIR, "label.mat")

# Features disponibles en cada .mat:
#   de_LDS, de_movingAve, psd_LDS, psd_movingAve
# de_LDS es la mejor opción según Zheng & Lu 2015
FEATURE_KEY = "de_LDS"

# ── Arquitectura SNN ──────────────────────────────────────────────────────────
INPUT_DIM  = 310        # 62 canales × 5 bandas
HIDDEN_DIM = 256        # neuronas en la capa oculta
OUTPUT_DIM = 3          # negativa / neutra / positiva
T          = 10         # pasos temporales (timesteps)

# ── Entrenamiento ─────────────────────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS     = 15
LR         = 1e-3
WEIGHT_DECAY = 1e-5
TEST_RATIO = 0.2        # fracción del dataset para test
SEED_RNG   = 42         # semilla de reproducibilidad

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Salidas ───────────────────────────────────────────────────────────────────
SAVE_MODEL  = "snn_seed_model.pth"     # guarda los pesos del modelo
SAVE_PLOT   = "training_curves.png"    # gráfica loss/accuracy
SAVE_CM     = "confusion_matrix.png"   # matriz de confusión

EMOTION_NAMES = ["negativa", "neutra", "positiva"]


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILIDAD
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE ETIQUETAS
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(label_file: str) -> np.ndarray:
    """
    Carga label.mat y devuelve un array de enteros [0,1,2] con 15 elementos.

    SEED usa -1 / 0 / 1.  Sumamos 1 para convertir a índices de clase.
    """
    raw = loadmat(label_file)
    user_keys = [k for k in raw.keys() if not k.startswith("__")]

    if not user_keys:
        raise ValueError(f"label.mat no contiene variables reconocibles: {label_file}")

    labels = raw[user_keys[0]].squeeze()  # shape (15,)
    labels_mapped = (labels + 1).astype(np.int64)

    print(f"  ✓ Etiquetas cargadas ({len(labels_mapped)} trials): {labels_mapped.tolist()}")
    return labels_mapped


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVERSIÓN DE UN TRIAL A VENTANAS
# ─────────────────────────────────────────────────────────────────────────────

def trial_to_windows(x: np.ndarray) -> np.ndarray:
    """
    Convierte un trial de forma (62, N_ventanas, 5) a (N_ventanas, 310).

    Los datos de SEED tienen:
        eje 0 → 62 canales EEG
        eje 1 → N ventanas temporales de 1 segundo
        eje 2 → 5 bandas de frecuencia (delta, theta, alpha, beta, gamma)

    El reshape aplana canales × bandas en un vector de 310 características.
    """
    if x.ndim != 3:
        raise ValueError(f"Se esperaba un array 3D, se recibió shape {x.shape}")

    # Caso estándar SEED: (62, N, 5)
    if x.shape[0] == 62 and x.shape[2] == 5:
        x = np.transpose(x, (1, 0, 2))   # → (N, 62, 5)
        return x.reshape(x.shape[0], -1)  # → (N, 310)

    # Por si el archivo ya estuviera como (N, 62, 5)
    if x.shape[1] == 62 and x.shape[2] == 5:
        return x.reshape(x.shape[0], -1)

    raise ValueError(f"Shape inesperado en un trial SEED: {x.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CARGA COMPLETA DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_dir: str, labels: np.ndarray, feature_key: str):
    """
    Lee todos los archivos .mat (excepto label.mat) y devuelve X e y.

    X.shape = (N_total_ventanas, 310)
    y.shape = (N_total_ventanas,)

    Las ventanas de cada trial heredan la etiqueta de ese trial.
    Los archivos se ordenan alfabéticamente para reproducibilidad.
    """
    mat_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
    mat_files = [f for f in mat_files
                 if os.path.basename(f).lower() != "label.mat"]

    if not mat_files:
        raise FileNotFoundError(
            f"No se encontraron archivos .mat en:\n  {data_dir}\n"
            "Comprueba que DATA_DIR es correcto."
        )

    print(f"\n  Archivos .mat encontrados: {len(mat_files)}")

    X_list, y_list = [], []
    files_missing_trials = []

    for mat_file in mat_files:
        fname = os.path.basename(mat_file)
        data  = loadmat(mat_file)

        windows_this_file = 0
        for trial_idx in range(1, 16):
            key = f"{feature_key}{trial_idx}"
            if key not in data:
                files_missing_trials.append((fname, key))
                continue

            windows = trial_to_windows(data[key])   # (N, 310)
            label   = labels[trial_idx - 1]
            labels_repeated = np.full(windows.shape[0], label, dtype=np.int64)

            X_list.append(windows)
            y_list.append(labels_repeated)
            windows_this_file += windows.shape[0]

        print(f"    {fname}: {windows_this_file} ventanas")

    if files_missing_trials:
        print(f"\n  ⚠ Trials no encontrados: {files_missing_trials[:10]}")

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)

    # Distribución de clases
    unique, counts = np.unique(y, return_counts=True)
    class_dist = {EMOTION_NAMES[c]: n for c, n in zip(unique, counts)}
    print(f"\n  X shape: {X.shape}  |  y shape: {y.shape}")
    print(f"  Distribución de clases: {class_dist}")

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 4. CODIFICACIÓN RATE CODING
# ─────────────────────────────────────────────────────────────────────────────

def rate_encode(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    Convierte un batch normalizado [0,1] en una secuencia de spikes.

    x.shape      → (batch, 310)
    salida.shape → (T, batch, 310)

    Cada valor x_i actúa como probabilidad de spike en cada timestep.
    Valores altos → más spikes → más activación en la SNN.
    """
    x = x.clamp(0.0, 1.0)
    return torch.stack(
        [torch.bernoulli(x) for _ in range(T)],
        dim=0
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODELO SNN
# ─────────────────────────────────────────────────────────────────────────────

class SNN_FC(nn.Module):
    """
    Red neuronal de spiking totalmente conectada.

    Arquitectura:
        Linear(310 → 256) → LIF → Dropout
        Linear(256 → 256) → LIF → Dropout
        Linear(256 → 3)

    La salida es la media temporal de los logits sobre los T timesteps.
    Esto es equivalente a contar cuántos spikes llegaron a la capa de salida.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.3):
        super().__init__()

        # Capa 1
        self.fc1      = nn.Linear(input_dim, hidden_dim)
        self.lif1     = neuron.LIFNode(
                            tau=2.0,
                            surrogate_function=surrogate.ATan(),
                            detach_reset=True
                        )
        self.drop1    = nn.Dropout(dropout)

        # Capa 2
        self.fc2      = nn.Linear(hidden_dim, hidden_dim)
        self.lif2     = neuron.LIFNode(
                            tau=2.0,
                            surrogate_function=surrogate.ATan(),
                            detach_reset=True
                        )
        self.drop2    = nn.Dropout(dropout)

        # Capa de salida (sin neurona LIF, salida continua para CrossEntropy)
        self.fc_out   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: (T, batch, input_dim)
        return: (batch, output_dim)   ← media temporal de logits
        """
        out_sum = torch.zeros(x_seq.shape[1], self.fc_out.out_features,
                              device=x_seq.device)

        for t in range(x_seq.shape[0]):
            h = self.fc1(x_seq[t])
            h = self.lif1(h)
            h = self.drop1(h)

            h = self.fc2(h)
            h = self.lif2(h)
            h = self.drop2(h)

            h = self.fc_out(h)
            out_sum = out_sum + h

        return out_sum / x_seq.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRENAMIENTO Y EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, loss_fn, train: bool):
    """Un epoch de entrenamiento o evaluación."""
    model.train(train)

    total_loss = 0.0
    preds_all, targets_all = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Resetear estados internos de las neuronas LIF entre batches
            functional.reset_net(model)

            # Codificación a spikes
            x_seq = rate_encode(x_batch, T)   # (T, batch, 310)

            logits = model(x_seq)
            loss   = loss_fn(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping: evita explosión de gradientes
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            preds_all.extend(logits.argmax(1).cpu().numpy())
            targets_all.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(targets_all, preds_all)
    f1       = f1_score(targets_all, preds_all, average="macro", zero_division=0)

    return avg_loss, acc, f1, np.array(targets_all), np.array(preds_all)


# ─────────────────────────────────────────────────────────────────────────────
# 7. GRÁFICAS
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"],  label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["test_acc"],  label="Test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_f1"], label="Train")
    axes[2].plot(epochs, history["test_f1"],  label="Test")
    axes[2].set_title("F1 Macro")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Gráfica guardada: {save_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Absoluta
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES,
                ax=axes[0])
    axes[0].set_title("Matriz de confusión (conteos)")
    axes[0].set_ylabel("Real")
    axes[0].set_xlabel("Predicha")

    # Normalizada
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES,
                ax=axes[1])
    axes[1].set_title("Matriz de confusión (normalizada)")
    axes[1].set_ylabel("Real")
    axes[1].set_xlabel("Predicha")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Matriz de confusión guardada: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. DIVISIÓN TRAIN / TEST RESPETANDO TRIALS
# ─────────────────────────────────────────────────────────────────────────────

def split_by_trial_ratio(X_trials: list, y_trials: list, test_ratio: float, rng_seed: int):
    """
    Divide a nivel de trial (no de ventana) para evitar data leakage.

    Un trial completo va entero a train o entero a test.
    Esto es más correcto que mezclar ventanas del mismo trial en ambos conjuntos.
    """
    n_trials = len(X_trials)
    rng = np.random.default_rng(rng_seed)
    idx = rng.permutation(n_trials)

    n_test  = max(1, int(n_trials * test_ratio))
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    X_train = np.vstack([X_trials[i] for i in train_idx]).astype(np.float32)
    y_train = np.concatenate([y_trials[i] for i in train_idx]).astype(np.int64)

    X_test  = np.vstack([X_trials[i] for i in test_idx]).astype(np.float32)
    y_test  = np.concatenate([y_trials[i] for i in test_idx]).astype(np.int64)

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED_RNG)
    print(f"\n{'='*60}")
    print("  SNN para reconocimiento de emociones - Dataset SEED")
    print(f"  Dispositivo: {DEVICE.upper()}")
    print(f"  Feature: {FEATURE_KEY}  |  T={T}  |  hidden={HIDDEN_DIM}")
    print(f"{'='*60}")

    # ── Carga de datos ────────────────────────────────────────────────────────
    print("\n[1/5] Cargando etiquetas...")
    labels = load_labels(LABEL_FILE)

    print("\n[2/5] Cargando características EEG...")
    # Necesitamos los trials por separado para hacer split correcto
    mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.mat")))
    mat_files = [f for f in mat_files if os.path.basename(f).lower() != "label.mat"]

    X_trials_all, y_trials_all = [], []

    for mat_file in mat_files:
        data = loadmat(mat_file)
        for trial_idx in range(1, 16):
            key = f"{FEATURE_KEY}{trial_idx}"
            if key not in data:
                continue
            windows = trial_to_windows(data[key])
            label   = labels[trial_idx - 1]
            X_trials_all.append(windows.astype(np.float32))
            y_trials_all.append(np.full(windows.shape[0], label, dtype=np.int64))

    total_windows = sum(x.shape[0] for x in X_trials_all)
    total_trials  = len(X_trials_all)
    print(f"  ✓ {total_trials} trials cargados  |  {total_windows} ventanas totales")

    # ── División train/test a nivel de trial ──────────────────────────────────
    print("\n[3/5] Dividiendo en train/test (por trial)...")
    X_train, X_test, y_train, y_test = split_by_trial_ratio(
        X_trials_all, y_trials_all, TEST_RATIO, SEED_RNG
    )
    print(f"  Train: {X_train.shape[0]} ventanas  |  Test: {X_test.shape[0]} ventanas")

    # ── Normalización MinMax a [0, 1] para rate coding ────────────────────────
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=(DEVICE == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=(DEVICE == "cuda"))

    # ── Modelo, optimizador, scheduler ───────────────────────────────────────
    print("\n[4/5] Entrenando la SNN...")
    model = SNN_FC(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, dropout=0.3).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Reduce LR si el test_loss no mejora durante 8 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    loss_fn = nn.CrossEntropyLoss()

    history = {k: [] for k in ["train_loss","train_acc","train_f1",
                                "test_loss", "test_acc", "test_f1"]}
    best_acc  = 0.0
    best_epoch = 0

    print(f"\n  {'Epoch':>5} | {'TrainLoss':>9} | {'TrainAcc':>8} | {'TrainF1':>7} | "
          f"{'TestLoss':>8} | {'TestAcc':>7} | {'TestF1':>6}")
    print("  " + "-"*65)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, train_loader, optimizer, loss_fn, train=True)
        te_loss, te_acc, te_f1, y_true, y_pred = run_epoch(model, test_loader, optimizer, loss_fn, train=False)

        scheduler.step(te_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["test_f1"].append(te_f1)

        marker = ""
        if te_acc > best_acc:
            best_acc   = te_acc
            best_epoch = epoch
            torch.save(model.state_dict(), SAVE_MODEL)
            marker = "  ← mejor"
            y_best_true = y_true.copy()
            y_best_pred = y_pred.copy()

        print(f"  {epoch:>5} | {tr_loss:>9.4f} | {tr_acc:>8.4f} | {tr_f1:>7.4f} | "
              f"{te_loss:>8.4f} | {te_acc:>7.4f} | {te_f1:>6.4f}{marker}")

    # ── Resultados finales ────────────────────────────────────────────────────
    print(f"\n[5/5] Resultados finales")
    print(f"  ✓ Mejor epoch: {best_epoch}  |  Mejor test accuracy: {best_acc:.4f}")

    print("\n  Classification report (mejor checkpoint):")
    print(classification_report(y_best_true, y_best_pred,
                                target_names=EMOTION_NAMES, digits=4))

    # ── Gráficas ──────────────────────────────────────────────────────────────
    plot_training_curves(history, SAVE_PLOT)
    plot_confusion_matrix(y_best_true, y_best_pred, SAVE_CM)

    print(f"\n  Modelo guardado en: {SAVE_MODEL}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()