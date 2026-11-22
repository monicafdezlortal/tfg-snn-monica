"""
Herramientas comunes del experimento 05: la ANN en Keras (tf_keras),
cuantizacion de la entrada a 4 bits, QuantizeML y conversion a Akida v1.

Solo funciona en el entorno .venv-akida (Python 3.11, TensorFlow 2.19.1,
QuantizeML 1.2.3, CNN2SNN/Akida 2.19.1). No necesita torch.

La receta sigue la de Alvaro Poveda (akida-model-deployment-pipeline,
scripts/run_small_matched_recipe.py): Dense sin bias -> ReLU(max_value=15)
-> Dense sin bias, entrada entera 0..15 (int8 en Keras/QuantizeML, uint8 en
Akida), QuantizeML a 4 bits y conversion bajo AkidaVersion.v1.
"""

import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers

from . import config as cfg
from .data import load_seed_features, zscore_per_subject, cross_subject_masks

# ============================================================
# HIPERPARAMETROS
# ============================================================
# Red: 1 capa oculta de 32 y dropout 0.35, la mejor configuracion de 04a
# (lr 3.25e-4 y batch 512 tambien vienen de 04a). Sin bias y con ReLU
# acotada a 15, como el modelo de Alvaro para Akida v1.
HIDDEN       = 32
DROPOUT      = 0.35
RELU_MAX     = 15.0
LR           = 3.3e-4
BATCH_SIZE   = 512
EPOCHS       = 50

# QuantizeML + QAT: configuracion de Alvaro (salvo el criterio de mejor
# epoca, que aqui es val_acc como en el resto de experimentos del TFG)
QPARAMS = {
    "input_dtype": "int8",
    "input_weight_bits": 4,
    "weight_bits": 4,
    "activation_bits": 4,
    "output_bits": 8,
    "per_tensor_activations": False,
}
CALIBRATION_SAMPLES = 1024
CALIBRATION_BATCH   = 128
QAT_LR       = 1e-4
QAT_EPOCHS   = 30
QAT_PATIENCE = 6
QAT_BATCH    = 128

# Cuantizacion de la entrada (z-score -> enteros 0..15)
N_LEVELS     = 16
CLIP_SIGMA   = 3.0
INPUT_QUANT  = ("uniforme3", "cuantiles")

DIFERENCIAS_CON_02 = (
    "05 cambia la arquitectura respecto a 02: 1 capa oculta de 32 (02: 128 -> 64), "
    "sin bias, sin BatchNorm, ReLU acotada a 15, dropout 0.35 (02: 0.3), Adam sin "
    "weight decay ni ReduceLROnPlateau, lr 3.3e-4 y batch 512 (02: 1e-3 y 256). "
    "La diferencia con 02 no se debe solo a pasar de PyTorch a Keras."
)


# ============================================================
# DATOS
# ============================================================
def set_seeds(seed):
    """Semillas de Python, numpy y TensorFlow, y operaciones deterministas."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def load_features_cached(channels):
    """(X float64, y, subject) de load_seed_features, guardados la primera vez
    en data/cache/ (fuera de git, como el propio SEED). Leer los 45 .mat en 10
    procesos a la vez agota la memoria; con la cache solo se leen una vez.
    Los valores son exactamente los de load_seed_features."""
    cache = cfg.REPO_ROOT / "data" / "cache" / f"{cfg.FEATURE_KEY}_{channels}ch.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"], d["subject"]
    X_raw, meta = load_seed_features(channels)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X_raw, y=meta["y"], subject=meta["subject"])
    return X_raw, meta["y"], meta["subject"]


def load_split(channels):
    """Features de_LDS con z-score por sujeto (float32) y split cross-subject
    del experimento 02: test 1-3, val 4-5, train 6-15."""
    X_raw, y, subject = load_features_cached(channels)
    Z = zscore_per_subject(X_raw, subject).astype(np.float32)
    tr, va, te = cross_subject_masks(subject)
    return {
        "train": (Z[tr], y[tr]), "val": (Z[va], y[va]), "test": (Z[te], y[te]),
        "subjects": {k: sorted(np.unique(subject[m]).tolist())
                     for k, m in (("train", tr), ("val", va), ("test", te))},
    }


def quantize_input(Z, method, Z_train):
    """z-score -> enteros 0..15 (uint8).
    uniforme3 : recorta a +-3 sigma y reparte [-3, 3] en 16 niveles iguales
                (cada nivel mide 0.375 sigma). No usa Z_train.
    cuantiles : 15 umbrales por feature en los cuantiles 1/16..15/16 del
                train, asi cada nivel recibe ~1/16 de los valores de train."""
    if method == "uniforme3":
        q = np.floor((np.clip(Z, -CLIP_SIGMA, CLIP_SIGMA) + CLIP_SIGMA)
                     / (2 * CLIP_SIGMA) * N_LEVELS)
        return np.clip(q, 0, N_LEVELS - 1).astype(np.uint8)
    if method == "cuantiles":
        edges = np.quantile(Z_train, np.linspace(0, 1, N_LEVELS + 1)[1:-1], axis=0)
        q = np.stack([np.searchsorted(edges[:, j], Z[:, j], side="right")
                      for j in range(Z.shape[1])], axis=1)
        return q.astype(np.uint8)
    raise ValueError(f"metodo de cuantizacion desconocido: {method}")


def quantized_split(split, method):
    """Aplica quantize_input a train/val/test. Devuelve (split_q, estadisticas)."""
    Z_train = split["train"][0]
    out = {k: (quantize_input(split[k][0], method, Z_train), split[k][1])
           for k in ("train", "val", "test")}
    stats = {"metodo": method}
    for k in ("train", "val", "test"):
        Z, q = split[k][0], out[k][0]
        stats[k] = {
            "recortado_pct": float(100 * np.mean(np.abs(Z) > CLIP_SIGMA))
                             if method == "uniforme3" else 0.0,
            "uso_niveles_pct": np.round(100 * np.bincount(q.ravel(), minlength=N_LEVELS)
                                        / q.size, 2).tolist(),
        }
    return out, stats


CONST_VALUE = 15


def add_const_input(X_u8):
    """Plan B (solo si 05b cae mucho): anade una columna constante = 15. La red
    no tiene bias, y con entradas 0..15 la primera capa no puede desplazar el
    umbral de cada neurona; el peso de esta columna hace de bias."""
    const = np.full((len(X_u8), 1), CONST_VALUE, dtype=np.uint8)
    return np.concatenate([X_u8, const], axis=1)


def int_inputs(channels, input_quant, const_input=False):
    """Split con la entrada cuantizada a 0..15 (uint8), opcionalmente con la
    columna constante. Devuelve (split_q, estadisticas, split float)."""
    split = load_split(channels)
    split_q, stats = quantized_split(split, input_quant)
    if const_input:
        split_q = {k: (add_const_input(X), y) for k, (X, y) in split_q.items()}
    split_q["subjects"] = split["subjects"]
    return split_q, stats, split


def class_weight_dict(y):
    """Pesos inversos a la frecuencia, con media 1 (misma formula que
    eeg_emotion.training.class_weights del experimento 02)."""
    counts = np.bincount(y, minlength=3).astype(np.float64)
    w = counts.sum() / (3.0 * counts)
    w = w / w.mean()
    return {i: float(v) for i, v in enumerate(w)}


# ============================================================
# MODELO
# ============================================================
def build_model(input_dim, input_dtype="float32", dropout=DROPOUT, hidden=HIDDEN,
                name="emotion_mlp_akida"):
    """entrada -> Dense(hidden, sin bias) -> ReLU(max 15) -> Dropout -> Dense(3, sin bias).
    El Dropout solo actua al entrenar: QuantizeML lo envuelve como
    QuantizedDropout y CNN2SNN lo elimina al convertir (comprobado)."""
    inp = keras.Input(shape=(input_dim,), dtype=input_dtype, name="entrada")
    x = layers.Dense(hidden, use_bias=False, name="dense1")(inp)
    x = layers.ReLU(max_value=RELU_MAX, name="relu1")(x)
    if dropout:
        x = layers.Dropout(dropout, name="dropout1")(x)
    out = layers.Dense(3, use_bias=False, name="dense2")(x)
    return keras.Model(inp, out, name=name)


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )


def n_params(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


class BestValAcc(keras.callbacks.Callback):
    """Guarda en memoria los pesos de la epoch con mejor val_acc y los
    restaura al terminar (mismo criterio que eeg_emotion.training.fit).
    Con patience, para si val_acc no mejora en `patience` epochs."""

    def __init__(self, patience=None):
        super().__init__()
        self.patience = patience
        self.best_val_acc, self.best_epoch = -1.0, 0
        self.best_weights, self.wait, self.history = None, 0, []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = float(logs["val_acc"])
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        self.history.append({"epoch": epoch + 1, "train_loss": float(logs["loss"]),
                             "train_acc": float(logs["acc"]),
                             "val_loss": float(logs["val_loss"]), "val_acc": val_acc,
                             "lr": lr})
        marker = ""
        if val_acc > self.best_val_acc:
            self.best_val_acc, self.best_epoch = val_acc, epoch + 1
            self.best_weights = self.model.get_weights()
            self.wait, marker = 0, "  *"
        else:
            self.wait += 1
            if self.patience is not None and self.wait >= self.patience:
                self.model.stop_training = True
        print(f"{epoch + 1:4d} {logs['loss']:10.4f} {logs['val_loss']:9.4f} "
              f"{val_acc:8.4f}{marker}")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        print(f"Mejor val_acc: {self.best_val_acc:.4f} (epoch {self.best_epoch})")


def train(model, train, val, epochs, batch_size, patience=None):
    """Entrena con pesos de clase y se queda con la mejor epoch por val_acc."""
    (X_tr, y_tr), (X_va, y_va) = train, val
    cb = BestValAcc(patience)
    print(f"{'Ep':>4} {'train_loss':>10} {'val_loss':>9} {'val_acc':>8}")
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
              batch_size=batch_size, shuffle=True, verbose=0,
              class_weight=class_weight_dict(y_tr), callbacks=[cb])
    return cb


def load_keras(path):
    """Carga un .keras (float o cuantizado). Importar quantizeml.layers registra
    sus capas en Keras; quantizeml.load_model solo acepta .h5/.onnx. Es como lo
    carga Alvaro en su notebook (keras.models.load_model, compile=False)."""
    import quantizeml.layers  # noqa: F401
    return keras.models.load_model(str(path), compile=False)


def predict_logits(model, X):
    return np.asarray(model.predict(X, batch_size=4096, verbose=0)).reshape(len(X), -1)


def predict_akida(ak_model, X_u8):
    """Logits del modelo Akida (simulador o placa) para una matriz (N, F) uint8."""
    out = ak_model.predict(np.ascontiguousarray(X_u8, dtype=np.uint8))
    return np.asarray(out, dtype=np.float32).reshape(len(X_u8), -1)


def chosen_variant(channels):
    """Variante de entrada de 05b elegida por val_acc para estos canales
    (results/05b_akida_int_input/seleccion.json, de scripts/05b_select_input_quant.py)."""
    import json
    path = os.path.join(str(cfg.RESULTS_DIR), "05b_akida_int_input", "seleccion.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["canales"][f"{channels}ch"]["elegida"]


def parse_variant(variant):
    """'uniforme3_const' -> ('uniforme3', True)"""
    const = variant.endswith("_const")
    return (variant[: -len("_const")] if const else variant), const


def stage_dir(experiment, channels, seed, input_quant=None):
    """results/<experimento>/<canales>[_<cuantizacion>]_seed<N>/"""
    from .results import run_dir
    ch = f"{channels}ch" + (f"_{input_quant}" if input_quant else "")
    return run_dir(experiment, ch, seed)


def env_versions():
    import akida, cnn2snn, quantizeml
    return {"tensorflow": tf.__version__, "tf_keras": keras.__version__,
            "quantizeml": quantizeml.__version__, "cnn2snn": cnn2snn.__version__,
            "akida": akida.__version__, "numpy": np.__version__,
            # en CPU el numero de hilos puede cambiar los ultimos decimales
            "TF_NUM_INTRAOP_THREADS": os.environ.get("TF_NUM_INTRAOP_THREADS", "por defecto"),
            "TF_NUM_INTEROP_THREADS": os.environ.get("TF_NUM_INTEROP_THREADS", "por defecto")}
