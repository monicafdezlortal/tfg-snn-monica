"""
Experimento 05a - ANN para Akida en Keras (tf_keras), en float.

Primera etapa del despliegue en AKD1000. Es la red que despues se cuantiza
(05b, 05c) y se convierte a Akida (05d), todavia con entrada float (z-score).

  - Red: entrada -> Dense(32, sin bias) -> ReLU(max 15) -> Dropout(0.35)
         -> Dense(3, sin bias). Sin BatchNorm (Akida v1 no la admite).
  - Mismo split que 02: test sujetos 1-3, val 4-5, train 6-15.
  - z-score por sujeto, pesos de clase, 50 epochs, mejor epoch por val_acc.

Ojo: la arquitectura no es la de 02 (ver DIFERENCIAS_CON_02 en
eeg_emotion/akida_tools.py), asi que la diferencia con 02 no es solo por
pasar de PyTorch a Keras.

Con --config 04a_4ch usa en su lugar la mejor configuracion de 4 canales de
04a (trial 33: 32 -> 16, dropout 0.138, lr 1.14e-4, batch 512; sin su weight
decay, como el resto de 05) y guarda en results/05a_akida_float_cfg4ch/.
Es una prueba para ver si con esa red se reduce la diferencia con 62 canales.

Uso (entorno .venv-akida, Python 3.11):
  .venv-akida\\Scripts\\python scripts/05a_akida_float.py --channels 4 --seed 0
  .venv-akida\\Scripts\\python scripts/05a_akida_float.py --channels 4 --seed 0 --config 04a_4ch
Salida:
  results/05a_akida_float/<4ch|62ch>_seed<N>/
    config.json, metrics.json, confusion_matrix.csv, model.keras
"""

import argparse
import os
import time

import numpy as np

from eeg_emotion import config as cfg
from eeg_emotion import akida_tools as at
from eeg_emotion.results import compute_metrics, save_run

# configuracion -> (experimento, modelo, capas ocultas, dropout, lr, batch, origen)
CONFIGS = {
    # mejor configuracion de 04a con 62 canales (trial 27): la de 05a-05d
    "04a_62ch": ("05a_akida_float", "MLP32_float", at.HIDDEN, at.DROPOUT, at.LR,
                 at.BATCH_SIZE, "04a trial 27 (mejor con 62 canales)"),
    # mejor configuracion de 04a con 4 canales (trial 33): prueba
    "04a_4ch": ("05a_akida_float_cfg4ch", "MLP32x16_float", (32, 16), 0.138, 1.1357e-4,
                512, "04a trial 33 (mejor con 4 canales); sin su weight decay 1.66e-4"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", type=int, choices=[4, 62], default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", choices=list(CONFIGS), default="04a_62ch",
                   help="red y entrenamiento (por defecto %(default)s, la de 05a-05d)")
    p.add_argument("--epochs", type=int, default=at.EPOCHS,
                   help="solo para pruebas rapidas (por defecto %(default)s)")
    p.add_argument("--out", default=None, help="carpeta de salida (por defecto en results/)")
    return p.parse_args()


def main():
    args = parse_args()
    EXPERIMENT, MODEL, hidden, dropout, lr, batch_size, origin = CONFIGS[args.config]
    at.set_seeds(args.seed)
    out_dir = args.out or at.stage_dir(EXPERIMENT, args.channels, args.seed)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    split = at.load_split(args.channels)
    X_tr, y_tr = split["train"]
    X_te, y_te = split["test"]
    input_dim = X_tr.shape[1]
    print(f"{EXPERIMENT}  canales={args.channels}  semilla={args.seed}  "
          f"train={len(y_tr)} val={len(split['val'][1])} test={len(y_te)}")

    model = at.build_model(input_dim, input_dtype="float32", dropout=dropout, hidden=hidden)
    at.compile_model(model, lr)
    print(f"Parametros entrenables: {at.n_params(model):,}\n")
    cb = at.train(model, split["train"], split["val"], args.epochs, batch_size)

    logits = at.predict_logits(model, X_te)
    preds = logits.argmax(1)
    test_loss = float(model.evaluate(X_te, y_te, batch_size=4096, verbose=0)[0])
    model.save(os.path.join(out_dir, "model.keras"))

    config = {
        "experiment": EXPERIMENT, "model": MODEL, "channels": args.channels,
        "seed": args.seed, "feature_key": cfg.FEATURE_KEY, "input_dim": input_dim,
        "input": "float32, z-score por sujeto",
        "normalization": "z-score por sujeto",
        "train_subjects": split["subjects"]["train"],
        "val_subjects": split["subjects"]["val"], "test_subjects": split["subjects"]["test"],
        "architecture": f"{input_dim} -> " + "".join(
            f"Dense({h}, sin bias) -> ReLU(max {at.RELU_MAX:g}) -> Dropout({dropout}) -> "
            for h in ([hidden] if isinstance(hidden, int) else hidden)) + "Dense(3, sin bias)",
        "hidden": hidden if isinstance(hidden, int) else list(hidden),
        "dropout": dropout, "relu_max": at.RELU_MAX, "use_bias": False,
        "batchnorm": False, "epochs": args.epochs, "batch_size": batch_size, "lr": lr,
        "optimizer": "Adam", "loss": "SparseCategoricalCrossentropy(from_logits=True)",
        "class_weights": at.class_weight_dict(y_tr),
        "seleccion": "mejor epoch por val_acc",
        "n_params": at.n_params(model),
        "diferencias_con_02": at.DIFERENCIAS_CON_02,
        **({} if args.config == "04a_62ch" else {"config": args.config, "origen": origin}),
        "versions": at.env_versions(),
    }
    metrics = {"seed": args.seed, **compute_metrics(y_te, preds), "test_loss": test_loss,
               "best_val_acc": cb.best_val_acc, "best_epoch": cb.best_epoch,
               "seconds": round(time.time() - t0, 1), "history": cb.history}
    print(f"\nTEST  acc={metrics['accuracy']:.4f}  F1 macro={metrics['f1_macro']:.4f}")
    print(np.array(metrics["confusion_matrix"]))
    save_run(out_dir, config, metrics)


if __name__ == "__main__":
    main()
