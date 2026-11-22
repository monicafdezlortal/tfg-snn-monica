"""
Experimento 05b - la misma red que 05a, reentrenada con la entrada cuantizada
a enteros 0..15 (4 bits), que es lo que recibe el AKD1000.

Como en la receta de Alvaro, la red float se entrena directamente con la
entrada entera (aqui int8 con valores 0..15, que es el tipo que espera
QuantizeML en 05c). Dos formas de pasar del z-score a 0..15:
  uniforme3 : recortar a +-3 sigma y repartir [-3, 3] en 16 niveles iguales
              (metodo principal)
  cuantiles : 15 umbrales por feature en los cuantiles del train (variante)
La que se usa en 05c se elige por val_acc con scripts/05b_select_input_quant.py.

--const-input (plan B): anade una entrada constante = 15 que hace de bias.

Uso (entorno .venv-akida):
  .venv-akida\\Scripts\\python scripts/05b_akida_int_input.py --channels 4 --seed 0 --input-quant uniforme3
Salida:
  results/05b_akida_int_input/<4ch|62ch>_<metodo>[_const]_seed<N>/
    config.json, metrics.json, confusion_matrix.csv, model.keras
"""

import argparse
import os
import time

import numpy as np

from eeg_emotion import config as cfg
from eeg_emotion import akida_tools as at
from eeg_emotion.results import compute_metrics, save_run

EXPERIMENT = "05b_akida_int_input"
MODEL      = "MLP32_int4in"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", type=int, choices=[4, 62], default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input-quant", choices=at.INPUT_QUANT, default="uniforme3")
    p.add_argument("--const-input", action="store_true",
                   help="plan B: entrada constante = 15 como bias")
    p.add_argument("--epochs", type=int, default=at.EPOCHS,
                   help="solo para pruebas rapidas (por defecto %(default)s)")
    p.add_argument("--out", default=None, help="carpeta de salida (por defecto en results/)")
    return p.parse_args()


def variant_name(input_quant, const_input):
    return input_quant + ("_const" if const_input else "")


def main():
    args = parse_args()
    at.set_seeds(args.seed)
    variant = variant_name(args.input_quant, args.const_input)
    out_dir = args.out or at.stage_dir(EXPERIMENT, args.channels, args.seed, variant)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    split, qstats, _ = at.int_inputs(args.channels, args.input_quant, args.const_input)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split["train"], split["val"], split["test"]
    as_i8 = lambda X: X.astype(np.int8)                   # 0..15 cabe en int8
    input_dim = X_tr.shape[1]
    print(f"{EXPERIMENT}  canales={args.channels}  semilla={args.seed}  entrada={variant}")
    for k in ("train", "val", "test"):
        print(f"  {k}: recortado {qstats[k]['recortado_pct']:.2f} %  "
              f"uso de niveles {qstats[k]['uso_niveles_pct']}")

    model = at.build_model(input_dim, input_dtype="int8")
    at.compile_model(model, at.LR)
    print(f"Parametros entrenables: {at.n_params(model):,}\n")
    cb = at.train(model, (as_i8(X_tr), y_tr), (as_i8(X_va), y_va), args.epochs, at.BATCH_SIZE)

    preds = at.predict_logits(model, as_i8(X_te)).argmax(1)
    test_loss = float(model.evaluate(as_i8(X_te), y_te, batch_size=4096, verbose=0)[0])
    model.save(os.path.join(out_dir, "model.keras"))

    config = {
        "experiment": EXPERIMENT, "model": MODEL, "channels": args.channels,
        "seed": args.seed, "feature_key": cfg.FEATURE_KEY, "input_dim": input_dim,
        "input": "int8 con valores 0..15 (z-score por sujeto cuantizado a 4 bits)",
        "input_quant": args.input_quant, "const_input": args.const_input,
        "input_quant_detail": {
            "uniforme3": "clip(z, -3, 3), 16 niveles de 0.375 sigma",
            "cuantiles": "15 umbrales por feature en los cuantiles k/16 del train",
        }[args.input_quant] + (" + columna constante = 15 (plan B, hace de bias)"
                               if args.const_input else ""),
        "normalization": "z-score por sujeto",
        "train_subjects": split["subjects"]["train"],
        "val_subjects": split["subjects"]["val"], "test_subjects": split["subjects"]["test"],
        "architecture": f"{input_dim} -> Dense({at.HIDDEN}, sin bias) -> ReLU(max {at.RELU_MAX:g}) "
                        f"-> Dropout({at.DROPOUT}) -> Dense(3, sin bias)",
        "hidden": at.HIDDEN, "dropout": at.DROPOUT, "relu_max": at.RELU_MAX, "use_bias": False,
        "batchnorm": False, "epochs": args.epochs, "batch_size": at.BATCH_SIZE, "lr": at.LR,
        "optimizer": "Adam", "loss": "SparseCategoricalCrossentropy(from_logits=True)",
        "class_weights": at.class_weight_dict(y_tr),
        "seleccion": "mejor epoch por val_acc",
        "n_params": at.n_params(model),
        "diferencias_con_02": at.DIFERENCIAS_CON_02,
        "versions": at.env_versions(),
    }
    metrics = {"seed": args.seed, **compute_metrics(y_te, preds), "test_loss": test_loss,
               "best_val_acc": cb.best_val_acc, "best_epoch": cb.best_epoch,
               "input_quant_stats": qstats,
               "seconds": round(time.time() - t0, 1), "history": cb.history}
    print(f"\nTEST  acc={metrics['accuracy']:.4f}  F1 macro={metrics['f1_macro']:.4f}")
    print(np.array(metrics["confusion_matrix"]))
    save_run(out_dir, config, metrics)


if __name__ == "__main__":
    main()
