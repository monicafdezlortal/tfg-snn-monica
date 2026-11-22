"""
Experimento 05c - cuantizacion con QuantizeML a 4 bits (pesos, activaciones y
entrada) y entrenamiento con cuantizacion (QAT).

Parte del modelo de 05b de la misma semilla, con la cuantizacion de entrada
elegida por val_acc (results/05b_akida_int_input/seleccion.json).
Configuracion de Alvaro (run_small_matched_recipe.py):
  QuantizationParams(input_dtype="int8", input_weight_bits=4, weight_bits=4,
                     activation_bits=4, output_bits=8, per_tensor_activations=False)
  PTQ: quantize(..., samples=1024 muestras, batch_size=128, epochs=1)
  QAT: Adam lr 1e-4, batch 128, hasta 30 epochs, paciencia 6
Diferencias con Alvaro:
  - mejor epoch por val_acc (el usaba val_loss), como el resto del TFG
  - las 1024 muestras de calibracion se eligen al azar del train (en SEED
    las primeras 1024 serian el primer trial de un solo sujeto)
  - pesos de clase, como en 02

Uso (entorno .venv-akida):
  .venv-akida\\Scripts\\python scripts/05c_akida_qat.py --channels 4 --seed 0
Salida:
  results/05c_akida_qat/<4ch|62ch>_<metodo>_seed<N>/
    config.json, metrics.json (incluye la PTQ antes de QAT), confusion_matrix.csv, model.keras
"""

import argparse
import os
import time

import numpy as np
from quantizeml.layers import QuantizationParams
from quantizeml.models import quantize

from eeg_emotion import config as cfg
from eeg_emotion import akida_tools as at
from eeg_emotion.results import compute_metrics, save_run

EXPERIMENT = "05c_akida_qat"
MODEL      = "MLP32_QAT4"
PREVIOUS   = "05b_akida_int_input"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", type=int, choices=[4, 62], default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default=None,
                   help="entrada de 05b (por defecto la de seleccion.json)")
    p.add_argument("--epochs", type=int, default=at.QAT_EPOCHS,
                   help="solo para pruebas rapidas (por defecto %(default)s)")
    p.add_argument("--out", default=None, help="carpeta de salida (por defecto en results/)")
    return p.parse_args()


def main():
    args = parse_args()
    at.set_seeds(args.seed)
    variant = args.variant or at.chosen_variant(args.channels)
    input_quant, const_input = at.parse_variant(variant)
    out_dir = args.out or at.stage_dir(EXPERIMENT, args.channels, args.seed, variant)
    os.makedirs(out_dir, exist_ok=True)
    src_dir = at.stage_dir(PREVIOUS, args.channels, args.seed, variant)
    t0 = time.time()

    split, qstats, _ = at.int_inputs(args.channels, input_quant, const_input)
    as_i8 = lambda X: X.astype(np.int8)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = [(as_i8(X), y) for X, y in
                                                (split["train"], split["val"], split["test"])]
    print(f"{EXPERIMENT}  canales={args.channels}  semilla={args.seed}  entrada={variant}")

    float_model = at.load_keras(os.path.join(src_dir, "model.keras"))
    float_preds = at.predict_logits(float_model, X_te).argmax(1)

    # --- PTQ: calibracion con 1024 muestras del train elegidas al azar ---
    rng = np.random.default_rng(args.seed)
    calib_idx = rng.choice(len(X_tr), size=at.CALIBRATION_SAMPLES, replace=False)
    qparams = QuantizationParams(**at.QPARAMS)
    qmodel = quantize(float_model, qparams=qparams, samples=X_tr[calib_idx],
                      batch_size=at.CALIBRATION_BATCH, epochs=1)
    ptq_preds = at.predict_logits(qmodel, X_te).argmax(1)
    ptq = compute_metrics(y_te, ptq_preds)
    print(f"PTQ (antes de QAT): test acc={ptq['accuracy']:.4f}  F1={ptq['f1_macro']:.4f}\n")

    # --- QAT ---
    at.compile_model(qmodel, at.QAT_LR)
    cb = at.train(qmodel, (X_tr, y_tr), (X_va, y_va), args.epochs, at.QAT_BATCH,
                  patience=at.QAT_PATIENCE)
    preds = at.predict_logits(qmodel, X_te).argmax(1)
    test_loss = float(qmodel.evaluate(X_te, y_te, batch_size=4096, verbose=0)[0])
    qmodel.save(os.path.join(out_dir, "model.keras"))

    config = {
        "experiment": EXPERIMENT, "model": MODEL, "channels": args.channels,
        "seed": args.seed, "feature_key": cfg.FEATURE_KEY, "input_dim": int(X_tr.shape[1]),
        "input": "int8 con valores 0..15", "variant": variant,
        "input_quant": input_quant, "const_input": const_input,
        "source_model": os.path.relpath(os.path.join(src_dir, "model.keras"), cfg.REPO_ROOT),
        "train_subjects": split["subjects"]["train"],
        "val_subjects": split["subjects"]["val"], "test_subjects": split["subjects"]["test"],
        "quantization_params": at.QPARAMS,
        "calibration": f"{at.CALIBRATION_SAMPLES} muestras de train al azar "
                       f"(np.random.default_rng({args.seed})), batch {at.CALIBRATION_BATCH}, 1 epoch",
        "qat": {"optimizer": "Adam", "lr": at.QAT_LR, "batch_size": at.QAT_BATCH,
                "max_epochs": args.epochs, "patience": at.QAT_PATIENCE,
                "seleccion": "mejor epoch por val_acc (Alvaro: val_loss)",
                "class_weights": at.class_weight_dict(y_tr)},
        "layers": [l.__class__.__name__ for l in qmodel.layers],
        "versions": at.env_versions(),
    }
    metrics = {"seed": args.seed, **compute_metrics(y_te, preds), "test_loss": test_loss,
               "best_val_acc": cb.best_val_acc, "best_epoch": cb.best_epoch,
               "ptq": {k: ptq[k] for k in ("accuracy", "f1_macro", "confusion_matrix")},
               "accuracy_05b": float(np.mean(float_preds == y_te)),
               "coincidencia_vs_05b": float(np.mean(preds == float_preds)),
               "seconds": round(time.time() - t0, 1), "history": cb.history}
    print(f"\nTEST  acc={metrics['accuracy']:.4f}  F1 macro={metrics['f1_macro']:.4f}  "
          f"(PTQ {ptq['accuracy']:.4f}, 05b {metrics['accuracy_05b']:.4f})")
    print(np.array(metrics["confusion_matrix"]))
    save_run(out_dir, config, metrics)


if __name__ == "__main__":
    main()
