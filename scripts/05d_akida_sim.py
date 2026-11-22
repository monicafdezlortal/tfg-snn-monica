"""
Experimento 05d - conversion a Akida con CNN2SNN (AkidaVersion.v1), exportacion
a .fbz y ejecucion en el simulador de Akida (sin placa).

Parte del modelo QAT de 05c de la misma semilla. Como Alvaro, la entrada a
Keras/QuantizeML es int8 y la de Akida uint8, con los mismos valores 0..15.
Compara con 05c: accuracy y porcentaje de predicciones que coinciden.
Ademas mapea el modelo a un AKD1000 virtual (hw_only=True) para comprobar que
cabe entero en el chip, y exporta lo que necesita 05e en la placa.

Uso (entorno .venv-akida):
  .venv-akida\\Scripts\\python scripts/05d_akida_sim.py --channels 4 --seed 0
Salida:
  results/05d_akida_sim/<4ch|62ch>_<metodo>_seed<N>/
    config.json, metrics.json, confusion_matrix.csv, model_summary.txt
    model.fbz, test_inputs_uint8.npy, test_labels.npy, sim_logits.npy
    (los .fbz y .npy no se suben a git: los .npy son datos de SEED)
"""

import argparse
import hashlib
import os
import time

import numpy as np
import akida
from cnn2snn import AkidaVersion, convert, set_akida_version

from eeg_emotion import config as cfg
from eeg_emotion import akida_tools as at
from eeg_emotion.results import compute_metrics, save_run

EXPERIMENT = "05d_akida_sim"
MODEL      = "MLP32_akida_sim"
PREVIOUS   = "05c_akida_qat"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channels", type=int, choices=[4, 62], default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default=None,
                   help="entrada de 05b (por defecto la de seleccion.json)")
    p.add_argument("--out", default=None, help="carpeta de salida (por defecto en results/)")
    return p.parse_args()


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    args = parse_args()
    at.set_seeds(args.seed)
    variant = args.variant or at.chosen_variant(args.channels)
    input_quant, const_input = at.parse_variant(variant)
    out_dir = args.out or at.stage_dir(EXPERIMENT, args.channels, args.seed, variant)
    os.makedirs(out_dir, exist_ok=True)
    src_dir = at.stage_dir(PREVIOUS, args.channels, args.seed, variant)
    fbz_path = os.path.join(out_dir, "model.fbz")
    t0 = time.time()

    split, _, _ = at.int_inputs(args.channels, input_quant, const_input)
    X_te_u8, y_te = split["test"]
    print(f"{EXPERIMENT}  canales={args.channels}  semilla={args.seed}  entrada={variant}")

    qat_model = at.load_keras(os.path.join(src_dir, "model.keras"))
    qat_preds = at.predict_logits(qat_model, X_te_u8.astype(np.int8)).argmax(1)

    with set_akida_version(AkidaVersion.v1):
        ak_model = convert(qat_model, file_path=fbz_path)
    ak_model = akida.Model(fbz_path)                   # se evalua el .fbz guardado
    t_sim = time.time()
    sim_logits = at.predict_akida(ak_model, X_te_u8)
    sim_seconds = time.time() - t_sim
    preds = sim_logits.argmax(1)
    summary = [f"{type(l).__name__}: {l}" for l in ak_model.layers]

    # Mapeo a un AKD1000 virtual: comprueba que el modelo cabe en el chip
    mapping = {}
    try:
        ak_hw = akida.Model(fbz_path)
        ak_hw.map(akida.AKD1000(), hw_only=True)
        mapping = {"status": "ok", "n_sequences": len(ak_hw.sequences),
                   "program_bytes": [len(s.program) for s in ak_hw.sequences]}
    except Exception as exc:                            # se registra, no se oculta
        mapping = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    np.save(os.path.join(out_dir, "test_inputs_uint8.npy"), X_te_u8)
    np.save(os.path.join(out_dir, "test_labels.npy"), y_te)
    np.save(os.path.join(out_dir, "sim_logits.npy"), sim_logits)
    with open(os.path.join(out_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n")

    config = {
        "experiment": EXPERIMENT, "model": MODEL, "channels": args.channels,
        "seed": args.seed, "feature_key": cfg.FEATURE_KEY, "input_dim": int(X_te_u8.shape[1]),
        "input": "uint8 con valores 0..15", "variant": variant,
        "input_quant": input_quant, "const_input": const_input,
        "source_model": os.path.relpath(os.path.join(src_dir, "model.keras"), cfg.REPO_ROOT),
        "test_subjects": split["subjects"]["test"],
        "conversion": "cnn2snn.convert dentro de set_akida_version(AkidaVersion.v1)",
        "backend": "simulador de Akida (software, sin placa)",
        "ip_version": str(ak_model.ip_version),
        "akida_layers": summary,
        "fbz": {"path": os.path.relpath(fbz_path, cfg.REPO_ROOT),
                "bytes": os.path.getsize(fbz_path), "sha256": sha256(fbz_path)},
        "mapeo_akd1000_virtual": mapping,
        "versions": at.env_versions(),
    }
    metrics = {"seed": args.seed, **compute_metrics(y_te, preds),
               "accuracy_05c": float(np.mean(qat_preds == y_te)),
               "coincidencia_vs_05c": float(np.mean(preds == qat_preds)),
               "sim_seconds_test": round(sim_seconds, 2),
               "seconds": round(time.time() - t0, 1)}
    print(f"\nTEST  acc={metrics['accuracy']:.4f}  F1 macro={metrics['f1_macro']:.4f}  "
          f"(05c {metrics['accuracy_05c']:.4f}, coincidencia {metrics['coincidencia_vs_05c']:.2%}, "
          f"mapeo AKD1000 virtual: {mapping['status']})")
    print(np.array(metrics["confusion_matrix"]))
    save_run(out_dir, config, metrics)


if __name__ == "__main__":
    main()
