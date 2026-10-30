"""
Agrega los resultados de varias semillas: media ± desviacion por experimento
y numero de canales.

Lee results/<experimento>/<canales>_seed<N>/metrics.json y escribe
results/summary.csv con una fila por (experimento, modelo, canales). Si un
metrics.json trae varios modelos (metrics["modelos"], p.ej. ANN y SVM en el
experimento 01), cada uno da su propia fila. La desviacion es la poblacional
(np.std, ddof=0), igual que en el resto de scripts.

Uso:
  python scripts/aggregate_seeds.py
  python scripts/aggregate_seeds.py --experiments 02_ann_cross_subject 03_snn_cross_subject
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from eeg_emotion import config as cfg

RUN_RE = re.compile(r"^(?P<channels>.+)_seed(?P<seed>-?\d+)$")
FIELDS = ["experimento", "modelo", "canales", "n_semillas", "semillas",
          "accuracy_media", "accuracy_std", "f1_macro_media", "f1_macro_std",
          "accuracy", "f1_macro"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(cfg.RESULTS_DIR))
    p.add_argument("--experiments", nargs="*", default=None,
                   help="por defecto, todas las carpetas de results/ con metrics.json")
    p.add_argument("--out", default=None, help="por defecto results/summary.csv")
    return p.parse_args()


def run_models(path, metrics):
    """{modelo: metricas} de un run: metrics["modelos"] si existe; si no, las
    metricas de primer nivel con el nombre del modelo de config.json."""
    if "modelos" in metrics:
        return metrics["modelos"]
    model = ""
    config_path = os.path.join(os.path.dirname(path), "config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            model = json.load(f).get("model", "")
    return {model: metrics}


def main():
    args = parse_args()
    groups = defaultdict(list)          # (experimento, modelo, canales) -> [(seed, metrics)]
    for path in sorted(glob.glob(os.path.join(args.results, "*", "*", "metrics.json"))):
        run = os.path.basename(os.path.dirname(path))
        exp = os.path.basename(os.path.dirname(os.path.dirname(path)))
        m = RUN_RE.match(run)
        if not m or (args.experiments and exp not in args.experiments):
            continue
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)
        for model, mm in run_models(path, metrics).items():
            if "accuracy" in mm and "f1_macro" in mm:
                groups[(exp, model, m["channels"])].append((int(m["seed"]), mm))

    rows = []
    for (exp, model, ch), runs in sorted(groups.items()):
        runs.sort()
        acc = np.array([r["accuracy"] for _, r in runs])
        f1 = np.array([r["f1_macro"] for _, r in runs])
        rows.append({
            "experimento": exp, "modelo": model, "canales": ch, "n_semillas": len(runs),
            "semillas": " ".join(str(s) for s, _ in runs),
            "accuracy_media": round(float(acc.mean()), 4),
            "accuracy_std": round(float(acc.std()), 4),
            "f1_macro_media": round(float(f1.mean()), 4),
            "f1_macro_std": round(float(f1.std()), 4),
            "accuracy": f"{acc.mean():.4f} ± {acc.std():.4f}",
            "f1_macro": f"{f1.mean():.4f} ± {f1.std():.4f}",
        })

    out = args.out or os.path.join(args.results, "summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"{'experimento':22s} {'modelo':11s} {'canales':9s} {'n':>2}  "
          f"{'accuracy':>17}  {'f1_macro':>17}")
    for r in rows:
        print(f"{r['experimento']:22s} {r['modelo']:11s} {r['canales']:9s} {r['n_semillas']:2d}  "
              f"{r['accuracy']:>17}  {r['f1_macro']:>17}")
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
