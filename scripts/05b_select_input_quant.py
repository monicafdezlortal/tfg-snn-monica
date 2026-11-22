"""
Elige la cuantizacion de la entrada de 05b (uniforme3 o cuantiles) para cada
numero de canales, por la val_acc media de las semillas (nunca por test).

Lee results/05b_akida_int_input/<canales>_<metodo>_seed<N>/metrics.json y
escribe results/05b_akida_int_input/seleccion.json, que usan 05c y 05d.

Uso:
  .venv-akida\\Scripts\\python scripts/05b_select_input_quant.py
"""

import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from eeg_emotion import config as cfg

EXPERIMENT = "05b_akida_int_input"
RUN_RE = re.compile(r"^(?P<ch>\d+)ch_(?P<variant>.+)_seed(?P<seed>\d+)$")


def main():
    root = os.path.join(str(cfg.RESULTS_DIR), EXPERIMENT)
    runs = defaultdict(list)                      # (canales, variante) -> [(seed, val, test)]
    for path in sorted(glob.glob(os.path.join(root, "*", "metrics.json"))):
        m = RUN_RE.match(os.path.basename(os.path.dirname(path)))
        if not m:
            continue
        with open(path, encoding="utf-8") as f:
            mm = json.load(f)
        runs[(int(m["ch"]), m["variant"])].append(
            (int(m["seed"]), mm["best_val_acc"], mm["accuracy"]))

    seleccion = {"criterio": "val_acc media entre semillas (sujetos 4-5); el test no se usa",
                 "canales": {}}
    for ch in sorted({c for c, _ in runs}):
        filas = {}
        for (c, variant), rr in sorted(runs.items()):
            if c != ch:
                continue
            rr.sort()
            filas[variant] = {
                "semillas": [s for s, _, _ in rr],
                "val_acc_media": round(float(np.mean([v for _, v, _ in rr])), 4),
                "val_acc_std": round(float(np.std([v for _, v, _ in rr])), 4),
                "test_acc_media (solo informativo)": round(float(np.mean([t for _, _, t in rr])), 4),
            }
        elegida = max(filas, key=lambda v: filas[v]["val_acc_media"])
        seleccion["canales"][f"{ch}ch"] = {"elegida": elegida, "variantes": filas}
        print(f"{ch} canales:")
        for v, f in filas.items():
            print(f"  {v:15s} val {f['val_acc_media']:.4f} ± {f['val_acc_std']:.4f}  "
                  f"(test {f['test_acc_media (solo informativo)']:.4f}, n={len(f['semillas'])})"
                  f"{'  <- elegida' if v == elegida else ''}")

    out = os.path.join(root, "seleccion.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(seleccion, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {out}")


if __name__ == "__main__":
    main()
