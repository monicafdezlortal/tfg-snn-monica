#!/usr/bin/env python3
"""
Experimento 05e - el modelo .fbz de 05d en la placa BrainChip AKD1000: latencia,
potencia y energia en las 9 combinaciones de ClockMode x MapMode.

Sigue la medida de Alvaro Poveda (akida-model-deployment-pipeline,
run_small_matched_recipe.py, run_hardware_sweep):
  - governor de la CPU en "performance" (se escribe en /sys: lanzar con sudo)
  - para cada ClockMode (Performance, Economy, LowPower) x MapMode (AllNps,
    HwPr, Minimal) y cada repeticion (5): cargar el .fbz, fijar el reloj,
    activar la medida de potencia, mapear con hw_only=True, 5 muestras de
    calentamiento, predict muestra a muestra midiendo la latencia, y una
    pasada en lote para la potencia
  - energia estimada (uJ) = potencia media (mW) x latencia media (ms)
La latencia es la observada desde el host alrededor de model.predict (no la
interna del chip) y la potencia es la que da la API de Akida, no la de todo
el sistema.

Solo necesita akida y numpy (el entorno de akd1000-bringup sirve tal cual);
no usa eeg_emotion, TensorFlow ni SEED en bruto. Copia a la Raspberry Pi la
carpeta de 05d del modelo que quieras medir (model.fbz, test_inputs_uint8.npy,
test_labels.npy, sim_logits.npy): los .npy son datos de SEED y no estan en git.

Uso (en la Raspberry Pi con la placa):
  sudo -E .venv/bin/python scripts/05e_akida_hardware.py \\
      --run-dir results/05d_akida_sim/4ch_uniforme3_seed0
  (--max-samples 1000 para una prueba rapida)
Salida:
  results/05e_akida_hardware/<nombre de la carpeta de 05d>/
    hardware_9modos.csv      una fila por combinacion (media y desviacion de las repeticiones)
    hardware_repeticiones.csv  una fila por repeticion
    hardware.json            entorno, governor, modo de menor energia y avisos
    latencias_ms.npy         latencia de cada muestra (modo, repeticion, muestra)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CLOCKS = ["Performance", "Economy", "LowPower"]
MAPS = ["AllNps", "HwPr", "Minimal"]
CLASS_NAMES = ["negativa", "neutra", "positiva"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="carpeta de 05d con model.fbz y los .npy de test")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=None,
                   help="usar solo las primeras N muestras de test (pruebas)")
    p.add_argument("--no-governor", action="store_true",
                   help="no tocar el governor de la CPU")
    p.add_argument("--out", default=None)
    return p.parse_args()


# ------------------------------------------------------------ utilidades
def now_iso():
    return datetime.now(timezone.utc).isoformat()


def is_number(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) \
        and math.isfinite(float(v))


def run_cmd(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else f"error: {r.stderr.strip()}"
    except Exception as exc:
        return f"no disponible: {exc!r}"


def cpu_files(name):
    return sorted(Path("/sys/devices/system/cpu").glob(f"cpu*/cpufreq/{name}"))


def read_cpu(name):
    out = {}
    for p in cpu_files(name):
        try:
            out[str(p)] = p.read_text(encoding="utf-8").strip()
        except Exception as exc:
            out[str(p)] = f"error: {exc!r}"
    return out


def set_governor_performance():
    before, errors = read_cpu("scaling_governor"), {}
    for p in cpu_files("scaling_governor"):
        try:
            p.write_text("performance\n", encoding="utf-8")
        except Exception as exc:
            errors[str(p)] = repr(exc)
    after = read_cpu("scaling_governor")
    return {"before": before, "after": after, "errors": errors,
            "all_performance": bool(after) and all(v == "performance" for v in after.values())}


def metrics(y, pred):
    acc = float(np.mean(pred == y))
    f1 = []
    for c in range(len(CLASS_NAMES)):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"accuracy": acc, "f1_macro": float(np.mean(f1))}


def power_summary(model):
    """Potencia de la API de Akida tras una pasada (como summarize_power de Alvaro)."""
    stats = getattr(model, "statistics", None)
    powers = getattr(stats, "powers", None)
    d = {}
    if isinstance(powers, dict):
        d = {str(k): float(v) for k, v in powers.items() if is_number(v)}
    elif powers is not None:
        for k in ("Avg", "Min", "Max", "avg", "min", "max"):
            v = getattr(powers, k, None)
            if is_number(v):
                d[k] = float(v)
    avg = d.get("Avg", d.get("avg"))
    mn = d.get("Min", d.get("min"))
    return {"power_mw_mean": avg, "power_mw_floor": mn, "power_raw": d,
            "statistics_repr": repr(stats)[:2000]}


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------ medida
def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    fbz = run_dir / "model.fbz"
    X = np.load(run_dir / "test_inputs_uint8.npy").astype(np.uint8)
    y = np.load(run_dir / "test_labels.npy")
    sim_pred = np.load(run_dir / "sim_logits.npy").argmax(1)
    if args.max_samples:
        X, y, sim_pred = X[: args.max_samples], y[: args.max_samples], sim_pred[: args.max_samples]
    out_dir = Path(args.out or Path("results") / "05e_akida_hardware" / run_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    import akida
    from akida import MapMode
    from akida.core.soc import ClockMode

    info = {
        "timestamp": now_iso(), "run_dir": str(run_dir), "fbz": str(fbz),
        "n_samples": int(len(y)), "repeats": args.repeats, "warmup": args.warmup,
        "akida": akida.__version__, "numpy": np.__version__, "python": sys.version,
        "platform": platform.platform(),
        "rpi": {"arm_clock": run_cmd(["vcgencmd", "measure_clock", "arm"]),
                "temp": run_cmd(["vcgencmd", "measure_temp"]),
                "throttled": run_cmd(["vcgencmd", "get_throttled"])},
        "cpu_freq_before": read_cpu("scaling_cur_freq"),
        "governor": None if args.no_governor else set_governor_performance(),
        "notas": ["latencia observada desde el host alrededor de model.predict, no la interna del chip",
                  "potencia de la API de Akida, no la de todo el sistema",
                  "energia_uJ = potencia_media_mW x latencia_media_ms (por repeticion)"],
    }
    if info["governor"] and not info["governor"]["all_performance"]:
        print("AVISO: no se pudo poner el governor en performance (lanzar con sudo).")

    devs = akida.devices()
    info["devices"] = repr(devs)
    if not devs:
        rows = [{"clock_mode": c, "map_mode": m, "status": "not_detected"} for c in CLOCKS for m in MAPS]
        write_csv(out_dir / "hardware_9modos.csv", rows)
        info["status"] = "not_detected"
        (out_dir / "hardware.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"No se ha detectado ningun AKD1000. Guardado en {out_dir}")
        return 1

    device = devs[0]
    info["device_version"] = repr(getattr(device, "version", None))
    original_clock = getattr(getattr(device, "soc", None), "clock_mode", None)
    lat_all = np.full((len(CLOCKS) * len(MAPS), args.repeats, len(y)), np.nan, dtype=np.float32)
    rows = []
    for ci, clock in enumerate(CLOCKS):
        for mi, mode in enumerate(MAPS):
            k = ci * len(MAPS) + mi
            for rep in range(args.repeats):
                row = {"clock_mode": clock, "map_mode": mode, "repeat": rep, "status": "running"}
                t_rep = time.perf_counter()
                try:
                    model = akida.Model(str(fbz))
                    # Como Alvaro: si no se puede fijar el reloj o activar la
                    # potencia, se apunta el aviso en la fila y se sigue
                    try:
                        device.soc.clock_mode = getattr(ClockMode, clock)
                    except Exception as exc:
                        row["aviso_reloj"] = f"{type(exc).__name__}: {exc}"
                        print(f"AVISO {clock}/{mode}: no se pudo fijar el reloj: {exc}")
                    try:
                        device.soc.power_measurement_enabled = True
                        row["power_measurement_enabled"] = True
                    except Exception as exc:
                        row["power_measurement_enabled"] = False
                        row["aviso_potencia"] = f"{type(exc).__name__}: {exc}"
                        print(f"AVISO {clock}/{mode}: no se pudo activar la medida de potencia: {exc}")
                    model.map(device, hw_only=True, mode=getattr(MapMode, mode))
                    for s in X[: args.warmup]:
                        model.predict(s.reshape(1, -1))
                    logits = []
                    for i, s in enumerate(X):
                        t0 = time.perf_counter()
                        out = model.predict(s.reshape(1, -1))
                        lat_all[k, rep, i] = (time.perf_counter() - t0) * 1000.0
                        logits.append(np.asarray(out, dtype=np.float32).reshape(-1))
                    pred = np.stack(logits).argmax(1)
                    lat = lat_all[k, rep]
                    row.update(metrics(y, pred))
                    row["coincidencia_vs_simulador"] = float(np.mean(pred == sim_pred))
                    row.update({"latency_mean_ms": float(np.mean(lat)), "latency_std_ms": float(np.std(lat)),
                                "latency_median_ms": float(np.median(lat)),
                                "latency_p95_ms": float(np.percentile(lat, 95)),
                                "latency_p99_ms": float(np.percentile(lat, 99))})
                    model.predict(X)                   # pasada en lote para la potencia
                    p = power_summary(model)
                    row["power_mw_mean"] = p["power_mw_mean"]
                    row["power_mw_floor"] = p["power_mw_floor"]
                    row["power_raw"] = json.dumps(p["power_raw"])
                    row["energy_uJ"] = (p["power_mw_mean"] * row["latency_mean_ms"]
                                        if is_number(p["power_mw_mean"]) else None)
                    row["throughput_ips"] = 1000.0 / row["latency_mean_ms"]
                    row["status"] = "completed"
                except Exception as exc:
                    row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc()[-2000:]})
                row["seconds"] = round(time.perf_counter() - t_rep, 2)
                rows.append(row)
                print(f"{clock:11s} {mode:7s} rep {rep}: {row['status']}  "
                      f"acc={row.get('accuracy', float('nan')):.4f}  "
                      f"lat={row.get('latency_mean_ms', float('nan')):.4f} ms  "
                      f"P={row.get('power_mw_mean')} mW")
                write_csv(out_dir / "hardware_repeticiones.csv", rows)
    if original_clock is not None:
        try:
            device.soc.clock_mode = original_clock
        except Exception:
            pass

    summary = []
    for clock in CLOCKS:
        for mode in MAPS:
            done = [r for r in rows if r["clock_mode"] == clock and r["map_mode"] == mode
                    and r["status"] == "completed"]
            s = {"clock_mode": clock, "map_mode": mode,
                 "status": "completed" if len(done) == args.repeats else ("partial" if done else "failed"),
                 "repeticiones": len(done)}
            for key in ("accuracy", "f1_macro", "coincidencia_vs_simulador", "latency_mean_ms",
                        "latency_p95_ms", "power_mw_mean", "power_mw_floor", "energy_uJ", "throughput_ips"):
                vals = [float(r[key]) for r in done if is_number(r.get(key))]
                s[f"{key}_media"] = float(np.mean(vals)) if vals else None
                s[f"{key}_std"] = float(np.std(vals)) if vals else None
            summary.append(s)
    write_csv(out_dir / "hardware_9modos.csv", summary)
    np.save(out_dir / "latencias_ms.npy", lat_all)
    ok = [s for s in summary if s["status"] == "completed" and is_number(s["energy_uJ_media"])]
    info["status"] = "completed" if ok else "failed"
    info["menor_energia"] = min(ok, key=lambda s: s["energy_uJ_media"]) if ok else None
    info["cpu_freq_after"] = read_cpu("scaling_cur_freq")
    (out_dir / "hardware.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {out_dir}")
    if info["menor_energia"]:
        b = info["menor_energia"]
        print(f"Menor energia: {b['clock_mode']} / {b['map_mode']}  {b['energy_uJ_media']:.1f} uJ  "
              f"({b['latency_mean_ms_media']:.4f} ms, {b['power_mw_mean_media']} mW)")
    else:
        print("Ninguna combinacion se completo: ver 'error' en hardware_repeticiones.csv")
    return 0 if info["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
