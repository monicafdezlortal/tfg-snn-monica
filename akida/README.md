# Medida en la placa BrainChip AKD1000 (experimento 05e)

Los experimentos 05a-05d (Keras → entrada de 4 bits → QuantizeML + QAT → Akida v1 en el simulador) están en `scripts/` y sus resultados en `results/05*/` (ver el README principal). Esta carpeta explica cómo lanzar el último paso, `scripts/05e_akida_hardware.py`, en el laboratorio, desde la Raspberry Pi 5 con la placa AKD1000.

La medida es la de Álvaro Poveda ([akida-model-deployment-pipeline](https://github.com/AlvaroPRuiiz/akida-model-deployment-pipeline)):

- Combinaciones: 3 ClockMode (Performance, Economy, LowPower) × 3 MapMode (AllNps, HwPr, Minimal).
- Por combinación: 5 repeticiones, con el governor de la CPU en `performance`.
- En cada repetición:
  1. Cargar el `.fbz`, fijar el reloj y activar la medida de potencia.
  2. Mapear con `hw_only=True`.
  3. Pasar 5 muestras de calentamiento.
  4. Medir la latencia de `predict` muestra a muestra con las 30 546 ventanas de test.
  5. Hacer una pasada en lote para la potencia.
- Energía estimada (µJ) = potencia media (mW) × latencia media (ms).

## 1. Preparar la Raspberry Pi

1. Pon en marcha la placa siguiendo el manual de [akd1000-bringup](https://github.com/AlvaroPRuiiz/akd1000-bringup): driver, `akida devices` y `python examples/01_hardware_smoke.py`.
2. Para 05e basta el entorno de akd1000-bringup (`akida==2.19.1`, `numpy`): no hace falta TensorFlow, torch ni este paquete.
3. Clona este repositorio en la Raspberry Pi (solo hace falta `scripts/05e_akida_hardware.py`).

## 2. Copiar el modelo y los datos de test

Los `.fbz` y los `.npy` **no están en git**. Los `.npy` son ventanas de SEED, que no se puede redistribuir, y el repositorio es público. Desde el PC, después de ejecutar 05d, copia la carpeta del modelo que quieras medir:

```bash
scp -r results/05d_akida_sim/4ch_cuantiles_seed0 pi@<ip>:~/tfg-snn-monica/results/05d_akida_sim/
```

Cada carpeta contiene:

| Fichero | Contenido |
|---|---|
| `model.fbz` | El modelo Akida v1 (2.6 KB con 4 canales, 13.4 KB con 62) |
| `test_inputs_uint8.npy` | Entradas de test (sujetos 1-3), ya cuantizadas a 0..15 |
| `test_labels.npy` | Etiquetas de test |
| `sim_logits.npy` | Salida del simulador, para comprobar que la placa da lo mismo |

## 3. Medir

```bash
cd ~/tfg-snn-monica
# prueba rápida (1000 muestras, 1 repetición)
sudo -E ~/akd1000-bringup/.venv/bin/python scripts/05e_akida_hardware.py \
    --run-dir results/05d_akida_sim/4ch_cuantiles_seed0 --max-samples 1000 --repeats 1
# medida completa (9 modos × 5 repeticiones × 30 546 muestras)
sudo -E ~/akd1000-bringup/.venv/bin/python scripts/05e_akida_hardware.py \
    --run-dir results/05d_akida_sim/4ch_cuantiles_seed0
```

- **`sudo`** hace falta para poner el governor en `performance` (se escribe en `/sys/devices/system/cpu/*/cpufreq/scaling_governor`). Sin permisos, el script avisa y sigue, y lo deja anotado en `hardware.json`.
- **Duración.** Con ~0.6 ms por inferencia (lo que midió Álvaro con modelos de este tamaño), 30 546 muestras × 45 repeticiones son unos 15-20 minutos por modelo.

Resultados en `results/05e_akida_hardware/<carpeta de 05d>/`:

- `hardware_9modos.csv`: una fila por combinación, con media y desviación de accuracy, F1, coincidencia con el simulador, latencia, potencia y energía.
- `hardware_repeticiones.csv`: una fila por repetición, con los avisos o errores si los hay.
- `hardware.json`: entorno, governor antes y después, reloj y temperatura de la Raspberry, y el modo de menor energía.
- `latencias_ms.npy`: latencia de cada muestra (no se sube a git).

Copia de vuelta al PC los `.csv` y el `.json` para añadirlos a `results/` y al README.

## Notas

- La latencia es la observada desde el host alrededor de `model.predict`, no la interna del chip.
- La potencia es la que da la API de Akida, no la del sistema completo.
- En el PC (sin placa) el script termina con `not_detected`. Se ha probado con el AKD1000 virtual del SDK y llega hasta la ejecución, que el SDK solo permite en un chip real.
