# Rutas históricas (evolución del trabajo)

Scripts y notebooks de las primeras fases del TFG, **tal cual estaban**.
No se usan para los resultados finales, pero documentan cómo se llegó al
protocolo actual (split cross-subject y normalización z-score por sujeto) y
se citan en la memoria como evolución del trabajo.

Se ejecutan desde la raíz del repositorio (usan rutas relativas a `data/`).

| Fichero | Qué hacía | Por qué se abandonó |
|---|---|---|
| `Prueba1SEED.ipynb` | Primera ANN sobre SEED: `train_test_split` **aleatorio por ventana**. | Ventanas contiguas del mismo trial (casi idénticas tras el suavizado LDS) caen en train y test a la vez, así que la accuracy sale inflada. |
| `trainsnn.py` | Primera SNN: 62 canales × 5 bandas = 310 features, split **aleatorio por trial** (mezcla sujetos en train y test), normalización MinMax ajustada sobre train. | Mezcla sujetos en train y test, así que no mide la generalización a sujetos nuevos. Necesita `seaborn`. La ruta a los datos es absoluta. |
| `build_dataset.py` | Generaba `data/seed_snn_dataset.npz` para la SNN: 4 canales, normalización min-max **global** (sobre todos los sujetos) y spikes precalculados con T=50. | La normalización global usa estadísticas de los sujetos de test. Ahora los datos se leen directamente de los `.mat` (`src/eeg_emotion/data.py`) y se normalizan por sujeto. |
| `train_snn_random_split.py` | SNN con test cross-subject (sujetos 1-3) pero validación con `random_split` sobre las ventanas de los sujetos 4-15. Recuperado del commit `014be81`. | val_acc ≈ 100 % por la fuga de datos entre ventanas contiguas, así que la validación no servía para elegir el modelo. Se sustituyó por sujetos de validación enteros (4-5). |
| `diagnostico_snn.py` | Medía la tasa de spikes por capa para detectar neuronas muertas (la loss se quedaba en ln 3). | Sirvió para introducir BatchNorm antes de cada LIF, bajar `v_threshold` a 0.5 y quitar la LIF de la capa de salida. **Ya no funciona**: importa `train_snn` y busca `lif3`, que ya no existen. |

Los resultados y pesos de estas rutas están en `results/legacy/`:
`baseline.csv` (tabla manual de las primeras pruebas, incluida la SNN con
MNIST), `ann_log.t` (log de la primera ANN cross-subject, en UTF-16) y los
`.pth`, que no se suben a git.
