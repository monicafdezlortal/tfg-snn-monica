# TFG clasificación de emociones con EEG con SNN

**¿Se puede reconocer la emoción (negativa / neutra / positiva) a partir del EEG con solo 4 canales temporales (FT7, FT8, T7, T8) y una precisión cercana a la de los 62 canales, en sujetos no vistos durante el entrenamiento?**
**¿Se mantiene esa precisión con una red de spikes (SNN) y al desplegar el modelo en hardware neuromórfico (BrainChip AKD1000)?**

Trabajo Fin de Grado de Mónica Fernández. Dataset: SEED (features `de_LDS`, ventanas de 1 s).

---

## 1. Estado actual

| Exp. | Descripción | Estado |
|---|---|---|
| 01 | Réplica del protocolo de Zheng & Lu 2015 (por sesión, trials 1-9 train / 10-15 test), ANN + SVM lineal | Hecho (5 semillas, 4 y 62 canales, 2 sesiones por sujeto) |
| 02 | ANN cross-subject (test 1-3, val 4-5, train 6-15), z-score por sujeto | Hecho (5 semillas, 4 y 62 canales) |
| 03 | SNN cross-subject con el mismo split | Hecho (4 canales) · 62 canales pendiente (a decidir) |
| 04 | Búsqueda de hiperparámetros (ANN, SNN) y de features de entrada (ANN) | Hecho |
| 05a-05d | ANN para Akida: Keras float → entrada de 4 bits → QuantizeML + QAT → Akida v1 (`.fbz`) en el simulador | Hecho (5 semillas, 4 y 62 canales) |
| 05e | Latencia, potencia y energía en la placa AKD1000 (9 modos de reloj y mapeo) | Script preparado; pendiente de ejecutar en el laboratorio |

## 2. Estructura del repositorio

```
tfg-snn-monica/
├── README.md
├── requirements.txt         # versiones con las que se obtuvieron los resultados (01-04)
├── requirements-akida.txt   # entorno .venv-akida del experimento 05 (Python 3.11)
├── pyproject.toml           # permite "pip install -e ." del paquete eeg_emotion
├── data/                    # SEED (no se sube a git); data/cache/: features en .npz para 05
├── src/eeg_emotion/
│   ├── config.py            # rutas, índices de canales, sujetos de cada split
│   ├── data.py              # carga de SEED, canales, normalización, splits (sin torch)
│   ├── spikes.py            # rate coding de la SNN (SpikeDataset, spike_collate)
│   ├── models.py            # EmotionANN, FlexANN, EmotionSNN, FlexSNN
│   ├── training.py          # bucle de entrenamiento, evaluate, semillas
│   ├── results.py           # métricas y guardado de resultados (sin torch)
│   └── akida_tools.py       # experimento 05: red Keras, entrada de 4 bits, QAT, Akida
├── scripts/                 # un script por experimento
│   ├── 01_zheng_lu_ann_svm.py
│   ├── 02_ann_cross_subject.py
│   ├── 03_snn_cross_subject.py
│   ├── 04a_ann_hparam_search.py
│   ├── 04b_snn_hparam_search.py
│   ├── 04c_ann_feature_experiments.py
│   ├── 05a_akida_float.py ... 05e_akida_hardware.py   # despliegue en Akida (ver 5)
│   ├── 05b_select_input_quant.py   # elige la entrada de 4 bits por val_acc
│   ├── run_05_seeds.sh      # lanza una etapa de 05 con las 5 semillas en paralelo
│   └── aggregate_seeds.py   # media ± desviación de varias semillas -> results/summary.csv
├── legacy/                  # rutas históricas tal cual estaban (ver legacy/README.md)
├── akida/                   # cómo medir en la placa AKD1000 (experimento 05e)
├── results/
│   ├── <experimento>/<canales>_seed<N>/   # config.json, metrics.json,
│   │                                      # confusion_matrix.csv, model.pth
│   │                                      # (05: model.keras / model.fbz)
│   ├── summary.csv          # tabla agregada (scripts/aggregate_seeds.py)
│   └── legacy/              # resultados de las rutas de legacy/
├── notebooks/               # exploración (ver notebooks/README.md)
└── docs/figures/            # figuras para la memoria
```

Cada script acepta `--channels 4|62` y `--seed N`, y guarda en su carpeta de `results/`:

- `config.json`: hiperparámetros, split y semilla.
- `metrics.json`: accuracy, F1 macro, F1 por clase, matriz de confusión e historial por epoch.
- `confusion_matrix.csv` y `model.pth` (los `.pth` no se suben a git).

El experimento 01 entrena un modelo por sesión, así que guarda en su lugar `sessions.csv` (accuracy, F1 y C del SVM de cada sesión) y una matriz de confusión acumulada por modelo (`confusion_matrix_EmotionANN.csv`, `confusion_matrix_LinearSVC.csv`). Con `--sessions 3` la carpeta es `<canales>_3ses_seed<N>`.

## 3. Instalación

Probado con Python 3.14 en Windows 11, solo con CPU.

```bash
git clone https://github.com/monicafdezlortal/tfg-snn-monica.git
cd tfg-snn-monica
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .                  # instala src/eeg_emotion en modo editable
```

Los comandos se ejecutan desde la raíz del repositorio.

**Entorno del experimento 05 (Akida).** El SDK de BrainChip necesita Python 3.11, así que va en un entorno aparte, `.venv-akida`, sin torch:

```bash
py -3.11 -m venv .venv-akida      # Linux: python3.11 -m venv .venv-akida
.venv-akida\Scripts\python -m pip install -r requirements-akida.txt
.venv-akida\Scripts\python -m pip install -e . --no-deps
```

Las versiones del SDK son las de [akida-model-deployment-pipeline](https://github.com/AlvaroPRuiiz/akida-model-deployment-pipeline): akida y cnn2snn 2.19.1, quantizeml 1.2.3, tensorflow 2.19.1 y tf_keras 2.19.0. La única diferencia es numpy: akd1000-bringup fija 2.4.6, pero tensorflow 2.19.1 exige numpy < 2.2, así que se usa 2.1.3. Con este entorno, la prueba de humo del simulador de akd1000-bringup (`examples/00_simulator_smoke.py`) da `[5, 5]`.

## 4. Datos

1. Solicita SEED al BCMI Lab (SJTU) en <https://bcmi.sjtu.edu.cn/home/seed/>. Hay que firmar un acuerdo de licencia, así que **el dataset no se distribuye con este repositorio**.
2. Del paquete descargado solo hace falta `SEED_EEG/ExtractedFeatures_1s/` (unos 2.7 GB). Contiene 45 ficheros `<sujeto>_<fecha>.mat` (15 sujetos × 3 sesiones) y `label.mat`.
3. Cópiala como `data/ExtractedFeatures_1s/`:

```
data/ExtractedFeatures_1s/
├── 1_20131027.mat  ...  15_20131105.mat
└── label.mat
```

Cada `.mat` tiene, para cada trial t = 1..15, la clave `de_LDS<t>` con forma `(62 canales, n ventanas, 5 bandas)`. El resto del paquete SEED (EEG en bruto, `Preprocessed_EEG`, `SEED_Multimodal`…) solo lo usan los scripts de exploración de `notebooks/`.

## 5. Cómo reproducir cada experimento

| Exp. | Comando | Memoria |
|---|---|---|
| 01 | `python scripts/01_zheng_lu_ann_svm.py --channels 4 --seed 0` (`--sessions 3` para usar las 3 sesiones) | 4.1 Réplica del protocolo de Zheng & Lu |
| 02 | `python scripts/02_ann_cross_subject.py --channels 4 --seed 0` | 4.2 Evaluación cross-subject; 4.4 Comparación 4 vs 62 canales |
| 03 | `python scripts/03_snn_cross_subject.py --channels 4 --seed 0` | 4.3 ANN frente a SNN |
| 04a | `python scripts/04a_ann_hparam_search.py` | 3.x Selección de hiperparámetros (provisional) |
| 04b | `python scripts/04b_snn_hparam_search.py` | 3.x Selección de hiperparámetros (provisional) |
| 04c | `python scripts/04c_ann_feature_experiments.py` | 3.x Selección de hiperparámetros (provisional) |
| 05a | `.venv-akida\Scripts\python scripts/05a_akida_float.py --channels 4 --seed 0` | 4.5 Despliegue en Akida |
| 05b | `... scripts/05b_akida_int_input.py --channels 4 --seed 0 --input-quant uniforme3` (y `cuantiles`); después `scripts/05b_select_input_quant.py` | 4.5 |
| 05c | `... scripts/05c_akida_qat.py --channels 4 --seed 0` (usa la entrada elegida en 05b) | 4.5 |
| 05d | `... scripts/05d_akida_sim.py --channels 4 --seed 0` | 4.5 |
| 05e | En la Raspberry Pi con la placa: `sudo -E .venv/bin/python scripts/05e_akida_hardware.py --run-dir results/05d_akida_sim/4ch_cuantiles_seed0` (ver [akida/README.md](akida/README.md)) | 4.6 Medidas en AKD1000 |
| Tabla | `python scripts/aggregate_seeds.py` | Tablas de resultados de 4.1-4.5 |

Para varias semillas y los dos conjuntos de canales (01, 02 y 03):

```bash
for ch in 4 62; do for s in 0 1 2 3 4; do
  python scripts/02_ann_cross_subject.py --channels $ch --seed $s
done; done
python scripts/aggregate_seeds.py
```

Para el experimento 05, cada etapa con las 5 semillas y los dos conjuntos de canales (en Git Bash, una etapa detrás de otra porque cada una parte de la anterior):

```bash
bash scripts/run_05_seeds.sh 05a_akida_float
bash scripts/run_05_seeds.sh 05b_akida_int_input --input-quant uniforme3
bash scripts/run_05_seeds.sh 05b_akida_int_input --input-quant cuantiles
.venv-akida/Scripts/python scripts/05b_select_input_quant.py
bash scripts/run_05_seeds.sh 05c_akida_qat
bash scripts/run_05_seeds.sh 05d_akida_sim
python scripts/aggregate_seeds.py
```

`run_05_seeds.sh` usa 1 hilo de TensorFlow por proceso. Con 62 canales cada proceso necesita ~1 GB de RAM: con poca memoria libre, `CHANNELS="62" MAXJOBS=2 bash scripts/run_05_seeds.sh ...`.

- **Pruebas rápidas.** `--epochs 2` en 01/02/03, 05a, 05b y 05c, y `--trials 1 --max-epochs 1` en las búsquedas.
- **Reanudar.** Las búsquedas (04a/04b/04c) se reanudan solas desde su CSV si se interrumpen.
- **Duración.** El experimento 01 tarda unos 5 min por semilla (4 canales) y unos 6 min (62 canales). La SNN tarda unos 2 min por epoch con 4 canales en CPU, así que 50 epochs llevan más de 1.5 h por semilla. Las 5 semillas de la tabla se ejecutaron en paralelo con `OMP_NUM_THREADS=3`. El número de hilos queda en `config.json` porque en CPU puede afectar a los últimos decimales. Conviene **desactivar la suspensión del equipo** durante las ejecuciones largas: si se suspende, la columna `seconds` de los CSV incluye el tiempo suspendido.

## 6. Resultados actuales

Media ± desviación típica (poblacional) entre semillas. Lo genera `scripts/aggregate_seeds.py` en `results/summary.csv`.

**Réplica de Zheng & Lu (experimento 01, 5 semillas: 0-4)**

Protocolo *subject-dependent* del paper: 15 sujetos × 2 sesiones = 30 experimentos; en cada uno, trials 1-9 para entrenar y 10-15 para test. La accuracy de cada semilla es la media de las 30 sesiones. La última columna es la desviación entre sesiones (media de las 5 semillas), que es la que reporta el paper.

| Modelo | Canales | Accuracy | F1 macro | Accuracy (%) ± std entre sesiones | Paper (Zheng & Lu 2015) |
|---|---|---|---|---|---|
| ANN (`EmotionANN`, 50 epochs) | 4 | 0.7824 ± 0.0043 | 0.7633 ± 0.0048 | 78.24 ± 13.20 | 82.88 ± 10.92 |
| SVM lineal | 4 | 0.7667 ± 0.0000 | 0.7455 ± 0.0000 | 76.67 ± 14.98 | 82.88 ± 10.92 |
| ANN (`EmotionANN`, 50 epochs) | 62 | 0.7829 ± 0.0045 | 0.7597 ± 0.0063 | 78.29 ± 11.83 | DBN 86.08 ± 8.34 |
| SVM lineal | 62 | 0.7911 ± 0.0000 | 0.7707 ± 0.0000 | 79.11 ± 12.80 | SVM 83.99 ± 9.72 |

- La SVM no depende de la semilla (desviación 0 entre semillas): solo varía la ANN.
- Todos quedan por debajo del paper, con una dispersión entre sesiones parecida: con 4 canales, la ANN 4.6 puntos y la SVM 6.2; con 62 canales, la SVM 4.9 puntos por debajo de la SVM del paper (y la ANN 7.8 por debajo de su DBN).
- En este protocolo, 4 canales rinden casi igual que 62 (ANN 78.2 frente a 78.3; SVM 76.7 frente a 79.1), en línea con la conclusión del paper.

**Modelos base cross-subject (experimentos 02 y 03, 5 semillas: 0-4)**

Test con los sujetos 1-3 (30 546 ventanas), no vistos en el entrenamiento.

| Modelo | Exp. | Canales | Accuracy | F1 macro |
|---|---|---|---|---|
| ANN (`EmotionANN`, 20/310 → 128 → 64 → 3) | 02 | 4 | 0.6236 ± 0.0072 | 0.6223 ± 0.0065 |
| ANN (`EmotionANN`) | 02 | 62 | 0.6508 ± 0.0081 | 0.6514 ± 0.0077 |
| SNN (`EmotionSNN`, LIF, T = 50) | 03 | 4 | 0.6026 ± 0.0041 | 0.6018 ± 0.0043 |
| SNN (`EmotionSNN`) | 03 | 62 | pendiente (a decidir) | |

**Búsquedas (experimento 04, ejecutadas antes de la reorganización)**

Configuración elegida por validación (nunca por test) y evaluada en test una sola vez. En 04a y 04c es la media de 3 reentrenos (semillas 0-2); en 04b es una sola ejecución. Los resultados completos están en `results/04*/`.

| Búsqueda | Configuración elegida | Canales | Test accuracy |
|---|---|---|---|
| 04a ANN, random search (41 configuraciones, CV de 4 folds) | 1 capa oculta de 32, dropout 0.35, lr 3.3e-4 | 62 | 0.6984 ± 0.0132 |
| 04a: configuración de 02 reentrenada con el protocolo de 04a | 128 → 64, dropout 0.3, lr 1e-3 | 4 | 0.6424 ± 0.0127 |
| 04b SNN, random search (15 configuraciones, val sujetos 4-5) | 2 capas de 242, umbral 1.47, τ 2.05, lr 1.5e-3 | 4 | 0.6146 |
| 04c ANN, referencia de_LDS (red fija 256 → 128) | de_LDS, sin contexto ni regularización | 62 | 0.6879 ± 0.0099 |
| 04c ANN, greedy de features, contexto y regularización | de + dasm + rasm, ruido gaussiano σ = 0.5 | 62 | 0.6748 ± 0.0029 (ensemble de 5: 0.6927) |

Al pasar del protocolo por sesión (01, ~78 %) al cross-subject (02, 62-65 %), la ANN pierde unos 15 puntos: generalizar a personas nuevas es mucho más difícil que a trials nuevos de la misma persona. Con 4 canales, la SNN pierde unos 2.1 puntos respecto a la ANN equivalente (0.603 frente a 0.624). Con la misma red, pasar de 4 a 62 canales mejora la ANN en unos 2.7 puntos. En las búsquedas, la mejor ANN de 62 canales llega a ~0.70. En 04c, añadir features de asimetría mejora la validación pero no el test.

**Despliegue en Akida (experimento 05, 5 semillas: 0-4)**

Mismo split cross-subject que 02 (test sujetos 1-3). Cada etapa parte de la anterior con la misma semilla.

> **05a no es la red de 02 pasada a Keras.** Cambia la arquitectura: 1 capa oculta de 32 (02: 128 → 64), sin bias, sin BatchNorm, ReLU acotada a 15, dropout 0.35 (02: 0.3), lr 3.3e-4 y batch 512 (02: 1e-3 y 256), sin weight decay ni ReduceLROnPlateau. Es la mejor configuración de 04a adaptada a lo que admite Akida v1. La diferencia entre 02 y 05a, por tanto, no se debe solo a pasar de PyTorch a Keras.

| Etapa | Qué cambia | Canales | Entrada | Accuracy | F1 macro | Coincide con la etapa anterior |
|---|---|---|---|---|---|---|
| 02 (referencia) | PyTorch, `EmotionANN` 128 → 64, con BN y bias | 4 | float | 0.6236 ± 0.0072 | 0.6223 ± 0.0065 | |
| 05a | Keras, Dense(32) sin bias, sin BN | 4 | float (z-score) | 0.6228 ± 0.0103 | 0.6214 ± 0.0106 | |
| 05b | Entrada cuantizada a 0..15 | 4 | cuantiles (elegida por val) | 0.6079 ± 0.0111 | 0.5990 ± 0.0130 | |
| 05b (variante) | | 4 | ±3σ uniforme | 0.6123 ± 0.0186 | 0.6029 ± 0.0176 | |
| 05c | QuantizeML 4 bits + QAT | 4 | cuantiles | 0.6041 ± 0.0134 | 0.5973 ± 0.0141 | 89.8 % con 05b |
| 05d | Akida v1 (`.fbz`) en el simulador | 4 | cuantiles | **0.6037 ± 0.0135** | 0.5967 ± 0.0144 | 99.1 % con 05c |
| 02 (referencia) | PyTorch, `EmotionANN` 128 → 64, con BN y bias | 62 | float | 0.6508 ± 0.0081 | 0.6514 ± 0.0077 | |
| 05a | Keras, Dense(32) sin bias, sin BN | 62 | float (z-score) | 0.6900 ± 0.0149 | 0.6899 ± 0.0149 | |
| 05b | Entrada cuantizada a 0..15 | 62 | ±3σ uniforme (elegida por val) | 0.6820 ± 0.0215 | 0.6807 ± 0.0222 | |
| 05b (variante) | | 62 | cuantiles | 0.6758 ± 0.0087 | 0.6751 ± 0.0090 | |
| 05c | QuantizeML 4 bits + QAT | 62 | ±3σ uniforme | 0.6736 ± 0.0220 | 0.6724 ± 0.0224 | 92.6 % con 05b |
| 05d | Akida v1 (`.fbz`) en el simulador | 62 | ±3σ uniforme | **0.6736 ± 0.0220** | 0.6724 ± 0.0224 | 99.7 % con 05c |

- **De la red float al simulador de Akida se pierden 1.9 puntos con 4 canales (0.623 → 0.604) y 1.6 con 62 (0.690 → 0.674).**
  - Con 4 canales, casi todo (1.5 puntos) se pierde al cuantizar la entrada a 4 bits (05b).
  - Con 62, la pérdida se reparte entre la entrada (0.8) y la QAT de pesos y activaciones (0.8).
  - La conversión a Akida no cuesta prácticamente nada.
- **El simulador coincide con el modelo QAT en el 99.1 % (4 canales) y el 99.7 % (62) de las predicciones.** Álvaro obtuvo un 94.6 % con su modelo 8-957-180.
- **Recorte de la entrada a ±3σ.** Afecta al 0.26/0.63/0.73 % de los valores (train/val/test) con 4 canales y al 0.38/0.55/0.64 % con 62. Con ±3σ, los niveles extremos casi no se usan (unos 3.4 bits efectivos de 4). La variante por cuantiles usa los 16 niveles por igual. La validación elige cuantiles con 4 canales y ±3σ con 62. En test las dos variantes quedan a menos de 1 punto: con 4 canales la no elegida (±3σ) sale 0.4 puntos mejor, y con 62 la elegida es también la mejor.
- **El modelo Akida es `InputData → FullyConnected → FullyConnected`** (el dropout desaparece al convertir). El `.fbz` ocupa 2.6 KB con 4 canales y 13.4 KB con 62, y cabe entero en un AKD1000 virtual.
- **Ninguna etapa pierde más de 5 puntos respecto a la anterior**, así que no ha hecho falta el plan B (añadir una entrada constante como bias, opción `--const-input` de 05b).

**4 frente a 62 canales en cada etapa**

Media de test de 5 semillas; en 05b-05d, con la entrada elegida por validación (cuantiles con 4 canales, ±3σ con 62).

| Etapa | Accuracy 4 canales | Accuracy 62 canales | 62 − 4, accuracy (puntos) | 62 − 4, F1 macro (puntos) |
|---|---|---|---|---|
| 02 (referencia, otra red) | 0.6236 | 0.6508 | +2.7 | +2.9 |
| 05a float | 0.6228 | 0.6900 | +6.7 | +6.9 |
| 05b entrada 0..15 | 0.6079 | 0.6820 | +7.4 | +8.2 |
| 05c QAT 4 bits | 0.6041 | 0.6736 | +7.0 | +7.5 |
| 05d simulador Akida | 0.6037 | 0.6736 | +7.0 | +7.6 |

| Tamaño del modelo (igual en las 5 semillas) | 4 canales | 62 canales |
|---|---|---|
| Features de entrada | 20 | 310 |
| Pesos (4 bits) | 736 | 10 016 |
| `.fbz` | 2 612 B (2.6 KB) | 13 439 B (13.4 KB) |
| Programa al mapear en el AKD1000 virtual | 2 376 B | 9 288 B |

- **La diferencia sube de 2.7 puntos en 02 a ~7 en 05 por la red, no por Akida.**
  - La red de 05 es la mejor de 04a con 62 canales. Con 62 canales mejora a 02 en 3.9 puntos; con 4 canales queda igual (−0.1).
  - Después, cuantizar y convertir apenas mueve la diferencia: se queda entre 7.0 y 7.4 puntos.
- **Con 62 canales el modelo es 5 veces más grande** (13.6 veces más pesos), pero los dos caben con mucha holgura en el chip.
- **Prueba con la mejor red de 4 canales de 04a no mejora.**
  - Red de la prueba: trial 33, 32 → 16, dropout 0.138, lr 1.14e-4.
  - 05a con 4 canales y esa red da 0.6182 ± 0.0066 (val 0.728), frente a 0.6228 ± 0.0103 (val 0.736) con la red de 32. Es 0.5 puntos peor y no reduce la diferencia con 62 canales.
  - Por eso 05b-05d siguen con la red de 32. Para repetirla: `05a_akida_float.py --config 04a_4ch`; los resultados están en `results/05a_akida_float_cfg4ch/`.

## 7. Decisiones metodológicas clave

- **Evaluación cross-subject (sujetos de test 1-3 no vistos; validación con sujetos 4-5).**
  - Con un split aleatorio por ventana, las ventanas contiguas de un mismo trial, casi idénticas tras el suavizado LDS, caen a la vez en train y en test o validación.
  - El resultado es una accuracy inflada. Validando con `random_split` se llegó a val_acc ≈ 100 % (ver `legacy/`).
  - Separar sujetos enteros mide la generalización a personas nuevas, que es el caso de uso real.
  - El test solo se usa una vez al final. Los hiperparámetros se eligen con validación por sujetos: los sujetos 4-5 en 02/03 y una validación cruzada de 4 folds sobre los sujetos 4-15 en 04.
- **Normalización z-score por sujeto.**
  - La amplitud del EEG varía mucho entre personas. Por eso cada sujeto se normaliza con su propia media y desviación por feature, sin usar etiquetas.
  - Así no se mezclan las estadísticas de los sujetos de test con las de train, cosa que sí pasaba con la normalización min-max global del antiguo `build_dataset.py`.
  - Para la SNN, el z-score se recorta a ±4σ y se mapea a [0, 1], que es la probabilidad de spike del rate coding.
- **4 canales: FT7, FT8, T7 y T8.**
  - Zheng & Lu (2015) identificaron las regiones temporales laterales y las bandas beta/gamma como las más discriminativas.
  - Mostraron que un perfil de solo 4 canales (FT7, FT8, T7, T8) consigue una precisión comparable a la de 62 canales.
  - Pocos canales hacen viable un dispositivo portátil y un modelo pequeño para hardware neuromórfico. Por eso se compara sistemáticamente 4 frente a 62 canales.
- **Features `de_LDS`.**
  - Entropía diferencial por banda (Duan et al., 2013), suavizada con un sistema dinámico lineal (Shi & Lu, 2010).
  - Es la feature con la que Zheng & Lu obtuvieron sus mejores resultados. En 04c se comprobó que psd, de_movingAve y las asimetrías no la mejoran de forma clara en test.
- **SNN: diseño para que aprenda.**
  - BatchNorm antes de cada neurona LIF y umbral 0.5 (con el umbral 1.0 por defecto las neuronas no disparaban y la loss se quedaba en ln 3).
  - Capa de salida sin LIF, para tener logits continuos para CrossEntropy.
  - Los spikes se regeneran en cada epoch, lo que actúa como data augmentation.
- **Pesos de clase** inversos a la frecuencia en train, en todos los modelos salvo en la réplica de Zheng & Lu (01), que sigue el script original sin pesos.
- **Réplica de Zheng & Lu (01).**
  - Sirve para comprobar que la ANN y el pipeline de datos dan resultados del orden del paper antes de pasar al protocolo cross-subject, más exigente.
  - Se usan 2 sesiones por sujeto, como en el paper (el SEED público trae 3; `--sessions 3` da 45 experimentos).
  - Sin validación: la ANN entrena 50 epochs fijas, y el test no se usa para elegir epoch. La normalización usa solo las estadísticas del train de cada sesión.
  - El C de la SVM lineal se elige en cada sesión, entre 2⁻¹⁶ y 2¹⁰ (potencias pares de 2), con validación cruzada `GroupKFold` de 3 folds agrupada por trial dentro del train, para que ventanas de un mismo trial no caigan a la vez en train y validación.
- **Despliegue en Akida (05).**
  - **Receta de Álvaro Poveda** (`run_small_matched_recipe.py`): Dense sin bias → ReLU(max 15) → Dense sin bias; entrada entera 0..15 (int8 en Keras y QuantizeML, uint8 en Akida).
    - Cuantización: `QuantizationParams(input_dtype="int8", input_weight_bits=4, weight_bits=4, activation_bits=4, output_bits=8, per_tensor_activations=False)`.
    - PTQ con 1024 muestras. QAT con Adam lr 1e-4, batch 128, hasta 30 epochs y paciencia 6.
    - Conversión con `cnn2snn.convert` dentro de `set_akida_version(AkidaVersion.v1)`.
  - **Tamaño de la red** (32 neuronas ocultas, dropout 0.35, lr y batch): la mejor configuración de 04a. QuantizeML y CNN2SNN aceptan la capa Dropout (QuantizeML la envuelve y CNN2SNN la quita al convertir), así que se mantiene.
  - **Diferencias con Álvaro:**
    - la mejor epoch se elige por val_acc también en QAT, como en el resto del TFG (él usaba val_loss);
    - las 1024 muestras de calibración se eligen al azar del train, porque en SEED las primeras serían un único trial de un sujeto;
    - se usan pesos de clase, como en 02.
  - **La red float de 05b se entrena ya con la entrada entera** (como Álvaro), en vez de reutilizar los pesos de 05a.
  - **Cuantización de la entrada:** el z-score por sujeto se pasa a 16 niveles. Dos métodos: recortar a ±3σ en niveles de 0.375σ, o 15 umbrales por feature en los cuantiles del train. Se elige uno por cada número de canales por la val_acc media (`results/05b_akida_int_input/seleccion.json`), nunca por test.

**Referencias**

- W.-L. Zheng y B.-L. Lu, "Investigating Critical Frequency Bands and Channels for EEG-Based Emotion Recognition with Deep Neural Networks", *IEEE Trans. Autonomous Mental Development*, 7(3), 162-175, 2015.
- R.-N. Duan, J.-Y. Zhu y B.-L. Lu, "Differential entropy feature for EEG-based emotion classification", *IEEE/EMBS NER*, 2013.
- L.-C. Shi y B.-L. Lu, "Off-line and on-line vigilance estimation based on linear dynamical system and manifold learning", *IEEE EMBC*, 2010.
- W. Fang et al., "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence", *Science Advances*, 9(40), 2023.

## 8. Limitaciones

- **Solo 3 sujetos de test.** La accuracy depende mucho de qué sujetos tocan. Por eso se reportan varias semillas, aunque la semilla no captura la variabilidad entre sujetos (en la CV por sujetos de la ANN de referencia, los folds van de 0.70 a 0.75 de val_acc).
- **La normalización por sujeto usa los datos (sin etiquetas) de cada sujeto de test.** En uso real haría falta una sesión de calibración del usuario nuevo.
- **El suavizado LDS de las features de SEED no es causal**: usa todo el trial. Un sistema en tiempo real tendría que usar un filtro causal (p. ej. media móvil o Kalman sin suavizado).
- **Las etiquetas son por película** (15 trials por sesión) y se asignan a todas sus ventanas, aunque la emoción no sea constante a lo largo del clip.
- **Réplica de Zheng & Lu: entre 4.6 y 6.2 puntos por debajo del paper.** El paper no detalla todo el preprocesado ni cómo eligió los hiperparámetros.
  - Con la rejilla inicial de C (2⁻¹⁰ a 2¹⁰), el C elegido caía a menudo en el extremo inferior: en 10 de 30 sesiones con 4 canales y en 6 de 30 con 62. Por eso se amplió a 2⁻¹⁶ - 2¹⁰. Ahora solo 5 de 30 sesiones con 4 canales quedan en un extremo (2 en 2⁻¹⁶ y 3 en 2¹⁰) y ninguna con 62.
  - La ampliación no mejora la SVM: baja de 77.71 a 76.67 % con 4 canales y de 79.40 a 79.11 % con 62. La validación cruzada de 3 folds (3 trials por fold) es ruidosa y a veces elige un C muy pequeño que generaliza peor al test. La diferencia con el paper, por tanto, no se debe a la rejilla de C.
- **La SNN se entrena y evalúa en software** (SpikingJelly, CPU). No se han medido latencia ni energía en hardware.
- **La ANN de Akida (05) solo se ha ejecutado en el simulador.** Falta medir accuracy, latencia, potencia y energía en el AKD1000 (05e, preparado). La latencia de 05e será la observada desde el host alrededor de `predict`, y la potencia es la que da la API de Akida, no la de todo el sistema.
- **La cuantización de la entrada usa las estadísticas por sujeto** (z-score) del sujeto de test, igual que en 02. En el dispositivo haría falta una calibración previa del usuario.
- **El entorno de Akida usa numpy 2.1.3 en lugar del 2.4.6 de akd1000-bringup**, por la restricción de tensorflow 2.19.1. La prueba de humo del simulador pasa igualmente.

## 9. Relación con los repositorios de Álvaro Poveda

El despliegue en hardware (experimentos 05+) se apoya en dos repositorios de Álvaro Poveda, también de un TFG en la Universidad Politécnica de Madrid:

- **[akd1000-bringup](https://github.com/AlvaroPRuiiz/akd1000-bringup).**
  - Puesta en marcha de la placa BrainChip AKD1000 por PCIe: instalación, driver, entorno y pruebas de humo en simulador y en hardware (`examples/00_simulator_smoke.py`, `examples/01_hardware_smoke.py`).
  - Benchmark de modelos `.fbz`.
  - Es el paso previo para tener la placa operativa.
- **[akida-model-deployment-pipeline](https://github.com/AlvaroPRuiiz/akida-model-deployment-pipeline).**
  - Pipeline para llevar un clasificador Keras al AKD1000:
    1. Cuantización a 4 bits con QuantizeML (PTQ o QAT).
    2. Conversión explícita a Akida v1 con CNN2SNN y exportación a `.fbz`.
    3. Validación en software.
    4. Mapeo y medida en el chip.
  - Lo desarrolló para modelos de ángulo de llegada (AoA) con una entrada de 8 enteros.

En este TFG se sigue ese pipeline (experimento 05) con un MLP pequeño (20 entradas con 4 canales, 310 con 62). Las tres diferencias respecto al caso AoA se han resuelto así:

1. **Framework.** El modelo se ha reimplementado en Keras (`tf_keras`), con la arquitectura que admite Akida v1 (`eeg_emotion/akida_tools.py`).
2. **Entrada.** Las features continuas (z-score) se cuantizan a enteros 0..15 (05b).
3. **Entorno.** Hay un entorno aparte, `.venv-akida`, con Python 3.11 y sin torch (`requirements-akida.txt`). La carga de datos de `eeg_emotion` ya no importa torch.
