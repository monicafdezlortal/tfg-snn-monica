# tfg-eeg-emotion

**¿Se puede reconocer la emoción (negativa / neutra / positiva) a partir del EEG con solo 4 canales temporales (FT7, FT8, T7, T8) y una precisión cercana a la de los 62 canales, en sujetos no vistos durante el entrenamiento?**
**¿Se mantiene esa precisión con una red de spikes (SNN) y al desplegar el modelo en hardware neuromórfico (BrainChip AKD1000)?**

Trabajo Fin de Grado de Mónica Fernández. Dataset: SEED (features `de_LDS`, ventanas de 1 s).

---

## 1. Estado actual

| Exp. | Descripción | Estado |
|---|---|---|
| 01 | Réplica del protocolo de Zheng & Lu 2015 (por sesión, trials 1-9 train / 10-15 test), ANN + SVM lineal | ✅ Hecho (5 semillas, 4 y 62 canales, 2 sesiones por sujeto) |
| 02 | ANN cross-subject (test 1-3, val 4-5, train 6-15), z-score por sujeto | ✅ Hecho (5 semillas, 4 y 62 canales) |
| 03 | SNN cross-subject con el mismo split | ✅ Hecho (4 canales) · ⏳ 62 canales pendiente (a decidir con el tutor) |
| 04 | Búsqueda de hiperparámetros (ANN, SNN) y de features de entrada (ANN) | ✅ Hecho |
| 05+ | Despliegue de la ANN en BrainChip AKD1000 | ⏳ Pendiente (carpeta `akida/` preparada) |

## 2. Estructura del repositorio

```
tfg-eeg-emotion/
├── README.md
├── requirements.txt         # versiones con las que se obtuvieron los resultados
├── pyproject.toml           # permite "pip install -e ." del paquete eeg_emotion
├── data/                    # SEED (no se sube a git)
├── src/eeg_emotion/
│   ├── config.py            # rutas, índices de canales, sujetos de cada split
│   ├── data.py              # carga de SEED, canales, normalización, splits, rate coding
│   ├── models.py            # EmotionANN, FlexANN, EmotionSNN, FlexSNN
│   └── training.py          # bucle de entrenamiento, evaluate, métricas, semillas
├── scripts/                 # un script por experimento
│   ├── 01_zheng_lu_ann_svm.py
│   ├── 02_ann_cross_subject.py
│   ├── 03_snn_cross_subject.py
│   ├── 04a_ann_hparam_search.py
│   ├── 04b_snn_hparam_search.py
│   ├── 04c_ann_feature_experiments.py
│   └── aggregate_seeds.py   # media ± desviación de varias semillas -> results/summary.csv
├── legacy/                  # rutas históricas tal cual estaban (ver legacy/README.md)
├── akida/                   # despliegue en AKD1000 (vacía por ahora)
├── results/<experimento>/<canales>_seed<N>/   # config.json, metrics.json,
│                                              # confusion_matrix.csv, model.pth
├── notebooks/               # exploración (notebooks y scripts explorar_*.py)
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
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .                  # instala src/eeg_emotion en modo editable
```

Los comandos se ejecutan desde la raíz del repositorio.

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
| Tabla | `python scripts/aggregate_seeds.py` | Tablas de resultados de 4.1-4.4 |

Para varias semillas y los dos conjuntos de canales (01, 02 y 03):

```bash
for ch in 4 62; do for s in 0 1 2 3 4; do
  python scripts/02_ann_cross_subject.py --channels $ch --seed $s
done; done
python scripts/aggregate_seeds.py
```

- **Pruebas rápidas.** `--epochs 2` en 01/02/03, y `--trials 1 --max-epochs 1` en las búsquedas.
- **Reanudar.** Las búsquedas (04a/04b/04c) se reanudan solas desde su CSV si se interrumpen.
- **Duración.** El experimento 01 tarda unos 5 min por semilla (4 canales) y unos 6 min (62 canales). La SNN tarda unos 2 min por epoch con 4 canales en CPU, así que 50 epochs llevan más de 1.5 h por semilla. Las 5 semillas de la tabla se ejecutaron en paralelo con `OMP_NUM_THREADS=3`. El número de hilos queda en `config.json` porque en CPU puede afectar a los últimos decimales. Conviene **desactivar la suspensión del equipo** durante las ejecuciones largas: si se suspende, la columna `seconds` de los CSV incluye el tiempo suspendido.

## 6. Resultados actuales

Media ± desviación típica (poblacional) entre semillas. Lo genera `scripts/aggregate_seeds.py` en `results/summary.csv`.

**Réplica de Zheng & Lu (experimento 01, 5 semillas: 0-4)**

Protocolo *subject-dependent* del paper: 15 sujetos × 2 sesiones = 30 experimentos; en cada uno, trials 1-9 para entrenar y 10-15 para test. La accuracy de cada semilla es la media de las 30 sesiones. La última columna es la desviación entre sesiones (media de las 5 semillas), que es la que reporta el paper.

| Modelo | Canales | Accuracy | F1 macro | Accuracy (%) ± std entre sesiones | Paper (Zheng & Lu 2015) |
|---|---|---|---|---|---|
| ANN (`EmotionANN`, 50 epochs) | 4 | 0.7824 ± 0.0043 | 0.7633 ± 0.0048 | 78.24 ± 13.20 | 82.88 ± 10.92 |
| SVM lineal | 4 | 0.7771 ± 0.0000 | 0.7588 ± 0.0000 | 77.71 ± 14.33 | 82.88 ± 10.92 |
| ANN (`EmotionANN`, 50 epochs) | 62 | 0.7829 ± 0.0045 | 0.7597 ± 0.0063 | 78.29 ± 11.83 | DBN 86.08 ± 8.34 |
| SVM lineal | 62 | 0.7940 ± 0.0000 | 0.7726 ± 0.0000 | 79.40 ± 12.84 | SVM 83.99 ± 9.72 |

- La SVM no depende de la semilla (desviación 0 entre semillas): solo varía la ANN.
- Ambos modelos quedan unos 4-5 puntos por debajo del paper, con una dispersión entre sesiones parecida.
- En este protocolo, 4 canales rinden casi igual que 62 (ANN 78.2 frente a 78.3), en línea con la conclusión del paper.

**Modelos base cross-subject (experimentos 02 y 03, 5 semillas: 0-4)**

Test con los sujetos 1-3 (30 546 ventanas), no vistos en el entrenamiento.

| Modelo | Exp. | Canales | Accuracy | F1 macro |
|---|---|---|---|---|
| ANN (`EmotionANN`, 20/310 → 128 → 64 → 3) | 02 | 4 | 0.6236 ± 0.0072 | 0.6223 ± 0.0065 |
| ANN (`EmotionANN`) | 02 | 62 | 0.6508 ± 0.0081 | 0.6514 ± 0.0077 |
| SNN (`EmotionSNN`, LIF, T = 50) | 03 | 4 | 0.6026 ± 0.0041 | 0.6018 ± 0.0043 |
| SNN (`EmotionSNN`) | 03 | 62 | pendiente (a decidir con el tutor) | |

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
  - El C de la SVM lineal se elige en cada sesión con validación cruzada `GroupKFold` de 3 folds agrupada por trial dentro del train, para que ventanas de un mismo trial no caigan a la vez en train y validación.

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
- **Réplica de Zheng & Lu: ~4-5 puntos por debajo del paper.** El paper no detalla todo el preprocesado ni cómo eligió los hiperparámetros. Además, el C elegido para la SVM cae a menudo en el extremo inferior de la rejilla (2⁻¹⁰): en 10 de 30 sesiones con 4 canales y en 6 de 30 con 62. Ampliar la rejilla podría mejorar algo la SVM.
- **La SNN se entrena y evalúa en software** (SpikingJelly, CPU). No se han medido latencia ni energía en hardware.
- **El despliegue en AKD1000 exige convertir el modelo** de PyTorch a Keras/QuantizeML, cuantizarlo a 4 bits y cuantizar la entrada a enteros (uint8). Esto puede costar accuracy respecto a la ANN en float32.

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

En este TFG se seguirá ese pipeline con la ANN de los experimentos 02/04 (un MLP pequeño, con 20 entradas en la versión de 4 canales). Hay tres diferencias respecto al caso AoA que habrá que resolver en `akida/`:

1. **Framework.** El modelo está en PyTorch y hay que reimplementarlo en Keras (`tf_keras`).
2. **Entrada.** Las features son continuas (z-score) y hay que cuantizarlas a enteros.
3. **Entorno.** El pipeline usa Python 3.11 y el entorno de este repositorio usa 3.14, así que el despliegue necesitará un entorno virtual separado.
