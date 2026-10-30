# Exploración

Notebooks y scripts sueltos de las primeras semanas del TFG. No forman parte
de ningún experimento: sirvieron para conocer SpikingJelly y el formato de
SEED, y para localizar los índices de los canales FT7, FT8, T7 y T8.

Se ejecutan desde la raíz del repositorio.

| Fichero | Qué hace | Datos que necesita |
|---|---|---|
| `SpikingJelly1.ipynb` | Primera SNN con SpikingJelly (LIF, T = 6) sobre MNIST, en Google Colab. Su resultado aparece en `results/legacy/baseline.csv`. | Ninguno (descarga MNIST) |
| `explorar_mat.py` | Muestra las claves y formas de un `.mat` de features y de `label.mat`. | `data/ExtractedFeatures_1s/` |
| `explorar_canales.py` | Lee los nombres de los 62 canales de un `.cnt` en bruto. Necesita `mne`. | `SEED/SEED_EEG/SEED_RAW_EEG/` |
| `explorar_canales2.py` | Busca información de canales en `Preprocessed_EEG` y obtiene los índices de FT7, FT8, T7 y T8 (14, 22, 23, 31) a partir de la lista de canales de SEED. | `SEED/SEED_EEG/Preprocessed_EEG/` |
