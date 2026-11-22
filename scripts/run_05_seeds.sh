#!/usr/bin/env bash
# Lanza una etapa del experimento 05 con semillas 0-4, en paralelo y con 1 hilo
# de TensorFlow por proceso (queda en config.json).
# Uso (Git Bash, desde la raiz del repo):
#   bash scripts/run_05_seeds.sh 05a_akida_float
#   bash scripts/run_05_seeds.sh 05b_akida_int_input --input-quant cuantiles
#   CHANNELS="62" MAXJOBS=2 bash scripts/run_05_seeds.sh 05a_akida_float
# CHANNELS: canales a lanzar (por defecto "4 62").
# MAXJOBS: procesos a la vez (por defecto 5). Con 62 canales cada proceso usa
#          ~1 GB de RAM; con poca memoria libre conviene MAXJOBS=2.
# Los logs quedan en results/<etapa>/logs/ (no se suben a git).
set -u
STAGE="$1"; shift
PY=".venv-akida/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv-akida/bin/python"
LOGS="results/${STAGE}/logs"
mkdir -p "$LOGS"
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 PYTHONIOENCODING=utf-8
SUFFIX="$(echo "$*" | tr -c 'a-z0-9' '_')"
for ch in ${CHANNELS:-4 62}; do
  for s in 0 1 2 3 4; do
    while [ "$(jobs -rp | wc -l)" -ge "${MAXJOBS:-5}" ]; do wait -n; done
    "$PY" "scripts/${STAGE}.py" --channels "$ch" --seed "$s" "$@" \
      > "$LOGS/${ch}ch_seed${s}${SUFFIX}.log" 2>&1 &
  done
done
wait
grep -h -E "TEST|Error" "$LOGS"/*"${SUFFIX}".log
