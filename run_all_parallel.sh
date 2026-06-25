#!/bin/bash
# Pipeline Pan-Andino PARALELO — baja N areas a la vez (cada area usa ~1MB/s, hay ~4.4MB/s).
# Modo orquestador (sin args):   corre fases 1-3 de las 15 areas con CONCURRENCY en paralelo,
#                                luego fases 4-5 (features + modelo) UNA vez.
# Modo worker (con 1 area):      corre fases 1-3 de esa area (lo invoca xargs, no llamar a mano).
#
# Uso:
#   nohup bash run_all_parallel.sh > logs/all.log 2>&1 &
#   CONCURRENCY=4 nohup bash run_all_parallel.sh > logs/all.log 2>&1 &   # ajustar paralelismo
#   bash monitor.sh

VENV=/home/andre/Documents/GLOF_Andes_Project-Paper/.venv_glof
PROJECT=/mnt/discoD/Research-Dev/AUP_Papers/03.-ML_BasedPanAndean
CONCURRENCY=${CONCURRENCY:-4}

source "$VENV/bin/activate"
cd "$PROJECT" || exit 1
export PYTHONUNBUFFERED=1

# --- Optimizacion GDAL para COG remotos en alta latencia ---
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export GDAL_HTTP_MULTIPLEX=YES
export GDAL_HTTP_VERSION=2
export GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES
export GDAL_HTTP_MAX_RETRY=5
export GDAL_HTTP_RETRY_DELAY=2
export CPL_VSIL_CURL_CHUNK_SIZE=10485760
export CPL_VSIL_CURL_CACHE_SIZE=536870912
export VSI_CACHE=TRUE
export VSI_CACHE_SIZE=268435456
export GDAL_NUM_THREADS=2
mkdir -p logs

ts () { date '+%Y-%m-%d %H:%M:%S'; }

# ---------- MODO WORKER: una sola area (fases 1-3) ----------
if [ -n "$1" ]; then
  AREA="$1"
  alog="logs/${AREA}.log"
  {
    echo "### $AREA inicio $(ts)"
    python scripts/01_download_s2.py "$AREA"      && \
    python scripts/02_download_landsat.py "$AREA" && \
    python scripts/03_process_dem.py "$AREA"      && \
    python scripts/04_detect_lakes_s2.py "$AREA"  && \
    python scripts/05_detect_lakes_landsat.py "$AREA"
    rc=$?
    echo "### $AREA fin $(ts) rc=$rc"
    exit $rc
  } >"$alog" 2>&1
  exit $?
fi

# ---------- MODO ORQUESTADOR ----------
# EXCLUDE="area1 area2" para saltar areas (ej. patagonia_norte por bbox gigante)
AREAS=$(EXCLUDE="$EXCLUDE" python - <<'PY'
import os
from config_expanded_study_areas import EXPANDED_STUDY_AREAS
excl = set(os.environ.get('EXCLUDE','').split())
print("\n".join(a for a in sorted(EXPANDED_STUDY_AREAS.keys()) if a not in excl))
PY
)

echo "############################################################"
echo "# PIPELINE PARALELO  CONCURRENCY=$CONCURRENCY   inicio $(ts)"
echo "############################################################"
echo "$AREAS" | tr '\n' ' '; echo ""

# Fases por-area en paralelo (xargs -P). Cada area -> este mismo script en modo worker.
echo "$AREAS" | xargs -P "$CONCURRENCY" -I {} bash run_all_parallel.sh {}

echo ""
echo "=== Resultado por area ==="
for a in $AREAS; do
  last=$(grep -E "rc=" "logs/${a}.log" 2>/dev/null | tail -1)
  echo "  $a : ${last:-sin log}"
done

# Fases globales 4-5 UNA vez
echo ""
echo "================= FASE GLOBAL: features + modelo $(ts) ================="
python scripts/06_extract_features.py --source all          2>&1 | tee logs/_features.log
python scripts/07_match_glofs.py --buffer 5000 --pre2017-buffer 10000  2>&1 | tee logs/_match.log
python scripts/08_train_model.py                            2>&1 | tee logs/_train.log

echo ""
echo "############################################################"
echo "# DONE $(ts)   ->  models/model_comparison.csv"
echo "############################################################"
