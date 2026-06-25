#!/bin/bash
VENV=/home/andre/Documents/GLOF_Andes_Project-Paper/.venv_glof
PROJECT=/mnt/discoD/Research-Dev/AUP_Papers/03.-ML_BasedPanAndean
source "$VENV/bin/activate"
cd "$PROJECT" || exit 1
export PYTHONUNBUFFERED=1
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export GDAL_HTTP_MULTIPLEX=YES
export GDAL_HTTP_VERSION=2
export GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES
export CPL_VSIL_CURL_CHUNK_SIZE=10485760
export VSI_CACHE=TRUE
mkdir -p logs

ts () { date '+%Y-%m-%d %H:%M:%S'; }

echo "[finalize] esperando a que termine el orquestador de descargas... $(ts)"
while pgrep -f "bash run_all_parallel.sh$" >/dev/null; do sleep 60; done
while pgrep -f "scripts/0[1-3]_" >/dev/null; do sleep 30; done
echo "[finalize] descargas terminadas $(ts)"

AREAS=$(EXCLUDE="patagonia_norte" python - <<'PY'
import os
from config_expanded_study_areas import EXPANDED_STUDY_AREAS
excl = set(os.environ.get('EXCLUDE','').split())
print("\n".join(a for a in sorted(EXPANDED_STUDY_AREAS.keys()) if a not in excl))
PY
)

echo "[finalize] regenerar terreno (fix VRM) $(ts)"
for a in $AREAS; do rm -rf "data/interim/terrain/$a"; done
echo "$AREAS" | xargs -P 5 -I {} bash -c 'python scripts/03_process_dem.py {} > logs/redem_{}.log 2>&1'

echo "[finalize] re-deteccion limpia de lagos $(ts)"
rm -f data/processed/lakes_s2/*.gpkg data/processed/lakes_landsat/*.gpkg 2>/dev/null
echo "$AREAS" | xargs -P 5 -I {} bash -c '
  python scripts/04_detect_lakes_s2.py {} > logs/redetect_s2_{}.log 2>&1
  python scripts/05_detect_lakes_landsat.py {} > logs/redetect_ls_{}.log 2>&1
'

echo "[finalize] resumen lagos por area:"
for a in $AREAS; do
  s2=$(grep -oE "[0-9]+ lakes total" logs/redetect_s2_$a.log 2>/dev/null | tail -1)
  ls=$(grep -oE "[0-9]+ lakes total" logs/redetect_ls_$a.log 2>/dev/null | tail -1)
  printf "  %-26s S2: %-16s Landsat: %s\n" "$a" "${s2:-?}" "${ls:-?}"
done

echo "[finalize] features $(ts)"
python scripts/06_extract_features.py --source all 2>&1 | tee logs/_features.log | tail -5
echo "[finalize] match GLOFs $(ts)"
python scripts/07_match_glofs.py --buffer 5000 --pre2017-buffer 10000 2>&1 | tee logs/_match.log | tail -8
echo "[finalize] entrenamiento $(ts)"
python scripts/08_train_model.py 2>&1 | tee logs/_train.log | tail -20

echo "[finalize] DONE $(ts)"
echo "=== model_comparison.csv ==="
cat models/model_comparison.csv 2>/dev/null
