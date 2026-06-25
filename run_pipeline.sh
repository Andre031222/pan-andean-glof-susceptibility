#!/bin/bash
set -e

AREA=${1:-all}
VENV=/home/andre/Documents/GLOF_Andes_Project-Paper/.venv_glof
PROJECT=/mnt/discoD/Research-Dev/AUP_Papers/03.-ML_BasedPanAndean

source "$VENV/bin/activate"
cd "$PROJECT"

if [ "$AREA" = "all" ]; then
    AREA_ARG="--all"
else
    AREA_ARG="$AREA"
fi

echo "=========================================="
echo " GLOF Susceptibility Pipeline  area=$AREA"
echo "=========================================="

echo ""
echo "=== Phase 1: Download ==="
python scripts/01_download_s2.py $AREA_ARG
python scripts/02_download_landsat.py $AREA_ARG

echo ""
echo "=== Phase 2: DEM + Terrain ==="
python scripts/03_process_dem.py $AREA_ARG

echo ""
echo "=== Phase 3: Lake Detection ==="
python scripts/04_detect_lakes_s2.py $AREA_ARG
python scripts/05_detect_lakes_landsat.py $AREA_ARG

echo ""
echo "=== Phase 4: Features + Labels ==="
python scripts/06_extract_features.py --source all
python scripts/07_match_glofs.py --buffer 5000 --pre2017-buffer 10000

echo ""
echo "=== Phase 5: Model ==="
python scripts/08_train_model.py

echo ""
echo "=========================================="
echo " DONE. Check models/model_comparison.csv"
echo "=========================================="
