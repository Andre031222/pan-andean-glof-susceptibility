#!/bin/bash
# Monitor de progreso REAL basado en disco (no en logs con buffer).
# Uso:  bash monitor.sh           -> resumen de todas las areas
#       bash monitor.sh AREA      -> detalle de una area
#       watch -n 10 bash monitor.sh   -> refresco automatico cada 10s
PROJECT=/mnt/discoD/Research-Dev/AUP_Papers/03.-ML_BasedPanAndean
cd "$PROJECT" || exit 1

AREAS=$(python - <<'PY'
from config_expanded_study_areas import EXPANDED_STUDY_AREAS
print(" ".join(sorted(EXPANDED_STUDY_AREAS.keys())))
PY
)

count_tif () { find "$1" -name '*.tif' 2>/dev/null | wc -l; }

echo "============================================================"
echo " ESTADO PIPELINE GLOF        $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "Procesos activos:"
ps -eo pid,etimes,cmd | grep -E "0[0-9]_(download|process|detect|extract|match|train)" | grep -v grep \
  | awk '{printf "  PID %-7s %4ss  %s\n", $1, $2, $3" "$4}' || echo "  (ninguno)"
echo ""

detail () {
  local a="$1"
  local s2=$(count_tif "data/raw/sentinel2/$a")
  local ls=$(count_tif "data/raw/landsat/$a")
  local dem=$(count_tif "data/processed/terrain/$a")
  printf "  %-28s S2:%3s/54  Landsat:%3s  DEM:%2s\n" "$a" "$s2" "$ls" "$dem"
}

if [ -n "$1" ]; then
  echo "Detalle: $1"
  for y in 2017 2018 2019 2020 2021 2022 2023 2024 2025; do
    n=$(count_tif "data/raw/sentinel2/$1/$y")
    st="pendiente"; [ "$n" -ge 6 ] && st="completo" || { [ "$n" -gt 0 ] && st="$n/6 parcial"; }
    printf "    S2 %s : %s .tif  (%s)\n" "$y" "$n" "$st"
  done
  echo "    Peso: $(du -sh data/raw/sentinel2/$1 2>/dev/null | cut -f1)"
else
  for a in $AREAS; do detail "$a"; done
fi
echo ""
echo "Peso total data/: $(du -sh data 2>/dev/null | cut -f1)"
