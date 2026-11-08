#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
data=data/mvtec
encoder=weights/dinov2_vitb14_reg4_pretrain.pth
output=${1:-runs/reproduction}
mkdir -p "$(dirname "$output")"
mkdir "$output"
export OMP_NUM_THREADS=6 PYTHONNOUSERSITE=1 MPLCONFIGDIR="$output/mpl-cache"
python run.py validate-data --data "$data" --output "$output/data-validation.json"
evaluations=()
for run in B-s1 F-s1 A1-s1 A2-s1 A3-s1 B-s2 F-s2 B-s3 F-s3; do
    variant=${run%-s*}
    seed=${run##*-s}
    python run.py train --config "configs/$variant.json" --data "$data" --encoder "$encoder"         --seed "$seed" --device cuda:0 --output "$output/$run"
    python run.py evaluate --checkpoint "$output/$run/inference.pt" --data "$data"         --device cuda:0 --output "$output/$run/eval"
    evaluations+=("$output/$run/eval/evaluation.json")
done
python run.py summarize "${evaluations[@]}" --output "$output/summary"
