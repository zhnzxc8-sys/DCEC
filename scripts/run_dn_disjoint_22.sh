#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/CVIP2026_DCEC_v11_JOURNAL

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com

VAL="/root/autodl-tmp/flickr30k_trainval/data/validation-00000-of-00001.parquet"
TRAIN_GLOB="/root/autodl-tmp/flickr30k_trainval/data/train-*-of-00008.parquet"
CACHE="/root/autodl-tmp/hf_cache"
OUT="results/journal_baselines/dn"
LOG="$OUT/DN_DISJOINT_22_RUN.log"

conditions=(
"clean 0"
"jpeg 1"
"jpeg 2"
"jpeg 3"
"gaussian_blur 1"
"gaussian_blur 2"
"gaussian_blur 3"
"gaussian_noise 1"
"gaussian_noise 2"
"gaussian_noise 3"
"low_light 1"
"low_light 2"
"low_light 3"
"motion_blur 1"
"motion_blur 2"
"motion_blur 3"
"contrast 1"
"contrast 2"
"contrast 3"
"pixelate 1"
"pixelate 2"
"pixelate 3"
)

echo "===== DN DISJOINT 22 START =====" | tee -a "$LOG"
date | tee -a "$LOG"

for item in "${conditions[@]}"; do
    read -r DEG SEV <<< "$item"

    if [[ "$DEG" == "clean" ]]; then
        TAG="clean"
    else
        TAG="${DEG}_s${SEV}"
    fi

    RESULT="$OUT/formal_disjoint_${TAG}_5seed.json"

    echo | tee -a "$LOG"
    echo "===== $TAG =====" | tee -a "$LOG"

    if [[ -s "$RESULT" ]]; then
        echo "[SKIP EXISTING] $RESULT" | tee -a "$LOG"
        continue
    fi

    python src/eval_distribution_normalization.py \
      --local-parquet "$VAL" \
      --stats-parquet-glob "$TRAIN_GLOB" \
      --split validation \
      --degradation "$DEG" \
      --severity "$SEV" \
      --degradation-seed 2026 \
      --stats-samples 100 \
      --stats-seeds 9501 9502 9503 9504 9505 \
      --batch-size 64 \
      --cache-dir "$CACHE" \
      --output "$RESULT" \
      2>&1 | tee -a "$LOG"

    echo "[DONE] $TAG" | tee -a "$LOG"
done

echo
echo "===== DN DISJOINT 22 COMPLETE ====="
date
