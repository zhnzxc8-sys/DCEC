#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/CVIP2026_DCEC_v11_JOURNAL

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com

VAL="/root/autodl-tmp/flickr30k_trainval/data/validation-00000-of-00001.parquet"
CACHE="/root/autodl-tmp/hf_cache"
OUT="results/journal_baselines/dcec_vs_dn"

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

declare -A CKPTS
CKPTS[2027]="results/multiseed/a3_no_sem_seed2027/best_dcec_ablation.pt"
CKPTS[2028]="results/multiseed/a3_no_sem_seed2028/best_dcec_ablation.pt"

for MODEL_SEED in 2027 2028; do

    CKPT="${CKPTS[$MODEL_SEED]}"
    LOG="$OUT/DCEC_A3_MODELSEED${MODEL_SEED}_VALIDATION22.log"

    echo "===== MODEL SEED ${MODEL_SEED} START =====" | tee -a "$LOG"
    date | tee -a "$LOG"

    for item in "${conditions[@]}"; do
        read -r DEG SEV <<< "$item"

        if [[ "$DEG" == "clean" ]]; then
            TAG="clean"
        else
            TAG="${DEG}_s${SEV}"
        fi

        RESULT="$OUT/a3_seed${MODEL_SEED}_${TAG}.json"

        echo | tee -a "$LOG"
        echo "===== ${MODEL_SEED} / ${TAG} =====" | tee -a "$LOG"

        if [[ -s "$RESULT" ]]; then
            echo "[SKIP EXISTING] $RESULT" | tee -a "$LOG"
            continue
        fi

        python src/eval_dcec_ablation_validation_b32rev.py \
          --variant no_sem \
          --checkpoint "$CKPT" \
          --val-file "$VAL" \
          --cache-dir "$CACHE" \
          --degradation "$DEG" \
          --severity "$SEV" \
          --seed 2026 \
          --batch-size 64 \
          --output "$RESULT" \
          2>&1 | tee -a "$LOG"

        echo "[DONE] ${MODEL_SEED} / ${TAG}" | tee -a "$LOG"
    done

    echo "===== MODEL SEED ${MODEL_SEED} COMPLETE =====" | tee -a "$LOG"
done

echo "===== ALL DCEC SEEDS COMPLETE ====="
