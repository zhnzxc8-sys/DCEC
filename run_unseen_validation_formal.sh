#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/CVIP2026_DCEC_v8

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com

VAL="/root/autodl-tmp/flickr30k_trainval/data/validation-00000-of-00001.parquet"
CACHE="/root/autodl-tmp/hf_cache"
CKPT="results/ablation_formal/no_sem/best_dcec_ablation.pt"

BASE_OUT="results/unseen_formal/frozen_clip"
A3_OUT="results/unseen_formal/a3_no_sem_seed2026"
PROV_OUT="results/provenance/unseen"

mkdir -p "$BASE_OUT" "$A3_OUT" "$PROV_OUT"

if [[ "$VAL" == *"/test"* ]] || [[ "$VAL" == *"test.parquet"* ]]; then
    echo "ERROR: test access forbidden"
    exit 1
fi

if [ ! -f "$VAL" ]; then
    echo "ERROR: validation parquet missing: $VAL"
    exit 1
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: A3 checkpoint missing: $CKPT"
    exit 1
fi

sha256sum \
    configs/unseen_degradation_sweep.json \
    src/image_degradations.py \
    src/run_degradation_baseline.py \
    src/eval_dcec_ablation_validation.py \
    "$CKPT" \
    > "$PROV_OUT/formal_unseen_sha256.txt"

DEGRADATIONS=(
    motion_blur
    contrast
    pixelate
)

echo "=================================================="
echo "FORMAL UNSEEN DEGRADATION VALIDATION"
echo "Validation only - Test remains frozen"
echo "=================================================="

echo
echo "========== FROZEN CLIP =========="

for DEG in "${DEGRADATIONS[@]}"
do
    for SEV in 1 2 3
    do
        OUT="${BASE_OUT}/${DEG}_s${SEV}.json"

        if [ -s "$OUT" ]; then
            echo "SKIP Frozen CLIP ${DEG} S${SEV}: already exists"
            continue
        fi

        echo
        echo "----------------------------------------"
        echo "Frozen CLIP | ${DEG} | S${SEV}"
        echo "----------------------------------------"

        python -u src/run_degradation_baseline.py \
            --local-parquet "$VAL" \
            --split validation \
            --degradation "$DEG" \
            --severity "$SEV" \
            --seed 2026 \
            --batch-size 32 \
            --cache-dir "$CACHE" \
            --output "$OUT"
    done
done

touch "${BASE_OUT}/COMPLETED"

echo
echo "========== A3 NO_SEM SEED 2026 =========="

for DEG in "${DEGRADATIONS[@]}"
do
    for SEV in 1 2 3
    do
        OUT="${A3_OUT}/${DEG}_s${SEV}.json"

        if [ -s "$OUT" ]; then
            echo "SKIP A3 ${DEG} S${SEV}: already exists"
            continue
        fi

        echo
        echo "----------------------------------------"
        echo "A3 no_sem | ${DEG} | S${SEV}"
        echo "----------------------------------------"

        python -u src/eval_dcec_ablation_validation.py \
            --variant no_sem \
            --checkpoint "$CKPT" \
            --val-file "$VAL" \
            --cache-dir "$CACHE" \
            --degradation "$DEG" \
            --severity "$SEV" \
            --seed 2026 \
            --batch-size 32 \
            --output "$OUT"
    done
done

touch "${A3_OUT}/COMPLETED"

echo
echo "=================================================="
echo "ALL FORMAL UNSEEN VALIDATION COMPLETED"
echo "=================================================="
