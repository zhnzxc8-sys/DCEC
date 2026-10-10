#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/CVIP2026_DCEC_v7

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com

TRAIN_DIR="/root/autodl-tmp/flickr30k_trainval/data"
VAL_FILE="/root/autodl-tmp/flickr30k_trainval/data/validation-00000-of-00001.parquet"
CACHE_DIR="/root/autodl-tmp/hf_cache"

mkdir -p results/multiseed

run_a3 () {
    SEED=$1
    OUT="results/multiseed/a3_no_sem_seed${SEED}"

    if [ -f "${OUT}/COMPLETED" ]; then
        echo "SKIP A3 no_sem seed ${SEED}: already completed"
        return
    fi

    mkdir -p "$OUT"

    echo "=================================================="
    echo "A3 NO_SEM - SEED ${SEED}"
    echo "=================================================="

    python -u src/train_dcec_ablation.py \
        --variant no_sem \
        --train-dir "$TRAIN_DIR" \
        --val-file "$VAL_FILE" \
        --cache-dir "$CACHE_DIR" \
        --output-dir "$OUT" \
        --config configs/dcec_ablation.json \
        --epochs 5 \
        --batch-size 32 \
        --lr 0.001 \
        --weight-decay 0.0001 \
        --seed "$SEED" \
        --num-workers 4 \
        2>&1 | tee "${OUT}/run.log"

    touch "${OUT}/COMPLETED"
    echo "FINISHED: A3 no_sem seed ${SEED}"
}

run_a5 () {
    SEED=$1
    OUT="results/multiseed/a5_full_dcec_seed${SEED}"

    if [ -f "${OUT}/COMPLETED" ]; then
        echo "SKIP A5 full_dcec seed ${SEED}: already completed"
        return
    fi

    mkdir -p "$OUT"

    echo "=================================================="
    echo "A5 FULL DCEC - SEED ${SEED}"
    echo "=================================================="

    python -u src/train_dcec.py \
        --train-dir "$TRAIN_DIR" \
        --val-file "$VAL_FILE" \
        --cache-dir "$CACHE_DIR" \
        --output-dir "$OUT" \
        --config configs/dcec_v1.json \
        --epochs 5 \
        --batch-size 32 \
        --lr 0.001 \
        --weight-decay 0.0001 \
        --seed "$SEED" \
        --num-workers 4 \
        --lambda-sem 1.0 \
        --lambda-rank 1.0 \
        --lambda-res 0.001 \
        2>&1 | tee "${OUT}/run.log"

    touch "${OUT}/COMPLETED"
    echo "FINISHED: A5 full_dcec seed ${SEED}"
}

run_a3 2027
run_a3 2028
run_a5 2027
run_a5 2028

echo "=================================================="
echo "ALL DCEC MULTISEED TRAINING COMPLETED"
echo "=================================================="
