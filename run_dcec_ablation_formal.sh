#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/CVIP2026_DCEC_v7

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com

for VAR in residual_only quality_no_gate no_sem no_rank
do
    OUT="results/ablation_formal/${VAR}"

    if [ -f "${OUT}/COMPLETED" ]; then
        echo "========================================"
        echo "SKIP completed: ${VAR}"
        echo "========================================"
        continue
    fi

    mkdir -p "$OUT"

    echo "========================================"
    echo "FORMAL ABLATION: ${VAR}"
    echo "========================================"

    python -u src/train_dcec_ablation.py \
      --variant "$VAR" \
      --train-dir /root/autodl-tmp/flickr30k_trainval/data \
      --val-file /root/autodl-tmp/flickr30k_trainval/data/validation-00000-of-00001.parquet \
      --cache-dir /root/autodl-tmp/hf_cache \
      --output-dir "$OUT" \
      --epochs 5 \
      --batch-size 32 \
      --lr 0.001 \
      --weight-decay 0.0001 \
      --seed 2026 \
      --num-workers 4 \
      2>&1 | tee "${OUT}/formal_training.log"

    touch "${OUT}/COMPLETED"

    echo "FINISHED: ${VAR}"
done

echo "ALL FORMAL ABLATIONS COMPLETED"
