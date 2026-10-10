"""Internal latency decomposition of the original DCEC quality descriptor.

Validation data only.
No final-test access.
The original descriptor implementation is not modified.
"""

import json
import time
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from dcec_dataset import DCECDataset, dcec_collate
from dcec_quality import (
    _grayscale_luminance,
    _laplacian_absolute_mean,
    _block_boundary_response,
)


VAL_FILE = Path(
    "/root/autodl-tmp/flickr30k_trainval/data/"
    "validation-00000-of-00001.parquet"
)

CACHE_DIR = Path("/root/autodl-tmp/hf_cache")

OUTPUT = Path(
    "results/journal_efficiency/"
    "quality_internal_profile.json"
)

BATCH_SIZE = 32
WARMUP = 3
REPEATS = 20


def profile_once(images):
    times = {
        "rgb_to_luminance": 0.0,
        "mean_std": 0.0,
        "laplacian": 0.0,
        "gray_uint8_conversion": 0.0,
        "pil_gaussian_blur": 0.0,
        "high_frequency_residual": 0.0,
        "block_boundary_response": 0.0,
    }

    total_start = time.perf_counter()

    for image in images:
        t0 = time.perf_counter()
        luminance = _grayscale_luminance(image)
        times["rgb_to_luminance"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = float(np.mean(luminance))
        _ = float(np.std(luminance))
        times["mean_std"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = _laplacian_absolute_mean(luminance)
        times["laplacian"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        grayscale_image = Image.fromarray(
            np.rint(luminance * 255.0).astype(np.uint8),
            mode="L",
        )
        times["gray_uint8_conversion"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        blurred = np.asarray(
            grayscale_image.filter(
                ImageFilter.GaussianBlur(radius=1.0)
            ),
            dtype=np.float32,
        ) / 255.0
        times["pil_gaussian_blur"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = float(np.mean(np.abs(luminance - blurred)))
        times["high_frequency_residual"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = _block_boundary_response(luminance)
        times["block_boundary_response"] += time.perf_counter() - t0

    total_ms = (time.perf_counter() - total_start) * 1000.0

    return {
        key: value * 1000.0
        for key, value in times.items()
    }, total_ms


def main():
    dataset = DCECDataset(
        split="validation",
        path=VAL_FILE,
        seed=2026,
        cache_dir=CACHE_DIR,
        degradation="clean",
        severity=0,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=dcec_collate,
    )

    batch = next(iter(loader))
    images = batch["clean_image"]

    print("images:", len(images))

    for _ in range(WARMUP):
        profile_once(images)

    records = []

    for repeat in range(REPEATS):
        components, total_ms = profile_once(images)

        records.append({
            **components,
            "total_ms": total_ms,
        })

        print(
            f"[{repeat + 1:02d}/{REPEATS}] "
            f"total={total_ms:.3f} ms"
        )

    keys = [
        "rgb_to_luminance",
        "mean_std",
        "laplacian",
        "gray_uint8_conversion",
        "pil_gaussian_blur",
        "high_frequency_residual",
        "block_boundary_response",
    ]

    summary = {}

    for key in keys:
        values = [record[key] for record in records]

        summary[key] = {
            "batch_ms_mean": mean(values),
            "batch_ms_std": pstdev(values),
        }

    total_values = [
        record["total_ms"]
        for record in records
    ]

    total_mean = mean(total_values)

    for key in keys:
        summary[key]["share_percent"] = (
            summary[key]["batch_ms_mean"]
            / total_mean
            * 100.0
        )

    result = {
        "experiment": (
            "Original DCEC quality descriptor internal latency"
        ),
        "dataset_split": "Flickr30K validation",
        "final_test_accessed": False,
        "batch_size": BATCH_SIZE,
        "measurement_repeats": REPEATS,
        "total_batch_ms_mean": total_mean,
        "total_batch_ms_std": pstdev(total_values),
        "components": summary,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("QUALITY DESCRIPTOR INTERNAL PROFILE")
    print("=" * 80)

    print(
        f"Total: {total_mean:.3f} ms / batch{BATCH_SIZE}"
    )

    print()

    ranked = sorted(
        summary.items(),
        key=lambda item: item[1]["batch_ms_mean"],
        reverse=True,
    )

    for name, data in ranked:
        print(
            f"{name:28s} "
            f"{data['batch_ms_mean']:9.3f} ms "
            f"| {data['share_percent']:6.2f}%"
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
