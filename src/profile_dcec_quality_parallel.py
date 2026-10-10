"""Exact parallel quality-descriptor profiling.

Uses the original frozen extract_quality_descriptor() implementation.
No descriptor definition is changed.

Validation data only. Final test is never accessed.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch.utils.data import DataLoader

from dcec_dataset import DCECDataset, dcec_collate
from dcec_quality import extract_quality_descriptor


VAL_FILE = Path(
    "/root/autodl-tmp/flickr30k_trainval/data/"
    "validation-00000-of-00001.parquet"
)

CACHE_DIR = Path("/root/autodl-tmp/hf_cache")

OUTPUT = Path(
    "results/journal_efficiency/"
    "quality_parallel_profile.json"
)

BATCH_SIZE = 32
WORKER_COUNTS = [1, 2, 4, 8, 16]
WARMUP = 3
REPEATS = 30


def extract_serial(images):
    return torch.stack(
        [
            extract_quality_descriptor(image)
            for image in images
        ]
    )


def extract_parallel(images, executor):
    descriptors = list(
        executor.map(
            extract_quality_descriptor,
            images,
        )
    )

    return torch.stack(descriptors)


def summarize(values):
    return {
        "latency_ms_mean": mean(values),
        "latency_ms_std": pstdev(values),
        "latency_ms_min": min(values),
        "latency_ms_max": max(values),
    }


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

    if len(images) != BATCH_SIZE:
        raise RuntimeError(
            f"Expected batch size {BATCH_SIZE}, got {len(images)}"
        )

    print("images:", len(images))

    print()
    print("Computing serial reference...")

    reference = extract_serial(images)

    print("reference shape:", tuple(reference.shape))

    results = []

    serial_times = []

    for _ in range(WARMUP):
        extract_serial(images)

    for _ in range(REPEATS):
        start = time.perf_counter()

        current = extract_serial(images)

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        serial_times.append(elapsed)

    serial_summary = summarize(serial_times)

    print()
    print(
        "Serial:",
        f"{serial_summary['latency_ms_mean']:.3f}"
        " ms/batch"
    )

    results.append(
        {
            "workers": 1,
            **serial_summary,
            "speedup_vs_serial": 1.0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "exact_equal": True,
        }
    )

    for workers in WORKER_COUNTS[1:]:
        print()
        print(f"Testing workers={workers} ...")

        with ThreadPoolExecutor(
            max_workers=workers
        ) as executor:

            for _ in range(WARMUP):
                extract_parallel(
                    images,
                    executor,
                )

            parallel_reference = extract_parallel(
                images,
                executor,
            )

            difference = (
                parallel_reference
                - reference
            ).abs()

            max_abs_error = float(
                difference.max().item()
            )

            mean_abs_error = float(
                difference.mean().item()
            )

            exact_equal = bool(
                torch.equal(
                    parallel_reference,
                    reference,
                )
            )

            times = []

            for _ in range(REPEATS):
                start = time.perf_counter()

                current = extract_parallel(
                    images,
                    executor,
                )

                elapsed = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                times.append(elapsed)

        current_summary = summarize(times)

        speedup = (
            serial_summary["latency_ms_mean"]
            / current_summary["latency_ms_mean"]
        )

        row = {
            "workers": workers,
            **current_summary,
            "speedup_vs_serial": speedup,
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "exact_equal": exact_equal,
        }

        results.append(row)

        print(
            "latency:",
            f"{row['latency_ms_mean']:.3f} ms/batch"
        )

        print(
            "speedup:",
            f"{speedup:.3f}x"
        )

        print(
            "exact_equal:",
            exact_equal
        )

        print(
            "max_abs_error:",
            max_abs_error
        )

    fastest = min(
        results,
        key=lambda row: row["latency_ms_mean"],
    )

    result = {
        "experiment": (
            "Exact parallel DCEC quality descriptor profiling"
        ),
        "dataset_split": "Flickr30K validation",
        "final_test_accessed": False,
        "batch_size": BATCH_SIZE,
        "warmup_repeats": WARMUP,
        "measurement_repeats": REPEATS,
        "descriptor_implementation": (
            "original extract_quality_descriptor unchanged"
        ),
        "results": results,
        "fastest": fastest,
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
    print("QUALITY DESCRIPTOR PARALLEL PROFILE")
    print("=" * 80)

    for row in results:
        print(
            f"workers={row['workers']:2d} | "
            f"{row['latency_ms_mean']:.3f} ms | "
            f"speedup={row['speedup_vs_serial']:.3f}x | "
            f"exact={row['exact_equal']} | "
            f"max_error={row['max_abs_error']}"
        )

    print()
    print(
        "FASTEST WORKERS:",
        fastest["workers"],
    )

    print(
        "FASTEST LATENCY:",
        f"{fastest['latency_ms_mean']:.3f} ms/batch",
    )

    print(
        "FASTEST SPEEDUP:",
        f"{fastest['speedup_vs_serial']:.3f}x",
    )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
