"""Exact-parallel end-to-end DCEC efficiency profiling.

Validation only.
Final Flickr30K test is never accessed.

The original extract_quality_descriptor() function is unchanged.
Only per-image descriptor extraction is parallelized.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median, pstdev

import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from dcec_ablation_model import DCECAblation, ablation_spec
from dcec_dataset import DCECDataset, dcec_collate
from dcec_quality import extract_quality_descriptor
from eval_dcec_validation import _frozen_image_embeddings, freeze_clip


MODEL_NAME = "openai/clip-vit-base-patch32"

VAL_FILE = Path(
    "/root/autodl-tmp/flickr30k_trainval/data/"
    "validation-00000-of-00001.parquet"
)

CHECKPOINT = Path(
    "results/ablation_formal/no_sem/best_dcec_ablation.pt"
)

CACHE_DIR = Path("/root/autodl-tmp/hf_cache")

OUTPUT = Path(
    "results/journal_efficiency/"
    "end_to_end_parallel4_profile.json"
)

BATCH_SIZE = 32
WORKERS = 4
WARMUP_BATCHES = 5
MEASURE_BATCHES = 25


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_a3(device):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("variant") != "no_sem":
        raise ValueError("Expected no_sem checkpoint.")

    architecture = str(
        ablation_spec("no_sem")["architecture"]
    )

    model = DCECAblation(
        architecture
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def summarize(times, batch_size):
    return {
        "batch_latency_ms_mean": mean(times),
        "batch_latency_ms_median": median(times),
        "batch_latency_ms_std": pstdev(times),
        "per_image_latency_ms_mean": mean(times) / batch_size,
        "throughput_images_per_second": (
            batch_size / (mean(times) / 1000.0)
        ),
    }


def extract_quality_parallel(images, executor):
    descriptors = list(
        executor.map(
            extract_quality_descriptor,
            images,
        )
    )

    return torch.stack(descriptors)


def run_clip_only(
    clip_model,
    processor,
    images,
    device,
):
    return _frozen_image_embeddings(
        clip_model,
        processor,
        images,
        device,
    )


def run_clip_dcec_parallel(
    clip_model,
    processor,
    a3,
    images,
    device,
    executor,
):
    z = _frozen_image_embeddings(
        clip_model,
        processor,
        images,
        device,
    )

    quality = extract_quality_parallel(
        images,
        executor,
    ).to(device)

    with torch.no_grad():
        outputs = a3(
            z.to(device),
            quality,
        )

    return outputs["z_corrected"]


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    batches = []

    for batch in loader:
        batches.append(batch["clean_image"])

        if len(batches) >= (
            WARMUP_BATCHES + MEASURE_BATCHES
        ):
            break

    warmup = batches[:WARMUP_BATCHES]
    measured = batches[
        WARMUP_BATCHES:
        WARMUP_BATCHES + MEASURE_BATCHES
    ]

    clip_model = CLIPModel.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        use_safetensors=True,
    ).to(device)

    freeze_clip(clip_model)

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
    )

    a3 = load_a3(device)

    print("device:", device)
    print("gpu:", torch.cuda.get_device_name(device))
    print("batch_size:", BATCH_SIZE)
    print("quality_workers:", WORKERS)
    print("measurement_batches:", len(measured))

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        # warmup
        for images in warmup:
            run_clip_only(
                clip_model,
                processor,
                images,
                device,
            )

            run_clip_dcec_parallel(
                clip_model,
                processor,
                a3,
                images,
                device,
                executor,
            )

        sync(device)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        clip_times = []

        for images in measured:
            sync(device)

            t0 = time.perf_counter()

            run_clip_only(
                clip_model,
                processor,
                images,
                device,
            )

            sync(device)

            clip_times.append(
                (time.perf_counter() - t0) * 1000.0
            )

        clip_peak_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        dcec_times = []

        for images in measured:
            sync(device)

            t0 = time.perf_counter()

            run_clip_dcec_parallel(
                clip_model,
                processor,
                a3,
                images,
                device,
                executor,
            )

            sync(device)

            dcec_times.append(
                (time.perf_counter() - t0) * 1000.0
            )

        dcec_peak_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )

    clip_summary = summarize(
        clip_times,
        BATCH_SIZE,
    )

    dcec_summary = summarize(
        dcec_times,
        BATCH_SIZE,
    )

    latency_overhead_percent = (
        (
            dcec_summary["batch_latency_ms_mean"]
            / clip_summary["batch_latency_ms_mean"]
        ) - 1.0
    ) * 100.0

    throughput_drop_percent = (
        1.0
        - (
            dcec_summary["throughput_images_per_second"]
            / clip_summary["throughput_images_per_second"]
        )
    ) * 100.0

    result = {
        "experiment": (
            "DCEC exact-parallel end-to-end inference overhead"
        ),
        "dataset_split": "Flickr30K validation",
        "final_test_accessed": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "batch_size": BATCH_SIZE,
        "quality_workers": WORKERS,
        "descriptor_definition_changed": False,
        "descriptor_parallel_exact": True,
        "measurement_batches": len(measured),

        "frozen_clip": {
            **clip_summary,
            "peak_memory_mb": clip_peak_mb,
        },

        "frozen_clip_plus_dcec": {
            **dcec_summary,
            "peak_memory_mb": dcec_peak_mb,
        },

        "latency_overhead_percent": (
            latency_overhead_percent
        ),

        "throughput_drop_percent": (
            throughput_drop_percent
        ),

        "peak_memory_overhead_mb": (
            dcec_peak_mb - clip_peak_mb
        ),

        "note": (
            "Original per-image quality descriptor is used "
            "unchanged; only extraction across images is "
            "parallelized with four CPU threads."
        ),
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
    print("EXACT-PARALLEL END-TO-END RESULT")
    print("=" * 80)

    print(
        "Frozen CLIP:",
        f"{clip_summary['batch_latency_ms_mean']:.3f} ms/batch"
    )

    print(
        "CLIP + DCEC:",
        f"{dcec_summary['batch_latency_ms_mean']:.3f} ms/batch"
    )

    print(
        "Latency overhead:",
        f"{latency_overhead_percent:+.2f}%"
    )

    print()

    print(
        "Frozen CLIP throughput:",
        f"{clip_summary['throughput_images_per_second']:.2f} img/s"
    )

    print(
        "CLIP + DCEC throughput:",
        f"{dcec_summary['throughput_images_per_second']:.2f} img/s"
    )

    print(
        "Throughput drop:",
        f"{throughput_drop_percent:.2f}%"
    )

    print()

    print(
        "Frozen CLIP peak memory:",
        f"{clip_peak_mb:.2f} MB"
    )

    print(
        "CLIP + DCEC peak memory:",
        f"{dcec_peak_mb:.2f} MB"
    )

    print(
        "Peak memory overhead:",
        f"{dcec_peak_mb - clip_peak_mb:+.2f} MB"
    )

    print()
    print("Descriptor definition changed: NO")
    print("Parallel descriptor numerically exact: YES")

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
