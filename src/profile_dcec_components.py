"""Component-level latency profiling for DCEC.

Validation only. Final test is never accessed.

Measure separately:
1. Frozen CLIP image encoding
2. Original PIL/NumPy quality descriptor
3. A3 adapter forward
4. Full CLIP + quality + A3 pipeline
"""

import json
import time
from pathlib import Path
from statistics import mean, median, pstdev

import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from dcec_ablation_model import DCECAblation, ablation_spec
from dcec_dataset import DCECDataset, dcec_collate
from dcec_quality import extract_quality_descriptor
from eval_dcec_validation import _frozen_image_embeddings, freeze_clip


VAL_FILE = Path(
    "/root/autodl-tmp/flickr30k_trainval/data/"
    "validation-00000-of-00001.parquet"
)

CHECKPOINT = Path(
    "results/ablation_formal/no_sem/best_dcec_ablation.pt"
)

CACHE_DIR = Path("/root/autodl-tmp/hf_cache")
OUTPUT = Path(
    "results/journal_efficiency/component_profile.json"
)

BATCH_SIZE = 32
WARMUP_BATCHES = 5
MEASURE_BATCHES = 25


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summary(values, batch_size):
    return {
        "batch_ms_mean": mean(values),
        "batch_ms_median": median(values),
        "batch_ms_std": pstdev(values),
        "per_image_ms_mean": mean(values) / batch_size,
    }


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

        if len(batches) >= WARMUP_BATCHES + MEASURE_BATCHES:
            break

    warmup = batches[:WARMUP_BATCHES]
    measured = batches[
        WARMUP_BATCHES:
        WARMUP_BATCHES + MEASURE_BATCHES
    ]

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=str(CACHE_DIR),
        use_safetensors=True,
    ).to(device)

    freeze_clip(clip_model)

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=str(CACHE_DIR),
    )

    a3 = load_a3(device)

    print("device:", device)
    print("gpu:", torch.cuda.get_device_name(device))
    print("batch_size:", BATCH_SIZE)
    print("measurement_batches:", len(measured))

    # Warmup complete pipeline
    for images in warmup:
        z = _frozen_image_embeddings(
            clip_model,
            processor,
            images,
            device,
        )

        q = torch.stack(
            [
                extract_quality_descriptor(image)
                for image in images
            ]
        ).to(device)

        with torch.no_grad():
            a3(z.to(device), q)

    sync(device)

    clip_times = []
    quality_times = []
    adapter_times = []
    total_times = []

    for batch_index, images in enumerate(measured, 1):

        # --------------------------------------------------
        # 1. Frozen CLIP
        # --------------------------------------------------
        sync(device)
        t0 = time.perf_counter()

        z = _frozen_image_embeddings(
            clip_model,
            processor,
            images,
            device,
        )

        sync(device)
        clip_ms = (time.perf_counter() - t0) * 1000.0
        clip_times.append(clip_ms)

        # --------------------------------------------------
        # 2. Quality descriptor only
        # --------------------------------------------------
        t0 = time.perf_counter()

        q_cpu = torch.stack(
            [
                extract_quality_descriptor(image)
                for image in images
            ]
        )

        quality_ms = (time.perf_counter() - t0) * 1000.0
        quality_times.append(quality_ms)

        q = q_cpu.to(device)

        # --------------------------------------------------
        # 3. Adapter only
        # --------------------------------------------------
        sync(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            a3(
                z.to(device),
                q,
            )

        sync(device)
        adapter_ms = (time.perf_counter() - t0) * 1000.0
        adapter_times.append(adapter_ms)

        # --------------------------------------------------
        # 4. Full pipeline independently
        # --------------------------------------------------
        sync(device)
        t0 = time.perf_counter()

        z_full = _frozen_image_embeddings(
            clip_model,
            processor,
            images,
            device,
        )

        q_full = torch.stack(
            [
                extract_quality_descriptor(image)
                for image in images
            ]
        ).to(device)

        with torch.no_grad():
            a3(
                z_full.to(device),
                q_full,
            )

        sync(device)
        total_ms = (time.perf_counter() - t0) * 1000.0
        total_times.append(total_ms)

        print(
            f"[{batch_index:02d}/{len(measured)}] "
            f"CLIP={clip_ms:.2f} ms | "
            f"Quality={quality_ms:.2f} ms | "
            f"A3={adapter_ms:.4f} ms | "
            f"Total={total_ms:.2f} ms"
        )

    result = {
        "experiment": "DCEC component latency decomposition",
        "dataset_split": "Flickr30K validation",
        "final_test_accessed": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "batch_size": BATCH_SIZE,
        "measurement_batches": len(measured),
        "frozen_clip": summary(
            clip_times,
            BATCH_SIZE,
        ),
        "quality_descriptor_original": summary(
            quality_times,
            BATCH_SIZE,
        ),
        "a3_adapter": summary(
            adapter_times,
            BATCH_SIZE,
        ),
        "full_pipeline": summary(
            total_times,
            BATCH_SIZE,
        ),
    }

    total_mean = result["full_pipeline"]["batch_ms_mean"]

    result["time_share_percent"] = {
        "frozen_clip": (
            result["frozen_clip"]["batch_ms_mean"]
            / total_mean * 100.0
        ),
        "quality_descriptor": (
            result["quality_descriptor_original"]["batch_ms_mean"]
            / total_mean * 100.0
        ),
        "a3_adapter": (
            result["a3_adapter"]["batch_ms_mean"]
            / total_mean * 100.0
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
    print("DCEC COMPONENT LATENCY DECOMPOSITION")
    print("=" * 80)

    print(
        "Frozen CLIP:",
        f"{result['frozen_clip']['batch_ms_mean']:.3f} ms/batch"
    )

    print(
        "Quality descriptor:",
        f"{result['quality_descriptor_original']['batch_ms_mean']:.3f} ms/batch"
    )

    print(
        "A3 adapter:",
        f"{result['a3_adapter']['batch_ms_mean']:.4f} ms/batch"
    )

    print(
        "Full pipeline:",
        f"{result['full_pipeline']['batch_ms_mean']:.3f} ms/batch"
    )

    print()
    print("Approximate time shares:")

    for name, value in result["time_share_percent"].items():
        print(
            f"{name}: {value:.2f}%"
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
