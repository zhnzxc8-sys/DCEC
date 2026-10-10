"""End-to-end inference overhead profiling for DCEC A3.

Validation data only.
No Flickr30K final-test data is accessed.

Compare:
1. Frozen CLIP image encoding
2. Frozen CLIP + quality descriptor + A3 correction
"""

import argparse
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


MODEL_NAME = "openai/clip-vit-base-patch32"

DEFAULT_VAL = Path(
    "/root/autodl-tmp/flickr30k_trainval/data/"
    "validation-00000-of-00001.parquet"
)

DEFAULT_CHECKPOINT = Path(
    "results/ablation_formal/no_sem/best_dcec_ablation.pt"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val-file",
        type=Path,
        default=DEFAULT_VAL,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/root/autodl-tmp/hf_cache"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/journal_efficiency/end_to_end_profile.json"
        ),
    )
    return parser.parse_args()


def load_a3(path, device):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("variant") != "no_sem":
        raise ValueError(
            f"Expected no_sem checkpoint, got "
            f"{checkpoint.get('variant')!r}"
        )

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


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def run_clip_dcec(
    clip_model,
    processor,
    a3_model,
    images,
    device,
):
    z = _frozen_image_embeddings(
        clip_model,
        processor,
        images,
        device,
    )

    quality = torch.stack(
        [
            extract_quality_descriptor(image)
            for image in images
        ]
    ).to(device)

    with torch.no_grad():
        outputs = a3_model(
            z.to(device),
            quality,
        )

    return outputs["z_corrected"]


def summarize(times_ms, images_per_batch):
    return {
        "batch_latency_ms_mean": mean(times_ms),
        "batch_latency_ms_median": median(times_ms),
        "batch_latency_ms_std": pstdev(times_ms),
        "per_image_latency_ms_mean": (
            mean(times_ms) / images_per_batch
        ),
        "throughput_images_per_second": (
            images_per_batch
            / (mean(times_ms) / 1000.0)
        ),
    }


def main():
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = DCECDataset(
        split="validation",
        path=args.val_file,
        seed=2026,
        cache_dir=args.cache_dir,
        degradation="clean",
        severity=0,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dcec_collate,
    )

    clip_model = CLIPModel.from_pretrained(
        MODEL_NAME,
        cache_dir=str(args.cache_dir),
        use_safetensors=True,
    ).to(device)

    freeze_clip(clip_model)

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=str(args.cache_dir),
    )

    a3_model = load_a3(
        args.checkpoint,
        device,
    )

    batches = []

    for batch in loader:
        batches.append(batch["clean_image"])

        if len(batches) >= (
            args.warmup_batches + args.num_batches
        ):
            break

    if len(batches) <= args.warmup_batches:
        raise RuntimeError("Not enough validation batches.")

    warmup = batches[:args.warmup_batches]
    measured = batches[
        args.warmup_batches:
        args.warmup_batches + args.num_batches
    ]

    print("device:", device)
    print("gpu:", torch.cuda.get_device_name(device))
    print("batch_size:", args.batch_size)
    print("warmup_batches:", len(warmup))
    print("measurement_batches:", len(measured))

    with torch.no_grad():
        for images in warmup:
            run_clip_only(
                clip_model,
                processor,
                images,
                device,
            )

            run_clip_dcec(
                clip_model,
                processor,
                a3_model,
                images,
                device,
            )

    synchronize(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    clip_times = []

    with torch.no_grad():
        for images in measured:
            synchronize(device)

            start = time.perf_counter()

            run_clip_only(
                clip_model,
                processor,
                images,
                device,
            )

            synchronize(device)

            clip_times.append(
                (time.perf_counter() - start) * 1000.0
            )

    clip_peak_mb = (
        torch.cuda.max_memory_allocated(device)
        / (1024 ** 2)
        if device.type == "cuda"
        else None
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    dcec_times = []

    with torch.no_grad():
        for images in measured:
            synchronize(device)

            start = time.perf_counter()

            run_clip_dcec(
                clip_model,
                processor,
                a3_model,
                images,
                device,
            )

            synchronize(device)

            dcec_times.append(
                (time.perf_counter() - start) * 1000.0
            )

    dcec_peak_mb = (
        torch.cuda.max_memory_allocated(device)
        / (1024 ** 2)
        if device.type == "cuda"
        else None
    )

    effective_batch_size = len(measured[0])

    clip_summary = summarize(
        clip_times,
        effective_batch_size,
    )

    dcec_summary = summarize(
        dcec_times,
        effective_batch_size,
    )

    latency_overhead_pct = (
        (
            dcec_summary["batch_latency_ms_mean"]
            / clip_summary["batch_latency_ms_mean"]
        ) - 1.0
    ) * 100.0

    memory_overhead_mb = (
        dcec_peak_mb - clip_peak_mb
        if (
            dcec_peak_mb is not None
            and clip_peak_mb is not None
        )
        else None
    )

    result = {
        "experiment": "DCEC end-to-end inference overhead",
        "dataset_split": "Flickr30K validation",
        "final_test_accessed": False,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "batch_size": effective_batch_size,
        "measurement_batches": len(measured),
        "frozen_clip": {
            **clip_summary,
            "peak_memory_mb": clip_peak_mb,
        },
        "frozen_clip_plus_dcec": {
            **dcec_summary,
            "peak_memory_mb": dcec_peak_mb,
        },
        "latency_overhead_percent": latency_overhead_pct,
        "peak_memory_overhead_mb": memory_overhead_mb,
        "scope": (
            "Includes image preprocessing, frozen CLIP vision "
            "encoding, PIL quality descriptor extraction, "
            "and DCEC adapter inference."
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("END-TO-END EFFICIENCY RESULT")
    print("=" * 80)

    print(
        "Frozen CLIP latency:",
        f"{clip_summary['batch_latency_ms_mean']:.3f} ms/batch",
    )

    print(
        "CLIP + DCEC latency:",
        f"{dcec_summary['batch_latency_ms_mean']:.3f} ms/batch",
    )

    print(
        "Latency overhead:",
        f"{latency_overhead_pct:+.2f}%",
    )

    print(
        "Frozen CLIP throughput:",
        f"{clip_summary['throughput_images_per_second']:.2f} img/s",
    )

    print(
        "CLIP + DCEC throughput:",
        f"{dcec_summary['throughput_images_per_second']:.2f} img/s",
    )

    print(
        "Frozen CLIP peak memory:",
        f"{clip_peak_mb:.2f} MB",
    )

    print(
        "CLIP + DCEC peak memory:",
        f"{dcec_peak_mb:.2f} MB",
    )

    print(
        "Peak memory overhead:",
        f"{memory_overhead_mb:+.2f} MB",
    )

    print()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
