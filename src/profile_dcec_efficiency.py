"""Efficiency profiling for frozen DCEC A3 journal extension.

This script measures adapter-only:
1. trainable parameter count
2. checkpoint size
3. linear-layer MACs
4. GPU forward latency
5. peak CUDA memory

No Flickr30K test data is accessed.
"""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import torch
import torch.nn as nn

from dcec_ablation_model import DCECAblation, ablation_spec


DEFAULT_CHECKPOINT = Path(
    "results/ablation_formal/no_sem/best_dcec_ablation.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile DCEC A3 adapter efficiency."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 16, 32, 64],
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/journal_efficiency/adapter_profile.json"),
    )
    return parser.parse_args()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable


def checkpoint_size_mb(path):
    return path.stat().st_size / (1024 ** 2)


def count_linear_macs(model, z, quality):
    macs = 0
    handles = []

    def hook(module, inputs, output):
        nonlocal macs

        x = inputs[0]

        if not isinstance(x, torch.Tensor):
            return

        vectors = x.numel() // x.shape[-1]

        macs += (
            vectors
            * module.in_features
            * module.out_features
        )

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(
                module.register_forward_hook(hook)
            )

    with torch.no_grad():
        model(z, quality)

    for handle in handles:
        handle.remove()

    return macs


def benchmark_latency(
    model,
    batch_size,
    device,
    warmup,
    repeats,
):
    z = torch.randn(
        batch_size,
        512,
        device=device,
    )

    quality = torch.randn(
        batch_size,
        5,
        device=device,
    )

    z = torch.nn.functional.normalize(
        z,
        p=2,
        dim=1,
    )

    with torch.no_grad():
        for _ in range(warmup):
            model(z, quality)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        times_ms = []

        with torch.no_grad():
            for _ in range(repeats):
                start = torch.cuda.Event(
                    enable_timing=True
                )
                end = torch.cuda.Event(
                    enable_timing=True
                )

                start.record()
                model(z, quality)
                end.record()

                torch.cuda.synchronize(device)

                times_ms.append(
                    start.elapsed_time(end)
                )

        peak_memory_mb = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )

    else:
        import time

        times_ms = []

        with torch.no_grad():
            for _ in range(repeats):
                start = time.perf_counter()
                model(z, quality)
                end = time.perf_counter()

                times_ms.append(
                    (end - start) * 1000.0
                )

        peak_memory_mb = None

    return {
        "batch_size": batch_size,
        "latency_ms_mean": mean(times_ms),
        "latency_ms_std": pstdev(times_ms),
        "latency_ms_per_image": mean(times_ms) / batch_size,
        "throughput_images_per_second": (
            batch_size / (mean(times_ms) / 1000.0)
        ),
        "peak_memory_mb": peak_memory_mb,
    }


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    variant = checkpoint.get("variant")

    if variant != "no_sem":
        raise ValueError(
            f"Expected no_sem checkpoint, got {variant!r}"
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

    total_params, trainable_params = (
        count_parameters(model)
    )

    sample_z = torch.randn(
        1,
        512,
        device=device,
    )

    sample_z = torch.nn.functional.normalize(
        sample_z,
        p=2,
        dim=1,
    )

    sample_quality = torch.randn(
        1,
        5,
        device=device,
    )

    macs_per_image = count_linear_macs(
        model,
        sample_z,
        sample_quality,
    )

    profile_rows = []

    for batch_size in args.batch_sizes:
        print()
        print(
            f"Profiling batch_size={batch_size} ..."
        )

        row = benchmark_latency(
            model=model,
            batch_size=batch_size,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        profile_rows.append(row)

        print(
            f"latency={row['latency_ms_mean']:.6f}"
            f" ± {row['latency_ms_std']:.6f} ms"
        )

        print(
            f"per-image={row['latency_ms_per_image']:.6f} ms"
        )

        print(
            f"throughput="
            f"{row['throughput_images_per_second']:.2f} img/s"
        )

        if row["peak_memory_mb"] is not None:
            print(
                f"peak_memory="
                f"{row['peak_memory_mb']:.3f} MB"
            )

    result = {
        "experiment": "DCEC A3 adapter efficiency profile",
        "variant": "no_sem",
        "architecture": architecture,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "checkpoint": str(args.checkpoint),
        "checkpoint_size_mb": checkpoint_size_mb(
            args.checkpoint
        ),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameter_size_fp32_mb": (
            trainable_params * 4 / (1024 ** 2)
        ),
        "linear_macs_per_image": macs_per_image,
        "linear_mmacs_per_image": macs_per_image / 1e6,
        "warmup_iterations": args.warmup,
        "measurement_repeats": args.repeats,
        "profiles": profile_rows,
        "scope": (
            "adapter-only; excludes frozen CLIP encoding "
            "and PIL quality-descriptor extraction"
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
    print("DCEC A3 EFFICIENCY PROFILE")
    print("=" * 80)
    print("device:", result["gpu_name"])
    print(
        "trainable_parameters:",
        result["trainable_parameters"],
    )
    print(
        "checkpoint_size_mb:",
        f"{result['checkpoint_size_mb']:.4f}",
    )
    print(
        "parameter_size_fp32_mb:",
        f"{result['parameter_size_fp32_mb']:.4f}",
    )
    print(
        "linear_macs_per_image:",
        result["linear_macs_per_image"],
    )
    print(
        "linear_mmacs_per_image:",
        f"{result['linear_mmacs_per_image']:.4f}",
    )
    print()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
