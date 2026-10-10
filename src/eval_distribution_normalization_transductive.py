"""Controlled Flickr30K evaluation of Distribution Normalization (DN).

DN protocol:
1. Encode CLIP image/text representations and L2-normalize them.
2. Estimate modality means from fixed unlabeled samples.
3. Subtract lambda * modality mean.
4. Compute retrieval by direct dot product WITHOUT re-normalizing
   the DN-corrected representations.

The frozen path is checked against the project's canonical retrieval evaluator.
"""

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Sequence

import torch
from transformers import CLIPModel, CLIPProcessor

from flickr30k_dataset import (
    DATASET_NAME,
    SPLITS,
    load_flickr30k_split,
    prepare_retrieval_samples,
)
from image_degradations import apply_degradation, get_degradation_metadata
from retrieval_metrics import compute_retrieval_metrics
from run_degradation_baseline import MODEL_NAME, _encode_images, _encode_texts


LAMBDA = 0.25
KS = (1, 5, 10)
DEFAULT_STATS_SEEDS = (9501, 9502, 9503, 9504, 9505)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _recall_at_k(
    similarity: torch.Tensor,
    correct_indices: Sequence[Sequence[int]],
    k: int,
) -> float:
    rankings = similarity.argsort(dim=1, descending=True)
    hits = 0
    for query_index, positives in enumerate(correct_indices):
        top_k = set(rankings[query_index, :k].tolist())
        hits += bool(top_k.intersection(positives))
    return 100.0 * hits / len(correct_indices)


def _metrics_from_similarity(
    similarity: torch.Tensor,
    image_to_text: Sequence[Sequence[int]],
    text_to_image: Sequence[int],
) -> dict[str, float]:
    text_to_image_groups = [[image_index] for image_index in text_to_image]

    metrics = {
        f"i2t_R@{k}": _recall_at_k(similarity, image_to_text, k)
        for k in KS
    }
    metrics.update(
        {
            f"t2i_R@{k}": _recall_at_k(
                similarity.T,
                text_to_image_groups,
                k,
            )
            for k in KS
        }
    )
    return metrics


def _primary(metrics: dict[str, float]) -> float:
    return 0.5 * (metrics["i2t_R@1"] + metrics["t2i_R@1"])


def _check_frozen_parity(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_to_text: Sequence[Sequence[int]],
    text_to_image: Sequence[int],
) -> tuple[dict[str, float], dict[str, float]]:
    canonical = compute_retrieval_metrics(
        image_features,
        text_features,
        image_to_text,
        text_to_image,
    )

    direct_similarity = image_features @ text_features.T
    direct = _metrics_from_similarity(
        direct_similarity,
        image_to_text,
        text_to_image,
    )

    for key in canonical:
        if abs(float(canonical[key]) - float(direct[key])) > 1e-12:
            raise RuntimeError(
                f"Frozen parity failure for {key}: "
                f"canonical={canonical[key]} direct={direct[key]}"
            )

    return canonical, direct


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty list.")

    return {
        "mean": float(statistics.mean(values)),
        "std_population": float(statistics.pstdev(values)),
        "std_sample": (
            float(statistics.stdev(values))
            if len(values) > 1
            else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dataset-id", default=DATASET_NAME)
    parser.add_argument("--local-parquet", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=sorted(SPLITS),
        default="validation",
    )

    parser.add_argument(
        "--degradation",
        required=True,
        choices=(
            "clean",
            "jpeg",
            "gaussian_blur",
            "gaussian_noise",
            "low_light",
            "motion_blur",
            "contrast",
            "pixelate",
        ),
    )
    parser.add_argument(
        "--severity",
        type=int,
        required=True,
        choices=(0, 1, 2, 3),
    )

    parser.add_argument("--degradation-seed", type=int, default=2026)
    parser.add_argument("--stats-samples", type=_positive_int, default=100)
    parser.add_argument(
        "--stats-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_STATS_SEEDS),
    )

    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata = get_degradation_metadata(
        args.degradation,
        args.severity,
    )

    if args.degradation == "clean" and args.severity != 0:
        raise ValueError("clean requires severity=0")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = load_flickr30k_split(
        split=args.split,
        dataset_id=args.dataset_id,
        local_parquet=args.local_parquet,
        cache_dir=args.cache_dir,
        streaming=False,
    )

    sample_count = (
        args.limit
        if args.limit is not None
        else len(dataset)
    )

    images, captions, image_to_text, text_to_image = (
        prepare_retrieval_samples(
            dataset,
            sample_count,
        )
    )

    if args.stats_samples > len(images):
        raise ValueError(
            f"stats_samples={args.stats_samples} exceeds "
            f"number of images={len(images)}"
        )

    if args.stats_samples > len(captions):
        raise ValueError(
            f"stats_samples={args.stats_samples} exceeds "
            f"number of captions={len(captions)}"
        )

    degraded_images = [
        apply_degradation(
            image,
            degradation=args.degradation,
            severity=args.severity,
            seed=args.degradation_seed,
            image_index=image_index,
        )
        for image_index, image in enumerate(images)
    ]

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=args.cache_dir,
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME,
        revision="d15b5f29721ca72dac15f8526b284be910de18be",
        cache_dir=args.cache_dir,
        use_safetensors=True,
    ).to(device)
    model.eval()

    image_features = _encode_images(
        model,
        processor,
        degraded_images,
        device,
        args.batch_size,
    )

    text_features = _encode_texts(
        model,
        processor,
        captions,
        device,
        args.batch_size,
    )

    if image_features.shape[-1] != text_features.shape[-1]:
        raise ValueError(
            "Image/text embedding dimensions do not match."
        )

    frozen_metrics, frozen_direct_metrics = (
        _check_frozen_parity(
            image_features,
            text_features,
            image_to_text,
            text_to_image,
        )
    )

    frozen_primary = _primary(frozen_metrics)

    print("===== FROZEN PARITY PASS =====")
    print(
        json.dumps(
            frozen_metrics,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Frozen primary: {frozen_primary:.6f}")

    dn_runs = []

    for stats_seed in args.stats_seeds:
        rng = random.Random(stats_seed)

        image_indices = rng.sample(
            range(len(image_features)),
            args.stats_samples,
        )

        text_indices = rng.sample(
            range(len(text_features)),
            args.stats_samples,
        )

        image_mean = image_features[image_indices].mean(dim=0)
        text_mean = text_features[text_indices].mean(dim=0)

        dn_image_features = (
            image_features - LAMBDA * image_mean
        )

        dn_text_features = (
            text_features - LAMBDA * text_mean
        )

        # IMPORTANT:
        # Do NOT call compute_retrieval_metrics here.
        # That function L2-normalizes its inputs.
        # Official DN uses direct dot products after mean subtraction.
        similarity = (
            dn_image_features @ dn_text_features.T
        )

        metrics = _metrics_from_similarity(
            similarity,
            image_to_text,
            text_to_image,
        )

        primary = _primary(metrics)

        run = {
            "stats_seed": stats_seed,
            **metrics,
            "primary": primary,
            "delta_primary_vs_frozen": (
                primary - frozen_primary
            ),
        }
        dn_runs.append(run)

        print()
        print(
            f"===== DN stats_seed={stats_seed} ====="
        )
        print(
            json.dumps(
                run,
                ensure_ascii=False,
                indent=2,
            )
        )

    metric_names = [
        "i2t_R@1",
        "i2t_R@5",
        "i2t_R@10",
        "t2i_R@1",
        "t2i_R@5",
        "t2i_R@10",
        "primary",
        "delta_primary_vs_frozen",
    ]

    dn_summary = {
        name: _summary(
            [float(run[name]) for run in dn_runs]
        )
        for name in metric_names
    }

    result = {
        "method": "Distribution Normalization",
        "protocol": "controlled_dcec_validation",
        "model": MODEL_NAME,
        "dataset": args.dataset_id,
        "dataset_split": args.split,
        "local_parquet": str(args.local_parquet),
        "number_of_images": len(images),
        "number_of_captions": len(captions),
        "device": str(device),

        "degradation": metadata["degradation"],
        "severity": metadata["severity"],
        "degradation_parameter_name": metadata["parameter_name"],
        "degradation_parameter_value": metadata["parameter_value"],
        "degradation_seed": args.degradation_seed,

        "dn_lambda": LAMBDA,
        "stats_samples_image": args.stats_samples,
        "stats_samples_text": args.stats_samples,
        "stats_seeds": args.stats_seeds,
        "statistics_source": (
            "unlabeled samples from the same "
            "validation condition"
        ),
        "dn_renormalize_after_correction": False,

        "frozen_parity_pass": True,
        "frozen_metrics": frozen_metrics,
        "frozen_direct_metrics": frozen_direct_metrics,
        "frozen_primary": frozen_primary,

        "dn_runs": dn_runs,
        "dn_summary": dn_summary,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("===== DN SUMMARY =====")
    print(
        json.dumps(
            dn_summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print()
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
