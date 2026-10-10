"""Distribution Normalization baseline under the controlled DCEC protocol.

Formal protocol:
- Evaluation: full Flickr30K validation split.
- DN statistics: disjoint Flickr30K training split.
- 100 unlabeled image samples + 100 unlabeled text samples per estimate.
- Five fixed statistics seeds.
- CLIP ViT-B/32.
- DN lambda = 0.25.
- Features are L2-normalized BEFORE mean estimation.
- DN-corrected features are NOT re-normalized.
- Retrieval uses direct dot-product ranking after DN correction.
"""

import argparse
import glob
import json
import random
import statistics
from pathlib import Path
from typing import Sequence

import torch
from datasets import load_dataset
from transformers import CLIPModel, CLIPProcessor

from flickr30k_dataset import (
    DATASET_NAME,
    SPLITS,
    load_flickr30k_split,
    prepare_retrieval_samples,
)
from image_degradations import apply_degradation, get_degradation_metadata
from retrieval_metrics import compute_retrieval_metrics
from run_degradation_baseline import (
    MODEL_NAME,
    _encode_images,
    _encode_texts,
)


MODEL_REVISION = "d15b5f29721ca72dac15f8526b284be910de18be"
LAMBDA = 0.25
KS = (1, 5, 10)
DEFAULT_STATS_SEEDS = (9501, 9502, 9503, 9504, 9505)


def _positive_int(value: str) -> int:
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


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
    t2i_groups = [[image_index] for image_index in text_to_image]

    metrics = {
        f"i2t_R@{k}": _recall_at_k(
            similarity,
            image_to_text,
            k,
        )
        for k in KS
    }

    metrics.update(
        {
            f"t2i_R@{k}": _recall_at_k(
                similarity.T,
                t2i_groups,
                k,
            )
            for k in KS
        }
    )

    return metrics


def _primary(metrics: dict[str, float]) -> float:
    return 0.5 * (
        metrics["i2t_R@1"] + metrics["t2i_R@1"]
    )


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "std_population": float(statistics.pstdev(values)),
        "std_sample": (
            float(statistics.stdev(values))
            if len(values) > 1
            else 0.0
        ),
    }


def _check_frozen_parity(
    image_features,
    text_features,
    image_to_text,
    text_to_image,
):
    canonical = compute_retrieval_metrics(
        image_features,
        text_features,
        image_to_text,
        text_to_image,
    )

    direct = _metrics_from_similarity(
        image_features @ text_features.T,
        image_to_text,
        text_to_image,
    )

    for key in canonical:
        if abs(
            float(canonical[key]) - float(direct[key])
        ) > 1e-12:
            raise RuntimeError(
                f"Frozen parity failure: {key} "
                f"{canonical[key]} vs {direct[key]}"
            )

    return canonical


def _load_stats_dataset(
    pattern: str,
    cache_dir: Path | None,
):
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No training parquet files match: {pattern}"
        )

    print("===== DN STATS FILES =====")
    for file in files:
        print(file)

    dataset = load_dataset(
        "parquet",
        data_files={"train": files},
        split="train",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    return dataset, files


def _sample_statistics_data(
    dataset,
    stats_samples: int,
    stats_seed: int,
    degradation: str,
    severity: int,
    degradation_seed: int,
):
    if stats_samples > len(dataset):
        raise ValueError("stats_samples exceeds train images")

    rng = random.Random(stats_seed)

    image_indices = rng.sample(
        range(len(dataset)),
        stats_samples,
    )

    total_captions = len(dataset) * 5

    text_flat_indices = rng.sample(
        range(total_captions),
        stats_samples,
    )

    image_view = dataset.select_columns(["image"])
    caption_view = dataset.select_columns(["captions"])

    stats_images = []

    for image_index in image_indices:
        image = image_view[image_index]["image"]

        degraded = apply_degradation(
            image,
            degradation=degradation,
            severity=severity,
            seed=degradation_seed,
            image_index=image_index,
        )

        stats_images.append(degraded)

    stats_captions = []

    for flat_index in text_flat_indices:
        image_index, caption_index = divmod(
            flat_index,
            5,
        )

        captions = caption_view[image_index]["captions"]

        if len(captions) != 5:
            raise ValueError(
                "Statistics source must have five captions per image."
            )

        stats_captions.append(
            captions[caption_index]
        )

    return (
        stats_images,
        stats_captions,
        image_indices,
        text_flat_indices,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--local-parquet",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--stats-parquet-glob",
        required=True,
    )

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

    parser.add_argument(
        "--degradation-seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--stats-samples",
        type=_positive_int,
        default=100,
    )

    parser.add_argument(
        "--stats-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_STATS_SEEDS),
    )

    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=64,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if (
        args.degradation == "clean"
        and args.severity != 0
    ):
        raise ValueError("clean requires severity=0")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    metadata = get_degradation_metadata(
        args.degradation,
        args.severity,
    )

    eval_dataset = load_flickr30k_split(
        split=args.split,
        dataset_id=DATASET_NAME,
        local_parquet=args.local_parquet,
        cache_dir=args.cache_dir,
        streaming=False,
    )

    images, captions, image_to_text, text_to_image = (
        prepare_retrieval_samples(
            eval_dataset,
            len(eval_dataset),
        )
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

    stats_dataset, stats_files = _load_stats_dataset(
        args.stats_parquet_glob,
        args.cache_dir,
    )

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=args.cache_dir,
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        cache_dir=args.cache_dir,
        use_safetensors=True,
    ).to(device)

    model.eval()

    print("===== ENCODING FULL VALIDATION =====")

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

    frozen_metrics = _check_frozen_parity(
        image_features,
        text_features,
        image_to_text,
        text_to_image,
    )

    frozen_primary = _primary(frozen_metrics)

    print("===== FROZEN PARITY PASS =====")
    print(json.dumps(frozen_metrics, indent=2))
    print(f"Frozen primary: {frozen_primary:.6f}")

    runs = []

    for stats_seed in args.stats_seeds:
        print()
        print(
            f"===== DN DISJOINT stats_seed={stats_seed} ====="
        )

        (
            stats_images,
            stats_captions,
            image_indices,
            text_flat_indices,
        ) = _sample_statistics_data(
            stats_dataset,
            args.stats_samples,
            stats_seed,
            args.degradation,
            args.severity,
            args.degradation_seed,
        )

        stats_image_features = _encode_images(
            model,
            processor,
            stats_images,
            device,
            args.batch_size,
        )

        stats_text_features = _encode_texts(
            model,
            processor,
            stats_captions,
            device,
            args.batch_size,
        )

        image_mean = stats_image_features.mean(dim=0)
        text_mean = stats_text_features.mean(dim=0)

        dn_image_features = (
            image_features - LAMBDA * image_mean
        )

        dn_text_features = (
            text_features - LAMBDA * text_mean
        )

        # Official DN operation:
        # direct dot product after mean subtraction.
        # No post-correction re-normalization.
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
            "stats_image_indices": image_indices,
            "stats_text_flat_indices": text_flat_indices,
        }

        runs.append(run)

        print(
            json.dumps(
                {
                    k: v
                    for k, v in run.items()
                    if not k.endswith("_indices")
                },
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

    summary = {
        metric: _summary(
            [float(run[metric]) for run in runs]
        )
        for metric in metric_names
    }

    result = {
        "method": "Distribution Normalization",
        "protocol": "disjoint_train_statistics_full_validation",
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "dn_lambda": LAMBDA,
        "dn_post_correction_renormalization": False,

        "evaluation_split": "validation",
        "evaluation_parquet": str(args.local_parquet),
        "number_of_eval_images": len(images),
        "number_of_eval_captions": len(captions),

        "statistics_split": "train",
        "statistics_files": stats_files,
        "statistics_samples_images": args.stats_samples,
        "statistics_samples_texts": args.stats_samples,
        "statistics_seeds": args.stats_seeds,

        "degradation": metadata["degradation"],
        "severity": metadata["severity"],
        "degradation_parameter_name": metadata["parameter_name"],
        "degradation_parameter_value": metadata["parameter_value"],
        "degradation_seed": args.degradation_seed,

        "frozen_parity_pass": True,
        "frozen_metrics": frozen_metrics,
        "frozen_primary": frozen_primary,

        "dn_runs": runs,
        "dn_summary": summary,

        "final_test_accessed": False,
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
    print("===== DN FORMAL SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print()
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
