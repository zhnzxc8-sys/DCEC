"""Final frozen Flickr30K test evaluation.

IMPORTANT:
- A3 architecture/checkpoints are frozen before this script is used.
- No training, finetuning, or hyperparameter tuning is performed.
- Test results must not be used to modify the method.
"""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import torch
from transformers import CLIPModel, CLIPProcessor

from dcec_ablation_model import DCECAblation, ablation_spec
from dcec_quality import extract_quality_descriptor
from eval_dcec_validation import (
    _frozen_image_embeddings,
    _frozen_text_embeddings,
    freeze_clip,
)
from flickr30k_dataset import (
    load_flickr30k_split,
    prepare_retrieval_samples,
)
from image_degradations import apply_degradation, get_degradation_metadata
from retrieval_metrics import compute_retrieval_metrics


MODEL_NAME = "openai/clip-vit-base-patch32"

TEST_FILE = Path("/root/autodl-tmp/flickr30k/test.parquet")

CHECKPOINTS = {
    "2026": Path("results/ablation_formal/no_sem/best_dcec_ablation.pt"),
    "2027": Path("results/multiseed/a3_no_sem_seed2027/best_dcec_ablation.pt"),
    "2028": Path("results/multiseed/a3_no_sem_seed2028/best_dcec_ablation.pt"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FINAL frozen A3 evaluation on Flickr30K Karpathy test."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/flickr30k_final_test_protocol.json"),
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
        "--output-dir",
        type=Path,
        default=Path("results/flickr30k_final_test"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only after technical interruption; never for tuning.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit dataset/mappings only. Computes NO retrieval metrics.",
    )
    return parser.parse_args()


def load_a3(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    variant = checkpoint.get("variant")

    if variant != "no_sem":
        raise ValueError(
            f"Expected no_sem checkpoint, got {variant!r}: {checkpoint_path}"
        )

    architecture = str(ablation_spec("no_sem")["architecture"])

    model = DCECAblation(architecture).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def primary(metrics):
    return (
        float(metrics["i2t_R@1"])
        + float(metrics["t2i_R@1"])
    ) / 2.0


def encode_all_texts(
    clip_model,
    processor,
    captions,
    device,
    batch_size,
):
    chunks = []

    for start in range(0, len(captions), batch_size):
        stop = min(start + batch_size, len(captions))

        chunk = _frozen_text_embeddings(
            clip_model,
            processor,
            captions[start:stop],
            device,
        )

        chunks.append(chunk.cpu())

    return torch.cat(chunks, dim=0)


def evaluate_condition(
    images,
    text_features,
    image_to_text,
    text_to_image,
    degradation,
    severity,
    clip_model,
    processor,
    a3_models,
    device,
    batch_size,
):
    baseline_chunks = []
    corrected_chunks = {
        seed: [] for seed in a3_models
    }

    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))

        degraded_images = []

        for image_index in range(start, stop):
            degraded = apply_degradation(
                images[image_index],
                degradation=degradation,
                severity=severity,
                seed=2026,
                image_index=image_index,
            )
            degraded_images.append(degraded)

        z_degraded = _frozen_image_embeddings(
            clip_model,
            processor,
            degraded_images,
            device,
        )

        baseline_chunks.append(z_degraded.cpu())

        quality = torch.stack(
            [
                extract_quality_descriptor(image)
                for image in degraded_images
            ]
        ).to(device)

        z_device = z_degraded.to(device)

        with torch.no_grad():
            for seed, model in a3_models.items():
                outputs = model(z_device, quality)

                corrected_chunks[seed].append(
                    outputs["z_corrected"].cpu()
                )

        if start == 0 or stop == len(images) or stop % 256 == 0:
            print(f"images processed: {stop}/{len(images)}")

    baseline_image_features = torch.cat(
        baseline_chunks,
        dim=0,
    )

    baseline_metrics = compute_retrieval_metrics(
        baseline_image_features,
        text_features,
        image_to_text,
        text_to_image,
    )

    seed_results = {}

    for seed, chunks in corrected_chunks.items():
        corrected_features = torch.cat(chunks, dim=0)

        metrics = compute_retrieval_metrics(
            corrected_features,
            text_features,
            image_to_text,
            text_to_image,
        )

        seed_results[seed] = {
            **metrics,
            "primary_component": primary(metrics),
        }

    return baseline_metrics, seed_results


def main():
    args = parse_args()

    if not args.protocol.exists():
        raise FileNotFoundError(args.protocol)

    if not TEST_FILE.exists():
        raise FileNotFoundError(TEST_FILE)

    protocol = json.loads(
        args.protocol.read_text(encoding="utf-8")
    )

    if protocol.get("split") != "test":
        raise ValueError("Final protocol must explicitly specify split=test.")

    if protocol.get("post_test_method_changes") is not False:
        raise ValueError(
            "Protocol must forbid post-test method changes."
        )

    conditions = [
        (str(item[0]), int(item[1]))
        for item in protocol["conditions"]
    ]

    if len(conditions) != 22:
        raise ValueError(
            f"Expected exactly 22 frozen conditions, got {len(conditions)}"
        )

    print("=" * 84)
    print("FLICKR30K FINAL FROZEN TEST")
    print("=" * 84)
    print("TEST FILE:", TEST_FILE)
    print("No training / finetuning / hyperparameter tuning")
    print("No post-test method changes permitted")
    print("Frozen conditions:", len(conditions))

    dataset = load_flickr30k_split(
        split="test",
        local_parquet=TEST_FILE,
        cache_dir=args.cache_dir,
        streaming=False,
    )

    if len(dataset) != 1000:
        raise ValueError(
            f"Expected exactly 1000 test images, got {len(dataset)}"
        )

    (
        images,
        captions,
        image_to_text,
        text_to_image,
    ) = prepare_retrieval_samples(
        dataset,
        len(dataset),
    )

    print()
    print("===== TEST DATA AUDIT =====")
    print("images:", len(images))
    print("captions:", len(captions))
    print("image_to_text:", len(image_to_text))
    print("text_to_image:", len(text_to_image))

    if len(images) != 1000:
        raise AssertionError("Expected 1000 images.")

    if len(captions) != 5000:
        raise AssertionError("Expected 5000 captions.")

    if len(image_to_text) != 1000:
        raise AssertionError("Expected 1000 image_to_text entries.")

    if len(text_to_image) != 5000:
        raise AssertionError("Expected 5000 text_to_image entries.")

    if any(len(indices) != 5 for indices in image_to_text):
        raise AssertionError(
            "Every Flickr30K test image must have exactly 5 captions."
        )

    print("Flickr30K FINAL TEST DATA MAPPING: OK")

    if args.dry_run:
        print()
        print("=" * 84)
        print("DRY RUN COMPLETE")
        print("NO RETRIEVAL METRICS WERE COMPUTED")
        print("TEST RESULT REMAINS UNOBSERVED BY THIS COMMAND")
        print("=" * 84)
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("device:", device)

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

    print()
    print("Encoding 5000 Flickr30K test captions once...")

    text_features = encode_all_texts(
        clip_model,
        processor,
        captions,
        device,
        max(64, args.batch_size * 4),
    )

    print("text_features:", tuple(text_features.shape))

    a3_models = {}

    for seed, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        print(
            f"Loading frozen A3 seed {seed}: "
            f"{checkpoint_path}"
        )

        a3_models[seed] = load_a3(
            checkpoint_path,
            device,
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for condition_index, (degradation, severity) in enumerate(
        conditions,
        1,
    ):
        condition_name = f"{degradation}_s{severity}"
        output_path = args.output_dir / f"{condition_name}.json"

        if args.resume and output_path.exists():
            print()
            print("SKIP completed:", condition_name)

            saved = json.loads(
                output_path.read_text(encoding="utf-8")
            )

            all_results.append(saved)
            continue

        print()
        print("=" * 84)
        print(
            f"[{condition_index}/{len(conditions)}] "
            f"{degradation} S{severity}"
        )
        print("=" * 84)

        metadata = get_degradation_metadata(
            degradation,
            severity,
        )

        baseline_metrics, seed_results = evaluate_condition(
            images=images,
            text_features=text_features,
            image_to_text=image_to_text,
            text_to_image=text_to_image,
            degradation=degradation,
            severity=severity,
            clip_model=clip_model,
            processor=processor,
            a3_models=a3_models,
            device=device,
            batch_size=args.batch_size,
        )

        seed_primary = [
            seed_results[seed]["primary_component"]
            for seed in sorted(seed_results)
        ]

        baseline_primary = primary(
            baseline_metrics
        )

        a3_primary_mean = mean(
            seed_primary
        )

        a3_primary_std = pstdev(
            seed_primary
        )

        result = {
            "experiment": "final frozen Flickr30K test",
            "dataset": "Flickr30K Karpathy",
            "split": "test",
            "number_of_images": len(images),
            "number_of_captions": len(captions),
            "degradation": metadata["degradation"],
            "severity": metadata["severity"],
            "degradation_parameter_name": metadata["parameter_name"],
            "degradation_parameter_value": metadata["parameter_value"],
            "baseline": {
                **baseline_metrics,
                "primary_component": baseline_primary,
            },
            "a3_seeds": seed_results,
            "a3_primary_mean": a3_primary_mean,
            "a3_primary_std": a3_primary_std,
            "delta_primary_mean": (
                a3_primary_mean
                - baseline_primary
            ),
        }

        output_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

        all_results.append(result)

        print(
            "Frozen primary:",
            f"{baseline_primary:.4f}",
        )

        print(
            "A3 mean±std:",
            f"{a3_primary_mean:.4f}",
            "±",
            f"{a3_primary_std:.4f}",
        )

        print(
            "Delta:",
            f"{result['delta_primary_mean']:+.4f}",
        )

        print(
            "Saved:",
            output_path,
        )

    baseline_mean = mean(
        result["baseline"]["primary_component"]
        for result in all_results
    )

    per_seed_condition_means = {}

    for seed in sorted(CHECKPOINTS):
        per_seed_condition_means[seed] = mean(
            result["a3_seeds"][seed]["primary_component"]
            for result in all_results
        )

    a3_mean = mean(
        per_seed_condition_means.values()
    )

    a3_std = pstdev(
        per_seed_condition_means.values()
    )

    wins = sum(
        result["delta_primary_mean"] > 1e-8
        for result in all_results
    )

    ties = sum(
        abs(result["delta_primary_mean"]) <= 1e-8
        for result in all_results
    )

    losses = sum(
        result["delta_primary_mean"] < -1e-8
        for result in all_results
    )

    seen_names = {
        "jpeg",
        "gaussian_blur",
        "gaussian_noise",
        "low_light",
    }

    unseen_names = {
        "motion_blur",
        "contrast",
        "pixelate",
    }

    seen_results = [
        result
        for result in all_results
        if result["degradation"] in seen_names
    ]

    unseen_results = [
        result
        for result in all_results
        if result["degradation"] in unseen_names
    ]

    clean_result = next(
        result
        for result in all_results
        if result["degradation"] == "clean"
        and result["severity"] == 0
    )

    def group_summary(results):
        if not results:
            return None

        frozen = mean(
            result["baseline"]["primary_component"]
            for result in results
        )

        a3 = mean(
            result["a3_primary_mean"]
            for result in results
        )

        return {
            "number_of_conditions": len(results),
            "frozen_mean": frozen,
            "a3_mean": a3,
            "delta": a3 - frozen,
            "wins": sum(
                result["delta_primary_mean"] > 1e-8
                for result in results
            ),
            "ties": sum(
                abs(result["delta_primary_mean"]) <= 1e-8
                for result in results
            ),
            "losses": sum(
                result["delta_primary_mean"] < -1e-8
                for result in results
            ),
        }

    summary = {
        "experiment": "final frozen Flickr30K test",
        "dataset": "Flickr30K Karpathy",
        "split": "test",
        "number_of_conditions": len(all_results),

        "overall": {
            "frozen_condition_mean": baseline_mean,
            "a3_per_seed_condition_means": per_seed_condition_means,
            "a3_condition_mean": a3_mean,
            "a3_condition_std_across_seeds": a3_std,
            "delta_condition_mean": a3_mean - baseline_mean,
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },

        "clean": {
            "frozen_primary": clean_result["baseline"]["primary_component"],
            "a3_primary_mean": clean_result["a3_primary_mean"],
            "a3_primary_std": clean_result["a3_primary_std"],
            "delta": clean_result["delta_primary_mean"],
        },

        "seen_degradations": group_summary(
            seen_results
        ),

        "unseen_degradations": group_summary(
            unseen_results
        ),

        "post_test_method_changes_permitted": False,
    }

    summary_path = args.output_dir / "summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    completed_path = args.output_dir / "FINAL_TEST_COMPLETED"

    completed_path.write_text(
        "Final frozen Flickr30K test evaluation completed.\n"
        "No post-test method changes permitted.\n",
        encoding="utf-8",
    )

    print()
    print("=" * 84)
    print("FINAL FLICKR30K TEST SUMMARY")
    print("=" * 84)

    print(
        "Frozen 22-condition mean:",
        f"{baseline_mean:.4f}",
    )

    print(
        "A3 22-condition mean:",
        f"{a3_mean:.4f}",
        "±",
        f"{a3_std:.4f}",
    )

    print(
        "Delta:",
        f"{a3_mean - baseline_mean:+.4f}",
    )

    print(
        "W/T/L:",
        f"{wins}/{ties}/{losses}",
    )

    print()
    print("Clean:")
    print(
        f"  Frozen={summary['clean']['frozen_primary']:.4f}"
    )
    print(
        f"  A3={summary['clean']['a3_primary_mean']:.4f}"
        f" ± {summary['clean']['a3_primary_std']:.4f}"
    )
    print(
        f"  Delta={summary['clean']['delta']:+.4f}"
    )

    print()
    print("Seen degradations:")
    print(
        json.dumps(
            summary["seen_degradations"],
            indent=2,
        )
    )

    print()
    print("Unseen degradations:")
    print(
        json.dumps(
            summary["unseen_degradations"],
            indent=2,
        )
    )

    print()
    print("Saved:", summary_path)
    print("FINAL TEST COMPLETED")
    print("A3 IS NOW PERMANENTLY FROZEN")


if __name__ == "__main__":
    main()
