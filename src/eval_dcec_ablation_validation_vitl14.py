"""Validation-only evaluator for DCEC controlled ablations."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from dcec_ablation_model_vitl14 import ABLATION_VARIANTS, DCECAblation, ablation_spec
from dcec_dataset import DCECDataset, dcec_collate
from dcec_quality import extract_quality_descriptor
from eval_dcec_validation import _frozen_image_embeddings, _frozen_text_embeddings, freeze_clip
from image_degradations import get_degradation_metadata
from retrieval_metrics import compute_retrieval_metrics


MODEL_NAME = "openai/clip-vit-large-patch14"
DEGRADATIONS = (
    "clean",
    "jpeg",
    "gaussian_blur",
    "gaussian_noise",
    "low_light",
    "motion_blur",
    "contrast",
    "pixelate",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def evaluate_ablation_validation_condition(
    variant: str,
    ablation_model: DCECAblation,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    val_file: str | Path,
    degradation: str,
    severity: int,
    seed: int,
    batch_size: int,
    cache_dir: str | Path | None = None,
    limit: int | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate a single fixed condition on validation data only."""
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unsupported ablation variant {variant!r}.")
    metadata = get_degradation_metadata(degradation, severity)
    if device is None:
        device = next(ablation_model.parameters()).device
    validation_dataset = DCECDataset(
        split="validation",
        path=val_file,
        seed=seed,
        cache_dir=cache_dir,
        degradation=degradation,
        severity=severity,
        limit=limit,
    )
    loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dcec_collate,
    )
    ablation_model.eval()
    image_features: list[torch.Tensor] = []
    text_features: list[torch.Tensor] = []
    image_to_text: list[list[int]] = []
    text_to_image: list[int] = []
    image_offset = 0
    text_offset = 0

    with torch.no_grad():
        for batch in loader:
            caption_groups = batch["captions"]
            if any(len(captions) != 5 for captions in caption_groups):
                raise ValueError("Validation requires exactly five captions per image.")
            captions = [caption for captions in caption_groups for caption in captions]
            z_degraded = _frozen_image_embeddings(
                clip_model, processor, batch["degraded_image"], device
            )
            batch_text_features = _frozen_text_embeddings(
                clip_model, processor, captions, device
            )
            quality = None
            if ablation_model.variant != "residual_only":
                quality = torch.stack(
                    [extract_quality_descriptor(image) for image in batch["degraded_image"]]
                ).to(device)
            outputs = ablation_model(z_degraded, quality)
            image_features.append(outputs["z_corrected"].cpu())
            text_features.append(batch_text_features.cpu())

            for local_index in range(len(batch["degraded_image"])):
                image_to_text.append(list(range(text_offset, text_offset + 5)))
                text_to_image.extend([image_offset + local_index] * 5)
                text_offset += 5
            image_offset += len(batch["degraded_image"])

    metrics = compute_retrieval_metrics(
        torch.cat(image_features, dim=0),
        torch.cat(text_features, dim=0),
        image_to_text,
        text_to_image,
    )
    return {
        "variant": variant,
        "dataset_split": "validation",
        "number_of_images": image_offset,
        "number_of_captions": text_offset,
        "degradation": metadata["degradation"],
        "severity": metadata["severity"],
        "seed": seed,
        "device": str(device),
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=ABLATION_VARIANTS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--degradation", choices=DEGRADATIONS, required=True)
    parser.add_argument("--severity", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("variant") != args.variant:
        raise ValueError("Checkpoint variant does not match --variant.")
    architecture = str(ablation_spec(args.variant)["architecture"])
    ablation_model = DCECAblation(architecture).to(device)
    ablation_model.load_state_dict(checkpoint["model_state_dict"])
    clip_model = CLIPModel.from_pretrained(
        MODEL_NAME, cache_dir=args.cache_dir, use_safetensors=True
    ).to(device)
    freeze_clip(clip_model)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir)

    result = evaluate_ablation_validation_condition(
        variant=args.variant,
        ablation_model=ablation_model,
        clip_model=clip_model,
        processor=processor,
        val_file=args.val_file,
        degradation=args.degradation,
        severity=args.severity,
        seed=args.seed,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
        limit=args.limit,
        device=device,
    )
    result["checkpoint"] = str(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
