"""Validation-only evaluation for a trained DCEC checkpoint."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from dcec_dataset import DCECDataset, dcec_collate
from dcec_model import DCEC
from dcec_quality import extract_quality_descriptor
from flickr30k_dataset import DATASET_NAME
from image_degradations import get_degradation_metadata
from retrieval_metrics import compute_retrieval_metrics


MODEL_NAME = "openai/clip-vit-base-patch32"
DEGRADATIONS = ("clean", "jpeg", "gaussian_blur", "gaussian_noise", "low_light")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def freeze_clip(model: CLIPModel) -> None:
    """Freeze every CLIP parameter for DCEC-only optimization/evaluation."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()


def _frozen_image_embeddings(
    model: CLIPModel, processor: CLIPProcessor, images: list[Any], device: torch.device
) -> torch.Tensor:
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(device)
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision_outputs.pooler_output
        if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
            raise ValueError("vision_model.pooler_output must be a 2D Tensor.")
        if pooled.shape[-1] != model.config.vision_config.hidden_size:
            raise ValueError("Unexpected frozen CLIP vision hidden size.")
        features = model.visual_projection(pooled)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("visual_projection must return a 2D Tensor.")
        if features.shape[-1] != model.config.projection_dim:
            raise ValueError("Unexpected frozen CLIP image projection size.")
        return functional.normalize(features, p=2, dim=1)


def _frozen_text_embeddings(
    model: CLIPModel, processor: CLIPProcessor, captions: list[str], device: torch.device
) -> torch.Tensor:
    with torch.no_grad():
        inputs = processor(
            text=captions, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        pooled = text_outputs.pooler_output
        if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
            raise ValueError("text_model.pooler_output must be a 2D Tensor.")
        if pooled.shape[-1] != model.config.text_config.hidden_size:
            raise ValueError("Unexpected frozen CLIP text hidden size.")
        features = model.text_projection(pooled)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("text_projection must return a 2D Tensor.")
        if features.shape[-1] != model.config.projection_dim:
            raise ValueError("Unexpected frozen CLIP text projection size.")
        return functional.normalize(features, p=2, dim=1)


def evaluate_validation_condition(
    dcec: DCEC,
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
    """Evaluate one fixed condition on validation only and return Recall@K metrics."""
    metadata = get_degradation_metadata(degradation, severity)
    if device is None:
        device = next(dcec.parameters()).device

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
    dcec.eval()
    corrected_embeddings: list[torch.Tensor] = []
    text_embeddings: list[torch.Tensor] = []
    image_to_text: list[list[int]] = []
    text_to_image: list[int] = []
    image_offset = 0
    text_offset = 0

    with torch.no_grad():
        for batch in loader:
            captions_by_image = batch["captions"]
            if any(len(captions) != 5 for captions in captions_by_image):
                raise ValueError("Validation requires exactly five captions per image.")
            captions = [caption for group in captions_by_image for caption in group]
            z_degraded = _frozen_image_embeddings(
                clip_model, processor, batch["degraded_image"], device
            )
            text_features = _frozen_text_embeddings(clip_model, processor, captions, device)
            quality = torch.stack(
                [extract_quality_descriptor(image) for image in batch["degraded_image"]]
            ).to(device)
            outputs = dcec(z_degraded, quality)
            corrected_embeddings.append(outputs["z_corrected"].cpu())
            text_embeddings.append(text_features.cpu())

            for local_index in range(len(batch["degraded_image"])):
                indices = list(range(text_offset, text_offset + 5))
                image_to_text.append(indices)
                text_to_image.extend([image_offset + local_index] * 5)
                text_offset += 5
            image_offset += len(batch["degraded_image"])

    image_features = torch.cat(corrected_embeddings, dim=0)
    text_features = torch.cat(text_embeddings, dim=0)
    metrics = compute_retrieval_metrics(
        image_features, text_features, image_to_text, text_to_image
    )
    return {
        "dataset": DATASET_NAME,
        "dataset_split": "validation",
        "number_of_images": image_features.shape[0],
        "number_of_captions": text_features.shape[0],
        "degradation": metadata["degradation"],
        "severity": metadata["severity"],
        "seed": seed,
        "device": str(device),
        **metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--degradation", choices=DEGRADATIONS, required=True)
    parser.add_argument("--severity", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "dcec_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain dcec_state_dict.")

    dcec = DCEC().to(device)
    dcec.load_state_dict(checkpoint["dcec_state_dict"])
    clip_model = CLIPModel.from_pretrained(
        MODEL_NAME, cache_dir=args.cache_dir, use_safetensors=True
    ).to(device)
    freeze_clip(clip_model)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir)

    result = evaluate_validation_condition(
        dcec=dcec,
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
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
