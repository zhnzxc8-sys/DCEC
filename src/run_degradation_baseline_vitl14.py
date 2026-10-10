"""Evaluate frozen CLIP retrieval under deterministic image degradations."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from transformers import CLIPModel, CLIPProcessor

from flickr30k_dataset import (
    DATASET_NAME,
    SPLITS,
    load_flickr30k_split,
    prepare_retrieval_samples,
)
from image_degradations import apply_degradation, get_degradation_metadata
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
DEFAULT_OUTPUT = Path("results") / "degradation_baseline.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _normalize_embeddings(embeddings: torch.Tensor, name: str) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {embeddings.shape}.")
    if torch.any(embeddings.norm(dim=1) == 0):
        raise ValueError(f"{name} contains a zero-norm embedding.")
    return functional.normalize(embeddings, p=2, dim=1)


def _encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: list[Any],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    embeddings = []
    with torch.no_grad():
        for image_batch in _batched(images, batch_size):
            inputs = processor(images=image_batch, return_tensors="pt").to(device)
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled_image = vision_outputs.pooler_output
            if not isinstance(pooled_image, torch.Tensor) or pooled_image.ndim != 2:
                raise ValueError("vision_model.pooler_output must be a 2D Tensor.")
            if pooled_image.shape[-1] != model.config.vision_config.hidden_size:
                raise ValueError("Vision pooled dimension does not match model configuration.")
            image_embedding = model.visual_projection(pooled_image)
            if not isinstance(image_embedding, torch.Tensor) or image_embedding.ndim != 2:
                raise ValueError("visual_projection must return a 2D Tensor.")
            if image_embedding.shape[-1] != model.config.projection_dim:
                raise ValueError("Image embedding dimension does not match model configuration.")
            embeddings.append(image_embedding.cpu())
    return _normalize_embeddings(torch.cat(embeddings, dim=0), "image_features")


def _encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    captions: list[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    embeddings = []
    with torch.no_grad():
        for caption_batch in _batched(captions, batch_size):
            inputs = processor(
                text=caption_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            text_outputs = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            pooled_text = text_outputs.pooler_output
            if not isinstance(pooled_text, torch.Tensor) or pooled_text.ndim != 2:
                raise ValueError("text_model.pooler_output must be a 2D Tensor.")
            if pooled_text.shape[-1] != model.config.text_config.hidden_size:
                raise ValueError("Text pooled dimension does not match model configuration.")
            text_embedding = model.text_projection(pooled_text)
            if not isinstance(text_embedding, torch.Tensor) or text_embedding.ndim != 2:
                raise ValueError("text_projection must return a 2D Tensor.")
            if text_embedding.shape[-1] != model.config.projection_dim:
                raise ValueError("Text embedding dimension does not match model configuration.")
            embeddings.append(text_embedding.cpu())
    return _normalize_embeddings(torch.cat(embeddings, dim=0), "text_features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DATASET_NAME)
    parser.add_argument("--local-parquet", type=Path, default=None)
    parser.add_argument("--split", choices=sorted(SPLITS), default="test")
    parser.add_argument("--degradation", choices=DEGRADATIONS, required=True)
    parser.add_argument("--severity", type=int, required=True, choices=(0, 1, 2, 3))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = get_degradation_metadata(args.degradation, args.severity)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_flickr30k_split(
        split=args.split,
        dataset_id=args.dataset_id,
        local_parquet=args.local_parquet,
        cache_dir=args.cache_dir,
        streaming=False,
    )
    sample_count = args.limit if args.limit is not None else len(dataset)
    images, captions, image_to_text, text_to_image = prepare_retrieval_samples(
        dataset, sample_count
    )

    degraded_images = [
        apply_degradation(
            image,
            degradation=args.degradation,
            severity=args.severity,
            seed=args.seed,
            image_index=image_index,
        )
        for image_index, image in enumerate(images)
    ]

    processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir)
    model = CLIPModel.from_pretrained(
        MODEL_NAME,
        cache_dir=args.cache_dir,
        use_safetensors=True,
    ).to(device)
    model.eval()

    image_features = _encode_images(
        model, processor, degraded_images, device, args.batch_size
    )
    text_features = _encode_texts(model, processor, captions, device, args.batch_size)
    if image_features.shape[-1] != text_features.shape[-1]:
        raise ValueError("Image and text embedding dimensions must match.")
    metrics = compute_retrieval_metrics(
        image_features, text_features, image_to_text, text_to_image
    )

    result = {
        "model": MODEL_NAME,
        "dataset": args.dataset_id,
        "dataset_id": args.dataset_id,
        "local_parquet": str(args.local_parquet) if args.local_parquet else None,
        "split": args.split,
        "number_of_images": len(images),
        "number_of_captions": len(captions),
        "device": str(device),
        "run_type": (
            "degradation_smoke_test" if args.limit is not None else "full_degradation_evaluation"
        ),
        "degradation": metadata["degradation"],
        "severity": metadata["severity"],
        "degradation_parameter_name": metadata["parameter_name"],
        "degradation_parameter_value": metadata["parameter_value"],
        "seed": args.seed,
        "batch_size": args.batch_size,
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
