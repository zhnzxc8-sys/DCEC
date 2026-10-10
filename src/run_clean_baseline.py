"""Evaluate a clean, pretrained CLIP baseline on Flickr30K Karpathy splits."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import CLIPModel, CLIPProcessor

from flickr30k_dataset import (
    DATASET_NAME,
    SPLITS,
    load_flickr30k_split,
    prepare_retrieval_samples,
)
from retrieval_metrics import compute_retrieval_metrics


MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_OUTPUT = Path("results") / "clean_clip_baseline.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _batched(items: list[Any], batch_size: int):
    """Yield fixed-size lists so embeddings are extracted in manageable batches."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: list[Any],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    embeddings = []
    with torch.no_grad():  # Inference only: never build a training graph.
        for image_batch in _batched(images, batch_size):
            inputs = processor(images=image_batch, return_tensors="pt").to(device)
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled_image = vision_outputs.pooler_output
            if not isinstance(pooled_image, torch.Tensor):
                raise TypeError("vision_model.pooler_output must be a PyTorch Tensor.")
            assert pooled_image.ndim == 2, (
                f"vision pooled output must be 2D, got {pooled_image.shape}."
            )
            assert pooled_image.shape[-1] == model.config.vision_config.hidden_size, (
                "vision pooled dimension does not match vision_config.hidden_size: "
                f"{pooled_image.shape[-1]} != {model.config.vision_config.hidden_size}."
            )

            image_embedding = model.visual_projection(pooled_image)
            if not isinstance(image_embedding, torch.Tensor):
                raise TypeError("visual_projection must return a PyTorch Tensor.")
            assert image_embedding.ndim == 2, (
                f"image embedding must be 2D, got {image_embedding.shape}."
            )
            assert image_embedding.shape[-1] == model.config.projection_dim, (
                "image embedding dimension does not match config.projection_dim: "
                f"{image_embedding.shape[-1]} != {model.config.projection_dim}."
            )
            if not embeddings:
                print(f"vision pooled shape: {tuple(pooled_image.shape)}")
                print(f"image embedding shape: {tuple(image_embedding.shape)}")
            embeddings.append(image_embedding.cpu())
    image_features = torch.cat(embeddings, dim=0)
    assert image_features.ndim == 2, (
        f"image_features must be 2D, got {image_features.shape}."
    )
    return image_features


def _encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    captions: list[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    embeddings = []
    with torch.no_grad():  # Inference only: never build a training graph.
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
            if not isinstance(pooled_text, torch.Tensor):
                raise TypeError("text_model.pooler_output must be a PyTorch Tensor.")
            assert pooled_text.ndim == 2, (
                f"text pooled output must be 2D, got {pooled_text.shape}."
            )
            assert pooled_text.shape[-1] == model.config.text_config.hidden_size, (
                "text pooled dimension does not match text_config.hidden_size: "
                f"{pooled_text.shape[-1]} != {model.config.text_config.hidden_size}."
            )

            text_embedding = model.text_projection(pooled_text)
            if not isinstance(text_embedding, torch.Tensor):
                raise TypeError("text_projection must return a PyTorch Tensor.")
            assert text_embedding.ndim == 2, (
                f"text embedding must be 2D, got {text_embedding.shape}."
            )
            assert text_embedding.shape[-1] == model.config.projection_dim, (
                "text embedding dimension does not match config.projection_dim: "
                f"{text_embedding.shape[-1]} != {model.config.projection_dim}."
            )
            if not embeddings:
                print(f"text pooled shape: {tuple(pooled_text.shape)}")
                print(f"text embedding shape: {tuple(text_embedding.shape)}")
            embeddings.append(text_embedding.cpu())
    text_features = torch.cat(embeddings, dim=0)
    assert text_features.ndim == 2, (
        f"text_features must be 2D, got {text_features.shape}."
    )
    return text_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), default="test")
    parser.add_argument("--dataset-id", default=DATASET_NAME)
    parser.add_argument("--local-parquet", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream a limited local inspection instead of using the regular dataset mode.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Use only this many images for a local smoke test; default evaluates the full split.",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.streaming and args.limit is None:
        raise ValueError("--streaming requires --limit for a bounded local inspection.")

    dataset = load_flickr30k_split(
        split=args.split,
        dataset_id=args.dataset_id,
        local_parquet=args.local_parquet,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )
    sample_count = args.limit if args.limit is not None else len(dataset)
    images, captions, image_to_text, text_to_image = prepare_retrieval_samples(
        dataset, sample_count
    )

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(
    MODEL_NAME,
    cache_dir=args.cache_dir,
    use_safetensors=True,
).to(device)
    model.eval()

    image_features = _encode_images(model, processor, images, device, args.batch_size)
    text_features = _encode_texts(model, processor, captions, device, args.batch_size)
    assert image_features.ndim == 2, (
        f"image_features must be 2D, got {image_features.shape}."
    )
    assert text_features.ndim == 2, (
        f"text_features must be 2D, got {text_features.shape}."
    )
    assert image_features.shape[-1] == model.config.projection_dim, (
        "image_features dimension does not match config.projection_dim: "
        f"{image_features.shape[-1]} != {model.config.projection_dim}."
    )
    assert text_features.shape[-1] == model.config.projection_dim, (
        "text_features dimension does not match config.projection_dim: "
        f"{text_features.shape[-1]} != {model.config.projection_dim}."
    )
    assert image_features.shape[-1] == text_features.shape[-1], (
        "Image and text embedding dimensions must match: "
        f"{image_features.shape[-1]} != {text_features.shape[-1]}."
    )
    metrics = compute_retrieval_metrics(
        image_features, text_features, image_to_text, text_to_image
    )

    result = {
        "model": MODEL_NAME,
        "dataset": args.dataset_id,
        "dataset_id": args.dataset_id,
        "local_parquet": str(args.local_parquet) if args.local_parquet else None,
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "streaming": args.streaming,
        "split": args.split,
        "number_of_images": len(images),
        "number_of_captions": len(captions),
        "device": str(device),
        "run_type": "local_smoke_test" if args.limit is not None else "full_split_evaluation",
        "official_benchmark": False,
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
