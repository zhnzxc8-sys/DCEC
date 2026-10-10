"""Train DCEC v1 on local Flickr30K train data and select by validation only."""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as functional
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from dcec_dataset import DCECDataset, dcec_collate
from dcec_losses import compute_dcec_losses
from dcec_model import DCEC
from dcec_quality import extract_quality_descriptor
from eval_dcec_validation import evaluate_validation_condition, freeze_clip


MODEL_NAME = "openai/clip-vit-base-patch32"
VALIDATION_CONDITIONS = (
    ("clean", 0),
    ("jpeg", 2),
    ("gaussian_blur", 2),
    ("gaussian_noise", 2),
    ("low_light", 2),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _frozen_image_embeddings(
    model: CLIPModel, processor: CLIPProcessor, images: list[Any], device: torch.device
) -> torch.Tensor:
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(device)
        pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
            raise ValueError("vision_model.pooler_output must be a 2D Tensor.")
        features = model.visual_projection(pooled)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("visual_projection must return a 2D Tensor.")
        return functional.normalize(features, p=2, dim=1)


def _frozen_text_embeddings(
    model: CLIPModel, processor: CLIPProcessor, captions: list[str], device: torch.device
) -> torch.Tensor:
    with torch.no_grad():
        inputs = processor(
            text=captions, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        pooled = model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).pooler_output
        if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
            raise ValueError("text_model.pooler_output must be a 2D Tensor.")
        features = model.text_projection(pooled)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("text_projection must return a 2D Tensor.")
        return functional.normalize(features, p=2, dim=1)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _config_or_cli(value: Any, config: dict[str, Any], section: str, key: str) -> Any:
    return value if value is not None else config[section][key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dcec_v1.json"))
    parser.add_argument("--epochs", type=_positive_int, default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lambda-sem", type=float, default=None)
    parser.add_argument("--lambda-rank", type=float, default=None)
    parser.add_argument("--lambda-res", type=float, default=None)
    parser.add_argument("--limit-train", type=_positive_int, default=None)
    parser.add_argument("--limit-val", type=_positive_int, default=None)
    return parser.parse_args()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)
    args.epochs = _config_or_cli(args.epochs, config, "default_training", "epochs")
    args.batch_size = _config_or_cli(args.batch_size, config, "default_training", "batch_size")
    args.lr = _config_or_cli(args.lr, config, "default_training", "lr")
    args.weight_decay = _config_or_cli(
        args.weight_decay, config, "default_training", "weight_decay"
    )
    args.seed = _config_or_cli(args.seed, config, "default_training", "seed")
    args.lambda_sem = _config_or_cli(args.lambda_sem, config, "loss", "lambda_sem")
    args.lambda_rank = _config_or_cli(args.lambda_rank, config, "loss", "lambda_rank")
    args.lambda_res = _config_or_cli(args.lambda_res, config, "loss", "lambda_res")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = DCECDataset(
        split="train",
        path=args.train_dir,
        seed=args.seed,
        cache_dir=args.cache_dir,
        limit=args.limit_train,
    )
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=args.num_workers,
        collate_fn=dcec_collate,
    )

    clip_model = CLIPModel.from_pretrained(
        MODEL_NAME, cache_dir=args.cache_dir, use_safetensors=True
    ).to(device)
    freeze_clip(clip_model)
    if any(parameter.requires_grad for parameter in clip_model.parameters()):
        raise RuntimeError("All CLIP parameters must be frozen for DCEC training.")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir)
    dcec = DCEC().to(device)
    print(f"total DCEC trainable parameters: {dcec.count_trainable_parameters()}")

    optimizer = AdamW(dcec.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    clip_parameter_ids = {id(parameter) for parameter in clip_model.parameters()}
    if any(
        id(parameter) in clip_parameter_ids
        for group in optimizer.param_groups
        for parameter in group["params"]
    ):
        raise RuntimeError("Optimizer must not contain frozen CLIP parameters.")

    train_log = args.output_dir / "train_metrics.jsonl"
    validation_log = args.output_dir / "validation_metrics.jsonl"
    best_score = float("-inf")
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        dcec.train()
        epoch_losses: dict[str, list[float]] = {
            "total_loss": [],
            "sem_loss": [],
            "rank_loss": [],
            "res_loss": [],
        }
        for batch in train_loader:
            captions_by_image = batch["captions"]
            if any(len(captions) != 5 for captions in captions_by_image):
                raise ValueError("Training requires exactly five captions per image.")
            captions = [caption for group in captions_by_image for caption in group]
            z_clean = _frozen_image_embeddings(
                clip_model, processor, batch["clean_image"], device
            )
            z_degraded = _frozen_image_embeddings(
                clip_model, processor, batch["degraded_image"], device
            )
            text_features = _frozen_text_embeddings(clip_model, processor, captions, device)
            quality = torch.stack(
                [extract_quality_descriptor(image) for image in batch["degraded_image"]]
            ).to(device)

            outputs = dcec(z_degraded, quality)
            losses = compute_dcec_losses(
                z_corrected=outputs["z_corrected"],
                z_clean=z_clean,
                text_features=text_features,
                delta_z=outputs["delta_z"],
                scale=clip_model.logit_scale.exp().detach(),
                lambda_sem=args.lambda_sem,
                lambda_rank=args.lambda_rank,
                lambda_res=args.lambda_res,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            optimizer.step()
            for key in epoch_losses:
                epoch_losses[key].append(float(losses[key].detach().cpu()))

        train_record = {
            "epoch": epoch,
            "mean_total_loss": mean(epoch_losses["total_loss"]),
            "mean_sem_loss": mean(epoch_losses["sem_loss"]),
            "mean_rank_loss": mean(epoch_losses["rank_loss"]),
            "mean_res_loss": mean(epoch_losses["res_loss"]),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        _append_jsonl(train_log, train_record)

        validation_records = []
        for degradation, severity in VALIDATION_CONDITIONS:
            record = evaluate_validation_condition(
                dcec=dcec,
                clip_model=clip_model,
                processor=processor,
                val_file=args.val_file,
                degradation=degradation,
                severity=severity,
                seed=2026,
                batch_size=args.batch_size,
                cache_dir=args.cache_dir,
                limit=args.limit_val,
                device=device,
            )
            record["epoch"] = epoch
            record["primary_component"] = 0.5 * (record["i2t_R@1"] + record["t2i_R@1"])
            _append_jsonl(validation_log, record)
            validation_records.append(record)
        val_primary_score = mean(record["primary_component"] for record in validation_records)

        checkpoint = {
            "dcec_state_dict": dcec.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_primary_score": val_primary_score,
            "config": config,
            "seed": args.seed,
        }
        torch.save(checkpoint, args.output_dir / "last_dcec.pt")
        if val_primary_score > best_score:
            best_score = val_primary_score
            torch.save(checkpoint, args.output_dir / "best_dcec.pt")


if __name__ == "__main__":
    main()
