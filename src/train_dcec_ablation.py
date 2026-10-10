"""Train pre-registered DCEC ablations using train and validation only."""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from dcec_ablation_model import (
    ABLATION_VARIANTS,
    DCECAblation,
    ablation_spec,
    validation_primary_score,
)
from dcec_dataset import DCECDataset, dcec_collate
from dcec_losses import compute_dcec_losses
from dcec_quality import extract_quality_descriptor
from eval_dcec_ablation_validation import evaluate_ablation_validation_condition
from eval_dcec_validation import _frozen_image_embeddings, _frozen_text_embeddings, freeze_clip


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=ABLATION_VARIANTS, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/dcec_ablation.json"))
    parser.add_argument("--epochs", type=_positive_int, default=5)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train", type=_positive_int, default=None)
    parser.add_argument("--limit-val", type=_positive_int, default=None)
    return parser.parse_args()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    with args.config.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    spec = ablation_spec(args.variant)
    architecture = str(spec["architecture"])
    lambda_sem = float(spec["lambda_sem"])
    lambda_rank = float(spec["lambda_rank"])
    lambda_res = float(spec["lambda_res"])

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers,
        collate_fn=dcec_collate,
    )

    clip_model = CLIPModel.from_pretrained(
        MODEL_NAME, cache_dir=args.cache_dir, use_safetensors=True
    ).to(device)
    freeze_clip(clip_model)
    if any(parameter.requires_grad for parameter in clip_model.parameters()):
        raise RuntimeError("All CLIP parameters must be frozen.")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, cache_dir=args.cache_dir)
    ablation_model = DCECAblation(architecture).to(device)
    print(
        f"total {args.variant} trainable parameters: "
        f"{ablation_model.count_trainable_parameters()}"
    )
    optimizer = AdamW(ablation_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    clip_parameter_ids = {id(parameter) for parameter in clip_model.parameters()}
    if any(
        id(parameter) in clip_parameter_ids
        for group in optimizer.param_groups
        for parameter in group["params"]
    ):
        raise RuntimeError("Optimizer must not include CLIP parameters.")

    train_log = args.output_dir / "train_metrics.jsonl"
    validation_log = args.output_dir / "validation_metrics.jsonl"
    best_score = float("-inf")
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        ablation_model.train()
        epoch_losses: dict[str, list[float]] = {
            "total_loss": [],
            "sem_loss": [],
            "rank_loss": [],
            "res_loss": [],
        }
        for batch in train_loader:
            caption_groups = batch["captions"]
            if any(len(captions) != 5 for captions in caption_groups):
                raise ValueError("Training requires exactly five captions per image.")
            captions = [caption for captions in caption_groups for caption in captions]
            z_clean = _frozen_image_embeddings(
                clip_model, processor, batch["clean_image"], device
            )
            z_degraded = _frozen_image_embeddings(
                clip_model, processor, batch["degraded_image"], device
            )
            text_features = _frozen_text_embeddings(clip_model, processor, captions, device)
            quality = None
            if architecture != "residual_only":
                quality = torch.stack(
                    [extract_quality_descriptor(image) for image in batch["degraded_image"]]
                ).to(device)
            outputs = ablation_model(z_degraded, quality)
            losses = compute_dcec_losses(
                z_corrected=outputs["z_corrected"],
                z_clean=z_clean,
                text_features=text_features,
                delta_z=outputs["delta_z"],
                scale=clip_model.logit_scale.exp().detach(),
                lambda_sem=lambda_sem,
                lambda_rank=lambda_rank,
                lambda_res=lambda_res,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            optimizer.step()
            for key in epoch_losses:
                epoch_losses[key].append(float(losses[key].detach().cpu()))

        _append_jsonl(
            train_log,
            {
                "variant": args.variant,
                "epoch": epoch,
                "mean_total_loss": mean(epoch_losses["total_loss"]),
                "mean_sem_loss": mean(epoch_losses["sem_loss"]),
                "mean_rank_loss": mean(epoch_losses["rank_loss"]),
                "mean_res_loss": mean(epoch_losses["res_loss"]),
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
        )

        validation_records = []
        for degradation, severity in VALIDATION_CONDITIONS:
            record = evaluate_ablation_validation_condition(
                variant=args.variant,
                ablation_model=ablation_model,
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
        primary_score = validation_primary_score(validation_records)

        checkpoint = {
            "variant": args.variant,
            "model_state_dict": ablation_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_primary_score": primary_score,
            "config": config,
            "seed": args.seed,
        }
        torch.save(checkpoint, args.output_dir / "last_dcec_ablation.pt")
        if primary_score > best_score:
            best_score = primary_score
            torch.save(checkpoint, args.output_dir / "best_dcec_ablation.pt")


if __name__ == "__main__":
    main()
