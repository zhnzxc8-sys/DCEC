"""Mechanism analysis for frozen DCEC A3.

Validation data only.
Final Flickr30K test is never accessed.

Measures:
- degradation-induced embedding drift
- corrected-to-clean similarity
- correction magnitude
- raw residual magnitude
- gate statistics
- retrieval performance
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from dcec_ablation_model import DCECAblation, ablation_spec
from dcec_dataset import DCECDataset, dcec_collate
from dcec_quality import extract_quality_descriptor
from eval_dcec_validation import (
    _frozen_image_embeddings,
    _frozen_text_embeddings,
    freeze_clip,
)
from retrieval_metrics import compute_retrieval_metrics


MODEL_NAME = "openai/clip-vit-base-patch32"

VAL_FILE = Path(
    "/root/autodl-tmp/flickr30k_trainval/data/"
    "validation-00000-of-00001.parquet"
)

CHECKPOINT = Path(
    "results/ablation_formal/no_sem/best_dcec_ablation.pt"
)

CACHE_DIR = Path("/root/autodl-tmp/hf_cache")

OUTPUT = Path(
    "results/journal_mechanism/"
    "seen_degradation_mechanism.json"
)

CONDITIONS = [
    ("clean", 0),

    ("jpeg", 1),
    ("jpeg", 2),
    ("jpeg", 3),

    ("gaussian_blur", 1),
    ("gaussian_blur", 2),
    ("gaussian_blur", 3),

    ("gaussian_noise", 1),
    ("gaussian_noise", 2),
    ("gaussian_noise", 3),

    ("low_light", 1),
    ("low_light", 2),
    ("low_light", 3),
]

BATCH_SIZE = 32


def primary(metrics):
    return (
        float(metrics["i2t_R@1"])
        + float(metrics["t2i_R@1"])
    ) / 2.0


def load_a3(device):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("variant") != "no_sem":
        raise ValueError("Expected frozen no_sem checkpoint.")

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

    return model


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    clip_model = CLIPModel.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        use_safetensors=True,
    ).to(device)

    freeze_clip(clip_model)

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
    )

    a3 = load_a3(device)

    results = []

    for condition_index, (degradation, severity) in enumerate(
        CONDITIONS,
        1,
    ):
        print()
        print("=" * 84)
        print(
            f"[{condition_index}/{len(CONDITIONS)}] "
            f"{degradation} S{severity}"
        )
        print("=" * 84)

        dataset = DCECDataset(
            split="validation",
            path=VAL_FILE,
            seed=2026,
            cache_dir=CACHE_DIR,
            degradation=degradation,
            severity=severity,
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=dcec_collate,
        )

        clean_features = []
        degraded_features = []
        corrected_features = []
        text_features = []

        image_to_text = []
        text_to_image = []

        gate_values = []
        residual_norms = []
        correction_norms = []

        image_offset = 0
        text_offset = 0

        with torch.no_grad():

            for batch in loader:

                captions = [
                    caption
                    for group in batch["captions"]
                    for caption in group
                ]

                z_clean = _frozen_image_embeddings(
                    clip_model,
                    processor,
                    batch["clean_image"],
                    device,
                )

                z_degraded = _frozen_image_embeddings(
                    clip_model,
                    processor,
                    batch["degraded_image"],
                    device,
                )

                z_text = _frozen_text_embeddings(
                    clip_model,
                    processor,
                    captions,
                    device,
                )

                quality = torch.stack(
                    [
                        extract_quality_descriptor(image)
                        for image in batch["degraded_image"]
                    ]
                ).to(device)

                outputs = a3(
                    z_degraded.to(device),
                    quality,
                )

                z_corrected = outputs["z_corrected"]
                delta_z = outputs["delta_z"]
                gate = outputs["gate"]

                clean_features.append(
                    z_clean.cpu()
                )

                degraded_features.append(
                    z_degraded.cpu()
                )

                corrected_features.append(
                    z_corrected.cpu()
                )

                text_features.append(
                    z_text.cpu()
                )

                residual_norms.extend(
                    delta_z.norm(
                        p=2,
                        dim=1,
                    ).cpu().tolist()
                )

                correction_norms.extend(
                    (
                        z_corrected
                        - z_degraded.to(device)
                    ).norm(
                        p=2,
                        dim=1,
                    ).cpu().tolist()
                )

                gate_values.extend(
                    gate.reshape(-1).cpu().tolist()
                )

                batch_size = len(
                    batch["degraded_image"]
                )

                for local_index in range(batch_size):

                    image_to_text.append(
                        list(
                            range(
                                text_offset,
                                text_offset + 5,
                            )
                        )
                    )

                    text_to_image.extend(
                        [image_offset + local_index] * 5
                    )

                    text_offset += 5

                image_offset += batch_size

        z_clean = torch.cat(
            clean_features,
            dim=0,
        )

        z_degraded = torch.cat(
            degraded_features,
            dim=0,
        )

        z_corrected = torch.cat(
            corrected_features,
            dim=0,
        )

        z_text = torch.cat(
            text_features,
            dim=0,
        )

        degraded_clean_cos = F.cosine_similarity(
            z_degraded,
            z_clean,
            dim=1,
        )

        corrected_clean_cos = F.cosine_similarity(
            z_corrected,
            z_clean,
            dim=1,
        )

        frozen_metrics = compute_retrieval_metrics(
            z_degraded,
            z_text,
            image_to_text,
            text_to_image,
        )

        corrected_metrics = compute_retrieval_metrics(
            z_corrected,
            z_text,
            image_to_text,
            text_to_image,
        )

        frozen_primary = primary(
            frozen_metrics
        )

        corrected_primary = primary(
            corrected_metrics
        )

        row = {
            "degradation": degradation,
            "severity": severity,

            "number_of_images": int(
                z_clean.shape[0]
            ),

            "degraded_clean_cosine_mean": float(
                degraded_clean_cos.mean().item()
            ),

            "degraded_clean_cosine_std": float(
                degraded_clean_cos.std(
                    unbiased=False
                ).item()
            ),

            "corrected_clean_cosine_mean": float(
                corrected_clean_cos.mean().item()
            ),

            "corrected_clean_cosine_std": float(
                corrected_clean_cos.std(
                    unbiased=False
                ).item()
            ),

            "semantic_recovery_cosine": float(
                (
                    corrected_clean_cos
                    - degraded_clean_cos
                ).mean().item()
            ),

            "raw_residual_norm_mean": float(
                torch.tensor(
                    residual_norms
                ).mean().item()
            ),

            "correction_norm_mean": float(
                torch.tensor(
                    correction_norms
                ).mean().item()
            ),

            "gate_mean": float(
                torch.tensor(
                    gate_values
                ).mean().item()
            ),

            "gate_std": float(
                torch.tensor(
                    gate_values
                ).std(
                    unbiased=False
                ).item()
            ),

            "frozen_primary": frozen_primary,

            "corrected_primary": corrected_primary,

            "retrieval_delta": (
                corrected_primary
                - frozen_primary
            ),
        }

        results.append(row)

        print(
            "deg→clean cosine:",
            f"{row['degraded_clean_cosine_mean']:.6f}"
        )

        print(
            "corr→clean cosine:",
            f"{row['corrected_clean_cosine_mean']:.6f}"
        )

        print(
            "semantic recovery:",
            f"{row['semantic_recovery_cosine']:+.6f}"
        )

        print(
            "correction norm:",
            f"{row['correction_norm_mean']:.6f}"
        )

        print(
            "gate:",
            f"{row['gate_mean']:.6f}"
            f" ± {row['gate_std']:.6f}"
        )

        print(
            "retrieval:",
            f"{frozen_primary:.4f}"
            " → "
            f"{corrected_primary:.4f}"
            f" ({row['retrieval_delta']:+.4f})"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            {
                "experiment": (
                    "DCEC degradation mechanism analysis"
                ),
                "dataset_split": (
                    "Flickr30K validation"
                ),
                "final_test_accessed": False,
                "checkpoint": str(CHECKPOINT),
                "conditions": results,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("MECHANISM SUMMARY")
    print("=" * 100)

    print(
        f"{'Condition':22s} "
        f"{'Deg→Clean':>10s} "
        f"{'Corr→Clean':>11s} "
        f"{'Recovery':>10s} "
        f"{'CorrNorm':>10s} "
        f"{'Gate':>8s} "
        f"{'ΔRetr':>8s}"
    )

    for row in results:

        name = (
            f"{row['degradation']} "
            f"S{row['severity']}"
        )

        print(
            f"{name:22s} "
            f"{row['degraded_clean_cosine_mean']:10.6f} "
            f"{row['corrected_clean_cosine_mean']:11.6f} "
            f"{row['semantic_recovery_cosine']:+10.6f} "
            f"{row['correction_norm_mean']:10.6f} "
            f"{row['gate_mean']:8.4f} "
            f"{row['retrieval_delta']:+8.4f}"
        )

    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
