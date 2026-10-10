"""Image-level mechanism analysis for frozen DCEC A3.

Validation only. Final Flickr30K test is never accessed.

Goal:
Test whether degradation drift / semantic recovery / correction magnitude
remain associated with retrieval-oriented margin improvement after
removing condition-level means.
"""

import json
from pathlib import Path

import numpy as np
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


MODEL_NAME = "openai/clip-vit-base-patch32"

VAL_FILE = Path(
    "/root/autodl-tmp/flickr30k_trainval/data/"
    "validation-00000-of-00001.parquet"
)

CHECKPOINT = Path(
    "results/ablation_formal/no_sem/best_dcec_ablation.pt"
)

CACHE_DIR = Path("/root/autodl-tmp/hf_cache")

OUTPUT_JSON = Path(
    "results/journal_mechanism/"
    "image_level_mechanism_summary.json"
)

OUTPUT_NPZ = Path(
    "results/journal_mechanism/"
    "image_level_mechanism_arrays.npz"
)

BATCH_SIZE = 32

BOOTSTRAP_REPEATS = 500
BOOTSTRAP_SEED = 2026

SEEN_CONDITIONS = [
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

UNSEEN_CONDITIONS = [
    ("motion_blur", 1),
    ("motion_blur", 2),
    ("motion_blur", 3),

    ("contrast", 1),
    ("contrast", 2),
    ("contrast", 3),

    ("pixelate", 1),
    ("pixelate", 2),
    ("pixelate", 3),
]

CONDITIONS = SEEN_CONDITIONS + UNSEEN_CONDITIONS


def load_a3(device):
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("variant") != "no_sem":
        raise ValueError("Expected frozen A3 no_sem checkpoint.")

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


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if x.size != y.size:
        raise ValueError("Correlation arrays must have equal size.")

    if x.size < 2:
        return float("nan")

    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def retrieval_margins(
    similarity,
    image_to_text,
    text_to_image,
):
    similarity = similarity.cpu()

    number_of_images = similarity.shape[0]
    number_of_captions = similarity.shape[1]

    i2t_margins = torch.empty(
        number_of_images,
        dtype=similarity.dtype,
    )

    for image_index, positives in enumerate(image_to_text):
        row = similarity[image_index]

        positive_indices = torch.tensor(
            positives,
            dtype=torch.long,
        )

        positive_best = row[
            positive_indices
        ].max()

        negative_row = row.clone()
        negative_row[positive_indices] = -torch.inf

        negative_best = negative_row.max()

        i2t_margins[image_index] = (
            positive_best - negative_best
        )

    positive_images = torch.tensor(
        text_to_image,
        dtype=torch.long,
    )

    caption_indices = torch.arange(
        number_of_captions,
        dtype=torch.long,
    )

    positive_scores = similarity[
        positive_images,
        caption_indices,
    ]

    negative_similarity = similarity.clone()

    negative_similarity[
        positive_images,
        caption_indices,
    ] = -torch.inf

    hardest_negative_images = (
        negative_similarity.max(dim=0).values
    )

    caption_t2i_margins = (
        positive_scores
        - hardest_negative_images
    )

    t2i_sum = torch.zeros(
        number_of_images,
        dtype=similarity.dtype,
    )

    t2i_count = torch.zeros(
        number_of_images,
        dtype=similarity.dtype,
    )

    t2i_sum.scatter_add_(
        0,
        positive_images,
        caption_t2i_margins,
    )

    t2i_count.scatter_add_(
        0,
        positive_images,
        torch.ones_like(caption_t2i_margins),
    )

    t2i_image_margins = (
        t2i_sum
        / t2i_count.clamp_min(1.0)
    )

    combined = (
        i2t_margins
        + t2i_image_margins
    ) / 2.0

    return combined


def build_reference(
    clip_model,
    processor,
    device,
):
    dataset = DCECDataset(
        split="validation",
        path=VAL_FILE,
        seed=2026,
        cache_dir=CACHE_DIR,
        degradation="clean",
        severity=0,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=dcec_collate,
    )

    clean_features = []
    text_features = []

    image_to_text = []
    text_to_image = []

    image_offset = 0
    text_offset = 0

    print("Building frozen clean/text reference...")

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

            z_text = _frozen_text_embeddings(
                clip_model,
                processor,
                captions,
                device,
            )

            clean_features.append(z_clean.cpu())
            text_features.append(z_text.cpu())

            batch_size = len(batch["clean_image"])

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

    z_clean = torch.cat(clean_features, dim=0)
    z_text = torch.cat(text_features, dim=0)

    if z_clean.shape[0] != 1014:
        raise ValueError(
            f"Expected 1014 validation images, got {z_clean.shape[0]}"
        )

    if z_text.shape[0] != 5070:
        raise ValueError(
            f"Expected 5070 captions, got {z_text.shape[0]}"
        )

    print(
        "reference:",
        tuple(z_clean.shape),
        tuple(z_text.shape),
    )

    return (
        z_clean,
        z_text,
        image_to_text,
        text_to_image,
    )


def evaluate_condition(
    degradation,
    severity,
    clip_model,
    processor,
    a3,
    z_clean,
    z_text,
    image_to_text,
    text_to_image,
    device,
):
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

    degraded_features = []
    corrected_features = []

    correction_norms = []
    residual_norms = []
    gates = []

    with torch.no_grad():
        for batch in loader:
            z_degraded = _frozen_image_embeddings(
                clip_model,
                processor,
                batch["degraded_image"],
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

            degraded_features.append(
                z_degraded.cpu()
            )

            corrected_features.append(
                z_corrected.cpu()
            )

            correction_norms.append(
                (
                    z_corrected
                    - z_degraded.to(device)
                ).norm(
                    p=2,
                    dim=1,
                ).cpu()
            )

            residual_norms.append(
                delta_z.norm(
                    p=2,
                    dim=1,
                ).cpu()
            )

            gates.append(
                gate.reshape(-1).cpu()
            )

    z_degraded = torch.cat(
        degraded_features,
        dim=0,
    )

    z_corrected = torch.cat(
        corrected_features,
        dim=0,
    )

    correction_norm = torch.cat(
        correction_norms,
        dim=0,
    )

    residual_norm = torch.cat(
        residual_norms,
        dim=0,
    )

    gate = torch.cat(
        gates,
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

    drift = 1.0 - degraded_clean_cos

    recovery = (
        corrected_clean_cos
        - degraded_clean_cos
    )

    similarity_degraded = (
        z_degraded
        @ z_text.T
    )

    similarity_corrected = (
        z_corrected
        @ z_text.T
    )

    degraded_margin = retrieval_margins(
        similarity_degraded,
        image_to_text,
        text_to_image,
    )

    corrected_margin = retrieval_margins(
        similarity_corrected,
        image_to_text,
        text_to_image,
    )

    margin_gain = (
        corrected_margin
        - degraded_margin
    )

    return {
        "drift": drift.numpy(),
        "recovery": recovery.numpy(),
        "correction_norm": correction_norm.numpy(),
        "residual_norm": residual_norm.numpy(),
        "gate": gate.numpy(),
        "degraded_margin": degraded_margin.numpy(),
        "corrected_margin": corrected_margin.numpy(),
        "margin_gain": margin_gain.numpy(),
    }


def cluster_bootstrap_ci(
    x_matrix,
    y_matrix,
    repeats,
    seed,
):
    rng = np.random.default_rng(seed)

    condition_count, image_count = x_matrix.shape

    x_residual = (
        x_matrix
        - x_matrix.mean(
            axis=1,
            keepdims=True,
        )
    )

    y_residual = (
        y_matrix
        - y_matrix.mean(
            axis=1,
            keepdims=True,
        )
    )

    values = []

    for _ in range(repeats):
        indices = rng.integers(
            0,
            image_count,
            size=image_count,
        )

        x_boot = x_residual[
            :,
            indices,
        ].reshape(-1)

        y_boot = y_residual[
            :,
            indices,
        ].reshape(-1)

        value = pearson(
            x_boot,
            y_boot,
        )

        if np.isfinite(value):
            values.append(value)

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "bootstrap_repeats": int(
            values.size
        ),
        "ci95_low": float(
            np.percentile(values, 2.5)
        ),
        "ci95_high": float(
            np.percentile(values, 97.5)
        ),
    }


def summarize_relation(
    x_matrix,
    y_matrix,
    bootstrap_seed,
):
    raw = pearson(
        x_matrix.reshape(-1),
        y_matrix.reshape(-1),
    )

    x_residual = (
        x_matrix
        - x_matrix.mean(
            axis=1,
            keepdims=True,
        )
    )

    y_residual = (
        y_matrix
        - y_matrix.mean(
            axis=1,
            keepdims=True,
        )
    )

    residualized = pearson(
        x_residual.reshape(-1),
        y_residual.reshape(-1),
    )

    within_condition = []

    for index in range(
        x_matrix.shape[0]
    ):
        value = pearson(
            x_matrix[index],
            y_matrix[index],
        )

        if np.isfinite(value):
            within_condition.append(
                value
            )

    bootstrap = cluster_bootstrap_ci(
        x_matrix,
        y_matrix,
        repeats=BOOTSTRAP_REPEATS,
        seed=bootstrap_seed,
    )

    return {
        "pooled_raw_pearson": raw,
        "condition_demeaned_pearson": residualized,
        "mean_within_condition_pearson": float(
            np.mean(within_condition)
        ),
        "median_within_condition_pearson": float(
            np.median(within_condition)
        ),
        "condition_demeaned_cluster_bootstrap_ci95": bootstrap,
    }


def summarize_group(
    arrays,
    indices,
    seed_offset,
):
    relations = [
        (
            "drift_vs_margin_gain",
            "drift",
            "margin_gain",
        ),
        (
            "recovery_vs_margin_gain",
            "recovery",
            "margin_gain",
        ),
        (
            "correction_norm_vs_margin_gain",
            "correction_norm",
            "margin_gain",
        ),
        (
            "gate_vs_margin_gain",
            "gate",
            "margin_gain",
        ),
        (
            "drift_vs_correction_norm",
            "drift",
            "correction_norm",
        ),
        (
            "drift_vs_recovery",
            "drift",
            "recovery",
        ),
    ]

    output = {}

    for relation_index, (
        name,
        x_name,
        y_name,
    ) in enumerate(relations):

        x = arrays[x_name][indices]
        y = arrays[y_name][indices]

        output[name] = summarize_relation(
            x,
            y,
            bootstrap_seed=(
                BOOTSTRAP_SEED
                + seed_offset
                + relation_index
            ),
        )

    return output


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("device:", device)

    if device.type == "cuda":
        print(
            "gpu:",
            torch.cuda.get_device_name(device),
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

    (
        z_clean,
        z_text,
        image_to_text,
        text_to_image,
    ) = build_reference(
        clip_model,
        processor,
        device,
    )

    collected = {
        "drift": [],
        "recovery": [],
        "correction_norm": [],
        "residual_norm": [],
        "gate": [],
        "degraded_margin": [],
        "corrected_margin": [],
        "margin_gain": [],
    }

    condition_names = []

    for condition_index, (
        degradation,
        severity,
    ) in enumerate(CONDITIONS, 1):

        name = (
            f"{degradation}_s{severity}"
        )

        condition_names.append(name)

        print()
        print("=" * 88)
        print(
            f"[{condition_index}/{len(CONDITIONS)}] "
            f"{degradation} S{severity}"
        )
        print("=" * 88)

        row = evaluate_condition(
            degradation=degradation,
            severity=severity,
            clip_model=clip_model,
            processor=processor,
            a3=a3,
            z_clean=z_clean,
            z_text=z_text,
            image_to_text=image_to_text,
            text_to_image=text_to_image,
            device=device,
        )

        for key in collected:
            collected[key].append(
                row[key]
            )

        print(
            "mean drift:",
            f"{np.mean(row['drift']):.6f}"
        )

        print(
            "mean recovery:",
            f"{np.mean(row['recovery']):+.6f}"
        )

        print(
            "mean correction norm:",
            f"{np.mean(row['correction_norm']):.6f}"
        )

        print(
            "mean retrieval-margin gain:",
            f"{np.mean(row['margin_gain']):+.6f}"
        )

    arrays = {
        key: np.stack(
            values,
            axis=0,
        )
        for key, values in collected.items()
    }

    seen_indices = np.arange(
        len(SEEN_CONDITIONS)
    )

    unseen_indices = np.arange(
        len(SEEN_CONDITIONS),
        len(CONDITIONS),
    )

    all_indices = np.arange(
        len(CONDITIONS)
    )

    summary = {
        "experiment": (
            "DCEC image-level mechanism analysis"
        ),
        "dataset_split": (
            "Flickr30K validation"
        ),
        "final_test_accessed": False,
        "checkpoint": str(CHECKPOINT),
        "number_of_images": int(
            arrays["drift"].shape[1]
        ),
        "seen_conditions": [
            f"{d}_s{s}"
            for d, s in SEEN_CONDITIONS
        ],
        "unseen_conditions": [
            f"{d}_s{s}"
            for d, s in UNSEEN_CONDITIONS
        ],
        "number_of_seen_conditions": len(
            SEEN_CONDITIONS
        ),
        "number_of_unseen_conditions": len(
            UNSEEN_CONDITIONS
        ),
        "number_of_total_conditions": len(
            CONDITIONS
        ),
        "retrieval_margin_definition": (
            "per-image average of bidirectional retrieval margins; "
            "I2T uses best positive minus hardest negative caption; "
            "T2I uses correct image minus hardest negative image "
            "for each caption, averaged over captions"
        ),
        "statistics_note": (
            "condition-demeaned correlations remove each "
            "degradation/severity condition mean before pooling. "
            "95% confidence intervals use image-cluster bootstrap. "
            "Associations are descriptive and do not establish causality."
        ),
        "seen": summarize_group(
            arrays,
            seen_indices,
            seed_offset=100,
        ),
        "unseen": summarize_group(
            arrays,
            unseen_indices,
            seed_offset=200,
        ),
        "all_nonclean": summarize_group(
            arrays,
            all_indices,
            seed_offset=300,
        ),
    }

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        condition_names=np.array(
            condition_names,
            dtype=str,
        ),
        **arrays,
    )

    print()
    print("=" * 100)
    print("IMAGE-LEVEL MECHANISM SUMMARY")
    print("=" * 100)

    for group_name in [
        "seen",
        "unseen",
        "all_nonclean",
    ]:
        print()
        print(
            f"===== {group_name.upper()} ====="
        )

        group = summary[group_name]

        for relation_name, stats in group.items():
            ci = stats[
                "condition_demeaned_cluster_bootstrap_ci95"
            ]

            print(
                f"{relation_name:34s} "
                f"raw={stats['pooled_raw_pearson']:+.4f} | "
                f"demeaned={stats['condition_demeaned_pearson']:+.4f} | "
                f"within={stats['mean_within_condition_pearson']:+.4f} | "
                f"CI95=[{ci['ci95_low']:+.4f}, "
                f"{ci['ci95_high']:+.4f}]"
            )

    print()
    print("Saved:", OUTPUT_JSON)
    print("Saved:", OUTPUT_NPZ)
    print()
    print("FINAL TEST ACCESSED: NO")


if __name__ == "__main__":
    main()
