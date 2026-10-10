"""Local-Parquet train/validation data access with a strict frozen-test guard."""

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset
from torch.utils.data import Dataset

from image_degradations import apply_degradation, get_degradation_metadata


TRAINING_DEGRADATIONS = (
    "clean",
    "jpeg",
    "gaussian_blur",
    "gaussian_noise",
    "low_light",
)


def validate_train_validation_path(path: str | Path) -> Path:
    """Reject any path component that explicitly refers to a frozen test split."""
    resolved = Path(path)
    for component in str(resolved).replace("\\", "/").lower().split("/"):
        if component == "test" or component == "test.parquet" or (
            component.startswith("test-") and component.endswith(".parquet")
        ):
            raise ValueError("DCEC training and validation must never read a test path.")
    return resolved


def _resolve_parquet_files(path: str | Path, split: str) -> list[Path]:
    validated_path = validate_train_validation_path(path)
    if validated_path.is_file():
        if validated_path.suffix.lower() != ".parquet":
            raise ValueError("DCEC data files must use the .parquet extension.")
        return [validated_path]
    if not validated_path.is_dir():
        raise FileNotFoundError(f"DCEC data path does not exist: {validated_path}")

    files = sorted(validated_path.glob(f"{split}-*.parquet"))
    single_file = validated_path / f"{split}.parquet"
    if single_file.is_file():
        files.append(single_file)
    if not files:
        raise FileNotFoundError(f"No {split} Parquet files found in {validated_path}")
    return files


def load_dcec_split(
    split: str, path: str | Path, cache_dir: str | Path | None = None
):
    """Load only local train or validation Parquet files; test is forbidden."""
    if split not in {"train", "validation"}:
        raise ValueError("DCEC only supports split='train' or split='validation'; test is frozen.")
    parquet_files = _resolve_parquet_files(path, split)
    return load_dataset(
        "parquet",
        data_files={split: [str(file_path) for file_path in parquet_files]},
        split=split,
        cache_dir=cache_dir,
    )


def deterministic_degradation_choice(seed: int, epoch: int, image_index: int) -> tuple[str, int]:
    """Choose one train-time degradation with a private, reproducible RNG."""
    condition_seed = seed + epoch * 1_000_003 + image_index
    generator = random.Random(condition_seed)
    degradation = generator.choice(TRAINING_DEGRADATIONS)
    severity = 0 if degradation == "clean" else generator.choice((1, 2, 3))
    return degradation, severity


class DCECDataset(Dataset):
    """Return clean/degraded image pairs and all five captions without test access."""

    def __init__(
        self,
        split: str,
        path: str | Path,
        seed: int = 2026,
        cache_dir: str | Path | None = None,
        degradation: str | None = None,
        severity: int | None = None,
        limit: int | None = None,
    ) -> None:
        if split == "test":
            raise ValueError("DCEC test access is forbidden.")
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")
        if split == "validation" and (degradation is None or severity is None):
            raise ValueError("Validation requires an explicit degradation and severity.")

        self.split = split
        self.seed = seed
        self.epoch = 0
        self.degradation = degradation
        self.severity = severity
        self.dataset = load_dcec_split(split, path, cache_dir=cache_dir)
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive when provided.")
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by deterministic train-time degradation sampling."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = epoch

    def condition_for_index(self, image_index: int) -> tuple[str, int, int]:
        """Return deterministic condition and degradation seed for one image index."""
        if self.split == "train":
            degradation, severity = deterministic_degradation_choice(
                self.seed, self.epoch, image_index
            )
            condition_seed = self.seed + self.epoch * 1_000_003 + image_index
            return degradation, severity, condition_seed
        assert self.degradation is not None and self.severity is not None
        return self.degradation, self.severity, self.seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, image_index: int) -> dict[str, Any]:
        sample = self.dataset[image_index]
        captions = sample.get("captions")
        if not isinstance(captions, Sequence) or isinstance(captions, str) or len(captions) != 5:
            raise ValueError("Every DCEC image must retain exactly five captions.")
        if not all(isinstance(caption, str) for caption in captions):
            raise ValueError("Captions must be strings.")
        image = sample.get("image")
        if image is None or not hasattr(image, "convert"):
            raise ValueError("Sample is missing a PIL-compatible image.")

        clean_image = image.convert("RGB")
        degradation, severity, condition_seed = self.condition_for_index(image_index)
        degraded_image = apply_degradation(
            clean_image,
            degradation=degradation,
            severity=severity,
            seed=condition_seed,
            image_index=image_index,
        )
        degradation_metadata = get_degradation_metadata(degradation, severity)
        degradation_metadata["seed"] = condition_seed
        return {
            "clean_image": clean_image,
            "degraded_image": degraded_image,
            "captions": list(captions),
            "image_id": sample.get("image_id"),
            "degradation": degradation,
            "severity": severity,
            "degradation_seed": condition_seed,
            "degradation_metadata": degradation_metadata,
        }


def dcec_collate(samples: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Keep PIL images and per-image five-caption groups intact for frozen CLIP."""
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    return {key: [sample[key] for sample in samples] for key in samples[0]}
