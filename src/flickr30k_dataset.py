"""Flickr30K Karpathy-split loading and retrieval mapping helpers."""

from collections.abc import Iterable, Mapping
from itertools import islice
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "noonamkha/flickr30k-karpathy"
SPLITS = frozenset({"train", "validation", "test"})
REQUIRED_FIELDS = frozenset({"image_id", "image", "captions"})
CAPTIONS_PER_IMAGE = 5


def _validate_schema(dataset: Any) -> None:
    """Fail early when a data source does not expose the expected fields."""
    columns = getattr(dataset, "column_names", None)
    if columns is None:
        features = getattr(dataset, "features", None)
        columns = list(features) if features is not None else []
    missing = REQUIRED_FIELDS.difference(columns)
    if missing:
        raise ValueError(
            f"Dataset schema is missing required fields: {sorted(missing)}. "
            f"Available fields: {list(columns)}"
        )


def load_flickr30k_split(
    split: str = "test",
    dataset_id: str = DATASET_NAME,
    local_parquet: str | Path | None = None,
    cache_dir: str | Path | None = None,
    streaming: bool = False,
):
    """Load exactly one Flickr30K Karpathy split while retaining all fields.

    ``local_parquet`` takes priority and restricts loading to that one Parquet
    file. Otherwise, ``split`` is passed directly to the Hugging Face loader,
    so requesting ``test`` does not prepare train or validation splits.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}.")

    if local_parquet is not None:
        parquet_path = Path(local_parquet)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Local Parquet file not found: {parquet_path}")
        dataset = load_dataset(
            "parquet",
            data_files={split: str(parquet_path)},
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )
    else:
        dataset = load_dataset(
            dataset_id,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
        )
    _validate_schema(dataset)
    return dataset


def _validate_sample(sample: Mapping[str, Any], sample_index: int) -> list[str]:
    missing = REQUIRED_FIELDS.difference(sample)
    if missing:
        raise ValueError(
            f"Sample {sample_index} is missing required fields: {sorted(missing)}."
        )
    if sample["image_id"] is None or sample["image"] is None:
        raise ValueError(f"Sample {sample_index} has an empty image_id or image.")

    captions = sample["captions"]
    if not isinstance(captions, (list, tuple)):
        raise ValueError(f"Sample {sample_index} captions must be a list or tuple.")
    if len(captions) != CAPTIONS_PER_IMAGE:
        raise ValueError(
            f"Sample {sample_index} must have exactly {CAPTIONS_PER_IMAGE} captions; "
            f"got {len(captions)}."
        )
    if not all(isinstance(caption, str) and caption for caption in captions):
        raise ValueError(f"Sample {sample_index} contains an invalid caption.")
    return list(captions)


def prepare_retrieval_samples(
    dataset: Iterable[Mapping[str, Any]], num_samples: int
) -> tuple[list[Any], list[str], list[list[int]], list[int]]:
    """Flatten the first ``num_samples`` images and all captions for retrieval.

    The returned ``image_to_text`` and ``text_to_image`` mappings are directly
    compatible with ``compute_retrieval_metrics`` in ``retrieval_metrics.py``.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    images: list[Any] = []
    captions_flat: list[str] = []
    image_to_text: list[list[int]] = []
    text_to_image: list[int] = []

    # islice limits streaming datasets to the requested small inspection subset.
    for image_index, sample in enumerate(islice(dataset, num_samples)):
        captions = _validate_sample(sample, image_index)
        text_indices = list(range(len(captions_flat), len(captions_flat) + len(captions)))

        images.append(sample["image"])
        captions_flat.extend(captions)  # Keep every one of the five captions.
        image_to_text.append(text_indices)
        text_to_image.extend([image_index] * len(captions))

    if len(images) != num_samples:
        raise ValueError(
            f"Requested {num_samples} samples, but the dataset yielded only {len(images)}."
        )
    return images, captions_flat, image_to_text, text_to_image
