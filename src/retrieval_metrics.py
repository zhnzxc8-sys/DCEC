"""Recall@K metrics for image-to-text and text-to-image retrieval."""

from collections.abc import Sequence

import torch


_KS = (1, 5, 10)


def _normalize_features(features: torch.Tensor, name: str) -> torch.Tensor:
    """L2-normalize feature rows and reject undefined cosine similarities."""
    norms = features.norm(dim=1, keepdim=True)
    if torch.any(norms == 0):
        raise ValueError(f"{name} contains a zero-norm feature vector.")
    return features / norms


def _validate_inputs(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_to_text: Sequence[Sequence[int]],
    text_to_image: Sequence[int],
) -> None:
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("image_features and text_features must be 2D tensors.")
    if image_features.shape[0] == 0 or text_features.shape[0] == 0:
        raise ValueError("image_features and text_features must not be empty.")
    if image_features.shape[1] != text_features.shape[1]:
        raise ValueError("Image and text feature dimensions must match.")
    if len(image_to_text) != image_features.shape[0]:
        raise ValueError("image_to_text must have one entry per image.")
    if len(text_to_image) != text_features.shape[0]:
        raise ValueError("text_to_image must have one entry per text.")

    text_count = text_features.shape[0]
    image_count = image_features.shape[0]
    for positives in image_to_text:
        if not positives:
            raise ValueError("Every image must have at least one matching text.")
        if any(index < 0 or index >= text_count for index in positives):
            raise ValueError("image_to_text contains an out-of-range text index.")
    if any(index < 0 or index >= image_count for index in text_to_image):
        raise ValueError("text_to_image contains an out-of-range image index.")


def _recall_at_k(
    similarity: torch.Tensor, correct_indices: Sequence[Sequence[int]], k: int
) -> float:
    """Return percentage of queries with any correct item in their top-k list."""
    rankings = similarity.argsort(dim=1, descending=True)
    hits = 0
    for query_index, positives in enumerate(correct_indices):
        top_k = set(rankings[query_index, :k].tolist())
        hits += bool(top_k.intersection(positives))
    return 100.0 * hits / len(correct_indices)


def compute_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    image_to_text: Sequence[Sequence[int]],
    text_to_image: Sequence[int],
) -> dict[str, float]:
    """Compute cosine-similarity Recall@1, @5, and @10 for both directions.

    ``image_to_text[i]`` contains every relevant text index for image ``i``;
    ``text_to_image[j]`` contains the one relevant image index for text ``j``.
    """
    _validate_inputs(image_features, text_features, image_to_text, text_to_image)

    # Convert both feature sets to unit vectors so dot products are cosine scores.
    image_features = _normalize_features(image_features, "image_features")
    text_features = _normalize_features(
        text_features.to(image_features.device, dtype=image_features.dtype),
        "text_features",
    )
    similarity = image_features @ text_features.T

    # Rows query texts for I2T; transposing makes each text query images for T2I.
    text_to_image_groups = [[image_index] for image_index in text_to_image]
    metrics = {
        f"i2t_R@{k}": _recall_at_k(similarity, image_to_text, k) for k in _KS
    }
    metrics.update(
        {
            f"t2i_R@{k}": _recall_at_k(similarity.T, text_to_image_groups, k)
            for k in _KS
        }
    )
    return metrics
