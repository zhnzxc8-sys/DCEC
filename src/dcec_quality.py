"""Image-only, label-free visual quality descriptors for DCEC."""

import numpy as np
import torch
from PIL import Image, ImageFilter


def _grayscale_luminance(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to RGB luminance in the closed interval [0, 1]."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _laplacian_absolute_mean(luminance: np.ndarray) -> float:
    padded = np.pad(luminance, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * luminance
    )
    return float(np.mean(np.abs(laplacian)))


def _block_boundary_response(luminance: np.ndarray) -> float:
    """Measure absolute discontinuities across internal 8-pixel JPEG-like blocks."""
    height, width = luminance.shape
    responses: list[np.ndarray] = []
    for column in range(8, width, 8):
        responses.append(np.abs(luminance[:, column] - luminance[:, column - 1]))
    for row in range(8, height, 8):
        responses.append(np.abs(luminance[row, :] - luminance[row - 1, :]))
    if not responses:
        return 0.0
    return float(np.mean(np.concatenate([response.ravel() for response in responses])))


def extract_quality_descriptor(image: Image.Image) -> torch.Tensor:
    """Return five finite float32 descriptors computed only from ``image``.

    The descriptor comprises luminance mean and standard deviation, Laplacian
    magnitude, high-frequency residual magnitude, and 8-pixel boundary response.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance.")

    luminance = _grayscale_luminance(image)
    grayscale_image = Image.fromarray(
        np.rint(luminance * 255.0).astype(np.uint8), mode="L"
    )
    blurred = np.asarray(
        grayscale_image.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32
    ) / 255.0
    descriptor = np.asarray(
        [
            np.mean(luminance),
            np.std(luminance),
            _laplacian_absolute_mean(luminance),
            np.mean(np.abs(luminance - blurred)),
            _block_boundary_response(luminance),
        ],
        dtype=np.float32,
    )
    if descriptor.shape != (5,) or not np.all(np.isfinite(descriptor)):
        raise ValueError("Quality descriptor must be finite with shape (5,).")
    return torch.from_numpy(descriptor)
