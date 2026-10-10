"""Deterministic image-only degradations for retrieval sensitivity experiments."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


_JPEG_QUALITY = {1: 75, 2: 50, 3: 25}
_BLUR_RADIUS = {1: 1.0, 2: 2.0, 3: 4.0}
_NOISE_SIGMA = {1: 0.02, 2: 0.05, 3: 0.10}
_BRIGHTNESS_FACTOR = {1: 0.75, 2: 0.50, 3: 0.25}

# Validation-only unseen degradations.
# These are intentionally excluded from TRAINING_DEGRADATIONS in dcec_dataset.py.
_MOTION_BLUR_KERNEL = {1: 3, 2: 7, 3: 15}
_CONTRAST_FACTOR = {1: 0.75, 2: 0.50, 3: 0.25}
_PIXELATE_BLOCK = {1: 2, 2: 4, 3: 8}
_DEGRADATIONS = {
    "clean": ("none", {0: None}),
    "jpeg": ("quality", _JPEG_QUALITY),
    "gaussian_blur": ("radius", _BLUR_RADIUS),
    "gaussian_noise": ("sigma", _NOISE_SIGMA),
    "low_light": ("brightness_factor", _BRIGHTNESS_FACTOR),
    "motion_blur": ("kernel_size", _MOTION_BLUR_KERNEL),
    "contrast": ("contrast_factor", _CONTRAST_FACTOR),
    "pixelate": ("block_size", _PIXELATE_BLOCK),
}


def _validate_condition(degradation: str, severity: int) -> None:
    if degradation not in _DEGRADATIONS:
        raise ValueError(
            f"Unsupported degradation {degradation!r}. "
            f"Choose from {sorted(_DEGRADATIONS)}."
        )
    if not isinstance(severity, int) or isinstance(severity, bool):
        raise ValueError("severity must be an integer in {0, 1, 2, 3}.")
    if severity not in (0, 1, 2, 3):
        raise ValueError("severity must be one of 0, 1, 2, or 3.")
    if degradation == "clean" and severity != 0:
        raise ValueError("clean degradation only supports severity=0.")


def get_degradation_metadata(degradation: str, severity: int) -> dict[str, object]:
    """Return the actual parameter used for one degradation condition."""
    _validate_condition(degradation, severity)
    if severity == 0:
        return {
            "degradation": degradation,
            "severity": severity,
            "parameter_name": "none",
            "parameter_value": None,
        }

    parameter_name, parameter_values = _DEGRADATIONS[degradation]
    return {
        "degradation": degradation,
        "severity": severity,
        "parameter_name": parameter_name,
        "parameter_value": parameter_values[severity],
    }


def apply_degradation(
    image: Image.Image,
    degradation: str,
    severity: int,
    seed: int = 2026,
    image_index: int = 0,
) -> Image.Image:
    """Apply one deterministic degradation and return an RGB image of the same size."""
    _validate_condition(degradation, severity)
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance.")

    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    if severity == 0:
        return rgb_image

    metadata = get_degradation_metadata(degradation, severity)
    parameter_value = metadata["parameter_value"]

    if degradation == "jpeg":
        # Keep JPEG encoding fully in memory; no temporary image files are written.
        buffer = BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=parameter_value)
        buffer.seek(0)
        with Image.open(buffer) as decoded_image:
            return decoded_image.convert("RGB")

    if degradation == "gaussian_blur":
        return rgb_image.filter(ImageFilter.GaussianBlur(radius=parameter_value)).convert(
            "RGB"
        )

    if degradation == "gaussian_noise":
        pixels = np.asarray(rgb_image, dtype=np.float32) / 255.0
        # A private generator makes each image condition reproducible and independent.
        generator = np.random.default_rng(seed + image_index)
        noisy_pixels = np.clip(
            pixels + generator.normal(0.0, parameter_value, size=pixels.shape), 0.0, 1.0
        )
        return Image.fromarray(
            np.rint(noisy_pixels * 255.0).astype(np.uint8), mode="RGB"
        )

    if degradation == "low_light":
        return ImageEnhance.Brightness(rgb_image).enhance(parameter_value).convert("RGB")

    if degradation == "motion_blur":
        kernel_size = int(parameter_value)
        pixels = np.asarray(rgb_image, dtype=np.float32)
        height, width, _ = pixels.shape
        pad = kernel_size // 2
        padded = np.pad(
            pixels,
            ((0, 0), (pad, pad), (0, 0)),
            mode="edge",
        )
        blurred = np.zeros_like(pixels)
        for offset in range(kernel_size):
            blurred += padded[:, offset:offset + width, :]
        blurred /= float(kernel_size)
        return Image.fromarray(
            np.rint(np.clip(blurred, 0.0, 255.0)).astype(np.uint8),
            mode="RGB",
        )

    if degradation == "contrast":
        return ImageEnhance.Contrast(rgb_image).enhance(
            float(parameter_value)
        ).convert("RGB")

    if degradation == "pixelate":
        block_size = int(parameter_value)
        width, height = rgb_image.size
        small_width = max(1, width // block_size)
        small_height = max(1, height // block_size)

        reduced = rgb_image.resize(
            (small_width, small_height),
            resample=Image.Resampling.BILINEAR,
        )
        return reduced.resize(
            (width, height),
            resample=Image.Resampling.NEAREST,
        ).convert("RGB")

    raise ValueError(f"Unsupported degradation {degradation!r}.")
