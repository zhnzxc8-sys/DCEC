"""Offline assertions for deterministic image degradations."""

import numpy as np
from PIL import Image

from image_degradations import apply_degradation


def _make_test_image() -> Image.Image:
    """Create a small deterministic RGB image without external data or randomness."""
    height, width = 9, 11
    pixels = np.fromfunction(
        lambda y, x, channel: (17 * y + 13 * x + 31 * channel) % 256,
        (height, width, 3),
        dtype=int,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _assert_image_properties(image: Image.Image, reference: Image.Image) -> None:
    assert image.size == reference.size
    assert image.mode == "RGB"


def main() -> None:
    image = _make_test_image()

    clean = apply_degradation(image, "clean", severity=0)
    _assert_image_properties(clean, image)
    assert np.array_equal(np.asarray(clean), np.asarray(image))

    for degradation in ("jpeg", "gaussian_blur", "low_light", "gaussian_noise"):
        for severity in (1, 2, 3):
            degraded = apply_degradation(
                image, degradation, severity, seed=2026, image_index=4
            )
            _assert_image_properties(degraded, image)

    noise_a = apply_degradation(image, "gaussian_noise", 2, seed=2026, image_index=4)
    noise_b = apply_degradation(image, "gaussian_noise", 2, seed=2026, image_index=4)
    noise_other_image = apply_degradation(
        image, "gaussian_noise", 2, seed=2026, image_index=5
    )
    assert np.array_equal(np.asarray(noise_a), np.asarray(noise_b))
    assert not np.array_equal(np.asarray(noise_a), np.asarray(noise_other_image))

    try:
        apply_degradation(image, "invalid_degradation", 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid degradation must raise ValueError.")

    try:
        apply_degradation(image, "jpeg", 4)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid severity must raise ValueError.")

    print("Image degradation assertions passed.")


if __name__ == "__main__":
    main()
