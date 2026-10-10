"""Fully offline component tests for DCEC v1."""

import math

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, ImageEnhance

from dcec_dataset import deterministic_degradation_choice, validate_train_validation_path
from dcec_losses import compute_dcec_losses
from dcec_model import DCEC
from dcec_quality import extract_quality_descriptor
from image_degradations import apply_degradation


def _checkerboard_image(size: int = 16) -> Image.Image:
    values = np.indices((size, size)).sum(axis=0) % 2 * 255
    rgb = np.stack((values, values, values), axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _expect_value_error(function) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Expected ValueError was not raised.")


def main() -> None:
    image = _checkerboard_image()
    descriptor = extract_quality_descriptor(image)
    repeated_descriptor = extract_quality_descriptor(image)
    assert descriptor.shape == (5,)
    assert descriptor.dtype == torch.float32
    assert torch.isfinite(descriptor).all()
    assert torch.equal(descriptor, repeated_descriptor)

    darker_image = ImageEnhance.Brightness(image).enhance(0.25)
    darker_descriptor = extract_quality_descriptor(darker_image)
    assert darker_descriptor[0] < descriptor[0]
    blurred_image = apply_degradation(image, "gaussian_blur", severity=2)
    blurred_descriptor = extract_quality_descriptor(blurred_image)
    assert not torch.allclose(descriptor[2:4], blurred_descriptor[2:4])

    generator = torch.Generator().manual_seed(2026)
    z_degraded = torch.randn(4, 512, generator=generator)
    z_clean = torch.randn(4, 512, generator=generator)
    text_features = torch.randn(20, 512, generator=generator)
    assert not z_degraded.requires_grad
    assert not z_clean.requires_grad
    assert not text_features.requires_grad

    dcec = DCEC()
    assert dcec.count_trainable_parameters() == 283_713
    outputs = dcec(z_degraded, torch.stack([descriptor] * 4))
    assert outputs["z_corrected"].shape == (4, 512)
    assert outputs["delta_z"].shape == (4, 512)
    assert outputs["gate"].shape == (4, 1)
    assert torch.all((outputs["gate"] >= 0) & (outputs["gate"] <= 1))
    assert torch.allclose(
        outputs["z_corrected"].norm(dim=1), torch.ones(4), atol=1e-6
    )
    assert torch.allclose(
        outputs["z_corrected"], functional.normalize(z_degraded, p=2, dim=1), atol=1e-6
    )

    losses = compute_dcec_losses(
        z_corrected=outputs["z_corrected"],
        z_clean=z_clean,
        text_features=text_features,
        delta_z=outputs["delta_z"],
        scale=torch.tensor(2.0),
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total_loss"].backward()
    assert any(
        parameter.grad is not None for parameter in dcec.parameters() if parameter.requires_grad
    )
    assert z_degraded.grad is None
    assert z_clean.grad is None
    assert text_features.grad is None

    first_choice = deterministic_degradation_choice(2026, epoch=0, image_index=7)
    assert first_choice == deterministic_degradation_choice(2026, epoch=0, image_index=7)
    alternatives = {
        deterministic_degradation_choice(2026, epoch=epoch, image_index=index)
        for epoch in range(1, 5)
        for index in range(1, 5)
    }
    assert any(choice != first_choice for choice in alternatives)

    _expect_value_error(lambda: validate_train_validation_path("test.parquet"))
    _expect_value_error(lambda: validate_train_validation_path("/data/test-00000.parquet"))
    _expect_value_error(lambda: validate_train_validation_path("C:\\data\\test.parquet"))
    assert validate_train_validation_path("/data/train-00000.parquet").name == "train-00000.parquet"

    print("DCEC component assertions passed.")


if __name__ == "__main__":
    main()
