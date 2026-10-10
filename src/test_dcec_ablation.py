"""Fully offline unit tests for pre-registered DCEC ablations."""

import torch
import torch.nn.functional as functional

from dcec_ablation_model import DCECAblation, ablation_spec, validation_primary_score
from dcec_dataset import validate_train_validation_path
from dcec_losses import compute_dcec_losses


def _expect_value_error(callback) -> None:
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Expected ValueError was not raised.")


def main() -> None:
    generator = torch.Generator().manual_seed(2026)
    z_degraded = torch.randn(4, 512, generator=generator)
    z_clean = torch.randn(4, 512, generator=generator)
    text_features = torch.randn(20, 512, generator=generator)
    quality = torch.randn(4, 5, generator=generator)
    assert not z_degraded.requires_grad
    assert not z_clean.requires_grad
    assert not text_features.requires_grad

    expected_parameters = {
        "residual_only": 262_912,
        "quality_no_gate": 281_600,
        "full": 283_713,
    }
    models = {}
    for variant, expected_count in expected_parameters.items():
        model = DCECAblation(variant)
        models[variant] = model
        outputs = model(z_degraded, None if variant == "residual_only" else quality)
        assert outputs["z_corrected"].shape == (4, 512)
        assert outputs["delta_z"].shape == (4, 512)
        assert outputs["gate"].shape == (4, 1)
        assert model.count_trainable_parameters() == expected_count
        assert torch.allclose(outputs["z_corrected"].norm(dim=1), torch.ones(4), atol=1e-6)
        assert torch.allclose(
            outputs["z_corrected"], functional.normalize(z_degraded, p=2, dim=1), atol=1e-6
        )
        if variant == "full":
            assert torch.all((outputs["gate"] >= 0) & (outputs["gate"] <= 1))
            assert any("gate_mlp" in name for name, _ in model.named_parameters())
        else:
            assert torch.equal(outputs["gate"], torch.ones(4, 1))
            assert not any("gate" in name for name, _ in model.named_parameters())

    outputs = models["full"](z_degraded, quality)
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
        parameter.grad is not None
        for parameter in models["full"].parameters()
        if parameter.requires_grad
    )
    assert z_degraded.grad is None
    assert z_clean.grad is None
    assert text_features.grad is None

    no_sem = ablation_spec("no_sem")
    no_rank = ablation_spec("no_rank")
    assert no_sem["architecture"] == "full" and no_sem["lambda_sem"] == 0.0
    assert no_rank["architecture"] == "full" and no_rank["lambda_rank"] == 0.0

    metrics = [
        {"i2t_R@1": 10.0, "t2i_R@1": 20.0},
        {"i2t_R@1": 20.0, "t2i_R@1": 30.0},
        {"i2t_R@1": 30.0, "t2i_R@1": 40.0},
        {"i2t_R@1": 40.0, "t2i_R@1": 50.0},
        {"i2t_R@1": 50.0, "t2i_R@1": 60.0},
    ]
    assert validation_primary_score(metrics) == 35.0

    _expect_value_error(lambda: DCECAblation("invalid"))
    _expect_value_error(lambda: ablation_spec("invalid"))
    _expect_value_error(lambda: validation_primary_score(metrics[:4]))
    _expect_value_error(lambda: validate_train_validation_path("test.parquet"))
    _expect_value_error(lambda: validate_train_validation_path("/data/test-000.parquet"))
    _expect_value_error(lambda: validate_train_validation_path("C:\\data\\test.parquet"))

    print("DCEC ablation assertions passed.")


if __name__ == "__main__":
    main()
