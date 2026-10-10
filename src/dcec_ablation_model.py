"""Fixed-structure DCEC v1 ablation architectures."""

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as functional
from torch import nn


ARCHITECTURES = ("residual_only", "quality_no_gate", "full")
ABLATION_VARIANTS = ("residual_only", "quality_no_gate", "no_sem", "no_rank")


def ablation_spec(variant: str) -> dict[str, float | str]:
    """Return the pre-registered architecture and loss weights for one variant."""
    specs: dict[str, dict[str, float | str]] = {
        "residual_only": {
            "architecture": "residual_only",
            "lambda_sem": 1.0,
            "lambda_rank": 1.0,
            "lambda_res": 0.001,
        },
        "quality_no_gate": {
            "architecture": "quality_no_gate",
            "lambda_sem": 1.0,
            "lambda_rank": 1.0,
            "lambda_res": 0.001,
        },
        "no_sem": {
            "architecture": "full",
            "lambda_sem": 0.0,
            "lambda_rank": 1.0,
            "lambda_res": 0.001,
        },
        "no_rank": {
            "architecture": "full",
            "lambda_sem": 1.0,
            "lambda_rank": 0.0,
            "lambda_res": 0.001,
        },
    }
    if variant not in specs:
        raise ValueError(f"Unsupported ablation variant {variant!r}.")
    return dict(specs[variant])


def validation_primary_score(condition_metrics: Sequence[Mapping[str, float]]) -> float:
    """Apply DCEC v1's fixed five-condition validation selection rule."""
    if len(condition_metrics) != 5:
        raise ValueError("Validation primary score requires exactly five conditions.")
    return sum(
        0.5 * (metrics["i2t_R@1"] + metrics["t2i_R@1"])
        for metrics in condition_metrics
    ) / len(condition_metrics)


class DCECAblation(nn.Module):
    """Residual-only, quality-without-gate, or full DCEC v1 architecture."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        if variant not in ARCHITECTURES:
            raise ValueError(f"variant must be one of {ARCHITECTURES}, got {variant!r}.")
        self.variant = variant
        self.embedding_dim = 512
        self.quality_dim = 5

        if variant != "residual_only":
            self.quality_mlp = nn.Sequential(
                nn.Linear(5, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
            )
        else:
            self.quality_mlp = None

        residual_input_dim = 512 if variant == "residual_only" else 512 + 64
        self.residual_corrector = nn.Sequential(
            nn.Linear(residual_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )
        if variant == "full":
            self.gate_mlp = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
        else:
            self.gate_mlp = None

        final_residual_layer = self.residual_corrector[-1]
        assert isinstance(final_residual_layer, nn.Linear)
        nn.init.zeros_(final_residual_layer.weight)
        nn.init.zeros_(final_residual_layer.bias)

    def forward(
        self, z_degraded: torch.Tensor, quality_descriptor: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Correct one batch of frozen CLIP image embeddings."""
        if z_degraded.ndim != 2 or z_degraded.shape[-1] != self.embedding_dim:
            raise ValueError("z_degraded must have shape [B, 512].")

        if self.variant == "residual_only":
            residual_input = z_degraded
            gate = torch.ones(
                (z_degraded.shape[0], 1), dtype=z_degraded.dtype, device=z_degraded.device
            )
        else:
            if quality_descriptor is None or quality_descriptor.shape != (
                z_degraded.shape[0],
                self.quality_dim,
            ):
                raise ValueError("quality_descriptor must have shape [B, 5].")
            assert self.quality_mlp is not None
            condition = self.quality_mlp(quality_descriptor)
            residual_input = torch.cat((z_degraded, condition), dim=1)
            if self.variant == "full":
                assert self.gate_mlp is not None
                gate = self.gate_mlp(condition)
            else:
                gate = torch.ones(
                    (z_degraded.shape[0], 1),
                    dtype=z_degraded.dtype,
                    device=z_degraded.device,
                )

        delta_z = self.residual_corrector(residual_input)
        z_corrected = functional.normalize(z_degraded + gate * delta_z, p=2, dim=1)
        return {"z_corrected": z_corrected, "delta_z": delta_z, "gate": gate}

    def count_trainable_parameters(self) -> int:
        """Return the number of trainable parameters in this ablation only."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
