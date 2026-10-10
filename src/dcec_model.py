"""Lightweight degradation-conditioned embedding correction model."""

import torch
import torch.nn.functional as functional
from torch import nn


class DCEC(nn.Module):
    """Correct frozen CLIP image embeddings using image-only quality descriptors."""

    def __init__(self, embedding_dim: int = 512, quality_dim: int = 5) -> None:
        super().__init__()
        if embedding_dim != 512 or quality_dim != 5:
            raise ValueError("DCEC v1 requires embedding_dim=512 and quality_dim=5.")

        self.embedding_dim = embedding_dim
        self.quality_dim = quality_dim
        self.quality_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.residual_corrector = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Start from the frozen CLIP behavior: the residual is exactly zero.
        final_residual_layer = self.residual_corrector[-1]
        assert isinstance(final_residual_layer, nn.Linear)
        nn.init.zeros_(final_residual_layer.weight)
        nn.init.zeros_(final_residual_layer.bias)

    def forward(
        self, z_degraded: torch.Tensor, quality_descriptor: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return normalized corrected embeddings, residuals, and scalar gates."""
        if z_degraded.ndim != 2 or z_degraded.shape[-1] != self.embedding_dim:
            raise ValueError("z_degraded must have shape [B, 512].")
        if quality_descriptor.ndim != 2 or quality_descriptor.shape != (
            z_degraded.shape[0],
            self.quality_dim,
        ):
            raise ValueError("quality_descriptor must have shape [B, 5].")

        condition = self.quality_mlp(quality_descriptor)
        delta_z = self.residual_corrector(torch.cat((z_degraded, condition), dim=-1))
        gate = self.gate_mlp(condition)
        z_corrected = functional.normalize(z_degraded + gate * delta_z, p=2, dim=-1)
        return {"z_corrected": z_corrected, "delta_z": delta_z, "gate": gate}

    def count_trainable_parameters(self) -> int:
        """Return the number of trainable DCEC parameters only."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
