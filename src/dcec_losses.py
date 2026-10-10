"""Losses for anchoring DCEC to frozen clean CLIP semantics."""

from typing import Any

import torch
import torch.nn.functional as functional


def _normalized(features: torch.Tensor, name: str) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"{name} must be a 2D Tensor.")
    if torch.any(features.norm(dim=1) == 0):
        raise ValueError(f"{name} contains a zero-norm feature.")
    return functional.normalize(features, p=2, dim=1)


def clean_semantic_anchor_loss(
    z_corrected: torch.Tensor, z_clean: torch.Tensor
) -> torch.Tensor:
    """Penalize angular deviation from the frozen clean image embedding."""
    corrected = _normalized(z_corrected, "z_corrected")
    clean = _normalized(z_clean.detach(), "z_clean")
    if corrected.shape != clean.shape:
        raise ValueError("z_corrected and z_clean must have matching shapes.")
    return (1.0 - functional.cosine_similarity(corrected, clean, dim=1)).mean()


def bidirectional_ranking_preservation_loss(
    z_corrected: torch.Tensor,
    z_clean: torch.Tensor,
    text_features: torch.Tensor,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    """Distill clean I2T and T2I similarity distributions into corrected images."""
    corrected = _normalized(z_corrected, "z_corrected")
    clean = _normalized(z_clean.detach(), "z_clean")
    text = _normalized(text_features.detach(), "text_features")
    if corrected.shape != clean.shape or corrected.shape[1] != text.shape[1]:
        raise ValueError("Image and text embeddings must have compatible dimensions.")
    if text.shape[0] != 5 * corrected.shape[0]:
        raise ValueError("DCEC ranking preservation requires exactly five captions per image.")

    scale_tensor = torch.as_tensor(scale, device=corrected.device, dtype=corrected.dtype).detach()
    clean_similarity = scale_tensor * (clean @ text.T)
    corrected_similarity = scale_tensor * (corrected @ text.T)

    def kl_teacher_to_student(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        teacher_distribution = functional.softmax(teacher_logits, dim=1).detach()
        return functional.kl_div(
            functional.log_softmax(student_logits, dim=1),
            teacher_distribution,
            reduction="batchmean",
        )

    i2t_loss = kl_teacher_to_student(clean_similarity, corrected_similarity)
    t2i_loss = kl_teacher_to_student(clean_similarity.T, corrected_similarity.T)
    return 0.5 * (i2t_loss + t2i_loss)


def residual_regularization(delta_z: torch.Tensor) -> torch.Tensor:
    """Penalize unnecessarily large embedding residuals."""
    if delta_z.ndim != 2:
        raise ValueError("delta_z must be a 2D Tensor.")
    return torch.mean(delta_z**2)


def compute_dcec_losses(
    z_corrected: torch.Tensor,
    z_clean: torch.Tensor,
    text_features: torch.Tensor,
    delta_z: torch.Tensor,
    scale: torch.Tensor | float,
    lambda_sem: float = 1.0,
    lambda_rank: float = 1.0,
    lambda_res: float = 0.001,
) -> dict[str, torch.Tensor]:
    """Compute weighted DCEC training loss terms."""
    for name, value in {
        "lambda_sem": lambda_sem,
        "lambda_rank": lambda_rank,
        "lambda_res": lambda_res,
    }.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

    sem_loss = clean_semantic_anchor_loss(z_corrected, z_clean)
    rank_loss = bidirectional_ranking_preservation_loss(
        z_corrected, z_clean, text_features, scale
    )
    res_loss = residual_regularization(delta_z)
    total_loss = lambda_sem * sem_loss + lambda_rank * rank_loss + lambda_res * res_loss
    return {
        "total_loss": total_loss,
        "sem_loss": sem_loss,
        "rank_loss": rank_loss,
        "res_loss": res_loss,
    }
