from __future__ import annotations
from typing import Tuple

import torch


@torch.no_grad()
def project_stiefel_tangent(u: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Project Euclidean gradient onto the tangent space of St(D, K)."""
    sym = 0.5 * (u.transpose(-1, -2) @ grad + grad.transpose(-1, -2) @ u)
    return grad - u @ sym


@torch.no_grad()
def qr_retraction(u: torch.Tensor) -> torch.Tensor:
    """Retract a tangent step back to the Stiefel manifold via QR."""
    q, r = torch.linalg.qr(u, mode='reduced')
    diag = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    diag = torch.where(diag == 0, torch.ones_like(diag), diag)
    q = q * diag.unsqueeze(-2)
    return q


@torch.no_grad()
def stiefel_sgd_step(u: torch.Tensor, grad: torch.Tensor, lr: float) -> torch.Tensor:
    tangent_grad = project_stiefel_tangent(u, grad)
    return qr_retraction(u - lr * tangent_grad)


@torch.no_grad()
def stiefel_batch_sgd_step(u: torch.Tensor, grad: torch.Tensor, lr: float) -> torch.Tensor:
    """Apply a Stiefel SGD step to a batch of class-specific bases.

    Args:
        u: Tensor of shape (C, D, K)
        grad: Tensor of shape (C, D, K)
        lr: Learning rate
    """
    updated = []
    for uc, gc in zip(u, grad):
        updated.append(stiefel_sgd_step(uc, gc, lr))
    return torch.stack(updated, dim=0)


@torch.no_grad()
def prox_nonnegative_l1_step(x: torch.Tensor, grad: torch.Tensor, lr: float, lam: float) -> torch.Tensor:
    """One proximal step for a nonnegative L1-regularized variable."""
    return torch.clamp(x - lr * (grad + lam), min=0.0)
