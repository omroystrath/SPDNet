"""
spd_net_util.py  -  Modernised SPD manifold utilities
=====================================================

Updated from the original SPDNet codebase to use:
    * torch.linalg.svd / torch.linalg.eigh  instead of torch.svd
    * @staticmethod + ctx  pattern instead of legacy Function style
    * torch.linalg.qr  instead of np.linalg.qr
    * Pure PyTorch (no numpy in forward/backward paths)
"""

import torch
import numpy as np
from torch.autograd import Function

EPS_SVD = 1e-7


def _compute_K_batched(eigenvalues: torch.Tensor) -> torch.Tensor:
    """K_ij = 1/(sigma_i - sigma_j) with K_ii = 0, safe for near-degenerate eigenvalues."""
    s1 = eigenvalues.unsqueeze(2)
    s2 = eigenvalues.unsqueeze(1)
    diff = s1 - s2
    safe = diff.clone()
    safe[safe.abs() < EPS_SVD] = EPS_SVD
    K = 1.0 / safe
    mask = torch.eye(eigenvalues.shape[1], device=eigenvalues.device, dtype=torch.bool)
    K[:, mask] = 0.0
    return K


class RecFunction(Function):
    """
    Eigenvalue rectification for batched SPD matrices.
    Forward:   X -> U max(eps, ?) U^T
    Uses torch.linalg.eigh (symmetric -> real eigenvalues, ascending).
    """

    @staticmethod
    def forward(ctx, X: torch.Tensor, eps: float = 1e-4):
        X = 0.5 * (X + X.transpose(-2, -1))
        eigvals, eigvecs = torch.linalg.eigh(X)           # ascending
        rectified = eigvals.clamp(min=eps)
        indicator = (eigvals >= eps).to(X.dtype)

        result = eigvecs @ torch.diag_embed(rectified) @ eigvecs.transpose(-2, -1)

        ctx.save_for_backward(eigvecs, eigvals)
        ctx.rectified = rectified
        ctx.indicator = indicator
        return result

    @staticmethod
    def backward(ctx, grad_output):
        U, S = ctx.saved_tensors
        rect_S = ctx.rectified
        Q = ctx.indicator
        Ut = U.transpose(-2, -1)

        dLdC = 0.5 * (grad_output + grad_output.transpose(-2, -1))

        dLdU = 2.0 * dLdC @ U @ torch.diag_embed(rect_S)
        dLdS_inner = Ut @ dLdC @ U
        dLdS = torch.diag_embed(Q) @ dLdS_inner
        dLdS_diag = torch.diagonal(dLdS, dim1=-2, dim2=-1)

        K = _compute_K_batched(S)
        tmp = K.transpose(-2, -1) * (Ut @ dLdU)
        tmp = 0.5 * (tmp + tmp.transpose(-2, -1)) + torch.diag_embed(dLdS_diag)
        grad = U @ tmp @ Ut
        return grad, None  # None for eps


class LogFunction(Function):
    """
    Matrix logarithm for batched SPD matrices.
    Forward:   X -> U log(?) U^T
    """

    @staticmethod
    def forward(ctx, X: torch.Tensor):
        X = 0.5 * (X + X.transpose(-2, -1))
        eigvals, eigvecs = torch.linalg.eigh(X)
        log_eigvals = torch.log(eigvals.clamp(min=1e-7))
        inv_eigvals = 1.0 / eigvals.clamp(min=1e-7)

        result = eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-2, -1)

        ctx.save_for_backward(eigvecs, eigvals, log_eigvals, inv_eigvals)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        U, S, logS, invS = ctx.saved_tensors
        Ut = U.transpose(-2, -1)

        dLdC = 0.5 * (grad_output + grad_output.transpose(-2, -1))

        dLdU = 2.0 * dLdC @ U @ torch.diag_embed(logS)
        dLdS_inner = Ut @ dLdC @ U
        dLdS = torch.diag_embed(invS) @ dLdS_inner
        dLdS_diag = torch.diagonal(dLdS, dim1=-2, dim2=-1)

        K = _compute_K_batched(S)
        tmp = K.transpose(-2, -1) * (Ut @ dLdU)
        tmp = 0.5 * (tmp + tmp.transpose(-2, -1)) + torch.diag_embed(dLdS_diag)
        grad = U @ tmp @ Ut
        return grad


# ---------------------------------------------------------
#  Convenience wrappers
# ---------------------------------------------------------
def rec_mat(X: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Rectification: X -> U max(eps, ?) U^T  (batched)."""
    return RecFunction.apply(X, eps)


def log_mat(X: torch.Tensor) -> torch.Tensor:
    """Matrix logarithm: X -> U log(?) U^T  (batched)."""
    return LogFunction.apply(X)


# ---------------------------------------------------------
#  Stiefel manifold operations  (pure PyTorch)
# ---------------------------------------------------------
def cal_riemann_grad(W: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
    """
    Project Euclidean gradient onto the tangent space of the Stiefel manifold.

    Parameters
    ----------
    W     : (d_in, d_out)  current weight on St(d_out, d_in)
    egrad : (d_in, d_out)  Euclidean gradient dL/dW

    Returns
    -------
    rgrad : (d_in, d_out)  Riemannian gradient
    """
    WtU = W.T @ egrad
    sym = 0.5 * (WtU + WtU.T)
    return egrad - W @ sym


def cal_retraction(W: torch.Tensor, rgrad: torch.Tensor, lr: float) -> torch.Tensor:
    """
    Retraction on the Stiefel manifold via QR decomposition.

    Y = W - lr * rgrad
    Q, R = qr(Y)
    W_new = Q * sign(diag(R))
    """
    Y = W - lr * rgrad
    Q, R = torch.linalg.qr(Y, mode="reduced")
    sign_R = torch.sign(torch.diag(R))
    sign_R[sign_R == 0] = 1.0
    return Q * sign_R.unsqueeze(0)


def update_para_riemann(W: torch.Tensor, egrad: torch.Tensor, lr: float) -> torch.Tensor:
    """Full Riemannian update: project gradient + retract."""
    rgrad = cal_riemann_grad(W, egrad)
    return cal_retraction(W, rgrad, lr)
