"""
Modern SPD manifold layers for PyTorch (>= 2.0).

Provides:
    - BiMapLayer   : bilinear mapping  W^T X W  on compact Stiefel manifold
    - ReEigLayer   : eigenvalue rectification (non-linearity)
    - LogEigLayer  : matrix logarithm map -> flat Euclidean space

All custom autograd Functions use @staticmethod forward/backward with ctx.
Eigendecomposition uses torch.linalg.eigh (symmetric-specific, ascending order).
"""

import torch
import torch.nn as nn
from torch.autograd import Function

EPS_EIGH = 1e-7  # floor added to eigenvalue gaps to avoid 1/0 in K matrix


# -----------------------------------------------------------------
#  Helper: stable K-matrix   K_ij = 1 / (sigma_i ? sigma_j)  with safe diagonal
# -----------------------------------------------------------------
def _compute_K(eigenvalues: torch.Tensor) -> torch.Tensor:
    """
    eigenvalues : (B, d)   ascending eigenvalues from eigh
    returns     : (B, d, d)  K matrix with K_ii = 0
    """
    s1 = eigenvalues.unsqueeze(2)          # (B, d, 1)
    s2 = eigenvalues.unsqueeze(1)          # (B, 1, d)
    diff = s1 - s2                          # (B, d, d)
    # clamp small gaps to avoid inf
    safe = diff.clone()
    safe[safe.abs() < EPS_EIGH] = EPS_EIGH
    K = 1.0 / safe
    # zero the diagonal
    mask = torch.eye(eigenvalues.shape[1], device=eigenvalues.device,
                     dtype=torch.bool).unsqueeze(0)
    K.masked_fill_(mask, 0.0)
    return K


# -----------------------------------------------------------------
#  ReEig  -  eigenvalue rectification
# -----------------------------------------------------------------
class _ReEigFn(Function):
    """max(eps, eigenvalue) rectification on batched SPD matrices."""

    @staticmethod
    def forward(ctx, X: torch.Tensor, epsilon: float):
        # force exact symmetry (float32 accumulates asymmetry)
        X = 0.5 * (X + X.transpose(-2, -1))
        eigvals, eigvecs = torch.linalg.eigh(X)
        rectified = eigvals.clamp(min=epsilon)
        indicator = (eigvals >= epsilon).to(X.dtype)
        result = eigvecs @ torch.diag_embed(rectified) @ eigvecs.transpose(-2, -1)
        ctx.save_for_backward(eigvecs, eigvals, rectified, indicator)
        return result

    @staticmethod
    def backward(ctx, dLdY):
        U, S, S_rect, Q_diag = ctx.saved_tensors
        # symmetrise incoming gradient
        dLdY = 0.5 * (dLdY + dLdY.transpose(-2, -1))
        Ut = U.transpose(-2, -1)

        # dL/dU
        dLdU = 2.0 * dLdY @ U @ torch.diag_embed(S_rect)
        # dL/dS  (only flows through rectified eigenvalues)
        inner = Ut @ dLdY @ U
        dLdS_full = torch.diag_embed(Q_diag) @ inner
        dLdS_diag = torch.diagonal(dLdS_full, dim1=-2, dim2=-1)

        K = _compute_K(S)
        tmp = K.transpose(-2, -1) * (Ut @ dLdU)
        tmp = 0.5 * (tmp + tmp.transpose(-2, -1)) + torch.diag_embed(dLdS_diag)
        grad = U @ tmp @ Ut
        return grad, None  # None for epsilon


class ReEigLayer(nn.Module):
    """Eigenvalue rectification layer -- the SPDNet non-linearity."""

    def __init__(self, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return _ReEigFn.apply(X, self.epsilon)

    def extra_repr(self):
        return f"epsilon={self.epsilon}"


# -----------------------------------------------------------------
#  LogEig  -  matrix logarithm via eigen-decomposition
# -----------------------------------------------------------------
class _LogEigFn(Function):
    """Log-Euclidean map:  X  ->  U log(?) U^T."""

    @staticmethod
    def forward(ctx, X: torch.Tensor):
        X = 0.5 * (X + X.transpose(-2, -1))
        eigvals, eigvecs = torch.linalg.eigh(X)
        log_eigvals = torch.log(eigvals.clamp(min=1e-7))
        result = eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-2, -1)
        ctx.save_for_backward(eigvecs, eigvals, log_eigvals)
        return result

    @staticmethod
    def backward(ctx, dLdY):
        U, S, logS = ctx.saved_tensors
        dLdY = 0.5 * (dLdY + dLdY.transpose(-2, -1))
        Ut = U.transpose(-2, -1)

        # dL/dU  via log eigenvalues
        dLdU = 2.0 * dLdY @ U @ torch.diag_embed(logS)
        # dL/dS  via  ?^{-1}
        inv_S = torch.diag_embed(1.0 / S.clamp(min=1e-7))
        inner = Ut @ dLdY @ U
        dLdS_full = inv_S @ inner
        dLdS_diag = torch.diagonal(dLdS_full, dim1=-2, dim2=-1)

        K = _compute_K(S)
        tmp = K.transpose(-2, -1) * (Ut @ dLdU)
        tmp = 0.5 * (tmp + tmp.transpose(-2, -1)) + torch.diag_embed(dLdS_diag)
        grad = U @ tmp @ Ut
        return grad


class LogEigLayer(nn.Module):
    """Matrix logarithm layer -- maps SPD manifold to tangent (flat) space."""

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return _LogEigFn.apply(X)


# -----------------------------------------------------------------
#  BiMap  -  bilinear mapping on compact Stiefel manifold
# -----------------------------------------------------------------
class BiMapLayer(nn.Module):
    """
    Bilinear mapping layer:  X_out = W^T X_in W
    W in St(d_out, d_in)  is kept on the compact Stiefel manifold.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        assert d_out <= d_in, "d_out must be <= d_in for dimensionality reduction"
        W = torch.empty(d_in, d_out)
        nn.init.orthogonal_(W)
        self.weight = nn.Parameter(W)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        W = self.weight  # (d_in, d_out)
        Wt = W.T.unsqueeze(0)  # (1, d_out, d_in)
        Wb = W.unsqueeze(0)    # (1, d_in, d_out)
        return Wt @ X @ Wb     # (B, d_out, d_out)

    def extra_repr(self):
        return f"d_in={self.weight.shape[0]}, d_out={self.weight.shape[1]}"


# -----------------------------------------------------------------
#  Manifold-aware SGD optimiser
# -----------------------------------------------------------------
def _collect_stiefel_ids(model: nn.Module) -> set[int]:
    """
    Walk the module tree and collect parameter ids that must be
    optimised on the Stiefel manifold.

    Sources of Stiefel params:
        * BiMapLayer.weight
        * LearnableSPDSupport.stiefel_ids  (eigenvalue / bimap methods)
    """
    ids = set()
    for m in model.modules():
        # BiMap weights
        if isinstance(m, BiMapLayer):
            ids.add(id(m.weight))
        # LearnableSPDSupport may flag some of its own params as Stiefel
        # (imported lazily to avoid circular dependency)
        if hasattr(m, "stiefel_ids"):
            ids |= m.stiefel_ids
    return ids


class StiefelSGD:
    """
    Mini-batch SGD that keeps designated parameters on the compact
    Stiefel manifold  St(d_out, d_in)  via Riemannian gradient projection
    + QR retraction, and does standard Euclidean SGD on everything else.

    Stiefel parameters are detected automatically from:
        * BiMapLayer.weight
        * LearnableSPDSupport  ('eigenvalue' -> eigvecs,  'bimap' -> bimap_W)

    Parameters
    ----------
    model : nn.Module
    lr    : learning rate (same for Stiefel and Euclidean params)
    lr_support : optional separate learning rate for LearnableSPDSupport
                 Euclidean params (log_diag, off_diag, S_raw, raw_eigvals).
                 If None, uses ``lr``.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        lr_support: float | None = None,
    ):
        self.lr = lr
        self.lr_support = lr_support if lr_support is not None else lr

        stiefel_ids = _collect_stiefel_ids(model)

        # also collect ids of Euclidean support params for separate lr
        support_euclid_ids: set[int] = set()
        for m in model.modules():
            if hasattr(m, "stiefel_ids"):  # is a LearnableSPDSupport
                for p in m.parameters():
                    pid = id(p)
                    if pid not in stiefel_ids:
                        support_euclid_ids.add(pid)

        self.stiefel_params: list[nn.Parameter] = []
        self.support_euclid_params: list[nn.Parameter] = []
        self.euclid_params: list[nn.Parameter] = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in stiefel_ids:
                self.stiefel_params.append(p)
            elif pid in support_euclid_ids:
                self.support_euclid_params.append(p)
            else:
                self.euclid_params.append(p)

    def zero_grad(self):
        all_p = self.stiefel_params + self.support_euclid_params + self.euclid_params
        for p in all_p:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        # -- Euclidean params (backbone FC, attention, etc.) --
        for p in self.euclid_params:
            if p.grad is None:
                continue
            p.data -= self.lr * p.grad.data

        # -- Euclidean support params (separate lr) --
        for p in self.support_euclid_params:
            if p.grad is None:
                continue
            p.data -= self.lr_support * p.grad.data

        # -- Stiefel params (BiMap weights + SPD support eigvecs/bimap_W) --
        for W in self.stiefel_params:
            if W.grad is None:
                continue
            egrad = W.grad.data
            # project to tangent space of Stiefel manifold
            sym = W.data.T @ egrad
            sym = 0.5 * (sym + sym.T)
            rgrad = egrad - W.data @ sym
            # retraction via QR
            Y = W.data - self.lr * rgrad
            Q, R = torch.linalg.qr(Y, mode='reduced')
            sign = torch.sign(torch.diag(R))
            sign[sign == 0] = 1.0
            W.data.copy_(Q * sign.unsqueeze(0))
