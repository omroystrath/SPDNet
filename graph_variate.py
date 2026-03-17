"""
Graph-Variate Signal Analysis (GVSA) for PyTorch.

Given a multivariate signal  X in ?^{B x C x T}  and a stable graph
adjacency (support) matrix  C in ?^{B x C x C}, graph-variate signal
analysis computes

    Theta_{ijt} = c_{ij} * F_V(x_i(t), x_j(t))

for a chosen node-space function F_V.  The result is a 4-D tensor
(B, C, C, T) of instantaneous weighted adjacency matrices --
one SPD matrix per time step when the support is SPD and F_V
produces PSD outputs.

Learnable support parameterisations (all guarantee SPD output):
    * log_cholesky  :  C = L L^T           (L lower-tri, diag = exp(d))
    * matrix_exp    :  C = expm(S)          (S free symmetric)
    * eigenvalue    :  C = U Lambda U^T          (U on Stiefel, Lambda = softplus(lambda))
    * bimap         :  C = W^T C? W + epsI    (W on Stiefel, C? data-derived)

Reference
---------
K. Smith, L. Spyrou, J. Escudero,
"Graph-Variate Signal Analysis", arXiv:1703.06687v4, 2018.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as Func

EPS = 1e-8


# =================================================================
#  Utilities
# =================================================================
def pearson_correlation_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Pearson correlation along the time axis.

    Parameters
    ----------
    x : (B, C, T)

    Returns
    -------
    C : (B, C, C)  symmetric PSD.
    """
    x_c = x - x.mean(dim=2, keepdim=True)
    cov = torch.bmm(x_c, x_c.transpose(1, 2)) / (x.shape[2] - 1)
    std = x_c.pow(2).sum(2).div(x.shape[2] - 1).clamp(min=EPS).sqrt()
    denom = (std.unsqueeze(2) * std.unsqueeze(1)).clamp(min=EPS)
    corr = cov / denom
    return corr.clamp(-1.0, 1.0)


def ensure_spd(M: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Add epsI to make PSD -> strictly SPD."""
    I = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
    return M + eps * I


def _sym(M: torch.Tensor) -> torch.Tensor:
    """Symmetrise a (..., d, d) tensor."""
    return 0.5 * (M + M.transpose(-2, -1))


# =================================================================
#  Learnable SPD support -- four parameterisations
# =================================================================
class LearnableSPDSupport(nn.Module):
    """
    Learnable SPD matrix with a hard SPD constraint.

    Every parameterisation maps unconstrained real parameters to a
    C x C symmetric positive-definite matrix.  All are differentiable
    and work with standard (Euclidean) or manifold-aware optimisers.

    Parameters
    ----------
    n_channels     : matrix dimension C
    method         : 'log_cholesky' | 'matrix_exp' | 'eigenvalue' | 'bimap'
    spd_eps        : diagonal regularisation
    init_data      : (B, C, T) tensor used to compute an initial
                     correlation matrix.  If None, initialise from identity.
    bimap_rank     : output rank for the 'bimap' method  (default = C)

    Attributes exposed for the optimiser
    -------------------------------------
    .stiefel_ids : set of id(p) for parameters that live on Stiefel
                   (only non-empty for 'eigenvalue' and 'bimap')
    """

    _SPD_METHODS = ("log_cholesky", "matrix_exp", "eigenvalue", "bimap")

    def __init__(
        self,
        n_channels: int,
        method: str = "log_cholesky",
        spd_eps: float = 1e-3,
        init_data: torch.Tensor | None = None,
        bimap_rank: int | None = None,
    ):
        super().__init__()
        assert method in self._SPD_METHODS, (
            f"method must be one of {self._SPD_METHODS}, got '{method}'"
        )
        self.n = n_channels
        self.method = method
        self.spd_eps = spd_eps
        self.stiefel_ids: set[int] = set()

        # -- data-derived initialisation --
        if init_data is not None:
            with torch.no_grad():
                C0 = pearson_correlation_matrix(init_data).mean(0)  # (C, C)
                C0 = ensure_spd(C0, spd_eps)
        else:
            C0 = None  # will init from identity

        # -- build parameters per method --
        if method == "log_cholesky":
            self._init_log_cholesky(C0)
        elif method == "matrix_exp":
            self._init_matrix_exp(C0)
        elif method == "eigenvalue":
            self._init_eigenvalue(C0)
        elif method == "bimap":
            self._init_bimap(C0, bimap_rank)

    # ---------------------------------------------------------
    #  (1) Log-Cholesky:  C = L L^T,  diag(L) = exp(d)
    #
    #  Unconstrained -> SPD.  Gradient flows through exp() on
    #  diagonal and linearly through off-diagonal.  Standard
    #  Euclidean optimiser (Adam, SGD) works directly.
    # ---------------------------------------------------------
    def _init_log_cholesky(self, C0: torch.Tensor | None):
        n = self.n
        if C0 is not None:
            L0 = torch.linalg.cholesky(C0)
            d_init = torch.log(torch.diag(L0).clamp(min=1e-12))
            mask = torch.tril(torch.ones(n, n, dtype=torch.bool), diagonal=-1)
            off_init = L0[mask]
        else:
            d_init = torch.zeros(n)
            off_init = torch.zeros(n * (n - 1) // 2)

        self.log_diag = nn.Parameter(d_init)
        self.off_diag = nn.Parameter(off_init)

    def _build_log_cholesky(self, device, dtype) -> torch.Tensor:
        n = self.n
        # build L functionally: diag + strict lower triangle
        diag_vals = torch.exp(self.log_diag)                       # (n,)
        # indices for strict lower triangle
        rows, cols = torch.tril_indices(n, n, offset=-1, device=device)
        # scatter off-diag values into a flat n*n vector, then reshape
        L_flat = torch.zeros(n * n, device=device, dtype=dtype)
        # diagonal entries
        diag_idx = torch.arange(n, device=device) * (n + 1)       # [0, n+1, 2n+2, ...]
        L_flat = L_flat.scatter(0, diag_idx, diag_vals)
        # off-diagonal entries
        off_idx = rows * n + cols
        L_flat = L_flat.scatter(0, off_idx, self.off_diag)
        L = L_flat.reshape(n, n)
        C = L @ L.T + self.spd_eps * torch.eye(n, device=device, dtype=dtype)
        return C

    # ---------------------------------------------------------
    #  (2) Matrix exponential:  C = expm(S),  S free symmetric
    #
    #  expm(symmetric) is always SPD.  S is unconstrained.
    #  Initialise S = logm(C?).  More expensive (eigh in forward)
    #  but the most natural Riemannian parameterisation.
    # ---------------------------------------------------------
    def _init_matrix_exp(self, C0: torch.Tensor | None):
        n = self.n
        if C0 is not None:
            eigvals, eigvecs = torch.linalg.eigh(C0)
            S0 = eigvecs @ torch.diag(torch.log(eigvals.clamp(min=1e-12))) @ eigvecs.T
        else:
            S0 = torch.zeros(n, n)

        self.S_raw = nn.Parameter(S0)

    def _build_matrix_exp(self, device, dtype) -> torch.Tensor:
        S = _sym(self.S_raw)
        eigvals, eigvecs = torch.linalg.eigh(S)
        C = eigvecs @ torch.diag_embed(torch.exp(eigvals)) @ eigvecs.T
        return C

    # ---------------------------------------------------------
    #  (3) Eigenvalue:  C = U diag(softplus(lambda)) U^T
    #      U in St(C, C)  on Stiefel,  lambda unconstrained
    #
    #  Gives direct control over both the eigenvalues (through lambda)
    #  and the eigenbasis (through U).  The Stiefel constraint on
    #  U is respected by StiefelSGD.  softplus(lambda) > 0 ensures SPD.
    # ---------------------------------------------------------
    def _init_eigenvalue(self, C0: torch.Tensor | None):
        n = self.n
        if C0 is not None:
            eigvals, eigvecs = torch.linalg.eigh(C0)
            # invert softplus:  lambda = log(exp(sigma) - 1)
            lam_init = torch.log(torch.exp(eigvals.clamp(min=1e-6)) - 1.0)
            U_init = eigvecs
        else:
            lam_init = torch.zeros(n)
            U_init = torch.eye(n)

        self.eigvecs = nn.Parameter(U_init)
        self.raw_eigvals = nn.Parameter(lam_init)
        self.stiefel_ids.add(id(self.eigvecs))

    def _build_eigenvalue(self, device, dtype) -> torch.Tensor:
        lam = Func.softplus(self.raw_eigvals) + self.spd_eps
        U = self.eigvecs
        return U @ torch.diag(lam) @ U.T

    # ---------------------------------------------------------
    #  (4) BiMap:  C = W^T C? W + epsI
    #      W in St(r, C)  on Stiefel,  C? fixed from data
    #
    #  The SPDNet philosophy applied to the support itself.
    #  W^T (SPD) W  is SPD by Schur complement / congruence.
    #  r < C  simultaneously learns the support AND reduces its
    #  dimension, so the downstream SPDNet sees smaller matrices.
    # ---------------------------------------------------------
    def _init_bimap(self, C0: torch.Tensor | None, rank: int | None):
        n = self.n
        r = rank if rank is not None else n
        assert r <= n, f"bimap_rank ({r}) must be <= n_channels ({n})"
        self.bimap_rank = r

        if C0 is not None:
            self.register_buffer("C0", C0)
        else:
            self.register_buffer("C0", torch.eye(n))

        W = torch.empty(n, r)
        nn.init.orthogonal_(W)
        self.bimap_W = nn.Parameter(W)
        self.stiefel_ids.add(id(self.bimap_W))

    def _build_bimap(self, device, dtype) -> torch.Tensor:
        W = self.bimap_W
        r = self.bimap_rank
        C = W.T @ self.C0 @ W
        C = C + self.spd_eps * torch.eye(r, device=device, dtype=dtype)
        return C

    # ---------------------------------------------------------
    #  Forward  -- dispatch to the active method
    # ---------------------------------------------------------
    def forward(self) -> torch.Tensor:
        """
        Returns
        -------
        C : (d, d) SPD matrix  (d = C for all except bimap where d = r)
        """
        d = next(self.parameters())
        device, dtype = d.device, d.dtype

        if self.method == "log_cholesky":
            return self._build_log_cholesky(device, dtype)
        elif self.method == "matrix_exp":
            return self._build_matrix_exp(device, dtype)
        elif self.method == "eigenvalue":
            return self._build_eigenvalue(device, dtype)
        elif self.method == "bimap":
            return self._build_bimap(device, dtype)
        else:
            raise RuntimeError(f"Unknown method: {self.method}")

    @property
    def support_dim(self) -> int:
        """Spatial dimension of the output SPD support."""
        if self.method == "bimap":
            return self.bimap_rank
        return self.n

    def extra_repr(self):
        info = f"n={self.n}, method='{self.method}', eps={self.spd_eps}"
        if self.method == "bimap":
            info += f", rank={self.bimap_rank}"
        n_free = sum(p.numel() for p in self.parameters())
        info += f", free_params={n_free}"
        return info

    @torch.no_grad()
    def reinit_from_data(self, x: torch.Tensor):
        """
        Re-initialise parameter VALUES in-place from a data batch.

        This copies new values into existing nn.Parameter tensors so that
        parameter identities (and any optimizer references) are preserved.

        Parameters
        ----------
        x : (B, C, T)  data batch used to compute initial correlation.
        """
        C0 = pearson_correlation_matrix(x).mean(0)
        C0 = ensure_spd(C0, self.spd_eps)
        n = self.n

        if self.method == "log_cholesky":
            L0 = torch.linalg.cholesky(C0)
            self.log_diag.data.copy_(torch.log(torch.diag(L0).clamp(min=1e-12)))
            mask = torch.tril(torch.ones(n, n, dtype=torch.bool), diagonal=-1)
            self.off_diag.data.copy_(L0[mask])

        elif self.method == "matrix_exp":
            eigvals, eigvecs = torch.linalg.eigh(C0)
            S0 = eigvecs @ torch.diag(torch.log(eigvals.clamp(min=1e-12))) @ eigvecs.T
            self.S_raw.data.copy_(S0)

        elif self.method == "eigenvalue":
            eigvals, eigvecs = torch.linalg.eigh(C0)
            lam = torch.log(torch.exp(eigvals.clamp(min=1e-6)) - 1.0)
            self.raw_eigvals.data.copy_(lam)
            self.eigvecs.data.copy_(eigvecs)

        elif self.method == "bimap":
            self.C0.copy_(C0)
            # re-orthogonalise W via QR just in case
            Q, R = torch.linalg.qr(self.bimap_W.data, mode="reduced")
            sign = torch.sign(torch.diag(R))
            sign[sign == 0] = 1.0
            self.bimap_W.data.copy_(Q * sign.unsqueeze(0))


# =================================================================
#  Node-space functions  (instantaneous bivariate)
# =================================================================
def graph_variate(
    x: torch.Tensor,
    fun: str = "corr",
    z_normalise: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Instantaneous node-space function tensor  J in R^{BxCxCxT}.
    """
    if z_normalise:
        mu = x.mean(dim=1, keepdim=True)
        sig = x.std(dim=1, keepdim=True, unbiased=True).clamp(min=1e-5)
        x = (x - mu) / sig
        # clamp to prevent float32 overflow in outer product
        x = x.clamp(-10.0, 10.0)

    if fun == "sqd":
        return (x.unsqueeze(2) - x.unsqueeze(1)).pow(2)
    elif fun == "corr":
        d = x - x.mean(dim=2, keepdim=True)
        return d.unsqueeze(2) * d.unsqueeze(1)
    elif fun == "abs":
        d = x - x.mean(dim=2, keepdim=True)
        return (d.unsqueeze(2) * d.unsqueeze(1)).abs()
    else:
        raise ValueError(f"Unknown node function: {fun}")


# =================================================================
#  Graph-Variate Transform  (front-end module)
# =================================================================
class GraphVariateTransform(nn.Module):
    """
    Turn a (B, C, T) signal into batched SPD matrices (B*T, d, d).

    Pipeline
    --------
    1.  Stable support  C  -- from data or learnable  (SPD-constrained)
    2.  Instantaneous node-function tensor  J  (B, C, C, T)
    3.  GVD connectivity  Theta(t) = C (.) J(t)
    4.  Regularise  Theta(t) += epsI
    5.  Reshape -> (B*T, d, d)

    Parameters
    ----------
    n_channels      : signal channels C
    node_fun        : 'corr' | 'sqd' | 'abs'
    support         : 'data' | 'log_cholesky' | 'matrix_exp' |
                      'eigenvalue' | 'bimap'
    spd_eps         : diagonal regularisation
    z_normalise     : z-score channels per time step
    bimap_rank      : rank for the 'bimap' support  (default = C)
    init_from_data  : if True, the first batch initialises learnable support
                      from data correlation  (one-shot, not repeated)
    """

    def __init__(
        self,
        n_channels: int,
        node_fun: str = "corr",
        support: str = "data",
        spd_eps: float = 1e-3,
        z_normalise: bool = True,
        bimap_rank: int | None = None,
        init_from_data: bool = True,
        n_windows: int | None = None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.node_fun = node_fun
        self.support_mode = support
        self.spd_eps = spd_eps
        self.z_normalise = z_normalise
        self.n_windows = n_windows  # None = every time step

        # -- learnable support --
        self.learnable_support: LearnableSPDSupport | None = None
        if support == "fixed":
            # frozen precomputed support -- set via set_fixed_support()
            self.register_buffer(
                "_fixed_support",
                torch.eye(n_channels),
            )
        elif support != "data":
            assert support in LearnableSPDSupport._SPD_METHODS, (
                f"support must be 'data', 'fixed', or one of "
                f"{LearnableSPDSupport._SPD_METHODS}, got '{support}'"
            )
            self._init_from_data = init_from_data
            self._support_initialised = not init_from_data

            self.learnable_support = LearnableSPDSupport(
                n_channels=n_channels,
                method=support,
                spd_eps=spd_eps,
                init_data=None,
                bimap_rank=bimap_rank,
            )

    def set_fixed_support(self, C: torch.Tensor):
        """
        Set a precomputed frozen SPD support matrix.

        Parameters
        ----------
        C : (C, C) SPD matrix -- e.g. correlation from training data.
        """
        assert self.support_mode == "fixed", "set_fixed_support requires support='fixed'"
        C = ensure_spd(C.detach(), self.spd_eps)
        self._fixed_support.copy_(C)

    # ---------------------------------------------------------
    def _maybe_init_support_from_data(self, x: torch.Tensor):
        """One-shot: reinitialise support parameter VALUES from first batch."""
        if self._support_initialised:
            return
        self._support_initialised = True
        self.learnable_support.reinit_from_data(x)

    # ---------------------------------------------------------
    def _compute_support(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, d, d)  SPD support matrix.
        """
        B = x.shape[0]
        if self.support_mode == "data":
            C = pearson_correlation_matrix(x)
            return ensure_spd(C, self.spd_eps)

        if self.support_mode == "fixed":
            return self._fixed_support.unsqueeze(0).expand(B, -1, -1)

        # learnable
        self._maybe_init_support_from_data(x)
        C = self.learnable_support()                        # (d, d)
        return C.unsqueeze(0).expand(B, -1, -1)             # (B, d, d)

    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        spd_batch : (B*T, d, d)  SPD matrices
        meta      : dict with B, T, support, support_dim
        """
        B, C, T = x.shape

        # 1. support
        support = self._compute_support(x)                 # (B, d, d)
        d = support.shape[-1]

        # 2. instantaneous node function
        J = graph_variate(x, fun=self.node_fun,
                          z_normalise=self.z_normalise)     # (B, C, C, T)

        # window averaging: reduce T -> n_windows
        if self.n_windows is not None and self.n_windows < T:
            nw = self.n_windows
            win_size = T // nw
            # trim to exact multiple
            J_trim = J[..., :nw * win_size]                 # (B, C, C, nw*ws)
            J = J_trim.reshape(B, C, C, nw, win_size).mean(dim=-1)  # (B, C, C, nw)
            T = nw

        # if bimap reduced the support dim, project J through the same W
        if d < C:
            W = self.learnable_support.bimap_W              # (C, r)
            # (B, C, C, T) -> project both spatial dims: W^T J W
            J_flat = J.permute(0, 3, 1, 2).reshape(B * T, C, C)
            Wt = W.T.unsqueeze(0)                           # (1, r, C)
            Wb = W.unsqueeze(0)                             # (1, C, r)
            J_red = Wt @ J_flat @ Wb                        # (BT, r, r)
            J = J_red.reshape(B, T, d, d).permute(0, 2, 3, 1)  # (B, d, d, T)

        # 3. Hadamard product -- support (.) J(t)
        theta = support.unsqueeze(-1) * J                   # (B, d, d, T)

        # 4. symmetrise spatial dims + regularise
        theta = 0.5 * (theta + theta.transpose(1, 2))  # swap dim1,dim2 (spatial)
        I = torch.eye(d, device=x.device, dtype=x.dtype)
        theta = theta + self.spd_eps * I.reshape(1, d, d, 1)

        # NaN/Inf guard (float32 can overflow in outer products)
        if torch.is_floating_point(theta) and theta.dtype != torch.float64:
            theta = torch.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
            theta = theta + self.spd_eps * I.reshape(1, d, d, 1)

        # 5. reshape -> (B*T, d, d)
        spd_batch = theta.permute(0, 3, 1, 2).reshape(B * T, d, d)

        meta = {
            "B": B,
            "T": T,
            "support": support.detach(),
            "support_dim": d,
        }
        return spd_batch, meta

    def extra_repr(self):
        parts = [
            f"n_channels={self.n_channels}",
            f"node_fun='{self.node_fun}'",
            f"support='{self.support_mode}'",
        ]
        if self.learnable_support is not None:
            parts.append(f"support_dim={self.learnable_support.support_dim}")
        return ", ".join(parts)
