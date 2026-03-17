"""
GVSPD-Net  -  Graph-Variate SPD Network
========================================

End-to-end model that:
    1. Takes raw multivariate signals  (B, C, T)
    2. Computes graph-variate SPD matrices  (B*T, C, C)
    3. Passes them through SPDNet layers (BiMap -> ReEig -> ... -> LogEig)
    4. Pools across the temporal dimension
    5. Classifies with a fully-connected + softmax head

Supports:
    * variable depth  (n BiMap/ReEig blocks)
    * temporal pooling strategies:  mean | attention | last
    * Stiefel-aware optimisation  (StiefelSGD)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spd_layers import BiMapLayer, ReEigLayer, LogEigLayer, StiefelSGD
from graph_variate import GraphVariateTransform, pearson_correlation_matrix, ensure_spd


# =============================================================
#  SPDNet Baseline -- per-trial covariance, no graph-variate
# =============================================================
class SPDNetBaseline(nn.Module):
    """
    Ablation baseline: per-trial covariance -> SPDNet -> FC.

    No graph-variate transform, no temporal dimension.
    Each trial (B, C, T) produces one covariance matrix (B, C, C)
    which goes through BiMap/ReEig/LogEig then a classifier.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        bimap_dims: list[int] | None = None,
        epsilon: float = 1e-4,
        spd_eps: float = 1e-3,
    ):
        super().__init__()
        self.spd_eps = spd_eps
        self.lr_support = None

        if bimap_dims is None:
            bimap_dims = [max(n_channels // 2, 4), max(n_channels // 4, 3)]

        layers = []
        d_in = n_channels
        for d_out in bimap_dims:
            layers.append(BiMapLayer(d_in, d_out))
            layers.append(ReEigLayer(epsilon=epsilon))
            d_in = d_out
        layers.append(LogEigLayer())
        self.spd_backbone = nn.Sequential(*layers)

        feat_dim = d_in * d_in
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T) -> logits (B, n_classes)
        """
        # per-trial covariance
        x_c = x - x.mean(dim=2, keepdim=True)
        cov = torch.bmm(x_c, x_c.transpose(1, 2)) / (x.shape[2] - 1)
        I = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
        cov = cov + self.spd_eps * I  # ensure SPD

        log_spd = self.spd_backbone(cov)
        feats = log_spd.reshape(x.shape[0], -1)
        return self.fc(feats)

    def make_optimiser(self, lr=1e-2, lr_support=None):
        return StiefelSGD(self, lr=lr)


# =============================================================
#  GVSPD-Net
# =============================================================


class GVSPDNet(nn.Module):
    """
    Parameters
    ----------
    n_channels     : number of EEG / signal channels  (= spatial dim C)
    n_classes      : number of output classes
    bimap_dims     : list of output dimensions for successive BiMap layers,
                     e.g. [16, 10]  means  C -> 16 -> 10
    epsilon        : ReEig rectification threshold
    node_fun       : 'corr' or 'sqd'  (graph-variate node function)
    support_mode   : 'data'           -- Pearson correlation from batch (no params)
                     'log_cholesky'   -- learnable, L L^T with log-diagonal
                     'matrix_exp'     -- learnable, expm(S) with free symmetric S
                     'eigenvalue'     -- learnable, U softplus(lambda) U^T (U on Stiefel)
                     'bimap'          -- learnable, W^T C? W (W on Stiefel, C? fixed)
    temporal_pool  : 'mean' | 'attention' | 'last'
    spd_eps        : regularisation eps added to ensure strict PD
    bimap_rank     : only for support_mode='bimap' -- rank of learnable support
    lr_support     : separate learning rate for learnable support params
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        bimap_dims: list[int] | None = None,
        epsilon: float = 1e-4,
        node_fun: str = "corr",
        support_mode: str = "data",
        temporal_pool: str = "mean",
        spd_eps: float = 1e-3,
        bimap_rank: int | None = None,
        lr_support: float | None = None,
        n_windows: int | None = None,
    ):
        super().__init__()
        self.lr_support = lr_support

        # -- graph-variate front-end --
        self.gv = GraphVariateTransform(
            n_channels=n_channels,
            node_fun=node_fun,
            support=support_mode,
            spd_eps=spd_eps,
            bimap_rank=bimap_rank,
            n_windows=n_windows,
        )

        # work out the spatial dim entering the SPDNet backbone
        # (bimap support may reduce it)
        if support_mode == "bimap" and bimap_rank is not None:
            spd_dim = bimap_rank
        else:
            spd_dim = n_channels

        if bimap_dims is None:
            bimap_dims = [max(spd_dim // 2, 4), max(spd_dim // 4, 3)]

        # -- SPDNet backbone --
        layers = []
        d_in = spd_dim
        for d_out in bimap_dims:
            layers.append(BiMapLayer(d_in, d_out))
            layers.append(ReEigLayer(epsilon=epsilon))
            d_in = d_out
        layers.append(LogEigLayer())
        self.spd_backbone = nn.Sequential(*layers)

        self.final_spd_dim = d_in

        # -- temporal pooling --
        self.temporal_pool = temporal_pool
        feat_dim = self.final_spd_dim * self.final_spd_dim
        if temporal_pool == "attention":
            self.attn = nn.Sequential(
                nn.Linear(feat_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

        # -- classifier head --
        self.fc = nn.Linear(feat_dim, n_classes)

    # ---------------------------------------------------------
    def _pool_temporal(self, feats: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        feats : (B*T, D)   flattened log-mapped SPD features
        returns : (B, D)
        """
        D = feats.shape[-1]
        feats = feats.reshape(B, T, D)  # (B, T, D)

        if self.temporal_pool == "mean":
            return feats.mean(dim=1)

        elif self.temporal_pool == "last":
            return feats[:, -1, :]

        elif self.temporal_pool == "attention":
            scores = self.attn(feats).squeeze(-1)  # (B, T)
            weights = torch.softmax(scores, dim=1)  # (B, T)
            return (weights.unsqueeze(-1) * feats).sum(dim=1)

        else:
            raise ValueError(f"Unknown temporal pooling: {self.temporal_pool}")

    # ---------------------------------------------------------
    def _run_backbone_chunked(self, spd: torch.Tensor, chunk_size: int = 512):
        """Run SPD backbone in chunks to avoid cusolver batch limits."""
        BT = spd.shape[0]
        if BT <= chunk_size:
            return self.spd_backbone(spd)

        outputs = []
        for start in range(0, BT, chunk_size):
            chunk = spd[start : start + chunk_size]
            outputs.append(self.spd_backbone(chunk))
        return torch.cat(outputs, dim=0)

    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T)  raw multivariate signal

        Returns
        -------
        logits : (B, n_classes)
        """
        # 1. graph-variate -> batched SPD matrices
        spd, meta = self.gv(x)                        # (B*T, C, C)
        B, T = meta["B"], meta["T"]

        # 2. SPDNet backbone in chunks (cusolver batched eigh has limits)
        log_spd = self._run_backbone_chunked(spd)      # (B*T, d, d)

        # 3. vectorise
        BT, d, _ = log_spd.shape
        feats = log_spd.reshape(BT, d * d)             # (B*T, D)

        # 4. temporal pooling
        pooled = self._pool_temporal(feats, B, T)       # (B, D)

        # 5. classify
        logits = self.fc(pooled)                          # (B, n_classes)
        return logits

    # ---------------------------------------------------------
    def make_optimiser(self, lr: float = 1e-2, lr_support: float | None = None):
        """Return a StiefelSGD that respects manifold geometry."""
        ls = lr_support if lr_support is not None else self.lr_support
        return StiefelSGD(self, lr=lr, lr_support=ls)

    def set_fixed_support(self, C: torch.Tensor):
        """Set the frozen SPD support (only for support_mode='fixed')."""
        self.gv.set_fixed_support(C)


# -----------------------------------------------------------------
#  Convenience factory for BCI-IV-2a
# -----------------------------------------------------------------
def build_gvspd_bci2a(
    n_channels: int = 22,
    n_classes: int = 4,
    bimap_dims: list[int] | None = None,
    temporal_pool: str = "mean",
) -> GVSPDNet:
    """Pre-configured GVSPD-Net for 22-channel motor-imagery BCI data."""
    if bimap_dims is None:
        bimap_dims = [16, 10]
    return GVSPDNet(
        n_channels=n_channels,
        n_classes=n_classes,
        bimap_dims=bimap_dims,
        epsilon=1e-4,
        node_fun="corr",
        support_mode="data",
        temporal_pool=temporal_pool,
        spd_eps=1e-3,
    ).float()


# -----------------------------------------------------------------
if __name__ == "__main__":
    # quick smoke test -- all support modes, both dtypes
    for dt in [torch.float32, torch.float64]:
        torch.manual_seed(42)
        B, C, T, n_cls = 4, 22, 64, 4
        x = torch.randn(B, C, T, dtype=dt)

        for mode in ["data", "fixed", "log_cholesky", "matrix_exp",
                      "eigenvalue", "bimap"]:
            rank = 16 if mode == "bimap" else None
            model = GVSPDNet(
                n_channels=C, n_classes=n_cls,
                bimap_dims=None,
                support_mode=mode,
                bimap_rank=rank,
            ).to(dtype=dt)

            if mode == "fixed":
                model.set_fixed_support(torch.eye(C, dtype=dt))

            logits = model(x)
            loss = F.cross_entropy(logits.float(), torch.randint(0, n_cls, (B,)))
            loss.backward()

            opt = model.make_optimiser(lr=0.01)
            opt.step()

            print(f"  {str(dt):14s}  {mode:15s}  logits={logits.shape}  "
                  f"loss={loss.item():.4f}  ok")
