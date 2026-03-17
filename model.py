"""
model.py  -  Original SPDNetwork modernised for PyTorch >= 2.0
==============================================================

Changes from the original:
    * Uses nn.Module / nn.Parameter properly  (no bare Variables)
    * Loads initial weights from .mat files OR random initialisation
    * Uses updated spd_net_util  (torch.linalg, @staticmethod, ctx)
    * update_para() uses pure-torch Stiefel retraction
    * Compatible with standard PyTorch training loops
"""

import os
import torch
import torch.nn as nn
import numpy as np
import spd_net_util as util

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class SPDNetwork(nn.Module):
    """
    Original SPDNet architecture:
        Input (d?xd?) -> BiMap(d?) -> ReEig -> BiMap(d?) -> ReEig -> BiMap(d?) -> LogEig -> FC

    Parameters
    ----------
    dims        : list of 4 ints [d?, d?, d?, d?]  e.g. [400, 200, 100, 50]
    n_classes   : number of output classes
    init_dir    : path to directory with w_1.mat, w_2.mat, w_3.mat, fc.mat
                  (if None, uses random orthogonal initialisation)
    epsilon     : ReEig rectification threshold
    """

    def __init__(
        self,
        dims: list[int] = (400, 200, 100, 50),
        n_classes: int = 7,
        init_dir: str | None = None,
        epsilon: float = 1e-4,
    ):
        super().__init__()
        d0, d1, d2, d3 = dims
        self.epsilon = epsilon

        # -- BiMap weights on Stiefel manifold --
        if init_dir is not None and HAS_SCIPY and os.path.isdir(init_dir):
            w1 = torch.from_numpy(sio.loadmat(f"{init_dir}/w_1.mat")["w_1"])
            w2 = torch.from_numpy(sio.loadmat(f"{init_dir}/w_2.mat")["w_2"])
            w3 = torch.from_numpy(sio.loadmat(f"{init_dir}/w_3.mat")["w_3"])
            fc = torch.from_numpy(
                sio.loadmat(f"{init_dir}/fc.mat")["theta"].astype(np.float64)
            )
        else:
            w1 = torch.empty(d0, d1, dtype=torch.float64)
            nn.init.orthogonal_(w1)
            w2 = torch.empty(d1, d2, dtype=torch.float64)
            nn.init.orthogonal_(w2)
            w3 = torch.empty(d2, d3, dtype=torch.float64)
            nn.init.orthogonal_(w3)
            fc = torch.randn(d3 * d3, n_classes, dtype=torch.float64) * 0.01

        self.w_1 = nn.Parameter(w1)
        self.w_2 = nn.Parameter(w2)
        self.w_3 = nn.Parameter(w3)
        self.fc_w = nn.Parameter(fc)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : (B, d?, d?)  batch of SPD matrices

        Returns
        -------
        logits : (B, n_classes)
        """
        B = X.shape[0]

        # -- Layer 1: BiMap + ReEig --
        W1 = self.w_1.unsqueeze(0)                                 # (1, d0, d1)
        X1 = W1.transpose(1, 2) @ X @ W1                           # (B, d1, d1)
        X1 = util.rec_mat(X1, self.epsilon)

        # -- Layer 2: BiMap + ReEig --
        W2 = self.w_2.unsqueeze(0)
        X2 = W2.transpose(1, 2) @ X1 @ W2
        X2 = util.rec_mat(X2, self.epsilon)

        # -- Layer 3: BiMap + LogEig --
        W3 = self.w_3.unsqueeze(0)
        X3 = W3.transpose(1, 2) @ X2 @ W3
        X3 = util.log_mat(X3)

        # -- FC --
        feat = X3.reshape(B, -1)                                   # (B, d3*d3)
        logits = feat @ self.fc_w                                   # (B, n_classes)
        return logits

    @torch.no_grad()
    def update_para(self, lr: float):
        """
        Manual parameter update with Riemannian SGD on Stiefel manifold
        for BiMap weights and standard SGD for FC weights.
        """
        for W in [self.w_1, self.w_2, self.w_3]:
            if W.grad is None:
                continue
            new_W = util.update_para_riemann(W.data, W.grad.data, lr)
            W.data.copy_(new_W)
            W.grad.zero_()

        if self.fc_w.grad is not None:
            self.fc_w.data -= lr * self.fc_w.grad.data
            self.fc_w.grad.zero_()
