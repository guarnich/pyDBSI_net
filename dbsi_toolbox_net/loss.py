"""
Hybrid Physics-Informed Loss for DBSINet
=========================================

Total loss:
    L = L_physics + λ(epoch) · L_supervised

L_physics: MSE between predicted and observed S/S₀ (self-supervised, always active)
L_supervised: parameter-space MSE on synthetic ground truth (annealed to 0)

λ(epoch) = λ_start · max(0, 1 − epoch/n_anneal)²   — quadratic decay

Fiber direction loss uses angular distance:
    dist = 1 − |cos θ| = 1 − |dot(v_pred, v_true)|
so that antipodal directions (+v, −v) both score 0 (physically equivalent).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .forward import dbsi_forward_2iso, AD_MAX, RD_MAX, D_NONRF_MAX


_AD_NORM    = float(AD_MAX)
_RD_NORM    = float(RD_MAX)
_DN_NORM    = float(D_NONRF_MAX)


class DBSILoss(nn.Module):
    """
    Hybrid physics-informed + supervised loss with annealing.

    Parameters
    ----------
    lambda_start : float
        Initial supervised weight. Default: 0.5.
    n_anneal_epochs : int
        Epochs over which λ decays to 0. Default: 20.
    w_frac, w_diff, w_dnonrf, w_dir : float
        Relative weights of supervised sub-terms. Defaults: 1.0, 0.5, 0.3, 0.2.
    """

    def __init__(
        self,
        lambda_start:    float = 0.5,
        n_anneal_epochs: int   = 20,
        w_frac:          float = 1.0,
        w_diff:          float = 0.5,
        w_dnonrf:        float = 0.3,
        w_dir:           float = 0.2,
    ):
        super().__init__()
        self.lambda_start    = lambda_start
        self.n_anneal_epochs = n_anneal_epochs
        self.w_frac          = w_frac
        self.w_diff          = w_diff
        self.w_dnonrf        = w_dnonrf
        self.w_dir           = w_dir

    def get_lambda(self, epoch: int) -> float:
        if epoch >= self.n_anneal_epochs:
            return 0.0
        return self.lambda_start * (1.0 - epoch / self.n_anneal_epochs) ** 2

    def forward(
        self,
        pred:   dict,
        s_obs:  torch.Tensor,          # (B, N) observed S/S₀
        bvals:  torch.Tensor,          # (N,)
        bvecs:  torch.Tensor,          # (N, 3)
        epoch:  int = 0,
        gt:     Optional[dict] = None,
    ) -> tuple:
        """
        Parameters
        ----------
        pred   : dict — DBSINet output
        s_obs  : (B, N) observed normalized signal
        bvals  : (N,) protocol b-values (same device as pred)
        bvecs  : (N, 3) protocol gradient vectors
        epoch  : current epoch (controls λ)
        gt     : ground-truth parameter dict (only from synthetic data)

        Returns
        -------
        loss_total : scalar Tensor
        loss_dict  : dict[str, float]  for logging
        """
        # ── Physics loss ──────────────────────────────────────────────────
        s_pred = dbsi_forward_2iso(
            pred['ff'], pred['rf'], pred['nrf'],
            pred['ad'], pred['rd'], pred['d_nonrf'],
            pred['fiber_dir'], bvals, bvecs,
        )
        loss_physics = F.mse_loss(s_pred, s_obs)

        loss_dict  = {'physics': loss_physics.item()}
        loss_total = loss_physics

        # ── Supervised regularization (annealed) ──────────────────────────
        lam = self.get_lambda(epoch)
        if lam > 0.0 and gt is not None:
            loss_sup = self._supervised(pred, gt)
            loss_total = loss_total + lam * loss_sup
            loss_dict['supervised'] = loss_sup.item()
        else:
            loss_dict['supervised'] = 0.0

        loss_dict['lambda'] = lam
        loss_dict['total']  = loss_total.item()
        return loss_total, loss_dict

    def _supervised(self, pred: dict, gt: dict) -> torch.Tensor:
        # Fractions
        p_frac = torch.stack([pred['ff'], pred['rf'], pred['nrf']], dim=1)
        g_frac = torch.stack([gt['ff'],   gt['rf'],   gt['nrf']],  dim=1)
        l_frac = F.mse_loss(p_frac, g_frac)

        # Diffusivities (normalized to [0,1])
        l_ad  = F.mse_loss(pred['ad'] / _AD_NORM, gt['ad'] / _AD_NORM)
        l_rd  = F.mse_loss(pred['rd'] / _RD_NORM, gt['rd'] / _RD_NORM)

        # D_nonrf
        l_dn  = F.mse_loss(pred['d_nonrf'] / _DN_NORM, gt['d_nonrf'] / _DN_NORM)

        # Fiber direction: antipodally symmetric angular distance
        cos   = (pred['fiber_dir'] * gt['fiber_dir']).sum(dim=-1)  # (B,)
        l_dir = (1.0 - cos.abs()).mean()

        return (self.w_frac   * l_frac
                + self.w_diff * (l_ad + l_rd)
                + self.w_dnonrf * l_dn
                + self.w_dir  * l_dir)
