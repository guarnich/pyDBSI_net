"""
DBSI Differentiable Forward Model (PyTorch)
============================================

Implements the 2-ISO DBSI signal model in fully differentiable PyTorch.
This module is the physical core of the neural pipeline: it maps estimated
biophysical parameters to a predicted DWI signal, enabling gradient-based
optimization via a physics-informed reconstruction loss.

2-ISO Signal Model
------------------
    S(b, g) / S₀ = FF · exp(−b · D_app(g, v, AD, RD))
                 + RF · exp(−b · D_res)
                 + NRF · exp(−b · D_nonrf)

where:
    D_app(g, v, AD, RD) = RD + (AD − RD) · cos²θ,   cos θ = dot(g, v)
    D_res   = 0.10×10⁻³ mm²/s  (fixed — Wang et al. 2011)
    D_nonrf ∈ [0.3×10⁻³, 3.5×10⁻³] mm²/s  (estimated per voxel)

Why D_res is fixed
------------------
The restricted pool (ADC ≤ 0.3e-3) has a centroid consistently near
0.10e-3 mm²/s in normal and inflammatory tissue (Wang 2011; Ye 2020).
At b_max = 2000 s/mm², the signal difference between D_res = 0.08e-3
and D_res = 0.15e-3 is < 0.3% of S₀ — indistinguishable at in-vivo SNR.
Fixing D_res removes one degree of freedom from the severely ill-posed
inverse problem without any measurable cost in accuracy.

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590-3601.
Ye Z, et al. (2020). NeuroImage, 221:117228.
"""

import torch


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

D_RES_FIXED  = 0.10e-3   # mm²/s — restricted pool ADC (fixed, Wang 2011)
THRESH_RES   = 0.3e-3    # mm²/s — RF / NRF boundary
ISO_ADC_MAX  = 3.5e-3    # mm²/s — iso_grid upper bound (must match pyDBSI)

# Physiological bounds — used both here and in model.py output heads
RD_MIN      = 0.05e-3   # mm²/s
RD_MAX      = 3.0e-3    # mm²/s
AD_MAX      = 3.5e-3    # mm²/s
D_NONRF_MIN = THRESH_RES
D_NONRF_MAX = ISO_ADC_MAX


def dbsi_forward_2iso(
    ff:        torch.Tensor,   # (B,)
    rf:        torch.Tensor,   # (B,)
    nrf:       torch.Tensor,   # (B,)
    ad:        torch.Tensor,   # (B,)   mm²/s
    rd:        torch.Tensor,   # (B,)   mm²/s
    d_nonrf:   torch.Tensor,   # (B,)   mm²/s
    fiber_dir: torch.Tensor,   # (B, 3) unit vector
    bvals:     torch.Tensor,   # (N,)   s/mm²
    bvecs:     torch.Tensor,   # (N, 3)
) -> torch.Tensor:             # (B, N)  predicted S/S₀
    """
    Differentiable DBSI 2-ISO forward model.

    All tensors must be on the same device. bvals and bvecs describe the
    acquisition protocol; the remaining tensors describe per-voxel tissue.

    Physical constraints (enforced by model.py, not here):
        ff + rf + nrf = 1,  ff/rf/nrf ≥ 0   (softmax)
        AD ≥ RD ≥ 0                          (parameterization)
        fiber_dir is a unit vector, z ≥ 0   (L2 norm + sign flip)

    At b = 0: exp(0·D) = 1 for any D → S = FF + RF + NRF = 1.
    b=0 measurements contribute zero gradient to diffusivity estimation;
    they only constrain the sum-to-one normalization (already enforced by
    softmax). This is physically correct.
    """
    # cos²θ[b_vox, vol] = (fiber_dir[b_vox] · bvecs[vol])²
    cos_t = torch.einsum('bi,ni->bn', fiber_dir, bvecs)   # (B, N)
    cos2  = cos_t.pow(2)                                   # (B, N)

    # Apparent diffusivity for fiber compartment
    D_app = rd.unsqueeze(1) + (ad - rd).unsqueeze(1) * cos2   # (B, N)

    b = bvals.unsqueeze(0)   # (1, N)

    S_fiber = ff.unsqueeze(1)  * torch.exp(-b * D_app)
    S_rf    = rf.unsqueeze(1)  * torch.exp(-b * D_RES_FIXED)
    S_nrf   = nrf.unsqueeze(1) * torch.exp(-b * d_nonrf.unsqueeze(1))

    return S_fiber + S_rf + S_nrf   # (B, N)


def reconstruct_adc_iso(
    rf:      torch.Tensor,   # (B,)
    nrf:     torch.Tensor,   # (B,)
    d_nonrf: torch.Tensor,   # (B,)
) -> torch.Tensor:           # (B,)
    """
    Mean isotropic ADC consistent with pyDBSI channel 8.

        ADC_iso = (RF · D_res + NRF · D_nonrf) / (RF + NRF)
    """
    denom = rf + nrf + 1e-10
    return (rf * D_RES_FIXED + nrf * d_nonrf) / denom
