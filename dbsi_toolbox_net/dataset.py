"""
Synthetic Dataset Generator for DBSINet
========================================

Generates physically realistic DWI signals from the 14 tissue scenarios
used in pyDBSI calibration (Wang 2011; Vavasour 2022). Each sample consists
of a noisy normalized signal S/S₀ and the ground-truth biophysical parameters
that generated it.

This module is intentionally self-contained (no imports from pyDBSI) so that
dbsi_toolbox_net can be installed and used independently.

Protocol-Conditioned Training
------------------------------
Because DBSINet is protocol-conditioned (via the Deep Sets encoder), the
dataset generator can be called with ANY protocol (bvals, bvecs). During
training, one or more protocols can be mixed in the same DataLoader by
returning (signal, bvals, bvecs, gt_params) tuples. The collate_fn handles
protocols with different N by treating each batch as a single protocol.

Multi-Protocol Training Strategy
----------------------------------
Option 1 — Train on one protocol at a time (simpler):
    One DataLoader per protocol, alternate batches during training.

Option 2 — Mixed-protocol batches (recommended for generalization):
    Each batch contains samples from a single protocol (sampled uniformly
    from a pool of known protocols). The DataLoader is re-instantiated
    each epoch with a freshly sampled protocol.

The Trainer in trainer.py implements Option 2.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple

from .forward import D_RES_FIXED, RD_MIN, RD_MAX, AD_MAX, D_NONRF_MIN, D_NONRF_MAX


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_D_FREE    = 3.05e-3    # mm²/s — CSF at 37°C
_D_AX_NOM  = 1.60e-3
_D_AX_STD  = 0.10e-3
_D_RAD_NOM = 0.40e-3
_D_RAD_STD = 0.07e-3


# ─────────────────────────────────────────────────────────────────────────────
# 14 TISSUE SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = {
    'WM_normal': dict(
        f_fiber_mu=0.50, f_fiber_sd=0.05,
        f_cell_mu=0.03,  f_cell_sd=0.015,
        f_hin_mu=0.44,   f_free_mu=0.03,
        d_hin_mu=0.80e-3, d_hin_sd=0.03e-3, weight=1.0),
    'WM_CC': dict(
        f_fiber_mu=0.63, f_fiber_sd=0.05,
        f_cell_mu=0.02,  f_cell_sd=0.01,
        f_hin_mu=0.32,   f_free_mu=0.03,
        d_hin_mu=0.76e-3, d_hin_sd=0.03e-3, weight=1.0),
    'WM_subcortical': dict(
        f_fiber_mu=0.37, f_fiber_sd=0.06,
        f_cell_mu=0.04,  f_cell_sd=0.02,
        f_hin_mu=0.52,   f_free_mu=0.07,
        d_hin_mu=0.81e-3, d_hin_sd=0.04e-3, weight=0.8),
    'GM_cortex': dict(
        f_fiber_mu=0.00, f_fiber_sd=0.00,
        f_cell_mu=0.03,  f_cell_sd=0.01,
        f_hin_mu=0.87,   f_free_mu=0.10,
        d_hin_mu=0.88e-3, d_hin_sd=0.05e-3, weight=2.0),
    'GM_deep': dict(
        f_fiber_mu=0.08, f_fiber_sd=0.04,
        f_cell_mu=0.04,  f_cell_sd=0.015,
        f_hin_mu=0.79,   f_free_mu=0.09,
        d_hin_mu=0.82e-3, d_hin_sd=0.04e-3, weight=1.5),
    'GM_cerebellum': dict(
        f_fiber_mu=0.22, f_fiber_sd=0.06,
        f_cell_mu=0.05,  f_cell_sd=0.02,
        f_hin_mu=0.65,   f_free_mu=0.08,
        d_hin_mu=0.80e-3, d_hin_sd=0.04e-3, weight=1.0),
    'CSF_pure': dict(
        f_fiber_mu=0.00, f_fiber_sd=0.00,
        f_cell_mu=0.00,  f_cell_sd=0.00,
        f_hin_mu=0.02,   f_free_mu=0.98,
        d_hin_mu=0.90e-3, d_hin_sd=0.00e-3, weight=2.0),
    'NAWM': dict(
        f_fiber_mu=0.44, f_fiber_sd=0.05,
        f_cell_mu=0.09,  f_cell_sd=0.03,
        f_hin_mu=0.41,   f_free_mu=0.06,
        d_hin_mu=0.83e-3, d_hin_sd=0.04e-3, weight=0.8),
    'Lesion_active': dict(
        f_fiber_mu=0.17, f_fiber_sd=0.05,
        f_cell_mu=0.40,  f_cell_sd=0.05,
        f_hin_mu=0.30,   f_free_mu=0.13,
        d_hin_mu=1.05e-3, d_hin_sd=0.06e-3, weight=1.2),
    'Lesion_chronic': dict(
        f_fiber_mu=0.15, f_fiber_sd=0.04,
        f_cell_mu=0.08,  f_cell_sd=0.03,
        f_hin_mu=0.45,   f_free_mu=0.32,
        d_hin_mu=1.08e-3, d_hin_sd=0.07e-3, weight=1.0),
    'Lesion_cortical': dict(
        f_fiber_mu=0.04, f_fiber_sd=0.03,
        f_cell_mu=0.17,  f_cell_sd=0.04,
        f_hin_mu=0.65,   f_free_mu=0.14,
        d_hin_mu=0.93e-3, d_hin_sd=0.05e-3, weight=1.2),
    'PV_WM_GM': dict(
        f_fiber_mu=0.26, f_fiber_sd=0.05,
        f_cell_mu=0.04,  f_cell_sd=0.015,
        f_hin_mu=0.62,   f_free_mu=0.08,
        d_hin_mu=0.85e-3, d_hin_sd=0.04e-3, weight=0.75),
    'PV_WM_CSF': dict(
        f_fiber_mu=0.23, f_fiber_sd=0.05,
        f_cell_mu=0.02,  f_cell_sd=0.01,
        f_hin_mu=0.25,   f_free_mu=0.50,
        d_hin_mu=0.84e-3, d_hin_sd=0.04e-3, weight=0.75),
    'PV_GM_CSF': dict(
        f_fiber_mu=0.00, f_fiber_sd=0.00,
        f_cell_mu=0.02,  f_cell_sd=0.01,
        f_hin_mu=0.47,   f_free_mu=0.51,
        d_hin_mu=0.89e-3, d_hin_sd=0.04e-3, weight=1.5),
}

_SC_NAMES  = list(SCENARIOS.keys())
_SC_W      = np.array([SCENARIOS[n]['weight'] for n in _SC_NAMES])
_SC_PROBS  = _SC_W / _SC_W.sum()


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _sample_fractions(sc: dict, rng: np.random.Generator) -> tuple:
    f_fiber = float(np.clip(rng.normal(sc['f_fiber_mu'], sc['f_fiber_sd']), 0.0, 1.0))
    rem = max(0.0, 1.0 - f_fiber)
    f_cell_max = min(sc['f_cell_mu'] + 3.0 * sc['f_cell_sd'], rem)
    f_cell = float(np.clip(rng.normal(sc['f_cell_mu'], sc['f_cell_sd']), 0.0, f_cell_max))
    rem = max(0.0, 1.0 - f_fiber - f_cell)
    total = sc['f_hin_mu'] + sc['f_free_mu']
    if total > 1e-10:
        f_hin  = rem * sc['f_hin_mu'] / total
        f_free = rem * sc['f_free_mu'] / total
    else:
        f_hin = rem; f_free = 0.0
    return f_fiber, f_cell, f_hin, f_free


def _hemisphere_dir(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3)
    v /= np.linalg.norm(v) + 1e-12
    if v[2] < 0:
        v = -v
    return v.astype(np.float32)


def generate_samples(
    bvals:      np.ndarray,
    bvecs:      np.ndarray,
    snr:        float,
    n_samples:  int,
    rng:        Optional[np.random.Generator] = None,
    snr_jitter: float = 0.20,
) -> dict:
    """
    Generate n_samples synthetic DWI voxels for a given protocol.

    Each sample is drawn from a scenario weighted by tissue prevalence.
    Rician noise is added with ±snr_jitter relative variation to improve
    SNR robustness at inference.

    Parameters
    ----------
    bvals : (N,)  b-values in s/mm²
    bvecs : (N, 3) gradient unit vectors
    snr   : float  base SNR (from estimate_snr_robust)
    n_samples : int
    rng   : optional seeded Generator
    snr_jitter : float  relative SNR variation across samples

    Returns
    -------
    dict:
        'signal'      : (n_samples, N) float32  noisy S/S₀
        'signal_clean': (n_samples, N) float32  clean S/S₀
        'ff', 'rf', 'nrf', 'ad', 'rd', 'd_nonrf' : (n_samples,) float32
        'fiber_dir'   : (n_samples, 3) float32
        'scenario'    : list[str]
    """
    if rng is None:
        rng = np.random.default_rng()

    N       = len(bvals)
    b0_mask = bvals < 100.0

    sig_out   = np.zeros((n_samples, N), dtype=np.float32)
    sig_clean = np.zeros((n_samples, N), dtype=np.float32)
    ff_out    = np.zeros(n_samples, dtype=np.float32)
    rf_out    = np.zeros(n_samples, dtype=np.float32)
    nrf_out   = np.zeros(n_samples, dtype=np.float32)
    ad_out    = np.zeros(n_samples, dtype=np.float32)
    rd_out    = np.zeros(n_samples, dtype=np.float32)
    dn_out    = np.zeros(n_samples, dtype=np.float32)
    dir_out   = np.zeros((n_samples, 3), dtype=np.float32)
    sc_list   = []

    sc_idx = rng.choice(len(_SC_NAMES), size=n_samples, p=_SC_PROBS)

    for i in range(n_samples):
        sc = SCENARIOS[_SC_NAMES[sc_idx[i]]]
        sc_list.append(_SC_NAMES[sc_idx[i]])

        f_fiber, f_cell, f_hin, f_free = _sample_fractions(sc, rng)
        d_hin = float(np.clip(rng.normal(sc['d_hin_mu'], sc['d_hin_sd']),
                               D_NONRF_MIN, D_NONRF_MAX))
        d_ax  = float(np.clip(rng.normal(_D_AX_NOM,  _D_AX_STD),  0.8e-3, AD_MAX))
        d_rad = float(np.clip(rng.normal(_D_RAD_NOM, _D_RAD_STD), RD_MIN, RD_MAX))
        if d_ax < d_rad:
            d_ax, d_rad = d_rad, d_ax

        f_nrf = f_hin + f_free
        d_nonrf = float(np.clip(
            (f_hin * d_hin + f_free * _D_FREE) / max(f_nrf, 1e-10),
            D_NONRF_MIN, D_NONRF_MAX,
        ))

        v = _hemisphere_dir(rng)
        cos2  = (bvecs @ v) ** 2
        D_app = d_rad + (d_ax - d_rad) * cos2

        s = (f_fiber * np.exp(-bvals * D_app)
             + f_cell  * np.exp(-bvals * D_RES_FIXED)
             + f_nrf   * np.exp(-bvals * d_nonrf))

        snr_i  = snr * float(rng.uniform(1 - snr_jitter, 1 + snr_jitter))
        sigma  = 1.0 / max(snr_i, 1.0)
        s_noisy = np.sqrt((s + rng.normal(0, sigma, N))**2
                          + rng.normal(0, sigma, N)**2)

        n_b0 = int(b0_mask.sum())
        s0n  = float(np.mean(s_noisy[b0_mask])) if n_b0 > 0 else float(s_noisy[0])
        s0c  = float(np.mean(s[b0_mask]))       if n_b0 > 0 else float(s[0])
        if s0n < 1e-6: s0n = 1.0
        if s0c < 1e-6: s0c = 1.0

        sig_out[i]   = (s_noisy / s0n).astype(np.float32)
        sig_clean[i] = (s       / s0c).astype(np.float32)
        ff_out[i]    = f_fiber
        rf_out[i]    = f_cell
        nrf_out[i]   = f_nrf
        ad_out[i]    = d_ax
        rd_out[i]    = d_rad
        dn_out[i]    = d_nonrf
        dir_out[i]   = v

    return dict(signal=sig_out, signal_clean=sig_clean,
                ff=ff_out, rf=rf_out, nrf=nrf_out,
                ad=ad_out, rd=rd_out, d_nonrf=dn_out,
                fiber_dir=dir_out, scenario=sc_list)


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticDBSIDataset(Dataset):
    """
    PyTorch Dataset wrapping synthetic DBSI signals for a single protocol.

    Parameters
    ----------
    data : dict  — output of generate_samples()
    bvals : (N,) ndarray
    bvecs : (N, 3) ndarray
    supervised : bool  — include ground-truth parameters in each item
    """

    def __init__(self, data: dict,
                 bvals: np.ndarray,
                 bvecs: np.ndarray,
                 supervised: bool = True):
        self.signal = torch.from_numpy(data['signal'])
        self.bvals  = torch.from_numpy(bvals.astype(np.float32))
        self.bvecs  = torch.from_numpy(bvecs.astype(np.float32))
        self.supervised = supervised

        if supervised:
            self.gt = {k: torch.from_numpy(data[k])
                       for k in ('ff','rf','nrf','ad','rd','d_nonrf','fiber_dir')}

    def __len__(self) -> int:
        return len(self.signal)

    def __getitem__(self, idx) -> dict:
        item = {
            'signal': self.signal[idx],
            'bvals':  self.bvals,
            'bvecs':  self.bvecs,
        }
        if self.supervised:
            item['gt'] = {k: v[idx] for k, v in self.gt.items()}
        return item


def make_dataloader(
    bvals:       np.ndarray,
    bvecs:       np.ndarray,
    snr:         float,
    n_samples:   int,
    batch_size:  int,
    supervised:  bool = True,
    seed:        int  = 42,
    num_workers: int  = 0,
) -> DataLoader:
    """
    Build a DataLoader of synthetic voxels for one protocol.
    """
    rng  = np.random.default_rng(seed)
    data = generate_samples(bvals, bvecs, snr, n_samples, rng=rng)
    ds   = SyntheticDBSIDataset(data, bvals, bvecs, supervised=supervised)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True,
                      drop_last=True)
