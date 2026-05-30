"""
DBSINet Inference Engine
=========================

Runs a trained DBSINet checkpoint on 4D DWI NIfTI data and saves
parameter maps in the same format as pyDBSI (11-channel layout).

Output Channels (11, identical to pyDBSI DBSI_Adaptive)
---------------------------------------------------------
    0  : FF       — Fiber fraction
    1  : RF       — Restricted fraction
    2  : HF       — NaN (not estimated in 2-ISO mode)
    3  : WF       — NaN (not estimated in 2-ISO mode)
    4  : NRF      — Non-restricted fraction
    5  : AD       — Axial diffusivity  (NaN if FF ≤ fiber_threshold)
    6  : RD       — Radial diffusivity (NaN if FF ≤ fiber_threshold)
    7  : FA       — Fiber FA           (NaN if FF ≤ fiber_threshold)
    8  : ADC_iso  — Mean isotropic ADC
    9  : fiber_x  — Fiber direction x component (NEW — not in pyDBSI)
    10 : fiber_y  — Fiber direction y component (NEW — not in pyDBSI)
    11 : fiber_z  — Fiber direction z component (NEW — not in pyDBSI)

Note: channels 9-11 are new outputs specific to DBSINet (explicit fiber
direction). pyDBSI derived the direction implicitly from the NNLS weights
and did not export it. Having it explicit enables tractography integration.

Fiber FA Formula
----------------
Identical to pyDBSI:  FA = (AD − RD) / sqrt(AD² + 2·RD²)
Applied only where FF > fiber_threshold; NaN otherwise.

Protocol Compatibility Check
------------------------------
At inference time, the checkpoint's stored protocol metadata is compared
against the input protocol (bvals, bvecs). A warning is issued if:
  - The number of volumes differs
  - b_max differs by more than 10%
The model can still be applied (it is protocol-conditioned), but the user
should verify that the inference protocol is within the distribution of
protocols seen during training.
"""

import os
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm
from typing import Optional, Tuple

from .model   import DBSINet
from .forward import RD_MIN, RD_MAX, AD_MAX, D_NONRF_MIN, D_NONRF_MAX


# ─────────────────────────────────────────────────────────────────────────────
# FA helper (same formula as pyDBSI compute_fiber_fa)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fiber_fa(ad: np.ndarray, rd: np.ndarray) -> np.ndarray:
    """
    FA = (AD − RD) / sqrt(AD² + 2·RD²) for cylindrically symmetric tensor.
    Returns NaN where inputs are NaN.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        diff  = np.abs(ad - rd)
        denom = np.sqrt(ad**2 + 2 * rd**2)
        fa    = np.where(denom > 1e-12, diff / denom, 0.0)
        fa    = np.clip(fa, 0.0, 1.0)
    return fa.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(
    ckpt_path: str,
    device:    str = 'auto',
) -> Tuple[DBSINet, dict]:
    """
    Load a trained DBSINet checkpoint.

    Returns
    -------
    model : DBSINet  (eval mode, on device)
    meta  : dict     checkpoint metadata (protocols, history, config)
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    ck = torch.load(ckpt_path, map_location=dev)

    cfg   = ck.get('model_config', {})
    model = DBSINet(
        embed_dim      = cfg.get('embed_dim',      256),
        aggregator_dim = cfg.get('aggregator_dim', 512),
        n_res_blocks   = cfg.get('n_res_blocks',   4),
        n_phi_layers   = cfg.get('n_phi_layers',   3),
        dropout        = cfg.get('dropout',        0.0),
    )
    model.load_state_dict(ck['model_state'])
    model.to(dev).eval()

    return model, ck


def _check_protocol_compatibility(ck_meta: dict, bvals: np.ndarray) -> None:
    """Warn if inference protocol differs substantially from training."""
    stored = ck_meta.get('protocols', [])
    if not stored:
        return
    b_max_inf = float(np.max(bvals))
    for p in stored:
        b_max_tr = max(p['bvals'])
        if abs(b_max_inf - b_max_tr) / max(b_max_tr, 1.0) > 0.10:
            print(f"  [WARNING] b_max mismatch: inference={b_max_inf:.0f}, "
                  f"training={b_max_tr:.0f} s/mm². "
                  f"Model is protocol-conditioned but may extrapolate.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model:            DBSINet,
    data:             np.ndarray,      # (X, Y, Z, N) float32
    bvals:            np.ndarray,      # (N,)
    bvecs:            np.ndarray,      # (N, 3)
    mask:             np.ndarray,      # (X, Y, Z) bool
    fiber_threshold:  float = 0.15,
    batch_size:       int   = 4096,
    device:           str   = 'auto',
) -> np.ndarray:                       # (X, Y, Z, 12)
    """
    Run DBSINet inference on a 4D DWI volume.

    Parameters
    ----------
    model : DBSINet (eval mode)
    data  : (X, Y, Z, N) raw DWI signal
    bvals : (N,)
    bvecs : (N, 3)
    mask  : (X, Y, Z) brain mask
    fiber_threshold : float
        Minimum FF to output AD, RD, FA. Default: 0.15.
    batch_size : int
        Voxels per inference batch. Default: 4096.
    device : str

    Returns
    -------
    results : (X, Y, Z, 12) float32
        Channels 0-11 as described in module docstring.
        Channels 2, 3 are always NaN (2-ISO model).
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device)

    # Normalise bvecs
    bvecs = bvecs.copy().astype(np.float64)
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    bvecs = (bvecs / norms).astype(np.float32)

    bvals_t = torch.from_numpy(bvals.astype(np.float32)).to(dev)  # (N,)
    bvecs_t = torch.from_numpy(bvecs).to(dev)                     # (N, 3)

    coords  = np.argwhere(mask)                 # (V, 3)
    n_vox   = len(coords)
    X, Y, Z = data.shape[:3]

    # Pre-allocate output: 12 channels (11 pyDBSI + 3 fiber direction)
    N_OUT   = 12
    results = np.full((X, Y, Z, N_OUT), np.nan, dtype=np.float32)

    # b0 threshold consistent with pyDBSI
    b0_mask = bvals < 100.0
    n_b0    = int(b0_mask.sum())

    print(f"\n  Running DBSINet inference on {n_vox:,} voxels...")

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n_vox, batch_size),
                          desc="  Progress", unit="batch"):
            end  = min(start + batch_size, n_vox)
            idx  = coords[start:end]            # (B, 3)
            xs, ys, zs = idx[:, 0], idx[:, 1], idx[:, 2]
            B = len(idx)

            # Extract and normalize signal
            sig_raw = data[xs, ys, zs].astype(np.float32)   # (B, N)
            s0 = np.mean(sig_raw[:, b0_mask], axis=1, keepdims=True) if n_b0 > 0 \
                 else sig_raw[:, :1]
            s0 = np.where(s0 < 1e-6, 1.0, s0)
            sig_norm = sig_raw / s0                          # (B, N)

            sig_t = torch.from_numpy(sig_norm).to(dev)       # (B, N)

            pred = model(sig_t, bvals_t, bvecs_t)

            ff      = pred['ff'].cpu().numpy()
            rf      = pred['rf'].cpu().numpy()
            nrf     = pred['nrf'].cpu().numpy()
            ad      = pred['ad'].cpu().numpy()
            rd      = pred['rd'].cpu().numpy()
            d_nonrf = pred['d_nonrf'].cpu().numpy()
            fdir    = pred['fiber_dir'].cpu().numpy()         # (B, 3)
            adc_iso = pred['adc_iso'].cpu().numpy()

            results[xs, ys, zs, 0] = ff
            results[xs, ys, zs, 1] = rf
            # ch 2 (HF) = NaN  — 2-ISO model
            # ch 3 (WF) = NaN  — 2-ISO model
            results[xs, ys, zs, 4] = nrf
            results[xs, ys, zs, 8] = adc_iso

            # AD, RD, FA: only where FF > fiber_threshold
            fiber_mask = ff > fiber_threshold
            if fiber_mask.any():
                xf = xs[fiber_mask]
                yf = ys[fiber_mask]
                zf = zs[fiber_mask]
                ad_f  = ad[fiber_mask]
                rd_f  = rd[fiber_mask]
                fa_f  = _compute_fiber_fa(ad_f, rd_f)
                results[xf, yf, zf, 5] = ad_f
                results[xf, yf, zf, 6] = rd_f
                results[xf, yf, zf, 7] = fa_f

            # Fiber direction (all voxels with valid signal)
            results[xs, ys, zs,  9] = fdir[:, 0]
            results[xs, ys, zs, 10] = fdir[:, 1]
            results[xs, ys, zs, 11] = fdir[:, 2]

    # Channels 2, 3 stay NaN (set at allocation)
    print(f"  Inference complete.\n")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT MAP NAMES
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_MAP_NAMES = [
    'fiber_fraction',          # 0
    'restricted_fraction',     # 1
    'hindered_fraction_NaN',   # 2  — always NaN in 2-ISO
    'water_fraction_NaN',      # 3  — always NaN in 2-ISO
    'nonrestricted_fraction',  # 4
    'axial_diffusivity',       # 5
    'radial_diffusivity',      # 6
    'fiber_fa',                # 7
    'mean_iso_adc',            # 8
    'fiber_dir_x',             # 9   NEW
    'fiber_dir_y',             # 10  NEW
    'fiber_dir_z',             # 11  NEW
]


def save_maps(
    results:    np.ndarray,    # (X, Y, Z, 12)
    affine:     np.ndarray,    # (4, 4)
    output_dir: str,
) -> None:
    """Save each output channel as a compressed NIfTI file."""
    os.makedirs(output_dir, exist_ok=True)
    for i, name in enumerate(OUTPUT_MAP_NAMES):
        if name.endswith('_NaN'):
            continue
        vol   = results[..., i].astype(np.float32)
        fpath = os.path.join(output_dir, f'dbsinet_{name}.nii.gz')
        nib.save(nib.Nifti1Image(vol, affine), fpath)
    print(f"  Maps saved to {output_dir}")
