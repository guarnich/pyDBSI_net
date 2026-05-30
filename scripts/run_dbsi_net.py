#!/usr/bin/env python
"""
DBSINet Inference Script
=========================

Runs a trained DBSINet checkpoint on a 4D DWI NIfTI volume and saves
parameter maps compatible with pyDBSI output format.

Example
-------
    python scripts/run_dbsi_net.py \\
        --dwi  data.nii.gz \\
        --bval data.bval \\
        --bvec data.bvec \\
        --mask mask.nii.gz \\
        --ckpt checkpoints/dbsinet_final.pt \\
        --out  results/ \\
        --fiber-threshold 0.15
"""

import argparse
import numpy as np
import nibabel as nib
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dbsi_toolbox_net import load_checkpoint, run_inference, save_maps


def main():
    parser = argparse.ArgumentParser(description='DBSINet Inference')
    parser.add_argument('--dwi',  required=True, help='4D DWI NIfTI')
    parser.add_argument('--bval', required=True)
    parser.add_argument('--bvec', required=True)
    parser.add_argument('--mask', default=None, help='Brain mask NIfTI')
    parser.add_argument('--ckpt', required=True, help='Trained checkpoint (.pt)')
    parser.add_argument('--out',  required=True, help='Output directory')
    parser.add_argument('--fiber-threshold', type=float, default=0.15)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\n  Loading {args.dwi} ...")
    img    = nib.load(args.dwi)
    data   = img.get_fdata().astype(np.float32)
    affine = img.affine

    bvals = np.loadtxt(args.bval).astype(np.float32)
    bvecs = np.loadtxt(args.bvec).astype(np.float32)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    bvecs = bvecs / norms

    if args.mask:
        mask = nib.load(args.mask).get_fdata().astype(bool)
    else:
        mean_vol = np.mean(data, axis=-1)
        thresh   = np.percentile(mean_vol[mean_vol > 0], 10)
        mask     = mean_vol > thresh
        print("  No mask provided — using threshold mask")

    print(f"  Data shape : {data.shape}")
    print(f"  Mask voxels: {mask.sum():,}")
    print(f"  b_max      : {bvals.max():.0f} s/mm²")
    print(f"  N volumes  : {len(bvals)}")

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n  Loading checkpoint {args.ckpt} ...")
    model, ck_meta = load_checkpoint(args.ckpt, device=args.device)

    # Protocol compatibility check
    from dbsi_toolbox_net.inference import _check_protocol_compatibility
    _check_protocol_compatibility(ck_meta, bvals)

    print(f"  {model}")
    print(f"  Trained for {ck_meta.get('epoch', '?')} epochs")

    # ── Inference ─────────────────────────────────────────────────────────
    results = run_inference(
        model            = model,
        data             = data,
        bvals            = bvals,
        bvecs            = bvecs,
        mask             = mask,
        fiber_threshold  = args.fiber_threshold,
        batch_size       = args.batch_size,
        device           = args.device,
    )

    # ── Save maps ─────────────────────────────────────────────────────────
    save_maps(results, affine, args.out)
    print("\n  Done.")


if __name__ == '__main__':
    main()
