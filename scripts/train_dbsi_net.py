#!/usr/bin/env python
"""
DBSINet Training Script
========================

Trains a DBSINet model on synthetic data generated from one or more
DWI protocols. Protocols are specified as (bval, bvec) file pairs; SNR
is estimated automatically from the corresponding DWI data files.

Examples
--------
Single protocol:
    python scripts/train_dbsi_net.py \\
        --protocol data.bval data.bvec \\
        --snr 30 \\
        --out checkpoints/ \\
        --epochs 100

Multiple protocols:
    python scripts/train_dbsi_net.py \\
        --protocol p1.bval p1.bvec \\
        --protocol p2.bval p2.bvec \\
        --snr 28 35 \\
        --out checkpoints/ \\
        --epochs 100

Resume from checkpoint:
    python scripts/train_dbsi_net.py \\
        --protocol data.bval data.bvec \\
        --snr 30 \\
        --resume checkpoints/dbsinet_epoch_0050.pt \\
        --out checkpoints/ \\
        --epochs 100
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dbsi_toolbox_net import DBSINet, Trainer


def load_protocol(bval_path: str, bvec_path: str):
    bvals = np.loadtxt(bval_path).astype(np.float32)
    bvecs = np.loadtxt(bvec_path).astype(np.float32)
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        bvecs = bvecs.T
    norms = np.linalg.norm(bvecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    bvecs = bvecs / norms
    return bvals, bvecs


def main():
    parser = argparse.ArgumentParser(description='Train DBSINet')

    parser.add_argument('--protocol', nargs=2, action='append',
                        metavar=('BVAL', 'BVEC'), required=True,
                        help='Protocol bval/bvec files (repeat for multiple)')
    parser.add_argument('--snr', type=float, nargs='+', required=True,
                        help='SNR for each protocol (one value or one per protocol)')
    parser.add_argument('--out', default='./checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n-samples', type=int, default=50_000,
                        help='Synthetic voxels per protocol per epoch')
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--aggregator-dim', type=int, default=512)
    parser.add_argument('--n-res-blocks', type=int, default=4)
    parser.add_argument('--lambda-start', type=float, default=0.5,
                        help='Initial supervised loss weight')
    parser.add_argument('--n-anneal', type=int, default=20,
                        help='Epochs to anneal supervised loss to 0')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save-every', type=int, default=10)

    args = parser.parse_args()

    # ── Load protocols ────────────────────────────────────────────────────
    protocols = []
    snr_list  = args.snr if len(args.snr) > 1 else args.snr * len(args.protocol)
    if len(snr_list) != len(args.protocol):
        parser.error(f'--snr must have 1 value or one per --protocol '
                     f'({len(args.protocol)} protocols provided)')

    for (bval_f, bvec_f), snr in zip(args.protocol, snr_list):
        bvals, bvecs = load_protocol(bval_f, bvec_f)
        protocols.append((bvals, bvecs, float(snr)))
        print(f"  Protocol: {bval_f} / {bvec_f}  "
              f"(N={len(bvals)}, b_max={bvals.max():.0f}, SNR={snr:.1f})")

    # ── Build model ───────────────────────────────────────────────────────
    model = DBSINet(
        embed_dim      = args.embed_dim,
        aggregator_dim = args.aggregator_dim,
        n_res_blocks   = args.n_res_blocks,
    )
    print(f"\n  {model}")

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(
        model                  = model,
        protocols              = protocols,
        n_samples_per_protocol = args.n_samples,
        batch_size             = args.batch_size,
        lr                     = args.lr,
        n_epochs               = args.epochs,
        lambda_start           = args.lambda_start,
        n_anneal_epochs        = args.n_anneal,
        device                 = args.device,
        output_dir             = args.out,
        save_every             = args.save_every,
        seed                   = args.seed,
    )
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
