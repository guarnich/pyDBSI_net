"""
DBSINet Trainer
================

Training loop for the protocol-conditioned DBSINet.

Multi-Protocol Strategy
------------------------
At each epoch, training alternates through all registered protocols.
For each protocol, a fresh batch of synthetic signals is generated on the
fly (or from a pre-generated pool). This ensures the network sees all
protocols every epoch and does not overfit to any single acquisition scheme.

The protocol pool is defined as a list of (bvals, bvecs, snr) tuples.
At least one protocol must be provided. For single-protocol training,
pass a list with one entry.

Checkpointing
--------------
Checkpoints are saved every `save_every` epochs and at the end of training.
Each checkpoint stores:
    - model state dict
    - optimizer state dict
    - scheduler state dict
    - epoch number
    - training config (hyperparameters)
    - loss history
    - protocol metadata (bvals, bvecs for all registered protocols)

The protocol metadata is critical: at inference time, the checkpoint must
carry the protocol used at training to verify that the model was trained
on a compatible acquisition scheme (or to confirm protocol-conditioned
generalization).
"""

import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional

from .model   import DBSINet
from .loss    import DBSILoss
from .dataset import generate_samples, SyntheticDBSIDataset


# Protocol type: (bvals, bvecs, snr)
Protocol = Tuple[np.ndarray, np.ndarray, float]


class Trainer:
    """
    Training manager for DBSINet.

    Parameters
    ----------
    model : DBSINet
    protocols : list of (bvals, bvecs, snr)
        At least one protocol required. For each protocol, synthetic data
        is generated on-the-fly every epoch.
    n_samples_per_protocol : int
        Synthetic voxels generated per protocol per epoch. Default: 50_000.
    batch_size : int
        Mini-batch size. Default: 2048.
    lr : float
        Initial learning rate. Default: 3e-4.
    weight_decay : float
        AdamW weight decay. Default: 1e-5.
    n_epochs : int
        Total training epochs. Default: 100.
    lambda_start : float
        Initial supervised loss weight. Default: 0.5.
    n_anneal_epochs : int
        Epochs for supervised loss annealing. Default: 20.
    device : str or torch.device
        Training device. Default: 'cuda' if available, else 'cpu'.
    output_dir : str
        Directory for checkpoints and loss history. Default: './checkpoints'.
    save_every : int
        Save checkpoint every N epochs. Default: 10.
    seed : int
        Global random seed. Default: 42.
    num_workers : int
        DataLoader workers. Default: 0.
    """

    def __init__(
        self,
        model:                    DBSINet,
        protocols:                List[Protocol],
        n_samples_per_protocol:   int   = 50_000,
        batch_size:               int   = 2048,
        lr:                       float = 3e-4,
        weight_decay:             float = 1e-5,
        n_epochs:                 int   = 100,
        lambda_start:             float = 0.5,
        n_anneal_epochs:          int   = 20,
        device:                   str   = 'auto',
        output_dir:               str   = './checkpoints',
        save_every:               int   = 10,
        seed:                     int   = 42,
        num_workers:              int   = 0,
    ):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.model     = model.to(self.device)
        self.protocols = protocols
        self.n_samples = n_samples_per_protocol
        self.batch_sz  = batch_size
        self.n_epochs  = n_epochs
        self.out_dir   = output_dir
        self.save_every = save_every
        self.seed       = seed
        self.num_workers = num_workers

        os.makedirs(output_dir, exist_ok=True)

        self.loss_fn = DBSILoss(
            lambda_start=lambda_start,
            n_anneal_epochs=n_anneal_epochs,
        ).to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Cosine annealing with warm restarts: T_0=n_epochs so one full cycle
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )

        self.history = {'train_total': [], 'train_physics': [],
                        'train_supervised': [], 'lr': []}

        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n{'='*60}")
        print(f"  DBSINet Trainer")
        print(f"{'='*60}")
        print(f"  Model parameters : {model.n_parameters:,}")
        print(f"  Protocols        : {len(protocols)}")
        print(f"  Samples/protocol : {n_samples_per_protocol:,}")
        print(f"  Batch size       : {batch_size}")
        print(f"  Epochs           : {n_epochs}")
        print(f"  Device           : {self.device}")
        print(f"  Output dir       : {output_dir}")
        print(f"{'='*60}\n")

    # ─────────────────────────────────────────────────────────────────────────
    def _make_loader(self, protocol_idx: int, epoch: int) -> DataLoader:
        """
        Generate fresh synthetic data for protocol `protocol_idx` and
        return a DataLoader. Using a new seed each epoch ensures the
        network never sees the same signal twice.
        """
        bvals, bvecs, snr = self.protocols[protocol_idx]
        seed_ep = self.seed + epoch * len(self.protocols) + protocol_idx
        rng  = np.random.default_rng(seed_ep)
        data = generate_samples(bvals, bvecs, snr, self.n_samples, rng=rng)
        ds   = SyntheticDBSIDataset(data, bvals, bvecs, supervised=True)
        return DataLoader(
            ds, batch_size=self.batch_sz, shuffle=True,
            num_workers=self.num_workers, pin_memory=(self.device.type == 'cuda'),
            drop_last=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        totals = {'total': 0.0, 'physics': 0.0, 'supervised': 0.0}
        n_batches = 0

        for p_idx in range(len(self.protocols)):
            loader = self._make_loader(p_idx, epoch)
            for batch in loader:
                signal = batch['signal'].to(self.device)    # (B, N)
                bvals  = batch['bvals'][0].to(self.device)  # (N,)  same for all in batch
                bvecs  = batch['bvecs'][0].to(self.device)  # (N, 3)
                gt     = {k: v.to(self.device)
                          for k, v in batch['gt'].items()}

                pred = self.model(signal, bvals, bvecs)
                loss, ld = self.loss_fn(pred, signal, bvals, bvecs, epoch, gt)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                totals['total']      += ld['total']
                totals['physics']    += ld['physics']
                totals['supervised'] += ld['supervised']
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    # ─────────────────────────────────────────────────────────────────────────
    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Run the full training loop.

        Parameters
        ----------
        resume_from : str or None
            Path to a checkpoint to resume from. If None, trains from scratch.
        """
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self._load_checkpoint(resume_from)
            print(f"  Resuming from epoch {start_epoch}\n")

        for epoch in range(start_epoch, self.n_epochs):
            t0 = time.time()
            metrics = self._train_one_epoch(epoch)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            self.history['train_total'].append(metrics['total'])
            self.history['train_physics'].append(metrics['physics'])
            self.history['train_supervised'].append(metrics['supervised'])
            self.history['lr'].append(lr)

            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:4d}/{self.n_epochs}  "
                  f"loss={metrics['total']:.5f}  "
                  f"phys={metrics['physics']:.5f}  "
                  f"sup={metrics['supervised']:.5f}  "
                  f"λ={self.loss_fn.get_lambda(epoch):.3f}  "
                  f"lr={lr:.2e}  "
                  f"t={elapsed:.1f}s")

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch + 1)

        self._save_checkpoint(self.n_epochs, final=True)
        print(f"\n  Training complete. Final checkpoint saved to {self.out_dir}")

    # ─────────────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, final: bool = False) -> None:
        tag  = 'final' if final else f'epoch_{epoch:04d}'
        path = os.path.join(self.out_dir, f'dbsinet_{tag}.pt')

        # Store protocol metadata so inference can verify compatibility
        protocol_meta = [
            {'bvals': bv.tolist(), 'bvecs': bvec.tolist(), 'snr': snr}
            for bv, bvec, snr in self.protocols
        ]

        torch.save({
            'epoch':         epoch,
            'model_state':   self.model.state_dict(),
            'optim_state':   self.optimizer.state_dict(),
            'sched_state':   self.scheduler.state_dict(),
            'history':       self.history,
            'protocols':     protocol_meta,
            'model_config': {
                'embed_dim':      self.model.embed_dim,
                'aggregator_dim': self.model.aggregator_dim,
                'n_res_blocks':   self.model.n_res_blocks,
                'n_phi_layers':   self.model.n_phi_layers,
                'dropout':        self.model.dropout,
            },
        }, path)

    def _load_checkpoint(self, path: str) -> int:
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck['model_state'])
        self.optimizer.load_state_dict(ck['optim_state'])
        self.scheduler.load_state_dict(ck['sched_state'])
        self.history = ck.get('history', self.history)
        return ck.get('epoch', 0)
