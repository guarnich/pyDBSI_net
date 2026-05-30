"""
dbsi_toolbox_net — Physics-Informed Neural DBSI Parameter Estimation
=====================================================================

Protocol-conditioned Deep Sets architecture for voxel-wise estimation of
DBSI biophysical parameters from any multi-shell DWI acquisition.

Main components
---------------
DBSINet
    Protocol-conditioned residual MLP using a Deep Sets encoder.
    Each DWI measurement is encoded as a token (S/S₀, b_norm, gx, gy, gz);
    mean pooling produces a protocol-invariant voxel embedding.

DBSILoss
    Hybrid physics-informed + supervised loss with quadratic annealing.

Trainer
    Full training loop with multi-protocol support, checkpointing,
    and cosine learning rate scheduling.

run_inference
    Batched NIfTI inference engine compatible with pyDBSI output layout.

load_checkpoint
    Load a trained checkpoint and return (model, metadata).

Output channels (12)
--------------------
    0  : fiber_fraction
    1  : restricted_fraction
    2  : hindered_fraction      (always NaN — 2-ISO model)
    3  : water_fraction         (always NaN — 2-ISO model)
    4  : nonrestricted_fraction
    5  : axial_diffusivity      (NaN if FF ≤ fiber_threshold)
    6  : radial_diffusivity     (NaN if FF ≤ fiber_threshold)
    7  : fiber_fa               (NaN if FF ≤ fiber_threshold)
    8  : mean_iso_adc
    9  : fiber_dir_x
    10 : fiber_dir_y
    11 : fiber_dir_z

References
----------
Wang Y, et al. (2011). Brain, 134(12):3590-3601.
Zaheer M, et al. (2017). Deep Sets. NeurIPS 30.
"""

__version__ = "1.0.0"
__author__  = "DBSI Toolbox Contributors"

from .model     import DBSINet
from .loss      import DBSILoss
from .trainer   import Trainer
from .inference import run_inference, load_checkpoint, save_maps, OUTPUT_MAP_NAMES
from .dataset   import (generate_samples, SyntheticDBSIDataset,
                         make_dataloader, SCENARIOS)

__all__ = [
    "DBSINet",
    "DBSILoss",
    "Trainer",
    "run_inference",
    "load_checkpoint",
    "save_maps",
    "OUTPUT_MAP_NAMES",
    "generate_samples",
    "SyntheticDBSIDataset",
    "make_dataloader",
    "SCENARIOS",
]
