# pyDBSI_net
<<<<<<< HEAD

**Physics-Informed Neural DBSI Parameter Estimation — Protocol-Conditioned Deep Sets Architecture**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`pyDBSI_net` is a physics-informed neural network implementation of Diffusion Basis Spectrum Imaging (DBSI) that extends the classical pyDBSI toolbox with a learned parameter estimation approach.

### Key differences from pyDBSI

| | pyDBSI | pyDBSI_net |
|---|---|---|
| **Estimation** | NNLS + grid search | Deep Sets MLP |
| **Protocol** | Fixed design matrix (AD=1.5e-3, RD=0.5e-3) | Protocol-conditioned (any b-values/vectors) |
| **Fiber direction** | argmax of NNLS weights | Explicit output head |
| **Speed (inference)** | ~1 ms/voxel (CPU, Numba) | ~0.1 ms/voxel (GPU) |
| **Uncertainty** | None | Planned (v2) |
| **Output** | 11 channels | 12 channels (+fiber direction) |

### Architecture: Deep Sets

Each DWI measurement is treated as an independent set element:

```
token_i = (S_i/S₀,  b_i/b_max,  gx_i,  gy_i,  gz_i)  ∈ ℝ⁵
```

A shared MLP φ encodes each token; mean pooling produces a protocol-invariant
voxel embedding; a residual MLP ρ maps it to biophysical parameters.

This design:
- Works with **any N** (any number of DWI volumes)
- Works with **any b-values and gradient scheme**
- Is trained once and applied to new protocols without retraining

### Loss: Hybrid Physics-Informed

```
L = L_physics + λ(epoch) · L_supervised
```

- **L_physics**: MSE between predicted and observed S/S₀ (always active)
- **L_supervised**: parameter-space MSE on synthetic ground truth (annealed to 0 over 20 epochs)
- Training is fully **synthetic** — no labelled real data required

## Installation

```bash
git clone https://github.com/guarnich/pyDBSI_toolbox_net
cd pyDBSI_toolbox_net
pip install .
```

## Quick Start

### Training

```bash
# Single protocol
python scripts/train_dbsi_net.py \
    --protocol data.bval data.bvec \
    --snr 30 \
    --out checkpoints/ \
    --epochs 100

# Multiple protocols (protocol-conditioned training)
python scripts/train_dbsi_net.py \
    --protocol p1.bval p1.bvec \
    --protocol p2.bval p2.bvec \
    --snr 28 35 \
    --out checkpoints/ \
    --epochs 100
```

### Inference

```bash
python scripts/run_dbsi_net.py \
    --dwi  data.nii.gz \
    --bval data.bval \
    --bvec data.bvec \
    --mask mask.nii.gz \
    --ckpt checkpoints/dbsinet_final.pt \
    --out  results/
```

### Python API

```python
from dbsi_toolbox_net import DBSINet, Trainer, load_checkpoint, run_inference
import numpy as np

# Build and train
model   = DBSINet(embed_dim=256, aggregator_dim=512)
trainer = Trainer(model, protocols=[(bvals, bvecs, snr)], n_epochs=100)
trainer.train()

# Inference
model, _ = load_checkpoint('checkpoints/dbsinet_final.pt')
results  = run_inference(model, data, bvals, bvecs, mask)
```

## Output Maps (12 channels)

| Channel | Name | Notes |
|---------|------|-------|
| 0 | `fiber_fraction` | FF |
| 1 | `restricted_fraction` | RF |
| 2 | `hindered_fraction_NaN` | Always NaN (2-ISO) |
| 3 | `water_fraction_NaN` | Always NaN (2-ISO) |
| 4 | `nonrestricted_fraction` | NRF = HF + WF |
| 5 | `axial_diffusivity` | NaN if FF ≤ threshold |
| 6 | `radial_diffusivity` | NaN if FF ≤ threshold |
| 7 | `fiber_fa` | NaN if FF ≤ threshold |
| 8 | `mean_iso_adc` | Weighted centroid ADC |
| 9 | `fiber_dir_x` | Explicit fiber direction |
| 10 | `fiber_dir_y` | |
| 11 | `fiber_dir_z` | |

## Model Architecture

```
Input: S/S₀ ∈ ℝᴺ,  bvals ∈ ℝᴺ,  bvecs ∈ ℝᴺˣ³

  Build tokens: (S_i, b_norm_i, gx_i, gy_i, gz_i)  shape (N, 5)
        ↓
  φ [shared MLP, N×(5→256)]
        ↓
  Mean pooling  →  ℝ²⁵⁶
        ↓
  ρ [Linear(256→512) + 4 ResBlocks]
        ↓
  Bottleneck [512→256]
        ↓
  ┌─────────────┬──────────────┬───────────┬──────────────┐
  │ frac_head   │ diff_head    │ dnonrf    │ dir_head     │
  │ Linear→3    │ Linear→2     │ Linear→1  │ Linear→3     │
  │ softmax     │ sigmoid/     │ sigmoid   │ L2 norm      │
  │ FF,RF,NRF   │ softplus     │ D_nonrf   │ fiber_dir    │
  │             │ RD, AD       │           │              │
  └─────────────┴──────────────┴───────────┴──────────────┘
```

Default model size: ~3.2M parameters.

## Physical Constraints (enforced by architecture)

| Constraint | Implementation |
|-----------|---------------|
| FF + RF + NRF = 1, ≥ 0 | softmax |
| RD ∈ [0.05e-3, 3.0e-3] | sigmoid |
| AD ≥ RD | RD + softplus(Δ) |
| AD ≤ 3.5e-3 | clamp |
| D_nonrf ∈ [0.3e-3, 3.5e-3] | sigmoid |
| fiber_dir unit vector, z ≥ 0 | L2 norm + sign flip |

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- NumPy ≥ 1.20
- NiBabel ≥ 3.2
- tqdm ≥ 4.60
- PyYAML ≥ 6.0

## References

1. Wang Y, et al. (2011). Quantification of increased cellularity during inflammatory demyelination. *Brain*, 134(12), 3590–3601.
2. Zaheer M, et al. (2017). Deep Sets. *NeurIPS* 30.
3. Ye Z, et al. (2020). Improved DBSI. *NeuroImage*, 221, 117228.
=======
>>>>>>> 38654767f7cdac199e8c2f29f57ce47a7dc72f40
