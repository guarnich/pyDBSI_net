"""
DBSINet — Protocol-Conditioned Deep Sets Architecture
======================================================

Architecture Overview
---------------------
The core challenge of protocol conditioning is that different DWI acquisitions
have different numbers of measurements N and different (b, g) configurations.
A standard MLP with fixed input dimension cannot generalize across protocols.

Solution: Deep Sets (Zaheer et al. 2017)
-----------------------------------------
Each DWI measurement is treated as an independent element of a SET:

    token_i = (S_i/S₀, b_i_norm, gx_i, gy_i, gz_i)  ∈ ℝ⁵

Key insight: the token includes BOTH the signal value AND the acquisition
parameter (b, g) that produced it. The network therefore learns a function
of "signal-in-context", not just signal alone. A gradient direction producing
a highly attenuated signal tells a very different story depending on whether
b = 500 or b = 2000 s/mm².

The Deep Sets architecture has two components:

    φ (token encoder):  shared MLP applied to each token independently
                        ℝ⁵ → ℝ^d_embed

    ρ (set aggregator): MLP applied to the mean-pooled embedding
                        ℝ^d_embed → biophysical parameters

Mean pooling (∑ φ(token_i) / N) is:
    • Permutation invariant   — volume ordering does not matter
    • Protocol invariant      — N can be any integer ≥ 1
    • Differentiable          — gradients flow through pooling

b-value Normalization
---------------------
b-values are normalized by b_max of the protocol before being fed as input:
    b_norm = b / b_max  ∈ [0, 1]

This makes the token encoder parameters independent of the absolute b scale,
allowing a single trained model to generalize to protocols with different
b_max without retraining.

Output Parameterization (Option A — explicit fiber direction)
-------------------------------------------------------------
All physical constraints are enforced by architecture:

    1. Fractions (FF, RF, NRF): softmax → non-negative, sum = 1 exactly
    2. RD: RD_MIN + sigmoid(r) · (RD_MAX − RD_MIN)
    3. AD: RD + softplus(a) · scale, clamped to AD_MAX  →  AD ≥ RD always
    4. D_nonrf: D_NONRF_MIN + sigmoid(d) · (D_NONRF_MAX − D_NONRF_MIN)
    5. fiber_dir: raw ℝ³ → L2 normalize → flip z ≥ 0 convention

Fiber direction (Option A rationale)
-------------------------------------
The fiber direction is estimated explicitly as a 3D unit vector. This allows:
  - The forward model to compute cos²θ exactly in the physics loss
  - Direct inspection of the estimated direction as an output map
  - Extension to multi-fiber (crossing) models in future versions

Alternative B (spherical harmonics input features) would be protocol-invariant
too, but requires a minimum number of directions per shell and loses the
direct relationship between individual measurements and the forward model.
Option A is simpler, more interpretable, and directly integrable.

References
----------
Zaheer M, et al. (2017). Deep Sets. NeurIPS 30.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .forward import RD_MIN, RD_MAX, AD_MAX, D_NONRF_MIN, D_NONRF_MAX


_AD_DELTA_SCALE = 2.0e-3   # mm²/s — maps softplus output to physiological ΔD

# Input token dimension: (S/S₀, b_norm, gx, gy, gz)
_TOKEN_DIM = 5


# ─────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    """
    Feedforward MLP with LayerNorm + GELU activations.

    Used both as the token encoder φ (shared, applied per measurement)
    and as intermediate projection layers.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim),
                       nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock(nn.Module):
    """
    Pre-activation residual block: LayerNorm → Linear → GELU → Linear → skip.
    Used in the set aggregator ρ.
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class DBSINet(nn.Module):
    """
    Protocol-conditioned DBSINet using the Deep Sets architecture.

    Each DWI volume is encoded as a token (S/S₀, b_norm, gx, gy, gz).
    A shared MLP φ encodes each token; mean pooling aggregates the set.
    A residual MLP ρ maps the aggregate to biophysical parameters.

    Parameters
    ----------
    embed_dim : int
        Dimension of the per-token embedding produced by φ. Default: 256.
    n_phi_layers : int
        Depth of the token encoder φ. Default: 3.
    aggregator_dim : int
        Width of the set aggregator ρ. Default: 512.
    n_res_blocks : int
        Number of residual blocks in ρ. Default: 4.
    dropout : float
        Dropout probability. Default: 0.0.
    """

    def __init__(
        self,
        embed_dim:      int   = 256,
        n_phi_layers:   int   = 3,
        aggregator_dim: int   = 512,
        n_res_blocks:   int   = 4,
        dropout:        float = 0.0,
    ):
        super().__init__()
        # Store all constructor args as attributes so trainer.py can read
        # them back into the checkpoint model_config dict.
        self.embed_dim      = embed_dim
        self.aggregator_dim = aggregator_dim
        self.n_res_blocks   = n_res_blocks
        self.n_phi_layers   = n_phi_layers
        self.dropout        = dropout

        # ── φ: token encoder (shared across all measurements) ─────────────
        # Input:  (S/S₀, b_norm, gx, gy, gz) ∈ ℝ⁵
        # Output: embedding ∈ ℝ^embed_dim
        # Applied identically to every volume in the protocol.
        self.phi = _MLP(
            in_dim=_TOKEN_DIM,
            hidden_dim=embed_dim,
            out_dim=embed_dim,
            n_layers=n_phi_layers,
            dropout=dropout,
        )

        # ── ρ: set aggregator (processes mean-pooled embedding) ───────────
        # Input:  ℝ^embed_dim  (from mean pooling)
        # Output: ℝ^aggregator_dim  (compact voxel representation)
        self.rho_proj = nn.Sequential(
            nn.Linear(embed_dim, aggregator_dim),
            nn.LayerNorm(aggregator_dim),
            nn.GELU(),
        )
        self.rho_blocks = nn.Sequential(
            *[_ResBlock(aggregator_dim, dropout) for _ in range(n_res_blocks)]
        )

        # ── Output bottleneck ─────────────────────────────────────────────
        _out_dim = aggregator_dim // 2
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(aggregator_dim),
            nn.Linear(aggregator_dim, _out_dim),
            nn.GELU(),
        )

        # ── Output heads ──────────────────────────────────────────────────
        self.frac_head   = nn.Linear(_out_dim, 3)   # → softmax → FF, RF, NRF
        self.diff_head   = nn.Linear(_out_dim, 2)   # → RD, AD
        self.dnonrf_head = nn.Linear(_out_dim, 1)   # → D_nonrf
        self.dir_head    = nn.Linear(_out_dim, 3)   # → fiber direction

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small output head initialization: avoids saturated sigmoids at start
        for head in [self.frac_head, self.diff_head,
                     self.dnonrf_head, self.dir_head]:
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

    def encode(
        self,
        signal: torch.Tensor,   # (B, N)   normalized S/S₀
        bvals:  torch.Tensor,   # (N,)     b-values [s/mm²]
        bvecs:  torch.Tensor,   # (N, 3)   gradient unit vectors
    ) -> torch.Tensor:          # (B, aggregator_dim)
        """
        Encode a set of DWI measurements into a fixed-size voxel embedding.

        This is the core protocol-conditioned step:
          1. Build tokens: cat(S_i/S₀, b_i/b_max, g_i) for each volume i
          2. Apply φ independently to each token
          3. Mean-pool across volumes → one embedding per voxel
          4. Apply ρ (projection + residual blocks)

        The resulting embedding captures all protocol-specific signal
        information in a protocol-invariant representation.
        """
        B, N = signal.shape
        device = signal.device

        # ── b-value normalization ─────────────────────────────────────────
        b_max  = bvals.max().clamp(min=1.0)
        b_norm = (bvals / b_max).unsqueeze(0).expand(B, -1)   # (B, N)

        # ── Build tokens: (S, b_norm, gx, gy, gz) ────────────────────────
        # bvecs: (N, 3) → (B, N, 3)
        bvecs_exp = bvecs.unsqueeze(0).expand(B, -1, -1)

        tokens = torch.cat([
            signal.unsqueeze(-1),       # (B, N, 1)
            b_norm.unsqueeze(-1),        # (B, N, 1)
            bvecs_exp,                   # (B, N, 3)
        ], dim=-1)                       # (B, N, 5)

        # ── Apply φ to each token ─────────────────────────────────────────
        # Reshape to (B·N, 5) for batched linear layers, then reshape back
        tokens_flat = tokens.reshape(B * N, _TOKEN_DIM)
        emb_flat    = self.phi(tokens_flat)                # (B·N, embed_dim)
        emb         = emb_flat.reshape(B, N, self.embed_dim)  # (B, N, embed_dim)

        # ── Mean pooling ──────────────────────────────────────────────────
        # Permutation invariant aggregation: Σ φ(token_i) / N
        pooled = emb.mean(dim=1)   # (B, embed_dim)

        # ── Apply ρ ───────────────────────────────────────────────────────
        h = self.rho_proj(pooled)       # (B, aggregator_dim)
        h = self.rho_blocks(h)          # (B, aggregator_dim)
        return h                        # (B, aggregator_dim)

    def forward(
        self,
        signal: torch.Tensor,   # (B, N)
        bvals:  torch.Tensor,   # (N,)
        bvecs:  torch.Tensor,   # (N, 3)
    ) -> dict:
        """
        Full forward pass: DWI set → biophysical parameters.

        Parameters
        ----------
        signal : Tensor (B, N)
            Normalized DWI signal S/S₀.
        bvals : Tensor (N,)
            B-values in s/mm² for the current protocol.
        bvecs : Tensor (N, 3)
            Gradient unit vectors.

        Returns
        -------
        dict with keys:
            'ff'        : (B,) fiber fraction ∈ [0,1]
            'rf'        : (B,) restricted fraction ∈ [0,1]
            'nrf'       : (B,) non-restricted fraction ∈ [0,1]
                          ff + rf + nrf = 1 by construction
            'ad'        : (B,) axial diffusivity [mm²/s], ≥ rd
            'rd'        : (B,) radial diffusivity [mm²/s]
            'd_nonrf'   : (B,) NRF centroid ADC [mm²/s]
            'fiber_dir' : (B, 3) unit fiber direction, z ≥ 0
            'adc_iso'   : (B,) mean isotropic ADC [mm²/s] (for map saving)
        """
        # ── Encode ────────────────────────────────────────────────────────
        h = self.encode(signal, bvals, bvecs)   # (B, aggregator_dim)
        h = self.bottleneck(h)                  # (B, aggregator_dim//2)

        # ── Fractions ─────────────────────────────────────────────────────
        fracs = torch.softmax(self.frac_head(h), dim=-1)
        ff  = fracs[:, 0]
        rf  = fracs[:, 1]
        nrf = fracs[:, 2]

        # ── Diffusivities ─────────────────────────────────────────────────
        diff_raw = self.diff_head(h)
        rd    = RD_MIN + torch.sigmoid(diff_raw[:, 0]) * (RD_MAX - RD_MIN)
        delta = F.softplus(diff_raw[:, 1]) * _AD_DELTA_SCALE
        ad    = torch.clamp(rd + delta, max=AD_MAX)

        # ── D_nonrf ───────────────────────────────────────────────────────
        d_nonrf = (D_NONRF_MIN
                   + torch.sigmoid(self.dnonrf_head(h).squeeze(-1))
                   * (D_NONRF_MAX - D_NONRF_MIN))

        # ── Fiber direction (Option A: explicit unit vector) ──────────────
        dir_raw   = self.dir_head(h)
        fiber_dir = F.normalize(dir_raw, p=2, dim=-1)
        # Enforce z ≥ 0 hemisphere convention (cos²θ is antipodally symmetric,
        # so this does not change the forward model — purely cosmetic convention)
        sign      = torch.sign(fiber_dir[:, 2:3])
        sign      = torch.where(sign == 0, torch.ones_like(sign), sign)
        fiber_dir = fiber_dir * sign

        # ── Mean isotropic ADC (pyDBSI channel 8 equivalent) ─────────────
        from .forward import reconstruct_adc_iso
        adc_iso = reconstruct_adc_iso(rf, nrf, d_nonrf)

        return {
            'ff':        ff,
            'rf':        rf,
            'nrf':       nrf,
            'ad':        ad,
            'rd':        rd,
            'd_nonrf':   d_nonrf,
            'fiber_dir': fiber_dir,
            'adc_iso':   adc_iso,
        }

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (f"DBSINet(protocol_conditioned=True, "
                f"embed_dim={self.embed_dim}, "
                f"aggregator_dim={self.aggregator_dim}, "
                f"n_params={self.n_parameters:,})")
