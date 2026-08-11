"""
Deep residual 1D CNN for G2Net gravitational wave detection.

Phase 3, Steps 1b + 2: V2-style two-stage fusion on top of ~10 residual backbone
blocks, with S2 spherical harmonic sky features. After the shared backbone,
4 parallel paths (H1, L1, V1, joint) are processed through individual branch blocks,
then merged and processed through 4 fusion blocks. Sky features (SH coefficients from
cross-detector consistency maps on S2) modulate the pooled CNN features through a
residual FiLM layer, so the geometric evidence conditions the detection decision
rather than sitting alongside it as additional concatenated bins.
LIGO H1/L1 share extractor and branch weights; Virgo has separate weights.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sky_feasibility import sh_block_index, bispectrum_terms


# ============================================================================================
#                                        POOLING
# ============================================================================================

class GeM(nn.Module):
    """
    Generalized Mean pooling with learnable exponent.

    Computes (mean(x^p))^(1/p) with learnable p (init=3). Interpolates between
    average pooling (p=1) and max pooling (p->inf). Negative activations from
    BatchNorm are clamped to eps. Forces float32 to prevent NaN under AMP.

    The effective p is clamped to [p_min, p_max] inside the forward pass so
    that drift in the underlying learnable parameter cannot push the pooling
    operation into a numerically unstable regime. The clamp does not block
    gradients -- p still trains freely via the straight-through nature of
    the clamp's gradient (zero only at the boundary) -- but the forward
    computation always sees a valid exponent.

    Parameters
    ----------
    kernel_size : int
        Pooling window size.
    p : float
        Initial value for the learnable exponent.
    eps : float
        Clamping floor for negative/zero activations.
    p_min : float
        Lower bound on the effective exponent.
    p_max : float
        Upper bound on the effective exponent.
    """

    def __init__(self, kernel_size: int, p: float = 3.0, eps: float = 1e-6,
                 p_min: float = 1.0, p_max: float = 10.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps
        self.p_min = p_min
        self.p_max = p_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            p = self.p.clamp(self.p_min, self.p_max)
            x = x.clamp(min=self.eps).pow(p)
            x = F.avg_pool1d(x, self.kernel_size)
            return x.pow(1.0 / p)


class AdaptiveConcatPool1d(nn.Module):
    """Concatenation of adaptive average pooling and adaptive max pooling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            F.adaptive_avg_pool1d(x, 1),
            F.adaptive_max_pool1d(x, 1),
        ], dim=1)


# ============================================================================================
#                                      SKY READOUT
# ============================================================================================

SKY_READOUTS = ("none", "mlp121", "power", "bispectrum", "scramble")


def sky_conditioning_dim(mode: str, n_sky_features: int, l_max: int, l_bisp: int = 4) -> int:
    """
    Input dimension of the FiLM conditioning MLP for a given readout mode.

    mlp121/scramble consume the full flattened a_lm vector; power consumes the
    (l_max + 1) angular power values; bispectrum consumes power plus the
    invariant bispectrum features. Returns 0 for the disabled pathway.
    """
    if mode == "none":
        return 0
    if mode in ("mlp121", "scramble"):
        return n_sky_features
    if mode == "power":
        return l_max + 1
    if mode == "bispectrum":
        return (l_max + 1) + bispectrum_terms(l_max, l_bisp)["n_features"]
    raise ValueError(f"unknown sky_readout mode: {mode!r}")


def hidden_for_param_count(d_in: int, feat_dim: int, target: int) -> int:
    """
    Solve for the FiLM hidden width that hits a target conditioning-path
    parameter count. Conditioning params (BN + fc1 + fc2) are
        2*d_in + hidden*(d_in + 1) + 2*feat_dim*(hidden + 1),
    linear in hidden. Used by match_params so power (d_in=11) and mlp121
    (d_in=121) carry the same capacity in the conditioning path.
    """
    hidden = round((target - 2 * d_in - 2 * feat_dim) / (d_in + 1 + 2 * feat_dim))
    return max(1, hidden)


class SkyReadout(nn.Module):
    """
    FiLM modulation of pooled CNN features by S2 sky features, with a
    selectable readout that determines what SO(3) structure survives.

    Modes (flag ``sky_readout``):
    - ``mlp121``: the flattened a_lm vector (121 coefs for l_max=10) into
      BatchNorm and a 2-layer MLP. Covariant under SO(3) but not invariant --
      the MLP mixes the (2l+1)-blocks, so a rotation of the sky changes the
      output. This is the current/original readout.
    - ``power``: the angular power spectrum p_l = sum_m a_lm^2 (l_max + 1
      values). Each p_l is the squared norm of one multipole block, invariant
      under the block rotation, so the readout is exactly SO(3)-invariant.
    - ``bispectrum``: power spectrum concatenated with a subset of the SH
      bispectrum (Clebsch-Gordan / Wigner-3j contractions). Also SO(3)-invariant
      but retains relative phase between multipoles that the power spectrum discards.
    - ``scramble``: identical to ``mlp121`` but the a_lm vector is randomly
      permuted per sample. Preserves the per-sample marginal statistics of the
      coefficients while destroying the physical block structure. A control, not
      a method: if it matches ``mlp121`` the network was never using the geometry.

    In every mode the conditioning vector passes through BatchNorm and a 2-layer
    MLP producing tanh-bounded (gamma, beta); the pooled features are modulated as
    (1 + gamma) * features + beta. The output projection is zero-initialized, so
    the module starts as the identity and the FiLM path only contributes if the
    optimizer learns to use it. The tanh bound keeps (1 + gamma) in (0, 2): an
    early unbounded-gamma run diverged at epoch 5 with NaN loss.

    Parameters
    ----------
    mode : str
        One of SKY_READOUTS minus ``none``.
    n_sky_features : int
        Length of the input a_lm vector.
    feat_dim : int
        Dimension of the pooled CNN feature vector to modulate.
    l_max : int
        Maximum SH degree (defines the power-spectrum length and blocks).
    hidden_dim : int
        Hidden width of the conditioning MLP.
    l_bisp : int
        Maximum degree of the bispectrum subset (bispectrum mode only).
    scramble_seed : int
        Seed for the per-sample permutation (scramble mode only).
    """

    def __init__(self, mode: str, n_sky_features: int, feat_dim: int, l_max: int,
                 hidden_dim: int = 128, l_bisp: int = 4, scramble_seed: int = 0):
        super().__init__()
        if mode not in ("mlp121", "power", "bispectrum", "scramble"):
            raise ValueError(f"SkyReadout does not handle mode {mode!r}")
        self.mode = mode
        self.feat_dim = feat_dim
        self.l_max = l_max
        self.scramble_seed = scramble_seed
        self._generators: dict = {}

        if mode in ("power", "bispectrum"):
            self.register_buffer("block_index", torch.tensor(sh_block_index(l_max)))
        if mode == "bispectrum":
            bt = bispectrum_terms(l_max, l_bisp)
            self.n_bispec = bt["n_features"]
            self.register_buffer("bispec_feature", torch.tensor(bt["feature"]))
            self.register_buffer("bispec_i", torch.tensor(bt["i"]))
            self.register_buffer("bispec_j", torch.tensor(bt["j"]))
            self.register_buffer("bispec_p", torch.tensor(bt["p"]))
            self.register_buffer("bispec_w", torch.tensor(bt["w"]))

        d_in = sky_conditioning_dim(mode, n_sky_features, l_max, l_bisp)
        self.d_in = d_in
        self.bn = nn.BatchNorm1d(d_in, momentum=0.1, eps=1e-5)
        self.fc1 = nn.Linear(d_in, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, 2 * feat_dim)

    def reset_to_identity(self):
        """Zero the output projection so the module is identity at init."""
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def conditioning_param_count(self) -> int:
        """Learnable parameters in the conditioning path (BN + fc1 + fc2)."""
        mods = [self.bn, self.fc1, self.fc2]
        return sum(p.numel() for m in mods for p in m.parameters())

    def _power(self, sky: torch.Tensor) -> torch.Tensor:
        p = sky.new_zeros(sky.shape[0], self.l_max + 1)
        return p.index_add_(1, self.block_index, sky * sky)

    def _bispectrum(self, sky: torch.Tensor) -> torch.Tensor:
        contrib = self.bispec_w * sky[:, self.bispec_i] * sky[:, self.bispec_j] * sky[:, self.bispec_p]
        b = sky.new_zeros(sky.shape[0], self.n_bispec)
        return b.index_add_(1, self.bispec_feature, contrib)

    def _scramble(self, sky: torch.Tensor) -> torch.Tensor:
        # independent random permutation of the coefficient vector per sample.
        # preserves each sample's coefficient multiset (marginals) but destroys
        # the fixed index -> (l, m) correspondence the physical structure lives in.
        # a per-device generator seeded once keeps runs reproducible given the
        # (seeded) data order.
        dev = sky.device
        if dev not in self._generators:
            g = torch.Generator(device=dev)
            g.manual_seed(self.scramble_seed)
            self._generators[dev] = g
        g = self._generators[dev]
        noise = torch.rand(sky.shape, generator=g, device=dev)
        perm = noise.argsort(dim=1)
        return torch.gather(sky, 1, perm)

    def _reduce(self, sky: torch.Tensor) -> torch.Tensor:
        if self.mode == "mlp121":
            return sky
        if self.mode == "scramble":
            return self._scramble(sky)
        if self.mode == "power":
            return self._power(sky)
        # bispectrum: power spectrum concatenated with bispectrum features
        return torch.cat([self._power(sky), self._bispectrum(sky)], dim=1)

    def forward(self, features: torch.Tensor, sky: torch.Tensor) -> torch.Tensor:
        cond = self._reduce(sky)
        gamma_beta = self.fc2(self.act(self.fc1(self.bn(cond))))
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        return (1.0 + gamma) * features + beta


# ============================================================================================
#                                   STOCHASTIC DEPTH
# ============================================================================================

def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """
    Stochastic depth: randomly drop the entire residual branch per sample.

    Uses inverted dropout: surviving samples are scaled by 1/(1-p) during training
    so that no scaling is needed at inference.

    Parameters
    ----------
    x : torch.Tensor
        Residual branch output of shape (B, C, T).
    drop_prob : float
        Probability of dropping this branch for each sample.
    training : bool
        Whether model is in training mode.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    mask = torch.rand(x.shape[0], 1, 1, dtype=x.dtype, device=x.device)
    mask = torch.floor(mask + keep_prob)
    return x * mask / keep_prob


# ============================================================================================
#                                    RESIDUAL BLOCK
# ============================================================================================

class ResBlock(nn.Module):
    """
    Residual block with two Conv1d layers, optional GeM downsampling, and stochastic depth.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size for both convolutions.
    downsample_factor : int | None
        Spatial downsampling factor via GeM pooling. None for identity blocks.
    drop_prob : float
        Stochastic depth drop probability for this block.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, downsample_factor: int | None = None, drop_prob: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-5)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-5)
        self.act = nn.SiLU()
        self.pool = GeM(downsample_factor) if downsample_factor else None
        self.drop_prob = drop_prob

        # shortcut: project channels and/or downsample to match main path
        needs_proj = (in_channels != out_channels)
        layers = []
        if needs_proj:
            layers.append(nn.Conv1d(in_channels, out_channels, 1))
            layers.append(nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-5))
        if downsample_factor:
            layers.append(GeM(downsample_factor))
        self.shortcut = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))         # first conv + BN + activation
        out = self.bn2(self.conv2(out))                 # second conv + BN (no activation yet)
        if self.pool is not None:
            out = self.pool(out)
        out = drop_path(out, self.drop_prob, self.training)
        return self.act(out + residual)


# ============================================================================================
#                                      DIY MODEL
# ============================================================================================

class DIYModel(nn.Module):
    """
    Deep residual 1D CNN for binary classification of GW signals.

    Architecture (Phase 3, Steps 1b + 2 -- V2 fusion + S2 sky features via FiLM):
    - Separate extractors for LIGO (H1/L1 shared) and Virgo
    - 10 residual backbone blocks with GeM downsampling and channel widening
    - V2 fusion: 4 parallel branch paths (H1, L1, V1, joint) x 2 ResBlocks each
    - 4 fusion blocks processing merged branch outputs
    - AdaptiveConcatPool1d (avg + max) on fused features
    - S2 sky features (SH coefficients) -> SkyReadout FiLM modulation of pooled features
    - 3-layer classifier head
    - Optional auxiliary per-branch heads (disabled in the final config with
      aux_loss_weight=0; kept for compatibility with earlier ablation runs)
    - All outputs are raw logits (no sigmoid); use BCEWithLogitsLoss

    Backbone:  4096 -> 2048 (extractor) -> 512 -> 128 -> 32
    Channels:  1 -> n -> n -> 2n -> 4n (backbone) -> 4n (branches) -> 16n (fused)
    Fusion:    4 paths x 4n -> concat 16n -> 4 ResBlocks -> ConcatPool -> 32n features
    Sky:       (l_max + 1)^2 SH coefficients -> BN -> MLP -> (gamma, beta) -> FiLM

    Input shape: (batch_size, 3, 4096) - 3 detectors, 4096 time samples
    Sky input:   (batch_size, n_sky_features) - SH coefficients from S2 sky map
    """

    def __init__(self, n_channels: int = 16, dropout_rate: float = 0.5, drop_path_rate: float = 0.0,
                 n_sky_features: int = 121, sky_readout: str = "mlp121", h1l1_merge: str = "concat",
                 h1l1_pool: str = "mean", match_params: bool = False, l_max: int = 10,
                 l_bisp: int = 4, scramble_seed: int = 0):
        """
        Parameters
        ----------
        n_channels : int
            Base channel width (n). Channels progress as n -> 2n -> 4n.
        dropout_rate : float
            Dropout rate for classifier head.
        drop_path_rate : float
            Maximum stochastic depth drop probability (applied to last block;
            linearly increases from 0 at the first block).
        n_sky_features : int
            Number of S2 sky features (SH coefficients). With l_max=10,
            this is (10+1)^2 = 121.
        sky_readout : str
            Sky readout variant, one of SKY_READOUTS. ``none`` disables the
            sky pathway (FiLM bypassed). See SkyReadout for the rest.
        h1l1_merge : str
            ``concat`` for the ordered four-path fusion (breaks the H1<->L1 swap
            symmetry) or ``symmetric`` for swap-invariant pooling of the two
            LIGO branches before fusion.
        h1l1_pool : str
            Symmetric pooling op when h1l1_merge=='symmetric': ``mean`` or ``max``.
        match_params : bool
            Equalize the conditioning-path parameter count across readouts by
            adjusting the FiLM hidden width to match mlp121 at hidden=128.
        l_max : int
            Maximum SH degree (must satisfy (l_max + 1)^2 == n_sky_features when
            a sky readout is active).
        l_bisp : int
            Maximum degree of the bispectrum subset (bispectrum readout only).
        scramble_seed : int
            Seed for the per-sample permutation (scramble readout only).
        """
        super().__init__()
        n = n_channels
        if sky_readout not in SKY_READOUTS:
            raise ValueError(f"sky_readout must be one of {SKY_READOUTS}, got {sky_readout!r}")
        if h1l1_merge not in ("concat", "symmetric"):
            raise ValueError(f"h1l1_merge must be 'concat' or 'symmetric', got {h1l1_merge!r}")
        if sky_readout in ("power", "bispectrum", "mlp121", "scramble") and n_sky_features != (l_max + 1) ** 2:
            raise ValueError(
                f"n_sky_features ({n_sky_features}) must equal (l_max+1)^2 ({(l_max + 1) ** 2})"
            )
        self.sky_readout_mode = sky_readout
        self.h1l1_merge = h1l1_merge
        self.h1l1_pool = h1l1_pool

        # ---- extractors ----
        # Conv(1,n,64) -> BN -> SiLU -> Conv(n,n,64) -> GeM(2)
        # H1/L1 share weights (same instrument type); Virgo is separate
        self.ligo_extractor = nn.Sequential(
            nn.Conv1d(1, n, 64, padding='same'),
            nn.BatchNorm1d(n, momentum=0.1, eps=1e-5),
            nn.SiLU(),
            nn.Conv1d(n, n, 64, padding='same'),
            GeM(2),
        )
        self.virgo_extractor = nn.Sequential(
            nn.Conv1d(1, n, 64, padding='same'),
            nn.BatchNorm1d(n, momentum=0.1, eps=1e-5),
            nn.SiLU(),
            nn.Conv1d(n, n, 64, padding='same'),
            GeM(2),
        )

        # ---- residual backbone (shared by all 3 branches) ----
        # 5 groups x 2 blocks = 10 residual blocks
        # (out_channels, kernel_size, downsample_factor)
        groups = [
            (n,     31, 4),     # group 1: 2048 -> 512
            (n,     31, None),  # group 2: 512 (no downsample)
            (2 * n, 15, 4),     # group 3: 512 -> 128, widen to 2n
            (4 * n,  7, 4),     # group 4: 128 -> 32, widen to 4n
            (4 * n,  7, None),  # group 5: 32 (no downsample)
        ]

        # stochastic depth schedule across all blocks:
        # 10 backbone + 2 branch-depth (parallel) + 4 fusion = 16 depth levels
        n_depth_levels = 16
        drop_probs = np.linspace(0.0, drop_path_rate, n_depth_levels)

        blocks = []
        in_ch = n
        block_idx = 0
        for out_ch, k, ds in groups:
            blocks.append(ResBlock(in_ch, out_ch, k, downsample_factor=ds, drop_prob=drop_probs[block_idx]))
            in_ch = out_ch
            block_idx += 1
            blocks.append(ResBlock(in_ch, out_ch, k, drop_prob=drop_probs[block_idx]))
            block_idx += 1
        self.backbone = nn.Sequential(*blocks)

        # ---- V2 two-stage fusion: individual branch paths ----
        # H1/L1 share branch weights (same instrument); Virgo separate
        self.ligo_branch = nn.Sequential(
            ResBlock(4 * n, 4 * n, 7, drop_prob=drop_probs[10]),
            ResBlock(4 * n, 4 * n, 7, drop_prob=drop_probs[11]),
        )
        self.virgo_branch = nn.Sequential(
            ResBlock(4 * n, 4 * n, 7, drop_prob=drop_probs[10]),
            ResBlock(4 * n, 4 * n, 7, drop_prob=drop_probs[11]),
        )

        # ---- V2 two-stage fusion: joint branch ----
        # detector backbone outputs concatenated on channels -> project down -> 2 ResBlocks.
        # concat feeds all 3 (12n); symmetric feeds the swap-pooled LIGO pair + Virgo (8n).
        # fused-channel count: concat = 4 paths x 4n = 16n; symmetric = 3 paths x 4n = 12n.
        if h1l1_merge == "concat":
            joint_in_ch = 3 * 4 * n
            fused_ch = 16 * n
        else:
            joint_in_ch = 2 * 4 * n
            fused_ch = 12 * n

        self.joint_project = nn.Sequential(
            nn.Conv1d(joint_in_ch, 4 * n, 1),
            nn.BatchNorm1d(4 * n, momentum=0.1, eps=1e-5),
        )
        self.joint_branch = nn.Sequential(
            ResBlock(4 * n, 4 * n, 7, drop_prob=drop_probs[10]),
            ResBlock(4 * n, 4 * n, 7, drop_prob=drop_probs[11]),
        )

        # ---- V2 two-stage fusion: post-merge processing ----
        self.fusion = nn.Sequential(
            ResBlock(fused_ch, fused_ch, 7, drop_prob=drop_probs[12]),
            ResBlock(fused_ch, fused_ch, 7, drop_prob=drop_probs[13]),
            ResBlock(fused_ch, fused_ch, 7, drop_prob=drop_probs[14]),
            ResBlock(fused_ch, fused_ch, 7, drop_prob=drop_probs[15]),
        )

        # ---- pooling and heads ----
        self.global_pool = AdaptiveConcatPool1d()

        # auxiliary per-branch heads: ConcatPool on 4n -> 8n features
        self.branch_head = nn.Linear(8 * n, 1)

        # AdaptiveConcatPool doubles the channel count (avg + max)
        feat_dim = 2 * fused_ch

        # ---- S2 sky features (Phase 3, Step 2): FiLM modulation ----
        if sky_readout == "none":
            self.sky_readout = None
        else:
            hidden = 128
            if match_params:
                # match the conditioning-path parameter count to mlp121 @ hidden=128
                ref_d_in = sky_conditioning_dim("mlp121", n_sky_features, l_max)
                target = 2 * ref_d_in + 128 * (ref_d_in + 1) + 2 * feat_dim * (128 + 1)
                d_in = sky_conditioning_dim(sky_readout, n_sky_features, l_max, l_bisp)
                hidden = hidden_for_param_count(d_in, feat_dim, target)
            self.sky_readout = SkyReadout(
                sky_readout, n_sky_features, feat_dim, l_max,
                hidden_dim=hidden, l_bisp=l_bisp, scramble_seed=scramble_seed,
            )

        # classifier head: takes the (optionally FiLM-modulated) pooled features
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256, momentum=0.1, eps=1e-5),
            nn.Dropout(dropout_rate),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64, momentum=0.1, eps=1e-5),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

        self._init_weights()
        # restore identity init for the FiLM output projection so the sky path
        # starts as a no-op (gamma=0, beta=0); _init_weights has just
        # overwritten it with Kaiming above.
        if self.sky_readout is not None:
            self.sky_readout.reset_to_identity()

    def conditioning_param_count(self) -> int:
        """Learnable parameters in the sky conditioning path (0 if disabled)."""
        return 0 if self.sky_readout is None else self.sky_readout.conditioning_param_count()

    def _sym_pool(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Swap-invariant pooling of the two LIGO branch tensors."""
        if self.h1l1_pool == "mean":
            return 0.5 * (a + b)
        return torch.maximum(a, b)

    def _init_weights(self):
        """Apply Kaiming normal initialization to conv and linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _extract_pooled(self, X: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Run extractor -> backbone -> branches -> fusion -> global pool.

        Returns the pooled feature vector together with the per-detector
        branch tensors (needed by the auxiliary heads). Manifold mixup
        interposes between this method and ``classify_from_pooled``.
        """
        # per-detector extraction
        h1 = self.ligo_extractor(X[:, 0:1, :])
        l1 = self.ligo_extractor(X[:, 1:2, :])
        v1 = self.virgo_extractor(X[:, 2:3, :])

        # batched backbone pass: (3*B, C, T)
        stacked = torch.cat([h1, l1, v1], dim=0)
        stacked = self.backbone(stacked)
        h1_feat, l1_feat, v1_feat = stacked.chunk(3, dim=0)

        # individual branch paths (H1/L1 batched through shared LIGO branch)
        ligo_stacked = torch.cat([h1_feat, l1_feat], dim=0)
        ligo_stacked = self.ligo_branch(ligo_stacked)
        h1_branch, l1_branch = ligo_stacked.chunk(2, dim=0)
        v1_branch = self.virgo_branch(v1_feat)

        if self.h1l1_merge == "concat":
            # ordered four-path fusion (breaks the H1<->L1 swap symmetry)
            joint_in = torch.cat([h1_feat, l1_feat, v1_feat], dim=1)
            joint_branch = self.joint_branch(self.joint_project(joint_in))
            fused = torch.cat([h1_branch, l1_branch, v1_branch, joint_branch], dim=1)
        else:
            # symmetric merge: swap-invariant pooling of the LIGO pair before fusion.
            # both the joint-branch input and the fused stack use the pooled pair, so
            # exchanging the H1/L1 input channels leaves the pooled features unchanged.
            joint_in = torch.cat([self._sym_pool(h1_feat, l1_feat), v1_feat], dim=1)
            joint_branch = self.joint_branch(self.joint_project(joint_in))
            fused = torch.cat([self._sym_pool(h1_branch, l1_branch), v1_branch, joint_branch], dim=1)

        fused = self.fusion(fused)
        pooled = self.global_pool(fused).squeeze(-1)

        return pooled, [h1_branch, l1_branch, v1_branch]

    def classify_from_pooled(self, pooled: torch.Tensor, sky_features: torch.Tensor) -> torch.Tensor:
        """Apply the sky readout (if any) and the classifier head to pooled features."""
        if self.sky_readout is None:
            return self.classifier(pooled)
        return self.classifier(self.sky_readout(pooled, sky_features))

    def forward(self, X: torch.Tensor, sky_features: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Parameters
        ----------
        X : torch.Tensor
            Input of shape (batch_size, 3, 4096).
        sky_features : torch.Tensor
            S2 spherical harmonic coefficients, shape (batch_size, n_sky_features).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]
            Inference: logits (batch_size, 1).
            Training: (logits, [h1_logits, l1_logits, v1_logits]).
        """
        pooled, branches = self._extract_pooled(X)
        main_logits = self.classify_from_pooled(pooled, sky_features)

        if self.training:
            h1_branch, l1_branch, v1_branch = branches
            branch_logits = [
                self.branch_head(self.global_pool(h1_branch).squeeze(-1)),
                self.branch_head(self.global_pool(l1_branch).squeeze(-1)),
                self.branch_head(self.global_pool(v1_branch).squeeze(-1)),
            ]
            return main_logits, branch_logits

        return main_logits

    def compute_loss(
        self,
        y_true: torch.Tensor,
        logits: torch.Tensor,
        branch_logits: list[torch.Tensor] | None = None,
        aux_loss_weight: float = 0.0,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute total loss: main BCE + weighted auxiliary per-branch BCE.

        All inputs are raw logits (pre-sigmoid). Uses BCEWithLogitsLoss for
        numerical stability.

        Parameters
        ----------
        y_true : torch.Tensor
            True labels of shape (batch_size, 1).
        logits : torch.Tensor
            Main logits of shape (batch_size, 1).
        branch_logits : list[torch.Tensor] | None
            Per-detector logits, each of shape (batch_size, 1).
        aux_loss_weight : float
            Weight for auxiliary branch losses (lambda). 0 disables.
        label_smoothing : float
            Smoothing factor. Targets become eps and 1-eps instead of 0 and 1.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if label_smoothing > 0:
            y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing

        main_loss = F.binary_cross_entropy_with_logits(logits, y_true)

        if branch_logits is not None and aux_loss_weight > 0:
            aux_loss = sum(
                F.binary_cross_entropy_with_logits(bl, y_true) for bl in branch_logits
            ) / len(branch_logits)
            return main_loss + aux_loss_weight * aux_loss

        return main_loss

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, sky_features: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Predict probabilities for input samples.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, 3, n_time_samples).
        sky_features : np.ndarray
            S2 SH coefficients of shape (n_samples, n_sky_features).
        batch_size : int
            Batch size for inference to avoid OOM on large inputs.

        Returns
        -------
        np.ndarray
            Predicted probabilities of shape (n_samples,).
        """
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device

        n_samples = X.shape[0]
        if n_samples <= batch_size:
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            sky_t = torch.tensor(sky_features, dtype=torch.float32, device=device)
            logits = self.forward(X_t, sky_features=sky_t)
            result = torch.sigmoid(logits).cpu().numpy().flatten()
        else:
            all_predictions = []
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = torch.tensor(X[start_idx:end_idx], dtype=torch.float32, device=device)
                sky_batch = torch.tensor(sky_features[start_idx:end_idx], dtype=torch.float32, device=device)
                logits = self.forward(X_batch, sky_features=sky_batch)
                all_predictions.append(torch.sigmoid(logits).cpu().numpy())
            result = np.concatenate(all_predictions, axis=0).flatten()

        if was_training:
            self.train()
        return result

    def predict(self, X: np.ndarray, sky_features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels for input samples. 1 = BH merger, 0 = not

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features).
        sky_features : np.ndarray
            S2 SH coefficients of shape (n_samples, n_sky_features).
        threshold : float
            Classification threshold.

        Returns
        -------
        np.ndarray
            Predicted binary labels of shape (n_samples,).
        """
        probas = self.predict_proba(X, sky_features)
        return (probas >= threshold).astype(int)

    def save_weights(self, filepath: str) -> None:
        """
        Save model weights to a file.

        Parameters
        ----------
        filepath : str
            Path to save weights (.pt file).
        """
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath: str) -> None:
        """
        Load model weights from a file.

        Parameters
        ----------
        filepath : str
            Path to load weights from (.pt file).
        """
        state_dict = torch.load(filepath, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict)


# ============================================================================================
#                                     SKY HEAD MODEL
# ============================================================================================

class SkyHeadModel(nn.Module):
    """
    Sky features into a small classifier, with no CNN pathway at all.

    Config 5 of the sweep: measures how much of the label signal lives in the
    geometric features on their own, independent of the waveform CNN. The
    forward signature matches DIYModel (it accepts and ignores the raw signal
    X) so the training loop and evaluation code need no special-casing. Outputs
    are raw logits.

    Parameters
    ----------
    n_sky_features : int
        Length of the input SH coefficient vector.
    hidden_dim : int
        Width of the first hidden layer.
    dropout_rate : float
        Dropout rate in the classifier.
    """

    def __init__(self, n_sky_features: int = 121, hidden_dim: int = 128, dropout_rate: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_sky_features, momentum=0.1, eps=1e-5),
            nn.Linear(n_sky_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.1, eps=1e-5),
            nn.Dropout(dropout_rate),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64, momentum=0.1, eps=1e-5),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def conditioning_param_count(self) -> int:
        """Parameters in the sky path (the whole model, here)."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, X: torch.Tensor, sky_features: torch.Tensor):
        logits = self.net(sky_features)
        if self.training:
            return logits, None
        return logits

    def compute_loss(self, y_true, logits, branch_logits=None, aux_loss_weight=0.0, label_smoothing=0.0):
        """BCE-with-logits loss. Branch/aux arguments accepted for interface parity."""
        if label_smoothing > 0:
            y_true = y_true * (1 - label_smoothing) + 0.5 * label_smoothing
        return F.binary_cross_entropy_with_logits(logits, y_true)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, sky_features: np.ndarray, batch_size: int = 256) -> np.ndarray:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        n_samples = sky_features.shape[0]
        preds = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            sky_t = torch.tensor(sky_features[start:end], dtype=torch.float32, device=device)
            logits = self.net(sky_t)
            preds.append(torch.sigmoid(logits).cpu().numpy())
        if was_training:
            self.train()
        return np.concatenate(preds, axis=0).flatten()

    def predict(self, X: np.ndarray, sky_features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X, sky_features) >= threshold).astype(int)

    def save_weights(self, filepath: str) -> None:
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath: str) -> None:
        state_dict = torch.load(filepath, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict)
