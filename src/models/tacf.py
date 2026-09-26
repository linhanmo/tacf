from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregator import AggregatorAgent, AggregatorOutput
from .agent import AgentGroup, AgentsOutput
from .consensus import ConsensusLayer, ConsensusOutput
from ..data.dataset import AGENT_NAMES
from ..data.freq_decomp import DecomposerOutput, LearnableDecomposer


__all__ = ["TACFOutput", "TACF", "_RevIN"]


class _RevIN(nn.Module):
    """Reversible Instance Normalization (RevIN, Kim et al. ICLR 2022).

    Standard on all recent ETT-SOTA models (DLinear / iTransformer /
    PatchTST / Bi-Mamba4TS / TEFN).  Subtracts per-channel, per-sample,
    lookback-time mean/std and restores them on the output via denorm so
    that magnitude/trend information never leaves the pipeline, even
    when the backbone operates in z-score space.

    Ported from Time-Series-Library/layers/StandardNorm.py so it stays
    self-contained in src/ and doesn't depend on third-party imports.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        self._last_mean: Optional[torch.Tensor] = None
        self._last_stdev: Optional[torch.Tensor] = None
        self._last_last_val: Optional[torch.Tensor] = None

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self._last_last_val = x[:, -1:, :].detach()
            self._last_mean = None
        else:
            self._last_mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        v = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
        self._last_stdev = torch.sqrt(v + self.eps).detach()

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        self._get_statistics(x)
        if self.subtract_last:
            x = x - self._last_last_val
        else:
            x = x - self._last_mean
        x = x / self._last_stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self._last_stdev.to(x.device).to(x.dtype)
        if self.subtract_last:
            x = x + self._last_last_val.to(x.device).to(x.dtype)
        else:
            x = x + self._last_mean.to(x.device).to(x.dtype)
        return x


@dataclass
class TACFOutput:
    """Full forward output of the TACF MOA model.

    Top-level predictions
    ---------------------
    y_hat  : (B, P, D)  fused mean forecast
    sigma  : (B, P, D)  fused standard deviation
    alpha  : (B, 3)     learned agent mixture weights (softmax-normalised)
    reject : (B, 3)     learned agent reject signals (in [0, 1])
    effective_weights : (B, 3)  effective renormalised w = alpha·(1 - r)

    Auxiliary outputs (used for losses / inspection / logging)
    ----------------------------------------------------------
    decomposer_out, specialists_out, consensus_out, aggregator_out
        Full intermediate dataclass outputs.
    aux_losses : dict
        Scalar auxiliary losses keyed by name (``orthogonality`` is always
        present; ``_mse_diag`` is added if ``y`` was passed to ``forward``).
    """

    y_hat: torch.Tensor
    sigma: torch.Tensor
    alpha: torch.Tensor
    reject: torch.Tensor
    effective_weights: torch.Tensor
    decomposer_out: DecomposerOutput
    specialists_out: AgentsOutput
    consensus_out: ConsensusOutput
    aggregator_out: AggregatorOutput
    aux_losses: Dict[str, torch.Tensor] = field(default_factory=dict)

    def dict_for_logging(self) -> Dict[str, float]:
        return {
            k: float(v.detach().mean().cpu().item()) for k, v in self.aux_losses.items()
        }


class TACF(nn.Module):
    """Top-level TACF Mixture-of-Agents model.

    Pipeline exactly mirrors the ASCII diagram in ``/home/lin/tacf/Architecture.md``::

        Input X ∈ R^(B, T, D)
          └─▶ LearnableDecomposer
                 3 Conv1D branches (k=64 trend / 32 cycle / 16 local)
                 └─▶ X_trend, X_cycle, X_local ∈ R^(B, T, D) × 3
                      └─▶ AgentGroup (3 × IndependentAgent w/ own Bi-Mamba θ_k)
                             └─▶ {μ_k, σ_k, h_k} ∈ R^(B,P,D)×2 × R^(B,d)
                                  └─▶ ConsensusLayer (2-round MHA + IVW correction)
                                         └─▶ {μ_k^R, σ_k^R, h_k^R}
                                              └─▶ AggregatorAgent
                                                     LightMamba([h_t,h_c,h_l])
                                                     ├─▶ α ∈ softmax, r ∈ sigmoid
                                                     └─▶ precision-w fusion → ŷ,σ

    Notes
    -----
    * All Bi-Mamba backbones use CUDA selective-scan kernels when available
      and fall back to a differentiable gated-conv mixer on CPU, so the same
      weights can be sanity-tested on a developer laptop and trained on GPU.
    * All three specialists share the same architecture but never share
      parameters, matching the architecture's "independently parameterised
      specialists, coordinated only through consensus + aggregator" design.
    * Weight initialisation follows standard time-series conventions:
      trunc_normal for Linear (σ=0.02), Kaiming for Conv1d, identity for Norms.
    """

    agent_names = tuple(AGENT_NAMES)

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        seq_len: int = 336,
        pred_len: int = 168,
        d_hidden: int = 512,
        # ---------- Decomposer ----------
        kernel_trend: int = 64,
        kernel_cycle: int = 32,
        kernel_local: int = 16,
        decomp_residual: bool = True,
        # ---------- Specialist Bi-Mamba ----------
        specialist_d_model: int = 512,
        specialist_n_layers: int = 4,
        specialist_d_state: int = 16,
        specialist_d_conv: int = 4,
        specialist_expand: int = 2,
        specialist_dropout: float = 0.1,
        specialist_bidirectional: bool = True,
        specialist_share_embeddings: bool = False,
        # ---------- Consensus ----------
        consensus_n_rounds: int = 2,
        consensus_num_heads: int = 4,
        consensus_attn_dropout: float = 0.05,
        consensus_use_layer_norm: bool = True,
        consensus_residual: bool = True,
        consensus_use_ivw: bool = True,
        # ---------- Aggregator LightMamba ----------
        aggregator_d_model: int = 256,
        aggregator_n_layers: int = 2,
        aggregator_d_state: int = 16,
        aggregator_d_conv: int = 4,
        aggregator_expand: int = 2,
        aggregator_dropout: float = 0.05,
        aggregator_bidirectional: bool = True,
        # ---------- Sigma calibration (global multiplier, learnable scalar) ----
        sigma_global_multiplier_init: float = 1.0,
        # ---------- SOTA-grade normalisation / residual shortcuts 2026-09-26 ----------
        # RevIN (Reversible Instance Normalization) — DLinear / iTransformer /
        # PatchTST / Bi-Mamba4TS all use per-sample instance norm on the time
        # axis because ETT / weather / electricity series have strong non-stationary
        # per-window distribution shifts (oil temperature jumps after maintenance,
        # diurnal/weekly cycles not captured by the global StandardScaler).
        use_revin: bool = True,
        revin_affine: bool = True,
        revin_subtract_last: bool = False,
        # DLinear-style trend residual head — add a linear (2-layer MLP over last T
        # timesteps of each instance-normalised channel directly to the final y_hat.
        # DLinear takes ~MSE=0.44 on this exact setting (ETTh1 336→168) so
        # giving TACF a dedicated linear trend shortcut avoids the MoA having
        # to re-learn trivial linear extrapolation.
        use_dlinear_trend_head: bool = True,
        # ---------- Misc ----------
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.d_hidden = int(d_hidden)
        self._sigma_mult_init = float(sigma_global_multiplier_init)
        self.use_revin = bool(use_revin)
        self.use_dlinear_trend_head = bool(use_dlinear_trend_head)

        # --- RevIN: per-sample reversible instance norm on the (B, T, D) input.
        if self.use_revin:
            self.revin = _RevIN(
                num_features=self.in_channels,
                eps=1e-5,
                affine=bool(revin_affine),
                subtract_last=bool(revin_subtract_last),
            )
        else:
            self.revin = None

        # --- DLinear-style direct trend shortcut.
        if self.use_dlinear_trend_head:
            # Simple T→P linear projection per channel (DLinear-individual style).
            # Equivalent to fitting an AR(seq_len) per channel, residual on top of MoA.
            # einsum "b t d, d p t -> b p d" contracts dim T:
            #   out[b,p,d] = sum_t x_norm[b,t,d] * W[d,p,t] + b[d,p]
            # So W.shape = (D, P, T).
            self._trend_W = nn.Parameter(
                torch.empty(self.out_channels, self.pred_len, self.seq_len)
            )
            self._trend_b = nn.Parameter(torch.zeros(self.out_channels, self.pred_len))
            nn.init.kaiming_normal_(self._trend_W, nonlinearity="linear")
        else:
            self._trend_W = None
            self._trend_b = None

        self.decomposer = LearnableDecomposer(
            in_channels=in_channels,
            d_hidden=max(64, in_channels * 4),
            kernel_trend=kernel_trend,
            kernel_cycle=kernel_cycle,
            kernel_local=kernel_local,
            residual=decomp_residual,
        )
        self.specialists = AgentGroup(
            in_channels=in_channels,
            out_channels=out_channels,
            seq_len=seq_len,
            pred_len=pred_len,
            d_hidden=d_hidden,
            d_model=specialist_d_model,
            n_layers=specialist_n_layers,
            d_state=specialist_d_state,
            d_conv=specialist_d_conv,
            expand=specialist_expand,
            dropout=specialist_dropout,
            bidirectional=specialist_bidirectional,
            norm_eps=norm_eps,
            share_embeddings=specialist_share_embeddings,
        )
        self.consensus = ConsensusLayer(
            d_hidden=d_hidden,
            out_channels=out_channels,
            pred_len=pred_len,
            num_heads=consensus_num_heads,
            attn_dropout=consensus_attn_dropout,
            n_rounds=consensus_n_rounds,
            use_layer_norm=consensus_use_layer_norm,
            residual=consensus_residual,
            use_inverse_variance=consensus_use_ivw,
        )
        self.aggregator = AggregatorAgent(
            d_hidden=d_hidden,
            out_channels=out_channels,
            pred_len=pred_len,
            d_model=aggregator_d_model,
            n_layers=aggregator_n_layers,
            d_state=aggregator_d_state,
            d_conv=aggregator_d_conv,
            expand=aggregator_expand,
            dropout=aggregator_dropout,
            bidirectional=aggregator_bidirectional,
            norm_eps=norm_eps,
        )
        # Global learnable sigma multiplier.  8-dataset smoke runs showed
        # systemic under-coverage (all 8 datasets Q95 coverage << 0.95); a
        # positive scalar multiplier is the simplest, least-biased correction
        # and can be shrunk back toward 1.0 if the coverage penalty loss feels
        # the band becomes too wide.
        self.sigma_global_multiplier = nn.Parameter(
            torch.tensor(float(sigma_global_multiplier_init), dtype=torch.float32)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm,)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        x_stamp: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> TACFOutput:
        """Run the full TACF pipeline.

        Parameters
        ----------
        x : (B, T, D_in)     look-back window
        x_stamp : (B, T, S)  optional continuous time features
        y : (B, P, D_out)    optional ground-truth (stored as diag _mse aux loss only)

        Returns
        -------
        :class:`TACFOutput` — fused ŷ/σ + α/r/eff_w + full auxiliary outputs.
        """
        B, T, D_in = x.shape
        # ---------------------------------------------------------------
        # 2026-09-26: RevIN on input (DLinear / iTransformer SOTA style).
        # Captures per-window distribution shift that the dataset-level
        # StandardScaler misses (ETTh1 oil-temp maintenance jumps etc).
        # ---------------------------------------------------------------
        if self.use_revin:
            x_norm = self.revin.norm(x)
        else:
            x_norm = x

        # ---------------------------------------------------------------
        # 2026-09-26: DLinear-style direct trend residual shortcut.
        # Computed on the *normalised* input so initial MoA predictions
        # are already anchored at the correct AR(seq_len) baseline, then
        # MoA only has to predict the *residual* nonlinear pattern
        # (seasonal, spike, regime change) on top.
        # ---------------------------------------------------------------
        if self.use_dlinear_trend_head:
            # DLinear-individual-style per-channel T→P linear projection.
            # For each sample b, channel d: pred[b,:,d] = x[b,:,d] @ W[d].T + b[d], where W[d] is (P,T).
            # We do: (B, T, D) → (B, D, T) batch-wise matmul with W.T=(D, T, P) → (B, D, P) → (B, P, D).
            B = x_norm.shape[0]
            x_norm_bd = x_norm.transpose(1, 2)                          # (B, D, T)
            w_dtp = self._trend_W.transpose(-1, -2)                      # (D, T, P)
            # w_dtp.expand(B, D, T, P) would create (B, D, T, P); we want per-D matmul so use einsum
            # (bmm can't broadcast D dim, so we reshape + permute back after contracting T)
            # Alternative: loop over D or use matmul on (B,D,T) × (B,D,T,P)  — use einsum safely:
            # "b d t, d t p -> b d p" contracts t over matching dims.
            trend_bdp = torch.einsum("b d t, d t p -> b d p", x_norm_bd, w_dtp)   # (B, D, P)
            # _trend_b shape: (D, P).  Need (1, P, D).
            trend_pred = trend_bdp.transpose(1, 2) + self._trend_b.T.reshape(1, self.pred_len, self.out_channels)
        else:
            trend_pred = None

        decomp = self.decomposer(x_norm)
        spec_out = self.specialists(decomp.as_dict(), x_stamp)

        mu_raw = spec_out.stack_mu()
        sigma_raw = spec_out.stack_sigma()
        h_raw = spec_out.stack_h()

        consensus_out = self.consensus(mu_raw, sigma_raw, h_raw)

        aggregator_out = self.aggregator(
            h_seq=consensus_out.h,
            mu=consensus_out.mu,
            sigma=consensus_out.sigma,
        )

        # ---- Apply the learnable global sigma multiplier.
        mult = self.sigma_global_multiplier.clamp(min=0.2, max=20.0)
        sigma_scaled = (aggregator_out.sigma * mult).clamp(min=1e-4)

        # ---- Add DLinear trend shortcut to aggregator output.
        if trend_pred is not None:
            y_hat_final = aggregator_out.y_hat + trend_pred
        else:
            y_hat_final = aggregator_out.y_hat

        # ---------------------------------------------------------------
        # 2026-09-26: RevIN denorm on output (y_hat and sigma both).
        # Because sigma is a std, scaling by instance stdev recovers its
        # physical units consistently with y_hat.
        # ---------------------------------------------------------------
        if self.use_revin:
            y_hat_final = self.revin.denorm(y_hat_final)
            sigma_scaled = self.revin.denorm(sigma_scaled)
            # sigma is always positive; denorm may have added the mean
            # component (because denorm is mean + stdev * z).  Subtract
            # the mean shift that was meant for y_hat to keep sigma a
            # pure scale.
            _m_shift = getattr(self.revin, "_last_mean", None)
            _use_last = getattr(self.revin, "subtract_last", False)
            if _use_last:
                _m_shift = getattr(self.revin, "_last_last_val", _m_shift)
            if _m_shift is not None:
                # broadcast _m_shift from (B,1,D) along P dim
                sigma_scaled = sigma_scaled - _m_shift.to(sigma_scaled.device).to(sigma_scaled.dtype)
            sigma_scaled = sigma_scaled.clamp(min=1e-4)

        aux_losses: Dict[str, torch.Tensor] = {}
        aux_losses["orthogonality"] = self.decomposer.orthogonality_loss(x_norm)
        aux_losses["sigma_multiplier"] = mult.detach().reshape(())
        if y is not None:
            with torch.no_grad():
                se = (y_hat_final - y).pow(2)
                aux_losses["_mse_diag"] = se.mean()
            if self.use_dlinear_trend_head:
                # Diagnostics: how much does the linear shortcut contribute?
                with torch.no_grad():
                    tgt_normed = y
                    if self.use_revin and getattr(self.revin, "_last_mean", None) is not None:
                        # compare trend_head vs full on raw y scale by
                        # computing MSE of trend-only on whatever space
                        # we used to compute y_hat_final above — since
                        # y_hat_final went through denorm, use denormed trend.
                        trend_full = self.revin.denorm(trend_pred) if self.use_revin else trend_pred
                        aux_losses["_mse_trend_only"] = (trend_full - y).pow(2).mean()

        return TACFOutput(
            y_hat=y_hat_final,
            sigma=sigma_scaled,
            alpha=aggregator_out.alpha,
            reject=aggregator_out.reject,
            effective_weights=aggregator_out.effective_weights,
            decomposer_out=decomp,
            specialists_out=spec_out,
            consensus_out=consensus_out,
            aggregator_out=aggregator_out,
            aux_losses=aux_losses,
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> Dict[str, int]:
        return {
            "decomposer": self.decomposer.n_params,
            "specialists": self.specialists.n_params,
            **{f"agent_{n}": self.specialists.agents[n].n_params for n in self.agent_names},
            "consensus": self.consensus.n_params,
            "aggregator": self.aggregator.n_params,
            "total": self.n_params,
        }

    def __repr__(self) -> str:
        bd = self.param_breakdown()
        lines = ["TACF("]
        for k, v in bd.items():
            lines.append(f"  {k:<16s}: {v:>12,} params")
        lines.append(")")
        return "\n".join(lines)


TACFMOA = TACF
MOAForwardOutput = TACFOutput
