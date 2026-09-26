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


__all__ = ["TACFOutput", "TACF"]


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
        sigma_global_multiplier_init: float = 3.0,
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
        decomp = self.decomposer(x)
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
        # Parameter is stored as a plain scalar (initialised to the user-supplied
        # init value, typically 3.0).  We clamp it to a sensible positive range
        # so early-stage instabilities can't blow sigma up, and then multiply
        # sigma = sigma_aggregator * mult, clamp floor 1e-4 from head.
        mult = self.sigma_global_multiplier.clamp(min=0.2, max=20.0)
        sigma_scaled = (aggregator_out.sigma * mult).clamp(min=1e-4)

        aux_losses: Dict[str, torch.Tensor] = {}
        aux_losses["orthogonality"] = self.decomposer.orthogonality_loss(x)
        # Store sigma multiplier as a tensor-typed scalar for later logging.
        # Do NOT multiply with sigma_scaled; just detach and reshape to () so
        # downstream can `float()` it cleanly.
        aux_losses["sigma_multiplier"] = mult.detach().reshape(())
        if y is not None:
            with torch.no_grad():
                se = (aggregator_out.y_hat - y).pow(2)
                aux_losses["_mse_diag"] = se.mean()

        return TACFOutput(
            y_hat=aggregator_out.y_hat,
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
