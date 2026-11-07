from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bimamba import BiMambaStack
from ..utils._layers import RMSNorm


__all__ = ["AggregatorOutput", "AggregatorAgent"]


@dataclass
class AggregatorOutput:
    y_hat: torch.Tensor
    sigma: torch.Tensor
    alpha: torch.Tensor
    reject: torch.Tensor
    effective_weights: torch.Tensor
    comm_state: torch.Tensor


class AggregatorAgent(nn.Module):
    """Light-weight aggregator that arbitrates across the three specialists.

    Input sequence
    --------------
    h_seq ∈ R^(B, 3, d)  – the stacked, post-consensus latent vectors of the
    K=3 agents ordered (trend, cycle, local).

    Processing
    ----------
    A *tiny* BiMamba stack (``d_model=256, n_layers=2``, ~0.5M parameters)
    treats the agent axis as the sequence dimension.  Because the sequence
    length is just 3, LightMamba operates here as a learnable cross-agent
    feature mixer rather than as a long-range selective scan.

    Three heads
    -----------
    alpha_k ∈ R^3  (softmax) — global mixture weights across specialists.
    reject_k ∈ R^3 (sigmoid) — per-agent reject signal in [0, 1].
    fusion_scale  (sigmoid)  — global scalar applied to the stacked μ before
                               the precision-weighted fusion (lets the model
                               compensate for magnitude drift between the
                               specialists trained in isolation).

    Outputs
    -------
    ŷ ∈ R^(B, P, D), σ ∈ R^(B, P, D) produced via a precision-weighted
    mixture using ``w_k ∝ α_k · (1 - r_k) · Prec_k`` renormalised to sum 1
    per batch item.  This way rejected agents are smoothly down-weighted but
    never hard-disconnected, which keeps the gradient flowing through r_k
    even when ``r_k → 1``.

    Also returned are the raw ``alpha`` and ``reject`` vectors plus the
    effective mixture weights ``w_k`` and the final LightMamba state
    ``comm_state ∈ R^(B, 3, d_model)`` (useful for downstream analyses).
    """

    def __init__(
        self,
        d_hidden: int,
        out_channels: int,
        pred_len: int,
        d_model: int = 256,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
        bidirectional: bool = True,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_hidden = int(d_hidden)
        self.out_channels = int(out_channels)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)

        self.input_proj = nn.Sequential(
            nn.Linear(d_hidden, d_model),
            RMSNorm(d_model, eps=norm_eps),
        )
        self.light_mamba = BiMambaStack(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            bidirectional=bidirectional,
            norm_eps=norm_eps,
            use_ffn=True,
            ff_mult=2,
        )

        flat_dim = d_model * 3
        self.head_alpha = nn.Sequential(
            nn.Linear(flat_dim, d_model),
            nn.GELU(),
            RMSNorm(d_model, eps=norm_eps),
            nn.Linear(d_model, 3),
        )
        self.head_reject = nn.Sequential(
            nn.Linear(flat_dim, d_model // 2),
            nn.GELU(),
            RMSNorm(d_model // 2, eps=norm_eps),
            nn.Linear(d_model // 2, 3),
        )
        self.fusion_scale = nn.Sequential(
            nn.Linear(flat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.min_sigma = 1e-4

    @staticmethod
    def _flatten(h: torch.Tensor) -> torch.Tensor:
        B, K, _ = h.shape
        return h.reshape(B, -1)

    def _fuse(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        alpha: torch.Tensor,
        reject: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Variance-weighted mixture fusion.

        mu, sigma : (B, K, P, D)
        alpha     : (B, K)   softmax weights
        reject    : (B, K)   sigmoid rejection signals
        """
        keep = 1.0 - reject
        eff = alpha * keep
        eff = eff / (eff.sum(dim=-1, keepdim=True) + 1e-6)
        eff4 = eff.unsqueeze(-1).unsqueeze(-1)

        precision = 1.0 / (sigma.pow(2) + 1e-6)
        w = eff4 * precision
        denom = w.sum(dim=1) + 1e-6
        y_hat = (w * mu).sum(dim=1) / denom
        sigma_out = 1.0 / torch.sqrt(denom)
        sigma_out = torch.clamp(sigma_out, min=self.min_sigma)
        return y_hat, sigma_out, eff

    def forward(
        self,
        h_seq: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> AggregatorOutput:
        """
        h_seq : (B, K, d)   K=3 stacked post-consensus agent latents
        mu    : (B, K, P, D)
        sigma : (B, K, P, D)
        """
        B, K, d = h_seq.shape
        if K != 3:
            raise ValueError(
                f"AggregatorAgent expects K=3 stacked latents, got K={K}"
            )
        h = self.input_proj(h_seq)
        h = self.light_mamba(h)
        h_flat = self._flatten(h)

        alpha_logits = self.head_alpha(h_flat)
        alpha = F.softmax(alpha_logits, dim=-1)

        reject_logits = self.head_reject(h_flat)
        reject = torch.sigmoid(reject_logits)

        scale = torch.sigmoid(self.fusion_scale(h_flat)).view(B, 1, 1, 1)
        mu_scaled = mu * scale

        y_hat, sigma_out, eff_w = self._fuse(mu_scaled, sigma, alpha, reject)

        return AggregatorOutput(
            y_hat=y_hat,
            sigma=sigma_out,
            alpha=alpha,
            reject=reject,
            effective_weights=eff_w,
            comm_state=h,
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
