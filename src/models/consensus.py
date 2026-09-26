from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils._layers import RMSNorm


__all__ = ["ConsensusOutput", "MultiHeadAgentAttention", "ConsensusLayer"]


@dataclass
class ConsensusOutput:
    mu: torch.Tensor
    sigma: torch.Tensor
    h: torch.Tensor
    comm_weights: torch.Tensor
    delta_mu: torch.Tensor
    delta_sigma: torch.Tensor


def _inverse_variance_fusion(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Attention-weighted inverse-variance fusion.

    mu, sigma : (B, K, P, D)    predictions / uncertainties across K agents
    weights   : (B, K_q, K_k)   per-query mixing kernel (e.g. row-normalised attn)
    returns   : (B, K_q, P, D)  fused μ for each query agent
    """
    precision = 1.0 / (sigma.pow(2) + eps)
    w_prec = weights.unsqueeze(-1).unsqueeze(-1) * precision.unsqueeze(1)
    denom = w_prec.sum(dim=2) + eps
    num = (w_prec * mu.unsqueeze(1)).sum(dim=2)
    return num / denom


class MultiHeadAgentAttention(nn.Module):
    """Multi-head self-attention over the K=3 agent "tokens".

    Each "token" is a per-agent latent vector ``h_k ∈ R^d``.  Attention runs
    purely over the agent axis, treating the batch as independent sequences of
    length K.
    """

    def __init__(
        self, d: int, d_attn: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if d_attn % num_heads != 0:
            raise ValueError(
                f"d_attn ({d_attn}) must be divisible by num_heads ({num_heads})"
            )
        self.d = int(d)
        self.d_attn = int(d_attn)
        self.num_heads = int(num_heads)
        self.d_head = self.d_attn // self.num_heads

        self.q_proj = nn.Linear(d, d_attn, bias=False)
        self.k_proj = nn.Linear(d, d_attn, bias=False)
        self.v_proj = nn.Linear(d, d_attn, bias=False)
        self.o_proj = nn.Linear(d_attn, d, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = self.d_head ** -0.5

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h : (B, K, d) → updated_h (B, K, d), mean_attn_weights (B, K, K)."""
        B, K, _ = h.shape
        q = (
            self.q_proj(h)
            .view(B, K, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(h)
            .view(B, K, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(h)
            .view(B, K, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn.mean(dim=1)

        attn_d = self.dropout(attn)
        out = torch.matmul(attn_d, v)
        out = out.transpose(1, 2).contiguous().view(B, K, self.d_attn)
        out = self.o_proj(out)
        return out, attn_weights


class ConsensusLayer(nn.Module):
    """Two-round inter-agent consensus + inverse-variance correction.

    Round 1 (broadcast)
        Agents make their ``(μ, σ, h)`` tuples globally visible.
    Round 2 (refine)
        (a) :class:`MultiHeadAgentAttention` computes context vectors ``Δh``
            and a per-agent mixing kernel ``W_attn ∈ [0,1]^{K×K}``.
        (b) ``W_attn`` is used as the mixing kernel for an **inverse-
            variance-weighted** consensus ``μ_iv = Σ w_{q,k}·Prec_k·μ_k / Σ w_{q,k}·Prec_k``,
            and a precision-fused ``σ_iv = 1 / sqrt(Σ w_{q,k}·Prec_k)``.
        (c) Parameterised sigmoid gates blend the consensus deltas back into
            each agent's original estimate, preserving full gradient flow
            through the gating logits (which are a function of ``h_q + Δh_q``).

    The output ``{μ_k^R, σ_k^R, h_k^R}`` has exactly the same tensor shapes
    as the inputs but has been "averaged in the neighbourhood" shaped by both
    agent-agent affinity and per-agent uncertainty.
    """

    def __init__(
        self,
        d_hidden: int,
        out_channels: int,
        pred_len: int,
        d_attn: Optional[int] = None,
        num_heads: int = 4,
        attn_dropout: float = 0.05,
        n_rounds: int = 2,
        use_layer_norm: bool = True,
        residual: bool = True,
        use_inverse_variance: bool = True,
    ) -> None:
        super().__init__()
        self.d_hidden = int(d_hidden)
        self.out_channels = int(out_channels)
        self.pred_len = int(pred_len)
        self.n_rounds = max(1, int(n_rounds))
        self.residual = bool(residual)
        self.use_inverse_variance = bool(use_inverse_variance)

        d_attn = int(d_attn) if d_attn else d_hidden
        self.attn = MultiHeadAgentAttention(
            d=d_hidden,
            d_attn=d_attn,
            num_heads=num_heads,
            dropout=attn_dropout,
        )

        self.norm_h = RMSNorm(d_hidden, eps=1e-5) if use_layer_norm else nn.Identity()
        self.norm_mu = (
            RMSNorm(out_channels, eps=1e-5) if use_layer_norm else nn.Identity()
        )

        self.gate_h = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.Sigmoid(),
        )
        self.gate_mu = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 4),
            nn.SiLU(),
            nn.Linear(d_hidden // 4, 1),
            nn.Sigmoid(),
        )
        self.delta_sigma_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 4),
            nn.SiLU(),
            nn.Linear(d_hidden // 4, pred_len * out_channels),
        )
        self.min_sigma = 1e-4

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        h: torch.Tensor,
    ) -> ConsensusOutput:
        """
        mu    : (B, K, P, D)
        sigma : (B, K, P, D)
        h     : (B, K, d)
        """
        B, K, P, D = mu.shape
        if sigma.shape != (B, K, P, D):
            raise ValueError(
                f"sigma shape mismatch: expected {(B, K, P, D)}, got {tuple(sigma.shape)}"
            )
        if h.shape[:2] != (B, K):
            raise ValueError(
                f"h shape mismatch: leading dims {(B, K)} expected, got {tuple(h.shape[:2])}"
            )

        h_in = h
        mu_in = mu
        for _ in range(self.n_rounds):
            dh, attn_w = self.attn(h_in)
            h_res = (
                self.norm_h(h_in + dh)
                if self.residual
                else self.norm_h(dh)
            )

            h_cat = torch.cat([h_in, h_res], dim=-1)
            g_h = self.gate_h(h_cat)
            h_new = h_in + g_h * (h_res - h_in)

            if self.use_inverse_variance:
                mu_iv = _inverse_variance_fusion(mu_in, sigma, attn_w)
            else:
                mu_weighted = (
                    attn_w.unsqueeze(-1).unsqueeze(-1) * mu_in.unsqueeze(1)
                ).sum(dim=2)
                mu_iv = mu_weighted

            delta_mu = self.norm_mu(mu_iv - mu_in)
            g_mu = self.gate_mu(h_new).view(B, K, 1, 1)
            mu_new = mu_in + g_mu * delta_mu

            precision = 1.0 / (sigma.pow(2) + 1e-6)
            prec_weighted = (
                attn_w.unsqueeze(-1).unsqueeze(-1) * precision.unsqueeze(1)
            ).sum(dim=2)
            sigma_iv = 1.0 / torch.sqrt(prec_weighted + 1e-6)
            delta_sigma = sigma_iv - sigma

            delta_sigma_raw = self.delta_sigma_head(h_new).view(B, K, P, D)
            g_sigma = torch.sigmoid(delta_sigma_raw)
            sigma_new = sigma + g_sigma * delta_sigma
            sigma_new = torch.clamp(sigma_new, min=self.min_sigma)

            # Prepare for next round
            h_in = h_new
            mu_in = mu_new
            sigma = sigma_new

        return ConsensusOutput(
            mu=mu_new,
            sigma=sigma_new,
            h=h_new,
            comm_weights=attn_w,
            delta_mu=delta_mu,
            delta_sigma=delta_sigma_raw,
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
