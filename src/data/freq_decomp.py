from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import AGENT_NAMES
from ..utils._layers import Conv1dSame, RMSNorm


__all__ = ["DecomposerOutput", "LearnableDecomposer"]


@dataclass
class DecomposerOutput:
    trend: torch.Tensor
    cycle: torch.Tensor
    local: torch.Tensor

    def as_list(self) -> List[torch.Tensor]:
        return [self.trend, self.cycle, self.local]

    def as_dict(self) -> dict:
        return dict(zip(AGENT_NAMES, self.as_list()))


class _ComponentBranch(nn.Module):
    """Single branch of the learnable frequency decomposer.

    Channel projection → large-kernel depthwise Conv → SE-style gating →
    channel-mixing 1×1 Conv → 1×1 output projection → residual + RMSNorm.
    """

    def __init__(
        self,
        in_channels: int,
        d_hidden: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        causal: bool = False,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.d_hidden = int(d_hidden)
        self.kernel_size = int(kernel_size)
        self.residual = bool(residual)

        self.input_proj = nn.Conv1d(in_channels, d_hidden, kernel_size=1, bias=bias)
        self.conv = Conv1dSame(
            in_channels=d_hidden,
            out_channels=d_hidden,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=d_hidden,
            bias=bias,
            causal=causal,
        )
        self.mix = nn.Sequential(
            nn.Conv1d(d_hidden, d_hidden, kernel_size=1, bias=bias),
            nn.GELU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size=1, bias=bias),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_hidden, max(16, d_hidden // 4), kernel_size=1),
            nn.GELU(),
            nn.Conv1d(max(16, d_hidden // 4), d_hidden, kernel_size=1),
            nn.Sigmoid(),
        )
        self.output_proj = nn.Conv1d(d_hidden, in_channels, kernel_size=1, bias=bias)
        self.norm = RMSNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C_in) → (B, L, C_in)."""
        h = x.transpose(1, 2).contiguous()
        h = self.input_proj(h)
        h = self.conv(h)
        g = self.gate(h)
        h = self.mix(h) * g + h
        out = self.output_proj(h).transpose(1, 2).contiguous()
        if self.residual:
            out = out + x
        return self.norm(out)


class LearnableDecomposer(nn.Module):
    """Learnable frequency decomposer via 3 parallel Conv1D branches.

    Architecture (follows the canonical ASCII diagram in ``Architecture.md``):

    Input X ∈ R^(B, T, D)
        ┌─ Conv1D(W_t, k=64) → low-pass emphasis  → X_trend  ∈ R^(B, T, D)
        ├─ Conv1D(W_c, k=32) → band-pass emphasis → X_cycle  ∈ R^(B, T, D)
        └─ Conv1D(W_l, k=16) → high-pass emphasis → X_local  ∈ R^(B, T, D)

    Each branch uses depthwise-large-kernel convolution with SE-style gating
    and a channel-mixing 1×1 block.  A per-channel softmax mixture gate is
    applied across the three branches to enforce the constraint that the
    three components recombine additively onto the original signal manifold.
    An optional pairwise orthogonality loss pushes components apart in the
    (B·T, D) product space.
    """

    agent_names: Tuple[str, ...] = AGENT_NAMES

    def __init__(
        self,
        in_channels: int,
        d_hidden: Optional[int] = None,
        kernel_trend: int = 64,
        kernel_cycle: int = 32,
        kernel_local: int = 16,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        causal: bool = False,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        if d_hidden is None:
            d_hidden = max(64, in_channels * 4)
        self.d_hidden = int(d_hidden)
        self.kernel_sizes = (int(kernel_trend), int(kernel_cycle), int(kernel_local))

        self.trend_branch = _ComponentBranch(
            in_channels=in_channels,
            d_hidden=d_hidden,
            kernel_size=kernel_trend,
            stride=stride,
            dilation=dilation,
            bias=bias,
            causal=causal,
            residual=residual,
        )
        self.cycle_branch = _ComponentBranch(
            in_channels=in_channels,
            d_hidden=d_hidden,
            kernel_size=kernel_cycle,
            stride=stride,
            dilation=dilation,
            bias=bias,
            causal=causal,
            residual=residual,
        )
        self.local_branch = _ComponentBranch(
            in_channels=in_channels,
            d_hidden=d_hidden,
            kernel_size=kernel_local,
            stride=stride,
            dilation=dilation,
            bias=bias,
            causal=causal,
            residual=residual,
        )
        self.mix_gate = nn.Sequential(
            nn.Conv1d(3 * in_channels, 3 * in_channels, kernel_size=1, groups=3),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> DecomposerOutput:
        raw_trend = self.trend_branch(x)
        raw_cycle = self.cycle_branch(x)
        raw_local = self.local_branch(x)

        B, L, D = x.shape
        stacked = torch.cat([raw_trend, raw_cycle, raw_local], dim=-1)
        gates = self.mix_gate(stacked.transpose(1, 2)).transpose(1, 2)
        g_t, g_c, g_l = gates.chunk(3, dim=-1)

        return DecomposerOutput(
            trend=raw_trend * g_t, cycle=raw_cycle * g_c, local=raw_local * g_l
        )

    def orthogonality_loss(self, x: torch.Tensor) -> torch.Tensor:
        d = self(x)
        loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        a, b, c = d.trend, d.cycle, d.local
        for u, v in ((a, b), (a, c), (b, c)):
            u_flat = u.flatten(1)
            v_flat = v.flatten(1)
            u_n = F.normalize(u_flat, p=2, dim=-1)
            v_n = F.normalize(v_flat, p=2, dim=-1)
            sim = (u_n * v_n).sum(dim=-1).pow(2).mean()
            loss = loss + sim
        return loss

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
