from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "RMSNorm",
    "LayerNorm",
    "FeedForward",
    "Conv1dSame",
    "SinusoidalPositionalEncoding",
    "PositionalEmbedding",
    "DataEmbedding",
]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp = x.float()
        rms = x_fp.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = x_fp * rms
        if self.weight is not None:
            out = out * self.weight.view(1, 1, -1) if out.dim() > 2 else out * self.weight
        return out.to(in_dtype)


class LayerNorm(nn.Module):
    """FP32-safe LayerNorm that always computes in float32."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = float(eps)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp = x.float()
        mu = x_fp.mean(dim=-1, keepdim=True)
        va = x_fp.sub(mu).pow(2).mean(dim=-1, keepdim=True)
        out = x_fp.sub(mu).mul(va.add(self.eps).rsqrt())
        if self.weight is not None:
            w = self.weight.view(*([1] * (out.dim() - 1)), -1)
            b = self.bias.view(*([1] * (out.dim() - 1)), -1)
            out = out.mul(w).add(b)
        return out.to(in_dtype)


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        d_out: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        d_out = d_out or d_model
        activation = (activation or "swiglu").lower()
        if activation in ("swiglu", "glu"):
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_model, d_ff, bias=False)
            self.w3 = nn.Linear(d_ff, d_out, bias=False)
            self.act = nn.SiLU()
            self.forward = self._forward_swiglu  # type: ignore[assignment]
        else:
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w3 = nn.Linear(d_ff, d_out, bias=False)
            self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
            self.forward = self._forward_ff  # type: ignore[assignment]
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _forward_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(self.act(self.w1(x)) * self.w2(x)))

    def _forward_ff(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(self.act(self.w1(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError  # overridden in __init__


class Conv1dSame(nn.Module):
    """Conv1d with SAME padding (asymmetric when kernel is even), supports causal."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        d = int(dilation)
        s = int(stride)
        if causal:
            pad_total = d * (k - 1)
            self.pad_left = pad_total
            self.pad_right = 0
        else:
            pad_total = d * (k - 1) - (s - 1)
            self.pad_left = pad_total // 2
            self.pad_right = pad_total - self.pad_left
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k,
            stride=s,
            dilation=d,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_left + self.pad_right > 0:
            x = F.pad(x, (self.pad_left, self.pad_right))
        return self.conv(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) or (B, L)."""
        L = x.size(1)
        return self.drop(x + self.pe[:, :L])


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape[0], x.shape[1]
        idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        return self.emb(idx)


class DataEmbedding(nn.Module):
    """Value projection + positional (optional) + token embedding (optional)."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        dropout: float = 0.0,
        max_len: int = 5000,
        use_pos: bool = True,
        use_token: bool = True,
    ) -> None:
        super().__init__()
        self.value_emb = nn.Sequential(
            nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, padding_mode="replicate"),
        )
        self.use_pos = bool(use_pos)
        self.use_token = bool(use_token)
        if self.use_pos:
            self.position_emb = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        else:
            self.position_emb = None
        if self.use_token:
            self.token_emb = PositionalEmbedding(d_model, max_len=max_len)
        else:
            self.token_emb = None
        self.time_proj: Optional[nn.Linear] = None
        self.norm = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def bind_time_features(self, stamp_dim: int) -> None:
        if stamp_dim > 0:
            self.time_proj = nn.Linear(stamp_dim, self.value_emb[0].out_channels)
        else:
            self.time_proj = None

    def forward(
        self,
        x: torch.Tensor,
        x_stamp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.value_emb(x.transpose(1, 2)).transpose(1, 2)
        if self.position_emb is not None:
            h = self.position_emb(h)
        if self.token_emb is not None:
            h = h + self.token_emb(x)
        if self.time_proj is not None and x_stamp is not None and x_stamp.shape[-1] > 0:
            h = h + self.time_proj(x_stamp)
        return self.drop(self.norm(h))
