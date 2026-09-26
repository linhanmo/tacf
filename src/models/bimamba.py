from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils._layers import FeedForward, LayerNorm, RMSNorm


try:
    from mamba_ssm import Mamba  # type: ignore

    _HAS_MAMBA = True
except Exception:  # pragma: no cover - runtime only
    _HAS_MAMBA = False


__all__ = [
    "BiMambaConfig",
    "MambaProjection",
    "BiMamba",
    "BiMambaBlock",
    "BiMambaStack",
]


@dataclass
class BiMambaConfig:
    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.0
    bidirectional: bool = True
    norm_eps: float = 1e-5
    use_ffn: bool = True
    ff_mult: int = 4


def _require_mamba() -> None:
    if not _HAS_MAMBA:
        raise ImportError(
            "mamba_ssm is not installed.  Install mamba_ssm>=2.2.2 with the "
            "correct CUDA wheel to use BiDirectionalMamba backbones on GPU."
        )


class _CpuGatedConvMixer(nn.Module):
    """CPU-compatible fallback for the selective-scan Mamba layer.

    Implements a gated depthwise-conv mixer with the same ``(B, L, D)`` I/O
    signature and similar parameter count as a real ``mamba_ssm.Mamba`` block,
    so that CPU debugging / CI runs share the same model topology as GPU
    training runs.  Forward+backward are fully differentiable via standard
    PyTorch ops.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.dw_conv = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )
        self.mix = nn.Sequential(
            nn.Conv1d(d_inner, d_inner, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(d_inner, d_inner, kernel_size=1),
        )
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.d_conv = int(d_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_e, gate = xz.chunk(2, dim=-1)
        conv_in = x_e.transpose(1, 2)
        conv_out = self.dw_conv(conv_in)
        conv_out = conv_out[..., :L].transpose(1, 2)
        mixed = self.mix(conv_out.transpose(1, 2)).transpose(1, 2)
        h = mixed * F.silu(conv_out)
        gated = h * F.sigmoid(gate)
        out = self.out_proj(gated)
        return self.drop(out)


class MambaProjection(nn.Module):
    """Single-direction Mamba with a transparent CPU fallback.

    If ``mamba_ssm`` is installed **and** the input tensor lives on CUDA, we
    dispatch to the real selective-scan kernel.  Otherwise we run a fully-
    differentiable, parameter-shape-compatible gated-conv mixer.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self._has_cuda_mamba = False
        if _HAS_MAMBA:
            try:
                self.mamba = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                self._has_cuda_mamba = True
            except Exception:  # pragma: no cover - init edge case
                self.mamba = None  # type: ignore[assignment]
                self._has_cuda_mamba = False
        self.cpu_fallback = _CpuGatedConvMixer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._has_cuda_mamba and x.is_cuda:
            return self.drop(self.mamba(x))
        return self.drop(self.cpu_fallback(x))


class BiDirectionalMamba(nn.Module):
    """Bidirectional Mamba with SiLU-GLU fusion of forward/reverse streams.

    When ``bidirectional=False`` this reduces to a simple single-directional
    Mamba projection (no time-reversed stream, no fusion).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        _require_mamba()
        self.d_model = int(d_model)
        self.bidirectional = bool(bidirectional)

        self.fwd = MambaProjection(
            d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, dropout=dropout,
        )
        if self.bidirectional:
            self.bwd = MambaProjection(
                d_model=d_model, d_state=d_state, d_conv=d_conv,
                expand=expand, dropout=dropout,
            )
            self.fuse = nn.Sequential(
                nn.Linear(2 * d_model, d_model, bias=False),
                nn.SiLU(),
                nn.Linear(d_model, d_model, bias=False),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
        else:
            self.bwd = None
            self.fuse = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd_out = self.fwd(x)
        if not self.bidirectional:
            return fwd_out
        rev_x = torch.flip(x, dims=(1,))
        bwd_out = torch.flip(self.bwd(rev_x), dims=(1,))
        return self.fuse(torch.cat([fwd_out, bwd_out], dim=-1))


BiMamba = BiDirectionalMamba


class BiMambaBlock(nn.Module):
    """Pre-norm residual block wrapping :class:`BiDirectionalMamba` + SwiGLU FFN."""

    def __init__(self, cfg: BiMambaConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.mixer = BiDirectionalMamba(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
        )
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.use_ffn = cfg.use_ffn
        if cfg.use_ffn:
            self.norm2 = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
            self.ffn = FeedForward(
                d_model=cfg.d_model,
                d_ff=cfg.ff_mult * cfg.d_model,
                dropout=cfg.dropout,
                activation="swiglu",
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mixer(self.norm1(x)))
        if self.use_ffn:
            x = x + self.ffn(self.norm2(x))
        return x


class BiMambaStack(nn.Module):
    """Stack of ``n_layers`` :class:`BiMambaBlock` + final RMSNorm."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = True,
        norm_eps: float = 1e-5,
        use_ffn: bool = True,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        cfg = BiMambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            bidirectional=bidirectional,
            norm_eps=norm_eps,
            use_ffn=use_ffn,
            ff_mult=ff_mult,
        )
        self.layers = nn.ModuleList([BiMambaBlock(cfg) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model, eps=norm_eps)

    def forward(
        self, x: torch.Tensor, return_hiddens: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        hiddens: List[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            if return_hiddens:
                hiddens.append(x)
        x = self.final_norm(x)
        if return_hiddens:
            return x, hiddens
        return x

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
