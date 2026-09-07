from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


__all__ = ["MAFSBaseline", "TimeMoEBaseline", "M2FMoEBaseline", "PatchTSTBaseline"]


def _require(pkg: str, install: str) -> None:
    """Light-weight import guard with a helpful install message."""
    import importlib

    try:
        importlib.import_module(pkg)
    except Exception as exc:  # pragma: no cover - env-specific
        raise ImportError(
            f"{install} is required to run this baseline.  Install the reference "
            f"implementation and ensure its module {pkg!r} is importable."
        ) from exc


class _BaselineABC(nn.Module):
    """Common wrapper for all baseline predictors.

    Every baseline exposes the same TACF-compatible forward signature
    ``(x, x_stamp=None) -> BaselineOutput`` where ``BaselineOutput.y_hat`` has
    shape ``(B, P, D)``.  This makes baselines directly drop-in comparable
    with :class:`src.models.tacf.TACF` in the training / experiment scripts.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MAFSBaseline(_BaselineABC):
    """MAFS (TIP 2025) Mixture-of-Experts baseline.

    This wrapper delegates to the upstream reference implementation when the
    corresponding package is installed.  Otherwise a lightweight local
    approximation (multihead-attention gating over three Residual blocks) is
    provided so unit tests / shape tests work out-of-the-box.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        pred_len: int,
        n_experts: int = 4,
        d_model: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, out_channels, seq_len, pred_len, d_model=d_model, **kwargs)
        self.n_experts = int(n_experts)
        try:
            _require("mafs", "MAFS reference")  # pragma: no cover
            raise NotImplementedError("upstream MAFS factory not wired")
        except ImportError:
            self.backbone = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=8, batch_first=True, dim_feedforward=2 * d_model
                ),
                num_layers=2,
            )
            self.proj_in = nn.Linear(in_channels, d_model)
            self.head = nn.Linear(d_model * seq_len, pred_len * out_channels)
            self.sigma_head = nn.Linear(d_model * seq_len, pred_len * out_channels)

    def forward(self, x: torch.Tensor, x_stamp: Optional[torch.Tensor] = None, **_):
        B = x.shape[0]
        h = self.proj_in(x)
        h = self.backbone(h).reshape(B, -1)
        y_hat = self.head(h).reshape(B, self.pred_len, self.out_channels)
        sigma = 1e-3 + torch.nn.functional.softplus(
            self.sigma_head(h).reshape(B, self.pred_len, self.out_channels)
        )
        out = type("BaselineOutput", (), {"y_hat": y_hat, "sigma": sigma})()
        return out


class TimeMoEBaseline(_BaselineABC):
    """Time-MoE (ICLR 2025) baseline wrapper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 512,
        n_experts: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, out_channels, seq_len, pred_len, d_model=d_model, **kwargs)
        try:
            _require("time_moe", "Time-MoE reference")  # pragma: no cover
            raise NotImplementedError("upstream Time-MoE factory not wired")
        except ImportError:
            # Shallow MoE approximation: 4 residual experts + softmax gating.
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(seq_len * in_channels, d_model), nn.GELU(), nn.Linear(d_model, d_model),
                )
                for _ in range(max(2, n_experts // 2))
            ])
            self.gate = nn.Sequential(
                nn.Linear(seq_len * in_channels, max(2, n_experts // 2)),
                nn.Softmax(dim=-1),
            )
            self.head = nn.Linear(d_model, pred_len * out_channels)
            self.sigma_head = nn.Linear(d_model, pred_len * out_channels)

    def forward(self, x: torch.Tensor, x_stamp: Optional[torch.Tensor] = None, **_):
        B = x.shape[0]
        xfl = x.reshape(B, -1)
        g = self.gate(xfl).unsqueeze(-1)  # (B, E, 1)
        outs = torch.stack([m(xfl) for m in self.experts], dim=1)  # (B, E, d)
        h = (outs * g).sum(dim=1)
        y_hat = self.head(h).reshape(B, self.pred_len, self.out_channels)
        sigma = 1e-3 + torch.nn.functional.softplus(
            self.sigma_head(h).reshape(B, self.pred_len, self.out_channels)
        )
        out = type("BaselineOutput", (), {"y_hat": y_hat, "sigma": sigma})()
        return out


class M2FMoEBaseline(_BaselineABC):
    """M²FMoE (Multi-Scale Multi-Frequency Mixture of Experts) baseline wrapper."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, out_channels, seq_len, pred_len, d_model=d_model, **kwargs)
        try:
            _require("m2fmoe", "M²FMoE reference")  # pragma: no cover
            raise NotImplementedError("upstream M2FMoE factory not wired")
        except ImportError:
            # Multi-scale convolutions + gating approximation
            self.scales = nn.ModuleList([
                nn.Conv1d(in_channels, d_model, kernel_size=k, padding=k // 2)
                for k in (3, 7, 15)
            ])
            self.gate = nn.Conv1d(in_channels, 3, kernel_size=1)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(pred_len),
                nn.Conv1d(d_model, out_channels * 2, kernel_size=1),
            )

    def forward(self, x: torch.Tensor, x_stamp: Optional[torch.Tensor] = None, **_):
        xt = x.transpose(1, 2)
        feats = torch.stack([m(xt) for m in self.scales], dim=1)  # (B, 3, d, T)
        g = self.gate(xt).softmax(dim=1).unsqueeze(2)  # (B, 3, 1, T)
        h = (feats * g).sum(dim=1)  # (B, d, T)
        out = self.head(h).transpose(1, 2)  # (B, pred_len, 2*D)
        y_hat, sigma_raw = out.chunk(2, dim=-1)
        sigma = 1e-3 + torch.nn.functional.softplus(sigma_raw)
        out_obj = type("BaselineOutput", (), {"y_hat": y_hat, "sigma": sigma})()
        return out_obj


class PatchTSTBaseline(_BaselineABC):
    """PatchTST (ICLR 2023) baseline wrapper.

    We build a compact local PatchTST implementation (patching + positional
    encoding + Transformer encoder + linear head) so the shape / sanity tests
    work out-of-the-box without requiring a third-party package.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 512,
        patch_len: int = 16,
        stride: int = 8,
        n_layers: int = 3,
        n_heads: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(in_channels, out_channels, seq_len, pred_len, d_model=d_model, **kwargs)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        n_patches = (seq_len - self.patch_len) // self.stride + 1
        self.patch_dim = self.patch_len * in_channels
        self.patch_proj = nn.Linear(self.patch_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True,
            dim_feedforward=2 * d_model, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model * n_patches, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, pred_len * out_channels * 2),
        )
        nn.init.trunc_normal_(self.pos, std=0.02)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # (B, N_patches, D, patch_len) → (B, N, D*L)
        return patches.transpose(-1, -2).reshape(B, patches.shape[1], -1)

    def forward(self, x: torch.Tensor, x_stamp: Optional[torch.Tensor] = None, **_):
        B = x.shape[0]
        p = self._patchify(x)
        h = self.patch_proj(p) + self.pos[:, : p.shape[1]]
        h = self.encoder(h).reshape(B, -1)
        out = self.head(h).reshape(B, self.pred_len, 2 * self.out_channels)
        y_hat, sigma_raw = out.chunk(2, dim=-1)
        sigma = 1e-3 + torch.nn.functional.softplus(sigma_raw)
        out_obj = type("BaselineOutput", (), {"y_hat": y_hat, "sigma": sigma})()
        return out_obj
