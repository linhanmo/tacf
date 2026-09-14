from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from ..data.dataset import AGENT_NAMES


__all__ = [
    "AgentLossesOutput",
    "total_variation_loss",
    "seasonal_fourier_l1_loss",
    "local_sparsity_l1",
    "build_agent_heterogeneous_losses",
]


@dataclass
class AgentLossesOutput:
    tv_trend: torch.Tensor
    seasonal_cycle: torch.Tensor
    sparse_local: torch.Tensor
    total: torch.Tensor


def total_variation_loss(
    y_hat_k: torch.Tensor, reduce: str = "mean"
) -> torch.Tensor:
    """Total-variation loss — encourages smoothly-varying (trend-like) output.

    TV = Σ |y_t+1 - y_t| across the prediction horizon and features.
    """
    diff = y_hat_k[..., 1:, :] - y_hat_k[..., :-1, :]
    v = diff.abs()
    if reduce == "mean":
        return v.mean()
    if reduce == "sum":
        return v.sum()
    return v


def seasonal_fourier_l1_loss(
    y_hat_k: torch.Tensor,
    target: torch.Tensor,
    freq_bands: int = 8,
    reduce: str = "mean",
) -> torch.Tensor:
    """Encourage a cycle-specialist to fit the low-rank seasonal Fourier basis.

    Computes the (real) FFT over the prediction horizon, compares normalised
    magnitudes of y_hat_k vs target on the lowest ``freq_bands`` frequency
    bins, and returns an L1 discrepancy on those amplitudes.  This biases the
    cycle agent towards fitting seasonal structure rather than trend.

    Notes
    -----
    cuFFT has two well-known limitations under AMP / float16:
      (1) `ComplexHalf` support is experimental and emits warnings.
      (2) Real FFTs only accept signal lengths that are powers of two when
          executed in half precision on CUDA.
    Because common TACF prediction horizons (96, 168, 192, 336, 720, ...)
    are typically *not* powers of two, we promote inputs to ``float32`` just
    around the FFT call, then cast the final scalar loss back to the input
    dtype.  This keeps autocast / GradScaler happy elsewhere and preserves
    gradient flow through the FFT (all ops are real-valued after ``.abs()``).
    """
    orig_dtype = y_hat_k.dtype
    need_promote = orig_dtype != torch.float32 and orig_dtype != torch.float64
    B, P, D = y_hat_k.shape
    yhat = y_hat_k.transpose(1, 2).reshape(-1, P)
    tgt = target.transpose(1, 2).reshape(-1, P)
    if need_promote:
        yhat = yhat.float()
        tgt = tgt.float()
    Fh = torch.fft.rfft(yhat, n=P, dim=-1).abs()
    Ft = torch.fft.rfft(tgt, n=P, dim=-1).abs()
    if freq_bands > Fh.shape[-1]:
        freq_bands = Fh.shape[-1]
    Fh_n = Fh[..., :freq_bands] / (
        Fh[..., :freq_bands].sum(dim=-1, keepdim=True) + 1e-6
    )
    Ft_n = Ft[..., :freq_bands] / (
        Ft[..., :freq_bands].sum(dim=-1, keepdim=True) + 1e-6
    )
    v = (Fh_n - Ft_n).abs()
    if reduce == "mean":
        loss = v.mean()
    elif reduce == "sum":
        loss = v.sum()
    else:
        loss = v
    if need_promote and torch.is_tensor(loss):
        loss = loss.to(orig_dtype)
    return loss


def local_sparsity_l1(
    y_hat_k: torch.Tensor, target: torch.Tensor, reduce: str = "mean"
) -> torch.Tensor:
    """L1 loss on the residual of the local agent.

    Penalises the absolute error of the local-specialist against the target;
    combined with the trend/TV loss on the trend agent this encourages the
    local agent to absorb the sparse transient deviations left by the
    trend+cycle decomposition.
    """
    v = (y_hat_k - target).abs()
    if reduce == "mean":
        return v.mean()
    if reduce == "sum":
        return v.sum()
    return v


def build_agent_heterogeneous_losses(
    specialist_mu: Mapping[str, torch.Tensor] | Iterable[Tuple[str, torch.Tensor]],
    y: torch.Tensor,
    tv_weight: float = 0.1,
    seasonal_weight: float = 0.1,
    sparse_weight: float = 1.0,
    freq_bands: int = 8,
) -> AgentLossesOutput:
    """Compute a structure-aware, per-agent-type heterogeneous loss bundle.

    Parameters
    ----------
    specialist_mu : mapping or iterable of (name, mu)
        Predicted μ_k ∈ R^(B, P, D) for each of the three specialists.
    y : (B, P, D)
        Ground-truth forecast target.
    tv_weight : float
        Weight of the total-variation regulariser applied to the *trend* agent.
    seasonal_weight : float
        Weight of the seasonal Fourier L1 term applied to the *cycle* agent.
    sparse_weight : float
        Weight of the L1 residual term applied to the *local* agent.
    freq_bands : int
        Number of low Fourier frequency bins used in the seasonal term.

    Returns
    -------
    :class:`AgentLossesOutput` — per-term scalars plus a ``total`` scalar for
    convenience in the overall training objective.
    """
    if not isinstance(specialist_mu, Mapping):
        specialist_mu = dict(specialist_mu)

    trend_mu = specialist_mu.get(AGENT_NAMES[0])
    cycle_mu = specialist_mu.get(AGENT_NAMES[1])
    local_mu = specialist_mu.get(AGENT_NAMES[2])

    loss_tv = (
        total_variation_loss(trend_mu) * tv_weight
        if trend_mu is not None
        else torch.zeros((), device=y.device, dtype=y.dtype)
    )
    loss_seasonal = (
        seasonal_fourier_l1_loss(cycle_mu, y, freq_bands=freq_bands) * seasonal_weight
        if cycle_mu is not None
        else torch.zeros((), device=y.device, dtype=y.dtype)
    )
    loss_local = (
        local_sparsity_l1(local_mu, y) * sparse_weight
        if local_mu is not None
        else torch.zeros((), device=y.device, dtype=y.dtype)
    )
    total = loss_tv + loss_seasonal + loss_local
    return AgentLossesOutput(
        tv_trend=loss_tv,
        seasonal_cycle=loss_seasonal,
        sparse_local=loss_local,
        total=total,
    )
