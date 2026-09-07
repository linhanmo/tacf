from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


__all__ = [
    "MetricsResult",
    "mse",
    "mae",
    "rmse",
    "mape",
    "correlation",
    "relative_squared_error",
    "quantile_coverage_probability",
    "gaussian_nll_np",
    "compute_metrics",
]


@dataclass
class MetricsResult:
    mse: float
    mae: float
    rmse: float
    mape: float
    corr: float
    rse: float
    q95_coverage: float
    q95_width: float
    nll: float

    def format(self, short: bool = False) -> str:
        if short:
            return (
                f"MSE={self.mse:.4f}  MAE={self.mae:.4f}  RMSE={self.rmse:.4f}  "
                f"MAPE={self.mape:.2f}%  CORR={self.corr:.4f}  RSE={self.rse:.4f}  "
                f"Q95Cov={self.q95_coverage:.3f}  NLL={self.nll:.4f}"
            )
        return (
            f"MSE={self.mse:.6f}  MAE={self.mae:.6f}  RMSE={self.rmse:.6f}  "
            f"MAPE={self.mape:.4f}%  CORR={self.corr:.4f}  RSE={self.rse:.4f}  "
            f"Q95Cov={self.q95_coverage:.3f}  Q95W={self.q95_width:.4f}  NLL={self.nll:.4f}"
        )

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@torch.no_grad()
def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).mean()


@torch.no_grad()
def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


@torch.no_grad()
def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).mean().sqrt()


@torch.no_grad()
def mape(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    t = target.abs()
    mask = t > eps
    if mask.sum() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    err = ((pred[mask] - target[mask]).abs() / t[mask]).mean()
    return err * 100.0


@torch.no_grad()
def correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = pred.flatten()
    t = target.flatten()
    pm = p - p.mean()
    tm = t - t.mean()
    num = (pm * tm).sum()
    den = (pm.pow(2).sum().sqrt() * tm.pow(2).sum().sqrt()) + 1e-8
    return num / den


@torch.no_grad()
def relative_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = (pred - target).pow(2).sum()
    den = (target - target.mean()).pow(2).sum() + 1e-8
    return (num / den).sqrt()


@torch.no_grad()
def quantile_coverage_probability(
    pred: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    q: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (coverage probability at q, average interval width)."""
    z = torch.special.erfinv(
        torch.tensor(float(q), device=pred.device, dtype=torch.float64)
    )
    z = z.to(pred.dtype) * torch.sqrt(
        torch.tensor(2.0, device=pred.device, dtype=pred.dtype)
    )
    lo = pred - z * sigma
    hi = pred + z * sigma
    covered = ((target >= lo) & (target <= hi)).float().mean()
    width = (hi - lo).mean()
    return covered, width


def _gaussian_nll_np(
    mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, eps: float = 1e-6
) -> float:
    var = np.clip(np.asarray(sigma).astype(np.float64), a_min=eps, a_max=None) ** 2
    diff = np.asarray(mu).astype(np.float64) - np.asarray(y).astype(np.float64)
    per = 0.5 * (np.log(2 * math.pi * var) + diff**2 / var)
    return float(per.mean())


@torch.no_grad()
def compute_metrics(
    preds_list: Sequence[torch.Tensor],
    targets_list: Sequence[torch.Tensor],
    sigmas_list: Optional[Sequence[torch.Tensor]] = None,
    q: float = 0.95,
    eps_nll: float = 1e-6,
) -> MetricsResult:
    """Compute aggregate metrics across a list of mini-batch outputs.

    Parameters
    ----------
    preds_list : sequence of tensors of shape ``(B, P, D)``
    targets_list : sequence of tensors matching ``preds_list``
    sigmas_list : optional sequence of tensors matching ``preds_list`` — if
        omitted NLL and coverage are reported as NaN / 0 respectively.
    q : coverage quantile (default 0.95)
    eps_nll : small floor for sigma before computing NLL
    """
    pred = torch.cat([p.detach() for p in preds_list], dim=0)
    target = torch.cat([t.detach() for t in targets_list], dim=0)
    if sigmas_list is None or len(sigmas_list) == 0:
        sigma = torch.ones_like(pred)
        has_sigma = False
    else:
        sigma = torch.cat([s.detach() for s in sigmas_list], dim=0)
        sigma = torch.clamp(sigma, min=eps_nll)
        has_sigma = True

    mse_v = float(mse(pred, target).cpu().item())
    mae_v = float(mae(pred, target).cpu().item())
    rmse_v = float(rmse(pred, target).cpu().item())
    mape_v = float(mape(pred, target).cpu().item())
    corr_v = float(correlation(pred, target).cpu().item())
    rse_v = float(relative_squared_error(pred, target).cpu().item())

    if has_sigma:
        q95_c, q95_w = quantile_coverage_probability(pred, sigma, target, q=q)
        q95_cv = float(q95_c.cpu().item())
        q95_wv = float(q95_w.cpu().item())
        nll_v = float(
            _gaussian_nll_np(
                pred.cpu().numpy(), sigma.cpu().numpy(), target.cpu().numpy()
            )
        )
    else:
        q95_cv = float("nan")
        q95_wv = float("nan")
        nll_v = float("nan")

    return MetricsResult(
        mse=mse_v,
        mae=mae_v,
        rmse=rmse_v,
        mape=mape_v,
        corr=corr_v,
        rse=rse_v,
        q95_coverage=q95_cv,
        q95_width=q95_wv,
        nll=nll_v,
    )
