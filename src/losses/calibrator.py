from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F


__all__ = [
    "CalibratorOutput",
    "gaussian_nll",
    "mixture_nll",
    "consensus_regularization",
    "reject_regularization",
    "ece_loss",
    "build_total_loss",
]


@dataclass
class CalibratorOutput:
    nll: torch.Tensor
    mse: torch.Tensor
    consensus: torch.Tensor
    reject: torch.Tensor
    orthogonality: torch.Tensor
    agent_hetero: torch.Tensor
    total: torch.Tensor

    def logging_dict(self, prefix: str = "") -> Dict[str, float]:
        d: Dict[str, float] = {}
        for k, v in (
            ("nll", self.nll),
            ("mse", self.mse),
            ("consensus", self.consensus),
            ("reject", self.reject),
            ("orthogonality", self.orthogonality),
            ("agent_hetero", self.agent_hetero),
            ("total", self.total),
        ):
            if torch.is_tensor(v):
                try:
                    d[f"{prefix}{k}"] = float(v.detach().cpu().item())
                except Exception:
                    pass
        return d


def gaussian_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """Factorised Gaussian negative log-likelihood (per-channel, per-step)."""
    sigma = torch.clamp(sigma, min=eps)
    var = sigma.pow(2)
    per = 0.5 * (
        torch.log(2 * torch.pi * var) + (mu - y).pow(2) / var
    )
    if reduction == "mean":
        return per.mean()
    if reduction == "sum":
        return per.sum()
    return per


def mixture_nll(
    mu_stack: torch.Tensor,
    sigma_stack: torch.Tensor,
    alpha: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mixture-of-Gaussians NLL with K components (one per agent).

    mu_stack    : (B, K, P, D)
    sigma_stack : (B, K, P, D)
    alpha       : (B, K)      mixture weights (softmax-normalised)
    y           : (B, P, D)
    """
    B, K, P, D = mu_stack.shape
    sigma = torch.clamp(sigma_stack, min=eps)
    y4 = y.unsqueeze(1).expand(B, K, P, D)
    var = sigma.pow(2)
    log_pdf_comp = (
        -0.5 * torch.log(2 * torch.pi * var) - 0.5 * (mu_stack - y4).pow(2) / var
    )
    log_weights = alpha.log().view(B, K, 1, 1)
    log_mix = torch.logsumexp(log_weights + log_pdf_comp, dim=1)
    if reduction == "mean":
        return -log_mix.mean()
    if reduction == "sum":
        return -log_mix.sum()
    return -log_mix


def consensus_regularization(
    delta_mu: torch.Tensor,
    delta_sigma: torch.Tensor,
    comm_weights: Optional[torch.Tensor] = None,
    weight: float = 0.01,
    entropy_bonus: float = 0.01,
) -> torch.Tensor:
    """Penalises large consensus updates + encourages attention diversity.

    ||Δμ||² + ||Δσ||² − entropy_bonus · H(attn_weights)
    """
    loss = delta_mu.pow(2).mean() + delta_sigma.pow(2).mean()
    if comm_weights is not None:
        h = -(comm_weights * (comm_weights + 1e-8).log()).sum(dim=-1).mean()
        loss = loss - entropy_bonus * h
    return weight * loss


def reject_regularization(
    reject: torch.Tensor,
    weight: float = 0.01,
    target_usage: float = 0.9,
) -> torch.Tensor:
    """Penalise excessive rejection while discouraging r_k ≡ 0 (BCE push-pull)."""
    usage = 1.0 - reject.mean()
    loss = weight * F.binary_cross_entropy(
        reject.clamp(1e-5, 1 - 1e-5),
        torch.full_like(reject, 1.0 - target_usage),
    ) + 0.1 * weight * F.relu(0.5 - usage).pow(2)
    return loss


def ece_loss(
    sigma: torch.Tensor,
    mu: torch.Tensor,
    y: torch.Tensor,
    n_bins: int = 10,
    q_target: float = 0.95,
    weight: float = 0.0,
) -> torch.Tensor:
    """Approximate expected calibration error for Gaussian predictive intervals.

    Only contributes if ``weight > 0``; added here for reference / ablation.
    """
    if weight <= 0:
        return torch.zeros((), device=mu.device, dtype=mu.dtype)
    z = torch.special.erfinv(
        torch.tensor(float(q_target), device=mu.device, dtype=torch.float64)
    ).to(mu.dtype) * torch.sqrt(torch.tensor(2.0, device=mu.device, dtype=mu.dtype))
    lo = mu - z * sigma
    hi = mu + z * sigma
    covered = ((y >= lo) & (y <= hi)).float().mean()
    return weight * (covered - q_target).abs()


def build_total_loss(
    output,  # TACFOutput or MOAForwardOutput-like
    y: torch.Tensor,
    lambda_nll: float = 1.0,
    lambda_mse: float = 1.0,
    lambda_consensus: float = 0.1,
    lambda_reject: float = 0.01,
    lambda_orthogonality: float = 0.01,
    lambda_agent: float = 0.05,
    use_mixture_nll: bool = False,
    agent_hetero: Optional[torch.Tensor] = None,
) -> CalibratorOutput:
    """Build the full scalar training objective for a TACF forward pass.

    Total = λ_NLL·NLL + λ_MSE·MSE + λ_C·ConsensusReg + λ_R·RejectReg
          + λ_O·Orthogonality + λ_A·(agent heterogeneous structure loss)

    Parameters
    ----------
    output : TACFOutput (or any MOAForwardOutput-alike dataclass)
    y      : (B, P, D) ground-truth forecast target
    lambda_* : per-component scalar weights
    use_mixture_nll : bool
        If ``True`` compute mixture-NLL over the 3 agent components using the
        aggregator's ``alpha`` weights instead of the default single-Gaussian
        NLL on the aggregated μ/σ.
    agent_hetero : optional scalar tensor
        Pre-computed output of :func:`build_agent_heterogeneous_losses.total`.
        If supplied the value is multiplied by ``lambda_agent`` and added to
        the total loss.
    """
    if use_mixture_nll:
        try:
            nll = mixture_nll(
                output.consensus_out.mu,
                output.consensus_out.sigma,
                output.alpha,
                y,
            )
        except Exception:
            nll = gaussian_nll(output.y_hat, output.sigma, y)
    else:
        nll = gaussian_nll(output.y_hat, output.sigma, y)

    mse = F.mse_loss(output.y_hat, y)

    consensus = consensus_regularization(
        delta_mu=output.consensus_out.delta_mu,
        delta_sigma=output.consensus_out.delta_sigma,
        comm_weights=getattr(output.consensus_out, "comm_weights", None),
        weight=1.0,
    )
    reject = reject_regularization(output.reject, weight=1.0)

    ortho = output.aux_losses.get(
        "orthogonality",
        torch.zeros((), device=y.device, dtype=y.dtype),
    )
    if not torch.is_tensor(ortho) or ortho.numel() != 1:
        ortho = torch.zeros((), device=y.device, dtype=y.dtype)

    if agent_hetero is None:
        agent_term = torch.zeros((), device=y.device, dtype=y.dtype)
    else:
        agent_term = agent_hetero

    total = (
        lambda_nll * nll
        + lambda_mse * mse
        + lambda_consensus * consensus
        + lambda_reject * reject
        + lambda_orthogonality * ortho
        + lambda_agent * agent_term
    )
    return CalibratorOutput(
        nll=nll,
        mse=mse,
        consensus=consensus,
        reject=reject,
        orthogonality=ortho,
        agent_hetero=agent_term,
        total=total,
    )
