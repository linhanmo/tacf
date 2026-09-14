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
    coverage_penalty: torch.Tensor
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
            ("coverage_penalty", self.coverage_penalty),
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
    """Penalise excessive rejection while discouraging r_k ≡ 0 (BCE push-pull).

    Notes
    -----
    ``torch.nn.functional.binary_cross_entropy`` on post-sigmoid probabilities
    is explicitly **not safe** under ``torch.cuda.amp.autocast`` (PyTorch will
    raise a ``RuntimeError`` suggesting a move to the *logits* variant).

    Because call sites currently only hold the post-sigmoid ``reject`` tensor
    (``AggregatorAgent.forward`` applies ``sigmoid`` before returning and does
    **not** surface the raw logits), we invert the probabilities back to
    logits via ``torch.logit`` inside a small float32 promotion window, then
    compute ``F.binary_cross_entropy_with_logits`` — which is fully
    AMP/GradScaler safe and numerically more stable than applying BCE directly
    to clamped probabilities.  Call sites and function signatures remain
    unchanged.
    """
    usage = 1.0 - reject.mean()
    target_p = 1.0 - target_usage
    orig_dtype = reject.dtype
    need_promote = orig_dtype != torch.float32 and orig_dtype != torch.float64
    r = reject.float() if need_promote else reject
    eps = 1e-5
    r_clamped = torch.clamp(r, min=eps, max=1.0 - eps)
    logits = torch.logit(r_clamped, eps=eps)
    target = torch.full_like(logits, float(target_p))
    bce = F.binary_cross_entropy_with_logits(logits, target)
    if need_promote:
        bce = bce.to(orig_dtype)
        usage = usage.to(orig_dtype)
    loss = weight * bce + 0.1 * weight * F.relu(0.5 - usage).pow(2)
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


def coverage_penalty(
    sigma: torch.Tensor,
    mu: torch.Tensor,
    y: torch.Tensor,
    target_q: float = 0.95,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-batch differentiable coverage Hinge loss.

    For target_q=0.95 we want the symmetric gaussian interval length
    ``± z_q·σ`` to contain ``y`` at least 95% of the time. Since
    ``coverage`` is discrete and non-differentiable, we use the
    *sample-wise surrogate*:

        width_surrogate = 2·z_q·σ   (the total interval width we pay for)
        margin = |y − μ|
        hinge = ReLU( margin − z_q·σ )²   (push σ up whenever it under-covers)

    and add a *global batch-level coverage gap* term:

        gap = ReLU( target_q − batch_coverage_est )²

    where ``batch_coverage_est`` is approximated via a straight-through
    estimator using Gaussian CDF probability of coverage per sample
    ``Φ((y−μ)/σ) − Φ((μ−y)/σ)``, which is differentiable and equals the
    true coverage in expectation for correctly-specified Gaussians.

    Both terms are averaged over (B,P,D) so the magnitude is comparable to
    MSE in the 0.01–1.0 range (usually needs λ_cov ∈ [0.05, 0.5]).
    """
    z_q = torch.special.erfinv(
        torch.tensor(float(target_q), device=mu.device, dtype=torch.float64)
    ).to(mu.dtype) * torch.sqrt(torch.tensor(2.0, device=mu.device, dtype=mu.dtype))
    sigma = torch.clamp(sigma, min=eps)
    margin = (y - mu).abs()
    # sample-wise under-cover hinge (σ too small ⇒ margin > z·σ ⇒ penalty)
    hinge_sample = F.relu(margin - z_q * sigma).pow(2)

    # Expected coverage probability (differentiable):
    #   p_cover = Φ((margin)/σ) − Φ(−(margin)/σ) = 2Φ(margin/σ) − 1
    # which is equivalent to erf( margin / (√2 σ) )
    p_cover = torch.erf(margin / (torch.sqrt(torch.tensor(2.0, device=mu.device, dtype=mu.dtype)) * sigma))
    # batch-level gap: target_q minus the mean expected coverage (only penalise under-coverage)
    gap = target_q - p_cover.mean()
    hinge_batch = F.relu(gap).pow(2)

    return hinge_sample.mean() + 2.0 * hinge_batch


def build_total_loss(
    output,  # TACFOutput or MOAForwardOutput-like
    y: torch.Tensor,
    lambda_nll: float = 2.0,
    lambda_mse: float = 1.0,
    lambda_consensus: float = 0.1,
    lambda_reject: float = 0.01,
    lambda_orthogonality: float = 0.01,
    lambda_agent: float = 0.05,
    lambda_cov_penalty: float = 0.25,
    target_q: float = 0.95,
    use_mixture_nll: bool = False,
    agent_hetero: Optional[torch.Tensor] = None,
) -> CalibratorOutput:
    """Build the full scalar training objective for a TACF forward pass.

    Total = λ_NLL·NLL + λ_MSE·MSE + λ_C·ConsensusReg + λ_R·RejectReg
          + λ_O·Orthogonality + λ_A·(agent heterogeneous structure loss)
          + λ_cov·CoveragePenalty   (NEW per smoke-run calibration fix)

    **Post smoke-run defaults (updated):** λ_NLL raised from 1.0→2.0, new
    λ_cov_penalty=0.25, target_q=0.95.  They force σ to be honest rather
    than systematically under-estimated (8/8 datasets under-covered on the
    initial smoke runs: worst-case exchange_rate Q95 coverage only 12.2%).

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

    cov_loss = coverage_penalty(output.sigma, output.y_hat, y, target_q=target_q)

    total = (
        lambda_nll * nll
        + lambda_mse * mse
        + lambda_consensus * consensus
        + lambda_reject * reject
        + lambda_orthogonality * ortho
        + lambda_agent * agent_term
        + lambda_cov_penalty * cov_loss
    )
    return CalibratorOutput(
        nll=nll,
        mse=mse,
        consensus=consensus,
        reject=reject,
        orthogonality=ortho,
        agent_hetero=agent_term,
        coverage_penalty=cov_loss,
        total=total,
    )
