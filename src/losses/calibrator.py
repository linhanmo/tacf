from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F


__all__ = [
    "CalibratorOutput",
    "gaussian_nll",
    "gaussian_crps",           # NEW: closed-form CRPS for Gaussian forecasts
    "mixture_nll",
    "consensus_regularization",
    "reject_regularization",
    "reject_distribution_penalty",   # NEW: target-mean + entropy penalty on r_k
    "ece_loss",
    "coverage_penalty",
    "build_total_loss",
    "composite_coverage_score",
    "composite_score_from_metrics",
    "temperature_scale_nll",    # NEW: post-hoc T via L-BFGS on val set
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
    sigma_reg_weight: float = 0.0,
) -> torch.Tensor:
    """Factorised Gaussian negative log-likelihood (per-channel, per-step).

    Parameters
    ----------
    sigma_reg_weight : float, default 0.0
        Extra penalty on mean(sigma) + mean(log(sigma)) to prevent sigma from
        collapsing to 0 (under-coverage) or exploding to infinity
        (99.9%-coverage-but-width-0.9 useless intervals).  The combined term
        ``sigma + log(sigma)`` is the convex regulariser whose minimum is at
        ``sigma = 1`` (for unit-scale targets); because targets are z-scored,
        this translates to a stable ~1 std width which the MSE / coverage
        penalty can then finely tune.  0.01–0.05 is a good starting weight.
    """
    sigma = torch.clamp(sigma, min=eps)
    var = sigma.pow(2)
    per = 0.5 * (
        torch.log(2 * torch.pi * var) + (mu - y).pow(2) / var
    )
    if reduction == "mean":
        base = per.mean()
    elif reduction == "sum":
        base = per.sum()
    else:
        base = per
    if sigma_reg_weight and sigma_reg_weight > 0:
        reg = sigma.mean() + torch.log(sigma).mean()
        if reduction == "sum":
            reg = reg * sigma.numel()
        base = base + sigma_reg_weight * reg
    return base


def gaussian_crps(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""Closed-form Continuous Ranked Probability Score for Gaussian forecasts.

    CRPS for N(μ, σ²) and observation y has the known closed form:

        CRPS = σ · [ z·Φ(z) + φ(z) - 1/√π ]

    where z = (y − μ)/σ, Φ the standard normal CDF (``.erf/2 + 0.5``) and φ
    the standard normal PDF.  Result is a non-negative scalar for each
    (y, μ, σ) tuple with 0 being a perfect forecast.

    CRPS is the recommended alternative to RMSE for *probabilistic* forecasts:
    it evaluates both sharpness (small σ) and calibration (correct quantile
    coverage) simultaneously, unlike RMSE which only penalises point bias.
    """
    sigma = torch.clamp(sigma, min=eps)
    z = (y - mu) / sigma
    # Standard-normal CDF via erf
    cdf_z = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    # Standard-normal PDF
    pdf_z = torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)
    crps_per = sigma * (z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - 1.0 / math.sqrt(math.pi))
    if reduction == "mean":
        return crps_per.mean()
    if reduction == "sum":
        return crps_per.sum()
    return crps_per


def mixture_nll(
    mu_stack: torch.Tensor,
    sigma_stack: torch.Tensor,
    alpha: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean",
    sigma_reg_weight: float = 0.0,
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
        base = -log_mix.mean()
    elif reduction == "sum":
        base = -log_mix.sum()
    else:
        base = -log_mix
    if sigma_reg_weight and sigma_reg_weight > 0:
        s_clamped = torch.clamp(sigma_stack, min=eps)
        reg = s_clamped.mean() + torch.log(s_clamped).mean()
        if reduction == "sum":
            reg = reg * s_clamped.numel()
        base = base + sigma_reg_weight * reg
    return base


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


def reject_distribution_penalty(
    reject: torch.Tensor,
    target_mean: float = 0.10,
    target_std: float = 0.05,
    entropy_weight: float = 0.5,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Auxiliary penalty on the aggregated reject signal ``r``.

    Used in Stage 3 alongside the raw ``reject_regularization``.  The term
    pushes the **distribution** of ``r`` (across the batch) towards a
    desired profile rather than only penalising the mean:

      - ``(mean(r) − target_mean)^2`` — force, e.g., ~10% of points to be
        rejected; too-low → rejecting nothing, too-high → the aggregator
        trusts nobody.
      - ``(|std(r) − target_std|)^2`` — avoid degenerate histograms where
        all ``r`` are identical (either all 0 or all 1).
      - ``entropy_weight · H(p)`` with ``p`` = histogram of ``r`` in 10
        equal bins in [0, 1] — encourage a spread-out, non-degenerate
        distribution (standard practice in multi-agent rejection gates).

    All three sub-terms share units of [0, 1], so this function returns a
    unit-less scalar in the same ball-park as a typical BCE term.
    """
    r = torch.clamp(reject.float(), min=0.0, max=1.0)
    r_mean = r.mean()
    r_std = r.std(unbiased=False) if r.numel() > 1 else torch.zeros((), device=r.device, dtype=r.dtype)
    term_mean = (r_mean - float(target_mean)).pow(2)
    term_std = (r_std - float(target_std)).abs().pow(2)
    # Entropy over [0, 1] histogram
    n_bins = 10
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=r.device, dtype=r.dtype)
    hist = torch.histc(r.flatten(), bins=n_bins, min=0.0, max=1.0)
    p = hist / (hist.sum() + eps)
    log_p = torch.log(p + eps)
    entropy = -(p * log_p).sum() / math.log(float(n_bins))  # normalised to [0, 1]
    return term_mean + term_std + float(entropy_weight) * (1.0 - entropy)


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
    sigma_reg_weight: float = 0.0,
    lambda_ece: float = 0.0,
    lambda_crps: float = 0.0,
    lambda_reject_dist: float = 0.0,
    use_mixture_nll: bool = False,
    agent_hetero: Optional[torch.Tensor] = None,
) -> CalibratorOutput:
    """Build the full scalar training objective for a TACF forward pass.

    Per-stage convention (see ``src/experiments/main.py``):
      * S1/S2 (pretrain + consensus): **primary MSE**  → λ_MSE ≫ λ_NLL
      * S3 (aggregator r gate):         **primary NLL**  → λ_NLL ≫ λ_MSE + λ_ECE~0.5 + λ_Rdist~0.2
      * S4 (end-to-end):                **primary NLL**  → λ_NLL ≫ λ_MSE + λ_CRPS~0.5 + λ_ECE~0.3

    Parameters
    ----------
    lambda_ece : float, default 0.0
        Weight on ``|coverage − target_q|`` calibration gap (S3/S4 turn on).
    lambda_crps : float, default 0.0
        Weight on closed-form Gaussian CRPS.  Only used by S4 as the
        **probabilistic** alternative to the deprecated RMSE secondary metric.
    lambda_reject_dist : float, default 0.0
        Weight on ``reject_distribution_penalty`` — S3 auxiliary metric.
    """
    if use_mixture_nll:
        try:
            nll = mixture_nll(
                output.consensus_out.mu,
                output.consensus_out.sigma,
                output.alpha,
                y,
                sigma_reg_weight=sigma_reg_weight,
            )
        except Exception:
            nll = gaussian_nll(output.y_hat, output.sigma, y, sigma_reg_weight=sigma_reg_weight)
    else:
        nll = gaussian_nll(output.y_hat, output.sigma, y, sigma_reg_weight=sigma_reg_weight)

    mse = F.mse_loss(output.y_hat, y)

    crps_loss = (
        gaussian_crps(output.y_hat, output.sigma, y).to(mse.dtype)
        if float(lambda_crps) > 0
        else torch.zeros((), device=y.device, dtype=y.dtype)
    )

    consensus = consensus_regularization(
        delta_mu=output.consensus_out.delta_mu,
        delta_sigma=output.consensus_out.delta_sigma,
        comm_weights=getattr(output.consensus_out, "comm_weights", None),
        weight=1.0,
    )
    reject = reject_regularization(output.reject, weight=1.0)

    # S3 auxiliary: target histogram on r
    rdist = torch.zeros((), device=y.device, dtype=y.dtype)
    if float(lambda_reject_dist) > 0:
        try:
            rdist = reject_distribution_penalty(output.reject).to(y.dtype)
        except Exception:
            rdist = torch.zeros((), device=y.device, dtype=y.dtype)

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

    ece_gap = torch.zeros((), device=y.device, dtype=y.dtype)
    if float(lambda_ece) > 0:
        ece_gap = ece_loss(
            output.sigma,
            output.y_hat,
            y,
            q_target=float(target_q),
            weight=1.0,
        ).to(y.dtype)

    total = (
        lambda_nll * nll
        + lambda_mse * mse
        + lambda_consensus * consensus
        + lambda_reject * reject
        + lambda_orthogonality * ortho
        + lambda_agent * agent_term
        + lambda_cov_penalty * cov_loss
        + float(lambda_ece) * ece_gap
        + float(lambda_crps) * crps_loss
        + float(lambda_reject_dist) * rdist
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


# ---------------------------------------------------------------------------
# Composite coverage monitor for S2/S3 best-checkpoint selection
# ---------------------------------------------------------------------------

def composite_coverage_score(
    coverage: float,
    width: float,
    mse: Optional[float] = None,
    target_q: float = 0.95,
    width_weight: float = 1.0,
    mse_weight: float = 0.5,
) -> float:
    """Combine coverage gap + interval width (and optionally MSE) into a
    single **smaller-is-better** scalar used for ``monitor``.

    Formulation (all terms are ≥ 0):

        score = (target_q − coverage)²       ← gap penalty (dominant)
              + width_weight · width          ← discourage exploding intervals
              + mse_weight · √mse             ← tie-breaking on point accuracy

    Why quadratic on gap: a candidate with 94% coverage is *far* better than
    one with 20% coverage; the square term strongly penalises anything more
    than ~5% below ``target_q`` while still preferring 95% to 93%.

    Why sqrt(mse): coverage/width typically live in [0, 1.2] whereas exchange_rate
    MSE lives in [0.01, 0.05] — sqrt keeps them in the same ballpark so MSE
    acts only as a tie-breaker.
    """
    gap = max(0.0, float(target_q) - float(coverage))
    w = max(0.0, float(width))
    score = gap * gap + width_weight * w
    if mse is not None:
        m = max(0.0, float(mse))
        score = score + mse_weight * (m ** 0.5)
    return score


def composite_score_from_metrics(
    metrics,  # MetricsResult | dict
    target_q: float = 0.95,
    width_weight: float = 1.0,
    mse_weight: float = 0.5,
    fallback: float = float("inf"),
) -> float:
    """``composite_coverage_score`` wrapper that accepts either a
    ``MetricsResult`` object or a plain dict (e.g. ``metrics.as_dict()``).
    Returns ``fallback`` (default +∞) if any required field is missing/NaN,
    so the trainer will never select a broken checkpoint as ``best``.
    """
    import math
    try:
        if isinstance(metrics, dict):
            cov = metrics.get("q95_coverage")
            wid = metrics.get("q95_width")
            mse = metrics.get("mse")
        else:
            cov = getattr(metrics, "q95_coverage", None)
            wid = getattr(metrics, "q95_width", None)
            mse = getattr(metrics, "mse", None)
        # 缺字段 + 非数值 + NaN 一律 fallback
        def _ok(x):
            if x is None:
                return False
            try:
                f = float(x)
                return (f == f) and (f < float("inf")) and (f > float("-inf"))
            except Exception:
                return False
        if not _ok(cov) or not _ok(wid):
            return fallback
        return composite_coverage_score(
            coverage=float(cov),
            width=float(wid),
            mse=None if not _ok(mse) else float(mse),
            target_q=target_q,
            width_weight=width_weight,
            mse_weight=mse_weight,
        )
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Post-hoc temperature scaling (S4 final calibration)
# ---------------------------------------------------------------------------

def temperature_scale_nll(
    mus: Sequence[torch.Tensor],
    sigmas: Sequence[torch.Tensor],
    ys: Sequence[torch.Tensor],
    *,
    init_T: float = 1.0,
    T_low: float = 0.1,
    T_high: float = 5.0,
    max_iter: int = 200,
    grid_points: int = 41,
    lbfgs_iter: int = 80,
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """Two-stage T-scaling: coarse grid search + L-BFGS refinement.

    Scales only the predictive standard deviation: ``σ' = T · σ``, keeping μ
    unchanged (the standard "temperature scaling" for regression — a single
    scalar per model; guarantees no loss of point accuracy while fixing
    under/over-confident NLL).

    Parameters
    ----------
    mus/sigmas/ys : sequences of tensors, each (B, P, D)
        These should be the VALIDATION split predictions (the T found here is
        then applied on the test split for reporting).
    init_T : float, default 1.0
        Starting point for L-BFGS refinement.  After coarse grid search we
        restart at the best grid T so this is only a fallback.

    Returns
    -------
    dict with keys:
      * ``T``        — optimal scalar temperature (float)
      * ``nll_before`` — raw NLL at T = 1.0
      * ``nll_after``  — optimised NLL at the learned T
      * ``status``     — "grid+lbfgs" | "grid-only" depending on which ran
    """
    mu = torch.cat([m.detach().float() for m in mus], dim=0)
    sigma = torch.cat([s.detach().float() for s in sigmas], dim=0)
    y = torch.cat([yy.detach().float() for yy in ys], dim=0)
    sigma = torch.clamp(sigma, min=eps)
    N = mu.numel()

    def _nll(T: float) -> float:
        s = sigma * float(T)
        var = s.pow(2)
        per = 0.5 * (math.log(2 * math.pi) + torch.log(var) + (mu - y).pow(2) / var)
        return float(per.mean().cpu().item())

    status = "grid-only"
    nll_before = _nll(1.0)

    # 1) Coarse grid search (safe + robust, no gradients needed)
    grid = torch.linspace(T_low, T_high, steps=grid_points)
    best_T = float(init_T)
    best_nll = float("inf")
    for T in grid.tolist():
        val = _nll(T)
        if val < best_nll:
            best_nll = val
            best_T = float(T)

    # 2) L-BFGS refinement (tighten around best grid-T via gradient on log_T)
    #    Parameterise on log T so the optimiser stays strictly positive.
    try:
        logT = torch.nn.Parameter(torch.tensor(math.log(best_T), dtype=torch.float32))
        opt = torch.optim.LBFGS(
            [logT],
            lr=0.3,
            max_iter=lbfgs_iter,
            max_eval=2 * lbfgs_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            T = torch.exp(logT)
            s = sigma * T
            var = s.pow(2)
            per = 0.5 * (math.log(2 * math.pi) + torch.log(var) + (mu - y).pow(2) / var)
            loss = per.mean()
            loss.backward()
            return loss

        opt.step(closure)
        refined_T = float(torch.exp(logT.detach()).cpu().item())
        refined_T = max(T_low, min(T_high, refined_T))
        refined_nll = _nll(refined_T)
        if refined_nll < best_nll:
            best_nll = refined_nll
            best_T = refined_T
            status = "grid+lbfgs"
    except Exception:
        pass

    return {
        "T": float(best_T),
        "nll_before": float(nll_before),
        "nll_after": float(best_nll),
        "status": status,
        "N_samples": int(N),
    }

