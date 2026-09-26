from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

from .trainer import Trainer, TrainerConfig, freeze_module, unfreeze_module
from ..losses.calibrator import (
    consensus_regularization,
    gaussian_nll,
)
from ..utils.logger import ExperimentLogger


__all__ = ["run_stage2_comm"]


def _stage2_loss_builder(cfg: TrainerConfig) -> Callable[[Any, torch.Tensor], Any]:
    @dataclass
    class _Stage2Loss:
        consensus_nll: torch.Tensor
        consensus_mse: torch.Tensor
        consensus_reg: torch.Tensor
        total: torch.Tensor

        def logging_dict(self, prefix: str = "") -> Dict[str, float]:
            return {
                f"{prefix}consensus_nll": float(self.consensus_nll.detach().cpu().item()),
                f"{prefix}consensus_mse": float(self.consensus_mse.detach().cpu().item()),
                f"{prefix}consensus_reg": float(self.consensus_reg.detach().cpu().item()),
                f"{prefix}total": float(self.total.detach().cpu().item()),
            }

    def _loss_fn(out: Any, y: torch.Tensor) -> _Stage2Loss:
        # FIXED 2026-09-26: stage2 loss now operates on out.y_hat / out.sigma
        # (the full pipeline through the *frozen* aggregator) rather than on
        # out.consensus_out.mu/sigma.  Previously the loss path and the val/test
        # eval path were fully decoupled: _evaluate() always returns
        # out.y_hat = aggregator(consensus_out).  Because the aggregator is
        # random-init + FROZEN during S2, any improvements to consensus were
        # not visible in val_mse at all → stage2 looked like a total no-op.
        # Aligning loss to y_hat/sigma makes the training objective observable
        # through the same frozen aggregator window that val metrics use.
        y_hat = out.y_hat
        sigma = out.sigma
        sreg = float(getattr(cfg, "sigma_reg_weight", 0.0) or 0.0)
        nll = gaussian_nll(
            y_hat,
            sigma,
            y,
            sigma_reg_weight=sreg,
        ).to(torch.float32)
        mse = F.mse_loss(y_hat.float(), y.float())
        # Consensus regularization still acts on consensus internals to keep
        # deltas bounded and attention diverse, even when the supervised loss
        # is moved to y_hat.
        try:
            creg = consensus_regularization(
                out.consensus_out.delta_mu,
                out.consensus_out.delta_sigma,
                getattr(out.consensus_out, "comm_weights", None),
                weight=1.0,
            )
            if torch.is_tensor(creg):
                creg = creg.to(device=y.device, dtype=torch.float32)
            else:
                creg = torch.as_tensor(creg, device=y.device, dtype=torch.float32)
        except Exception:
            creg = torch.zeros((), device=y.device, dtype=torch.float32)
        l_cons = float(getattr(cfg, "lambda_consensus", 0.1) or 0.1)
        total = (
            float(cfg.lambda_nll) * nll
            + float(cfg.lambda_mse) * mse
            + l_cons * creg
        ).to(dtype=y.dtype)
        return _Stage2Loss(
            consensus_nll=nll,
            consensus_mse=mse,
            consensus_reg=creg,
            total=total,
        )

    return _loss_fn


def run_stage2_comm(
    model,
    train_loader,
    val_loader=None,
    test_loader=None,
    cfg: Optional[TrainerConfig] = None,
    logger: Optional[ExperimentLogger] = None,
    dm=None,
    **kwargs,
):
    """Stage 2: Train consensus layer only.

    Decomposer, specialists and aggregator are all frozen.  We monitor the
    consensus-refined μ/σ against the ground truth using NLL + MSE plus the
    consensus regularisation that keeps communication deltas small and
    attention weights diverse.

    As in S1 we freeze the sigma_global_multiplier scalar so consensus learns
    via the specialist-level (per-agent, per-step) sigma corrections, not a
    single global dial that could mask the quality of consensus learning.
    """
    if cfg is None:
        cfg = TrainerConfig(**kwargs)
    freeze_module(model.decomposer)
    freeze_module(model.specialists)
    freeze_module(model.aggregator)
    unfreeze_module(model.consensus)
    # FIXED 2026-09-26: S2 σ stability — keep sigma_global_multiplier frozen.
    if hasattr(model, "sigma_global_multiplier") and isinstance(
        getattr(model, "sigma_global_multiplier"), torch.nn.Parameter
    ):
        model.sigma_global_multiplier.requires_grad_(False)
    trainer = Trainer(
        model=model,
        cfg=cfg,
        logger=logger,
        build_loss_fn=_stage2_loss_builder(cfg),
        inverse_transform_fn=getattr(dm, "inverse_transform", None),
        trainable_names=("consensus.",),
    )
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        tag="stage2_comm",
    )
    result["stage"] = "stage2_comm"
    result["trainer_config"] = asdict(cfg)
    return result
