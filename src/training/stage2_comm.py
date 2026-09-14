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
        B, K, P, D = out.consensus_out.mu.shape
        mu = out.consensus_out.mu
        sigma = out.consensus_out.sigma
        y4 = y.unsqueeze(1).expand(B, K, P, D)
        nll = gaussian_nll(mu, sigma, y4)
        mse = F.mse_loss(mu, y4)
        creg = consensus_regularization(
            out.consensus_out.delta_mu,
            out.consensus_out.delta_sigma,
            getattr(out.consensus_out, "comm_weights", None),
            weight=1.0,
        )
        total = (
            cfg.lambda_nll * nll
            + cfg.lambda_mse * mse
            + cfg.lambda_consensus * creg
        )
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
    """
    if cfg is None:
        cfg = TrainerConfig(**kwargs)
    freeze_module(model.decomposer)
    freeze_module(model.specialists)
    freeze_module(model.aggregator)
    unfreeze_module(model.consensus)
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
