from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

from .trainer import Trainer, TrainerConfig, freeze_module
from ..losses.agent_losses import build_agent_heterogeneous_losses
from ..losses.calibrator import gaussian_nll
from ..utils.logger import ExperimentLogger


__all__ = ["run_stage1_pretrain"]


def _stage1_loss_builder(model, cfg: TrainerConfig) -> Callable[[Any, torch.Tensor], Any]:
    from dataclasses import dataclass

    @dataclass
    class _Stage1Loss:
        specialist_nll: torch.Tensor
        specialist_mse: torch.Tensor
        agent_hetero: torch.Tensor
        orthogonality: torch.Tensor
        total: torch.Tensor

        def logging_dict(self, prefix: str = "") -> Dict[str, float]:
            return {
                f"{prefix}specialist_nll": float(self.specialist_nll.detach().cpu().item()),
                f"{prefix}specialist_mse": float(self.specialist_mse.detach().cpu().item()),
                f"{prefix}agent_hetero": float(self.agent_hetero.detach().cpu().item()),
                f"{prefix}orthogonality": float(self.orthogonality.detach().cpu().item()),
                f"{prefix}total": float(self.total.detach().cpu().item()),
            }

    def _loss_fn(out: Any, y: torch.Tensor) -> _Stage1Loss:
        nll_sum = 0.0
        mse_sum = 0.0
        mu_dict: Dict[str, torch.Tensor] = {}
        for name, spec in out.specialists_out.outputs.items():
            mu = spec.mu
            sigma = spec.sigma
            nll_sum = nll_sum + gaussian_nll(mu, sigma, y)
            mse_sum = mse_sum + F.mse_loss(mu, y)
            mu_dict[name] = mu
        nll = nll_sum / max(1, len(out.specialists_out.outputs))
        mse = mse_sum / max(1, len(out.specialists_out.outputs))
        hetero = build_agent_heterogeneous_losses(mu_dict, y)
        ortho = out.aux_losses.get(
            "orthogonality",
            torch.zeros((), device=y.device, dtype=y.dtype),
        )
        total = (
            cfg.lambda_nll * nll
            + cfg.lambda_mse * mse
            + cfg.lambda_agent * hetero.total
            + cfg.lambda_orthogonality * ortho
        )
        return _Stage1Loss(
            specialist_nll=nll,
            specialist_mse=mse,
            agent_hetero=hetero.total,
            orthogonality=ortho,
            total=total,
        )

    return _loss_fn


def run_stage1_pretrain(
    model,
    train_loader,
    val_loader=None,
    test_loader=None,
    cfg: Optional[TrainerConfig] = None,
    logger: Optional[ExperimentLogger] = None,
    dm=None,
    **kwargs,
):
    """Stage 1: Pretrain decomposer + 3 independent specialists.

    Consensus and aggregator are completely frozen.  Each agent is supervised
    independently with Gaussian NLL + MSE, reinforced with the structural
    heterogeneous losses (TV on trend, Fourier-seasonal on cycle, L1-residual
    on local) plus the decomposer orthogonality penalty.
    """
    if cfg is None:
        cfg = TrainerConfig(**kwargs)
    freeze_module(model.consensus)
    freeze_module(model.aggregator)
    trainer = Trainer(
        model=model,
        cfg=cfg,
        logger=logger,
        build_loss_fn=_stage1_loss_builder(model, cfg),
        inverse_transform_fn=getattr(dm, "inverse_transform", None),
        trainable_names=("decomposer.", "specialists."),
    )
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        tag="stage1_pretrain",
    )
    result["stage"] = "stage1_pretrain"
    result["trainer_config"] = asdict(cfg)
    return result
