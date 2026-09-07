from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import torch

from .trainer import Trainer, TrainerConfig, unfreeze_module
from ..losses.agent_losses import build_agent_heterogeneous_losses
from ..losses.calibrator import build_total_loss
from ..utils.logger import ExperimentLogger


__all__ = ["run_stage4_finetune"]


def _wrap_with_hetero_loss(trainer_cfg: TrainerConfig) -> Any:
    def _loss_fn(out: Any, y: torch.Tensor) -> Any:
        mu_dict = {name: spec.mu for name, spec in out.specialists_out.outputs.items()}
        hetero = build_agent_heterogeneous_losses(mu_dict, y)
        return build_total_loss(
            out,
            y,
            lambda_nll=trainer_cfg.lambda_nll,
            lambda_mse=trainer_cfg.lambda_mse,
            lambda_consensus=trainer_cfg.lambda_consensus,
            lambda_reject=trainer_cfg.lambda_reject,
            lambda_orthogonality=trainer_cfg.lambda_orthogonality,
            lambda_agent=trainer_cfg.lambda_agent,
            use_mixture_nll=trainer_cfg.use_mixture_nll,
            agent_hetero=hetero.total,
        )

    return _loss_fn


def run_stage4_finetune(
    model,
    train_loader,
    val_loader=None,
    test_loader=None,
    cfg: Optional[TrainerConfig] = None,
    logger: Optional[ExperimentLogger] = None,
    dm=None,
    lr_mult: float = 0.1,
    **kwargs,
):
    """Stage 4: End-to-end fine-tuning with a reduced learning rate.

    Unfreezes every component and optimises the full training objective from
    ``losses.calibrator.build_total_loss`` including the heterogeneous per-
    specialist structure loss.
    """
    if cfg is None:
        cfg = TrainerConfig(**kwargs)
    else:
        cfg = TrainerConfig(**{**asdict(cfg), **kwargs})
    cfg.lr = float(cfg.lr) * float(lr_mult)
    cfg.min_lr_factor = float(cfg.min_lr_factor)
    unfreeze_module(model.decomposer)
    unfreeze_module(model.specialists)
    unfreeze_module(model.consensus)
    unfreeze_module(model.aggregator)
    trainer = Trainer(
        model=model,
        cfg=cfg,
        logger=logger,
        build_loss_fn=_wrap_with_hetero_loss(cfg),
        inverse_transform_fn=getattr(dm, "inverse_transform", None),
    )
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        tag="stage4_finetune",
    )
    result["stage"] = "stage4_finetune"
    result["trainer_config"] = asdict(cfg)
    return result
