from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

from .trainer import MSELoss, Trainer, TrainerConfig, freeze_module
from ..utils.logger import ExperimentLogger


__all__ = ["run_stage1_pretrain"]


def _stage1_loss_builder(model, cfg: TrainerConfig) -> Callable[[Any, torch.Tensor], Any]:
    def _loss_fn(out: Any, y: torch.Tensor) -> MSELoss:
        # Supervise each specialist independently while fusion is frozen.
        losses = [
            F.mse_loss(spec.mu.float(), y.float())
            for spec in out.specialists_out.outputs.values()
        ]
        return MSELoss(torch.stack(losses).mean())

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
    independently with MSE, averaged across all specialists. No probabilistic
    or auxiliary losses participate in training.
    """
    if cfg is None:
        cfg = TrainerConfig(**kwargs)
    freeze_module(model.consensus)
    freeze_module(model.aggregator)
    # FIXED 2026-09-26: S1 σ stability — freeze sigma_global_multiplier.
    if hasattr(model, "sigma_global_multiplier") and isinstance(
        getattr(model, "sigma_global_multiplier"), torch.nn.Parameter
    ):
        model.sigma_global_multiplier.requires_grad_(False)
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
