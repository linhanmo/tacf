from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Optional

import torch
from .trainer import Trainer, TrainerConfig, build_mse_loss, freeze_module, unfreeze_module
from ..utils.logger import ExperimentLogger


__all__ = ["run_stage2_comm"]


def _stage2_loss_builder(cfg: TrainerConfig) -> Callable[[Any, torch.Tensor], Any]:
    # Use the final prediction through the frozen aggregator, as in evaluation.
    return build_mse_loss


def run_stage2_comm(
    model,
    train_loader,
    val_loader=None,
    test_loader=None,
    cfg: Optional[TrainerConfig] = None,
    logger: Optional[ExperimentLogger] = None,
    dm=None,
    lr_mult: float = 1.0,
    **kwargs,
):
    """Stage 2: Train consensus layer only.

    Decomposer, specialists and aggregator are all frozen.  We monitor the
    final prediction against the ground truth using MSE only.
    """
    if cfg is None:
        cfg = TrainerConfig(**kwargs)
    else:
        cfg = TrainerConfig(**{**asdict(cfg), **kwargs})
    cfg.lr = float(cfg.lr) * float(lr_mult)
    cfg.min_lr_factor = float(cfg.min_lr_factor)
    freeze_module(model.decomposer)
    freeze_module(model.specialists)
    freeze_module(model.aggregator)
    unfreeze_module(model.consensus)
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
    result["lr_mult_used"] = float(lr_mult)
    return result
