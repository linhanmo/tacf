from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

import torch

from .trainer import Trainer, TrainerConfig, build_mse_loss, unfreeze_module
from ..utils.logger import ExperimentLogger


__all__ = ["run_stage4_finetune"]


def run_stage4_finetune(
    model,
    train_loader,
    val_loader=None,
    test_loader=None,
    cfg: Optional[TrainerConfig] = None,
    logger: Optional[ExperimentLogger] = None,
    dm=None,
    lr_mult: float = 0.03,
    early_stop_override: Optional[int] = 5,
    **kwargs,
):
    """Stage 4: End-to-end MSE-only fine-tuning with a reduced learning rate.

    Analysis on the 8-dataset smoke runs showed S4 regresses on 50% datasets
    (aggregator α/r weights disturbed by full-update noise). We therefore
    (a) drop the default lr multiplier from 0.10 → **0.03**,
    (b) tighten early_stop from TrainerConfig.default (10) → **5**
        (can be switched off by passing early_stop_override=None).
    """
    if cfg is None:
        cfg = TrainerConfig(**kwargs)
    else:
        cfg = TrainerConfig(**{**asdict(cfg), **kwargs})
    cfg.lr = float(cfg.lr) * float(lr_mult)
    cfg.min_lr_factor = float(cfg.min_lr_factor)
    if early_stop_override is not None:
        cfg.early_stop = int(early_stop_override)
    unfreeze_module(model.decomposer)
    unfreeze_module(model.specialists)
    unfreeze_module(model.consensus)
    unfreeze_module(model.aggregator)
    trainer = Trainer(
        model=model,
        cfg=cfg,
        logger=logger,
        build_loss_fn=build_mse_loss,
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
    result["lr_mult_used"] = float(lr_mult)
    return result
