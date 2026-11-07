from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Optional

from .trainer import Trainer, TrainerConfig, build_mse_loss, freeze_module, unfreeze_module
import torch
from ..utils.logger import ExperimentLogger


__all__ = ["run_stage3_aggregator"]


def _stage3_loss_builder(cfg: TrainerConfig) -> Callable[[Any, torch.Tensor], Any]:
    return build_mse_loss


def run_stage3_aggregator(
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
    """Stage 3: Train aggregator only.

    We keep decomposer / specialists / consensus frozen and teach the
    LightMamba aggregator to emit sensible mixture weights α and rejection
    signals r using MSE on the final prediction only.
    """
    if cfg is None:
        cfg = TrainerConfig(**kwargs)
    else:
        cfg = TrainerConfig(**{**asdict(cfg), **kwargs})
    cfg.lr = float(cfg.lr) * float(lr_mult)
    cfg.min_lr_factor = float(cfg.min_lr_factor)
    freeze_module(model.decomposer)
    freeze_module(model.specialists)
    freeze_module(model.consensus)
    unfreeze_module(model.aggregator)
    if hasattr(model, "sigma_global_multiplier") and isinstance(
        getattr(model, "sigma_global_multiplier"), torch.nn.Parameter
    ):
        model.sigma_global_multiplier.requires_grad_(True)
    trainer = Trainer(
        model=model,
        cfg=cfg,
        logger=logger,
        build_loss_fn=_stage3_loss_builder(cfg),
        inverse_transform_fn=getattr(dm, "inverse_transform", None),
        trainable_names=("aggregator.",),
    )
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        tag="stage3_aggregator",
    )
    result["stage"] = "stage3_aggregator"
    result["trainer_config"] = asdict(cfg)
    result["lr_mult_used"] = float(lr_mult)
    return result
