from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

from .trainer import Trainer, TrainerConfig, freeze_module, unfreeze_module
from ..losses.calibrator import reject_regularization, gaussian_nll
import torch
import torch.nn.functional as F
from ..utils.logger import ExperimentLogger


__all__ = ["run_stage3_aggregator"]


def _stage3_loss_builder(cfg: TrainerConfig) -> Callable[[Any, torch.Tensor], Any]:
    @dataclass
    class _Stage3Loss:
        nll: torch.Tensor
        mse: torch.Tensor
        reject_reg: torch.Tensor
        total: torch.Tensor

        def logging_dict(self, prefix: str = "") -> Dict[str, float]:
            return {
                f"{prefix}nll": float(self.nll.detach().cpu().item()),
                f"{prefix}mse": float(self.mse.detach().cpu().item()),
                f"{prefix}reject_reg": float(self.reject_reg.detach().cpu().item()),
                f"{prefix}total": float(self.total.detach().cpu().item()),
            }

    def _loss_fn(out: Any, y: torch.Tensor) -> _Stage3Loss:
        nll = gaussian_nll(out.y_hat, out.sigma, y)
        mse = F.mse_loss(out.y_hat, y)
        rreg = reject_regularization(out.reject, weight=1.0)
        total = (
            cfg.lambda_nll * nll
            + cfg.lambda_mse * mse
            + cfg.lambda_reject * rreg
        )
        return _Stage3Loss(nll=nll, mse=mse, reject_reg=rreg, total=total)

    return _loss_fn


def run_stage3_aggregator(
    model,
    train_loader,
    val_loader=None,
    test_loader=None,
    cfg: Optional[TrainerConfig] = None,
    logger: Optional[ExperimentLogger] = None,
    dm=None,
    **kwargs,
):
    """Stage 3: Train aggregator only.

    We keep decomposer / specialists / consensus frozen and teach the
    LightMamba aggregator to emit sensible mixture weights α and rejection
    signals r using a simple NLL + MSE + reject regulariser loss.
    """
    if cfg is None:
        cfg = TrainerConfig(**kwargs)
    freeze_module(model.decomposer)
    freeze_module(model.specialists)
    freeze_module(model.consensus)
    unfreeze_module(model.aggregator)
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
    return result
