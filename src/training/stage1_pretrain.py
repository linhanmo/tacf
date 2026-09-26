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

    from ..losses.calibrator import coverage_penalty

    @dataclass
    class _Stage1Loss:
        specialist_nll: torch.Tensor
        specialist_mse: torch.Tensor
        agent_hetero: torch.Tensor
        orthogonality: torch.Tensor
        coverage_penalty: torch.Tensor
        total: torch.Tensor

        def logging_dict(self, prefix: str = "") -> Dict[str, float]:
            return {
                f"{prefix}specialist_nll": float(self.specialist_nll.detach().cpu().item()),
                f"{prefix}specialist_mse": float(self.specialist_mse.detach().cpu().item()),
                f"{prefix}agent_hetero": float(self.agent_hetero.detach().cpu().item()),
                f"{prefix}orthogonality": float(self.orthogonality.detach().cpu().item()),
                f"{prefix}coverage_penalty": float(self.coverage_penalty.detach().cpu().item()),
                f"{prefix}total": float(self.total.detach().cpu().item()),
            }

    def _loss_fn(out: Any, y: torch.Tensor) -> _Stage1Loss:
        dev = y.device
        dt = y.dtype
        nll_sum = torch.zeros((), device=dev, dtype=torch.float32)
        mse_sum = torch.zeros((), device=dev, dtype=torch.float32)
        cov_sum = torch.zeros((), device=dev, dtype=torch.float32)
        n_spec = 0
        mu_dict: Dict[str, torch.Tensor] = {}
        sreg = float(getattr(cfg, "sigma_reg_weight", 0.0) or 0.0)
        for name, spec in out.specialists_out.outputs.items():
            mu = spec.mu
            sigma = spec.sigma
            nll_sum = nll_sum + gaussian_nll(mu, sigma, y, sigma_reg_weight=sreg).to(torch.float32)
            mse_sum = mse_sum + F.mse_loss(mu.float(), y.float())
            cov_sum = cov_sum + coverage_penalty(sigma, mu, y, target_q=cfg.target_q).to(torch.float32)
            mu_dict[name] = mu
            n_spec += 1
        denom = max(1, n_spec)
        nll = nll_sum / denom
        mse = mse_sum / denom
        cov_loss = cov_sum / denom
        hetero = build_agent_heterogeneous_losses(mu_dict, y)
        ortho = out.aux_losses.get(
            "orthogonality",
            torch.zeros((), device=dev, dtype=torch.float32),
        )
        if not torch.is_tensor(ortho):
            ortho = torch.as_tensor(ortho, device=dev, dtype=torch.float32)
        else:
            ortho = ortho.to(device=dev, dtype=torch.float32)
        total = (
            float(cfg.lambda_nll) * nll
            + float(cfg.lambda_mse) * mse
            + float(cfg.lambda_agent) * hetero.total.to(device=dev, dtype=torch.float32)
            + float(cfg.lambda_orthogonality) * ortho
            + float(cfg.lambda_cov_penalty) * cov_loss
        ).to(dtype=dt)
        return _Stage1Loss(
            specialist_nll=nll,
            specialist_mse=mse,
            agent_hetero=hetero.total,
            orthogonality=ortho,
            coverage_penalty=cov_loss,
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

    Additionally we freeze the learnable sigma_global_multiplier during S1 so
    that σ calibration is fully driven by the specialist heads +
    sigma_reg_weight, and only left to the aggregator (S3) and E2E finetune
    (S4) to fine-tune the overall scale.  This prevents the model from using
    the global scalar as a "cheat code" to compensate for specialist collapse
    in the first few epochs.
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
