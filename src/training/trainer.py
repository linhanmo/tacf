from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    _LRScheduler,
)

from ..utils.logger import ExperimentLogger, LoggedValues, load_checkpoint, save_checkpoint
from ..utils.metrics import MetricsResult, compute_metrics


__all__ = [
    "TrainerConfig",
    "Trainer",
    "seed_everything",
    "make_optimizer",
    "make_scheduler",
    "freeze_module",
    "unfreeze_module",
]


@dataclass
class TrainerConfig:
    max_epochs: int = 100
    early_stop: int = 10
    monitor: str = "val_mse"
    mode: str = "min"
    amp: bool = True
    device: Optional[str] = None
    grad_clip: float = 1.0
    accum_steps: int = 1
    warmup_epochs: int = 5
    min_lr_factor: float = 0.01
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    lambda_nll: float = 2.0             # post-smoke-run upgrade: was 1.0
    lambda_mse: float = 1.0
    lambda_consensus: float = 0.1
    lambda_reject: float = 0.01
    lambda_orthogonality: float = 0.01
    lambda_agent: float = 0.05
    lambda_cov_penalty: float = 0.25    # NEW: per-sample under-coverage hinge + batch gap
    target_q: float = 0.95              # NEW: target quantile for λ_cov_penalty (95% interval)
    use_mixture_nll: bool = False
    inverse_transform_eval: bool = True


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_optimizer(
    params: Iterable[torch.Tensor],
    optimizer: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
) -> torch.optim.Optimizer:
    opt_cls = AdamW if optimizer.lower() == "adamw" else Adam
    return opt_cls(
        params,
        lr=lr,
        weight_decay=weight_decay if opt_cls is AdamW else 0.0,
        betas=(beta1, beta2),
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    warmup_epochs: int = 5,
    min_lr_factor: float = 0.01,
) -> _LRScheduler:
    warmup = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(1, warmup_epochs)
    )
    remaining = max(1, max_epochs - warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=optimizer.defaults["lr"] * min_lr_factor)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)


def _named_filter(
    model: nn.Module,
    trainable_names: Optional[Sequence[str]] = None,
    frozen_names: Optional[Sequence[str]] = None,
) -> List[torch.Tensor]:
    params: List[torch.Tensor] = []
    for name, p in model.named_parameters():
        keep = True
        if trainable_names is not None:
            keep = any(tok in name for tok in trainable_names)
        if keep and frozen_names is not None:
            keep = not any(tok in name for tok in frozen_names)
        p.requires_grad_(keep)
        if keep:
            params.append(p)
    return params


class Trainer:
    """General-purpose TACF trainer.

    The trainer is intentionally small and composable — per-stage logic is
    written in the corresponding ``stage*.py`` module and invokes this trainer
    with a specific parameter filter (``trainable_names`` / ``frozen_names``),
    dataloader pair, loss-weight configuration, and a custom hook (if any).
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainerConfig,
        logger: Optional[ExperimentLogger] = None,
        build_loss_fn: Optional[Callable[[Any, torch.Tensor], Any]] = None,
        inverse_transform_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        trainable_names: Optional[Sequence[str]] = None,
        frozen_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.logger = logger
        self.build_loss_fn = build_loss_fn
        self.inverse_transform_fn = inverse_transform_fn
        self.device = torch.device(
            cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.trainable_parameters = _named_filter(
            self.model, trainable_names=trainable_names, frozen_names=frozen_names
        )
        self.optimizer = make_optimizer(
            self.trainable_parameters,
            optimizer=cfg.optimizer,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            beta1=cfg.beta1,
            beta2=cfg.beta2,
        )
        self.scheduler = make_scheduler(
            self.optimizer,
            max_epochs=cfg.max_epochs,
            warmup_epochs=cfg.warmup_epochs,
            min_lr_factor=cfg.min_lr_factor,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda" and cfg.amp))
        self.best_monitor: Optional[float] = None
        self.best_epoch = 0
        self.best_ckpt_path: Optional[str] = None
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ core
    def _forward_loss(
        self, batch: Tuple[torch.Tensor, ...],
    ) -> Tuple[Any, Any, torch.Tensor, torch.Tensor]:
        x, x_stamp, y, y_stamp = [t.to(self.device) for t in batch[:4]]
        out = self.model(x, x_stamp, y)
        if self.build_loss_fn is not None:
            loss_obj = self.build_loss_fn(out, y)
            total = loss_obj.total if hasattr(loss_obj, "total") else loss_obj
        else:
            from ..losses.calibrator import build_total_loss

            loss_obj = build_total_loss(
                out,
                y,
                lambda_nll=self.cfg.lambda_nll,
                lambda_mse=self.cfg.lambda_mse,
                lambda_consensus=self.cfg.lambda_consensus,
                lambda_reject=self.cfg.lambda_reject,
                lambda_orthogonality=self.cfg.lambda_orthogonality,
                lambda_agent=self.cfg.lambda_agent,
                lambda_cov_penalty=self.cfg.lambda_cov_penalty,
                target_q=self.cfg.target_q,
                use_mixture_nll=self.cfg.use_mixture_nll,
            )
            total = loss_obj.total
        return out, loss_obj, y, total

    def _train_one_epoch(
        self,
        loader: Iterable[Any],
        epoch: int,
    ) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0
        self.optimizer.zero_grad(set_to_none=True)
        prog = (
            self.logger.iter_with_progress(loader, desc=f"train {epoch}")
            if self.logger is not None
            else loader
        )
        for step, batch in enumerate(prog):
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                out, loss_obj, y, loss = self._forward_loss(batch)
                loss = loss / self.cfg.accum_steps
            self.scaler.scale(loss).backward()
            if (step + 1) % self.cfg.accum_steps == 0:
                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable_parameters, self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu().item() * self.cfg.accum_steps)
            with torch.no_grad():
                total_mse += float(F.mse_loss(out.y_hat.detach(), y).cpu().item())
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)
        avg_mse = total_mse / max(1, n_batches)
        return avg_loss, avg_mse

    @torch.no_grad()
    def _evaluate(self, loader: Iterable[Any]) -> Tuple[MetricsResult, List[Any], List[Any], List[Any]]:
        self.model.eval()
        preds: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        sigmas: List[torch.Tensor] = []
        for batch in loader:
            x, x_stamp, y, y_stamp = [t.to(self.device) for t in batch[:4]]
            out = self.model(x, x_stamp)
            y_hat = out.y_hat.detach()
            sigma = out.sigma.detach()
            y_t = y.detach()
            if self.cfg.inverse_transform_eval and self.inverse_transform_fn is not None:
                y_hat = self.inverse_transform_fn(y_hat)
                sigma = self.inverse_transform_fn(sigma) - self.inverse_transform_fn(torch.zeros_like(sigma))
                y_t = self.inverse_transform_fn(y_t)
            preds.append(y_hat.cpu())
            sigmas.append(sigma.cpu())
            targets.append(y_t.cpu())
        metrics = compute_metrics(preds, targets, sigmas if len(sigmas) else None)
        return metrics, preds, targets, sigmas

    # ------------------------------------------------------------------ main
    def fit(
        self,
        train_loader: Iterable[Any],
        val_loader: Optional[Iterable[Any]] = None,
        test_loader: Optional[Iterable[Any]] = None,
        tag: str = "",
    ) -> Dict[str, Any]:
        cfg = self.cfg
        best_payload: Dict[str, Any] | None = None
        best_epoch = -1
        best_monitor: Optional[float] = None
        for epoch in range(1, cfg.max_epochs + 1):
            t0 = time.time()
            if self.logger is not None:
                self.logger.start_epoch()
            train_loss, train_mse = self._train_one_epoch(train_loader, epoch)
            self.scheduler.step()

            val_metrics: Optional[MetricsResult] = None
            test_metrics: Optional[MetricsResult] = None
            if val_loader is not None:
                val_metrics, *_ = self._evaluate(val_loader)
            if test_loader is not None:
                test_metrics, *_ = self._evaluate(test_loader)

            monitor_value = (
                float(getattr(val_metrics, cfg.monitor, train_loss))
                if val_metrics is not None
                else float(train_loss)
            )

            row = LoggedValues(
                epoch=epoch,
                elapsed_s=time.time() - t0,
                train_loss=float(train_loss),
                train_mse=float(train_mse),
                val_mse=None if val_metrics is None else float(val_metrics.mse),
                val_mae=None if val_metrics is None else float(val_metrics.mae),
                val_rmse=None if val_metrics is None else float(val_metrics.rmse),
                val_corr=None if val_metrics is None else float(val_metrics.corr),
                val_q95=None if val_metrics is None else float(val_metrics.q95_coverage),
                val_nll=None if val_metrics is None else float(val_metrics.nll),
                test_mse=None if test_metrics is None else float(test_metrics.mse),
                test_mae=None if test_metrics is None else float(test_metrics.mae),
                test_rmse=None if test_metrics is None else float(test_metrics.rmse),
                test_q95=None if test_metrics is None else float(test_metrics.q95_coverage),
                test_nll=None if test_metrics is None else float(test_metrics.nll),
            )
            if self.logger is not None:
                self.logger.log_row(row, monitor=cfg.monitor, mode=cfg.mode)
                ckpt_state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "history": self.history,
                    "tag": tag,
                    "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(vars(cfg)),
                }
                self.logger.save_checkpoint(
                    ckpt_state,
                    name="last.pt",
                    monitor_value=monitor_value,
                    mode=cfg.mode,
                )
            # keep trainer-local history copy for return dict (best_stage_selection etc.)
            try:
                self.history.append(asdict(row) if hasattr(row, "__dataclass_fields__") else dict(vars(row)))
            except Exception:
                self.history.append({"epoch": epoch, "train_loss": row.train_loss, "train_mse": row.train_mse})

            better = False
            if best_monitor is None:
                better = True
            elif cfg.mode == "min" and monitor_value < best_monitor:
                better = True
            elif cfg.mode == "max" and monitor_value > best_monitor:
                better = True
            if better:
                best_monitor = monitor_value
                best_epoch = epoch
                best_payload = {
                    "train_loss": row.train_loss,
                    "train_mse": row.train_mse,
                    "val_metrics": None if val_metrics is None else val_metrics.as_dict(),
                    "test_metrics": None if test_metrics is None else test_metrics.as_dict(),
                    "monitor_value": monitor_value,
                }
                self.best_epoch = best_epoch
                self.best_monitor = best_monitor
            elif cfg.early_stop > 0 and (epoch - best_epoch) >= cfg.early_stop:
                break

        if self.logger is not None and self.logger.best_checkpoint_path is not None:
            try:
                load_checkpoint(
                    self.logger.best_checkpoint_path,
                    self.model,
                    device=self.device,
                )
                self.best_ckpt_path = str(self.logger.best_checkpoint_path)
            except Exception:
                pass
        return {
            "best_epoch": best_epoch,
            "best_monitor": best_monitor,
            "best": best_payload,
            "history": self.history,
            "best_checkpoint": self.best_ckpt_path,
        }
