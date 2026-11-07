from __future__ import annotations

import math
import os
import random
import time
import warnings
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
    beta2: float = 0.99
    lambda_nll: float = 0.0
    lambda_mse: float = 1.0
    lambda_consensus: float = 0.0
    lambda_reject: float = 0.0
    lambda_orthogonality: float = 0.0
    lambda_agent: float = 0.0
    lambda_cov_penalty: float = 0.0
    target_q: float = 0.95              # target quantile for λ_cov_penalty (95% interval)
    sigma_reg_weight: float = 0.0       # convex σ regulariser (E[σ]+E[log σ]) inside NLL
    lambda_ece: float = 0.0
    lambda_crps: float = 0.0
    lambda_reject_dist: float = 0.0
    # Composite-coverage monitor (S2 previously used; kept for ablation)
    composite_width_weight: float = 1.0
    composite_mse_weight: float = 0.5
    # NaN self-recovery: roll back + decay LR
    nan_recovery_threshold: int = 2
    nan_recovery_max: int = 3
    nan_lr_decay: float = 0.5
    # S2 warm-start consensus (lr warm-up)
    s2_warm_epochs: int = 0
    s2_warm_lr_mult: float = 0.1
    # Temperature scaling applied in _evaluate for val/test metrics reporting.
    # Usually set by main.py AFTER S4 returns (via post-hoc L-BFGS on val preds).
    eval_temperature: float = 1.0
    use_mixture_nll: bool = False
    inverse_transform_eval: bool = True


@dataclass
class MSELoss:
    mse: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.mse

    def logging_dict(self, prefix: str = "") -> Dict[str, float]:
        value = float(self.mse.detach().cpu().item())
        return {f"{prefix}mse": value, f"{prefix}total": value}


def build_mse_loss(out: Any, y: torch.Tensor) -> MSELoss:
    # Accumulate in FP32 even when model predictions use AMP.
    return MSELoss(F.mse_loss(out.y_hat.float(), y.float()))


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
            loss_obj = build_mse_loss(out, y)
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
        # Release cached CUDA blocks at epoch boundary to reduce allocator
        # fragmentation that causes non-deterministic SIGSEGV on long runs.
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return avg_loss, avg_mse

    @torch.no_grad()
    def _evaluate(self, loader: Iterable[Any]) -> Tuple[MetricsResult, List[Any], List[Any], List[Any]]:
        # ---- Double-guard against empty val/test splits (small datasets + long
        #      horizons can leave the precomputed val/test arrays with 0 windows).
        #      In that case we immediately return a NaN-filled metrics dict so the
        #      training run can continue instead of crashing in torch.cat([]).
        _NAN = float("nan")
        _EMPTY_METRICS: MetricsResult = {
            "mse": _NAN, "mae": _NAN, "rmse": _NAN, "mape": _NAN,
            "corr": _NAN, "rse": _NAN,
            "nll": _NAN, "crps": _NAN, "ece": _NAN,
            "q50": 0.0, "q50_width": _NAN,
            "q90": 0.0, "q90_width": _NAN,
            "q95": 0.0, "q95_width": _NAN,
            "composite_coverage": 0.0,
        }
        ds = getattr(loader, "dataset", None)
        if ds is not None:
            try:
                n = len(ds)
            except (TypeError, NotImplementedError):
                n = None
            if n is not None and n <= 0:
                return _EMPTY_METRICS, [], [], []
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
        metrics = compute_metrics(
            preds,
            targets,
            sigmas if len(sigmas) else None,
            temperature=float(getattr(self.cfg, "eval_temperature", 1.0) or 1.0),
        )
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
        self.current_stage: str = str(tag or "")

        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"torch\.optim\.lr_scheduler",
        )

        # ---------------------------------------------------------------------
        # Composite monitor: when cfg.monitor == "composite_coverage", build a
        # single smaller-is-better score from (q95 gap, q95 width, √MSE).
        # Any other string (e.g. "val_mse") is read straight off val_metrics.
        # ---------------------------------------------------------------------
        use_composite = (cfg.monitor == "composite_coverage")
        # mode forced to "min" because composite is smaller-is-better; keep
        # cfg.mode unchanged for downstream logging.
        effective_mode = "min" if use_composite else cfg.mode

        def _compute_monitor(val_m: Optional[MetricsResult], fallback_loss: float) -> float:
            if use_composite:
                if val_m is None:
                    return float("inf")
                from ..losses.calibrator import composite_score_from_metrics
                return composite_score_from_metrics(
                    val_m,
                    target_q=float(cfg.target_q),
                    width_weight=float(getattr(cfg, "composite_width_weight", 1.0)),
                    mse_weight=float(getattr(cfg, "composite_mse_weight", 0.5)),
                    fallback=float("inf"),
                )
            # cfg.monitor is usually written as "val_mse" / "val_nll" / ... but
            # MetricsResult stores plain names ("mse" / "nll" / "crps" / ...).
            # Also tolerate users passing the plain name.  NaN maps to ±inf.
            if val_m is not None:
                key = str(cfg.monitor)
                for candidate in (key, key[4:] if key.startswith("val_") else None,
                                  ("val_" + key) if not key.startswith("val_") else None):
                    if candidate and hasattr(val_m, candidate):
                        raw = getattr(val_m, candidate)
                        break
                else:
                    raw = fallback_loss
            else:
                raw = fallback_loss
            try:
                f = float(raw)
                if f != f:
                    return float("inf") if effective_mode == "min" else float("-inf")
                return f
            except Exception:
                return float("inf") if effective_mode == "min" else float("-inf")

        # NaN recovery state --------------------------------------------------
        nan_streak: int = 0
        nan_rollbacks: int = 0
        recoveries_log: List[Dict[str, Any]] = []

        # S2 warm start ------------------------------------------------------
        s2_warm_epochs = max(0, int(getattr(cfg, "s2_warm_epochs", 0) or 0))
        s2_warm_mult = float(getattr(cfg, "s2_warm_lr_mult", 1.0) or 1.0)
        # Remember base LR (before scheduler warmup) so we can re-apply
        # warm-multiplicative factor without permanently mutating cfg.
        base_lr_before_fit = float(self.optimizer.defaults.get("lr", cfg.lr))

        for epoch in range(1, cfg.max_epochs + 1):
            # ---- Optional S2 warm override for this epoch only ------------
            warm_active = (s2_warm_epochs > 0 and epoch <= s2_warm_epochs)
            if warm_active:
                base_before = float(self.optimizer.param_groups[0]["lr"])
                # Scheduler already computed the intended LR (its warmup is
                # linear multiplicative). We just multiply by s2_warm_mult on
                # top: for epoch <= s2_warm_epochs → LR ↓ to 10%.
                for pg in self.optimizer.param_groups:
                    pg["lr"] = float(pg["lr"]) * s2_warm_mult
                applied_lr = float(self.optimizer.param_groups[0]["lr"])
            else:
                applied_lr = float(self.optimizer.param_groups[0]["lr"])
                base_before = applied_lr

            t0 = time.time()
            if self.logger is not None:
                self.logger.start_epoch()
            train_loss, train_mse = self._train_one_epoch(train_loader, epoch)
            self.scheduler.step()

            # -----------------------------------------------------------------
            # NaN self-recovery: if train_loss is finite → reset streak.
            # Otherwise streak++; if streak >= cfg.nan_recovery_threshold AND
            # we have a best checkpoint: roll back model + optimizer, drop LR
            # × cfg.nan_lr_decay, re-init scheduler with new LR, continue.
            # -----------------------------------------------------------------
            loss_finite = (train_loss == train_loss)
            if loss_finite:
                nan_streak = 0
            else:
                nan_streak += 1
                if (
                    nan_streak >= int(cfg.nan_recovery_threshold)
                    and nan_rollbacks < int(cfg.nan_recovery_max)
                    and self.logger is not None
                    and self.logger.best_checkpoint_path is not None
                ):
                    print(
                        f"  ⚠ [{tag or 'trainer'}] epoch {epoch}: train_loss NaN "
                        f"× {nan_streak} → rollback to best@ep{self.best_epoch} "
                        f"(LR × {cfg.nan_lr_decay})",
                        flush=True,
                    )
                    try:
                        load_checkpoint(
                            self.logger.best_checkpoint_path,
                            self.model,
                            self.optimizer,
                            device=self.device,
                        )
                        new_lr = max(
                            1e-8,
                            float(self.optimizer.param_groups[0]["lr"]) * float(cfg.nan_lr_decay),
                        )
                        for pg in self.optimizer.param_groups:
                            pg["lr"] = new_lr
                        self.optimizer.defaults["lr"] = new_lr
                        self.scheduler = make_scheduler(
                            self.optimizer,
                            max_epochs=max(1, cfg.max_epochs - epoch + 1),
                            warmup_epochs=min(
                                max(1, cfg.warmup_epochs // 2),
                                max(1, cfg.max_epochs - epoch + 1),
                            ),
                            min_lr_factor=cfg.min_lr_factor,
                        )
                        nan_streak = 0
                        nan_rollbacks += 1
                        recoveries_log.append({
                            "epoch": epoch,
                            "new_lr": new_lr,
                            "prev_best_epoch": int(self.best_epoch),
                        })
                    except Exception as exc:
                        print(f"    rollback failed: {exc}", flush=True)

            val_metrics: Optional[MetricsResult] = None
            test_metrics: Optional[MetricsResult] = None
            if val_loader is not None:
                val_metrics = self._evaluate(val_loader)[0]
            if test_loader is not None:
                test_metrics = self._evaluate(test_loader)[0]

            monitor_value = _compute_monitor(val_metrics, train_loss)

            row = LoggedValues(
                stage=self.current_stage or None,
                epoch=epoch,
                elapsed_s=time.time() - t0,
                train_loss=float(train_loss) if loss_finite else float("nan"),
                train_mse=float(train_mse) if (train_mse == train_mse) else float("nan"),
                val_mse=None if val_metrics is None else float(val_metrics.mse),
                val_mae=None if val_metrics is None else float(val_metrics.mae),
                val_corr=None if val_metrics is None else float(val_metrics.corr),
                val_nll=None if val_metrics is None else float(val_metrics.nll),
                val_crps=None if val_metrics is None else float(val_metrics.crps),
                val_ece=None if val_metrics is None else float(val_metrics.ece),
                val_q50=None if val_metrics is None else float(val_metrics.q50_coverage),
                val_q90=None if val_metrics is None else float(val_metrics.q90_coverage),
                val_q95=None if val_metrics is None else float(val_metrics.q95_coverage),
                val_q50_width=None if val_metrics is None else float(val_metrics.q50_width),
                val_q90_width=None if val_metrics is None else float(val_metrics.q90_width),
                val_q95_width=None if val_metrics is None else float(val_metrics.q95_width),
                val_rmse=None if val_metrics is None else float(val_metrics.rmse),
                test_mse=None if test_metrics is None else float(test_metrics.mse),
                test_mae=None if test_metrics is None else float(test_metrics.mae),
                test_nll=None if test_metrics is None else float(test_metrics.nll),
                test_crps=None if test_metrics is None else float(test_metrics.crps),
                test_ece=None if test_metrics is None else float(test_metrics.ece),
                test_q50=None if test_metrics is None else float(test_metrics.q50_coverage),
                test_q90=None if test_metrics is None else float(test_metrics.q90_coverage),
                test_q95=None if test_metrics is None else float(test_metrics.q95_coverage),
                test_q50_width=None if test_metrics is None else float(test_metrics.q50_width),
                test_q90_width=None if test_metrics is None else float(test_metrics.q90_width),
                test_q95_width=None if test_metrics is None else float(test_metrics.q95_width),
                test_rmse=None if test_metrics is None else float(test_metrics.rmse),
            )
            if self.logger is not None:
                self.logger.log_row(row, monitor=cfg.monitor, mode=cfg.mode)
                # Stage checkpoint tensors on CPU, then release the snapshot.
                def _to_cpu(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.detach().cpu()
                    if isinstance(obj, dict):
                        return {k: _to_cpu(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return type(obj)(_to_cpu(v) for v in obj)
                    return obj
                ckpt_state = {
                    "epoch": epoch,
                    "model_state_dict": _to_cpu(self.model.state_dict()),
                    "optimizer_state_dict": _to_cpu(self.optimizer.state_dict()),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "history": self.history,
                    "tag": tag,
                    "config": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(vars(cfg)),
                    "nan_recoveries": list(recoveries_log),
                }
                self.logger.save_checkpoint(
                    ckpt_state,
                    name="last.pt",
                    monitor_value=monitor_value,
                    mode=effective_mode,
                )
                del ckpt_state
                # Release unused cached CUDA blocks at the epoch boundary.
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            try:
                self.history.append(asdict(row) if hasattr(row, "__dataclass_fields__") else dict(vars(row)))
            except Exception:
                self.history.append({"epoch": epoch, "train_loss": row.train_loss, "train_mse": row.train_mse})

            better = False
            if best_monitor is None:
                better = True
            elif effective_mode == "min" and monitor_value < best_monitor:
                better = True
            elif effective_mode == "max" and monitor_value > best_monitor:
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
                # ----------------------------------------------------------------- 指标打印（早停触发）
                elapsed_str = f"{row.elapsed_s:>6.1f}s"
                tag_str = f" [{tag}]" if tag else ""
                header = (
                    f"\n  EPOCH{tag_str} {epoch:>3}/{cfg.max_epochs}  {elapsed_str}  "
                    f"LR={applied_lr:.3e}  best@{best_epoch} {cfg.monitor}={best_monitor:.4f}"
                    f"  ⏹  early stop triggered"
                )
                train_str = (
                    f"    train: loss={row.train_loss:.4f}  mse={row.train_mse:.4f}"
                    + (f"  (WARM ×{s2_warm_mult:.2f})" if warm_active else "")
                    + (f"  (rollbacks={nan_rollbacks})" if nan_rollbacks else "")
                )
                val_str = ""
                if val_metrics is not None:
                    val_str = (
                        f"    val:   mse={row.val_mse:.4f} mae={row.val_mae:.4f} "
                        f"corr={row.val_corr:+.4f} | "
                        f"nll={row.val_nll:.3f} crps={row.val_crps:.4f} ece={row.val_ece:.3f} | "
                        f"Q50={row.val_q50:.2f} Q90={row.val_q90:.2f} Q95={row.val_q95:.2f} "
                        f"(W50={row.val_q50_width:.3f} W90={row.val_q90_width:.3f} W95={row.val_q95_width:.3f}"
                    )
                test_str = ""
                if test_metrics is not None:
                    test_str = (
                        f"    test:  mse={row.test_mse:.4f} mae={row.test_mae:.4f} | "
                        f"nll={row.test_nll:.3f} crps={row.test_crps:.4f} ece={row.test_ece:.3f} | "
                        f"Q50={row.test_q50:.2f} Q90={row.test_q90:.2f} Q95={row.test_q95:.2f} "
                        f"W50={row.test_q50_width:.3f} W90={row.test_q90_width:.3f} W95={row.test_q95_width:.3f}"
                    )
                lines = [l for l in (header, train_str, val_str, test_str) if l]
                print("\n".join(lines), flush=True)
                break

            # --------------------------------------------------------------------- 每个 epoch 的指标打印
            best_hint = "  ★ new best" if better else ""
            monitor_delta = ""
            if not better and best_monitor is not None:
                delta = monitor_value - best_monitor
                if effective_mode == "min":
                    delta = -delta  # 越小越好，显示比 best 差多少
                monitor_delta = f"  Δbest={delta:+.4e}"
            elapsed_str = f"{row.elapsed_s:>6.1f}s"
            tag_str = f" [{tag}]" if tag else ""
            header = (
                f"  EPOCH{tag_str} {epoch:>3}/{cfg.max_epochs}  {elapsed_str}  "
                f"LR={applied_lr:.3e}  {cfg.monitor}={monitor_value:.4f}"
                f"{monitor_delta}{best_hint}"
                + (f"  (WARM ×{s2_warm_mult:.2f})" if warm_active else "")
                + (f"  rollback×{nan_rollbacks}" if nan_rollbacks else "")
            )
            train_str = (
                f"    train: loss={row.train_loss:.4f}  mse={row.train_mse:.4f}"
            )
            val_str = ""
            if val_metrics is not None:
                val_str = (
                    f"    val:   mse={row.val_mse:.4f} mae={row.val_mae:.4f} "
                    f"corr={row.val_corr:+.4f} | "
                    f"nll={row.val_nll:.3f} crps={row.val_crps:.4f} ece={row.val_ece:.3f} | "
                    f"Q50={row.val_q50:.2f} Q90={row.val_q90:.2f} Q95={row.val_q95:.2f} "
                    f"W50={row.val_q50_width:.3f} W90={row.val_q90_width:.3f} W95={row.val_q95_width:.3f}"
                )
            test_str = ""
            if test_metrics is not None:
                test_str = (
                    f"    test:  mse={row.test_mse:.4f} mae={row.test_mae:.4f} | "
                    f"nll={row.test_nll:.3f} crps={row.test_crps:.4f} ece={row.test_ece:.3f} | "
                    f"Q50={row.test_q50:.2f} Q90={row.test_q90:.2f} Q95={row.test_q95:.2f} "
                    f"W50={row.test_q50_width:.3f} W90={row.test_q90_width:.3f} W95={row.test_q95_width:.3f}"
                )
            lines = [l for l in (header, train_str, val_str, test_str) if l]
            print("\n".join(lines), flush=True)

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
        result: Dict[str, Any] = {
            "best_epoch": best_epoch,
            "best_monitor": best_monitor,
            "best": best_payload,
            "history": self.history,
            "best_checkpoint": self.best_ckpt_path,
        }
        if recoveries_log:
            result["nan_recoveries"] = recoveries_log
        return result
