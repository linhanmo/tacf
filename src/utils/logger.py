from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


__all__ = ["ExperimentLogger", "save_checkpoint", "load_checkpoint"]


@dataclass
class LoggedValues:
    stage: Optional[str] = None
    epoch: int
    elapsed_s: float
    train_loss: float
    train_mse: Optional[float] = None
    val_mse: Optional[float] = None
    val_mae: Optional[float] = None
    val_rmse: Optional[float] = None
    val_corr: Optional[float] = None
    val_q95: Optional[float] = None
    val_nll: Optional[float] = None
    test_mse: Optional[float] = None
    test_mae: Optional[float] = None
    test_rmse: Optional[float] = None
    test_q95: Optional[float] = None
    test_nll: Optional[float] = None
    extra: Dict[str, Any] | None = None


def _default_project_name() -> str:
    return datetime.now().strftime("tacf_%Y%m%d_%H%M%S")


class ExperimentLogger:
    """Lightweight CSV + JSON + TensorBoard-friendly experiment logger."""

    def __init__(
        self,
        log_dir: str | os.PathLike,
        project: Optional[str] = None,
        use_tqdm: bool = True,
    ) -> None:
        self.project = project or _default_project_name()
        self.log_dir = Path(log_dir) / self.project
        self.ckpt_dir = self.log_dir / "checkpoints"
        self.vis_dir = self.log_dir / "figures"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / "history.csv"
        self.json_path = self.log_dir / "history.json"
        self.config_path = self.log_dir / "config.json"
        self._rows: List[Dict[str, Any]] = []
        self._best_monitor: Optional[float] = None
        self._best_ckpt: Optional[Path] = None
        self._epoch_start = time.time()
        self._use_tqdm = use_tqdm

        self._tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            self._tb_writer = SummaryWriter(log_dir=str(self.log_dir))
        except Exception:
            self._tb_writer = None

    # ------------------------------------------------------------------ config
    def log_config(self, cfg: Any) -> None:
        cfg_s: Dict[str, Any]
        if hasattr(cfg, "__dict__"):
            cfg_s = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg.__dict__)
        elif isinstance(cfg, dict):
            cfg_s = dict(cfg)
        else:
            cfg_s = {"repr": repr(cfg)}
        with open(self.config_path, "w") as f:
            json.dump(cfg_s, f, indent=2, default=str)

    # ------------------------------------------------------------------ rows
    @property
    def best_checkpoint_path(self) -> Optional[Path]:
        return self._best_ckpt

    def start_epoch(self) -> None:
        self._epoch_start = time.time()

    def log_row(
        self,
        row: LoggedValues,
        monitor: Optional[str] = "val_mse",
        mode: str = "min",
    ) -> None:
        now = time.time()
        payload = asdict(row)
        if payload["extra"] is None:
            del payload["extra"]
        else:
            for k, v in payload["extra"].items():
                payload[f"extra_{k}"] = float(v) if isinstance(v, (int, float)) else v
            del payload["extra"]
        payload.setdefault("elapsed_s", now - self._epoch_start)
        self._rows.append(payload)
        self._append_csv(payload)
        if monitor in payload and payload[monitor] is not None:
            self._maybe_update_best(payload, monitor=monitor, mode=mode)
        if self._tb_writer is not None:
            for k, v in payload.items():
                if isinstance(v, (int, float)):
                    try:
                        self._tb_writer.add_scalar(k, float(v), row.epoch)
                    except Exception:
                        pass

    def _append_csv(self, payload: Dict[str, Any]) -> None:
        all_keys: set = set()
        for r in self._rows:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)
        # 如果之前写入过 CSV，但旧的字段集合与当前完整字段集合不匹配（通常因为
        # 某个 stage 新增了列，比如 ``stage``、``extra_*``），则把 CSV 整体
        # 重写一次，这样所有历史行 + 新行的列完全对齐，没有错位/缺列风险。
        rewrite_needed = False
        if self.csv_path.exists():
            try:
                with open(self.csv_path, "r", newline="") as f:
                    reader = csv.reader(f)
                    existing_header = next(reader, [])
            except Exception:
                existing_header = []
            rewrite_needed = (set(existing_header) != set(fieldnames))
        if rewrite_needed or (not self.csv_path.exists()):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in self._rows:
                    w.writerow({k: r.get(k, "") for k in fieldnames})
        else:
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writerow({k: payload.get(k, "") for k in fieldnames})
        with open(self.json_path, "w") as f:
            json.dump(self._rows, f, indent=2, default=str)

    def _maybe_update_best(
        self, payload: Dict[str, Any], monitor: str, mode: str
    ) -> None:
        value = float(payload[monitor])
        better = False
        if self._best_monitor is None:
            better = True
        elif mode == "min" and value < self._best_monitor:
            better = True
        elif mode == "max" and value > self._best_monitor:
            better = True
        if better:
            self._best_monitor = value

    # ------------------------------------------------------------------ tqdm
    def iter_with_progress(
        self, iterable: Iterable, desc: str = ""
    ) -> Iterable:
        if not self._use_tqdm:
            return iterable
        try:
            from tqdm.auto import tqdm

            return tqdm(iterable, desc=desc, leave=False, file=sys.stdout)
        except Exception:
            return iterable

    # ------------------------------------------------------------------ ckpt
    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        name: str = "last.pt",
        monitor_value: Optional[float] = None,
        mode: str = "min",
    ) -> Path:
        path = self.ckpt_dir / name
        torch.save(state_dict, path)
        if monitor_value is None:
            return path
        better = False
        if self._best_ckpt is None:
            better = True
        elif mode == "min" and monitor_value < (self._best_monitor or float("inf")):
            better = True
        elif mode == "max" and monitor_value > (self._best_monitor or -float("inf")):
            better = True
        if better:
            best_path = self.ckpt_dir / "best.pt"
            shutil.copy(path, best_path)
            self._best_ckpt = best_path
            self._best_monitor = monitor_value
            best_meta = best_path.with_suffix(".json")
            with open(best_meta, "w") as f:
                json.dump(
                    {"path": str(best_path), "monitor_value": monitor_value, "mode": mode},
                    f,
                    indent=2,
                )
        return path

    def close(self) -> None:
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass


def save_checkpoint(
    ckpt_path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        payload.update(extra)
    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device or "cpu")
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt
