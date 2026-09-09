from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Ensure project root on path
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent
_PROJ_DIR = _SRC_DIR.parent
for _p in (_PROJ_DIR, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.data.dataloader import TACFDataModule
from src.models.tacf import TACF
from src.training import (
    TrainerConfig,
    run_stage1_pretrain,
    run_stage2_comm,
    run_stage3_aggregator,
    run_stage4_finetune,
    seed_everything,
)
from src.utils.logger import ExperimentLogger


def build_model_from_dm(dm: TACFDataModule, **override: Any) -> TACF:
    return TACF(
        in_channels=dm.n_features,
        out_channels=dm.n_features,
        seq_len=dm.seq_len,
        pred_len=dm.pred_len,
        **override,
    )


def main(argv=None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="TACF main 4-stage experiment runner")
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--seq-len", type=int, default=336)
    parser.add_argument("--pred-len", type=int, default=168)
    parser.add_argument("--label-len", type=int, default=168)
    parser.add_argument("--root", type=str, default=str(_PROJ_DIR / "preprocess"))
    parser.add_argument("--log-dir", type=str, default=str(_PROJ_DIR / "logs"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs-stage1", type=int, default=30)
    parser.add_argument("--max-epochs-stage2", type=int, default=10)
    parser.add_argument("--max-epochs-stage3", type=int, default=10)
    parser.add_argument("--max-epochs-stage4", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--agg-d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--early-stop", type=int, default=10)
    args = parser.parse_args(argv)

    seed_everything(args.seed)

    dm = TACFDataModule(
        root=args.root,
        dataset=args.dataset,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        label_len=args.label_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        use_time_features=True,
    )
    print(dm)
    train_loader, val_loader, test_loader = dm.dataloaders()

    model_override = dict(
        specialist_d_model=args.d_model,
        specialist_n_layers=args.n_layers,
        aggregator_d_model=args.agg_d_model,
    )
    model = build_model_from_dm(dm, **model_override)
    print(model)

    cfg_common = dict(
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        amp=bool(args.amp and torch.cuda.is_available()),
        lr=args.lr,
        grad_clip=1.0,
        monitor="val_mse",
        mode="min",
        lambda_agent=0.05,
    )

    logger = ExperimentLogger(log_dir=args.log_dir)
    logger.log_config({"args": vars(args), "stats": asdict(dm.stats)})

    results: Dict[str, Any] = {}
    stages = [
        ("stage1", run_stage1_pretrain, TrainerConfig(max_epochs=args.max_epochs_stage1, early_stop=args.early_stop, **cfg_common)),
        ("stage2", run_stage2_comm, TrainerConfig(max_epochs=args.max_epochs_stage2, early_stop=max(3, args.early_stop//2), **cfg_common)),
        ("stage3", run_stage3_aggregator, TrainerConfig(max_epochs=args.max_epochs_stage3, early_stop=max(3, args.early_stop//2), **cfg_common)),
        ("stage4", run_stage4_finetune, TrainerConfig(max_epochs=args.max_epochs_stage4, early_stop=args.early_stop, **cfg_common)),
    ]
    for tag, runner, trainer_cfg in stages:
        print(f"\n===== Running {tag} =====")
        res = runner(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            cfg=trainer_cfg,
            logger=logger,
            dm=dm,
        )
        results[tag] = {k: (str(v) if isinstance(v, Path) else v) for k, v in res.items()}
        best = res.get("best")
        if best is not None:
            print(f"  best_epoch={res.get('best_epoch')}  tag={tag}")
        print(f"  best_monitor={res.get('best_monitor')}  best_checkpoint={res.get('best_checkpoint')}")

    with open(logger.log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()
